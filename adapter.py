#!/usr/bin/env python3
"""Alpha Auditor JSONL Adapter for the AgentVerse evaluation harness.

Connects the AgentVerse CLI runner to Alpha Auditor's deterministic engine:
- 'market_step': causal sequential walk-forward market decision engine
- 'audit_single_asset': deterministic single-asset backtest verification
- 'audit_pairs_spread': deterministic pairs OLS & z-score verification
"""

from __future__ import annotations

import json
import math
import sys

import pandas as pd

from engine.backtester import compute_causal_signals, simulate_strategy_core
from engine.schemas import (
    ExecutionAssumptions,
    RiskControls,
    StrategyParameters,
    StrategySpec,
)
from engine.walk_forward_stepper import decide_market_step


def _handle_audit_single_asset(data: dict) -> dict:
    """Deterministic verification endpoint for custom single-asset backtest cases."""
    prices = data.get("prices") if "prices" in data else data.get("close")
    dates = data.get("dates")
    if (
        not isinstance(prices, list)
        or not isinstance(dates, list)
        or len(prices) != len(dates)
        or len(prices) < 10
        or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0 for p in prices)
    ):
        raise ValueError("Invalid prices or dates input")

    df = pd.DataFrame(
        {
            "date": [str(d) for d in dates],
            "open_a": [float(p) for p in prices],
            "high_a": [float(p) for p in prices],
            "low_a": [float(p) for p in prices],
            "close_a": [float(p) for p in prices],
            "volume_a": [1000] * len(prices),
            "open_bench": [float(p) for p in prices],
            "close_bench": [float(p) for p in prices],
        }
    )
    spec = StrategySpec(
        hypothesis_text=str(data.get("hypothesis_text", "Test return threshold")),
        title="Adapter Single-Asset Verification",
        strategy_family="single_asset",
        template_id=data.get("template_id", "return_threshold"),
        signal_variant=data.get("signal_variant", "cum_return_drop"),
        asset_a="TEST",
        benchmark_ticker="TEST",
        parameters=StrategyParameters(
            lookback_window=int(data.get("lookback_window", 1)),
            streak_length=int(data.get("streak_length", 3)),
            entry_threshold=float(data.get("entry_threshold", 2.0)),
            exit_threshold=float(data.get("exit_threshold", 0.0)),
            holding_period=int(data.get("holding_period", 1)),
            exit_mode=data.get("exit_mode", "fixed_hold"),
            direction=data.get("direction", "long"),
        ),
        execution=ExecutionAssumptions(
            initial_capital=100_000.0,
            execution_timing="next_close",
            commission_bps=float(data.get("commission_bps", 0.0)),
            slippage_bps=float(data.get("slippage_bps", 0.0)),
        ),
        risk_controls=RiskControls(
            max_drawdown_circuit_breaker_pct=None,
            per_trade_stop_loss_pct=None,
            max_holding_days=100,
        ),
        human_confirmed=True,
    )
    split_idx = len(df) // 2
    sim = simulate_strategy_core(df, spec, split_idx)
    trades = sim["trades"]
    final_eq = int(round(float(sim["equity_net"][-1])))
    return {
        "status": "ok",
        "trade_count": len(trades),
        "final_equity_rounded": final_eq,
        "entry_dates": [t.entry_date for t in trades],
        "exit_dates": [t.exit_date for t in trades],
    }


def _handle_audit_pairs_spread(data: dict) -> dict:
    """Deterministic verification endpoint for custom pairs-trading OLS & z-score cases."""
    prices_a = data.get("prices_a") if "prices_a" in data else data.get("close_a")
    prices_b = data.get("prices_b") if "prices_b" in data else data.get("close_b")
    dates = data.get("dates")
    if (
        not isinstance(prices_a, list)
        or not isinstance(prices_b, list)
        or not isinstance(dates, list)
        or len(prices_a) != len(dates)
        or len(prices_b) != len(dates)
        or len(prices_a) < 30
        or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0 for p in prices_a)
        or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p <= 0 for p in prices_b)
    ):
        raise ValueError("Invalid prices_a, prices_b, or dates input")

    df = pd.DataFrame(
        {
            "date": [str(d) for d in dates],
            "open_a": [float(p) for p in prices_a],
            "high_a": [float(p) for p in prices_a],
            "low_a": [float(p) for p in prices_a],
            "close_a": [float(p) for p in prices_a],
            "volume_a": [1000] * len(prices_a),
            "open_b": [float(p) for p in prices_b],
            "high_b": [float(p) for p in prices_b],
            "low_b": [float(p) for p in prices_b],
            "close_b": [float(p) for p in prices_b],
            "volume_b": [1000] * len(prices_b),
            "open_bench": [float(p) for p in prices_a],
            "close_bench": [float(p) for p in prices_a],
        }
    )
    spec = StrategySpec(
        hypothesis_text=str(data.get("hypothesis_text", "Test pairs spread reversion")),
        title="Adapter Pairs Verification",
        strategy_family="pairs_trading",
        template_id=data.get("template_id", "pairs_cointegration_reversion"),
        signal_variant=data.get("signal_variant", "ols_log"),
        asset_a=str(data.get("asset_a", "TEST_A")),
        asset_b=str(data.get("asset_b", "TEST_B")),
        benchmark_ticker=str(data.get("asset_a", "TEST_A")),
        parameters=StrategyParameters(
            lookback_window=int(data.get("lookback_window", 20)),
            entry_threshold=float(data.get("entry_threshold", 2.0)),
            exit_threshold=float(data.get("exit_threshold", 0.25)),
            stop_z_threshold=float(data.get("stop_z_threshold", 4.0)),
            holding_period=int(data.get("holding_period", 15)),
            exit_mode=data.get("exit_mode", "threshold_reversion"),
            direction=data.get("direction", "both"),
        ),
        execution=ExecutionAssumptions(
            initial_capital=100_000.0,
            execution_timing=data.get("execution_timing", "next_close"),
            commission_bps=float(data.get("commission_bps", 0.0)),
            slippage_bps=float(data.get("slippage_bps", 0.0)),
        ),
        risk_controls=RiskControls(
            max_drawdown_circuit_breaker_pct=None,
            per_trade_stop_loss_pct=None,
            max_holding_days=int(data.get("max_holding_days", 60)),
        ),
        human_confirmed=True,
    )
    split_idx = max(25, len(df) // 2)
    precomputed = compute_causal_signals(df, spec, split_idx)
    sim = simulate_strategy_core(df, spec, split_idx, precomputed_signals=precomputed)
    trades = sim["trades"]
    final_eq = int(round(float(sim["equity_net"][-1])))
    return {
        "status": "ok",
        "signal_variant": spec.signal_variant,
        "trade_count": len(trades),
        "final_equity_rounded": final_eq,
        "ols_beta_formation": round(float(precomputed[4]), 4),
        "entry_dates": [t.entry_date for t in trades],
        "exit_dates": [t.exit_date for t in trades],
    }


def solve(request: dict) -> dict:
    if not isinstance(request, dict):
        return {"error": "invalid_input"}
    task = request.get("task")
    if "input" in request:
        data = request.get("input")
        if not isinstance(data, dict):
            return {"error": "invalid_input"}
    else:
        data = request

    try:
        if task == "market_step":
            return decide_market_step(data)
        if task == "audit_single_asset":
            return _handle_audit_single_asset(data)
        if task == "audit_pairs_spread":
            return _handle_audit_pairs_spread(data)
    except Exception:
        return {"error": "invalid_input"}

    return {"error": "unsupported_task"}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except ValueError:
            request = None
        request_id = request.get("request_id") if isinstance(request, dict) else None
        output = solve(request) if request_id is not None else {"error": "invalid_input"}
        print(
            json.dumps({"request_id": request_id, "output": output}),
            flush=True,
        )


if __name__ == "__main__":
    main()
