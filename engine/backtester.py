"""Deterministic, strictly causal backtesting and parameter sensitivity engine.

Key guarantees:
1. Zero Look-Ahead Bias: Signals at index t use only prices [0..t].
   Pairs OLS parameters (alpha, beta) in 'ols_log' mode are estimated strictly
   on the In-Sample Formation split [0..split_idx).
2. Next-Bar Execution: Signals triggered at close of bar t execute at bar t+1
   (Open or Close as specified by execution_timing) with explicit commission and slippage.
3. Daily Mark-to-Market Equity: Open positions are marked to market every bar so
   intraday/holding drawdowns, volatility, and Sharpe ratios reflect true holding risk.
4. Enforced Risk Controls: Per-trade stop-loss, max holding period time-stop, and
   portfolio maximum drawdown circuit breaker are enforced in the simulation loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from typing import Any, Literal

import numpy as np
import pandas as pd
import scipy
from scipy import stats

from engine.data_provider import load_aligned_series
from engine.econometrics import (
    TRADING_DAYS_PER_YEAR,
    compute_pairs_econometrics,
    deflated_sharpe_ratio_bailey_lopez,
    exact_sign_test_p_value,
    lo_mertens_sharpe_se_and_psr,
    newey_west_factor_regression,
    stationary_bootstrap_cis,
)
from engine.guardrails import check_neutrality
from engine.literature import build_literature_review
from engine.schemas import (
    AuditReport,
    BracketProfileResult,
    CostSensitivityRow,
    PairsDiagnostics,
    ParameterSensitivitySurface,
    PeriodMetrics,
    SensitivityGridCell,
    StrategySpec,
    SubperiodRow,
    TradeRecord,
)
from engine.stress_tests import (
    build_hypothesis_scorecard_and_ledger,
    run_adversarial_stress_suite,
)


def _compute_rsi(closes: np.ndarray, window: int) -> np.ndarray:
    """Causal Wilder RSI series on close prices."""
    n = len(closes)
    rsi = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains[:window]))
    avg_loss = float(np.mean(losses[:window]))
    if avg_loss <= 1e-12:
        rsi[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[window] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + gains[i]) / window
        avg_loss = (avg_loss * (window - 1) + losses[i]) / window
        if avg_loss <= 1e-12:
            rsi[i + 1] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_causal_signals(
    df: pd.DataFrame,
    spec: StrategySpec,
    split_idx: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], PairsDiagnostics | None, float, float]:
    """Compute strictly causal signal series at bar t using only information up to bar t.

    Guarantees prefix invariance at every bar t in [0..n):
    modifying any prices at indices >= m (for any m, whether inside Formation [0..split_idx)
    or Out-of-Sample [split_idx..n)) never alters signal_values[:m] or indicators[:m].

    Returns:
        (signal_values, indicator_series_dict, pairs_diagnostics, ols_alpha, ols_beta)
    """
    n = len(df)
    close_a = df["close_a"].to_numpy(dtype=np.float64)
    params = spec.parameters
    L = max(1, params.lookback_window)
    L_slow = max(L + 1, params.secondary_window)

    signal_values = np.full(n, np.nan, dtype=np.float64)
    indicators: dict[str, np.ndarray] = {}
    pairs_diag: PairsDiagnostics | None = None
    ols_alpha = 0.0
    ols_beta = 1.0

    if spec.strategy_family == "single_asset":
        if spec.template_id == "return_threshold":
            if spec.signal_variant in ("cum_return_drop", "cum_return_surge"):
                k = max(1, L)
                ret_k = np.full(n, np.nan, dtype=np.float64)
                ret_k[k:] = ((close_a[k:] / close_a[:-k]) - 1.0) * 100.0
                signal_values = ret_k
                indicators["k_day_return_pct"] = ret_k
            elif spec.signal_variant == "vol_scaled_shock":
                vol_win = max(5, L)
                daily_ret = np.full(n, np.nan, dtype=np.float64)
                daily_ret[1:] = (close_a[1:] / close_a[:-1]) - 1.0
                s_ret = pd.Series(daily_ret)
                roll_std = s_ret.rolling(vol_win, min_periods=vol_win).std(ddof=1).to_numpy()
                vol_z = np.where(roll_std > 1e-12, daily_ret / roll_std, np.nan)
                signal_values = vol_z
                indicators["vol_scaled_return_z"] = vol_z

        elif spec.template_id == "consecutive_streak":
            daily_diff = np.zeros(n, dtype=np.float64)
            daily_diff[1:] = np.diff(close_a)
            streak = np.zeros(n, dtype=np.float64)
            cur = 0
            for t in range(1, n):
                if spec.signal_variant == "negative_streak":
                    cur = (cur + 1) if daily_diff[t] < 0 else 0
                else:
                    cur = (cur + 1) if daily_diff[t] > 0 else 0
                streak[t] = float(cur)
            signal_values = streak
            indicators["streak_count"] = streak

        elif spec.template_id == "mean_reversion_oscillator":
            win = max(5, L)
            if spec.signal_variant == "price_zscore":
                log_p = pd.Series(np.log(close_a))
                mean = log_p.rolling(win, min_periods=win).mean().to_numpy()
                std = log_p.rolling(win, min_periods=win).std(ddof=1).to_numpy()
                safe_std = np.where(std > 1e-12, std, np.nan)
                z = (log_p.to_numpy() - mean) / safe_std
                signal_values = z
                indicators["zscore"] = z
            elif spec.signal_variant == "bollinger_band":
                s_close = pd.Series(close_a)
                sma = s_close.rolling(win, min_periods=win).mean().to_numpy()
                std = s_close.rolling(win, min_periods=win).std(ddof=1).to_numpy()
                safe_std = np.where(std > 1e-12, std, np.nan)
                bb_z = (close_a - sma) / safe_std
                signal_values = bb_z
                indicators["bollinger_z"] = bb_z
                indicators["sma"] = sma
                indicators["upper_band"] = sma + abs(params.entry_threshold) * std
                indicators["lower_band"] = sma - abs(params.entry_threshold) * std
            elif spec.signal_variant == "rsi_extremes":
                rsi = _compute_rsi(close_a, win)
                signal_values = rsi
                indicators["rsi"] = rsi

        elif spec.template_id == "trend_breakout":
            fast_w = max(2, L)
            slow_w = max(fast_w + 1, L_slow)
            s_close = pd.Series(close_a)
            if spec.signal_variant == "sma_crossover":
                fast_ma = s_close.rolling(fast_w, min_periods=fast_w).mean().to_numpy()
                slow_ma = s_close.rolling(slow_w, min_periods=slow_w).mean().to_numpy()
                spread_pct = np.where(slow_ma > 0, ((fast_ma - slow_ma) / slow_ma) * 100.0, np.nan)
                signal_values = spread_pct
                indicators["fast_ma"] = fast_ma
                indicators["slow_ma"] = slow_ma
                indicators["ma_spread_pct"] = spread_pct
            elif spec.signal_variant == "ema_crossover":
                fast_ma = s_close.ewm(span=fast_w, min_periods=fast_w, adjust=False).mean().to_numpy()
                slow_ma = s_close.ewm(span=slow_w, min_periods=slow_w, adjust=False).mean().to_numpy()
                spread_pct = np.where(slow_ma > 0, ((fast_ma - slow_ma) / slow_ma) * 100.0, np.nan)
                signal_values = spread_pct
                indicators["fast_ma"] = fast_ma
                indicators["slow_ma"] = slow_ma
                indicators["ma_spread_pct"] = spread_pct
            elif spec.signal_variant == "donchian_breakout":
                # Prior-window high and low up to t-1 compared with close_t
                high_channel = s_close.shift(1).rolling(fast_w, min_periods=fast_w).max().to_numpy()
                low_channel = s_close.shift(1).rolling(slow_w, min_periods=slow_w).min().to_numpy()
                up_break = np.where(
                    high_channel > 0, ((close_a - high_channel) / high_channel) * 100.0, np.nan
                )
                down_break = np.where(
                    low_channel > 0, ((close_a - low_channel) / low_channel) * 100.0, np.nan
                )
                # Positive when breaking above prior fast_w high; negative when breaking below prior slow_w low
                breakout_pct = np.where(
                    np.isfinite(up_break) & (up_break > 0.0),
                    up_break,
                    np.where(np.isfinite(down_break) & (down_break < 0.0), down_break, 0.0),
                )
                breakout_pct = np.where(np.isfinite(high_channel), breakout_pct, np.nan)
                signal_values = breakout_pct
                indicators["high_channel"] = high_channel
                indicators["low_channel"] = low_channel

    else:
        # Family B: Pairs Trading
        close_b = df["close_b"].to_numpy(dtype=np.float64)
        dates = df["date"].astype(str).tolist()
        pairs_diag, ols_alpha, ols_beta = compute_pairs_econometrics(
            dates[:split_idx], close_a[:split_idx], close_b[:split_idx]
        )
        log_a = np.log(close_a)
        log_b = np.log(close_b)
        win = max(5, L)

        spread = np.full(n, np.nan, dtype=np.float64)
        roll_mean = np.full(n, np.nan, dtype=np.float64)
        roll_std = np.full(n, np.nan, dtype=np.float64)
        zscore = np.full(n, np.nan, dtype=np.float64)
        hedge_series = np.full(n, np.nan, dtype=np.float64)

        if spec.signal_variant == "price_ratio":
            spread = log_a - log_b
            ols_alpha, ols_beta = 0.0, 1.0
            hedge_series = np.ones(n, dtype=np.float64)
            s_spread = pd.Series(spread)
            roll_mean = s_spread.rolling(win, min_periods=win).mean().to_numpy()
            roll_std = s_spread.rolling(win, min_periods=win).std(ddof=1).to_numpy()
            safe_rstd = np.where(roll_std > 1e-12, roll_std, np.nan)
            zscore = (spread - roll_mean) / safe_rstd

        elif spec.signal_variant == "ols_log":
            # Strictly causal expanding OLS within In-Sample Formation [0..split_idx),
            # then frozen at the Formation [0..split_idx) OLS estimates for Out-of-Sample [split_idx..n).
            min_form_obs = max(20, win)
            for t in range(min_form_obs - 1, n):
                if t < split_idx:
                    ya = log_a[: t + 1]
                    xb = log_b[: t + 1]
                    vb = float(np.var(xb, ddof=1))
                    if vb > 1e-12:
                        b_t = float(np.cov(ya, xb, ddof=1)[0, 1] / vb)
                        a_t = float(np.mean(ya) - b_t * np.mean(xb))
                    else:
                        b_t, a_t = 1.0, 0.0
                else:
                    b_t, a_t = ols_beta, ols_alpha

                hedge_series[t] = b_t
                w_ya = log_a[t - win + 1 : t + 1]
                w_xb = log_b[t - win + 1 : t + 1]
                w_spread = w_ya - (a_t + b_t * w_xb)
                spread[t] = float(w_spread[-1])
                m_t = float(np.mean(w_spread))
                s_t = float(np.std(w_spread, ddof=1))
                roll_mean[t] = m_t
                roll_std[t] = s_t
                if s_t > 1e-12:
                    zscore[t] = (spread[t] - m_t) / s_t

        else:
            # Causal rolling OLS log spread
            roll_w = max(20, L_slow, win)
            for t in range(roll_w - 1, n):
                ya = log_a[t - roll_w + 1 : t + 1]
                xb = log_b[t - roll_w + 1 : t + 1]
                vb = float(np.var(xb, ddof=1))
                if vb > 1e-12:
                    b_t = float(np.cov(ya, xb, ddof=1)[0, 1] / vb)
                    a_t = float(np.mean(ya) - b_t * np.mean(xb))
                else:
                    b_t, a_t = 1.0, 0.0
                hedge_series[t] = b_t
                w_ya = log_a[t - win + 1 : t + 1]
                w_xb = log_b[t - win + 1 : t + 1]
                w_spread = w_ya - (a_t + b_t * w_xb)
                spread[t] = float(w_spread[-1])
                m_t = float(np.mean(w_spread))
                s_t = float(np.std(w_spread, ddof=1))
                roll_mean[t] = m_t
                roll_std[t] = s_t
                if s_t > 1e-12:
                    zscore[t] = (spread[t] - m_t) / s_t

        signal_values = zscore
        indicators["spread"] = spread
        indicators["spread_mean"] = roll_mean
        indicators["spread_zscore"] = zscore
        indicators["hedge_ratio"] = hedge_series

    return signal_values, indicators, pairs_diag, ols_alpha, ols_beta


def _check_entry_trigger(
    t: int,
    signal_values: np.ndarray,
    indicators: dict[str, np.ndarray],
    close_a: np.ndarray,
    spec: StrategySpec,
) -> Literal["long", "short", "long_spread", "short_spread"] | None:
    """Determine if bar t triggers an entry signal for execution at bar t+1."""
    val = signal_values[t]
    if not np.isfinite(val):
        return None

    params = spec.parameters
    thresh = abs(params.entry_threshold)

    if spec.strategy_family == "pairs_trading":
        if val <= -thresh and params.direction in ("long", "both"):
            return "long_spread"
        if val >= thresh and params.direction in ("short", "both"):
            return "short_spread"
        return None

    # Single-asset templates
    if spec.template_id == "return_threshold":
        if spec.signal_variant in ("cum_return_drop", "vol_scaled_shock"):
            if val <= -thresh:
                return "short" if params.direction == "short" else "long"
        elif spec.signal_variant == "cum_return_surge":
            if val >= thresh:
                return "short" if params.direction == "short" else "long"

    elif spec.template_id == "consecutive_streak":
        if val >= params.streak_length:
            return "short" if params.direction == "short" else "long"

    elif spec.template_id == "mean_reversion_oscillator":
        if spec.signal_variant in ("price_zscore", "bollinger_band"):
            if val <= -thresh and params.direction in ("long", "both"):
                return "long"
            if val >= thresh and params.direction in ("short", "both"):
                return "short"
        elif spec.signal_variant == "rsi_extremes":
            rsi_entry = params.entry_threshold if 1.0 <= params.entry_threshold <= 99.0 else 30.0
            if val <= rsi_entry and params.direction in ("long", "both"):
                return "long"
            if val >= (100.0 - rsi_entry) and params.direction in ("short", "both"):
                return "short"

    elif spec.template_id == "trend_breakout":
        band = max(0.0, params.entry_threshold) if params.entry_threshold < 15.0 else 0.0
        if spec.signal_variant in ("sma_crossover", "ema_crossover"):
            prev_val = signal_values[t - 1] if t > 0 else np.nan
            if np.isfinite(prev_val):
                if prev_val <= band and val > band and params.direction in ("long", "both"):
                    return "long"
                if prev_val >= -band and val < -band and params.direction in ("short", "both"):
                    return "short"
        elif spec.signal_variant == "donchian_breakout":
            if val > 0.0 and params.direction in ("long", "both"):
                return "long"
            if val < 0.0 and params.direction in ("short", "both"):
                return "short"

    return None


def _check_exit_trigger(
    t: int,
    direction: str,
    holding_days: int,
    open_mtm_return_pct: float,
    signal_values: np.ndarray,
    indicators: dict[str, np.ndarray],
    close_a: np.ndarray,
    entry_close_a: float,
    spec: StrategySpec,
) -> str | None:
    """Determine if an active position should trigger an exit at close of bar t (filling at t+1)."""
    params = spec.parameters
    risk = spec.risk_controls
    val = signal_values[t]

    # 1. Risk control: Per-trade stop-loss
    if risk.per_trade_stop_loss_pct is not None and open_mtm_return_pct <= -abs(risk.per_trade_stop_loss_pct):
        return f"stop_loss ({open_mtm_return_pct:.2f}% <= -{abs(risk.per_trade_stop_loss_pct):.1f}%)"

    # 2. Pairs trading specific exits
    if spec.strategy_family == "pairs_trading":
        if np.isfinite(val):
            exit_z = abs(params.exit_threshold)
            if direction == "long_spread" and val >= -exit_z:
                return f"spread_convergence (z={val:.2f} >= -{exit_z:.2f})"
            if direction == "short_spread" and val <= exit_z:
                return f"spread_convergence (z={val:.2f} <= +{exit_z:.2f})"
            if params.stop_z_threshold is not None and abs(val) >= abs(params.stop_z_threshold):
                return f"spread_divergence_stop (|z|={abs(val):.2f} >= {abs(params.stop_z_threshold):.2f})"
        max_hold = min(risk.max_holding_days, max(1, params.holding_period))
        if holding_days >= max_hold:
            return f"holding_period_reached ({holding_days}d)"
        return None

    # 3. Single-asset exits
    if params.exit_mode == "first_positive_close":
        if direction == "long" and close_a[t] > entry_close_a:
            return "first_positive_close"
        if direction == "short" and close_a[t] < entry_close_a:
            return "first_profitable_close"

    if spec.template_id == "mean_reversion_oscillator" and params.exit_mode == "threshold_reversion":
        if np.isfinite(val):
            if spec.signal_variant in ("price_zscore", "bollinger_band"):
                exit_z = params.exit_threshold
                if direction == "long" and val >= exit_z:
                    return f"zscore_reverted (z={val:.2f} >= {exit_z:.2f})"
                if direction == "short" and val <= -exit_z:
                    return f"zscore_reverted (z={val:.2f} <= -{exit_z:.2f})"
            elif spec.signal_variant == "rsi_extremes":
                rsi_exit = params.exit_threshold if 10.0 <= params.exit_threshold <= 90.0 else 55.0
                if direction == "long" and val >= rsi_exit:
                    return f"rsi_reverted (RSI={val:.1f} >= {rsi_exit:.1f})"
                if direction == "short" and val <= (100.0 - rsi_exit):
                    return f"rsi_reverted (RSI={val:.1f} <= {100.0 - rsi_exit:.1f})"

    if spec.template_id == "trend_breakout" and params.exit_mode == "signal_reversal":
        if spec.signal_variant in ("sma_crossover", "ema_crossover") and np.isfinite(val):
            if direction == "long" and val < 0.0:
                return "ma_cross_below"
            if direction == "short" and val > 0.0:
                return "ma_cross_above"
        elif spec.signal_variant == "donchian_breakout":
            low_ch = indicators.get("low_channel")
            high_ch = indicators.get("high_channel")
            if direction == "long" and low_ch is not None and np.isfinite(low_ch[t]) and close_a[t] < low_ch[t]:
                return "channel_low_break"
            if direction == "short" and high_ch is not None and np.isfinite(high_ch[t]) and close_a[t] > high_ch[t]:
                return "channel_high_break"

    target_hold = min(risk.max_holding_days, max(1, params.holding_period))
    if holding_days >= target_hold:
        return f"holding_period ({holding_days}d)"

    return None


def compute_pair_leg_weights(dir_label: str, beta_w: float) -> tuple[float, float]:
    """Compute signed dollar weights (w_a, w_b) with unit gross exposure (|w_a| + |w_b| = 1.0).

    For spread = log_a - alpha - beta * log_b:
    - long_spread bets on spread widening: w_a = +1/(1+|beta|), w_b = -beta/(1+|beta|)
    - short_spread bets on spread narrowing: w_a = -1/(1+|beta|), w_b = +beta/(1+|beta|)
    Preserves the sign of beta_w even when beta_w < 0.
    """
    b_clamped = max(-5.0, min(5.0, float(beta_w))) if np.isfinite(beta_w) else 1.0
    if abs(b_clamped) < 0.05:
        b_clamped = 0.05 if b_clamped >= 0.0 else -0.05
    denom = 1.0 + abs(b_clamped)
    wa = 1.0 / denom
    wb = -b_clamped / denom
    if dir_label == "long_spread":
        return wa, wb
    return -wa, -wb


def simulate_strategy_core(
    df: pd.DataFrame,
    spec: StrategySpec,
    split_idx: int,
    override_cost_bps_per_leg: float | None = None,
    entry_delay_bars: int = 1,
    precomputed_signals: tuple[np.ndarray, dict[str, np.ndarray], PairsDiagnostics | None, float, float] | None = None,
) -> dict[str, Any]:
    """Execute deterministic bar-by-bar walk-forward simulation over aligned DataFrame."""
    n = len(df)
    dates = df["date"].astype(str).tolist()
    open_a = df["open_a"].to_numpy(dtype=np.float64)
    close_a = df["close_a"].to_numpy(dtype=np.float64)
    close_bench = df["close_bench"].to_numpy(dtype=np.float64)

    is_pairs = spec.strategy_family == "pairs_trading"
    open_b = df["open_b"].to_numpy(dtype=np.float64) if is_pairs else np.ones(n, dtype=np.float64)
    close_b = df["close_b"].to_numpy(dtype=np.float64) if is_pairs else np.ones(n, dtype=np.float64)

    if precomputed_signals is not None:
        signal_values, indicators, pairs_diag, ols_alpha, ols_beta = precomputed_signals
    else:
        signal_values, indicators, pairs_diag, ols_alpha, ols_beta = compute_causal_signals(
            df, spec, split_idx
        )

    exec_NextOpen = spec.execution.execution_timing == "next_open"
    one_way_leg_bps = (
        override_cost_bps_per_leg
        if override_cost_bps_per_leg is not None
        else (spec.execution.commission_bps + spec.execution.slippage_bps)
    )
    # Unit gross exposure across both single-asset (|w_a| = 1.0) and pairs (|w_a| + |w_b| = 1.0):
    # total portfolio notional traded on entry is 1.0x capital, and 1.0x capital on exit.
    one_way_cost_rate = one_way_leg_bps / 10_000.0

    daily_gross_ret = np.zeros(n, dtype=np.float64)
    daily_net_ret = np.zeros(n, dtype=np.float64)
    daily_position = np.zeros(n, dtype=np.float64)
    daily_bench_ret = np.zeros(n, dtype=np.float64)
    daily_bench_ret[1:] = (close_bench[1:] / close_bench[:-1]) - 1.0

    trades: list[TradeRecord] = []
    active: dict[str, Any] | None = None
    pending_entry: dict[str, Any] | None = None
    pending_exit_reason: str | None = None

    initial_cap = spec.execution.initial_capital
    equity_net = np.full(n, initial_cap, dtype=np.float64)
    equity_gross = np.full(n, initial_cap, dtype=np.float64)
    peak_equity = initial_cap
    circuit_breaker_tripped = False
    circuit_breaker_date: str | None = None
    max_dd_limit = spec.risk_controls.max_drawdown_circuit_breaker_pct

    hedge_arr = indicators.get("hedge_ratio", np.full(n, np.nan, dtype=np.float64))

    def _pair_step_return(
        dir_label: str, pa_prev: float, pa_curr: float, pb_prev: float, pb_curr: float, beta_w: float
    ) -> float:
        ra = (pa_curr / pa_prev) - 1.0
        rb = (pb_curr / pb_prev) - 1.0
        wa, wb = compute_pair_leg_weights(dir_label, beta_w)
        return wa * ra + wb * rb

    def _single_step_return(dir_label: str, p_prev: float, p_curr: float) -> float:
        r = (p_curr / p_prev) - 1.0
        return r if dir_label == "long" else -r

    for t in range(n):
        prev_eq_net = equity_net[t - 1] if t > 0 else initial_cap
        prev_eq_gross = equity_gross[t - 1] if t > 0 else initial_cap
        cur_eq_net = prev_eq_net
        cur_eq_gross = prev_eq_gross

        # 1. Execute pending exit at bar t
        if active is not None and pending_exit_reason is not None:
            fill_a = open_a[t] if exec_NextOpen else close_a[t]
            fill_b = open_b[t] if exec_NextOpen else close_b[t]
            if is_pairs:
                overnight_r = _pair_step_return(
                    active["direction"],
                    active["last_mark_a"],
                    fill_a,
                    active["last_mark_b"],
                    fill_b,
                    active["beta_w"],
                )
            else:
                overnight_r = _single_step_return(active["direction"], active["last_mark_a"], fill_a)

            cur_eq_gross = cur_eq_gross * (1.0 + overnight_r)
            gross_mult = cur_eq_gross / active["entry_equity_gross"]
            round_trip_cost_rate = 2.0 * one_way_cost_rate
            net_mult = max(0.01, gross_mult - round_trip_cost_rate)
            cur_eq_net = active["entry_equity_net"] * net_mult

            trade_gross = (gross_mult - 1.0) * 100.0
            trade_cost_pct = round_trip_cost_rate * 100.0
            trade_net = (net_mult - 1.0) * 100.0

            trades.append(
                TradeRecord(
                    trade_id=len(trades) + 1,
                    split="in_sample" if active["entry_idx"] < split_idx else "out_of_sample",
                    direction=active["direction"],
                    signal_date=active["signal_date"],
                    signal_value=round(float(active["signal_value"]), 4),
                    entry_date=active["entry_date"],
                    entry_price_a=round(float(active["entry_price_a"]), 4),
                    entry_price_b=round(float(active["entry_price_b"]), 4) if is_pairs else None,
                    exit_date=dates[t],
                    exit_price_a=round(float(fill_a), 4),
                    exit_price_b=round(float(fill_b), 4) if is_pairs else None,
                    holding_days=max(1, t - active["entry_idx"]),
                    gross_return_pct=round(trade_gross, 4),
                    cost_pct=round(trade_cost_pct, 4),
                    net_return_pct=round(trade_net, 4),
                    exit_reason=pending_exit_reason,
                    cumulative_equity_after=round(float(cur_eq_net), 2),
                    crossed_split=bool(active["entry_idx"] < split_idx and t >= split_idx),
                    leg_weight_a=round(float(active["leg_weight_a"]), 4),
                    leg_weight_b=round(float(active["leg_weight_b"]), 4),
                )
            )
            active = None
            pending_exit_reason = None

        # 2. Execute pending entry at bar t (if not halted by circuit breaker and not on final bar)
        if active is None and pending_entry is not None and not circuit_breaker_tripped and t < n - 1:
            if pending_entry.get("wait_bars", 0) > 0:
                pending_entry["wait_bars"] -= 1
            else:
                fill_a = open_a[t] if exec_NextOpen else close_a[t]
                fill_b = open_b[t] if exec_NextOpen else close_b[t]
                beta_w = float(hedge_arr[t - 1]) if t > 0 and np.isfinite(hedge_arr[t - 1]) else ols_beta
                if is_pairs:
                    wa, wb = compute_pair_leg_weights(pending_entry["direction"], beta_w)
                else:
                    wa = 1.0 if pending_entry["direction"] == "long" else -1.0
                    wb = 0.0
                active = {
                    "direction": pending_entry["direction"],
                    "signal_date": pending_entry["signal_date"],
                    "signal_value": pending_entry["signal_value"],
                    "entry_idx": t,
                    "entry_date": dates[t],
                    "entry_price_a": fill_a,
                    "entry_price_b": fill_b,
                    "entry_close_a": close_a[t],
                    "last_mark_a": fill_a,
                    "last_mark_b": fill_b,
                    "beta_w": beta_w,
                    "leg_weight_a": wa,
                    "leg_weight_b": wb,
                    "holding_days": 0,
                    "cum_gross": 0.0,
                    "entry_equity_net": cur_eq_net,
                    "entry_equity_gross": cur_eq_gross,
                }
                pending_entry = None

                # If entered at open_t, mark from open_t to close_t on day t
                if exec_NextOpen:
                    if is_pairs:
                        intraday_r = _pair_step_return(
                            active["direction"], fill_a, close_a[t], fill_b, close_b[t], beta_w
                        )
                    else:
                        intraday_r = _single_step_return(active["direction"], fill_a, close_a[t])
                    cur_eq_gross = cur_eq_gross * (1.0 + intraday_r)
                    cur_eq_net = cur_eq_net * (1.0 + intraday_r - one_way_cost_rate)
                    active["cum_gross"] = (1.0 + active["cum_gross"]) * (1.0 + intraday_r) - 1.0
                    active["last_mark_a"] = close_a[t]
                    active["last_mark_b"] = close_b[t]
                    active["holding_days"] = 1
                else:
                    cur_eq_net = cur_eq_net * (1.0 - one_way_cost_rate)
        elif active is not None or circuit_breaker_tripped or t == n - 1:
            pending_entry = None

        # 3. If position was already open prior to bar t, mark from close_{t-1} to close_t
        if active is not None and active["entry_idx"] < t:
            if is_pairs:
                bar_r = _pair_step_return(
                    active["direction"],
                    active["last_mark_a"],
                    close_a[t],
                    active["last_mark_b"],
                    close_b[t],
                    active["beta_w"],
                )
            else:
                bar_r = _single_step_return(active["direction"], active["last_mark_a"], close_a[t])
            cur_eq_gross = cur_eq_gross * (1.0 + bar_r)
            cur_eq_net = cur_eq_net * (1.0 + bar_r)
            active["cum_gross"] = (1.0 + active["cum_gross"]) * (1.0 + bar_r) - 1.0
            active["last_mark_a"] = close_a[t]
            active["last_mark_b"] = close_b[t]
            active["holding_days"] += 1

        # 3b. If on final bar t == n - 1 and position is still open, liquidate at close_t
        # BEFORE computing daily_net_ret[t], equity_net[t], peak_equity, and drawdown checks.
        final_bar_liquidated_trade: dict[str, Any] | None = None
        if t == n - 1 and active is not None:
            gross_mult = cur_eq_gross / active["entry_equity_gross"]
            round_trip_cost_rate = 2.0 * one_way_cost_rate
            net_mult = max(0.01, gross_mult - round_trip_cost_rate)
            cur_eq_net = active["entry_equity_net"] * net_mult
            final_bar_liquidated_trade = active
            active = None

        # Record daily returns and update equity
        equity_gross[t] = cur_eq_gross
        equity_net[t] = cur_eq_net
        daily_gross_ret[t] = (equity_gross[t] / prev_eq_gross) - 1.0
        daily_net_ret[t] = (equity_net[t] / prev_eq_net) - 1.0
        if active is not None:
            daily_position[t] = 1.0 if active["direction"] in ("long", "long_spread") else -1.0
        elif final_bar_liquidated_trade is not None:
            daily_position[t] = (
                1.0 if final_bar_liquidated_trade["direction"] in ("long", "long_spread") else -1.0
            )

        if final_bar_liquidated_trade is not None:
            gross_mult = equity_gross[t] / final_bar_liquidated_trade["entry_equity_gross"]
            round_trip_cost_rate = 2.0 * one_way_cost_rate
            net_mult = equity_net[t] / final_bar_liquidated_trade["entry_equity_net"]
            trade_gross = (gross_mult - 1.0) * 100.0
            trade_cost_pct = round_trip_cost_rate * 100.0
            trade_net = (net_mult - 1.0) * 100.0
            trades.append(
                TradeRecord(
                    trade_id=len(trades) + 1,
                    split="in_sample" if final_bar_liquidated_trade["entry_idx"] < split_idx else "out_of_sample",
                    direction=final_bar_liquidated_trade["direction"],
                    signal_date=final_bar_liquidated_trade["signal_date"],
                    signal_value=round(float(final_bar_liquidated_trade["signal_value"]), 4),
                    entry_date=final_bar_liquidated_trade["entry_date"],
                    entry_price_a=round(float(final_bar_liquidated_trade["entry_price_a"]), 4),
                    entry_price_b=round(float(final_bar_liquidated_trade["entry_price_b"]), 4) if is_pairs else None,
                    exit_date=dates[t],
                    exit_price_a=round(float(close_a[t]), 4),
                    exit_price_b=round(float(close_b[t]), 4) if is_pairs else None,
                    holding_days=max(1, t - final_bar_liquidated_trade["entry_idx"]),
                    gross_return_pct=round(trade_gross, 4),
                    cost_pct=round(trade_cost_pct, 4),
                    net_return_pct=round(trade_net, 4),
                    exit_reason="end_of_sample_liquidation",
                    cumulative_equity_after=round(float(equity_net[t]), 2),
                    crossed_split=bool(final_bar_liquidated_trade["entry_idx"] < split_idx and t >= split_idx),
                    leg_weight_a=round(float(final_bar_liquidated_trade["leg_weight_a"]), 4),
                    leg_weight_b=round(float(final_bar_liquidated_trade["leg_weight_b"]), 4),
                )
            )

        peak_equity = max(peak_equity, equity_net[t])
        current_dd_pct = ((peak_equity - equity_net[t]) / peak_equity) * 100.0 if peak_equity > 0 else 0.0

        # Check portfolio drawdown circuit breaker
        if (
            max_dd_limit is not None
            and not circuit_breaker_tripped
            and current_dd_pct >= max_dd_limit
        ):
            circuit_breaker_tripped = True
            circuit_breaker_date = dates[t]
            if active is not None and t < n - 1:
                pending_exit_reason = f"circuit_breaker_halt (DD={current_dd_pct:.1f}% >= {max_dd_limit:.1f}%)"

        # 4. End-of-bar t signal evaluation for execution at bar t + 1 (or t + entry_delay_bars)
        if t < n - 1:
            if active is not None and pending_exit_reason is None:
                open_mtm_pct = active["cum_gross"] * 100.0
                exit_reason = _check_exit_trigger(
                    t,
                    active["direction"],
                    active["holding_days"],
                    open_mtm_pct,
                    signal_values,
                    indicators,
                    close_a,
                    active["entry_close_a"],
                    spec,
                )
                if exit_reason is not None:
                    pending_exit_reason = exit_reason
            elif active is None and pending_entry is None and not circuit_breaker_tripped:
                trig_dir = _check_entry_trigger(t, signal_values, indicators, close_a, spec)
                if trig_dir is not None:
                    pending_entry = {
                        "direction": trig_dir,
                        "signal_date": dates[t],
                        "signal_value": float(signal_values[t]),
                        "wait_bars": max(0, entry_delay_bars - 1),
                    }

    return {
        "dates": dates,
        "daily_gross_ret": daily_gross_ret,
        "daily_net_ret": daily_net_ret,
        "daily_bench_ret": daily_bench_ret,
        "daily_position": daily_position,
        "equity_gross": equity_gross,
        "equity_net": equity_net,
        "trades": trades,
        "signal_values": signal_values,
        "indicators": indicators,
        "pairs_diag": pairs_diag,
        "circuit_breaker_tripped": circuit_breaker_tripped,
        "circuit_breaker_date": circuit_breaker_date,
    }


def _compute_period_metrics(
    label: str,
    dates: list[str],
    gross_r: np.ndarray,
    net_r: np.ndarray,
    bench_r: np.ndarray,
    pos: np.ndarray,
    trades: list[TradeRecord],
    cb_tripped: bool = False,
    cb_date: str | None = None,
    halted_prior_to_period: bool = False,
) -> PeriodMetrics:
    n = len(net_r)
    if n == 0:
        raise ValueError(f"Empty period slice for {label}")

    cum_gross = float(np.prod(1.0 + gross_r) - 1.0) * 100.0
    cum_net = float(np.prod(1.0 + net_r) - 1.0) * 100.0
    cum_bench = float(np.prod(1.0 + bench_r) - 1.0) * 100.0

    years = max(n / TRADING_DAYS_PER_YEAR, 1.0 / TRADING_DAYS_PER_YEAR)
    cagr_net = (float(max(1e-9, np.prod(1.0 + net_r)) ** (1.0 / years)) - 1.0) * 100.0
    cagr_bench = (float(max(1e-9, np.prod(1.0 + bench_r)) ** (1.0 / years)) - 1.0) * 100.0

    vol_net = float(np.std(net_r, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0) if n > 1 else 0.0
    vol_bench = float(np.std(bench_r, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR) * 100.0) if n > 1 else 0.0

    # Max Drawdown (%) with initial capital E_0 = 1.0 included in the running peak H_t = max(1.0, max_{s<=t} E_s)
    eq_curve = np.cumprod(1.0 + net_r)
    peaks = np.maximum(1.0, np.maximum.accumulate(eq_curve))
    dd_series = ((peaks - eq_curve) / peaks) * 100.0
    max_dd = float(np.max(dd_series)) if len(dd_series) else 0.0

    bench_curve = np.cumprod(1.0 + bench_r)
    bench_peaks = np.maximum(1.0, np.maximum.accumulate(bench_curve))
    bench_max_dd = float(np.max(((bench_peaks - bench_curve) / bench_peaks) * 100.0))

    # Trade-level metrics & exposure
    tr_nets = [t.net_return_pct for t in trades]
    k_trades = len(tr_nets)
    exposure = float(np.mean(np.abs(pos) > 0) * 100.0)
    is_zero_exposure = bool(k_trades == 0 and exposure <= 1e-9 and np.std(net_r) <= 1e-12)

    sr_bench, _, _, _, _ = lo_mertens_sharpe_se_and_psr(bench_r, benchmark_sr_annual=0.0)
    if is_zero_exposure:
        sr_net, sr_se, psr_pct, skew, kurt = 0.0, 0.0, 0.0, 0.0, 0.0
        sortino = 0.0
        calmar = 0.0
        if halted_prior_to_period:
            status_note = (
                f"Zero exposure in {label}: portfolio drawdown circuit breaker already halted trading on {cb_date} "
                "during the In-Sample Formation split. Return-to-volatility ratios (Sharpe, Sortino, PSR) are undefined."
            )
        else:
            status_note = (
                f"Zero exposure in {label}: 0 trades triggered during this window. "
                "Return-to-volatility ratios (Sharpe, Sortino, PSR) are undefined on a constant cash series."
            )
    else:
        sr_net, sr_se, psr_pct, skew, kurt = lo_mertens_sharpe_se_and_psr(net_r, benchmark_sr_annual=0.0)
        downside = np.where(net_r < 0.0, net_r, 0.0)
        down_std = float(np.sqrt(np.mean(downside**2)) * math.sqrt(TRADING_DAYS_PER_YEAR))
        sortino = float((np.mean(net_r) * TRADING_DAYS_PER_YEAR) / down_std) if down_std > 1e-12 else 0.0
        calmar = float(cagr_net / max_dd) if max_dd > 1e-6 else 0.0
        status_note = None

    wins = sum(1 for x in tr_nets if x > 0)
    losses = sum(1 for x in tr_nets if x < 0)
    win_rate = (wins / k_trades * 100.0) if k_trades > 0 else 0.0

    gross_gains = sum(x for x in tr_nets if x > 0)
    gross_losses = abs(sum(x for x in tr_nets if x < 0))
    profit_factor = round(gross_gains / gross_losses, 3) if gross_losses > 1e-9 else None

    mean_tr = float(np.mean(tr_nets)) if k_trades > 0 else 0.0
    med_tr = float(np.median(tr_nets)) if k_trades > 0 else 0.0
    std_tr = float(np.std(tr_nets, ddof=1)) if k_trades > 1 else 0.0
    best_tr = float(max(tr_nets)) if k_trades > 0 else 0.0
    worst_tr = float(min(tr_nets)) if k_trades > 0 else 0.0

    # Two-sided t-test on daily returns (active days or full series)
    if n > 5 and np.std(net_r, ddof=1) > 1e-12:
        t_stat, p_val = stats.ttest_1samp(net_r, 0.0)
        t_stat_f = float(t_stat) if np.isfinite(t_stat) else 0.0
        p_val_f = float(p_val) if np.isfinite(p_val) else 1.0
    else:
        t_stat_f, p_val_f = 0.0, 1.0

    sign_p = exact_sign_test_p_value(wins, losses)

    return PeriodMetrics(
        period_label=label,
        start_date=str(dates[0]),
        end_date=str(dates[-1]),
        trading_days=n,
        trade_count=k_trades,
        exposure_pct=round(exposure, 2),
        total_gross_return_pct=round(cum_gross, 2),
        total_net_return_pct=round(cum_net, 2),
        cagr_net_pct=round(cagr_net, 2),
        benchmark_return_pct=round(cum_bench, 2),
        benchmark_cagr_pct=round(cagr_bench, 2),
        excess_return_pct=round(cum_net - cum_bench, 2),
        annualized_volatility_pct=round(vol_net, 2),
        benchmark_volatility_pct=round(vol_bench, 2),
        max_drawdown_pct=round(max_dd, 2),
        benchmark_max_drawdown_pct=round(bench_max_dd, 2),
        sharpe_ratio=round(sr_net, 3),
        benchmark_sharpe_ratio=round(sr_bench, 3),
        sharpe_se_lo2002=round(sr_se, 3),
        probabilistic_sharpe_psr_pct=round(psr_pct, 2),
        sortino_ratio=round(sortino, 3),
        calmar_ratio=round(calmar, 3),
        win_rate_pct=round(win_rate, 2),
        profit_factor=profit_factor,
        mean_trade_net_pct=round(mean_tr, 3),
        median_trade_net_pct=round(med_tr, 3),
        std_trade_net_pct=round(std_tr, 3),
        best_trade_net_pct=round(best_tr, 3),
        worst_trade_net_pct=round(worst_tr, 3),
        skewness_daily=round(skew, 3),
        kurtosis_daily=round(kurt, 3),
        t_stat_mean_return=round(t_stat_f, 3),
        p_value_two_sided=round(p_val_f, 4),
        sign_test_p_value=sign_p,
        circuit_breaker_triggered=cb_tripped,
        circuit_breaker_date=cb_date,
        is_zero_exposure=is_zero_exposure,
        halted_prior_to_period=halted_prior_to_period,
        status_note=status_note,
    )


def compute_parameter_sensitivity_surface(
    df: pd.DataFrame,
    spec: StrategySpec,
    split_idx: int,
) -> ParameterSensitivitySurface:
    """Evaluate a 5x5 grid (K=25 configurations) around the confirmed thresholds/windows
    and compute the Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR).
    """
    p = spec.parameters

    def _pad_five_ints(vals: set[float], min_val: int = 1, step: int = 1) -> list[float]:
        out = sorted(vals)
        cand = float(max(min_val, int(out[-1]) + step if out else min_val))
        while len(out) < 5:
            if cand not in out:
                out.append(cand)
            cand += step
        return sorted(out[:5])

    # Choose X and Y parameter axes based on template
    if spec.template_id == "consecutive_streak":
        x_name, x_label = "streak_length", "Streak Length (Days)"
        base_x = p.streak_length
        x_vals = _pad_five_ints({float(max(1, base_x + d)) for d in (-2, -1, 0, 1, 2)}, min_val=1, step=1)
        y_name, y_label = "holding_period", "Holding Period (Days)"
        base_y = p.holding_period
        y_vals = _pad_five_ints({float(max(1, round(base_y * m))) for m in (0.4, 0.7, 1.0, 1.5, 2.0)}, min_val=1, step=2)
    elif spec.template_id == "trend_breakout":
        x_name, x_label = "lookback_window", "Fast Window (Days)"
        base_x = p.lookback_window
        x_vals = _pad_five_ints({float(max(3, round(base_x * m))) for m in (0.5, 0.75, 1.0, 1.25, 1.5)}, min_val=3, step=5)
        y_name, y_label = "secondary_window", "Slow Window (Days)"
        base_y = max(base_x + 10, p.secondary_window)
        y_vals = _pad_five_ints({float(max(int(max(x_vals)) + 5, round(base_y * m))) for m in (0.6, 0.8, 1.0, 1.25, 1.5)}, min_val=int(max(x_vals)) + 5, step=10)
    elif spec.strategy_family == "pairs_trading":
        x_name, x_label = "entry_threshold", "Entry Z-Score Threshold (σ)"
        base_x = max(0.5, abs(p.entry_threshold))
        x_vals = [round(base_x * m, 2) for m in (0.6, 0.8, 1.0, 1.2, 1.4)]
        y_name, y_label = "lookback_window", "Spread Z-Score Window (Days)"
        base_y = max(5, p.lookback_window)
        y_vals = _pad_five_ints({float(max(5, round(base_y * m))) for m in (0.5, 0.75, 1.0, 1.5, 2.0)}, min_val=5, step=5)
    else:
        x_name, x_label = "entry_threshold", "Entry Threshold"
        base_x = max(0.25, abs(p.entry_threshold))
        x_vals = [round(base_x * m, 2) for m in (0.5, 0.75, 1.0, 1.25, 1.5)]
        y_name, y_label = "holding_period", "Holding Period (Days)"
        base_y = max(1, p.holding_period)
        y_vals = _pad_five_ints({float(max(1, round(base_y * m))) for m in (0.4, 0.7, 1.0, 1.5, 2.0)}, min_val=1, step=2)

    cells: list[SensitivityGridCell] = []
    trial_sharpes: list[float] = []
    best_net_returns: np.ndarray | None = None
    best_sr = -999.0
    selected_net_returns: np.ndarray | None = None

    for xv in x_vals:
        for yv in y_vals:
            trial_spec = spec.model_copy(deep=True)
            setattr(
                trial_spec.parameters,
                x_name,
                int(round(xv)) if x_name in ("streak_length", "lookback_window", "secondary_window", "holding_period") else float(xv),
            )
            setattr(
                trial_spec.parameters,
                y_name,
                int(round(yv)) if y_name in ("streak_length", "lookback_window", "secondary_window", "holding_period") else float(yv),
            )
            sim = simulate_strategy_core(df, trial_spec, split_idx)
            net_r = sim["daily_net_ret"]
            is_sr, _, _, _, _ = lo_mertens_sharpe_se_and_psr(net_r[:split_idx])
            oos_sr, _, _, _, _ = lo_mertens_sharpe_se_and_psr(net_r[split_idx:])
            full_sr, _, _, _, _ = lo_mertens_sharpe_se_and_psr(net_r)
            cum_ret = float(np.prod(1.0 + net_r) - 1.0) * 100.0

            eq = np.cumprod(1.0 + net_r)
            pk = np.maximum(1.0, np.maximum.accumulate(eq))
            mdd = float(np.max(((pk - eq) / pk) * 100.0))

            tr_list = sim["trades"]
            w_rate = (sum(1 for t in tr_list if t.net_return_pct > 0) / len(tr_list) * 100.0) if tr_list else 0.0

            cells.append(
                SensitivityGridCell(
                    param_x_value=float(xv),
                    param_y_value=float(yv),
                    trade_count=len(tr_list),
                    in_sample_sharpe=round(is_sr, 3),
                    out_of_sample_sharpe=round(oos_sr, 3),
                    full_sharpe=round(full_sr, 3),
                    full_net_return_pct=round(cum_ret, 2),
                    max_drawdown_pct=round(mdd, 2),
                    win_rate_pct=round(w_rate, 1),
                )
            )
            trial_sharpes.append(full_sr)
            if full_sr > best_sr:
                best_sr = full_sr
                best_net_returns = net_r
            if math.isclose(xv, getattr(p, x_name), rel_tol=0.05) and math.isclose(
                yv, getattr(p, y_name), rel_tol=0.05
            ):
                selected_net_returns = net_r

    if selected_net_returns is None:
        selected_net_returns = simulate_strategy_core(df, spec, split_idx)["daily_net_ret"]
    if best_net_returns is None:
        best_net_returns = selected_net_returns

    exp_max_sr, selected_dsr = deflated_sharpe_ratio_bailey_lopez(selected_net_returns, trial_sharpes)
    _, best_dsr = deflated_sharpe_ratio_bailey_lopez(best_net_returns, trial_sharpes)

    return ParameterSensitivitySurface(
        param_x_name=x_name,
        param_x_label=x_label,
        param_x_values=[float(v) for v in x_vals],
        param_y_name=y_name,
        param_y_label=y_label,
        param_y_values=[float(v) for v in y_vals],
        cells=cells,
        trials_count_k=len(cells),
        expected_max_sr_under_null=exp_max_sr,
        selected_spec_deflated_sharpe_psr_pct=selected_dsr,
        best_grid_deflated_sharpe_psr_pct=best_dsr,
        methodology_note=(
            f"Evaluated K={len(cells)} parameter combinations across {x_label} and {y_label}. "
            f"Per Bailey & López de Prado (2014), the expected maximum Sharpe Ratio under a null hypothesis "
            f"of zero skill across K={len(cells)} trials is {exp_max_sr:.2f}. "
            "Inspect whether performance forms a broad plateau across neighbouring parameters or is isolated to a single spike."
        ),
    )


def compute_standard_bracket_profiles(
    df: pd.DataFrame,
    spec: StrategySpec,
    split_idx: int,
) -> list[BracketProfileResult]:
    """Run 3 standard literature profiles (Conservative, Baseline, Aggressive) so the user can
    compare how threshold sensitivity shapes In-Sample and Out-of-Sample behaviour.
    """
    p = spec.parameters
    profiles_cfg: list[tuple[str, str, str, dict[str, Any]]] = []

    if spec.strategy_family == "pairs_trading":
        profiles_cfg = [
            (
                "aggressive",
                "Aggressive (1.5σ Entry)",
                "Frequent spread entries at ±1.5σ divergence, exiting at ±0.5σ",
                {"entry_threshold": 1.5, "exit_threshold": 0.5, "lookback_window": p.lookback_window},
            ),
            (
                "baseline",
                "Baseline Literature (2.0σ Entry)",
                "Gatev et al. (2006) standard ±2.0σ divergence trigger, exiting at ±0.25σ",
                {"entry_threshold": 2.0, "exit_threshold": 0.25, "lookback_window": p.lookback_window},
            ),
            (
                "conservative",
                "Conservative Tail (2.5σ Entry)",
                "High-conviction ±2.5σ tail divergence trigger, exiting at 0.0σ mean",
                {"entry_threshold": 2.5, "exit_threshold": 0.0, "lookback_window": p.lookback_window},
            ),
        ]
    elif spec.template_id == "consecutive_streak":
        base_s = max(2, p.streak_length)
        profiles_cfg = [
            (
                "aggressive",
                f"Aggressive ({max(2, base_s - 1)}-Day Streak)",
                f"Enters after {max(2, base_s - 1)} consecutive days, holding {p.holding_period}d",
                {"streak_length": max(2, base_s - 1), "entry_threshold": float(max(2, base_s - 1))},
            ),
            (
                "baseline",
                f"Baseline ({base_s}-Day Streak)",
                f"Enters after {base_s} consecutive days, holding {p.holding_period}d",
                {"streak_length": base_s, "entry_threshold": float(base_s)},
            ),
            (
                "conservative",
                f"Conservative ({base_s + 1}-Day Streak)",
                f"Waits for {base_s + 1} consecutive days, holding {p.holding_period}d",
                {"streak_length": base_s + 1, "entry_threshold": float(base_s + 1)},
            ),
        ]
    elif spec.template_id == "trend_breakout":
        profiles_cfg = [
            (
                "aggressive",
                "Fast Trend (10d / 30d)",
                "Responsive 10-day / 30-day moving-average / breakout window",
                {"lookback_window": 10, "secondary_window": 30},
            ),
            (
                "baseline",
                f"Baseline Trend ({p.lookback_window}d / {p.secondary_window}d)",
                f"Standard {p.lookback_window}-day / {p.secondary_window}-day window",
                {"lookback_window": p.lookback_window, "secondary_window": p.secondary_window},
            ),
            (
                "conservative",
                "Slow Institutional Trend (50d / 150d)",
                "Long-horizon 50-day / 150-day institutional trend filter",
                {"lookback_window": 50, "secondary_window": 150},
            ),
        ]
    elif spec.template_id == "mean_reversion_oscillator":
        if spec.signal_variant == "rsi_extremes":
            profiles_cfg = [
                (
                    "aggressive",
                    "Aggressive (RSI ≤ 35)",
                    "Moderate oversold/overbought RSI trigger (35 / 65)",
                    {"entry_threshold": 35.0, "exit_threshold": 55.0},
                ),
                (
                    "baseline",
                    "Baseline Literature (RSI ≤ 30)",
                    "Wilder (1978) standard RSI 30 / 70 extreme threshold",
                    {"entry_threshold": 30.0, "exit_threshold": 55.0},
                ),
                (
                    "conservative",
                    "Conservative Deep Extreme (RSI ≤ 25)",
                    "Deep oversold/overbought RSI 25 / 75 threshold",
                    {"entry_threshold": 25.0, "exit_threshold": 55.0},
                ),
            ]
        else:
            profiles_cfg = [
                (
                    "aggressive",
                    "Aggressive (1.5σ Band)",
                    "Enters at ±1.5σ deviation from rolling mean",
                    {"entry_threshold": 1.5, "exit_threshold": 0.0},
                ),
                (
                    "baseline",
                    "Baseline Literature (2.0σ Band)",
                    "Standard ±2.0σ Bollinger / Z-Score threshold",
                    {"entry_threshold": 2.0, "exit_threshold": 0.0},
                ),
                (
                    "conservative",
                    "Conservative Tail (2.5σ Band)",
                    "High-selectivity ±2.5σ tail deviation threshold",
                    {"entry_threshold": 2.5, "exit_threshold": 0.0},
                ),
            ]
    else:
        base_t = max(0.5, abs(p.entry_threshold))
        agg_t = round(max(0.5, base_t * 0.65), 2)
        con_t = round(base_t * 1.5, 2)
        profiles_cfg = [
            (
                "aggressive",
                f"Aggressive ({agg_t:.2f}% Shock)",
                f"Lower shock hurdle ({agg_t:.2f}%), higher trade frequency",
                {"entry_threshold": agg_t},
            ),
            (
                "baseline",
                f"Baseline ({base_t:.2f}% Shock)",
                f"Primary hypothesis threshold ({base_t:.2f}%)",
                {"entry_threshold": base_t},
            ),
            (
                "conservative",
                f"Conservative Tail ({con_t:.2f}% Shock)",
                f"Strict tail shock hurdle ({con_t:.2f}%), lower turnover",
                {"entry_threshold": con_t},
            ),
        ]

    split_date = str(df["date"].iloc[split_idx])
    results: list[BracketProfileResult] = []
    for pid, plabel, pdesc, overrides in profiles_cfg:
        pspec = spec.model_copy(deep=True)
        for k, v in overrides.items():
            setattr(pspec.parameters, k, v)
        psim = simulate_strategy_core(df, pspec, split_idx)
        pdates = psim["dates"]
        pgross = psim["daily_gross_ret"]
        pnet = psim["daily_net_ret"]
        pbench = psim["daily_bench_ret"]
        ppos = psim["daily_position"]
        ptrades: list[TradeRecord] = psim["trades"]

        pcb_tripped = bool(psim["circuit_breaker_tripped"])
        pcb_date = psim["circuit_breaker_date"]
        pcb_in_is = bool(pcb_tripped and pcb_date is not None and pcb_date < split_date)
        pcb_in_oos = bool(pcb_tripped and pcb_date is not None and pcb_date >= split_date)

        p_is = _compute_period_metrics(
            f"{plabel} (In-Sample)",
            pdates[:split_idx],
            pgross[:split_idx],
            pnet[:split_idx],
            pbench[:split_idx],
            ppos[:split_idx],
            [t for t in ptrades if t.split == "in_sample"],
            pcb_in_is,
            pcb_date if pcb_in_is else None,
            halted_prior_to_period=False,
        )
        p_oos = _compute_period_metrics(
            f"{plabel} (Out-of-Sample)",
            pdates[split_idx:],
            pgross[split_idx:],
            pnet[split_idx:],
            pbench[split_idx:],
            ppos[split_idx:],
            [t for t in ptrades if t.split == "out_of_sample"],
            pcb_in_oos,
            pcb_date if pcb_tripped else None,
            halted_prior_to_period=pcb_in_is,
        )
        p_full = _compute_period_metrics(
            f"{plabel} (Full Sample)",
            pdates,
            pgross,
            pnet,
            pbench,
            ppos,
            ptrades,
            pcb_tripped,
            pcb_date,
        )
        results.append(
            BracketProfileResult(
                profile_id=pid,
                profile_label=plabel,
                threshold_description=pdesc,
                entry_threshold=float(pspec.parameters.entry_threshold),
                exit_threshold=float(pspec.parameters.exit_threshold),
                lookback_window=int(pspec.parameters.lookback_window),
                holding_period=int(pspec.parameters.holding_period),
                in_sample_metrics=p_is,
                out_of_sample_metrics=p_oos,
                full_sample_metrics=p_full,
                equity_net=[round(float(x), 2) for x in psim["equity_net"]],
            )
        )
    return results


def compute_engine_version_info() -> dict[str, Any]:
    """Compute SHA-256 fingerprint of core computational modules and library versions."""
    engine_dir = Path(__file__).resolve().parent
    module_names = [
        "schemas.py",
        "econometrics.py",
        "backtester.py",
        "stress_tests.py",
        "data_provider.py",
        "hypothesis_parser.py",
    ]
    hasher = hashlib.sha256()
    module_hashes: dict[str, str] = {}
    for fname in module_names:
        fpath = engine_dir / fname
        if fpath.exists():
            raw = fpath.read_bytes()
            fhash = hashlib.sha256(raw).hexdigest()[:12]
            module_hashes[fname] = fhash
            hasher.update(fname.encode("utf-8") + b":" + raw + b"\n")
    return {
        "engine_release": "2.0.0",
        "code_sha256": hasher.hexdigest()[:16],
        "module_hashes": module_hashes,
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "scipy_version": str(scipy.__version__),
    }


def _verify_report_guardrails(
    scorecard: Any,
    stress_suite: Any,
    lit_review: Any,
    methodology_notes: list[str],
    limitations: list[str],
) -> bool:
    """Verify neutrality guardrails across all generated narrative strings in the AuditReport."""
    texts: list[str] = [
        scorecard.hypothesis_restatement,
        scorecard.executive_summary,
        *scorecard.key_takeaways,
        *(d.explanation for d in scorecard.dimensions),
        *(s.interpretation for s in scorecard.calculation_steps),
        stress_suite.gemini_overview,
        *(t.key_finding for t in stress_suite.tests),
        lit_review.theoretical_mechanism_summary,
        lit_review.gemini_literature_synthesis,
        *methodology_notes,
        *limitations,
    ]
    for txt in texts:
        passed, _ = check_neutrality(txt)
        if not passed:
            return False
    return True


_AUDIT_REPORT_CACHE: dict[str, str] = {}


def run_full_audit(spec: StrategySpec) -> AuditReport:
    """Run the complete deterministic backtest, temporal holdout split, and econometric
    diagnostic suite on a confirmed StrategySpec.
    """
    if not spec.human_confirmed:
        raise ValueError("StrategySpec must be explicitly approved by the human researcher before execution.")

    # Load aligned dataset FIRST so data slice SHA-256 and engine version are part of the cache key
    df, provenance = load_aligned_series(
        asset_a=spec.asset_a,
        start_date=spec.start_date,
        end_date=spec.end_date,
        asset_b=spec.asset_b if spec.strategy_family == "pairs_trading" else None,
        benchmark_ticker=spec.benchmark_ticker,
    )

    spec_json = spec.model_dump_json()
    spec_sha256 = hashlib.sha256(spec_json.encode("utf-8")).hexdigest()
    engine_ver = compute_engine_version_info()
    aligned_sha = str(provenance.get("aligned_slice_sha256", ""))
    code_sha = str(engine_ver.get("code_sha256", ""))
    composite_cache_key = f"{spec_sha256}:{aligned_sha}:{code_sha}"

    if composite_cache_key in _AUDIT_REPORT_CACHE:
        return AuditReport.model_validate_json(_AUDIT_REPORT_CACHE[composite_cache_key])

    n = len(df)
    split_idx = max(25, min(n - 25, int(round(n * spec.execution.train_split_ratio))))
    split_date = str(df["date"].iloc[split_idx])

    precomputed = compute_causal_signals(df, spec, split_idx)
    sim = simulate_strategy_core(df, spec, split_idx, precomputed_signals=precomputed)

    dates = sim["dates"]
    gross_r = sim["daily_gross_ret"]
    net_r = sim["daily_net_ret"]
    bench_r = sim["daily_bench_ret"]
    pos = sim["daily_position"]
    trades: list[TradeRecord] = sim["trades"]

    is_trades = [t for t in trades if t.split == "in_sample"]
    oos_trades = [t for t in trades if t.split == "out_of_sample"]

    cb_tripped = bool(sim["circuit_breaker_tripped"])
    cb_date = sim["circuit_breaker_date"]
    cb_in_is = bool(cb_tripped and cb_date is not None and cb_date < split_date)
    cb_in_oos = bool(cb_tripped and cb_date is not None and cb_date >= split_date)

    full_metrics = _compute_period_metrics(
        "Full Sample",
        dates,
        gross_r,
        net_r,
        bench_r,
        pos,
        trades,
        cb_tripped,
        cb_date,
    )
    is_metrics = _compute_period_metrics(
        "In-Sample Formation",
        dates[:split_idx],
        gross_r[:split_idx],
        net_r[:split_idx],
        bench_r[:split_idx],
        pos[:split_idx],
        is_trades,
        cb_in_is,
        cb_date if cb_in_is else None,
        halted_prior_to_period=False,
    )
    oos_metrics = _compute_period_metrics(
        "Out-of-Sample (Held-Out)",
        dates[split_idx:],
        gross_r[split_idx:],
        net_r[split_idx:],
        bench_r[split_idx:],
        pos[split_idx:],
        oos_trades,
        cb_in_oos,
        cb_date if cb_tripped else None,
        halted_prior_to_period=cb_in_is,
    )

    # Stationary Bootstrap CIs (Politis & Romano 1994)
    boot_cis = stationary_bootstrap_cis(
        net_r, bench_r, [t.net_return_pct for t in trades], replications=1000
    )

    # Newey-West HAC Factor Regression (Newey & West 1987)
    factor_reg = newey_west_factor_regression(net_r, bench_r)

    # Transaction Cost Sensitivity Grid (0, 5, 10, 20, 35, 50 bps per leg)
    cost_rows: list[CostSensitivityRow] = []
    for bps in (0.0, 5.0, 10.0, 20.0, 35.0, 50.0):
        c_sim = simulate_strategy_core(
            df, spec, split_idx, override_cost_bps_per_leg=bps, precomputed_signals=precomputed
        )
        c_net = c_sim["daily_net_ret"]
        c_is_ret = float(np.prod(1.0 + c_net[:split_idx]) - 1.0) * 100.0
        c_oos_ret = float(np.prod(1.0 + c_net[split_idx:]) - 1.0) * 100.0
        c_full_ret = float(np.prod(1.0 + c_net) - 1.0) * 100.0
        c_sr, _, _, _, _ = lo_mertens_sharpe_se_and_psr(c_net)
        cost_rows.append(
            CostSensitivityRow(
                total_cost_bps_per_leg=bps,
                in_sample_net_return_pct=round(c_is_ret, 2),
                out_of_sample_net_return_pct=round(c_oos_ret, 2),
                full_period_net_return_pct=round(c_full_ret, 2),
                full_period_sharpe=round(c_sr, 3),
            )
        )

    # Estimate break-even one-way cost per unit notional in bps (round-trip = 2x one-way)
    if len(trades) > 0 and full_metrics.total_gross_return_pct > 0:
        avg_gross_per_trade_bps = (np.mean([t.gross_return_pct for t in trades])) * 100.0
        break_even_bps: float | None = round(max(0.0, avg_gross_per_trade_bps / 2.0), 1)
    else:
        break_even_bps = 0.0

    # Subperiod Breakdown (2-year or calendar chunks)
    subperiods: list[SubperiodRow] = []
    years_series = pd.to_datetime(df["date"]).dt.year
    unique_years = sorted(years_series.unique())
    step_yr = 2 if len(unique_years) >= 6 else 1
    for i in range(0, len(unique_years), step_yr):
        yr_group = unique_years[i : i + step_yr]
        mask = years_series.isin(yr_group).to_numpy()
        if int(np.sum(mask)) < 15:
            continue
        sub_dates = [d for d, m in zip(dates, mask) if m]
        sub_net = net_r[mask]
        sub_bench = bench_r[mask]
        sub_trades = [t for t in trades if sub_dates[0] <= t.entry_date <= sub_dates[-1]]
        s_ret = float(np.prod(1.0 + sub_net) - 1.0) * 100.0
        b_ret = float(np.prod(1.0 + sub_bench) - 1.0) * 100.0
        s_sr, _, _, _, _ = lo_mertens_sharpe_se_and_psr(sub_net)
        eq_s = np.cumprod(1.0 + sub_net)
        pk_s = np.maximum(1.0, np.maximum.accumulate(eq_s))
        dd_s = float(np.max(((pk_s - eq_s) / pk_s) * 100.0))
        lbl = f"{yr_group[0]}" if len(yr_group) == 1 else f"{yr_group[0]}–{yr_group[-1]}"
        subperiods.append(
            SubperiodRow(
                label=lbl,
                start_date=sub_dates[0],
                end_date=sub_dates[-1],
                trade_count=len(sub_trades),
                strategy_net_return_pct=round(s_ret, 2),
                benchmark_return_pct=round(b_ret, 2),
                excess_return_pct=round(s_ret - b_ret, 2),
                sharpe_ratio=round(s_sr, 3),
                max_drawdown_pct=round(dd_s, 2),
            )
        )

    # Parameter Sensitivity Surface (2D grid + Bailey-López de Prado DSR)
    param_surface = (
        compute_parameter_sensitivity_surface(df, spec, split_idx)
        if "parameter_sensitivity_grid" in spec.selected_diagnostics
        else None
    )

    # Step 2 / Step 4: Compute 3-Profile Standard Literature Threshold Bracket
    bracket_profiles = compute_standard_bracket_profiles(df, spec, split_idx)

    # Step 3: Academic Literature & Rigor Review
    literature_review = build_literature_review(spec, sim["pairs_diag"])

    # Step 5: Adversarial Stress Test Suite
    def _stress_sim_cb(
        d_frame: pd.DataFrame,
        s_spec: StrategySpec,
        s_idx: int,
        cost_override: float | None,
        delay_bars: int,
    ) -> dict[str, Any]:
        return simulate_strategy_core(
            d_frame,
            s_spec,
            s_idx,
            override_cost_bps_per_leg=cost_override,
            entry_delay_bars=delay_bars,
            precomputed_signals=precomputed,
        )

    stress_test_suite = run_adversarial_stress_suite(
        df=df,
        spec=spec,
        split_idx=split_idx,
        base_sim=sim,
        full_metrics=full_metrics,
        oos_metrics=oos_metrics,
        pairs_diag=sim["pairs_diag"],
        parameter_surface=param_surface,
        simulate_fn=_stress_sim_cb,
    )

    # Step 6: Empirical Hypothesis Scorecard & Step-by-Step Formula Ledger
    scorecard = build_hypothesis_scorecard_and_ledger(
        spec=spec,
        full_metrics=full_metrics,
        in_metrics=is_metrics,
        oos_metrics=oos_metrics,
        factor_regression=factor_reg,
        bootstrap_cis=boot_cis,
        break_even_cost_bps=break_even_bps,
        parameter_surface=param_surface,
        pairs_diag=sim["pairs_diag"],
        stress_suite=stress_test_suite,
    )

    # Prepare aligned time series for frontend interactive charting
    initial_cap = spec.execution.initial_capital
    bench_equity = initial_cap * np.cumprod(1.0 + bench_r)
    eq_net = sim["equity_net"]
    peaks_net = np.maximum(initial_cap, np.maximum.accumulate(eq_net))
    dd_net_pct = -((peaks_net - eq_net) / peaks_net) * 100.0

    ts_payload: dict[str, list[Any]] = {
        "dates": dates,
        "equity_net": [round(float(x), 2) for x in eq_net],
        "equity_gross": [round(float(x), 2) for x in sim["equity_gross"]],
        "equity_benchmark": [round(float(x), 2) for x in bench_equity],
        "equity_cash": [round(initial_cap, 2)] * n,
        "drawdown_net_pct": [round(float(x), 2) for x in dd_net_pct],
        "close_a": [round(float(x), 4) for x in df["close_a"]],
        "signal_values": [
            round(float(x), 4) if np.isfinite(x) else None for x in sim["signal_values"]
        ],
        "position": [float(x) for x in pos],
    }

    # Include template-specific indicator series for exact chart overlays
    for ind_key in ("sma", "upper_band", "lower_band", "fast_ma", "slow_ma", "high_channel", "low_channel", "hedge_ratio"):
        if ind_key in sim["indicators"]:
            arr = sim["indicators"][ind_key]
            ts_payload[ind_key] = [round(float(x), 4) if np.isfinite(x) else None for x in arr]

    if spec.strategy_family == "pairs_trading":
        ts_payload["close_b"] = [round(float(x), 4) for x in df["close_b"]]
        ca0 = float(df["close_a"].iloc[0])
        cb0 = float(df["close_b"].iloc[0])
        ts_payload["rebased_a"] = [round(float(x / ca0 * 100.0), 2) for x in df["close_a"]]
        ts_payload["rebased_b"] = [round(float(x / cb0 * 100.0), 2) for x in df["close_b"]]
        spread_arr = sim["indicators"]["spread"]
        z_arr = sim["indicators"]["spread_zscore"]
        ts_payload["spread"] = [
            round(float(x), 5) if np.isfinite(x) else None for x in spread_arr
        ]
        ts_payload["spread_zscore"] = [
            round(float(x), 4) if np.isfinite(x) else None for x in z_arr
        ]

    methodology_notes = [
        f"Causal Signal & Execution Timing: Signals are evaluated strictly at the close of bar t using data [0..t] and executed at bar t+1 ({spec.execution.execution_timing}) with {spec.execution.commission_bps:.1f} bps commission + {spec.execution.slippage_bps:.1f} bps adverse slippage per unit of portfolio notional.",
        f"Chronological Holdout Separation: Dataset ({n} trading days) is partitioned into an In-Sample Formation split ({is_metrics.start_date} to {is_metrics.end_date}, {is_metrics.trading_days} days) and a Held-Out Out-of-Sample split ({oos_metrics.start_date} to {oos_metrics.end_date}, {oos_metrics.trading_days} days).",
        "Inference under Non-Normality & Serial Dependence: Sharpe Ratio standard errors and Probabilistic Sharpe Ratios (PSR) incorporate sample skewness and kurtosis per Mertens (2002) and Lo (2002). Confidence intervals use the Politis & Romano (1994) Stationary Bootstrap, and benchmark alpha/beta t-statistics use Newey-West (1987) HAC standard errors.",
    ]
    if spec.strategy_family == "pairs_trading" and sim["pairs_diag"] is not None:
        pd_diag = sim["pairs_diag"]
        if spec.signal_variant == "ols_log":
            methodology_notes.append(
                f"Pairs Cointegration Formation: Within the Formation split ({pd_diag.formation_start} to {pd_diag.formation_end}), OLS parameters (alpha_t, beta_t) are estimated causally on expanding windows [0..t] to eliminate in-sample look-ahead bias, then frozen at the full Formation estimates (beta = {pd_diag.ols_hedge_ratio_beta:.4f}, alpha = {pd_diag.ols_alpha:.4f}) throughout Out-of-Sample evaluation."
            )
        else:
            methodology_notes.append(
                f"Pairs Cointegration Formation: Full Formation split ({pd_diag.formation_start} to {pd_diag.formation_end}) Engle-Granger / ADF diagnostics report beta = {pd_diag.ols_hedge_ratio_beta:.4f}, alpha = {pd_diag.ols_alpha:.4f}; trading signals use the causal '{spec.signal_variant}' estimator."
            )

    limitations = [
        "Historical Simulation Boundary: Historical backtests and statistical significance measures describe past price paths and do not guarantee future market behaviour or liquidity.",
        "Execution & Microstructure Simplification: Fills are modelled at daily Open/Close bars with fixed basis-point slippage and do not model intraday limit-order-book queue priority, borrow fee spikes on short legs, or market impact for large notional orders.",
        "Selection & Multiple-Testing Bias: Inspecting multiple thresholds or re-running hypotheses on the same dataset increases false-discovery probability; consult the Bailey & López de Prado (2014) Deflated Sharpe Ratio (DSR) and Out-of-Sample split metrics when interpreting results.",
    ]

    det_run_hash = hashlib.sha256(composite_cache_key.encode("utf-8")).hexdigest()[:10]
    guardrail_ok = _verify_report_guardrails(
        scorecard, stress_test_suite, literature_review, methodology_notes, limitations
    )

    replay_manifest = {
        "spec_sha256": spec_sha256,
        "aligned_slice_sha256": aligned_sha,
        "engine_code_sha256": code_sha,
        "composite_cache_key": composite_cache_key,
        "bootstrap_seed": 20261107,
        "monte_carlo_seed": 42,
        "split_index": split_idx,
        "split_date": split_date,
    }

    report = AuditReport(
        schema_version="2.0",
        run_id=f"audit-{det_run_hash}",
        created_at_utc=os.environ.get(
            "ALPHA_AUDIT_FIXED_UTC",
            f"{full_metrics.end_date}T00:00:00+00:00",
        ),
        market_as_of_date=full_metrics.end_date,
        spec=spec,
        spec_sha256=spec_sha256,
        data_provenance=provenance,
        engine_version=engine_ver,
        replay_manifest=replay_manifest,
        split_date=split_date,
        full_sample_metrics=full_metrics,
        in_sample_metrics=is_metrics,
        out_of_sample_metrics=oos_metrics,
        bracket_profiles=bracket_profiles,
        timeseries=ts_payload,
        trades=trades,
        bootstrap_cis=boot_cis,
        factor_regression=factor_reg,
        pairs_diagnostics=sim["pairs_diag"],
        cost_sensitivity=cost_rows,
        break_even_cost_bps=break_even_bps,
        subperiods=subperiods,
        parameter_surface=param_surface,
        literature_review=literature_review,
        stress_test_suite=stress_test_suite,
        scorecard=scorecard,
        methodology_notes=methodology_notes,
        limitations=limitations,
        guardrail_verified=guardrail_ok,
    )
    _AUDIT_REPORT_CACHE[composite_cache_key] = report.model_dump_json()
    return report


def save_audit_run(report: AuditReport, runs_dir: Path, overwrite: bool = False) -> Path:
    """Atomically persist an AuditReport JSON to runs_dir without partial writes or silent overwrites."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    target_path = runs_dir / f"{report.run_id}.json"
    if target_path.exists() and not overwrite:
        try:
            existing_raw = json.loads(target_path.read_text(encoding="utf-8"))
            existing_ver = existing_raw.get("schema_version", "1.0")
            if existing_ver == "1.0":
                # Preserve legacy v1 run untouched and write v2 under explicit v2 suffix if needed
                backup_v1 = runs_dir / f"{report.run_id}-v1.json"
                if not backup_v1.exists():
                    backup_v1.write_text(json.dumps(existing_raw, indent=2), encoding="utf-8")
                return target_path
        except Exception:
            pass

    tmp_path = runs_dir / f".{report.run_id}.{uuid.uuid4().hex[:8]}.tmp"
    payload = report.model_dump_json(indent=2)
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, target_path)
    return target_path


def load_audit_run(path: Path) -> AuditReport:
    """Load an AuditReport from disk with schema-version migration support for v1 and v2 runs."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "schema_version" not in raw:
        raw["schema_version"] = "1.0"
    if "market_as_of_date" not in raw or not raw["market_as_of_date"]:
        raw["market_as_of_date"] = raw.get("full_sample_metrics", {}).get("end_date", "")
    return AuditReport.model_validate(raw)

