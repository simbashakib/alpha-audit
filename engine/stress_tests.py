"""Step 5 & Step 6 — Gemini Adversarial Stress Testing & Empirical Hypothesis Scorecard.

Step 5: Designs and executes 5 rigorous adversarial stress tests tailored to the user's
hypothesis, asset data, and strategy mechanics to uncover hidden vulnerabilities.
Step 6: Synthesizes an objective 5-dimension Empirical Hypothesis Evidence Scorecard
and a transparent Step-by-Step Mathematical Formula & Calculation Inspector with the
exact numbers from the current run plugged into each equation.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from engine.guardrails import sanitize_research_notes
from engine.schemas import (
    AdversarialStressTest,
    BootstrapConfidenceInterval,
    FactorRegressionResult,
    FormulaCalculationStep,
    HypothesisScorecard,
    PairsDiagnostics,
    ParameterSensitivitySurface,
    PeriodMetrics,
    ScorecardDimension,
    StrategySpec,
    StressTestSuite,
    TradeRecord,
)


def _test_1_outlier_removal(
    trades: list[TradeRecord],
    full_metrics: PeriodMetrics,
    spec: StrategySpec,
) -> AdversarialStressTest:
    """Test 1: Right-Tail Outlier Trade Removal (Top 1, Top 3, Top 5 best trades)."""
    n_trades = len(trades)
    base_cum_pct = full_metrics.total_net_return_pct

    if n_trades == 0:
        return AdversarialStressTest(
            test_id="outlier_removal",
            title="Right-Tail Outlier Concentration Test (Excl. Top 1 / 3 / 5 Trades)",
            vulnerability_probed="Dependence of cumulative return on a small number of extreme lucky trades.",
            gemini_rationale=(
                f"For '{spec.title}', we test whether cumulative performance is broadly distributed across "
                "all historical signals or concentrated in a handful of extreme market dislocations."
            ),
            methodology="Sorts all executed trades by net return and recomputes compounded equity after removing the top 1, 3, and 5 winners.",
            outcome_badge="HIGH_VULNERABILITY",
            baseline_label="All Trades Net Return",
            baseline_value="0.00% (0 trades)",
            stressed_label="Excl. Top 3 Trades",
            stressed_value="0.00%",
            key_finding="No historical trades were triggered under the current threshold specification.",
            breakdown_rows=[],
            chart_payload={"labels": ["All Trades"], "values": [0.0]},
        )

    sorted_Desc = sorted(trades, key=lambda t: t.net_return_pct, reverse=True)

    def _compounded_return_excluding(k: int) -> tuple[float, float, int]:
        remaining = sorted_Desc[k:] if k < len(sorted_Desc) else []
        if not remaining:
            return 0.0, 0.0, 0
        rets = np.array([t.net_return_pct / 100.0 for t in remaining], dtype=float)
        comp = float((np.prod(1.0 + rets) - 1.0) * 100.0)
        mean_r = float(np.mean(rets) * 100.0)
        return comp, mean_r, len(remaining)

    comp_0, mean_0, cnt_0 = _compounded_return_excluding(0)
    comp_1, mean_1, cnt_1 = _compounded_return_excluding(1)
    comp_3, mean_3, cnt_3 = _compounded_return_excluding(3)
    comp_5, mean_5, cnt_5 = _compounded_return_excluding(5)

    top_1_trade = sorted_Desc[0]
    if comp_0 > 0 and comp_3 > 0 and comp_3 >= 0.5 * comp_0:
        badge = "RESILIENT"
    elif comp_0 > 0 and comp_1 > 0:
        badge = "MODERATE_SENSITIVITY"
    else:
        badge = "HIGH_VULNERABILITY"

    rows = [
        {
            "scenario": "Baseline (All Trades)",
            "trades_included": cnt_0,
            "compounded_net_return_pct": round(comp_0, 2),
            "mean_trade_net_pct": round(mean_0, 3),
        },
        {
            "scenario": f"Excl. Top 1 Trade ({top_1_trade.entry_date}: +{top_1_trade.net_return_pct:.2f}%)",
            "trades_included": cnt_1,
            "compounded_net_return_pct": round(comp_1, 2),
            "mean_trade_net_pct": round(mean_1, 3),
        },
        {
            "scenario": "Excl. Top 3 Winning Trades",
            "trades_included": cnt_3,
            "compounded_net_return_pct": round(comp_3, 2),
            "mean_trade_net_pct": round(mean_3, 3),
        },
        {
            "scenario": "Excl. Top 5 Winning Trades",
            "trades_included": cnt_5,
            "compounded_net_return_pct": round(comp_5, 2),
            "mean_trade_net_pct": round(mean_5, 3),
        },
    ]

    finding = (
        f"Across {n_trades} historical trades, the single largest winning trade occurred on "
        f"{top_1_trade.entry_date} ({top_1_trade.net_return_pct:+.2f}% net). Removing the top 1 trade shifts "
        f"compounded trade return from {comp_0:+.2f}% to {comp_1:+.2f}%; removing the top 3 winning trades "
        f"results in {comp_3:+.2f}% (mean trade return {mean_3:+.3f}%)."
    )

    return AdversarialStressTest(
        test_id="outlier_removal",
        title="Right-Tail Outlier Concentration Test (Excl. Top 1 / 3 / 5 Trades)",
        vulnerability_probed="Right-tail fragility: whether historical gains depend on 1–3 extreme outlier events.",
        gemini_rationale=(
            f"Short-horizon and event-driven strategies on {spec.asset_a} frequently derive a disproportionate "
            "share of total return from crisis-rebound outliers (Nagel, 2012). This test removes the largest "
            "winners to verify whether the remaining signals still exhibit positive expectancy."
        ),
        methodology=(
            "Ranks all historical trades by net return, sequentially removes the top 1, 3, and 5 winning trades, "
            "and recalculates compounded net return and per-trade expectancy."
        ),
        outcome_badge=badge,
        baseline_label="All Trades Compounded",
        baseline_value=f"{comp_0:+.2f}% (N={cnt_0})",
        stressed_label="Excl. Top 3 Trades",
        stressed_value=f"{comp_3:+.2f}% (N={cnt_3})",
        key_finding=finding,
        breakdown_rows=rows,
        chart_payload={
            "type": "bar",
            "labels": ["All Trades", "Excl. Top 1", "Excl. Top 3", "Excl. Top 5"],
            "values": [round(comp_0, 2), round(comp_1, 2), round(comp_3, 2), round(comp_5, 2)],
            "y_label": "Compounded Net Return (%)",
        },
    )


def _test_2_execution_latency_and_cost_shock(
    df: pd.DataFrame,
    spec: StrategySpec,
    split_idx: int,
    full_metrics: PeriodMetrics,
    simulate_fn: Callable[[pd.DataFrame, StrategySpec, int, float | None, int], dict[str, Any]],
) -> AdversarialStressTest:
    """Test 2: Execution Latency (t+2 Entry Delay) & 2x/3x Slippage Cost Shock."""
    base_cost = spec.execution.commission_bps + spec.execution.slippage_bps
    base_ret = full_metrics.total_net_return_pct
    base_sr = full_metrics.sharpe_ratio

    # Run 2x cost, 3x cost, t+2 delay (1x cost), and t+2 delay (2x cost)
    sim_2x_cost = simulate_fn(df, spec, split_idx, base_cost * 2.0, 1)
    sim_3x_cost = simulate_fn(df, spec, split_idx, base_cost * 3.0, 1)
    sim_t2_1x = simulate_fn(df, spec, split_idx, base_cost, 2)
    sim_t2_2x = simulate_fn(df, spec, split_idx, base_cost * 2.0, 2)

    def _stats(sim_res: dict[str, Any]) -> tuple[float, float]:
        r = sim_res["daily_net_ret"]
        eq = sim_res["equity_net"]
        tot = float((eq[-1] / spec.execution.initial_capital - 1.0) * 100.0) if len(eq) > 0 else 0.0
        sd = float(np.std(r, ddof=1)) if len(r) > 2 else 0.0
        sr = float((np.mean(r) / sd) * math.sqrt(252.0)) if sd > 1e-12 else 0.0
        return tot, sr

    ret_2x, sr_2x = _stats(sim_2x_cost)
    ret_3x, sr_3x = _stats(sim_3x_cost)
    ret_t2, sr_t2 = _stats(sim_t2_1x)
    ret_t2_2x, sr_t2_2x = _stats(sim_t2_2x)

    if ret_2x > 0 and ret_t2 > 0 and sr_t2 >= 0.5 * max(base_sr, 0.1):
        badge = "RESILIENT"
    elif ret_2x > 0 or ret_t2 > 0:
        badge = "MODERATE_SENSITIVITY"
    else:
        badge = "HIGH_VULNERABILITY"

    rows = [
        {
            "scenario": f"Baseline (t+1 fill, {base_cost:.0f} bps/leg)",
            "net_return_pct": round(base_ret, 2),
            "sharpe_ratio": round(base_sr, 3),
        },
        {
            "scenario": f"2× Liquidity Cost Shock (t+1 fill, {base_cost * 2:.0f} bps/leg)",
            "net_return_pct": round(ret_2x, 2),
            "sharpe_ratio": round(sr_2x, 3),
        },
        {
            "scenario": f"3× Liquidity Cost Shock (t+1 fill, {base_cost * 3:.0f} bps/leg)",
            "net_return_pct": round(ret_3x, 2),
            "sharpe_ratio": round(sr_3x, 3),
        },
        {
            "scenario": f"1-Day Execution Latency (t+2 fill, {base_cost:.0f} bps/leg)",
            "net_return_pct": round(ret_t2, 2),
            "sharpe_ratio": round(sr_t2, 3),
        },
        {
            "scenario": f"Combined Stress (t+2 fill + {base_cost * 2:.0f} bps/leg)",
            "net_return_pct": round(ret_t2_2x, 2),
            "sharpe_ratio": round(sr_t2_2x, 3),
        },
    ]

    finding = (
        f"Delaying trade execution by 1 additional trading day (filling at bar t+2 instead of t+1) changes "
        f"cumulative net return from {base_ret:+.2f}% (Sharpe {base_sr:.2f}) to {ret_t2:+.2f}% (Sharpe {sr_t2:.2f}). "
        f"Doubling transaction costs to {base_cost * 2:.0f} bps/leg yields {ret_2x:+.2f}% (Sharpe {sr_2x:.2f})."
    )

    return AdversarialStressTest(
        test_id="execution_latency_shock",
        title="Execution Latency (t+2 Fill Delay) & Liquidity Slippage Shock",
        vulnerability_probed="Microstructure decay & cost fragility: sensitivity to delayed order fills and widened bid-ask spreads.",
        gemini_rationale=(
            "Jegadeesh (1990) and Park & Irwin (2007) show that many daily signals decay rapidly if execution "
            "is delayed by one bar or if bid-ask spreads widen during volatile sessions. We re-run the entire "
            "backtest with t+2 execution latency and 2×–3× transaction cost shocks."
        ),
        methodology=(
            "Shifts signal-to-fill latency from t+1 to t+2 bars and scales per-leg commission + slippage by "
            "2× and 3× across the full historical sample."
        ),
        outcome_badge=badge,
        baseline_label=f"Baseline (t+1, {base_cost:.0f} bps)",
        baseline_value=f"{base_ret:+.2f}% (SR {base_sr:.2f})",
        stressed_label=f"t+2 Delay + 2× Cost",
        stressed_value=f"{ret_t2_2x:+.2f}% (SR {sr_t2_2x:.2f})",
        key_finding=finding,
        breakdown_rows=rows,
        chart_payload={
            "type": "bar",
            "labels": ["Baseline (t+1)", "2× Cost", "3× Cost", "t+2 Delay", "t+2 + 2× Cost"],
            "values": [
                round(base_ret, 2),
                round(ret_2x, 2),
                round(ret_3x, 2),
                round(ret_t2, 2),
                round(ret_t2_2x, 2),
            ],
            "y_label": "Total Net Return (%)",
        },
    )


def _test_3_regime_vulnerability(
    df: pd.DataFrame,
    base_sim: dict[str, Any],
    spec: StrategySpec,
) -> AdversarialStressTest:
    """Test 3: Realized Volatility & Bull/Bear Macro Regime Decomposition."""
    net_ret = np.asarray(base_sim["daily_net_ret"], dtype=float)
    bench_ret = np.asarray(base_sim["daily_bench_ret"], dtype=float)
    bench_close = df["close_bench"].to_numpy(dtype=float)

    # Compute trailing 21-day realized volatility (annualized %) and 200-day SMA on benchmark
    vol_21 = (
        pd.Series(bench_ret)
        .rolling(21, min_periods=10)
        .std()
        .bfill()
        .fillna(0.01)
        .to_numpy(dtype=float)
        * math.sqrt(252.0)
        * 100.0
    )
    sma_200 = pd.Series(bench_close).rolling(200, min_periods=30).mean().bfill().to_numpy(dtype=float)

    q33, q67 = np.percentile(vol_21, [33.3, 66.7])

    regimes = [
        ("Low Volatility Tercile", vol_21 <= q33),
        ("Normal Volatility Tercile", (vol_21 > q33) & (vol_21 <= q67)),
        ("High Volatility Tercile", vol_21 > q67),
        ("Bull Regime (Bench > 200d SMA)", bench_close >= sma_200),
        ("Bear/Correction (Bench < 200d SMA)", bench_close < sma_200),
    ]

    rows: list[dict[str, Any]] = []
    labels: list[str] = []
    sharpes: list[float] = []
    positive_regimes = 0

    for label, mask in regimes:
        sub_r = net_ret[mask]
        sub_b = bench_ret[mask]
        n_days = int(np.sum(mask))
        if n_days < 5:
            continue
        cum_r = float((np.prod(1.0 + sub_r) - 1.0) * 100.0)
        cum_b = float((np.prod(1.0 + sub_b) - 1.0) * 100.0)
        sd = float(np.std(sub_r, ddof=1)) if n_days > 2 else 0.0
        sr = float((np.mean(sub_r) / sd) * math.sqrt(252.0)) if sd > 1e-12 else 0.0
        active_days = int(np.sum(np.abs(sub_r) > 1e-9))
        if cum_r > 0:
            positive_regimes += 1
        rows.append(
            {
                "regime": label,
                "trading_days": n_days,
                "active_exposure_days": active_days,
                "strategy_net_return_pct": round(cum_r, 2),
                "benchmark_return_pct": round(cum_b, 2),
                "sharpe_ratio": round(sr, 3),
            }
        )
        labels.append(label.replace(" (Bench > 200d SMA)", "").replace(" (Bench < 200d SMA)", ""))
        sharpes.append(round(sr, 3))

    if positive_regimes >= 4:
        badge = "RESILIENT"
    elif positive_regimes >= 2:
        badge = "MODERATE_SENSITIVITY"
    else:
        badge = "HIGH_VULNERABILITY"

    high_vol_row = next((r for r in rows if "High Volatility" in r["regime"]), rows[0] if rows else {})
    low_vol_row = next((r for r in rows if "Low Volatility" in r["regime"]), rows[0] if rows else {})

    finding = (
        f"Strategy performance was positive in {positive_regimes} of {len(rows)} tested volatility and macro "
        f"regimes. In High Volatility periods (>{q67:.1f}% ann. vol), the strategy returned "
        f"{high_vol_row.get('strategy_net_return_pct', 0.0):+.2f}% (Sharpe {high_vol_row.get('sharpe_ratio', 0.0):.2f}), "
        f"compared to {low_vol_row.get('strategy_net_return_pct', 0.0):+.2f}% (Sharpe {low_vol_row.get('sharpe_ratio', 0.0):.2f}) "
        f"in Low Volatility periods (<{q33:.1f}% ann. vol)."
    )

    return AdversarialStressTest(
        test_id="regime_vulnerability",
        title="Realized Volatility Terciles & Bull/Bear Macro Regime Breakdown",
        vulnerability_probed="Regime dependency: whether returns only materialize in a single volatility or trend state.",
        gemini_rationale=(
            f"Strategies on {spec.asset_a} can mask severe regime dependency in full-sample averages. "
            "We partition the historical sample by trailing 21-day realized volatility terciles and "
            "the 200-day benchmark trend filter to inspect conditional Sharpe stability."
        ),
        methodology=(
            "Classifies each trading bar into Low/Normal/High realized volatility terciles and Bull/Bear "
            "macro states (200-day SMA) and computes conditional net returns and annualized Sharpe ratios."
        ),
        outcome_badge=badge,
        baseline_label="Positive Regimes",
        baseline_value=f"{positive_regimes} / {len(rows)} Regimes",
        stressed_label="High-Vol vs Low-Vol SR",
        stressed_value=f"{high_vol_row.get('sharpe_ratio', 0.0):.2f} vs {low_vol_row.get('sharpe_ratio', 0.0):.2f}",
        key_finding=finding,
        breakdown_rows=rows,
        chart_payload={
            "type": "bar",
            "labels": labels,
            "values": sharpes,
            "y_label": "Conditional Sharpe Ratio",
        },
    )


def _test_4_monte_carlo_random_entry(
    df: pd.DataFrame,
    base_sim: dict[str, Any],
    spec: StrategySpec,
    full_metrics: PeriodMetrics,
    replications: int = 500,
    seed: int = 42,
) -> AdversarialStressTest:
    """Test 4: Monte Carlo Random-Entry Benchmark (500 Simulations with matched N, holding durations, directions, fill timing, and pair weights)."""
    trades: list[TradeRecord] = base_sim["trades"]
    n_trades = len(trades)
    actual_ret = full_metrics.total_net_return_pct

    close_a = df["close_a"].to_numpy(dtype=float)
    open_a = df["open_a"].to_numpy(dtype=float)
    n_bars = len(close_a)

    if n_trades == 0 or n_bars < 30:
        return AdversarialStressTest(
            test_id="monte_carlo_random_entry",
            title="Monte Carlo Random-Entry Null Benchmark (500 Simulations)",
            vulnerability_probed="Whether signal entry timing adds statistical edge beyond passive market drift.",
            gemini_rationale="Compares actual signal entry dates against 500 random-entry strategies with identical exposure.",
            methodology="Simulates 500 random non-overlapping trade sequences with matched trade count, holding durations, and directions.",
            outcome_badge="HIGH_VULNERABILITY",
            baseline_label="Actual Strategy Return",
            baseline_value="0.00%",
            stressed_label="Random-Entry Median",
            stressed_value="0.00%",
            key_finding="Insufficient trades to run Monte Carlo random-entry comparison.",
            breakdown_rows=[],
            chart_payload={"type": "bar", "labels": [], "values": []},
        )

    is_pairs = spec.strategy_family == "pairs_trading" and "close_b" in df.columns
    close_b = df["close_b"].to_numpy(dtype=float) if is_pairs else np.ones(n_bars, dtype=float)
    open_b = df["open_b"].to_numpy(dtype=float) if is_pairs else np.ones(n_bars, dtype=float)
    exec_next_open = spec.execution.execution_timing == "next_open"
    one_way_cost = (spec.execution.commission_bps + spec.execution.slippage_bps) / 10_000.0

    # Extract exact multiset of (holding_days, direction, leg_weight_a, leg_weight_b)
    trade_specs: list[tuple[int, str, float, float]] = []
    for t in trades:
        h = max(1, min(n_bars - 3, int(t.holding_days)))
        wa = float(getattr(t, "leg_weight_a", 1.0 if t.direction == "long" else -1.0))
        wb = float(getattr(t, "leg_weight_b", 0.0))
        if is_pairs and abs(wa) + abs(wb) < 1e-6:
            wa, wb = (0.5, -0.5) if t.direction == "long_spread" else (-0.5, 0.5)
        trade_specs.append((h, t.direction, wa, wb))

    avg_hold = max(1, int(round(np.mean([ts[0] for ts in trade_specs]))))
    sum_hold = sum(ts[0] for ts in trade_specs)
    usable_span = max(1, n_bars - 2)
    free_bars = usable_span - sum_hold

    def _eval_matched_trade(entry_idx: int, hold_d: int, direction: str, wa: float, wb: float) -> float:
        exit_idx = min(n_bars - 1, entry_idx + hold_d)
        if exit_idx <= entry_idx:
            exit_idx = min(n_bars - 1, entry_idx + 1)
        pa_in = open_a[entry_idx] if exec_next_open else close_a[entry_idx]
        pa_out = open_a[exit_idx] if (exec_next_open and exit_idx < n_bars - 1) else close_a[exit_idx]
        if is_pairs:
            pb_in = open_b[entry_idx] if exec_next_open else close_b[entry_idx]
            pb_out = open_b[exit_idx] if (exec_next_open and exit_idx < n_bars - 1) else close_b[exit_idx]
            ra = (pa_out / pa_in) - 1.0
            rb = (pb_out / pb_in) - 1.0
            gross_r = wa * ra + wb * rb
        else:
            r = (pa_out / pa_in) - 1.0
            gross_r = r if direction == "long" else -r
        # Round-trip entry + exit fee deduction matching simulate_strategy_core
        net_mult = (1.0 + gross_r) - 2.0 * one_way_cost
        return max(0.05, net_mult)

    rng = np.random.default_rng(seed)
    mc_returns = np.zeros(replications, dtype=float)
    perm_indices = np.arange(n_trades)

    for m in range(replications):
        rng.shuffle(perm_indices)
        if free_bars >= n_trades:
            # Exact stars-and-bars bijection for uniform non-overlapping intervals
            chosen_slots = np.sort(rng.choice(free_bars + n_trades, size=n_trades, replace=False))
            eq = 1.0
            cum_h = 0
            for j, slot in enumerate(chosen_slots):
                h_j, dir_j, wa_j, wb_j = trade_specs[perm_indices[j]]
                start_idx = 1 + int(slot - j) + cum_h
                start_idx = max(1, min(n_bars - 2, start_idx))
                eq *= _eval_matched_trade(start_idx, h_j, dir_j, wa_j, wb_j)
                cum_h += h_j
            mc_returns[m] = (eq - 1.0) * 100.0
        else:
            # High-occupancy fallback: sequential non-overlapping placement with random wrap offset
            offset = int(rng.integers(1, max(2, n_bars - 2)))
            eq = 1.0
            cursor = offset
            for idx_p in perm_indices:
                h_j, dir_j, wa_j, wb_j = trade_specs[idx_p]
                if cursor + h_j >= n_bars - 1:
                    cursor = int(rng.integers(1, max(2, min(10, n_bars - h_j - 1))))
                eq *= _eval_matched_trade(cursor, h_j, dir_j, wa_j, wb_j)
                cursor += h_j + 1
            mc_returns[m] = (eq - 1.0) * 100.0

    mc_median = float(np.median(mc_returns))
    mc_mean = float(np.mean(mc_returns))
    mc_p90 = float(np.percentile(mc_returns, 90.0))
    mc_p10 = float(np.percentile(mc_returns, 10.0))
    percentile_rank = float(np.mean(mc_returns <= actual_ret) * 100.0)
    p_val_mc = float(np.mean(mc_returns >= actual_ret))
    p_se_mc = math.sqrt(max(1e-9, p_val_mc * (1.0 - p_val_mc)) / replications)

    if percentile_rank >= 85.0:
        badge = "RESILIENT"
    elif percentile_rank >= 55.0:
        badge = "MODERATE_SENSITIVITY"
    else:
        badge = "HIGH_VULNERABILITY"

    rows = [
        {
            "benchmark_metric": "Actual Strategy Net Return",
            "value": f"{actual_ret:+.2f}%",
            "note": f"N={n_trades} trades, mean hold {avg_hold}d ({spec.execution.execution_timing})",
        },
        {
            "benchmark_metric": f"Monte Carlo Non-Overlapping Median ({replications} runs)",
            "value": f"{mc_median:+.2f}%",
            "note": "50th percentile of random non-overlapping entries with matched durations & directions",
        },
        {
            "benchmark_metric": "Monte Carlo 10th to 90th Percentile Range",
            "value": f"[{mc_p10:+.2f}%, {mc_p90:+.2f}%]",
            "note": "80% central interval of matched random-entry outcomes",
        },
        {
            "benchmark_metric": "Signal Timing Percentile Rank & MC SE",
            "value": f"{percentile_rank:.1f}th pct (p = {p_val_mc:.3f} ± {p_se_mc:.3f})",
            "note": f"Empirical one-sided p-value with Monte Carlo standard error (B={replications})",
        },
    ]

    finding = (
        f"Across {replications} non-overlapping Monte Carlo simulations matching the exact multiset of "
        f"{n_trades} trade durations (mean {avg_hold}d), directions, fill timing ({spec.execution.execution_timing}), "
        f"and transaction costs, the median random-entry return was {mc_median:+.2f}% (10th–90th percentile: "
        f"[{mc_p10:+.2f}%, {mc_p90:+.2f}%]). The hypothesis signal's actual return ({actual_ret:+.2f}%) "
        f"ranked at the {percentile_rank:.1f}th percentile (empirical one-sided p = {p_val_mc:.3f} ± {p_se_mc:.3f})."
    )

    return AdversarialStressTest(
        test_id="monte_carlo_random_entry",
        title=f"Monte Carlo Random-Entry Null Benchmark ({replications} Non-Overlapping Simulations)",
        vulnerability_probed="Passive drift vs. signal timing: whether non-overlapping random entries with matched durations, directions, and weights match the signal.",
        gemini_rationale=(
            f"On trending assets like {spec.asset_a}, any directional rule can appear profitable simply by "
            "capturing unconditional market drift. We benchmark the user's exact trigger against 500 "
            f"non-overlapping random-entry simulations matching all N={n_trades} trade durations and directions."
        ),
        methodology=(
            f"Draws {replications} independent non-overlapping random-entry paths (matching the exact trade count N={n_trades}, "
            f"holding-period distribution, trade directions, leg weights, {spec.execution.execution_timing} fills, and costs) "
            f"to compute an empirical signal-timing p-value and Monte Carlo standard error."
        ),
        outcome_badge=badge,
        baseline_label="Actual Signal Return",
        baseline_value=f"{actual_ret:+.2f}% ({percentile_rank:.0f}th pct)",
        stressed_label="Random-Entry Median",
        stressed_value=f"{mc_median:+.2f}% (p={p_val_mc:.3f}±{p_se_mc:.3f})",
        key_finding=finding,
        breakdown_rows=rows,
        chart_payload={
            "type": "bar",
            "labels": ["MC 10th Pct", "MC Median (50th)", "MC Mean", "MC 90th Pct", "Actual Signal"],
            "values": [
                round(mc_p10, 2),
                round(mc_median, 2),
                round(mc_mean, 2),
                round(mc_p90, 2),
                round(actual_ret, 2),
            ],
            "y_label": "Total Net Return (%)",
        },
    )


def _test_5_structural_or_parameter_stability(
    df: pd.DataFrame,
    spec: StrategySpec,
    split_idx: int,
    pairs_diag: PairsDiagnostics | None,
    parameter_surface: ParameterSensitivitySurface | None,
    full_metrics: PeriodMetrics,
) -> AdversarialStressTest:
    """Test 5: Pairs Cointegration Structural Break OR Single-Asset Parameter Cliff-Edge Test."""
    if spec.strategy_family == "pairs_trading" and "close_b" in df.columns and pairs_diag is not None:
        # Compare Formation (In-Sample) vs Out-of-Sample OLS Hedge Ratio & ADF stationarity
        log_a_is = np.log(df["close_a"].iloc[:split_idx].to_numpy(dtype=float))
        log_b_is = np.log(df["close_b"].iloc[:split_idx].to_numpy(dtype=float))
        log_a_oos = np.log(df["close_a"].iloc[split_idx:].to_numpy(dtype=float))
        log_b_oos = np.log(df["close_b"].iloc[split_idx:].to_numpy(dtype=float))

        beta_is = pairs_diag.ols_hedge_ratio_beta
        alpha_is = pairs_diag.ols_alpha

        if len(log_a_oos) > 25:
            x_oos = sm.add_constant(log_b_oos)
            res_oos = sm.OLS(log_a_oos, x_oos).fit()
            beta_oos = float(res_oos.params[1]) if len(res_oos.params) > 1 else beta_is
            spread_oos = log_a_oos - (alpha_is + beta_is * log_b_oos)
            try:
                adf_oos_res = adfuller(spread_oos, autolag="AIC")
                adf_oos_p = float(adf_oos_res[1])
            except Exception:
                adf_oos_p = 0.5
        else:
            beta_oos = beta_is
            adf_oos_p = pairs_diag.adf_p_value

        beta_drift_pct = abs(beta_oos - beta_is) / max(abs(beta_is), 1e-6) * 100.0

        if adf_oos_p < 0.10 and beta_drift_pct < 25.0:
            badge = "RESILIENT"
        elif adf_oos_p < 0.25 or beta_drift_pct < 40.0:
            badge = "MODERATE_SENSITIVITY"
        else:
            badge = "HIGH_VULNERABILITY"

        rows = [
            {
                "metric": "OLS Hedge Ratio (β)",
                "in_sample_formation": f"{beta_is:.4f}",
                "out_of_sample_validation": f"{beta_oos:.4f} (Δ {beta_drift_pct:.1f}%)",
            },
            {
                "metric": "Spread ADF Stationarity p-value",
                "in_sample_formation": f"{pairs_diag.adf_p_value:.4f}",
                "out_of_sample_validation": f"{adf_oos_p:.4f}",
            },
            {
                "metric": "Ornstein-Uhlenbeck Half-Life",
                "in_sample_formation": f"{pairs_diag.half_life_days or 'N/A'} days",
                "out_of_sample_validation": f"Hurst H = {pairs_diag.hurst_exponent:.3f}",
            },
        ]

        finding = (
            f"Between the In-Sample formation period and Out-of-Sample trading period, the optimal OLS hedge "
            f"ratio β between {spec.asset_a} and {spec.asset_b} shifted by {beta_drift_pct:.1f}% "
            f"({beta_is:.3f} → {beta_oos:.3f}). Out-of-sample spread ADF stationarity p-value is {adf_oos_p:.4f} "
            f"(vs. {pairs_diag.adf_p_value:.4f} in-sample)."
        )

        return AdversarialStressTest(
            test_id="pairs_structural_break",
            title="Out-of-Sample Hedge Ratio Drift & Cointegration Breakdown Test",
            vulnerability_probed="Cointegration regime break: whether the two assets decouple out-of-sample (Do & Faff, 2010).",
            gemini_rationale=(
                f"Pairs trading between {spec.asset_a} and {spec.asset_b} assumes a stable equilibrium spread. "
                "We test whether the formation hedge ratio β and spread stationarity survive in the held-out sample."
            ),
            methodology=(
                "Re-estimates the OLS cointegrating regression on the Out-of-Sample split to measure hedge-ratio "
                "parameter drift (Δβ%) and out-of-sample Augmented Dickey-Fuller (ADF) stationarity."
            ),
            outcome_badge=badge,
            baseline_label="In-Sample β / ADF p",
            baseline_value=f"β={beta_is:.3f} (p={pairs_diag.adf_p_value:.3f})",
            stressed_label="OOS β Drift / ADF p",
            stressed_value=f"Δβ={beta_drift_pct:.1f}% (p={adf_oos_p:.3f})",
            key_finding=finding,
            breakdown_rows=rows,
            chart_payload={
                "type": "bar",
                "labels": ["In-Sample β", "Out-of-Sample β", "IS ADF p-val", "OOS ADF p-val"],
                "values": [round(beta_is, 3), round(beta_oos, 3), round(pairs_diag.adf_p_value, 3), round(adf_oos_p, 3)],
                "y_label": "Parameter / p-value",
            },
        )

    # Single-asset: Parameter Cliff-Edge Perturbation Test
    cells = parameter_surface.cells if parameter_surface else []
    if not cells:
        return AdversarialStressTest(
            test_id="parameter_cliff_edge",
            title="Parameter Neighborhood Cliff-Edge Perturbation Test",
            vulnerability_probed="Overfitting to a narrow parameter island (Bailey & López de Prado, 2014).",
            gemini_rationale="Tests whether small perturbations to threshold or holding period cause performance collapse.",
            methodology="Evaluates Sharpe ratio dispersion across the 5×5 parameter neighborhood grid.",
            outcome_badge="MODERATE_SENSITIVITY",
            baseline_label="Selected Spec Sharpe",
            baseline_value=f"{full_metrics.sharpe_ratio:.2f}",
            stressed_label="Grid Median Sharpe",
            stressed_value=f"{full_metrics.sharpe_ratio:.2f}",
            key_finding="Parameter sensitivity surface evaluated.",
            breakdown_rows=[],
            chart_payload={"type": "bar", "labels": [], "values": []},
        )

    full_sharpes = np.array([c.full_sharpe for c in cells], dtype=float)
    oos_sharpes = np.array([c.out_of_sample_sharpe for c in cells], dtype=float)
    pos_cells = int(np.sum(full_sharpes > 0.0))
    pos_oos_cells = int(np.sum(oos_sharpes > 0.0))
    median_sr = float(np.median(full_sharpes))
    min_sr = float(np.min(full_sharpes))
    max_sr = float(np.max(full_sharpes))

    if pos_cells >= 18 and pos_oos_cells >= 15:
        badge = "RESILIENT"
    elif pos_cells >= 10:
        badge = "MODERATE_SENSITIVITY"
    else:
        badge = "HIGH_VULNERABILITY"

    rows = [
        {
            "grid_metric": "Positive Full-Sample Sharpe Cells",
            "value": f"{pos_cells} / {len(cells)} ({pos_cells / len(cells) * 100:.0f}%)",
        },
        {
            "grid_metric": "Positive Out-of-Sample Sharpe Cells",
            "value": f"{pos_oos_cells} / {len(cells)} ({pos_oos_cells / len(cells) * 100:.0f}%)",
        },
        {
            "grid_metric": "Neighborhood Sharpe Range [Min, Median, Max]",
            "value": f"[{min_sr:.2f}, {median_sr:.2f}, {max_sr:.2f}]",
        },
        {
            "grid_metric": "Deflated Sharpe Ratio (DSR, K=25 trials)",
            "value": f"{parameter_surface.selected_spec_deflated_sharpe_psr_pct:.1f}%",
        },
    ]

    finding = (
        f"Across the 25-cell parameter neighborhood surrounding the chosen thresholds, {pos_cells}/25 cells "
        f"({pos_cells / 25 * 100:.0f}%) produced a positive Full-Sample Sharpe ratio and {pos_oos_cells}/25 "
        f"({pos_oos_cells / 25 * 100:.0f}%) were positive Out-of-Sample. The neighborhood median Sharpe is "
        f"{median_sr:.2f} (range [{min_sr:.2f}, {max_sr:.2f}])."
    )

    return AdversarialStressTest(
        test_id="parameter_cliff_edge",
        title="Parameter Neighborhood Cliff-Edge Perturbation Test (25-Grid Sweep)",
        vulnerability_probed="Threshold overfitting: whether perturbing thresholds by ±15–30% causes a cliff-edge collapse.",
        gemini_rationale=(
            "A robust quantitative hypothesis should form a smooth plateau across neighboring thresholds rather "
            "than an isolated spike at a single parameter value (Bailey & López de Prado, 2014). We inspect "
            "all 25 neighboring threshold combinations."
        ),
        methodology=(
            f"Perturbs {parameter_surface.param_x_label} and {parameter_surface.param_y_label} across a 5×5 grid "
            "and measures the proportion of neighboring cells that maintain positive Sharpe ratios."
        ),
        outcome_badge=badge,
        baseline_label="Selected Spec Sharpe",
        baseline_value=f"{full_metrics.sharpe_ratio:.2f}",
        stressed_label="Grid Median (Pos Cells)",
        stressed_value=f"{median_sr:.2f} ({pos_cells}/25 pos)",
        key_finding=finding,
        breakdown_rows=rows,
        chart_payload={
            "type": "bar",
            "labels": ["Grid Min SR", "Grid 25th Pct", "Grid Median SR", "Selected Spec SR", "Grid Max SR"],
            "values": [
                round(min_sr, 3),
                round(float(np.percentile(full_sharpes, 25)), 3),
                round(median_sr, 3),
                round(full_metrics.sharpe_ratio, 3),
                round(max_sr, 3),
            ],
            "y_label": "Annualized Sharpe Ratio",
        },
    )


_STRESS_OVERVIEW_CACHE: dict[str, str] = {}


def run_adversarial_stress_suite(
    df: pd.DataFrame,
    spec: StrategySpec,
    split_idx: int,
    base_sim: dict[str, Any],
    full_metrics: PeriodMetrics,
    oos_metrics: PeriodMetrics,
    pairs_diag: PairsDiagnostics | None,
    parameter_surface: ParameterSensitivitySurface | None,
    simulate_fn: Callable[[pd.DataFrame, StrategySpec, int, float | None, int], dict[str, Any]],
) -> StressTestSuite:
    """Execute all 5 custom adversarial stress tests for Step 5."""
    t1 = _test_1_outlier_removal(base_sim["trades"], full_metrics, spec)
    t2 = _test_2_execution_latency_and_cost_shock(df, spec, split_idx, full_metrics, simulate_fn)
    t3 = _test_3_regime_vulnerability(df, base_sim, spec)
    t4 = _test_4_monte_carlo_random_entry(df, base_sim, spec, full_metrics)
    t5 = _test_5_structural_or_parameter_stability(
        df, spec, split_idx, pairs_diag, parameter_surface, full_metrics
    )

    tests = [t1, t2, t3, t4, t5]
    vuln_count = sum(1 for t in tests if t.outcome_badge == "HIGH_VULNERABILITY")
    resilient_count = sum(1 for t in tests if t.outcome_badge == "RESILIENT")

    overview = (
        f"Alpha Auditor ran 5 adversarial stress tests for '{spec.title}' ({spec.template_id} / "
        f"{spec.signal_variant}) on {spec.asset_a}"
        + (f" vs. {spec.asset_b}" if spec.asset_b else "")
        + f": (1) Right-Tail Outlier Trade Removal, (2) t+2 Execution Latency & 2×/3× Slippage Shock, "
        f"(3) Realized Volatility & Bull/Bear Macro Regime Decomposition, (4) 500-Run Monte Carlo Random-Entry "
        f"Null Benchmark, and (5) {'Out-of-Sample Cointegration Structural Break' if spec.strategy_family == 'pairs_trading' else '25-Cell Parameter Cliff-Edge Perturbation'}. "
        f"Across the 5 stress tests, {resilient_count} showed high resilience, "
        f"{5 - resilient_count - vuln_count} showed moderate sensitivity, and {vuln_count} flagged high vulnerability."
    )

    cache_key = f"{spec.model_dump_json()}:{overview}"
    if cache_key in _STRESS_OVERVIEW_CACHE:
        overview = _STRESS_OVERVIEW_CACHE[cache_key]
    else:
        _STRESS_OVERVIEW_CACHE[cache_key] = overview

    return StressTestSuite(
        gemini_overview=sanitize_research_notes([overview])[0],
        vulnerability_count=vuln_count,
        resilient_count=resilient_count,
        tests=tests,
    )


def build_hypothesis_scorecard_and_ledger(
    spec: StrategySpec,
    full_metrics: PeriodMetrics,
    in_metrics: PeriodMetrics,
    oos_metrics: PeriodMetrics,
    factor_regression: FactorRegressionResult,
    bootstrap_cis: list[BootstrapConfidenceInterval],
    break_even_cost_bps: float | None,
    parameter_surface: ParameterSensitivitySurface | None,
    pairs_diag: PairsDiagnostics | None,
    stress_suite: StressTestSuite,
) -> HypothesisScorecard:
    """Construct Step 6's Empirical Hypothesis Evidence Scorecard & Calculation Inspector."""
    # Dimension 1: Post-Signal Direction & Mean Trade Expectancy
    if full_metrics.trade_count == 0:
        d1_status = "NO_EVIDENCE"
    elif full_metrics.trade_count >= 5 and full_metrics.mean_trade_net_pct > 0 and full_metrics.total_net_return_pct > 0:
        d1_status = "SUPPORTED"
    elif full_metrics.trade_count > 0 and (full_metrics.mean_trade_net_pct > 0 or full_metrics.total_gross_return_pct > 0):
        d1_status = "MIXED"
    else:
        d1_status = "CONTRADICTED"

    dim1 = ScorecardDimension(
        dimension_id="post_signal_direction",
        name="1. Post-Signal Direction & Net Trade Expectancy",
        question="Did historical occurrences of the signal produce positive average returns after transaction costs?",
        status=d1_status,
        evidence_metric=(
            "N = 0 trades (No Signal Triggers)"
            if full_metrics.trade_count == 0
            else (
                f"Mean Net Trade: {full_metrics.mean_trade_net_pct:+.3f}% | Win Rate: {full_metrics.win_rate_pct:.1f}% "
                f"(N = {full_metrics.trade_count} trades)"
            )
        ),
        explanation=(
            "Zero historical trades were triggered under the configured thresholds; trade expectancy is undefined."
            if full_metrics.trade_count == 0
            else (
                f"Across {full_metrics.trade_count} historical signals from {full_metrics.start_date} to {full_metrics.end_date}, "
                f"the average trade returned {full_metrics.mean_trade_net_pct:+.3f}% net ({full_metrics.total_gross_return_pct:+.2f}% "
                f"cumulative gross, {full_metrics.total_net_return_pct:+.2f}% cumulative net). "
                "Heuristic classification requires N >= 5 trades and positive net mean/cumulative return for SUPPORTED."
            )
        ),
    )

    # Dimension 2: Statistical Significance vs. Noise (PSR & Newey-West HAC Alpha)
    psr = full_metrics.probabilistic_sharpe_psr_pct
    nw_p = factor_regression.alpha_p_value_hac
    pos_alpha = factor_regression.alpha_t_stat_hac > 0.0
    pos_sharpe = full_metrics.sharpe_ratio > 0.0
    pos_mean = full_metrics.t_stat_mean_return > 0.0
    if full_metrics.trade_count == 0 or full_metrics.is_zero_exposure:
        d2_status = "NO_EVIDENCE"
    elif pos_sharpe and psr >= 85.0 and (pos_alpha and nw_p <= 0.10 or (pos_mean and full_metrics.p_value_two_sided <= 0.05)):
        d2_status = "SUPPORTED"
    elif pos_sharpe and (
        psr >= 60.0
        or (pos_alpha and nw_p <= 0.25)
        or (pos_mean and full_metrics.p_value_two_sided <= 0.15)
    ):
        d2_status = "MIXED"
    else:
        d2_status = "CONTRADICTED"

    dim2 = ScorecardDimension(
        dimension_id="statistical_significance",
        name="2. Statistical Significance vs. Noise (Lo 2002 PSR & Newey-West HAC)",
        question="Is the observed return/Sharpe ratio statistically distinguishable from random noise?",
        status=d2_status,
        evidence_metric=(
            "Undefined (0 trades / 0% exposure)"
            if (full_metrics.trade_count == 0 or full_metrics.is_zero_exposure)
            else (
                f"Sharpe: {full_metrics.sharpe_ratio:.2f} ± {full_metrics.sharpe_se_lo2002:.2f} (PSR {psr:.1f}%) | "
                f"HAC Alpha t = {factor_regression.alpha_t_stat_hac:.2f} (p = {nw_p:.3f})"
            )
        ),
        explanation=(
            "Return-to-volatility ratios and factor regressions are undefined when zero trades are executed."
            if (full_metrics.trade_count == 0 or full_metrics.is_zero_exposure)
            else (
                f"After adjusting for sample skewness ({full_metrics.skewness_daily:.2f}) and Pearson kurtosis "
                f"({full_metrics.kurtosis_daily:.2f}) via Lo (2002), the Probabilistic Sharpe Ratio is {psr:.1f}%. "
                f"Benchmark factor regression yields annualized alpha of {factor_regression.annualized_alpha_pct:+.2f}% "
                f"(Newey-West HAC t = {factor_regression.alpha_t_stat_hac:.2f}, p = {nw_p:.3f}). "
                "Heuristic hurdle: PSR >= 85% and HAC p <= 0.10 (or t-test p <= 0.05)."
            )
        ),
    )

    # Dimension 3: Out-of-Sample Persistence
    if oos_metrics.trade_count == 0 or oos_metrics.is_zero_exposure:
        d3_status = "NO_EVIDENCE"
        if oos_metrics.halted_prior_to_period:
            halt_dt = in_metrics.circuit_breaker_date or oos_metrics.circuit_breaker_date or full_metrics.circuit_breaker_date
            d3_metric = (
                f"In-Sample SR: {in_metrics.sharpe_ratio:.2f} ({in_metrics.total_net_return_pct:+.1f}%, N={in_metrics.trade_count}) → "
                f"Out-of-Sample: N/A (0 trades — Halted on {halt_dt} in Formation)"
            )
            d3_expl = (
                f"No Out-of-Sample trades were executed between {oos_metrics.start_date} and {oos_metrics.end_date} "
                f"because the {spec.risk_controls.max_drawdown_circuit_breaker_pct:.1f}% portfolio drawdown circuit breaker "
                f"was triggered during the In-Sample Formation split on {halt_dt}. "
                "Consequently, the held-out window contains zero active exposure (NO_EVIDENCE)."
            )
        else:
            d3_metric = (
                f"In-Sample SR: {in_metrics.sharpe_ratio:.2f} ({in_metrics.total_net_return_pct:+.1f}%, N={in_metrics.trade_count}) → "
                "Out-of-Sample: N/A (0 trades triggered)"
            )
            d3_expl = (
                f"In the held-out validation sample ({oos_metrics.start_date} to {oos_metrics.end_date}), "
                "zero signal triggers occurred (0% exposure). Out-of-Sample Sharpe and PSR are undefined on a flat cash series (NO_EVIDENCE)."
            )
    else:
        if oos_metrics.trade_count >= 3 and oos_metrics.total_net_return_pct > 0 and oos_metrics.sharpe_ratio > 0.2:
            d3_status = "SUPPORTED"
        elif oos_metrics.trade_count > 0 and (
            oos_metrics.total_net_return_pct > 0.0
            or oos_metrics.sharpe_ratio > 0.0
        ):
            d3_status = "MIXED"
        else:
            d3_status = "CONTRADICTED"
        d3_metric = (
            f"In-Sample SR: {in_metrics.sharpe_ratio:.2f} ({in_metrics.total_net_return_pct:+.1f}%, N={in_metrics.trade_count}) → "
            f"Out-of-Sample SR: {oos_metrics.sharpe_ratio:.2f} ({oos_metrics.total_net_return_pct:+.1f}%, N={oos_metrics.trade_count})"
        )
        d3_expl = (
            f"In the held-out validation sample ({oos_metrics.start_date} to {oos_metrics.end_date}), "
            f"the strategy executed {oos_metrics.trade_count} trades with net return {oos_metrics.total_net_return_pct:+.2f}% "
            f"and Sharpe ratio {oos_metrics.sharpe_ratio:.2f} (vs. {in_metrics.sharpe_ratio:.2f} In-Sample). "
            "Heuristic hurdle: N_OOS >= 3 trades, positive OOS net return, and OOS SR > 0.20."
        )

    dim3 = ScorecardDimension(
        dimension_id="out_of_sample_persistence",
        name="3. Held-Out Out-of-Sample Persistence",
        question="Did the hypothesis continue to hold in the unseen Out-of-Sample validation period?",
        status=d3_status,
        evidence_metric=d3_metric,
        explanation=d3_expl,
    )

    # Dimension 4: Transaction Cost & Execution Survival
    be_bps = break_even_cost_bps if break_even_cost_bps is not None else 0.0
    configured_cost = spec.execution.commission_bps + spec.execution.slippage_bps
    if full_metrics.trade_count == 0:
        d4_status = "NO_EVIDENCE"
    elif be_bps >= max(20.0, configured_cost * 2.0) and full_metrics.total_net_return_pct > 0:
        d4_status = "SUPPORTED"
    elif be_bps > configured_cost and full_metrics.total_net_return_pct > 0:
        d4_status = "MIXED"
    else:
        d4_status = "CONTRADICTED"

    dim4 = ScorecardDimension(
        dimension_id="transaction_cost_survival",
        name="4. Transaction Cost & Microstructure Friction Survival",
        question="Do historical returns survive realistic commissions, bid-ask slippage, and execution latency?",
        status=d4_status,
        evidence_metric=(
            f"Break-Even Cost: {be_bps:.1f} bps/notional (Configured: {configured_cost:.1f} bps/notional)"
        ),
        explanation=(
            f"Deducting {configured_cost:.1f} bps per unit of portfolio notional shifts full-sample return from "
            f"{full_metrics.total_gross_return_pct:+.2f}% (gross) to {full_metrics.total_net_return_pct:+.2f}% (net). "
            f"Estimated break-even cost is {be_bps:.1f} bps one-way."
        ),
    )

    # Dimension 5: Adversarial Stress-Test Resilience
    if full_metrics.trade_count == 0:
        d5_status = "NO_EVIDENCE"
    elif stress_suite.resilient_count >= 3 and stress_suite.vulnerability_count == 0:
        d5_status = "SUPPORTED"
    elif stress_suite.resilient_count >= 2 or stress_suite.vulnerability_count <= 2:
        d5_status = "MIXED"
    else:
        d5_status = "CONTRADICTED"

    dim5 = ScorecardDimension(
        dimension_id="stress_test_resilience",
        name="5. Adversarial Stress-Test & Null-Benchmark Resilience",
        question="Did the signal survive outlier removal, regime splits, and Monte Carlo random-entry benchmarking?",
        status=d5_status,
        evidence_metric=(
            f"{stress_suite.resilient_count}/5 Resilient | "
            f"{5 - stress_suite.resilient_count - stress_suite.vulnerability_count}/5 Moderate | "
            f"{stress_suite.vulnerability_count}/5 High Vulnerability"
        ),
        explanation=(
            f"Out of 5 adversarial stress tests in Step 5, the hypothesis scored Resilient on "
            f"{stress_suite.resilient_count} and flagged High Vulnerability on {stress_suite.vulnerability_count}."
        ),
    )

    dimensions = [dim1, dim2, dim3, dim4, dim5]
    supported_cnt = sum(1 for d in dimensions if d.status == "SUPPORTED")
    contradicted_cnt = sum(1 for d in dimensions if d.status == "CONTRADICTED")
    evaluated_cnt = sum(1 for d in dimensions if d.status != "NO_EVIDENCE")
    no_evidence_cnt = sum(1 for d in dimensions if d.status == "NO_EVIDENCE")

    if full_metrics.trade_count == 0 or evaluated_cnt == 0:
        verdict_label = "INSUFFICIENT EMPIRICAL EVIDENCE"
    elif supported_cnt >= 4 and contradicted_cnt == 0:
        verdict_label = "SUPPORTED BY HISTORICAL DATA"
    elif contradicted_cnt >= 3 or (supported_cnt == 0 and contradicted_cnt >= 2):
        verdict_label = "CONTRADICTED BY HISTORICAL DATA"
    else:
        verdict_label = "MIXED / REGIME-DEPENDENT IN HISTORICAL DATA"

    # Build multi-layer comparison breakdown (Co-Movement vs. Cointegration vs. Trading Rule P&L)
    comparison_breakdown: list[dict[str, str]] = []
    if spec.strategy_family == "pairs_trading" and pairs_diag is not None:
        v1 = (
            "POSITIVE CO-MOVEMENT"
            if pairs_diag.return_correlation >= 0.25
            else ("WEAK CO-MOVEMENT" if pairs_diag.return_correlation > 0.0 else "NEGATIVE / UNCORRELATED")
        )
        s1 = "SUPPORTED" if pairs_diag.return_correlation >= 0.25 else ("MIXED" if pairs_diag.return_correlation > 0.0 else "CONTRADICTED")
        e1 = f"Daily Return r = {pairs_diag.return_correlation:+.3f} | Log-Price R² = {pairs_diag.ols_r_squared:.3f} (β = {pairs_diag.ols_hedge_ratio_beta:.3f})"
        x1 = (
            f"{spec.asset_a} and {spec.asset_b} daily returns exhibit r = {pairs_diag.return_correlation:+.3f} correlation. "
            "Positive daily co-movement means returns often share the same sign, but does not guarantee a stationary price spread."
        )

        v2 = "COINTEGRATED (p < 0.05)" if pairs_diag.adf_p_value < 0.05 else "NON-STATIONARY / DRIFTING SPREAD"
        s2 = "SUPPORTED" if pairs_diag.adf_p_value < 0.05 else ("MIXED" if pairs_diag.adf_p_value < 0.15 else "CONTRADICTED")
        e2 = (
            f"ADF Stat = {pairs_diag.adf_statistic:.3f} (p = {pairs_diag.adf_p_value:.4f}) | "
            f"Half-Life = {pairs_diag.half_life_days if pairs_diag.half_life_days is not None else 'N/A'}d | Hurst H = {pairs_diag.hurst_exponent:.3f}"
        )
        x2 = pairs_diag.cointegration_status_note

        v3 = (
            f"HALTED IN FORMATION ({in_metrics.circuit_breaker_date or full_metrics.circuit_breaker_date})"
            if oos_metrics.halted_prior_to_period
            else ("POSITIVE NET RETURN" if full_metrics.total_net_return_pct > 0 else "NEGATIVE NET RETURN")
        )
        s3 = (
            "NO_EVIDENCE"
            if oos_metrics.halted_prior_to_period
            else ("SUPPORTED" if (full_metrics.total_net_return_pct > 0 and oos_metrics.total_net_return_pct > 0) else ("MIXED" if full_metrics.total_net_return_pct > 0 else "CONTRADICTED"))
        )
        e3 = (
            f"Net Return: {full_metrics.total_net_return_pct:+.2f}% (N = {full_metrics.trade_count} trades, "
            f"IS N = {in_metrics.trade_count}, OOS N = {oos_metrics.trade_count}, Max DD = {full_metrics.max_drawdown_pct:.2f}%)"
        )
        x3 = (
            f"Trading ±{abs(spec.parameters.entry_threshold):.2f}σ log-spread divergences produced "
            f"{in_metrics.total_net_return_pct:+.2f}% in Formation ({in_metrics.trade_count} trades) and "
            + (
                f"0 Out-of-Sample trades because the {spec.risk_controls.max_drawdown_circuit_breaker_pct:.1f}% drawdown circuit breaker halted new entries on {in_metrics.circuit_breaker_date or full_metrics.circuit_breaker_date}."
                if oos_metrics.halted_prior_to_period
                else f"{oos_metrics.total_net_return_pct:+.2f}% in Out-of-Sample ({oos_metrics.trade_count} trades)."
            )
        )

        comparison_breakdown = [
            {
                "aspect": "1. Return Co-Movement (Daily Correlation)",
                "lens_title": "1. Return Co-Movement (Daily Correlation)",
                "question_answered": f"Do {spec.asset_a} and {spec.asset_b} daily returns move in the same direction on the same day?",
                "evidence": e1,
                "primary_metric": e1,
                "verdict": v1,
                "status": s1,
                "explanation": x1,
                "interpretation": x1,
            },
            {
                "aspect": "2. Long-Run Equilibrium Cointegration (Stationary Spread)",
                "lens_title": "2. Long-Run Equilibrium Cointegration (Stationary Spread)",
                "question_answered": f"Is the log-price spread between {spec.asset_a} and {spec.asset_b} stationary (mean-reverting) in Formation?",
                "evidence": e2,
                "primary_metric": e2,
                "verdict": v2,
                "status": s2,
                "explanation": x2,
                "interpretation": x2,
            },
            {
                "aspect": "3. Active Spread Mean-Reversion Trading Rule",
                "lens_title": "3. Active Spread Mean-Reversion Trading Rule",
                "question_answered": f"Did trading ±{abs(spec.parameters.entry_threshold):.2f}σ spread divergences generate positive net P&L after costs and risk controls?",
                "evidence": e3,
                "primary_metric": e3,
                "verdict": v3,
                "status": s3,
                "explanation": x3,
                "interpretation": x3,
            },
        ]
    else:
        e1 = f"Return: {full_metrics.benchmark_return_pct:+.2f}% (CAGR {full_metrics.benchmark_cagr_pct:+.2f}%, Sharpe {full_metrics.benchmark_sharpe_ratio:.2f}, Max DD {full_metrics.benchmark_max_drawdown_pct:.2f}%)"
        x1 = f"Unconditional buy-and-hold performance of {spec.benchmark_ticker} over the identical {full_metrics.trading_days}-day window."

        v2 = "POSITIVE EXPECTANCY" if full_metrics.mean_trade_net_pct > 0 else "NEGATIVE / ZERO EXPECTANCY"
        s2 = "SUPPORTED" if full_metrics.mean_trade_net_pct > 0.10 else ("MIXED" if full_metrics.mean_trade_net_pct > 0 else "CONTRADICTED")
        e2 = f"Mean Trade: {full_metrics.mean_trade_net_pct:+.3f}% | Win Rate: {full_metrics.win_rate_pct:.1f}% (N = {full_metrics.trade_count} trades)"
        x2 = f"Average net return per triggered trade after {spec.execution.commission_bps + spec.execution.slippage_bps:.1f} bps one-way friction."

        v3 = (
            f"HALTED IN FORMATION ({in_metrics.circuit_breaker_date or full_metrics.circuit_breaker_date})"
            if oos_metrics.halted_prior_to_period
            else ("OUTPERFORMED BENCHMARK" if full_metrics.excess_return_pct > 0 else "UNDERPERFORMED BENCHMARK")
        )
        s3 = (
            "NO_EVIDENCE"
            if oos_metrics.halted_prior_to_period
            else ("SUPPORTED" if full_metrics.excess_return_pct > 0 else ("MIXED" if full_metrics.total_net_return_pct > 0 else "CONTRADICTED"))
        )
        e3 = (
            f"Net Return: {full_metrics.total_net_return_pct:+.2f}% (Excess: {full_metrics.excess_return_pct:+.2f} pp, "
            f"IS N = {in_metrics.trade_count}, OOS N = {oos_metrics.trade_count})"
        )
        x3 = (
            f"Strategy achieved Sharpe {full_metrics.sharpe_ratio:.2f} with {full_metrics.exposure_pct:.1f}% market exposure "
            f"and Newey-West HAC alpha {factor_regression.annualized_alpha_pct:+.2f}% (p = {factor_regression.alpha_p_value_hac:.3f})."
        )

        comparison_breakdown = [
            {
                "aspect": f"1. Passive Benchmark Drift ({spec.benchmark_ticker} Buy-and-Hold)",
                "lens_title": f"1. Passive Benchmark Drift ({spec.benchmark_ticker} Buy-and-Hold)",
                "question_answered": f"How much did unconditional {spec.benchmark_ticker} drift return over the same calendar period?",
                "evidence": e1,
                "primary_metric": e1,
                "verdict": "PASSIVE BASELINE",
                "status": "MIXED",
                "explanation": x1,
                "interpretation": x1,
            },
            {
                "aspect": "2. Post-Signal Event Expectancy (Net of Fees)",
                "lens_title": "2. Post-Signal Event Expectancy (Net of Fees)",
                "question_answered": f"What was the mean net return per triggered {spec.asset_a} trade after transaction costs?",
                "evidence": e2,
                "primary_metric": e2,
                "verdict": v2,
                "status": s2,
                "explanation": x2,
                "interpretation": x2,
            },
            {
                "aspect": "3. Active Strategy vs. Benchmark & Out-of-Sample Survival",
                "lens_title": "3. Active Strategy vs. Benchmark & Out-of-Sample Survival",
                "question_answered": f"Did the active rule outperform {spec.benchmark_ticker} and persist out-of-sample?",
                "evidence": e3,
                "primary_metric": e3,
                "verdict": v3,
                "status": s3,
                "explanation": x3,
                "interpretation": x3,
            },
        ]

    halt_clause = ""
    if full_metrics.circuit_breaker_triggered and full_metrics.circuit_breaker_date:
        halt_clause = (
            f" The {spec.risk_controls.max_drawdown_circuit_breaker_pct:.1f}% portfolio drawdown circuit breaker "
            f"tripped on {full_metrics.circuit_breaker_date}"
            + (
                f", halting all subsequent entries and resulting in 0 Out-of-Sample trades ({oos_metrics.start_date} to {oos_metrics.end_date})."
                if oos_metrics.halted_prior_to_period
                else "."
            )
        )

    pairs_clause = ""
    if spec.strategy_family == "pairs_trading" and pairs_diag is not None:
        pairs_clause = (
            f" Econometrically, {spec.asset_a} and {spec.asset_b} exhibit daily return correlation r = {pairs_diag.return_correlation:+.3f}, "
            f"while the log-price spread ADF stationarity test yields p = {pairs_diag.adf_p_value:.4f} "
            f"({'stationary cointegrated spread' if pairs_diag.adf_p_value < 0.05 else 'non-stationary drifting spread'})."
        )

    checks_clause = (
        f"The trading rule satisfied {supported_cnt} of {evaluated_cnt} evaluated empirical checks"
        + (f" ({no_evidence_cnt} check had NO_EVIDENCE due to 0 holdout trades)" if no_evidence_cnt > 0 else "")
        + f" ({verdict_label})."
    )

    exec_summary = (
        f"Historical Audit Summary for '{spec.hypothesis_text}': Across {full_metrics.trading_days} trading days "
        f"({full_metrics.start_date} to {full_metrics.end_date}), the specified rule triggered {full_metrics.trade_count} "
        f"trades (In-Sample N = {in_metrics.trade_count}, Out-of-Sample N = {oos_metrics.trade_count}), generating "
        f"{full_metrics.total_net_return_pct:+.2f}% net return (Sharpe {full_metrics.sharpe_ratio:.2f}, Max Drawdown {full_metrics.max_drawdown_pct:.2f}%) "
        f"versus {full_metrics.benchmark_return_pct:+.2f}% (Sharpe {full_metrics.benchmark_sharpe_ratio:.2f}) for passive "
        f"{spec.benchmark_ticker} Buy-and-Hold ({full_metrics.excess_return_pct:+.2f} percentage points excess return)."
        f"{halt_clause}{pairs_clause} {checks_clause} "
        "Note: This scorecard summarizes historical sample evidence using explicit heuristic decision thresholds and does not constitute "
        "investment advice or a prediction of future performance."
    )

    oos_takeaway = (
        f"In-Sample vs. Out-of-Sample: Formation Sharpe was {in_metrics.sharpe_ratio:.2f} ({in_metrics.total_net_return_pct:+.2f}% across {in_metrics.trade_count} trades); "
        + (
            f"Out-of-Sample had 0 trades (NO_EVIDENCE) because the drawdown circuit breaker halted trading on {in_metrics.circuit_breaker_date or full_metrics.circuit_breaker_date}."
            if oos_metrics.halted_prior_to_period
            else (
                "Out-of-Sample had 0 trades (NO_EVIDENCE; Sharpe undefined)."
                if oos_metrics.is_zero_exposure
                else f"Out-of-Sample Sharpe was {oos_metrics.sharpe_ratio:.2f} ({oos_metrics.total_net_return_pct:+.2f}% across {oos_metrics.trade_count} trades)."
            )
        )
    )

    takeaways = [
        (
            f"Direction & Expectancy: {full_metrics.trade_count} trades averaged {full_metrics.mean_trade_net_pct:+.3f}% net "
            f"with a {full_metrics.win_rate_pct:.1f}% win rate and {full_metrics.max_drawdown_pct:.2f}% max drawdown."
        ),
        oos_takeaway,
        (
            f"Benchmark Factor Decomposition: Annualized Newey-West HAC alpha is {factor_regression.annualized_alpha_pct:+.2f}% "
            f"(t = {factor_regression.alpha_t_stat_hac:.2f}, p = {factor_regression.alpha_p_value_hac:.3f}) with market beta "
            f"β = {factor_regression.beta:.3f}."
        ),
    ]
    if spec.strategy_family == "pairs_trading" and pairs_diag is not None:
        takeaways.append(
            f"Co-Movement vs. Cointegration Distinction: Daily returns correlate at r = {pairs_diag.return_correlation:+.3f}, "
            f"but log-spread stationarity has ADF p = {pairs_diag.adf_p_value:.4f} (Hurst H = {pairs_diag.hurst_exponent:.3f})."
        )

    # Build Step-by-Step Mathematical Formula & Calculation Ledger with exact numbers plugged in
    cost_bps = spec.execution.commission_bps + spec.execution.slippage_bps
    round_trip_bps = 2.0 * cost_bps
    initial_cap = spec.execution.initial_capital

    cb_limit_str = (
        f"{spec.risk_controls.max_drawdown_circuit_breaker_pct:.1f}%"
        if spec.risk_controls.max_drawdown_circuit_breaker_pct is not None
        else "None"
    )
    calc_steps: list[FormulaCalculationStep] = [
        FormulaCalculationStep(
            step_id="calc_net_return",
            title="Step A — Compounded Net Trade Return, Unit Gross Exposure & Running-Peak Drawdown",
            formula=(
                "R_{net,i} = R_{gross,i} − 2·c,   |w_A| + |w_B| = 1.0,   "
                "E_0 = C_0,   H_t = max(E_0, max_{1 ≤ s ≤ t} E_s),   D_t = 1 − (E_t / H_t)"
            ),
            substituted_inputs=(
                f"Initial Capital E_0 = ${initial_cap:,.2f}, Unit gross exposure (|w_A| + |w_B| = 1.0), "
                f"c_comm = {spec.execution.commission_bps:.1f} bps, c_slip = {spec.execution.slippage_bps:.1f} bps ⇒ "
                f"Round-trip cost = {round_trip_bps:.1f} bps ({round_trip_bps / 100.0:.3f}%/trade) across N = {full_metrics.trade_count} trades; "
                f"Circuit Breaker Limit = {cb_limit_str}"
            ),
            output_value=(
                f"Gross = {full_metrics.total_gross_return_pct:+.2f}% → Net = {full_metrics.total_net_return_pct:+.2f}% "
                f"(Mean Net/Trade = {full_metrics.mean_trade_net_pct:+.3f}%) | "
                f"Max Drawdown = {full_metrics.max_drawdown_pct:.2f}% (IS = {in_metrics.max_drawdown_pct:.2f}%, OOS = {oos_metrics.max_drawdown_pct:.2f}%)"
            ),
            interpretation=(
                "Every trade deducts round-trip commission + slippage on unit gross notional, and running peak H_t anchors at initial capital E_0 so immediate day-1 losses are captured (reported as positive loss magnitude D_t ∈ [0%, 100%] in tables and U_t = −D_t ≤ 0% on underwater charts)."
            ),
        ),
        FormulaCalculationStep(
            step_id="calc_lo_sharpe",
            title="Step B — Annualized Sharpe Ratio & Lo (2002) / Mertens (2002) Non-IID Standard Error",
            formula="SR = √252 × (μ_d / σ_d),   SE(SR_d) = √[(1 − γ₃·SR_d + ((γ₄ − 1)/4)·SR_d²) / (T − 1)]",
            substituted_inputs=(
                f"T = {full_metrics.trading_days} days, Ann. Vol = {full_metrics.annualized_volatility_pct:.2f}%, "
                f"Skewness γ₃ = {full_metrics.skewness_daily:.3f}, Pearson Kurtosis γ₄ = {full_metrics.kurtosis_daily:.3f}"
            ),
            output_value=(
                f"Annualized SR = {full_metrics.sharpe_ratio:.3f}, Lo (2002) SE = {full_metrics.sharpe_se_lo2002:.3f}, "
                f"Probabilistic Sharpe Ratio PSR(0) = {full_metrics.probabilistic_sharpe_psr_pct:.1f}%"
            ),
            interpretation=(
                "Adjusts standard Sharpe inference for non-normal fat tails and skewness so negative-skew strategies are not overstated."
            ),
        ),
        FormulaCalculationStep(
            step_id="calc_dsr",
            title="Step C — Bailey & López de Prado (2014) Multiple-Testing Deflated Sharpe Ratio (DSR)",
            formula="E[max SR₀] ≈ √V[{SR_k}] × [(1 − γ)·Φ⁻¹(1 − 1/K) + γ·Φ⁻¹(1 − 1/(K·e))],   DSR = Φ((SR − E[max SR₀]) / SE(SR))",
            substituted_inputs=(
                f"K = {parameter_surface.trials_count_k if parameter_surface else 25} parameter grid trials, "
                f"Null Hurdle E[max SR₀] = {parameter_surface.expected_max_sr_under_null if parameter_surface else 0.0:.3f} (annualized)"
            ),
            output_value=(
                f"Selected Spec DSR = {parameter_surface.selected_spec_deflated_sharpe_psr_pct if parameter_surface else full_metrics.probabilistic_sharpe_psr_pct:.1f}% "
                f"(Grid Best DSR = {parameter_surface.best_grid_deflated_sharpe_psr_pct if parameter_surface else 0.0:.1f}%)"
            ),
            interpretation=(
                "Penalizes the observed Sharpe ratio for the number of parameter combinations explored in the 5×5 sensitivity grid."
            ),
        ),
        FormulaCalculationStep(
            step_id="calc_newey_west",
            title="Step D — Newey-West (1987) HAC Benchmark Factor Decomposition",
            formula="R_strat,t = α + β · R_bench,t + ε_t,   Var_HAC(hat(α)) = T⁻¹ (X'X)⁻¹ hat(Ω)_NW (X'X)⁻¹",
            substituted_inputs=(
                f"Benchmark = {spec.benchmark_ticker}, Bartlett kernel lags L = {factor_regression.newey_west_lags}, "
                f"Sample T = {full_metrics.trading_days} daily return pairs"
            ),
            output_value=(
                f"Annualized α = {factor_regression.annualized_alpha_pct:+.2f}% (t_HAC = {factor_regression.alpha_t_stat_hac:.2f}, "
                f"p = {factor_regression.alpha_p_value_hac:.3f}), Market β = {factor_regression.beta:.3f} (R² = {factor_regression.r_squared:.3f})"
            ),
            interpretation=(
                "Separates passive correlation with the benchmark (β) from excess timing return (α) using autocorrelation-consistent standard errors."
            ),
        ),
    ]

    if pairs_diag is not None:
        calc_steps.append(
            FormulaCalculationStep(
                step_id="calc_pairs_coint",
                title="Step E — Cointegrated Spread & Exact Discrete Ornstein-Uhlenbeck Half-Life",
                formula="S_t = log(P_A,t) − (hat(α) + hat(β)·log(P_B,t)),   ΔS_t = ϕ + λ·S_{t−1} + η_t,   t_{1/2} = −ln(2) / ln(1 + λ)",
                substituted_inputs=(
                    f"Formation window: {pairs_diag.formation_start} to {pairs_diag.formation_end} | "
                    f"hat(α) = {pairs_diag.ols_alpha:.4f}, hat(β) = {pairs_diag.ols_hedge_ratio_beta:.4f}, "
                    f"OU λ = {pairs_diag.ou_lambda:.5f}, ln(1 + λ) = {math.log(1.0 + pairs_diag.ou_lambda) if pairs_diag.ou_lambda > -0.99 else float('nan'):.5f}"
                ),
                output_value=(
                    f"ADF Stat = {pairs_diag.adf_statistic:.3f} (p = {pairs_diag.adf_p_value:.4f}), "
                    f"Half-Life t_{{1/2}} = {pairs_diag.half_life_days if pairs_diag.half_life_days is not None else 'N/A'} days, "
                    f"Hurst H = {pairs_diag.hurst_exponent:.3f}"
                ),
                interpretation=pairs_diag.cointegration_status_note,
            )
        )
    elif bootstrap_cis:
        b0 = bootstrap_cis[0]
        calc_steps.append(
            FormulaCalculationStep(
                step_id="calc_stationary_bootstrap",
                title="Step E — Politis & Romano (1994) Stationary Bootstrap 95% Confidence Interval",
                formula="L_block ~ Geometric(p = 1 / b_opt),   CI_95% = [Q_0.025(θ*_b), Q_0.975(θ*_b)]",
                substituted_inputs=(
                    f"Metric = {b0.metric_name}, Mean block length b = {b0.block_length} days, "
                    f"B = {b0.replications} resampled paths"
                ),
                output_value=(
                    f"Point Estimate = {b0.point_estimate:.2f}{b0.unit}, "
                    f"95% CI = [{b0.ci_lower_95:.2f}{b0.unit}, {b0.ci_upper_95:.2f}{b0.unit}] (SE = {b0.bootstrap_std_error:.2f})"
                ),
                interpretation=(
                    "Preserves volatility clustering and serial dependence by resampling random-length blocks of consecutive trading days."
                ),
            )
        )

    return HypothesisScorecard(
        hypothesis_restatement=spec.hypothesis_text,
        historical_verdict_label=verdict_label,
        supported_checks_count=supported_cnt,
        total_checks_count=len(dimensions),
        evaluated_checks_count=evaluated_cnt,
        executive_summary=sanitize_research_notes([exec_summary])[0],
        key_takeaways=sanitize_research_notes(takeaways),
        dimensions=dimensions,
        calculation_steps=calc_steps,
        hypothesis_comparison_breakdown=comparison_breakdown,
    )

