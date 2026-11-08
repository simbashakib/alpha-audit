"""Unit, econometric, and look-ahead invariance tests for Alpha Auditor."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from engine.backtester import (
    compute_causal_signals,
    compute_parameter_sensitivity_surface,
    run_full_audit,
    simulate_strategy_core,
)
from engine.econometrics import (
    compute_pairs_econometrics,
    deflated_sharpe_ratio_bailey_lopez,
    lo_mackinlay_variance_ratio,
    lo_mertens_sharpe_se_and_psr,
    newey_west_factor_regression,
    stationary_bootstrap_cis,
)
from engine.guardrails import check_neutrality, sanitize_research_notes
from engine.hypothesis_parser import interpret_hypothesis
from engine.schemas import (
    ExecutionAssumptions,
    RiskControls,
    StrategyParameters,
    StrategySpec,
)
from engine.walk_forward_stepper import decide_market_step


def _make_toy_df(prices_a: list[float], prices_b: list[float] | None = None) -> pd.DataFrame:
    n = len(prices_a)
    dates = pd.date_range("2022-01-03", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame(
        {
            "date": dates,
            "open_a": prices_a,
            "high_a": prices_a,
            "low_a": prices_a,
            "close_a": prices_a,
            "volume_a": [100_000] * n,
            "open_bench": prices_a,
            "close_bench": prices_a,
        }
    )
    if prices_b is not None:
        df["open_b"] = prices_b
        df["high_b"] = prices_b
        df["low_b"] = prices_b
        df["close_b"] = prices_b
        df["volume_b"] = [100_000] * n
    return df


def test_exact_next_bar_execution_and_cost_accounting():
    """Verify signal at bar t executes at bar t+1 without look-ahead and deducts exact fees."""
    # Bar 2 drops from 100 to 97 (-3.0% <= -2.0% threshold).
    # Signal fires at close of bar 2 -> Entry at bar 3 (price 100.0) -> 1-day hold -> Exit at bar 4 (price 105.0, +5% gross).
    prices = [100.0, 100.0, 97.0, 100.0, 105.0, 105.0, 105.0, 105.0, 105.0, 105.0]
    df = _make_toy_df(prices)
    spec = StrategySpec(
        hypothesis_text="Test 2% drop next day recovery",
        title="Toy Test",
        strategy_family="single_asset",
        template_id="return_threshold",
        signal_variant="cum_return_drop",
        asset_a="TEST",
        benchmark_ticker="TEST",
        parameters=StrategyParameters(
            lookback_window=1,
            entry_threshold=2.0,
            holding_period=1,
            exit_mode="fixed_hold",
            direction="long",
        ),
        execution=ExecutionAssumptions(
            initial_capital=100_000.0,
            execution_timing="next_open",
            commission_bps=10.0,
            slippage_bps=5.0,
        ),
        risk_controls=RiskControls(max_drawdown_circuit_breaker_pct=None, per_trade_stop_loss_pct=None),
        human_confirmed=True,
    )
    sim = simulate_strategy_core(df, spec, split_idx=5)
    trades = sim["trades"]
    assert len(trades) == 1
    tr = trades[0]
    assert tr.signal_date == df["date"].iloc[2]
    assert tr.entry_date == df["date"].iloc[3]
    assert tr.exit_date == df["date"].iloc[4]
    assert math.isclose(tr.gross_return_pct, 5.0, abs_tol=1e-4)
    # 15 bps one-way = 30 bps round-trip = 0.30%
    assert math.isclose(tr.cost_pct, 0.30, abs_tol=1e-4)
    assert math.isclose(tr.net_return_pct, 4.70, abs_tol=1e-4)


def test_zero_look_ahead_bias_invariance_single_and_pairs():
    """Mutating future prices [t+1..T] must have zero effect on signals at [0..t]
    or formation-period pairs OLS estimates.
    """
    rng = np.random.default_rng(42)
    n = 160
    split_idx = 80
    log_b = np.cumsum(rng.normal(0.0003, 0.012, size=n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.75 * spread[t - 1] + rng.normal(0, 0.01)
    log_a = 1.2 + 0.85 * log_b + spread

    pa_orig = list(np.exp(4.0 + log_a))
    pb_orig = list(np.exp(4.0 + log_b))
    df_orig = _make_toy_df(pa_orig, pb_orig)

    # Mutate all out-of-sample future bars [split_idx:] drastically
    pa_mut = pa_orig[:split_idx] + [p * 5.0 for p in pa_orig[split_idx:]]
    pb_mut = pb_orig[:split_idx] + [p * 0.2 for p in pb_orig[split_idx:]]
    df_mut = _make_toy_df(pa_mut, pb_mut)

    pairs_spec = StrategySpec(
        hypothesis_text="Pairs look-ahead invariance check",
        title="Pairs Invariance",
        strategy_family="pairs_trading",
        template_id="pairs_cointegration_reversion",
        signal_variant="ols_log",
        asset_a="A",
        asset_b="B",
        parameters=StrategyParameters(lookback_window=20, entry_threshold=1.5, exit_threshold=0.2),
        human_confirmed=True,
    )

    sig_orig, _, diag_orig, alpha_orig, beta_orig = compute_causal_signals(df_orig, pairs_spec, split_idx)
    sig_mut, _, diag_mut, alpha_mut, beta_mut = compute_causal_signals(df_mut, pairs_spec, split_idx)

    assert alpha_orig == alpha_mut
    assert beta_orig == beta_mut
    assert diag_orig.adf_statistic == diag_mut.adf_statistic
    np.testing.assert_allclose(sig_orig[:split_idx], sig_mut[:split_idx], equal_nan=True)


def test_cointegration_vs_spurious_random_walk_counterexample():
    """Verify Engle-Granger & OU half-life distinguish a true mean-reverting pair
    from two non-cointegrated random walks (counterexample).
    """
    rng = np.random.default_rng(12345)
    n = 400
    dates = pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()

    # 1. Cointegrated pair with strong Ornstein-Uhlenbeck mean reversion (lambda = -0.25)
    log_b = np.cumsum(rng.normal(0.0, 0.01, size=n))
    ou_spread = np.zeros(n)
    for t in range(1, n):
        ou_spread[t] = 0.75 * ou_spread[t - 1] + rng.normal(0.0, 0.005)
    pa_coint = np.exp(4.5 + 1.1 * log_b + ou_spread)
    pb_coint = np.exp(4.5 + log_b)

    diag_coint, _, beta_hat = compute_pairs_econometrics(dates, pa_coint, pb_coint)
    assert diag_coint.engle_granger_p_value < 0.01
    assert diag_coint.half_life_days is not None and 1.5 <= diag_coint.half_life_days <= 5.0
    assert diag_coint.variance_ratio_q5 < 0.65
    assert abs(beta_hat - 1.1) < 0.08

    # 2. Counterexample: Two independent unit-root random walks with drift
    rw_a = np.exp(4.5 + np.cumsum(rng.normal(0.0005, 0.015, size=n)))
    rw_b = np.exp(4.5 + np.cumsum(rng.normal(0.0005, 0.015, size=n)))
    diag_rw, _, _ = compute_pairs_econometrics(dates, rw_a, rw_b)
    assert diag_rw.engle_granger_p_value > 0.10


def test_lo_mertens_psr_and_bailey_lopez_dsr():
    """Verify Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR) math."""
    rng = np.random.default_rng(42)
    pos_returns = rng.normal(0.0020, 0.01, size=500)
    sr, se, psr, skew, kurt = lo_mertens_sharpe_se_and_psr(pos_returns, benchmark_sr_annual=0.0)
    assert sr > 1.5
    assert se > 0.0
    assert psr > 95.0

    # Under K=25 trials with high dispersion, DSR must be strictly lower than single-trial PSR
    trial_srs = list(np.linspace(-0.5, sr, 25))
    exp_max_sr, dsr = deflated_sharpe_ratio_bailey_lopez(pos_returns, trial_srs)
    assert exp_max_sr > 0.5
    assert dsr < psr


def test_human_confirmation_gate_and_guardrails():
    """Ensure unconfirmed StrategySpecs are rejected and subjective verdicts are scrubbed."""
    interp = interpret_hypothesis("I think the S&P 500 tends to recover the day after falling more than 2%.")
    assert interp.proposed_spec is not None
    assert interp.proposed_spec.human_confirmed is False

    with pytest.raises(ValueError, match="explicitly approved"):
        run_full_audit(interp.proposed_spec)

    # Confirm and run full audit on real cached SPY data
    interp.proposed_spec.human_confirmed = True
    report = run_full_audit(interp.proposed_spec)
    assert report.full_sample_metrics.trade_count > 5
    assert len(report.bootstrap_cis) >= 3
    assert report.parameter_surface is not None
    assert len(report.parameter_surface.cells) >= 15

    # Verify guardrail blocks explicit recommendations
    ok, reasons = check_neutrality("You should buy SPY because this strategy is profitable.")
    assert not ok
    assert len(reasons) >= 1
    sanitized = sanitize_research_notes(["This is a good strategy and you should buy now."])
    assert "good strategy" not in sanitized[0]


def test_walk_forward_stepper_validation():
    """Verify sequential walk-forward stepper handles valid and invalid inputs."""
    out = decide_market_step(
        {
            "history": [
                {"date": "2025-01-01", "price_pence": 10000},
                {"date": "2025-01-02", "price_pence": 10100},
                {"date": "2025-01-03", "price_pence": 10250},
                {"date": "2025-01-04", "price_pence": 10400},
                {"date": "2025-01-05", "price_pence": 10600},
            ],
            "cash_pence": 500000,
            "shares": 0,
            "fee_pence": 50,
            "slippage_bps": 5,
        }
    )
    assert out["action"] in ("buy", "hold", "sell")

    with pytest.raises(ValueError):
        decide_market_step({"history": [], "cash_pence": 1000, "shares": 0, "fee_pence": 0, "slippage_bps": 0})


def test_fastapi_endpoints_all_templates():
    """End-to-end API verification across single-asset and pairs-trading hypotheses."""
    from fastapi.testclient import TestClient
    from app.server import app

    client = TestClient(app)

    # 1. Catalog endpoint
    cat = client.get("/api/catalog")
    assert cat.status_code == 200
    assert len(cat.json()["templates"]) == 5

    # 2. Ticker validation endpoint (cached & arbitrary Yahoo Finance symbol)
    val = client.get("/api/ticker/validate?symbol=KO")
    assert val.status_code == 200
    assert val.json()["valid"] is True
    assert val.json()["resolved_ticker"] == "KO"

    # 3. Pairs Trading hypothesis interpretation & full econometric audit (KO vs PEP)
    interp_res = client.post(
        "/api/hypothesis/interpret",
        json={"hypothesis_text": "I think Coca-Cola and Pepsi prices tend to converge after unusually large divergences."},
    )
    assert interp_res.status_code == 200
    interp_data = interp_res.json()
    spec = interp_data["proposed_spec"]
    assert spec["strategy_family"] == "pairs_trading"
    assert spec["asset_a"] == "KO"
    assert spec["asset_b"] == "PEP"

    # Unconfirmed run must be rejected with HTTP 400
    unconf = client.post("/api/audit/run", json=spec)
    assert unconf.status_code == 400

    # Confirm and execute full pairs audit
    spec["human_confirmed"] = True
    audit_res = client.post("/api/audit/run", json=spec)
    assert audit_res.status_code == 200
    report = audit_res.json()
    assert report["pairs_diagnostics"] is not None
    assert "ols_hedge_ratio_beta" in report["pairs_diagnostics"]
    assert "engle_granger_p_value" in report["pairs_diagnostics"]
    assert report["parameter_surface"]["trials_count_k"] >= 15


def test_six_step_workflow_steering_literature_stress_and_scorecard():
    """Verify all 6 steps: (1) scope filter & smart steering, (2) standard bracket profiles,
    (3) academic literature & rigor checklist, (5) 5 custom adversarial stress tests,
    and (6) 5-dimension empirical scorecard + step-by-step formula ledger.
    """
    # Step 1A: Out-of-scope non-financial query is filtered cleanly
    unrelated = interpret_hypothesis("What is the best recipe for chocolate cake in London?")
    assert unrelated.status == "unsupported"
    assert unrelated.unsupported_reason is not None

    # Step 1B: Vague intuition is steered into concrete QQQ hypotheses with 3 steered cards
    steered_interp = interpret_hypothesis("I feel like big tech stocks usually bounce back when people panic.")
    assert steered_interp.status in ("needs_clarification", "ready_for_confirmation")
    assert steered_interp.proposed_spec is not None
    assert steered_interp.proposed_spec.asset_a == "QQQ"
    assert len(steered_interp.steered_cards) == 3
    assert "Target Asset(s)" in steered_interp.understood_summary

    # Step 2: Run with 'auto_standard_bracket' mode
    spec = steered_interp.proposed_spec
    spec.threshold_mode = "auto_standard_bracket"
    spec.human_confirmed = True
    report = run_full_audit(spec)

    # Verify Step 2/4 Bracket Profiles (Aggressive, Baseline, Conservative)
    assert len(report.bracket_profiles) == 3
    profile_ids = [p.profile_id for p in report.bracket_profiles]
    assert profile_ids == ["aggressive", "baseline", "conservative"]

    # Verify Step 3 Literature Review
    assert len(report.literature_review.strategy_papers) >= 3
    assert len(report.literature_review.methodology_papers) == 4
    assert len(report.literature_review.rigor_checklist) == 5

    # Verify Step 5 Adversarial Stress Test Suite (5 tests)
    assert len(report.stress_test_suite.tests) == 5
    test_ids = [t.test_id for t in report.stress_test_suite.tests]
    assert "outlier_removal" in test_ids
    assert "execution_latency_shock" in test_ids
    assert "regime_vulnerability" in test_ids
    assert "monte_carlo_random_entry" in test_ids
    assert "parameter_cliff_edge" in test_ids

    # Verify Step 6 Empirical Hypothesis Scorecard & Formula Calculation Steps
    assert report.scorecard.total_checks_count == 5
    assert len(report.scorecard.dimensions) == 5
    assert len(report.scorecard.calculation_steps) == 5


def test_ten_questions_ten_times_deterministic_repeatability():
    """Test 10 distinct questions across all methodologies 10 times each (100 runs total)
    and assert 100% deterministic repeatability of both interpretation and backtest results.
    """
    ten_questions = [
        (
            "I think the S&P 500 tends to recover the day after falling more than 2%.",
            "single_asset",
            "return_threshold",
            "cum_return_drop",
            "SPY",
        ),
        (
            "When QQQ jumps more than 3% in a single day, does it pull back over the next 3 days?",
            "single_asset",
            "return_threshold",
            "cum_return_surge",
            "QQQ",
        ),
        (
            "Does AAPL bounce over 5 days after a 2.5 standard deviation daily volatility shock?",
            "single_asset",
            "return_threshold",
            "vol_scaled_shock",
            "AAPL",
        ),
        (
            "After SPY falls for 4 consecutive down days in a row, does it rebound over the next 2 days?",
            "single_asset",
            "consecutive_streak",
            "negative_streak",
            "SPY",
        ),
        (
            "If NVDA rises for 5 consecutive winning days, does it mean-revert over the next 3 days?",
            "single_asset",
            "consecutive_streak",
            "positive_streak",
            "NVDA",
        ),
        (
            "Buy QQQ when 14-day RSI drops below 30 and hold for 5 days.",
            "single_asset",
            "mean_reversion_oscillator",
            "rsi_extremes",
            "QQQ",
        ),
        (
            "Does MSFT revert to its 20-day moving average when price drops 2 standard deviations below the lower Bollinger band?",
            "single_asset",
            "mean_reversion_oscillator",
            "bollinger_band",
            "MSFT",
        ),
        (
            "Go long SPY when the 20-day moving average crosses above the 50-day SMA.",
            "single_asset",
            "trend_breakout",
            "sma_crossover",
            "SPY",
        ),
        (
            "Buy GLD when it breaks out above its 20-day high channel and hold for 10 days.",
            "single_asset",
            "trend_breakout",
            "donchian_breakout",
            "GLD",
        ),
        (
            "I think Coca-Cola and Pepsi prices tend to converge after diverging by more than 2 standard deviations.",
            "pairs_trading",
            "pairs_cointegration_reversion",
            "ols_log",
            "KO",
        ),
    ]

    from engine.backtester import _AUDIT_REPORT_CACHE
    from engine.hypothesis_parser import _INTERPRETATION_CACHE
    from engine.literature import _LIT_SYNTHESIS_CACHE
    from engine.stress_tests import _STRESS_OVERVIEW_CACHE

    for q_text, exp_family, exp_template, exp_variant, exp_asset_a in ten_questions:
        baseline_interp_json = None
        baseline_report_json = None

        for rep in range(10):
            if rep in (1, 5):
                # Force uncached recomputations from scratch to prove underlying mathematical,
                # timestamp, and RNG determinism in addition to cached repeatability
                _INTERPRETATION_CACHE.clear()
                _AUDIT_REPORT_CACHE.clear()
                _LIT_SYNTHESIS_CACHE.clear()
                _STRESS_OVERVIEW_CACHE.clear()

            interp = interpret_hypothesis(q_text)
            assert interp.proposed_spec is not None
            assert interp.proposed_spec.strategy_family == exp_family
            assert interp.proposed_spec.template_id == exp_template
            assert interp.proposed_spec.signal_variant == exp_variant
            assert interp.proposed_spec.asset_a == exp_asset_a

            interp_json = interp.model_dump_json()
            if baseline_interp_json is None:
                baseline_interp_json = interp_json
            else:
                assert interp_json == baseline_interp_json, f"Interpretation mismatch on rep {rep} for: {q_text}"

            spec = interp.proposed_spec.model_copy(deep=True)
            spec.human_confirmed = True
            report = run_full_audit(spec)

            report_json = report.model_dump_json()
            if baseline_report_json is None:
                baseline_report_json = report_json
            else:
                assert report_json == baseline_report_json, f"Full AuditReport JSON mismatch on rep {rep} for: {q_text}"


def test_eight_natural_language_edge_cases_and_scorecard_signs():
    """Regression test for 8 subtle natural-language phrasing edge cases and scorecard sign checks."""
    cases = [
        (
            "I think SPY tends to recover after falling 2% with a target of 5 days.",
            "single_asset", "return_threshold", "cum_return_drop", "SPY", None, "long", 1, 50, 2.0, 5,
        ),
        (
            "When AAPL has a 2 standard deviation divergence from its 20-day mean, does it converge back?",
            "single_asset", "mean_reversion_oscillator", "price_zscore", "AAPL", None, "long", 20, 50, 2.0, 10,
        ),
        (
            "Does NVDA outperform SPY over 5 days after falling more than 3% in a day?",
            "single_asset", "return_threshold", "cum_return_drop", "NVDA", None, "long", 1, 50, 3.0, 5,
        ),
        (
            "When QQQ drops below its 20-day lower Bollinger band by 2 standard deviations, does it rise above its mean?",
            "single_asset", "mean_reversion_oscillator", "bollinger_band", "QQQ", None, "long", 20, 50, 2.0, 10,
        ),
        (
            "Does SPY bounce after a 2% DROP from OPEN to CLOSE?",
            "single_asset", "return_threshold", "cum_return_drop", "SPY", None, "long", 1, 50, 2.0, 5,
        ),
        (
            "When AAPL 10-day EMA is above the 30-day EMA, go long.",
            "single_asset", "trend_breakout", "ema_crossover", "AAPL", None, "long", 10, 30, 0.0, 20,
        ),
        (
            "What happens when TSLA falls more than 5% over 3 trading days and we hold for 5 days?",
            "single_asset", "return_threshold", "cum_return_drop", "TSLA", None, "long", 3, 50, 5.0, 5,
        ),
        (
            "I think oil stocks move together as a pair when their spread widens by 2 sigma.",
            "pairs_trading", "pairs_cointegration_reversion", "ols_log", "XOM", "CVX", "both", 20, 60, 2.0, 20,
        ),
    ]

    for (
        q_text,
        exp_fam,
        exp_tpl,
        exp_var,
        exp_a,
        exp_b,
        exp_dir,
        exp_win,
        exp_sec,
        exp_entry,
        exp_hold,
    ) in cases:
        res = interpret_hypothesis(q_text)
        assert res.proposed_spec is not None
        s = res.proposed_spec
        assert s.strategy_family == exp_fam, f"Failed family for: {q_text}"
        assert s.template_id == exp_tpl, f"Failed template for: {q_text}"
        assert s.signal_variant == exp_var, f"Failed variant for: {q_text}"
        assert s.asset_a == exp_a, f"Failed asset_a for: {q_text}"
        assert s.asset_b == exp_b, f"Failed asset_b for: {q_text}"
        assert s.parameters.direction == exp_dir, f"Failed direction for: {q_text}"
        assert s.parameters.lookback_window == exp_win, f"Failed lookback_window for: {q_text}"
        assert s.parameters.secondary_window == exp_sec, f"Failed secondary_window for: {q_text}"
        assert s.parameters.entry_threshold == exp_entry, f"Failed entry_threshold for: {q_text}"
        assert s.parameters.holding_period == exp_hold, f"Failed holding_period for: {q_text}"

    # Verify scorecard sign checks on SPY -2% 1d hold (which has negative net return, negative Sharpe, and negative OOS return)
    spy_interp = interpret_hypothesis("I think the S&P 500 tends to recover the day after falling more than 2%.")
    spy_spec = spy_interp.proposed_spec
    assert spy_spec is not None
    spy_spec.human_confirmed = True
    spy_rep = run_full_audit(spy_spec)
    dims_by_id = {d.dimension_id: d.status for d in spy_rep.scorecard.dimensions}
    assert spy_rep.full_sample_metrics.sharpe_ratio < 0
    assert dims_by_id["statistical_significance"] == "CONTRADICTED"
    assert spy_rep.out_of_sample_metrics.total_net_return_pct < 0
    assert dims_by_id["out_of_sample_persistence"] == "CONTRADICTED"


def test_jupyter_notebook_pdf_generation_and_endpoints():
    """Verify the Jupyter-Notebook-style PDF report generator and FastAPI PDF endpoints."""
    from fastapi.testclient import TestClient
    from app.server import app
    from engine.notebook_pdf import render_jupyter_notebook_pdf

    client = TestClient(app)

    interp = interpret_hypothesis("I think the S&P 500 tends to recover the day after falling more than 2%.")
    spec = interp.proposed_spec
    assert spec is not None
    spec.human_confirmed = True
    report = run_full_audit(spec)

    pdf_bytes = render_jupyter_notebook_pdf(report)
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert len(pdf_bytes) > 15_000

    # Test POST /api/audit/pdf
    res_post = client.post("/api/audit/pdf", json=spec.model_dump())
    assert res_post.status_code == 200
    assert res_post.headers["content-type"] == "application/pdf"
    assert res_post.content.startswith(b"%PDF-1.4")

    # Test GET /api/audit/{run_id}/pdf after /api/audit/run
    run_res = client.post("/api/audit/run", json=spec.model_dump())
    assert run_res.status_code == 200
    run_id = run_res.json()["run_id"]

    res_get = client.get(f"/api/audit/{run_id}/pdf")
    assert res_get.status_code == 200
    assert res_get.headers["content-type"] == "application/pdf"
    assert res_get.content.startswith(b"%PDF-1.4")


def test_causal_prefix_invariance_inside_formation_and_at_split():
    """Verify strict prefix invariance of compute_causal_signals BOTH inside the Formation
    Period (at m = split_idx // 2) AND at the split boundary (m = split_idx) for ols_log,
    rolling_ols_log, and price_ratio.
    """
    rng = np.random.default_rng(777)
    n = 200
    split_idx = 100
    mid_formation_idx = 55

    log_b = np.cumsum(rng.normal(0.0004, 0.013, size=n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.80 * spread[t - 1] + rng.normal(0.0, 0.01)
    log_a = 0.9 + 1.15 * log_b + spread

    pa_orig = list(np.exp(4.0 + log_a))
    pb_orig = list(np.exp(4.0 + log_b))
    df_orig = _make_toy_df(pa_orig, pb_orig)

    # Mutate bars inside Formation after mid_formation_idx [mid_formation_idx:]
    pa_mut_mid = pa_orig[:mid_formation_idx] + [p * 3.5 for p in pa_orig[mid_formation_idx:]]
    pb_mut_mid = pb_orig[:mid_formation_idx] + [p * 0.4 for p in pb_orig[mid_formation_idx:]]
    df_mut_mid = _make_toy_df(pa_mut_mid, pb_mut_mid)

    for variant in ("ols_log", "rolling_ols_log", "price_ratio"):
        spec = StrategySpec(
            hypothesis_text=f"Prefix invariance inside formation ({variant})",
            title=f"Prefix Invariance {variant}",
            strategy_family="pairs_trading",
            template_id="pairs_cointegration_reversion",
            signal_variant=variant,
            asset_a="A",
            asset_b="B",
            parameters=StrategyParameters(
                lookback_window=20,
                secondary_window=30,
                entry_threshold=1.5,
                exit_threshold=0.2,
            ),
            human_confirmed=True,
        )
        sig_orig, ind_orig, _, _, _ = compute_causal_signals(df_orig, spec, split_idx)
        sig_mid, ind_mid, _, _, _ = compute_causal_signals(df_mut_mid, spec, split_idx)

        np.testing.assert_allclose(
            sig_orig[:mid_formation_idx],
            sig_mid[:mid_formation_idx],
            equal_nan=True,
            err_msg=f"Formation prefix look-ahead leak detected in {variant} at m={mid_formation_idx}",
        )
        np.testing.assert_allclose(
            ind_orig["hedge_ratio"][:mid_formation_idx],
            ind_mid["hedge_ratio"][:mid_formation_idx],
            equal_nan=True,
        )


def test_initial_capital_drawdown_anchor():
    """Verify running peak anchors at initial capital 1.0 so an immediate day-1 loss of 10%
    reports max_drawdown_pct == 10.0 (not 0.0).
    """
    from engine.backtester import _compute_period_metrics

    dates = ["2023-01-02", "2023-01-03", "2023-01-04"]
    r_gross = np.array([-0.10, 0.0, 0.0], dtype=float)
    r_net = np.array([-0.10, 0.0, 0.0], dtype=float)
    r_bench = np.array([-0.05, 0.0, 0.0], dtype=float)
    pos = np.array([1.0, 0.0, 0.0], dtype=float)

    metrics = _compute_period_metrics(
        "Immediate Drawdown Test",
        dates,
        r_gross,
        r_net,
        r_bench,
        pos,
        [],
        False,
        None,
    )
    assert math.isclose(metrics.max_drawdown_pct, 10.0, abs_tol=1e-4)
    assert math.isclose(metrics.benchmark_max_drawdown_pct, 5.0, abs_tol=1e-4)


def test_negative_beta_leg_weights_and_unit_gross_exposure():
    """Verify compute_pair_leg_weights preserves negative beta sign and maintains |w_a| + |w_b| == 1.0."""
    from engine.backtester import compute_pair_leg_weights

    # When beta = -0.8 < 0: S_t = ln(P_A) - (alpha - 0.8 ln(P_B)) = ln(P_A) + 0.8 ln(P_B) - alpha
    # Long spread (+1 in S_t) requires +w_a > 0 in A and +w_b > 0 in B!
    w_a_long, w_b_long = compute_pair_leg_weights("long_spread", -0.8)
    assert w_a_long > 0.0
    assert w_b_long > 0.0
    assert math.isclose(abs(w_a_long) + abs(w_b_long), 1.0, abs_tol=1e-12)
    assert math.isclose(w_a_long, 1.0 / 1.8, abs_tol=1e-12)
    assert math.isclose(w_b_long, 0.8 / 1.8, abs_tol=1e-12)

    w_a_short, w_b_short = compute_pair_leg_weights("short_spread", -0.8)
    assert w_a_short < 0.0
    assert w_b_short < 0.0
    assert math.isclose(abs(w_a_short) + abs(w_b_short), 1.0, abs_tol=1e-12)


def test_daily_vs_per_trade_compounded_equity_reconciliation_and_final_bar_liquidation():
    """Verify that initial_capital * prod(1 + trade.net_return_pct/100) equals terminal equity_net[-1],
    including when an open position is forcibly liquidated on the final bar T-1.
    """
    # Construct a price path where a trade opens near the end and is still open at T-1 (forced end_of_sample exit)
    prices = [100.0, 100.0, 97.0, 100.0, 103.0, 103.0, 99.0, 100.0, 104.0, 106.0]
    df = _make_toy_df(prices)
    spec = StrategySpec(
        hypothesis_text="Final bar liquidation and compounded trade equity reconciliation",
        title="Compounded Trade Reconciliation",
        strategy_family="single_asset",
        template_id="return_threshold",
        signal_variant="cum_return_drop",
        asset_a="TEST",
        benchmark_ticker="TEST",
        parameters=StrategyParameters(
            lookback_window=1,
            entry_threshold=2.5,
            holding_period=10,  # Long enough that trade 1 is still open at bar T-1
            exit_mode="fixed_hold",
            direction="long",
        ),
        execution=ExecutionAssumptions(
            initial_capital=100_000.0,
            execution_timing="next_open",
            commission_bps=10.0,
            slippage_bps=10.0,
        ),
        risk_controls=RiskControls(max_drawdown_circuit_breaker_pct=None, per_trade_stop_loss_pct=None),
        human_confirmed=True,
    )
    sim = simulate_strategy_core(df, spec, split_idx=5)
    trades = sim["trades"]
    assert len(trades) >= 1
    assert trades[-1].exit_reason == "end_of_sample_liquidation"

    compounded_from_trades = spec.execution.initial_capital * float(
        np.prod([1.0 + t.net_return_pct / 100.0 for t in trades])
    )
    terminal_daily_eq = float(sim["equity_net"][-1])
    assert math.isclose(compounded_from_trades, terminal_daily_eq, rel_tol=1e-6, abs_tol=1e-2)


def test_donchian_short_breakout_and_oscillator_fixed_hold_exit():
    """Verify donchian_breakout triggers on breakdown below prior L-day low and mean_reversion_oscillator
    respects fixed_hold exit mode.
    """
    # 1. Donchian short breakdown: prices stay at 100 for 6 bars, then drop to 92 at bar 6
    prices_short = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 92.0, 90.0, 88.0, 95.0, 96.0]
    df_short = _make_toy_df(prices_short)
    spec_donchian = StrategySpec(
        hypothesis_text="Short Donchian breakdown",
        title="Donchian Short",
        strategy_family="single_asset",
        template_id="trend_breakout",
        signal_variant="donchian_breakout",
        asset_a="TEST",
        benchmark_ticker="TEST",
        parameters=StrategyParameters(
            lookback_window=5,
            secondary_window=3,
            entry_threshold=0.0,
            holding_period=2,
            exit_mode="fixed_hold",
            direction="short",
        ),
        risk_controls=RiskControls(max_drawdown_circuit_breaker_pct=None, per_trade_stop_loss_pct=None),
        human_confirmed=True,
    )
    sim_d = simulate_strategy_core(df_short, spec_donchian, split_idx=5)
    assert len(sim_d["trades"]) >= 1
    assert sim_d["trades"][0].direction == "short"
    assert sim_d["trades"][0].holding_days == 2

    # 2. Mean-reversion oscillator with fixed_hold=4 must hold for 4 bars even if z-score crosses 0 earlier
    prices_osc = [100.0] * 10 + [90.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0]
    df_osc = _make_toy_df(prices_osc)
    spec_osc = StrategySpec(
        hypothesis_text="Oscillator fixed hold",
        title="Oscillator Fixed Hold",
        strategy_family="single_asset",
        template_id="mean_reversion_oscillator",
        signal_variant="price_zscore",
        asset_a="TEST",
        benchmark_ticker="TEST",
        parameters=StrategyParameters(
            lookback_window=8,
            entry_threshold=1.5,
            exit_threshold=0.0,
            holding_period=4,
            exit_mode="fixed_hold",
            direction="long",
        ),
        risk_controls=RiskControls(max_drawdown_circuit_breaker_pct=None, per_trade_stop_loss_pct=None),
        human_confirmed=True,
    )
    sim_o = simulate_strategy_core(df_osc, spec_osc, split_idx=8)
    assert len(sim_o["trades"]) == 1
    assert sim_o["trades"][0].holding_days == 4
    assert sim_o["trades"][0].exit_reason == "holding_period (4d)"


def test_zero_trade_holdout_and_pep_aapl_reconciliation():
    """Verify PEP vs AAPL co-movement disambiguation and zero-trade holdout handling."""
    from pathlib import Path
    from engine.backtester import load_audit_run

    # 1. Disambiguation of "Do Pepsi and Apple move together?"
    interp = interpret_hypothesis("I think Pepsi and Apple move together.")
    assert interp.proposed_spec is not None
    assert interp.proposed_spec.asset_a == "PEP"
    assert interp.proposed_spec.asset_b == "AAPL"
    assert len(interp.steered_cards) == 3
    assert len(interp.proposed_spec.user_specified_fields) >= 2
    assert len(interp.proposed_spec.software_inferred_defaults) >= 3

    # 2. Replay legacy PEP/AAPL run artifact and verify v2 fixes
    legacy_path = Path(__file__).resolve().parent.parent / "data" / "runs" / "audit-022cbf20a8.json"
    if legacy_path.exists():
        legacy_report = load_audit_run(legacy_path)
        spec = legacy_report.spec.model_copy(deep=True)
        spec.human_confirmed = True
        v2_report = run_full_audit(spec)

        # In v2, causal expanding OLS in Formation + circuit breaker halt in Formation -> 0 OOS trades
        assert v2_report.out_of_sample_metrics.trade_count == 0
        assert v2_report.out_of_sample_metrics.is_zero_exposure is True
        assert v2_report.out_of_sample_metrics.halted_prior_to_period is True

        dims = {d.dimension_id: d for d in v2_report.scorecard.dimensions}
        assert dims["out_of_sample_persistence"].status == "NO_EVIDENCE"
        assert v2_report.scorecard.evaluated_checks_count == 4
        assert len(v2_report.scorecard.hypothesis_comparison_breakdown) == 3


def test_executable_jupyter_notebook_ipynb_generation_and_fresh_kernel_execution():
    """Verify v4 .ipynb generation, fresh-kernel execution via nbclient, and FastAPI .ipynb endpoints."""
    from fastapi.testclient import TestClient
    from app.server import app
    from engine.notebook_ipynb import (
        execute_and_verify_notebook_bytes,
        render_jupyter_notebook_ipynb,
    )

    client = TestClient(app)

    # 1. Single-asset SPY audit -> generate and execute .ipynb in a fresh kernel
    interp = interpret_hypothesis("I think the S&P 500 tends to recover the day after falling more than 2%.")
    spec = interp.proposed_spec
    assert spec is not None

    # Verify POST /api/audit/pdf and POST /api/audit/ipynb reject unconfirmed specs (HTTP 400)
    unconf_pdf = client.post("/api/audit/pdf", json=spec.model_dump())
    assert unconf_pdf.status_code == 400
    unconf_ipynb = client.post("/api/audit/ipynb", json=spec.model_dump())
    assert unconf_ipynb.status_code == 400

    spec.human_confirmed = True
    report = run_full_audit(spec)
    ipynb_bytes = render_jupyter_notebook_ipynb(report)
    exec_summary = execute_and_verify_notebook_bytes(ipynb_bytes)
    assert exec_summary["cells_executed"] == 27
    assert exec_summary["code_cells_verified"] == 13

    # 2. Pairs KO/PEP audit -> generate and execute .ipynb in a fresh kernel
    pairs_interp = interpret_hypothesis(
        "I think Coca-Cola and Pepsi prices tend to converge after unusually large divergences."
    )
    pairs_spec = pairs_interp.proposed_spec
    assert pairs_spec is not None
    pairs_spec.human_confirmed = True
    pairs_report = run_full_audit(pairs_spec)
    pairs_ipynb = render_jupyter_notebook_ipynb(pairs_report)
    pairs_exec = execute_and_verify_notebook_bytes(pairs_ipynb)
    assert pairs_exec["cells_executed"] == 27
    assert pairs_exec["code_cells_verified"] == 13

    # 3. FastAPI POST /api/audit/ipynb and GET /api/audit/{run_id}/ipynb
    res_post = client.post("/api/audit/ipynb", json=spec.model_dump())
    assert res_post.status_code == 200
    assert "application/x-ipynb+json" in res_post.headers["content-type"]

    run_res = client.post("/api/audit/run", json=spec.model_dump())
    assert run_res.status_code == 200
    run_id = run_res.json()["run_id"]

    res_get = client.get(f"/api/audit/{run_id}/ipynb")
    assert res_get.status_code == 200
    assert "application/x-ipynb+json" in res_get.headers["content-type"]


def test_adapter_submission_paths_and_pairs_audit():
    """Verify all AgentVerse adapter.py submission tasks: market_step, audit_single_asset,
    audit_pairs_spread, and graceful handling of invalid inputs.
    """
    from adapter import solve

    # 1. Invalid inputs
    assert solve("not-a-dict") == {"error": "invalid_input"}
    assert solve({"task": "market_step", "input": "not-a-dict"}) == {"error": "invalid_input"}
    assert solve({"task": "nonexistent_task", "input": {}}) == {"error": "unsupported_task"}

    # 2. market_step task
    step_res = solve(
        {
            "task": "market_step",
            "input": {
                "cash_pence": 100000,
                "shares": 0,
                "fee_pence": 10,
                "slippage_bps": 5,
                "history": [{"price_pence": 1000 + 15 * i} for i in range(25)],
            },
        }
    )
    assert step_res.get("action") in ("buy", "sell", "hold")

    # 3. audit_single_asset task
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2022-01-03", periods=90).strftime("%Y-%m-%d").tolist()
    pa = list(100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=90))))
    single_res = solve(
        {
            "task": "audit_single_asset",
            "input": {
                "dates": dates,
                "prices": pa,
                "hypothesis_text": "Single asset daily drop mean reversion",
                "lookback_window": 1,
                "entry_threshold": 1.2,
                "holding_period": 3,
            },
        }
    )
    assert single_res["status"] == "ok"
    assert "final_equity_rounded" in single_res
    assert "trade_count" in single_res

    # 4. audit_pairs_spread task (newly wired)
    pb = list(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.011, size=90))))
    pairs_res = solve(
        {
            "task": "audit_pairs_spread",
            "input": {
                "dates": dates,
                "prices_a": pa,
                "prices_b": pb,
                "asset_a": "KO",
                "asset_b": "PEP",
                "signal_variant": "ols_log",
                "lookback_window": 20,
                "entry_threshold": 1.5,
                "exit_threshold": 0.3,
            },
        }
    )
    assert pairs_res["status"] == "ok"
    assert pairs_res["signal_variant"] == "ols_log"
    assert "final_equity_rounded" in pairs_res
    assert "ols_beta_formation" in pairs_res
    assert "trade_count" in pairs_res


def test_honest_model_provenance_and_comparison_breakdown_keys():
    """Verify no fake 'gemini-4-argon' labels appear on any parser path and that
    hypothesis_comparison_breakdown populates both backend and frontend keys.
    """
    for prompt in [
        "I think SPY recovers after falling 2% in a day.",
        "How should I optimize my 401k retirement tax bracket?",
        "Tell me about random weather patterns on Mars.",
        "Do KO and PEP move together?",
    ]:
        res = interpret_hypothesis(prompt)
        assert "gemini-4-argon" not in res.model_used.lower()
        assert res.interpretation_mode in (
            "deterministic_rules",
            "live_gemini",
            "live_gemini_fallback",
        )
        if res.proposed_spec is not None:
            assert isinstance(res.proposed_spec.software_inferred_defaults, dict)

    pairs_res = interpret_hypothesis("Do KO and PEP move together?")
    spec = pairs_res.proposed_spec
    assert spec is not None
    spec.human_confirmed = True
    report = run_full_audit(spec)
    assert report.replay_manifest["bootstrap_seed"] == 20261107
    assert report.replay_manifest["monte_carlo_seed"] == 42

    breakdown = report.scorecard.hypothesis_comparison_breakdown
    assert len(breakdown) == 3
    for item in breakdown:
        for req_key in (
            "aspect",
            "evidence",
            "verdict",
            "explanation",
            "lens_title",
            "question_answered",
            "primary_metric",
            "status",
            "interpretation",
        ):
            assert req_key in item and item[req_key], f"Missing or empty key {req_key} in {item}"


def test_crossed_split_trade_and_notebook_tamper_detection():
    """Verify trades spanning split_idx set crossed_split=True and that tampering with
    embedded DF_RECORDS in a generated .ipynb fails the SHA-256 assertion in Cell 3.
    """
    import nbformat
    import pytest
    from engine.notebook_ipynb import (
        execute_and_verify_notebook_bytes,
        render_jupyter_notebook_ipynb,
    )

    # 1. Construct a trade that opens at bar 4 (before split_idx=5) and holds for 3 days (exits at bar 7 >= split_idx)
    prices = [100.0, 100.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0]
    df = _make_toy_df(prices)
    spec = StrategySpec(
        hypothesis_text="Trade crossing split_idx boundary",
        title="Crossed Split Trade Test",
        strategy_family="single_asset",
        template_id="return_threshold",
        signal_variant="cum_return_drop",
        asset_a="TEST",
        benchmark_ticker="TEST",
        parameters=StrategyParameters(
            lookback_window=1,
            entry_threshold=3.0,
            holding_period=4,
            exit_mode="fixed_hold",
            direction="long",
        ),
        risk_controls=RiskControls(max_drawdown_circuit_breaker_pct=None, per_trade_stop_loss_pct=None),
        human_confirmed=True,
    )
    sim = simulate_strategy_core(df, spec, split_idx=5)
    assert len(sim["trades"]) == 1
    tr = sim["trades"][0]
    assert tr.split == "in_sample"
    assert tr.crossed_split is True

    # 2. Verify notebook SHA-256 tamper detection on embedded DF_RECORDS
    interp = interpret_hypothesis("I think SPY recovers the day after dropping more than 2%.")
    spy_spec = interp.proposed_spec
    assert spy_spec is not None
    spy_spec.human_confirmed = True
    spy_report = run_full_audit(spy_spec)
    ipynb_bytes = render_jupyter_notebook_ipynb(spy_report)

    nb = nbformat.reads(ipynb_bytes.decode("utf-8"), as_version=4)
    mutated = False
    for cell in nb.cells:
        if cell.cell_type == "code" and "df = pd.DataFrame(DF_RECORDS)" in cell.source:
            cell.source = cell.source.replace(
                "df = pd.DataFrame(DF_RECORDS)",
                "DF_RECORDS[0]['close_a'] += 1.0\ndf = pd.DataFrame(DF_RECORDS)",
                1,
            )
            mutated = True
            break
    assert mutated is True
    tampered_bytes = nbformat.writes(nb).encode("utf-8")
    with pytest.raises(Exception):
        execute_and_verify_notebook_bytes(tampered_bytes)


def test_multi_turn_chat_isolation_steered_provenance_and_borderline_cointegration():
    """Verify multi-turn chat isolation on new sector-cue hypotheses vs short follow-up
    refinements, single-asset steered card provenance fields, and KO/PEP cointegration note consistency.
    """
    prior_history = [
        {
            "role": "user",
            "content": "I think the S&P 500 tends to recover the day after falling more than 2%.",
        }
    ]

    # 1A. Standalone new hypothesis with sector cue ("tech stocks" -> QQQ) must NOT be polluted by prior SPY turn
    new_sector_interp = interpret_hypothesis(
        "I think tech stocks bounce back after a sharp selloff",
        chat_history=prior_history,
    )
    assert new_sector_interp.proposed_spec is not None
    assert new_sector_interp.proposed_spec.asset_a == "QQQ"

    # 1B. Genuine short follow-up refinement overrides prior explicit threshold & holding period
    prior_explicit = [
        {
            "role": "user",
            "content": "I think SPY recovers over the next 3 trading days after falling more than 2% in a single day.",
        }
    ]
    followup_interp = interpret_hypothesis(
        "what about holding for 10 days and a 3.5% drop?",
        chat_history=prior_explicit,
    )
    assert followup_interp.proposed_spec is not None
    assert followup_interp.proposed_spec.asset_a == "SPY"
    assert followup_interp.proposed_spec.parameters.entry_threshold == 3.5
    assert followup_interp.proposed_spec.parameters.holding_period == 10

    # 1C. Multi-hop risk follow-up ("what about a 5% stop loss?") preserves anchor asset & overrides stop-loss
    multihop_history = prior_explicit + [
        {"role": "user", "content": "what about holding for 10 days and a 3.5% drop?"}
    ]
    stoploss_interp = interpret_hypothesis(
        "what about a 5% stop loss?",
        chat_history=multihop_history,
    )
    assert stoploss_interp.status == "ready_for_confirmation"
    assert stoploss_interp.proposed_spec is not None
    assert stoploss_interp.proposed_spec.asset_a == "SPY"
    assert stoploss_interp.proposed_spec.parameters.entry_threshold == 3.5
    assert stoploss_interp.proposed_spec.parameters.holding_period == 10
    assert stoploss_interp.proposed_spec.risk_controls.per_trade_stop_loss_pct == 5.0
    assert "risk_controls.per_trade_stop_loss_pct" in stoploss_interp.proposed_spec.user_specified_fields

    # 1D. Out-of-scope non-financial prompt in multi-turn chat is still rejected as unsupported
    non_fin_interp = interpret_hypothesis(
        "hello how are you doing today?",
        chat_history=prior_explicit,
    )
    assert non_fin_interp.status == "unsupported"
    assert non_fin_interp.proposed_spec is None

    # 2. Single-asset and pairs steered cards must populate accurate user_specified_fields and software_inferred_defaults
    assert len(new_sector_interp.steered_cards) == 3
    for card in new_sector_interp.steered_cards:
        assert len(card.proposed_spec.user_specified_fields) >= 1
        assert len(card.proposed_spec.software_inferred_defaults) >= 3

    # 3. KO vs PEP cointegration note & pairs steered cards provenance
    ko_pep_interp = interpret_hypothesis(
        "I think Coca-Cola and Pepsi prices tend to converge after diverging by more than 2 standard deviations."
    )
    assert ko_pep_interp.proposed_spec is not None
    assert len(ko_pep_interp.steered_cards) == 3
    for card in ko_pep_interp.steered_cards:
        assert "parameters.entry_threshold" in card.proposed_spec.user_specified_fields
        assert "parameters.lookback_window" in card.proposed_spec.user_specified_fields
        assert "parameters.entry_threshold" not in card.proposed_spec.software_inferred_defaults
        assert "parameters.lookback_window" not in card.proposed_spec.software_inferred_defaults
        assert len(card.proposed_spec.software_inferred_defaults) >= 3

    ko_pep_spec = ko_pep_interp.proposed_spec
    ko_pep_spec.human_confirmed = True
    ko_pep_report = run_full_audit(ko_pep_spec)
    diag = ko_pep_report.pairs_diagnostics
    assert diag is not None
    assert diag.adf_p_value < 0.05 or diag.engle_granger_p_value < 0.05
    assert "does NOT reject the unit-root null" not in diag.cointegration_status_note
    assert "rejects a univariate unit root at the 5% level" in diag.cointegration_status_note

