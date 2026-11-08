"""Strict Pydantic v2 schemas for the 6-Step Alpha Auditor Research Workflow:
Step 1: Conversational Gemini Intake, Smart Steering & Scope Filtering
Step 2: Strategy Interpretation & Dual-Mode Threshold Configuration (Specific vs. Standard Bracket)
Step 3: Peer-Reviewed Quantitative Literature & Methodology Grounding
Step 4: Interactive Performance, Multi-Threshold Overlay & Risk Analysis
Step 5: Adversarial Stress Testing
Step 6: Empirical Hypothesis Scorecard & Full Step-by-Step Calculation Audit Ledger
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


StrategyFamily = Literal["single_asset", "pairs_trading"]

SingleTemplateId = Literal[
    "return_threshold",
    "consecutive_streak",
    "mean_reversion_oscillator",
    "trend_breakout",
]

PairsTemplateId = Literal["pairs_cointegration_reversion"]

TemplateId = Literal[
    "return_threshold",
    "consecutive_streak",
    "mean_reversion_oscillator",
    "trend_breakout",
    "pairs_cointegration_reversion",
]

SignalVariant = Literal[
    # return_threshold variants
    "cum_return_drop",
    "cum_return_surge",
    "vol_scaled_shock",
    # consecutive_streak variants
    "negative_streak",
    "positive_streak",
    # mean_reversion_oscillator variants
    "price_zscore",
    "bollinger_band",
    "rsi_extremes",
    # trend_breakout variants
    "sma_crossover",
    "ema_crossover",
    "donchian_breakout",
    # pairs_cointegration_reversion variants
    "ols_log",
    "rolling_ols_log",
    "price_ratio",
]

ExecutionTiming = Literal["next_open", "next_close"]
PositionDirection = Literal["long", "short", "both"]
ExitMode = Literal["fixed_hold", "threshold_reversion", "first_positive_close", "signal_reversal"]
ThresholdMode = Literal["user_specified", "auto_standard_bracket"]


class ExecutionAssumptions(BaseModel):
    """Explicit trade execution, cost, and capital assumptions."""

    initial_capital: float = Field(default=100_000.0, ge=1_000.0, le=100_000_000.0)
    execution_timing: ExecutionTiming = Field(
        default="next_open",
        description="Fill timing on bar t+1 after signal observed at close of bar t.",
    )
    commission_bps: float = Field(
        default=5.0,
        ge=0.0,
        le=500.0,
        description="Commission per leg execution in basis points (1 bps = 0.01%).",
    )
    slippage_bps: float = Field(
        default=5.0,
        ge=0.0,
        le=500.0,
        description="Adverse price slippage per leg execution in basis points.",
    )
    train_split_ratio: float = Field(
        default=0.5,
        ge=0.2,
        le=0.8,
        description="Chronological fraction of dataset allocated to In-Sample Formation.",
    )


class RiskControls(BaseModel):
    """Deterministic risk limits enforced in code during simulation."""

    max_drawdown_circuit_breaker_pct: float | None = Field(
        default=25.0,
        ge=2.0,
        le=95.0,
        description="Halts new entries if strategy drawdown reaches this percentage.",
    )
    per_trade_stop_loss_pct: float | None = Field(
        default=15.0,
        ge=0.5,
        le=90.0,
        description="Closes an active trade if open mark-to-market loss exceeds this percentage.",
    )
    max_holding_days: int = Field(
        default=20,
        ge=1,
        le=252,
        description="Maximum bars a position can stay open before time-stop exit.",
    )


class StrategyParameters(BaseModel):
    """Canonical parameter set across single-asset and pairs strategy templates."""

    lookback_window: int = Field(
        default=20,
        ge=1,
        le=504,
        description="Primary lookback window in trading days.",
    )
    secondary_window: int = Field(
        default=50,
        ge=2,
        le=756,
        description="Secondary window in trading days (e.g., slow MA window or rolling hedge window).",
    )
    streak_length: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of consecutive positive/negative days required for streak signal.",
    )
    entry_threshold: float = Field(
        default=2.0,
        description="Primary entry threshold (e.g., % drop such as 2.0, z-score 2.0, or RSI 30.0).",
    )
    exit_threshold: float = Field(
        default=0.0,
        description="Primary exit threshold (e.g., z-score 0.0 for mean reversion, or RSI 55.0).",
    )
    stop_z_threshold: float | None = Field(
        default=4.0,
        ge=1.0,
        le=10.0,
        description="For z-score / pairs strategies: stop-loss z-score if divergence widens beyond bounds.",
    )
    holding_period: int = Field(
        default=5,
        ge=1,
        le=252,
        description="Target holding period in trading days.",
    )
    exit_mode: ExitMode = Field(
        default="fixed_hold",
        description="Condition that triggers position exit.",
    )
    direction: PositionDirection = Field(
        default="long",
        description="Allowed trade direction (long, short, or both).",
    )


class StrategySpec(BaseModel):
    """Complete, reproducible specification of a quantitative hypothesis experiment."""

    hypothesis_text: str = Field(..., min_length=3, description="Original natural-language hypothesis.")
    title: str = Field(..., description="Concise institutional research title for the experiment.")
    strategy_family: StrategyFamily
    template_id: TemplateId
    signal_variant: SignalVariant
    asset_a: str = Field(..., min_length=1, max_length=20, description="Primary ticker symbol.")
    asset_b: str | None = Field(
        default=None,
        description="Secondary ticker symbol for pairs trading.",
    )
    benchmark_ticker: str = Field(
        default="SPY",
        description="Reference market benchmark ticker.",
    )
    start_date: str = Field(default="2018-01-01", description="ISO start date YYYY-MM-DD.")
    end_date: str = Field(default="2025-12-31", description="ISO end date YYYY-MM-DD.")
    threshold_mode: ThresholdMode = Field(
        default="user_specified",
        description="'user_specified' tests exact thresholds; 'auto_standard_bracket' tests Conservative, Baseline, and Aggressive literature thresholds.",
    )
    parameters: StrategyParameters = Field(default_factory=StrategyParameters)
    execution: ExecutionAssumptions = Field(default_factory=ExecutionAssumptions)
    risk_controls: RiskControls = Field(default_factory=RiskControls)
    selected_diagnostics: list[str] = Field(
        default_factory=lambda: [
            "stationary_bootstrap",
            "newey_west_alpha",
            "subperiod_regimes",
            "cost_sensitivity",
            "parameter_sensitivity_grid",
        ]
    )
    human_confirmed: bool = Field(
        default=False,
        description="Must be explicitly set to True by user confirmation before execution.",
    )
    user_specified_fields: list[str] = Field(
        default_factory=list,
        description="Parameter keys explicitly stated by the user in natural language.",
    )
    software_inferred_defaults: dict[str, Any] = Field(
        default_factory=dict,
        description="Default values added by software when compiling an underspecified prompt.",
    )
    evaluation_stage: Literal["confirmatory", "exploratory_tuner"] = Field(
        default="confirmatory",
        description="Whether this run is the initial confirmatory audit or a re-run after live parameter tuning on the same holdout.",
    )
    prior_holdout_touches: int = Field(
        default=0,
         ge=0,
        description="Number of prior parameter evaluations on this asset/date holdout slice.",
    )

    @model_validator(mode="after")
    def validate_pairs_requirements(self) -> "StrategySpec":
        self.asset_a = self.asset_a.strip().upper()
        if self.asset_b:
            self.asset_b = self.asset_b.strip().upper()
        self.benchmark_ticker = self.benchmark_ticker.strip().upper()
        if self.strategy_family == "pairs_trading":
            if not self.asset_b:
                raise ValueError("Pairs trading requires two distinct asset tickers (asset_a and asset_b).")
            if self.asset_a == self.asset_b:
                raise ValueError("Pairs trading requires asset_a and asset_b to be distinct tickers.")
        return self


class AmbiguityQuestion(BaseModel):
    """Structured parameter clarification prompt presented when a hypothesis is underspecified."""

    parameter_key: str = Field(..., description="Field path in StrategySpec.")
    label: str = Field(..., description="Research-grade parameter label.")
    question: str = Field(..., description="Explanation of why this parameter requires specification.")
    current_value: Any = Field(..., description="Default/initial value pre-populated in the form.")
    suggested_options: list[Any] = Field(default_factory=list, description="Literature standard values.")
    unit: str = Field(default="", description="Unit label (%, σ, days, bps).")


class SteeredHypothesisCard(BaseModel):
    """Concrete, testable hypothesis suggestion from the specification compiler when the user's input is vague or slightly off."""

    label: str = Field(..., description="Short title for the steered hypothesis.")
    hypothesis_text: str = Field(..., description="Refined, directly testable English hypothesis statement.")
    rationale: str = Field(..., description="Why this refinement makes the user's intuition empirically testable.")
    target_assets: str = Field(..., description="Asset(s) used in this refinement (e.g., 'SPY', 'KO / PEP').")
    template_badge: str = Field(..., description="Strategy family / signal badge.")
    proposed_spec: StrategySpec


BasketDecision = TemplateId | Literal["needs_detail", "no_match"]


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    basket: BasketDecision | None = None


class InterpretationResponse(BaseModel):
    """Output of Step 1 & Step 2 (Conversational Intake, Steering & Specification Compilation)."""

    status: Literal[
        "ready_for_confirmation",
        "needs_clarification",
        "steered_suggestions",
        "unsupported",
    ]
    gemini_reply: str = Field(
        default="",
        description="Natural conversational response explaining the interpretation or steering suggestions.",
    )
    understood_summary: dict[str, str] = Field(
        default_factory=dict,
        description="Structured summary of what was understood (Assets, Signal Trigger, Holding Rule, Benchmark).",
    )
    proposed_spec: StrategySpec | None = None
    steered_cards: list[SteeredHypothesisCard] = Field(
        default_factory=list,
        description="Clickable steered hypotheses if the user's prompt was vague, incomplete, or slightly off.",
    )
    ambiguities: list[AmbiguityQuestion] = Field(default_factory=list)
    interpretation_notes: list[str] = Field(default_factory=list)
    unsupported_reason: str | None = None
    model_used: str = Field(default="deterministic-spec-compiler-v2")
    interpretation_mode: Literal["live_gemini", "cached_gemini", "deterministic_rules"] = Field(
        default="deterministic_rules",
        description="Whether this response came from a live Gemini API call, cached fixture, or deterministic rule compiler.",
    )
    chat_history: list[ChatTurn] = Field(default_factory=list)


class TradeRecord(BaseModel):
    """Exact audit record for a single historical trade."""

    trade_id: int
    split: Literal["in_sample", "out_of_sample"]
    direction: Literal["long", "short", "long_spread", "short_spread"]
    signal_date: str
    signal_value: float
    entry_date: str
    entry_price_a: float
    entry_price_b: float | None = None
    exit_date: str
    exit_price_a: float
    exit_price_b: float | None = None
    holding_days: int
    gross_return_pct: float
    cost_pct: float
    net_return_pct: float
    exit_reason: str
    cumulative_equity_after: float
    crossed_split: bool = Field(
        default=False,
        description="True if position entered during In-Sample Formation and exited during Out-of-Sample Holdout.",
    )
    leg_weight_a: float = Field(
        default=1.0,
        description="Signed portfolio capital weight on Asset A.",
    )
    leg_weight_b: float = Field(
        default=0.0,
        description="Signed portfolio capital weight on Asset B (for pairs strategies).",
    )


class PeriodMetrics(BaseModel):
    """Comprehensive performance & statistical metrics for a chronological split."""

    period_label: str
    start_date: str
    end_date: str
    trading_days: int
    trade_count: int
    exposure_pct: float
    total_gross_return_pct: float
    total_net_return_pct: float
    cagr_net_pct: float
    benchmark_return_pct: float
    benchmark_cagr_pct: float
    excess_return_pct: float
    annualized_volatility_pct: float
    benchmark_volatility_pct: float
    max_drawdown_pct: float
    benchmark_max_drawdown_pct: float
    sharpe_ratio: float
    benchmark_sharpe_ratio: float
    sharpe_se_lo2002: float
    probabilistic_sharpe_psr_pct: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate_pct: float
    profit_factor: float | None
    mean_trade_net_pct: float
    median_trade_net_pct: float
    std_trade_net_pct: float
    best_trade_net_pct: float
    worst_trade_net_pct: float
    skewness_daily: float
    kurtosis_daily: float
    t_stat_mean_return: float
    p_value_two_sided: float
    sign_test_p_value: float | None
    circuit_breaker_triggered: bool = False
    circuit_breaker_date: str | None = None
    is_zero_exposure: bool = Field(
        default=False,
        description="True when 0 trades and 0% exposure occurred in this period, making Sharpe/Sortino/PSR undefined.",
    )
    halted_prior_to_period: bool = Field(
        default=False,
        description="True when the portfolio circuit breaker tripped in an earlier split prior to this period.",
    )
    status_note: str | None = Field(
        default=None,
        description="Diagnostic explanation when period has zero exposure or prior halt.",
    )


class BracketProfileResult(BaseModel):
    """Result for one profile in Step 2's 'Auto-Test Standard Thresholds' bracket."""

    profile_id: str
    profile_label: str
    threshold_description: str
    entry_threshold: float
    exit_threshold: float
    lookback_window: int
    holding_period: int
    in_sample_metrics: PeriodMetrics
    out_of_sample_metrics: PeriodMetrics
    full_sample_metrics: PeriodMetrics
    equity_net: list[float]


class PairsDiagnostics(BaseModel):
    """Econometric diagnostics for two-asset cointegration and spread stationarity."""

    formation_start: str
    formation_end: str
    ols_alpha: float
    ols_hedge_ratio_beta: float
    ols_r_squared: float
    return_correlation: float
    level_correlation: float
    adf_statistic: float
    adf_p_value: float
    adf_critical_values: dict[str, float]
    engle_granger_stat: float
    engle_granger_p_value: float
    ou_lambda: float
    half_life_days: float | None
    hurst_exponent: float
    variance_ratio_q5: float
    variance_ratio_p_value: float
    cointegration_status_note: str


class BootstrapConfidenceInterval(BaseModel):
    """Politis & Romano (1994) Stationary Bootstrap 95% confidence interval."""

    metric_name: str
    point_estimate: float
    ci_lower_95: float
    ci_upper_95: float
    bootstrap_std_error: float
    block_length: int
    replications: int
    unit: str


class FactorRegressionResult(BaseModel):
    """Newey-West (1987) HAC factor regression against benchmark."""

    annualized_alpha_pct: float
    alpha_t_stat_hac: float
    alpha_p_value_hac: float
    beta: float
    beta_t_stat_hac: float
    r_squared: float
    information_ratio: float
    newey_west_lags: int


class CostSensitivityRow(BaseModel):
    """Performance across varying transaction cost assumptions (commission + slippage)."""

    total_cost_bps_per_leg: float
    in_sample_net_return_pct: float
    out_of_sample_net_return_pct: float
    full_period_net_return_pct: float
    full_period_sharpe: float


class SubperiodRow(BaseModel):
    """Performance across distinct historical subperiods and volatility regimes."""

    label: str
    start_date: str
    end_date: str
    trade_count: int
    strategy_net_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float


class SensitivityGridCell(BaseModel):
    """Single cell in a 2D threshold/parameter sensitivity sweep."""

    param_x_value: float
    param_y_value: float
    trade_count: int
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    full_sharpe: float
    full_net_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float


class ParameterSensitivitySurface(BaseModel):
    """2D parameter sweep + Bailey & López de Prado (2014) Deflated Sharpe Ratio."""

    param_x_name: str
    param_x_label: str
    param_x_values: list[float]
    param_y_name: str
    param_y_label: str
    param_y_values: list[float]
    cells: list[SensitivityGridCell]
    trials_count_k: int
    expected_max_sr_under_null: float
    selected_spec_deflated_sharpe_psr_pct: float
    best_grid_deflated_sharpe_psr_pct: float
    methodology_note: str


class LiteraturePaper(BaseModel):
    """Foundational academic paper relevant to the strategy and backtesting methodology."""

    authors: str
    year: int
    title: str
    journal: str
    category: Literal["Strategy Mechanism", "Econometric & Backtest Rigor"]
    core_finding: str
    empirical_pitfall_warned: str
    how_alpha_auditor_addresses: str


class LiteratureReviewSection(BaseModel):
    """Step 3: Academic literature grounding and methodological safeguards."""

    theoretical_mechanism_summary: str
    gemini_literature_synthesis: str
    strategy_papers: list[LiteraturePaper]
    methodology_papers: list[LiteraturePaper]
    rigor_checklist: list[dict[str, str]]


class AdversarialStressTest(BaseModel):
    """Step 5: Individual custom stress test designed to probe a specific vulnerability."""

    test_id: str
    title: str
    vulnerability_probed: str
    gemini_rationale: str
    methodology: str
    outcome_badge: Literal["RESILIENT", "MODERATE_SENSITIVITY", "HIGH_VULNERABILITY"]
    baseline_label: str
    baseline_value: str
    stressed_label: str
    stressed_value: str
    key_finding: str
    breakdown_rows: list[dict[str, Any]] = Field(default_factory=list)
    chart_payload: dict[str, Any] = Field(default_factory=dict)


class StressTestSuite(BaseModel):
    """Step 5: Adversarial stress tests computed by the audit engine."""

    gemini_overview: str
    vulnerability_count: int
    resilient_count: int
    tests: list[AdversarialStressTest]


class ScorecardDimension(BaseModel):
    """Step 6: Single objective evidence check evaluating whether the hypothesis held up."""

    dimension_id: str
    name: str
    question: str
    status: Literal["SUPPORTED", "MIXED", "CONTRADICTED", "NO_EVIDENCE"]
    evidence_metric: str
    explanation: str


class FormulaCalculationStep(BaseModel):
    """Step 6: Inspectable mathematical formula with the exact run numbers substituted."""

    step_id: str
    title: str
    formula: str
    substituted_inputs: str
    output_value: str
    interpretation: str


class HypothesisScorecard(BaseModel):
    """Step 6: Clear, non-advisory synthesis of whether the user's hypothesis held in the data."""

    hypothesis_restatement: str
    historical_verdict_label: Literal[
        "SUPPORTED BY HISTORICAL DATA",
        "MIXED / REGIME-DEPENDENT IN HISTORICAL DATA",
        "CONTRADICTED BY HISTORICAL DATA",
        "NO HOLDOUT EVIDENCE / HALTED IN FORMATION",
        "INSUFFICIENT EMPIRICAL EVIDENCE",
    ]
    supported_checks_count: int
    total_checks_count: int
    evaluated_checks_count: int = Field(
        default=5,
        description="Number of dimensions with empirical evidence (excluding NO_EVIDENCE).",
    )
    executive_summary: str
    key_takeaways: list[str]
    dimensions: list[ScorecardDimension]
    calculation_steps: list[FormulaCalculationStep]
    hypothesis_comparison_breakdown: list[dict[str, str]] = Field(
        default_factory=list,
        description="Explicit separation of Return Co-Movement, Long-Run Cointegration, and Active Trading Rule P&L when applicable.",
    )


class AuditReport(BaseModel):
    """Complete, self-contained, reproducible research output across all 6 steps."""

    schema_version: str = Field(default="2.0")
    run_id: str
    created_at_utc: str
    market_as_of_date: str = Field(
        default="",
        description="Latest bar date in the aligned historical market dataset snapshot.",
    )
    spec: StrategySpec
    spec_sha256: str
    data_provenance: dict[str, Any]
    engine_version: dict[str, Any] = Field(
        default_factory=dict,
        description="Engine version and SHA-256 hashes of core computational modules.",
    )
    replay_manifest: dict[str, Any] = Field(
        default_factory=dict,
        description="Self-contained replay provenance including spec hash, data hash, and RNG seeds.",
    )
    split_date: str

    # Core performance splits
    full_sample_metrics: PeriodMetrics
    in_sample_metrics: PeriodMetrics
    out_of_sample_metrics: PeriodMetrics

    # Step 2/4 Multi-Threshold Standard Bracket Results (when threshold_mode == 'auto_standard_bracket' or always computed for comparison)
    bracket_profiles: list[BracketProfileResult] = Field(default_factory=list)

    # Time series for interactive charting
    timeseries: dict[str, list[Any]]

    # Trade ledger
    trades: list[TradeRecord]

    # Econometric & statistical diagnostics
    bootstrap_cis: list[BootstrapConfidenceInterval]
    factor_regression: FactorRegressionResult
    pairs_diagnostics: PairsDiagnostics | None = None
    cost_sensitivity: list[CostSensitivityRow]
    break_even_cost_bps: float | None = None
    subperiods: list[SubperiodRow]
    parameter_surface: ParameterSensitivitySurface | None = None

    # Step 3: Academic Literature & Methodology Grounding
    literature_review: LiteratureReviewSection

    # Step 5: Adversarial stress tests and optional model commentary
    stress_test_suite: StressTestSuite
    commentary_model: str | None = None
    commentary_error: str | None = None

    # Step 6: Empirical Hypothesis Scorecard & Calculation Ledger
    scorecard: HypothesisScorecard

    # Objective methodological notes & limitations (strictly neutral)
    methodology_notes: list[str]
    limitations: list[str]
    guardrail_verified: bool = True

