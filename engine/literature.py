"""Step 3 — Peer-Reviewed Quantitative Literature & Backtesting Rigor Grounding.

Maps each strategy family and signal variant to foundational financial economics and
econometric papers, explains the documented empirical pitfalls in academic literature,
and details how Alpha Auditor's deterministic backtester guards against them.
"""

from __future__ import annotations


from engine.guardrails import sanitize_research_notes
from engine.schemas import (
    LiteraturePaper,
    LiteratureReviewSection,
    PairsDiagnostics,
    StrategySpec,
)


STRATEGY_PAPERS_BY_TEMPLATE: dict[str, list[LiteraturePaper]] = {
    "return_threshold": [
        LiteraturePaper(
            authors="Jegadeesh, N.",
            year=1990,
            title="Evidence of Predictable Behavior of Security Returns",
            journal="Journal of Finance, 45(3), 881–898",
            category="Strategy Mechanism",
            core_finding=(
                "Documents statistically significant short-horizon return reversals at 1-week to 1-month "
                "horizons driven by temporary inventory imbalances and liquidity provision."
            ),
            empirical_pitfall_warned=(
                "Bid-ask bounce and execution latency can account for a substantial fraction of apparent "
                "short-horizon reversal returns if trades are assumed to execute at the signal close."
            ),
            how_alpha_auditor_addresses=(
                "Enforces strict t+1 next-bar execution after signal observation on bar t and evaluates "
                "sensitivity across 0 to 50 bps per-leg transaction costs plus t+2 latency stress tests."
            ),
        ),
        LiteraturePaper(
            authors="Lehmann, B. N.",
            year=1990,
            title="Fads, Martingales, and Market Efficiency",
            journal="Quarterly Journal of Economics, 105(1), 1–28",
            category="Strategy Mechanism",
            core_finding=(
                "Demonstrates that securities experiencing sharp short-term price drops or surges exhibit "
                "subsequent mean reversion as market makers replenish liquidity."
            ),
            empirical_pitfall_warned=(
                "Reversal magnitude is highly regime-dependent and can suffer severe left-tail drawdowns "
                "during structural liquidity cascades or macro regime shifts."
            ),
            how_alpha_auditor_addresses=(
                "Separates In-Sample and Out-of-Sample periods, applies per-trade and portfolio drawdown "
                "circuit breakers, and tests performance across high- vs. low-volatility regimes."
            ),
        ),
        LiteraturePaper(
            authors="Nagel, S.",
            year=2012,
            title="Evaporating Liquidity",
            journal="Review of Financial Studies, 25(7), 2005–2039",
            category="Strategy Mechanism",
            core_finding=(
                "Shows that short-term reversal strategies proxy for the returns to liquidity provision "
                "and vary strongly with market-wide volatility (VIX / realized volatility)."
            ),
            empirical_pitfall_warned=(
                "Apparent alpha may be concentrated in a handful of extreme volatility spikes rather than "
                "representing a stationary everyday edge."
            ),
            how_alpha_auditor_addresses=(
                "Runs an explicit Right-Tail Outlier Removal stress test (removing top 1, 3, and 5 trades) "
                "and a Realized Volatility Regime Breakdown in Step 5."
            ),
        ),
    ],
    "consecutive_streak": [
        LiteraturePaper(
            authors="De Bondt, W. F. M., & Thaler, R.",
            year=1985,
            title="Does the Stock Market Overreact?",
            journal="Journal of Finance, 40(3), 793–805",
            category="Strategy Mechanism",
            core_finding=(
                "Establishes the behavioral overreaction hypothesis: investors systematically overweight "
                "recent streaks of gains or losses, creating subsequent price reversals."
            ),
            empirical_pitfall_warned=(
                "Multi-day streaks can coincide with fundamental news announcements where price continuation "
                "(post-earnings drift) dominates behavioral mean reversion."
            ),
            how_alpha_auditor_addresses=(
                "Evaluates both fixed holding horizons and first-positive-close exits, and compares streak "
                "timing against 500 Monte Carlo random-entry simulations with identical exposure."
            ),
        ),
        LiteraturePaper(
            authors="Barberis, N., Shleifer, A., & Vishny, R.",
            year=1998,
            title="A Model of Investor Sentiment",
            journal="Journal of Financial Economics, 49(3), 307–343",
            category="Strategy Mechanism",
            core_finding=(
                "Models how conservatism bias and representativeness heuristics cause underreaction to "
                "isolated shocks and overreaction to consecutive streaks of identical-sign returns."
            ),
            empirical_pitfall_warned=(
                "Selecting the exact streak length (e.g., 3 vs. 4 days) ex-post on full-sample data "
                "introduces severe data-snooping bias."
            ),
            how_alpha_auditor_addresses=(
                "Sweeps a 2D parameter grid across streak lengths and holding periods and applies the "
                "Bailey & López de Prado (2014) Deflated Sharpe Ratio penalty for multiple testing."
            ),
        ),
        LiteraturePaper(
            authors="Lo, A. W., & MacKinlay, A. C.",
            year=1988,
            title="Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test",
            journal="Review of Financial Studies, 1(1), 41–66",
            category="Strategy Mechanism",
            core_finding=(
                "Develops overlapping variance-ratio tests to distinguish genuine serial dependence in "
                "return streaks from heteroskedastic random walks."
            ),
            empirical_pitfall_warned=(
                "Time-varying volatility (GARCH effects) inflates standard t-statistics on streak returns "
                "even when conditional mean returns are zero."
            ),
            how_alpha_auditor_addresses=(
                "Computes Newey-West (1987) HAC standard errors and Politis & Romano (1994) Stationary "
                "Bootstrap confidence intervals to account for serial correlation and heteroskedasticity."
            ),
        ),
    ],
    "mean_reversion_oscillator": [
        LiteraturePaper(
            authors="Poterba, J. M., & Summers, L. H.",
            year=1988,
            title="Mean Reversion in Stock Prices: Evidence and Implications",
            journal="Journal of Financial Economics, 22(1), 27–59",
            category="Strategy Mechanism",
            core_finding=(
                "Analyzes transitory components in asset prices and tests whether deviations from moving "
                "equilibrium bands revert toward fundamental value."
            ),
            empirical_pitfall_warned=(
                "In small samples, slow mean reversion is statistically difficult to separate from a "
                "random walk with drift."
            ),
            how_alpha_auditor_addresses=(
                "Benchmarks oscillator timing against passive Buy-and-Hold and reports Newey-West HAC "
                "alpha t-statistics alongside Stationary Bootstrap 95% confidence intervals."
            ),
        ),
        LiteraturePaper(
            authors="Brock, W., Lakonishok, J., & LeBaron, B.",
            year=1992,
            title="Simple Technical Trading Rules and the Stochastic Properties of Stock Returns",
            journal="Journal of Finance, 47(5), 1731–1764",
            category="Strategy Mechanism",
            core_finding=(
                "Evaluates moving-average and band-based rules using bootstrap distributions to test "
                "whether conditional returns following technical triggers differ from unconditional returns."
            ),
            empirical_pitfall_warned=(
                "Standard parametric t-tests overstate significance when daily returns exhibit skewness, "
                "leptokurtosis (fat tails), and conditional heteroskedasticity."
            ),
            how_alpha_auditor_addresses=(
                "Implements Lo (2002) / Mertens (2002) non-IID Sharpe standard errors correcting for "
                "3rd and 4th moments (skewness and kurtosis) and Stationary Bootstrap resampling."
            ),
        ),
        LiteraturePaper(
            authors="Park, C. H., & Irwin, S. H.",
            year=2007,
            title="What Do We Know About the Profitability of Technical Analysis?",
            journal="Journal of Economic Surveys, 21(4), 786–826",
            category="Strategy Mechanism",
            core_finding=(
                "Comprehensive meta-study showing that many oscillator strategies lose statistical "
                "significance once realistic transaction costs and out-of-sample splits are enforced."
            ),
            empirical_pitfall_warned=(
                "High-turnover oscillator rules degrade rapidly under realistic bid-ask spread and "
                "slippage assumptions."
            ),
            how_alpha_auditor_addresses=(
                "Computes exact break-even transaction cost in bps per leg and tests Conservative, "
                "Baseline, and Aggressive oscillator thresholds across In-Sample and Out-of-Sample splits."
            ),
        ),
    ],
    "trend_breakout": [
        LiteraturePaper(
            authors="Jegadeesh, N., & Titman, S.",
            year=1993,
            title="Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency",
            journal="Journal of Finance, 48(1), 65–91",
            category="Strategy Mechanism",
            core_finding=(
                "Documents cross-sectional and medium-horizon momentum where assets exhibiting strong "
                "recent trend performance continue to outperform over subsequent months."
            ),
            empirical_pitfall_warned=(
                "Momentum strategies experience sharp 'momentum crashes' during rapid market rebounds "
                "following bear states."
            ),
            how_alpha_auditor_addresses=(
                "Tracks maximum drawdown underwater curves, Calmar and Sortino ratios, and stress-tests "
                "performance across Bull vs. Bear and High- vs. Low-Volatility regimes."
            ),
        ),
        LiteraturePaper(
            authors="Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H.",
            year=2012,
            title="Time Series Momentum",
            journal="Journal of Financial Economics, 104(2), 228–250",
            category="Strategy Mechanism",
            core_finding=(
                "Demonstrates significant time-series momentum (trend following on an asset's own past "
                "returns) across equity indices, currencies, commodities, and bond futures."
            ),
            empirical_pitfall_warned=(
                "Trend-following return distributions are positively skewed with low win rates, relying "
                "heavily on a small subset of sustained trend trades."
            ),
            how_alpha_auditor_addresses=(
                "Executes a Right-Tail Outlier Removal stress test in Step 5 to quantify how much of "
                "cumulative performance depends on the single top 1, 3, or 5 trend trades."
            ),
        ),
        LiteraturePaper(
            authors="Daniel, K., & Moskowitz, T. J.",
            year=2016,
            title="Momentum Crashes",
            journal="Journal of Financial Economics, 122(2), 221–247",
            category="Strategy Mechanism",
            core_finding=(
                "Shows that trend/momentum strategies have time-varying beta exposure that can produce "
                "severe drawdowns during inflection points."
            ),
            empirical_pitfall_warned=(
                "Comparing raw trend returns without adjusting for market beta conflates passive bull-market "
                "exposure with genuine timing skill."
            ),
            how_alpha_auditor_addresses=(
                "Runs a Newey-West HAC factor regression against the benchmark to decompose strategy "
                "returns into market beta and annualized Jensen's alpha."
            ),
        ),
    ],
    "pairs_cointegration_reversion": [
        LiteraturePaper(
            authors="Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G.",
            year=2006,
            title="Pairs Trading: Performance of a Relative-Value Arbitrage Rule",
            journal="Review of Financial Studies, 19(3), 797–827",
            category="Strategy Mechanism",
            core_finding=(
                "Seminal empirical study testing two-standard-deviation spread divergence triggers between "
                "historically co-moving equities with strict formation vs. trading split separation."
            ),
            empirical_pitfall_warned=(
                "Estimating hedge ratios or spread standard deviations on the full sample introduces severe "
                "look-ahead bias that artificially guarantees mean reversion."
            ),
            how_alpha_auditor_addresses=(
                "Strictly estimates OLS hedge ratios on the In-Sample formation window (or causal rolling "
                "window) and applies two-leg transaction costs (4 executions per round-trip)."
            ),
        ),
        LiteraturePaper(
            authors="Engle, R. F., & Granger, C. W. J.",
            year=1987,
            title="Co-Integration and Error Correction: Representation, Estimation, and Testing",
            journal="Econometrica, 55(2), 251–276",
            category="Strategy Mechanism",
            core_finding=(
                "Establishes the formal two-step cointegration test demonstrating when a linear combination "
                "of two non-stationary I(1) price series forms a stationary I(0) equilibrium spread."
            ),
            empirical_pitfall_warned=(
                "High return correlation does not imply cointegration; two correlated random walks can "
                "diverge permanently without an error-correction mechanism."
            ),
            how_alpha_auditor_addresses=(
                "Computes Augmented Dickey-Fuller (ADF), Engle-Granger cointegration p-values, Ornstein-Uhlenbeck "
                "half-life of mean reversion, Hurst exponent, and Lo-MacKinlay Variance Ratio."
            ),
        ),
        LiteraturePaper(
            authors="Do, B., & Faff, R.",
            year=2010,
            title="Does Simple Pairs Trading Still Work?",
            journal="Financial Analysts Journal, 66(4), 83–95",
            category="Strategy Mechanism",
            core_finding=(
                "Shows that cointegration relationships frequently experience structural breaks out-of-sample, "
                "causing diverging spreads that fail to revert."
            ),
            empirical_pitfall_warned=(
                "Pairs that appear cointegrated in-sample can suffer regime divergence out-of-sample when "
                "company fundamentals decouple."
            ),
            how_alpha_auditor_addresses=(
                "Enforces stop-loss z-score bounds (e.g., |z| >= 4.0) and executes a dedicated Out-of-Sample "
                "Spread Stability & Structural Break Stress Test in Step 5."
            ),
        ),
    ],
}


METHODOLOGY_RIGOR_PAPERS: list[LiteraturePaper] = [
    LiteraturePaper(
        authors="Lo, A. W.",
        year=2002,
        title="The Statistics of Sharpe Ratios",
        journal="Financial Analysts Journal, 58(4), 36–52",
        category="Econometric & Backtest Rigor",
        core_finding=(
            "Derives the asymptotic distribution and standard error of the Sharpe ratio under non-IID "
            "returns, accounting for serial correlation and higher-order moments."
        ),
        empirical_pitfall_warned=(
            "Assuming IID normal returns understates Sharpe ratio estimation error when strategies exhibit "
            "negative skewness or fat tails."
        ),
        how_alpha_auditor_addresses=(
            "Reports Lo (2002) / Mertens (2002) non-IID Sharpe standard errors and Probabilistic Sharpe "
            "Ratios (PSR) explicitly adjusted for sample skewness and kurtosis."
        ),
    ),
    LiteraturePaper(
        authors="Bailey, D. H., & López de Prado, M.",
        year=2014,
        title="The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
        journal="Journal of Portfolio Management, 40(5), 94–107",
        category="Econometric & Backtest Rigor",
        core_finding=(
            "Quantifies the expected maximum Sharpe ratio under the null hypothesis across K independent "
            "parameter trials and deflates observed Sharpe ratios for multiple testing."
        ),
        empirical_pitfall_warned=(
            "Testing multiple parameter thresholds and cherry-picking the best historical setting inflates "
            "false-positive discovery rates."
        ),
        how_alpha_auditor_addresses=(
            "Tracks the exact number of parameter combinations K evaluated in the sensitivity grid and "
            "computes the Deflated Sharpe Ratio (DSR) above the multiple-testing threshold E[max SR_0]."
        ),
    ),
    LiteraturePaper(
        authors="Politis, D. N., & Romano, J. P.",
        year=1994,
        title="The Stationary Bootstrap",
        journal="Journal of the American Statistical Association, 89(428), 1303–1313",
        category="Econometric & Backtest Rigor",
        core_finding=(
            "Constructs stationary resampled time series using geometrically distributed block lengths, "
            "preserving temporal dependence and volatility clustering."
        ),
        empirical_pitfall_warned=(
            "Standard IID bootstrap resampling destroys autocorrelation and GARCH volatility clusters in "
            "financial return series."
        ),
        how_alpha_auditor_addresses=(
            "Generates 95% confidence intervals for annualized return, Sharpe ratio, and mean trade return "
            "using Stationary Bootstrap resampling with automatic block-length selection."
        ),
    ),
    LiteraturePaper(
        authors="Newey, W. K., & West, K. D.",
        year=1987,
        title="A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix",
        journal="Econometrica, 55(3), 703–708",
        category="Econometric & Backtest Rigor",
        core_finding=(
            "Provides a consistent covariance estimator for regression alpha and beta under serially "
            "correlated and heteroskedastic disturbances."
        ),
        empirical_pitfall_warned=(
            "OLS regression t-statistics on daily strategy returns are biased upward when residuals "
            "exhibit autocorrelation or volatility clustering."
        ),
        how_alpha_auditor_addresses=(
            "Estimates benchmark factor regressions using Newey-West HAC standard errors with automatic "
            "Bartlett kernel lag selection."
        ),
    ),
]


def build_literature_review(
    spec: StrategySpec,
    pairs_diag: PairsDiagnostics | None = None,
) -> LiteratureReviewSection:
    """Construct Step 3's peer-reviewed literature & methodological rigor section."""
    strategy_papers = STRATEGY_PAPERS_BY_TEMPLATE.get(
        spec.template_id,
        STRATEGY_PAPERS_BY_TEMPLATE["return_threshold"],
    )

    assets_str = f"{spec.asset_a} and {spec.asset_b}" if spec.asset_b else spec.asset_a

    if spec.template_id == "return_threshold":
        direction_desc = (
            "short-horizon return contrarianism (mean reversion following sharp price drops)"
            if spec.signal_variant == "cum_return_drop"
            else "short-horizon return continuation / shock response"
        )
        mechanism = (
            f"This experiment investigates {direction_desc} on {assets_str} using the '{spec.signal_variant}' "
            f"trigger ({spec.parameters.entry_threshold:.2f} threshold over a {spec.parameters.lookback_window}-day "
            f"window). In empirical market microstructure literature (Jegadeesh, 1990; Lehmann, 1990; Nagel, 2012), "
            f"sharp short-term price dislocations often reflect temporary order-flow imbalances and inventory "
            f"constraints among liquidity providers rather than permanent shifts in fundamental value."
        )
        synthesis = (
            f"Academic literature indicates that short-horizon threshold triggers on {assets_str} are highly "
            f"sensitive to execution timing (bid-ask bounce on the signal bar) and volatility clustering. "
            f"To prevent false discovery, this audit enforces t+1 execution latency ({spec.execution.execution_timing}), "
            f"evaluates net performance across 0–50 bps per-leg transaction costs, and tests whether returns "
            f"survive removal of extreme volatility outlier days."
        )
    elif spec.template_id == "consecutive_streak":
        mechanism = (
            f"This experiment tests multi-day return streak behavior on {assets_str} after "
            f"{spec.parameters.streak_length} consecutive days ({spec.signal_variant}). Behavioral finance "
            f"models (De Bondt & Thaler, 1985; Barberis, Shleifer & Vishny, 1998) posit that consecutive "
            f"same-sign daily returns can induce short-term investor overreaction or liquidity exhaustion, "
            f"while random-walk variance-ratio tests (Lo & MacKinlay, 1988) test whether such streaks differ "
            f"from heteroskedastic noise."
        )
        synthesis = (
            f"Empirical studies warn that selecting a specific streak length ({spec.parameters.streak_length} days) "
            f"can suffer from sample-selection bias and time-varying volatility. Alpha Auditor mitigates this by "
            f"comparing the selected rule against a 25-cell parameter grid with Deflated Sharpe Ratio (DSR) "
            f"adjustment and benchmarking signal timing against 500 Monte Carlo random-entry simulations."
        )
    elif spec.template_id == "mean_reversion_oscillator":
        mechanism = (
            f"This experiment evaluates oscillator-based mean reversion on {assets_str} via '{spec.signal_variant}' "
            f"(lookback {spec.parameters.lookback_window} days, entry threshold {spec.parameters.entry_threshold:.2f}). "
            f"Following Brock, Lakonishok & LeBaron (1992) and Poterba & Summers (1988), standardized price "
            f"oscillators test whether temporary deviations from a rolling equilibrium band revert once "
            f"short-term buying or selling pressure subsides."
        )
        synthesis = (
            f"Meta-analyses of technical oscillators (Park & Irwin, 2007) demonstrate that apparent in-sample "
            f"edges frequently erode out-of-sample or under realistic transaction costs due to trade turnover. "
            f"Alpha Auditor addresses this via a strict {int(spec.execution.train_split_ratio * 100)}/"
            f"{100 - int(spec.execution.train_split_ratio * 100)} chronological split, Stationary Bootstrap "
            f"confidence intervals, and break-even transaction cost estimation."
        )
    elif spec.template_id == "trend_breakout":
        mechanism = (
            f"This experiment examines time-series momentum and trend continuation on {assets_str} using "
            f"'{spec.signal_variant}' (primary window {spec.parameters.lookback_window}d, secondary window "
            f"{spec.parameters.secondary_window}d). Time-series momentum literature (Moskowitz, Ooi & Pedersen, "
            f"2012; Jegadeesh & Titman, 1993) documents that slow diffusion of fundamental information and "
            f"institutional herding can generate persistent medium-horizon price trends."
        )
        synthesis = (
            "Trend-following strategies typically exhibit positive return skewness—characterized by a lower "
            "win rate offset by a small number of large trend-capture trades—and vulnerability to sudden "
            "rebound crashes (Daniel & Moskowitz, 2016). Alpha Auditor decomposes returns into passive "
            "market beta versus Newey-West HAC alpha and stress-tests right-tail trade concentration."
        )
    else:
        coint_detail = ""
        if pairs_diag is not None:
            coint_detail = (
                f" On the formation sample ({pairs_diag.formation_start} to {pairs_diag.formation_end}), "
                f"the pair exhibits return correlation {pairs_diag.return_correlation:.2f}, ADF p-value "
                f"{pairs_diag.adf_p_value:.4f}, and Ornstein-Uhlenbeck half-life "
                f"{pairs_diag.half_life_days if pairs_diag.half_life_days is not None else 'N/A'} days."
            )
        mechanism = (
            f"This experiment investigates statistical arbitrage and relative-value convergence between "
            f"{assets_str} using '{spec.signal_variant}' spread construction (entry |z| >= "
            f"{spec.parameters.entry_threshold:.2f}, exit |z| <= {spec.parameters.exit_threshold:.2f}). "
            f"Following Gatev, Goetzmann & Rouwenhorst (2006) and Engle & Granger (1987), pairs trading "
            f"requires not merely price correlation, but a stationary cointegrated spread with a finite "
            f"Ornstein-Uhlenbeck mean-reversion half-life.{coint_detail}"
        )
        synthesis = (
            f"Empirical pairs trading studies (Do & Faff, 2010) emphasize two critical failure modes: "
            f"(1) look-ahead bias when estimating hedge ratios on the full sample, and (2) out-of-sample "
            f"structural breaks where diverging spreads fail to revert. Alpha Auditor locks OLS hedge ratio "
            f"estimation strictly to the In-Sample formation window, charges 4 leg executions per round-trip, "
            f"and enforces a |z| >= {spec.parameters.stop_z_threshold or 4.0:.1f} divergence stop-loss."
        )


    rigor_checklist: list[dict[str, str]] = [
        {
            "safeguard": "Zero Look-Ahead Signal & Execution Separation",
            "status": "ENFORCED",
            "detail": (
                f"Signals are computed strictly using data up to close of bar t; trade entry executes on bar t+1 "
                f"({spec.execution.execution_timing}) with {spec.execution.commission_bps + spec.execution.slippage_bps:.1f} "
                f"bps per-leg transaction cost."
            ),
        },
        {
            "safeguard": "Chronological In-Sample / Out-of-Sample Split",
            "status": "ENFORCED",
            "detail": (
                f"First {int(spec.execution.train_split_ratio * 100)}% of bars allocated to Formation/In-Sample; "
                f"remaining {100 - int(spec.execution.train_split_ratio * 100)}% held out for Out-of-Sample validation."
            ),
        },
        {
            "safeguard": "Non-IID Higher-Moment & Serial Correlation Corrections",
            "status": "ENFORCED",
            "detail": (
                "Applies Lo (2002) / Mertens (2002) skewness-kurtosis Sharpe SE, Newey-West (1987) HAC alpha "
                "t-statistics, and Politis & Romano (1994) Stationary Bootstrap 95% confidence intervals."
            ),
        },
        {
            "safeguard": "Multiple-Testing Deflation (Selection Bias Guard)",
            "status": "ENFORCED",
            "detail": (
                "Evaluates a 5×5 parameter neighborhood grid (K=25 trials) and reports the Bailey & López de Prado "
                "(2014) Deflated Sharpe Ratio (DSR) relative to the expected maximum null Sharpe ratio."
            ),
        },
        {
            "safeguard": "Corporate-Action Adjusted Price Series",
            "status": "ENFORCED",
            "detail": (
                "All open, high, low, and close series are adjusted using split and dividend adjustment factors "
                "(AdjClose / Close) so historical splits do not create artificial price jumps."
            ),
        },
    ]

    cleaned_notes = sanitize_research_notes([mechanism, synthesis])

    return LiteratureReviewSection(
        theoretical_mechanism_summary=cleaned_notes[0],
        gemini_literature_synthesis=cleaned_notes[1],
        strategy_papers=strategy_papers,
        methodology_papers=METHODOLOGY_RIGOR_PAPERS,
        rigor_checklist=rigor_checklist,
    )
