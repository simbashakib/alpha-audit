"""Step 1 & Step 2 — Deterministic Hypothesis Intake, Smart Steering,
Scope Filtering & Structured Specification Compiler.

1. Explains what can be asked and filters out non-financial or unsupported prompts.
2. Conversational Gemini interpretation (`gemini_reply`) + structured `understood_summary`.
3. Smart Steering (`steered_cards`): When a user's prompt is vague, informal, or slightly off,
   steers it toward 3 concrete, literature-backed testable hypotheses with clickable cards.
"""

from __future__ import annotations

import re
from typing import Any

from engine.data_provider import ENTITY_TO_TICKER, resolve_ticker
from engine.guardrails import sanitize_research_notes
from engine.schemas import (
    AmbiguityQuestion,
    ChatTurn,
    ExecutionAssumptions,
    InterpretationResponse,
    RiskControls,
    SteeredHypothesisCard,
    StrategyParameters,
    StrategySpec,
    TemplateId,
)

UNSUPPORTED_KEYWORDS = (
    (
        r"\b(option|options|call option|put option|0dte|gamma|iv crush|implied volatility surface|black-scholes)\b",
        "Options chain and implied volatility surface backtesting requires historical strike/expiry tick data not present in the verified daily OHLCV store. Formulate the hypothesis on underlying equity, ETF, or volatility index price series (e.g. SPY, QQQ, AAPL).",
    ),
    (
        r"\b(order book|level 2|level ii|l2 book|high-frequency|hft|microsecond|millisecond|bid-ask queue)\b",
        "Limit-order-book and sub-second microstructure hypotheses require L2/L3 tick data. Alpha Auditor evaluates daily bar hypotheses with explicit basis-point slippage.",
    ),
    (
        r"\b(insider trading|non-public|front-run|manipulat|wash trad)\b",
        "Alpha Auditor only evaluates lawful historical price-based market hypotheses.",
    ),
    (
        r"\b(predict tomorrow's exact price|what will .* price be tomorrow|should i buy .* right now|tell me what stock to buy|guaranteed profit)\b",
        "Alpha Auditor is a historical hypothesis research workbench, not a price predictor or investment adviser. State a testable historical rule (e.g., 'I think the S&P 500 tends to recover the day after falling more than 2%').",
    ),
    (
        r"\b(recipe|weather|football|soccer|premier league|movie|restaurant|homework|poem|joke|write an essay|translate|horoscope|medical|doctor)\b",
        "This query is outside the scope of Alpha Auditor. Alpha Auditor is dedicated exclusively to empirical quantitative finance and historical trading hypothesis research.",
    ),
)

FINANCIAL_DOMAIN_TERMS = re.compile(
    r"\b(stock|stocks|share|shares|etf|index|market|price|prices|return|returns|drop|drops|fall|falls|falling|"
    r"rise|rises|rising|rally|bounce|recover|revert|reversion|converge|convergence|diverge|divergence|pair|pairs|"
    r"spread|cointegrat|moving average|sma|ema|rsi|bollinger|z-score|zscore|volatility|breakout|trend|momentum|"
    r"streak|consecutive|positive|negative|up|down|buy|sell|long|short|hold|holding|day|days|trading|trade|"
    r"tech|bank|oil|energy|gold|crypto|bitcoin|ethereum|treasury|bond|sector|panic|dip|crash|surge|jump|overbought|oversold|"
    r"stop|stop-loss|stoploss|loss|drawdown|slippage|commission|fee|fees|bps|basis points|lookback|window|threshold|horizon)\b",
    re.I,
)


def _extract_tickers_from_text(text: str) -> list[str]:
    """Identify distinct asset tickers from entity names or uppercase symbols."""
    lower = text.lower()
    found: list[tuple[int, str]] = []

    for alias in sorted(ENTITY_TO_TICKER.keys(), key=len, reverse=True):
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        for m in re.finditer(pattern, lower):
            ticker = ENTITY_TO_TICKER[alias]
            if ticker not in [t for _, t in found]:
                found.append((m.start(), ticker))
                break

    STOP_UPPER = {
        "I", "A", "AN", "THE", "AND", "OR", "IF", "WHEN", "AFTER", "BEFORE",
        "BUY", "SELL", "LONG", "SHORT", "DAY", "DAYS", "SMA", "EMA", "RSI",
        "MA", "STD", "SD", "Z", "ETF", "ETFS", "US", "UK", "VS", "OVER", "UNDER",
        "MORE", "LESS", "THAN", "BY", "IN", "ON", "TO", "FOR", "WITH", "AT",
        "BPS", "PCT", "OLS", "ADF", "AI", "IT", "IS", "BE", "DO", "GO", "SO",
        "BB", "MACD", "ATR", "VWAP", "USD", "GBP", "EUR", "JPY", "ATH", "ATL",
        "VIX", "GDP", "CPI", "FED", "FOMC", "PE", "EPS", "OOS", "SR", "DD",
        "CI", "SE", "PSR", "DSR", "HAC", "OU", "VR", "HIGH", "HIGHS", "HOLD",
        "DROP", "DROPS", "FALL", "FALLS", "RISE", "RISES", "UP", "DOWN", "TEST",
        "DOES", "WHAT", "HOW", "WHY", "CAN", "WILL", "FROM", "INTO", "NEXT",
        "LAST", "PAST", "MEAN", "BAND", "BANDS", "PAIR", "PAIRS", "BOTH", "EACH",
        "WEEK", "WEEKS", "YEAR", "YEARS", "BAR", "BARS", "OPEN", "CLOSE", "PRICE",
        "STOCK", "STOCKS", "SHARE", "SHARES", "TREND", "CROSS", "ABOVE", "BELOW",
        "UPPER", "LOWER", "DAILY", "SHOCK", "PULL", "BACK", "GAIN", "GAINS",
        "LOSS", "LOSSES", "MOVE", "MOVES", "RULE", "RULES", "RATE", "RATES",
        "RATIO", "SPREAD", "LEVEL", "INDEX", "VALUE", "GROWTH", "RETURN",
        "RETURNS", "PROFIT", "RISK", "ALPHA", "BETA", "SIGMA", "DELTA", "GAMMA",
        "THETA", "VEGA", "BULL", "BEAR", "CRASH", "RALLY", "SURGE", "JUMP",
        "JUMPS", "BOUNCE", "REVERT", "REVERTS", "ENTER", "EXIT", "START", "END",
        "FIRST", "THIRD", "FIFTH", "SPLIT", "TOTAL", "NET", "GROSS", "FIXED",
        "WINDOW", "PERIOD", "TARGET", "STOP", "LIMIT", "TRUE", "FALSE", "NULL",
    }
    for m in re.finditer(r"(?<![A-Za-z0-9&'])\$?(\^[A-Z]{2,6}|[A-Z]{1,5}(?:-[A-Z]{1,3}|\.[A-Z]{1,2})?)\b", text):
        cand = m.group(1)
        if cand in STOP_UPPER:
            continue
        # Single-letter tickers (like V, C, F) require '$' prefix unless paired as 'V and MA' / 'V / MA'
        if len(cand) == 1 and not m.group(0).startswith("$"):
            if cand == "V" and re.search(r"\bV\s*(?:/|and|vs\.?|versus)\s*MA\b", text):
                pass
            else:
                continue
        # Avoid matching 'LOW' when used as "20-day LOW" / "LOW to HIGH" unless paired with HD or $LOW
        if cand == "LOW" and not m.group(0).startswith("$"):
            if not re.search(r"\bHD\b", text):
                continue
        # Avoid matching 'COST' when used as "transaction COST" unless paired with WMT or $COST
        if cand == "COST" and not m.group(0).startswith("$"):
            prefix = text[max(0, m.start() - 20) : m.start()].lower()
            if any(w in prefix for w in ("transaction", "trading", "slippage", "zero", "high", "low", "no")):
                continue
        if m.group(0).startswith("$") or cand.isupper():
            resolved = resolve_ticker(cand)
            if resolved not in [t for _, t in found]:
                found.append((m.start(), resolved))

    found.sort(key=lambda x: x[0])
    return [t for _, t in found]


def _build_understood_summary(spec: StrategySpec) -> dict[str, str]:
    """Build Step 2's structured 'What Gemini Understood' summary table."""
    if spec.strategy_family == "pairs_trading":
        assets_str = f"{spec.asset_a} (Leg A) vs. {spec.asset_b} (Leg B)"
        trigger_str = (
            f"Enter Long/Short spread when {spec.signal_variant} spread |z-score| ≥ "
            f"{spec.parameters.entry_threshold:.2f}σ (rolling {spec.parameters.lookback_window}d window)"
        )
        exit_str = (
            f"Exit when spread reverts to |z| ≤ {spec.parameters.exit_threshold:.2f}σ, "
            f"stop-loss at |z| ≥ {spec.parameters.stop_z_threshold or 4.0:.1f}σ, or max {spec.risk_controls.max_holding_days}d"
        )
    elif spec.template_id == "return_threshold":
        assets_str = f"{spec.asset_a} (Single-Asset Equity/ETF)"
        verb = "falls ≤ -" if spec.signal_variant == "cum_return_drop" else "rises ≥ +"
        trigger_str = (
            f"{spec.asset_a} {verb}{spec.parameters.entry_threshold:.2f}% over "
            f"{spec.parameters.lookback_window} trading day(s) ({spec.signal_variant})"
        )
        exit_str = (
            f"Enter {spec.parameters.direction.upper()} at next bar ({spec.execution.execution_timing}) "
            f"and hold for {spec.parameters.holding_period} trading day(s) ({spec.parameters.exit_mode})"
        )
    elif spec.template_id == "consecutive_streak":
        assets_str = f"{spec.asset_a} (Single-Asset Equity/ETF)"
        streak_kind = "negative (down)" if spec.signal_variant == "negative_streak" else "positive (up)"
        trigger_str = f"{spec.parameters.streak_length} consecutive {streak_kind} trading days on {spec.asset_a}"
        exit_str = (
            f"Enter {spec.parameters.direction.upper()} at next bar ({spec.execution.execution_timing}) "
            f"and hold for {spec.parameters.holding_period} trading day(s) ({spec.parameters.exit_mode})"
        )
    elif spec.template_id == "mean_reversion_oscillator":
        assets_str = f"{spec.asset_a} (Single-Asset Equity/ETF)"
        trigger_str = (
            f"{spec.signal_variant} over {spec.parameters.lookback_window}d window crosses entry threshold "
            f"{spec.parameters.entry_threshold:.2f}"
        )
        exit_str = (
            f"Hold {spec.parameters.direction.upper()} until oscillator reverts to {spec.parameters.exit_threshold:.2f} "
            f"(or max {spec.parameters.holding_period}d)"
        )
    else:
        assets_str = f"{spec.asset_a} (Single-Asset Equity/ETF)"
        trigger_str = (
            f"{spec.signal_variant} with fast window {spec.parameters.lookback_window}d and slow window "
            f"{spec.parameters.secondary_window}d"
        )
        exit_str = f"Hold {spec.parameters.direction.upper()} until signal reversal or max {spec.risk_controls.max_holding_days}d"

    total_cost = spec.execution.commission_bps + spec.execution.slippage_bps
    return {
        "Target Asset(s)": assets_str,
        "Strategy Family & Template": f"{spec.strategy_family.replace('_', ' ').title()} — {spec.template_id.replace('_', ' ').title()}",
        "Signal Trigger Condition": trigger_str,
        "Holding & Exit Rule": exit_str,
        "Execution & Benchmark": (
            f"Fill at t+1 ({spec.execution.execution_timing}), {total_cost:.0f} bps/leg cost, "
            f"benchmarked vs. {spec.benchmark_ticker} ({spec.start_date} to {spec.end_date})"
        ),
    }


def _build_steered_cards(
    base_spec: StrategySpec,
    lower_text: str,
) -> list[SteeredHypothesisCard]:
    """Generate 3 concrete, clickable steered hypothesis cards so the user can refine
    vague intuitions or explore closely related institutional formulations.
    """
    a = base_spec.asset_a
    b = base_spec.asset_b or ("PEP" if a == "KO" else ("CVX" if a == "XOM" else "QQQ"))
    cards: list[SteeredHypothesisCard] = []

    if base_spec.strategy_family == "pairs_trading":
        base_pair_fields = (
            list(base_spec.user_specified_fields)
            if base_spec.user_specified_fields
            else ["asset_a", "asset_b"]
        )
        spec_ols = base_spec.model_copy(deep=True)
        spec_ols.signal_variant = "ols_log"
        spec_ols.parameters.entry_threshold = 2.0
        spec_ols.parameters.exit_threshold = 0.25
        spec_ols.parameters.lookback_window = 20
        spec_ols.hypothesis_text = (
            f"I think {a} and {b} log-price spread tends to revert to the mean when their 20-day OLS spread z-score exceeds 2.0 standard deviations."
        )
        spec_ols.title = f"{a} / {b} Static OLS Cointegration Reversion (2.0σ Entry)"
        spec_ols.user_specified_fields = list(
            dict.fromkeys(
                base_pair_fields
                + [
                    "signal_variant",
                    "parameters.entry_threshold",
                    "parameters.exit_threshold",
                    "parameters.lookback_window",
                ]
            )
        )
        spec_ols.software_inferred_defaults = {
            "parameters.holding_period": f"{spec_ols.parameters.holding_period}d max",
            "execution.execution_timing": "next_open (t+1)",
            "execution.cost_bps": "10 bps (5 comm + 5 slip)",
            "risk_controls.max_drawdown_circuit_breaker_pct": "25.0%",
        }
        cards.append(
            SteeredHypothesisCard(
                label=f"{a}/{b} Static OLS Spread (2.0σ → 0.25σ)",
                hypothesis_text=spec_ols.hypothesis_text,
                rationale="Gatev et al. (2006) & Engle-Granger (1987) formation-locked log-price cointegration spread.",
                target_assets=f"{a} / {b}",
                template_badge="Pairs Cointegration (OLS)",
                proposed_spec=spec_ols,
            )
        )

        spec_roll = base_spec.model_copy(deep=True)
        spec_roll.signal_variant = "rolling_ols_log"
        spec_roll.parameters.entry_threshold = 2.0
        spec_roll.parameters.exit_threshold = 0.0
        spec_roll.parameters.lookback_window = 30
        spec_roll.parameters.secondary_window = 60
        spec_roll.hypothesis_text = (
            f"I think {a} and {b} converge using a time-varying 60-day rolling OLS hedge ratio when the 30-day spread z-score diverges beyond 2.0σ."
        )
        spec_roll.title = f"{a} / {b} Dynamic Rolling-Beta Spread Convergence"
        spec_roll.user_specified_fields = list(
            dict.fromkeys(
                base_pair_fields
                + [
                    "signal_variant",
                    "parameters.entry_threshold",
                    "parameters.exit_threshold",
                    "parameters.lookback_window",
                    "parameters.secondary_window",
                ]
            )
        )
        spec_roll.software_inferred_defaults = {
            "parameters.holding_period": f"{spec_roll.parameters.holding_period}d max",
            "execution.execution_timing": "next_open (t+1)",
            "execution.cost_bps": "10 bps (5 comm + 5 slip)",
            "risk_controls.max_drawdown_circuit_breaker_pct": "25.0%",
        }
        cards.append(
            SteeredHypothesisCard(
                label=f"{a}/{b} Dynamic Rolling Hedge Ratio (60d β)",
                hypothesis_text=spec_roll.hypothesis_text,
                rationale="Adapts to gradual structural drift in the relative hedge ratio between the two legs.",
                target_assets=f"{a} / {b}",
                template_badge="Pairs Rolling Beta",
                proposed_spec=spec_roll,
            )
        )

        spec_tight = base_spec.model_copy(deep=True)
        spec_tight.signal_variant = "ols_log"
        spec_tight.parameters.entry_threshold = 2.5
        spec_tight.parameters.exit_threshold = 0.25
        spec_tight.parameters.lookback_window = 20
        spec_tight.hypothesis_text = (
            f"I think {a} and {b} revert reliably only after extreme 2.5 standard deviation spread divergences."
        )
        spec_tight.title = f"{a} / {b} High-Conviction 2.5σ Divergence Filter"
        spec_tight.user_specified_fields = list(
            dict.fromkeys(
                base_pair_fields
                + [
                    "signal_variant",
                    "parameters.entry_threshold",
                    "parameters.exit_threshold",
                    "parameters.lookback_window",
                ]
            )
        )
        spec_tight.software_inferred_defaults = {
            "parameters.holding_period": f"{spec_tight.parameters.holding_period}d max",
            "execution.execution_timing": "next_open (t+1)",
            "execution.cost_bps": "10 bps (5 comm + 5 slip)",
            "risk_controls.max_drawdown_circuit_breaker_pct": "25.0%",
        }
        cards.append(
            SteeredHypothesisCard(
                label=f"{a}/{b} High-Threshold Tail Divergence (2.5σ)",
                hypothesis_text=spec_tight.hypothesis_text,
                rationale="Reduces trade turnover and transaction cost drag by waiting for 2.5σ tail dislocations.",
                target_assets=f"{a} / {b}",
                template_badge="Pairs High-Selectivity",
                proposed_spec=spec_tight,
            )
        )
        return cards

    # Single-asset steered cards
    base_user_fields = list(base_spec.user_specified_fields) if base_spec.user_specified_fields else ["asset_a"]
    # Card 1: Short-Horizon Percentage Drop Reversion (Jegadeesh 1990)
    spec_shock = StrategySpec(
        hypothesis_text=f"I think {a} tends to recover over the next 3 trading days after falling more than 2% in a single day.",
        title=f"{a} -2.0% Daily Shock Mean Reversion (3d Hold)",
        strategy_family="single_asset",
        template_id="return_threshold",
        signal_variant="cum_return_drop",
        asset_a=a,
        benchmark_ticker="SPY",
        parameters=StrategyParameters(
            lookback_window=1,
            entry_threshold=2.0,
            exit_threshold=0.0,
            holding_period=3,
            exit_mode="fixed_hold",
            direction="long",
        ),
        user_specified_fields=list(dict.fromkeys(base_user_fields + ["parameters.entry_threshold", "parameters.holding_period"])),
        software_inferred_defaults={
            "execution.execution_timing": "next_open (t+1)",
            "execution.cost_bps": "10 bps (5 comm + 5 slip)",
            "risk_controls.max_drawdown_circuit_breaker_pct": "25.0%",
        },
    )
    cards.append(
        SteeredHypothesisCard(
            label=f"{a} Single-Day -2% Shock Rebound (3d Hold)",
            hypothesis_text=spec_shock.hypothesis_text,
            rationale="Tests short-horizon liquidity-provision bounce after >2% daily selloffs (Jegadeesh, 1990).",
            target_assets=a,
            template_badge="Return Shock Threshold",
            proposed_spec=spec_shock,
        )
    )

    # Card 2: Multi-Day Consecutive Negative Streak (De Bondt & Thaler 1985)
    spec_streak = StrategySpec(
        hypothesis_text=f"I think {a} tends to rise over the next 3 trading days after 3 consecutive negative trading days.",
        title=f"{a} 3-Day Consecutive Negative Streak Reversal",
        strategy_family="single_asset",
        template_id="consecutive_streak",
        signal_variant="negative_streak",
        asset_a=a,
        benchmark_ticker="SPY",
        parameters=StrategyParameters(
            lookback_window=3,
            streak_length=3,
            entry_threshold=3.0,
            holding_period=3,
            exit_mode="fixed_hold",
            direction="long",
        ),
        user_specified_fields=list(dict.fromkeys(base_user_fields + ["parameters.streak_length", "parameters.holding_period"])),
        software_inferred_defaults={
            "execution.execution_timing": "next_open (t+1)",
            "execution.cost_bps": "10 bps (5 comm + 5 slip)",
            "risk_controls.max_drawdown_circuit_breaker_pct": "25.0%",
        },
    )
    cards.append(
        SteeredHypothesisCard(
            label=f"{a} 3-Consecutive Down-Days Reversal",
            hypothesis_text=spec_streak.hypothesis_text,
            rationale="Tests whether multi-day selling streaks induce short-term overreaction and subsequent mean reversion.",
            target_assets=a,
            template_badge="Consecutive Streak",
            proposed_spec=spec_streak,
        )
    )

    # Card 3: Rolling Z-Score / Bollinger Band Mean Reversion (Brock, Lakonishok & LeBaron 1992)
    spec_z = StrategySpec(
        hypothesis_text=f"I think {a} reverts toward its 20-day moving average when its price falls more than 2.0 standard deviations below the 20-day mean.",
        title=f"{a} 20-Day Bollinger / Z-Score (-2.0σ) Mean Reversion",
        strategy_family="single_asset",
        template_id="mean_reversion_oscillator",
        signal_variant="price_zscore",
        asset_a=a,
        benchmark_ticker="SPY",
        parameters=StrategyParameters(
            lookback_window=20,
            entry_threshold=2.0,
            exit_threshold=0.0,
            holding_period=10,
            exit_mode="threshold_reversion",
            direction="long",
        ),
        user_specified_fields=list(dict.fromkeys(base_user_fields + ["parameters.lookback_window", "parameters.entry_threshold"])),
        software_inferred_defaults={
            "parameters.exit_threshold": "0.0σ",
            "parameters.holding_period": "10d max",
            "execution.execution_timing": "next_open (t+1)",
            "execution.cost_bps": "10 bps (5 comm + 5 slip)",
        },
    )
    cards.append(
        SteeredHypothesisCard(
            label=f"{a} Volatility-Normalized -2.0σ Z-Score Reversion",
            hypothesis_text=spec_z.hypothesis_text,
            rationale="Normalizes the trigger by 20-day rolling volatility so thresholds adapt across calm and turbulent regimes.",
            target_assets=a,
            template_badge="Z-Score Oscillator",
            proposed_spec=spec_z,
        )
    )
    return cards


def _compile_hypothesis_deterministic(text: str, template: TemplateId | None = None) -> InterpretationResponse:
    """High-precision deterministic parser translating English hypotheses into StrategySpec,
    conversational Gemini guidance, structured understood summary, and steered cards.
    """
    cleaned = text.strip()
    lower = cleaned.lower()

    # 1. Check explicit unsupported or out-of-scope keywords
    for pattern, reason in UNSUPPORTED_KEYWORDS:
        if re.search(pattern, lower):
            return InterpretationResponse(
                status="unsupported",
                gemini_reply=reason,
                unsupported_reason=reason,
                interpretation_notes=[reason],
                model_used="deterministic-spec-compiler-v2 (scope-guardrail)",
                interpretation_mode="deterministic_rules",
            )

    tickers = _extract_tickers_from_text(cleaned)

    # 2. Check if prompt is completely unrelated to finance/markets/assets
    if template is None and not tickers and not FINANCIAL_DOMAIN_TERMS.search(lower):
        reason = (
            "Your input does not appear to describe a financial market or price hypothesis. "
            "Alpha Auditor tests historical hypotheses on equities, ETFs, indices, crypto, and asset pairs "
            "(for example: 'I think the S&P 500 tends to recover the day after falling more than 2%' or "
            "'I think Coca-Cola and Pepsi converge after their spread diverges beyond 2 standard deviations')."
        )
        return InterpretationResponse(
            status="unsupported",
            gemini_reply=reason,
            unsupported_reason=reason,
            interpretation_notes=[reason],
            model_used="deterministic-spec-compiler-v2 (scope-guardrail)",
            interpretation_mode="deterministic_rules",
        )

    ambiguities: list[AmbiguityQuestion] = []
    notes: list[str] = []

    # Map sector/theme keywords if no explicit ticker was given
    if not tickers:
        if any(w in lower for w in ("tech", "nasdaq", "semiconductor", "ai stock")):
            tickers = ["QQQ"]
        elif any(w in lower for w in ("gold", "precious metal", "safe haven")):
            tickers = ["GLD"]
        elif any(w in lower for w in ("bond", "treasury", "rates", "yield")):
            tickers = ["TLT"]
        elif any(w in lower for w in ("small cap", "russell")):
            tickers = ["IWM"]
        elif any(w in lower for w in ("oil", "energy")):
            tickers = ["XOM", "CVX"] if "pair" in lower or "together" in lower else ["XOM"]
        elif any(w in lower for w in ("bank", "financial")):
            tickers = ["JPM", "BAC"] if "pair" in lower or "together" in lower else ["JPM"]

    # Separate benchmark comparison tickers (e.g. "Does NVDA outperform SPY...") from two-leg pair tickers
    benchmark_ticker = "SPY"
    explicit_pair_terms = bool(
        re.search(
            r"\b(pair|pairs|cointegrat|spread between|ratio between|revert towards each other|"
            r"move together|track each other|two stocks|two assets)\b",
            lower,
        )
    )
    if len(tickers) >= 2 and not explicit_pair_terms:
        bench_match = re.search(
            r"\b(?:outperform(?:s|ing)?|beat(?:s|ing)?|underperform(?:s|ing)?|excess\s+return\s+over|"
            r"alpha\s+over|benchmark(?:ed)?\s+(?:to|against))\s+(?:the\s+)?(s&p\s*500|s&p|spy|qqq|iwm|dia|market)\b",
            lower,
        )
        if bench_match:
            raw_bench = bench_match.group(1).strip()
            inferred_bench = "SPY" if raw_bench in ("s&p 500", "s&p", "spy", "market") else raw_bench.upper()
            if inferred_bench in tickers and tickers[0] != inferred_bench:
                benchmark_ticker = inferred_bench
                tickers = [t for t in tickers if t != inferred_bench]

    # Detect if this is a Pairs Trading hypothesis
    single_asset_self_ref = bool(
        re.search(
            r"\b(its\s+(?:\d+(?:-|\s)?day\s+)?(?:mean|average|moving average|sma|ema|band|bollinger|high|low|trend)|"
            r"the\s+\d+(?:-|\s)?day\s+(?:mean|average|moving average|sma|ema|bollinger|band))\b",
            lower,
        )
    )
    is_pairs = bool(
        len(tickers) >= 2
        or explicit_pair_terms
        or (
            not single_asset_self_ref
            and re.search(
                r"\b(spread\s+diverg|spread\s+converg|converge\s+toward\s+each\s+other|"
                r"relative\s+to\s+each\s+other)\b",
                lower,
            )
        )
    )

    lower_no_risk = re.sub(
        r"(?:(?:per[-\s]?trade\s+)?stop[-\s]?loss(?:\s+of|\s+at)?\s+\d+(?:\.\d+)?\s*(?:%|percent|pct)|\d+(?:\.\d+)?\s*(?:%|percent|pct)\s+(?:per[-\s]?trade\s+)?stop[-\s]?loss|"
        r"max(?:imum)?\s+drawdown(?:\s+limit|\s+breaker)?(?:\s+of|\s+at)?\s+\d+(?:\.\d+)?\s*(?:%|percent|pct)|\d+(?:\.\d+)?\s*(?:%|percent|pct)\s+(?:max(?:imum)?\s+)?drawdown(?:\s+limit|\s+breaker)?)",
        " ",
        lower,
    )
    pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)", lower_no_risk)
    sigma_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:std|standard deviation|sigma|z-score|z\b|σ)", lower_no_risk)
    streak_match = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|\d+)\s+(?:(?:consecutive|straight)\b|(?:[a-z]+\s+){0,2}(?:days?|bars?|sessions?)\s+in\s+a\s+row)",
        lower,
    )
    ma_pair_match = re.search(
        r"(\d{1,3})\s*(?:-|\s)?(?:day|d|bar)?\s*(?:(?:simple|exponential|moving|average|sma|ema)\s+)*(?:and|vs\.?|/|to|(?:is\s+|cross(?:es|ing)?\s+|rises?\s+|falls?\s+|drops?\s+|moves?\s+)?(?:above|below|over|under)\s+(?:its\s+|the\s+)?)(\d{1,3})\s*(?:-|\s)?(?:day|d)?",
        lower,
    )
    shock_window_match = re.search(
        r"(?:fall(?:s|ing)?|drop(?:s|ping)?|lose(?:s|ing)?|declin(?:e|es|ing)|ris(?:e|es|ing)|jump(?:s|ing)?|surg(?:e|es|ing)|gain(?:s|ing)?|rally|rallies|up|down)\s+"
        r"(?:more\s+than\s+|at\s+least\s+|by\s+)?\d+(?:\.\d+)?\s*(?:%|percent|pct)\s+(?:over|in|across|within)\s+(\d{1,3})\s*(?:trading\s+)?(?:day|days|bar|bars)",
        lower,
    )
    explicit_hold_match = re.search(
        r"(?:hold(?:ing)?\s+(?:for\s+)?|(?:over\s+)?the\s+next\s+|following\s+|for\s+the\s+next\s+|target\s+of\s+)(\d{1,3})\s*(?:trading\s+)?(?:day|days|bar|bars)",
        lower,
    )
    fallback_over_match = re.search(
        r"\bover\s+(\d{1,3})\s*(?:trading\s+)?(?:day|days|bar|bars)",
        lower,
    )
    if explicit_hold_match:
        hold_match = explicit_hold_match
    elif fallback_over_match and (not shock_window_match or fallback_over_match.start() != shock_window_match.start(1) - 5):
        # Only use fallback 'over N days' as holding period if it wasn't the shock window itself
        if not shock_window_match or fallback_over_match.group(1) != shock_window_match.group(1):
            hold_match = fallback_over_match
        else:
            hold_match = None
    else:
        hold_match = None

    next_day_mention = bool(re.search(r"\b(the day after|next day|following day|1 day|one day)\b", lower))

    WORD_TO_INT = {
        "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8,
    }
    DEFAULT_PAIR_PARTNER = {
        "KO": "PEP", "PEP": "KO", "XOM": "CVX", "CVX": "XOM",
        "JPM": "BAC", "BAC": "JPM", "V": "MA", "MA": "V",
        "HD": "LOW", "LOW": "HD", "AAPL": "MSFT", "MSFT": "AAPL",
        "NVDA": "AMD", "GOOGL": "META", "META": "GOOGL",
        "SPY": "QQQ", "QQQ": "SPY", "GLD": "SLV",
    }

    if template == "pairs_cointegration_reversion" or (template is None and is_pairs):
        asset_a = tickers[0] if len(tickers) >= 1 else "KO"
        asset_b = tickers[1] if len(tickers) >= 2 else DEFAULT_PAIR_PARTNER.get(asset_a, "PEP" if asset_a != "PEP" else "KO")

        if len(tickers) < 2:
            ambiguities.append(
                AmbiguityQuestion(
                    parameter_key="asset_pair",
                    label="Asset Pair Selection (Leg A / Leg B)",
                    question="Which two assets would you like to pair against each other?",
                    current_value=f"{asset_a} / {asset_b}",
                    suggested_options=["KO / PEP", "XOM / CVX", "V / MA", "JPM / BAC", "HD / LOW"],
                    unit="tickers",
                )
            )

        is_move_together_vague = bool(
            re.search(r"\b(move together|track each other|correlated|co-move)\b", lower)
            and not sigma_match
            and "cointegrat" not in lower
            and "spread" not in lower
        )
        if is_move_together_vague:
            ambiguities.append(
                AmbiguityQuestion(
                    parameter_key="signal_variant",
                    label="Co-Movement Formulation (Spread Model)",
                    question=(
                        "'Move together' can mean (1) positive daily return correlation, (2) a stationary cointegrated spread (ols_log), "
                        "or (3) a time-varying rolling-beta spread (rolling_ols_log). Which spread model should drive trade signals?"
                    ),
                    current_value="ols_log",
                    suggested_options=["ols_log", "rolling_ols_log", "price_ratio"],
                    unit="model",
                )
            )

        entry_z = float(sigma_match.group(1)) if sigma_match else 2.0
        if not sigma_match:
            ambiguities.append(
                AmbiguityQuestion(
                    parameter_key="parameters.entry_threshold",
                    label="Spread Divergence Entry Threshold (z_entry)",
                    question="How wide should the spread divergence be before entering the pair trade?",
                    current_value=2.0,
                    suggested_options=[1.5, 2.0, 2.5, 3.0],
                    unit="σ (z-score)",
                )
            )

        lookback = 20
        lookback_match = re.search(r"(\d{1,3})\s*(?:-|\s)?(?:day|d)\s+(?:window|lookback|z-score|rolling|average|spread)", lower)
        if lookback_match:
            lookback = int(lookback_match.group(1))
        else:
            ambiguities.append(
                AmbiguityQuestion(
                    parameter_key="parameters.lookback_window",
                    label="Spread Z-Score Rolling Lookback Window (L)",
                    question="What rolling window length should standardize the spread?",
                    current_value=20,
                    suggested_options=[15, 20, 40, 60],
                    unit="trading days",
                )
            )

        ambiguities.append(
            AmbiguityQuestion(
                parameter_key="parameters.exit_threshold",
                label="Spread Convergence Exit Threshold (z_exit)",
                question="At what z-score level is the spread considered converged?",
                current_value=0.25,
                suggested_options=[0.0, 0.25, 0.5, 0.75],
                unit="σ (z-score)",
            )
        )

        spread_variant: Any = "ols_log"
        if "rolling" in lower and ("hedge" in lower or "beta" in lower or "ols" in lower):
            spread_variant = "rolling_ols_log"
        elif "ratio" in lower:
            spread_variant = "price_ratio"

        notes.append(
            f"Mapped two-asset relative-value hypothesis to Pairs Cointegration Reversion ({asset_a} vs. {asset_b}) using '{spread_variant}' spread construction."
        )
        notes.append(
            "Scorecard separately audits (1) Daily Return Correlation, (2) Engle-Granger / ADF Long-Run Cointegration, and (3) Active Mean-Reversion Trading Rule P&L."
        )
        notes.append(
            "In 'ols_log' mode, OLS parameters are estimated causally on expanding windows within Formation and frozen at the split boundary for Out-of-Sample evaluation."
        )

        user_fields: list[str] = []
        inferred_defaults: dict[str, Any] = {}
        if len(tickers) >= 1:
            user_fields.append("asset_a")
        else:
            inferred_defaults["asset_a"] = asset_a
        if len(tickers) >= 2:
            user_fields.append("asset_b")
        else:
            inferred_defaults["asset_b"] = asset_b
        if sigma_match:
            user_fields.append("parameters.entry_threshold")
        else:
            inferred_defaults["parameters.entry_threshold"] = f"±{entry_z:.1f}σ"
        if lookback_match:
            user_fields.append("parameters.lookback_window")
        else:
            inferred_defaults["parameters.lookback_window"] = f"{lookback}d"
        if hold_match:
            user_fields.append("parameters.holding_period")
        else:
            inferred_defaults["parameters.holding_period"] = "20d max"
        inferred_defaults["parameters.exit_threshold"] = "±0.25σ"
        inferred_defaults["execution.execution_timing"] = "next_open (t+1)"
        inferred_defaults["execution.cost_bps"] = "10 bps (5 comm + 5 slip)"
        inferred_defaults["risk_controls.max_drawdown_circuit_breaker_pct"] = "25.0%"

        spec = StrategySpec(
            hypothesis_text=cleaned,
            title=f"{asset_a} / {asset_b} Statistical Arbitrage Spread Convergence",
            strategy_family="pairs_trading",
            template_id="pairs_cointegration_reversion",
            signal_variant=spread_variant,
            asset_a=asset_a,
            asset_b=asset_b,
            benchmark_ticker=benchmark_ticker,
            parameters=StrategyParameters(
                lookback_window=lookback,
                secondary_window=60,
                entry_threshold=entry_z,
                exit_threshold=0.25,
                stop_z_threshold=4.0,
                holding_period=int(hold_match.group(1)) if hold_match else 20,
                exit_mode="threshold_reversion",
                direction="both",
            ),
            execution=ExecutionAssumptions(),
            risk_controls=RiskControls(max_holding_days=30, per_trade_stop_loss_pct=12.0),
            selected_diagnostics=[
                "pairs_cointegration",
                "stationary_bootstrap",
                "newey_west_alpha",
                "cost_sensitivity",
                "subperiod_regimes",
                "parameter_sensitivity_grid",
            ],
            user_specified_fields=user_fields,
            software_inferred_defaults=inferred_defaults,
        )
        steered = _build_steered_cards(spec, lower)
        summary = _build_understood_summary(spec)
        if is_move_together_vague:
            reply = (
                f"Saying **{asset_a}** and **{asset_b}** 'move together' spans three distinct empirical questions that Alpha Auditor will evaluate side by side:\n"
                f"1. **Daily Return Co-Movement**: Do their daily returns correlate positively?\n"
                f"2. **Long-Run Cointegration**: Does their log-price spread stay stationary around an equilibrium hedge ratio (Engle-Granger / ADF)?\n"
                f"3. **Pairs Trading Rule P&L**: Does trading temporary **±{entry_z:.1f}σ** spread divergences (over a {lookback}-day window, exiting at **0.25σ**) produce positive net returns after fees and a 25% drawdown circuit breaker?\n\n"
                f"Review or adjust the software-filled defaults in Step 2, or pick one of the refined formulations below."
            )
        else:
            reply = (
                f"We will test whether **{asset_a}** and **{asset_b}** converge after their spread diverges unusually far, while also reporting daily return correlation and Engle-Granger cointegration. "
                f"The specification uses a {lookback}-day rolling spread z-score (entering beyond **±{entry_z:.1f}σ** and exiting at **0.25σ**). "
                f"Review the parameters in Step 2 or select **'Auto-Test Standard Thresholds'** to compare ±1.5σ, ±2.0σ, and ±2.5σ."
            )
        return InterpretationResponse(
            status="needs_clarification" if ambiguities else "ready_for_confirmation",
            gemini_reply=reply,
            understood_summary=summary,
            proposed_spec=spec,
            steered_cards=steered,
            ambiguities=ambiguities,
            interpretation_notes=sanitize_research_notes(notes),
            model_used="deterministic-spec-compiler-v2",
            interpretation_mode="deterministic_rules",
        )

    # Single-Asset Hypothesis Classification
    asset_a = tickers[0] if tickers else "SPY"
    if not tickers:
        ambiguities.append(
            AmbiguityQuestion(
                parameter_key="asset_a",
                label="Target Asset Ticker",
                question="Which ticker symbol would you like to test?",
                current_value="SPY",
                suggested_options=["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "BTC-USD"],
                unit="ticker",
            )
        )

    # Template 2: Consecutive Streak
    has_streak = streak_match or re.search(
        r"\b(consecutive|in a row|straight\s+(?:up|down|positive|negative|red|green|winning|losing))\b", lower
    )
    if template == "consecutive_streak" or (template is None and has_streak):
        raw_n = streak_match.group(1) if streak_match else "3"
        n_days = WORD_TO_INT.get(raw_n, int(raw_n) if raw_n.isdigit() else 3)
        trigger_clause = lower.split("tend")[0] if "tend" in lower else (lower.split(",")[0] if "," in lower else lower)
        is_pos_streak = bool(re.search(r"\b(positive|up|gain|green|rising|higher|winning)\b", trigger_clause))
        variant: Any = "positive_streak" if is_pos_streak else "negative_streak"
        outcome_clause = lower.split("after")[0] if "after" in lower and "when" not in lower[:10] else lower.split(",")[-1]
        if "when" in lower[:15] and "," in lower:
            outcome_clause = lower.split(",", 1)[1]
        elif "after" in lower:
            parts_after = lower.split("after", 1)
            outcome_clause = parts_after[0] if len(parts_after[0]) > 12 else parts_after[1]
        go_short = bool(re.search(r"\b(fall|falls|drop|drops|decline|declines|revert down|pull back|pulls back|short|lose|lower)\b", outcome_clause))
        direction: Any = "short" if go_short else "long"

        hold_days = int(hold_match.group(1)) if hold_match else (1 if next_day_mention else 3)
        if not hold_match and not next_day_mention:
            ambiguities.append(
                AmbiguityQuestion(
                    parameter_key="parameters.holding_period",
                    label="Post-Signal Holding Period (H)",
                    question="How many trading days should we hold after the streak completes?",
                    current_value=hold_days,
                    suggested_options=[1, 3, 5, 10],
                    unit="trading days",
                )
            )

        user_fields = (["asset_a"] if tickers else []) + (["parameters.streak_length"] if streak_match else []) + (["parameters.holding_period"] if (hold_match or next_day_mention) else [])
        inferred_defaults = {}
        if not tickers:
            inferred_defaults["asset_a"] = asset_a
        if not streak_match:
            inferred_defaults["parameters.streak_length"] = f"{n_days} days"
        if not (hold_match or next_day_mention):
            inferred_defaults["parameters.holding_period"] = f"{hold_days} days"
        inferred_defaults["execution.execution_timing"] = "next_open (t+1)"
        inferred_defaults["execution.cost_bps"] = "10 bps (5 comm + 5 slip)"

        notes.append(
            f"Mapped hypothesis to Consecutive Streak template ({variant}: {n_days} consecutive days on {asset_a}, entering {direction} at t+1 for {hold_days} trading day(s))."
        )
        spec = StrategySpec(
            hypothesis_text=cleaned,
            title=f"{asset_a} {n_days}-Day {'Positive' if is_pos_streak else 'Negative'} Streak Response",
            strategy_family="single_asset",
            template_id="consecutive_streak",
            signal_variant=variant,
            asset_a=asset_a,
            benchmark_ticker=benchmark_ticker,
            parameters=StrategyParameters(
                lookback_window=n_days,
                streak_length=n_days,
                entry_threshold=float(n_days),
                holding_period=hold_days,
                exit_mode="fixed_hold",
                direction=direction,
            ),
            user_specified_fields=user_fields,
            software_inferred_defaults=inferred_defaults,
        )
        steered = _build_steered_cards(spec, lower)
        summary = _build_understood_summary(spec)
        reply = (
            f"We will test what happens to **{asset_a}** after **{n_days} consecutive {'up' if is_pos_streak else 'down'} days** — "
            f"entering **{direction.upper()}** on the next trading day and holding for **{hold_days} trading day(s)**. "
            f"Review Step 2 to run this directly or compare 2-day, 3-day, and 4-day streaks side by side."
        )
        return InterpretationResponse(
            status="needs_clarification" if ambiguities else "ready_for_confirmation",
            gemini_reply=reply,
            understood_summary=summary,
            proposed_spec=spec,
            steered_cards=steered,
            ambiguities=ambiguities,
            interpretation_notes=sanitize_research_notes(notes),
            model_used="deterministic-spec-compiler-v2",
            interpretation_mode="deterministic_rules",
        )

    # Template 4: Trend / Moving Average Crossover or Channel Breakout
    has_oscillator_terms = bool(sigma_match or re.search(r"\b(rsi|bollinger|z-score|z score|standard deviation|oversold|overbought)\b", lower))
    has_trend = (
        ma_pair_match
        or re.search(r"\b(sma|ema|golden cross|death cross|breakout|donchian|channel high|day high)\b", lower)
        or ("moving average" in lower and not has_oscillator_terms)
    )
    if template == "trend_breakout" or (template is None and has_trend):
        if "donchian" in lower or "breakout" in lower or "channel" in lower or "day high" in lower:
            t_var: Any = "donchian_breakout"
            ch_match = re.search(r"(\d{1,3})\s*(?:-|\s)?(?:day|d)\s+(?:high|low|breakout|channel|donchian)", lower)
            fast_w = int(ch_match.group(1)) if ch_match else (int(hold_match.group(1)) if hold_match else 20)
            slow_w = max(10, fast_w // 2)
            windows_explicit = bool(ch_match)
        else:
            t_var = "ema_crossover" if "ema" in lower or "exponential" in lower else "sma_crossover"
            all_ma_windows = [
                int(m.group(1))
                for m in re.finditer(
                    r"\b(\d{1,3})\s*(?:-|\s)?(?:day|d|bar)?\s*(?:simple|exponential|moving\s+average|sma|ema)",
                    lower,
                )
            ]
            if ma_pair_match:
                w1, w2 = int(ma_pair_match.group(1)), int(ma_pair_match.group(2))
                fast_w, slow_w = min(w1, w2), max(w1, w2)
                windows_explicit = True
            elif len(all_ma_windows) >= 2:
                fast_w, slow_w = min(all_ma_windows[0], all_ma_windows[1]), max(all_ma_windows[0], all_ma_windows[1])
                windows_explicit = True
            elif "golden cross" in lower or "death cross" in lower:
                fast_w, slow_w = 50, 200
                windows_explicit = True
            else:
                fast_w, slow_w = 20, 50
                windows_explicit = False
                ambiguities.append(
                    AmbiguityQuestion(
                        parameter_key="parameters.lookback_window",
                        label="Fast Moving Average Window",
                        question="What fast moving-average window (in trading days) should we test?",
                        current_value=20,
                        suggested_options=[10, 20, 50],
                        unit="trading days",
                    )
                )
                ambiguities.append(
                    AmbiguityQuestion(
                        parameter_key="parameters.secondary_window",
                        label="Slow Moving Average Window",
                        question="What slow moving-average window (in trading days) should we test?",
                        current_value=50,
                        suggested_options=[50, 100, 200],
                        unit="trading days",
                    )
                )

        user_fields = (["asset_a"] if tickers else []) + (["parameters.lookback_window", "parameters.secondary_window"] if windows_explicit else [])
        inferred_defaults = {}
        if not tickers:
            inferred_defaults["asset_a"] = asset_a
        if not windows_explicit:
            inferred_defaults["parameters.lookback_window"] = f"{fast_w}d"
            inferred_defaults["parameters.secondary_window"] = f"{slow_w}d"
        inferred_defaults["execution.execution_timing"] = "next_open (t+1)"
        inferred_defaults["execution.cost_bps"] = "10 bps (5 comm + 5 slip)"

        notes.append(
            f"Mapped hypothesis to Trend / Breakout template ({t_var} with windows {fast_w}d / {slow_w}d on {asset_a})."
        )
        spec = StrategySpec(
            hypothesis_text=cleaned,
            title=f"{asset_a} ({fast_w}d / {slow_w}d) {t_var.replace('_', ' ').title()}",
            strategy_family="single_asset",
            template_id="trend_breakout",
            signal_variant=t_var,
            asset_a=asset_a,
            benchmark_ticker=benchmark_ticker,
            parameters=StrategyParameters(
                lookback_window=fast_w,
                secondary_window=slow_w,
                entry_threshold=0.0,
                exit_threshold=0.0,
                holding_period=int(hold_match.group(1)) if hold_match else 20,
                exit_mode="signal_reversal",
                direction="short" if "death cross" in lower or "crosses below" in lower else "long",
            ),
            risk_controls=RiskControls(max_holding_days=120),
            user_specified_fields=user_fields,
            software_inferred_defaults=inferred_defaults,
        )
        steered = _build_steered_cards(spec, lower)
        summary = _build_understood_summary(spec)
        if t_var == "donchian_breakout":
            reply = (
                f"Configured a **Channel Breakout** test on **{asset_a}** triggered whenever price breaks above its **{fast_w}-day high** "
                f"(exiting on a **{slow_w}-day low** break or time-stop). Continue to Step 2 to review or compare fast vs. slow breakout windows."
            )
        else:
            reply = (
                f"Configured a **Moving-Average Trend** test on **{asset_a}** comparing the **{fast_w}-day** and **{slow_w}-day** "
                f"{'exponential ' if t_var == 'ema_crossover' else ''}moving averages. Continue to Step 2 to confirm or test standard fast/medium/slow windows."
            )
        return InterpretationResponse(
            status="needs_clarification" if ambiguities else "ready_for_confirmation",
            gemini_reply=reply,
            understood_summary=summary,
            proposed_spec=spec,
            steered_cards=steered,
            ambiguities=ambiguities,
            interpretation_notes=sanitize_research_notes(notes),
            model_used="deterministic-spec-compiler-v2",
            interpretation_mode="deterministic_rules",
        )

    # Check if user explicitly asked for volatility-scaled daily shock
    if template in (None, "return_threshold") and re.search(r"\b(vol(?:-|\s)?scaled|volatility(?:-|\s)?scaled|volatility\s+shock|vol\s+shock)\b", lower):
        entry_val = float(sigma_match.group(1)) if sigma_match else 2.0
        win_match = re.search(r"(\d{1,3})\s*(?:-|\s)?(?:day|d)\s+(?:vol|realized|window|lookback)", lower)
        win = int(win_match.group(1)) if win_match else 20
        hold_days = int(hold_match.group(1)) if hold_match else (1 if next_day_mention else 5)
        user_fields = (["asset_a"] if tickers else []) + (["parameters.entry_threshold"] if sigma_match else []) + (["parameters.lookback_window"] if win_match else []) + (["parameters.holding_period"] if (hold_match or next_day_mention) else [])
        inferred_defaults = {}
        if not sigma_match:
            inferred_defaults["parameters.entry_threshold"] = f"{entry_val:.1f}σ"
        if not win_match:
            inferred_defaults["parameters.lookback_window"] = f"{win}d"
        if not (hold_match or next_day_mention):
            inferred_defaults["parameters.holding_period"] = f"{hold_days}d"
        inferred_defaults["execution.execution_timing"] = "next_open (t+1)"
        inferred_defaults["execution.cost_bps"] = "10 bps (5 comm + 5 slip)"

        notes.append(
            f"Mapped hypothesis to Volatility-Scaled Return Shock (vol_scaled_shock, window={win}d, threshold={entry_val}σ, hold={hold_days}d on {asset_a})."
        )
        spec = StrategySpec(
            hypothesis_text=cleaned,
            title=f"{asset_a} {entry_val:.1f}σ Volatility-Scaled Shock Reversion ({hold_days}d Hold)",
            strategy_family="single_asset",
            template_id="return_threshold",
            signal_variant="vol_scaled_shock",
            asset_a=asset_a,
            benchmark_ticker=benchmark_ticker,
            parameters=StrategyParameters(
                lookback_window=win,
                entry_threshold=entry_val,
                exit_threshold=0.0,
                holding_period=hold_days,
                exit_mode="fixed_hold",
                direction="long",
            ),
            user_specified_fields=user_fields,
            software_inferred_defaults=inferred_defaults,
        )
        steered = _build_steered_cards(spec, lower)
        summary = _build_understood_summary(spec)
        reply = (
            f"Configured a **Volatility-Scaled Return Shock** on **{asset_a}**, entering when a daily drop exceeds **{entry_val:.1f}×** its {win}-day realized volatility and holding for **{hold_days} trading day(s)**."
        )
        return InterpretationResponse(
            status="needs_clarification" if ambiguities else "ready_for_confirmation",
            gemini_reply=reply,
            understood_summary=summary,
            proposed_spec=spec,
            steered_cards=steered,
            ambiguities=ambiguities,
            interpretation_notes=sanitize_research_notes(notes),
            model_used="deterministic-spec-compiler-v2",
            interpretation_mode="deterministic_rules",
        )

    # Template 3: Mean Reversion Oscillator (RSI / Bollinger / Z-Score)
    if template == "mean_reversion_oscillator" or (template is None and has_oscillator_terms):
        win_match = re.search(r"(\d{1,3})\s*(?:-|\s)?(?:day|d)\s+(?:rsi|bollinger|lower\s+bollinger|upper\s+bollinger|band|z-score|z score|moving average|mean|lookback|window)", lower)
        osc_trigger = lower.split(",")[0] if "," in lower else (lower.split("does it")[0] if "does it" in lower else lower)
        has_explicit_thresh = bool(sigma_match or pct_match)
        if "rsi" in lower or "oversold" in lower or "overbought" in lower:
            m_var: Any = "rsi_extremes"
            rsi_num_match = re.search(r"(?:rsi|below|under|less than|above|over|exceeds|at)\s+(?:is\s+|drops\s+below\s+|falls\s+below\s+)?(\d{1,2}(?:\.\d+)?)", lower)
            has_explicit_thresh = bool(pct_match or rsi_num_match)
            raw_rsi = float(pct_match.group(1)) if pct_match else (float(rsi_num_match.group(1)) if rsi_num_match else 30.0)
            entry_val = (100.0 - raw_rsi) if raw_rsi > 50.0 else raw_rsi
            exit_val = 55.0
            win = int(win_match.group(1)) if win_match else 14
            is_short_osc = bool("overbought" in lower or raw_rsi > 50.0 or re.search(r"rsi.*?\b(above|over|exceeds)\b", osc_trigger))
        elif "bollinger" in lower:
            m_var = "bollinger_band"
            entry_val = float(sigma_match.group(1)) if sigma_match else 2.0
            exit_val = 0.0
            win = int(win_match.group(1)) if win_match else 20
            has_lower = bool(re.search(r"\b(lower|below|drops?|falls?|under|oversold)\b", osc_trigger))
            has_upper = bool(re.search(r"\b(upper|overbought|breaks?\s+above|rises?\s+above|trades?\s+above)\b", osc_trigger))
            is_short_osc = bool(has_upper and not has_lower)
        else:
            m_var = "price_zscore"
            entry_val = float(sigma_match.group(1)) if sigma_match else 2.0
            exit_val = 0.0
            win = int(win_match.group(1)) if win_match else 20
            has_lower = bool(re.search(r"\b(below|drops?|falls?|under|negative|oversold)\b", osc_trigger))
            has_upper = bool(re.search(r"\b(overbought|above\s+its|above\s+the|positive\s+z|rises?\s+above|jumps?\s+above)\b", osc_trigger))
            is_short_osc = bool(has_upper and not has_lower)

        if not sigma_match and m_var != "rsi_extremes":
            ambiguities.append(
                AmbiguityQuestion(
                    parameter_key="parameters.entry_threshold",
                    label="Entry Standard-Deviation Threshold",
                    question="How many standard deviations from the rolling mean should trigger entry?",
                    current_value=entry_val,
                    suggested_options=[1.5, 2.0, 2.5, 3.0],
                    unit="σ",
                )
            )

        hold_days = int(hold_match.group(1)) if hold_match else 10
        user_fields = (["asset_a"] if tickers else []) + (["parameters.entry_threshold"] if has_explicit_thresh else []) + (["parameters.lookback_window"] if win_match else []) + (["parameters.holding_period"] if hold_match else [])
        inferred_defaults = {}
        if not has_explicit_thresh:
            inferred_defaults["parameters.entry_threshold"] = f"{entry_val}"
        if not win_match:
            inferred_defaults["parameters.lookback_window"] = f"{win}d"
        if not hold_match:
            inferred_defaults["parameters.holding_period"] = f"{hold_days}d"
        inferred_defaults["parameters.exit_threshold"] = f"{exit_val}"
        inferred_defaults["execution.execution_timing"] = "next_open (t+1)"
        inferred_defaults["execution.cost_bps"] = "10 bps (5 comm + 5 slip)"

        notes.append(
            f"Mapped hypothesis to Mean-Reversion Oscillator ({m_var}, lookback={win}d, entry={entry_val}, exit={exit_val} on {asset_a})."
        )
        spec = StrategySpec(
            hypothesis_text=cleaned,
            title=f"{asset_a} {m_var.replace('_', ' ').title()} Mean Reversion",
            strategy_family="single_asset",
            template_id="mean_reversion_oscillator",
            signal_variant=m_var,
            asset_a=asset_a,
            benchmark_ticker=benchmark_ticker,
            parameters=StrategyParameters(
                lookback_window=win,
                entry_threshold=entry_val,
                exit_threshold=exit_val,
                holding_period=hold_days,
                exit_mode="threshold_reversion",
                direction="short" if is_short_osc else "long",
            ),
            user_specified_fields=user_fields,
            software_inferred_defaults=inferred_defaults,
        )
        steered = _build_steered_cards(spec, lower)
        summary = _build_understood_summary(spec)
        if m_var == "rsi_extremes":
            reply = (
                f"Configured an **RSI Mean-Reversion** rule on **{asset_a}** using a {win}-day RSI "
                f"(entering **{'SHORT' if is_short_osc else 'LONG'}** when RSI reaches extreme threshold **{100.0 - entry_val if is_short_osc else entry_val:.0f}** and holding up to {hold_days} days or until RSI reverts to {exit_val:.0f})."
            )
        else:
            reply = (
                f"Configured a **{win}-Day {'Bollinger Band' if m_var == 'bollinger_band' else 'Z-Score'} Mean-Reversion** rule on **{asset_a}**, "
                f"entering **{'SHORT' if is_short_osc else 'LONG'}** when price moves **{entry_val:.1f}σ** from its {win}-day average and holding up to **{hold_days} trading days** as it reverts toward the mean."
            )
        return InterpretationResponse(
            status="needs_clarification" if ambiguities else "ready_for_confirmation",
            gemini_reply=reply,
            understood_summary=summary,
            proposed_spec=spec,
            steered_cards=steered,
            ambiguities=ambiguities,
            interpretation_notes=sanitize_research_notes(notes),
            model_used="deterministic-spec-compiler-v2",
            interpretation_mode="deterministic_rules",
        )

    # Default Template 1: Return Threshold Shock
    thresh_pct = float(pct_match.group(1)) if pct_match else 2.0
    if not pct_match:
        ambiguities.append(
            AmbiguityQuestion(
                parameter_key="parameters.entry_threshold",
                label="Return Shock Threshold (%)",
                question="What minimum percentage price move should trigger the signal?",
                current_value=2.0,
                suggested_options=[1.0, 1.5, 2.0, 3.0, 5.0],
                unit="%",
            )
        )

    shock_lookback = int(shock_window_match.group(1)) if shock_window_match else 1
    hold_days = int(hold_match.group(1)) if hold_match else (1 if next_day_mention else 5)
    if not hold_match and not next_day_mention:
        ambiguities.append(
            AmbiguityQuestion(
                parameter_key="parameters.holding_period",
                label="Holding Period After Signal (Days)",
                question="How many trading days should we hold the position after the price move?",
                current_value=hold_days,
                suggested_options=[1, 3, 5, 10, 20],
                unit="trading days",
            )
        )

    # Check if trigger clause describes a surge or drop
    trigger_part = lower.split("tend")[0] if "tend" in lower else lower
    if "after" in lower:
        trigger_part = lower.split("after", 1)[1]
    elif "when" in lower:
        trigger_part = lower.split("when", 1)[1].split(",")[0]

    is_surge = bool(re.search(r"\b(ris(?:e|es|ing)|jump|jumps|jumping|surge|surges|surging|gain|gains|up more than|rally|rallies)\b", trigger_part))
    r_var: Any = "cum_return_surge" if is_surge else "cum_return_drop"

    # Determine expected post-shock direction (long vs short)
    outcome_part = lower.split("after")[0] if "after" in lower else (lower.split(",")[1] if "," in lower else lower)
    if is_surge:
        go_short = bool(re.search(r"\b(pull(?:s)?\s+back|fall|falls|drop|drops|revert|reverts|mean-revert|decline|declines|short)\b", outcome_part))
        dir_shock: Any = "short" if go_short else "long"
    else:
        go_short = bool(re.search(r"\b(keep(?:s)?\s+falling|continue(?:s)?\s+to\s+fall|drop\s+further|short)\b", outcome_part))
        dir_shock = "short" if go_short else "long"

    user_fields = (
        (["asset_a"] if tickers else [])
        + (["parameters.entry_threshold"] if pct_match else [])
        + (["parameters.lookback_window"] if shock_window_match else [])
        + (["parameters.holding_period"] if (hold_match or next_day_mention) else [])
    )
    inferred_defaults = {}
    if not tickers:
        inferred_defaults["asset_a"] = asset_a
    if not pct_match:
        inferred_defaults["parameters.entry_threshold"] = f"{thresh_pct:.1f}%"
    if not shock_window_match:
        inferred_defaults["parameters.lookback_window"] = f"{shock_lookback}d"
    if not (hold_match or next_day_mention):
        inferred_defaults["parameters.holding_period"] = f"{hold_days}d"
    inferred_defaults["execution.execution_timing"] = "next_open (t+1)"
    inferred_defaults["execution.cost_bps"] = "10 bps (5 comm + 5 slip)"

    shock_horizon_label = f"{shock_lookback}-Day" if shock_lookback > 1 else "Daily"
    notes.append(
        f"Mapped hypothesis to Return Threshold Shock ({r_var}: {'+' if is_surge else '-'}{thresh_pct:.2f}% move over {shock_lookback} trading day(s) on {asset_a}, entering {dir_shock} and holding for {hold_days} trading day(s))."
    )
    spec = StrategySpec(
        hypothesis_text=cleaned,
        title=f"{asset_a} {'+' if is_surge else '-'}{thresh_pct:.1f}% {shock_horizon_label} Shock {'Pullback' if dir_shock == 'short' else 'Rebound'} ({hold_days}d Hold)",
        strategy_family="single_asset",
        template_id="return_threshold",
        signal_variant=r_var,
        asset_a=asset_a,
        benchmark_ticker=benchmark_ticker,
        parameters=StrategyParameters(
            lookback_window=shock_lookback,
            entry_threshold=thresh_pct,
            exit_threshold=0.0,
            holding_period=hold_days,
            exit_mode="fixed_hold",
            direction=dir_shock,
        ),
        user_specified_fields=user_fields,
        software_inferred_defaults=inferred_defaults,
    )
    steered = _build_steered_cards(spec, lower)
    summary = _build_understood_summary(spec)

    if len(ambiguities) >= 2:
        reply = (
            f"To test that idea on **{asset_a}**, I have configured a **Short-Term Shock Rebound** rule "
            f"(buying the next day after a **-{thresh_pct:.1f}%** drop and holding for **{hold_days} trading days**). "
            f"Since exact numbers were not specified, you can adjust the software-filled defaults in Step 2 or select **'Auto-Test Standard Thresholds'** to evaluate multiple thresholds side by side."
        )
    else:
        action_desc = "pulls back (SHORT)" if dir_shock == "short" else "recovers (LONG)"
        window_desc = f"{shock_lookback}-day move" if shock_lookback > 1 else "single-day move"
        reply = (
            f"We will test whether **{asset_a}** {action_desc} over the next **{hold_days} trading day(s)** after a {window_desc} of "
            f"**{'+' if is_surge else '-'}{thresh_pct:.1f}%**. Review Step 2 to confirm the parameters and run the audit."
        )

    return InterpretationResponse(
        status="needs_clarification" if ambiguities else "ready_for_confirmation",
        gemini_reply=reply,
        understood_summary=summary,
        proposed_spec=spec,
        steered_cards=steered,
        ambiguities=ambiguities,
        interpretation_notes=sanitize_research_notes(notes),
        model_used="deterministic-spec-compiler-v2",
        interpretation_mode="deterministic_rules",
    )


_INTERPRETATION_CACHE: dict[str, str] = {}


def _mark_user_field(spec: StrategySpec, field_key: str) -> None:
    if field_key not in spec.user_specified_fields:
        spec.user_specified_fields.append(field_key)
    short_key = field_key.replace("parameters.", "")
    if isinstance(spec.software_inferred_defaults, dict):
        spec.software_inferred_defaults.pop(field_key, None)
        spec.software_inferred_defaults.pop(short_key, None)


def _apply_risk_and_cost_overrides(spec: StrategySpec, lower: str) -> list[str]:
    """Extract explicit stop-loss, max-drawdown, and transaction cost (bps) overrides from text."""
    applied_notes: list[str] = []
    sl_match = re.search(
        r"(?:(?:per[-\s]?trade\s+)?stop[-\s]?loss(?:\s+of|\s+at|\s+to)?\s+(\d+(?:\.\d+)?)\s*(?:%|percent|pct)|"
        r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\s+(?:per[-\s]?trade\s+)?stop[-\s]?loss)",
        lower,
    )
    if sl_match:
        sl_val = max(0.5, min(90.0, float(sl_match.group(1) or sl_match.group(2))))
        spec.risk_controls.per_trade_stop_loss_pct = sl_val
        _mark_user_field(spec, "risk_controls.per_trade_stop_loss_pct")
        applied_notes.append(f"Applied explicit per-trade stop-loss override: -{sl_val:.1f}%.")

    dd_match = re.search(
        r"(?:max(?:imum)?\s+drawdown(?:\s+limit|\s+breaker)?(?:\s+of|\s+at|\s+to)?\s+(\d+(?:\.\d+)?)\s*(?:%|percent|pct)|"
        r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)\s+(?:max(?:imum)?\s+)?drawdown(?:\s+limit|\s+breaker)?)",
        lower,
    )
    if dd_match:
        dd_val = max(2.0, min(95.0, float(dd_match.group(1) or dd_match.group(2))))
        spec.risk_controls.max_drawdown_circuit_breaker_pct = dd_val
        _mark_user_field(spec, "risk_controls.max_drawdown_circuit_breaker_pct")
        applied_notes.append(f"Applied explicit portfolio max-drawdown circuit breaker override: -{dd_val:.1f}%.")

    bps_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:bps|basis\s+points?)", lower)
    if bps_match:
        bps_val = max(0.0, min(500.0, float(bps_match.group(1))))
        if "commission" in lower and "slippage" not in lower:
            spec.execution.commission_bps = bps_val
        elif "slippage" in lower and "commission" not in lower:
            spec.execution.slippage_bps = bps_val
        else:
            spec.execution.commission_bps = bps_val / 2.0
            spec.execution.slippage_bps = bps_val / 2.0
        _mark_user_field(spec, "execution.cost_bps")
        applied_notes.append(
            f"Applied explicit transaction friction override: {spec.execution.commission_bps + spec.execution.slippage_bps:.1f} bps/leg."
        )
    return applied_notes


def _apply_followup_refinement(
    base_resp: InterpretationResponse,
    followup_text: str,
    template: TemplateId | None = None,
) -> InterpretationResponse:
    """Apply a short conversational follow-up refinement on top of an existing compiled StrategySpec."""
    if base_resp.proposed_spec is None:
        return _compile_hypothesis_deterministic(followup_text)

    cleaned_f = followup_text.strip()
    lower_f = cleaned_f.lower()
    base_spec = base_resp.proposed_spec

    # Check if the follow-up switches to a different strategy template on the same asset(s)
    has_streak_cue = bool(
        re.search(r"\b(consecutive|in a row|straight\s+(?:up|down|positive|negative|red|green|winning|losing)|streak)\b", lower_f)
    )
    has_osc_cue = bool(re.search(r"\b(rsi|bollinger|z-score|z score|oversold|overbought)\b", lower_f))
    has_trend_cue = bool(re.search(r"\b(sma|ema|golden cross|death cross|breakout|donchian|moving average)\b", lower_f))
    switch_by_cue = (
        (has_streak_cue and base_spec.template_id != "consecutive_streak")
        or (has_osc_cue and base_spec.template_id not in ("mean_reversion_oscillator", "pairs_cointegration_reversion"))
        or (has_trend_cue and base_spec.template_id != "trend_breakout")
    )
    if (template is not None and template != base_spec.template_id) or (template is None and switch_by_cue):
        asset_prefix = (
            f"{base_spec.asset_a} and {base_spec.asset_b}"
            if base_spec.strategy_family == "pairs_trading" and base_spec.asset_b
            else base_spec.asset_a
        )
        switched = _compile_hypothesis_deterministic(f"{asset_prefix}: {cleaned_f}", template)
        if switched.proposed_spec is not None:
            switched.proposed_spec.hypothesis_text = f"{base_spec.hypothesis_text.rstrip('.')} ({cleaned_f})"
        return switched

    spec = base_spec.model_copy(deep=True)
    changes: list[str] = []

    risk_notes = _apply_risk_and_cost_overrides(spec, lower_f)
    changes.extend(risk_notes)

    lower_no_risk = re.sub(
        r"(?:(?:per[-\s]?trade\s+)?stop[-\s]?loss(?:\s+of|\s+at|\s+to)?\s+\d+(?:\.\d+)?\s*(?:%|percent|pct)|\d+(?:\.\d+)?\s*(?:%|percent|pct)\s+(?:per[-\s]?trade\s+)?stop[-\s]?loss|"
        r"max(?:imum)?\s+drawdown(?:\s+limit|\s+breaker)?(?:\s+of|\s+at|\s+to)?\s+\d+(?:\.\d+)?\s*(?:%|percent|pct)|\d+(?:\.\d+)?\s*(?:%|percent|pct)\s+(?:max(?:imum)?\s+)?drawdown(?:\s+limit|\s+breaker)?)",
        " ",
        lower_f,
    )

    pct_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent|pct)", lower_no_risk)
    if pct_m:
        val = float(pct_m.group(1))
        spec.parameters.entry_threshold = val
        _mark_user_field(spec, "parameters.entry_threshold")
        changes.append(f"entry threshold = {val}%")

    sigma_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:std|standard deviation|sigma|z-score|z\b|σ)", lower_no_risk)
    if sigma_m:
        val = float(sigma_m.group(1))
        if "exit" in lower_no_risk and "entry" not in lower_no_risk:
            spec.parameters.exit_threshold = val
            _mark_user_field(spec, "parameters.exit_threshold")
            changes.append(f"exit threshold = {val}σ")
        else:
            spec.parameters.entry_threshold = val
            _mark_user_field(spec, "parameters.entry_threshold")
            changes.append(f"entry threshold = {val}σ")

    word_to_int = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
    streak_m = re.search(
        r"\b(one|two|three|four|five|six|seven|eight|\d+)\s+(?:(?:consecutive|straight)\b|(?:[a-z]+\s+){0,2}(?:days?|bars?|sessions?)\s+in\s+a\s+row)",
        lower_f,
    )
    if streak_m and spec.template_id == "consecutive_streak":
        raw_n = streak_m.group(1)
        n_days = word_to_int.get(raw_n, int(raw_n) if raw_n.isdigit() else spec.parameters.streak_length)
        spec.parameters.streak_length = n_days
        spec.parameters.entry_threshold = float(n_days)
        spec.parameters.lookback_window = n_days
        _mark_user_field(spec, "parameters.streak_length")
        changes.append(f"streak length = {n_days}d")

    hold_m = re.search(
        r"(?:hold(?:ing)?(?:\s+period)?(?:\s+of|\s+for|\s+to)?\s+|(?:over\s+)?the\s+next\s+|following\s+|for\s+the\s+next\s+|for\s+)(\d{1,3})\s*(?:trading\s+)?(?:day|days|bar|bars)",
        lower_f,
    )
    if hold_m:
        h_days = max(1, min(252, int(hold_m.group(1))))
        spec.parameters.holding_period = h_days
        spec.risk_controls.max_holding_days = max(30, h_days * 2)
        _mark_user_field(spec, "parameters.holding_period")
        changes.append(f"holding period = {h_days}d")

    lb_m = re.search(
        r"(?:(\d{1,3})\s*(?:-|\s)?(?:day|d)\s+(?:lookback|window|rolling|rsi|bollinger|z-score|spread)|"
        r"(?:lookback|window)(?:\s+of|\s+to)?\s+(\d{1,3})\s*(?:trading\s+)?(?:day|days|d)?)",
        lower_f,
    )
    if lb_m:
        lb_days = max(1, min(500, int(lb_m.group(1) or lb_m.group(2))))
        spec.parameters.lookback_window = lb_days
        _mark_user_field(spec, "parameters.lookback_window")
        changes.append(f"lookback window = {lb_days}d")

    if re.search(r"\b(go\s+short|short\s+only|short\s+instead)\b", lower_f):
        spec.parameters.direction = "short"
        changes.append("direction = SHORT")
    elif re.search(r"\b(go\s+long|long\s+only|long\s+instead)\b", lower_f):
        spec.parameters.direction = "long"
        changes.append("direction = LONG")

    spec.hypothesis_text = f"{base_spec.hypothesis_text.rstrip('.')} ({cleaned_f})"
    remaining_ambs = [
        q for q in (base_resp.ambiguities or []) if q.parameter_key not in spec.user_specified_fields
    ]
    summary = _build_understood_summary(spec)
    steered = _build_steered_cards(spec, spec.hypothesis_text.lower())
    notes = list(base_resp.interpretation_notes or [])
    if changes:
        notes.append(f"Applied conversational follow-up refinement ({cleaned_f}): {', '.join(changes)}.")
    notes = sanitize_research_notes(notes)

    asset_label = f"{spec.asset_a}/{spec.asset_b}" if spec.strategy_family == "pairs_trading" and spec.asset_b else spec.asset_a
    change_summary = ", ".join(changes) if changes else cleaned_f
    reply = (
        f"Updated the **{asset_label}** specification with your refinement (**{change_summary}**): "
        f"entry threshold = **{spec.parameters.entry_threshold}**, lookback = **{spec.parameters.lookback_window}d**, "
        f"holding period = **{spec.parameters.holding_period}d**, and per-trade stop-loss = **-{spec.risk_controls.per_trade_stop_loss_pct}%**. "
        f"Review Step 2 to confirm and run the audit."
    )
    return InterpretationResponse(
        status="needs_clarification" if remaining_ambs else "ready_for_confirmation",
        gemini_reply=reply,
        understood_summary=summary,
        proposed_spec=spec,
        steered_cards=steered,
        ambiguities=remaining_ambs,
        interpretation_notes=notes,
        model_used=base_resp.model_used,
        interpretation_mode=base_resp.interpretation_mode,
    )


def _is_standalone_or_sector_prompt(text: str) -> bool:
    lower_t = text.strip().lower()
    has_sector_cue = any(
        w in lower_t
        for w in (
            "tech", "nasdaq", "semiconductor", "ai stock", "gold", "precious metal",
            "safe haven", "bond", "treasury", "rates", "yield", "small cap",
            "russell", "oil", "energy", "bank", "financial", "stocks", "equities",
        )
    )
    starts_new = bool(
        re.match(r"^(?:i\s+think|i\s+believe|hypothesi[sz]e|test\s+whether|do\s+|does\s+|when\s+|if\s+)", lower_t)
    )
    return bool(_extract_tickers_from_text(text) or has_sector_cue or starts_new)


def _resolve_clarifications(turns: list[ChatTurn]) -> list[ChatTurn]:
    """Carry stated assets or triggers into a clarification without rewriting the chat transcript."""
    resolved = []
    pending = []
    for turn in turns:
        if turn.role != "user":
            continue
        if turn.basket == "needs_detail":
            pending.append(turn.content)
            continue
        if pending and turn.basket != "no_match":
            context = " ".join(pending)
            assets = _extract_tickers_from_text(turn.content)
            if not assets:
                # A trigger-only answer should retain the asset(s) that prompted the question.
                prior_assets = _extract_tickers_from_text(context)
                if prior_assets:
                    text = " and ".join(prior_assets) + ": " + turn.content
                    turn = turn.model_copy(update={"content": text})
            elif not _extract_tickers_from_text(context) and not FINANCIAL_DOMAIN_TERMS.search(turn.content):
                # An asset-only answer completes the user's earlier trigger.
                turn = turn.model_copy(update={"content": context + " " + turn.content})
        pending.clear()
        resolved.append(turn)
    return resolved


def interpret_hypothesis(
    hypothesis_text: str,
    chat_history: list[Any] | None = None,
    template: TemplateId | None = None,
) -> InterpretationResponse:
    """Interpret a natural-language trading hypothesis with 100% deterministic repeatability.
    Supports multi-turn refinement when chat_history is provided.
    """
    prior_turns: list[ChatTurn] = []
    if chat_history:
        for item in chat_history:
            if isinstance(item, ChatTurn):
                prior_turns.append(item)
            elif isinstance(item, dict) and "role" in item and "content" in item:
                prior_turns.append(ChatTurn.model_validate(item))

    effective_text = hypothesis_text.strip()
    compiler_history = prior_turns
    if template is not None and any(turn.basket == "needs_detail" for turn in prior_turns):
        resolved = _resolve_clarifications([*prior_turns, ChatTurn(role="user", content=effective_text, basket=template)])
        compiler_history = resolved[:-1]
        effective_text = resolved[-1].content
    if template is not None and len(effective_text) < 3:
        return InterpretationResponse(
            status="needs_clarification",
            gemini_reply="What price movement would trigger the test for that asset?",
        )
    lower_eff = effective_text.lower()

    # Check if this is a short follow-up parameter refinement on an active multi-turn conversation
    is_unsupported_kw = any(re.search(pat, lower_eff) for pat, _ in UNSUPPORTED_KEYWORDS)
    has_fin_or_param_cue = bool(FINANCIAL_DOMAIN_TERMS.search(lower_eff))
    is_followup_refinement = bool(
        compiler_history
        and not is_unsupported_kw
        and not _is_standalone_or_sector_prompt(effective_text)
        and has_fin_or_param_cue
    )

    norm_key = " ".join(effective_text.strip().split()).lower()
    cacheable = template is None and not prior_turns
    if cacheable and norm_key in _INTERPRETATION_CACHE:
        return InterpretationResponse.model_validate_json(_INTERPRETATION_CACHE[norm_key])

    baseline: InterpretationResponse | None = None
    if is_followup_refinement:
        prior_inputs = [t for t in compiler_history if t.role == "user" and t.content.strip()]
        active_base: InterpretationResponse | None = None
        for turn in prior_inputs:
            if turn.basket in ("no_match", "needs_detail"):
                continue
            u_txt = turn.content
            u_low = u_txt.strip().lower()
            if any(re.search(pat, u_low) for pat, _ in UNSUPPORTED_KEYWORDS):
                continue
            if _is_standalone_or_sector_prompt(u_txt) or active_base is None:
                cand = _compile_hypothesis_deterministic(u_txt, turn.basket)
                if cand.status != "unsupported" and cand.proposed_spec is not None:
                    if cand.proposed_spec is not None:
                        _apply_risk_and_cost_overrides(cand.proposed_spec, u_low)
                    active_base = cand
            elif active_base is not None and FINANCIAL_DOMAIN_TERMS.search(u_low):
                active_base = _apply_followup_refinement(active_base, u_txt, turn.basket)
        if active_base is not None and active_base.proposed_spec is not None:
            baseline = _apply_followup_refinement(active_base, effective_text, template)

    if baseline is None:
        baseline = _compile_hypothesis_deterministic(effective_text, template)
        if baseline.proposed_spec is not None:
            extra_notes = _apply_risk_and_cost_overrides(baseline.proposed_spec, lower_eff)
            if extra_notes:
                baseline.understood_summary = _build_understood_summary(baseline.proposed_spec)
                baseline.interpretation_notes = sanitize_research_notes(
                    list(baseline.interpretation_notes or []) + extra_notes
                )

    updated_history = list(prior_turns) + [
        ChatTurn(role="user", content=hypothesis_text.strip()),
        ChatTurn(role="assistant", content=baseline.gemini_reply),
    ]
    baseline.chat_history = updated_history

    if cacheable:
        _INTERPRETATION_CACHE[norm_key] = baseline.model_dump_json()
    return baseline
