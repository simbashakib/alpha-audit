"""FastAPI backend server for Alpha Auditor Quantitative Research Platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from engine.backtester import load_audit_run, run_full_audit, save_audit_run
from engine.data_provider import CATALOG_TICKERS, validate_ticker_symbol
from engine.gemini_chat import GeminiChatError, chat_about_hypothesis
from engine.notebook_ipynb import render_jupyter_notebook_ipynb
from engine.notebook_pdf import render_jupyter_notebook_pdf
from engine.schemas import AuditReport, ChatTurn, InterpretationResponse, StrategySpec

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=False)
FRONTEND_DIR = ROOT_DIR / "frontend"
RUNS_DIR = ROOT_DIR / "data" / "runs"
EVAL_DIR = ROOT_DIR / "eval-runs"

app = FastAPI(
    title="Alpha Auditor — Quantitative Research & Hypothesis Verification Workbench",
    version="2.0.0",
)


class HypothesisRequest(BaseModel):
    hypothesis_text: str = Field(max_length=8000)
    chat_history: list[ChatTurn] = Field(default_factory=list, max_length=128)


TEMPLATE_METADATA: list[dict[str, Any]] = [
    {
        "template_id": "return_threshold",
        "family": "single_asset",
        "name": "Return Shock / Threshold Response",
        "literature_ref": "De Bondt & Thaler (1985); Lehmann (1990); Andersen et al. (2001)",
        "variants": [
            {
                "id": "cum_return_drop",
                "label": "Cumulative k-Day Return Drop (R_{t,k} <= -θ%)",
                "formula": "R_{t,k} = (C_t / C_{t-k} - 1) × 100 <= -θ",
            },
            {
                "id": "cum_return_surge",
                "label": "Cumulative k-Day Return Surge (R_{t,k} >= +θ%)",
                "formula": "R_{t,k} = (C_t / C_{t-k} - 1) × 100 >= +θ",
            },
            {
                "id": "vol_scaled_shock",
                "label": "Realized-Volatility Scaled Shock (r_t / σ_{t,L} <= -z)",
                "formula": "z_t = r_t / SD(r_{t-L+1..t}) <= -z_entry",
            },
        ],
    },
    {
        "template_id": "consecutive_streak",
        "family": "single_asset",
        "name": "Consecutive Run-Length Streak",
        "literature_ref": "Fama (1965); Jegadeesh (1990) Short-Horizon Return Autocorrelation",
        "variants": [
            {
                "id": "negative_streak",
                "label": "N Consecutive Negative Closes (C_s < C_{s-1})",
                "formula": "∑_{i=0}^{N-1} I(ΔC_{t-i} < 0) = N",
            },
            {
                "id": "positive_streak",
                "label": "N Consecutive Positive Closes (C_s > C_{s-1})",
                "formula": "∑_{i=0}^{N-1} I(ΔC_{t-i} > 0) = N",
            },
        ],
    },
    {
        "template_id": "mean_reversion_oscillator",
        "family": "single_asset",
        "name": "Statistical Mean-Reversion Oscillator",
        "literature_ref": "Poterba & Summers (1988); Wilder (1978); Bollinger (2001)",
        "variants": [
            {
                "id": "price_zscore",
                "label": "Rolling Log-Price Z-Score (|z_t| >= z_entry)",
                "formula": "z_t = (ln C_t - μ_L) / σ_L",
            },
            {
                "id": "bollinger_band",
                "label": "Bollinger Volatility Band Excursion",
                "formula": "b_t = (C_t - SMA_L) / SD_L(C)",
            },
            {
                "id": "rsi_extremes",
                "label": "Wilder Relative Strength Index (RSI_L Extremes)",
                "formula": "RSI_L = 100 - 100 / (1 + AvgGain_L / AvgLoss_L)",
            },
        ],
    },
    {
        "template_id": "trend_breakout",
        "family": "single_asset",
        "name": "Moving-Average Regime & Channel Breakout",
        "literature_ref": "Brock, Lakonishok & LeBaron (1992); Moskowitz, Ooi & Pedersen (2012)",
        "variants": [
            {
                "id": "sma_crossover",
                "label": "Simple Moving Average Crossover (SMA_fast vs SMA_slow)",
                "formula": "(SMA_fast(t) - SMA_slow(t)) / SMA_slow(t) > θ%",
            },
            {
                "id": "ema_crossover",
                "label": "Exponential Moving Average Crossover (EMA_fast vs EMA_slow)",
                "formula": "(EMA_fast(t) - EMA_slow(t)) / EMA_slow(t) > θ%",
            },
            {
                "id": "donchian_breakout",
                "label": "Donchian Channel High/Low Breakout",
                "formula": "C_t > max(C_{t-L..t-1})",
            },
        ],
    },
    {
        "template_id": "pairs_cointegration_reversion",
        "family": "pairs_trading",
        "name": "Two-Asset Cointegration & Spread Reversion",
        "literature_ref": "Engle & Granger (1987); Gatev, Goetzmann & Rouwenhorst (2006); Vidyamurthy (2004)",
        "variants": [
            {
                "id": "ols_log",
                "label": "Causal Expanding Formation OLS / Frozen OOS Hedge Ratio (β̂_t)",
                "formula": "S_t = ln(P_{A,t}) - (α̂_t + β̂_t ln(P_{B,t})) [expanding in Formation, frozen in OOS]",
            },
            {
                "id": "rolling_ols_log",
                "label": "Causal Rolling Window OLS Dynamic Hedge Ratio (β̂_t)",
                "formula": "S_t = ln(P_{A,t}) - (α̂_t + β̂_t ln(P_{B,t})) over window L_hedge",
            },
            {
                "id": "price_ratio",
                "label": "Log Price Ratio Spread (β ≡ 1)",
                "formula": "S_t = ln(P_{A,t} / P_{B,t})",
            },
        ],
    },
]

PRESET_HYPOTHESES: list[dict[str, str]] = [
    {
        "id": "spy_dip",
        "category": "Single-Asset Return Shock",
        "label": "S&P 500 >2% Daily Drop Mean Reversion",
        "text": "I think the S&P 500 tends to recover the day after falling more than 2%.",
    },
    {
        "id": "ko_pep_pairs",
        "category": "Pairs Statistical Arbitrage",
        "label": "Coca-Cola vs. Pepsi Spread Convergence",
        "text": "I think Coca-Cola and Pepsi prices tend to converge after unusually large divergences.",
    },
    {
        "id": "aapl_streak",
        "category": "Single-Asset Run-Length",
        "label": "Apple 3-Day Negative Streak Rebound",
        "text": "I think Apple tends to rise after three consecutive negative trading days.",
    },
    {
        "id": "xom_cvx_pairs",
        "category": "Pairs Statistical Arbitrage",
        "label": "ExxonMobil vs. Chevron 2σ Spread Reversion",
        "text": "ExxonMobil and Chevron tend to revert towards each other when their 20-day spread diverges by more than 2 standard deviations.",
    },
    {
        "id": "nvda_zscore",
        "category": "Single-Asset Oscillator",
        "label": "NVIDIA 2σ Bollinger Band Mean Reversion",
        "text": "I think NVIDIA reverts over 5 trading days when its price drops 2 standard deviations below its 20-day Bollinger Band.",
    },
    {
        "id": "qqq_ma",
        "category": "Single-Asset Trend Regime",
        "label": "Nasdaq-100 (QQQ) 20d/50d SMA Crossover",
        "text": "I think QQQ trends higher when the 20-day moving average crosses above the 50-day moving average.",
    },
    {
        "id": "vague_tech_panic",
        "category": "Vague Intuition (Steering Demo)",
        "label": "Informal Hunch: Tech Stocks After Panic Selloffs",
        "text": "I feel like big tech stocks usually bounce back pretty quickly when people panic.",
    },
]


@app.get("/api/catalog")
def get_catalog() -> dict[str, Any]:
    return {
        "tickers": CATALOG_TICKERS,
        "templates": TEMPLATE_METADATA,
        "presets": PRESET_HYPOTHESES,
    }


@app.get("/api/ticker/validate")
def api_validate_ticker(symbol: str = Query(..., min_length=1, max_length=20)) -> dict[str, Any]:
    return validate_ticker_symbol(symbol)


@app.post("/api/hypothesis/interpret", response_model=InterpretationResponse)
def api_interpret_hypothesis(req: HypothesisRequest) -> InterpretationResponse:
    if not req.hypothesis_text or not req.hypothesis_text.strip():
        raise HTTPException(status_code=400, detail="Provide a non-empty trading hypothesis.")
    try:
        baskets = {tpl["template_id"]: tpl["name"] for tpl in TEMPLATE_METADATA}
        return chat_about_hypothesis(req.hypothesis_text.strip(), req.chat_history, baskets)
    except GeminiChatError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@app.post("/api/audit/run", response_model=AuditReport)
def api_run_audit(spec: StrategySpec) -> AuditReport:
    if not spec.human_confirmed:
        raise HTTPException(
            status_code=400,
            detail="Human researcher confirmation is required before running the backtest.",
        )
    try:
        report = run_full_audit(spec)
        save_audit_run(report, RUNS_DIR)
        return report
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/audit/{run_id}")
def api_get_audit(run_id: str) -> JSONResponse:
    safe_id = "".join(c for c in run_id if c.isalnum() or c == "-")
    path = RUNS_DIR / f"{safe_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit run not found.")
    report = load_audit_run(path)
    return JSONResponse(content=report.model_dump(mode="json"))


@app.get("/api/audit/{run_id}/pdf")
def api_get_audit_pdf(run_id: str) -> Response:
    safe_id = "".join(c for c in run_id if c.isalnum() or c == "-")
    path = RUNS_DIR / f"{safe_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit run not found.")
    report = load_audit_run(path)
    pdf_bytes = render_jupyter_notebook_pdf(report)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="Alpha_Auditor_Notebook_{safe_id}.pdf"'
        },
    )


def _report_for_export(payload: dict[str, Any]) -> AuditReport:
    try:
        if "spec" in payload and "in_sample_metrics" in payload:
            return AuditReport.model_validate(payload)
        spec = StrategySpec.model_validate(payload)
        if not spec.human_confirmed:
            raise HTTPException(status_code=400, detail="Confirm the strategy before running an audit.")
        return run_full_audit(spec)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_input=False, include_context=False)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/audit/pdf")
def api_post_audit_pdf(payload: dict[str, Any]) -> Response:
    report = _report_for_export(payload)
    pdf_bytes = render_jupyter_notebook_pdf(report)
    safe_id = "".join(c for c in report.run_id if c.isalnum() or c == "-") or "report"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="Alpha_Auditor_Notebook_{safe_id}.pdf"'
        },
    )


@app.get("/api/audit/{run_id}/ipynb")
def api_get_audit_ipynb(run_id: str) -> Response:
    safe_id = "".join(c for c in run_id if c.isalnum() or c == "-")
    path = RUNS_DIR / f"{safe_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audit run not found.")
    report = load_audit_run(path)
    ipynb_bytes = render_jupyter_notebook_ipynb(report)
    return Response(
        content=ipynb_bytes,
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="Alpha_Auditor_Notebook_{safe_id}.ipynb"'
        },
    )


@app.post("/api/audit/ipynb")
def api_post_audit_ipynb(payload: dict[str, Any]) -> Response:
    report = _report_for_export(payload)
    ipynb_bytes = render_jupyter_notebook_ipynb(report)
    safe_id = "".join(c for c in report.run_id if c.isalnum() or c == "-") or "report"
    return Response(
        content=ipynb_bytes,
        media_type="application/x-ipynb+json",
        headers={
            "Content-Disposition": f'attachment; filename="Alpha_Auditor_Notebook_{safe_id}.ipynb"'
        },
    )


@app.get("/api/challenge/summary")
def api_challenge_summary() -> dict[str, Any]:
    """Return archived, self-serve AgentVerse challenge results if present."""
    out: dict[str, Any] = {}
    for key, folder in (
        ("real_markets", EVAL_DIR / "challenge-real-markets"),
        ("synthetic_markets", EVAL_DIR / "challenge-markets"),
    ):
        summary_file = folder / "summary.json"
        if summary_file.exists():
            try:
                data = json.loads(summary_file.read_text(encoding="utf-8"))
                out[key] = {
                    "summary": data,
                }
            except Exception:
                pass
    return out


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))
