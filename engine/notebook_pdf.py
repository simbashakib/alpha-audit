"""Jupyter-Notebook-Style PDF Report Generator for Alpha Auditor (Step 6).

Generates a multi-page, high-resolution vector PDF styled like a Jupyter Notebook
(`In [1]:` methodology/code cells + `Out [1]:` empirical tables, substituted formulas,
and vector equity/drawdown/stress-test plots) so researchers can see the exact
methodology, what was tested, and where every number comes from.
"""

from __future__ import annotations

import io
import textwrap
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from engine.schemas import AuditReport


def _wrap(text: str, width: int = 96) -> str:
    lines: list[str] = []
    for para in str(text).split("\n"):
        wrapped = textwrap.wrap(para, width=width, replace_whitespace=False)
        lines.extend(wrapped if wrapped else [""])
    return "\n".join(lines)


def _draw_notebook_page_header(fig: plt.Figure, report: AuditReport, page_num: int, total_pages: int) -> None:
    """Draw a clean Jupyter Notebook top bar on the figure without text overlap."""
    ax_hdr = fig.add_axes([0.04, 0.948, 0.92, 0.036])
    ax_hdr.set_axis_off()
    rect = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor="#0f172a",
        edgecolor="#cbd5e1",
        linewidth=0.8,
    )
    ax_hdr.add_patch(rect)
    ax_hdr.text(
        0.02,
        0.5,
        f"Jupyter Notebook  |  Alpha_Auditor_{report.run_id}.ipynb",
        color="#f8fafc",
        fontsize=8.6,
        fontweight="bold",
        family="monospace",
        va="center",
        ha="left",
    )
    ax_hdr.text(
        0.98,
        0.5,
        f"Python 3 (Deterministic Kernel)  •  Page {page_num} of {total_pages}",
        color="#38bdf8",
        fontsize=8.0,
        family="monospace",
        va="center",
        ha="right",
    )


def _draw_cell_box(
    fig: plt.Figure,
    rect_coords: list[float],
    prompt_label: str,
    prompt_color: str,
    bg_color: str,
    border_color: str,
    title_text: str,
    body_text: str,
    is_code: bool = True,
    fontsize: float = 8.0,
    linespacing: float = 1.28,
) -> None:
    """Render a Jupyter-style `In [n]:` or `Out [n]:` or Markdown cell with consistent physical padding."""
    x, y, w, h = rect_coords
    ax = fig.add_axes([x, y, w, h])
    ax.set_axis_off()

    top_pad = min(0.12, 0.020 / max(h, 0.08))
    top_y = 0.98 - top_pad

    # Prompt gutter on the left (0..0.082) and cell content box (0.088..1.0)
    if prompt_label:
        ax.text(
            0.076,
            top_y,
            prompt_label,
            color=prompt_color,
            fontsize=8.2,
            fontweight="bold",
            family="monospace",
            ha="right",
            va="top",
        )
        box_x = 0.088
        box_w = 0.912
    else:
        box_x = 0.0
        box_w = 1.0

    box = FancyBboxPatch(
        (box_x, 0.02),
        box_w,
        0.96,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=0.9,
    )
    ax.add_patch(box)

    if title_text:
        ax.text(
            box_x + 0.016,
            top_y,
            title_text,
            color="#0f172a",
            fontsize=fontsize + 0.8,
            fontweight="bold",
            family="monospace" if is_code else "sans-serif",
            va="top",
        )
        top_y -= min(0.20, 0.028 / max(h, 0.08))

    ax.text(
        box_x + 0.016,
        top_y,
        body_text,
        color="#1e293b",
        fontsize=fontsize,
        family="monospace" if is_code else "sans-serif",
        va="top",
        linespacing=linespacing,
    )


def render_jupyter_notebook_pdf(report: AuditReport) -> bytes:
    """Render a 4-page Jupyter-Notebook-style PDF report showing the methodology,
    code/formulas, empirical charts, stress tests, and exact number derivations.
    """
    buf = io.BytesIO()
    spec = report.spec
    full_m = report.full_sample_metrics
    is_m = report.in_sample_metrics
    oos_m = report.out_of_sample_metrics
    sc = report.scorecard
    prov = report.data_provenance or {}
    total_pages = 4

    with PdfPages(buf) as pdf:
        # =====================================================================
        # PAGE 1: NOTEBOOK OVERVIEW, METHODOLOGY CODE CELL [1] & SCORECARD OUT [1]
        # =====================================================================
        fig1 = plt.figure(figsize=(8.5, 11.0), facecolor="#ffffff")
        _draw_notebook_page_header(fig1, report, page_num=1, total_pages=total_pages)

        # Markdown Cell: Title, Hypothesis & Plain-English Summary
        md_summary = (
            f"Hypothesis Tested: \"{spec.hypothesis_text}\"\n"
            f"Empirical Verdict: {sc.historical_verdict_label} ({sc.supported_checks_count}/{sc.evaluated_checks_count} Evaluated Checks Supported)\n\n"
            f"{_wrap(sc.executive_summary, 95)}"
        )
        _draw_cell_box(
            fig1,
            [0.04, 0.755, 0.92, 0.175],
            prompt_label="MD",
            prompt_color="#64748b",
            bg_color="#f8fafc",
            border_color="#cbd5e1",
            title_text=f"# 1. Research Hypothesis & Executive Summary — {spec.title}",
            body_text=md_summary,
            is_code=False,
            fontsize=7.8,
            linespacing=1.28,
        )

        # In [1]: Strategy Specification & Causal Signal Methodology Code Cell
        assets_expr = f"asset_a='{spec.asset_a}', asset_b='{spec.asset_b}'" if spec.asset_b else f"asset_a='{spec.asset_a}'"
        code_cell_1 = (
            f"# Exact Reproducible Strategy Specification & Causal Execution Setup\n"
            f"spec = StrategySpec(\n"
            f"    strategy_family='{spec.strategy_family}', template_id='{spec.template_id}',\n"
            f"    signal_variant='{spec.signal_variant}', {assets_expr}, benchmark='{spec.benchmark_ticker}',\n"
            f"    start_date='{spec.start_date}', end_date='{spec.end_date}', split={spec.execution.train_split_ratio},\n"
            f"    parameters=dict(lookback={spec.parameters.lookback_window}, entry={spec.parameters.entry_threshold},\n"
            f"                    exit={spec.parameters.exit_threshold}, hold={spec.parameters.holding_period},\n"
            f"                    exit_mode='{spec.parameters.exit_mode}', direction='{spec.parameters.direction}'),\n"
            f"    execution=dict(timing='{spec.execution.execution_timing}',\n"
            f"                   commission_bps={spec.execution.commission_bps}, slippage_bps={spec.execution.slippage_bps})\n"
            f")\n"
            f"# Signals S_t use strictly [0..t] data; trades fill at bar t+1 ({spec.execution.execution_timing})."
        )
        _draw_cell_box(
            fig1,
            [0.04, 0.525, 0.92, 0.215],
            prompt_label="In [1]:",
            prompt_color="#0284c7",
            bg_color="#f1f5f9",
            border_color="#94a3b8",
            title_text="",
            body_text=code_cell_1,
            is_code=True,
            fontsize=7.3,
            linespacing=1.24,
        )

        # Out [1]: 5-Dimension Empirical Scorecard & Data Provenance
        dim_lines = [
            f"Dataset Provenance: {prov.get('aligned_trading_days', full_m.trading_days)} aligned bars "
            f"({full_m.start_date} .. {full_m.end_date}) | Split: {report.split_date}",
            f"Spec SHA-256: {report.spec_sha256[:24]}... | Data SHA-256: {str(prov.get('aligned_slice_sha256', ''))[:24]}...",
            "-" * 88,
        ]
        for d in sc.dimensions:
            dim_lines.append(f"[{d.status:12s}] {d.name}")
            dim_lines.append(f"   Metric  : {_wrap(d.evidence_metric, 76).replace(chr(10), chr(10) + '             ')}")
            dim_lines.append(f"   Detail  : {_wrap(d.explanation, 76).replace(chr(10), chr(10) + '             ')}")
        _draw_cell_box(
            fig1,
            [0.04, 0.045, 0.92, 0.465],
            prompt_label="Out[1]:",
            prompt_color="#dc2626",
            bg_color="#ffffff",
            border_color="#cbd5e1",
            title_text="Empirical Hypothesis Scorecard (5 Objective Verification Dimensions)",
            body_text="\n".join(dim_lines),
            is_code=True,
            fontsize=6.9,
            linespacing=1.20,
        )

        pdf.savefig(fig1)
        plt.close(fig1)

        # =====================================================================
        # PAGE 2: WALK-FORWARD SIMULATION, EQUITY/DRAWDOWN PLOTS & SPLIT TABLE
        # =====================================================================
        fig2 = plt.figure(figsize=(8.5, 11.0), facecolor="#ffffff")
        _draw_notebook_page_header(fig2, report, page_num=2, total_pages=total_pages)

        code_cell_2 = (
            "# Execute Walk-Forward Backtest & Compare In-Sample vs. Held-Out Out-of-Sample\n"
            "sim = simulate_strategy_core(df, spec, split_idx=split_idx)\n"
            "brackets = compute_standard_bracket_profiles(df, spec, split_idx=split_idx)\n"
            "plot_equity_and_drawdown(sim, brackets, split_date=report.split_date)"
        )
        _draw_cell_box(
            fig2,
            [0.04, 0.815, 0.92, 0.115],
            prompt_label="In [2]:",
            prompt_color="#0284c7",
            bg_color="#f1f5f9",
            border_color="#94a3b8",
            title_text="",
            body_text=code_cell_2,
            is_code=True,
            fontsize=7.3,
            linespacing=1.26,
        )

        # Out [2] Top: Cumulative Equity Curve Plot
        ax_eq = fig2.add_axes([0.13, 0.545, 0.81, 0.225])
        dates_dt = pd.to_datetime(report.timeseries["dates"])
        ax_eq.plot(dates_dt, report.timeseries["equity_net"], color="#0284c7", lw=1.8, label="Confirmed Strategy (Net)")
        ax_eq.plot(
            dates_dt,
            report.timeseries["equity_benchmark"],
            color="#64748b",
            lw=1.2,
            alpha=0.85,
            label=f"Buy & Hold ({spec.benchmark_ticker})",
        )
        prof_colors = {"aggressive": "#d97706", "baseline": "#059669", "conservative": "#7c3aed"}
        for prof in report.bracket_profiles or []:
            if len(prof.equity_net) == len(dates_dt):
                ax_eq.plot(
                    dates_dt,
                    prof.equity_net,
                    color=prof_colors.get(prof.profile_id, "#0ea5e9"),
                    lw=1.0,
                    ls="--",
                    alpha=0.8,
                    label=prof.profile_label,
                )
        split_dt = pd.to_datetime(report.split_date)
        ax_eq.axvline(split_dt, color="#059669", ls="-.", lw=1.3, label=f"OOS Split ({report.split_date})")
        if full_m.circuit_breaker_triggered and full_m.circuit_breaker_date:
            cb_dt = pd.to_datetime(full_m.circuit_breaker_date)
            ax_eq.axvline(cb_dt, color="#e11d48", ls=":", lw=1.3, label=f"Circuit Breaker ({full_m.circuit_breaker_date})")
        ax_eq.set_title("Out [2a]: Cumulative Portfolio Equity ($) — In-Sample vs. Held-Out Out-of-Sample", fontsize=8.8, fontweight="bold")
        ax_eq.set_ylabel("Equity ($)", fontsize=8)
        ax_eq.tick_params(labelsize=7.5)
        ax_eq.grid(True, alpha=0.3)
        ax_eq.legend(loc="upper left", fontsize=6.4, ncol=2, framealpha=0.9)

        # Out [2] Middle: Underwater Drawdown Plot
        ax_dd = fig2.add_axes([0.13, 0.365, 0.81, 0.125])
        dd_vals = np.asarray(report.timeseries["drawdown_net_pct"], dtype=float)
        ax_dd.fill_between(dates_dt, dd_vals, 0, color="#e11d48", alpha=0.22)
        ax_dd.plot(dates_dt, dd_vals, color="#e11d48", lw=1.2, label="Strategy Underwater Drawdown (%)")
        ax_dd.axvline(split_dt, color="#059669", ls="-.", lw=1.1)
        ax_dd.set_title("Out [2b]: Underwater Drawdown Profile (U_t = -D_t <= 0%)", fontsize=8.5, fontweight="bold")
        ax_dd.set_ylabel("U_t (%)", fontsize=7.5)
        ax_dd.tick_params(labelsize=7.5)
        ax_dd.grid(True, alpha=0.3)

        def _fmt_sr_cell(m):
            return f"{'N/A (0 tr)':>12s}" if m.is_zero_exposure else f"{m.sharpe_ratio:>12.3f}"

        def _fmt_se_cell(m):
            return f"{'N/A (0 tr)':>12s}" if m.is_zero_exposure else f"{'±' + f'{m.sharpe_se_lo2002:.3f}':>12s}"

        def _fmt_psr_cell(m):
            return f"{'N/A (0 tr)':>12s}" if m.is_zero_exposure else f"{m.probabilistic_sharpe_psr_pct:>11.1f}%"

        # Out [2] Bottom: Chronological Split Table (strictly <= 88 chars per line)
        split_table_text = (
            f"{'METRIC / ESTIMATOR':<28s} | {'IN-SAMPLE':>12s} | {'HELD-OUT OOS':>12s} | {'FULL SAMPLE':>12s} | {'BENCHMARK':>11s}\n"
            + ("-" * 87) + "\n"
            f"{'Sample Window (Trading Days)':<28s} | {is_m.trading_days:>12d} | {oos_m.trading_days:>12d} | {full_m.trading_days:>12d} | {full_m.trading_days:>11d}\n"
            f"{'Trades (Time Exposure %)':<28s} | {f'{is_m.trade_count} ({is_m.exposure_pct:.1f}%)':>12s} | {f'{oos_m.trade_count} ({oos_m.exposure_pct:.1f}%)':>12s} | {f'{full_m.trade_count} ({full_m.exposure_pct:.1f}%)':>12s} | {'1 (100%)':>11s}\n"
            f"{'Cumulative Gross Return (%)':<28s} | {is_m.total_gross_return_pct:>+11.2f}% | {oos_m.total_gross_return_pct:>+11.2f}% | {full_m.total_gross_return_pct:>+11.2f}% | {'—':>11s}\n"
            f"{'Cumulative Net Return (%)':<28s} | {is_m.total_net_return_pct:>+11.2f}% | {oos_m.total_net_return_pct:>+11.2f}% | {full_m.total_net_return_pct:>+11.2f}% | {full_m.benchmark_return_pct:>+10.2f}%\n"
            f"{'Compound Annual Growth (%)':<28s} | {is_m.cagr_net_pct:>+11.2f}% | {oos_m.cagr_net_pct:>+11.2f}% | {full_m.cagr_net_pct:>+11.2f}% | {full_m.benchmark_cagr_pct:>+10.2f}%\n"
            f"{'Annualized Volatility (%)':<28s} | {is_m.annualized_volatility_pct:>11.2f}% | {oos_m.annualized_volatility_pct:>11.2f}% | {full_m.annualized_volatility_pct:>11.2f}% | {full_m.benchmark_volatility_pct:>10.2f}%\n"
            f"{'Annualized Sharpe Ratio':<28s} | {_fmt_sr_cell(is_m)} | {_fmt_sr_cell(oos_m)} | {_fmt_sr_cell(full_m)} | {full_m.benchmark_sharpe_ratio:>11.3f}\n"
            f"{'Lo (2002) Non-IID Sharpe SE':<28s} | {_fmt_se_cell(is_m)} | {_fmt_se_cell(oos_m)} | {_fmt_se_cell(full_m)} | {'—':>11s}\n"
            f"{'Probabilistic Sharpe PSR(0)':<28s} | {_fmt_psr_cell(is_m)} | {_fmt_psr_cell(oos_m)} | {_fmt_psr_cell(full_m)} | {'—':>11s}\n"
            f"{'Maximum Drawdown Loss (%)':<28s} | {is_m.max_drawdown_pct:>11.2f}% | {oos_m.max_drawdown_pct:>11.2f}% | {full_m.max_drawdown_pct:>11.2f}% | {full_m.benchmark_max_drawdown_pct:>10.2f}%\n"
            f"{'Win Rate / Mean Trade Net':<28s} | {f'{is_m.win_rate_pct:.0f}%/{is_m.mean_trade_net_pct:+.2f}%':>12s} | {f'{oos_m.win_rate_pct:.0f}%/{oos_m.mean_trade_net_pct:+.2f}%':>12s} | {f'{full_m.win_rate_pct:.0f}%/{full_m.mean_trade_net_pct:+.2f}%':>12s} | {'—':>11s}"
        )
        _draw_cell_box(
            fig2,
            [0.04, 0.045, 0.92, 0.29],
            prompt_label="Out[2]:",
            prompt_color="#dc2626",
            bg_color="#ffffff",
            border_color="#cbd5e1",
            title_text="Chronological Partition Summary (In-Sample Formation vs. Held-Out Out-of-Sample)",
            body_text=split_table_text,
            is_code=True,
            fontsize=6.9,
            linespacing=1.22,
        )

        pdf.savefig(fig2)
        plt.close(fig2)

        # =====================================================================
        # PAGE 3: STEP-BY-STEP MATHEMATICAL FORMULAS & SUBSTITUTED NUMBERS
        # =====================================================================
        fig3 = plt.figure(figsize=(8.5, 11.0), facecolor="#ffffff")
        _draw_notebook_page_header(fig3, report, page_num=3, total_pages=total_pages)

        code_cell_3 = (
            "# Exact Mathematical Estimators & Where Every Number Comes From\n"
            "sr, se_lo2002, psr, skew, kurt = lo_mertens_sharpe_se_and_psr(daily_net_returns)\n"
            "exp_max_sr_null, dsr = deflated_sharpe_ratio_bailey_lopez(daily_net_returns, k25_sharpes)\n"
            "hac_reg = newey_west_factor_regression(daily_net_returns, daily_benchmark_returns)\n"
            "boot_cis = stationary_bootstrap_cis(daily_net_returns, daily_benchmark_returns, B=1000)"
        )
        _draw_cell_box(
            fig3,
            [0.04, 0.815, 0.92, 0.115],
            prompt_label="In [3]:",
            prompt_color="#0284c7",
            bg_color="#f1f5f9",
            border_color="#94a3b8",
            title_text="",
            body_text=code_cell_3,
            is_code=True,
            fontsize=7.3,
            linespacing=1.26,
        )

        calc_lines: list[str] = []
        for step in sc.calculation_steps:
            calc_lines.append(f"■ {step.title}")
            calc_lines.append(f"  Equation : {_wrap(step.formula, 76).replace(chr(10), chr(10) + '             ')}")
            calc_lines.append(f"  Inputs   : {_wrap(step.substituted_inputs, 76).replace(chr(10), chr(10) + '             ')}")
            calc_lines.append(f"  Result   : {_wrap(step.output_value, 76).replace(chr(10), chr(10) + '             ')}")
            calc_lines.append(f"  Meaning  : {_wrap(step.interpretation, 76).replace(chr(10), chr(10) + '             ')}")
            calc_lines.append("-" * 88)

        _draw_cell_box(
            fig3,
            [0.04, 0.045, 0.92, 0.75],
            prompt_label="Out[3]:",
            prompt_color="#dc2626",
            bg_color="#ffffff",
            border_color="#cbd5e1",
            title_text="Step-by-Step Formula Audit (Exact Run Numbers Substituted into Each Equation)",
            body_text="\n".join(calc_lines),
            is_code=True,
            fontsize=7.0,
            linespacing=1.22,
        )

        pdf.savefig(fig3)
        plt.close(fig3)

        # =====================================================================
        # PAGE 4: ADVERSARIAL STRESS TESTS, LITERATURE & SAMPLE TRADE BLOTTER
        # =====================================================================
        fig4 = plt.figure(figsize=(8.5, 11.0), facecolor="#ffffff")
        _draw_notebook_page_header(fig4, report, page_num=4, total_pages=total_pages)

        code_cell_4 = (
            "# Run 5 Custom Adversarial Stress Tests & Inspect Trade Blotter (Signal t -> Fill t+1)\n"
            "stress_suite = run_adversarial_stress_suite(df, spec, split_idx)\n"
            "display(stress_suite.tests, report.trades[:10])"
        )
        _draw_cell_box(
            fig4,
            [0.04, 0.84, 0.92, 0.09],
            prompt_label="In [4]:",
            prompt_color="#0284c7",
            bg_color="#f1f5f9",
            border_color="#94a3b8",
            title_text="",
            body_text=code_cell_4,
            is_code=True,
            fontsize=7.3,
            linespacing=1.24,
        )

        # Out [4a]: Stress Test Suite Table
        st_lines = [
            f"Stress Summary: {report.stress_test_suite.resilient_count}/5 Resilient  |  "
            f"{5 - report.stress_test_suite.resilient_count - report.stress_test_suite.vulnerability_count}/5 Moderate  |  "
            f"{report.stress_test_suite.vulnerability_count}/5 High Vulnerability",
            "-" * 88,
        ]
        for idx, t in enumerate(report.stress_test_suite.tests, start=1):
            st_lines.append(
                f"Test 0{idx} [{t.outcome_badge:20s}] {t.title}\n"
                f"   Baseline: {t.baseline_value:<24s} -> Stressed: {t.stressed_value}\n"
                f"   Finding : {_wrap(t.key_finding, 76).replace(chr(10), chr(10) + '             ')}"
            )
        _draw_cell_box(
            fig4,
            [0.04, 0.40, 0.92, 0.425],
            prompt_label="Out[4]:",
            prompt_color="#dc2626",
            bg_color="#ffffff",
            border_color="#cbd5e1",
            title_text="Adversarial Stress-Testing Suite (5 Vulnerability Probes)",
            body_text="\n".join(st_lines),
            is_code=True,
            fontsize=6.6,
            linespacing=1.16,
        )

        # Out [4b]: Sample Trade Audit Blotter & Literature References
        blotter_lines = [
            f"{'#':>3s} | {'SPLIT':<9s} | {'DIR':<6s} | {'SIGNAL (t)':<10s} | {'SIG VAL':>7s} | {'ENTRY(t+1)':<10s} | {'EXIT DATE':<10s} | {'H':>3s} | {'GROSS%':>7s} | {'COST%':>6s} | {'NET%':>7s}",
            "-" * 88,
        ]
        sample_trades = report.trades[:9]
        if not sample_trades:
            blotter_lines.append("No historical trades triggered under this threshold specification.")
        else:
            for tr in sample_trades:
                sp_lbl = "OOS" if tr.split == "out_of_sample" else "IN-SAMPLE"
                dir_short = tr.direction[:6]
                blotter_lines.append(
                    f"{tr.trade_id:>3d} | {sp_lbl:<9s} | {dir_short:<6s} | {tr.signal_date:<10s} | "
                    f"{tr.signal_value:>7.2f} | {tr.entry_date:<10s} | {tr.exit_date:<10s} | "
                    f"{tr.holding_days:>2d}d | {tr.gross_return_pct:>+6.2f}% | {-tr.cost_pct:>5.2f}% | {tr.net_return_pct:>+6.2f}%"
                )
            if len(report.trades) > 9:
                blotter_lines.append(f"... plus {len(report.trades) - 9} more historical trades (all {len(report.trades)} trades included in full audit).")

        lit_refs = "; ".join(
            f"{p.authors} ({p.year})" for p in (report.literature_review.strategy_papers[:2] + report.literature_review.methodology_papers[:2])
        )
        blotter_lines.append("-" * 88)
        blotter_lines.append(_wrap(f"Key Literature Grounding: {lit_refs}", 86))

        _draw_cell_box(
            fig4,
            [0.04, 0.04, 0.92, 0.34],
            prompt_label="Out[5]:",
            prompt_color="#dc2626",
            bg_color="#ffffff",
            border_color="#cbd5e1",
            title_text="Sample Historical Trade Blotter (Causal Signal at Bar t -> Fill at Bar t+1)",
            body_text="\n".join(blotter_lines),
            is_code=True,
            fontsize=6.7,
            linespacing=1.18,
        )

        pdf.savefig(fig4)
        plt.close(fig4)

    return buf.getvalue()
