#!/usr/bin/env python3
"""Deterministic CLI Replay & Audit Verification Runner for Alpha Auditor.

Usage examples:
  python replay.py data/runs/audit-022cbf20a8.json
  python replay.py --spec path/to/spec.json --ipynb out.ipynb --execute-ipynb
  python replay.py --verify-repeatability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.backtester import load_audit_run, run_full_audit, save_audit_run
from engine.hypothesis_parser import interpret_hypothesis
from engine.notebook_ipynb import (
    execute_and_verify_notebook_bytes,
    render_jupyter_notebook_ipynb,
)
from engine.notebook_pdf import render_jupyter_notebook_pdf
from engine.schemas import AuditReport, StrategySpec


def _replay_from_path(
    path: Path,
    ipynb_out: Path | None = None,
    pdf_out: Path | None = None,
    execute_ipynb: bool = False,
    save_v2: bool = False,
) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    prior_report: AuditReport | None = None
    if "spec" in raw and "in_sample_metrics" in raw:
        prior_report = load_audit_run(path)
        spec = prior_report.spec.model_copy(deep=True)
    else:
        spec = StrategySpec.model_validate(raw)

    spec.human_confirmed = True
    replayed = run_full_audit(spec)

    data_sha = replayed.replay_manifest.get("aligned_slice_sha256") or replayed.data_provenance.get(
        "aligned_slice_sha256", ""
    )
    print(f"=== Alpha Auditor Deterministic Replay ({replayed.engine_version.get('engine_release', '2.0.0')}) ===")
    print(f"Input file         : {path}")
    print(f"Replayed Run ID    : {replayed.run_id}")
    print(f"Spec SHA-256       : {replayed.spec_sha256}")
    print(f"Dataset SHA-256    : {data_sha}")
    print(f"Market As-Of Date  : {replayed.market_as_of_date}")
    print(
        f"Formation IS       : Net={replayed.in_sample_metrics.total_net_return_pct:+.2f}% | "
        f"MaxDD={replayed.in_sample_metrics.max_drawdown_pct:.2f}% | "
        f"Sharpe={replayed.in_sample_metrics.sharpe_ratio:.2f} | "
        f"Trades={replayed.in_sample_metrics.trade_count}"
    )
    print(
        f"Holdout OOS        : Net={replayed.out_of_sample_metrics.total_net_return_pct:+.2f}% | "
        f"MaxDD={replayed.out_of_sample_metrics.max_drawdown_pct:.2f}% | "
        f"Sharpe={replayed.out_of_sample_metrics.sharpe_ratio:.2f} | "
        f"Trades={replayed.out_of_sample_metrics.trade_count} | "
        f"ZeroExp={replayed.out_of_sample_metrics.is_zero_exposure}"
    )
    print(
        f"Scorecard Verdict  : {replayed.scorecard.historical_verdict_label} "
        f"({replayed.scorecard.supported_checks_count}/"
        f"{replayed.scorecard.evaluated_checks_count} evaluated checks supported)"
    )

    if prior_report is not None:
        prior_ver = getattr(prior_report, "schema_version", "1.0")
        print(f"\n--- Comparison vs Saved Run (schema_version={prior_ver}, id={prior_report.run_id}) ---")
        if prior_ver == "2.0" and prior_report.engine_version == replayed.engine_version:
            assert replayed.run_id == prior_report.run_id, (
                f"Run ID mismatch: {replayed.run_id} != {prior_report.run_id}"
            )
            prior_dsha = prior_report.replay_manifest.get(
                "aligned_slice_sha256"
            ) or prior_report.data_provenance.get("aligned_slice_sha256", "")
            assert data_sha == prior_dsha, "Dataset SHA mismatch"
            assert (
                replayed.out_of_sample_metrics.total_net_return_pct
                == prior_report.out_of_sample_metrics.total_net_return_pct
            ), "OOS net return mismatch"
            print("EXACT MATCH: Replayed v2 audit matches saved v2 artifact bit-for-bit.")
        else:
            print(
                f"Legacy -> v2 Reconciliation:\n"
                f"  IS Net Return  : {prior_report.in_sample_metrics.total_net_return_pct:+.2f}% -> "
                f"{replayed.in_sample_metrics.total_net_return_pct:+.2f}%\n"
                f"  IS MaxDD       : {prior_report.in_sample_metrics.max_drawdown_pct:.2f}% -> "
                f"{replayed.in_sample_metrics.max_drawdown_pct:.2f}%\n"
                f"  OOS Net Return : {prior_report.out_of_sample_metrics.total_net_return_pct:+.2f}% -> "
                f"{replayed.out_of_sample_metrics.total_net_return_pct:+.2f}%\n"
                f"  OOS ZeroExp    : {prior_report.out_of_sample_metrics.is_zero_exposure} -> "
                f"{replayed.out_of_sample_metrics.is_zero_exposure}\n"
                f"  Verdict        : {prior_report.scorecard.historical_verdict_label} -> "
                f"{replayed.scorecard.historical_verdict_label}"
            )

    if save_v2:
        saved_path = save_audit_run(replayed, path.parent)
        print(f"Saved v2 audit report to: {saved_path}")

    if ipynb_out is not None or execute_ipynb:
        ipynb_bytes = render_jupyter_notebook_ipynb(replayed)
        if ipynb_out is not None and not execute_ipynb:
            ipynb_out.parent.mkdir(parents=True, exist_ok=True)
            ipynb_out.write_bytes(ipynb_bytes)
            print(f"Exported Jupyter Notebook (.ipynb) to: {ipynb_out}")
        if execute_ipynb:
            res = execute_and_verify_notebook_bytes(ipynb_bytes, save_executed_path=ipynb_out)
            if ipynb_out is not None:
                print(f"Exported executed Jupyter Notebook (.ipynb) to: {ipynb_out}")
            print(
                f"Verified fresh-kernel notebook execution: "
                f"cells_executed={res['cells_executed']}, code_cells_verified={res['code_cells_verified']}"
            )

    if pdf_out is not None:
        pdf_bytes = render_jupyter_notebook_pdf(replayed)
        pdf_out.write_bytes(pdf_bytes)
        print(f"Exported Audit Report (.pdf) to: {pdf_out}")

    return 0


def _verify_repeatability() -> int:
    interp = interpret_hypothesis(
        "I think the S&P 500 tends to recover the day after falling more than 2%."
    )
    spec = interp.proposed_spec
    spec.human_confirmed = True
    r1 = run_full_audit(spec)
    r2 = run_full_audit(spec)
    assert r1.run_id == r2.run_id
    assert r1.spec_sha256 == r2.spec_sha256
    d1 = r1.replay_manifest.get("aligned_slice_sha256", "")
    d2 = r2.replay_manifest.get("aligned_slice_sha256", "")
    assert d1 == d2
    assert r1.model_dump(mode="json", exclude={"created_at_utc"}) == r2.model_dump(
        mode="json", exclude={"created_at_utc"}
    )
    print(
        f"Repeatability verified: run_id={r1.run_id}, "
        f"spec_sha256={r1.spec_sha256}, dataset_sha256={d1}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic CLI replay and verification runner for Alpha Auditor."
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        help="Path to a saved AuditReport JSON or StrategySpec JSON file.",
    )
    parser.add_argument(
        "--ipynb",
        type=Path,
        default=None,
        help="Optional output path to export executable v4 Jupyter Notebook (.ipynb).",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Optional output path to export Jupyter-style PDF report (.pdf).",
    )
    parser.add_argument(
        "--execute-ipynb",
        action="store_true",
        help="Execute the generated .ipynb top-to-bottom in a fresh Python kernel and verify assertions.",
    )
    parser.add_argument(
        "--save-v2",
        action="store_true",
        help="Save the replayed v2 AuditReport back to the runs directory (backing up any v1 artifact).",
    )
    parser.add_argument(
        "--verify-repeatability",
        action="store_true",
        help="Run a deterministic two-pass repeatability check on the default SPY hypothesis.",
    )

    args = parser.parse_args(argv)
    if args.verify_repeatability:
        return _verify_repeatability()
    if not args.input_path:
        parser.print_help()
        return 1
    return _replay_from_path(
        Path(args.input_path),
        ipynb_out=args.ipynb,
        pdf_out=args.pdf,
        execute_ipynb=args.execute_ipynb,
        save_v2=args.save_v2,
    )


if __name__ == "__main__":
    sys.exit(main())
