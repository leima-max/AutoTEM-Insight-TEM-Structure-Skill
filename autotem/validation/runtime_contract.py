"""Runtime contract validator for CI and local smoke checks."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from autotem import TEMAnalysisSkill
from autotem.core import RunLogEntry, RunSummary, from_json, read_json


def validate_runtime_contract(output_dir: Path) -> None:
    """Validate run_summary and run.log files against stable runtime schemas."""
    summary_path = output_dir / "run_summary.json"
    log_path = output_dir / "run.log"
    report_path = output_dir / "analysis_report.json"
    workflow_used_path = output_dir / "workflow_used.yaml"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing runtime summary: {summary_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"Missing runtime log: {log_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report: {report_path}")
    if not workflow_used_path.exists():
        raise FileNotFoundError(f"Missing effective workflow: {workflow_used_path}")

    summary = read_json(summary_path, RunSummary)
    if summary.warning_count != len(summary.warnings):
        raise ValueError("run_summary warning_count does not match warnings length.")
    if Path(summary.report_path) != report_path:
        raise ValueError("run_summary report_path does not match expected output path.")
    if Path(summary.log_path) != log_path:
        raise ValueError("run_summary log_path does not match expected output path.")
    if Path(summary.workflow_used_path) != workflow_used_path:
        raise ValueError("run_summary workflow_used_path does not match expected output path.")

    lines = log_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("run.log is empty.")
    for line in lines:
        entry = from_json(RunLogEntry, line)
        if not entry.timestamp_utc:
            raise ValueError("run.log entry missing timestamp.")


def run_smoke_check() -> None:
    """Generate a tiny workflow run for contract validation."""
    with tempfile.TemporaryDirectory(prefix="autotem-runtime-contract-") as tmp:
        root = Path(tmp)
        input_path = root / "input.npy"
        output_dir = root / "output"
        image = np.zeros((24, 24), dtype=np.float32)
        image[12, 16] = 4.0
        image[12, 8] = 4.0
        np.save(input_path, image)

        TEMAnalysisSkill().run(
            input_path=input_path,
            workflow_path=Path("configs/default.yaml"),
            output_dir=output_dir,
        )
        validate_runtime_contract(output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AutoTEM runtime output contracts.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Existing output directory containing run_summary.json and run.log.",
    )
    args = parser.parse_args(argv)

    if args.output_dir is not None:
        validate_runtime_contract(args.output_dir)
        print(f"Runtime contract validation passed for: {args.output_dir}")
        return 0

    run_smoke_check()
    print("Runtime contract smoke validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
