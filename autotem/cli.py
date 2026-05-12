"""Command-line entry points for AutoTEM-Insight."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotem.workflows.pipeline import TEMAnalysisSkill


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autotem", description="Run AutoTEM analysis workflows.")
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze a TEM input file.")
    analyze.add_argument("input", type=Path, help="Input image or microscopy data file.")
    analyze.add_argument("--workflow", type=Path, default=None, help="YAML workflow recipe.")
    analyze.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/run"),
        help="Output directory.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        skill = TEMAnalysisSkill()
        result = skill.run(
            input_path=args.input,
            workflow_path=args.workflow,
            output_dir=args.output,
        )
        print(result["status"])
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
