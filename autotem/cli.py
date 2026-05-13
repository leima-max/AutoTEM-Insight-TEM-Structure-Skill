"""Command-line entry points for AutoTEM-Insight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autotem.workflows.pipeline import TEMAnalysisSkill

EXIT_OK = 0
EXIT_CONFIG_ERROR = 2
EXIT_FILE_ERROR = 3
EXIT_RUNTIME_ERROR = 4


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
    analyze.add_argument(
        "--detect-atoms",
        action="store_true",
        help="Force-run detect_atoms reserved step in the workflow.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        try:
            skill = TEMAnalysisSkill()
            result = skill.run(
                input_path=args.input,
                workflow_path=args.workflow,
                output_dir=args.output,
                detect_atoms_step=True if args.detect_atoms else None,
            )
        except ValueError as exc:
            print(f"AutoTEM analyze configuration error: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
        except FileNotFoundError as exc:
            print(f"AutoTEM analyze file error: {exc}", file=sys.stderr)
            return EXIT_FILE_ERROR
        except Exception as exc:  # noqa: BLE001 - CLI should report all runtime failures.
            print(f"AutoTEM analyze runtime error: {exc}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR

        print(result["status"])
        if "atom_count" in result:
            print(f"Detected atoms: {result['atom_count']}")
        print(json.dumps(result, indent=2))
        return EXIT_OK

    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
