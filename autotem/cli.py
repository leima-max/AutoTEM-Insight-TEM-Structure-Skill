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
    analyze.add_argument(
        "--formats",
        type=str,
        default=None,
        help="Comma-separated output formats override, e.g. json,csv,html,svg,pdf.",
    )
    analyze.add_argument(
        "--report-title",
        type=str,
        default=None,
        help="Optional report summary/title shown in exported report outputs.",
    )
    analyze.add_argument(
        "--report-sections",
        type=str,
        default=None,
        help=(
            "Comma-separated HTML sections, e.g. "
            "overview,workflow_steps,workflow_details,interface_checks,figures."
        ),
    )
    analyze.add_argument(
        "--open-step-sections",
        action="store_true",
        help="Open workflow step detail panels by default in HTML report.",
    )
    analyze.add_argument(
        "--pdf-sections",
        type=str,
        default=None,
        help=(
            "Comma-separated PDF sections, e.g. "
            "cover,overview,workflow_steps,interface_checks,warnings,provenance,figures."
        ),
    )
    analyze.add_argument(
        "--pdf-max-figures",
        type=int,
        default=None,
        help="Maximum number of figure pages to include in PDF report.",
    )
    analyze.add_argument(
        "--pdf-cover-field",
        action="append",
        default=None,
        help="Repeatable cover key=value entry for PDF cover page.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        try:
            skill = TEMAnalysisSkill()
            format_override = None
            if args.formats:
                format_override = [
                    item.strip().lower() for item in args.formats.split(",") if item.strip()
                ]
            report_sections_override = None
            if args.report_sections:
                report_sections_override = [
                    item.strip().lower()
                    for item in args.report_sections.split(",")
                    if item.strip()
                ]
            pdf_sections_override = None
            if args.pdf_sections:
                pdf_sections_override = [
                    item.strip().lower()
                    for item in args.pdf_sections.split(",")
                    if item.strip()
                ]
            pdf_cover_fields: dict[str, str] | None = None
            if args.pdf_cover_field:
                parsed: dict[str, str] = {}
                for entry in args.pdf_cover_field:
                    if "=" not in entry:
                        raise ValueError(
                            "Invalid --pdf-cover-field value. Use key=value format."
                        )
                    key, value = entry.split("=", 1)
                    key = key.strip()
                    if not key:
                        raise ValueError(
                            "Invalid --pdf-cover-field value. Key cannot be empty."
                        )
                    parsed[key] = value.strip()
                pdf_cover_fields = parsed
            result = skill.run(
                input_path=args.input,
                workflow_path=args.workflow,
                output_dir=args.output,
                detect_atoms_step=True if args.detect_atoms else None,
                output_formats_override=format_override,
                report_title=args.report_title,
                report_sections_override=report_sections_override,
                open_step_sections=True if args.open_step_sections else None,
                pdf_sections_override=pdf_sections_override,
                pdf_max_figure_pages=args.pdf_max_figures,
                pdf_cover_fields=pdf_cover_fields,
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
