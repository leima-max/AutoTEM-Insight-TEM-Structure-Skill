"""Command-line entry points for AutoTEM-Insight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import TypeAdapter

from autotem.diffraction.phase_lookup import (
    materials_project_api_config_path,
    materials_project_api_setup_guidance,
    resolve_materials_project_api_key,
    write_materials_project_api_config,
)
from autotem.harness.contracts import HarnessError
from autotem.validation import run_benchmark_suite_from_file, run_diffsims_smoke_validation
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
        "--harness",
        action="store_true",
        help="Execute an explicit headless recipe through the registered domain services.",
    )
    analyze.add_argument(
        "--fail-on-blocking",
        action="store_true",
        help=(
            "Return a nonzero exit code for incomplete execution "
            "or failed quality/acceptance gates."
        ),
    )
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
            "overview,workflow_steps,workflow_details,interface_checks,simulation,figures."
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

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run A4 benchmark suite from YAML config.",
    )
    benchmark.add_argument(
        "--suite",
        type=Path,
        required=True,
        help="Path to benchmark suite YAML.",
    )
    benchmark.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional benchmark suite output directory override.",
    )
    benchmark.add_argument(
        "--fail-on-failed",
        action="store_true",
        help="Return non-zero exit code when any benchmark case/criteria fail.",
    )

    preview = subparsers.add_parser(
        "preview-b1b3",
        aliases=["preview-b"],
        help="Launch Streamlit UI for B1-B5 diffraction debugging.",
    )
    preview.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Streamlit server address (default: 127.0.0.1).",
    )
    preview.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501).",
    )
    preview.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default output root directory shown in UI.",
    )

    configure_mp = subparsers.add_parser(
        "configure-mp-api",
        help="Configure or inspect Materials Project API key setup.",
    )
    configure_mp.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional AutoTEM Materials Project config path override.",
    )
    configure_mp.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="Read an API key from stdin and store it in the local AutoTEM config file.",
    )
    configure_mp.add_argument(
        "--check",
        action="store_true",
        help="Check whether the API key is configured without printing the key.",
    )
    configure_mp.add_argument(
        "--show",
        action="store_true",
        help="Show first-use setup guidance and the active config path.",
    )

    validate_diffsims = subparsers.add_parser(
        "validate-diffsims",
        help="Run a standalone real-diffsims simulation and phase-matching smoke check.",
    )
    validate_diffsims.add_argument(
        "--materials",
        type=str,
        default="si,cu",
        help="Comma-separated local built-in materials to validate (default: si,cu).",
    )
    validate_diffsims.add_argument(
        "--voltage-kv",
        type=float,
        default=200.0,
        help="Accelerating voltage for diffsims simulation.",
    )
    validate_diffsims.add_argument(
        "--reciprocal-radius",
        type=float,
        default=1.5,
        help="diffsims reciprocal radius in inverse Angstrom.",
    )
    validate_diffsims.add_argument(
        "--max-simulated-spots",
        type=int,
        default=128,
        help="Maximum simulated spots to extract per material.",
    )
    validate_diffsims.add_argument(
        "--max-measured-spots",
        type=int,
        default=6,
        help="Maximum synthetic measured spots used for closed-loop matching.",
    )
    validate_diffsims.add_argument(
        "--min-matched-spots",
        type=int,
        default=3,
        help="Minimum matched spots required for the closed-loop check.",
    )
    validate_diffsims.add_argument(
        "--timeout-s",
        type=float,
        default=30.0,
        help="Maximum seconds per material before the real diffsims smoke subprocess is stopped.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        try:
            skill = TEMAnalysisSkill()
            if args.harness and any((
                args.detect_atoms, args.report_title, args.report_sections, args.open_step_sections,
                args.pdf_sections, args.pdf_max_figures is not None, args.pdf_cover_field,
            )):
                raise ValueError(
                    "Configure operations in the Harness plan. Report layout options require "
                    "a legacy workflow."
                )
            format_override = None
            if args.formats:
                format_override = [
                    item.strip().lower() for item in args.formats.split(",") if item.strip()
                ]
            report_sections_override = None
            if args.report_sections:
                report_sections_override = [
                    item.strip().lower() for item in args.report_sections.split(",") if item.strip()
                ]
            pdf_sections_override = None
            if args.pdf_sections:
                pdf_sections_override = [
                    item.strip().lower() for item in args.pdf_sections.split(",") if item.strip()
                ]
            pdf_cover_fields: dict[str, str] | None = None
            if args.pdf_cover_field:
                parsed: dict[str, str] = {}
                for entry in args.pdf_cover_field:
                    if "=" not in entry:
                        raise ValueError("Invalid --pdf-cover-field value. Use key=value format.")
                    key, value = entry.split("=", 1)
                    key = key.strip()
                    if not key:
                        raise ValueError("Invalid --pdf-cover-field value. Key cannot be empty.")
                    parsed[key] = value.strip()
                pdf_cover_fields = parsed
            result = (
                skill.run_harness(
                    input_path=args.input,
                    workflow_path=args.workflow,
                    output_dir=args.output,
                    output_formats=format_override,
                )
                if args.harness
                else skill.run(
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
            )
            serialized_result = TypeAdapter(dict[str, object]).dump_json(result, indent=2).decode()
        except HarnessError as exc:
            print(f"AutoTEM {exc.code}: {exc.message}", file=sys.stderr)
            if exc.code == "NEEDS_INPUT" and exc.details:
                print("Set workflow.parameters.load in the recipe (--workflow).", file=sys.stderr)
                print(json.dumps(exc.details, ensure_ascii=True), file=sys.stderr)
            if exc.code.startswith("LEGACY_") or exc.code.startswith("INVALID_"):
                return EXIT_CONFIG_ERROR
            return EXIT_RUNTIME_ERROR
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
        print(serialized_result)
        if args.harness or "run_id" in result:
            return (
                EXIT_OK
                if result["execution_status"] == "completed"
                and (not args.fail_on_blocking or result.get("acceptance_status") != "blocked")
                else EXIT_RUNTIME_ERROR
            )
        if args.fail_on_blocking and (
            result.get("execution_status") != "completed"
            or result.get("quality_gates_passed") is not True
            or result.get("validation_status") in {"failed", "invalid"}
            or result.get("benchmark_passed") is False
        ):
            return EXIT_RUNTIME_ERROR
        return EXIT_OK

    if args.command == "benchmark":
        try:
            summary = run_benchmark_suite_from_file(
                suite_path=args.suite,
                output_dir=args.output,
            )
        except ValueError as exc:
            print(f"AutoTEM benchmark configuration error: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR
        except FileNotFoundError as exc:
            print(f"AutoTEM benchmark file error: {exc}", file=sys.stderr)
            return EXIT_FILE_ERROR
        except Exception as exc:  # noqa: BLE001
            print(f"AutoTEM benchmark runtime error: {exc}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR

        acceptance = summary.get("acceptance", {})
        all_passed = (
            bool(acceptance.get("all_passed", False)) if isinstance(acceptance, dict) else False
        )
        status = "A4 benchmark suite completed."
        print(status)
        print(json.dumps(summary, indent=2))
        if args.fail_on_failed and not all_passed:
            return EXIT_RUNTIME_ERROR
        return EXIT_OK

    if args.command in {"preview-b1b3", "preview-b"}:
        if importlib.util.find_spec("streamlit") is None:
            print(
                "AutoTEM preview configuration error: streamlit is not installed. "
                'Install UI extras via `pip install "autotem-insight[ui]"`.',
                file=sys.stderr,
            )
            return EXIT_CONFIG_ERROR

        app_path = Path(__file__).resolve().parents[1] / "apps" / "b1_b3_preview_ui.py"
        if not app_path.exists():
            print(
                f"AutoTEM preview file error: UI script not found: {app_path}",
                file=sys.stderr,
            )
            return EXIT_FILE_ERROR

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.address",
            str(args.host),
            "--server.port",
            str(args.port),
            "--server.headless",
            "false",
        ]
        env = dict(os.environ)
        if args.output_root is not None:
            env["AUTOTEM_UI_OUTPUT_ROOT"] = str(args.output_root)

        print(
            f"Launching B1-B5 diffraction debug UI on http://{args.host}:{args.port} ...",
        )
        try:
            completed = subprocess.run(cmd, check=False, env=env)
        except Exception as exc:  # noqa: BLE001
            print(f"AutoTEM preview runtime error: {exc}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR
        return EXIT_OK if completed.returncode == 0 else EXIT_RUNTIME_ERROR

    if args.command == "configure-mp-api":
        config_path = args.config or materials_project_api_config_path()
        if args.api_key_stdin:
            api_key = sys.stdin.read().strip()
            if not api_key:
                print("AutoTEM MP API configuration error: empty API key.", file=sys.stderr)
                return EXIT_CONFIG_ERROR
            try:
                written_path = write_materials_project_api_config(
                    api_key,
                    config_path=config_path,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"AutoTEM MP API configuration error: {exc}", file=sys.stderr)
                return EXIT_CONFIG_ERROR
            print(f"Materials Project API key stored in {written_path}.")
            return EXIT_OK

        resolved, info = resolve_materials_project_api_key(config_path=config_path)
        if args.check:
            if resolved:
                print(f"Materials Project API key configured via {info.get('source')}.")
                return EXIT_OK
            print(materials_project_api_setup_guidance(config_path), file=sys.stderr)
            return EXIT_CONFIG_ERROR

        print(materials_project_api_setup_guidance(config_path))
        print(
            "To configure without putting the key in shell history, run:\n"
            "  autotem configure-mp-api --api-key-stdin\n"
            "Then paste the key and press Ctrl+Z then Enter on Windows PowerShell, "
            "or Ctrl+D on POSIX shells."
        )
        return EXIT_OK

    if args.command == "validate-diffsims":
        materials = [item.strip() for item in str(args.materials).split(",") if item.strip()]
        try:
            result = run_diffsims_smoke_validation(
                materials=materials,
                voltage_kv=float(args.voltage_kv),
                reciprocal_radius_inv_angstrom=float(args.reciprocal_radius),
                max_simulated_spots=max(1, int(args.max_simulated_spots)),
                max_measured_spots=max(1, int(args.max_measured_spots)),
                min_matched_spots=max(1, int(args.min_matched_spots)),
                timeout_s=max(1.0, float(args.timeout_s)),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"AutoTEM diffsims validation runtime error: {exc}", file=sys.stderr)
            return EXIT_RUNTIME_ERROR

        print("AutoTEM diffsims validation " + str(result["status"]) + ".")
        print(json.dumps(result, indent=2))
        return EXIT_OK if result.get("ok") else EXIT_RUNTIME_ERROR

    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
