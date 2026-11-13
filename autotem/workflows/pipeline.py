"""Workflow orchestration facade for AutoTEM-Insight."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from autotem.atomic.atom_detection import detect_atoms
from autotem.calibration import calibrate_dataset
from autotem.core import (
    AnalysisReport,
    AtomLattice,
    CalibrationResult,
    DiffractionResult,
    OutputArtifact,
    PreprocessingResult,
    QualityAssessment,
    RunLogEntry,
    RunSummary,
    RunSummaryStep,
    SimulationResult,
    TEMDataset,
    WorkflowConfig,
    WorkflowStep,
    write_json,
)
from autotem.diffraction import run_diffraction_analysis
from autotem.io import load_tem_dataset, to_temdataset
from autotem.io.readers import read_workflow_plane
from autotem.preprocessing import assess_quality, preprocess_dataset
from autotem.reporting import (
    ensure_a3_output_layout,
    evaluate_bc_interface_completeness,
    export_csv_tables_with_models,
    export_html_report,
    export_json_bundle,
    export_pdf_report,
    persist_intermediate_artifacts,
    summarize_bc_interface_rows,
    write_output_manifest,
)
from autotem.simulation import simulate_from_diffraction
from autotem.validation import run_benchmark_summary, run_validation_summary
from autotem.visualization import build_core_overlays, build_core_plots
from autotem.workflows.recipes import (
    WORKFLOW_PARAMETER_SCHEMA_VERSION,
    load_workflow_recipe,
    save_effective_workflow,
)

_UNIMPLEMENTED_A2_STEPS: set[str] = {
    "detect_atoms",
    "run_gpa",
    "atom_strain",
    "classify_sublattices",
    "segment_microstructure",
    "detect_defects",
    "quantify_z_contrast",
    "analyze_spectroscopy",
    "analyze_4dstem",
    "ai_inference",
}
_RESERVED_STEP_CONTRACTS: dict[str, dict[str, object]] = {
    "detect_atoms": {
        "owner": "C",
        "expected_inputs": ["TEMDataset", "PreprocessingResult", "CalibrationResult"],
        "expected_outputs": ["AtomLattice", "atom_coordinates.csv", "atom_overlay.png"],
    },
    "run_gpa": {
        "owner": "C",
        "expected_inputs": ["TEMDataset", "PreprocessingResult", "DiffractionResult"],
        "expected_outputs": ["StrainResult", "strain_maps", "gpa_profiles.csv"],
    },
    "atom_strain": {
        "owner": "C",
        "expected_inputs": ["AtomLattice", "TEMDataset"],
        "expected_outputs": ["StrainResult", "atomic_displacement.csv"],
    },
    "classify_sublattices": {
        "owner": "C",
        "expected_inputs": ["AtomLattice", "DiffractionResult"],
        "expected_outputs": ["AtomLattice", "sublattice_table.csv"],
    },
    "segment_microstructure": {
        "owner": "C",
        "expected_inputs": ["TEMDataset", "PreprocessingResult", "DiffractionResult"],
        "expected_outputs": ["MicrostructureResult", "grain_masks"],
    },
    "detect_defects": {
        "owner": "C",
        "expected_inputs": ["TEMDataset", "MicrostructureResult", "AtomLattice"],
        "expected_outputs": ["MicrostructureResult", "defect_table.csv"],
    },
    "quantify_z_contrast": {
        "owner": "C",
        "expected_inputs": ["TEMDataset", "AtomLattice", "CalibrationResult"],
        "expected_outputs": ["ZContrastResult", "z_contrast_table.csv"],
    },
    "analyze_spectroscopy": {
        "owner": "C",
        "expected_inputs": ["TEMDataset", "CalibrationResult"],
        "expected_outputs": ["SpectroscopyResult", "element_maps"],
    },
    "analyze_4dstem": {
        "owner": "B",
        "expected_inputs": ["TEMDataset", "CalibrationResult"],
        "expected_outputs": ["FourDSTEMResult", "bragg_disk_table.csv"],
    },
    "ai_inference": {
        "owner": "C",
        "expected_inputs": ["TEMDataset", "PreprocessingResult"],
        "expected_outputs": ["AIInferenceResult", "ai_masks"],
    },
    "validate": {
        "owner": "A",
        "expected_inputs": ["AnalysisReport"],
        "expected_outputs": ["ValidationMetric[]", "validation_summary.json"],
    },
    "benchmark": {
        "owner": "A",
        "expected_inputs": ["AnalysisReport", "benchmark_config"],
        "expected_outputs": ["BenchmarkResult", "benchmark_report.json"],
    },
}

_STEP_TO_PARAM_KEYS: dict[str, tuple[str, ...]] = {
    "assess_quality": ("quality", "assess_quality"),
    "calibrate": ("calibrate", "calibration"),
    "preprocess": ("preprocess", "preprocessing"),
    "fft": ("diffraction", "fft", "analyze"),
    "diffraction_peaks": ("diffraction", "fft", "analyze"),
    "index_diffraction": ("diffraction", "analyze"),
    "phase_match": ("diffraction", "analyze"),
    "simulate": ("simulation", "simulate"),
    "validate": ("validation", "validate"),
    "benchmark": ("benchmark",),
}
_SUPPORTED_EXPORT_FORMATS: set[str] = {"json", "csv", "html", "svg", "pdf"}
_RUNTIME_DEPENDENCY_NAMES: tuple[str, ...] = (
    "numpy",
    "scipy",
    "matplotlib",
    "pydantic",
    "pyyaml",
    "Pillow",
    "imageio",
    "tifffile",
    "ncempy",
    "h5py",
)


def _collect_warnings(*warning_lists: list[str]) -> list[str]:
    collected: list[str] = []
    for warnings in warning_lists:
        for warning in warnings:
            if warning not in collected:
                collected.append(warning)
    return collected


def _write_run_summary(summary_path: Path, summary: RunSummary) -> Path:
    write_json(summary_path, summary)
    return summary_path


def _extract_stage_parameters(
    workflow_parameters: dict[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    for key in keys:
        value = workflow_parameters.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}


def _parameter_signature(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, default=str)


def _legacy_atom_options(
    workflow: WorkflowConfig, overrides: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve both legacy YAML aliases once and retain an effective, replayable recipe."""
    groups = ("detect_atoms", "atom_detection")
    supplied = [name for name in groups if name in workflow.parameters]
    if "detect_atoms" not in {step.name for step in workflow.steps}:
        if supplied or overrides:
            raise ValueError("Atom detection parameters require a detect_atoms step.")
        return {}, {}
    from autotem.harness.contracts import HarnessError
    from autotem.services.atomic import DetectParameters

    values: dict[str, Any] = {}
    for name in supplied:
        group = workflow.parameters[name]
        if not isinstance(group, dict):
            raise ValueError(f"workflow.parameters.{name} must be a mapping.")
        for key, value in group.items():
            if key not in DetectParameters.model_fields or key in values:
                raise HarnessError("LEGACY_PARAMETER_REQUIRES_MIGRATION", f"{name}.{key}")
            values[key] = value
    options = dict(overrides or {})
    # These file-output and input-reader options belonged to the Python call before
    # YAML mapping was supported. They are not registered scientific parameters.
    api_options = {
        key: options.pop(key)
        for key in ("overlay_path", "return_overlay_array", "slice_index")
        if key in options
    }
    if options.pop("coordinate_unit", "pixel") != "pixel":
        raise ValueError(
            "detect_atoms returns pixel coordinates. Convert them before changing units."
        )
    if not values and not options:
        return {}, api_options
    effective = DetectParameters.model_validate({**values, **options}).model_dump()
    if effective["image_role"] != "raw":
        raise ValueError(
            "Legacy detect_atoms reads raw. Use a registered workflow for image_role=processed."
        )
    if supplied or overrides:
        for name in supplied:
            del workflow.parameters[name]
        workflow.parameters[supplied[0] if supplied else "detect_atoms"] = effective
    return effective, api_options


def _ensure_forced_step(workflow: WorkflowConfig, step_name: str) -> None:
    existing = [step.name for step in workflow.steps]
    if step_name in existing:
        return
    report_index = next(
        (idx for idx, step in enumerate(workflow.steps) if step.name == "report"),
        None,
    )
    step = WorkflowStep(name=step_name)
    if report_index is None:
        workflow.steps.append(step)
    else:
        workflow.steps.insert(report_index, step)


def _sha256_of_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_selected_dataset(path: Path, selection: dict, directory: Path) -> TEMDataset:
    import numpy as np

    from autotem.services.data_quality import read_selected_input

    pixels, dataset = read_selected_input(path, selection, dataset_id=path.stem)
    # Consumers reopen data.uri after the strict report is reloaded. The source container
    # alone cannot reproduce the chosen channel, leading axes or crop.
    directory.mkdir(parents=True, exist_ok=True)
    selected_file = NamedTemporaryFile(
        dir=directory, prefix="selected-", suffix=".npy", delete=False
    )
    plane = Path(selected_file.name)
    try:
        with selected_file:
            np.save(selected_file, pixels, allow_pickle=False)
    except BaseException:
        plane.unlink(missing_ok=True)
        raise
    dataset.data.uri = str(plane.resolve())
    dataset.provenance["selected_npy_sha256"] = _sha256_of_file(plane)
    return dataset


def _runtime_dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _RUNTIME_DEPENDENCY_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def _runtime_context() -> dict[str, object]:
    try:
        package_version = importlib.metadata.version("autotem-insight")
    except importlib.metadata.PackageNotFoundError:
        package_version = "0.1.0"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "autotem_insight_version": package_version,
        "dependency_versions": _runtime_dependency_versions(),
    }


def _hash_json_payload(payload: object) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _write_json_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _evaluate_quality_gates(
    workflow: WorkflowConfig,
    report: AnalysisReport,
    *,
    reporting_parameters: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, object]]:
    gate_warnings: list[str] = []
    gate_results: dict[str, dict[str, object]] = {}
    step_names = {step.name for step in workflow.steps}
    reporting_parameters = reporting_parameters or {}

    def _cfg_float(value: Any, *, default: float) -> float:
        if isinstance(value, bool):
            return float(default)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return float(default)
        return float(default)

    any_failed_step = any(step.status == "failed" for step in workflow.steps)
    gate_results["workflow_no_failed_steps"] = {
        "passed": not any_failed_step,
        "required": True,
    }
    if any_failed_step:
        gate_warnings.append(
            "BLOCKING: Workflow contains failed step(s); report is not acceptance-ready."
        )

    diffraction_required = bool(
        step_names.intersection({"fft", "diffraction_peaks", "index_diffraction", "phase_match"})
    )
    if diffraction_required:
        has_diffraction = report.diffraction is not None
        has_peaks = has_diffraction and len(report.diffraction.peaks) > 0
        has_d_spacings = has_diffraction and len(report.diffraction.d_spacings) > 0
        gate_results["diffraction_result_present"] = {
            "passed": bool(has_diffraction),
            "required": True,
        }
        gate_results["diffraction_peak_table_present"] = {
            "passed": bool(has_peaks),
            "required": True,
        }
        gate_results["d_spacing_table_present"] = {
            "passed": bool(has_d_spacings),
            "required": True,
        }
        if not has_diffraction:
            gate_warnings.append(
                "BLOCKING: Diffraction step requested but DiffractionResult is missing."
            )
        if has_diffraction and not has_peaks:
            gate_warnings.append("BLOCKING: Diffraction step requested but no peaks were detected.")
        if has_diffraction and not has_d_spacings:
            gate_warnings.append(
                "BLOCKING: Diffraction step requested but d-spacing table is empty."
            )
        if has_diffraction and has_peaks and has_d_spacings:
            diffraction_gate_cfg_raw = reporting_parameters.get("diffraction_gates")
            diffraction_gate_cfg = (
                dict(diffraction_gate_cfg_raw) if isinstance(diffraction_gate_cfg_raw, dict) else {}
            )
            min_peak_count = int(
                max(1, _cfg_float(diffraction_gate_cfg.get("min_peak_count"), default=4.0))
            )
            min_unc_coverage = _cfg_float(
                diffraction_gate_cfg.get("min_uncertainty_coverage"),
                default=0.80,
            )
            max_mean_rel_unc = _cfg_float(
                diffraction_gate_cfg.get("max_mean_relative_uncertainty"),
                default=0.35,
            )

            peak_count = len(report.diffraction.peaks)
            peak_count_pass = peak_count >= min_peak_count
            gate_results["diffraction_peak_count_minimum"] = {
                "passed": peak_count_pass,
                "required": True,
                "observed": peak_count,
                "threshold": min_peak_count,
            }
            if not peak_count_pass:
                gate_warnings.append(
                    "BLOCKING: Diffraction peak count is below gate threshold "
                    f"({peak_count} < {min_peak_count})."
                )

            center_offset = None
            center_offset_pass = False
            max_center_offset_override = diffraction_gate_cfg.get("max_center_offset_px")
            max_center_offset = _cfg_float(max_center_offset_override, default=-1.0)
            fft_payload = report.diffraction.provenance.get("fft", {})
            if isinstance(fft_payload, dict):
                center_payload = fft_payload.get("center", {})
                if isinstance(center_payload, dict):
                    raw_offset = center_payload.get("offset_px")
                    if isinstance(raw_offset, (int, float)):
                        center_offset = float(raw_offset)
                    image_shape = fft_payload.get("image_shape", [])
                    if (
                        max_center_offset < 0
                        and isinstance(image_shape, list)
                        and len(image_shape) >= 2
                    ):
                        try:
                            h = int(image_shape[0])
                            w = int(image_shape[1])
                        except (TypeError, ValueError):
                            h = 0
                            w = 0
                        if h > 0 and w > 0:
                            max_center_offset = float(max(6.0, 0.20 * min(h, w)))
            if center_offset is not None and max_center_offset >= 0:
                center_offset_pass = (
                    math.isfinite(center_offset) and center_offset <= max_center_offset
                )
            gate_results["diffraction_center_offset_in_range"] = {
                "passed": center_offset_pass,
                "required": True,
                "observed": center_offset,
                "threshold": (None if max_center_offset < 0 else max_center_offset),
            }
            if not center_offset_pass:
                gate_warnings.append(
                    "BLOCKING: Diffraction center offset is missing, nonfinite, "
                    "or outside the validated gate threshold "
                    f"(observed={center_offset}, threshold={max_center_offset})."
                )

            rel_uncertainties: list[float] = []
            for item in report.diffraction.d_spacings:
                if item.uncertainty is None:
                    continue
                if not isinstance(item.value, (int, float)):
                    continue
                if not isinstance(item.uncertainty, (int, float)):
                    continue
                value = float(item.value)
                uncertainty = float(item.uncertainty)
                if (
                    not math.isfinite(value)
                    or not math.isfinite(uncertainty)
                    or value <= 1e-12
                    or uncertainty < 0
                ):
                    continue
                rel_uncertainties.append(uncertainty / value)
            unc_coverage = len(rel_uncertainties) / max(1, len(report.diffraction.d_spacings))
            mean_rel_unc = (
                float(sum(rel_uncertainties) / len(rel_uncertainties)) if rel_uncertainties else 0.0
            )
            unc_cov_pass = unc_coverage >= min_unc_coverage
            unc_mean_pass = mean_rel_unc <= max_mean_rel_unc
            gate_results["d_spacing_uncertainty_coverage_minimum"] = {
                "passed": unc_cov_pass,
                "required": True,
                "observed": unc_coverage,
                "threshold": min_unc_coverage,
            }
            gate_results["d_spacing_uncertainty_mean_relative_maximum"] = {
                "passed": unc_mean_pass,
                "required": True,
                "observed": mean_rel_unc,
                "threshold": max_mean_rel_unc,
            }
            if not unc_cov_pass:
                gate_warnings.append(
                    "BLOCKING: d-spacing uncertainty coverage below gate threshold "
                    f"({unc_coverage:.3f} < {min_unc_coverage:.3f})."
                )
            if not unc_mean_pass:
                gate_warnings.append(
                    "BLOCKING: Mean d-spacing relative uncertainty above gate threshold "
                    f"({mean_rel_unc:.3f} > {max_mean_rel_unc:.3f})."
                )

            if step_names.intersection({"index_diffraction", "phase_match"}):
                physical_calibration = all(
                    item.unit.strip().lower() in {"nm", "angstrom", "å", "pm", "um", "µm", "μm"}
                    and isinstance(item.value, (int, float))
                    and not isinstance(item.value, bool)
                    and math.isfinite(float(item.value))
                    and float(item.value) > 0
                    for item in report.diffraction.d_spacings
                )
                gate_results["phase_matching_physical_calibration"] = {
                    "passed": physical_calibration,
                    "required": True,
                }
                if not physical_calibration:
                    gate_warnings.append(
                        "BLOCKING: Phase indexing/matching requires finite positive "
                        "d-spacings with physical length units; "
                        "pixel/unknown values are diagnostic."
                    )
                consistency_payload = report.diffraction.provenance.get("b3_b4_consistency", {})
                indexed_fraction = None
                if isinstance(consistency_payload, dict):
                    raw_fraction = consistency_payload.get("indexed_nonzero_fraction")
                    if isinstance(raw_fraction, (int, float)):
                        indexed_fraction = float(raw_fraction)
                min_indexed_fraction = _cfg_float(
                    diffraction_gate_cfg.get("min_indexed_nonzero_fraction"),
                    default=0.20,
                )
                indexed_fraction_pass = (
                    indexed_fraction is not None
                    and math.isfinite(indexed_fraction)
                    and indexed_fraction >= min_indexed_fraction
                )
                gate_results["b3_b4_indexed_fraction_minimum"] = {
                    "passed": indexed_fraction_pass,
                    "required": True,
                    "observed": indexed_fraction,
                    "threshold": min_indexed_fraction,
                }
                if not indexed_fraction_pass:
                    gate_warnings.append(
                        "BLOCKING: B3/B4 indexed non-zero hkl fraction is missing, nonfinite, "
                        "or below gate threshold "
                        f"(observed={indexed_fraction}, threshold={min_indexed_fraction})."
                    )

    atom_required = "detect_atoms" in step_names
    if atom_required:
        atom_present = report.atom_lattice is not None
        atom_coordinates_present = atom_present and len(report.atom_lattice.atoms) > 0
        gate_results["atom_lattice_present"] = {
            "passed": bool(atom_present),
            "required": True,
        }
        gate_results["atom_coordinates_present"] = {
            "passed": bool(atom_coordinates_present),
            "required": True,
        }
        if not atom_present:
            gate_warnings.append(
                "BLOCKING: detect_atoms step requested but AtomLattice is missing."
            )
        elif not atom_coordinates_present:
            gate_warnings.append(
                "BLOCKING: detect_atoms step requested but atom coordinates are empty."
            )

    interface_gate_cfg_raw = reporting_parameters.get("interface_gates")
    interface_gate_cfg = (
        dict(interface_gate_cfg_raw) if isinstance(interface_gate_cfg_raw, dict) else {}
    )
    interface_gates_enabled = bool(interface_gate_cfg.get("enabled", True))
    required_items_override_raw = interface_gate_cfg.get("required_items")
    required_items_override = (
        [str(item) for item in required_items_override_raw]
        if isinstance(required_items_override_raw, list)
        else []
    )

    default_required_items: list[str] = []
    if diffraction_required:
        default_required_items.extend(
            ["B.DiffractionResult", "B.Diffraction peaks", "B.d-spacing table"]
        )
    strain_required = bool(step_names.intersection({"run_gpa", "atom_strain"}))
    if atom_required or strain_required:
        default_required_items.extend(["C.AtomLattice", "C.Atom coordinates"])
    if strain_required:
        default_required_items.append("C.StrainResult")

    required_items = required_items_override if required_items_override else default_required_items
    if interface_gates_enabled and required_items:
        checks = evaluate_bc_interface_completeness(report)
        check_map = {f"{item['owner']}.{item['item']}": item["status"] for item in checks}
        missing_items = [
            item for item in required_items if check_map.get(item, "missing") != "pass"
        ]
        interface_passed = len(missing_items) == 0
        gate_results["bc_interface_required_items_present"] = {
            "passed": interface_passed,
            "required": True,
            "required_items": required_items,
            "missing_items": missing_items,
        }
        if not interface_passed:
            gate_warnings.append(
                "BLOCKING: B/C interface completeness gate failed; missing required items: "
                + ", ".join(missing_items)
            )

    passed = all(
        bool(result.get("passed", False))
        for result in gate_results.values()
        if bool(result.get("required", False))
    )
    summary = {
        "passed": passed,
        "blocking_warning_count": len(gate_warnings),
        "gates": gate_results,
    }
    return gate_warnings, summary


def _input_fingerprint(input_path: Path) -> dict[str, object]:
    if not input_path.exists() or not input_path.is_file():
        return {"path": str(input_path), "exists": False, "sha256": None, "size_bytes": None}
    return {
        "path": str(input_path),
        "exists": True,
        "sha256": _sha256_of_file(input_path),
        "size_bytes": input_path.stat().st_size,
    }


def _append_artifact_unique(
    artifacts: list[OutputArtifact],
    artifact: OutputArtifact,
) -> None:
    if any(item.uri == artifact.uri for item in artifacts):
        return
    artifacts.append(artifact)


class _RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: list[RunLogEntry] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def log(self, level: str, message: str, context: dict[str, Any] | None = None) -> None:
        entry = RunLogEntry(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            level=level,  # type: ignore[arg-type]
            message=message,
            context=context or {},
        )
        self._entries.append(entry)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{entry.model_dump_json()}\n")
            handle.flush()

    def info(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.log("INFO", message, context=context)

    def warning(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.log("WARNING", message, context=context)

    def error(self, message: str, context: dict[str, Any] | None = None) -> None:
        self.log("ERROR", message, context=context)

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(f"{entry.model_dump_json()}\n" for entry in self._entries),
            encoding="utf-8",
        )


@dataclass(slots=True)
class TEMAnalysisSkill:
    """Facade that coordinates load, calibration, analysis, and reporting steps."""

    def load_data(self, path: str | Path, metadata: dict[str, object] | None = None) -> TEMDataset:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        dataset = load_tem_dataset(path)
        if metadata:
            merged_metadata = dict(dataset.metadata)
            merged_metadata.update(metadata)
            dataset.metadata = merged_metadata
        return dataset

    def assess_quality(
        self,
        dataset: TEMDataset,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> QualityAssessment:
        return assess_quality(dataset, parameters=parameters)

    def calibrate(
        self,
        dataset: TEMDataset,
        method: str = "auto",
        reference: dict[str, object] | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> CalibrationResult:
        calibration = calibrate_dataset(
            dataset,
            method=method,
            reference=reference,
            parameters=parameters,
        )
        return calibration

    def preprocess(
        self,
        dataset: TEMDataset,
        recipe: str = "auto",
        *,
        quality_after: QualityAssessment | None = None,
        parameters: dict[str, Any] | None = None,
        output_dir: str | Path | None = None,
    ) -> PreprocessingResult:
        return preprocess_dataset(
            dataset,
            quality_after=quality_after,
            recipe=recipe,
            parameters=parameters,
            output_dir=output_dir,
        )

    def analyze_fft(
        self,
        preprocessing: PreprocessingResult,
        calibration: CalibrationResult,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> DiffractionResult:
        return run_diffraction_analysis(preprocessing, calibration, parameters=parameters)

    def index_diffraction(self, diffraction: DiffractionResult) -> DiffractionResult:
        # Indexing is currently bundled in `run_diffraction_analysis` for A2 baseline.
        return diffraction

    def simulate_from_diffraction(
        self,
        diffraction: DiffractionResult,
        *,
        parameters: dict[str, Any] | None = None,
    ) -> SimulationResult:
        return simulate_from_diffraction(diffraction, parameters=parameters)

    def run_harness(
        self,
        input_path: str | Path,
        workflow_path: str | Path | None = None,
        output_dir: str | Path = "outputs/harness",
        *,
        plan: dict | None = None,
        output_formats: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run a registered headless recipe; retain the legacy facade and strict core models."""
        from autotem.harness import Harness
        from autotem.harness.compiler import compile_legacy_recipe, compile_plan

        if plan is not None and workflow_path is not None:
            raise ValueError("Select one plan or workflow file.")
        formats = ["json", "html"] if output_formats is None else list(output_formats)
        if plan is None:
            if workflow_path is None:
                raise ValueError("Select an explicit traditional recipe or plan")
            import yaml

            source = yaml.safe_load(Path(workflow_path).read_text(encoding="utf-8-sig"))
            if not isinstance(source, dict):
                raise ValueError("Workflow file must contain one plan or legacy workflow object.")
            if "workflow" in source:
                plan = compile_legacy_recipe(input_path, workflow_path, output_formats)
                formats = plan["user_input"]["legacy_recipe"]["report_formats"]
            else:
                source["input_path"] = str(input_path)
                plan = compile_plan(source).model_dump(mode="json")
        else:
            plan = {**plan, "input_path": str(input_path)}
        if not formats or not set(formats) <= {"json", "html", "pdf"}:
            raise ValueError("Choose at least one registered report format from json, html or pdf.")
        harness = Harness(output_dir)
        state = harness.submit_analysis(plan, asynchronous=False)
        report = harness.export_report(state["run_id"], formats=formats)
        return {**state, "execution_status": state["status"], "report": report}

    def run(
        self,
        input_path: str | Path,
        workflow_path: str | Path | None = None,
        output_dir: str | Path = "outputs/run",
        metadata: dict[str, object] | None = None,
        detect_atoms_step: bool | None = None,
        atom_detection_options: dict[str, Any] | None = None,
        output_formats_override: list[str] | None = None,
        report_title: str | None = None,
        report_sections_override: list[str] | None = None,
        open_step_sections: bool | None = None,
        pdf_sections_override: list[str] | None = None,
        pdf_max_figure_pages: int | None = None,
        pdf_cover_fields: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run the configured A2 workflow and emit structured outputs."""
        if workflow_path is not None and Path(workflow_path).is_file():
            import yaml

            try:
                explicit = yaml.safe_load(Path(workflow_path).read_text(encoding="utf-8-sig"))
            except yaml.YAMLError:
                explicit = None  # The legacy entry records its YAML failure in run.log.
            if isinstance(explicit, dict) and "workflow" not in explicit and (
                "steps" in explicit or "envelope_version" in explicit
            ):
                if any(value is not None for value in (
                    metadata, detect_atoms_step, atom_detection_options, report_title,
                    report_sections_override, open_step_sections, pdf_sections_override,
                    pdf_max_figure_pages, pdf_cover_fields,
                )):
                    raise ValueError(
                        "For registered recipes put analysis options in plan.steps. "
                        "Legacy report-layout overrides require a legacy workflow."
                    )
                return self.run_harness(
                    input_path, output_dir=output_dir, plan=explicit,
                    output_formats=output_formats_override,
                )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ensure_a3_output_layout(output_dir)
        log_path = output_dir / "run.log"
        logger = _RunLogger(log_path)
        logger.info(
            "Starting run: "
            f"input='{Path(input_path)}', workflow='{workflow_path}', output='{output_dir}'."
        )

        try:
            recipe = load_workflow_recipe(workflow_path)
            if detect_atoms_step is True:
                _ensure_forced_step(recipe.workflow, "detect_atoms")
            detector_parameters, detector_api_options = _legacy_atom_options(
                recipe.workflow, atom_detection_options
            )
            if any(step.name == "simulate" for step in recipe.workflow.steps):
                simulation_parameters = _extract_stage_parameters(
                    recipe.workflow.parameters,
                    _STEP_TO_PARAM_KEYS.get("simulate", ("simulation",)),
                )
                if not any(
                    key in simulation_parameters for key in ("diffsims_timeout_s", "timeout_s")
                ):
                    # Optional native imports may stall before the solver starts.
                    # The default workflow must use a bounded worker, even without a Harness.
                    simulation_parameters["diffsims_timeout_s"] = 30.0
                    recipe.workflow.parameters["simulation"] = simulation_parameters
            workflow_parameters = dict(recipe.workflow.parameters)
            reporting_parameters = _extract_stage_parameters(
                workflow_parameters,
                ("reporting", "report"),
            )
            workflow_used_path = save_effective_workflow(recipe, output_dir)
            logger.info(
                f"Loaded workflow '{recipe.workflow.name}' with "
                f"{len(recipe.workflow.steps)} steps.",
            )

            dataset: TEMDataset | None = None
            quality: QualityAssessment | None = None
            calibration: CalibrationResult | None = None
            preprocessing: PreprocessingResult | None = None
            diffraction: DiffractionResult | None = None
            simulation: SimulationResult | None = None
            atom_lattice: AtomLattice | None = None

            step_records: list[WorkflowStep] = []
            run_warnings: list[str] = []
            diffraction_signature: str | None = None
            diffraction_source_step: str | None = None
            validate_step_indices: list[int] = []
            benchmark_step_indices: list[int] = []

            for step in recipe.workflow.steps:
                step_record = WorkflowStep(
                    name=step.name,
                    status="pending",
                    parameters=dict(step.parameters),
                    inputs=list(step.inputs),
                    outputs=list(step.outputs),
                    warnings=list(step.warnings),
                )
                logger.info(f"Executing step '{step_record.name}'.")

                if step_record.name == "load":
                    load_options = workflow_parameters.get("load", {})
                    if not isinstance(load_options, dict):
                        raise ValueError("workflow.parameters.load must be a mapping.")
                    source = Path(input_path)
                    if not source.is_file():
                        raise FileNotFoundError(f"Input file does not exist: {source}")
                    if load_options or source.suffix.lower() == ".npz":
                        directory = output_dir / (
                            "intermediates" if recipe.workflow.save_intermediates else ".runtime"
                        )
                        dataset = _load_selected_dataset(source, load_options, directory)
                        if metadata:
                            dataset.metadata = {
                                **dataset.metadata,
                                **metadata,
                                "selection": dataset.metadata["selection"],
                            }
                        step_record.parameters = dict(load_options)
                        step_record.outputs = [dataset.data.uri]
                    elif (plane := read_workflow_plane(source)) is not None:
                        dataset = to_temdataset(plane)
                        if metadata:
                            dataset.metadata.update(metadata)
                    else:
                        dataset = self.load_data(source, metadata=metadata)
                    step_record.status = "warning" if dataset.warnings else "done"
                    step_record.warnings.extend(dataset.warnings)
                elif step_record.name == "assess_quality":
                    if dataset is None:
                        step_record.status = "failed"
                        step_record.warnings.append(
                            "Dataset is required before quality assessment."
                        )
                    else:
                        stage_parameters = _extract_stage_parameters(
                            workflow_parameters,
                            _STEP_TO_PARAM_KEYS.get("assess_quality", ("quality",)),
                        )
                        step_record.parameters = stage_parameters
                        quality = self.assess_quality(
                            dataset,
                            parameters=stage_parameters,
                        )
                        step_record.status = "warning" if quality.warnings else "done"
                        step_record.warnings.extend(quality.warnings)
                elif step_record.name == "calibrate":
                    if dataset is None:
                        step_record.status = "failed"
                        step_record.warnings.append("Dataset is required before calibration.")
                    else:
                        stage_parameters = _extract_stage_parameters(
                            workflow_parameters,
                            _STEP_TO_PARAM_KEYS.get("calibrate", ("calibrate",)),
                        )
                        method = str(stage_parameters.get("method", "auto"))
                        reference = stage_parameters.get("reference")
                        reference_mapping = reference if isinstance(reference, dict) else None
                        step_record.parameters = stage_parameters
                        step_record.status = "done"
                        calibration = self.calibrate(
                            dataset,
                            method=method,
                            reference=reference_mapping,
                            parameters=stage_parameters,
                        )
                        if calibration.warnings:
                            step_record.status = "warning"
                        step_record.warnings.extend(calibration.warnings)
                elif step_record.name == "preprocess":
                    if dataset is None:
                        step_record.status = "failed"
                        step_record.warnings.append("Dataset is required before preprocessing.")
                    else:
                        stage_parameters = _extract_stage_parameters(
                            workflow_parameters,
                            _STEP_TO_PARAM_KEYS.get("preprocess", ("preprocess",)),
                        )
                        prep_recipe = str(stage_parameters.get("recipe", "auto"))
                        step_record.parameters = stage_parameters
                        preprocessing = self.preprocess(
                            dataset,
                            recipe=prep_recipe,
                            quality_after=quality,
                            parameters=stage_parameters,
                            output_dir=output_dir
                            / (
                                "intermediates"
                                if recipe.workflow.save_intermediates
                                else ".runtime"
                            ),
                        )
                        step_record.status = "warning" if preprocessing.warnings else "done"
                        step_record.warnings.extend(preprocessing.warnings)
                elif step_record.name in {
                    "fft",
                    "diffraction_peaks",
                    "index_diffraction",
                    "phase_match",
                }:
                    if preprocessing is None or calibration is None:
                        step_record.status = "failed"
                        step_record.warnings.append(
                            "Preprocessing and calibration are required before diffraction steps."
                        )
                    else:
                        stage_parameters = _extract_stage_parameters(
                            workflow_parameters,
                            _STEP_TO_PARAM_KEYS.get(step_record.name, ("diffraction",)),
                        )
                        step_record.parameters = stage_parameters
                        signature = _parameter_signature(stage_parameters)
                        should_recompute = diffraction is None or signature != diffraction_signature
                        if should_recompute:
                            diffraction = self.analyze_fft(
                                preprocessing,
                                calibration,
                                parameters=stage_parameters,
                            )
                            diffraction_signature = signature
                            diffraction_source_step = step_record.name
                        else:
                            step_record.parameters = {
                                **stage_parameters,
                                "_reused_diffraction_result_from": diffraction_source_step,
                            }
                        step_record.status = "warning" if diffraction.warnings else "done"
                        step_record.warnings.extend(diffraction.warnings)
                elif step_record.name == "simulate":
                    if diffraction is None:
                        step_record.status = "failed"
                        step_record.warnings.append(
                            "Diffraction result is required before simulation."
                        )
                    else:
                        stage_parameters = _extract_stage_parameters(
                            workflow_parameters,
                            _STEP_TO_PARAM_KEYS.get("simulate", ("simulation",)),
                        )
                        step_record.parameters = stage_parameters
                        simulation = self.simulate_from_diffraction(
                            diffraction,
                            parameters=stage_parameters,
                        )
                        step_record.status = "warning" if simulation.warnings else "done"
                        step_record.warnings.extend(simulation.warnings)
                elif step_record.name == "validate":
                    stage_parameters = _extract_stage_parameters(
                        workflow_parameters,
                        _STEP_TO_PARAM_KEYS.get("validate", ("validation",)),
                    )
                    step_record.parameters = stage_parameters
                    step_record.outputs = ["ValidationMetric[]", "validation_summary.json"]
                    step_record.status = "done"
                    validate_step_indices.append(len(step_records))
                elif step_record.name == "benchmark":
                    stage_parameters = _extract_stage_parameters(
                        workflow_parameters,
                        _STEP_TO_PARAM_KEYS.get("benchmark", ("benchmark",)),
                    )
                    step_record.parameters = stage_parameters
                    step_record.outputs = ["BenchmarkResult", "benchmark_report.json"]
                    step_record.status = "done"
                    benchmark_step_indices.append(len(step_records))
                elif step_record.name == "report":
                    step_record.status = "done"
                elif step_record.name in _UNIMPLEMENTED_A2_STEPS:
                    if step_record.name == "detect_atoms":
                        contract = _RESERVED_STEP_CONTRACTS.get(step_record.name, {})
                        expected_inputs = contract.get("expected_inputs")
                        expected_outputs = contract.get("expected_outputs")
                        owner = contract.get("owner")
                        if isinstance(expected_inputs, list):
                            step_record.inputs = [str(item) for item in expected_inputs]
                        if isinstance(expected_outputs, list):
                            step_record.outputs = [str(item) for item in expected_outputs]
                        step_record.parameters = {**detector_parameters, **detector_api_options}
                        if "overlay_path" in step_record.parameters:
                            step_record.parameters["overlay_path"] = str(
                                step_record.parameters["overlay_path"]
                            )
                        try:
                            plane_hash = (
                                dataset.provenance.get("selected_npy_sha256") if dataset else None
                            )
                            detector_input = input_path
                            options = detector_api_options
                            if plane_hash:
                                import numpy as np

                                detector_input = np.load(dataset.data.uri, allow_pickle=False)
                                requested_slice = options.get("slice_index", 0)
                                if requested_slice != 0:
                                    step_record.warnings.append(
                                        f"load already selected the 2D input. Detector slice_index="
                                        f"{requested_slice} was ignored for this plane."
                                    )
                                    options = {**options, "slice_index": 0}
                                    step_record.parameters["slice_index"] = 0
                            detection_result = detect_atoms(
                                detector_input,
                                **{
                                    key: value for key, value in detector_parameters.items()
                                    if key != "image_role"
                                },
                                **options,
                            )
                            atom_lattice = detection_result.to_atom_lattice("pixel")
                            atom_lattice.provenance["image_role"] = detector_parameters.get(
                                "image_role", "raw"
                            )
                            if plane_hash:
                                atom_lattice.provenance["selected_npy_sha256"] = plane_hash
                                atom_lattice.provenance["selection"] = dataset.metadata["selection"]
                            step_record.warnings.extend(detection_result.warnings)
                            step_record.status = "warning" if step_record.warnings else "done"
                            step_record.outputs.extend(["AtomLattice", "atom_coordinates.csv"])
                        except Exception as exc:  # noqa: BLE001
                            step_record.status = "warning"
                            step_record.warnings.append(f"detect_atoms step fallback: {exc}")
                            step_record.warnings.append(
                                "detect_atoms integration fallback used; "
                                "step kept as contract warning."
                            )
                        if owner is not None:
                            step_record.warnings.append(f"Reserved ownership: Member {owner}.")
                    else:
                        step_record.status = "warning"
                        contract = _RESERVED_STEP_CONTRACTS.get(step_record.name, {})
                        expected_inputs = contract.get("expected_inputs")
                        expected_outputs = contract.get("expected_outputs")
                        owner = contract.get("owner")
                        if isinstance(expected_inputs, list):
                            step_record.inputs = [str(item) for item in expected_inputs]
                        if isinstance(expected_outputs, list):
                            step_record.outputs = [str(item) for item in expected_outputs]
                        step_record.warnings.append(
                            f"Step '{step_record.name}' is reserved for B/C integration and "
                            "not implemented in A2 baseline."
                        )
                        if owner is not None:
                            step_record.warnings.append(f"Reserved ownership: Member {owner}.")
                else:
                    step_record.status = "skipped"
                    step_record.warnings.append(
                        f"Unsupported step '{step_record.name}' was skipped."
                    )

                logger.info(
                    f"Step '{step_record.name}' finished with status={step_record.status}, "
                    f"warnings={len(step_record.warnings)}."
                )
                if step_record.warnings:
                    logger.warning(
                        f"Step '{step_record.name}' produced warnings.",
                        context={"warnings": step_record.warnings},
                    )
                run_warnings = _collect_warnings(run_warnings, step_record.warnings)
                step_records.append(step_record)

            workflow = WorkflowConfig(
                name=recipe.workflow.name,
                description=recipe.workflow.description,
                steps=step_records,
                parameters=recipe.workflow.parameters,
                output_formats=recipe.workflow.output_formats,
                save_intermediates=recipe.workflow.save_intermediates,
                provenance=recipe.workflow.provenance,
            )

            report_path = output_dir / "analysis_report.json"
            artifacts = [
                OutputArtifact(
                    kind="json",
                    uri=str(report_path),
                    label="Analysis report",
                    mime_type="application/json",
                ),
                OutputArtifact(
                    kind="other",
                    uri=str(workflow_used_path),
                    label="Effective workflow recipe",
                    mime_type="application/x-yaml",
                ),
            ]
            if dataset is not None and "selected_npy_sha256" in dataset.provenance:
                artifacts.append(
                    OutputArtifact(
                        kind="array",
                        uri=dataset.data.uri,
                        label="Selected input plane",
                        mime_type="application/octet-stream",
                        provenance={"sha256": dataset.provenance["selected_npy_sha256"]},
                    )
                )
            requested_formats = {item.lower() for item in recipe.outputs.formats}
            requested_formats.update(item.lower() for item in recipe.workflow.output_formats)
            if output_formats_override:
                requested_formats.update(item.lower() for item in output_formats_override)
            unknown_formats = sorted(requested_formats.difference(_SUPPORTED_EXPORT_FORMATS))
            if unknown_formats:
                raise ValueError(
                    "Unsupported output format override(s): "
                    f"{unknown_formats}. Supported: {sorted(_SUPPORTED_EXPORT_FORMATS)}."
                )

            requested_sections = reporting_parameters.get("sections")
            if isinstance(requested_sections, list):
                requested_sections = [str(item) for item in requested_sections]
            else:
                requested_sections = None
            if report_sections_override:
                requested_sections = list(report_sections_override)
            open_sections_from_params = reporting_parameters.get("open_step_sections")
            if isinstance(open_sections_from_params, bool):
                open_sections_setting = open_sections_from_params
            else:
                open_sections_setting = False
            if isinstance(open_step_sections, bool):
                open_sections_setting = open_step_sections

            pdf_parameters = reporting_parameters.get("pdf")
            if not isinstance(pdf_parameters, dict):
                pdf_parameters = {}
            pdf_sections = pdf_parameters.get("sections")
            if isinstance(pdf_sections, list):
                pdf_sections = [str(item) for item in pdf_sections]
            else:
                pdf_sections = None
            if pdf_sections_override:
                pdf_sections = list(pdf_sections_override)
            pdf_max_pages = pdf_parameters.get("max_figure_pages")
            if isinstance(pdf_max_pages, bool):
                pdf_max_pages = None
            elif isinstance(pdf_max_pages, (int, float)):
                pdf_max_pages = int(pdf_max_pages)
            else:
                pdf_max_pages = None
            if isinstance(pdf_max_figure_pages, int):
                pdf_max_pages = pdf_max_figure_pages
            pdf_cover = pdf_parameters.get("cover_fields")
            if not isinstance(pdf_cover, dict):
                pdf_cover = {}
            if pdf_cover_fields:
                merged_cover = dict(pdf_cover)
                merged_cover.update(pdf_cover_fields)
                pdf_cover = merged_cover

            input_path_obj = Path(input_path)
            report_summary = report_title or "A2 baseline workflow completed."
            report_options_payload = {
                "report_title": report_title,
                "output_formats_override": output_formats_override or [],
                "report_sections_override": report_sections_override or [],
                "open_step_sections": open_sections_setting,
                "pdf_sections_override": pdf_sections_override or [],
                "pdf_max_figure_pages": pdf_max_pages,
                "pdf_cover_fields": pdf_cover,
            }
            report = AnalysisReport(
                summary=report_summary,
                workflow=workflow,
                dataset=dataset,
                quality=quality,
                calibration=calibration,
                preprocessing=preprocessing,
                diffraction=diffraction,
                simulation=simulation,
                atom_lattice=atom_lattice,
                figures=[],
                tables=[],
                artifacts=artifacts,
                warnings=_collect_warnings(
                    run_warnings,
                    dataset.warnings if dataset is not None else [],
                    quality.warnings if quality is not None else [],
                    calibration.warnings if calibration is not None else [],
                    preprocessing.warnings if preprocessing is not None else [],
                    diffraction.warnings if diffraction is not None else [],
                    simulation.warnings if simulation is not None else [],
                ),
                provenance={
                    "pipeline": "TEMAnalysisSkill.run",
                    "runtime": _runtime_context(),
                    "input_fingerprint": _input_fingerprint(input_path_obj),
                    "workflow_fingerprint": {
                        "path": str(workflow_used_path),
                        "sha256": _sha256_of_file(workflow_used_path),
                    },
                    "workflow_parameters_snapshot": recipe.workflow.parameters,
                    "workflow_parameters_hash_sha256": _hash_json_payload(
                        recipe.workflow.parameters
                    ),
                    "requested_formats": sorted(requested_formats),
                    "requested_formats_hash_sha256": _hash_json_payload(sorted(requested_formats)),
                    "workflow_parameter_schema_version": recipe.workflow.provenance.get(
                        "workflow_parameter_schema_version",
                        WORKFLOW_PARAMETER_SCHEMA_VERSION,
                    ),
                    "report_options": report_options_payload,
                    "report_options_hash_sha256": _hash_json_payload(report_options_payload),
                },
            )
            export_svg = "svg" in requested_formats
            figure_artifacts, figures, figure_warnings = build_core_plots(
                report,
                output_dir,
                export_svg=export_svg,
            )
            overlay_artifacts, overlay_figures, overlay_warnings = build_core_overlays(
                report,
                output_dir,
                export_svg=export_svg,
            )
            report.figures = [*figures, *overlay_figures]
            report.warnings = _collect_warnings(
                report.warnings,
                figure_warnings,
                overlay_warnings,
            )
            artifacts.extend(figure_artifacts)
            artifacts.extend(overlay_artifacts)

            if recipe.workflow.save_intermediates:
                artifacts.extend(persist_intermediate_artifacts(report, output_dir))
            gate_warnings, gate_summary = _evaluate_quality_gates(
                workflow,
                report,
                reporting_parameters=reporting_parameters,
            )
            report.warnings = _collect_warnings(report.warnings, gate_warnings)
            report.provenance["quality_gates"] = gate_summary
            report.provenance["bc_interface_completeness"] = summarize_bc_interface_rows(
                [
                    {
                        "owner": item["owner"],
                        "item": item["item"],
                        "status": item["status"],
                        "detail": item["detail"],
                    }
                    for item in evaluate_bc_interface_completeness(report)
                ]
            )

            validation_metrics = []
            validation_step_params: dict[str, Any] = {}
            if validate_step_indices:
                validation_step_params = dict(step_records[validate_step_indices[0]].parameters)
            if validate_step_indices or benchmark_step_indices:
                validation_metrics, validation_payload, validation_warnings = (
                    run_validation_summary(
                        report,
                        validate_parameters=validation_step_params,
                    )
                )
                report.warnings = _collect_warnings(report.warnings, validation_warnings)
                report.provenance["validation_summary"] = {
                    "metric_count": validation_payload.get("metric_count", 0),
                    "acceptance": validation_payload.get("acceptance", {}),
                }
            else:
                validation_payload = {}

            if validate_step_indices:
                validation_path = output_dir / "json" / "validation_summary.json"
                _write_json_payload(validation_path, validation_payload)
                artifacts.append(
                    OutputArtifact(
                        kind="json",
                        uri=str(validation_path),
                        label="A4 validation summary",
                        mime_type="application/json",
                    )
                )
                acceptance = validation_payload.get("acceptance", {})
                for idx in validate_step_indices:
                    step_record = workflow.steps[idx]
                    step_record.outputs = [
                        "ValidationMetric[]",
                        str(validation_path),
                    ]
                    step_record.warnings = _collect_warnings(
                        step_record.warnings,
                        validation_warnings,
                    )
                    if isinstance(acceptance, dict) and acceptance.get("all_passed") is False:
                        step_record.status = "warning"

            if benchmark_step_indices:
                benchmark_step_params = dict(step_records[benchmark_step_indices[0]].parameters)
                benchmark_result, benchmark_payload, benchmark_warnings = run_benchmark_summary(
                    report,
                    benchmark_parameters=benchmark_step_params,
                    baseline_metrics=validation_metrics if validation_metrics else None,
                )
                report.benchmark = benchmark_result
                report.warnings = _collect_warnings(report.warnings, benchmark_warnings)
                benchmark_path = output_dir / "json" / "benchmark_report.json"
                _write_json_payload(benchmark_path, benchmark_payload)
                artifacts.append(
                    OutputArtifact(
                        kind="json",
                        uri=str(benchmark_path),
                        label="A4 benchmark report",
                        mime_type="application/json",
                    )
                )
                report.provenance["benchmark_summary"] = {
                    "case_id": benchmark_payload.get("case_id", ""),
                    "metric_count": benchmark_payload.get("metric_count", 0),
                    "passed": benchmark_payload.get("passed", None),
                    "acceptance": benchmark_payload.get("acceptance", {}),
                }
                for idx in benchmark_step_indices:
                    step_record = workflow.steps[idx]
                    step_record.outputs = [
                        "BenchmarkResult",
                        str(benchmark_path),
                    ]
                    step_record.warnings = _collect_warnings(
                        step_record.warnings,
                        benchmark_warnings,
                    )
                    if benchmark_result.passed is False:
                        step_record.status = "warning"

            # Execution, quality checks and reference-backed acceptance are separate facts.
            # Preserve the legacy summary string for existing consumers.
            incomplete_steps = [
                step.name
                for step in workflow.steps
                if step.status == "skipped"
                or any(
                    "not implemented in A2 baseline" in warning
                    or "integration fallback used" in warning
                    for warning in step.warnings
                )
            ]
            if simulation is not None:
                adapter_status = simulation.provenance.get("adapter", {})
                if isinstance(adapter_status, dict) and adapter_status.get("active") is False:
                    if "simulate" not in incomplete_steps:
                        incomplete_steps.append("simulate")
            execution_status = (
                "failed"
                if any(step.status == "failed" for step in workflow.steps)
                else "partial"
                if incomplete_steps
                else "completed"
            )
            validation_acceptance = validation_payload.get("acceptance", {})
            validation_status = (
                str(validation_acceptance.get("status", "not_evaluated"))
                if isinstance(validation_acceptance, dict)
                else "not_evaluated"
            )
            benchmark_passed = report.benchmark.passed if report.benchmark is not None else None
            outcome = {
                "execution_status": execution_status,
                "incomplete_steps": incomplete_steps,
                "quality_gates_passed": bool(gate_summary["passed"]),
                "validation_status": validation_status,
                "benchmark_passed": benchmark_passed,
            }
            report.provenance["run_outcome"] = outcome

            # Export every user-facing snapshot from the same finalized evidence.
            # CSV precedes the bundle so its table count is accurate.
            if "csv" in requested_formats:
                csv_artifacts, report_tables = export_csv_tables_with_models(report, output_dir)
                report.tables = report_tables
                artifacts.extend(csv_artifacts)
            if "json" in requested_formats:
                artifacts.extend(export_json_bundle(report, output_dir))
            if "html" in requested_formats:
                artifacts.append(
                    export_html_report(
                        report,
                        output_dir,
                        options={
                            "sections": requested_sections,
                            "open_step_sections": open_sections_setting,
                        },
                    )
                )
            if "pdf" in requested_formats:
                artifacts.append(
                    export_pdf_report(
                        report,
                        output_dir,
                        options={
                            "sections": pdf_sections,
                            "max_figure_pages": pdf_max_pages if pdf_max_pages is not None else 8,
                            "cover_fields": pdf_cover,
                        },
                    )
                )

            run_summary_path = output_dir / "run_summary.json"
            summary = RunSummary(
                status="A2 baseline workflow completed.",
                workflow_name=workflow.name,
                input_path=str(Path(input_path)),
                workflow_path=str(Path(workflow_path)) if workflow_path else "",
                output_dir=str(output_dir),
                report_path=str(report_path),
                log_path=str(log_path),
                workflow_used_path=str(workflow_used_path),
                workflow_parameter_schema_version=report.provenance.get(
                    "workflow_parameter_schema_version",
                    WORKFLOW_PARAMETER_SCHEMA_VERSION,
                ),
                steps=[
                    RunSummaryStep(
                        name=step.name,
                        status=step.status,
                        warnings=list(step.warnings),
                        warning_count=len(step.warnings),
                    )
                    for step in workflow.steps
                ],
                warnings=list(report.warnings),
                warning_count=len(report.warnings),
                runtime=report.provenance.get("runtime", {}),
                input_fingerprint=report.provenance.get("input_fingerprint", {}),
                workflow_fingerprint=report.provenance.get("workflow_fingerprint", {}),
                workflow_parameters_snapshot=report.provenance.get(
                    "workflow_parameters_snapshot",
                    {},
                ),
                bc_interface_summary=report.provenance.get("bc_interface_completeness", {}),
            )
            _write_run_summary(run_summary_path, summary)
            logger.info(f"Run summary written to '{run_summary_path}'.")

            _append_artifact_unique(
                artifacts,
                OutputArtifact(
                    kind="log",
                    uri=str(log_path),
                    label="Run log",
                    mime_type="application/x-ndjson",
                ),
            )
            _append_artifact_unique(
                artifacts,
                OutputArtifact(
                    kind="json",
                    uri=str(run_summary_path),
                    label="Run summary",
                    mime_type="application/json",
                    provenance={"sha256": _sha256_of_file(run_summary_path)},
                ),
            )
            _append_artifact_unique(
                artifacts,
                OutputArtifact(
                    kind="other",
                    uri=str(workflow_used_path),
                    label="Effective workflow recipe",
                    mime_type="application/x-yaml",
                    provenance={"sha256": _sha256_of_file(workflow_used_path)},
                ),
            )
            _append_artifact_unique(
                artifacts,
                OutputArtifact(
                    kind="json",
                    uri=str(report_path),
                    label="Analysis report",
                    mime_type="application/json",
                ),
            )

            report.artifacts = artifacts
            manifest_artifact = write_output_manifest(
                report,
                output_dir,
                requested_formats=requested_formats,
            )
            report.artifacts.append(manifest_artifact)
            write_json(report_path, report)
            logger.info(f"Report written to '{report_path}'.")

            return {
                "status": summary.status,
                "input_path": summary.input_path,
                "workflow_path": summary.workflow_path,
                "output_dir": summary.output_dir,
                "report_path": summary.report_path,
                "workflow_used_path": summary.workflow_used_path,
                "run_summary_path": str(run_summary_path),
                "log_path": summary.log_path,
                "atom_lattice": atom_lattice,
                "atom_count": len(atom_lattice.atoms) if atom_lattice is not None else 0,
                **outcome,
            }
        except Exception as exc:
            logger.error(f"Run failed: {exc}")
            logger.error(traceback.format_exc())
            raise
        finally:
            logger.flush()
