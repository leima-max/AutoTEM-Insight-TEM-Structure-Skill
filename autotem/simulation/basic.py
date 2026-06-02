"""B5 simulation adapters from phase-candidate evidence."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from autotem.core import DiffractionResult, Measurement, SimulationResult
from autotem.diffraction.phase_matching import match_diffsims_patterns_to_diffraction
from autotem.simulation.diffsims_adapter import simulate_diffraction_with_diffsims


def _to_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _zone_axis_tuple(raw_axis: object) -> tuple[int, int, int] | None:
    if not isinstance(raw_axis, (list, tuple)) or len(raw_axis) < 3:
        return None
    try:
        axis = tuple(int(round(float(raw_axis[idx]))) for idx in range(3))
    except (TypeError, ValueError):
        return None
    if axis == (0, 0, 0):
        return None
    return axis


def _zone_axis_candidates(raw_axes: object) -> list[tuple[int, int, int]] | None:
    if raw_axes is None:
        return None
    single_axis = _zone_axis_tuple(raw_axes)
    if single_axis is not None:
        return [single_axis]
    if not isinstance(raw_axes, list):
        return None
    axes: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for raw_axis in raw_axes:
        axis = _zone_axis_tuple(raw_axis)
        if axis is None or axis in seen:
            continue
        seen.add(axis)
        axes.append(axis)
    return axes or None


def _diffsims_timeout_s(params: dict[str, object]) -> float | None:
    raw = params.get("diffsims_timeout_s", params.get("timeout_s"))
    if raw is None:
        return None
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _diffsims_fallback_result(
    diffraction: DiffractionResult,
    *,
    warning: str,
    reason: str,
    timeout_s: float | None = None,
) -> SimulationResult:
    provenance: dict[str, object] = {
        "method": "diffsims",
        "adapter": {
            "adapter": "diffsims",
            "active": False,
            "reason": reason,
        },
        "phase_matching": {
            "active": False,
            "reason": reason,
        },
    }
    if timeout_s is not None:
        provenance["adapter"]["timeout_s"] = float(timeout_s)  # type: ignore[index]
    return SimulationResult(
        matched_phase_candidates=list(diffraction.phase_candidates),
        confidence=0.2,
        warnings=[*diffraction.warnings, warning],
        provenance=provenance,
    )


def _tail_text(text: str, *, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _temporary_directory_root() -> str | None:
    root = os.environ.get("AUTOTEM_TMP_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        return root
    if os.name == "nt":
        windows_root = Path("C:/tmp")
        try:
            windows_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return str(windows_root)
    return None


def _simulate_diffsims_with_timeout(
    diffraction: DiffractionResult,
    *,
    parameters: dict[str, object] | None = None,
    timeout_s: float,
) -> SimulationResult:
    timeout = max(1.0, float(timeout_s))
    with tempfile.TemporaryDirectory(
        prefix="autotem_diffsims_",
        dir=_temporary_directory_root(),
        ignore_cleanup_errors=True,
    ) as tmp_dir:
        input_path = Path(tmp_dir) / "request.json"
        output_path = Path(tmp_dir) / "response.json"
        stdout_path = Path(tmp_dir) / "stdout.txt"
        stderr_path = Path(tmp_dir) / "stderr.txt"
        input_path.write_text(
            json.dumps(
                {
                    "diffraction": diffraction.model_dump(mode="json"),
                    "parameters": dict(parameters or {}),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle:
                with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "autotem.simulation.diffsims_worker",
                            "--input",
                            str(input_path),
                            "--output",
                            str(output_path),
                        ],
                        cwd=Path.cwd(),
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
        except subprocess.TimeoutExpired:
            warning = (
                "B5 diffsims simulation timed out in subprocess "
                f"after {timeout:.1f}s."
            )
            return _diffsims_fallback_result(
                diffraction,
                warning=warning,
                reason="timeout",
                timeout_s=timeout,
            )
        if completed.returncode != 0:
            stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
            stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
            detail = _tail_text((stderr or stdout).strip())
            warning = (
                "B5 diffsims subprocess failed"
                + (f": {detail}" if detail else f" with exit code {completed.returncode}")
            )
            return _diffsims_fallback_result(
                diffraction,
                warning=warning,
                reason="worker_failed",
            )
        if not output_path.exists():
            warning = "B5 diffsims subprocess exited without writing a result."
            return _diffsims_fallback_result(
                diffraction,
                warning=warning,
                reason="worker_empty",
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload.get("ok"):
            error = (
                str(payload.get("error", "unknown diffsims subprocess error"))
                if isinstance(payload, dict)
                else "invalid diffsims subprocess payload"
            )
            warning = f"B5 diffsims subprocess failed: {error}"
            return _diffsims_fallback_result(
                diffraction,
                warning=warning,
                reason="worker_failed",
            )
        simulation = SimulationResult.model_validate(payload["simulation"])
        updated_diffraction = DiffractionResult.model_validate(payload["diffraction"])
        diffraction.phase_candidates = list(updated_diffraction.phase_candidates)
        return simulation


def simulate_from_diffraction(
    diffraction: DiffractionResult,
    *,
    parameters: dict[str, object] | None = None,
) -> SimulationResult:
    """Simulate diffraction evidence with optional external engines."""
    params = dict(parameters or {})
    engine = str(params.get("simulation_engine", "diffsims")).strip().lower()
    if engine != "diffsims":
        warning = (
            f"B5 simulation engine '{engine}' is no longer supported; "
            "use 'diffsims' to generate simulated diffraction spots."
        )
        return SimulationResult(
            matched_phase_candidates=list(diffraction.phase_candidates),
            confidence=0.0,
            warnings=[*diffraction.warnings, warning],
            provenance={
                "method": "unsupported_simulation_engine",
                "engine": engine,
                "adapter": {"active": False, "reason": "unsupported_engine"},
                "phase_matching": {"active": False, "reason": "unsupported_engine"},
            },
        )
    timeout = _diffsims_timeout_s(params)
    if timeout is not None:
        return _simulate_diffsims_with_timeout(
            diffraction,
            parameters=params,
            timeout_s=timeout,
        )
    return _simulate_diffsims_in_process(diffraction, params)


def _simulate_diffsims_in_process(
    diffraction: DiffractionResult,
    params: dict[str, object],
) -> SimulationResult:
    """Run the real diffsims adapter in the current process."""
    voltage_kv = float(params.get("voltage_kv", 200.0))
    reciprocal_radius = float(params.get("reciprocal_radius_inv_angstrom", 1.0))
    if params.get("max_candidates") is None:
        max_candidates = max(1, len(diffraction.phase_candidates) or 3)
    else:
        max_candidates = int(params.get("max_candidates", 3))
    calibration = float(params.get("calibration", 0.01))
    max_simulated_spots = int(params.get("max_simulated_spots", 256))
    minimum_simulated_g = float(params.get("minimum_simulated_g_inv_nm", 1e-6))
    rotation_axis_raw = params.get("rotation_axis")
    rotation_axis = None
    if isinstance(rotation_axis_raw, (list, tuple)) and len(rotation_axis_raw) >= 3:
        rotation_axis = (
            float(rotation_axis_raw[0]),
            float(rotation_axis_raw[1]),
            float(rotation_axis_raw[2]),
        )
    rotation_angles_raw = params.get("rotation_angles_deg")
    rotation_angles = None
    if isinstance(rotation_angles_raw, (list, tuple)):
        rotation_angles = [float(value) for value in rotation_angles_raw]
    pattern_shape_raw = params.get("pattern_shape")
    pattern_shape = None
    if isinstance(pattern_shape_raw, list) and len(pattern_shape_raw) >= 2:
        pattern_shape = (int(pattern_shape_raw[0]), int(pattern_shape_raw[1]))
    elif isinstance(pattern_shape_raw, tuple) and len(pattern_shape_raw) >= 2:
        pattern_shape = (int(pattern_shape_raw[0]), int(pattern_shape_raw[1]))

    result = simulate_diffraction_with_diffsims(
        diffraction.phase_candidates,
        voltage_kv=voltage_kv,
        reciprocal_radius_inv_angstrom=reciprocal_radius,
        max_candidates=max(1, max_candidates),
        pattern_shape=pattern_shape,
        calibration=calibration,
        minimum_simulated_g_inv_nm=max(0.0, minimum_simulated_g),
        max_simulated_spots=max(1, max_simulated_spots),
        with_direct_beam=_to_bool(params.get("with_direct_beam"), default=True),
        max_excitation_error=float(params.get("max_excitation_error", 0.01)),
        shape_factor_width=(
            None
            if params.get("shape_factor_width") is None
            else float(params.get("shape_factor_width"))
        ),
        zone_axis_candidates=_zone_axis_candidates(params.get("zone_axis_candidates")),
        rotation_axis=rotation_axis,
        rotation_angles_deg=rotation_angles,
        fallback_single_atom=_to_bool(params.get("fallback_single_atom"), default=True),
        pattern_sigma=float(params.get("pattern_sigma", 10.0)),
        pattern_in_plane_angle_deg=float(params.get("pattern_in_plane_angle_deg", 0.0)),
        pattern_mirrored=_to_bool(params.get("pattern_mirrored"), default=False),
        pattern_normalize=_to_bool(params.get("pattern_normalize"), default=True),
        pattern_clip_threshold=float(params.get("pattern_clip_threshold", 1.0)),
    )
    warnings = [*diffraction.warnings, *result.warnings]
    metrics: list[Measurement] = []
    matched_candidates = list(diffraction.phase_candidates)
    matched_patterns = result.patterns
    phase_matching_provenance = {"active": False, "reason": "disabled"}
    if _to_bool(params.get("phase_match_enabled"), default=True):
        match_result = match_diffsims_patterns_to_diffraction(
            diffraction,
            result.patterns,
            d_tolerance_rel=float(params.get("phase_match_d_tolerance_rel", 0.10)),
            angle_tolerance_deg=float(params.get("phase_match_angle_tolerance_deg", 12.0)),
            min_matched_spots=int(params.get("phase_match_min_spots", 2)),
            top_n=int(params.get("phase_match_top_n", max_candidates)),
            previous_score_weight=float(params.get("phase_match_previous_score_weight", 0.35)),
            max_rotation_candidates=int(params.get("phase_match_max_rotation_candidates", 64)),
            preserve_candidate_zone_axis=_to_bool(
                params.get("preserve_candidate_zone_axis"),
                default=False,
            ),
            preserve_candidate_ranking=_to_bool(
                params.get("preserve_candidate_ranking"),
                default=False,
            ),
        )
        matched_candidates = match_result.candidates
        matched_patterns = match_result.patterns
        metrics.extend(match_result.comparison_metrics)
        warnings.extend(match_result.warnings)
        phase_matching_provenance = match_result.provenance
        if matched_candidates:
            diffraction.phase_candidates = matched_candidates

    if not metrics and matched_candidates and matched_candidates[0].mean_d_error is not None:
        best_error = matched_candidates[0].mean_d_error
        if isinstance(best_error.value, (int, float)):
            metrics.append(
                Measurement(
                    value=float(best_error.value),
                    unit=best_error.unit,
                    method="best_phase_mean_d_error",
                )
            )
    return SimulationResult(
        patterns=matched_patterns,
        matched_phase_candidates=matched_candidates,
        comparison_metrics=metrics,
        confidence=(
            max((float(candidate.score or 0.0) for candidate in matched_candidates), default=0.0)
            if phase_matching_provenance.get("active")
            else (0.5 if result.patterns else 0.2)
        ),
        warnings=warnings,
        provenance={
            "method": "diffsims",
            "adapter": result.provenance,
            "phase_matching": phase_matching_provenance,
        },
    )
