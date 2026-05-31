"""A2 baseline simulation adapters from phase-candidate evidence."""

from __future__ import annotations

from autotem.core import DiffractionResult, Measurement, SimulatedPattern, SimulationResult
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


def _baseline_simulation(diffraction: DiffractionResult) -> SimulationResult:
    """Create simple simulation placeholders linked to matched phase candidates."""
    warnings = list(diffraction.warnings)
    if not diffraction.phase_candidates:
        warnings.append("No phase candidates available; simulation outputs are empty.")
        return SimulationResult(
            confidence=0.2,
            warnings=warnings,
            provenance={"method": "a2_baseline_no_phase"},
        )

    patterns: list[SimulatedPattern] = []
    metrics: list[Measurement] = []

    for candidate in diffraction.phase_candidates:
        patterns.append(
            SimulatedPattern(
                kind="diffraction",
                structure_id=candidate.identifier or candidate.name,
                zone_axis=candidate.zone_axis or (0, 0, 1),
                microscope_parameters={"voltage_kv": 200},
                simulation_parameters={"engine": "a2_baseline"},
                score=Measurement(
                    value=float(candidate.score or 0.0),
                    unit="confidence",
                    method="phase_match_score",
                ),
            )
        )

    best = diffraction.phase_candidates[0]
    if best.mean_d_error is not None:
        metrics.append(
            Measurement(
                value=float(best.mean_d_error.value),
                unit=best.mean_d_error.unit,
                method="best_phase_mean_d_error",
            )
        )

    return SimulationResult(
        patterns=patterns,
        matched_phase_candidates=diffraction.phase_candidates,
        comparison_metrics=metrics,
        confidence=max(0.3, min(0.9, float(best.score or 0.0))),
        warnings=warnings,
        provenance={"method": "a2_baseline"},
    )


def simulate_from_diffraction(
    diffraction: DiffractionResult,
    *,
    parameters: dict[str, object] | None = None,
) -> SimulationResult:
    """Simulate diffraction evidence with optional external engines."""
    params = dict(parameters or {})
    engine = str(params.get("simulation_engine", "baseline")).strip().lower()
    if engine != "diffsims":
        return _baseline_simulation(diffraction)

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
