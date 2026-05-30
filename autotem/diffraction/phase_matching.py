"""Experimental-vs-simulated diffraction phase matching helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin
from typing import Any

import numpy as np

from autotem.core import (
    DiffractionResult,
    GVector,
    Measurement,
    PhaseCandidate,
    SimulatedPattern,
    SimulatedSpot,
)


@dataclass(slots=True)
class _MeasuredSpot:
    index: int
    gx: float
    gy: float
    magnitude: float
    intensity: float


@dataclass(slots=True)
class _TemplateSpot:
    index: int
    gx: float
    gy: float
    magnitude: float
    intensity: float


@dataclass(slots=True)
class _MatchSummary:
    matched_count: int
    mean_d_error: float | None
    mean_angle_error_deg: float | None
    score: float
    rotation_deg: float
    matched_pairs: list[dict[str, Any]]


@dataclass(slots=True)
class PhaseMatchingResult:
    """Diffraction phase matching result derived from simulated patterns."""

    candidates: list[PhaseCandidate]
    patterns: list[SimulatedPattern]
    comparison_metrics: list[Measurement]
    warnings: list[str]
    provenance: dict[str, Any]


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _measurement_float(measurement: Measurement | None) -> float | None:
    if measurement is None:
        return None
    return _as_float(measurement.value)


def _gvector_components(g_vector: GVector) -> tuple[float, float] | None:
    gx = _measurement_float(g_vector.gx)
    gy = _measurement_float(g_vector.gy)
    if gx is None or gy is None:
        return None
    if not np.isfinite([gx, gy]).all():
        return None
    return gx, gy


def _measured_spots(diffraction: DiffractionResult) -> list[_MeasuredSpot]:
    measured: list[_MeasuredSpot] = []
    for idx, g_vector in enumerate(diffraction.g_vectors):
        components = _gvector_components(g_vector)
        if components is None:
            continue
        gx, gy = components
        magnitude = _measurement_float(g_vector.magnitude)
        if magnitude is None:
            magnitude = float(np.hypot(gx, gy))
        if magnitude <= 1e-12:
            continue
        intensity = 1.0
        peak_index = g_vector.source_peak_index if g_vector.source_peak_index is not None else idx
        if 0 <= peak_index < len(diffraction.peaks):
            intensity = float(diffraction.peaks[peak_index].intensity or 1.0)
        measured.append(
            _MeasuredSpot(
                index=idx,
                gx=float(gx),
                gy=float(gy),
                magnitude=float(magnitude),
                intensity=max(1e-9, intensity),
            )
        )
    return measured


def _spot_components(spot: SimulatedSpot) -> tuple[float, float] | None:
    gx = _measurement_float(spot.gx)
    gy = _measurement_float(spot.gy)
    if gx is None or gy is None:
        return None
    if not np.isfinite([gx, gy]).all():
        return None
    return gx, gy


def _template_spots(pattern: SimulatedPattern) -> list[_TemplateSpot]:
    template: list[_TemplateSpot] = []
    for idx, spot in enumerate(pattern.spots):
        components = _spot_components(spot)
        if components is None:
            continue
        gx, gy = components
        magnitude = _measurement_float(spot.magnitude)
        if magnitude is None:
            magnitude = float(np.hypot(gx, gy))
        if magnitude <= 1e-12:
            continue
        intensity = float(spot.intensity if spot.intensity is not None else 1.0)
        template.append(
            _TemplateSpot(
                index=idx,
                gx=float(gx),
                gy=float(gy),
                magnitude=float(magnitude),
                intensity=max(1e-9, intensity),
            )
        )
    return template


def _angle_rad(gx: float, gy: float) -> float:
    return float(atan2(gy, gx))


def _normalize_rotation_rad(angle: float) -> float:
    angle = (float(angle) + pi) % pi
    if angle < 0:
        angle += pi
    return angle


def _rotation_candidates(
    measured: list[_MeasuredSpot],
    template: list[_TemplateSpot],
    *,
    max_rotation_candidates: int,
) -> list[float]:
    measured_ranked = sorted(measured, key=lambda item: item.intensity, reverse=True)
    template_ranked = sorted(template, key=lambda item: item.intensity, reverse=True)
    anchor_count = max(1, min(8, int(np.sqrt(max(1, max_rotation_candidates)))))

    rotations = [0.0]
    for meas in measured_ranked[:anchor_count]:
        theta_m = _angle_rad(meas.gx, meas.gy)
        for sim in template_ranked[:anchor_count]:
            theta_s = _angle_rad(sim.gx, sim.gy)
            rotations.append(_normalize_rotation_rad(theta_m - theta_s))

    unique: list[float] = []
    seen: set[int] = set()
    for rotation in rotations:
        key = int(round(_normalize_rotation_rad(rotation) * 100000.0))
        if key in seen:
            continue
        seen.add(key)
        unique.append(_normalize_rotation_rad(rotation))
        if len(unique) >= max(1, int(max_rotation_candidates)):
            break
    return unique


def _rotate(gx: float, gy: float, rotation: float) -> tuple[float, float]:
    cr = cos(rotation)
    sr = sin(rotation)
    return gx * cr - gy * sr, gx * sr + gy * cr


def _friedel_angle_error_deg(
    meas: _MeasuredSpot,
    sim: _TemplateSpot,
    *,
    rotation: float,
) -> float:
    sx, sy = _rotate(sim.gx, sim.gy, rotation)
    denom = max(1e-12, float(meas.magnitude * sim.magnitude))
    cosang = (meas.gx * sx + meas.gy * sy) / denom
    cosang = max(-1.0, min(1.0, float(cosang)))
    # Experimental diffraction generally treats Friedel pairs as equivalent.
    return float(np.degrees(np.arccos(abs(cosang))))


def _d_error(meas: _MeasuredSpot, sim: _TemplateSpot) -> float:
    d_meas = 1.0 / max(meas.magnitude, 1e-12)
    d_sim = 1.0 / max(sim.magnitude, 1e-12)
    return abs(d_sim - d_meas) / max(d_meas, 1e-12)


def _score_template_at_rotation(
    measured: list[_MeasuredSpot],
    template: list[_TemplateSpot],
    *,
    rotation: float,
    d_tolerance_rel: float,
    angle_tolerance_deg: float,
) -> _MatchSummary:
    used_simulated: set[int] = set()
    d_errors: list[float] = []
    angle_errors: list[float] = []
    pairs: list[dict[str, Any]] = []

    measured_ranked = sorted(measured, key=lambda item: item.intensity, reverse=True)
    for meas in measured_ranked:
        best: tuple[float, _TemplateSpot, float, float] | None = None
        for sim in template:
            if sim.index in used_simulated:
                continue
            d_err = _d_error(meas, sim)
            if d_err > d_tolerance_rel:
                continue
            angle_err = _friedel_angle_error_deg(meas, sim, rotation=rotation)
            if angle_err > angle_tolerance_deg:
                continue
            pair_score = d_err / max(d_tolerance_rel, 1e-9)
            pair_score += angle_err / max(angle_tolerance_deg, 1e-9)
            if best is None or pair_score < best[0]:
                best = (pair_score, sim, d_err, angle_err)
        if best is None:
            continue
        _, sim, d_err, angle_err = best
        used_simulated.add(sim.index)
        d_errors.append(float(d_err))
        angle_errors.append(float(angle_err))
        pairs.append(
            {
                "measured_g_index": meas.index,
                "simulated_spot_index": sim.index,
                "d_error_rel": float(d_err),
                "angle_error_deg": float(angle_err),
            }
        )

    if not pairs:
        return _MatchSummary(
            matched_count=0,
            mean_d_error=None,
            mean_angle_error_deg=None,
            score=0.0,
            rotation_deg=float(np.degrees(rotation)),
            matched_pairs=[],
        )

    mean_d_error = float(np.mean(d_errors))
    mean_angle_error = float(np.mean(angle_errors))
    coverage = len(pairs) / max(1, len(measured))
    score = coverage
    score *= float(np.exp(-mean_d_error / max(d_tolerance_rel, 1e-9)))
    score *= float(np.exp(-mean_angle_error / max(angle_tolerance_deg, 1e-9)))
    return _MatchSummary(
        matched_count=len(pairs),
        mean_d_error=mean_d_error,
        mean_angle_error_deg=mean_angle_error,
        score=max(0.0, min(1.0, score)),
        rotation_deg=float(np.degrees(rotation)),
        matched_pairs=pairs,
    )


def _best_template_match(
    measured: list[_MeasuredSpot],
    template: list[_TemplateSpot],
    *,
    d_tolerance_rel: float,
    angle_tolerance_deg: float,
    max_rotation_candidates: int,
) -> _MatchSummary:
    best: _MatchSummary | None = None
    for rotation in _rotation_candidates(
        measured,
        template,
        max_rotation_candidates=max_rotation_candidates,
    ):
        summary = _score_template_at_rotation(
            measured,
            template,
            rotation=rotation,
            d_tolerance_rel=d_tolerance_rel,
            angle_tolerance_deg=angle_tolerance_deg,
        )
        if best is None:
            best = summary
            continue
        if summary.score > best.score:
            best = summary
        elif summary.score == best.score and summary.matched_count > best.matched_count:
            best = summary
    return best or _MatchSummary(0, None, None, 0.0, 0.0, [])


def _candidate_key_values(candidate: PhaseCandidate) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (candidate.identifier, candidate.name)
        if value is not None and str(value).strip()
    }


def _pattern_key_values(pattern: SimulatedPattern) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (pattern.structure_id,)
        if value is not None and str(value).strip()
    }


def _patterns_for_candidate(
    candidate: PhaseCandidate,
    patterns: list[SimulatedPattern],
) -> list[tuple[int, SimulatedPattern]]:
    candidate_keys = _candidate_key_values(candidate)
    matches: list[tuple[int, SimulatedPattern]] = []
    for idx, pattern in enumerate(patterns):
        if candidate_keys.intersection(_pattern_key_values(pattern)):
            matches.append((idx, pattern))
    return matches


def _zone_axis_payload(axis: tuple[int, int, int] | None) -> list[int] | None:
    return list(axis) if axis is not None else None


def _updated_candidate(
    candidate: PhaseCandidate,
    pattern: SimulatedPattern,
    summary: _MatchSummary,
    *,
    previous_score_weight: float,
) -> PhaseCandidate:
    previous_score = float(candidate.score or 0.0)
    raw_score = float(summary.score)
    weight = max(0.0, min(1.0, float(previous_score_weight)))
    if candidate.score is None:
        score = raw_score
    else:
        score = (1.0 - weight) * raw_score + weight * previous_score
    score = max(0.0, min(1.0, score))

    mean_d_error = (
        None
        if summary.mean_d_error is None
        else Measurement(
            value=summary.mean_d_error,
            unit="relative",
            method="diffsims_phase_match",
            provenance={
                "previous_score": previous_score,
                "raw_diffsims_score": raw_score,
                "rotation_deg": summary.rotation_deg,
                "zone_axis": _zone_axis_payload(pattern.zone_axis),
            },
        )
    )
    mean_angle_error = (
        None
        if summary.mean_angle_error_deg is None
        else Measurement(
            value=summary.mean_angle_error_deg,
            unit="deg",
            method="diffsims_phase_match",
            provenance={
                "previous_score": previous_score,
                "raw_diffsims_score": raw_score,
                "rotation_deg": summary.rotation_deg,
                "zone_axis": _zone_axis_payload(pattern.zone_axis),
            },
        )
    )

    warnings = list(candidate.warnings)
    if summary.matched_count == 0:
        warnings.append("diffsims phase matching found no spot matches.")

    return candidate.model_copy(
        update={
            "score": score,
            "matched_spot_count": summary.matched_count,
            "mean_d_error": mean_d_error,
            "mean_angle_error": mean_angle_error,
            "zone_axis": pattern.zone_axis or candidate.zone_axis,
            "warnings": list(dict.fromkeys(warnings)),
        }
    )


def _updated_pattern(pattern: SimulatedPattern, summary: _MatchSummary) -> SimulatedPattern:
    return pattern.model_copy(
        update={
            "score": Measurement(
                value=float(summary.score),
                unit="confidence",
                method="diffsims_phase_match_score",
                parameters={
                    "matched_spot_count": summary.matched_count,
                    "mean_d_error_rel": summary.mean_d_error,
                    "mean_angle_error_deg": summary.mean_angle_error_deg,
                    "rotation_deg": summary.rotation_deg,
                    "zone_axis": _zone_axis_payload(pattern.zone_axis),
                },
            ),
            "simulation_parameters": {
                **pattern.simulation_parameters,
                "phase_match": {
                    "matched_spot_count": summary.matched_count,
                    "mean_d_error_rel": summary.mean_d_error,
                    "mean_angle_error_deg": summary.mean_angle_error_deg,
                    "rotation_deg": summary.rotation_deg,
                    "zone_axis": _zone_axis_payload(pattern.zone_axis),
                    "score": summary.score,
                    "matched_pairs": summary.matched_pairs,
                },
            },
        }
    )


def _comparison_metrics(best_candidate: PhaseCandidate | None) -> list[Measurement]:
    if best_candidate is None:
        return []
    metrics = [
        Measurement(
            value=float(best_candidate.score or 0.0),
            unit="confidence",
            method="diffsims_phase_match_score",
        ),
        Measurement(
            value=int(best_candidate.matched_spot_count or 0),
            unit="spots",
            method="diffsims_phase_matched_spot_count",
        ),
    ]
    if best_candidate.mean_d_error is not None:
        metrics.append(
            Measurement(
                value=best_candidate.mean_d_error.value,
                unit=best_candidate.mean_d_error.unit,
                method="diffsims_phase_mean_d_error",
            )
        )
    if best_candidate.mean_angle_error is not None:
        metrics.append(
            Measurement(
                value=best_candidate.mean_angle_error.value,
                unit=best_candidate.mean_angle_error.unit,
                method="diffsims_phase_mean_angle_error",
            )
        )
    return metrics


def match_diffsims_patterns_to_diffraction(
    diffraction: DiffractionResult,
    patterns: list[SimulatedPattern],
    *,
    d_tolerance_rel: float = 0.10,
    angle_tolerance_deg: float = 12.0,
    min_matched_spots: int = 2,
    top_n: int = 3,
    previous_score_weight: float = 0.35,
    max_rotation_candidates: int = 64,
) -> PhaseMatchingResult:
    """Match diffsims patterns against measured g-vectors and rank phases."""
    warnings: list[str] = []
    measured = _measured_spots(diffraction)
    if not measured:
        return PhaseMatchingResult(
            candidates=list(diffraction.phase_candidates),
            patterns=patterns,
            comparison_metrics=[],
            warnings=["diffsims phase matching skipped: no measured g-vectors available."],
            provenance={"active": False, "reason": "no_measured_spots"},
        )
    if not patterns:
        return PhaseMatchingResult(
            candidates=list(diffraction.phase_candidates),
            patterns=[],
            comparison_metrics=[],
            warnings=["diffsims phase matching skipped: no simulated patterns available."],
            provenance={"active": False, "reason": "no_simulated_patterns"},
        )

    summaries: dict[str, _MatchSummary] = {}
    updated_patterns = list(patterns)
    updated_candidates: list[PhaseCandidate] = []
    per_phase: list[dict[str, Any]] = []

    for candidate in diffraction.phase_candidates:
        candidate_patterns = _patterns_for_candidate(candidate, patterns)
        if not candidate_patterns:
            candidate_warning = f"diffsims phase matching skipped '{candidate.name}': no pattern."
            warnings.append(candidate_warning)
            updated_candidates.append(
                candidate.model_copy(
                    update={
                        "warnings": list(dict.fromkeys([*candidate.warnings, candidate_warning]))
                    }
                )
            )
            continue

        best_pattern_idx = -1
        best_pattern: SimulatedPattern | None = None
        best_template_count = 0
        best_summary: _MatchSummary | None = None
        empty_template_count = 0
        for pattern_idx, pattern in candidate_patterns:
            template = _template_spots(pattern)
            if not template:
                empty_template_count += 1
                continue
            summary = _best_template_match(
                measured,
                template,
                d_tolerance_rel=max(1e-6, float(d_tolerance_rel)),
                angle_tolerance_deg=max(0.1, float(angle_tolerance_deg)),
                max_rotation_candidates=max(1, int(max_rotation_candidates)),
            )
            if best_summary is None or summary.score > best_summary.score:
                best_pattern_idx = pattern_idx
                best_pattern = pattern
                best_template_count = len(template)
                best_summary = summary
            elif best_summary is not None and summary.score == best_summary.score:
                if summary.matched_count > best_summary.matched_count:
                    best_pattern_idx = pattern_idx
                    best_pattern = pattern
                    best_template_count = len(template)
                    best_summary = summary

        if best_pattern is None or best_summary is None:
            candidate_warning = (
                f"diffsims phase matching skipped '{candidate.name}': "
                "pattern contains no simulated spots."
            )
            warnings.append(candidate_warning)
            updated_candidates.append(
                candidate.model_copy(
                    update={
                        "warnings": list(dict.fromkeys([*candidate.warnings, candidate_warning]))
                    }
                )
            )
            continue

        summary = best_summary
        pattern = best_pattern
        if summary.matched_count < max(1, int(min_matched_spots)):
            warnings.append(
                f"diffsims phase matching low support for '{candidate.name}': "
                f"{summary.matched_count} matched spots."
            )
        updated_candidate = _updated_candidate(
            candidate,
            pattern,
            summary,
            previous_score_weight=previous_score_weight,
        )
        updated_pattern = _updated_pattern(pattern, summary)
        updated_candidates.append(updated_candidate)
        updated_patterns[best_pattern_idx] = updated_pattern
        summaries[candidate.name] = summary
        per_phase.append(
            {
                "phase_name": candidate.name,
                "structure_id": pattern.structure_id,
                "simulated_pattern_count": len(candidate_patterns),
                "empty_simulated_pattern_count": empty_template_count,
                "best_pattern_index": best_pattern_idx,
                "simulated_spot_count": best_template_count,
                "measured_spot_count": len(measured),
                "matched_spot_count": summary.matched_count,
                "mean_d_error_rel": summary.mean_d_error,
                "mean_angle_error_deg": summary.mean_angle_error_deg,
                "rotation_deg": summary.rotation_deg,
                "simulation_rotation_angle_deg": pattern.simulation_parameters.get(
                    "rotation_angle_deg"
                ),
                "simulation_rotation_axis": pattern.simulation_parameters.get("rotation_axis"),
                "zone_axis": _zone_axis_payload(pattern.zone_axis),
                "zone_axis_index": pattern.simulation_parameters.get("zone_axis_index"),
                "score": summary.score,
            }
        )

    ranked = sorted(updated_candidates, key=lambda item: float(item.score or 0.0), reverse=True)
    ranked = ranked[: max(1, int(top_n))]
    best_candidate = ranked[0] if ranked else None

    active = any(summary.matched_count > 0 for summary in summaries.values())
    return PhaseMatchingResult(
        candidates=ranked,
        patterns=updated_patterns,
        comparison_metrics=_comparison_metrics(best_candidate),
        warnings=list(dict.fromkeys(warnings)),
        provenance={
            "active": active,
            "method": "diffsims_template_spot_match",
            "measured_spot_count": len(measured),
            "simulated_pattern_count": len(patterns),
            "d_tolerance_rel": float(d_tolerance_rel),
            "angle_tolerance_deg": float(angle_tolerance_deg),
            "min_matched_spots": int(min_matched_spots),
            "previous_score_weight": max(0.0, min(1.0, float(previous_score_weight))),
            "max_rotation_candidates": int(max_rotation_candidates),
            "selected_phase": best_candidate.name if best_candidate is not None else None,
            "selected_zone_axis": _zone_axis_payload(best_candidate.zone_axis)
            if best_candidate is not None
            else None,
            "per_phase": per_phase,
        },
    )
