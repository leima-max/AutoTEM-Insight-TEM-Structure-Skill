"""A2 baseline simulation adapters from phase-candidate evidence."""

from __future__ import annotations

from autotem.core import DiffractionResult, Measurement, SimulatedPattern, SimulationResult


def simulate_from_diffraction(diffraction: DiffractionResult) -> SimulationResult:
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
