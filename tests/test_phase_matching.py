from __future__ import annotations

import numpy as np
import pytest

from autotem.core import (
    DiffractionResult,
    GVector,
    Measurement,
    PhaseCandidate,
    SimulatedPattern,
    SimulatedSpot,
)
from autotem.diffraction.phase_matching import match_diffsims_patterns_to_diffraction


def _physical_inputs():
    coordinates = [(2.5, 0.0), (0.0, 2.5), (2.5, 2.5)]
    vectors = [
        GVector(
            gx=Measurement(value=gx, unit="1/nm"),
            gy=Measurement(value=gy, unit="1/nm"),
            magnitude=Measurement(value=float(np.hypot(gx, gy)), unit="1/nm"),
        )
        for gx, gy in coordinates
    ]
    spots = [
        SimulatedSpot(
            gx=Measurement(value=gx, unit="1/nm"),
            gy=Measurement(value=gy, unit="1/nm"),
            magnitude=Measurement(value=float(np.hypot(gx, gy)), unit="1/nm"),
            d_spacing=Measurement(value=1.0 / float(np.hypot(gx, gy)), unit="nm"),
        )
        for gx, gy in coordinates
    ]
    diffraction = DiffractionResult(
        g_vectors=vectors, phase_candidates=[PhaseCandidate(name="cubic", score=0.1)]
    )
    patterns = [SimulatedPattern(kind="diffraction", structure_id="cubic", spots=spots)]
    return diffraction, patterns


@pytest.mark.parametrize("unit", ["1/pixel", "unknown", "rad/nm"])
def test_phase_matching_rejects_uncalibrated_measured_vectors(unit) -> None:
    diffraction, patterns = _physical_inputs()
    for vector in diffraction.g_vectors:
        vector.gx.unit = unit
        vector.gy.unit = unit
    result = match_diffsims_patterns_to_diffraction(diffraction, patterns)

    assert result.provenance["active"] is False
    assert result.provenance["reason"] == "no_valid_physical_measurements"
    assert result.candidates[0].name == diffraction.phase_candidates[0].name
    assert result.candidates[0].score is None
    assert not result.comparison_metrics
    assert result.warnings


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_phase_matching_rejects_invalid_measured_magnitudes(value) -> None:
    diffraction, patterns = _physical_inputs()
    for vector in diffraction.g_vectors:
        vector.magnitude.value = value
    result = match_diffsims_patterns_to_diffraction(diffraction, patterns)

    assert result.provenance["active"] is False
    assert not result.comparison_metrics


@pytest.mark.parametrize(
    ("value", "unit"),
    [(0.4, "pixel"), (0.4, "unknown"), (float("nan"), "nm"),
     (float("inf"), "nm"), (0.0, "nm"), (-0.4, "nm")],
)
def test_phase_matching_rejects_invalid_simulated_d_spacings(value, unit) -> None:
    diffraction, patterns = _physical_inputs()
    for spot in patterns[0].spots:
        spot.d_spacing = Measurement(value=value, unit=unit)
    result = match_diffsims_patterns_to_diffraction(diffraction, patterns)

    assert result.provenance["active"] is False
    assert result.provenance["reason"] == "no_valid_physical_simulated_spots"
    assert result.candidates[0].name == diffraction.phase_candidates[0].name
    assert result.candidates[0].score is None
    assert not result.comparison_metrics


@pytest.mark.parametrize("convert_templates", [False, True])
@pytest.mark.parametrize(
    ("length_unit", "reciprocal_unit", "nm_per_unit"),
    [("angstrom", "1/angstrom", 0.1), ("Å", "Å^-1", 0.1),
     ("pm", "pm^-1", 0.001), ("um", "cycles/um", 1000.0)],
)
def test_phase_matching_normalizes_both_input_sides(
    convert_templates, length_unit, reciprocal_unit, nm_per_unit
) -> None:
    diffraction, patterns = _physical_inputs()
    baseline = match_diffsims_patterns_to_diffraction(diffraction, patterns)
    records = patterns[0].spots if convert_templates else diffraction.g_vectors
    for record in records:
        for measurement in (record.gx, record.gy, record.magnitude):
            measurement.value *= nm_per_unit
            measurement.unit = reciprocal_unit
        if convert_templates:
            record.d_spacing.value /= nm_per_unit
            record.d_spacing.unit = length_unit
    result = match_diffsims_patterns_to_diffraction(diffraction, patterns)

    assert result.provenance["active"] is True
    assert result.candidates[0].matched_spot_count == 3
    assert result.candidates[0].score == pytest.approx(baseline.candidates[0].score)
    assert result.candidates[0].mean_d_error.value == pytest.approx(0.0, abs=1e-12)


def test_phase_matching_excludes_only_invalid_spots() -> None:
    diffraction, patterns = _physical_inputs()
    diffraction.g_vectors[0].gx = Measurement(value=float("nan"), unit="1/nm")
    patterns[0].spots[0].gx.unit = "1/pixel"
    result = match_diffsims_patterns_to_diffraction(diffraction, patterns)

    assert result.provenance["active"] is True
    assert result.candidates[0].matched_spot_count == 2
    assert result.provenance["rejected_measured_spot_count"] == 1
    assert result.provenance["rejected_simulated_spot_count"] == 1
    assert result.warnings


@pytest.mark.parametrize("unit", ["1/pixel", "unknown", "rad/nm"])
def test_phase_matching_rejects_nonphysical_simulated_vectors(unit) -> None:
    diffraction, patterns = _physical_inputs()
    for spot in patterns[0].spots:
        spot.gx.unit = unit
        spot.gy.unit = unit
    result = match_diffsims_patterns_to_diffraction(diffraction, patterns)

    assert result.provenance["active"] is False
    assert not result.comparison_metrics


def test_phase_matching_rejects_zero_vectors_with_positive_magnitude() -> None:
    diffraction, patterns = _physical_inputs()
    for vector in diffraction.g_vectors:
        vector.gx.value = vector.gy.value = 0.0
    result = match_diffsims_patterns_to_diffraction(diffraction, patterns)

    assert result.provenance["active"] is False
    assert not result.comparison_metrics


def test_phase_matching_invalid_inputs_withhold_previous_match_scores() -> None:
    diffraction, patterns = _physical_inputs()
    diffraction.phase_candidates[0].score = 0.99
    diffraction.phase_candidates[0].matched_spot_count = 20
    patterns[0].score = Measurement(value=0.99, unit="confidence")
    for vector in diffraction.g_vectors:
        vector.gx.unit = "1/pixel"
    result = match_diffsims_patterns_to_diffraction(diffraction, patterns)

    assert diffraction.phase_candidates[0].score == 0.99
    assert patterns[0].score.value == 0.99
    assert result.candidates[0].score is None
    assert result.candidates[0].matched_spot_count is None
    assert result.candidates[0].warnings
    assert result.patterns[0].score is None
    assert result.provenance["candidate_validation"] == "not_evaluated"
    assert result.provenance["withheld_candidate_scores"] == [{"name": "cubic", "score": 0.99}]
    assert not result.comparison_metrics
