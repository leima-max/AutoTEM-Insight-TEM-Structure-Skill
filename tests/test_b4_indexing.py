from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autotem.core import DiffractionPeak, GVector, IndexedSpot, Measurement
from autotem.diffraction.b4_indexing import run_b4_indexing


def _build_b4_inputs() -> tuple[list[DiffractionPeak], list[GVector], list[Measurement | None]]:
    # Reciprocal vectors for a cubic lattice with a = 0.4 nm:
    # (100), (010), (110), (-110)
    gxy = [
        (2.5, 0.0),
        (0.0, 2.5),
        (2.5, 2.5),
        (-2.5, 2.5),
    ]
    intensities = [100.0, 96.0, 80.0, 72.0]
    peaks = [
        DiffractionPeak(x=0.0, y=0.0, intensity=intensities[idx], confidence=0.8)
        for idx in range(len(gxy))
    ]
    g_vectors: list[GVector] = []
    d_by_peak: list[Measurement | None] = []
    for gx, gy in gxy:
        g_mag = float(np.hypot(gx, gy))
        g_vectors.append(
            GVector(
                gx=Measurement(value=gx, unit="1/nm", method="test"),
                gy=Measurement(value=gy, unit="1/nm", method="test"),
                magnitude=Measurement(value=g_mag, unit="1/nm", method="test"),
            )
        )
        d_by_peak.append(Measurement(value=1.0 / g_mag, unit="nm", method="test"))
    return peaks, g_vectors, d_by_peak


def test_b4_mvp_prefers_closest_lattice_candidate() -> None:
    peaks, g_vectors, d_by_peak = _build_b4_inputs()
    initial = [IndexedSpot(peak_index=i, h=0, k=0, l=0, confidence=0.1) for i in range(len(peaks))]
    result = run_b4_indexing(
        peaks=peaks,
        g_vectors=g_vectors,
        d_by_peak=d_by_peak,
        initial_indexed_spots=initial,
        candidate_phase_params=[
            {
                "name": "cubic_good",
                "a": 0.4,
                "b": 0.4,
                "c": 0.4,
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
                "space_group": "Pm-3m",
            },
            {
                "name": "cubic_bad",
                "a": 0.52,
                "b": 0.52,
                "c": 0.52,
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
                "space_group": "Pm-3m",
            },
        ],
        allowed_phase_names=None,
        max_hkl=3,
        d_tolerance_rel=0.08,
        angle_tolerance_deg=10.0,
        strict_mode=False,
        top_n=2,
    )

    assert result.phase_candidates
    assert result.phase_candidates[0].name == "cubic_good"
    assert (result.phase_candidates[0].matched_spot_count or 0) >= 3
    assert result.indexed_spots
    assert any((spot.h, spot.k, spot.l) != (0, 0, 0) for spot in result.indexed_spots)
    assert any((spot.confidence or 0.0) > 0.5 for spot in result.indexed_spots)
    assert bool(result.provenance.get("active"))


def test_b4_allowed_phase_filter_does_not_match_single_element_substrings() -> None:
    peaks, g_vectors, d_by_peak = _build_b4_inputs()
    initial = [IndexedSpot(peak_index=i, h=0, k=0, l=0, confidence=0.1) for i in range(len(peaks))]

    result = run_b4_indexing(
        peaks=peaks,
        g_vectors=g_vectors,
        d_by_peak=d_by_peak,
        initial_indexed_spots=initial,
        candidate_phase_params=[
            {
                "name": "Graphite",
                "formula": "C",
                "a": 0.246,
                "b": 0.246,
                "c": 0.6708,
                "alpha": 90,
                "beta": 90,
                "gamma": 120,
                "space_group": "P63/mmc",
            }
        ],
        allowed_phase_names={"CsPbBr3"},
        max_hkl=3,
        d_tolerance_rel=0.08,
        angle_tolerance_deg=10.0,
        strict_mode=False,
        top_n=1,
    )

    assert not result.phase_candidates
    assert result.provenance["reason"] == "no_phase_candidates"


def test_b4_allowed_phase_filter_accepts_equivalent_formula_order() -> None:
    peaks, g_vectors, d_by_peak = _build_b4_inputs()
    initial = [IndexedSpot(peak_index=i, h=0, k=0, l=0, confidence=0.1) for i in range(len(peaks))]

    result = run_b4_indexing(
        peaks=peaks,
        g_vectors=g_vectors,
        d_by_peak=d_by_peak,
        initial_indexed_spots=initial,
        candidate_phase_params=[
            {
                "name": "mp-cspbbr3-like",
                "formula": "Br3CsPb",
                "a": 0.4,
                "b": 0.4,
                "c": 0.4,
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
                "space_group": "Pm-3m",
                "source": "materials_project_optimade",
            },
            {
                "name": "cod-cspbbr3-like",
                "formula": "- Br3 Cs Pb -",
                "a": 0.4,
                "b": 0.4,
                "c": 0.4,
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
                "space_group": "Pm-3m",
                "source": "crystallography_open_database",
            },
        ],
        allowed_phase_names={"CsPbBr3"},
        max_hkl=3,
        d_tolerance_rel=0.08,
        angle_tolerance_deg=10.0,
        strict_mode=False,
        top_n=2,
    )

    assert [candidate.source for candidate in result.phase_candidates] == [
        "materials_project_optimade",
        "crystallography_open_database",
    ]


def test_b4_strict_reads_cif_candidates_and_reports_checks(tmp_path: Path) -> None:
    peaks, g_vectors, d_by_peak = _build_b4_inputs()
    initial = [IndexedSpot(peak_index=i, h=0, k=0, l=0, confidence=0.1) for i in range(len(peaks))]

    cif_dir = tmp_path / "cif_candidates"
    cif_dir.mkdir(parents=True, exist_ok=True)
    (cif_dir / "phase_cubic.cif").write_text(
        "\n".join(
            [
                "_cell_length_a 0.4",
                "_cell_length_b 0.4",
                "_cell_length_c 0.4",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 90",
                "_symmetry_space_group_name_H-M 'Pm-3m'",
                "_chemical_formula_sum 'X'",
            ]
        ),
        encoding="utf-8",
    )

    result = run_b4_indexing(
        peaks=peaks,
        g_vectors=g_vectors,
        d_by_peak=d_by_peak,
        initial_indexed_spots=initial,
        candidate_phase_params=[],
        allowed_phase_names={"phase_cubic"},
        max_hkl=3,
        d_tolerance_rel=0.1,
        angle_tolerance_deg=15.0,
        strict_mode=True,
        cif_candidates_dir=str(cif_dir),
        top_n=1,
    )

    assert result.phase_candidates
    top = result.phase_candidates[0]
    assert top.name == "phase_cubic"
    assert top.source == "cif_candidates"
    assert any("friedel_consistency=" in text for text in top.warnings)
    assert any("zone_axis_consistency=" in text for text in top.warnings)
    assert bool(result.provenance.get("active"))
    assert result.provenance.get("mode") == "b4_strict"


def test_b4_reads_direct_cif_candidate_paths_with_standard_angstrom_cell(
    tmp_path: Path,
) -> None:
    peaks, g_vectors, d_by_peak = _build_b4_inputs()
    initial = [IndexedSpot(peak_index=i, h=0, k=0, l=0, confidence=0.1) for i in range(len(peaks))]

    cif_path = tmp_path / "phase_cubic_angstrom.cif"
    cif_path.write_text(
        "\n".join(
            [
                "_cell_length_a 4.0",
                "_cell_length_b 4.0",
                "_cell_length_c 4.0",
                "_cell_angle_alpha 90",
                "_cell_angle_beta 90",
                "_cell_angle_gamma 90",
                "_space_group_name_H-M_alt 'Pm-3m'",
                "_space_group_IT_number 221",
                "_chemical_formula_sum 'X'",
                "loop_",
                "_atom_site_label",
                "_atom_site_type_symbol",
                "_atom_site_fract_x",
                "_atom_site_fract_y",
                "_atom_site_fract_z",
                "X1 X 0 0 0",
            ]
        ),
        encoding="utf-8",
    )

    result = run_b4_indexing(
        peaks=peaks,
        g_vectors=g_vectors,
        d_by_peak=d_by_peak,
        initial_indexed_spots=initial,
        candidate_phase_params=[],
        allowed_phase_names={"phase_cubic_angstrom"},
        max_hkl=3,
        d_tolerance_rel=0.08,
        angle_tolerance_deg=10.0,
        strict_mode=False,
        cif_candidate_paths=[str(cif_path)],
        top_n=1,
    )

    assert result.phase_candidates
    assert result.phase_candidates[0].source == "cif_candidates"
    assert result.phase_candidates[0].name == "phase_cubic_angstrom"
    assert result.phase_candidates[0].lattice_parameters["space_group_number"] == 221
    assert result.phase_candidates[0].lattice_parameters["atom_sites"][0]["element"] == "X"
    assert any((spot.h, spot.k, spot.l) != (0, 0, 0) for spot in result.indexed_spots)
    assert result.provenance.get("active") is True


def _run_physical_unit_case(
    g_vectors: list[GVector], d_by_peak: list[Measurement | None]
):
    peaks, _, _ = _build_b4_inputs()
    initial = [IndexedSpot(peak_index=i, h=0, k=0, l=0) for i in range(len(peaks))]
    result = run_b4_indexing(
        peaks=peaks,
        g_vectors=g_vectors,
        d_by_peak=d_by_peak,
        initial_indexed_spots=initial,
        candidate_phase_params=[{"name": "cubic", "a": 0.4, "b": 0.4, "c": 0.4}],
        allowed_phase_names=None,
        max_hkl=3,
        d_tolerance_rel=0.08,
        angle_tolerance_deg=10.0,
        strict_mode=False,
    )
    return result, initial


@pytest.mark.parametrize(
    ("value", "unit"),
    [(0.4, "pixel"), (0.4, "unknown"), (float("nan"), "nm"),
     (float("inf"), "nm"), (0.0, "nm"), (-0.4, "nm")],
)
def test_b4_rejects_uncalibrated_or_invalid_d_spacings(value, unit) -> None:
    _, vectors, distances = _build_b4_inputs()
    distances = [Measurement(value=value, unit=unit) for _ in distances]
    result, initial = _run_physical_unit_case(vectors, distances)

    assert result.indexed_spots == initial
    assert not result.phase_candidates
    assert result.provenance["active"] is False
    assert result.provenance["reason"] == "no_valid_physical_measurements"
    assert len(result.provenance["diagnostics"]["rejected_measurements"]) == 4
    assert result.warnings


@pytest.mark.parametrize("unit", ["1/pixel", "unknown", "rad/nm"])
def test_b4_rejects_g_vectors_without_physical_frequency_units(unit) -> None:
    _, vectors, distances = _build_b4_inputs()
    for vector in vectors:
        vector.gx.unit = unit
        vector.gy.unit = unit
    result, initial = _run_physical_unit_case(vectors, distances)

    assert result.indexed_spots == initial
    assert result.provenance["active"] is False
    assert not result.phase_candidates


@pytest.mark.parametrize(
    ("length_unit", "reciprocal_unit", "nm_per_unit"),
    [("angstrom", "1/angstrom", 0.1), ("Å", "Å^-1", 0.1),
     ("pm", "1/pm", 0.001), ("um", "um^{-1}", 1000.0)],
)
def test_b4_normalizes_physical_units_before_matching(
    length_unit, reciprocal_unit, nm_per_unit
) -> None:
    _, vectors, distances = _build_b4_inputs()
    baseline, _ = _run_physical_unit_case(vectors, distances)
    for vector in vectors:
        for measurement in (vector.gx, vector.gy, vector.magnitude):
            measurement.value *= nm_per_unit
            measurement.unit = reciprocal_unit
    for distance in distances:
        distance.value /= nm_per_unit
        distance.unit = length_unit
    converted, _ = _run_physical_unit_case(vectors, distances)

    assert converted.provenance["active"] is True
    assert converted.phase_candidates[0].score == pytest.approx(baseline.phase_candidates[0].score)
    assert [(s.h, s.k, s.l) for s in converted.indexed_spots] == [
        (s.h, s.k, s.l) for s in baseline.indexed_spots
    ]


def test_b4_invalid_peak_keeps_original_indices_without_receiving_hkl() -> None:
    _, vectors, distances = _build_b4_inputs()
    distances[1] = Measurement(value=0.4, unit="pixel")
    result, _ = _run_physical_unit_case(vectors, distances)

    assert result.provenance["active"] is True
    rejected = next(spot for spot in result.indexed_spots if spot.peak_index == 1)
    assert (rejected.h, rejected.k, rejected.l) == (0, 0, 0)
    assert rejected.d_spacing is None
    assert result.warnings


@pytest.mark.parametrize("field", ["gx", "gy", "magnitude"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_b4_rejects_nonfinite_g_vectors(field, value) -> None:
    _, vectors, distances = _build_b4_inputs()
    for vector in vectors:
        getattr(vector, field).value = value
    result, initial = _run_physical_unit_case(vectors, distances)

    assert result.indexed_spots == initial
    assert result.provenance["active"] is False
    assert not result.phase_candidates


@pytest.mark.parametrize("has_phases", [False, True])
@pytest.mark.parametrize("has_vectors", [False, True])
def test_b4_invalid_physical_inputs_withhold_preexisting_nonzero_hkl(
    has_phases, has_vectors
) -> None:
    peaks, vectors, distances = _build_b4_inputs()
    distances = [Measurement(value=0.4, unit="pixel") for _ in distances]
    initial = [IndexedSpot(peak_index=0, h=1, k=0, l=0, confidence=0.95)]
    result = run_b4_indexing(
        peaks=peaks, g_vectors=vectors if has_vectors else [], d_by_peak=distances,
        initial_indexed_spots=initial,
        candidate_phase_params=(
            [{"name": "cubic", "a": 0.4, "b": 0.4, "c": 0.4}] if has_phases else []
        ),
        allowed_phase_names=None, max_hkl=3, d_tolerance_rel=0.08,
        angle_tolerance_deg=10.0, strict_mode=False,
    )

    assert (initial[0].h, initial[0].k, initial[0].l) == (1, 0, 0)
    assert result.provenance["active"] is False
    returned_spot = result.indexed_spots[0]
    assert (returned_spot.h, returned_spot.k, returned_spot.l) == (0, 0, 0)
    assert result.indexed_spots[0].confidence == 0.0
    assert result.indexed_spots[0].d_spacing.unit == "pixel"
    assert result.provenance["diagnostics"]["withheld_initial_hkl"] == [
        {"peak_index": 0, "hkl": [1, 0, 0]}
    ]
