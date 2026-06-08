from __future__ import annotations

from pathlib import Path

import numpy as np

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
