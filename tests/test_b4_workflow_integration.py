from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from autotem import TEMAnalysisSkill
from autotem.core import AnalysisReport, Measurement, SimulatedPattern, SimulatedSpot, read_json
from autotem.diffraction.indexing import B4IndexingResult, run_b4_indexing
from autotem.diffraction.phase_lookup import PhaseLookupResult
from autotem.simulation.diffsims_adapter import DiffsimsSimulationResult


def _write_b4_workflow(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: b4_lattice_fit_integration",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    calibration:",
                "      method: reference",
                "      known_d_spacing: 0.4",
                "      known_d_spacing_unit: nm",
                "      measured_radius_px: 10",
                "    preprocessing:",
                "      recipe: min_max",
                "      background_method: none",
                "    diffraction:",
                "      top_k_peaks: 12",
                "      min_peak_distance: 3",
                "      center_exclusion_radius: 4",
                "      peak_threshold_strategy: relative_max",
                "      peak_threshold_value: 0.12",
                "      candidate_phases: [cubic_probe]",
                "      candidate_phase_params:",
                "        - name: cubic_probe",
                "          a: 0.4",
                "          b: 0.4",
                "          c: 0.4",
                "          alpha: 90",
                "          beta: 90",
                "          gamma: 90",
                "          space_group: Pm-3m",
                "      indexing_mode: lattice_fit",
                "      phase_top_n: 1",
                "      max_hkl: 3",
                "      d_tolerance_rel: 0.08",
                "      angle_tolerance_deg: 10.0",
                "outputs:",
                "  formats: [json, csv]",
                "  save_intermediates: true",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_b4_lookup_workflow(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: b4_lookup_integration",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    calibration:",
                "      method: reference",
                "      known_d_spacing: 0.4",
                "      known_d_spacing_unit: nm",
                "      measured_radius_px: 10",
                "    preprocessing:",
                "      recipe: min_max",
                "      background_method: none",
                "    diffraction:",
                "      top_k_peaks: 12",
                "      min_peak_distance: 3",
                "      center_exclusion_radius: 4",
                "      peak_threshold_strategy: relative_max",
                "      peak_threshold_value: 0.12",
                "      candidate_phases: [lookup_cubic]",
                "      phase_lookup_enabled: true",
                "      phase_lookup_materials: [lookup_cubic]",
                "      phase_lookup_sources: [local_builtin]",
                "      phase_lookup_limit_per_source: 1",
                "      phase_lookup_timeout_s: 1",
                "      indexing_mode: lattice_fit",
                "      phase_top_n: 1",
                "      max_hkl: 3",
                "      d_tolerance_rel: 0.08",
                "      angle_tolerance_deg: 10.0",
                "outputs:",
                "  formats: [json, csv]",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_b4_unified_lookup_workflow(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: b4_unified_lookup_integration",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, report]",
                "  parameters:",
                "    calibration:",
                "      method: reference",
                "      known_d_spacing: 0.4",
                "      known_d_spacing_unit: nm",
                "      measured_radius_px: 10",
                "    preprocessing:",
                "      recipe: min_max",
                "      background_method: none",
                "    diffraction:",
                "      top_k_peaks: 12",
                "      min_peak_distance: 3",
                "      center_exclusion_radius: 4",
                "      peak_threshold_strategy: relative_max",
                "      peak_threshold_value: 0.12",
                "      phase_candidate_materials: [lookup_cubic]",
                "      phase_lookup_sources: [local_builtin]",
                "      phase_lookup_limit_per_source: 1",
                "      phase_lookup_timeout_s: 1",
                "      indexing_mode: lattice_fit",
                "      phase_top_n: 1",
                "      max_hkl: 3",
                "      d_tolerance_rel: 0.08",
                "      angle_tolerance_deg: 10.0",
                "outputs:",
                "  formats: [json, csv]",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_real_diffsims_workflow(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: real_diffsims_closed_loop",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, simulate, report]",
                "  parameters:",
                "    calibration:",
                "      method: reference",
                "      known_d_spacing: 0.4",
                "      known_d_spacing_unit: nm",
                "      measured_radius_px: 10",
                "    preprocessing:",
                "      recipe: min_max",
                "      background_method: none",
                "    diffraction:",
                "      top_k_peaks: 12",
                "      min_peak_distance: 3",
                "      center_exclusion_radius: 4",
                "      peak_threshold_strategy: relative_max",
                "      peak_threshold_value: 0.12",
                "      candidate_phases: [cubic_probe]",
                "      candidate_phase_params:",
                "        - name: cubic_probe",
                "          formula: X",
                "          a: 0.4",
                "          b: 0.4",
                "          c: 0.4",
                "          alpha: 90",
                "          beta: 90",
                "          gamma: 90",
                "          space_group: Pm-3m",
                "          space_group_number: 221",
                "          atom_sites:",
                "            - element: Si",
                "              abc: [0, 0, 0]",
                "              occupancy: 1.0",
                "      indexing_mode: lattice_fit",
                "      phase_top_n: 1",
                "      max_hkl: 3",
                "      d_tolerance_rel: 0.08",
                "      angle_tolerance_deg: 10.0",
                "    simulation:",
                "      simulation_engine: diffsims",
                "      voltage_kv: 200",
                "      reciprocal_radius_inv_angstrom: 1.5",
                "      max_candidates: 1",
                "      rotation_axis: [0, 0, 1]",
                "      rotation_angles_deg: [0]",
                "      max_simulated_spots: 128",
                "      minimum_simulated_g_inv_nm: 0.001",
                "      phase_match_enabled: true",
                "      phase_match_d_tolerance_rel: 0.03",
                "      phase_match_angle_tolerance_deg: 3.0",
                "      phase_match_min_spots: 3",
                "      phase_match_previous_score_weight: 0.0",
                "      pattern_shape: [64, 64]",
                "outputs:",
                "  formats: [json, csv]",
                "  save_intermediates: true",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_mp_lookup_diffsims_workflow(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: mp_lookup_diffsims_closed_loop",
                "  steps: [load, assess_quality, calibrate, preprocess, analyze, simulate, report]",
                "  parameters:",
                "    calibration:",
                "      method: reference",
                "      known_d_spacing: 0.4",
                "      known_d_spacing_unit: nm",
                "      measured_radius_px: 10",
                "    preprocessing:",
                "      recipe: min_max",
                "      background_method: none",
                "    diffraction:",
                "      top_k_peaks: 12",
                "      min_peak_distance: 3",
                "      center_exclusion_radius: 4",
                "      peak_threshold_strategy: relative_max",
                "      peak_threshold_value: 0.12",
                "      candidate_phases: [mp-149]",
                "      phase_lookup_enabled: true",
                "      phase_lookup_materials: [mp-149]",
                "      phase_lookup_sources: [materials_project_api]",
                "      phase_lookup_limit_per_source: 1",
                "      phase_lookup_timeout_s: 1",
                "      materials_project_api_isolate: true",
                "      indexing_mode: lattice_fit",
                "      phase_top_n: 1",
                "      max_hkl: 3",
                "      d_tolerance_rel: 0.08",
                "      angle_tolerance_deg: 10.0",
                "    simulation:",
                "      simulation_engine: diffsims",
                "      max_candidates: 1",
                "      max_simulated_spots: 64",
                "      phase_match_enabled: true",
                "      phase_match_d_tolerance_rel: 0.03",
                "      phase_match_angle_tolerance_deg: 3.0",
                "      phase_match_min_spots: 3",
                "      phase_match_previous_score_weight: 0.0",
                "outputs:",
                "  formats: [json, csv]",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _synthetic_cubic_fft_image(size: int = 80) -> np.ndarray:
    spectrum = np.zeros((size, size), dtype=np.complex128)
    cy = size // 2
    cx = size // 2
    spectrum[cy, cx] = 2200.0
    for dy, dx, amp in [
        (0, 10, 320.0),
        (0, -10, 320.0),
        (10, 0, 300.0),
        (-10, 0, 300.0),
        (10, 10, 200.0),
        (-10, -10, 200.0),
    ]:
        spectrum[cy + dy, cx + dx] = amp
    image = np.fft.ifft2(np.fft.ifftshift(spectrum)).real
    image = image - float(image.min())
    image = image / max(float(image.max()), 1.0)
    return image.astype(np.float32)


def test_b4_public_import_and_workflow_lattice_fit_outputs_hkl_table(
    tmp_path: Path,
) -> None:
    assert B4IndexingResult is not None
    assert callable(run_b4_indexing)

    input_path = tmp_path / "b4_input.npy"
    np.save(input_path, _synthetic_cubic_fft_image())
    workflow_path = _write_b4_workflow(tmp_path / "b4_workflow.yaml")
    output_dir = tmp_path / "outputs"

    run_result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    report = read_json(Path(run_result["report_path"]), AnalysisReport)

    assert report.diffraction is not None
    assert report.diffraction.phase_candidates
    assert report.diffraction.phase_candidates[0].name == "cubic_probe"
    assert any(
        (spot.h, spot.k, spot.l) != (0, 0, 0)
        for spot in report.diffraction.indexed_spots
    )
    b4_provenance = report.diffraction.provenance.get("b4_indexing", {})
    assert b4_provenance.get("active") is True
    assert b4_provenance.get("selected_phase") == "cubic_probe"

    indexed_path = output_dir / "tables" / "indexed_spots.csv"
    assert indexed_path.exists()
    indexed_csv = indexed_path.read_text(encoding="utf-8")
    assert "peak_index,h,k,l,d_spacing" in indexed_csv
    assert "cubic_probe" in (output_dir / "tables" / "phase_candidates.csv").read_text(
        encoding="utf-8"
    )


@pytest.mark.real_diffsims
@pytest.mark.skipif(
    os.environ.get("AUTOTEM_RUN_REAL_DIFFSIMS_TESTS") != "1",
    reason=(
        "Real diffsims end-to-end smoke is opt-in because diffsims/orix/numba "
        "initialization can be slow or hang in ordinary pytest runs."
    ),
)
def test_real_diffsims_workflow_closes_simulation_phase_matching_loop(
    tmp_path: Path,
) -> None:
    pytest.importorskip("diffsims")
    pytest.importorskip("diffpy.structure")
    pytest.importorskip("orix")

    input_path = tmp_path / "real_diffsims_input.npy"
    np.save(input_path, _synthetic_cubic_fft_image())
    workflow_path = _write_real_diffsims_workflow(tmp_path / "real_diffsims.yaml")
    output_dir = tmp_path / "real_diffsims_outputs"

    run_result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    report = read_json(Path(run_result["report_path"]), AnalysisReport)

    assert report.diffraction is not None
    assert report.simulation is not None
    assert report.simulation.patterns
    assert report.simulation.patterns[0].spots
    assert report.simulation.provenance["adapter"]["active"] is True
    assert report.simulation.provenance["phase_matching"]["active"] is True
    assert report.simulation.provenance["phase_matching"]["selected_phase"] == "cubic_probe"
    assert report.simulation.matched_phase_candidates[0].matched_spot_count == 6
    assert (report.simulation.matched_phase_candidates[0].score or 0.0) > 0.95
    assert report.diffraction.phase_candidates[0].mean_d_error is not None
    assert report.diffraction.phase_candidates[0].mean_angle_error is not None

    simulated_spots_path = output_dir / "tables" / "simulated_spots.csv"
    simulation_matches_path = output_dir / "tables" / "simulation_phase_matches.csv"
    phase_candidates_path = output_dir / "tables" / "phase_candidates.csv"
    assert simulated_spots_path.exists()
    assert simulation_matches_path.exists()
    assert phase_candidates_path.exists()
    assert "rotation_angle_deg" in simulated_spots_path.read_text(encoding="utf-8")
    assert "cubic_probe" in simulation_matches_path.read_text(encoding="utf-8")
    assert "diffsims_phase_match" in phase_candidates_path.read_text(encoding="utf-8")


def test_b4_workflow_uses_material_lookup_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_lookup_phase_candidates(**kwargs) -> PhaseLookupResult:
        calls.append(dict(kwargs))
        return PhaseLookupResult(
            candidates=[
                {
                    "name": "lookup_cubic",
                    "formula": "X",
                    "a": 0.4,
                    "b": 0.4,
                    "c": 0.4,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                    "space_group": "Pm-3m",
                    "source": "test_lookup",
                    "identifier": "lookup_cubic",
                }
            ],
            warnings=[],
            provenance={"sources": ["test_lookup"], "material_names": ["lookup_cubic"]},
        )

    monkeypatch.setattr(
        "autotem.diffraction.basic.lookup_phase_candidates",
        fake_lookup_phase_candidates,
    )

    input_path = tmp_path / "b4_lookup_input.npy"
    np.save(input_path, _synthetic_cubic_fft_image())
    workflow_path = _write_b4_lookup_workflow(tmp_path / "b4_lookup_workflow.yaml")
    output_dir = tmp_path / "outputs_lookup"

    run_result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    report = read_json(Path(run_result["report_path"]), AnalysisReport)

    assert calls
    assert calls[0]["material_names"] == ["lookup_cubic"]
    assert report.diffraction is not None
    assert report.diffraction.phase_candidates[0].name == "lookup_cubic"
    assert report.diffraction.provenance["b4_phase_lookup"]["candidate_count"] == 1
    assert any(
        (spot.h, spot.k, spot.l) != (0, 0, 0)
        for spot in report.diffraction.indexed_spots
    )


def test_b4_workflow_uses_unified_phase_candidate_materials_for_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_lookup_phase_candidates(**kwargs) -> PhaseLookupResult:
        calls.append(dict(kwargs))
        return PhaseLookupResult(
            candidates=[
                {
                    "name": "lookup_cubic",
                    "formula": "X",
                    "a": 0.4,
                    "b": 0.4,
                    "c": 0.4,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                    "space_group": "Pm-3m",
                    "source": "test_lookup",
                    "identifier": "lookup_cubic",
                }
            ],
            warnings=[],
            provenance={"sources": ["test_lookup"], "material_names": ["lookup_cubic"]},
        )

    monkeypatch.setattr(
        "autotem.diffraction.basic.lookup_phase_candidates",
        fake_lookup_phase_candidates,
    )

    input_path = tmp_path / "b4_unified_lookup_input.npy"
    np.save(input_path, _synthetic_cubic_fft_image())
    workflow_path = _write_b4_unified_lookup_workflow(
        tmp_path / "b4_unified_lookup_workflow.yaml"
    )
    output_dir = tmp_path / "outputs_unified_lookup"

    run_result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    report = read_json(Path(run_result["report_path"]), AnalysisReport)

    assert calls
    assert calls[0]["material_names"] == ["lookup_cubic"]
    assert report.diffraction is not None
    assert report.diffraction.phase_candidates[0].name == "lookup_cubic"
    assert (
        report.diffraction.provenance["b4_phase_materials"]["source"]
        == "phase_candidate_materials"
    )
    assert (
        report.diffraction.provenance["b4_phase_materials"]["lookup_materials"]
        == ["lookup_cubic"]
    )


def test_mp_lookup_candidate_flows_into_diffsims_phase_matching(
    tmp_path: Path,
    monkeypatch,
) -> None:
    lookup_calls: list[dict[str, object]] = []

    def fake_lookup_phase_candidates(**kwargs) -> PhaseLookupResult:
        lookup_calls.append(dict(kwargs))
        return PhaseLookupResult(
            candidates=[
                {
                    "name": "mp-149",
                    "formula": "Si",
                    "a": 0.4,
                    "b": 0.4,
                    "c": 0.4,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                    "space_group": "Pm-3m",
                    "space_group_number": 221,
                    "source": "materials_project_api",
                    "identifier": "mp-149",
                    "atom_sites": [
                        {"element": "Si", "abc": [0, 0, 0], "occupancy": 1.0}
                    ],
                    "materials_project": {
                        "rank": 0,
                        "energy_above_hull": 0.0,
                        "is_stable": True,
                        "site_count": 1,
                    },
                }
            ],
            warnings=[],
            provenance={
                "sources": ["materials_project_api"],
                "material_names": ["mp-149"],
                "resolved_material_queries": [
                    {
                        "material": "mp-149",
                        "query_type": "materials_project_id",
                        "value": "mp-149",
                    }
                ],
            },
        )

    def fake_simulate_diffraction_with_diffsims(candidates, **kwargs):
        candidate = candidates[0]
        spots = []
        for gx, gy, intensity in [
            (2.5, 0.0, 100.0),
            (-2.5, 0.0, 98.0),
            (0.0, 2.5, 96.0),
            (0.0, -2.5, 94.0),
            (2.5, 2.5, 80.0),
            (-2.5, -2.5, 78.0),
        ]:
            magnitude = float(np.hypot(gx, gy))
            spots.append(
                SimulatedSpot(
                    gx=Measurement(value=gx, unit="1/nm"),
                    gy=Measurement(value=gy, unit="1/nm"),
                    magnitude=Measurement(value=magnitude, unit="1/nm"),
                    d_spacing=Measurement(value=1.0 / magnitude, unit="nm"),
                    intensity=intensity,
                )
            )
        return DiffsimsSimulationResult(
            patterns=[
                SimulatedPattern(
                    kind="diffraction",
                    structure_id=candidate.identifier or candidate.name,
                    zone_axis=(0, 0, 1),
                    simulation_parameters={
                        "engine": "diffsims",
                        "simulated_spot_count": len(spots),
                    },
                    spots=spots,
                )
            ],
            warnings=[],
            provenance={"adapter": "diffsims", "active": True, "pattern_count": 1},
        )

    monkeypatch.setattr(
        "autotem.diffraction.basic.lookup_phase_candidates",
        fake_lookup_phase_candidates,
    )
    monkeypatch.setattr(
        "autotem.simulation.basic.simulate_diffraction_with_diffsims",
        fake_simulate_diffraction_with_diffsims,
    )
    # Scope the fake adapter to this process. The separate worker is covered by real_diffsims.
    monkeypatch.setattr("autotem.simulation.basic._diffsims_timeout_s", lambda parameters: None)

    input_path = tmp_path / "mp_lookup_diffsims_input.npy"
    np.save(input_path, _synthetic_cubic_fft_image())
    workflow_path = _write_mp_lookup_diffsims_workflow(
        tmp_path / "mp_lookup_diffsims.yaml"
    )
    output_dir = tmp_path / "mp_lookup_diffsims_outputs"

    run_result = TEMAnalysisSkill().run(
        input_path=input_path,
        workflow_path=workflow_path,
        output_dir=output_dir,
    )
    report = read_json(Path(run_result["report_path"]), AnalysisReport)

    assert lookup_calls
    assert lookup_calls[0]["sources"] == ["materials_project_api"]
    assert lookup_calls[0]["materials_project_api_isolate"] is True
    assert report.diffraction is not None
    assert report.simulation is not None
    assert report.diffraction.phase_candidates[0].source == "materials_project_api"
    assert (
        report.diffraction.phase_candidates[0]
        .lattice_parameters["materials_project"]["is_stable"]
        is True
    )
    assert report.simulation.provenance["phase_matching"]["active"] is True
    assert report.simulation.matched_phase_candidates[0].identifier == "mp-149"
    assert report.simulation.matched_phase_candidates[0].matched_spot_count == 6
    matches_csv = (output_dir / "tables" / "simulation_phase_matches.csv").read_text(
        encoding="utf-8"
    )
    assert "materials_project_api" in matches_csv
    assert "mp-149" in matches_csv
    phase_csv = (output_dir / "tables" / "phase_candidates.csv").read_text(
        encoding="utf-8"
    )
    assert "mp_energy_above_hull" in phase_csv
    assert "0.0" in phase_csv
