from __future__ import annotations

from pathlib import Path

import pytest

from autotem.workflows.recipes import load_workflow_recipe


def _write_workflow(path: Path, body_lines: list[str]) -> Path:
    path.write_text("\n".join(body_lines), encoding="utf-8")
    return path


def test_b2b3_validation_rejects_invalid_denoise_method(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_denoise.yaml",
        [
            "workflow:",
            "  name: bad_denoise",
            "  parameters:",
            "    preprocessing:",
            "      denoise_method: unknown",
        ],
    )
    with pytest.raises(ValueError, match="denoise_method"):
        load_workflow_recipe(workflow)


def test_b2b3_validation_rejects_invalid_background_method(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_background.yaml",
        [
            "workflow:",
            "  name: bad_background",
            "  parameters:",
            "    preprocessing:",
            "      background_method: strange",
        ],
    )
    with pytest.raises(ValueError, match="background_method"):
        load_workflow_recipe(workflow)


def test_b2b3_validation_rejects_invalid_peak_detector(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_detector.yaml",
        [
            "workflow:",
            "  name: bad_detector",
            "  parameters:",
            "    diffraction:",
            "      peak_detector: weird",
        ],
    )
    with pytest.raises(ValueError, match="peak_detector"):
        load_workflow_recipe(workflow)


def test_b2b3_validation_rejects_invalid_peak_adapter_engine(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_peak_adapter_engine.yaml",
        [
            "workflow:",
            "  name: bad_peak_adapter_engine",
            "  parameters:",
            "    diffraction:",
            "      peak_adapter_engine: strange_adapter",
        ],
    )
    with pytest.raises(ValueError, match="peak_adapter_engine"):
        load_workflow_recipe(workflow)


def test_b2b3_validation_rejects_invalid_fft_center_mode(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_center_mode.yaml",
        [
            "workflow:",
            "  name: bad_center_mode",
            "  parameters:",
            "    diffraction:",
            "      fft_center_mode: manual",
        ],
    )
    with pytest.raises(ValueError, match="fft_center_mode"):
        load_workflow_recipe(workflow)


def test_b2b3_validation_rejects_invalid_auto_peak_tuning_type(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_auto_peak_tuning.yaml",
        [
            "workflow:",
            "  name: bad_auto_peak_tuning",
            "  parameters:",
            "    diffraction:",
            "      auto_peak_tuning: maybe",
        ],
    )
    with pytest.raises(ValueError, match="auto_peak_tuning"):
        load_workflow_recipe(workflow)


def test_b4_validation_rejects_invalid_candidate_phase_params(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_b4_candidate.yaml",
        [
            "workflow:",
            "  name: bad_b4_candidate",
            "  parameters:",
            "    diffraction:",
            "      candidate_phase_params:",
            "        - name: bad_phase",
            "          a: 0",
            "          b: 0.4",
            "          c: 0.4",
        ],
    )
    with pytest.raises(ValueError, match="candidate_phase_params"):
        load_workflow_recipe(workflow)


def test_b2b3_validation_rejects_invalid_diffraction_gate_threshold(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_diffraction_gate.yaml",
        [
            "workflow:",
            "  name: bad_diffraction_gate",
            "  parameters:",
            "    reporting:",
            "      diffraction_gates:",
            "        min_peak_count: 0",
        ],
    )
    with pytest.raises(ValueError, match="min_peak_count"):
        load_workflow_recipe(workflow)


def test_b2b3_validation_accepts_extended_preprocess_and_diffraction_params(
    tmp_path: Path,
) -> None:
    workflow = _write_workflow(
        tmp_path / "ok_b2b3.yaml",
        [
            "workflow:",
            "  name: ok_b2b3",
            "  steps: [load, preprocess, analyze, report]",
            "  parameters:",
            "    preprocessing:",
            "      denoise_method: wiener",
            "      wiener_window: 5",
            "      background_method: rolling_ball",
            "      background_opening_size: 21",
            "    diffraction:",
            "      auto_peak_tuning: true",
            "      peak_detector: dog",
            "      peak_adapter_engine: none",
            "      detector_sigma: 1.2",
            "      dog_sigma_ratio: 1.8",
            "      fft_center_mode: auto",
            "      direct_beam_search_radius: 12",
            "      direct_beam_refine_window: 3",
            "      direct_beam_center_uncertainty_px: 0.4",
            "      candidate_phases: [cubic_probe]",
            "      candidate_phase_params:",
            "        - name: cubic_probe",
            "          a: 0.4",
            "          b: 0.4",
            "          c: 0.4",
            "          alpha: 90",
            "          beta: 90",
            "          gamma: 90",
            "      indexing_mode: lattice_fit",
            "      max_hkl: 3",
            "      d_tolerance_rel: 0.08",
            "      angle_tolerance_deg: 10.0",
            "      b4_strict_mode: false",
            "      cif_candidate_paths: [cif_candidates/silicon_demo.cif]",
            "      phase_lookup_enabled: true",
            "      phase_lookup_materials: [Silicon]",
            "      phase_lookup_formula: Si",
            "      phase_lookup_elements: [Si]",
            "      phase_lookup_sources: [local_builtin]",
            "      materials_project_ids: [mp-149]",
            "      materials_project_config_path: mp.json",
            "      materials_project_sort_by: stability",
            "      materials_project_include_deprecated: false",
            "      materials_project_filter_stable: false",
            "      materials_project_max_energy_above_hull: 0.05",
            "      materials_project_prefer_experimental: true",
            "      materials_project_candidate_pool_size: 10",
            "      phase_lookup_limit_per_source: 2",
            "      phase_lookup_timeout_s: 1",
            "    simulation:",
            "      simulation_engine: diffsims",
            "      voltage_kv: 200",
            "      reciprocal_radius_inv_angstrom: 1.0",
            "      with_direct_beam: true",
            "      max_excitation_error: 0.01",
            "      shape_factor_width: 0.02",
            "      zone_axis_candidates:",
            "        - [0, 0, 1]",
            "        - [1, 1, 0]",
            "      rotation_axis: [0, 0, 1]",
            "      rotation_angles_deg: [0, 15]",
            "      max_candidates: 2",
            "      max_simulated_spots: 64",
            "      minimum_simulated_g_inv_nm: 0.001",
            "      fallback_single_atom: true",
            "      pattern_sigma: 8",
            "      pattern_in_plane_angle_deg: 0",
            "      pattern_mirrored: false",
            "      pattern_normalize: true",
            "      pattern_clip_threshold: 1",
            "      phase_match_enabled: true",
            "      phase_match_d_tolerance_rel: 0.08",
            "      phase_match_angle_tolerance_deg: 10.0",
            "      phase_match_min_spots: 2",
            "      phase_match_top_n: 2",
            "      phase_match_previous_score_weight: 0.25",
            "      phase_match_max_rotation_candidates: 32",
            "    reporting:",
            "      diffraction_gates:",
            "        min_peak_count: 4",
            "        min_uncertainty_coverage: 0.8",
            "outputs:",
            "  formats: [json]",
        ],
    )
    recipe = load_workflow_recipe(workflow)
    preprocessing = recipe.workflow.parameters.get("preprocessing", {})
    diffraction = recipe.workflow.parameters.get("diffraction", {})
    assert preprocessing.get("denoise_method") == "wiener"
    assert preprocessing.get("background_method") == "rolling_ball"
    assert diffraction.get("peak_detector") == "dog"
    assert diffraction.get("peak_adapter_engine") == "none"
    assert diffraction.get("auto_peak_tuning") is True
    assert diffraction.get("fft_center_mode") == "auto"
    assert diffraction.get("indexing_mode") == "lattice_fit"
    assert diffraction.get("candidate_phase_params")[0]["name"] == "cubic_probe"
    assert diffraction.get("phase_lookup_enabled") is True
    assert diffraction.get("phase_lookup_sources") == ["local_builtin"]
    assert diffraction.get("materials_project_sort_by") == "stability"
    assert diffraction.get("materials_project_candidate_pool_size") == 10
    simulation = recipe.workflow.parameters.get("simulation", {})
    assert simulation.get("simulation_engine") == "diffsims"
    assert simulation.get("phase_match_enabled") is True
    assert simulation.get("phase_match_top_n") == 2
    assert simulation.get("zone_axis_candidates") == [[0, 0, 1], [1, 1, 0]]
    assert simulation.get("rotation_angles_deg") == [0, 15]
    assert simulation.get("with_direct_beam") is True
    reporting = recipe.workflow.parameters.get("reporting", {})
    assert reporting.get("diffraction_gates", {}).get("min_peak_count") == 4


def test_diffsims_closed_loop_example_workflow_is_valid() -> None:
    recipe = load_workflow_recipe(Path("examples/diffsims_closed_loop.yaml"))
    assert recipe.workflow.name == "diffsims_closed_loop_example"
    assert "simulate" in [step.name for step in recipe.workflow.steps]
    simulation = recipe.workflow.parameters.get("simulation", {})
    assert simulation.get("simulation_engine") == "diffsims"
    assert simulation.get("zone_axis_candidates") == [[0, 0, 1], [1, 1, 0], [1, 1, 1]]
    assert simulation.get("phase_match_enabled") is True
    assert recipe.outputs.formats == ["json", "csv", "html"]


def test_materials_project_diffsims_lookup_example_workflow_is_valid() -> None:
    recipe = load_workflow_recipe(Path("examples/materials_project_diffsims_lookup.yaml"))
    diffraction = recipe.workflow.parameters.get("diffraction", {})
    assert diffraction.get("phase_lookup_sources") == ["materials_project_api"]
    assert diffraction.get("phase_candidate_materials") == ["mp-149"]
    simulation = recipe.workflow.parameters.get("simulation", {})
    assert simulation.get("simulation_engine") == "diffsims"
    assert simulation.get("zone_axis_candidates") == [[0, 0, 1], [1, 1, 0], [1, 1, 1]]
    assert simulation.get("fallback_single_atom") is False


def test_b5_validation_rejects_invalid_zone_axis_candidates(tmp_path: Path) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_zone_axis.yaml",
        [
            "workflow:",
            "  name: bad_zone_axis",
            "  parameters:",
            "    simulation:",
            "      simulation_engine: diffsims",
            "      zone_axis_candidates:",
            "        - [0, 0, 0]",
        ],
    )
    with pytest.raises(ValueError, match="zone_axis_candidates"):
        load_workflow_recipe(workflow)


def test_b5_validation_accepts_deprecated_baseline_engine_for_compatibility(
    tmp_path: Path,
) -> None:
    workflow = _write_workflow(
        tmp_path / "deprecated_baseline_engine.yaml",
        [
            "workflow:",
            "  name: deprecated_baseline_engine",
            "  parameters:",
            "    simulation:",
            "      simulation_engine: baseline",
        ],
    )

    recipe = load_workflow_recipe(workflow)

    assert recipe.workflow.parameters["simulation"]["simulation_engine"] == "baseline"


def test_b4_validation_rejects_invalid_materials_project_sort_mode(
    tmp_path: Path,
) -> None:
    workflow = _write_workflow(
        tmp_path / "bad_mp_sort.yaml",
        [
            "workflow:",
            "  name: bad_mp_sort",
            "  parameters:",
            "    diffraction:",
            "      materials_project_sort_by: random",
        ],
    )
    with pytest.raises(ValueError, match="materials_project_sort_by"):
        load_workflow_recipe(workflow)
