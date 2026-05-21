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
    reporting = recipe.workflow.parameters.get("reporting", {})
    assert reporting.get("diffraction_gates", {}).get("min_peak_count") == 4
