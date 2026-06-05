from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from autotem.core import ArrayReference, CalibrationResult, TEMDataset
from autotem.diffraction.basic import run_diffraction_analysis
from autotem.diffraction.fft import locate_direct_beam
from autotem.io import load_tem_dataset
from autotem.preprocessing.basic import preprocess_dataset


def _dataset_from_array(image: np.ndarray, path: Path) -> TEMDataset:
    np.save(path, image.astype(np.float32))
    return TEMDataset(
        data=ArrayReference(
            shape=tuple(int(v) for v in image.shape),
            dtype="float32",
            uri=str(path),
            description="test image",
        ),
        dataset_id=path.stem,
        source_path=str(path),
        modality="HRTEM",
        metadata={},
    )


def test_b2_preprocess_supports_multiple_denoise_methods(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:44, 20:44] = 1.0
    image = image + 0.20 * rng.standard_normal(image.shape).astype(np.float32)
    dataset = _dataset_from_array(image, tmp_path / "denoise_input.npy")

    for method in ("gaussian", "median", "wiener", "nlm"):
        result = preprocess_dataset(
            dataset,
            parameters={
                "denoise_method": method,
                "denoise_sigma": 0.8,
                "median_size": 3,
                "wiener_window": 5,
                "background_method": "none",
            },
        )
        assert result.image is not None
        assert result.image.shape == image.shape
        assert any(op.startswith("denoise:") for op in result.operations)


def test_b2_preprocess_background_subtraction_rolling_ball(tmp_path: Path) -> None:
    yy, xx = np.indices((80, 80))
    background = (yy + xx).astype(np.float32) / 200.0
    image = background.copy()
    image[30:45, 32:48] += 0.8
    dataset = _dataset_from_array(image, tmp_path / "bg_input.npy")

    result = preprocess_dataset(
        dataset,
        parameters={
            "denoise_method": "none",
            "background_method": "rolling_ball",
            "background_opening_size": 15,
        },
    )
    assert result.background is not None
    assert any(op.startswith("background:rolling_ball") for op in result.operations)


def test_b2_preprocess_can_bypass_to_raw_input(tmp_path: Path) -> None:
    image = np.linspace(10.0, 200.0, 64, dtype=np.float32).reshape(8, 8)
    dataset = _dataset_from_array(image, tmp_path / "raw_input.npy")

    result = preprocess_dataset(
        dataset,
        parameters={
            "enabled": False,
            "denoise_method": "wiener",
            "background_method": "gaussian",
        },
    )

    assert result.image is not None
    assert result.operations == ["raw_passthrough"]
    assert result.provenance["mode"] == "raw_passthrough"
    assert result.parameters["enabled"] is False
    assert not any(op.startswith("denoise:") for op in result.operations)
    assert not any(op.startswith("background:") for op in result.operations)
    assert np.allclose(np.load(result.image.uri, allow_pickle=False), image)


def test_b3_diffraction_accepts_tiff_raw_passthrough(tmp_path: Path) -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    image[16, 16] = 255
    image[10, 16] = 160
    image[22, 16] = 160
    image[16, 10] = 140
    image[16, 22] = 140
    path = tmp_path / "raw_input.tiff"
    Image.fromarray(image).save(path, format="TIFF")
    dataset = load_tem_dataset(path)
    preprocessing = preprocess_dataset(dataset, parameters={"enabled": False})

    result = run_diffraction_analysis(
        preprocessing,
        CalibrationResult(),
        parameters={"top_k_peaks": 4, "center_exclusion_radius": 2.0},
    )

    assert preprocessing.operations == ["raw_passthrough"]
    assert result.provenance["method"] == "run_diffraction_analysis"
    assert result.provenance.get("reason") != "missing_preprocessed"


def test_b2_fft_direct_beam_auto_localization() -> None:
    magnitude = np.zeros((64, 64), dtype=np.float64)
    magnitude[35, 29] = 120.0
    magnitude[35, 30] = 90.0
    magnitude[34, 29] = 85.0
    magnitude[36, 29] = 82.0

    center, diagnostics, warnings = locate_direct_beam(
        magnitude,
        center_mode="auto",
        search_radius_px=12.0,
        refine_window=2,
    )
    assert not warnings
    assert abs(center[0] - 29.0) < 1.0
    assert abs(center[1] - 35.0) < 1.0
    assert diagnostics["mode"] == "auto"
