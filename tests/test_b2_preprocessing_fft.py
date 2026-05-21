from __future__ import annotations

from pathlib import Path

import numpy as np

from autotem.core import ArrayReference, TEMDataset
from autotem.diffraction.fft import locate_direct_beam
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
