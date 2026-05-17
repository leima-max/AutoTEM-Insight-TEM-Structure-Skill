from __future__ import annotations

from pathlib import Path

import numpy as np

from autotem.core import ArrayReference, TEMDataset
from autotem.diffraction.basic import _load_preprocessed_array
from autotem.preprocessing import assess_quality, preprocess_dataset


def _make_dataset(tmp_path: Path, image: np.ndarray) -> TEMDataset:
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.asarray(image, dtype=np.float32), allow_pickle=False)
    return TEMDataset(
        data=ArrayReference(
            shape=tuple(int(v) for v in image.shape),
            dtype="float32",
            uri=str(input_path),
            description="test input",
        ),
        metadata={},
    )


def test_preprocess_dataset_persists_normalized_array(tmp_path: Path) -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    dataset = _make_dataset(tmp_path, image)

    result = preprocess_dataset(dataset, quality_after=assess_quality(dataset))

    assert result.image is not None
    assert result.image.uri is not None
    saved = np.load(result.image.uri, allow_pickle=False).astype(np.float32)
    assert saved.shape == image.shape
    assert np.min(saved) >= 0.0
    assert np.max(saved) <= 1.0
    assert not np.allclose(saved, image)


def test_diffraction_loader_reads_preprocessed_array(tmp_path: Path) -> None:
    image = np.zeros((16, 16), dtype=np.float32)
    image[4, 4] = 10.0
    image[12, 12] = 6.0
    dataset = _make_dataset(tmp_path, image)

    result = preprocess_dataset(
        dataset,
        quality_after=assess_quality(dataset),
        recipe="b2_fft",
        parameters={"denoise_sigma": 1.0, "background_sigma": 4.0},
    )
    loaded = _load_preprocessed_array(result)

    assert loaded is not None
    assert loaded.shape == image.shape
    assert loaded.dtype == np.float32
    assert not np.allclose(loaded, image)
