from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autotem.core.models import TEMDataset
from autotem.io.converters import to_temdataset
from autotem.io.readers import ReaderRegistry, build_default_registry, read_image


def _write_gray_image(path: Path, fmt: str) -> np.ndarray:
    data = (np.arange(64).reshape(8, 8) * 2).astype(np.uint8)
    Image.fromarray(data).save(path, format=fmt)
    return data


def test_read_png_jpg_tiff_success(tmp_path: Path) -> None:
    png_path = tmp_path / "sample.png"
    jpg_path = tmp_path / "sample.jpg"
    tiff_path = tmp_path / "sample.tiff"

    expected = _write_gray_image(png_path, "PNG")
    _write_gray_image(jpg_path, "JPEG")
    Image.fromarray(expected).save(tiff_path, format="TIFF", dpi=(25400, 25400))

    png_result = read_image(png_path)
    jpg_result = read_image(jpg_path)
    tiff_result = read_image(tiff_path)

    assert png_result.array.shape == (8, 8)
    assert jpg_result.array.shape == (8, 8)
    assert tiff_result.array.shape == (8, 8)
    assert np.allclose(png_result.array, expected.astype(np.float32))
    assert tiff_result.metadata_norm["pixel_size_nm"] == pytest.approx(1000.0, abs=1.0)


def test_reader_registry_dispatch_and_unsupported(tmp_path: Path) -> None:
    registry = ReaderRegistry()
    registry.register(".png", lambda _: None)  # type: ignore[arg-type]

    assert registry.get_reader("x.png") is not None
    with pytest.raises(ValueError):
        registry.get_reader("x.bmp")

    file_path = tmp_path / "s.png"
    _write_gray_image(file_path, "PNG")
    result = build_default_registry().read(file_path)
    assert result.array.shape == (8, 8)


def test_missing_metadata_has_warnings(tmp_path: Path) -> None:
    file_path = tmp_path / "meta_missing.png"
    _write_gray_image(file_path, "PNG")
    result = read_image(file_path)
    codes = {w["code"] for w in result.warnings}
    assert "META_MISSING_PIXEL_SIZE" in codes
    assert "META_MISSING_VOLTAGE" in codes


def test_to_temdataset_core_contract(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.png"
    _write_gray_image(file_path, "PNG")
    result = read_image(file_path)
    dataset = to_temdataset(result)

    assert isinstance(dataset, TEMDataset)
    assert dataset.data.shape == (8, 8)
    assert dataset.metadata["normalized"]["pixel_size_nm"] is None
    assert any("META_MISSING_PIXEL_SIZE" in warning for warning in dataset.warnings)

