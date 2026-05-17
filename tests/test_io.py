from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from autotem.core.models import TEMDataset
from autotem.io.converters import to_temdataset
from autotem.io.metadata import normalize_metadata
from autotem.io.readers import (
    ReaderRegistry,
    ReadResult,
    build_default_registry,
    read_image,
    read_series,
    read_series_as_temdataset,
)


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


def test_rgb_16bit_and_multiframe_tiff(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    rgb_data = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb_data[:, :, 0] = 100
    rgb_data[:, :, 1] = 150
    rgb_data[:, :, 2] = 200
    Image.fromarray(rgb_data).save(rgb_path, format="PNG")

    rgb_result = read_image(rgb_path)
    assert rgb_result.array.shape == (8, 8)
    assert any(w["code"] == "IMAGE_RGB_TO_GRAYSCALE" for w in rgb_result.warnings)

    tiff_16_path = tmp_path / "uint16.tiff"
    data_16 = (np.arange(64).reshape(8, 8) * 10).astype(np.uint16)
    Image.fromarray(data_16).save(tiff_16_path, format="TIFF")
    result_16 = read_image(tiff_16_path)
    assert result_16.array.dtype == np.float32
    assert result_16.array.shape == (8, 8)

    multi_path = tmp_path / "multi.tiff"
    frame1 = Image.fromarray(np.ones((8, 8), dtype=np.uint8))
    frame2 = Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 2)
    frame1.save(multi_path, format="TIFF", save_all=True, append_images=[frame2])
    multi_result = read_image(multi_path)
    assert multi_result.array.shape == (8, 8)
    assert any(w["code"] == "IMAGE_MULTIFRAME_TIFF" for w in multi_result.warnings)


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


def test_registry_reports_missing_optional_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    import autotem.io.readers as readers_module

    original_find_spec = readers_module.find_spec

    def fake_find_spec(name: str) -> object | None:
        if name in {"ncempy", "mrcfile", "h5py"}:
            return None
        return original_find_spec(name)

    monkeypatch.setattr(readers_module, "find_spec", fake_find_spec)
    registry = readers_module.build_default_registry()
    desc = registry.describe_formats()

    assert desc[".png"]["available"] is True
    assert desc[".dm3"]["available"] is False
    with pytest.raises(ValueError, match="Install: ncempy"):
        registry.get_reader("sample.dm3")


@pytest.mark.parametrize(
    ("raw", "expected_nm"),
    [
        ({"pixel_size": "0.5 nm"}, 0.5),
        ({"pixel_size": "500 pm"}, 0.5),
        ({"pixel_size": "5 angstrom"}, 0.5),
        ({"pixel_size": "0.0005 um"}, 0.5),
    ],
)
def test_metadata_unit_conversion_regression(raw: dict[str, str], expected_nm: float) -> None:
    normalized, _ = normalize_metadata(raw, source_format="dm")
    assert normalized["pixel_size_nm"] == pytest.approx(expected_nm)


@pytest.mark.parametrize(
    ("raw", "missing_code"),
    [
        ({}, "META_MISSING_PIXEL_SIZE"),
        ({"pixel_size": "0.2 nm"}, "META_MISSING_VOLTAGE"),
        (
            {"pixel_size": "0.2 nm", "accelerating_voltage_kv": 200},
            "META_MISSING_CAMERA_LENGTH",
        ),
    ],
)
def test_missing_metadata_warning_regression(
    raw: dict[str, object],
    missing_code: str,
) -> None:
    _, warnings = normalize_metadata(raw, source_format="dm")
    codes = {warning["code"] for warning in warnings}
    assert missing_code in codes


@pytest.mark.parametrize(
    ("raw", "min_confidence"),
    [
        (
            {
                "pixel_size": "0.2 nm",
                "accelerating_voltage_kv": 200,
                "camera_length_mm": 800,
                "magnification": 500000,
                "detector_inner_angle": "20 mrad",
            },
            0.95,
        ),
        ({}, 0.0),
    ],
)
def test_metadata_confidence_regression(raw: dict[str, object], min_confidence: float) -> None:
    normalized, _ = normalize_metadata(raw, source_format="dm")
    confidence = normalized["metadata_confidence"]
    completeness = normalized["metadata_completeness"]
    assert 0.0 <= confidence <= 1.0
    assert 0.0 <= completeness <= 1.0
    assert confidence >= min_confidence


def test_tiff_resolution_unit_parsing() -> None:
    normalized, warnings = normalize_metadata(
        {"XResolution": 1000.0, "ResolutionUnit": 2},
        source_format=".tiff",
    )
    assert normalized["pixel_size_nm"] is None
    assert any(w["code"] == "META_IMPLAUSIBLE_TIFF_SCALE" for w in warnings)
    assert any(w["code"] == "META_MISSING_PIXEL_SIZE" for w in warnings)

    normalized_cm, warnings_cm = normalize_metadata(
        {"XResolution": 1000.0, "ResolutionUnit": 3},
        source_format=".tiff",
    )
    assert normalized_cm["pixel_size_nm"] is None
    assert any(w["code"] == "META_IMPLAUSIBLE_TIFF_SCALE" for w in warnings_cm)

    normalized_unknown, warnings_unknown = normalize_metadata(
        {"XResolution": 1000.0, "ResolutionUnit": 1},
        source_format=".tiff",
    )
    assert normalized_unknown["pixel_size_nm"] is None
    assert any(w["code"] == "META_UNKNOWN_RESOLUTION_UNIT" for w in warnings_unknown)


def test_series_read_directory_and_dataset_conversion(tmp_path: Path) -> None:
    path_a = tmp_path / "a.png"
    path_b = tmp_path / "b.jpg"
    path_txt = tmp_path / "ignore.md"
    _write_gray_image(path_a, "PNG")
    _write_gray_image(path_b, "JPEG")
    path_txt.write_text("ignore")

    results = read_series(tmp_path)
    assert len(results) == 2
    assert all(isinstance(item, ReadResult) for item in results)
    paths = [item.provenance["path"] for item in results]
    assert str(path_a) in paths
    assert str(path_b) in paths

    datasets = read_series_as_temdataset(tmp_path)
    assert len(datasets) == 2
    assert all(isinstance(item, TEMDataset) for item in datasets)


def test_series_skip_errors_on_mixed_sequence(tmp_path: Path) -> None:
    good = tmp_path / "good.png"
    bad = tmp_path / "bad.xyz"
    _write_gray_image(good, "PNG")
    bad.write_text("nope")

    with pytest.raises(ValueError):
        read_series([good, bad], skip_errors=False)

    results = read_series([good, bad], skip_errors=True)
    assert len(results) == 1
    assert Path(results[0].provenance["path"]).name == "good.png"


def test_to_temdataset_core_contract_and_large_array_inline_policy() -> None:
    result = ReadResult(
        array=np.ones((80, 80), dtype=np.float32),
        metadata_raw={"format": "png"},
        metadata_norm={"pixel_size_nm": 0.2, "source_format": "png"},
        warnings=[],
        provenance={"path": "C:/data/sample.png", "format": "PNG", "reader": "pil"},
    )
    dataset = to_temdataset(result)

    assert isinstance(dataset, TEMDataset)
    assert dataset.data.shape == (80, 80)
    assert dataset.data.inline is None
    assert dataset.data.uri == "C:/data/sample.png"
    assert dataset.pixel_unit == "nm"
    assert dataset.modality == "TEM"

