from pathlib import Path

import numpy as np

from autotem.calibration import calibrate_dataset
from autotem.core import ArrayReference, TEMDataset
from autotem.preprocessing import assess_quality


def _make_dataset(tmp_path: Path, image: np.ndarray, *, metadata: dict | None = None) -> TEMDataset:
    image_path = tmp_path / "image.npy"
    np.save(image_path, image.astype(np.float32))
    return TEMDataset(
        data=ArrayReference(
            shape=tuple(int(axis) for axis in image.shape),
            dtype="float32",
            uri=str(image_path),
        ),
        metadata=metadata or {},
    )


def test_calibration_uses_metadata_normalized_pixel_size(tmp_path: Path) -> None:
    dataset = _make_dataset(
        tmp_path,
        np.zeros((32, 32), dtype=np.float32),
        metadata={"normalized": {"pixel_size_nm": 0.25, "accelerating_voltage_kv": 300}},
    )
    result = calibrate_dataset(dataset)
    assert result.pixel_size is not None
    assert result.pixel_size.value == 0.25
    assert result.pixel_size.method.startswith("metadata")
    assert result.voltage is not None
    assert result.voltage.value == 300


def test_calibration_supports_known_d_spacing_calibration(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, np.zeros((64, 64), dtype=np.float32))
    result = calibrate_dataset(
        dataset,
        parameters={
            "known_d_spacing": 0.2,
            "known_d_spacing_unit": "nm",
            "measured_radius_px": 20.0,
        },
    )
    assert result.diffraction_scale is not None
    assert result.diffraction_scale.method == "known_d_spacing_single_reference"
    assert float(result.diffraction_scale.value) == 0.25
    assert result.diffraction_scale.uncertainty is not None
    assert result.reference_measurements


def test_quality_assessment_emits_extended_b1_metrics(tmp_path: Path) -> None:
    image = np.zeros((64, 64), dtype=np.float32)
    image[32, 32] = 10.0
    image[20, 40] = 5.0
    dataset = _make_dataset(
        tmp_path,
        image,
        metadata={"normalized": {"metadata_completeness": 0.8}},
    )
    quality = assess_quality(dataset)
    assert quality.overall_score is not None
    assert 0.0 <= quality.overall_score <= 1.0
    assert quality.snr is not None
    assert quality.blur is not None
    assert quality.contamination is not None
    assert quality.astigmatism is not None
    assert quality.undersampling_risk is not None


def test_calibration_external_adapter_fallback_records_warning(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path, np.zeros((64, 64), dtype=np.float32))
    result = calibrate_dataset(
        dataset,
        parameters={
            "diffraction_calibration_engine": "py4dstem",
            "known_d_spacing": 0.2,
            "known_d_spacing_unit": "nm",
            "measured_radius_px": 20.0,
        },
    )
    assert result.diffraction_scale is not None
    assert result.transform.get("requested_adapter_engine") == "py4dstem"
    assert result.transform.get("calibration_source") == "parameters.known_d_spacing"
