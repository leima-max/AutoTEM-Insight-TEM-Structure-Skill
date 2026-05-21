from __future__ import annotations

import numpy as np
import pytest

from autotem.calibration.diffraction_calibration import (
    diffraction_scale_from_known_d_spacing,
    diffraction_scale_from_reference_pairs,
    resolve_diffraction_calibration,
)
from autotem.calibration.pixel_calibration import (
    convert_length,
    manual_pixel_size_from_reference,
    resolve_pixel_size_measurement,
)
from autotem.core import ArrayReference, Measurement, TEMDataset
from autotem.preprocessing.quality import assess_image_quality


def _dataset_from_image(
    image: np.ndarray,
    *,
    pixel_size_nm: float | None = None,
    metadata: dict[str, object] | None = None,
) -> TEMDataset:
    pixel_size = (
        None
        if pixel_size_nm is None
        else Measurement(value=float(pixel_size_nm), unit="nm", method="test")
    )
    return TEMDataset(
        data=ArrayReference(
            shape=tuple(int(v) for v in image.shape),
            dtype=str(image.dtype),
            inline=image.tolist(),
            description="test image",
        ),
        dataset_id="b1-test",
        source_path="",
        modality="HRTEM",
        pixel_size=pixel_size,
        pixel_unit=("nm" if pixel_size is not None else "unknown"),
        metadata=dict(metadata or {}),
    )


def test_b1_length_conversion_and_manual_pixel_calibration() -> None:
    assert convert_length(1.0, "um", "nm") == pytest.approx(1000.0)
    pixel_nm = manual_pixel_size_from_reference(
        reference_length=5.0,
        reference_pixels=100.0,
        reference_unit="nm",
    )
    assert pixel_nm == pytest.approx(0.05)


def test_b1_resolve_pixel_size_measurement_manual_reference() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    dataset = _dataset_from_image(image)
    measurement, source, warnings = resolve_pixel_size_measurement(
        dataset,
        reference={
            "reference_length": 5.0,
            "reference_pixels": 100.0,
            "reference_unit": "nm",
        },
    )
    assert measurement is not None
    assert measurement.value == pytest.approx(0.05)
    assert measurement.unit == "nm/pixel"
    assert source == "manual_reference_length"
    assert warnings == []


def test_b1_diffraction_scale_from_known_d_spacing() -> None:
    scale, uncertainty = diffraction_scale_from_known_d_spacing(
        known_d_spacing_nm=0.2,
        measured_radius_px=50.0,
        known_d_spacing_unit="nm",
        known_d_uncertainty_nm=0.002,
        measured_radius_uncertainty_px=1.0,
    )
    assert scale == pytest.approx(0.1)
    assert uncertainty is not None
    assert uncertainty > 0


def test_b1_multi_reference_diffraction_scale_rejects_outlier() -> None:
    scale, uncertainty, details, warnings = diffraction_scale_from_reference_pairs(
        [
            {"known_d_spacing": 0.20, "measured_radius_px": 50.0, "label": "ref0"},
            {"known_d_spacing": 0.25, "measured_radius_px": 40.0, "label": "ref1"},
            {"known_d_spacing": 0.22, "measured_radius_px": 20.0, "label": "outlier"},
        ],
        default_known_d_spacing_unit="nm",
        outlier_sigma=2.5,
        min_inlier_pairs=2,
    )
    assert scale == pytest.approx(0.1, abs=0.01)
    assert uncertainty is not None
    assert details["inlier_count"] == 2
    assert details["outlier_count"] == 1
    assert any("outlier" in item["label"] for item in details["pairs"])
    assert warnings


def test_b1_resolve_diffraction_calibration_pixel_fallback() -> None:
    resolved = resolve_diffraction_calibration(
        pixel_size_nm=0.05,
        voltage_kv=None,
        parameters={},
        reference={},
        fft_shape=(256, 256),
    )
    assert resolved["diffraction_scale_inv_nm_per_px"] == pytest.approx(1.0 / (256.0 * 0.05))
    assert resolved["source"] == "pixel_size_fft_fallback"
    assert resolved["warnings"]


def test_b1_resolve_diffraction_calibration_fallback_when_adapter_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from autotem.calibration import open_source_adapters

    def _fake_adapter(**_kwargs):
        raise RuntimeError("adapter unavailable in test")

    monkeypatch.setattr(
        open_source_adapters,
        "scale_from_known_d_spacing_with_adapter",
        _fake_adapter,
    )
    resolved = resolve_diffraction_calibration(
        pixel_size_nm=None,
        voltage_kv=None,
        parameters={
            "diffraction_calibration_engine": "py4dstem",
            "known_d_spacings": [0.2, 0.25, 0.22],
            "measured_radii_px": [50.0, 40.0, 20.0],
            "multi_reference_outlier_sigma": 2.5,
        },
        reference={},
        fft_shape=(256, 256),
    )
    assert resolved["diffraction_scale_inv_nm_per_px"] == pytest.approx(0.1, abs=0.01)
    assert resolved["source"] == "known_d_spacing.multi_reference"
    assert resolved["warnings"]
    details = resolved.get("reference_pair_details")
    assert isinstance(details, dict)
    assert details.get("outlier_count") == 1


def test_b1_quality_assessment_outputs_required_fields() -> None:
    rng = np.random.default_rng(42)
    image = rng.normal(loc=0.1, scale=0.03, size=(128, 128)).astype(np.float32)
    image[64, 20:108] += 0.7
    image = np.clip(image, 0.0, 1.0)
    dataset = _dataset_from_image(
        image,
        pixel_size_nm=0.05,
        metadata={"normalized": {"accelerating_voltage_kv": 200.0}},
    )

    quality = assess_image_quality(dataset)
    assert quality.overall_score is not None
    assert 0.0 <= quality.overall_score <= 1.0
    assert quality.snr is not None and quality.snr.unit == "dB"
    assert quality.drift is not None and quality.drift.unit == "pixel"
    assert quality.blur is not None and quality.blur.unit == "risk"
    assert quality.contamination is not None and quality.contamination.unit == "risk"
    assert quality.astigmatism is not None and quality.astigmatism.unit == "risk"
    assert quality.undersampling_risk is not None
    assert 0.0 <= quality.undersampling_risk <= 1.0


def test_b1_quality_modality_specific_threshold_profile() -> None:
    image = np.zeros((128, 128), dtype=np.float32)
    image[64, 60:68] = 1.0
    dataset_hrtem = _dataset_from_image(image, pixel_size_nm=0.06)
    dataset_hrtem.modality = "HRTEM"
    dataset_saed = _dataset_from_image(image, pixel_size_nm=0.06)
    dataset_saed.modality = "SAED"

    quality_hrtem = assess_image_quality(dataset_hrtem)
    quality_saed = assess_image_quality(dataset_saed)

    assert quality_hrtem.provenance.get("modality") == "HRTEM"
    assert quality_saed.provenance.get("modality") == "SAED"
    thresholds_hrtem = quality_hrtem.provenance.get("thresholds", {})
    thresholds_saed = quality_saed.provenance.get("thresholds", {})
    assert thresholds_hrtem.get("drift_px_max") != thresholds_saed.get("drift_px_max")
