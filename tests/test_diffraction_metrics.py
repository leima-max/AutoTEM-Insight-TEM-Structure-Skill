from pathlib import Path

import numpy as np

from autotem.calibration import calibrate_dataset
from autotem.core import ArrayReference, PreprocessingResult, TEMDataset
from autotem.diffraction import run_diffraction_analysis


def _make_periodic_image(size: int, kx: int) -> np.ndarray:
    y, x = np.indices((size, size))
    signal = 0.5 + 0.5 * np.cos(2.0 * np.pi * kx * x / float(size))
    return signal.astype(np.float32)


def _make_dataset_and_preprocess(tmp_path: Path, image: np.ndarray) -> tuple[TEMDataset, PreprocessingResult]:
    image_path = tmp_path / "pattern.npy"
    np.save(image_path, image)
    data_ref = ArrayReference(
        shape=tuple(int(axis) for axis in image.shape),
        dtype="float32",
        uri=str(image_path),
    )
    dataset = TEMDataset(data=data_ref, metadata={})
    preprocessing = PreprocessingResult(image=data_ref)
    return dataset, preprocessing


def _best_abs_error(d_spacings_nm: list[float], target_nm: float) -> float:
    return min(abs(value - target_nm) for value in d_spacings_nm)


def test_known_d_spacing_calibration_improves_d_spacing_accuracy(tmp_path: Path) -> None:
    size = 128
    kx = 10
    target_d_nm = 0.2
    image = _make_periodic_image(size=size, kx=kx)
    dataset, preprocessing = _make_dataset_and_preprocess(tmp_path, image)

    default_calibration = calibrate_dataset(
        dataset,
        parameters={"pixel_size_nm": 0.1, "voltage_kv": 200},
    )
    default_result = run_diffraction_analysis(preprocessing, default_calibration)
    default_d_values = [float(item.value) for item in default_result.d_spacings]
    assert default_d_values

    known_d_calibration = calibrate_dataset(
        dataset,
        parameters={
            "pixel_size_nm": 0.1,
            "voltage_kv": 200,
            "known_d_spacing": target_d_nm,
            "known_d_spacing_unit": "nm",
            "measured_radius_px": float(kx),
        },
    )
    calibrated_result = run_diffraction_analysis(preprocessing, known_d_calibration)
    calibrated_d_values = [float(item.value) for item in calibrated_result.d_spacings]
    assert calibrated_d_values

    default_error = _best_abs_error(default_d_values, target_d_nm)
    calibrated_error = _best_abs_error(calibrated_d_values, target_d_nm)

    assert calibrated_error < default_error
    assert calibrated_error <= 0.02
    assert (
        calibrated_result.provenance.get("calibration_scale_source")
        == "calibration.diffraction_scale"
    )


def test_d_spacing_uncertainty_propagates_from_calibration(tmp_path: Path) -> None:
    size = 128
    kx = 8
    image = _make_periodic_image(size=size, kx=kx)
    dataset, preprocessing = _make_dataset_and_preprocess(tmp_path, image)

    calibration = calibrate_dataset(
        dataset,
        parameters={
            "known_d_spacing": 0.25,
            "known_d_spacing_unit": "nm",
            "known_d_uncertainty_nm": 0.005,
            "measured_radius_px": float(kx),
            "measured_radius_uncertainty_px": 0.5,
        },
    )
    result = run_diffraction_analysis(preprocessing, calibration)
    assert result.d_spacings
    assert any(item.uncertainty is not None and item.uncertainty > 0 for item in result.d_spacings)
