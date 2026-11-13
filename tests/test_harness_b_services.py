"""Numerical and error contracts for existing B algorithms exposed through Harness."""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError

from autotem.core import (
    ArrayReference,
    CalibrationResult,
    Measurement,
    PhaseCandidate,
    SimulatedPattern,
    SimulatedSpot,
    SimulationResult,
    TEMDataset,
)
from autotem.harness.contracts import HarnessError, NeedsInput, OperationContext
from autotem.services import data_quality as dq
from autotem.services import diffraction as df

pytestmark = pytest.mark.smoke


class Context:
    def __init__(self, root: Path, input_path: Path | None = None, request: dict | None = None):
        self.output_dir = root
        root.mkdir(parents=True, exist_ok=True)
        self.input_path = input_path
        self.run_id = uuid.uuid4().hex
        self.request = request or {}
        self.values = {}
        self.arrays = {}

    def array(self, role):
        return self.arrays[role]

    def model(self, role, cls):
        value = self.values[role]
        return cls.model_validate(value.model_dump() if isinstance(value, BaseModel) else value)

    def value(self, role):
        value = self.values[role]
        return value.model_dump(mode="json") if isinstance(value, BaseModel) else value

    def emit_array(self, role, array, metadata=None):
        array = np.asarray(array)
        self.arrays[role] = array.copy()
        path = self.output_dir / f"{role}-{uuid.uuid4().hex}.npy"
        np.save(path, array, allow_pickle=False)
        return ArrayReference(shape=array.shape, dtype=str(array.dtype), uri=str(path))

    def emit_json(self, role, value, metadata=None):
        self.values[role] = value
        return uuid.uuid4().hex

    def emit_table(self, role, rows, metadata=None):
        self.values[role] = rows
        return uuid.uuid4().hex

    def emit_image(self, role, value, metadata=None):
        return uuid.uuid4().hex

    def call(self, handler, params=None):
        result = handler(self, params or {})
        self.values.update(result.outputs)
        return result


def loaded(tmp_path, array, modality="HRTEM"):
    source = tmp_path / "source.npy"
    np.save(source, array)
    ctx = Context(tmp_path / "run", source, {"modality": modality})
    ctx.call(dq.load_input)
    return ctx


def metric_context(tmp_path, *, direct=False, physical=True):
    ctx = loaded(tmp_path, np.ones((64, 128)), "SAED" if direct else "HRTEM")
    ctx.arrays["processed"] = ctx.arrays["raw"]
    if not direct:
        ctx.call(df.fft)
    ctx.call(df.center, {"center_mode": "manual", "center_x": 64.0, "center_y": 32.0})
    ctx.values["peaks"] = [{"x": 74.0, "y": 32.0}, {"x": 64.0, "y": 37.0}]
    calibration = CalibrationResult()
    if physical:
        if direct:
            calibration.diffraction_scale = Measurement(value=0.25, unit="1/nm/pixel")
        else:
            calibration.pixel_size = Measurement(value=0.025, unit="nm/pixel")
    ctx.values["calibration"] = calibration
    return ctx


def test_b_registry_has_only_callable_implemented_operations():
    definitions = dq.definitions() + df.definitions()
    assert len({definition.operation_id for definition in definitions}) == 16
    assert {definition.profile_id for definition in definitions} == {"data-quality", "diffraction"}
    for definition in definitions:
        assert callable(definition.handler)
        assert definition.implementation_status == "implemented"
        with pytest.raises(ValidationError):
            definition.parameter_model.model_validate({"not_a_parameter": True})


def test_load_preserves_full_pixels_dtype_and_source(tmp_path):
    raw = np.arange(256, dtype=np.uint16).reshape(16, 16)
    ctx = loaded(tmp_path, raw)
    dataset = ctx.model("dataset", TEMDataset)
    assert dataset.data.shape == raw.shape
    assert dataset.data.dtype == "uint16"
    np.testing.assert_array_equal(ctx.array("raw"), raw)
    np.testing.assert_array_equal(np.load(ctx.input_path), raw)
    assert list(tmp_path.glob("source_*")) == []


def test_load_merges_plan_selection_and_explicit_parameters(tmp_path):
    raw = np.arange(3 * 16 * 16).reshape(3, 16, 16)
    source = tmp_path / "stack.npy"
    np.save(source, raw)
    ctx = Context(tmp_path / "run", source, {"selection": {"frame": 0}, "modality": "STEM"})
    ctx.call(dq.load_input, {"frame": 2})
    np.testing.assert_array_equal(ctx.array("raw"), raw[2])
    assert ctx.model("dataset", TEMDataset).modality == "STEM"


def test_input_roi_is_shared_raw_and_records_source_coordinate_mapping(tmp_path):
    raw = np.arange(20 * 30).reshape(20, 30)
    ctx = loaded(tmp_path, raw)
    ctx.call(dq.load_input, {"roi": [3, 15, 4, 24]})
    ctx.call(dq.preprocess, {"enabled": False})
    np.testing.assert_array_equal(ctx.array("raw"), raw[3:15, 4:24])
    np.testing.assert_array_equal(ctx.array("processed"), raw[3:15, 4:24])
    metadata = ctx.model("dataset", TEMDataset).metadata["selection"]
    assert metadata["source_2d_shape"] == [20, 30]
    assert metadata["roi_origin_yx"] == [3, 4]
    assert metadata["coordinate_mapping"]["source_y"] == "selected_y + roi_origin_y"
    with pytest.raises(HarnessError) as error:
        ctx.call(dq.load_input, {"roi": [3, 21, 4, 24]})
    assert error.value.code == "INVALID_ROI"


def test_multidimensional_input_requires_complete_selection(tmp_path):
    array = np.arange(2 * 3 * 8 * 9).reshape(2, 3, 8, 9)
    source = tmp_path / "stack.npy"
    np.save(source, array)
    ctx = Context(tmp_path / "run", source)
    with pytest.raises(NeedsInput) as error:
        ctx.call(dq.load_input)
    assert error.value.details["fields"][0]["name"] == "selection_indices"
    ctx.call(dq.load_input, {"selection_indices": [1, 2]})
    np.testing.assert_array_equal(ctx.array("raw"), array[1, 2])
    for indices in ([1], [2, 0], [-1, 0]):
        with pytest.raises(HarnessError, match="Selection"):
            ctx.call(dq.load_input, {"selection_indices": indices})


@pytest.mark.parametrize(
    "array",
    [np.ones((4, 4)) * np.nan, np.ones((4, 4), dtype=complex), np.empty((0, 4)), np.ones((1, 4))],
)
def test_load_rejects_nonfinite_or_nonimage_arrays(tmp_path, array):
    path = tmp_path / "invalid.npy"
    np.save(path, array)
    with pytest.raises(HarnessError) as error:
        Context(tmp_path / "run", path).call(dq.load_input)
    assert error.value.code in {"INVALID_IMAGE", "INVALID_INPUT"}


def test_rgb_and_tiff_frames_are_never_implicitly_averaged_or_selected(tmp_path):
    rgb = np.zeros((8, 9, 3), dtype=np.uint8)
    rgb[..., 1] = 123
    rgb_path = tmp_path / "rgb.png"
    Image.fromarray(rgb).save(rgb_path)
    ctx = Context(tmp_path / "run", rgb_path)
    with pytest.raises(NeedsInput):
        ctx.call(dq.load_input)
    ctx.call(dq.load_input, {"channel": 1})
    np.testing.assert_array_equal(ctx.array("raw"), rgb[..., 1])
    path = tmp_path / "frames.tiff"
    Image.fromarray(np.zeros((8, 9), dtype=np.uint8)).save(
        path, save_all=True, append_images=[Image.fromarray(np.full((8, 9), 7, dtype=np.uint8))]
    )
    ctx.input_path = path
    with pytest.raises(NeedsInput):
        ctx.call(dq.load_input)
    ctx.call(dq.load_input, {"frame": 1})
    assert np.all(ctx.array("raw") == 7)


def test_npz_multiple_datasets_require_name(tmp_path):
    source = tmp_path / "images.npz"
    np.savez(source, first=np.ones((8, 9)), second=np.full((8, 9), 2))
    ctx = Context(tmp_path / "run", source)
    with pytest.raises(NeedsInput):
        ctx.call(dq.load_input)
    ctx.call(dq.load_input, {"dataset_path": "second"})
    assert np.all(ctx.array("raw") == 2)


def test_hdf_requires_dataset_and_frame_selection(tmp_path):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "images.h5"
    with h5py.File(source, "w") as file:
        file["data/a"] = np.ones((8, 9))
        file["data/b"] = np.arange(2 * 8 * 9).reshape(2, 8, 9)
    ctx = Context(tmp_path / "run", source)
    with pytest.raises(NeedsInput):
        ctx.call(dq.load_input)
    with pytest.raises(NeedsInput):
        ctx.call(dq.load_input, {"dataset_path": "/data/b"})
    ctx.call(dq.load_input, {"dataset_path": "/data/b", "frame": 1})
    np.testing.assert_array_equal(ctx.array("raw"), np.arange(144).reshape(2, 8, 9)[1])
    with pytest.raises(HarnessError):
        ctx.call(dq.load_input, {"dataset_path": "missing"})


def test_ser_requires_explicit_dataset_and_supports_frame_alias(tmp_path, monkeypatch):
    from ncempy.io import ser

    array = np.arange(80).reshape(8, 10)

    class File:
        head = {"ValidNumberElements": 3}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getDataset(self, index):
            return array + index, {"selection": index}

    monkeypatch.setattr(ser, "fileSER", lambda *args: File())
    source = tmp_path / "series.ser"
    source.write_bytes(b"reader fixture")
    ctx = Context(tmp_path / "run", source)
    with pytest.raises(NeedsInput):
        ctx.call(dq.load_input)
    ctx.call(dq.load_input, {"dataset_index": 2})
    np.testing.assert_array_equal(ctx.array("raw"), array + 2)
    ctx.call(dq.load_input, {"frame": 1})
    np.testing.assert_array_equal(ctx.array("raw"), array + 1)
    with pytest.raises(HarnessError) as error:
        ctx.call(dq.load_input, {"dataset_index": 3})
    assert error.value.code == "INVALID_SELECTION"


def test_dm_selection_does_not_collapse_anisotropic_spatial_units(tmp_path, monkeypatch):
    from ncempy.io import dm

    class File:
        numObjects = 3  # thumbnail + two image datasets

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getDataset(self, index):
            return {
                "data": np.ones((8, 9)) * index,
                "pixelSize": [0.01, 0.02],
                "pixelUnit": ["nm", "nm"],
            }

        def getMetadata(self, index):
            return {}

    monkeypatch.setattr(dm, "fileDM", lambda *args, **kwargs: File())
    source = tmp_path / "two.dm4"
    source.write_bytes(b"reader fixture")
    ctx = Context(tmp_path / "run", source)
    with pytest.raises(NeedsInput):
        ctx.call(dq.load_input)
    ctx.call(dq.load_input, {"dataset_index": 1})
    dataset = ctx.model("dataset", TEMDataset)
    assert dataset.pixel_size is None
    assert dataset.metadata["normalized"]["anisotropic_pixel_size_nm"] == [0.01, 0.02]
    with pytest.raises(NeedsInput):
        ctx.call(dq.resolve_calibration, {"require_physical": True})


def test_emi_requests_associated_image_instead_of_opening_metadata_as_ser(tmp_path):
    source = tmp_path / "metadata.emi"
    source.write_text("<metadata/>")
    ctx = Context(tmp_path / "run", source)
    with pytest.raises(NeedsInput) as error:
        ctx.call(dq.load_input)
    assert error.value.details["fields"][0]["format"] == "SER"


def test_quality_returns_finite_metrics_for_selected_pixels(tmp_path):
    ctx = loaded(tmp_path, np.random.default_rng(12).normal(size=(64, 64)))
    result = ctx.call(dq.assess)
    quality = result.outputs["quality"]
    assert 0 <= quality.overall_score <= 1
    assert np.isfinite([quality.snr.value, quality.blur.value, quality.drift.value]).all()


def test_quality_threshold_and_weight_configuration_is_recorded_and_strict(tmp_path):
    ctx = loaded(tmp_path, np.random.default_rng(12).normal(size=(64, 64)))
    result = ctx.call(
        dq.assess,
        {
            "thresholds": {"snr_min": 15.0},
            "thresholds_by_modality": {"HRTEM": {"snr_min": 20.0}},
            "metric_weights": {"blur": 2.0},
        },
    )
    quality = result.outputs["quality"]
    assert quality.provenance["thresholds"]["snr_min"] == 20
    assert quality.provenance["score_weights"]["blur"] > 0
    with pytest.raises(HarnessError) as error:
        ctx.call(dq.assess, {"thresholds": {"not_a_threshold": 12.0}})
    assert error.value.code == "INVALID_PARAMETERS"
    with pytest.raises(HarnessError):
        ctx.call(dq.assess, {"metric_weights": {"blur": float("nan")}})


def test_calibration_respects_manual_scale_and_preserves_unknown(tmp_path, monkeypatch):
    ctx = loaded(tmp_path, np.ones((16, 16)))
    import autotem.calibration.pixel_calibration as pixels

    def must_not_guess(*args, **kwargs):
        raise AssertionError("Implicit image/filename scale guessing is forbidden")

    monkeypatch.setattr(pixels, "infer_pixel_size_from_scalebar", must_not_guess)
    unknown = ctx.call(dq.resolve_calibration).outputs["calibration"]
    assert unknown.pixel_size is None
    with pytest.raises(NeedsInput):
        ctx.call(dq.resolve_calibration, {"require_physical": True})
    result = ctx.call(dq.resolve_calibration, {"reference_length": 4.0, "reference_pixels": 20.0})
    assert result.outputs["calibration"].pixel_size.value == pytest.approx(0.2)
    assert result.scientific_level == "quantitative"


def test_direct_diffraction_cannot_use_fft_calibration_fallback(tmp_path):
    ctx = loaded(tmp_path, np.ones((16, 16)), "SAED")
    result = ctx.call(dq.resolve_calibration, {"pixel_size_nm": 0.2})
    assert result.outputs["calibration"].diffraction_scale is None
    assert result.scientific_level == "diagnostic"
    result = ctx.call(dq.resolve_calibration, {"known_d_spacing": 0.25, "measured_radius_px": 20.0})
    assert result.outputs["calibration"].diffraction_scale.value == pytest.approx(0.2)
    assert "reciprocal_image" in ctx.arrays


def test_multiple_reference_calibration_and_rectangular_fft_evidence(tmp_path):
    ctx = loaded(tmp_path, np.ones((32, 64)))
    result = ctx.call(dq.resolve_calibration, {"pixel_size_nm": 0.1}).outputs["calibration"]
    assert result.diffraction_scale is None
    assert result.transform["fft_per_axis_inv_nm_per_px"] == pytest.approx(
        {"x": 1 / 6.4, "y": 1 / 3.2}
    )
    result = ctx.call(
        dq.resolve_calibration,
        {
            "known_d_spacing_refs": [
                {"known_d_spacing": 0.4, "measured_radius_px": 10.0},
                {"known_d_spacing": 0.2, "measured_radius_px": 20.0},
            ],
            "voltage_kv": 200.0,
            "detector_pixel_size_um": 15.0,
        },
    ).outputs["calibration"]
    assert result.diffraction_scale.value == pytest.approx(0.25)
    assert result.camera_length.value > 0
    assert result.transform["reference_pair_details"]["inlier_count"] == 2


def test_scale_bar_detector_returns_review_candidate_without_committing_scale(tmp_path):
    raw = np.random.default_rng(2).normal(0.1, 0.002, size=(128, 160))
    raw[108:112, 30:80] = 1.0
    ctx = loaded(tmp_path, raw)
    result = ctx.call(dq.scalebar_candidate, {"reference_length": 5.0})
    candidate = result.outputs["scalebar_candidate"]
    assert candidate["pixel_size_nm"] == pytest.approx(0.1, abs=0.02)
    assert candidate["requires_review"] is True
    assert ctx.model("dataset", TEMDataset).pixel_size is None
    assert result.scientific_level == "candidate"
    with pytest.raises(HarnessError) as error:
        ctx.call(dq.scalebar_candidate, {"reference_length": 5.0, "roi": [0, 129, 0, 50]})
    assert error.value.code == "INVALID_ROI"


def test_preprocessing_saves_processed_and_full_background_without_touching_raw(tmp_path):
    raw = np.random.default_rng(4).normal(size=(72, 80))
    ctx = loaded(tmp_path, raw)
    original = hashlib.sha256(ctx.input_path.read_bytes()).hexdigest()
    result = ctx.call(
        dq.preprocess, {"denoise_method": "gaussian", "background_method": "gaussian"}
    )
    processed = result.outputs["processed"]
    background = result.outputs["preprocessing"].background
    assert processed.shape == background.shape == raw.shape
    assert Path(background.uri).is_file()
    assert np.isfinite(ctx.array("processed")).all()
    assert ctx.array("processed").min() >= 0
    assert ctx.array("processed").max() <= 1
    assert hashlib.sha256(ctx.input_path.read_bytes()).hexdigest() == original
    assert list(tmp_path.glob("source_*")) == []


def test_preprocessing_disable_preserves_exact_pixels(tmp_path):
    raw = np.arange(80).reshape(8, 10)
    ctx = loaded(tmp_path, raw)
    ctx.call(dq.preprocess, {"enabled": False})
    np.testing.assert_array_equal(ctx.array("processed"), raw)


def test_fft_and_odd_center_match_numpy_exactly(tmp_path):
    image = np.random.default_rng(8).normal(size=(65, 127))
    ctx = loaded(tmp_path, image)
    ctx.arrays["processed"] = image
    ctx.call(df.fft)
    np.testing.assert_allclose(ctx.array("fft_complex"), np.fft.fftshift(np.fft.fft2(image)))
    ctx.call(df.center)
    assert ctx.value("center")["x"] == 63
    assert ctx.value("center")["y"] == 32
    assert list(tmp_path.glob("source_*")) == []


def test_direct_beam_and_peak_detection_are_real_algorithms(tmp_path):
    yy, xx = np.mgrid[:96, :96]
    image = 300 * np.exp(-((xx - 47.2) ** 2 + (yy - 48.3) ** 2) / 2)
    image += 80 * np.exp(-((xx - 67.4) ** 2 + (yy - 48.1) ** 2) / 2)
    image += 60 * np.exp(-((xx - 47.3) ** 2 + (yy - 28.2) ** 2) / 2)
    ctx = loaded(tmp_path, image, "SAED")
    ctx.call(df.pattern)
    ctx.call(df.center)
    assert ctx.value("center")["x"] == pytest.approx(47.2, abs=0.4)
    assert ctx.value("center")["y"] == pytest.approx(48.3, abs=0.4)
    ctx.call(
        df.peaks,
        {
            "threshold_strategy": "relative_max",
            "threshold_value": 0.3,
            "radius_outlier_filter": False,
        },
    )
    coordinates = np.array([[v["x"], v["y"]] for v in ctx.value("peaks")])
    assert np.min(np.linalg.norm(coordinates - [67.4, 48.1], axis=1)) < 0.6
    with pytest.raises(HarnessError) as error:
        ctx.call(df.fft)
    assert error.value.code == "MODALITY_MISMATCH"


def test_manual_peaks_and_parameter_suggestions_are_registered_evidence(tmp_path):
    ctx = metric_context(tmp_path)
    ctx.call(df.peak_suggestions)
    assert ctx.value("peak_parameter_suggestions")["top_k"] > 0
    ctx.call(df.manual_peaks, {"peaks": [{"x": 127.9, "y": 63.9}]})
    assert ctx.value("peaks")[0]["x"] == 127.9
    assert ctx.value("peak_diagnostics")["method"] == "manual_reviewed_points"
    with pytest.raises(HarnessError):
        ctx.call(df.manual_peaks, {"peaks": [{"x": 128.0, "y": 20.0}]})


def test_rectangular_fft_uses_distinct_axis_scales(tmp_path):
    ctx = metric_context(tmp_path)
    result = ctx.call(df.metrics, {"require_physical": True}).outputs["diffraction"]
    # 10/128 and 5/64 cycles/pixel both represent 3.125 1/nm at 0.025 nm/pixel.
    assert [v.magnitude.value for v in result.g_vectors] == pytest.approx([3.125, 3.125])
    assert [v.value for v in result.d_spacings] == pytest.approx([0.32, 0.32])
    assert result.provenance["scale_x"] != result.provenance["scale_y"]
    assert all(v.unit == "nm" for v in result.d_spacings)


def test_common_scale_uncertainty_is_correlated_across_axes(tmp_path):
    ctx = metric_context(tmp_path)
    ctx.values["peaks"] = [{"x": 74.0, "y": 37.0, "sigma_x": 0.0, "sigma_y": 0.0}]
    ctx.values["calibration"].pixel_size.uncertainty = 0.0025  # ten percent shared scale
    result = ctx.call(df.metrics, {"center_uncertainty_px": 0.0}).outputs["diffraction"]
    magnitude = result.g_vectors[0].magnitude
    assert magnitude.uncertainty == pytest.approx(magnitude.value * 0.1)
    assert result.d_spacings[0].uncertainty == pytest.approx(result.d_spacings[0].value * 0.1)
    assert result.g_vectors[0].angle.uncertainty == 0


def test_direct_pattern_metrics_use_detector_not_real_space_scale(tmp_path):
    ctx = metric_context(tmp_path, direct=True)
    result = ctx.call(df.metrics).outputs["diffraction"]
    assert [v.value for v in result.d_spacings] == pytest.approx([0.4, 0.8])


def test_uncalibrated_metrics_are_diagnostic_and_cannot_index(tmp_path):
    ctx = metric_context(tmp_path, physical=False)
    result = ctx.call(df.metrics)
    assert result.scientific_level == "diagnostic"
    diffraction = result.outputs["diffraction"]
    assert [v.value for v in diffraction.d_spacings] == pytest.approx([12.8, 12.8])
    assert all(v.unit == "pixel" for v in diffraction.d_spacings)
    assert all(v.magnitude.unit == "1/pixel" for v in diffraction.g_vectors)
    with pytest.raises(NeedsInput):
        ctx.call(df.index)
    with pytest.raises(NeedsInput):
        ctx.call(df.metrics, {"require_physical": True})


def test_uncalibrated_detector_reports_offsets_and_withholds_d_spacing(tmp_path):
    ctx = metric_context(tmp_path, direct=True, physical=False)
    ctx.values["peaks"] = [{"x": 61.0, "y": 36.0}, {"x": 64.0, "y": 32.0}]
    result = ctx.call(df.metrics)
    diffraction = result.outputs["diffraction"]
    assert result.scientific_level == "diagnostic"
    assert diffraction.d_spacings == [] and diffraction.provenance["d_by_peak"] == [None, None]
    assert diffraction.provenance["coordinate_space"] == "diffraction_detector"
    offset, center = diffraction.g_vectors
    assert [offset.gx.value, offset.gy.value, offset.magnitude.value] == [-3.0, 4.0, 5.0]
    assert offset.angle.value == pytest.approx(np.degrees(np.arctan2(4, -3)))
    assert center.magnitude.value == 0
    for vector in diffraction.g_vectors:
        assert all(m.unit == "pixel" for m in (vector.gx, vector.gy, vector.magnitude))
    with pytest.raises(NeedsInput):
        ctx.call(df.index)
    with pytest.raises(NeedsInput) as error:
        ctx.call(df.metrics, {"require_physical": True})
    assert error.value.details["fields"] == [{"name": "diffraction_scale_inv_nm_per_px"}]


def test_missing_uncertainty_is_not_reported_as_exact_zero(tmp_path):
    ctx = metric_context(tmp_path)
    result = ctx.call(df.metrics).outputs["diffraction"]
    assert result.g_vectors[0].magnitude.uncertainty is None
    assert result.d_spacings[0].uncertainty is None
    assert any("uncertainty is unknown" in warning for warning in result.warnings)


def test_metrics_reject_bad_units_uncertainty_peaks(tmp_path):
    ctx = metric_context(tmp_path)
    ctx.values["calibration"].pixel_size.unit = "pm/pixel"
    with pytest.raises(HarnessError) as error:
        ctx.call(df.metrics)
    assert error.value.code == "INVALID_CALIBRATION"
    ctx.values["calibration"].pixel_size.unit = "nm/pixel"
    ctx.values["calibration"].pixel_size.uncertainty = -0.1
    with pytest.raises(HarnessError):
        ctx.call(df.metrics)
    ctx.values["calibration"].pixel_size.uncertainty = None
    ctx.values["peaks"][0]["x"] = float("nan")
    with pytest.raises(HarnessError) as error:
        ctx.call(df.metrics)
    assert error.value.code == "INVALID_PEAKS"


def test_indexing_uses_calibrated_local_candidate(tmp_path):
    ctx = metric_context(tmp_path, direct=True)
    ctx.values["peaks"] = [
        {"x": 74.0, "y": 32.0},
        {"x": 64.0, "y": 42.0},
        {"x": 74.0, "y": 42.0},
        {"x": 54.0, "y": 32.0},
    ]
    ctx.call(df.metrics)
    with pytest.raises(NeedsInput):
        ctx.call(df.index)
    result = ctx.call(
        df.index,
        {
            "candidate_phase_params": [{"name": "simple_cubic", "a": 0.4, "b": 0.4, "c": 0.4}],
            "strict_mode": False,
        },
    )
    assert result.outputs["diffraction"].phase_candidates[0].name == "simple_cubic"
    assert result.outputs["diffraction"].indexed_spots
    assert result.scientific_level == "candidate"


def test_index_rejects_remote_cif_and_invalid_cell(tmp_path):
    ctx = metric_context(tmp_path)
    ctx.call(df.metrics)
    with pytest.raises(HarnessError) as error:
        ctx.call(df.index, {"cif_candidate_paths": ["https://example.com/a.cif"]})
    assert error.value.code == "INVALID_CANDIDATE"
    with pytest.raises(HarnessError) as error:
        ctx.call(
            df.index,
            {
                "candidate_phase_params": [
                    {
                        "name": "bad",
                        "a": 0.4,
                        "b": 0.4,
                        "c": 0.4,
                        "alpha": 1.0,
                        "beta": 179.0,
                        "gamma": 1.0,
                    }
                ]
            },
        )
    assert error.value.code == "INVALID_PARAMETERS"


@pytest.mark.parametrize("source", ["builtin", "local"])
def test_builtin_lookup_produces_complete_validated_structure(tmp_path, source):
    ctx = Context(tmp_path / "run")
    result = ctx.call(df.lookup, {"source": source, "formula": "Si"})
    assert result.outputs["phase_candidates"][0]["a"] == pytest.approx(0.5431)
    assert len(result.outputs["phase_candidates"][0]["atom_sites"]) == 8
    assert result.outputs["phase_lookup"]["network_used"] is False
    with pytest.raises(NeedsInput):
        ctx.call(df.lookup, {"source": source})


@pytest.mark.parametrize("source", ["cod", "optimade", "materials_project"])
def test_online_sources_require_two_explicit_network_gates(tmp_path, source):
    ctx = Context(tmp_path / "run")
    for request, params in (
        ({}, {}),
        ({}, {"allow_network": True}),
        ({"network_enabled": True}, {}),
    ):
        ctx.request = request
        with pytest.raises(HarnessError) as error:
            ctx.call(df.lookup, {"source": source, "formula": "Si", **params})
        assert error.value.code == "NETWORK_DISABLED"


@pytest.mark.parametrize("source", ["cod", "optimade", "materials_project"])
def test_online_lookup_contract_validates_output_without_live_network(
    tmp_path, monkeypatch, source
):
    from autotem.diffraction.phase_lookup import PhaseLookupResult

    ctx = Context(tmp_path / "run", request={"network_enabled": True})
    supplied = {}
    marker = "test-credential-marker"
    monkeypatch.setattr(
        df, "resolve_materials_project_api_key", lambda: (marker, {"configured": True})
    )
    monkeypatch.setattr(df.importlib, "import_module", lambda name: object())

    def provider(**kwargs):
        supplied.update(kwargs)
        return PhaseLookupResult(
            candidates=[{"name": "local", "a": 0.4, "b": 0.4, "c": 0.4}],
            warnings=[marker],
            provenance={"nested": marker},
        )

    monkeypatch.setattr(df, "lookup_phase_candidates", provider)
    result = ctx.call(
        df.lookup, {"source": source, "formula": "Si", "allow_network": True, "timeout_s": 10}
    )
    assert supplied["timeout_s"] == 10
    assert supplied["materials_project_api_isolate"] is False
    assert result.outputs["phase_lookup"]["network_used"] is True
    if source == "materials_project":
        assert marker not in str(result.outputs) + str(result.warnings)


def test_materials_project_requires_external_credential_configuration(tmp_path, monkeypatch):
    ctx = Context(tmp_path / "run", request={"network_enabled": True})
    monkeypatch.setattr(df, "resolve_materials_project_api_key", lambda: (None, {}))
    with pytest.raises(NeedsInput) as error:
        ctx.call(df.lookup, {"source": "materials_project", "formula": "Si", "allow_network": True})
    field = error.value.details["fields"][0]
    assert field["action"] == "configure-mp-api"
    assert field["secret_in_run_parameters"] is False
    with pytest.raises(HarnessError) as error:
        ctx.call(
            df.lookup, {"source": "materials_project", "formula": "Si", "api_key": "forbidden"}
        )
    assert error.value.code == "INVALID_PARAMETERS"


def test_local_cif_lookup_snapshots_input_and_normalizes_angstrom_to_nm(tmp_path):
    source = tmp_path / "simple.cif"
    source.write_text(
        """data_simple
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_space_group_name_H-M_alt 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si1 Si 0.0 0.0 0.0
""",
        encoding="utf-8",
    )
    original = source.read_bytes()
    ctx = Context(tmp_path / "run")
    result = ctx.call(df.lookup, {"source": "local", "cif_candidate_paths": [str(source)]})
    candidate = result.outputs["phase_candidates"][0]
    assert candidate["a"] == pytest.approx(0.4)
    assert candidate["atom_sites"][0]["element"] == "Si"
    evidence = result.outputs["phase_lookup"]["candidate_snapshots"][0]
    snapshot = Path(evidence["snapshot"])
    assert snapshot.read_bytes() == original
    assert evidence["sha256"] == hashlib.sha256(original).hexdigest()
    source.write_text("changed source")
    assert snapshot.read_bytes() == original
    assert ctx.value("candidate_cif_0")["text"].startswith("data_simple")
    with pytest.raises(HarnessError) as error:
        ctx.call(df.lookup, {"source": "builtin", "cif_candidate_paths": [str(source)]})
    assert error.value.code == "INVALID_PARAMETERS"


@pytest.mark.parametrize("source", ["builtin", "local", "cod", "optimade", "materials_project"])
def test_lookup_source_schemas_reject_invalid_fields_and_budgets(source):
    for parameters in ({"timeout_s": 31}, {"api_key": "forbidden"}, {"not_supported": True}):
        with pytest.raises(ValidationError):
            df.LookupParameters.model_validate({"source": source, **parameters})


def test_provider_failure_is_structured_and_does_not_expose_credentials(tmp_path, monkeypatch):
    ctx = Context(tmp_path / "run", request={"network_enabled": True})

    def failing_provider(**kwargs):
        raise RuntimeError("provider echoed a credential")

    monkeypatch.setattr(df, "lookup_phase_candidates", failing_provider)
    with pytest.raises(HarnessError) as error:
        ctx.call(df.lookup, {"source": "cod", "formula": "Si", "allow_network": True})
    assert error.value.code == "LOOKUP_FAILED"
    assert "credential" not in str(error.value.details)


def test_materials_project_import_failures_distinguish_missing_and_broken(tmp_path, monkeypatch):
    ctx = Context(tmp_path / "run", request={"network_enabled": True})
    monkeypatch.setattr(df, "resolve_materials_project_api_key", lambda: ("fixture", {}))

    def missing(name):
        raise ModuleNotFoundError("missing backend", name="mp_api")

    monkeypatch.setattr(df.importlib, "import_module", missing)
    with pytest.raises(HarnessError) as error:
        ctx.call(df.lookup, {"source": "materials_project", "formula": "Si", "allow_network": True})
    assert error.value.code == "DEPENDENCY_UNAVAILABLE"

    def broken(name):
        raise ModuleNotFoundError("missing transitive dependency", name="transitive")

    monkeypatch.setattr(df.importlib, "import_module", broken)
    with pytest.raises(HarnessError) as error:
        ctx.call(df.lookup, {"source": "materials_project", "formula": "Si", "allow_network": True})
    assert error.value.code == "DEPENDENCY_BROKEN"


def test_index_can_consume_lookup_artifact_in_same_run(tmp_path, monkeypatch):
    from autotem.diffraction.phase_lookup import PhaseLookupResult

    ctx = metric_context(tmp_path, direct=True)
    ctx.values["peaks"] = [
        {"x": 74.0, "y": 32.0},
        {"x": 64.0, "y": 42.0},
        {"x": 74.0, "y": 42.0},
        {"x": 54.0, "y": 32.0},
    ]
    ctx.call(df.metrics)
    monkeypatch.setattr(
        df,
        "lookup_phase_candidates",
        lambda **kwargs: PhaseLookupResult(
            candidates=[{"name": "simple_cubic", "a": 0.4, "b": 0.4, "c": 0.4}],
            warnings=[],
            provenance={"source": "local_fixture"},
        ),
    )
    ctx.call(df.lookup, {"formula": "Si"})
    result = ctx.call(df.index, {"strict_mode": False})
    assert result.outputs["diffraction"].phase_candidates[0].name == "simple_cubic"


def simulated_evidence():
    candidate = PhaseCandidate(
        name="local",
        identifier="local",
        lattice_parameters={
            "a": 0.4,
            "b": 0.4,
            "c": 0.4,
            "atom_sites": [{"element": "Si", "abc": [0, 0, 0], "occupancy": 1}],
        },
    )
    spots = [
        SimulatedSpot(
            gx=Measurement(value=x, unit="1/nm"),
            gy=Measurement(value=y, unit="1/nm"),
            intensity=1.0,
        )
        for x, y in ((2.5, 0), (0, 1.25))
    ]
    pattern = SimulatedPattern(
        kind="diffraction",
        structure_id="local",
        spots=spots,
        data=ArrayReference(shape=(8, 8), inline=np.ones((8, 8)).tolist()),
    )
    return candidate, SimulationResult(patterns=[pattern])


def test_simulation_requires_structure_and_dependencies(tmp_path, monkeypatch):
    ctx = metric_context(tmp_path, direct=True)
    ctx.call(df.metrics)
    ctx.values["diffraction"].phase_candidates = [PhaseCandidate(name="not_a_structure")]
    with pytest.raises(NeedsInput):
        ctx.call(df.simulate)
    candidate, _ = simulated_evidence()
    ctx.values["diffraction"].phase_candidates = [candidate]
    monkeypatch.setattr(df, "_dependencies", lambda: ["diffsims"])
    with pytest.raises(HarnessError) as error:
        ctx.call(df.simulate)
    assert error.value.code == "DEPENDENCY_UNAVAILABLE"


def test_simulation_forces_safe_parameters_and_persists_full_arrays(tmp_path, monkeypatch):
    ctx = metric_context(tmp_path, direct=True)
    ctx.call(df.metrics)
    candidate, simulation = simulated_evidence()
    ctx.values["diffraction"].phase_candidates = [candidate]
    monkeypatch.setattr(df, "_dependencies", lambda: [])
    supplied = {}

    def adapter(diffraction, *, parameters):
        supplied.update(parameters)
        return simulation

    monkeypatch.setattr(df, "simulate_from_diffraction", adapter)
    result = ctx.call(df.simulate)
    assert supplied["fallback_single_atom"] is False
    assert supplied["phase_match_enabled"] is False
    assert supplied["diffsims_timeout_s"] == 30
    assert Path(result.outputs["simulation"].patterns[0].data.uri).is_file()
    monkeypatch.setattr(
        df,
        "simulate_from_diffraction",
        lambda *a, **k: SimulationResult(provenance={"adapter": {"reason": "timeout"}}),
    )
    with pytest.raises(HarnessError) as error:
        ctx.call(df.simulate)
    assert error.value.code == "SIMULATION_TIMEOUT"


@pytest.mark.parametrize(
    "reason,expected_code",
    [
        ("import_failed", "DEPENDENCY_BROKEN"),
        ("dependency_missing", "DEPENDENCY_UNAVAILABLE"),
        ("timeout", "SIMULATION_TIMEOUT"),
        ("generator_failed", "SIMULATION_FAILED"),
        ("worker_failed", "SIMULATION_FAILED"),
        ("no_patterns", "SIMULATION_FAILED"),
    ],
)
def test_simulation_classifies_empty_adapter_evidence(
    tmp_path, monkeypatch, reason, expected_code
):
    ctx = metric_context(tmp_path, direct=True)
    ctx.call(df.metrics)
    candidate, _ = simulated_evidence()
    ctx.values["diffraction"].phase_candidates = [candidate]
    monkeypatch.setattr(df, "_dependencies", lambda: [])
    warning = "Real backend failure detail must remain available."
    monkeypatch.setattr(
        df,
        "simulate_from_diffraction",
        lambda *a, **k: SimulationResult(
            warnings=[warning], provenance={"adapter": {"reason": reason}}
        ),
    )
    with pytest.raises(HarnessError) as error:
        ctx.call(df.simulate)
    assert error.value.code == expected_code
    assert error.value.details["provenance"]["adapter"]["reason"] == reason
    assert error.value.details["warnings"] == [warning]


def test_matching_computes_from_physical_measured_and_simulated_spots(tmp_path):
    ctx = metric_context(tmp_path, direct=True)
    ctx.call(df.metrics)
    candidate, simulation = simulated_evidence()
    ctx.values["diffraction"].phase_candidates = [candidate]
    ctx.values["simulation"] = simulation
    result = ctx.call(df.match)
    assert result.outputs["diffraction"].phase_candidates[0].matched_spot_count == 2
    assert result.outputs["simulation"].comparison_metrics
    assert result.outputs["diffraction"].provenance["phase_matching"]["active"] is True
    ctx.values["simulation"] = SimulationResult()
    with pytest.raises(HarnessError) as error:
        ctx.call(df.match)
    assert error.value.code == "MATCHING_NOT_EVALUATED"


@pytest.mark.parametrize(
    "handler,params",
    [
        (dq.resolve_calibration, {"pixel_size_nm": float("nan")}),
        (dq.resolve_calibration, {"known_d_spacing": 0.2}),
        (dq.preprocess, {"clip_percentile_low": 90.0, "clip_percentile_high": 10.0}),
        (dq.preprocess, {"enabled": "false"}),
        (df.center, {"center_mode": "manual"}),
        (df.peaks, {"threshold_strategy": "quantile", "threshold_value": 2.0}),
        (df.metrics, {"center_uncertainty_px": -1.0}),
        (df.simulate, {"fallback_single_atom": True}),
        (df.simulate, {"zone_axis_candidates": [[0, 0, 0]]}),
        (df.match, {"previous_score_weight": float("inf")}),
    ],
)
def test_parameters_fail_before_accessing_artifacts(tmp_path, handler, params):
    with pytest.raises(HarnessError) as error:
        Context(tmp_path / "run").call(handler, params)
    assert error.value.code == "INVALID_PARAMETERS"


def test_real_store_and_context_round_trip_selected_raw_dataset(tmp_path):
    from autotem.harness.store import Store, encode

    source = tmp_path / "raw.npy"
    raw = np.arange(128, dtype=np.float32).reshape(8, 16)
    np.save(source, raw)
    store = Store(tmp_path / "harness")
    run_id = uuid.uuid4().hex
    with store.connect() as db:
        db.execute(
            "INSERT INTO runs(run_id,request_hash,request,status,created,updated,input_path) "
            "VALUES(?,?,?,'running',?,?,?)",
            (run_id, "hash", encode({}), time.time(), time.time(), str(source)),
        )
        db.execute(
            "INSERT INTO steps(run_id,step_id,operation_id,status) VALUES(?,?,?,'running')",
            (run_id, "load", "input.load"),
        )
    ctx = OperationContext(store, run_id, "load", {"modality": "SAED"})
    result = dq.load_input(ctx, {})
    ctx.emit_json("dataset", result.outputs["dataset"])
    ctx.emit_json("reciprocal_context", result.outputs["reciprocal_context"])
    store.commit_step(run_id, "load", {})
    reader = OperationContext(store, run_id, "downstream", {})
    np.testing.assert_array_equal(reader.array("raw"), raw)
    np.testing.assert_array_equal(reader.array("reciprocal_image"), raw)
    assert reader.model("dataset", TEMDataset).data.shape == raw.shape
    assert store.artifact(store.role(run_id, "raw"))["kind"] == "array"
