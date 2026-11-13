import numpy as np
import pytest

from tools.acceptance import verify_public_diffraction as public


def test_wrong_archive_is_rejected_before_extraction(tmp_path):
    source = tmp_path / "cau-36.zip"
    source.write_bytes(b"not the publicly pinned archive")
    with pytest.raises(ValueError, match="Wrong CAU-36 archive"):
        public.inputs(source, tmp_path)
    assert list(tmp_path.iterdir()) == [source]


def test_frame_identity_requires_dtype_shape_and_every_pixel():
    source = np.arange(516 * 516, dtype="<i4").reshape(516, 516)
    digest = public.hash_bytes(source.tobytes())
    assert public.pixels(source, "<i4", digest)["pixels_sha256"] == digest
    for changed in (source.view("<u4"), source.reshape(258, 1032)):
        with pytest.raises(ValueError, match="Wrong detector layout"):
            public.pixels(changed, "<i4", digest)
    source[-1, -1] -= 1
    with pytest.raises(ValueError, match="Detector pixels differ"):
        public.pixels(source, "<i4", digest)


def test_uncalibrated_detector_offsets_reject_old_units_and_inferred_d():
    coords = np.array([[6.0, 8.0]])
    vector = {
        "source_peak_index": 0,
        "gx": {"unit": "pixel", "value": 3},
        "gy": {"unit": "pixel", "value": 4},
        "magnitude": {"unit": "pixel", "value": 5},
        "angle": {"unit": "deg", "value": np.degrees(np.arctan2(4, 3))},
    }
    result = {
        "g_vectors": [vector],
        "provenance": {
            "physical": False,
            "d_by_peak": [None],
            "coordinate_space": "diffraction_detector",
        },
        "d_spacings": [],
    }
    center = {"x": 3.0, "y": 4.0}
    assert public.detector_geometry(result, coords, center) == [5, 5]
    with pytest.raises(ValueError, match="cannot yield d-spacings"):
        public.detector_geometry(
            {**result, "d_spacings": [{"unit": "pixel", "value": 0.2}]}, coords, center
        )
    vector["magnitude"]["unit"] = "1/pixel"
    with pytest.raises(ValueError, match="must use detector pixels"):
        public.detector_geometry(result, coords, center)
