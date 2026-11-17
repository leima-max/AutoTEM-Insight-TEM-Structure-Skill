"""Extended disk checks for diagnostic strain and direct reciprocal-space patterns."""

import numpy as np

from .cases import ATOM_REFERENCE, DIRECT_PEAKS, DIRECT_SCALE, GPA_REFERENCE, PERIOD_PX
from .evidence import RunEvidence, require


def check_gpa(data: RunEvidence) -> list[dict]:
    from autotem.strain.gpa import compute_gpa_fields

    raw = data.array("raw")
    direct = compute_gpa_fields(
        raw,
        reference_region=tuple(GPA_REFERENCE),
        g_vectors=[[1 / PERIOD_PX, 0], [0, 1 / PERIOD_PX]],
    )
    result = data.value("gpa")
    require(result["reference_region"] == GPA_REFERENCE, "GPA reference ROI changed")
    mask = data.array("gpa.quality_mask")
    np.testing.assert_array_equal(mask, direct.quality_mask)
    require(mask.any() and (~mask).any(), "Synthetic GPA must exercise valid and invalid regions")
    y0, y1, x0, x1 = GPA_REFERENCE
    require(mask[y0:y1, x0:x1].sum() >= 9, "Insufficient valid GPA reference")
    for field in (
        "displacement_x",
        "displacement_y",
        "strain_xx",
        "strain_yy",
        "strain_xy",
        "rotation_xy",
        "uncertainty",
    ):
        role = f"gpa.{field}"
        registered = data.array(role)
        require(
            registered.shape == raw.shape and result[field]["inline"] is None,
            f"Incomplete GPA {field}",
        )
        require(np.isfinite(registered[mask]).all(), f"Non-finite GPA {field} in the valid region")
        np.testing.assert_allclose(registered, getattr(direct, field), equal_nan=True, atol=1e-10)
    comparison = data.value("gpa_comparison")
    require(
        comparison["diagnosis"]["max_strain_mae_percent"] == 0,
        "Identical GPA configurations produced different strain",
    )
    return [
        {
            "claim": "Pixels in accepted GPA region",
            "value": int(mask.sum()),
            "unit": "pixels",
            "artifact_id": data.index["gpa.quality_mask"],
            "limitation": "Diagnostic strain",
        }
    ]


def check_atom_strain(data: RunEvidence) -> list[dict]:
    from autotem.strain.atom_strain import compute_atom_strain_fields

    coords = np.asarray([[a["y"], a["x"]] for a in data.value("refined_atoms")["atoms"]])
    direct = compute_atom_strain_fields(
        coords, data.array("raw").shape, reference_region=tuple(ATOM_REFERENCE)
    )
    payload = data.value("atom_strain")
    np.testing.assert_array_equal(data.array("atom_strain.coordinates_yx"), coords)
    mask = data.array("atom_strain.valid_atom_mask")
    np.testing.assert_array_equal(mask, direct.valid_atom_mask)
    require(
        mask.any() and payload["reference_region"] == ATOM_REFERENCE,
        "Missing valid atom strain or chosen reference",
    )
    for field in ("atom_strain_xx", "atom_strain_yy", "atom_strain_xy"):
        values = data.array(f"atom_strain.{field}")
        require(np.isfinite(values[mask]).all(), "Invalid authoritative atom values")
        np.testing.assert_allclose(values, getattr(direct, field), equal_nan=True, atol=1e-10)
    require(
        payload["provenance"]["map_semantics"].startswith("display_interpolation"),
        "Image-space display maps lost their interpretation",
    )
    np.testing.assert_allclose(
        data.array("atom_strain.strain_xx_map"), direct.strain_xx_map, equal_nan=True, atol=1e-10
    )
    return [
        {
            "claim": "Atoms with a valid local strain estimate",
            "value": int(mask.sum()),
            "unit": "columns",
            "artifact_id": data.index["atom_strain.valid_atom_mask"],
            "limitation": "Diagnostic per-atom estimate; interpolated maps are for display",
        }
    ]


def check_direct_pattern(data: RunEvidence) -> list[dict]:
    np.testing.assert_array_equal(data.array("raw"), data.array("reciprocal_image"))
    require(
        "fft_complex" not in data.index
        and data.value("reciprocal_context")["kind"] == "direct_diffraction",
        "Unexpected FFT",
    )
    result = data.value("diffraction")
    np.testing.assert_allclose(result["center"], [40, 32], rtol=0, atol=0.25)
    require(
        result["provenance"]["method"] == "direct_pattern_detector_scale",
        "Wrong detector calibration branch",
    )
    found = np.array([(peak["y"], peak["x"]) for peak in result["peaks"]])
    observed = []
    for y, x in DIRECT_PEAKS:
        offsets = np.linalg.norm(found - [y, x], axis=1)
        index = int(np.argmin(offsets))
        require(offsets[index] <= 0.25, "Missing detector peak")
        spacing = result["provenance"]["d_by_peak"][index]
        require(spacing["unit"] == "nm", "Direct detector spacing unit")
        radius = np.linalg.norm(found[index] - np.asarray(result["center"])[::-1])
        np.testing.assert_allclose(spacing["value"], 1 / (radius * DIRECT_SCALE), rtol=1e-12)
        observed.append(spacing["value"])
    return [
        {
            "claim": "Detector peak spacings",
            "value": observed,
            "unit": "nm",
            "method": "calibrated direct pattern",
            "artifact_id": data.index["diffraction"],
        }
    ]
