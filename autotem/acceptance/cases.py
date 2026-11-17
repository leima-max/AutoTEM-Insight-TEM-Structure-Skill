"""Deterministic analytic inputs and explicit analysis targets for release checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PIXEL_NM = 0.02
PERIOD_PX = 8
CORE_CASES = ("J01", "J02", "J06")
DIRECT_SCALE = 0.25
DIRECT_PEAKS = ((24, 40), (40, 40), (32, 30), (32, 50))
GPA_REFERENCE = [16, 48, 16, 48]
ATOM_REFERENCE = [8, 36, 8, 68]
CASE_MODALITIES = {
    "J01": "HRTEM",
    "J02": "HAADF-STEM",
    "J03": "HRTEM",
    "J04": "HAADF-STEM",
    "J05-SAED": "SAED",
    "J05-NBED": "NBED",
    "J06": "HRTEM",
}
REQUESTS = {
    "J01": "Measure the periodic lattice spacings in nm and index against synthetic-cubic.",
    "J02": "Refine columns on raw and measure nearest spacing in nm and relative intensity.",
    "J03": "Measure diagnostic GPA in the valid region and compare two identical GPA settings.",
    "J04": "Use raw-refined columns and the reference ROI to estimate per-atom diagnostic strain.",
    "J05-SAED": "Measure d-spacings from the direct diffraction pattern at 0.25 nm^-1/pixel.",
    "J05-NBED": "Measure d-spacings from the four supplied detector peaks at 0.25 nm^-1/pixel.",
    "J06": "Measure lattice spacings in nm. Ask for the acquisition scale if it is missing.",
}


def write_fixture(case_id: str, directory: Path) -> tuple[Path, dict]:
    """Write a non-square y,x image. The closed-form inputs do not require a random seed."""
    yy, xx = np.mgrid[:64, :80]
    modality = CASE_MODALITIES[case_id]
    calibration = {"pixel_size_nm": PIXEL_NM}
    if modality == "HRTEM":
        raw = np.cos(2 * np.pi * xx / PERIOD_PX) + np.cos(2 * np.pi * yy / PERIOD_PX)
        formula = "cos(2*pi*x/8) + cos(2*pi*y/8)"
        expected = {"axis_period_px": PERIOD_PX, "axis_period_nm": PERIOD_PX * PIXEL_NM}
    elif modality == "HAADF-STEM":
        raw = np.full(yy.shape, 10.0)
        centers = [(y + 0.21, x - 0.34) for y in range(10, 51, 8) for x in range(12, 61, 8)]
        for index, (y, x) in enumerate(centers):
            amplitude = 30.0 + 20.0 * (index % 2)
            raw += amplitude * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.1**2))
        formula = "10 + alternating 30/50 amplitude Gaussian columns, sigma 1.1 pixels"
        expected = {"centers_yx": centers, "nearest_spacing_nm": PERIOD_PX * PIXEL_NM}
    else:
        raw = np.exp(-((yy - 32) ** 2 + (xx - 40) ** 2) / 3)
        for y, x in DIRECT_PEAKS:
            raw += 0.3 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 3)
        formula = "Gaussian direct beam centered at (32,40) and four Gaussian detector peaks"
        expected = {"peaks_yx": DIRECT_PEAKS, "spacing_nm": [0.5, 0.5, 0.4, 0.4]}
        calibration = {"diffraction_scale_inv_nm_per_px": DIRECT_SCALE}
    suffix = {"J02": ".tif", "J06": ".tiff"}.get(case_id, ".npy")
    source = directory / f"{case_id} 合成 {modality}{suffix}"
    if suffix == ".npy":
        raw = raw.astype("<f8")
        np.save(source, raw, allow_pickle=False)
    else:
        raw = raw.astype("<f4")
        Image.fromarray(raw).save(source)
    if case_id == "J06":
        calibration = {"initial_state": "unknown", "user_reply_pm_per_pixel": 20}
    return source, {
        "case_id": case_id,
        "source": "autotem.acceptance.cases.write_fixture",
        "license": "Generated analytic data; no external dataset or third-party data license.",
        "formula": formula,
        "seed": None,
        "modality": CASE_MODALITIES[case_id],
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "axes": ["y", "x"],
        "selection": {},
        "calibration": {**calibration, "source": "analytic fixture definition"},
        "user_request": REQUESTS[case_id],
        "expected_status": "completed",
        "expected": expected,
        "scientific_accuracy": "not_evaluated",
    }


def read_fixture(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.array(image)


def plan_for(case_id: str, source: Path) -> dict:
    modality = CASE_MODALITIES[case_id]
    calibration = (
        {"diffraction_scale_inv_nm_per_px": DIRECT_SCALE}
        if modality in {"SAED", "NBED"}
        else {"pixel_size_nm": PIXEL_NM}
    )
    if case_id == "J06":
        calibration = {}
    operations = [
        ("input.load", {}),
        ("quality.assess", {}),
        ("calibration.resolve", {**calibration, "require_physical": True}),
    ]
    if case_id in ("J01", "J02", "J04", "J06"):
        operations.append(
            (
                "preprocessing.apply",
                {
                    "denoise_method": "none" if modality == "HRTEM" else "gaussian",
                },
            )
        )
    if case_id in ("J01", "J06"):
        operations += [
            ("diffraction.fft", {}),
            ("diffraction.center", {}),
            ("diffraction.peaks", {"threshold_strategy": "relative_max", "threshold_value": 0.15}),
            ("diffraction.metrics", {"require_physical": True}),
        ]
    if case_id == "J01":
        operations.append(
            (
                "diffraction.index",
                {
                    "candidate_phase_params": [
                        {"name": "synthetic-cubic", "a": 0.16, "b": 0.16, "c": 0.16}
                    ]
                },
            )
        )
    elif modality == "HAADF-STEM":
        operations += [
            ("atomic.detect", {"image_role": "processed"}),
            ("atomic.refine", {"method": "gaussian"}),
        ]
    if case_id == "J02":
        operations += [
            ("atomic.distances", {"output_unit": "nm"}),
            (
                "atomic.intensity",
                {
                    "integration_radius": 1.5,
                    "background_inner_radius": 2.0,
                    "background_outer_radius": 3.0,
                },
            ),
        ]
    elif case_id == "J03":
        gpa = {
            "reference_region": GPA_REFERENCE,
            "g_vector_unit": "1/pixel",
            "g_vectors": [[1 / PERIOD_PX, 0.0], [0.0, 1 / PERIOD_PX]],
        }
        operations += [
            ("strain.gpa", gpa),
            ("strain.gpa_compare", {"reference_config": gpa, "candidate_config": gpa}),
        ]
    elif case_id == "J04":
        operations.append(("strain.atom", {"reference_region": ATOM_REFERENCE}))
    elif modality in {"SAED", "NBED"}:
        operations += [
            ("diffraction.pattern", {}),
            ("diffraction.center", {}),
            ("diffraction.peaks.suggest", {}),
        ]
        if modality == "SAED":
            operations.append(
                (
                    "diffraction.peaks",
                    {"threshold_strategy": "relative_max", "threshold_value": 0.15},
                )
            )
        else:
            operations.append(
                (
                    "diffraction.peaks.manual",
                    {
                        "peaks": [{"x": x, "y": y} for y, x in DIRECT_PEAKS],
                    },
                )
            )
        operations.append(("diffraction.metrics", {"require_physical": True}))
    steps = []
    for operation_id, parameters in operations:
        steps.append(
            {
                "id": operation_id.replace(".", "_"),
                "operation_id": operation_id,
                "depends_on": [steps[-1]["id"]] if steps else [],
                "parameters": parameters,
            }
        )
    return {
        "input_path": str(source),
        "modality": CASE_MODALITIES[case_id],
        "steps": steps,
    }
