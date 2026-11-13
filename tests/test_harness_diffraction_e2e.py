"""Disk-to-report checks of FFT and direct pattern domain routing."""

from __future__ import annotations

import numpy as np
import pytest

from autotem.harness import Harness


@pytest.mark.parametrize("modality", ["HRTEM", "SAED", "NBED"])
def test_disk_reciprocal_chain_preserves_units_arrays_and_report(tmp_path, modality):
    yy, xx = np.mgrid[:64, :80]
    if modality == "HRTEM":
        raw = np.cos(2 * np.pi * xx / 8) + np.cos(2 * np.pi * yy / 8)
    else:
        raw = np.exp(-((yy - 32) ** 2 + (xx - 40) ** 2) / 3)
        for y, x in [(24, 40), (40, 40), (32, 30), (32, 50)]:
            raw += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 3) * 0.3
    source = tmp_path / "二维 input.npy"
    np.save(source, raw)
    steps = [
        {"id": "load", "operation_id": "input.load"},
        {
            "id": "calibration",
            "operation_id": "calibration.resolve",
            "parameters": {"pixel_size_nm": 0.02}
            if modality == "HRTEM"
            else {"diffraction_scale_inv_nm_per_px": 0.25},
        },
    ]
    if modality == "HRTEM":
        steps += [
            {
                "id": "preprocess",
                "operation_id": "preprocessing.apply",
                "parameters": {"denoise_method": "none"},
            },
            {"id": "fft", "operation_id": "diffraction.fft"},
        ]
    else:
        steps.append({"id": "pattern", "operation_id": "diffraction.pattern"})
    steps += [
        {"id": "center", "operation_id": "diffraction.center"},
        {
            "id": "peaks",
            "operation_id": "diffraction.peaks",
            "parameters": {"threshold_strategy": "relative_max", "threshold_value": 0.15},
        },
        {
            "id": "metrics",
            "operation_id": "diffraction.metrics",
            "parameters": {"require_physical": True},
        },
        {
            "id": "index",
            "operation_id": "diffraction.index",
            "parameters": {
                "candidate_phase_params": [
                    {"name": "synthetic-cubic", "a": 0.16, "b": 0.16, "c": 0.16}
                ]
            },
        },
    ]
    for previous, step in zip(steps, steps[1:], strict=False):
        step["depends_on"] = [previous["id"]]
    harness = Harness(tmp_path / "runs")
    state = harness.submit_analysis(
        {"input_path": str(source), "modality": modality, "steps": steps}, asynchronous=False
    )
    assert state["status"] == "completed", state
    artifacts = harness.list_artifacts(state["run_id"])
    roles = {a["role"] for a in artifacts}
    assert ("fft_complex" in roles) == (modality == "HRTEM")
    diffraction = harness.store.load_json(harness.store.role(state["run_id"], "diffraction"))
    assert diffraction["g_vectors"]
    assert all(g["magnitude"]["unit"] == "1/nm" for g in diffraction["g_vectors"])
    snapshot = next(a for a in artifacts if a["role"] == "raw")
    np.testing.assert_array_equal(harness.store.load_array(snapshot["artifact_id"]), raw)
    report = harness.export_report(state["run_id"])
    assert report["report_type"] == "analysis"
