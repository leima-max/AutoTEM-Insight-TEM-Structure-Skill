"""A4 wrappers retain honest missing-metric and result-role boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from autotem.harness import Harness, HarnessError
from autotem.harness.registry import default_registry


@pytest.mark.parametrize("operation", ["validation.evaluate", "benchmark.evaluate"])
def test_real_registered_metrics_do_not_pass_missing_criteria(tmp_path, operation):
    path = tmp_path / "input.npy"
    np.save(path, np.zeros((16, 16), np.float32))
    harness = Harness(tmp_path / "runs")
    state = harness.submit_analysis(
        {
            "input_path": str(path),
            "steps": [
                {"id": "load", "operation_id": "input.load"},
                {
                    "id": "measure",
                    "operation_id": operation,
                    "parameters": {
                        "acceptance_criteria": [{"metric": "missing_accuracy", "threshold": 0.9}]
                    },
                },
            ],
        },
        asynchronous=False,
    )
    assert state["status"] == "completed", state
    role = "validation" if operation.startswith("validation") else "benchmark"
    payload = harness.store.load_json(harness.store.role(state["run_id"], role))
    assert payload["acceptance"]["all_passed"] is False
    assert payload["acceptance"]["unevaluated"] == 1
    registry = default_registry()
    with pytest.raises(HarnessError):
        registry.validate_parameters(registry.get(operation), {"threshold": "invalid"})


def test_dependency_diagnostics_are_registered_real_health_probes(tmp_path):
    path = tmp_path / "input.npy"
    np.save(path, np.zeros((8, 8), np.float32))
    harness = Harness(tmp_path / "runs")
    state = harness.submit_analysis(
        {
            "input_path": str(path),
            "steps": [{"id": "dependencies", "operation_id": "maintenance.dependencies"}],
        },
        asynchronous=False,
    )
    assert state["status"] == "completed", state
    payload = harness.store.load_json(harness.store.role(state["run_id"], "dependency_diagnostics"))
    assert any(item["availability"]["state"] == "available" for item in payload)
    registry = default_registry()
    with pytest.raises(HarnessError):
        registry.validate_parameters(registry.get("maintenance.dependencies"), {"install": True})
