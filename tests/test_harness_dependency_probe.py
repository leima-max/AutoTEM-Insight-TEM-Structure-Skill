"""A long-lived host can recheck timed-out dependencies without repeated probes per inventory."""

import subprocess
from unittest.mock import Mock

import pytest

from autotem.harness import registry
from autotem.harness.contracts import OperationDefinition


@pytest.mark.parametrize(
    ("module", "returncode", "stdout", "stderr", "state", "reason"),
    [
        ("probe_ok", 0, '{"found":true}', "", "available", None),
        ("probe_absent", 0, '{"found":false}', "", "dependency_unavailable", None),
        (
            "probe_bad",
            1,
            "",
            "ModuleNotFoundError: No module named 'other'",
            "dependency_broken",
            "ModuleNotFoundError: No module named 'other'",
        ),
        ("probe_invalid", 0, "bad output", "", "dependency_broken", "invalid_probe"),
    ],
)
def test_only_timeouts_are_retried(monkeypatch, module, returncode, stdout, stderr, state, reason):
    completed = subprocess.CompletedProcess([], returncode, stdout, stderr)
    child = Mock(side_effect=[subprocess.TimeoutExpired("probe", 8), completed])
    monkeypatch.setattr(registry.subprocess, "run", child)

    assert registry.probe_dependency(module) == {
        "state": "dependency_broken",
        "dependency": module,
        "reason": "import_timeout",
    }
    expected = {"state": state, "dependency": module}
    if reason:
        expected["reason"] = reason
    assert registry.probe_dependency(module) == expected
    assert registry.probe_dependency(module) == expected
    assert child.call_count == 2


def test_shared_dependency_times_out_once_per_catalog(monkeypatch):
    operations = registry.OperationRegistry()
    for operation in ("test.first", "test.second"):
        operations.register(
            OperationDefinition(
                operation, "data-quality", lambda ctx, params: None, dependencies=("probe_shared",)
            )
        )
    child = Mock(
        side_effect=[
            subprocess.TimeoutExpired("probe", 8),
            subprocess.TimeoutExpired("probe", 8),
            subprocess.CompletedProcess([], 0, '{"found":true}', ""),
        ]
    )
    monkeypatch.setattr(registry.subprocess, "run", child)

    for count in (1, 2):
        rows = operations.capabilities()
        assert [row["availability"]["reason"] for row in rows] == [
            "import_timeout",
            "import_timeout",
        ]
        assert child.call_count == count
    for _ in range(2):
        assert [row["availability"] for row in operations.capabilities()] == [
            {"state": "available"}
        ] * 2
    assert child.call_count == 3
