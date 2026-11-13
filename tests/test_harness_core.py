"""Execution contracts use real disk inputs; scientific accuracy is not inferred here."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from autotem.harness import Harness, HarnessError
from autotem.harness.contracts import OperationDefinition, StrictParameters
from autotem.harness.registry import OperationRegistry
from autotem.harness.store import Store, encode


def image(tmp_path):
    path = tmp_path / "原始 image.npy"
    yy, xx = np.mgrid[:64, :80]
    array = np.cos(2 * np.pi * xx / 8) + np.cos(2 * np.pi * yy / 8)
    np.save(path, array)
    return path, array


def plan(path):
    return {
        "input_path": str(path),
        "modality": "HRTEM",
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "qa", "operation_id": "quality.assess", "depends_on": ["load"]},
        ],
    }


def wait(harness, run_id, maximum=40):
    deadline = time.monotonic() + maximum
    while time.monotonic() < deadline:
        state = harness.get_run_status(run_id)
        if state["status"] in {
            "completed",
            "failed",
            "needs_input",
            "cancelled",
            "timed_out",
            "interrupted",
        }:
            return state
        time.sleep(0.1)
    pytest.fail(f"Harness worker did not finish: {state}")


def test_registry_refuses_conflict_reserved_and_uncontrolled_parameters():
    registry = OperationRegistry()
    definition = OperationDefinition("test.one", "data-quality", lambda ctx, params: None)
    registry.register(definition)
    with pytest.raises(HarnessError, match="test.one") as error:
        registry.register(definition)
    assert error.value.code == "DUPLICATE_OPERATION"
    with pytest.raises(HarnessError) as error:
        registry.validate_parameters(definition, {"shell": "anything"})
    assert error.value.code == "INVALID_PARAMETERS"
    reserved = OperationDefinition(
        "test.reserved", "reserved", None, implementation_status="reserved"
    )
    registry.register(reserved)
    with pytest.raises(HarnessError) as error:
        registry.validate_parameters(reserved, {})
    assert error.value.code == "DEFERRED_THIS_PHASE"

    class OpenParameters(StrictParameters):
        model_config = {"extra": "allow"}

    with pytest.raises(HarnessError):
        registry.register(
            OperationDefinition(
                "test.open",
                "reserved",
                None,
                parameter_model=OpenParameters,
                implementation_status="reserved",
            )
        )


def test_array_store_roundtrip_hash_and_failed_registration(tmp_path, monkeypatch):
    store = Store(tmp_path / "store")
    run_id = uuid.uuid4().hex
    with store.connect() as db:
        db.execute(
            "INSERT INTO runs(run_id,request_hash,request,status,created,updated) "
            "VALUES(?,?,?,'running',0,0)",
            (run_id, "x", encode({})),
        )
    original = np.arange(480, dtype=np.float32).reshape(20, 24)
    artifact = store.write_array(run_id, "one", "array", original, {"unit": "pixel"})
    np.testing.assert_array_equal(store.load_array(artifact["artifact_id"]), original)
    assert artifact["metadata"]["shape"] == [20, 24]
    assert artifact["metadata"]["dtype"] == "float32"
    with open(artifact["path"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(HarnessError) as error:
        store.load_array(artifact["artifact_id"])
    assert error.value.code == "ARTIFACT_CHANGED"
    real_connect = store.connect

    def broken():
        raise RuntimeError("registration rejected")

    monkeypatch.setattr(store, "get_run", lambda value: {"run_id": value})
    monkeypatch.setattr(store, "connect", broken)
    # Reject registration after atomic publication: no orphan can count as evidence.
    before = set((store.run_dir(run_id) / "artifacts").iterdir())
    with pytest.raises(RuntimeError):
        store.write_json(run_id, "one", "uncommitted", {})
    monkeypatch.setattr(store, "connect", real_connect)
    assert set((store.run_dir(run_id) / "artifacts").iterdir()) == before
    with monkeypatch.context() as patch:
        def fail_readback(_artifact_id):
            raise OSError("interrupted response")

        patch.setattr(store, "artifact", fail_readback)
        with pytest.raises(OSError, match="interrupted response"):
            store.write_json(run_id, "one", "published", {"value": 7})
    (published,) = (a for a in store.artifacts(run_id) if a["role"] == "published")
    assert store.load_json(published["artifact_id"]) == {"value": 7}


def test_disk_pipeline_idempotency_child_and_reports(tmp_path):
    source, original = image(tmp_path)
    harness = Harness(tmp_path / "runs")
    request = plan(source)
    first = harness.submit_analysis(request, idempotency_key="first", asynchronous=False)
    assert first["status"] == "completed", first
    retry = harness.submit_analysis(request, idempotency_key="first", asynchronous=False)
    assert retry["run_id"] == first["run_id"]
    request["modality"] = "HAADF-STEM"
    with pytest.raises(HarnessError) as error:
        harness.submit_analysis(request, idempotency_key="first")
    assert error.value.code == "IDEMPOTENCY_CONFLICT"
    # The child recomputes the pinned source even after the original moves or changes.
    np.save(source, np.zeros_like(original))
    child = harness.submit_analysis(request, parent_run_id=first["run_id"], asynchronous=False)
    assert child["status"] == "completed", child
    assert child["parent_run_id"] == first["run_id"]
    assert child["parameter_diff"]
    child_raw = harness.store.role(child["run_id"], "raw")
    np.testing.assert_array_equal(harness.store.load_array(child_raw), original)
    raw = next(a for a in harness.list_artifacts(first["run_id"]) if a["role"] == "raw")
    np.testing.assert_array_equal(harness.store.load_array(raw["artifact_id"]), original)
    preview = harness.read_artifact(raw["artifact_id"], limit=10)
    assert preview["is_preview"] and len(preview["values"]) == 10
    exported = harness.export_report(first["run_id"], formats=["json", "html", "pdf"])
    assert exported["report_type"] == "analysis"
    manifest = next(a for a in exported["artifacts"] if a["role"] == "report_manifest")
    value = harness.store.load_json(manifest["artifact_id"])
    assert value["completeness"]["completed_steps"] == 2
    assert value["scientific_accuracy"] == "not_evaluated"
    assert harness.store._check(raw["artifact_id"])["sha256"] == raw["sha256"]


def test_missing_and_ambiguous_inputs_fail_closed(tmp_path):
    harness = Harness(tmp_path / "runs")
    with pytest.raises(HarnessError) as error:
        harness.inspect_input(str(tmp_path / "missing.npy"))
    assert error.value.code == "INPUT_NOT_FOUND"
    source = tmp_path / "stack.npy"
    np.save(source, np.ones((3, 16, 20)))
    inspection = harness.inspect_input(str(source))
    assert inspection["status"] == "needs_input", inspection
    resolved = harness.inspect_input(str(source), frame=1)
    assert resolved["status"] == "completed", resolved
    raw = next(a for a in resolved["artifacts"] if a["role"] == "raw")
    assert raw["metadata"]["shape"] == [16, 20]


def test_cancel_timeout_and_concurrent_submit_are_isolated(tmp_path):
    source, _ = image(tmp_path)
    harness = Harness(tmp_path / "runs")
    timeout_plan = plan(source)
    timeout_plan["budgets"] = {"run_seconds": 0.01}
    timed = harness.submit_analysis(timeout_plan, asynchronous=False)
    assert timed["status"] == "timed_out", timed
    # Concurrent requests use one scientific worker and retain independent raw evidence.
    with ThreadPoolExecutor(max_workers=3) as pool:
        submissions = list(
            pool.map(
                lambda i: harness.submit_analysis(plan(source), idempotency_key=f"request-{i}"),
                range(3),
            )
        )
    cancelled = harness.cancel_run(submissions[-1]["run_id"])
    assert cancelled["status"] in {"queued", "running", "cancelled"}
    states = [wait(harness, item["run_id"]) for item in submissions]
    assert states[-1]["status"] == "cancelled"
    assert all(state["status"] == "completed" for state in states[:-1]), states
    assert len({state["run_id"] for state in states}) == 3
    with harness.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM runs WHERE status='running'").fetchone()[0] == 0


def test_invalid_units_parameters_and_reserved_are_preflight_rejected(tmp_path):
    source, _ = image(tmp_path)
    harness = Harness(tmp_path / "runs")
    invalid = plan(source)
    invalid["steps"][0]["parameters"] = {"execute_script": "not allowed"}
    with pytest.raises(HarnessError):
        harness.validate_plan(invalid)
    invalid = plan(source)
    invalid["budgets"] = {"run_seconds": float("nan")}
    with pytest.raises(HarnessError):
        harness.validate_plan(invalid)
    invalid = plan(source)
    invalid["steps"].append({"id": "ai", "operation_id": "ai.atom_columns"})
    with pytest.raises(HarnessError) as error:
        harness.validate_plan(invalid)
    assert error.value.code == "DEFERRED_THIS_PHASE"
    assert json.loads(encode(harness.validate_plan(plan(source))))["valid"]


def test_same_run_work_packages_and_immutable_step_retry(tmp_path):
    source, _ = image(tmp_path)
    harness = Harness(tmp_path / "runs")
    initial = {
        "input_path": str(source),
        "modality": "HRTEM",
        "steps": [{"id": "load", "operation_id": "input.load"}],
    }
    first = harness.submit_analysis(initial, asynchronous=False)
    before = harness.store.role(first["run_id"], "raw")
    package = {
        "input_path": str(source),
        "modality": "HRTEM",
        "steps": [{"id": "qa", "operation_id": "quality.assess", "depends_on": ["load"]}],
    }
    harness.validate_plan(package, run_id=first["run_id"])
    complete = harness.submit_analysis(
        package, run_id=first["run_id"], idempotency_key="quality-package", asynchronous=False
    )
    assert complete["status"] == "completed", complete
    assert complete["run_id"] == first["run_id"]
    assert harness.store.role(first["run_id"], "raw") == before
    repeated = harness.submit_analysis(
        package, run_id=first["run_id"], idempotency_key="quality-package", asynchronous=False
    )
    assert len(repeated["steps"]) == 2
    with pytest.raises(HarnessError) as error:
        harness.submit_analysis(package, run_id=first["run_id"], asynchronous=False)
    assert error.value.code == "IMMUTABLE_STEP"


def test_installed_entry_point_plugin_is_opt_in_and_runs_without_executor_edits(
    tmp_path, isolated_harness_venv
):
    python, plugin_dir = isolated_harness_venv
    distribution = plugin_dir / "harness_test_plugin-1.0.dist-info"
    distribution.mkdir()
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: harness-test-plugin\nVersion: 1.0\n", encoding="utf-8"
    )
    (distribution / "entry_points.txt").write_text(
        "[autotem.operations]\ntest_plugin = harness_test_plugin:definitions\n", encoding="utf-8"
    )
    (plugin_dir / "harness_test_plugin.py").write_text(
        "import os, sys\n"
        "from autotem.harness.contracts import OperationDefinition, ServiceResult\n"
        "def execute(ctx, params):\n"
        "    count = int(ctx.array('raw').size)\n"
        "    return ServiceResult({'plugin_result': {'actual_pixels': count, 'module': __file__,\n"
        "        'isolated': sys.flags.isolated, 'worker_pid': os.getpid()}})\n"
        "def definitions():\n"
        "    return [OperationDefinition('plugin.pixel_count','data-quality',execute,"
        "input_roles=('raw',),output_roles=('plugin_result',))]\n",
        encoding="utf-8",
    )
    source, _ = image(tmp_path)
    request = plan(source)
    request["steps"].append(
        {"id": "plugin", "operation_id": "plugin.pixel_count", "depends_on": ["load"]}
    )
    enabled = {"enabled_plugins": ["test_plugin"]}
    calls = [
        ("validate_plan", {"plan": request}),
        ("list_capabilities", enabled),
        ("submit_analysis", {"plan": {**request, **enabled}, "asynchronous": False}),
    ]
    messages = "".join(
        json.dumps({"tool": tool, "arguments": arguments}) + "\n" for tool, arguments in calls
    )
    run_root = tmp_path / "runs"
    proc = subprocess.run(
        [str(python), "-I", "-m", "autotem.harness.cli", "--root", str(run_root), "stdio"],
        input=messages.encode("utf-8"),
        capture_output=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    disabled, available, submitted = map(json.loads, proc.stdout.splitlines())
    assert not disabled["ok"] and disabled["error"]["code"] == "UNKNOWN_OPERATION"
    assert available["ok"], available
    assert {
        "name": "test_plugin",
        "entry_point": "harness_test_plugin:definitions",
        "version": "1.0",
        "enabled": True,
    } in available["result"]["installed_plugins"]
    assert any(
        op["operation_id"] == "plugin.pixel_count" for op in available["result"]["operations"]
    )
    assert submitted["ok"], submitted
    state = submitted["result"]
    assert state["status"] == "completed", state
    store = Store(run_root)
    result = store.load_json(store.role(state["run_id"], "plugin_result"))
    assert result["actual_pixels"] == 64 * 80
    assert Path(result["module"]).resolve() == (plugin_dir / "harness_test_plugin.py").resolve()
    assert result["isolated"] == 1
    assert result["worker_pid"] == store.get_run(state["run_id"])["worker_pid"]


def test_scheduler_failure_and_queued_cancel_finish_jobs(tmp_path, monkeypatch):
    source, _ = image(tmp_path)
    harness = Harness(tmp_path / "runs")

    def rejected_start():
        raise OSError("process launch unavailable")

    monkeypatch.setattr(harness, "_start_scheduler", rejected_start)
    interrupted = harness.submit_analysis(plan(source), asynchronous=False)
    assert interrupted["status"] == "interrupted"
    assert interrupted["error"]["code"] == "SCHEDULER_UNAVAILABLE"
    assert all(s["status"] == "interrupted" for s in interrupted["steps"])
    assert interrupted["jobs"][-1]["status"] == "interrupted"
    monkeypatch.setattr(harness, "_start_scheduler", lambda: None)
    queued = harness.submit_analysis(plan(source))
    cancelled = harness.cancel_run(queued["run_id"])
    assert cancelled["status"] == "cancelled"
    assert cancelled["jobs"][-1]["status"] == "cancelled"


def test_pid_reuse_does_not_hide_interrupted_worker(tmp_path, monkeypatch):
    import os

    source, _ = image(tmp_path)
    harness = Harness(tmp_path / "runs")
    monkeypatch.setattr(harness, "_start_scheduler", lambda: None)
    queued = harness.submit_analysis(plan(source))
    harness.store.update_run(
        queued["run_id"], status="running", worker_pid=os.getpid(), worker_identity="stale"
    )
    state = harness.get_run_status(queued["run_id"])
    assert state["status"] == "interrupted"
    assert state["error"]["code"] == "WORKER_INTERRUPTED"
    assert state["jobs"][-1]["status"] == "interrupted"


def test_domain_work_package_budget_is_enforced(tmp_path):
    source, _ = image(tmp_path)
    request = plan(source)
    request["steps"] += [{"id": f"qa{i}", "operation_id": "quality.assess"} for i in range(7)]
    with pytest.raises(HarnessError) as error:
        Harness(tmp_path / "runs").validate_plan(request)
    assert error.value.code == "PROFILE_STEP_LIMIT"
    assert error.value.details == {"profile_id": "data-quality", "limit": 8}


def test_run_local_cache_supports_long_windows_names(tmp_path):
    import os
    from pathlib import Path

    from autotem.harness.worker import cache_path

    logical = tmp_path / ("cache_" + "n" * 80) / ("entry_" + "x" * 80 + ".nbi")
    filename = Path(cache_path(logical))
    filename.parent.mkdir(parents=True)
    filename.write_bytes(b"run-local compiled cache")
    assert filename.read_bytes() == b"run-local compiled cache"
    if os.name == "nt":
        assert str(filename).startswith("\\\\?\\")
    else:
        assert filename == logical.resolve()
