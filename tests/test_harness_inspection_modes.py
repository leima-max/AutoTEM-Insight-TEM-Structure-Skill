"""Inspection storage policy is explicit across API and public transports."""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from autotem.harness import Harness, HarnessError
from autotem.harness.cli import TOOL_SCHEMAS, dispatch_request, main, serve_stdio


def source_image(folder):
    folder.mkdir(exist_ok=True)
    source = folder / "selected source.npy"
    pixels = np.arange(160, dtype=np.float64).reshape(2, 8, 10)
    np.save(source, pixels)
    return source, pixels


def run_count(h):
    with h.store.connect() as db:
        return db.execute("SELECT count(*) FROM runs").fetchone()[0]


@pytest.mark.parametrize("mode", [None, "snapshot", "reference"])
def test_inspection_mode_preserves_selection_and_evidence(tmp_path, monkeypatch, mode):
    source, pixels = source_image(tmp_path)
    original = source.read_bytes()
    h = Harness(tmp_path / "store")
    if mode == "reference":
        def no_copy(*_args, **_kwargs):
            pytest.fail("Reference inspection attempted to copy the source")

        monkeypatch.setattr("autotem.harness.service.shutil.copyfile", no_copy)
    options = {} if mode is None else {"input_mode": mode}
    state = h.inspect_input(str(source), modality="HRTEM", frame=1, roi=[1, 7, 2, 9], **options)
    assert state["status"] == "completed", state
    run = h.store.get_run(state["run_id"])
    directory = h.store.run_dir(state["run_id"])
    assert run["request"]["input_mode"] == (mode or "snapshot")
    assert run["request"]["selection"] == {"frame": 1, "roi": [1, 7, 2, 9]}
    assert run["request"]["modality"] == "HRTEM"
    assert run["source_hash"] == hashlib.sha256(original).hexdigest()
    assert not list(directory.glob(".snapshot-*.tmp"))
    if mode == "reference":
        assert Path(run["input_path"]) == source.resolve()
        assert not (directory / "raw").exists()
    else:
        assert Path(run["input_path"]) == directory / "raw" / source.name
        assert Path(run["input_path"]).read_bytes() == original
    artifacts = {a["role"]: a for a in state["artifacts"]}
    assert {"raw", "raw_preview", "dataset"} <= artifacts.keys()
    np.testing.assert_array_equal(
        h.store.load_array(artifacts["raw"]["artifact_id"]), pixels[1, 1:7, 2:9]
    )
    assert h.read_artifact(artifacts["raw"]["artifact_id"], limit=4)["is_preview"]
    for role in ("raw", "raw_preview", "dataset"):
        assert h.store._check(artifacts[role]["artifact_id"])["sha256"] == artifacts[role]["sha256"]
    assert state["dataset"]["source_path"] == run["input_path"]
    assert state["dataset"]["metadata"]["selection"]["frame"] == 1
    assert state["dataset"]["metadata"]["selection"]["roi_origin_yx"] == [1, 2]
    assert source.read_bytes() == original


@pytest.mark.parametrize("mode", ["copy", "", None, True])
def test_invalid_api_mode_rejected_before_run_creation(tmp_path, mode):
    source, _ = source_image(tmp_path)
    h = Harness(tmp_path / "store")
    with pytest.raises(HarnessError) as caught:
        h.inspect_input(str(source), input_mode=mode, frame=0)
    assert caught.value.code == "INVALID_PLAN"
    assert run_count(h) == 0


class InspectionRecorder:
    def __init__(self):
        self.calls = []

    def inspect_input(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return {"run_id": "recorded", "status": "completed"}


@pytest.mark.parametrize("mode", [None, "snapshot", "reference"])
def test_jsonl_inspection_policy_matches_service_schema(mode):
    h = InspectionRecorder()
    arguments = {"path": "source.npy", "selection": {"frame": 1}}
    if mode is not None:
        arguments["input_mode"] = mode
    request = {"tool": "inspect_input", "arguments": arguments}
    output = io.StringIO()
    assert serve_stdio(h, io.StringIO(json.dumps(request) + "\n"), output) == 0
    response = json.loads(output.getvalue())
    assert response["ok"], response
    assert h.calls == [("source.npy", {"input_mode": mode or "snapshot", "frame": 1})]
    schema = TOOL_SCHEMAS["inspect_input"]
    assert schema["properties"]["input_mode"] == {
        "type": "string", "enum": ["snapshot", "reference"], "default": "snapshot"
    }
    assert schema["required"] == ["path"] and len(TOOL_SCHEMAS) == 9


@pytest.mark.parametrize("extra", [
    {"input_mode": "copy"}, {"input_mode": None}, {"input_mode": True},
    {"selection": {"input_mode": "reference"}},
    {"input_mode": "reference", "selection": {"input_mode": "snapshot"}},
])
def test_transport_rejects_invalid_or_nested_policy(extra):
    h = InspectionRecorder()
    response = dispatch_request(
        h, {"tool": "inspect_input", "arguments": {"path": "source.npy", **extra}}
    )
    assert not response["ok"] and response["error"]["code"] == "REQUEST_INVALID"
    assert h.calls == []


@pytest.mark.parametrize("mode", [None, "snapshot", "reference"])
def test_cli_inspection_policy_reaches_same_service(tmp_path, monkeypatch, capsys, mode):
    h = InspectionRecorder()
    monkeypatch.setattr("autotem.harness.Harness", lambda *_a, **_kw: h)
    argv = ["--root", str(tmp_path), "inspect", "source.npy", "--selection-json", '{"frame": 1}']
    if mode is not None:
        argv += ["--input-mode", mode]
    assert main(argv) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert h.calls == [("source.npy", {"input_mode": mode or "snapshot", "frame": 1})]


def test_cli_rejects_unknown_mode_before_service(tmp_path, monkeypatch):
    def forbidden(*_a, **_kw):
        pytest.fail("Invalid CLI input must not construct service")

    monkeypatch.setattr("autotem.harness.Harness", forbidden)
    with pytest.raises(SystemExit) as caught:
        main(["--root", str(tmp_path), "inspect", "source.npy", "--input-mode", "copy"])
    assert caught.value.code == 2


@pytest.mark.parametrize("mode", ["snapshot", "reference"])
def test_inspection_mode_cannot_bypass_workspace(tmp_path, mode):
    project = tmp_path / "project"
    project.mkdir()
    outside, _ = source_image(tmp_path / "outside")
    h = Harness(project / "store", workspace=project)
    with pytest.raises(HarnessError) as caught:
        h.inspect_input(str(outside), input_mode=mode, frame=0)
    assert caught.value.code == "WORKSPACE_INPUT_RESTRICTED"
    assert run_count(h) == 0


def test_reference_inspection_parent_rejects_changed_or_missing_source(tmp_path):
    source, pixels = source_image(tmp_path)
    h = Harness(tmp_path / "store")
    parent = h.inspect_input(str(source), input_mode="reference", frame=1)
    assert parent["status"] == "completed", parent
    plan = h.store.get_run(parent["run_id"])["request"]
    plan["steps"][0]["id"] = "reinspect"
    np.save(source, pixels + 1)
    with pytest.raises(HarnessError) as changed:
        h.submit_analysis(plan, parent_run_id=parent["run_id"], asynchronous=False)
    assert changed.value.code == "INPUT_CHANGED"
    source.unlink()
    with pytest.raises(HarnessError) as missing:
        h.submit_analysis(plan, parent_run_id=parent["run_id"], asynchronous=False)
    assert missing.value.code == "INPUT_NOT_FOUND"
    assert run_count(h) == 1
    assert h.get_run_status(parent["run_id"])["status"] == "completed"


def test_reference_inspection_rechecks_source_before_calculation(tmp_path, monkeypatch):
    source, pixels = source_image(tmp_path)
    h = Harness(tmp_path / "store")
    launch = h._launch_scheduler

    def changed_before_launch(run_id):
        np.save(source, pixels + 1)
        return launch(run_id)

    monkeypatch.setattr(h, "_launch_scheduler", changed_before_launch)
    result = h.inspect_input(str(source), input_mode="reference", frame=0)
    assert result["status"] == "failed", result
    assert result["error"]["code"] == "INPUT_CHANGED"
    assert not (h.store.run_dir(result["run_id"]) / "raw").exists()
    assert not {"raw", "raw_preview", "dataset"} & {a["role"] for a in result["artifacts"]}
