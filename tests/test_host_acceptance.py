"""Contract fixtures for the host evidence verifier. No real host journey is claimed."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from tools.host_acceptance import verify as verifier


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def sample(tmp_path):
    record = tmp_path / "case.json"
    verifier.main(["init", str(tmp_path), "--host", "codex", "--case", "J10"])
    spec = json.loads(record.read_text("utf-8"))
    now = datetime.now(timezone.utc)
    spec.update(
        observation="recorded",
        version_file="version.txt",
        host_version="1.2.3",
        started_at=now.isoformat(),
        ended_at=(now + timedelta(minutes=1)).isoformat(),
        turns=[{"trace": "host.jsonl", "prompt": "user.txt"}],
    )
    write_json(record, spec)
    (tmp_path / "version.txt").write_text("codex 1.2.3")
    (tmp_path / "user.txt").write_text("Use the reserved analysis capability.", encoding="utf-8")
    answer = {
        "schema_version": "1.0.0",
        "tool": "validate_plan",
        "ok": False,
        "result": None,
        "error": {"code": "DEFERRED_THIS_PHASE", "message": "reserved"},
    }
    events = [
        {"type": "thread.started", "thread_id": "test-only"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "mcp_tool_call",
                "tool": "validate_plan",
                "server": "autotem",
                "arguments": {"plan": {"steps": [{"id": "ai", "operation_id": "ai.denoise"}]}},
                "result": {"content": [{"type": "text", "text": json.dumps(answer)}]},
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "agent_message",
                "text": "The requested capability is reserved. FFT is available.",
            },
        },
        {"type": "turn.completed"},
    ]
    trace = tmp_path / "host.jsonl"
    trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return record, spec, trace, events


def test_unattempted_blocked_and_missing_proof_distinct(tmp_path):
    record, spec, _, _ = sample(tmp_path)
    partial = verifier.verify(record, root=tmp_path)
    assert (
        partial["case_verdict"] == "not_evaluated"
        and partial["checks"]["case_execution"] == "passed"
    )
    assert partial["reason"] == "no_verified_release_binding" and partial["release_ready"] is False
    assert "capability is reserved" not in json.dumps(partial)
    spec.update(
        observation="blocked",
        blocker_code="GATEWAY_UNAVAILABLE",
        blocker_file="gateway.log",
        host="openclaw",
    )
    write_json(record, spec)
    assert verifier.verify(record)["case_verdict"] == "not_evaluated"
    (tmp_path / "gateway.log").write_text("1006 closed", encoding="utf-8")
    blocked = verifier.verify(record)
    assert blocked["case_verdict"] == "blocked" and blocked["reason"] == "GATEWAY_UNAVAILABLE"
    spec.update(observation="not_evaluated")
    write_json(record, spec)
    assert verifier.verify(record)["case_verdict"] == "not_evaluated"


@pytest.mark.parametrize(
    "alter,reason",
    [
        ("fake-cli", "no_real_autotem_host_call"),
        ("different-server", "no_real_autotem_host_call"),
        ("different-tool", "tool_envelope_mismatch"),
        ("unfinished", "incomplete_host_turn"),
    ],
)
def test_discards_nonhost_cli_and_unmatched_results(tmp_path, alter, reason):
    record, _, trace, events = sample(tmp_path)
    call = events[2]["item"]
    if alter == "fake-cli":
        call["type"] = "command_execution"
    if alter == "different-server":
        call["server"] = "some_other_server"
    if alter == "different-tool":
        call["tool"] = "submit_analysis"
    if alter == "unfinished":
        events.pop()
    trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    report = verifier.verify(record, root=tmp_path)
    assert report["case_verdict"] != "passed" and report["reason"] == reason


def test_real_disk_hash_and_readonly_database_verification(tmp_path):
    import time
    import uuid

    from autotem.harness.store import Store, encode

    store = Store(tmp_path / "h")
    run_id, job_id = uuid.uuid4().hex, uuid.uuid4().hex
    directory = store.run_dir(run_id)
    raw = directory / "raw" / "input.npy"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"verifier unit fixture")
    now = time.time()
    plan = {
        "input_mode": "snapshot",
        "steps": [{"id": "load", "operation_version": "1.0.0"}],
        "budgets": {},
    }
    with store.connect() as db:
        db.execute(
            "INSERT INTO runs(run_id,request_hash,request,status,created,updated,input_path,"
            "source_hash,job_id) VALUES(?,?,?,'completed',?,?,?,?,?)",
            (run_id, "test", encode(plan), now, now, str(raw), verifier.digest(raw), job_id),
        )
        db.execute(
            "INSERT INTO jobs(job_id,run_id,status,created) VALUES(?,?,'completed',?)",
            (job_id, run_id, now),
        )
    a = store.write_json(run_id, "_request", "example", {"status": "diagnostic"})
    result = verifier.run_evidence(store.root, run_id, (now - 1, now + 1), None)
    assert result["artifacts"][a["artifact_id"]]["sha256"] == a["sha256"]
    from pathlib import Path

    Path(a["path"]).write_text("changed")
    with pytest.raises(verifier.EvidenceError, match="artifact_hash_mismatch"):
        verifier.run_evidence(store.root, run_id, (now - 1, now + 1), None)
    with pytest.raises(verifier.EvidenceError, match="run_outside_host_capture"):
        verifier.run_evidence(store.root, run_id, (now + 600, now + 800), None)


def test_manual_export_needs_original_observations_and_no_relabelled_native(tmp_path):
    record, spec, trace, events = sample(tmp_path)
    spec.update(host="workbuddy", capture_format="observed-host-export")
    write_json(record, spec)
    manual = {
        "host": "workbuddy",
        "session_id": "observed-test",
        "calls": [
            {
                "server": "autotem",
                "tool": "validate_plan",
                "arguments": events[2]["item"]["arguments"],
                "response": events[2]["item"]["result"],
                "original_evidence": "evidence.txt",
            }
        ],
        "answer": "Reserved capability.",
        "answer_evidence": "evidence.txt",
    }
    write_json(trace, manual)
    assert verifier.verify(record, root=tmp_path)["reason"] == "capture_file_missing_or_empty"
    (tmp_path / "evidence.txt").write_text("Synthetic test attachment")
    spec.update(delegation_mode="native")
    write_json(record, spec)
    assert verifier.verify(record, root=tmp_path)["reason"] == "incomplete_native_profile_traces"
    manual["calls"][0]["original_evidence"] = "../outside.txt"
    write_json(trace, manual)
    assert verifier.verify(record, root=tmp_path)["reason"] == "capture_path_outside_directory"


def test_automatic_host_proofs_are_scoped_without_a_reviewer(tmp_path):
    record, spec, trace, events = sample(tmp_path)
    events.insert(
        2,
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "Get-Content .agents/skills/autotem/SKILL.md",
                "exit_code": 0,
                "aggregated_output": "---\nname: autotem\n---",
            },
        },
    )
    trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    (tmp_path / "discovery.txt").write_text("\n".join(sorted(verifier.TOOLS)), encoding="utf-8")
    (tmp_path / "consent.txt").write_text("AutoTEM request was not approved", encoding="utf-8")
    spec["host_observations"] = {"discovery": "discovery.txt", "authorization": "consent.txt"}
    write_json(record, spec)
    result = verifier.verify(record, root=tmp_path)
    checks = result["checks"]
    assert result["observed_tools"] == ["validate_plan"]
    assert checks["effective_tool_permission"] == checks["full_tool_inventory"] == "passed"
    assert (
        checks["session_skill_discovery"] == "passed"
        and checks["first_authorization"] == "not_evaluated"
    )
    assert result["case_verdict"] == "not_evaluated"
    spec["host_version"] = "1.2"
    write_json(record, spec)
    assert verifier.verify(record, root=tmp_path)["reason"] == "host_version_evidence_mismatch"


def test_windows_capture_time_uses_all_available_microseconds():
    assert verifier.stamp("2026-11-12T13:17:38.1920650+08:00") == verifier.stamp(
        "2026-11-12T05:17:38.192065Z"
    )


def test_physical_units_and_case_negative_variants():
    assert (
        verifier.positive_distance_units(
            "atom_distances", {"nearest_distances": [1, 2], "unit": "pixel"}
        )
        == []
    )
    assert (
        verifier.positive_distance_units(
            "atom_distances", {"nearest_distances": [float("nan"), None], "unit": "nm"}
        )
        == []
    )
    result = {"d_spacings": [{"value": 0.29, "unit": "nm"}], "provenance": {"physical": True}}
    assert verifier.positive_distance_units("diffraction", result) == ["nm"]
    tool = {
        "tool": "validate_plan",
        "response": {"ok": False, "error": {"code": "DEPENDENCY_BROKEN"}},
    }
    verifier.check_case("J11", [tool], {}, [{}], {}, "broken_dependency")
    with pytest.raises(verifier.EvidenceError, match="fault_variant_not_observed"):
        verifier.check_case("J11", [tool], {}, [{}], {}, "missing_dependency")
    run = {
        "status": "needs_input",
        "requested_fields": {"q": ["reference_region"]},
        "artifacts": {"a": {"kind": "image"}},
        "question_evidence": {"q": ["a"]},
        "_children": [],
        "error_code": "NEEDS_INPUT",
    }
    retrieved = {
        "tool": "list_artifacts",
        "response": {"ok": True, "result": [{"artifact_id": "a"}]},
    }
    verifier.check_case("J08", [retrieved], {"r": run}, [{}], {}, "unanswered")
    run["_children"].append("unexpected")
    with pytest.raises(verifier.EvidenceError, match="reply_started_unexpected_child"):
        verifier.check_case("J08", [retrieved], {"r": run}, [{}], {}, "unanswered")


def test_matrix_missing_blocked_baselines_and_conflicts(tmp_path):
    from tools.host_acceptance import index

    release = tmp_path / "release"
    release.mkdir()
    manifest = {
        "release_id": "contract-test",
        "profiles": {"traditional-core-mcp": {"lock_sha256": "lock"}},
        "source": {"archive_sha256": "source"},
        "package_version": "1.0",
        "artifacts": [],
    }
    write_json(release / "release-manifest.json", manifest)
    record, spec, _, _ = sample(tmp_path)
    spec.update(
        observation="blocked", blocker_code="GATEWAY_UNAVAILABLE", blocker_file="gateway.txt"
    )
    write_json(record, spec)
    (tmp_path / "gateway.txt").write_text("closed with 1006", encoding="utf-8")
    summary = verifier.verify(record, bundle=release)
    public = tmp_path / "public.json"
    write_json(public, summary)
    matrix = index.build(release, [public])
    assert len(matrix["hosts"]) == 3
    assert all(len(h["cases"]) == 15 for h in matrix["hosts"].values())
    assert matrix["hosts"]["codex"]["cases"]["J10"]["verdict"] == "blocked"
    assert matrix["hosts"]["openclaw"]["cases"]["J01"]["verdict"] == "not_evaluated"
    assert matrix["release_ready"] is False and matrix["host_journey_matrix"] == "blocked"
    simulation = matrix["hosts"]["codex"]["cases"]["J15"]["variants"]
    assert [r["variant"] for r in simulation if r["required"]] == ["missing_dependency"]
    native = matrix["hosts"]["codex"]["cases"]["J14"]["variants"]
    assert [r["variant"] for r in native if r["required"]] == ["sequential"]
    with pytest.raises(ValueError, match="Mixed host versions"):
        index.build(release, [public], versions={"codex": "9.8.7"})
    summary["case_verdict"] = "passed"
    write_json(public, summary)
    with pytest.raises(ValueError, match="lacks verified release"):
        index.build(release, [public])
    summary["attempted_release_manifest_sha256"] = "0" * 64
    write_json(public, summary)
    with pytest.raises(ValueError, match="Mixed release identities"):
        index.build(release, [public])


def test_passed_verifier_requires_active_project_and_observed_skill(tmp_path, monkeypatch):
    import shutil
    import sys
    from pathlib import Path

    from autotem.acceptance import driver
    from autotem.installation import host_entry

    record, spec, trace, events = sample(tmp_path)
    events.insert(
        2,
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "exit_code": 0,
                "command": "Get-Content .agents/skills/autotem/SKILL.md",
                "aggregated_output": "---\nname: autotem\n---",
            },
        },
    )
    trace.write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    project = tmp_path / "project"
    root = project / ".autotem" / "runs"
    root.mkdir(parents=True)
    deployed = root.parent / "host-entry.py"
    shutil.copy(host_entry.__file__, deployed)
    adapter = project / "autotem-host" / "codex" / "adapter.json"
    adapter.parent.mkdir(parents=True)
    entry = {"command": sys.executable, "args": ["-I", str(deployed), "mcp"]}
    write_json(adapter, {"host": "codex", "mcp": {"mcpServers": {"autotem": entry}}})
    config = project / ".codex" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        "[mcp_servers.autotem]\n"
        + "\n".join(k + " = " + json.dumps(v) for k, v in {**entry, "cwd": str(project)}.items()),
        encoding="utf-8",
    )
    deployed_skill = project / ".agents" / "skills" / "autotem" / "SKILL.md"
    deployed_skill.parent.mkdir(parents=True)
    deployed_skill.write_text("synthetic verifier fixture", encoding="utf-8")
    bundle = tmp_path / "release"
    bundle.mkdir()
    write_json(bundle / "release-manifest.json", {"release_id": "test"})
    # These two controlled substitutes exercise the verifier contract only.
    monkeypatch.setattr(
        driver,
        "runtime_record",
        lambda *a: {
            "skill_resource_sha256": {"SKILL.md": verifier.digest(deployed_skill)},
            "release_identity": {"profile": "traditional-core-mcp"},
        },
    )
    monkeypatch.setattr(host_entry, "active_python", lambda c: Path(sys.executable))
    result = verifier.verify(record, bundle=bundle, root=root, project=project)
    assert result["case_verdict"] == "passed" and not result["release_ready"]
    assert result["checks"]["first_authorization"] == "not_evaluated"
    config.write_text('[mcp_servers.autotem]\ncommand="wrong"\n', encoding="utf-8")
    assert (
        verifier.verify(record, bundle=bundle, root=root, project=project)["reason"]
        == "active_host_mcp_entry_mismatch"
    )
