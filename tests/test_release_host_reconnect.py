"""Exercise reported host reconnect failures against project pointers, assets and store."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import test_release_activation

from autotem import bootstrap
from autotem.harness import Harness, runtime_binding
from autotem.harness.store import Store
from autotem.installation import managed

pytestmark = pytest.mark.smoke
release_project = test_release_activation.release_project
POINTER = ".autotem/active-runtime.json"
CODEX_ADAPTER = "autotem-host/codex/adapter.json"


@pytest.fixture(name="tmp_path")
def short_project_path(tmp_path_factory):
    # The default per-test name consumes Windows path space needed by historical artifacts.
    return tmp_path_factory.mktemp("hr")


def install(root, bundle, host="codex"):
    done = managed.manage(root, bundle, host=host, apply=True)
    assert done["ok"], done.get("error")
    return done


def transaction(root, installation):
    return root / ".autotem/installations" / installation["installation_id"] / "transaction.json"


def report_file(root, installation, stage="call_tool", **changes):
    report = {
        "schema_version": 1,
        "installation_id": installation["installation_id"],
        "release_id": installation["release_id"],
        "host": installation["host"],
        "stage": stage,
        "error_code": "CONNECTOR_DISCONNECTED",
        "message": "升级后连接断开，宿主无法完成此次调用。",
        **changes,
    }
    path = root.parent / "宿主 断线.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, report


def project_state(root, host):
    paths = [POINTER, ".autotem/host-entry.py", f"autotem-host/{host}/adapter.json"]
    if host == "codex":
        paths.append(".codex/config.toml")
    if host == "workbuddy":
        paths.append("autotem-host/workbuddy/autotem-skill.zip")
    return {name: (root / name).read_bytes() for name in paths}


@pytest.mark.parametrize(
    ("host", "stage"),
    [("codex", "connect"), ("openclaw", "discover_tools"), ("workbuddy", "call_tool")],
)
def test_report_restores_the_matching_predecessor_through_cli(release_project, capsys, host, stage):
    root, _ = release_project
    first = install(root, "first", host)
    original = project_state(root, host)
    store = Store(root / ".autotem/runs")
    run_id = "a" * 32
    with store.connect() as db:
        db.execute(
            "INSERT INTO runs(run_id,request_hash,request,status,created,updated) "
            "VALUES(?,'history','{}','completed',0,0)",
            (run_id,),
        )
    value = {"fixture": "historical spacing", "value": 0.16, "unit": "nm"}
    saved = store.write_json(run_id, "_history", "spacing", value)
    current = install(root, "second", host)
    assert current["installation_id"] != first["installation_id"]
    assert project_state(root, host)[POINTER] != original[POINTER]
    path, report = report_file(root, current, stage)
    status = managed.main(
        ["--project", str(root), "--release", "second", "--host-reconnect-failure", str(path)]
    )
    result = json.loads(capsys.readouterr().out)
    assert status == 0 and result["ok"] and result["state"] == "rolled_back", result
    assert result["host"] == host and result["host_reconnect_failure"] == report
    assert result["host_connection"] == "reported_failed"
    assert result["restored_host_connection"] == result["host_journey"] == "not_evaluated"
    assert result["release_ready"] is False
    assert result["rollback"]["status"] == "rolled_back"
    assert result["rollback"]["from_installation_id"] == current["installation_id"]
    assert result["rollback"]["to_installation_id"] == first["installation_id"]
    assert result["active_runtime"] == json.loads(original[POINTER])
    assert project_state(root, host) == original
    history = transaction(root, current)
    assert (root / result["journal_path"]).resolve() == history.resolve()
    journal = json.loads(history.read_bytes())["host_reconnect_recovery"]
    assert journal["report"] == report and journal["report_sha256"] == result["report_sha256"]
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":")).encode("ascii")
    assert result["report_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert journal["observed_by"] == "operator"
    assert store.artifact(saved["artifact_id"]) == saved
    assert store.load_json(saved["artifact_id"]) == value

    # Retry after a new promotion with the same release ID/hash, but a different installation ID.
    # The original report is sent again with regular JSON/UTF-8 BOM formatting.
    retry = managed.manage(root, "second", host_reconnect_failure=path)
    assert retry["ok"] and project_state(root, host) == original
    promoted_again = install(root, "second", host)
    assert promoted_again["manifest_sha256"] == current["manifest_sha256"]
    assert promoted_again["installation_id"] != current["installation_id"]
    fresh = project_state(root, host)
    recorded = history.read_bytes()
    path.write_text(json.dumps(report), encoding="utf-8-sig")
    replay = managed.manage(root, "second", host_reconnect_failure=path)
    assert replay["ok"] and replay["report_sha256"] == result["report_sha256"]
    assert replay["rollback"] == result["rollback"]
    assert replay["active_runtime"] == json.loads(fresh[POINTER])
    assert project_state(root, host) == fresh and history.read_bytes() == recorded


def test_delayed_report_for_inactive_installation_does_not_affect_newer_release(release_project):
    root, _ = release_project
    install(root, "first")
    earlier = install(root, "second")
    path, _ = report_file(root, earlier)
    active = install(root, "third")
    before = project_state(root, "codex")
    journal = transaction(root, active).read_bytes()
    result = managed.manage(root, "second", host_reconnect_failure=path)
    assert not result["ok"] and result["error"]["code"] == "ACTIVATION_CONFLICT"
    assert (
        project_state(root, "codex") == before and transaction(root, active).read_bytes() == journal
    )


@pytest.mark.parametrize(
    ("bundle", "host", "error"),
    [
        ("first", "codex", "HOST_RECONNECT_NO_PREDECESSOR"),
        ("second", "workbuddy", "HOST_RECONNECT_TARGET_MISMATCH"),
    ],
)
def test_first_install_or_host_without_a_predecessor_keeps_its_entry(
    release_project, bundle, host, error
):
    root, _ = release_project
    first = install(root, "first")
    current = first if bundle == "first" else install(root, bundle, host)
    path, _ = report_file(root, current)
    before = project_state(root, host)
    result = managed.manage(root, bundle, host_reconnect_failure=path)
    assert not result["ok"] and result["error"]["code"] == error
    assert project_state(root, host) == before


def test_pending_authorization_stale_connection_and_other_targets_never_roll_back(release_project):
    root, _ = release_project
    install(root, "first")
    current = install(root, "second")
    before = project_state(root, "codex")
    journal = transaction(root, current)
    original = journal.read_bytes()
    for code in (
        "HOST_APPROVAL_PENDING",
        "HOST_PERMISSION_DENIED",
        "HOST_RECONNECT_REQUIRED",
        "DEFERRED_THIS_PHASE",
        "ARTIFACT_NOT_FOUND",
        "DEPENDENCY_UNAVAILABLE",
        "UNKNOWN_OPERATION",
    ):
        path, _ = report_file(root, current, error_code=code)
        result = managed.manage(root, "second", host_reconnect_failure=path)
        assert not result["ok"] and result["error"]["code"] == "HOST_RECONNECT_NOT_FAILED", code
    path, _ = report_file(root, current)
    for options in ({"bundle": "first"}, {"bundle": "second", "host": "openclaw"}):
        result = managed.manage(root, host_reconnect_failure=path, **options)
        assert not result["ok"] and result["error"]["code"] == "HOST_RECONNECT_TARGET_MISMATCH"
    assert project_state(root, "codex") == before and journal.read_bytes() == original


def test_malformed_report_keeps_the_active_release_and_its_journal(release_project):
    root, _ = release_project
    install(root, "first")
    current = install(root, "second")
    path, report = report_file(root, current)
    before = project_state(root, "codex")
    journal = transaction(root, current)
    original = journal.read_bytes()
    invalid_fields = [
        {"schema_version": True},
        {"stage": "doctor"},
        {"installation_id": "unknown"},
        {"message": []},
        {"extra": "unexpected"},
    ]
    duplicate = json.dumps(report).replace(
        '"schema_version":', '"schema_version": 1, "schema_version":'
    )
    payloads = [json.dumps({**report, **changes}) for changes in invalid_fields]
    for raw in [*payloads, duplicate, " " * 16_385, "[]", "{"]:
        path.write_bytes(raw.encode("utf-8"))
        response = managed.manage(root, "second", host_reconnect_failure=path)
        assert not response["ok"] and response["error"]["code"] == "INVALID_HOST_RECONNECT_REPORT"
    assert project_state(root, "codex") == before and journal.read_bytes() == original


@pytest.mark.parametrize(
    ("missing", "code"),
    [("host_receipt", "HOST_RECEIPT_MISSING"), ("runtime_marker", "ROLLBACK_RUNTIME_FAILED")],
)
def test_unverifiable_predecessor_is_blocked_before_the_entry_changes(
    release_project, missing, code
):
    root, _ = release_project
    first = install(root, "first")
    previous = project_state(root, "codex")
    predecessor = json.loads(previous[POINTER])
    current = install(root, "second")
    before = project_state(root, "codex")
    control = root / ".autotem"
    if missing == "host_receipt":
        owner = predecessor["host_installations"]["codex"]
        evidence = control / "host-installations" / owner / "manifest.json"
    else:
        evidence = control / "runtimes" / first["runtime_id"] / bootstrap.MARKER_NAME
    original = evidence.read_bytes()
    evidence.unlink()
    path, report = report_file(root, current)
    result = managed.manage(root, "second", host_reconnect_failure=path)
    assert not result["ok"] and result["error"]["code"] == code
    assert result["rollback"]["status"] == "blocked" and result["host_reconnect_failure"] == report
    assert project_state(root, "codex") == before
    evidence.write_bytes(original)
    restored = managed.manage(root, "second", host_reconnect_failure=path)
    assert restored["ok"] and restored["report_sha256"] == result["report_sha256"]
    assert project_state(root, "codex") == previous


def test_submitted_job_stays_queued_until_the_owner_cancels_it(release_project, monkeypatch):
    root, staged = release_project
    install(root, "first")
    previous = project_state(root, "codex")
    current = install(root, "second")
    before = project_state(root, "codex")
    path, report = report_file(root, current)
    source = root / "image.npy"
    np.save(source, np.ones((8, 8), dtype=np.float32))
    harness = Harness(root / ".autotem/runs")
    plan = {
        "input_path": str(source),
        "modality": "HRTEM",
        "steps": [{"id": "load", "operation_id": "input.load"}],
    }
    # Submit as the staged interpreter and hold the scheduler before it can claim this job.
    launches = []
    runtime = SimpleNamespace(prefix=str(staged[-1].parent.parent), executable=str(staged[-1]))
    with monkeypatch.context() as setup:
        setup.setattr(runtime_binding, "sys", runtime)
        setup.setattr(harness, "_launch_scheduler", launches.append)
        queued = harness.submit_analysis(plan)
    run_id = queued["run_id"]
    assert queued["status"] == "queued" and launches == [run_id]
    result = managed.manage(root, "second", host_reconnect_failure=path)
    assert not result["ok"] and result["error"]["code"] == "WAIT_ACTIVE_RUNS"
    assert result["host_reconnect_failure"] == report and project_state(root, "codex") == before
    assert harness.store.get_run(run_id)["status"] == "queued"
    assert harness.cancel_run(run_id)["status"] == "cancelled"
    assert managed.manage(root, "second", host_reconnect_failure=path)["ok"]
    assert project_state(root, "codex") == previous
    assert harness.get_run_status(run_id)["status"] == "cancelled"
    assert source.is_file() and harness.list_artifacts(run_id)


@pytest.mark.parametrize("option", ["apply", "rollback", "uninstall"])
def test_reconnect_cannot_be_combined_with_another_action(release_project, option):
    root, _ = release_project
    install(root, "first")
    current = install(root, "second")
    path, _ = report_file(root, current)
    before = project_state(root, "codex")
    result = managed.manage(root, "second", host_reconnect_failure=path, **{option: True})
    assert not result["ok"] and project_state(root, "codex") == before
    args = ["--project", str(root), "--release", "second", "--host-reconnect-failure", str(path)]
    try:
        status = managed.main([*args, f"--{option}"])
    except SystemExit as exc:
        status = exc.code
    assert status == 2 and project_state(root, "codex") == before


@pytest.mark.parametrize("blocker", ["running_job", "changed_adapter", "unchanged_configuration"])
def test_blocked_rollback_records_the_original_failure_and_retries_when_resolved(
    release_project, blocker
):
    root, _ = release_project
    install(root, "first")
    previous = project_state(root, "codex")
    current = install(root, "second")
    before = project_state(root, "codex")
    path, report = report_file(root, current)
    store = Store(root / ".autotem/runs")
    if blocker == "running_job":
        with store.connect() as db:
            db.execute(
                "INSERT INTO jobs(job_id,run_id,created,status) VALUES('active','r',0,'running')"
            )
        expected = "WAIT_ACTIVE_RUNS"
    else:
        name = CODEX_ADAPTER if blocker == "changed_adapter" else ".codex/config.toml"
        if blocker == "unchanged_configuration":
            assert previous[name] == before[name]
        edited = before[name] + b"\n# local change to retain\n"
        (root / name).write_bytes(edited)
        expected = "HOST_FILE_CONFLICT"
    blocked_state = project_state(root, "codex")
    blocked = managed.manage(root, "second", host_reconnect_failure=path)
    assert not blocked["ok"] and blocked["state"] == "blocked"
    assert blocked["error"]["code"] == expected
    assert (
        blocked["host_reconnect_failure"] == report and blocked["rollback"]["status"] == "blocked"
    )
    journal = transaction(root, current)
    audit = json.loads(journal.read_bytes())["host_reconnect_recovery"]
    assert audit["report"] == report and audit["report_sha256"] == blocked["report_sha256"]
    assert project_state(root, "codex") == blocked_state

    # A different report cannot replace the source diagnostic on retry.
    other, _ = report_file(root, current, message="a different failure")
    refused = managed.manage(root, "second", host_reconnect_failure=other)
    assert not refused["ok"] and refused["error"]["code"] == "HOST_RECONNECT_TARGET_MISMATCH"
    assert project_state(root, "codex") == blocked_state
    path, _ = report_file(root, current)
    if blocker == "running_job":
        with store.connect() as db:
            assert (
                db.execute("SELECT status FROM jobs WHERE job_id='active'").fetchone()[0]
                == "running"
            )
            db.execute("UPDATE jobs SET status='completed' WHERE job_id='active'")
    else:
        conflict = next(f for f in blocked["conflict_resolution"]["files"] if f["path"] == name)
        assert conflict["current_sha256"] == hashlib.sha256(edited).hexdigest()
        assert conflict["expected_sha256"] == hashlib.sha256(before[name]).hexdigest()
        # The owner resolves the conflict in the project before asking for another rollback.
        (root / name).write_bytes(before[name])
    restored = managed.manage(root, "second", host_reconnect_failure=path)
    assert restored["ok"] and restored["host_reconnect_failure"] == report
    assert project_state(root, "codex") == previous
    persisted = json.loads(journal.read_bytes())["host_reconnect_recovery"]
    assert persisted["report_sha256"] == audit["report_sha256"] and persisted["report"] == report
    assert persisted["rollback"]["status"] == "rolled_back"
    assert persisted["last_blocked_error"]["code"] == expected
