"""Offline upgrade receipts must agree with their originating jobs and actual file bytes."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from tools.setup import verify_setup_retry, verify_upgrade
from tools.setup.verify_upgrade import (
    USER_CONFIG,
    GateError,
    conflict_copy,
    evidence,
    recheck,
    recovery_evidence,
    save,
    sha,
)

pytestmark = pytest.mark.smoke


@pytest.fixture
def codex_config_upgrade(tmp_path):
    project = tmp_path / "project"
    command = [str(project / ".autotem/python/base/python.exe"), "-I",
               str(project / ".autotem/host-entry.py"), "stdio"]
    route = {"command": command[0], "args": [*command[1:-1], "mcp"], "cwd": str(project)}

    def config(entry):
        return USER_CONFIG + b"\n\n[mcp_servers.autotem]\n" + (
            "\n".join(f"{key} = {json.dumps(value, ensure_ascii=False)}"
                      for key, value in entry.items()) + "\n"
        ).encode("utf-8")

    old = config({**route, "startup_timeout_sec": 60})
    new = config({**route, "required": True, "startup_timeout_sec": 180})
    stages = []
    for phase, data, owner in (
        ("before_upgrade", old, "install-" + "1" * 32),
        ("after_upgrade", new, "install-" + "2" * 32),
    ):
        stage = project / ".upgrade-verifier" / phase
        stage.mkdir(parents=True)
        (stage / "host_config_sha256").write_bytes(data)
        save(stage / "pointer_sha256", {"host_installations": {"codex": owner}})
        save(stage / "adapter_sha256", {
            "jsonl_command": command,
            "mcp": {"mcpServers": {"autotem": route}},
        })
        receipt = project / ".autotem/host-installations" / owner
        receipt.mkdir(parents=True)
        (receipt / "after").write_bytes(data)
        (receipt / "before").write_bytes(old if phase == "after_upgrade" else USER_CONFIG)
        save(receipt / "manifest.json", {
            "host": "codex", "installation_id": owner,
            "files": [{"path": ".codex/config.toml", "after_copy": "after",
                       "backup": "before", "after_sha256": sha(receipt / "after"),
                       "before_sha256": sha(receipt / "before")}],
        })
        stages.append({key: sha(stage / key) for key in (
            "host_config_sha256", "pointer_sha256", "adapter_sha256"
        )})
    return project, stages, config, route


def test_managed_startup_upgrade_reads_owned_exact_config_snapshots(codex_config_upgrade):
    project, stages, _, _ = codex_config_upgrade
    verify_upgrade.host_config_transition(project, *stages)


def test_historical_managed_receipt_binds_config_without_an_after_copy(codex_config_upgrade):
    # The pinned RC8 installer recorded after_sha256, before retained after-copies existed.
    project, stages, _, _ = codex_config_upgrade
    path = project / ".autotem/host-installations" / ("install-" + "1" * 32) / "manifest.json"
    receipt = json.loads(path.read_bytes())
    del receipt["files"][0]["after_copy"]
    save(path, receipt)
    verify_upgrade.host_config_transition(project, *stages)
    receipt["files"][0]["after_sha256"] = "0" * 64
    save(path, receipt)
    with pytest.raises(GateError, match="host_config"):
        verify_upgrade.host_config_transition(project, *stages)


@pytest.mark.parametrize("source_phase", ["before_upgrade", "after_upgrade"])
def test_unchanged_config_remains_compatible(codex_config_upgrade, source_phase):
    project, stages, _, _ = codex_config_upgrade
    local = project / ".upgrade-verifier"
    data = (local / source_phase / "host_config_sha256").read_bytes()
    for phase, stage in zip(("before_upgrade", "after_upgrade"), stages, strict=True):
        path = local / phase / "host_config_sha256"
        path.write_bytes(data)
        stage["host_config_sha256"] = sha(path)
    verify_upgrade.host_config_transition(project, *stages, allow_migration=False)


def test_historical_driver_cannot_reinterpret_changed_config(codex_config_upgrade):
    project, stages, _, _ = codex_config_upgrade
    assert verify_upgrade.PRE_STARTUP_VERIFIER_SHA256 not in verify_upgrade.EARLIER_DRIVERS
    with pytest.raises(GateError, match="host_config_historical_change"):
        verify_upgrade.host_config_transition(project, *stages, allow_migration=False)


@pytest.mark.parametrize("change", [
    "prefix", "prefix_newlines", "tail", "command", "args", "cwd", "extra",
    "required_int", "required_false", "timeout", "old_timeout", "old_timeout_float",
    "old_direct", "old_required", "old_extra", "snapshot_hash", "receipt_hash",
    "receipt_owner", "receipt_copy", "receipt_copy_escape", "receipt_backup",
    "receipt_before_hash",
])
def test_startup_migration_rejects_unowned_or_changed_bytes(codex_config_upgrade, change):
    project, stages, config, route = codex_config_upgrade
    old_change = change.startswith("old_")
    index = 0 if old_change else 1
    phase = ("before_upgrade", "after_upgrade")[index]
    stage = project / ".upgrade-verifier" / phase
    pointer = json.loads((stage / "pointer_sha256").read_bytes())
    owner = pointer["host_installations"]["codex"]
    receipt = project / ".autotem/host-installations" / owner
    record = json.loads((receipt / "manifest.json").read_bytes())
    path = stage / "host_config_sha256"
    entry = ({**route, "startup_timeout_sec": 60} if old_change else
             {**route, "required": True, "startup_timeout_sec": 180})
    if change in {"command", "args", "cwd", "extra"}:
        entry[change] = ["unrelated"] if change == "args" else "unrelated"
    elif change in {"required_int", "required_false"}:
        entry["required"] = 1 if change == "required_int" else False
    elif change in {"timeout", "old_timeout"}:
        entry["startup_timeout_sec"] = 90
    elif change == "old_timeout_float":
        entry["startup_timeout_sec"] = 60.0
    elif change == "old_direct":
        del entry["startup_timeout_sec"]
    elif change == "old_required":
        entry["required"] = True
    elif change == "old_extra":
        entry["unknown"] = "preserve me"
    data = config(entry)
    if change == "prefix":
        data = data.replace(b"16400", b"16401")
    elif change == "prefix_newlines":
        data = data.replace(b"\r\n", b"\n")
    elif change == "tail":
        data += b"# comment after the generated section\n"
    path.write_bytes(data)
    if change != "snapshot_hash":
        stages[index]["host_config_sha256"] = sha(path)
        record["files"][0]["after_sha256"] = sha(path)
        (receipt / "after").write_bytes(data)
    else:
        path.write_bytes(data + b"\n")
    if change == "receipt_hash":
        record["files"][0]["after_sha256"] = "0" * 64
    elif change == "receipt_owner":
        record["installation_id"] = "install-" + "3" * 32
    elif change == "receipt_copy":
        (receipt / "after").write_bytes(b"different")
    elif change == "receipt_copy_escape":
        (receipt.parent / "outside").write_bytes(data)
        record["files"][0]["after_copy"] = "../outside"
    elif change == "receipt_backup":
        (receipt / "before").write_bytes(b"different")
    elif change == "receipt_before_hash":
        record["files"][0]["before_sha256"] = "0" * 64
    save(receipt / "manifest.json", record)
    with pytest.raises(GateError, match="host_config"):
        verify_upgrade.host_config_transition(project, *stages)


def test_upgrade_retries_with_the_verified_after_copy(tmp_path):
    key = "install-" + "a" * 32
    directory = tmp_path / ".autotem/host-installations" / key
    directory.mkdir(parents=True)
    path = "autotem-host/codex/adapter.json"
    copy = directory / "after"
    copy.write_bytes(b"installed")
    expected = sha(copy)
    modified = hashlib.sha256(b"user edits").hexdigest()
    receipt = directory / "manifest.json"
    save(receipt, {"files": [{"path": path, "after_sha256": expected}]})
    row = dict(
        path=path,
        current_state="file",
        condition="recorded_file",
        current_sha256=modified,
        expected_sha256=expected,
        comparison_sha256=expected,
        comparison_path=copy.relative_to(tmp_path).as_posix(),
        receipt_path=receipt.relative_to(tmp_path).as_posix(),
    )
    reply = {"conflict_resolution": {"files": [row]}}
    assert conflict_copy(tmp_path, reply, key, expected, modified) == copy
    other = tmp_path / "another-copy"
    other.write_bytes(copy.read_bytes())
    row["comparison_path"] = other.name
    with pytest.raises(GateError, match="not_bound"):
        conflict_copy(tmp_path, reply, key, expected, modified)
    row["comparison_path"] = copy.relative_to(tmp_path).as_posix()
    copy.write_bytes(b"older before backup")
    with pytest.raises(GateError, match="not_bound"):
        conflict_copy(tmp_path, reply, key, expected, modified)


@pytest.fixture
def scientific_store(tmp_path):
    project = tmp_path / "sample project"
    root = project / ".autotem/runs"
    root.mkdir(parents=True)
    original = project / "synthetic.npy"
    original.write_bytes(b"synthetic fixture")
    source = {"input": str(original), "sha256": sha(original)}
    release = {
        "manifest_sha256": "a" * 64,
        "package_version": "0.1.0",
        "runtime_id": "rt-" + "a" * 20,
    }
    plan = '{"steps":[{"operation_id":"diffraction.metrics"}]}'
    binding = {
        "runtime_id": release["runtime_id"],
        "job_id": "job1",
        "plugins": {},
        "package_version": "0.1.0",
        "plan_sha256": hashlib.sha256(plan.encode()).hexdigest(),
    }
    dbfile = root / "harness.sqlite3"
    files = {}
    with sqlite3.connect(dbfile) as db:
        db.executescript("""
            CREATE TABLE runs(run_id, status, source_hash, input_path, request, job_id);
            CREATE TABLE jobs(job_id, run_id, status);
            CREATE TABLE events(run_id, kind, payload);
            CREATE TABLE steps(run_id, step_id, operation_id, status);
            CREATE TABLE artifacts(run_id,artifact_id,role,step_id,path,sha256,size,committed);
            """)
        db.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?,?)",
            ("run1", "completed", source["sha256"], str(original), plan, "job1"),
        )
        db.execute("INSERT INTO jobs VALUES('job1','run1','completed')")
        db.execute("INSERT INTO events VALUES('run1','runtime_bound',?)", (json.dumps(binding),))
        db.execute("INSERT INTO steps VALUES('run1','metrics','diffraction.metrics','completed')")
        for idx, role in enumerate(
            (
                "raw",
                "fft_complex",
                "g_vector_table",
                "diffraction",
                "report_manifest",
                "report_html",
            )
        ):
            path = root / f"artifact-{idx}.bin"
            files[role] = path
            if role == "diffraction":
                data = {"d_spacings": [{"unit": "nm", "value": 0.16} for _ in range(4)]}
            elif role == "report_manifest":
                data = {
                    "report_type": "analysis",
                    "run": {"run_id": "run1"},
                    "completeness": {"required_complete": True},
                }
            else:
                data = {"fixture_role": role}
            path.write_text(json.dumps(data), encoding="utf-8")
            db.execute(
                "INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,1)",
                (
                    "run1",
                    "artifact" + str(idx),
                    role,
                    "metrics",
                    str(path),
                    sha(path),
                    path.stat().st_size,
                ),
            )
    return project, source, release, files, dbfile


def test_reloads_every_artifact_and_preserves_input_science_checks(scientific_store):
    project, source, release, files, _ = scientific_store
    receipt = evidence(project, "run1", release, source, {"diffraction.metrics"})
    assert receipt["job_id"] == "job1" and len(receipt["artifacts"]) == 6
    assert receipt["d_spacings_nm"] == [0.16] * 4
    assert receipt["reports"]["report_html"] == sha(files["report_html"])
    files["fft_complex"].write_bytes(files["fft_complex"].read_bytes().replace(b"fft", b"old"))
    with pytest.raises(GateError, match="artifact_bytes"):
        evidence(project, "run1", release, source, {"diffraction.metrics"})


def test_rejects_another_runtime_and_stale_plan(scientific_store):
    project, source, release, _, dbfile = scientific_store
    wrong = {**release, "runtime_id": "rt-" + "b" * 20}
    with pytest.raises(GateError, match="bound_runtime_plan_job"):
        evidence(project, "run1", wrong, source, {"diffraction.metrics"})
    with sqlite3.connect(dbfile) as db:
        db.execute("UPDATE runs SET request='{}'")
    with pytest.raises(GateError, match="bound_runtime_plan_job"):
        evidence(project, "run1", release, source, {"diffraction.metrics"})


def test_refuses_external_artifact_path_even_when_digest_is_correct(scientific_store, tmp_path):
    project, source, release, files, dbfile = scientific_store
    external = tmp_path / "external.bin"
    external.write_bytes(files["raw"].read_bytes())
    with sqlite3.connect(dbfile) as db:
        db.execute("UPDATE artifacts SET path=? WHERE role='raw'", (str(external),))
    with pytest.raises(GateError, match="outside_or_missing"):
        evidence(project, "run1", release, source, {"diffraction.metrics"})


def test_unsubstantiated_public_pass_fails_offline_recheck(tmp_path):
    bundles = []
    for name in ("previous", "current"):
        bundle = tmp_path / name
        bundle.mkdir()
        launcher = bundle / "AutoTEM-Setup.exe"
        launcher.write_bytes(name.encode())
        (bundle / "release-manifest.json").write_text(
            json.dumps(
                {
                    "artifacts": [{"path": launcher.name, "sha256": sha(launcher)}],
                    "release_id": name,
                    "source": {"archive_sha256": "a" * 64},
                    "package_version": "0.1.0",
                }
            )
        )
        bundles.append(bundle)
    doc = tmp_path / "upgrade.json"
    doc.write_text(json.dumps({"schema_version": "1.0.0", "case_verdict": "passed", "checks": []}))
    with pytest.raises(GateError, match="upgrade_evidence_schema"):
        recheck(doc, bundles[1], bundles[0], tmp_path / "project")


def test_transport_recovery_requires_original_failure_and_reused_worker(tmp_path):
    project = tmp_path / "project"
    private = project / ".upgrade-verifier"
    digest = "c" * 64
    previous = {
        "manifest_sha256": digest,
        "runtime_id": "rt-" + digest[:20],
        "launcher_sha256": "a" * 64,
    }
    error = {"code": "MCP_HANDSHAKE_FAILED"}
    failure = {
        "installation_id": "prior",
        "manifest_sha256": digest,
        "error": error,
        "runtime_diagnostic": {"local_mcp": {"code": "MCP_PROBE_TIMEOUT"}},
        "checks": ["pip_check_passed", "installed_profile_versions_and_origins_passed"],
    }
    original = private / "maintenance/install_previous-mcp-timeout.json"
    save(original, failure)
    save(
        private / "initial-qualified-attempt.json",
        {
            "case_verdict": "failed",
            "failure": {"phase": "install_previous"},
            "previous_manifest_sha256": digest,
            "verifier_sha256": "d" * 64,
            "maintenance": [{"record_sha256": sha(original), "elapsed_seconds": 85.0}],
        },
    )
    save(
        project / ".autotem/installations/prior/transaction.json",
        {
            "state": "failed",
            "manifest_sha256": digest,
            "error": error,
        },
    )
    attempt = private / "recovery/attempt-001.json"
    marker = attempt.with_suffix(".marker")
    native = attempt.with_suffix(".native")
    save(
        marker,
        {"project": str(project), "state": "failed", "error": error, "manifest_sha256": digest},
    )
    save(
        native,
        {
            "ok": True,
            "manifest_sha256": digest,
            "runtime_id": previous["runtime_id"],
            "state": "host_setup_required",
            "local_transport": "passed",
        },
    )
    root = project / ".autotem/runtimes" / previous["runtime_id"]
    exe = root / "Scripts/python.exe"
    worker = root / "Lib/site-packages/autotem/harness/worker.py"
    for path in (exe, worker):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"original installed bytes")
    receipt = {
        **previous,
        "verifier_sha256": sha(Path(verify_setup_retry.__file__)),
        "native_sha256": sha(native),
        "before_marker_sha256": sha(marker),
        "initial_failure_sha256": sha(original),
        "before_pointer": "absent",
        "user_config_before_sha256": hashlib.sha256(USER_CONFIG).hexdigest(),
        "ok": True,
        "state": "host_setup_required",
        "error_code": None,
        "exit_code": 0,
        "probe_timeout_seconds": 60,
        "elapsed_seconds": 8.0,
        "installed_files_reused": True,
        "python_sha256": sha(exe),
        "worker_sha256": sha(worker),
    }
    save(attempt, receipt)
    assert recovery_evidence(project, previous)["attempts"] == [receipt]
    worker.write_bytes(b"different worker")
    with pytest.raises(GateError, match="reused_binary_or_worker"):
        recovery_evidence(project, previous)
    save(original, {**failure, "error": {"code": "DIFFERENT"}})
    with pytest.raises(GateError, match="first_native_probe_failure"):
        recovery_evidence(project, previous)
