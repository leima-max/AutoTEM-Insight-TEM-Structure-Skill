"""Offline upgrade receipts must agree with their originating jobs and actual file bytes."""

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from tools.setup import verify_setup_retry
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
