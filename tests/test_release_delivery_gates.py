import json
import zipfile
from types import SimpleNamespace

import pytest

from tools.release import ci_evidence as portable
from tools.release import qualify_release as release
from tools.setup import verify_install


def delivery_index(reconnect="passed"):
    gates = {
        "RG0": release.gate({"contracts": "passed"}),
        "RG1": release.gate({"installed_disk_journeys": "passed"}),
        "RG2": release.gate({"local_execution": "passed"}),
        "RG3": release.gate({"codex": "blocked"}),
        "RG4": release.gate(
            {
                "interrupted_install_recovery": "passed",
                "cross_release_upgrade_recovery": "passed",
                "post_switch_host_failure_recovery": reconnect,
                "codex.first_authorization": "not_evaluated",
            }
        ),
    }
    gates["RG5"] = release.gate(
        {
            **{name: row["verdict"] for name, row in gates.items()},
            "source_regression": "passed",
        }
    )
    return {"gates": gates, "release_ready": False}


@pytest.fixture
def cold_space(tmp_path, monkeypatch):
    bundle, evidence = tmp_path / "bundle", tmp_path / "evidence"
    bundle.mkdir()
    evidence.mkdir()
    (evidence / "project").mkdir()
    with zipfile.ZipFile(bundle / "python.zip", "w") as archive:
        archive.writestr("tools/python.exe", bytes(100))
    manifest = {
        "release_id": "fixture",
        "target": {},
        "source": {"archive_sha256": "c" * 64},
        "runtime": {"archive": "python.zip"},
        "artifacts": [],
    }
    path = bundle / "release-manifest.json"
    path.write_text(json.dumps(manifest))
    identity = release.sha(path)
    profile = "traditional-core-mcp"
    verifier = SimpleNamespace(
        PROFILE=profile, verify_release=lambda bundle: SimpleNamespace(manifest_sha256=identity)
    )
    monkeypatch.setattr(
        release,
        "load_module",
        lambda path: verifier if path.name == "release.py" else verify_install,
    )
    runtime = "rt-" + identity[:20]
    preview = dict(
        target_runtime_id=runtime,
        active_runtime_id=None,
        existing_project_control=False,
        minimum_free_bytes=100,
        estimated_new_payload_bytes=100,
        available_bytes=99,
        affected_project_paths=[
            ".codex/config.toml",
            ".agents/skills/autotem/",
            ".autotem/active-runtime.json",
            ".autotem/runtimes/" + runtime + "/",
            "autotem-host/codex/",
        ],
        host_authorization="宿主授权",
    )
    install = dict(case_verdict="failed", checks=[], release_manifest_sha256=identity)
    return bundle, evidence, preview, install


def test_low_space_receipt_is_bound_to_the_bundle_and_an_untouched_project(cold_space):
    bundle, evidence, preview, install = cold_space
    event = dict(
        state="blocked", applied=False, error={"code": "DISK_SPACE_LOW"}, space_preview=preview
    )
    (evidence / "installation.json").write_text(json.dumps({**install, "error": "DISK_SPACE_LOW"}))
    history = evidence / "maintenance.jsonl"
    history.write_text(json.dumps(event))
    report = release.failed_install(bundle, evidence)
    assert (
        report["gates"]["RG1"]["verdict"] == "failed" and report["failure_code"] == "DISK_SPACE_LOW"
    )
    assert report["gates"]["RG4"]["checks"]["post_switch_host_failure_recovery"] == "not_evaluated"
    (evidence / "project/.autotem").mkdir()
    with pytest.raises(ValueError, match="changed the project"):
        release.failed_install(bundle, evidence)
    preview["available_bytes"] = 101
    history.write_text(json.dumps(event))
    with pytest.raises(ValueError, match="Native preview"):
        release.failed_install(bundle, evidence)


def test_failed_transaction_is_still_consumed_after_successful_space_preview(cold_space):
    bundle, evidence, preview, install = cold_space
    preview["available_bytes"] = 100
    instance = "d" * 32
    install.update(
        checks=["read_only_preview", "project_space_and_host_write_preview"],
        project_preview=preview,
        error="VENV_FAILED",
    )
    (evidence / "installation.json").write_text(json.dumps(install))
    event = dict(
        state="blocked",
        applied=True,
        error={"code": "VENV_FAILED"},
        installation_id=instance,
        runtime_id=preview["target_runtime_id"],
    )
    (evidence / "maintenance.jsonl").write_text(json.dumps(event))
    receipt = evidence / "project/.autotem/installations" / instance / "transaction.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps(
            dict(
                state="failed",
                installation_id=instance,
                manifest_sha256=install["release_manifest_sha256"],
                error=event["error"],
            )
        )
    )
    report = release.failed_install(bundle, evidence)
    assert report["failure_code"] == "VENV_FAILED"
    assert report["disk_evidence_sha256"]["failed_transaction"] == release.sha(receipt)


@pytest.mark.parametrize(
    ("selector", "reconnect", "result"),
    [
        ("RG4.interrupted_install_recovery", "passed", 0),
        ("RG4.cross_release_upgrade_recovery", "passed", 0),
        ("RG4.post_switch_host_failure_recovery", "passed", 0),
        ("RG4.post_switch_host_failure_recovery", "not_evaluated", 1),
        ("RG5.source_regression", "passed", 0),
        ("RG4.codex.first_authorization", "passed", 1),
        ("RG4.check_without_evidence", "passed", 1),
    ],
)
def test_named_ci_requirement_preserves_actual_gate_verdicts(
    tmp_path, monkeypatch, selector, reconnect, result
):
    index = delivery_index(reconnect)
    monkeypatch.setattr(release, "evaluate", lambda args: index)
    output = tmp_path / "index.json"
    assert (
        release.main(
            [
                "--bundle",
                str(tmp_path),
                "--evidence",
                str(tmp_path),
                "--out",
                str(output),
                "--require",
                "RG0",
                "--require-check",
                selector,
            ]
        )
        == result
    )
    assert json.loads(output.read_bytes()) == index


def test_malformed_named_requirement_rejected_before_evaluation(tmp_path):
    output = tmp_path / "index.json"
    with pytest.raises(SystemExit) as error:
        release.main(
            [
                "--bundle",
                str(tmp_path),
                "--evidence",
                str(tmp_path),
                "--out",
                str(output),
                "--require-check",
                "RG4",
            ]
        )
    assert error.value.code == 2 and not output.exists()


def test_recovery_certificate_rechecked_from_selected_project(tmp_path, monkeypatch):
    observed = []
    identity, verifier_sha = "a" * 64, "b" * 64
    report = tmp_path / "recovery.json"
    project, bundle = tmp_path / "project", tmp_path / "bundle"
    args = SimpleNamespace(
        contract=None, local=None, headless=None, upgrade=None, recovery=(report, project)
    )
    receipt = dict(
        schema_version="1.0.0",
        case_verdict="passed",
        release_manifest_sha256=identity,
        verifier_sha256=verifier_sha,
    )
    report.write_text(json.dumps(receipt))

    def recheck(path, selected_bundle, selected_project):
        observed.append((path, selected_bundle, selected_project))
        return dict(case_verdict="passed", release_manifest_sha256=identity)

    monkeypatch.setattr(
        release,
        "tool",
        lambda name: {"setup/verify_recovery": SimpleNamespace(recheck=recheck)}[name],
    )
    assert release.certificates(args, bundle, tmp_path, identity) == {
        "recovery": dict(
            verdict="passed", report_sha256=release.sha(report), verifier_sha256=verifier_sha
        )
    }
    assert observed == [(report, bundle, project)]
    receipt["release_manifest_sha256"] = "c" * 64
    report.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="recovery certificate"):
        release.certificates(args, bundle, tmp_path, identity)


@pytest.mark.parametrize("has_reconnect", [True, False])
def test_upgrade_certificate_keeps_reconnect_with_the_original_report(
    tmp_path, monkeypatch, has_reconnect
):
    bundle, previous, project = (tmp_path / name for name in ("candidate", "previous", "project"))
    path = tmp_path / "upgrade.json"
    identity = "a" * 64
    report = dict(
        schema_version="1.0.0",
        release_manifest_sha256=identity,
        verifier_sha256="b" * 64,
        case_verdict="passed",
    )
    if has_reconnect:
        report["host_reconnect_recovery"] = {"verdict": "passed"}
    path.write_text(json.dumps(report))
    rechecked = dict(release_manifest_sha256=identity, case_verdict="passed")
    check = "post_switch_host_failure_recovery"
    verdict = "passed" if has_reconnect else "not_evaluated"
    rechecked[check] = verdict

    def recheck(*selected):
        assert selected == (path, bundle, previous, project)
        return rechecked

    monkeypatch.setattr(
        release,
        "tool",
        lambda name: {"setup/verify_upgrade": SimpleNamespace(recheck=recheck)}[name],
    )
    args = SimpleNamespace(
        contract=None, local=None, headless=None, upgrade=(path, previous, project), recovery=None
    )
    certificate = release.certificates(args, bundle, tmp_path, identity)["upgrade"]
    assert certificate == {
        "verdict": "passed",
        "verifier_sha256": report["verifier_sha256"],
        "report_sha256": release.sha(path),
        check: verdict,
    }
    rechecked[check] = "not_evaluated" if has_reconnect else "passed"
    with pytest.raises(ValueError, match="differs from its upgrade receipt"):
        release.certificates(args, bundle, tmp_path, identity)


def test_public_summary_derives_gate_and_release_state():
    index = delivery_index()
    assert portable.public_gate_verdicts(index)["RG5"] == "blocked"
    index["release_ready"] = True
    with pytest.raises(ValueError, match="readiness"):
        portable.public_gate_verdicts(index)
    index["release_ready"] = False
    index["gates"]["RG3"]["verdict"] = "passed"
    with pytest.raises(ValueError, match="RG3"):
        portable.public_gate_verdicts(index)
    index["gates"]["RG3"]["checks"]["codex"] = "passed"
    with pytest.raises(ValueError, match="RG5 differs from RG3"):
        portable.public_gate_verdicts(index)


@pytest.fixture
def maintenance_receipts(tmp_path):
    index = delivery_index()
    index["release_manifest_sha256"] = "a" * 64
    index["certificates"] = {}
    paths = []
    for name in ("upgrade", "recovery"):
        path = tmp_path / f"{name}.json"
        document = dict(
            release_manifest_sha256="a" * 64,
            verifier_sha256="b" * 64,
            case_verdict="passed",
            case=name,
        )
        if name == "upgrade":
            document["host_reconnect_recovery"] = {"verdict": "passed"}
        path.write_text(json.dumps(document))
        digest = portable.file_sha(path)
        index["certificates"][name] = dict(
            verdict="passed", verifier_sha256="b" * 64, report_sha256=digest
        )
        if name == "upgrade":
            index["certificates"][name]["post_switch_host_failure_recovery"] = "passed"
        paths.append(path)
    return index, [*paths, tmp_path / "source.json"]


@pytest.mark.parametrize("swapped", ["upgrade", "recovery"])
def test_public_delivery_refuses_a_different_maintenance_receipt(maintenance_receipts, swapped):
    index, paths = maintenance_receipts
    path = next(path for path in paths if path.stem == swapped)
    path.write_text(json.dumps({**json.loads(path.read_bytes()), "case": "another execution"}))
    with pytest.raises(ValueError, match=f"different {swapped} receipt"):
        portable.delivery_receipts(index, paths, {}, "b" * 64)


@pytest.mark.parametrize("origin", ["archived_source", "explicit_legacy_lock", None])
@pytest.mark.parametrize("historical", [False, True])
def test_public_delivery_requires_the_release_test_lock(
    maintenance_receipts, monkeypatch, origin, historical
):
    index, paths = maintenance_receipts
    check = "post_switch_host_failure_recovery"
    upgrade = paths[0]
    if historical:
        document = json.loads(upgrade.read_bytes())
        del document["host_reconnect_recovery"]
        upgrade.write_text(json.dumps(document))
        del index["gates"]["RG4"]["checks"][check]
        del index["certificates"]["upgrade"][check]
        index["certificates"]["upgrade"]["report_sha256"] = portable.file_sha(upgrade)
    simulation = "e" * 64
    collection = dict(verdict="passed", collected=2, deselected=0, sha256="c" * 64)
    environment = {"verdict": "not_evaluated"}
    if origin:
        environment = dict(
            verdict="passed",
            sha256="d" * 64,
            test_lock_sha256="f" * 64,
            simulation_manifest_sha256=simulation,
            test_lock_origin=origin,
        )
    tested = dict(
        verdict="passed",
        tests=2,
        passed=2,
        failures=0,
        errors=0,
        skipped=0,
        junit_sha256="a" * 64,
        report_sha256="b" * 64,
        collection=collection,
        environment=environment,
    )
    index["source_tests"] = tested
    monkeypatch.setattr(
        portable.runpy, "run_path", lambda path: {
            **vars(release), "source_tests": lambda *a, **k: tested
        }
    )
    core = {"source": {"archive_sha256": "c" * 64, "file_count": 1}}
    if origin == "archived_source":
        public = portable.delivery_receipts(index, paths, core, simulation)
        assert public[check] == {
            "verdict": "not_evaluated" if historical else "passed",
            "report_sha256": public["cross_release_upgrade_recovery"]["report_sha256"],
        }
        assert public[check]["report_sha256"] == portable.file_sha(upgrade)
        assert public["source_regression"]["collection"] == collection
        assert set(public["source_regression"]["environment"]) == {
            "verdict",
            "sha256",
            "simulation_manifest_sha256",
            "test_lock_sha256",
        }
    else:
        with pytest.raises(ValueError, match="isolated source test environment"):
            portable.delivery_receipts(index, paths, core, simulation)


@pytest.mark.parametrize("changed", ["gate", "certificate", "missing_summary", "invalid_summary"])
def test_public_delivery_rejects_an_unbound_reconnect_result(maintenance_receipts, changed):
    index, paths = maintenance_receipts
    check = "post_switch_host_failure_recovery"
    certificate = index["certificates"]["upgrade"]
    if changed == "gate":
        del index["gates"]["RG4"]["checks"][check]
    elif changed == "certificate":
        certificate[check] = "not_evaluated"
    else:
        upgrade = paths[0]
        document = json.loads(upgrade.read_bytes())
        if changed == "missing_summary":
            del document["host_reconnect_recovery"]
        else:
            document["host_reconnect_recovery"] = {}
        upgrade.write_text(json.dumps(document))
        certificate["report_sha256"] = portable.file_sha(upgrade)
    with pytest.raises(ValueError, match="(?i)host reconnect recovery"):
        portable.delivery_receipts(index, paths, {}, "b" * 64)
