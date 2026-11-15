"""The synthetic CI archive must survive relocation and exclude unregistered project files."""

import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import pytest
import yaml

from tools.release import build_release
from tools.release import ci_evidence as proof

pytestmark = pytest.mark.smoke
ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value).encode())
    return proof.file_sha(path)


@pytest.fixture()
def cli(tmp_path_factory):
    scratch = tmp_path_factory.mktemp("ci")
    bundle, output = scratch / "b", scratch / "e"
    lock = write(bundle / "locks/cli.txt", "test lock")
    source = write(bundle / "source/source.zip", "source test double")
    license_sha = write(bundle / "licenses/inventory.json", {"review": "not_evaluated"})
    wheel = "a" * 64
    manifest = {
        "profiles": {proof.CLI: {"lock": "locks/cli.txt", "lock_sha256": lock, "wheelhouse": "wh"}},
        "source": {"archive": "source/source.zip", "archive_sha256": source},
        "licenses": {"inventory": "licenses/inventory.json", "sha256": license_sha},
        "artifacts": [{"path": "wh/autotem_insight-0.1-py3-none-any.whl", "sha256": wheel}],
    }
    release = write(bundle / "release-manifest.json", manifest)
    identity = {
        "profile": proof.CLI,
        "manifest_sha256": release,
        "wheel_sha256": wheel,
        "source_archive_sha256": source,
        "lock_sha256": lock,
    }
    rid, aid, jid = "a" * 32, "b" * 32, "c" * 32
    store = output / "project/.autotem/runs"
    run = store / "runs" / rid
    raw, result = run / "raw/fixture.npy", run / "artifacts" / (aid + ".json")
    snapshot = write(raw, "test pixels")
    digest = write(
        result,
        {
            "phase_candidates": [{"name": "synthetic-cubic"}],
            "d_spacings": [{"value": 0.16, "unit": "nm"}] * 4,
        },
    )
    # Windows can relocate this fixture only after its SQLite handle is closed.
    with closing(sqlite3.connect(store / "harness.sqlite3")) as db:
        db.executescript("""
            CREATE TABLE runs(run_id,input_path,source_hash,job_id,status,error);
            CREATE TABLE jobs(ended);
            CREATE TABLE artifacts(run_id,artifact_id,path,sha256,size);
            INSERT INTO jobs VALUES (1);
        """)
        db.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?)",
            (rid, str(raw), snapshot, jid, "completed", None),
        )
        db.execute(
            "INSERT INTO artifacts VALUES (?,?,?,?,?)",
            (rid, aid, str(result), digest, result.stat().st_size),
        )
        db.commit()
    case = {
        "run_id": rid,
        "inspection_run_id": rid,
        "job_id": jid,
        "snapshot_sha256": snapshot,
        "claims": [
            {"value": [0.16] * 4, "unit": "nm", "artifact_id": aid},
            {"value": "synthetic-cubic"},
        ],
        "artifacts": [{"artifact_id": aid, "sha256": digest}],
    }
    science = {
        "case": case,
        "runtime": {"release_identity": identity},
        "local_mcp": "not_in_profile",
    }
    h = write(output / "science.json", science)
    write(
        output / "cli-profile.json",
        {
            "case_verdict": "passed",
            "release_manifest_sha256": release,
            "science_sha256": h,
            "science": {"run_id": rid},
        },
    )
    write(
        output / "installation.json",
        {
            "local_mcp": "not_in_profile",
            "profile": proof.CLI,
            "manifest_sha256": release,
            "transport": "jsonl",
        },
    )
    write(output / "tool-trace.json", [{"tool": "list_capabilities"}])
    write(output / "fixtures/fixture.npy", "test pixels")
    # These files must stay local even though one of them has a scientific data extension.
    write(run / "raw/unregistered.json", "private")
    write(output / "project/.codex/config.toml", "private")
    write(output / "project/.autotem/runtimes/rt-test/Scripts/python.exe", "mock interpreter")
    return bundle, output, raw


def test_archive_replays_without_project(cli, tmp_path):
    bundle, evidence, _ = cli
    out = tmp_path / "cli.zip"
    first = proof.collect(bundle, evidence, None, out)
    evidence.rename(tmp_path / "unavailable-project")
    bundle.rename(tmp_path / "unavailable-bundle")
    assert proof.verify(out, proof.file_sha(out)) == first
    assert first["runs"] == first["registered_artifacts"] == 1
    with ZipFile(out) as archive:
        names = archive.namelist()
        assert not any(n.endswith((".exe", "config.toml", "unregistered.json")) for n in names)
        assert any(n.endswith("/harness.sqlite3") for n in names)
    with pytest.raises(ValueError, match="ZIP digest"):
        proof.verify(out, "0" * 64)


@pytest.mark.parametrize(
    "fault", ["raw", "profile", "numeric", "pending", "unrelated-run", "overwrite"]
)
def test_invalid_or_unfinished_archive_is_rejected(cli, tmp_path, fault):
    bundle, evidence, raw = cli
    out = tmp_path / "cli.zip"
    if fault == "raw":
        raw.write_bytes(b"changed pixels")
    if fault in {"profile", "numeric"}:
        science = proof.read(evidence / "science.json")
        if fault == "profile":
            science["runtime"]["release_identity"]["profile"] = proof.SIM
        else:
            science["case"]["claims"][0]["value"][0] = 1.6
        receipt = proof.read(evidence / "cli-profile.json")
        receipt["science_sha256"] = write(evidence / "science.json", science)
        write(evidence / "cli-profile.json", receipt)
    if fault == "pending":
        with closing(sqlite3.connect(evidence / "project/.autotem/runs/harness.sqlite3")) as db:
            db.execute("UPDATE jobs SET ended=NULL")
            db.commit()
    if fault == "unrelated-run":
        with closing(sqlite3.connect(evidence / "project/.autotem/runs/harness.sqlite3")) as db:
            db.execute(
                "INSERT INTO runs SELECT ?,input_path,source_hash,job_id,status,error FROM runs",
                ("e" * 32,),
            )
            db.commit()
    if fault == "overwrite":
        out.write_bytes(b"historic evidence")
    with pytest.raises((ValueError, FileExistsError)):
        proof.collect(bundle, evidence, None, out)
    assert out.read_bytes() == b"historic evidence" if fault == "overwrite" else not out.exists()


def test_windows_release_triggers_on_every_frozen_source_rule():
    # BaseLoader preserves GitHub's literal key 'on'; YAML 1.1 otherwise reads it as True.
    workflow = yaml.load(
        (ROOT / ".github/workflows/windows-release.yml").read_bytes(), Loader=yaml.BaseLoader
    )
    triggers = set(workflow["on"]["pull_request"]["paths"])
    expected = {".github/release-baselines/**"}
    for pattern in build_release.SOURCE_PATHS:
        if (ROOT / pattern).is_dir():
            expected.add(pattern + "/**")
        elif pattern == "configs/*.yaml":
            # Git ls-files matches nested YAML with this pathspec. Actions' '*' stops at '/'.
            expected.add("configs/**/*.yaml")
        elif pattern.endswith("*") and "/" not in pattern:
            # The same Git pathspec includes entries in e.g. LICENSES/vendor.txt.
            expected.update({pattern, pattern + "/**"})
        else:
            expected.add(pattern)
    assert expected <= triggers, f"Missing Windows release triggers: {sorted(expected - triggers)}"
    assert {
        ".gitattributes", "apps/atom_detection_ui.py", "apps/b1_b3_preview_ui.py",
        "apps/b2_diffraction_ui.py", "apps/requirements.txt",
    } <= set(build_release.SOURCE_PATHS)


def test_shipped_app_sources_use_their_git_line_endings_on_windows(tmp_path):
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    (tmp_path / ".gitattributes").write_bytes((ROOT / ".gitattributes").read_bytes())
    shipped = [name for name in build_release.SOURCE_PATHS if name.startswith("apps/")]
    rules = subprocess.check_output(
        ["git", "-C", tmp_path, "-c", "core.autocrlf=true", "check-attr", "-z", "eol", "--", *shipped]
    ).decode().split("\0")[:-1]
    assert dict(zip(rules[::3], rules[2::3])) == dict.fromkeys(shipped, "lf")


def test_windows_release_triggers_on_source_suite_git_inputs():
    workflow = yaml.load(
        (ROOT / ".github/workflows/windows-release.yml").read_bytes(), Loader=yaml.BaseLoader
    )
    triggers = set(workflow["on"]["pull_request"]["paths"])
    suite_path = "benchmarks/suite_public_minimal.yaml"
    assert {
        "apps/b1_b3_preview_ui.py", "apps/b2_diffraction_ui.py", suite_path,
        "examples/diffsims_closed_loop.yaml", "examples/materials_project_diffsims_lookup.yaml",
    } <= triggers
    suite = yaml.safe_load((ROOT / suite_path).read_bytes())
    report_paths = [
        (PurePosixPath("benchmarks") / case["report_path"]).as_posix() for case in suite["cases"]
    ]
    assert report_paths
    report_pattern = "benchmarks/cases/*/analysis_report.json"
    for report in report_paths:
        # GitHub paths are case-sensitive and anchored at the repository root on every runner.
        assert report in triggers or (
            report_pattern in triggers and PurePosixPath("/", report).match("/" + report_pattern)
        ), report


def test_ci_joins_three_editions_before_publishing():
    workflow = yaml.safe_load((ROOT / ".github/workflows/windows-release.yml").read_bytes())
    [job] = workflow["jobs"].values()
    steps = {s["id"]: s for s in job["steps"] if "id" in s}
    assert list(steps) == [
        "build", "cold", "core", "cli", "sim", "source",
        "previous", "upgrade", "recovery", "combined",
    ]
    aggregate = steps["combined"]["run"]
    assert all(
        f"steps.{name}.outcome == 'success'" in steps["combined"]["if"]
        for name in ("core", "cli", "sim", "source", "upgrade", "recovery")
    )
    assert (
        "cq/project/.autotem" in aggregate
        and "/runtimes/$runtimeId/Scripts/python.exe" in aggregate
    )
    for flag in (
        "--simulation ",
        "--simulation-headless ",
        "--simulation-faults ",
        "--require RG2",
        "--source-tests ",
        "--upgrade ",
        "--recovery ",
        "--require-check RG4.cross_release_upgrade_recovery",
        "--require-check RG4.post_switch_host_failure_recovery",
        "--require-check RG4.interrupted_install_recovery",
        "--require-check RG5.source_regression",
        "--delivery-evidence ",
    ):
        assert flag in aggregate
    assert aggregate.index("qualify_release.py") < aggregate.index("ci_evidence.py")
    assert "steps.previous.outcome == 'success'" in steps["upgrade"]["if"]
    assert 'frozen/tools/setup/verify_upgrade.py' in steps["upgrade"]["run"]
    assert 'frozen/tools/setup/verify_recovery.py' in steps["recovery"]["run"]
    assert 'frozen/tools/release/run_source_tests.py' in steps["source"]["run"]
    assert '--support-repo "$env:GITHUB_WORKSPACE"' in steps["source"]["run"]
    uploads = [s for s in job["steps"] if "upload-artifact" in s.get("uses", "")]
    maintenance = next(
        s for s in uploads if s["with"]["name"] == "windows-private-delivery-records"
    )
    assert "github.event.repository.private" in maintenance["if"]
    assert "/up/.upgrade-verifier/reconnect/*" in maintenance["with"]["path"]
    assert "/up/.autotem/installations/*/transaction.json" in maintenance["with"]["path"]
    for step in uploads:
        paths = step["with"]["path"].splitlines()
        if "github.event.repository.private" not in step["if"]:
            assert len(paths) == 1 and paths[0].endswith("/public-summary.json")
        assert all("/simulation-fault-runtime/" not in p for p in paths)
        assert all(p.endswith("/rt-*/.autotem-runtime.json") for p in paths if "/runtimes/" in p)


def test_public_index_rejects_different_application_wheels(tmp_path, monkeypatch):
    archives = []
    for profile in (proof.CORE, proof.CLI, proof.SIM):
        path = tmp_path / (profile + ".zip")
        with ZipFile(path, "x") as archive:
            archive.writestr(proof.INDEX, json.dumps({"files": {}}))
        archives.append(path)

    def edition(path):
        return {
            "profile": path.stem,
            "release_manifest_sha256": path.stem,
            "source_archive_sha256": "same-source",
            "wheel_sha256": "other" if path.stem == proof.SIM else "same-wheel",
        }

    monkeypatch.setattr(proof, "verify", edition)
    index = tmp_path / "index.json"
    write(index, {**edition(archives[0]), "profile": proof.CORE})
    with pytest.raises(ValueError, match="Incompatible edition archives"):
        proof.publish(archives, index, tmp_path / "public.json")
    assert not (tmp_path / "public.json").exists()


def test_fault_archive_requires_its_actual_positive_and_fault_runs():
    prefix = "capture/simulation-faults/"
    files = {
        "capture/simulation/simulation.json": "positive",
        "capture/private-install.json": "setup",
        prefix + "backend-origin.json": "origin",
    }
    positive = {"run_id": "successful", "numerical_evidence": {"matched_spots": 4}}
    runs = {
        "child": {
            "status": "completed",
            "job": "child-job",
            "database": prefix + "missing_structure/harness/harness.sqlite3",
        }
    }
    cases = {}
    observed = {
        "missing_dependency": ("rejected", "DEPENDENCY_UNAVAILABLE"),
        "broken_dependency": ("failed", "DEPENDENCY_BROKEN"),
        "backend_timeout": ("failed", "SIMULATION_TIMEOUT"),
        "worker_timeout": ("timed_out", "TIMEOUT"),
        "missing_structure": ("needs_input", "NEEDS_INPUT"),
    }
    for name, (status, code) in observed.items():
        case = {
            "run_id": None,
            "verdict": "passed",
            "status": status,
            "code": code,
            "witness_sha256": "w",
            "plan_sha256": "p",
        }
        path = prefix + name + "/"
        database = path + "harness/harness.sqlite3"
        files.update(
            {
                path + "witness.json": "w",
                path + "plan.json": "p",
                path + "fixture.npy": "fixture",
                database: "db",
            }
        )
        if name != "missing_dependency":
            case.update(run_id=name, job_id=name, report_artifact_id="aid", report_sha256="report")
            runs[name] = dict(
                database=database,
                job=name,
                status=status,
                code=code,
                snapshot="fixture",
                artifacts={"aid": "report"},
            )
        cases[name] = case
    cases["missing_structure"]["child"] = {
        "run_id": "child",
        "job_id": "child-job",
        "numerical_evidence": positive["numerical_evidence"],
    }
    faults = {
        "release_manifest_sha256": "sim",
        "case_verdict": "passed",
        "positive_run_id": "successful",
        "positive_sha256": "positive",
        "installation_sha256": "setup",
        "backend_origin_sha256": "origin",
        "fixture_sha256": "fixture",
        "cases": cases,
    }
    proof.fault_evidence(faults, files, runs, "sim", positive)
    with pytest.raises(ValueError, match="different positive"):
        proof.fault_evidence(faults, files, runs, "core", positive)
    runs["worker_timeout"]["status"] = "completed"
    with pytest.raises(ValueError, match="persisted work"):
        proof.fault_evidence(faults, files, runs, "sim", positive)
