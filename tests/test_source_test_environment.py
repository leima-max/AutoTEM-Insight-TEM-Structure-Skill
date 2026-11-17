"""Contract tests for offline source-test setup and release qualification receipts."""

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
import zipfile

import pytest

from autotem.installation import release
from tools.release import build_release, qualify_release, run_source_tests
from tools.release import source_test_environment as env


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return env.sha(path)


def archive(path, files):
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for name, content in files.items():
            z.writestr(name, content)
    return env.sha(path)


def git(root, *args, input_text=None, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        input=input_text,
        capture_output=True,
        encoding="utf-8",
        check=check,
    )


def test_replay_uses_frozen_removals_and_preserves_git_support_files(tmp_path):
    repo = tmp_path / "源 worktree"
    repo.mkdir()
    git(repo, "init", "--quiet")
    initial = {
        "skills/autotem/SKILL.md": b"fixture\n",
        "autotem/harness/resources/skill/SKILL.md": b"fixture\n",
        "autotem/__init__.py": b"# fixture\n",
        "tests/test_retired.py": b"def test_retired(): pass\n",
        "configs/harness/obsolete.json": b"{}\n",
        "configs/harness/catalog/nested/entry.txt": b"previous\n",
        "configs/harness/single": b"previous\n",
        "support.txt": b"from the support commit\n",
    }
    for name, content in initial.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    git(repo, "-c", "core.autocrlf=false", "add", ".")
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "commit.gpgSign=false", "commit", "-qm", "frozen support")
    removed = [
        "tests/test_retired.py", "configs/harness/obsolete.json",
        "configs/harness/catalog/nested/entry.txt", "configs/harness/single",
    ]
    git(repo, "rm", "--quiet", "--", removed[1])
    for name in (removed[0], *removed[2:]):
        (repo / name).unlink()
    (repo / "configs/harness/catalog/nested").rmdir()
    (repo / "configs/harness/catalog").rmdir()
    (repo / "configs/harness/catalog").write_bytes(b"flat catalog\n")
    (repo / "configs/harness/single").mkdir()
    (repo / "configs/harness/single/child.txt").write_bytes(b"child\n")
    (repo / "tests/test_new.py").write_bytes(b"def test_new(): pass\n")
    (repo / "support.txt").write_bytes(b"uncommitted support\n")

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    frozen = bundle / "source.zip"
    index = build_release.snapshot(repo, frozen)
    assert index["deleted_from_commit"] == sorted(removed)
    source = {
        "git_commit": index["git_commit"], "archive": frozen.name,
        "archive_sha256": release.sha256(frozen), "inventory_member": build_release.SNAPSHOT_INDEX,
        "file_count": len(index["files"]),
    }
    checkout = tmp_path / "checkout"
    assert run_source_tests.stage_checkout(bundle, source, checkout, repo, release) == index
    assert (checkout / "support.txt").read_text(encoding="utf-8") == "from the support commit\n"
    for name in removed:
        assert not (checkout / name).is_file()
    for name in index["files"]:
        assert (checkout / name).read_bytes() == (repo / name).read_bytes()
    restored = checkout / removed[1]
    restored.write_bytes(b"stale\n")
    with pytest.raises(ValueError, match="Removed source file is still present"):
        run_source_tests.checked_sources(checkout, index["files"], index["deleted_from_commit"])


@pytest.mark.parametrize("deleted", ["bad", ["../invalid"], [".git/config"], ["kept"], ["x", "x"]])
def test_removals_are_unique_relative_files_outside_archive(tmp_path, deleted):
    path = tmp_path / "snapshot.zip"
    index = {"schema_version": release.SCHEMA, "git_commit": "a" * 40,
             "files": {"kept": hashlib.sha256(b"keep").hexdigest()}, "deleted_from_commit": deleted}
    archive(path, {"index.json": json.dumps(index), "kept": b"keep"})
    with zipfile.ZipFile(path) as frozen, pytest.raises(ValueError):
        release.verify_snapshot(frozen, "index.json")


@pytest.mark.parametrize("shallow", [False, True])
def test_archive_fetches_only_non_head_commit_from_local_support(tmp_path, shallow):
    repo, staged = tmp_path / "支持 worktree", tmp_path / "frozen"
    repo.mkdir()
    staged.mkdir()
    git(repo, "init", "--quiet")
    commit = [
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgSign=false",
        "commit",
        "--quiet",
        "-m",
    ]
    git(repo, *commit, "older support", "--allow-empty")
    (repo / "autotem").mkdir()
    (repo / "autotem/__init__.py").write_bytes(b"# Git base\n")
    (repo / "support.py").write_bytes(b"fixture support\n")
    git(repo, "add", "support.py", "autotem/__init__.py")
    git(repo, *commit, "frozen support")
    head = git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "support.py").write_bytes(b"newer support\n")
    git(repo, "add", "support.py")
    git(repo, *commit, "new HEAD")
    latest = git(repo, "rev-parse", "HEAD").stdout.strip()
    assert latest != head
    support = repo
    if shallow:
        support = tmp_path / "shallow support"
        support.mkdir()
        git(support, "init", "--quiet")
        git(support, "fetch", "--depth=2", "--no-tags", repo.as_uri(), latest)
        git(support, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        assert git(support, "rev-parse", "--is-shallow-repository").stdout.strip() == "true"
        assert not git(support, "for-each-ref", "--format=%(refname)", "refs/heads").stdout
    unused = git(
        support, "hash-object", "-w", "--stdin", input_text="unrelated private object"
    ).stdout.strip()
    files = {"tests/test_x.py": b"def test_x(): pass\n", "autotem/__init__.py": b"# source\n"}
    index = {
        "schema_version": "1.0.0",
        "git_commit": head,
        "files": {n: hashlib.sha256(v).hexdigest() for n, v in files.items()},
    }
    snap = staged / "src.zip"
    dig = archive(snap, {**files, "index.json": json.dumps(index)})
    info = {
        "archive": "src.zip",
        "archive_sha256": dig,
        "inventory_member": "index.json",
        "file_count": 2,
        "git_commit": head,
    }
    (support / "support.py").write_bytes(b"uncommitted support\n")
    checkout = tmp_path / "checkout"
    assert not (staged / ".git").exists()
    assert run_source_tests.stage_checkout(staged, info, checkout, support, release) == index
    assert git(checkout, "rev-parse", "HEAD").stdout.strip() == head
    assert git(checkout, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert git(checkout, "symbolic-ref", "-q", "HEAD", check=False).returncode == 1
    for obj in (unused, latest):
        assert git(checkout, "cat-file", "-e", obj, check=False).returncode != 0
    assert (checkout / "support.py").read_text(encoding="utf-8") == "fixture support\n"
    assert (checkout / "tests/test_x.py").read_bytes() == files["tests/test_x.py"]
    assert (checkout / "autotem/__init__.py").read_bytes() == files["autotem/__init__.py"]


@pytest.fixture
def receipts(tmp_path):
    report = tmp_path / "source-regression.json"
    extra = report.with_suffix(".test.lock.txt")
    extra.write_text("pytest==9 --hash=sha256:" + "1" * 64 + "\n", encoding="ascii")
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    source = {
        "archive": "source.zip",
        "file_count": 1,
        "git_commit": "abc",
        "archive_sha256": archive(bundle / "source.zip", {env.TEST_LOCK: extra.read_bytes()}),
    }
    runtime = {
        "archive": "psf.nupkg",
        "sha256": archive(bundle / "psf.nupkg", {"tools/python.exe": b"psf"}),
    }
    manifests = {}
    for name, profile in env.PROFILES.items():
        lock = report.with_suffix(f".{name}.lock.txt")
        all_pins = {"autotem-insight": ("1", "2" * 64)}
        if name == "simulation":
            all_pins["diffsims"] = ("0.7", "3" * 64)
        lock.write_text(
            "".join(f"{n}=={v} --hash=sha256:{h}\n" for n, (v, h) in all_pins.items()),
            encoding="ascii",
        )
        records = [
            {"path": "locks/deps.txt", "sha256": env.sha(lock)},
            {"path": runtime["archive"], "sha256": runtime["sha256"]},
        ]
        records.extend(
            {"path": f"wheelhouse/{n.replace('-', '_')}-{v}-py3-none-any.whl", "sha256": h}
            for n, (v, h) in all_pins.items()
        )
        raw = {
            "profiles": {
                profile: {
                    "lock": "locks/deps.txt",
                    "lock_sha256": env.sha(lock),
                    "wheelhouse": "wheelhouse",
                }
            },
            "source": source,
            "runtime": runtime,
            "pip_version": "25",
            "target": {"python_version": "3.12.10"},
            "artifacts": records,
        }
        manifests[name] = write(report.with_suffix(f".{name}-manifest.json"), raw)
        if name == "core":
            write(bundle / "release-manifest.json", raw)
    base_hash = hashlib.sha256(b"psf").hexdigest()
    observer = {
        "python_version": "3.12.10",
        "isolated": True,
        "user_site": False,
        "base_executable_sha256": base_hash,
        "editable_source": "autotem/__init__.py",
        "all_distributions_in_test_venv": True,
        "installed": {"pip": "25", "autotem-insight": "1", "diffsims": "0.7", "pytest": "9"},
    }
    environment = {
        "schema_version": "1.0.0",
        "execution_layer": "isolated_source_test_venv",
        "probe": observer,
        "core_manifest_sha256": manifests["core"],
        "simulation_manifest_sha256": manifests["simulation"],
        "test_lock_sha256": env.sha(extra),
        "test_lock_origin": "archived_source",
        "test_wheels": {"pytest": {"filename": "pytest-9-py3-none-any.whl", "sha256": "1" * 64}},
        "psf_executable_sha256": base_hash,
        "pip_check": "passed",
    }
    nodes = [
        "tests/test_b4_workflow_integration.py::test_real_diffsims_workflow_closes_simulation_phase_matching_loop",
        "tests/test_harness_release_operation_variants.py::test_optional_real_simulation_and_matching_from_disk",
    ]
    suite = ET.Element("testsuite", tests="2", failures="0", errors="0", skipped="0")
    for node in nodes:
        case = ET.SubElement(suite, "testcase", name=node.split("::")[1])
        props = ET.SubElement(case, "properties")
        ET.SubElement(props, "property", name="autotem_nodeid", value=node)
    junit = report.with_suffix(".junit.xml")
    ET.ElementTree(suite).write(junit, encoding="utf-8")
    data = {
        "schema_version": "1.2.0",
        "release_manifest_sha256": manifests["core"],
        "simulation_manifest_sha256": manifests["simulation"],
        "source_archive_sha256": source["archive_sha256"],
        "source_files_byte_verified": 1,
        "support_git_commit": "abc",
        "execution_layer": "source_checkout",
        "python_version": "3.12.10",
        "pytest_selection": "all",
        "pytest_default_smoke_overridden": True,
        "optional_diffsims_enabled": True,
        "optional_tests_passed": [n.split("::")[1] for n in nodes],
        "case_verdict": "passed",
        "results": {
            "collected": 2,
            "tests": 2,
            "passed": 2,
            "errors": 0,
            "failures": 0,
            "skipped": 0,
            "deselected": 0,
        },
        "environment_sha256": write(report.with_suffix(".environment.json"), environment),
        "junit_sha256": env.sha(junit),
        "collection_sha256": write(
            report.with_suffix(".collection.json"),
            {
                "selected": nodes,
                "deselected": [],
                "targets": ["tests"],
                "keyword": "",
                "marker": "",
            },
        ),
    }
    write(report, data)
    return report, bundle, data, environment


def check(report, data, **kwargs):
    return qualify_release.source_tests(
        report, data["release_manifest_sha256"], data["source_archive_sha256"], 1, **kwargs
    )


def test_isolated_regression_replays_with_and_without_original_bundle(receipts):
    report, bundle, data, _ = receipts
    portable = check(report, data, simulation_identity=data["simulation_manifest_sha256"])
    assert portable["verdict"] == portable["environment"]["verdict"] == "passed"
    assert portable == check(report, data, bundle=bundle)
    with pytest.raises(ValueError, match="simulation or core"):
        check(report, data, simulation_identity="0" * 64)
    report.with_suffix(".core.lock.txt").write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="release lock changed"):
        check(report, data)


@pytest.mark.parametrize("fault", [None, "skipped_startup", "mixed_import", "wrong_wheel"])
def test_wheel_startup_receipt_requires_source_import_and_exact_junit_node(receipts, fault):
    report, _, data, environment = receipts
    init = "a" * 64
    environment["schema_version"] = "1.1.0"
    observed = environment["probe"]
    del observed["editable_source"]
    observed.update(
        installed_module="autotem/__init__.py",
        checkout_on_sys_path=False,
        verified_package_members=2,
        wheel_sha256="e" * 64 if fault == "wrong_wheel" else "2" * 64,
        wheel_init_sha256=init,
    )
    data["schema_version"] = "1.3.0"
    data["environment_sha256"] = write(report.with_suffix(".environment.json"), environment)
    selection_path = report.with_suffix(".collection.json")
    selection = json.loads(selection_path.read_bytes())
    selection["source_import"] = {
        "path": "autotem/__init__.py",
        "sha256": "f" * 64 if fault == "mixed_import" else init,
    }
    selection["selected"].append(env.DEPLOYMENT_NODE)
    data["collection_sha256"] = write(selection_path, selection)
    junit = report.with_suffix(".junit.xml")
    suite = ET.parse(junit).getroot()
    suite.set("tests", "3")
    case = ET.SubElement(suite, "testcase", name=env.DEPLOYMENT_NODE.split("::")[1])
    props = ET.SubElement(case, "properties")
    ET.SubElement(props, "property", name="autotem_nodeid", value=env.DEPLOYMENT_NODE)
    skipped = fault == "skipped_startup"
    if skipped:
        ET.SubElement(case, "skipped")
    suite.set("skipped", str(int(skipped)))
    ET.ElementTree(suite).write(junit, encoding="utf-8")
    data["junit_sha256"] = env.sha(junit)
    data["results"].update(tests=3, collected=3, skipped=int(skipped), passed=3 - int(skipped))
    write(report, data)
    if fault in {"wrong_wheel", "mixed_import"}:
        with pytest.raises(ValueError, match="wheel|Wheel"):
            check(report, data)
    else:
        result = check(report, data)
        expected = "failed" if skipped else "passed"
        assert result["verdict"] == result["environment"]["wheel_entrypoint"] == expected
        assert result["collection"]["source_import"] == selection["source_import"]


@pytest.mark.parametrize("bad", ["global", "wheel", "user-site", "declaration", "psf"])
def test_changed_or_ambient_source_environment_cannot_pass(receipts, bad):
    report, bundle, data, environment = receipts
    if bad == "global":
        environment["probe"]["installed"]["outside"] = "1"
    if bad == "wheel":
        environment["test_wheels"]["pytest"]["sha256"] = "0" * 64
    if bad == "user-site":
        environment["probe"]["user_site"] = True
    if bad == "declaration":
        environment["test_lock_origin"] = "explicit_legacy_lock"
    if bad == "psf":
        environment["probe"]["base_executable_sha256"] = environment["psf_executable_sha256"] = (
            "0" * 64
        )
    data["environment_sha256"] = write(report.with_suffix(".environment.json"), environment)
    write(report, data)
    with pytest.raises(ValueError):
        check(report, data, bundle=bundle)


def test_test_lock_rejects_url_and_unhashed_package(tmp_path):
    p = tmp_path / "bad.txt"
    for content in [
        "--extra-index-url https://example.invalid",
        "pytest>=8",
        "pytest @ file:///tmp/fake.whl",
    ]:
        p.write_text(content, encoding="ascii")
        with pytest.raises(ValueError, match="hash-locked"):
            env.pins(p)


def test_source_runner_clears_ambient_test_selection_and_product_runtime(tmp_path, monkeypatch):
    for key in ("PYTEST_ADDOPTS", "AUTOTEM_RUNTIME_ID", "NUMBA_DISABLE_JIT", "PYTHONPATH"):
        monkeypatch.setenv(key, "foreign")
    configured = env.process_env(tmp_path)
    assert (
        not {"PYTEST_ADDOPTS", "AUTOTEM_RUNTIME_ID", "NUMBA_DISABLE_JIT", "PYTHONPATH"}
        & configured.keys()
    )
    assert (
        configured["AUTOTEM_RUN_REAL_DIFFSIMS_TESTS"]
        == configured["PYTEST_DISABLE_PLUGIN_AUTOLOAD"]
        == "1"
    )
