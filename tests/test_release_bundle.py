"""Offline release preflight, source provenance and dependency lock regression cases."""

from __future__ import annotations

import ensurepip
import hashlib
import json
import zipfile
from types import SimpleNamespace

import pytest

from autotem.installation import release
from tools.release import build_release as builder
from tools.setup import verify_cli_profile

pytestmark = pytest.mark.smoke


def wheel(directory, name, *, requirement="", body=b"# source\n"):
    stem = name.replace("-", "_") + "-1.0"
    path = directory / (stem + "-py3-none-any.whl")
    metadata = f"Name: {name}\nVersion: 1.0\n" + requirement
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(stem + ".dist-info/METADATA", metadata)
        if name == release.APP:
            z.writestr("autotem/__init__.py", body)
    return path


def test_declared_license_paths_use_distribution_root_and_legacy_layout(tmp_path):
    metadata = (
        "Metadata-Version: 2.4\nLicense-Expression: MIT\nClassifier: License :: Example\n"
        "License-File: people/AUTHORS\nLicense-File: AUTHORS.rst\n"
        "License-File: MISSING\nLicense-File: people/AUTHORS\n"
    )
    path = wheel(tmp_path, "example", requirement=metadata)
    files = {
        "example-1.0.dist-info/licenses/people/AUTHORS": b"Named contributor\r\n",
        "example-1.0.dist-info/AUTHORS.rst": b"Legacy attribution\n",
        "example-1.0.dist-info/licenses/NOTICE": b"A notice",
        "vendor/COPYING.txt": b"An existing notice",
    }
    with zipfile.ZipFile(path, "a") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("example/people/AUTHORS", b"Unrelated users with the same basename")
        archive.writestr("example-1.0.dist-info/unrelated/AUTHORS", b"Not the declared path")
        archive.writestr("example/MISSING", b"A different path")
    assert builder.license_record(path, "example", "1.0") == {
        "name": "example", "version": "1.0", "wheel": path.name, "declaration": "MIT",
        "license_classifiers": ["License :: Example"], "review_status": "requires_review",
        "texts_in_distributed_wheel": [
            {"wheel_member": name, "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in files.items()
        ],
    }


@pytest.mark.parametrize("header", ["../AUTHORS", "/AUTHORS", "C:\\AUTHORS"])
def test_license_file_must_be_a_relative_wheel_member(tmp_path, header):
    path = wheel(tmp_path, "example", requirement=f"License-File: {header}\n")
    with pytest.raises(ValueError, match="path"):
        builder.license_record(path, "example", "1.0")


def test_supplementary_notices_require_bounded_text_evidence(tmp_path, monkeypatch):
    path = wheel(tmp_path, "example", requirement="License-File: missing/OWNERS\n")
    clause = b"Copyright Example\r\nRedistribution and use in source and binary forms"
    padded = clause.ljust(builder.NOTICE_PREFIX_BYTES - 1, b" ")
    accepted = {
        "library/array/NUMPY_LICENSE.txt": clause,
        "library/Scintilla-License.txt": b"Copyright N\nPermission to use, copy, modify, "
        b"and distribute the software",
        "library/MAPIStubLibrary-License.txt": b"Copyright Example\nPermission is hereby granted",
        "library/fonts/readme.txt": b"These metrics may be used, copied, \r\nand distributed "
        b"provided that all copyright notices are retained.",
        "library/unicode/readme.md": padded + "界".encode(),
    }
    with zipfile.ZipFile(path, "a") as archive:
        for name, content in accepted.items():
            archive.writestr(name, content)
        archive.writestr("docs/README.md", b"Copyright Example. See the license for reuse.")
        archive.writestr("docs/USAGE.txt", clause)
        archive.writestr("docs/README.rst", b"x" * builder.NOTICE_PREFIX_BYTES + clause)
        archive.writestr("docs/sample-license.txt", b"\0" + clause)
        archive.writestr("docs/invalid-license.txt", clause + b"\xff" + padded)
        archive.writestr("docs/incomplete-license.txt", padded + b"\xe7")
        archive.writestr("docs/title-notice.txt", b"The Copyright Act is of interest.")
        archive.writestr("library/binary.dll", clause)
        archive.writestr("library/version.py", b"# " + clause)
    reads = []
    read = zipfile.ZipExtFile.read

    def record_read(stream, size=-1):
        reads.append((stream.name, size))
        return read(stream, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", record_read)
    result = builder.license_record(path, "example", "1.0")
    assert result["declaration"] is None and result["review_status"] == "requires_review"
    assert result["texts_in_distributed_wheel"] == [
        {"wheel_member": k, "sha256": hashlib.sha256(v).hexdigest()} for k, v in accepted.items()
    ]
    assert not any(n in {"library/binary.dll", "library/version.py"} for n, _ in reads)
    assert [size for n, size in reads if n == "docs/README.rst"] == [builder.NOTICE_PREFIX_BYTES]


def test_ply_embedded_evidence_preserves_entire_member_hash_without_declaring_license(tmp_path):
    path = wheel(tmp_path, "ply")
    grant = (
        b"# Copyright (C) Example\r\n# Redistribution and use in source and binary forms\r\n"
        b"# are permitted under the following conditions.\r\n"
    )
    body = b"answer = 42\n" * 10000
    expected = {"ply/lex.py": grant + body, "ply/yacc.py": grant + b"from . import lex\n"}
    with zipfile.ZipFile(path, "a") as archive:
        for name, content in expected.items():
            archive.writestr(name, content)
        archive.writestr("ply/other.py", grant + body)
    row = builder.license_record(path, "ply", "1.0")
    assert row["texts_in_distributed_wheel"] == [
        {"wheel_member": k, "sha256": hashlib.sha256(v).hexdigest()} for k, v in expected.items()
    ]
    assert row["review_status"] == "requires_review"
    assert row["declaration"] is None and row["license_classifiers"] == []
    assert builder.license_record(path, "other", "1.0")["texts_in_distributed_wheel"] == []


def test_ply_ordinary_source_and_missing_project_license_stay_missing(tmp_path):
    path = wheel(tmp_path, "ply", requirement="License-File: absent/OWNERS\n")
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(
            "ply/lex.py", b"print('example')\n# Copyright; Permission is hereby granted"
        )
        archive.writestr("ply/yacc.py", b"# Copyright the project authors\npass\n")
    for name in ("ply", release.APP):
        row = builder.license_record(path, name, "1.0")
        assert row["texts_in_distributed_wheel"] == [] and row["declaration"] is None
        assert row["review_status"] == "missing"


@pytest.fixture
def miniature(tmp_path, monkeypatch, request):
    # Build a tiny local release to exercise the schema. Pip resolution is tested on Windows CI.
    source = tmp_path / "fixture.zip"
    profile = getattr(request, "param", release.PROFILE)
    policy = {**release.current_target(), "profile_id": profile,
              "seed_pip_version": ensurepip.version(),
              "python_nupkg": {"url": "https://example.invalid", "filename": "python.nupkg"},
              "requested_scope": {"journeys": ["fixture"]},
              "build_inputs_lock": "requirements/release/pins.txt",
              "wheel_sources": "configs/release/sources.json"}
    pool = tmp_path / "pool"
    pool.mkdir()
    projects = (["numpy"] if profile == "traditional-cli" else ["mcp"])
    if profile == "traditional-simulation":
        projects += ["diffsims", "orix", "diffpy-structure"]
    deps = [wheel(pool, name) for name in projects]
    data = {
        "autotem/__init__.py": b"# source\n",
        "autotem/harness/resources/skill/SKILL.md": b"fixture skill",
        "skills/autotem/SKILL.md": b"fixture skill",
        f"configs/release/{profile}.json": json.dumps(policy).encode("utf-8"),
        "requirements/release/pins.txt": "".join(
            f"{name}==1.0 --hash=sha256:{release.sha256(dep)}\n"
            for name, dep in zip(projects, deps, strict=True)
        ).encode("ascii"),
        "configs/release/sources.json": json.dumps({"schema_version": release.SCHEMA, "wheels": {
            dep.name: {"url": f"https://files.pythonhosted.org/packages/{dep.name}",
                       "sha256": release.sha256(dep)} for dep in deps
        }}).encode("ascii"),
    }
    index = {"schema_version": release.SCHEMA, "git_commit": "1" * 40,
             "files": {n: hashlib.sha256(d).hexdigest() for n, d in data.items()}}
    with zipfile.ZipFile(source, "w") as z:
        z.writestr(builder.SNAPSHOT_INDEX, json.dumps(index))
        for name, content in data.items():
            z.writestr(name, content)

    def collect(pool, bundle, source, scratch, index, profile):
        directory = bundle / "wheelhouse" / profile
        directory.mkdir(parents=True)
        app = wheel(directory, release.APP, requirement='Requires-Dist: mcp; extra == "mcp"\n')
        with zipfile.ZipFile(app, "a") as archive:
            archive.writestr("autotem/harness/resources/skill/SKILL.md", b"fixture skill")
        for dep in deps:
            (directory / dep.name).write_bytes(dep.read_bytes())
        return directory

    monkeypatch.setattr(builder, "collect_wheels", collect)
    bundle = tmp_path / "Bundle 空格"
    bundle.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    builder.assemble(SimpleNamespace(
        source_archive=source, source_sha256=release.sha256(source), wheelhouse=pool,
        python_nupkg=None, launcher=None, installer_app=None, release_id="fixture", profile=profile,
    ), bundle, scratch)
    return bundle


@pytest.mark.parametrize("miniature", ["traditional-cli", "traditional-simulation"], indirect=True)
def test_single_profile_binds_default_choice_exact_extras_and_explicit_selection(miniature):
    verified = release.verify_release(miniature)
    profile = verified.profile
    assert profile in {"traditional-cli", "traditional-simulation"}
    assert verified == release.verify_release(miniature, profile=profile)
    assert verified.with_mcp == (profile == "traditional-simulation")
    root = verified.lock.read_text().splitlines()[0]
    assert root.startswith(f"autotem-insight{release.root_extra(profile)}==1.0 ")
    with pytest.raises(release.ReleaseError) as exc:
        release.verify_release(miniature, profile=release.PROFILE)
    assert exc.value.code == "RELEASE_UNSUPPORTED"
    manifest = miniature / "release-manifest.json"
    mixed = release.unique_json(manifest.read_bytes())
    mixed["profiles"][release.PROFILE] = mixed["profiles"][profile]
    builder.write_json(manifest, mixed)
    with pytest.raises(release.ReleaseError) as exc:
        release.verify_release(miniature)
    assert exc.value.code == "RELEASE_UNSUPPORTED"


def test_profile_lock_rejects_mcp_in_cli_or_partial_simulation():
    wheels = dict.fromkeys([release.APP, "mcp"], ("1.0", "a" * 64))
    with pytest.raises(ValueError, match="AutoTEM/MCP"):
        release.release_lock(wheels, "traditional-cli")
    with pytest.raises(ValueError, match="simulation wheels"):
        release.release_lock({**wheels, "diffsims": ("0.7.0", "b" * 64)}, "traditional-simulation")


def test_cli_qualification_rejects_a_core_manifest_before_setup(miniature, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("The CLI verifier must validate the edition before running any command")

    monkeypatch.setattr(verify_cli_profile, "execute", unexpected)
    output = tmp_path / "cli-proof"
    result = verify_cli_profile.run(miniature, output)
    assert result["case_verdict"] == "failed" and "traditional-cli bundle" in result["error"]
    assert not list((output / "project").iterdir())
    assert json.loads((output / "cli-profile.json").read_bytes()) == result


def test_verify_pins_root_extra_seed_pip_and_all_artifacts(miniature):
    result = release.verify_release(miniature)
    assert result.wheel.is_absolute() and result.wheel.is_file()
    assert result.wheelhouse == result.wheel.parent
    assert result.with_mcp and result.pip_version == ensurepip.version()
    assert result.manifest_sha256 == release.sha256(miniature / "release-manifest.json")
    assert result.lock.read_text().splitlines() == [
        f"autotem-insight[mcp]==1.0 --hash=sha256:{release.sha256(result.wheel)}",
        f"mcp==1.0 --hash=sha256:{release.sha256(next(result.wheelhouse.glob('mcp-*.whl')))}",
    ]
    assert json.loads((miniature / "release-manifest.json").read_text())["acceptance"] == {
        "gates": dict.fromkeys(release.GATES, "not_evaluated"), "release_ready": False,
    }


@pytest.mark.parametrize("change,code", [
    ("extra", "RELEASE_INVENTORY_MISMATCH"), ("missing", "RELEASE_INVENTORY_MISMATCH"),
    ("wheel_digest", "RELEASE_HASH_MISMATCH"), ("inconsistent_lock", "RELEASE_LOCK_MISMATCH"),
    ("target", "RELEASE_TARGET_MISMATCH"), ("duplicate_key", "RELEASE_MANIFEST_INVALID"),
    ("path", "RELEASE_MANIFEST_INVALID"), ("duplicate_record", "RELEASE_MANIFEST_INVALID"),
    ("source", "RELEASE_MANIFEST_INVALID"),
])
def test_release_preflight_rejects_inconsistent_bundle(miniature, change, code):
    manifest_file = miniature / "release-manifest.json"
    m = json.loads(manifest_file.read_text())
    dep = next(miniature.glob("wheelhouse/*/mcp-*.whl"))
    if change == "missing":
        dep.unlink()
    elif change == "wheel_digest":
        dep.write_bytes(b"corrupted")
    elif change == "extra":
        (miniature / "unexpected.exe").write_bytes(b"extra")
    elif change == "inconsistent_lock":
        lock = next(miniature.glob("locks/*.txt"))
        lock.write_bytes(lock.read_bytes().replace(b"[mcp]", b""))
        item = next(a for a in m["artifacts"] if a["path"].startswith("locks/"))
        item.update(size=lock.stat().st_size, sha256=release.sha256(lock))
        m["profiles"][release.PROFILE]["lock_sha256"] = item["sha256"]
    elif change == "duplicate_key":
        manifest_file.write_text('{"release_id":"one","release_id":"two"}')
    elif change == "path":
        m["artifacts"][0]["path"] = "../outside.whl"
    elif change == "duplicate_record":
        m["artifacts"].append(m["artifacts"][0])
    elif change == "target":
        m["target"]["python_version"] = "0.0.0"
    elif change == "source":
        m["source"]["git_commit"] = "2" * 40
    if change != "duplicate_key":
        builder.write_json(manifest_file, m)
    with pytest.raises(release.ReleaseError) as exc:
        release.verify_release(miniature)
    assert exc.value.code == code


def test_wheel_cannot_redirect_offline_pip_onto_network(tmp_path):
    path = wheel(tmp_path, "mcp", requirement="Requires-Dist: demo @ https://example.invalid/w\n")
    with pytest.raises(ValueError, match="Direct URL"):
        release.wheel_identity(path)


def test_snapshot_contains_working_sources_mirrors_and_rebuild_tools(tmp_path):
    first = tmp_path / "source.zip"
    index = builder.snapshot(builder.ROOT, first)
    path = "autotem/installation/release.py"
    assert index["files"][path] == release.sha256(builder.ROOT / path)
    assert "tools/release/build_release.py" in index["files"]
    assert not any(n.startswith(("outputs/", ".git/", "data/")) for n in index["files"])
    with zipfile.ZipFile(first) as archive:
        assert release.verify_snapshot(archive, builder.SNAPSHOT_INDEX) == index


def test_build_pool_reports_dependency_changes_before_pip(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    dep = wheel(pool, "mcp")
    lock = tmp_path / "pins.txt"
    lock.write_text(f"MCP==1.0 --hash=sha256:{release.sha256(dep)}\n")
    builder.verify_pool(pool, lock)
    lock.write_text(lock.read_text().replace("==1.0", "==1.1"))
    with pytest.raises(ValueError, match="differs"):
        builder.verify_pool(pool, lock)


def test_wheel_must_match_exact_source_bytes(tmp_path):
    path = wheel(tmp_path, release.APP, body=b"older code")
    index = {"files": {"autotem/__init__.py": hashlib.sha256(b"current code").hexdigest()}}
    with pytest.raises(ValueError, match="Wheel/source byte mismatch"):
        release.verify_source_wheel(path, index)


@pytest.mark.parametrize("state", ["original", "changed", "missing", "extra", "historical"])
def test_wheel_data_files_match_the_declarations_of_that_release(tmp_path, state):
    path = wheel(tmp_path, release.APP)
    content = b"print('shipped diffraction preview')"
    index = {
        "files": {
            "autotem/__init__.py": hashlib.sha256(b"# source\n").hexdigest(),
            "apps/preview.py": hashlib.sha256(content).hexdigest(),
            "examples/apps/preview.py": hashlib.sha256(b"other").hexdigest(),
        }
    }
    project = b'[tool.setuptools.data-files]\n"share/autotem-insight/apps" = ["apps/*.py"]'
    member = "autotem_insight-1.0.data/data/share/autotem-insight/apps/"
    with zipfile.ZipFile(path, "a") as z:
        if state not in ("missing", "historical"):
            z.writestr(member + "preview.py", b"wrong" if state == "changed" else content)
        if state == "extra":
            z.writestr(member + "extra.py", content)
    if state == "historical":
        release.verify_source_wheel(path, index, b'[project]\nname = "autotem-insight"')
    elif state == "original":
        release.verify_source_wheel(path, index, project)
    else:
        with pytest.raises(ValueError, match="mismatch|inventory differs"):
            release.verify_source_wheel(path, index, project)


def test_zipapp_must_match_snapshot_and_fixed_entrypoint(tmp_path):
    fixture = {
        "autotem/__init__.py": b"# init",
        "autotem/bootstrap.py": b"# bootstrap",
        "autotem/harness/__init__.py": b"# harness",
        "autotem/harness/hosts.py": b"# hosts",
        "autotem/installation/managed.py": b"def main(): return 0",
    }
    index = {"files": {n: hashlib.sha256(b).hexdigest() for n, b in fixture.items()}}
    path = tmp_path / "install.pyz"
    with zipfile.ZipFile(path, "w") as app:
        for name, content in fixture.items():
            app.writestr(name, content)
        app.writestr("__main__.py",
                     b"from autotem.installation.managed import main\nraise SystemExit(main())\n")
    release.verify_installer_app(path, index)
    index["files"]["autotem/installation/managed.py"] = hashlib.sha256(b"changed").hexdigest()
    with pytest.raises(ValueError, match="Installer/source byte mismatch"):
        release.verify_installer_app(path, index)


@pytest.mark.parametrize("name", ["../data", "a\\file", "a:b", "CON.txt", "a/./file", "name."])
def test_bundle_names_are_portable_relative_paths(name):
    with pytest.raises(ValueError):
        release.portable_path(name)
