"""CI needs an authenticated predecessor, different application code and bounded evidence."""

import hashlib
import json
import sqlite3
from contextlib import closing
from zipfile import ZipFile

import pytest

from tools.release import ci_delivery_records, ci_upgrade

pytestmark = pytest.mark.smoke


def digest(value):
    return hashlib.sha256(value).hexdigest()


def baseline(path, mutate=None):
    internal = {"source/autotem-source.zip": b"source", "wh/autotem_insight-0.1.whl": b"app"}
    external = {"wh/dep-0.1.whl": b"dependency", "runtime/python.nupkg": b"seed", "license": b"L"}
    meta = {
        "profiles": {ci_upgrade.PROFILE: {"wheelhouse": "wh"}},
        "runtime": {"archive": "runtime/python.nupkg", "license": "license"},
        "artifacts": [
            {"path": name, "size": len(data), "sha256": digest(data)}
            for name, data in {**internal, **external}.items()
        ],
    }
    raw = json.dumps(meta).encode()
    included = {ci_upgrade.MANIFEST: raw, **internal}
    if mutate:
        mutate(included)
    with ZipFile(path, "x") as archive:
        for name, data in included.items():
            archive.writestr(name, data)
    return digest(raw)


def test_sparse_predecessor_uses_external_inputs_and_both_pins(tmp_path):
    file = tmp_path / "previous.zip"
    pin = baseline(file)
    meta, files = ci_upgrade.sparse_files(file, pin, ci_upgrade.sha256(file))
    assert ci_upgrade.external_files(meta) == {"wh/dep-0.1.whl", "runtime/python.nupkg", "license"}
    assert set(files) == {
        ci_upgrade.MANIFEST,
        "source/autotem-source.zip",
        "wh/autotem_insight-0.1.whl",
    }
    with pytest.raises(ValueError, match="ZIP pin"):
        ci_upgrade.sparse_files(file, pin, "0" * 64)
    with pytest.raises(ValueError, match="manifest pin"):
        ci_upgrade.sparse_files(file, "0" * 64, ci_upgrade.sha256(file))
    with pytest.raises(ValueError, match="fixture is missing"):
        ci_upgrade.sparse_files(tmp_path / "absent.zip", pin, "0" * 64)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda f: f.pop("source/autotem-source.zip"), "missing, duplicate or unexpected"),
        (lambda f: f.update({"wh/dep-0.1.whl": b"third-party"}), "unexpected"),
        (lambda f: f.update({"source/autotem-source.zip": b"SOURCE"}), "Wrong hash"),
    ],
)
def test_sparse_predecessor_rejects_payload_substitution(tmp_path, mutate, reason):
    file = tmp_path / "previous.zip"
    pin = baseline(file, mutate)
    with pytest.raises(ValueError, match=reason):
        ci_upgrade.sparse_files(file, pin, ci_upgrade.sha256(file))


def test_a_renamed_or_repacked_application_is_not_a_code_upgrade(monkeypatch):
    old = {"manifest_sha256": "a", "source_sha256": "b", "wheel_sha256": "c"}
    new = {"manifest_sha256": "d", "source_sha256": "e", "wheel_sha256": "f"}
    code = {"autotem/module.py": "before"}
    records = {"previous": (old, code), "candidate": (new, code.copy())}
    monkeypatch.setattr(ci_upgrade, "generation", lambda path: records[path])
    with pytest.raises(ValueError, match="different shipped"):
        ci_upgrade.compare("previous", "candidate", "a")
    records["candidate"][1]["autotem/module.py"] = "after"
    with pytest.raises(ValueError, match="differs from its pin"):
        ci_upgrade.compare("previous", "candidate", "other")
    assert ci_upgrade.compare("previous", "candidate", "a")["changed_application_modules"] == [
        "autotem/module.py"
    ]


def test_private_maintenance_capture_includes_only_registered_bytes(tmp_path):
    store = tmp_path / "up/.autotem/runs"
    rid = "a" * 32
    raw = store / "runs" / rid / "raw/fixture.npy"
    art = store / "runs" / rid / "artifacts/result.json"
    for path in (raw, art, raw.with_name("unregistered.npy")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
    with closing(sqlite3.connect(store / "harness.sqlite3")) as db:
        db.executescript(
            "CREATE TABLE runs(run_id,input_path,source_hash);"
            "CREATE TABLE artifacts(run_id,path,sha256,committed);"
        )
        db.execute("INSERT INTO runs VALUES(?,?,?)", (rid, str(raw), ci_upgrade.sha256(raw)))
        db.execute("INSERT INTO artifacts VALUES(?,?,?,1)", (rid, str(art), ci_upgrade.sha256(art)))
        db.commit()
    target = tmp_path / "records.zip"
    assert ci_delivery_records.collect(tmp_path, target) == 3
    with ZipFile(target) as archive:
        index = json.loads(archive.read("registered-files.json"))
        assert len(index) == 3 and not any("unregistered" in member for member in index)
        assert index == {name: digest(archive.read(name)) for name in index}
    art.write_bytes(b"tampered")
    invalid = tmp_path / "invalid.zip"
    with pytest.raises(ValueError, match="Registered bytes changed"):
        ci_delivery_records.collect(tmp_path, invalid)
    with ZipFile(invalid) as archive:
        assert "collection-errors.json" in archive.namelist()
        assert archive.read(art.relative_to(tmp_path).as_posix()) == b"tampered"
