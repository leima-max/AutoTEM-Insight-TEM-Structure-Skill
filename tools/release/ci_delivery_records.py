"""Retain only SQLite-registered science from the two disposable maintenance projects."""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.release.ci_evidence import (  # noqa: E402
    SUFFIXES,
    checked_path,
    database_view,
    stored_path,
)


def collect(root, output):
    files = {}
    errors = []
    for project in ("up", "rp"):
        store = root / project / ".autotem/runs"
        if not (store / "harness.sqlite3").is_file():
            continue
        prefix = store.relative_to(root).as_posix() + "/"
        sqlite = checked_path(root, prefix + "harness.sqlite3").read_bytes()
        files[prefix + "harness.sqlite3"] = sqlite
        wal = b""
        if (store / "harness.sqlite3-wal").is_file():
            wal = checked_path(root, prefix + "harness.sqlite3-wal").read_bytes()
            if wal:
                files[prefix + "harness.sqlite3-wal"] = wal
        try:
            with database_view(sqlite, wal) as db:
                rows = db.execute(
                    "SELECT run_id, input_path, source_hash, 'raw' FROM runs "
                    "WHERE source_hash IS NOT NULL UNION ALL "
                    "SELECT run_id, path, sha256, 'artifacts' FROM artifacts WHERE committed=1"
                ).fetchall()
        except sqlite3.DatabaseError:
            errors.append(f"Unreadable preserved database in {project}")
            continue
        for rid, path, digest, part in rows:
            try:
                member = prefix + stored_path(str(store), rid, path, part)
                if Path(member).suffix.lower() not in SUFFIXES:
                    raise ValueError("Unexpected file type in the synthetic maintenance store")
                data = checked_path(root, member).read_bytes()
                files[member] = data
                if hashlib.sha256(data).hexdigest() != digest:
                    errors.append(f"Registered bytes changed in {project}/{rid}/{part}")
            except (TypeError, OSError, ValueError):
                errors.append(f"Registered path unavailable or unsafe in {project}/{rid}/{part}")
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    with ZipFile(output, "x", compression=ZIP_DEFLATED, compresslevel=3) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
        archive.writestr("registered-files.json", json.dumps(hashes, sort_keys=True))
        if errors:
            archive.writestr("collection-errors.json", json.dumps(errors))
    if errors:
        raise ValueError(f"{errors[0]}. Available original evidence was retained")
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps({"registered_store_files": collect(args.root.resolve(), args.out)}))
