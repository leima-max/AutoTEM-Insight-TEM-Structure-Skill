"""Pack and independently check synthetic Windows CI evidence without an installed runtime."""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import runpy
import sqlite3
import struct
import tempfile
from contextlib import closing, contextmanager
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath, PureWindowsPath
from posixpath import dirname, join, normpath
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZipFile

CORE, CLI, SIM = "traditional-core-mcp", "traditional-cli", "traditional-simulation"
BACKENDS = {"numpy", "diffsims", "orix", "diffpy-structure", "numba", "llvmlite", "pycifrw"}
INDEX = "portable-manifest.json"
SUFFIXES = {
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
    ".tif",
    ".tiff",
    ".png",
    ".csv",
    ".html",
    ".svg",
    ".yaml",
    ".yml",
    ".pdf",
    ".log",
    ".txt",
}
PLUGIN_FIXTURES = {
    "qualification_pixel.py",
    "qualification_broken.py",
    "METADATA",
    "entry_points.txt",
    "release-libraries.pth",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return stream_sha(stream)


def stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def relative_name(name):
    path = PurePosixPath(name)
    require(
        path.parts
        and path.as_posix() == name
        and not path.is_absolute()
        and ":" not in name
        and not PureWindowsPath(name).drive
        and "\\" not in name
        and not set(path.parts) & {".", ".."},
        f"Unsafe evidence member: {name}",
    )
    return name


def read(path):
    return json.loads(path.read_bytes())


def checked_path(root, name):
    path = root.joinpath(*PurePosixPath(relative_name(name)).parts)
    for part in (path, *path.parents):
        if part == root:
            break
        require(
            not part.is_symlink() and not getattr(part.lstat(), "st_file_attributes", 0) & 0x400,
            f"Linked evidence: {part}",
        )
    require(path.resolve().is_relative_to(root), f"Outside evidence root: {path}")
    return path


def project_runs(profile, document):
    if profile == CORE:
        return {document("installation.json")["managed_project_inspection"]["run_id"]}
    if profile == CLI:
        case = document("science.json")["case"]
        return {case["inspection_run_id"], case["run_id"]}
    return {document("simulation/simulation.json")["run_id"]}


def simulation_metadata(bundle, evidence, manifest, files):
    runtime = "rt-" + file_sha(bundle / "release-manifest.json")[:20]
    site = f"project/.autotem/runtimes/{runtime}/Lib/site-packages/"
    versions = read(evidence / "simulation/simulation.json")["installed_backends"]
    require(set(versions) == BACKENDS, "Missing simulation backend identity")
    for row in manifest["artifacts"]:
        path = PurePosixPath(row["path"])
        package = re.sub(r"[-_.]+", "-", path.name.split("-")[0]).lower()
        if path.suffix != ".whl" or package not in versions:
            continue
        wheel = checked_path(bundle, path.as_posix())
        require(file_sha(wheel) == row["sha256"], f"Wrong locked wheel: {package}")
        with ZipFile(wheel) as archive:
            (meta,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            installed = checked_path(evidence, site + meta)
            require(
                installed.read_bytes() == archive.read(meta), f"Wrong installed metadata: {package}"
            )
        files["backend-metadata/" + package + "/METADATA"] = installed
    require(
        all("backend-metadata/" + key + "/METADATA" in files for key in BACKENDS),
        "Missing installed simulation metadata",
    )


def inventory(bundle, evidence, qualification):
    manifest = read(bundle / "release-manifest.json")
    (profile,) = manifest["profiles"]
    require(profile in (CORE, CLI, SIM), "Unknown CI edition")
    files, stores = {}, {}

    def add(root, name, prefix=""):
        path = checked_path(root, name)
        member = relative_name(prefix + name)
        require(path.is_file() and member not in files, f"Missing or duplicate {member}")
        files[member] = path

    def tree(root, name, prefix="", *, audit=True):
        base = checked_path(root, name)
        require(base.is_dir(), f"Missing evidence directory {name}")
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(root).as_posix()
            checked_path(root, relative)
            if path.is_dir():
                continue
            if path.name == "harness.sqlite3" and audit:
                stores[prefix + relative] = str(path.parent)
            if (
                path.suffix.lower() in SUFFIXES
                or path.name == "harness.sqlite3"
                or (path.name == "harness.sqlite3-wal" and path.stat().st_size)
            ):
                add(root, relative, prefix)

    for name in (
        "release-manifest.json",
        manifest["source"]["archive"],
        manifest["profiles"][profile]["lock"],
        manifest["licenses"]["inventory"],
    ):
        add(bundle, name, "bundle/")
    store_name = "project/.autotem/runs"
    store = checked_path(evidence, store_name)
    db_name = store_name + "/harness.sqlite3"
    add(evidence, db_name, "capture/")
    stores["capture/" + db_name] = str(store)
    wal = store / "harness.sqlite3-wal"
    if wal.exists():
        checked_path(evidence, db_name + "-wal")
    if wal.is_file() and wal.stat().st_size:
        add(evidence, db_name + "-wal", "capture/")
    # The project can also contain credentials, adapters, unrelated raw files and a private
    # interpreter. Resolve only paths actually registered in this specific synthetic run store.
    with database_view(
        (store / "harness.sqlite3").read_bytes(), wal.read_bytes() if wal.exists() else b""
    ) as db:
        expected = project_runs(profile, lambda name: read(checked_path(evidence, name)))
        require(
            {r[0] for r in db.execute("SELECT run_id FROM runs")} == expected,
            "Project contains runs outside this CI fixture",
        )
        paths = db.execute(
            "SELECT run_id,input_path,'raw' FROM runs UNION ALL "
            "SELECT run_id,path,'artifacts' FROM artifacts"
        ).fetchall()
    for rid, path, part in paths:
        local = stored_path(str(store), rid, path, part)
        require(PurePosixPath(local).suffix.lower() in SUFFIXES, "Non-scientific project file")
        add(evidence, store_name + "/" + local, "capture/")
    if profile == CLI:
        for name in ("cli-profile.json", "installation.json", "science.json", "tool-trace.json"):
            add(evidence, name, "capture/")
        tree(evidence, "fixtures", "capture/")
    elif profile == CORE:
        require(qualification is not None, "Core qualification evidence is required")
        for name in ("installation.json", "maintenance.jsonl"):
            add(evidence, name, "capture/")
        for name in ("science", "variants"):
            tree(evidence, name, "capture/")
        for name in (
            "contract-freeze.json",
            "contract-probes.json",
            "local/local-execution.json",
            "local/supporting-files.json",
        ):
            add(qualification, name, "qualification/")
        if (qualification / "private-install.json").is_file():
            add(qualification, "private-install.json", "qualification/")
        tree(qualification, "contract-work", "qualification/", audit=False)
        local = qualification / "local"
        for name, expected in read(local / "supporting-files.json").items():
            parts = PurePosixPath(relative_name(name))
            permitted = parts.suffix in SUFFIXES or parts.name == "harness.sqlite3"
            if "extension-venv" in parts.parts:
                permitted = parts.name in PLUGIN_FIXTURES
            require(
                permitted,
                f"Unexpected local proof type: {name}",
            )
            add(local, name, "qualification/local/")
            require(
                file_sha(files["qualification/local/" + name]) == expected,
                f"Local support differs: {name}",
            )
    else:
        add(evidence, "private-install.json", "capture/")
        tree(evidence, "simulation", "capture/")
        tree(evidence, "simulation-faults", "capture/")
        simulation_metadata(bundle, evidence, manifest, files)
    if profile in (CORE, SIM):
        root = qualification if profile == CORE else evidence
        tree(root, "headless", "qualification/")
    return profile, files, stores


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.local = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("href", "src") and value:
                link = urlsplit(value)
                if not link.netloc and link.scheme not in ("http", "https", "mailto", "data"):
                    require(not link.scheme, f"Nonportable HTML link: {value}")
                    self.local.append(unquote(link.path))


@contextmanager
def database_view(data, wal):
    with tempfile.TemporaryDirectory() as scratch:
        dbfile = Path(scratch) / "harness.sqlite3"
        dbfile.write_bytes(data)
        if wal:
            Path(str(dbfile) + "-wal").write_bytes(wal)
        with closing(sqlite3.connect(dbfile)) as db:
            yield db


def stored_path(old, rid, path, part):
    require(re.fullmatch("[0-9a-f]{32}", rid), f"Invalid run ID {rid}")
    rel = PureWindowsPath(path).relative_to(PureWindowsPath(old))
    require(rel.parts[:3] == ("runs", rid, part) and len(rel.parts) == 4, f"Wrong run path: {rel}")
    return relative_name(rel.as_posix())


def registered(archive, files, stores):
    runs, local_links = {}, 0
    for member, old in stores.items():
        wal = archive.read(member + "-wal") if member + "-wal" in files else b""
        with database_view(archive.read(member), wal) as db:
            require(db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], member)
            require(
                not db.execute("SELECT 1 FROM jobs WHERE ended IS NULL LIMIT 1").fetchone(),
                f"Unsettled job in {member}",
            )
            snapshots = db.execute(
                "SELECT run_id,input_path,source_hash,job_id,status,error FROM runs"
            )
            for rid, path, digest, jid, status, error in snapshots:
                require(
                    re.fullmatch("[0-9a-f]{32}", rid) and rid not in runs,
                    f"Invalid or duplicate run {rid}",
                )
                runs[rid] = {
                    "snapshot": digest,
                    "job": jid,
                    "status": status,
                    "database": member,
                    "paths": {},
                    "code": (json.loads(error) or {}).get("code") if error else None,
                    "artifacts": {},
                }
                check_registered(files, member, old, rid, path, "raw", digest)
            rows = db.execute("SELECT run_id,artifact_id,path,sha256,size FROM artifacts")
            for rid, aid, path, digest, size in rows:
                name = check_registered(files, member, old, rid, path, "artifacts", digest)
                require(archive.getinfo(name).file_size == size, f"Wrong artifact size: {aid}")
                runs[rid]["artifacts"][aid] = digest
                runs[rid]["paths"][aid] = name
                if name.lower().endswith(".html"):
                    parser = Links()
                    parser.feed(archive.read(name).decode("utf-8"))
                    parser.close()
                    run_folder = dirname(dirname(name)) + "/"
                    for link in parser.local:
                        require(
                            not PureWindowsPath(link).drive and "\\" not in link,
                            f"Nonportable HTML link: {link}",
                        )
                        target = normpath(join(dirname(name), link)) if link else name
                        require(
                            target in files and target.startswith(run_folder),
                            f"Broken HTML link in {name}: {link}",
                        )
                        local_links += 1
    return runs, local_links


def check_registered(files, member, old, rid, path, part, digest):
    # The Windows store records its original absolute paths. Do not open those paths on replay.
    name = relative_name(join(dirname(member), stored_path(old, rid, path, part)))
    require(files.get(name) == digest, f"Missing or changed registered bytes: {name}")
    return name


def fault_evidence(faults, files, runs, release, positive):
    prefix = "capture/simulation-faults/"
    require(
        faults["release_manifest_sha256"] == release
        and faults["case_verdict"] == "passed"
        and faults["positive_run_id"] == positive["run_id"]
        and faults["positive_sha256"] == files["capture/simulation/simulation.json"]
        and faults["installation_sha256"] == files["capture/private-install.json"]
        and faults["backend_origin_sha256"] == files[prefix + "backend-origin.json"],
        "Simulation fault evidence uses a different positive or backend",
    )
    expected = {
        "missing_dependency": ("rejected", "DEPENDENCY_UNAVAILABLE"),
        "broken_dependency": ("failed", "DEPENDENCY_BROKEN"),
        "backend_timeout": ("failed", "SIMULATION_TIMEOUT"),
        "worker_timeout": ("timed_out", "TIMEOUT"),
        "missing_structure": ("needs_input", "NEEDS_INPUT"),
    }
    require(set(faults["cases"]) == set(expected), "Incomplete simulation fault matrix")
    for name, case in faults["cases"].items():
        start = prefix + name + "/"
        database = start + "harness/harness.sqlite3"
        require(
            case["verdict"] == "passed"
            and (case["status"], case["code"]) == expected[name]
            and case["witness_sha256"] == files[start + "witness.json"]
            and case["plan_sha256"] == files[start + "plan.json"]
            and faults["fixture_sha256"] == files[start + "fixture.npy"]
            and database in files,
            f"Changed simulation fault inputs: {name}",
        )
        if name == "missing_dependency":
            require(
                case["run_id"] is None
                and not any(r["database"] == database for r in runs.values()),
                "Rejected submission created a run",
            )
        else:
            run = runs[case["run_id"]]
            require(
                run["database"] == database
                and run["job"] == case["job_id"]
                and (run["status"], run["code"]) == expected[name]
                and run["snapshot"] == faults["fixture_sha256"]
                and run["artifacts"][case["report_artifact_id"]] == case["report_sha256"],
                f"Fault differs from persisted work: {name}",
            )
    child = faults["cases"]["missing_structure"]["child"]
    require(
        runs[child["run_id"]]["status"] == "completed"
        and runs[child["run_id"]]["job"] == child["job_id"]
        and child["numerical_evidence"] == positive["numerical_evidence"],
        "The structure response and child result differ",
    )


def simulated_result(archive, receipt, run):
    def artifact(operation, role):
        (aid,) = receipt["claim_artifacts"][operation][role]
        return archive.read(run["paths"][aid])

    values = receipt["numerical_evidence"]
    payload = artifact("simulation.diffraction", "simulated_pattern_0")
    require(payload[:8] == b"\x93NUMPY\x01\x00", "Wrong simulated NumPy fixture format")
    offset = 10 + int.from_bytes(payload[8:10], "little")
    header = ast.literal_eval(payload[10:offset].decode("ascii"))
    dtype = {"<f4": "<f", "<f8": "<d"}[header["descr"]]
    pixels = [v for (v,) in struct.iter_unpack(dtype, payload[offset:])]
    require(
        list(header["shape"]) == values["shape"] == [64, 64]
        and len(pixels) == 64 * 64
        and all(math.isfinite(v) for v in pixels)
        and max(pixels) > 0,
        "Invalid simulated image",
    )
    (pattern,) = json.loads(artifact("simulation.diffraction", "simulation"))["patterns"]
    require(len(pattern["spots"]) == values["simulated_spots"] > 0, "Wrong simulated spot count")
    for spot in pattern["spots"]:
        length = math.hypot(spot["gx"]["value"], spot["gy"]["value"])
        require(
            all(spot[k]["unit"] == "1/nm" for k in ("gx", "gy", "magnitude"))
            and length > 0
            and math.isclose(length, spot["magnitude"]["value"], rel_tol=1e-12)
            and spot["d_spacing"]["unit"] == "nm"
            and math.isclose(1 / length, spot["d_spacing"]["value"], rel_tol=1e-12),
            "Reciprocal scale differs",
        )
    best = json.loads(artifact("diffraction.match", "diffraction"))["phase_candidates"][0]
    require(
        best["name"] == values["candidate"] == "synthetic-cubic"
        and best["matched_spot_count"] == values["matched_spots"] == 4
        and all(best[k] == values[k] for k in ("mean_d_error", "mean_angle_error")),
        "Matched candidate differs",
    )


def audit(archive, metadata):
    files, stores = metadata["files"], metadata["stores"]
    names = archive.namelist()
    require(
        len(names) == len(set(names)) and set(names) == set(files) | {INDEX},
        "Archive inventory differs",
    )
    for name, expected in files.items():
        relative_name(name)
        with archive.open(name) as stream:
            require(stream_sha(stream) == expected, f"Changed {name}")

    def document(name):
        return json.loads(archive.read(name))

    manifest = document("bundle/release-manifest.json")
    (profile,) = manifest["profiles"]
    release = files["bundle/release-manifest.json"]
    spec = manifest["profiles"][profile]
    lock, source = files["bundle/" + spec["lock"]], manifest["source"]["archive_sha256"]
    [wheel] = [
        e["sha256"]
        for e in manifest["artifacts"]
        if e["path"].startswith(spec["wheelhouse"] + "/autotem_insight-")
        and e["path"].endswith(".whl")
    ]
    require(
        metadata["profile"] == profile
        and lock == spec["lock_sha256"]
        and source == files["bundle/" + manifest["source"]["archive"]]
        and manifest["licenses"]["sha256"] == files["bundle/" + manifest["licenses"]["inventory"]],
        "Wrong bundle or profile",
    )
    expected_stores = {
        n
        for n in files
        if n.endswith("/harness.sqlite3")
        and not n.startswith(("qualification/local/", "qualification/contract-work/"))
    }
    require(expected_stores == set(stores) and stores, "Missing scientific registry")
    runs, links = registered(archive, files, stores)
    managed_ids = {
        rid
        for rid, run in runs.items()
        if run["database"] == "capture/project/.autotem/runs/harness.sqlite3"
    }
    require(
        managed_ids == project_runs(profile, lambda name: document("capture/" + name)),
        "Unexpected project runs in portable capture",
    )
    if profile == CLI:
        receipt, science = document("capture/cli-profile.json"), document("capture/science.json")
        case = science["case"]
        install = document("capture/installation.json")
        require(
            receipt["case_verdict"] == "passed"
            and receipt["release_manifest_sha256"] == release
            and install["manifest_sha256"] == release
            and install["profile"] == CLI
            and install["transport"] == "jsonl"
            and receipt["science_sha256"] == files["capture/science.json"]
            and receipt["science"]["run_id"] == case["run_id"]
            and install["local_mcp"] == science["local_mcp"] == "not_in_profile"
            and document("capture/tool-trace.json"),
            "CLI receipt or JSONL trace differs",
        )
        measured = [case]
        spacing, candidate = case["claims"]
        diffraction = document(runs[case["run_id"]]["paths"][spacing["artifact_id"]])
        measured_spacings = [d["value"] for d in diffraction["d_spacings"] if d["unit"] == "nm"]
        require(
            spacing["unit"] == "nm"
            and len(spacing["value"]) == 4
            and all(math.isclose(v, 0.16, rel_tol=1e-10) for v in spacing["value"])
            and sum(math.isclose(v, 0.16, rel_tol=1e-10) for v in measured_spacings) >= 4
            and diffraction["phase_candidates"][0]["name"]
            == candidate["value"]
            == "synthetic-cubic",
            "Wrong CLI J01 result",
        )
    elif profile == CORE:
        receipt = document("capture/installation.json")
        science, variants = (
            document("capture/science/evidence.json"),
            document("capture/variants/operations.json"),
        )
        require(
            receipt["case_verdict"] == "passed"
            and receipt["release_manifest_sha256"] == release
            and receipt["science_evidence_sha256"] == files["capture/science/evidence.json"]
            and receipt["variant_evidence_sha256"] == files["capture/variants/operations.json"]
            and science["operations"]["sha256"] == files["capture/science/operations.json"]
            and variants["runtime"]["release_identity"] == science["runtime"]["release_identity"],
            "Cold disk evidence differs",
        )
        measured = science["cases"]
        inspection = receipt["managed_project_inspection"]
        managed = runs[inspection["run_id"]]
        require(
            managed["database"] == "capture/project/.autotem/runs/harness.sqlite3"
            and managed["job"] == inspection["job_id"]
            and managed["artifacts"] == inspection["artifacts"]
            and managed["snapshot"] == inspection["snapshot_sha256"],
            "Native installation run differs from its project database",
        )
        for entry in variants["cases"].values():
            name = "capture/variants/" + relative_name(entry["case_evidence"])
            require(
                entry["case_evidence_sha256"] == files[name]
                and entry["run_id"] == document(name)["run_id"]
                and entry["run_id"] in runs,
                "Wrong operation variant registry",
            )
        for name in (
            "qualification/contract-freeze.json",
            "qualification/local/local-execution.json",
        ):
            require(
                document(name)["release_manifest_sha256"] == release, f"Wrong release in {name}"
            )
        support = document("qualification/local/supporting-files.json")
        require(
            document("qualification/contract-freeze.json")["probe_sha256"]
            == files["qualification/contract-probes.json"]
            and document("qualification/local/local-execution.json")["supporting_files_sha256"]
            == files["qualification/local/supporting-files.json"]
            and all(files["qualification/local/" + n] == digest for n, digest in support.items()),
            "Local support manifest differs",
        )
    elif profile == SIM:
        receipt = document("capture/simulation/simulation.json")
        science = document("capture/simulation/variant-evidence.json")
        install = document("capture/private-install.json")
        require(set(receipt["installed_backends"]) == BACKENDS, "Wrong simulation dependency set")
        for package, version in receipt["installed_backends"].items():
            data = BytesParser().parsebytes(archive.read(f"backend-metadata/{package}/METADATA"))
            require(
                re.sub(r"[-_.]+", "-", data["Name"]).lower() == package
                and data["Version"] == version,
                f"Wrong archived backend metadata: {package}",
            )
        require(
            receipt["case_verdict"] == "passed"
            and install["manifest_sha256"] == release
            and install["profile"] == SIM
            and install["runtime"] == install["local_transport"] == "passed"
            and receipt["case_sha256"] == files["capture/simulation/variant-evidence.json"]
            and receipt["release_manifest_sha256"] == release
            and receipt["lock_sha256"] == lock
            and receipt["wheel_sha256"] == wheel
            and receipt["source_archive_sha256"] == source
            and receipt["run_id"] == science["run_id"]
            and receipt["job_id"] == runs[science["run_id"]]["job"]
            and all(
                runs[science["run_id"]]["artifacts"][a] == h
                for roles in receipt["claim_artifacts"].values()
                for by_id in roles.values()
                for a, h in by_id.items()
            ),
            "Simulation claims differ",
        )
        simulated_result(archive, receipt, runs[receipt["run_id"]])
        fault_evidence(
            document("capture/simulation-faults/faults.json"), files, runs, release, receipt
        )
        measured = []
    else:
        raise ValueError("Unknown profile")
    identity = science["runtime"]["release_identity"]
    require(
        (
            identity["profile"],
            identity["manifest_sha256"],
            identity["wheel_sha256"],
            identity["source_archive_sha256"],
            identity["lock_sha256"],
        )
        == (profile, release, wheel, source, lock),
        "Cross-profile scientific evidence",
    )
    for case in measured:
        run = runs[case["run_id"]]
        require(
            run["snapshot"] == case["snapshot_sha256"]
            and run["job"] == case["job_id"]
            and all(run["artifacts"][a["artifact_id"]] == a["sha256"] for a in case["artifacts"]),
            "Wrong scientific case or registered bytes",
        )
    if profile in (CORE, SIM):
        headless = document("qualification/headless/headless.json")
        require(
            headless["release_manifest_sha256"] == release
            and headless["case_verdict"] == "passed"
            and all(
                files["qualification/headless/" + k] == v
                for k, v in headless["evidence_sha256"].items()
            )
            and all(
                files["qualification/headless/processes/" + k] == v
                for k, v in headless["process_files_sha256"].items()
            ),
            "Wrong import audit or missing original process observations",
        )
    return {
        "profile": profile,
        "release_manifest_sha256": release,
        "source_archive_sha256": source,
        "wheel_sha256": wheel,
        "lock_sha256": lock,
        "registries": len(stores),
        "runs": len(runs),
        "registered_artifacts": sum(len(row["artifacts"]) for row in runs.values()),
        "relative_html_links": links,
    }


def verify(path, expected=None):
    if expected:
        require(file_sha(path) == expected, "Portable ZIP digest mismatch")
    with ZipFile(path) as archive:
        metadata = json.loads(archive.read(INDEX))
        require(metadata["schema_version"] == "1.0.0", "Unknown portable evidence schema")
        return audit(archive, metadata)


def collect(bundle, evidence, qualification, output):
    profile, files, stores = inventory(bundle, evidence, qualification)
    # Create a new file only after the roundtrip check has succeeded.
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        stage = Path(scratch) / output.name
        with ZipFile(stage, "x", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            manifest = {}
            for member, source in sorted(files.items()):
                data = source.read_bytes()
                manifest[member] = sha(data)
                archive.writestr(member, data)
            archive.writestr(
                INDEX,
                json.dumps(
                    dict(
                        schema_version="1.0.0",
                        profile=profile,
                        verifier_sha256=file_sha(Path(__file__)),
                        files=manifest,
                        stores=stores,
                    )
                ),
            )
        report = verify(stage)
        os.link(stage, output)
    return report


def public_gate_verdicts(report):
    order = ("failed", "blocked", "not_evaluated", "passed")
    gates = report["gates"]
    require(set(gates) == {f"RG{i}" for i in range(6)}, "Incomplete release gates")
    verdicts = {}
    for name, gate in gates.items():
        states = set(gate["checks"].values())
        require(states and states.issubset(order), f"Invalid {name} checks")
        verdict = next(state for state in order if state in states)
        require(verdict == gate["verdict"], f"{name} differs from its checks")
        verdicts[name] = verdict
    for name in ("RG0", "RG1", "RG2", "RG3", "RG4"):
        require(gates["RG5"]["checks"].get(name) == verdicts[name], f"RG5 differs from {name}")
    require(
        report["release_ready"] is all(state == "passed" for state in verdicts.values()),
        "Release readiness differs from the gates",
    )
    return verdicts


def delivery_receipts(report, paths, core_manifest, simulation_identity):
    upgrade, recovery, source = paths
    release = report["release_manifest_sha256"]
    checks = report["gates"]["RG4"]["checks"]
    qualifier = runpy.run_path(str(Path(__file__).with_name("qualify_release.py")))
    delivery = {}
    for name, key, path in (
        ("cross_release_upgrade_recovery", "upgrade", upgrade),
        ("interrupted_install_recovery", "recovery", recovery),
    ):
        document = read(path)
        certificate = report["certificates"][key]
        require(
            file_sha(path) == certificate["report_sha256"]
            and document["release_manifest_sha256"] == release
            and re.fullmatch(r"[0-9a-f]{64}", document["verifier_sha256"])
            and document["verifier_sha256"] == certificate["verifier_sha256"]
            and document["case_verdict"] == certificate["verdict"] == checks[name],
            f"Combined index uses a different {key} receipt",
        )
        delivery[name] = {
            "verdict": certificate["verdict"],
            "report_sha256": certificate["report_sha256"],
        }
        if key == "upgrade":
            reconnect = qualifier["HOST_RECONNECT_CHECK"]
            verdict = qualifier["upgrade_reconnect_verdict"](document, certificate)
            require(
                checks.get(reconnect, "not_evaluated") == verdict,
                "RG4 host reconnect recovery differs from its upgrade receipt",
            )
            delivery[reconnect] = {
                "verdict": verdict, "report_sha256": certificate["report_sha256"]
            }
    tested = qualifier["source_tests"](
        source,
        release,
        core_manifest["source"]["archive_sha256"],
        core_manifest["source"]["file_count"],
        simulation_identity=simulation_identity,
    )
    require(
        tested == report["source_tests"]
        and tested["verdict"] == report["gates"]["RG5"]["checks"]["source_regression"],
        "Combined index uses a different source regression",
    )
    collection, environment = tested["collection"], tested["environment"]
    require(
        collection["verdict"] == "passed"
        and environment["verdict"] == "passed"
        and environment["simulation_manifest_sha256"] == simulation_identity
        and environment["test_lock_origin"] == "archived_source",
        "Delivery needs a full collection and the matching isolated source test environment",
    )
    counts_and_hashes = (
        "verdict",
        "tests",
        "passed",
        "skipped",
        "failures",
        "errors",
        "report_sha256",
        "junit_sha256",
    )
    selection_keys = ("verdict", "collected", "deselected", "sha256")
    environment_keys = ("verdict", "sha256", "simulation_manifest_sha256", "test_lock_sha256")
    delivery["source_regression"] = {
        **{key: tested[key] for key in counts_and_hashes},
        "collection": {key: collection[key] for key in selection_keys},
        "environment": {key: environment[key] for key in environment_keys},
    }
    return delivery


def publish(archives, index, output, delivery_evidence=None):
    report = read(index)
    records, digests = {}, {}
    core_manifest = None
    for path in archives:
        replay = verify(path)
        profile = replay["profile"]
        require(profile not in records, "Duplicate edition")
        records[profile] = {**replay, "archive_sha256": file_sha(path)}
        with ZipFile(path) as archive:
            digests[profile] = json.loads(archive.read(INDEX))["files"]
            if profile == CORE and delivery_evidence:
                core_manifest = json.loads(archive.read("bundle/release-manifest.json"))
    require(
        set(records) == {CORE, CLI, SIM}
        and report["profile"] == CORE
        and report["release_manifest_sha256"] == records[CORE]["release_manifest_sha256"]
        and all(
            len({r[k] for r in records.values()}) == 1 and report[k] == records[CORE][k]
            for k in ("source_archive_sha256", "wheel_sha256")
        ),
        "Incompatible edition archives",
    )
    for key, profile, path in (
        ("contract", CORE, "qualification/contract-freeze.json"),
        ("local", CORE, "qualification/local/local-execution.json"),
        ("headless", CORE, "qualification/headless/headless.json"),
        ("simulation_edition", SIM, "capture/simulation/simulation.json"),
        ("simulation_headless", SIM, "qualification/headless/headless.json"),
        ("simulation_faults", SIM, "capture/simulation-faults/faults.json"),
    ):
        require(
            report["certificates"][key]["report_sha256"] == digests[profile][path],
            f"Combined index uses a different {key} certificate",
        )
    require(
        report["certificates"]["simulation_edition"]["release_manifest_sha256"]
        == records[SIM]["release_manifest_sha256"],
        "Wrong companion in release index",
    )
    result = {
        "schema_version": "1.0.0",
        "execution_layer": "synthetic_windows_ci",
        "release_index_sha256": file_sha(index),
        "portable_verifier_sha256": file_sha(Path(__file__)),
        "editions": records,
        "gates": public_gate_verdicts(report),
        "release_ready": report["release_ready"],
        "scientific_accuracy": report["scientific_accuracy"],
    }
    if delivery_evidence:
        result["delivery_checks"] = delivery_receipts(
            report, delivery_evidence, core_manifest, records[SIM]["release_manifest_sha256"]
        )
    with output.open("x", encoding="utf-8") as stream:
        print(json.dumps(result, indent=2), file=stream)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    for key in ("bundle", "evidence", "qualification", "out"):
        pack.add_argument("--" + key, required=key != "qualification", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("archive", type=Path)
    check.add_argument("--sha256")
    public = commands.add_parser("public")
    public.add_argument("--archives", nargs=3, type=Path, required=True)
    public.add_argument(
        "--delivery-evidence",
        nargs=3,
        type=Path,
        metavar=("UPGRADE", "RECOVERY", "SOURCE_TESTS"),
        help="Bind maintenance receipts and replay the source JUnit, selection and environment.",
    )
    for key in ("index", "out"):
        public.add_argument("--" + key, required=True, type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify(args.archive, args.sha256)
    elif args.command == "public":
        result = publish(args.archives, args.index, args.out, args.delivery_evidence)
    else:
        result = collect(
            args.bundle.resolve(),
            args.evidence.resolve(),
            args.qualification.resolve() if args.qualification else None,
            args.out.resolve(),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
