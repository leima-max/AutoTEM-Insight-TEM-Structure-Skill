"""Collect or independently replay the contracted interface of one installed AutoTEM release."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
import sqlite3
import subprocess
import sys
from contextlib import closing
from email.parser import BytesParser
from pathlib import Path
from zipfile import BadZipFile, ZipFile

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "harness" / "acceptance"
POLICY = CONFIG / "traditional-core-contract-v1.json"
SCHEMA = CONFIG / "harness-evidence-v1.schema.json"
HEX = r"[a-f0-9]{64}"
ID = r"[a-f0-9]{32}"
CATALOGUE_FIELDS = (
    "operation_id",
    "version",
    "profile_id",
    "implementation_status",
    "implementation_source",
    "input_roles",
    "output_roles",
    "parameter_schema",
    "modality",
    "coordinate_contract",
    "permission",
    "budget_seconds",
    "dependencies",
    "scientific_limitations",
    "result_level",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate key in evidence")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("Non-finite JSON evidence")

    return json.loads(
        path.read_text("utf-8-sig"), object_pairs_hook=unique, parse_constant=invalid_constant
    )


def local(root, relative):
    """Resolve only files in the evidence directory, including link targets."""
    name = Path(relative)
    require(
        not name.is_absolute() and ".." not in name.parts and "\\" not in relative,
        "Invalid evidence-relative path",
    )
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root.resolve()) and path.is_file(), "Evidence path outside root")
    return path


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def role_catalogue(row):
    return {k: row[k] for k in CATALOGUE_FIELDS}


def validators():
    from jsonschema import Draft202012Validator

    document = read(SCHEMA)
    Draft202012Validator.check_schema(document)
    return {
        name: Draft202012Validator({**document, "$ref": f"#/$defs/{name}"})
        for name in document["$defs"]
    }


def conform(validator, value, name):
    issue = next(validator.iter_errors(value), None)
    if issue is not None:
        location = "/".join(map(str, issue.absolute_path))
        raise ValueError(f"{name} schema mismatch at {location or '(root)'} ({issue.validator})")


def release_lock(bundle, manifest, profile, expected_identity):
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.tags import compatible_tags, cpython_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
    from packaging.version import Version

    lock = local(bundle, profile["lock"])
    require(
        sha(lock) == profile["lock_sha256"] == expected_identity["lock_sha256"],
        "Release lock mismatch",
    )
    pinned = {}
    for line in lock.read_text("ascii").splitlines():
        match = re.fullmatch(
            r"([a-z0-9][a-z0-9-]*)(\[mcp\])?==([^\s]+) --hash=sha256:([0-9a-f]{64})", line
        )
        require(match is not None, "Lock requires exact version and single wheel hash")
        name, extra, ver, digest = match.groups()
        require(
            name not in pinned and bool(extra) == (name == "autotem-insight"),
            "Duplicate or unexpected lock extra",
        )
        pinned[name] = (ver, digest)
    wheels = {}
    wheel_hashes = {}
    inventory = None
    target_version = tuple(map(int, manifest["target"]["python_version"].split(".")[:2]))
    python_version = Version(manifest["target"]["python_version"])
    python_constraints = 0
    abi = "cp" + "".join(map(str, target_version))
    permitted = set(cpython_tags(target_version, [abi], ["win_amd64"])) | set(
        compatible_tags(target_version, abi, ["win_amd64"])
    )
    for row in manifest["artifacts"]:
        if Path(row["path"]).parent.as_posix() != profile["wheelhouse"]:
            continue
        p = local(bundle, row["path"])
        if p.name == "wheelhouse-manifest.json":
            require(inventory is None and sha(p) == row["sha256"], "Invalid wheelhouse manifest")
            inventory = read(p)["wheels"]
            continue
        name, ver, _, tags = parse_wheel_filename(p.name)
        require(
            name not in wheels
            and name in pinned
            and pinned[name] == (str(ver), row["sha256"])
            and tags & permitted
            and p.stat().st_size == row["size"]
            and sha(p) == row["sha256"],
            "Wheel/lock mismatch",
        )
        with ZipFile(p) as z:
            entries = [
                m for m in z.namelist() if m.endswith(".dist-info/METADATA") and m.count("/") == 1
            ]
            require(len(entries) == 1, "Missing wheel metadata")
            meta = BytesParser().parsebytes(z.read(entries[0]))
        require(
            canonicalize_name(meta["Name"]) == name and Version(meta["Version"]) == ver,
            "Wheel identity mismatch",
        )
        if meta.get("Requires-Python"):
            require(
                python_version in SpecifierSet(meta["Requires-Python"]),
                f"Incompatible Requires-Python in {p.name}",
            )
            python_constraints += 1
        wheels[name] = [Requirement(v) for v in meta.get_all("Requires-Dist", [])]
        wheel_hashes[p.name] = row["sha256"]
    require(
        wheels.keys() == pinned.keys()
        and inventory == wheel_hashes
        and pinned["autotem-insight"][1] == expected_identity["wheel_sha256"],
        "Incomplete locked wheel inventory",
    )
    require(
        {p.name for p in lock.parent.parent.joinpath(profile["wheelhouse"]).iterdir()}
        == {
            Path(r["path"]).name
            for r in manifest["artifacts"]
            if Path(r["path"]).parent.as_posix() == profile["wheelhouse"]
        },
        "Unexpected wheelhouse entry",
    )
    # Resolve extras through the entire graph against the frozen Windows target on any verifier OS.
    full = manifest["target"]["python_version"]
    env = {
        "implementation_name": "cpython",
        "implementation_version": full,
        "os_name": "nt",
        "platform_machine": "AMD64",
        "platform_python_implementation": "CPython",
        "platform_system": "Windows",
        "python_full_version": full,
        "python_version": ".".join(full.split(".")[:2]),
        "sys_platform": "win32",
    }
    pending = [("autotem-insight", "mcp")]
    seen = set()
    while pending:
        name, extra = pending.pop()
        if (name, extra) in seen:
            continue
        seen.add((name, extra))
        for req in wheels[name]:
            if req.marker and not req.marker.evaluate({**env, "extra": extra}):
                continue
            target = canonicalize_name(req.name)
            require(
                not req.url and target in pinned and Version(pinned[target][0]) in req.specifier,
                f"Unsatisfied target dependency {target} for {name}",
            )
            pending.extend((target, x) for x in (req.extras or {""}))
    require({name for name, _ in seen} == set(pinned), "Lock contains an unreferenced package")
    return {
        "verdict": "passed",
        "wheel_count": len(wheels),
        "wheels_with_requires_python": python_constraints,
        "lock_sha256": sha(lock),
        "root_requirement": f"autotem-insight[mcp]=={pinned['autotem-insight'][0]}",
    }


def check_catalogue(policy, probe, stored, v):
    mcp = probe["mcp"]
    require(
        mcp["server"] == "autotem-harness" and mcp["tools"] == policy["tools"],
        "MCP nine-tool inventory or argument schema differs",
    )
    conform(v["protocolEnvelope"], mcp["capabilities"], "capability envelope")
    cap = mcp["capabilities"]
    require(
        cap["ok"]
        and cap["tool"] == "list_capabilities"
        and cap["result"]["envelope_version"] == policy["protocol_version"],
        "Capability call mismatch",
    )
    rows = cap["result"]["operations"]
    indexed = {r["operation_id"]: r for r in rows}
    require(
        len(rows) == len(indexed)
        and indexed.keys() == policy["operations"].keys() == stored.keys(),
        "Operation inventory differs",
    )
    result = {}
    from jsonschema import Draft202012Validator

    for name, row in indexed.items():
        fixed = policy["operations"][name]
        shape = role_catalogue(row)
        require(
            shape == role_catalogue(stored[name])
            and canonical(shape) == fixed["contract_sha256"]
            and row["version"] == fixed["version"]
            and row["profile_id"] == fixed["profile"]
            and row["implementation_status"] == fixed["status"]
            and row["input_roles"] == fixed["input_roles"]
            and row["output_roles"] == fixed["output_roles"],
            f"Changed operation contract: {name}",
        )
        require(
            set(row["input_roles"]) == row["input_schema"].keys()
            and set(row["output_roles"]) == row["output_schema"].keys()
            and canonical([row["input_schema"], row["output_schema"]])
            == fixed["role_schemas_sha256"],
            f"Role schema mismatch: {name}",
        )
        for s in (
            row["parameter_schema"],
            *row["input_schema"].values(),
            *row["output_schema"].values(),
        ):
            Draft202012Validator.check_schema(s)
        implemented = fixed["status"] == "implemented"
        require(
            implemented == bool(row["implementation_source"])
            and row["scientific_accuracy"] == "not_evaluated",
            f"Handler or science boundary mismatch: {name}",
        )
        error = stored[name]["negative_contract"]
        require(
            error["observed_code"]
            == ("INVALID_PARAMETERS" if implemented else "DEFERRED_THIS_PHASE")
            and error["verdict"] == "passed",
            f"Negative operation contract mismatch: {name}",
        )
        result[name] = {
            "version": row["version"],
            "status": fixed["status"],
            "domain": row["profile_id"],
            "input_roles": row["input_roles"],
            "output_roles": row["output_roles"],
            "availability": row["availability"]["state"],
            "negative_code": error["observed_code"],
            "contract_sha256": fixed["contract_sha256"],
            "role_schemas_sha256": canonical([row["input_schema"], row["output_schema"]]),
        }
    return {
        "verdict": "passed",
        "tools": sorted(mcp["tools"]),
        "protocol": mcp["protocol"],
        "tool_schemas_sha256": canonical(mcp["tools"]),
        "operation_count": len(rows),
        "implemented": sum(r["status"] == "implemented" for r in result.values()),
        "reserved": sum(r["status"] == "reserved" for r in result.values()),
        "operations": result,
    }


def check_protocol(policy, probe, v):
    from jsonschema import Draft202012Validator

    errors = probe["protocol_errors"]
    require(errors.keys() == policy["jsonl_errors"].keys(), "Incomplete protocol probes")
    for name, out in errors.items():
        conform(v["protocolEnvelope"], out, name)
        require(
            out["ok"] is False
            and out["error"]["code"] == policy["jsonl_errors"][name]
            and out["request_id"]
            == (
                None
                if name in {"top_level_field", "duplicate_key", "nonfinite", "oversize_line"}
                else name
            ),
            f"Error routing mismatch: {name}",
        )
    require(
        probe["terminal_states"] == policy["terminal_states"]
        and probe["max_request_bytes"] == policy["max_request_bytes"],
        "Changed status or protocol size",
    )
    plan, reply = (probe["models"][k] for k in ("plan", "input_response"))
    require(
        {n: canonical(s) for n, s in probe["models"].items()} == policy["model_schema_sha256"],
        "Plan or input-response schema differs from the frozen contract",
    )
    for schema in (plan, reply):
        Draft202012Validator.check_schema(schema)
    p = plan["properties"]
    require(
        set(p)
        == {
            "envelope_version",
            "input_path",
            "modality",
            "selection",
            "steps",
            "budgets",
            "network_enabled",
            "input_mode",
            "enabled_plugins",
            "user_input",
        }
        and p["envelope_version"]["default"] == "1.0.0"
        and p["input_mode"]["default"] == "snapshot"
        and set(p["input_mode"]["enum"]) == {"snapshot", "reference"}
        and p["network_enabled"]["default"] is False
        and p["steps"]["minItems"] == 1
        and p["steps"]["maxItems"] == 100
        and p["enabled_plugins"]["maxItems"] == 16,
        "Plan scope or bounds changed",
    )
    require(
        set(plan["$defs"]["Step"]["properties"])
        == {
            "id",
            "operation_id",
            "operation_version",
            "parameters",
            "depends_on",
            "inputs",
            "optional",
        },
        "Step contract mismatch",
    )
    budgets = {k: i["default"] for k, i in plan["$defs"]["Budgets"]["properties"].items()}
    require(
        budgets
        == {
            "run_seconds": 900,
            "operation_seconds": 120,
            "simulation_seconds": 30,
            "max_snapshot_bytes": 2147483648,
        },
        "Changed default budgets",
    )
    require(
        set(reply["required"]) == {"request_id", "source", "basis", "answers"}
        and set(reply["properties"]["source"]["enum"]) == {"user", "metadata", "artifact"}
        and probe["answer_units"] == policy["answer_units"],
        "Changed sourced answer or units",
    )
    return {
        "verdict": "passed",
        "negative_codes": {k: r["error"]["code"] for k, r in errors.items()},
        "max_request_bytes": probe["max_request_bytes"],
        "active_states": policy["active_states"],
        "terminal_states": probe["terminal_states"],
        "default_budgets": budgets,
        "answer_units": probe["answer_units"],
        "model_schema_sha256": {n: canonical(s) for n, s in probe["models"].items()},
    }


def database(path):
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def check_formats(policy, bundle, manifest, science, fixtures, work, probe, v):
    import numpy as np
    from PIL import Image

    with ZipFile(local(bundle, manifest["source"]["archive"])) as src:
        text = src.read("autotem/services/data_quality.py").decode("utf-8-sig")
    reader = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "_read_selected"
    )
    extensions = {
        n.value
        for n in ast.walk(reader)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and re.fullmatch(r"\.[a-z0-9]+", n.value)
    }
    formats = policy["formats"]
    require(
        extensions == set().union(*formats.values()),
        "Reader branch inventory differs from support policy",
    )
    core = set(formats["core_disk"])
    official = next(iter(manifest["profiles"].values()))["requested_scope"][
        "formats_for_first_acceptance"
    ]
    require(set(official) == core, "Declared release reader scope differs")
    tested = {}
    for fixture in fixtures:
        path = local(science, fixture["path"])
        require(
            sha(path) == fixture["sha256"]
            and fixture["license"]
            and fixture["source"]
            and fixture["axes"] == ["y", "x"],
            "Disk fixture source or axes changed",
        )
        tested.setdefault(path.suffix.lower(), []).append(fixture["case_id"])
    require(set(tested) == core, "Missing fixture for a released file extension")
    rows = probe["selections"]
    expected = {
        "npy_no_selection": ("needs_input", "selection_indices"),
        "npy_frame_roi": ("completed", None),
        "tif_no_frame": ("needs_input", "frame"),
        "tif_no_channel": ("needs_input", "channel"),
        "tif_frame_channel_roi": ("completed", None),
        "tif_bad_frame": ("failed", "INVALID_SELECTION"),
    }
    require(rows.keys() == expected.keys(), "Missing real reader selection probe")
    summary = {}
    with closing(database(work / "harness" / "harness.sqlite3")) as db:
        for name, (status, field) in expected.items():
            row = rows[name]
            raw = local(work, row["input"])
            saved = db.execute(
                "SELECT status,source_hash,input_path,job_id FROM runs WHERE run_id=?",
                (row["run_id"],),
            ).fetchone()
            require(
                saved
                and saved["status"] == row["status"] == status
                and saved["job_id"] == row["job_id"]
                and sha(raw) == saved["source_hash"] == row["snapshot_sha256"],
                "Selection run mismatch",
            )
            summary[name] = {
                "run_id": row["run_id"],
                "status": status,
                "source_sha256": row["snapshot_sha256"],
            }
            if status == "needs_input":
                (req,) = row["input_requests"]
                conform(v["requestRecord"], req, "selection request")
                recorded = db.execute(
                    "SELECT run_id,fields,status FROM input_requests WHERE request_id=?",
                    (req["request_id"],),
                ).fetchone()
                require(
                    req["run_id"] == row["run_id"]
                    and recorded
                    and recorded["run_id"] == row["run_id"]
                    and recorded["status"] == "open"
                    and json.loads(recorded["fields"]) == req["fields"]
                    and field in {f["name"] for f in req["fields"]}
                    and not row.get("raw"),
                    "Ambiguous input silently selected",
                )
                summary[name]["field"] = field
            elif status == "failed":
                require(row["error_code"] == field and not row.get("raw"), "Invalid frame accepted")
                summary[name]["error_code"] = field
            else:
                value = row["raw"]
                path = local(work, value["path"])
                stored = db.execute(
                    "SELECT role,sha256,committed FROM artifacts WHERE run_id=? AND artifact_id=?",
                    (row["run_id"], value["artifact_id"]),
                ).fetchone()
                require(
                    stored
                    and tuple(stored) == ("raw", sha(path), 1)
                    and sha(path) == value["sha256"],
                    "Selected image does not match the registered raw artifact",
                )
                if name.startswith("npy"):
                    image = np.load(raw, allow_pickle=False)[1]
                else:
                    with Image.open(raw) as source:
                        require(source.n_frames == 2, "Expected two TIFF frames")
                        source.seek(1)
                        image = np.asarray(source)[..., 2]
                expected_pixels = image[1:6, 2:9]
                result = np.load(path, allow_pickle=False)
                require(
                    result.shape == (5, 7)
                    and list(result.shape) == value["shape"]
                    and str(result.dtype) == value["dtype"]
                    and np.array_equal(result, expected_pixels)
                    and hashlib.sha256(result.tobytes()).hexdigest() == value["array_sha256"]
                    and value["selection"]["roi_origin_yx"] == [1, 2],
                    "Incorrect frame/channel/ROI pixels",
                )
                summary[name].update(
                    artifact_id=value["artifact_id"],
                    array_sha256=value["array_sha256"],
                    shape=value["shape"],
                    dtype=value["dtype"],
                    roi_yx=[1, 6, 2, 9],
                )
    return {
        "verdict": "passed",
        "release_qualified_formats": tested,
        "selection_probes": summary,
        "other_reader_branches": {
            "extensions": formats["reader_branches_outside_release_claim"],
            "release_qualification": "not_evaluated",
        },
        "metadata_only": formats["metadata_only"],
        "roi_coordinates": "source y0,y1,x0,x1; exclusive upper bounds",
        "reader_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def check_registered(science, db, case, run_id, entries):
    import numpy as np

    require(
        re.fullmatch(ID, run_id) and len({a["artifact_id"] for a in entries}) == len(entries),
        "Duplicate artifact or invalid run ID",
    )
    lookup = {}
    successful_steps = {
        step[0]
        for step in db.execute(
            "SELECT step_id FROM steps WHERE run_id=? AND status='completed'", (run_id,)
        )
    }
    for a in entries:
        path = local(science, a["path"])
        require(
            path.parent == (science / "harness" / "runs" / run_id / "artifacts").resolve()
            and path.stem == a["artifact_id"],
            "Cross-run or unregistered artifact path",
        )
        stored = db.execute(
            "SELECT role,step_id,kind,size,sha256,committed FROM artifacts "
            "WHERE run_id=? AND artifact_id=?",
            (run_id, a["artifact_id"]),
        ).fetchone()
        expected = (a["role"], a["step_id"], a["kind"], a["size"], a["sha256"], 1)
        require(
            stored
            and tuple(stored) == expected
            and a["execution_usable"] is (a["step_id"] in successful_steps)
            and path.stat().st_size == a["size"]
            and sha(path) == a["sha256"],
            f"Artifact registry or bytes mismatch for {case}",
        )
        if a["kind"] == "array":
            data = np.load(path, allow_pickle=False, mmap_mode="r")
            require(
                list(data.shape) == a["shape"]
                and str(data.dtype) == a["dtype"]
                and hashlib.sha256(data.tobytes(order="C")).hexdigest() == a["array_sha256"],
                f"Registered array shape, dtype or content mismatch for {case}",
            )
        lookup[a["artifact_id"]] = a
    return lookup


def cli_evidence(evidence, science, db, v, core_result_version):
    base = evidence / "science"
    conform(v["diskEvidence"], science, "disk evidence")
    fixtures = {f["case_id"]: f for f in science["fixtures"]}
    cases = {c["case_id"]: c for c in science["cases"]}
    require(
        len(cases) == len(science["cases"])
        and cases.keys() == fixtures.keys()
        and {"J01", "J02", "J03", "J04", "J05-SAED", "J05-NBED", "J06"} == cases.keys(),
        "Incorrect core disk case inventory",
    )
    n_artifacts = 0
    n_claims = 0
    for key, c in cases.items():
        require(
            c["run_id"] != c["inspection_run_id"] and c["case_verdict"] == "passed",
            "Inspection mistaken for scientific run",
        )
        r = db.execute(
            "SELECT status,source_hash,job_id,parent_run_id FROM runs WHERE run_id=?",
            (c["run_id"],),
        ).fetchone()
        require(
            r
            and tuple(r)
            == (c["execution_status"], c["snapshot_sha256"], c["job_id"], c["parent_run_id"])
            and c["snapshot_sha256"] == fixtures[key]["sha256"],
            "Disk run lineage mismatch",
        )
        saved = check_registered(base, db, key, c["run_id"], c["artifacts"])
        datasets = [a for a in saved.values() if a["role"] == "dataset" and a["execution_usable"]]
        require(
            len(datasets) == 1
            and read(local(base, datasets[0]["path"]))["schema_version"] == core_result_version,
            "Unexpected dataset result schema version",
        )
        require(
            c["report_artifact_id"] in saved
            and saved[c["report_artifact_id"]]["role"] == "report_manifest"
            and all(k["artifact_id"] in saved for k in c["claims"]),
            "Claim or report missing evidence",
        )
        n_artifacts += len(saved)
        n_claims += len(c["claims"])
    return {
        "verdict": "passed",
        "disk_cases": sorted(cases),
        "verified_artifact_files": n_artifacts,
        "claims_linked_to_artifacts": n_claims,
        "execution_layer": science["execution_layer"],
        "dataset_result_schema_version": core_result_version,
    }, cases


def response_evidence(science, db, child, v):
    from jsonschema import Draft202012Validator

    line = child["input_response"]
    parent = line["parent_run_id"]
    parent_record = db.execute("SELECT * FROM runs WHERE run_id=?", (parent,)).fetchone()
    require(
        parent == child["parent_run_id"]
        and parent_record
        and parent_record["status"] == "needs_input"
        and parent_record["source_hash"]
        == line["parent_snapshot_sha256"]
        == child["snapshot_sha256"],
        "Parent/child raw identity or terminal state mismatch",
    )
    question = db.execute(
        "SELECT * FROM input_requests WHERE request_id=? AND run_id=?", (line["request_id"], parent)
    ).fetchone()
    require(question is not None, "Missing real parent question")
    q = dict(question)
    for k in ("fields", "evidence_artifact_ids"):
        q[k] = json.loads(q[k])
    conform(v["requestRecord"], q, "calibration input request")
    artifacts = check_registered(science, db, "J06-parent", parent, line["parent_artifacts"])
    require(
        q["status"] == "answered"
        and line["request_fields"] == q["fields"]
        and set(q["evidence_artifact_ids"]) <= artifacts.keys()
        and q["operation_id"] == "calibration.resolve",
        "Question does not reference the recorded parent calibration",
    )
    outcome = {}
    for row in db.execute("SELECT * FROM input_responses WHERE request_id=?", (q["request_id"],)):
        reply = dict(row)
        for key in ("response", "validation", "adopted_values"):
            reply[key] = json.loads(reply[key]) if reply[key] is not None else None
        conform(v["responseRecord"], reply, "answer record")
        flag = reply["validation"]["status"]
        require(
            flag not in outcome and reply["parent_run_id"] == parent,
            "Duplicate or wrong parent response",
        )
        outcome[flag] = reply
    require(
        outcome.keys() == {"accepted", "rejected"}, "Missing source-checked rejection or acceptance"
    )
    good, bad = outcome["accepted"], outcome["rejected"]
    require(
        bad["validation"]["code"] == line["invalid_reply_code"] == "INVALID_INPUT_RESPONSE"
        and bad["response_id"] == line["invalid_response_id"]
        and bad["child_run_id"] is None
        and good["child_run_id"] == child["run_id"]
        and good["response_id"] == line["accepted_response_id"]
        and good["response"]["request_id"] == q["request_id"],
        "Answer audit or child link mismatch",
    )
    reply = good["response"]
    require(
        {k: reply[k] for k in line["accepted_reply"]} == line["accepted_reply"]
        and reply["basis"].strip()
        and set(reply["evidence_artifact_ids"]) <= artifacts.keys(),
        "Missing or changed answer provenance",
    )
    answer = reply["answers"]["pixel_size_nm"]
    spec = next(
        f["answer_schema"]["pixel_size_nm"] for f in q["fields"] if f["name"] == "pixel_size_nm"
    )
    Draft202012Validator(spec["value"]).validate(answer["value"])
    require(
        answer["unit"] == "pm/pixel" and answer["unit"] in spec["units"],
        "Unexpected unit in J06 fixture",
    )
    nm = answer["value"] / 1000
    require(
        good["adopted_values"]["pixel_size_nm"] == line["adopted"]
        and line["adopted"]["parameters"]["pixel_size_nm"] == nm,
        "Physical unit conversion differs",
    )
    calibration = next(a for a in child["artifacts"] if a["role"] == "calibration")
    calibrated = read(local(science, calibration["path"]))["pixel_size"]
    require(
        calibrated["value"] == nm and calibrated["unit"] == "nm/pixel",
        "Registered calibration differs from answer",
    )
    return {
        "verdict": "passed",
        "parent_run_id": parent,
        "child_run_id": child["run_id"],
        "request_id": q["request_id"],
        "rejected_code": bad["validation"]["code"],
        "response_source": reply["source"],
        "input_value": answer["value"],
        "input_unit": answer["unit"],
        "adopted_value": nm,
        "adopted_unit": "nm/pixel",
        "parent_evidence_count": len(artifacts),
        "calibration_artifact_id": calibration["artifact_id"],
    }


def check_host_matrix(bundle, manifest, path, v):
    """Rebuild the current-release matrix. Private host-session replay belongs to RG3."""
    sources = Path(__file__).resolve().parents[1] / "host_acceptance"
    with ZipFile(local(bundle, manifest["source"]["archive"])) as src:
        for name in ("verify.py", "matrix.json"):
            require(
                src.read("tools/host_acceptance/" + name) == (sources / name).read_bytes(),
                f"Host contract resource differs from the release: {name}",
            )
        archived_index_sha = hashlib.sha256(src.read("tools/host_acceptance/index.py")).hexdigest()
    spec = importlib.util.spec_from_file_location("autotem_host_index", sources / "index.py")
    index = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(index)
    require(
        set(index.MATRIX["cases"]) == {f"J{i:02}" for i in range(1, 16)}
        and set(index.MATRIX["hosts"]) == {"codex", "workbuddy", "openclaw"},
        "Missing planned host or journey",
    )
    rebuilt = index.build(bundle)
    conform(v["hostMatrix"], rebuilt, "unobserved host matrix")
    if path is not None:
        document = read(path)
        conform(v["hostMatrix"], document, "host evidence matrix")
        rows = [
            row
            for host in document["hosts"].values()
            for case in host["cases"].values()
            for row in [
                *case["variants"],
                *([case["unspecified_variant"]] if "unspecified_variant" in case else []),
            ]
        ]
        expected = {row["summary_sha256"] for row in rows if "summary_sha256" in row}
        reports = []
        if expected:
            # Public, redacted case summaries may be placed beside the chosen index under any name.
            for candidate in path.parent.glob("*.json"):
                if candidate != path and sha(candidate) in expected:
                    summary = read(candidate)
                    conform(v["hostCase"], summary, "host case summary")
                    require(
                        summary["evidence"]["verifier_sha256"] == sha(sources / "verify.py"),
                        "Host summary verifier differs from this release",
                    )
                    reports.append(candidate)
        require({sha(p) for p in reports} == expected, "Missing referenced host case summary")
        native = {
            host
            for host, entry in document["hosts"].items()
            if any(
                r["variant"] == "native" and r["required"]
                for r in entry["cases"]["J14"]["variants"]
            )
        }
        rebuilt = index.build(bundle, reports, versions=document["host_versions"], native=native)
        for field in (
            "release_id",
            "release_manifest_sha256",
            "matrix_sha256",
            "declared_profiles",
            "host_versions",
            "required_variant_counts",
            "host_journey_matrix",
        ):
            require(rebuilt[field] == document[field], f"Host matrix mismatch: {field}")
        require(set(document["hosts"]) == set(rebuilt["hosts"]), "Host inventory differs")
        for host, actual in rebuilt["hosts"].items():
            require(
                all(document["hosts"][host].get(k) == value for k, value in actual.items()),
                "Host rows differ from their release-bound summaries",
            )
    return {
        "host_matrix_source": "supplied_index" if path else "empty_schema_baseline",
        "host_matrix_sha256": sha(path) if path else None,
        "host_matrix_contract_sha256": rebuilt["matrix_sha256"],
        "host_indexer_sha256": sha(sources / "index.py"),
        "archived_host_indexer_sha256": archived_index_sha,
        "host_matrix_verdict": rebuilt["host_journey_matrix"],
        "host_required_variant_counts": rebuilt["required_variant_counts"],
        "host_private_session_replay": "not_evaluated",
    }


def assess(bundle, evidence, work, capture, host_index=None):
    """Replay the raw observations, archived run registry/files and exact target lock."""
    v = validators()
    policy, probe = read(POLICY), read(capture)
    manifest_file = local(bundle, "release-manifest.json")
    manifest = read(manifest_file)
    fingerprint = sha(manifest_file)
    science_dir = evidence / "science"
    install_file = local(evidence, "installation.json")
    science_file = local(evidence, "science/evidence.json")
    archive_file = local(evidence, "science/operations.json")
    install, science, ops = read(install_file), read(science_file), read(archive_file)
    runtime = probe["runtime"]
    identity = runtime["release_identity"]
    collector = Path(__file__).with_name("contract_probe.py")
    require(
        probe["schema_version"] == "1.0.0" and probe["collector_sha256"] == sha(collector),
        "Probe collector fingerprint differs from the recorded program",
    )
    require(
        set(manifest["profiles"]) == {policy["profile"]}
        and manifest["target"] == policy["target"]
        and manifest["pip_version"] == policy["seed_pip_version"],
        "Profile or runtime target mismatch",
    )
    scope = manifest["profiles"][policy["profile"]]["requested_scope"]
    require(
        set(scope["host_targets"]) == {"codex", "openclaw", "workbuddy"}
        and scope["operation_discovery"] == "list_capabilities"
        and sorted(scope["required_operations"]) == sorted(policy["required_operations"])
        and all(
            policy["operations"][n]["status"] == "implemented" for n in scope["required_operations"]
        )
        and manifest["profiles"][policy["profile"]]["extras"] == ["mcp"]
        and scope["optional_backends"] == [],
        "Profile declares a different host, tool or operation scope",
    )
    require(
        install["case_verdict"] == "passed"
        and install["release_manifest_sha256"] == fingerprint
        and install["science_evidence_sha256"] == sha(science_file)
        and science["operations"]["sha256"] == sha(archive_file)
        and runtime == science["runtime"]
        and runtime["isolated"]
        and runtime["installed_origin"]
        and runtime["private_install_verified"]
        and runtime["editable"] is False
        and runtime["python"] == manifest["target"]["python_version"]
        and runtime["implementation"].lower() == manifest["target"]["implementation"]
        and runtime["platform"] == "Windows"
        and runtime["architecture"].lower() == "amd64"
        and runtime["autotem_version"] == manifest["package_version"]
        and {
            "installed_profile_versions_and_origins_passed",
            "pip_check_passed",
            "local_mcp_handshake_passed",
        }
        <= set(install["runtime_checks"]),
        "Cold install, release or operation evidence differs",
    )
    require(
        identity["release_id"] == manifest["release_id"]
        and identity["manifest_sha256"] == fingerprint
        and identity["profile"] == policy["profile"]
        and identity["package_version"] == manifest["package_version"]
        and identity["source_archive_sha256"]
        == manifest["source"]["archive_sha256"]
        == sha(local(bundle, manifest["source"]["archive"])),
        "Release identity drift",
    )
    runtime_archive = local(bundle, manifest["runtime"]["archive"])
    require(
        sha(runtime_archive) == manifest["runtime"]["sha256"],
        "CPython runtime archive differs",
    )
    with ZipFile(runtime_archive) as runtime_package:
        seeds = [
            n
            for n in runtime_package.namelist()
            if n.startswith("tools/Lib/ensurepip/_bundled/pip-") and n.endswith(".whl")
        ]
        require(
            seeds
            == [f"tools/Lib/ensurepip/_bundled/pip-{manifest['pip_version']}-py3-none-any.whl"],
            "Unexpected ensurepip seed in the CPython runtime archive",
        )
    checks = {}
    checks["target_profile"] = {
        "verdict": "passed",
        "target": manifest["target"],
        "seed_pip_version": manifest["pip_version"],
        "qualified_profile": policy["profile"],
        "declared_host_targets": sorted(scope["host_targets"]),
        "required_operations": sorted(scope["required_operations"]),
        "limitations": scope["limitations"],
        "other_profiles_not_released": policy["unreleased_profiles"],
    }
    checks["dependency_lock"] = release_lock(
        bundle, manifest, manifest["profiles"][policy["profile"]], identity
    )
    checks["operations_and_tools"] = check_catalogue(policy, probe, ops["operations"], v)
    checks["plan_states_and_errors"] = check_protocol(policy, probe, v)
    traces = {r["tool"] for r in science["tool_trace"]}
    require(traces == set(policy["tools"]), "Disk journeys did not exercise all nine public tools")
    checks["readers_and_selection"] = check_formats(
        policy, bundle, manifest, science_dir, science["fixtures"], work, probe, v
    )
    with closing(database(local(evidence, "science/harness/harness.sqlite3"))) as db:
        require(
            db.execute("PRAGMA user_version").fetchone()[0] == 1,
            "Unsupported release store version",
        )
        checks["evidence_schema"], cases = cli_evidence(
            evidence, science, db, v, policy["core_result_version"]
        )
        checks["calibration_question_and_child"] = response_evidence(
            science_dir, db, cases["J06"], v
        )
    checks["evidence_schema"].update(
        json_schema_draft="2020-12",
        **check_host_matrix(bundle, manifest, host_index, v),
        schema_definitions=sorted(v),
    )
    result = {
        "schema_version": "1.0.0",
        "gate": "RG0",
        "case_verdict": "passed",
        "gate_scope": policy["profile"],
        "release_id": manifest["release_id"],
        "release_manifest_sha256": fingerprint,
        "source_archive_sha256": identity["source_archive_sha256"],
        "wheel_sha256": identity["wheel_sha256"],
        "lock_sha256": identity["lock_sha256"],
        "verifier_sha256": sha(Path(__file__)),
        "collector_sha256": sha(collector),
        "policy_sha256": sha(POLICY),
        "schema_sha256": sha(SCHEMA),
        "probe_sha256": sha(capture),
        "probe_workdir": work.name,
        "captured_utc": probe["captured_utc"],
        "checks": checks,
        "source_evidence_sha256": {
            "installation.json": sha(install_file),
            "science/evidence.json": sha(science_file),
            "science/operations.json": sha(archive_file),
        },
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
    }
    conform(v["certificate"], result, "RG0 certificate")
    return result


def recheck(bundle, evidence, certificate, work=None, host_index=None):
    """Return the verified certificate. Raise ValueError when any observed check differs."""
    from jsonschema.exceptions import SchemaError, ValidationError

    try:
        document = read(certificate)
        name = document.get("probe_workdir")
        require(
            isinstance(name, str)
            and name not in {".", ".."}
            and re.fullmatch(r"[A-Za-z0-9_.-]+", name),
            "Unsafe capture work directory",
        )
        expected = assess(
            bundle.resolve(),
            evidence.resolve(),
            (work or certificate.parent / name).resolve(),
            certificate.with_name("contract-probes.json"),
            host_index.resolve() if host_index else None,
        )
    except (
        OSError,
        KeyError,
        TypeError,
        IndexError,
        AttributeError,
        sqlite3.Error,
        BadZipFile,
        SchemaError,
        ValidationError,
    ) as exc:
        raise ValueError(f"Unavailable or malformed RG0 evidence: {exc}") from exc
    require(expected == document, "Contract certificate differs from independently replayed checks")
    return expected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("collect", "recheck", "summarize"):
        cmd = commands.add_parser(action)
        cmd.add_argument("--bundle", required=True, type=Path)
        cmd.add_argument("--evidence", required=True, type=Path)
        cmd.add_argument("--work", type=Path, required=action != "recheck")
        cmd.add_argument(
            "--host-index",
            type=Path,
            help="Optional same-release index with adjacent case summaries",
        )
        cmd.add_argument(
            "--certificate" if action == "recheck" else "--out", required=True, type=Path
        )
    args = parser.parse_args(argv)
    try:
        if args.action == "recheck":
            result = recheck(
                args.bundle, args.evidence, args.certificate, args.work, args.host_index
            )
        else:
            out = args.out.resolve()
            probe = out.with_name("contract-probes.json")
            require(not out.exists(), "Use a new certificate output")
            if args.action == "collect":
                require(not probe.exists(), "Use a new raw probe output")
                require(
                    sys.flags.isolated and sys.prefix != sys.base_prefix,
                    "Use the installed private interpreter with -I",
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(Path(__file__).with_name("contract_probe.py")),
                        "--bundle",
                        str(args.bundle.resolve()),
                        "--work",
                        str(args.work.resolve()),
                        "--out",
                        str(probe),
                    ],
                    check=True,
                    timeout=900,
                )
            result = assess(
                args.bundle.resolve(),
                args.evidence.resolve(),
                args.work.resolve(),
                probe,
                args.host_index.resolve() if args.host_index else None,
            )
            write_new(out, result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("gate", "case_verdict", "release_id", "release_manifest_sha256")
                }
            )
        )
        return 0
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        sqlite3.Error,
        subprocess.SubprocessError,
    ) as exc:
        parser.exit(1, f"RG0 verification failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
