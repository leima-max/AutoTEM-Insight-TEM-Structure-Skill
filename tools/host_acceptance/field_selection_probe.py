"""Exercise the three J07 field inputs through an installed scientific Harness.

Run with the active managed core Python and -I -B. The three input files are
read-only. Each output directory is new and contains independent stores and
public call transcripts. This probe does not collect native host observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sysconfig
from itertools import count
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_sha(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8"))


def runtime_identity():
    from autotem.harness import service
    from autotem.installation.host_entry import active_python

    prefix = Path(sys.prefix).resolve()
    control = prefix.parent.parent
    pointer = control / "active-runtime.json"
    marker = prefix / ".autotem-runtime.json"
    require(sys.flags.isolated and sys.dont_write_bytecode, "Launch the private Python with -I -B")
    require(active_python(control) == Path(sys.executable).resolve(), "Inactive runtime")
    package_root = Path(sysconfig.get_path("purelib")).resolve()
    origin = Path(service.__file__).resolve()
    require(
        package_root.is_relative_to(prefix) and origin.is_relative_to(package_root),
        "Harness must load from the private installation",
    )
    installation = json.loads(marker.read_bytes())
    active = json.loads(pointer.read_bytes())
    require(active["profile"] == "traditional-core-mcp", "Use the managed core profile")
    return {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "profile": active["profile"],
        "runtime_id": active["runtime_id"],
        "release_manifest_sha256": active["manifest_sha256"],
        "wheel_sha256": installation["wheel_sha256"],
        "package_version": installation["package_version"],
        "pointer": str(pointer),
        "pointer_sha256": sha(pointer.read_bytes()),
        "marker": str(marker),
        "marker_sha256": sha(marker.read_bytes()),
        "harness_origin": str(origin),
        "harness_sha256": sha(origin.read_bytes()),
    }


def selection(variant, source):
    """Decode the expected plane independently of the Harness reader."""
    import numpy as np
    from PIL import Image

    if variant == "dataset":
        with np.load(source, allow_pickle=False) as archive:
            require(set(archive.files) == {"first", "second"}, "Expected datasets first and second")
            first, second = archive["first"], archive["second"]
            require(first.ndim == second.ndim == 2, "Expected two 2D datasets")
            require(not np.array_equal(first, second), "Dataset choices must be distinguishable")
            return "dataset_path", "dataset_path", "second", "absent", {}, second
    if variant == "frame":
        stack = np.load(source, allow_pickle=False)
        require(stack.ndim == 3 and len(stack) >= 2, "Expected a stack with at least two frames")
        require(not np.array_equal(stack[0], stack[1]), "Frame choices must be distinguishable")
        return "selection_indices", "frame", 1, len(stack), {}, stack[1]
    with Image.open(source) as image:
        require(image.format == "TIFF" and image.n_frames >= 2, "Expected multipage RGB TIFF")
        require(image.getbands() == ("R", "G", "B"), "Expected RGB TIFF page zero")
        first = np.array(image)
        image.seek(1)
        require(image.getbands() == ("R", "G", "B"), "Expected RGB TIFF page one")
        second = np.array(image)
        blue = second[..., 2]
        require(not np.array_equal(first[..., 2], blue), "TIFF pages must be distinguishable")
        require(
            all(not np.array_equal(second[..., i], blue) for i in (0, 1)),
            "RGB channels must be distinguishable",
        )
        return "channel", "channel", 2, second.shape[-1], {"frame": 1}, blue


def counts(store):
    with store.connect() as db:
        row = db.execute(
            "SELECT (SELECT count(*) FROM runs), (SELECT count(*) FROM jobs)"
        ).fetchone()
    return {"runs": row[0], "jobs": row[1]}


def parent_evidence(harness, state, source_bytes):
    """Hash the immutable run row, step history, original snapshot and registered bytes."""
    run_id = state["run_id"]
    row = harness.store.get_run(run_id)
    snapshot = Path(row["input_path"])
    require(
        row["request"]["input_mode"] == "snapshot"
        and row["status"] == "needs_input"
        and snapshot.resolve().is_relative_to(harness.store.run_dir(run_id)),
        "Parent did not retain its own snapshot",
    )
    require(
        snapshot.read_bytes() == source_bytes and row["source_hash"] == sha(source_bytes),
        "Parent snapshot changed",
    )
    artifacts = []
    for item in harness.store.artifacts(run_id):
        path = Path(item["path"])
        require(
            path.resolve().is_relative_to(harness.store.run_dir(run_id)), "Artifact escaped parent"
        )
        digest = sha(path.read_bytes())
        require(
            digest == item["sha256"] and path.stat().st_size == item["size"],
            f"Parent artifact changed: {path}",
        )
        artifacts.append(
            {
                "role": item["role"],
                "artifact_id": item["artifact_id"],
                "path": str(path),
                "sha256": digest,
                "size": item["size"],
            }
        )
    return {
        "run_row_sha256": json_sha(row),
        "steps_sha256": json_sha(state["steps"]),
        "events_sha256": json_sha(state["events"]),
        "jobs_sha256": json_sha(state["jobs"]),
        "snapshot": str(snapshot),
        "snapshot_sha256": row["source_hash"],
        "artifacts": artifacts,
    }


def verify_plane(harness, child, expected):
    import numpy as np

    by_operation = {step["operation_id"]: step for step in child["steps"]}
    raw_id = harness.store.role(child["run_id"], "raw")
    raw = harness.store.artifact(raw_id)
    require(
        raw["role"] == "raw"
        and raw["kind"] == "array"
        and raw["run_id"] == child["run_id"]
        and raw["execution_usable"],
        "Invalid raw artifact",
    )
    load = by_operation["input.load"]
    require(raw_id in load["result"]["artifact_ids"], "Input loader did not publish this raw ID")
    path = Path(raw["path"])
    require(
        path.suffix == ".npy"
        and path.resolve().is_relative_to(harness.store.run_dir(child["run_id"])),
        "Expected a child-local NPY",
    )
    disk = np.load(path, allow_pickle=False)
    expected_bytes, actual_bytes = expected.tobytes(order="C"), disk.tobytes(order="C")
    require(
        disk.shape == expected.shape
        and disk.dtype == expected.dtype
        and actual_bytes == expected_bytes,
        "Stored pixels differ from the source plane",
    )
    require(
        sha(path.read_bytes()) == raw["sha256"]
        and sha(actual_bytes) == raw["metadata"]["array_sha256"]
        and list(disk.shape) == raw["metadata"]["shape"]
        and str(disk.dtype) == raw["metadata"]["dtype"],
        "Stored raw file or metadata mismatch",
    )
    consumers = {}
    for operation in ("quality.assess", "preprocessing.apply"):
        step = by_operation[operation]
        used = step["result"]["consumed_artifacts"]["raw"]
        require(
            step["status"] == "completed" and used == raw_id, f"{operation} consumed another plane"
        )
        consumers[operation] = used
    processed = harness.store.role(child["run_id"], "processed")
    fft = by_operation["diffraction.fft"]
    require(
        processed in by_operation["preprocessing.apply"]["result"]["artifact_ids"]
        and fft["status"] == "completed"
        and fft["result"]["consumed_artifacts"]["processed"] == processed,
        "FFT lineage does not follow preprocessing of the selected plane",
    )
    return {
        "artifact_id": raw_id,
        "path": str(path),
        "npy_sha256": raw["sha256"],
        "dtype": str(disk.dtype),
        "shape": list(disk.shape),
        "pixel_bytes": len(actual_bytes),
        "expected_pixels_sha256": sha(expected_bytes),
        "disk_pixels_sha256": sha(actual_bytes),
        "direct_raw_consumers": consumers,
        "fft_consumed_processed_artifact_id": processed,
        "fft_output_artifact_ids": fft["result"]["artifact_ids"],
    }


def scientific_job(state):
    (job,) = state["jobs"]
    (started,) = [event for event in state["events"] if event["kind"] == "worker_started"]
    worker = started["payload"]
    require(
        job["job_id"] == state["job_id"]
        and job["run_id"] == state["run_id"] == started["run_id"]
        and job["status"] == state["status"]
        and 0 < job["started"] <= started["created"] <= job["ended"]
        and worker["pid"] > 0
        and worker["process_identity"],
        "Expected a settled scientific job with a recorded worker identity",
    )
    return {**job, "worker": worker}


def probe(variant, source, root):
    from autotem.harness import Harness
    from autotem.harness.cli import dispatch_request

    original = source.read_bytes()
    field_name, answer_key, value, invalid, initial, expected = selection(variant, source)
    harness = Harness(root)
    trace = root / "calls.jsonl"
    call_ids = count(1)

    def call(tool, error_code=None, **arguments):
        request = {
            "schema_version": "1.0.0",
            "request_id": f"{variant}-{next(call_ids)}-{tool}",
            "tool": tool,
            "arguments": arguments,
        }
        reply = dispatch_request(harness, request)
        with trace.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps({"request": request, "response": reply}, allow_nan=False) + "\n"
            )
        require(reply["request_id"] == request["request_id"], "Unmatched tool reply")
        if error_code:
            require(
                not reply["ok"] and reply["error"]["code"] == error_code,
                f"{tool}: expected {error_code}, got {reply['error']}",
            )
            return reply["error"]
        require(reply["ok"], f"{tool}: {reply['error']}")
        return reply["result"]

    parent = call("inspect_input", path=str(source), selection=initial)
    require(
        parent["status"] == "needs_input",
        f"Inspection ended in {parent['status']}: {parent['error']}",
    )
    (question,) = parent["input_requests"]
    (field,) = question["fields"]
    request_id, parent_id = question["request_id"], parent["run_id"]
    evidence_id = field["evidence"]["selection_artifact_id"]
    require(
        field["name"] == field_name
        and answer_key in field["answer_schema"]
        and question["status"] == "open"
        and evidence_id in question["evidence_artifact_ids"]
        and question["run_id"] == parent_id
        and request_id == parent["error"]["details"]["request_id"],
        "Missing disk-backed selection question",
    )
    options = harness.store.load_json(evidence_id)["fields"]
    require(len(options) == 1 and options[0]["name"] == field_name, "Selection options differ")
    schema = field["answer_schema"][answer_key]["value"]
    if variant == "dataset":
        require(
            set(options[0]["choices"]) == set(schema["enum"]) == {"first", "second"},
            "Wrong datasets",
        )
    else:
        bound = options[0]["shape"][0] if variant == "frame" else options[0]["count"]
        require(
            bound == invalid and schema["minimum"] == 0 and schema["maximum"] == invalid - 1,
            "Wrong allowed choices",
        )
    old = parent_evidence(harness, parent, original)
    plan = {
        "input_path": str(source),
        "input_mode": "snapshot",
        "selection": initial,
        "steps": [
            {"id": key, "operation_id": operation}
            for key, operation in (
                ("inspect", "input.load"),
                ("quality", "quality.assess"),
                ("preprocess", "preprocessing.apply"),
                ("fft", "diffraction.fft"),
            )
        ],
    }

    def answer(selected):
        return {
            **plan,
            "user_input": {
                "response": {
                    "request_id": request_id,
                    "source": "artifact",
                    "evidence_artifact_ids": [evidence_id],
                    "basis": "Scripted J07 QA selection using the parent's file options.",
                    "answers": {answer_key: {"value": selected}},
                }
            },
        }

    before = counts(harness.store)
    denied = call(
        "submit_analysis",
        error_code="INVALID_INPUT_RESPONSE",
        plan=answer(invalid),
        parent_run_id=parent_id,
        idempotency_key="rejected",
        asynchronous=False,
    )
    after = counts(harness.store)
    refused = call("get_run_status", run_id=parent_id)
    (rejected,) = refused["input_responses"]
    require(
        before == after == {"runs": 1, "jobs": 1}
        and rejected["request_id"] == request_id
        and rejected["child_run_id"] is None
        and rejected["validation"]["status"] == "rejected"
        and rejected["validation"]["code"] == denied["code"]
        and refused["input_requests"][0]["status"] == "open",
        "Invalid choice changed the parent, created work, or lacks rejection evidence",
    )
    good = answer(value)
    validated = call("validate_plan", plan=good)
    require(
        validated["valid"]
        and validated["input_response_validation"]["parent_run_id"] == parent_id
        and validated["input_response_validation"]["request_id"] == request_id
        and validated["plan"]["steps"][0]["parameters"][answer_key] == value,
        "Validated response did not adopt the selected value",
    )
    child = call(
        "submit_analysis",
        plan=good,
        parent_run_id=parent_id,
        idempotency_key="accepted",
        asynchronous=False,
    )
    child_id = child["run_id"]
    require(
        child["status"] == "completed"
        and child["parent_run_id"] == parent_id
        and child_id != parent_id,
        f"Child failed to complete: {child['status']}, {child['error']}",
    )
    (accepted,) = child["input_responses"]
    require(
        accepted["request_id"] == request_id
        and accepted["parent_run_id"] == parent_id
        and accepted["child_run_id"] == child_id
        and accepted["validation"]["status"] == "accepted"
        and accepted["response"]["answers"] == good["user_input"]["response"]["answers"],
        "Missing accepted answer on the completed child",
    )
    final = call("get_run_status", run_id=parent_id)
    require(
        final["status"] == "needs_input"
        and final["input_requests"][0]["status"] == "answered"
        and final["latest_response_child_run_id"] == child_id
        and final["input_responses"] == [rejected, accepted],
        "Parent did not retain the original request and both reply receipts",
    )
    require(
        parent_evidence(harness, final, original) == old and source.read_bytes() == original,
        "Parent evidence or original field input changed",
    )
    child_snapshot = Path(harness.store.get_run(child_id)["input_path"])
    require(
        child_snapshot.resolve().is_relative_to(harness.store.run_dir(child_id))
        and child_snapshot.read_bytes() == original,
        "Child did not inherit the original bytes",
    )
    require(
        counts(harness.store) == {"runs": 2, "jobs": 2}, "Unexpected additional scientific work"
    )
    return {
        "variant": variant,
        "input": str(source),
        "input_sha256": sha(original),
        "input_bytes": len(original),
        "parent_run_id": parent_id,
        "parent_job": scientific_job(parent),
        "request_id": request_id,
        "initial_selection": initial,
        "question_field": field_name,
        "choices": options,
        "selection_artifact_id": evidence_id,
        "parent_immutable": old,
        "rejection": {
            "submitted": {answer_key: invalid},
            "error": denied,
            "response_id": rejected["response_id"],
            "before": before,
            "after": after,
            "child_run_id": rejected["child_run_id"],
        },
        "accepted": {
            "submitted": {answer_key: value},
            "response_id": accepted["response_id"],
            "child_run_id": child_id,
            "job": scientific_job(child),
            "snapshot": str(child_snapshot),
        },
        "selected_plane": verify_plane(harness, child, expected),
        "trace": str(trace),
        "trace_sha256": sha(trace.read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for variant, help_text in (
        ("dataset", "NPZ containing first and second"),
        ("frame", "NPY stack; select index 1"),
        ("channel", "RGB TIFF; select page 1, channel 2"),
    ):
        parser.add_argument("--" + variant, type=Path, required=True, help=help_text)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New output directory outside the installation project",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    runtime = Path(sys.prefix).resolve()
    if not sys.flags.isolated or not sys.dont_write_bytecode or runtime.parent.name != "runtimes":
        parser.error("Use the private active Python with -I -B")
    if out.is_relative_to(runtime.parent.parent.parent):
        parser.error("Choose an isolated output directory outside the installation project")
    try:
        out.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        parser.error(str(exc))
    scratch = out / "tmp"
    scratch.mkdir()
    for key in ("TEMP", "TMP", "TMPDIR", "AUTOTEM_TMP_DIR"):
        os.environ[key] = str(scratch)
    os.environ.update(
        MPLCONFIGDIR=str(scratch / "mpl"), MPLBACKEND="Agg", NUMBA_CACHE_DIR=str(scratch / "numba")
    )
    evidence = {
        "schema_version": 1,
        "execution_layer": "scientific_probe",
        "case_id": "J07",
        "native_host_journey": "not_evaluated",
        "scientific_accuracy": "not_evaluated",
        "probe_sha256": sha(Path(__file__).read_bytes()),
        "cases": [],
    }
    result = out / "selection-evidence.json"
    try:
        evidence["runtime"] = runtime_identity()
        for variant in ("dataset", "frame", "channel"):
            evidence["cases"].append(
                probe(variant, getattr(args, variant).resolve(strict=True), out / variant[0])
            )
        evidence["status"] = "passed"
    except Exception as exc:
        evidence["status"] = "failed"
        evidence["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(result, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
