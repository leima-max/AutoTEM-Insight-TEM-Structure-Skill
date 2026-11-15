"""Verify private host captures against the installed release and actual Harness evidence."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
from datetime import datetime, timezone
from decimal import Decimal
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path

MATRIX = json.loads(Path(__file__).with_name("matrix.json").read_text("utf-8"))
HOSTS = tuple(MATRIX["hosts"])
VARIANTS = MATRIX["cases"]
CASES = tuple(VARIANTS)
GPA_VARIANTS = ("gpa_answered", "gpa_unanswered", "gpa_invalid_reply")


def case_variants(case):
    return (*VARIANTS[case], *GPA_VARIANTS) if case == "J08" else VARIANTS[case]


TOOLS = {
    "list_capabilities",
    "inspect_input",
    "validate_plan",
    "submit_analysis",
    "get_run_status",
    "cancel_run",
    "list_artifacts",
    "read_artifact",
    "export_report",
}
DOMAINS = {"data-quality", "diffraction", "atomic-strain"}
OPERATIONS = {
    "J01": {
        "input.load",
        "calibration.resolve",
        "diffraction.fft",
        "diffraction.metrics",
        "diffraction.index",
    },
    "J02": {"atomic.detect", "atomic.refine", "atomic.distances", "atomic.intensity"},
    "J03": {"strain.gpa"},
    "J04": {"atomic.refine", "strain.atom"},
    "J05": {"diffraction.pattern", "diffraction.metrics"},
    "J06": {"calibration.resolve"},
    "J07": {"input.load"},
    "J15": {"simulation.diffraction"},
}
ROLES = {
    "J01": {"diffraction"},
    "J02": {"refined_atoms", "atom_distances", "atomic_intensity"},
    "J03": {
        "gpa",
        "gpa.quality_mask",
        "gpa.strain_xx",
        "gpa.strain_yy",
        "gpa.strain_xy",
        "gpa.displacement_x",
        "gpa.displacement_y",
        "gpa.rotation_xy",
        "gpa.uncertainty",
    },
    "J04": {
        "atom_strain",
        "atom_strain.valid_atom_mask",
        "atom_strain.atom_strain_xx",
        "atom_strain.atom_strain_yy",
        "atom_strain.atom_strain_xy",
    },
    "J05": {"diffraction", "reciprocal_image", "raw"},
    "J06": {"calibration"},
    "J07": {"raw"},
    "J15": {"simulation"},
}


class EvidenceError(ValueError):
    """An explicit contradiction or integrity failure."""


class NotEvaluated(EvidenceError):
    """A required observation has not been captured."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def observed(condition, message):
    if not condition:
        raise NotEvaluated(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def unique_json(value):
    def obj(pairs):
        result = {}
        for key, item in pairs:
            require(key not in result, "duplicate_json_key")
            result[key] = item
        return result

    def nonfinite(_):
        raise EvidenceError("nonfinite_json_number")

    return json.loads(value, object_pairs_hook=obj, parse_constant=nonfinite)


def stamp(value):
    # PowerShell DateTimeOffset "o" contains seven fractional digits. CPython
    # 3.10 accepts only three/six here; use microseconds for the DB time window.
    value = re.sub(
        r"(?<=\d)\.(\d{1,9})(?=Z$|[+-]\d\d:\d\d$)", lambda m: "." + m[1].ljust(6, "0")[:6], value
    )
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "capture_time_requires_offset")
    return parsed.timestamp()


def capture_file(base, name):
    observed(isinstance(name, str) and name, "missing_capture_reference")
    path = (base / name).resolve()
    require(
        not Path(name).is_absolute() and path.is_relative_to(base), "capture_path_outside_directory"
    )
    observed(path.is_file() and path.stat().st_size > 0, "capture_file_missing_or_empty")
    return path


def same_json(left, right):
    """Compare JSON values with numeric equality while keeping booleans distinct."""
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_json(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            same_json(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right and isinstance(left, bool) == isinstance(right, bool)


def envelope(tool, payload):
    if isinstance(payload, dict) and payload.get("tool") == tool:
        result = payload
    else:
        payload = payload or {}
        copies = [
            payload[key]
            for key in ("structuredContent", "structured_content")
            if payload.get(key) is not None
        ]
        contents = [c for c in payload.get("content", []) if c.get("type") == "text"]
        if contents or not copies:
            observed(len(contents) == 1, "actual_harness_response_not_visible")
            copies.append(unique_json(contents[0]["text"]))
        result = copies[0]
        require(
            all(same_json(result, other) for other in copies[1:]),
            "conflicting_harness_response_copies",
        )
    require(
        isinstance(result, dict)
        and result.get("tool") == tool
        and result.get("schema_version") == "1.0.0"
        and type(result.get("ok")) is bool,
        "tool_envelope_mismatch",
    )
    require(
        (result["ok"] and result.get("error") is None and result.get("result") is not None)
        or (not result["ok"] and isinstance(result.get("error"), dict)),
        "tool_envelope_inconsistent",
    )
    return result


def codex_mcp_response(item, started):
    """Codex also marks a received MCP isError business response as a failed item."""
    status, payload = item.get("status"), item.get("result")
    observed(
        status in {"completed", "failed"} and payload and item.get("error") is None,
        "host_did_not_receive_tool_result",
    )
    if status == "completed":
        return envelope(item["tool"], payload)

    # Bind a failed completion to the original in-flight call. An MCP error
    # response still has the tool-result wrapper; a transport failure has none.
    require(
        started is not None
        and started.get("status") == "in_progress"
        and item["tool"] == started["tool"]
        and isinstance(item.get("arguments"), dict)
        and same_json(item["arguments"], started["arguments"]),
        "failed_mcp_call_request_mismatch",
    )
    require(
        isinstance(payload, dict)
        and "tool" not in payload
        and all(payload.get(key, True) is True for key in ("isError", "is_error")),
        "failed_mcp_item_result_mismatch",
    )
    response = envelope(item["tool"], payload)
    error = response.get("error")
    require(
        response["ok"] is False
        and "result" in response
        and response["result"] is None
        and "request_id" in response
        and response["request_id"] is None  # MCP dispatch does not supply a CLI request_id.
        and isinstance(error, dict)
        and isinstance(error.get("code"), str)
        and re.fullmatch(r"[A-Z][A-Z0-9_]*", error["code"])
        and isinstance(error.get("message"), str)
        and error["message"].strip()
        and type(error.get("retryable")) is bool,
        "invalid_failed_mcp_error_envelope",
    )
    return response


def unsupported_resource_discovery(item, started, server):
    """Codex reports unsupported protocol discovery alongside named tool calls."""
    methods = {
        "list_mcp_resources": "resources/list",
        "list_mcp_resource_templates": "resources/templates/list",
    }
    method = methods.get(item["tool"])
    if method is None or started is None:
        return False
    # Only skip the paired native probe and its specific method-not-found reply.
    # A transport failure, a different target, or a hidden Harness result stays visible.
    return (
        started.get("tool") == item["tool"]
        and started.get("status") == "in_progress"
        and started.get("arguments") == item.get("arguments") == {"server": server}
        and started.get("result") is None and started.get("error") is None
        and item.get("status") == "failed" and item.get("result") is None
        and item.get("error") == {
            "message": f"{method} failed: {method} failed for `{server}`: Mcp error: -32601: Method not found"
        }
    )


def read_turn(base, turn, host, capture_format, server):
    prompt_file = capture_file(base, turn.get("prompt"))
    trace_file = capture_file(base, turn.get("trace"))
    prompt = prompt_file.read_text("utf-8-sig").strip()
    observed(prompt, "missing_user_input")
    calls, messages, attachments, sessions, delegation_events = [], [], [], [], []
    skill_opened = False
    if capture_format == "codex-exec-jsonl":
        require(host == "codex", "capture_format_host_mismatch")
        phase = "before_thread"
        completed_ids = set()
        started_calls = {}
        with trace_file.open(encoding="utf-8-sig") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                event = unique_json(line)
                kind = event.get("type")
                if kind == "thread.started":
                    require(phase == "before_thread", "host_turn_event_out_of_order")
                    sessions.append(event["thread_id"])
                    phase = "before_turn"
                elif kind == "turn.started":
                    require(phase == "before_turn", "host_turn_event_out_of_order")
                    phase = "active"
                elif kind == "turn.completed":
                    require(phase == "active", "host_turn_event_out_of_order")
                    phase = "completed"
                elif kind == "turn.failed":
                    raise NotEvaluated("host_turn_failed")
                elif kind in {"item.started", "item.completed"}:
                    item = event["item"]
                    # Codex can emit startup diagnostics before turn.started.
                    if item["type"] == "error":
                        continue
                    require(phase == "active", "host_item_outside_turn")
                    if kind == "item.started":
                        if item["type"] == "mcp_tool_call" and item.get("server") == server:
                            item_id = item.get("id")
                            if isinstance(item_id, str) and item_id:
                                require(
                                    item_id not in started_calls and item_id not in completed_ids,
                                    "duplicate_host_item_start",
                                )
                                started_calls[item_id] = item
                        continue
                    item_id = item.get("id")
                    if item_id is not None:
                        require(item_id not in completed_ids, "duplicate_host_item_completion")
                        completed_ids.add(item_id)
                    if item["type"] == "agent_message":
                        messages.append((line_no, item["text"]))
                    if item["type"] == "command_execution" and item.get("exit_code") == 0:
                        command = item.get("command", "").replace("\\", "/").lower()
                        output = item.get("aggregated_output", "")
                        if (
                            re.search(r"\b(get-content|cat|type|more)\s", command)
                            and "autotem/skill.md" in command
                        ) and re.search(r"(?im)^name:\s*autotem\s*$", output):
                            skill_opened = True
                    if item["type"] in {"collab_tool_call", "collab_agent_tool_call"}:
                        delegation_events.append(json.dumps(item, ensure_ascii=False))
                    if item["type"] == "mcp_tool_call" and item.get("server") == server:
                        tool = item["tool"]
                        started = started_calls.pop(item_id, None)
                        if unsupported_resource_discovery(item, started, server):
                            continue
                        require(tool in TOOLS, "unexpected_tool_in_autotem_namespace")
                        calls.append(
                            {
                                "tool": tool,
                                "arguments": item["arguments"],
                                "response": codex_mcp_response(item, started),
                                "line": line_no,
                            }
                        )
        observed(phase == "completed", "incomplete_host_turn")
    else:
        require(capture_format == "observed-host-export", "unsupported_capture_format")
        raw = unique_json(trace_file.read_text("utf-8-sig"))
        require(raw["host"] == host, "manual_observation_host_mismatch")
        sessions.append(raw["session_id"])
        for delegation in raw.get("delegation_events", []):
            attachments.append(digest(capture_file(base, delegation["original_evidence"])))
            delegation_events.append(json.dumps(delegation, ensure_ascii=False))
        for i, item in enumerate(raw["calls"]):
            if item["server"] != server:
                continue
            tool = item["tool"]
            require(tool in TOOLS, "unexpected_tool_in_autotem_namespace")
            attachments.append(digest(capture_file(base, item["original_evidence"])))
            calls.append(
                {
                    "tool": tool,
                    "arguments": item["arguments"],
                    "response": envelope(tool, item["response"]),
                    "line": i + 1,
                }
            )
        attachments.append(digest(capture_file(base, raw["answer_evidence"])))
        messages.append((len(calls) + 1, raw["answer"]))
    require(all(isinstance(x, str) and x for x in sessions), "invalid_host_session")
    observed(messages and messages[-1][1].strip(), "missing_host_answer")
    observed(not calls or messages[-1][0] > calls[-1]["line"], "missing_answer_after_tool_return")
    return {
        "session": sessions[0],
        "calls": calls,
        "answer": messages[-1][1],
        "prompt": prompt,
        "prompt_sha256": digest(prompt_file),
        "trace_sha256": digest(trace_file),
        "attachments": attachments,
        "delegation_events": delegation_events,
        "skill_opened": skill_opened,
    }


def run_evidence(root, run_id, window, runtime):
    require(isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id), "invalid_run_id")
    dbpath = root / "harness.sqlite3"
    observed(dbpath.is_file(), "harness_database_missing")
    db = sqlite3.connect(dbpath.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute("BEGIN")
        run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        require(
            run and window[0] - 1 <= run["created"] <= window[1] + 1, "run_outside_host_capture"
        )
        rows = db.execute(
            "SELECT * FROM artifacts WHERE run_id=? AND committed=1", (run_id,)
        ).fetchall()
        steps = db.execute(
            "SELECT step_id,operation_id,status,result,error FROM steps "
            "WHERE run_id=? ORDER BY rowid",
            (run_id,),
        ).fetchall()
        bindings = [
            unique_json(r[0])
            for r in db.execute(
                "SELECT payload FROM events WHERE run_id=? AND kind='runtime_bound' "
                "ORDER BY event_id",
                (run_id,),
            )
        ]
        jobs = [
            dict(r) for r in db.execute("SELECT job_id,status FROM jobs WHERE run_id=?", (run_id,))
        ]
        replies = [
            dict(r)
            for r in db.execute(
                "SELECT parent_run_id,child_run_id,request_id,response,validation,adopted_values "
                "FROM input_responses WHERE child_run_id=?",
                (run_id,),
            )
        ]
        question_records = {
            row["request_id"]: {
                "step_id": row["step_id"],
                "operation_id": row["operation_id"],
                "fields": unique_json(row["fields"]),
                "evidence": unique_json(row["evidence_artifact_ids"]),
            }
            for row in db.execute(
                "SELECT request_id,step_id,operation_id,fields,evidence_artifact_ids "
                "FROM input_requests WHERE run_id=?",
                (run_id,),
            )
        }
        all_children = [
            r[0] for r in db.execute("SELECT run_id FROM runs WHERE parent_run_id=?", (run_id,))
        ]
    finally:
        db.close()
    request = unique_json(run["request"])
    directory = (root / "runs" / run_id).resolve()
    require(directory.is_relative_to(root), "invalid_run_directory")
    if request["input_mode"] == "snapshot":
        raw = Path(run["input_path"]).resolve()
        require(raw.is_relative_to(directory / "raw"), "snapshot_outside_run")
    else:
        raw = Path(request["input_path"]).resolve()
        require(raw == Path(run["input_path"]).resolve(), "reference_source_mismatch")
    require(raw.is_file() and digest(raw) == run["source_hash"], "raw_hash_mismatch")
    artifacts = {}
    scientific_files = {}
    package_hashes = None
    environment = None
    physical_measurements = []
    modality = request.get("modality", "unknown")
    successful = {s["step_id"]: s["operation_id"] for s in steps if s["status"] == "completed"}
    for row in rows:
        path = Path(row["path"]).resolve()
        require(
            path.is_relative_to(directory / "artifacts") and path.is_file(),
            "invalid_registered_path",
        )
        require(
            path.stat().st_size == row["size"] and digest(path) == row["sha256"],
            "artifact_hash_mismatch",
        )
        meta = unique_json(row["metadata"])
        if row["kind"] == "array":
            import numpy as np

            array = np.load(path, mmap_mode="r", allow_pickle=False)
            require(
                list(array.shape) == meta["shape"] and str(array.dtype) == meta["dtype"],
                "registered_array_shape_or_dtype_changed",
            )
            del array
        if row["step_id"] in successful:
            scientific_files[(row["step_id"], row["role"])] = path
        artifacts[row["artifact_id"]] = {
            "role": row["role"],
            "step_id": row["step_id"],
            "kind": row["kind"],
            "file": path.relative_to(directory).as_posix(),
            "sha256": row["sha256"],
            "size": row["size"],
            "execution_usable": row["step_id"] in successful,
            **{
                k: meta[k]
                for k in (
                    "shape",
                    "dtype",
                    "unit",
                    "distance_unit",
                    "coordinate_unit",
                    "coordinate_convention",
                    "coordinates_sha256",
                    "array_sha256",
                    "image_role",
                    "validity_sha256",
                    "scientific_level",
                )
                if k in meta
            },
        }
        if row["role"] == "implementation_fingerprints":
            package_hashes = unique_json(path.read_text("utf-8"))
        if row["role"] == "environment":
            environment = unique_json(path.read_text("utf-8"))
        if row["role"] == "dataset" and row["step_id"] in successful:
            modality = unique_json(path.read_text("utf-8"))["modality"]
        if successful.get(row["step_id"]) in {"diffraction.metrics", "atomic.distances"}:
            if row["role"] in {"diffraction", "atom_distances"}:
                measurement = unique_json(path.read_text("utf-8"))
                units = positive_distance_units(row["role"], measurement)
                if units:
                    physical_measurements.append(
                        {"artifact_id": row["artifact_id"], "units": units}
                    )
    if runtime is not None:
        observed(bindings, "missing_managed_job_runtime_binding")
        require(
            {b["job_id"] for b in bindings} == {j["job_id"] for j in jobs}
            and all(
                b["runtime_id"] == Path(sys.prefix).name
                and b["package_version"] == runtime["autotem_version"]
                for b in bindings
            ),
            "job_runtime_mismatch",
        )
        require(isinstance(package_hashes, dict) and environment, "missing_runtime_provenance")
        import autotem

        package = Path(autotem.__file__).resolve().parent
        expected = {f.relative_to(package).as_posix(): digest(f) for f in package.rglob("*.py")}
        require(expected == package_hashes, "run_source_differs_from_release")
        require(environment["python"].split()[0] == runtime["python"], "run_python_mismatch")
        require(
            all(
                Path(op["source"]).resolve().is_relative_to(package)
                and digest(Path(op["source"])) == op["source_sha256"]
                for op in environment["operations"]
            ),
            "handler_origin_mismatch",
        )
    science = []
    parameters = {s["id"]: s.get("parameters", {}) for s in request["steps"]}
    for step in steps:
        step_id = step["step_id"]
        op = successful.get(step_id)
        if op in {"strain.gpa", "strain.atom"}:
            science.append(
                check_strain_fields(scientific_files, step, parameters[step_id], artifacts)
            )
        if op == "simulation.diffraction":
            simulation = unique_json(scientific_files[(step_id, "simulation")].read_text("utf-8"))
            require(
                simulation["patterns"] and all(p["spots"] for p in simulation["patterns"]),
                "empty_simulated_patterns",
            )
    return {
        "run_id": run_id,
        "parent_run_id": run["parent_run_id"],
        "status": run["status"],
        "error_code": unique_json(run["error"] or "{}").get("code"),
        "job_id": run["job_id"],
        "jobs": jobs,
        "source_sha256": run["source_hash"],
        "steps": [
            {
                **{key: s[key] for key in ("step_id", "operation_id", "status")},
                "warnings_sha256": [
                    text_digest(w) for w in unique_json(s["result"] or "{}").get("warnings", [])
                ],
                "consumed_artifacts": unique_json(s["result"] or "{}").get(
                    "consumed_artifacts", {}
                ),
                "error_code": unique_json(s["error"] or "{}").get("code"),
            }
            for s in steps
        ],
        "operations": {s["id"]: s["operation_version"] for s in request["steps"]},
        "budgets": request["budgets"],
        "parameter_diff_present": bool(run["parameter_diff"]),
        "artifacts": artifacts,
        "bindings": bindings,
        "modality": modality,
        "physical_measurements": physical_measurements,
        "strain_validity": science,
        "_replies": replies,
        "requested_fields": {
            key: [field["name"] for field in item["fields"]]
            for key, item in question_records.items()
        },
        "question_evidence": {key: item["evidence"] for key, item in question_records.items()},
        "_questions": question_records,
        "_children": all_children,
        "_request": request,
        "_directory": directory,
    }


def check_gpa_geometry(mask, source, config, consumed, artifacts):
    """Bind full fields and half-open reference boxes to the image actually used."""
    role = config.get("image_role", "raw")
    image_id = consumed.get(role)
    image = artifacts.get(image_id, {})
    provenance = source["provenance"]
    require(
        image.get("role") == role
        and image.get("execution_usable")
        and image.get("kind") == "array"
        and provenance.get("image_role", role) == role
        and mask.ndim == 2
        and list(mask.shape) == image.get("shape"),
        "gpa_fields_do_not_cover_consumed_image",
    )
    shared = config.get("reference_region")
    pair = config.get("reference_regions")
    expected = pair if pair is not None else [shared, shared]
    require(isinstance(expected, list) and len(expected) == 2, "invalid_gpa_reference_pair")
    regions = dict(zip(("g1", "g2"), expected, strict=True))
    common = expected[0] if expected[0] == expected[1] else None
    # Older records stored a common box alone. Distinct boxes need both labels.
    recorded = provenance.get(
        "reference_regions", dict.fromkeys(regions, source.get("reference_region"))
    )
    absolute = config.get("absolute_reference", False)
    require(
        recorded == regions
        and source.get("reference_region") == common
        and provenance.get("absolute_reference_accepted", False) is absolute,
        "gpa_reference_differs_from_request",
    )
    coverage = {}
    for label, box in regions.items():
        if box is None:
            require(absolute is True, "gpa_reference_without_valid_pixels")
            continue
        require(
            isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box),
            "invalid_gpa_reference_bounds",
        )
        y0, y1, x0, x1 = box
        require(
            0 <= y0 < y1 <= mask.shape[0] and 0 <= x0 < x1 <= mask.shape[1],
            "invalid_gpa_reference_bounds",
        )
        valid = int(mask[y0:y1, x0:x1].sum())
        require(valid >= 9, "gpa_reference_without_valid_pixels")
        coverage[label] = {"box": box, "valid_count": valid, "total_count": (y1 - y0) * (x1 - x0)}
    return {
        "image_artifact_id": image_id,
        "image_shape": image["shape"],
        "reference_coverage": coverage,
    }


def check_strain_fields(files, step, config, artifacts):
    import numpy as np

    step_id = step["step_id"]
    operation = step["operation_id"]
    prefix = "gpa" if operation == "strain.gpa" else "atom_strain"
    source = unique_json(files[(step_id, prefix)].read_text("utf-8"))

    def array(field):
        return np.load(files[(step_id, f"{prefix}.{field}")], mmap_mode="r", allow_pickle=False)

    if prefix == "gpa":
        mask = array("quality_mask")
        fields = (
            "displacement_x",
            "displacement_y",
            "strain_xx",
            "strain_yy",
            "strain_xy",
            "rotation_xy",
            "uncertainty",
        )
        require(
            source["provenance"].get("complete_arrays") is True,
            "strain_fields_not_declared_complete",
        )
    else:
        mask = array("valid_atom_mask")
        fields = ("atom_strain_xx", "atom_strain_yy", "atom_strain_xy")
        require(
            source["provenance"].get("map_semantics", "").startswith("display_interpolation")
            and source["provenance"].get("reference_atom_count", 0) >= 3,
            "missing_atomic_strain_reference_or_display_limits",
        )
    require(mask.dtype == np.bool_ and mask.any(), "strain_validity_mask_empty")
    geometry = {}
    if prefix == "gpa":
        consumed = unique_json(step["result"] or "{}").get("consumed_artifacts", {})
        geometry = check_gpa_geometry(mask, source, config, consumed, artifacts)
    for name in fields:
        values = array(name)
        require(
            values.shape == mask.shape and np.isfinite(values[mask]).all(),
            "invalid_strain_field_in_valid_region",
        )
    return {
        "step_id": step_id,
        "valid_count": int(mask.sum()),
        "total_count": mask.size,
        "scientific_level": "diagnostic",
        **geometry,
    }


def positive_distance_units(role, result):
    """Check real physical units and values inside the registered result itself."""
    physical_units = {"nm", "pm", "Å", "angstrom", "um", "µm"}
    if role == "diffraction":
        values = [(m["value"], m["unit"]) for m in result["d_spacings"]]
        if result.get("provenance", {}).get("physical") is not True:
            return []
    else:
        values = [(v, result["unit"]) for v in result["nearest_distances"]]
    return sorted(
        {
            unit
            for value, unit in values
            if unit in physical_units
            and type(value) in {int, float}
            and math.isfinite(value)
            and value > 0
        }
    )


# This verifier also runs as a stand-alone file extracted with matrix.json on the host.
_LENGTH = {
    "nm": 1.0,
    "nanometre": 1.0,
    "nanometer": 1.0,
    "纳米": 1.0,
    "pm": 0.001,
    "picometre": 0.001,
    "picometer": 0.001,
    "皮米": 0.001,
    "å": 0.1,
    "angstrom": 0.1,
    "埃": 0.1,
    "um": 1000.0,
    "µm": 1000.0,
    "μm": 1000.0,
    "micrometre": 1000.0,
    "micrometer": 1000.0,
    "微米": 1000.0,
}
_NUMBER = r"[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:[eE][+-]?\d+)?"
_UNIT = "(?:" + "|".join(_LENGTH) + r")(?:s)?(?![a-zA-Z⁻−/²³^]|\s*(?:/|per\s+pixel|每像素))"
_QUANTITY = re.compile(
    rf"(?<![a-zA-Z0-9_./])(?P<values>{_NUMBER}(?:\s*(?:,\s*(?:and\s+)?|、|\band\b|和|与)\s*{_NUMBER})*)"
    rf"(?:\s*(?:±|\+/-)\s*(?P<uncertainty>{_NUMBER}))?\s*(?P<unit>{_UNIT})",
    re.I,
)
_DISTANCE_LABEL = re.compile(
    r"间距|面距|晶格条纹周期|\b(?:d[\s-]*)?spacings?\b|lattice (?:fringe )?(?:spacing|period)|"
    r"interplanar distance|espacement|espaciado|Netzebenenabstand|"
    r"(?<![\w])d(?:[_₀-₉]?\d+)?\s*(?:[=≈≃:：]|[([])",
    re.I,
)
_UNCERTAINTY_LABEL = re.compile(
    r"不确定度|不确定性|uncertaint|incertidumbre|incertitude|Unsicherheit", re.I
)
_ID = re.compile(r"(?<![a-zA-Z0-9])[0-9a-f]{32}(?![a-zA-Z0-9])")


def length_scale(unit):
    return _LENGTH.get(unit.casefold().removesuffix("s").replace("μ", "µ"))


def spacing_statements(final):
    """Read d-spacing statements, labelled table columns and numerical uncertainties."""
    column = None
    for line in final.splitlines():
        line = line.replace("**", "").replace("`", "")
        if line.count("|") >= 2:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            for n, cell in enumerate(cells):
                unit = re.search(rf"[(\[/]\s*({_UNIT})\s*[])]?", cell, re.I)
                if _DISTANCE_LABEL.search(cell) and unit:
                    column = (n, unit[1])
                    break
            else:
                if column and column[0] < len(cells):
                    value = cells[column[0]].lstrip("≈~ ")
                    if re.fullmatch(_NUMBER, value):
                        yield value, column[1], False
                    elif re.match(r"[0-9+.\-]", value):
                        for match in _QUANTITY.finditer(value + " " + column[1]):
                            for val in re.findall(_NUMBER, match["values"]):
                                yield val, match["unit"], False
                            if match["uncertainty"] is not None:
                                yield match["uncertainty"], match["unit"], True
            continue
        column = None
        for sentence in re.split(r"[。；;!?！？]|(?<=\.)\s+", line):
            # Crystal parameters and per-pixel calibration are separate quantities.
            sentence = re.sub(
                r"(?<![\w])(?:[abc]\s*=\s*)+" + _NUMBER + r"\s*" + _UNIT, "", sentence, flags=re.I
            )
            spacing = _DISTANCE_LABEL.search(sentence)
            error = _UNCERTAINTY_LABEL.search(sentence)
            bare = _QUANTITY.fullmatch(sentence.strip(" •-~≈"))
            for match in _QUANTITY.finditer(sentence):
                if not spacing and not error and not bare:
                    continue
                error_value = bool(re.search(r"(?:±|\+/-)\s*$", sentence[: match.start()])) or bool(
                    error and not spacing
                )
                for val in re.findall(_NUMBER, match["values"]):
                    yield val, match["unit"], error_value
                if match["uncertainty"] is not None:
                    yield match["uncertainty"], match["unit"], True


def matches_displayed(number, scale, expected):
    """Allow rounding to the displayed decimal place, including scientific notation."""
    value = Decimal(number.replace(",", "."))
    if not (-300 <= value.as_tuple().exponent <= 300):
        return False
    shown = float(value) * scale
    half_place = 0.5 * 10.0 ** value.as_tuple().exponent * scale
    return (
        math.isfinite(shown)
        and math.isfinite(half_place)
        and abs(shown - expected) <= half_place + 4 * math.ulp(expected)
    )


_PIXELS = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
_COVERAGE_NUMBER = re.compile(
    rf"(?P<valid>{_PIXELS})\s*(?:/|／|\bof\b|\bout of\b)\s*(?P<total>{_PIXELS})"
    r"|(?P<percent>[+-]?\d+(?:\.\d+)?)\s*[%％]|(?P<fraction>(?<![\d.])[01]\.\d+)(?![\w.%％])",
    re.I,
)
_COVERAGE_LABEL = (
    r"(?:(?<![非不])有效(?:像素|像元)(?:数(?:量)?|覆盖率|占比|比例|百分比|率)?"
    r"|(?<![非不])有效(?:区域?|掩膜)?(?:覆盖率?|占比|比例)|(?:质量|有效)掩膜覆盖(?:率|全图)?"
    r"|(?:全图|整幅图像)(?:的)?(?:掩膜)?覆盖率"
    r"|(?:全图|整幅图像)(?:所有|全部)?像素(?:均|全部|都)有效"
    r"|\bvalid[- ]pixels?(?:\s+(?:fraction|coverage|percentage|count|ratio))?"
    r"|(?:quality[- ]mask|valid[- ]area)\s+coverage)"
)
_COVERAGE_AFTER_LABEL = re.compile(
    _COVERAGE_LABEL + r"(?:占全图的?)?[\s*_：（(≈=:]*(?:(?:约为|约占|约|为|是|占|的|共|总计"
    r"|is|was|of|about|approximately)"
    r"[\s*_≈=:]*)*$",
    re.I,
)
_COVERAGE_AFTER_VALUE = re.compile(
    r"^[\s*_]*(?:(?:of\s+)?(?:(?:all|the)\s+)*(?:image\s+)?pixels?\s+(?:are|were)\s+valid"
    r"|[个的]?(?:像素|像元)(?:为|是)?有效|个有效(?:像素|像元))",
    re.I,
)
_GPA_SCOPE = re.compile(
    r"(?P<image>全图|整幅(?:图像)?|整张图|整个图像|全场|原图所有|"
    r"\b(?:overall|whole[- ]image|full[- ](?:image|frame)|all\s+(?:image\s+)?pixels)\b)"
    r"|(?P<region>参考(?:区域?|框)|框[内中]|(?:最)?外圈|边缘|局部|选区|"
    r"\b(?:roi|reference (?:region|box)|local region|image edges?|outermost ring)\b)",
    re.I,
)


def gpa_coverage_statements(answer):
    """Recognize labeled whole-image percentages and valid/total pixel counts in the final text."""
    # A comma can separate an ROI, strain, or uncertainty measurement. Keep grouped integer digits.
    sentences = re.compile(r"[^。；;!！?？\n]+?(?=[。；;!！?？\n]|\.(?:\s|$)|$)")
    clauses = re.compile(r"(?:[^，,]|,(?=\d{3}(?!\d)))+")
    section = None
    for sentence in sentences.finditer(answer):
        heading = sentence[0].strip().strip("*").strip()
        if heading.startswith("#") or heading.endswith(("：", ":")):
            scopes = list(_GPA_SCOPE.finditer(heading))
            section = scopes[-1].lastgroup if scopes else None
        scope = section
        for clause in clauses.finditer(sentence[0]):
            words = clause[0]
            previous = None
            for number in _COVERAGE_NUMBER.finditer(words):
                prefix, suffix = words[: number.start()], words[number.end() :]
                regions = list(_GPA_SCOPE.finditer(prefix))
                active_scope = regions[-1].lastgroup if regions else scope
                # "valid pixel coverage 100% within the ROI" has a trailing scope.
                after = re.match(
                    r"[\s*_]*(?:(?:within|in|inside)\s+(?:the\s+)?|在|位于)", suffix, re.I
                )
                if after:
                    trailing = _GPA_SCOPE.match(suffix, after.end())
                    if trailing:
                        active_scope = trailing.lastgroup
                parenthetical = re.match(
                    r"[\s*_]*[（(]\s*(?:within\s+|in\s+)?(?:the\s+)?", suffix, re.I
                )
                if parenthetical:
                    trailing = _GPA_SCOPE.match(suffix, parenthetical.end())
                    if trailing and re.match(r"[\s内中]*[）)]", suffix[trailing.end() :]):
                        active_scope = trailing.lastgroup
                adjacent = previous is not None and re.fullmatch(
                    r"[\s*_]*(?:个?(?:像素|像元)|pixels?)?[\s*_]*(?:[（(]|即|=)[\s*_约≈]*",
                    words[previous : number.start()],
                    re.I,
                )
                if active_scope == "region" or not (
                    _COVERAGE_AFTER_LABEL.search(prefix)
                    or _COVERAGE_AFTER_VALUE.match(suffix)
                    or adjacent
                ):
                    previous = None
                    continue
                if number["fraction"] is not None and not re.search(
                    r"(?:占比|比例|\bfraction\b|\bratio\b)[\s*_：:=≈]*(?:约为|为|约|is|was|about)?"
                    r"[\s*_：:=≈]*$",
                    prefix,
                    re.I,
                ):
                    previous = None
                    continue
                observed(
                    not re.match(r"\s*(?:[-–—~～至到]|或|\bor\b)\s*\d", suffix, re.I),
                    "whole_image_gpa_coverage_range_not_resolved",
                )
                previous = number.end()
                offset = sentence.start() + clause.start()
                yield offset + number.start(), offset + number.end(), number, sentence[0]
            regions = list(_GPA_SCOPE.finditer(words))
            if regions:
                scope = regions[-1].lastgroup


class _LocalReportAnchors(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = set()
        self.has_base = False

    def handle_starttag(self, tag, attrs):
        href = dict(attrs).get("href")
        if tag == "base" and href:
            self.has_base = True
        if tag == "a" and href:
            self.hrefs.add(href)


def gpa_coverage_sources(run, cited):
    """Resolve a returned mask, GPA JSON or report to the specific registered masks it supports."""
    if run["status"] != "completed":
        return {}, {}
    artifacts, directory = run["artifacts"], run["_directory"]
    gpa_steps = {
        s["step_id"]
        for s in run["steps"]
        if s["operation_id"] == "strain.gpa" and s["status"] == "completed"
    }
    counts = {s["step_id"]: s for s in run["strain_validity"] if s["step_id"] in gpa_steps}
    masks, sources = {}, {}
    for aid, item in artifacts.items():
        step = item["step_id"]
        if item["role"] == "gpa.quality_mask" and item["execution_usable"] and step in counts:
            require(step not in masks, "ambiguous_registered_gpa_mask")
            masks[step] = aid
    evidence = {
        aid: {
            "run_id": run["run_id"],
            "step_id": step,
            "mask_sha256": artifacts[aid]["sha256"],
            "valid_pixels": counts[step]["valid_count"],
            "total_pixels": counts[step]["total_count"],
        }
        for step, aid in masks.items()
    }
    for ref in cited & artifacts.keys():
        item = artifacts[ref]
        step, role = item["step_id"], item["role"]
        path = directory / item["file"]
        if ref in evidence:
            ids = {ref}
        elif role == "gpa" and item["execution_usable"] and step in masks:
            mask_id = masks[step]
            pointer = unique_json(path.read_text("utf-8"))["provenance"].get("quality_mask", {})
            target = directory / artifacts[mask_id]["file"]
            ids = (
                {mask_id}
                if Path(pointer.get("uri") or "") in {target, Path(target.name)}
                else set()
            )
        elif role == "report_manifest":
            report = unique_json(path.read_text("utf-8"))
            require(report.get("run", {}).get("run_id") == run["run_id"], "gpa_report_run_mismatch")
            inventory = {a["artifact_id"]: a for a in report["artifacts"]}
            ids = set()
            for row in report.get("strain_results", {}).get("all", []):
                mask_id = (row.get("validity_mask") or {}).get("artifact_id")
                result = artifacts.get(row.get("artifact_id"), {})
                if (
                    mask_id in evidence
                    and row.get("operation_id") == "strain.gpa"
                    and row.get("step_id") == evidence[mask_id]["step_id"] == result.get("step_id")
                    and result.get("role") == "gpa"
                    and result.get("execution_usable")
                ):
                    a = inventory.get(mask_id, {})
                    require(
                        a.get("role") == "gpa.quality_mask"
                        and a.get("step_id") == row["step_id"]
                        and a.get("execution_usable")
                        and a.get("sha256") == evidence[mask_id]["mask_sha256"],
                        "gpa_report_mask_registration_mismatch",
                    )
                    ids.add(mask_id)
        elif role == "report_html":
            anchors = _LocalReportAnchors()
            anchors.feed(path.read_text("utf-8"))
            anchors.close()
            ids = {
                i
                for i in evidence
                if not anchors.has_base
                and (directory / artifacts[i]["file"]).parent == path.parent
                and Path(artifacts[i]["file"]).name in anchors.hrefs
            }
        else:
            continue
        sources[ref] = ids
    return evidence, sources


_REPEATED_MASK_LINK = re.compile(
    r"\[(?P<label>首次|第一次|初次|复算|重复|第二次|first|original|repeat(?:ed)?|second)"
    r"\s*(?:GPA\s*)?(?:质量|有效性?)?(?:掩膜|mask)\]\(\s*<?(?P<path>[^<>)\n]+)>?\s*\)",
    re.I,
)


def adjacent_gpa_masks(blocks, position, runs, evidence, cited, compared):
    """Resolve a joint result to two named mask links in the immediately following paragraph."""
    pair = next(
        ((a[0], b[0]) for a, b in pairwise(blocks) if a.start() <= position < a.end()),
        None,
    )
    if not pair:
        return ()
    result, citations = pair
    if not re.search(
        r"(?:两次|两组|两张|两个)(?:的|得到的)?(?:质量|有效性?)?掩膜(?:完全|逐像素)?(?:一致|相同)"
        r"|\bboth (?:quality )?masks (?:are|were) (?:identical|the same)\b",
        result,
        re.I,
    ):
        return ()
    if re.search(
        r"^\s*#{1,6}\s|(?:另[一份个次张]*|不同|其他)(?:样本|样品|运行|图像)"
        r"|\b(?:another|different|separate)\s+(?:sample|image|run)\b",
        citations,
        re.I | re.M,
    ):
        return ()
    links = list(_REPEATED_MASK_LINK.finditer(citations))
    if len(links) != 2:
        return ()
    by_label = {}
    for link in links:
        path = Path(link["path"].strip())
        aid = path.stem
        if aid not in cited or aid not in evidence:
            return ()
        run = runs[evidence[aid]["run_id"]]
        require(
            path == run["_directory"] / run["artifacts"][aid]["file"], "gpa_mask_link_path_mismatch"
        )
        label = link["label"].lower()
        which = "first" if label in {"首次", "第一次", "初次", "first", "original"} else "repeat"
        by_label[which] = aid
    if set(by_label) != {"first", "repeat"}:
        return ()
    first, repeat = by_label["first"], by_label["repeat"]
    require(first != repeat, "gpa_repeat_requires_distinct_registered_masks")
    run_id = evidence[first]["run_id"]
    mentioned = set(_ID.findall(result + "\n" + citations))
    require(
        evidence[repeat]["run_id"] == run_id
        and mentioned.intersection(runs) <= {run_id}
        and not any(
            mentioned.intersection(r["artifacts"]) for rid, r in runs.items() if rid != run_id
        ),
        "gpa_coverage_citation_mixes_runs",
    )
    observed(
        mentioned.intersection(evidence) <= {first, repeat}, "gpa_coverage_claim_has_ambiguous_mask"
    )
    run = runs[run_id]
    steps = [evidence[i]["step_id"] for i in (first, repeat)]
    recorded = [s for s in run["_request"]["steps"] if s["id"] in steps]
    images = {
        s["step_id"]: run["artifacts"][s["image_artifact_id"]]["sha256"]
        for s in run["strain_validity"]
        if s["step_id"] in steps
    }
    require(
        all((run_id, step) in compared for step in steps)
        and [s["id"] for s in recorded] == steps
        and images[steps[0]] == images[steps[1]]
        and recorded[0]["parameters"] == recorded[1]["parameters"],
        "gpa_masks_not_first_and_repeat_of_same_parameters",
    )
    require(
        evidence[first]["mask_sha256"] == evidence[repeat]["mask_sha256"],
        "answer_claims_identical_gpa_masks_with_different_content",
    )
    return first, repeat


def one_cited_gpa_mask(covering, mentioned, returned, cited, sources, evidence):
    observed(covering, "gpa_coverage_claim_has_no_linked_citation")
    size = min(len(c["quote"]) for c in covering)
    refs = {i for c in covering if len(c["quote"]) == size for i in c["artifact_ids"]} & cited
    inline = mentioned & returned
    if inline:
        refs &= inline
    elif len(evidence) > 1:
        observed(
            sum(m["run_id"] in mentioned for m in evidence.values()) == 1,
            "gpa_coverage_claim_has_no_local_mask_reference",
        )
    resolved = [sources[i] for i in refs if i in sources]
    observed(resolved and all(resolved), "gpa_coverage_claim_has_no_linked_mask")
    ids = set.intersection(*resolved)
    require(ids, "gpa_coverage_citations_disagree")
    observed(len(ids) == 1, "gpa_coverage_claim_has_ambiguous_mask")
    return next(iter(ids)), refs


def check_gpa_coverage_answer(final, claims, runs, returned, compared):
    """Check each whole-image claim against its cited GPA mask or identical pair."""
    statements = list(gpa_coverage_statements(final))
    observed(statements, "whole_image_gpa_coverage_not_observed")
    evidence, sources = {}, {}
    cited = {i for c in claims for i in c["artifact_ids"]} & returned
    for run in runs.values():
        measurements, linked = gpa_coverage_sources(run, cited)
        evidence.update(measurements)
        sources.update(linked)
    quantities, used = [], set()
    blocks = list(re.finditer(r"\S.*?(?=\r?\n[ \t]*\r?\n|\Z)", final, re.S))
    for start, end, number, sentence in statements:
        covering = []
        for claim in claims:
            for occurrence in re.finditer(re.escape(claim["quote"]), final):
                if occurrence.start() <= start and end <= occurrence.end():
                    covering.append(claim)
                    break
        mentioned = set(_ID.findall(sentence))
        pair = ()
        if not mentioned.intersection(sources):
            pair = adjacent_gpa_masks(blocks, start, runs, evidence, cited, compared)
        if pair:
            mask_id, refs = pair[0], set(pair)
            same_run = runs[evidence[mask_id]["run_id"]]["artifacts"]
            require(
                all(
                    i in same_run and (i not in sources or sources[i] & refs)
                    for c in covering
                    for i in c["artifact_ids"]
                ),
                "gpa_coverage_citations_disagree",
            )
        else:
            mask_id, refs = one_cited_gpa_mask(
                covering, mentioned, returned, cited, sources, evidence
            )
        measured = evidence[mask_id]
        run_id, step = measured["run_id"], measured["step_id"]
        require(
            refs <= runs[run_id]["artifacts"].keys() and mentioned.intersection(runs) <= {run_id},
            "gpa_coverage_citation_mixes_runs",
        )
        require((run_id, step) in compared, "gpa_coverage_step_was_not_compared")
        valid, total = measured["valid_pixels"], measured["total_pixels"]
        if number["percent"] is not None:
            ok = matches_displayed(number["percent"], 1, 100 * valid / total)
        elif number["fraction"] is not None:
            ok = matches_displayed(number["fraction"], 1, valid / total)
        else:
            ok = (int(number["valid"].replace(",", "")), int(number["total"].replace(",", ""))) == (
                valid,
                total,
            )
        require(ok, "answer_gpa_coverage_differs_from_registered_mask")
        item = {
            "as_displayed": number[0],
            "cited_artifacts": sorted(refs),
            "mask_artifact_id": mask_id,
        }
        if pair:
            item["first_and_repeat_mask_ids"] = list(pair)
        quantities.append(item)
        used.update(pair or [mask_id])
    return {
        "masks": {i: evidence[i] for i in sorted(used)},
        "quantities": quantities,
    }


def spacing_sources(run, cited):
    """Map cited JSON/table/report IDs to hash-checked d-spacing rows in this completed FFT run."""
    artifacts = run["artifacts"]
    ops = {s["step_id"]: s["operation_id"] for s in run["steps"] if s["status"] == "completed"}
    if (
        run["status"] != "completed"
        or run["modality"] != "HRTEM"
        or not OPERATIONS["J01"] <= set(ops.values())
    ):
        return {}
    physical = {m["artifact_id"] for m in run["physical_measurements"]}
    physical_steps = {artifacts[i]["step_id"] for i in physical}
    rows = {}
    for aid, a in artifacts.items():
        step = a["step_id"]
        role = a["role"]
        if ops.get(step) not in {"diffraction.metrics", "diffraction.index"}:
            continue
        path = run["_directory"] / a["file"]
        if role == "diffraction":
            result = unique_json(path.read_text("utf-8"))
            data = result["d_spacings"] if positive_distance_units(role, result) else []
        elif role in {"d_spacing_table", "g_vector_table"} and step in physical_steps:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                data = list(csv.DictReader(stream))
            if role == "g_vector_table":
                # The CSV serializes the measured magnitude as a Python measurement mapping.
                # Convert 1/length to length, without transferring position-only uncertainties.
                try:
                    data = [ast.literal_eval(row["magnitude"]) for row in data]
                except (SyntaxError, ValueError) as exc:
                    raise EvidenceError("invalid_registered_reciprocal_measurement") from exc
                data = [
                    {"value": 1 / row["value"], "unit": row["unit"][2:], "uncertainty": None}
                    for row in data
                    if row["unit"].startswith("1/") and row["value"] > 0
                ]
        else:
            continue
        values = []
        for row in data:
            scale = length_scale(row["unit"])
            if scale is not None and row["value"] not in (None, ""):
                val = float(row["value"]) * scale
                unc = row["uncertainty"]
                if math.isfinite(val) and val > 0:
                    values.append((val, None if unc in (None, "") else float(unc) * scale))
        if values:
            rows[aid] = values
    sources = {aid: {aid: rows[aid]} for aid in cited & rows.keys()}
    for aid in cited & artifacts.keys():
        a = artifacts[aid]
        if a["role"] not in {"report_manifest", "report_html"}:
            continue
        path = run["_directory"] / a["file"]
        if a["role"] == "report_manifest":
            report = unique_json(path.read_text("utf-8"))
            ids = {
                x["artifact_id"]
                for x in report["artifacts"]
                if x["artifact_id"] in rows
                and x["sha256"] == artifacts[x["artifact_id"]]["sha256"]
                and x["execution_usable"]
            }
        else:
            hrefs = set(re.findall(r"""href=["']([^"']+)["']""", path.read_text("utf-8"), re.I))
            ids = {i for i in rows if Path(artifacts[i]["file"]).name in hrefs}
        if ids:
            sources[aid] = {i: rows[i] for i in sorted(ids)}
    return sources


def check_spacing_answer(final, claims, runs, returned):
    """Check a quoted d-spacing against its sources without judging scientific accuracy."""
    known = set(runs) | {j["job_id"] for r in runs.values() for j in r["jobs"]}
    require(
        set(_ID.findall(final)) <= known | returned, "answer_mentions_unreturned_artifact_or_run"
    )
    cited = {aid for c in claims for aid in c["artifact_ids"]}
    sources = {}
    evidence = {}
    for run in runs.values():
        for aid, raw in spacing_sources(run, cited).items():
            sources[aid] = raw
            evidence[aid] = {
                "run_id": run["run_id"],
                "citation_sha256": run["artifacts"][aid]["sha256"],
                "measurements": {i: run["artifacts"][i]["sha256"] for i in raw},
            }
    statements = list(spacing_statements(final))
    observed(any(not unc for _, _, unc in statements), "spacing_value_and_unit_not_observed")
    require(sources, "spacing_source_not_cited")
    eligible = [
        r for r in runs.values() if r["run_id"] in {item["run_id"] for item in evidence.values()}
    ]
    measurement_products = {
        i for r in eligible for i, a in r["artifacts"].items() if a["execution_usable"]
    } | sources.keys()
    for line in final.splitlines():
        refs = set(_ID.findall(line)) & returned
        if refs & measurement_products:
            require(
                any(refs <= set(run["artifacts"]) for run in eligible),
                "spacing_citation_mixes_unrelated_runs",
            )
            if refs & sources.keys():
                match_spacing_statements(
                    spacing_statements(line), [sources[i] for i in refs if i in sources]
                )
    for claim in claims:
        refs = set(claim["artifact_ids"])
        # A declaration may cite supporting plots as well as the quantitative table.
        if any(not unc for _, _, unc in spacing_statements(claim["quote"])):
            require(sources.keys() & refs, "spacing_claim_has_no_quantitative_source")
            match_spacing_statements(
                spacing_statements(claim["quote"]), [sources[i] for i in refs if i in sources]
            )
    match_spacing_statements(statements, sources.values())
    return {
        "source_artifacts": evidence,
        "quantities": [
            {"value_as_displayed": value, "unit": unit, "is_uncertainty": unc}
            for value, unit, unc in statements
        ],
    }


def match_spacing_statements(statements, sources):
    values = [row for source in sources for rows in source.values() for row in rows]
    last = values
    for number, unit, uncertainty in statements:
        scale = length_scale(unit)
        if uncertainty:
            require(
                any(
                    err is not None
                    and float(number.replace(",", ".")) >= 0
                    and matches_displayed(number, scale, err)
                    for _, err in last
                ),
                "reported_spacing_uncertainty_mismatch",
            )
        else:
            last = [
                pair
                for pair in values
                if float(number.replace(",", ".")) > 0 and matches_displayed(number, scale, pair[0])
            ]
            require(last, "reported_spacing_value_or_unit_mismatch")


def gpa_submission(turn, runs):
    ids = {
        c["response"]["result"]["run_id"]
        for c in turn["calls"]
        if c["tool"] == "submit_analysis" and c["response"]["ok"]
    }
    require(len(ids) == 1, "gpa_turn_requires_one_submitted_run")
    return runs[ids.pop()]


def gpa_question(turn, run, name):
    require(
        run["status"] == "needs_input" and run["error_code"] == "NEEDS_INPUT", "gpa_not_waiting"
    )
    questions = [
        (qid, q)
        for qid, q in run["_questions"].items()
        if q["operation_id"] == "strain.gpa" and [f["name"] for f in q["fields"]] == [name]
    ]
    require(len(questions) == 1, "gpa_question_does_not_match_run")
    qid, question = questions[0]
    returned = [
        c["response"]["result"]
        for c in turn["calls"]
        if c["tool"] in {"submit_analysis", "get_run_status"} and c["response"]["ok"]
    ]
    observed(
        any(
            data.get("run_id") == run["run_id"]
            and data.get("status") == "needs_input"
            and any(
                q["request_id"] == qid
                and q["operation_id"] == "strain.gpa"
                and q["step_id"] == question["step_id"]
                and q["fields"] == question["fields"]
                and q["evidence_artifact_ids"] == question["evidence"]
                for q in data.get("input_requests", [])
            )
            for data in returned
        ),
        "gpa_question_not_observed_in_its_turn",
    )
    roles = {"g_peak_selection", "numbered_g_peaks"} if name == "g_vectors" else {"raw_preview"}
    candidates = {
        i
        for i in question["evidence"]
        if run["artifacts"][i]["role"] in roles
        and (name != "g_vectors" or run["artifacts"][i]["step_id"] == question["step_id"])
    }
    shown = set()
    for c in turn["calls"]:
        if not c["response"]["ok"]:
            continue
        data = c["response"]["result"]
        if c["tool"] == "read_artifact":
            entries = [data["artifact"]]
        elif c["tool"] == "list_artifacts" and c["arguments"]["run_id"] == run["run_id"]:
            entries = data
        else:
            continue
        for a in entries:
            i = a["artifact_id"]
            if i in candidates and all(a[k] == run["artifacts"][i][k] for k in ("role", "sha256")):
                shown.add(i)
    observed(shown, "gpa_prompt_evidence_not_returned_in_its_turn")
    return qid, question["step_id"], shown


def gpa_reply(turn, parent, child, question, names, excerpts):
    qid, step, shown = question
    require(
        child["parent_run_id"] == parent["run_id"]
        and child["parameter_diff_present"]
        and child["source_sha256"] == parent["source_sha256"]
        and child["_request"]["input_path"] == parent["_request"]["input_path"],
        "gpa_reply_parent_or_source_mismatch",
    )
    rows = [
        row
        for row in child["_replies"]
        if (row["parent_run_id"], row["child_run_id"], row["request_id"])
        == (parent["run_id"], child["run_id"], qid)
    ]
    require(len(rows) == 1, "gpa_reply_link_missing")
    row = rows[0]
    saved = unique_json(row["response"])
    adopted = unique_json(row["adopted_values"])
    answers = saved["answers"]
    require(
        unique_json(row["validation"])["status"] == "accepted"
        and len(answers) == 1
        and set(answers) <= names
        and adopted.keys() == answers.keys()
        and saved["request_id"] == qid
        and saved["source"] in {"user", "artifact"}
        and saved["basis"].strip()
        and shown.intersection(saved["evidence_artifact_ids"])
        and set(saved["evidence_artifact_ids"]) <= set(parent["question_evidence"][qid]),
        "gpa_reply_source_or_evidence_mismatch",
    )
    plans = [
        c["arguments"].get("plan")
        for c in turn["calls"]
        if c["tool"] == "submit_analysis"
        and c["response"]["ok"]
        and c["response"]["result"]["run_id"] == child["run_id"]
        and c["arguments"].get("parent_run_id") == parent["run_id"]
    ]
    require(
        any(
            isinstance(p, dict)
            and all(
                p.get("user_input", {}).get("response", {}).get(k) == saved.get(k)
                for k in (
                    "request_id",
                    "answers",
                    "source",
                    "basis",
                    "evidence_artifact_ids",
                    "raw_reply",
                )
            )
            for p in plans
        ),
        "gpa_observed_reply_differs_from_store",
    )
    quote = saved.get("raw_reply")
    if not quote:
        excerpt = excerpts.get(qid, {})
        require(excerpt.get("answers") == answers, "gpa_reply_excerpt_answers_mismatch")
        quote = excerpt.get("quote")
    observed(quote and quote in turn["prompt"], "gpa_reply_not_from_this_user_turn")
    applied = next(iter(adopted.values()))
    params = next(
        (
            s["parameters"]
            for s in child["_request"]["steps"]
            if s["id"] == step and s["operation_id"] == "strain.gpa"
        ),
        {},
    )
    require(
        applied["operation_id"] == "strain.gpa"
        and applied["step_id"] == step
        and applied["parameters"]
        and all(params.get(k) == v for k, v in applied["parameters"].items()),
        "gpa_reply_not_applied_to_child",
    )
    return {
        "request_id": qid,
        "parent_run_id": parent["run_id"],
        "child_run_id": child["run_id"],
        "answer": next(iter(answers)),
        "evidence_artifact_ids": sorted(shown),
        "reply_sha256": text_digest(quote),
    }


def check_gpa_followup(variant, turns, runs, excerpts):
    expected_turns = {"gpa_answered": 3, "gpa_unanswered": 1, "gpa_invalid_reply": 2}
    observed(len(turns) == expected_turns[variant], "gpa_followup_turn_count_mismatch")
    observed(
        len({t["session"] for t in turns}) == 1
        and len({t["trace_sha256"] for t in turns}) == len(turns),
        "gpa_requires_distinct_turns_in_one_session",
    )
    parent = gpa_submission(turns[0], runs)
    require(parent["parent_run_id"] is None, "gpa_first_submission_is_not_root")
    g_pair = gpa_question(turns[0], parent, "g_vectors")
    if variant != "gpa_answered":
        require(not parent["_children"], "reply_started_unexpected_child")
        if variant == "gpa_invalid_reply":
            rejected = []
            for c in turns[1]["calls"]:
                if c["tool"] not in {"validate_plan", "submit_analysis"}:
                    continue
                require(
                    not c["response"]["ok"] or c["tool"] != "submit_analysis",
                    "rejected_gpa_reply_submitted",
                )
                if c["response"]["ok"]:
                    continue
                args, error = c["arguments"], c["response"]["error"]
                plan = args.get("plan")
                reply = (
                    plan.get("user_input", {}).get("response", {}) if isinstance(plan, dict) else {}
                )
                answers = reply.get("answers", {})
                quote = reply.get("raw_reply") or excerpts.get(g_pair[0], {}).get("quote")
                if (
                    error["code"] == "INVALID_INPUT_RESPONSE"
                    and reply.get("request_id") == g_pair[0]
                    and plan.get("input_path") == parent["_request"]["input_path"]
                    and any(
                        s.get("id") == g_pair[1] and s.get("operation_id") == "strain.gpa"
                        for s in plan.get("steps", [])
                    )
                    and quote
                    and quote in turns[1]["prompt"]
                    and args.get("parent_run_id", parent["run_id"]) == parent["run_id"]
                    and (
                        reply.get("raw_reply")
                        or excerpts.get(g_pair[0], {}).get("answers") == answers
                    )
                    and reply.get("source") in {"user", "artifact"}
                    and reply.get("basis", "").strip()
                    and len(answers) == 1
                    and set(answers) <= {"g_vectors", "g_peak_indices"}
                    and error.get("details", {}).get("field") in {"g_vectors", "g_peak_indices"}
                    and g_pair[2].intersection(reply.get("evidence_artifact_ids", []))
                    and set(reply["evidence_artifact_ids"])
                    <= set(parent["question_evidence"][g_pair[0]])
                ):
                    rejected.append(error["code"])
            observed(rejected, "missing_linked_rejected_gpa_reply")
        return {
            "g_request_id": g_pair[0],
            "parent_run_id": parent["run_id"],
            "evidence_artifact_ids": sorted(g_pair[2]),
            "answered_edges": [],
        }
    middle = gpa_submission(turns[1], runs)
    first = gpa_reply(turns[1], parent, middle, g_pair, {"g_vectors", "g_peak_indices"}, excerpts)
    roi = gpa_question(turns[1], middle, "reference_region")
    require(g_pair[0] != roi[0] and g_pair[1] == roi[1], "gpa_questions_mixed")
    final = gpa_submission(turns[2], runs)
    second = gpa_reply(turns[2], middle, final, roi, {"reference_region"}, excerpts)
    middle_params = next(s["parameters"] for s in middle["_request"]["steps"] if s["id"] == roi[1])
    final_params = next(s["parameters"] for s in final["_request"]["steps"] if s["id"] == roi[1])
    require(
        all(middle_params[k] == final_params[k] for k in ("g_vectors", "g_vector_unit")),
        "gpa_selected_vectors_changed_at_roi_reply",
    )
    roles = {
        a["role"]
        for a in final["artifacts"].values()
        if a["execution_usable"] and a["step_id"] == roi[1]
    }
    require(
        final["status"] == "completed"
        and ROLES["J03"] <= roles
        and any(s["step_id"] == roi[1] and s["valid_count"] > 0 for s in final["strain_validity"]),
        "completed_gpa_fields_missing",
    )
    observed(
        any(
            c["tool"] == "export_report"
            and c["response"]["ok"]
            and c["response"]["result"]["run_id"] == final["run_id"]
            for c in turns[2]["calls"]
        ),
        "final_gpa_report_not_returned",
    )
    return {
        "g_request_id": g_pair[0],
        "roi_request_id": roi[0],
        "parent_run_id": parent["run_id"],
        "final_run_id": final["run_id"],
        "answered_edges": [first, second],
    }


def compared_gpa_steps(measurements, completed):
    """Match the card's comparison to the measured image and normalized plan parameters."""

    def outputs(runs, operation, role):
        for run in runs:
            parameters = {step["id"]: step["parameters"] for step in run["_request"]["steps"]}
            for step in run["steps"]:
                if step["operation_id"] != operation or step["status"] != "completed":
                    continue
                for artifact in run["artifacts"].values():
                    if (
                        artifact["role"] == role
                        and artifact["execution_usable"]
                        and artifact.get("step_id") == step["step_id"]
                    ):
                        yield run, step, parameters[step["step_id"]], artifact

    def image_hash(run, step, config):
        image_id = step["consumed_artifacts"].get(config.get("image_role", "raw"))
        return run["artifacts"].get(image_id, {}).get("sha256")

    measured = [
        (
            (run["run_id"], step["step_id"]),
            run["source_sha256"],
            image_hash(run, step, config),
            config,
        )
        for run, step, config, _ in outputs(measurements, "strain.gpa", "gpa")
    ]
    matched = set()
    for run, step, parameters, artifact in outputs(
        completed, "strain.gpa_compare", "gpa_comparison"
    ):
        config = parameters["reference_config"]
        image = image_hash(run, step, config)
        matching = {
            location
            for location, source, raw, plan in measured
            if (source, raw, plan) == (run["source_sha256"], image, config)
        }
        if not image or config != parameters["candidate_config"] or not matching:
            continue
        path = run["_directory"] / artifact["file"]
        require(digest(path) == artifact["sha256"], "gpa_comparison_artifact_changed")
        result = unique_json(path.read_text("utf-8"))
        overlap = result.get("quality", {}).get("intersection_fraction")
        diagnosis = result.get("diagnosis", {})
        error = diagnosis.get("max_strain_mae_percent")
        effective = result.get("config_a")
        # The service can resolve an omitted physical pixel size from image calibration.
        compared = {
            key: value
            for key, value in config.items()
            if key not in {"image_role", "absolute_reference", "reference_regions"}
            and not (key == "pixel_size_nm" and value is None)
        }
        if (
            isinstance(effective, dict)
            and effective == result.get("config_b")
            and all(key in effective and effective[key] == value for key, value in compared.items())
            and type(overlap) in (int, float)
            and 0 < overlap <= 1
            and diagnosis.get("configs_aligned") is True
            and type(error) in (int, float)
            and error == 0
        ):
            matched.update(matching)
    return matched


def has_raw_refined_atomic_measurements(run):
    """Trace the reported geometry and intensity to the same raw-image refinement."""
    successful = {s["step_id"]: s for s in run["steps"] if s["status"] == "completed"}

    def outputs(operation, role):
        for artifact_id, artifact in run["artifacts"].items():
            step = successful.get(artifact["step_id"])
            if artifact["role"] == role and artifact["execution_usable"] and step:
                if step["operation_id"] == operation:
                    yield artifact_id, artifact, step["consumed_artifacts"]

    detected = {aid for aid, _, _ in outputs("atomic.detect", "atoms")}
    physical = {m["artifact_id"] for m in run["physical_measurements"] if m["units"]}
    measured = {
        used.get("refined_atoms")
        for aid, _, used in outputs("atomic.distances", "atom_distances")
        if aid in physical
    }
    intensity = {
        (used.get("refined_atoms"), used.get("raw"))
        for _, _, used in outputs("atomic.intensity", "atomic_intensity")
    }
    for aid, artifact, used in outputs("atomic.refine", "refined_atoms"):
        raw_id = used.get("raw")
        raw = run["artifacts"].get(raw_id, {})
        if (
            aid not in measured
            or (aid, raw_id) not in intensity
            or used.get("atoms") not in detected
            or raw.get("role") != "raw"
            or not raw.get("execution_usable")
        ):
            continue
        lattice = unique_json((run["_directory"] / artifact["file"]).read_text("utf-8"))
        atoms = lattice.get("atoms", [])
        status = lattice.get("provenance", {}).get("raw_refinement", {}).get("point_status")
        if (
            atoms
            and isinstance(status, list)
            and len(status) == len(atoms)
            and all(
                s in ("accepted", "out_of_bounds", "max_shift_rejected", "fit_failed")
                for s in status
            )
        ):
            return True
    return False


def check_case(case, calls, runs, turns, excerpts, variant=None):
    completed = [r for r in runs.values() if r["status"] == "completed"]
    if variant is not None:
        require(variant in case_variants(case), "unknown_case_variant")
    if case == "J08" and variant in GPA_VARIANTS:
        return check_gpa_followup(variant, turns, runs, excerpts)
    if (
        case == "J08"
        and variant == "answered"
        and len(turns) == 3
        and {"g_vectors", "reference_region"}
        <= {
            f["name"]
            for run in runs.values()
            for q in run.get("_questions", {}).values()
            if q["operation_id"] == "strain.gpa"
            for f in q["fields"]
        }
    ):
        return check_gpa_followup("gpa_answered", turns, runs, excerpts)
    if case in OPERATIONS and not (case == "J15" and variant not in {None, "executed"}):
        candidates = [
            r
            for r in completed
            if OPERATIONS[case]
            <= {s["operation_id"] for s in r["steps"] if s["status"] == "completed"}
            and ROLES[case] <= {a["role"] for a in r["artifacts"].values() if a["execution_usable"]}
        ]
        require(candidates, "requested_operations_or_results_missing")
        if case == "J03":
            compared = compared_gpa_steps(candidates, completed)
            require(compared, "same_parameter_gpa_comparison_missing")
            return compared
        modalities = {"J01": {"HRTEM"}, "J02": {"HAADF-STEM"}, "J05": {"SAED", "NBED"}}
        if case in modalities:
            require(
                any(r["modality"] in modalities[case] for r in candidates),
                "requested_modality_not_observed",
            )
        if case in {"J01", "J05"}:
            require(any(r["physical_measurements"] for r in candidates), "physical_spacing_missing")
        if case == "J02":
            require(
                any(
                    has_raw_refined_atomic_measurements(r)
                    for r in candidates
                    if r["modality"] == "HAADF-STEM"
                ),
                "raw_refined_physical_geometry_or_intensity_missing",
            )
        if case == "J05":
            if variant:
                require(
                    any(r["modality"].lower() == variant for r in candidates),
                    "direct_pattern_modality_mismatch",
                )
            require(
                all(
                    "diffraction.fft" not in {s["operation_id"] for s in r["steps"]}
                    for r in candidates
                ),
                "direct_pattern_was_transformed",
            )
    codes = {c["response"]["error"]["code"] for c in calls if not c["response"]["ok"]}
    failed_codes = codes | {r["error_code"] for r in runs.values() if r["error_code"]}
    if case == "J08":
        field_names = {"g_vectors", "reference_region", "reference_atom_indices", "base_index"}
        parents = [
            r
            for r in runs.values()
            if r["status"] == "needs_input"
            and any(set(names) & field_names for names in r["requested_fields"].values())
        ]
        require(parents, "missing_reference_request")
        shown = {
            a["artifact_id"]
            for c in calls
            if c["response"]["ok"]
            for a in (
                c["response"]["result"]
                if c["tool"] == "list_artifacts"
                else [c["response"]["result"]["artifact"]]
                if c["tool"] == "read_artifact"
                else c["response"]["result"]["artifacts"]
                if c["tool"] == "inspect_input"
                else []
            )
        }
        # Asking from a numbered graphic, peak/coordinate table or original-frame preview
        # must leave an artifact the host actually received from the parent.
        require(
            any(
                i in shown and r["artifacts"][i]["kind"] in {"image", "table"}
                for r in parents
                for ids in r["question_evidence"].values()
                for i in ids
            ),
            "reference_evidence_not_returned_to_host",
        )
        if variant in {"unanswered", "invalid_reply"}:
            require(all(not r["_children"] for r in parents), "reply_started_unexpected_child")
            if variant == "unanswered":
                require(
                    len(turns) == 1
                    and sum(
                        c["tool"] in {"inspect_input", "submit_analysis"} and c["response"]["ok"]
                        for c in calls
                    )
                    <= 2,
                    "repeated_unanswered_submission",
                )
            else:
                observed(len(turns) >= 2, "missing_actual_followup_user_turn")
                require(
                    any(
                        c["tool"] in {"validate_plan", "submit_analysis"}
                        and not c["response"]["ok"]
                        and c["response"]["error"]["code"]
                        in {
                            "INVALID_INPUT_RESPONSE",
                            "CROSS_RUN_ARTIFACT",
                            "INPUT_REQUEST_PARENT_MISMATCH",
                            "INVALID_PARAMETERS",
                            "CREDENTIAL_PARAMETER_FORBIDDEN",
                        }
                        for t in turns[1:]
                        for c in t["calls"]
                    ),
                    "missing_rejected_reply",
                )
            return
    if case in {"J11", "J15"} and variant in {
        "invalid_file",
        "missing_dependency",
        "broken_dependency",
        "numerical_failure",
        "timeout",
    }:
        expected = {
            "invalid_file": {"INVALID_INPUT", "INPUT_NOT_FOUND", "UNSUPPORTED_INPUT"},
            "missing_dependency": {
                "BACKEND_UNAVAILABLE",
                "MISSING_DEPENDENCY",
                "DEPENDENCY_UNAVAILABLE",
            },
            "broken_dependency": {
                "BACKEND_IMPORT_FAILED",
                "DEPENDENCY_BROKEN",
                "BROKEN_DEPENDENCY",
            },
            "numerical_failure": {"NUMERICAL_FAILURE", "INVALID_METRICS"},
            "timeout": {"TIMEOUT", "SIMULATION_TIMEOUT"},
        }[variant]
        require(failed_codes & expected, "fault_variant_not_observed")
        return
    if case in {"J06", "J07", "J08", "J09"}:
        observed(len(turns) >= 2, "missing_actual_followup_user_turn")
        parents = {
            c["response"]["result"]["run_id"]
            for c in turns[0]["calls"]
            if c["response"]["ok"]
            and isinstance(c["response"]["result"], dict)
            and c["response"]["result"].get("status")
            == ("completed" if case == "J09" else "needs_input")
        }
        children = [r for r in completed if r["parent_run_id"] in parents]
        require(children, "missing_completed_child_of_observed_parent")
        for child in children:
            require(
                child["source_sha256"] == runs[child["parent_run_id"]]["source_sha256"]
                and child["parameter_diff_present"],
                "child_source_or_diff_mismatch",
            )
            if case == "J09":
                parent = runs[child["parent_run_id"]]
                first = turns[0]["calls"]
                before = [
                    a
                    for c in first
                    if c["tool"] == "list_artifacts"
                    and c["response"]["ok"]
                    and c["arguments"]["run_id"] == parent["run_id"]
                    for a in c["response"]["result"]
                ]
                before_hashes = [
                    e["payload"]["source_hash"]
                    for c in first
                    if c["response"]["ok"]
                    and isinstance(c["response"]["result"], dict)
                    and c["response"]["result"].get("run_id") == parent["run_id"]
                    for e in c["response"]["result"].get("events", [])
                    if e["kind"] == "submitted"
                ]
                observed(before and before_hashes, "parent_pre_revision_evidence_missing")
                require(
                    set(before_hashes) == {parent["source_sha256"]}
                    and all(
                        a["artifact_id"] in parent["artifacts"]
                        and a["sha256"] == parent["artifacts"][a["artifact_id"]]["sha256"]
                        for a in before
                    ),
                    "pre_revision_artifact_changed",
                )
            if case == "J09" and variant:
                require(
                    child["_request"]["input_mode"] == variant
                    and runs[child["parent_run_id"]]["_request"]["input_mode"] == variant,
                    "revision_input_mode_mismatch",
                )
        if case != "J09":
            later = [
                c
                for t in turns[1:]
                for c in t["calls"]
                if c["tool"] == "submit_analysis" and c["response"]["ok"]
            ]
            matched = False
            for child in children:
                for row in child["_replies"]:
                    reply = unique_json(row["response"])
                    asked = set(
                        runs[child["parent_run_id"]]["requested_fields"].get(row["request_id"], [])
                    )
                    expected = {
                        "J06": {
                            "pixel_size_nm",
                            "diffraction_scale_inv_nm_per_px",
                            "physical_calibration",
                        },
                        "J07": {
                            "frame",
                            "selection_indices",
                            "dataset_path",
                            "dataset_index",
                            "channel",
                        },
                        "J08": {
                            "g_vectors",
                            "reference_region",
                            "reference_atom_indices",
                            "base_index",
                        },
                    }[case]
                    if not asked & expected:
                        continue
                    if (
                        case == "J07"
                        and variant
                        and not asked
                        & {
                            "dataset": {"dataset_index", "dataset_path"},
                            "frame": {"frame", "selection_indices"},
                            "channel": {"channel"},
                        }[variant]
                    ):
                        continue
                    if any(
                        c["response"]["result"].get("run_id") == child["run_id"]
                        and c["arguments"]["plan"]["user_input"]["response"]["request_id"]
                        == row["request_id"]
                        for c in later
                    ):
                        if reply.get("raw_reply"):
                            require(
                                any(reply["raw_reply"] in t["prompt"] for t in turns[1:]),
                                "stored_reply_not_from_followup",
                            )
                        else:
                            excerpt = excerpts.get(row["request_id"], {})
                            observed(
                                excerpt.get("quote")
                                and any(excerpt["quote"] in t["prompt"] for t in turns[1:])
                                and excerpt.get("answers") == reply["answers"],
                                "unverified_followup_excerpt",
                            )
                        if case == "J06":
                            require(
                                child["physical_measurements"],
                                "no_child_physical_measurement",
                            )
                        if case == "J07":
                            used = [
                                s["consumed_artifacts"]["raw"]
                                for s in child["steps"]
                                if "raw" in s["consumed_artifacts"]
                            ]
                            require(
                                len(used) >= 2
                                and len(set(used)) == 1
                                and child["artifacts"][used[0]]["role"] == "raw",
                                "no_shared_selected_raw_evidence",
                            )
                        matched = True
            require(matched, "missing_recorded_user_response")
    if case == "J10":
        require("DEFERRED_THIS_PHASE" in codes, "missing_reserved_operation_refusal")
        answer = turns[-1]["answer"]
        observed(
            re.search(
                r"deferred|reserved|not (?:yet )?(?:available|implemented)"
                r"|延期|未实现|预留|暂不支持",
                answer,
                re.I,
            ),
            "reserved_limit_not_visible_in_answer",
        )
        observed(
            re.search(r"FFT|GPA|diffraction|atomic|traditional|衍射|原子|传统", answer, re.I),
            "available_alternative_not_visible_in_answer",
        )
    if case == "J11":
        require(
            codes
            or any(r["status"] in {"timed_out", "failed", "interrupted"} for r in runs.values()),
            "missing_fault_evidence",
        )
    if case == "J12":
        if variant == "interrupted":
            require(
                any(
                    r["status"] == "interrupted" and r["error_code"] == "WORKER_INTERRUPTED"
                    for r in runs.values()
                ),
                "missing_worker_interruption",
            )
            return
        require(
            any(
                c["tool"] == "cancel_run"
                and c["response"]["ok"]
                and runs.get(c["arguments"]["run_id"], {}).get("status") == "cancelled"
                for c in calls
            ),
            "missing_owned_cancelled_run",
        )
        if variant in {"queued_cancel", "running_cancel"}:
            stage = variant.split("_")[0]
            states = {}
            observed_stage = False
            for c in calls:
                if not c["response"]["ok"]:
                    continue
                data = c["response"]["result"]
                if c["tool"] == "cancel_run":
                    observed_stage |= states.get(c["arguments"]["run_id"]) == stage
                elif c["tool"] in {"submit_analysis", "get_run_status"}:
                    states[data["run_id"]] = data["status"]
            require(observed_stage, "pre_cancel_state_not_observed")
    if case == "J13":
        submissions = [
            (c["arguments"].get("idempotency_key"), c["response"]["result"]["run_id"])
            for c in calls
            if c["tool"] == "submit_analysis" and c["response"]["ok"]
        ]
        require(
            any(k and submissions.count((k, r)) >= 2 for k, r in submissions)
            and "IDEMPOTENCY_CONFLICT" in codes,
            "missing_replay_or_conflict",
        )
    if case == "J14":
        require(
            any(
                {s["operation_id"].split(".")[0] for s in r["steps"] if s["status"] == "completed"}
                >= {"input", "diffraction", "atomic"}
                for r in completed
            ),
            "missing_shared_domain_journey",
        )


def native_turns(spec, base, turns):
    if spec.get("delegation_mode", "sequential") == "sequential":
        return []
    require(spec["delegation_mode"] == "native", "invalid_delegation_mode")
    children, roles = [], set()
    for entry in spec.get("native_children", []):
        profile = entry["profile"]
        require(profile in DOMAINS, "unknown_delegated_profile")
        child = read_turn(
            base, entry, spec["host"], spec["capture_format"], spec.get("server", "autotem")
        )
        observed(
            child["session"] != turns[0]["session"]
            and child["calls"]
            and any(c["tool"] == "submit_analysis" and c["response"]["ok"] for c in child["calls"]),
            "no_delegated_scientific_tool_return",
        )
        related = [
            event for t in turns for event in t["delegation_events"] if child["session"] in event
        ]
        observed(
            any("spawn" in event.lower() for event in related)
            and any(
                word in event.lower()
                for event in related
                for word in ("wait", "return", "complete_task")
            ),
            "native_task_or_return_not_linked_to_parent",
        )
        child["profile"] = profile
        roles.add(profile)
        children.append(child)
    observed(
        children
        and len({c["session"] for c in children}) == len(children)
        and (spec["case_id"] != "J14" or roles == DOMAINS),
        "incomplete_native_profile_traces",
    )
    return children


def check_tool_runs(calls, runs):
    validated_files = {}
    for call in calls:
        if not call["response"]["ok"]:
            continue
        args, tool, data = call["arguments"], call["tool"], call["response"]["result"]
        if tool == "validate_plan" and isinstance(args.get("plan"), str):
            validated_files[args["plan"]] = data["plan"]
        if tool in {"inspect_input", "submit_analysis"}:
            run = runs[data["run_id"]]
            plan = args.get("plan", {})
            if isinstance(plan, str):
                observed(plan in validated_files, "file_plan_contents_not_recorded")
                plan = validated_files[plan]
            require(
                run["_request"]["input_path"]
                == (args["path"] if tool == "inspect_input" else plan["input_path"]),
                "submitted_file_differs_from_run",
            )
            require(
                {s["id"]: s["operation_id"] for s in plan.get("steps", [])}.items()
                <= {s["step_id"]: s["operation_id"] for s in run["steps"]}.items(),
                "submitted_steps_differ_from_run",
            )
            if "parent_run_id" in args:
                require(args["parent_run_id"] == run["parent_run_id"], "submitted_parent_mismatch")
            require(
                any(j["job_id"] == data.get("job_id") for j in run["jobs"]), "host_job_mismatch"
            )
        if tool in {"get_run_status", "cancel_run", "list_artifacts", "export_report"}:
            require(args["run_id"] in runs, "run_not_created_in_captured_journey")
            if isinstance(data, dict):
                require(data.get("run_id") == args["run_id"], "host_result_run_mismatch")
        if tool == "get_run_status" and data["status"] in {
            "completed",
            "failed",
            "needs_input",
            "cancelled",
            "interrupted",
            "timed_out",
        }:
            if data.get("job_id") == runs[args["run_id"]]["job_id"]:
                require(
                    data["status"] == runs[args["run_id"]]["status"], "host_terminal_state_mismatch"
                )


def structured_observation(text):
    """Return JSON/JSONL records, or None for a plain native text export."""
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        return [unique_json(text)]
    except ValueError:
        try:
            return [unique_json(line) for line in text.splitlines() if line.strip()]
        except ValueError:
            return []


def inventory_rows(rows):
    if not isinstance(rows, list):
        return set()
    found = set()
    for item in rows:
        name = item.get("name") if isinstance(item, dict) else item
        if isinstance(name, str):
            name = name.removeprefix("mcp__autotem__")
            if name in TOOLS:
                found.add(name)
    return found


def plain_inventory(text):
    found = set()
    for line in text.splitlines():
        tokens = re.split(r"[\s,，|]+", line.strip(" \t-*•"))
        names = [token.strip(" \t\x60") for token in tokens]
        if names and all(n.removeprefix("mcp__autotem__") in TOOLS for n in names):
            found.update(inventory_rows(names))
    return TOOLS <= found


def has_inventory(text):
    records = structured_observation(text)
    if records is None:
        return plain_inventory(text)
    for event in records:
        if isinstance(event, list) and TOOLS <= inventory_rows(event):
            return True
        if not isinstance(event, dict):
            continue
        listing = event.get("result", event)
        if any(
            entry[key] != "autotem"
            for entry in (event, listing)
            if isinstance(entry, dict)
            for key in ("server", "serverName", "server_name")
            if key in entry
        ):
            continue
        if isinstance(listing, dict) and TOOLS <= inventory_rows(listing.get("tools")):
            return True
        if isinstance(listing, dict) and isinstance(listing.get("data"), list):
            # Codex ListMcpServerStatusResponse groups a tool map under each server.
            for server in listing["data"]:
                if isinstance(server, dict) and server.get("name") == "autotem":
                    tools = server.get("tools")
                    if isinstance(tools, dict) and TOOLS <= inventory_rows(list(tools.values())):
                        return True
        # Codex Code Mode records each returned value as a distinct text block.
        # A separately returned ALL_TOOLS array is evidence; names in Skill prose are not.
        reply = event.get("payload", {})
        if isinstance(reply, dict) and reply.get("type") in {
            "custom_tool_call_output",
            "function_call_output",
        }:
            blocks = reply.get("output")
            if isinstance(blocks, str):
                blocks = [{"text": blocks}]
            for block in blocks if isinstance(blocks, list) else ():
                value = block.get("text", "") if isinstance(block, dict) else ""
                if not isinstance(value, str):
                    continue
                inner = structured_observation(value)
                if (inner is None and plain_inventory(value)) or any(
                    isinstance(row, list) and TOOLS <= inventory_rows(row) for row in inner or ()
                ):
                    return True
    return False


def granted_text(text):
    # Consent must describe the user's decision about this server in the same
    # statement. "accepted" in a fit table and approved CLI instructions are unrelated.
    service = r"\bautotem\b"
    resource = (
        r"(?:mcp[ -]+)?(?:server|connector|tools?|access|permissions?|"
        r"connection|authorization|consent|request)"
    )
    action = r"(?:approved|accepted|allowed|granted|authorized|permitted)"
    separator = r"[\s:=,，：—–-]+"
    english = (
        rf"(?:user\s+)?{action}\s+(?:(?:the|access|permission|request|for|to|connect|use|with)\s+)*"
        rf"{service}(?:{separator}{resource})*|"
        rf"(?:{resource}\s+(?:to|for|on)\s+)?{service}(?:['’]s)?(?:{separator}{resource})*"
        rf"{separator}(?:(?:is|was|has been|user)\s+)?{action}(?:\s+by\s+(?:the\s+)?user)?|"
        rf"{resource}{separator}{action}\s+(?:to|for)\s+{service}(?:{separator}{resource})*"
    )
    chinese = (
        r"(?:用户|操作者)?(?:已经|已)(?:授权|允许|同意|批准)(?:访问|连接|调用|使用)?"
        r"autotem(?:的)?(?:mcp)?(?:服务|连接器|工具|权限|访问|调用|请求)*|"
        r"autotem(?:的)?(?:mcp)?(?:服务|连接器|工具|权限|访问|调用|请求)*[：:，,]?"
        r"(?:已经|已)(?:由用户)?(?:授权|允许|同意|批准)"
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    statements = re.split(r"[.!?;。！？；\r\n]", text)
    for header, body in pairwise(lines):
        if re.fullmatch(r"autotem(?:\s+(?:mcp|server|connector|工具|服务))*", header, re.I):
            statements.append(header + " " + body)
    granted = False
    for statement in statements:
        s = statement.strip(" \t-•")
        if re.search(r"(?<![\w-])autotem(?![a-z0-9_-])", s, re.I | re.ASCII) and re.search(
            r"\b(denied|revoked|declined|not\s+(?:allowed|approved|granted))\b|已拒绝|未授权|未允许",
            s,
            re.I,
        ):
            return False
        explicit_access = re.search(rf"\b(?:user|{resource})\b", s, re.I)
        if explicit_access and re.fullmatch(
            rf"(?:{english})(?:\s+(?:for this session|for this project|once))?", s, re.I
        ):
            granted = True
        if re.fullmatch(chinese, re.sub(r"\s+", "", s), re.I):
            granted = True
    return granted


def has_grant(text, sessions, host=None):
    records = structured_observation(text)
    if records is None:
        if host not in HOSTS:
            return False
        # A service form or an isolated caption can contain the same words.
        # Preserve a visible host permissions heading or a host-labelled log entry.
        label = re.escape(host)
        header = rf"(?:OpenAI\s+)?{label}(?:\s*[-:·：]?\s*(?:MCP|tool|工具|服务))?"
        heading = rf"{header}\s*(?:[-:·：]\s*)?(?:permissions?|authorization|approval|权限|授权)"
        if re.search(rf"(?im)^\s*{heading}\s*$", text):
            return granted_text(text)
        for line in text.splitlines():
            prefixed = re.fullmatch(rf"\s*{label}\s*[:：]\s*(.+)", line, re.I)
            if prefixed and granted_text(prefixed[1]):
                return True
        return False
    for event in records[0] if len(records) == 1 and isinstance(records[0], list) else records:
        # An MCP server owns its elicitation form and can choose the question,
        # fields and labels. Even a correlated "accept" is a response to that
        # server, not a grant from the hosting application's permission UI.
        if not isinstance(event, dict) or any(
            k in event for k in ("jsonrpc", "method", "result", "error")
        ):
            continue
        body = event.get("item") if event.get("type") == "item.completed" else event
        if event.get("type") == "event_msg":
            body = event.get("payload")
        if not isinstance(body, dict):
            continue
        tag = re.sub(r"[^a-z]", "", str(body.get("type") or body.get("event") or "").lower())
        if not re.fullmatch(
            r"(?:mcp)?(?:server|tool)?(?:permission|consent|authorization|approval)"
            r"(?:decision|decided|granted|approved|allowed|accepted|response)?",
            tag,
        ):
            continue
        origin = body.get("origin", event.get("origin"))
        actor = body.get("actor", event.get("actor"))
        if host not in HOSTS or origin != host or actor != "user":
            continue
        session = body.get("session_id") or body.get("threadId") or event.get("session_id")
        if not isinstance(session, str) or session not in sessions:
            continue
        server = body.get("server") or body.get("server_name") or body.get("serverName")
        decision = body.get("decision") or body.get("action") or body.get("status")
        if server == "autotem" and (
            decision
            in ("allow", "accept", "approved", "allowed", "granted", "accepted", "authorized")
            or decision is None
            and tag.endswith(("granted", "approved", "allowed", "accepted"))
        ):
            return True
    return False


WIN_OCR = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
} | Select-Object -First 1
function Await($operation, [Type]$type) {
    $task = $asTask.MakeGenericMethod($type).Invoke($null, @($operation))
    $task.GetAwaiter().GetResult()
}
$fileType = [Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
function WinType($space, $name) { [type]("$space.$name, $space, ContentType=WindowsRuntime") }
$streamType = WinType Windows.Storage.Streams IRandomAccessStreamWithContentType
$decoderType = WinType Windows.Graphics.Imaging BitmapDecoder
$bitmapType = WinType Windows.Graphics.Imaging SoftwareBitmap
$engineType = [Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$resultType = [Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime]
$file = Await ($fileType::GetFileFromPathAsync($env:AUTOTEM_HOST_OBSERVATION_IMAGE)) $fileType
$stream = Await ($file.OpenReadAsync()) $streamType
$decoder = Await ($decoderType::CreateAsync($stream)) $decoderType
$bitmap = Await ($decoder.GetSoftwareBitmapAsync()) $bitmapType
foreach ($language in $engineType::AvailableRecognizerLanguages) {
    $engine = $engineType::TryCreateFromLanguage($language)
    if ($engine) {
        $result = Await ($engine.RecognizeAsync($bitmap)) $resultType
        $result.Lines | ForEach-Object { $_.Text }
    }
}
"""


def screenshot_text(path):
    """Read a real screenshot locally. An unavailable recognizer leaves the proof pending."""
    try:
        from PIL import Image
    except ImportError:
        return None
    formats = ("PNG", "JPEG", "BMP", "WEBP", "GIF", "TIFF")
    try:
        with Image.open(path, formats=formats) as image:
            image.verify()
        with Image.open(path, formats=formats) as image:
            image.load()
    except (Image.DecompressionBombError, EOFError, OSError, SyntaxError, ValueError):
        return None
    return windows_ocr(path)


def windows_ocr(path):
    import os
    import subprocess

    if sys.platform != "win32":
        return ""
    powershell = (
        Path(os.environ.get("SystemRoot", r"C:\Windows"))
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    try:
        process = subprocess.run(
            [str(powershell), "-NoProfile", "-NonInteractive", "-Command", WIN_OCR],
            env={**os.environ, "AUTOTEM_HOST_OBSERVATION_IMAGE": str(path)},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return process.stdout.decode("utf-8-sig", errors="replace") if process.returncode == 0 else ""


def evidence_text(path, ref, name, evidence):
    # Text/structured exports already contain a readable original. A transcription
    # may help a reviewer, but must not replace the original when checking the claim.
    if path.suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }:
        try:
            text = path.read_text(encoding="utf-8-sig")
            if "\0" not in text:
                return text
        except UnicodeDecodeError:
            pass
    if not isinstance(ref, dict):
        return ""
    recognized = screenshot_text(path)
    if recognized is not None:
        evidence[name + "_image_decoded"] = True
        if recognized:
            evidence[name + "_recognized_text_sha256"] = text_digest(recognized)
        else:
            evidence[name + "_visual_review"] = "not_evaluated"
    return recognized or ""


def host_observations(spec, base, turns, calls, result):
    """Recognize effective access separately from first consent and full inventory."""
    evidence = result["evidence"]
    checks = result["checks"]
    # A completed host MCP exchange proves the names actually routed and effective
    # access. It does not reveal a previous consent dialog or other available tools.
    seen = {c["tool"] for c in calls}
    result["observed_tools"] = sorted(seen)
    checks.update(
        tool_discovery="passed",
        effective_tool_permission="passed",
        full_tool_inventory="not_evaluated",
        first_authorization="not_evaluated",
        session_skill_discovery="not_evaluated",
    )
    source = spec.get("host_observations", {})
    for name in ("discovery", "authorization", "skill_discovery"):
        if not source.get(name):
            continue
        ref = source[name]
        file = capture_file(base, ref if isinstance(ref, str) else ref["original_evidence"])
        evidence[name + "_sha256"] = digest(file)
        transcript = None
        sessions = {t["session"] for t in turns}
        if isinstance(ref, dict):
            require(
                ref.get("session_id") in sessions,
                "host_observation_session_mismatch",
            )
            transcript = ref.get("observed_text", "")
            observed(
                isinstance(transcript, str) and transcript.strip(),
                "missing_observation_transcription",
            )
            evidence[name + "_transcription_sha256"] = text_digest(transcript)
        if name in {"discovery", "authorization"}:
            text = evidence_text(file, ref, name, evidence)
            if (
                name == "discovery"
                and has_inventory(text)
                and (transcript is None or has_inventory(transcript))
            ):
                checks["full_tool_inventory"] = "passed"
            if (
                name == "authorization"
                and has_grant(text, sessions, spec.get("host"))
                and (
                    transcript is None
                    or granted_text(transcript)
                    or has_grant(transcript, sessions, spec.get("host"))
                )
            ):
                checks["first_authorization"] = "passed"
            continue
        text = transcript if transcript is not None else file.read_text("utf-8-sig")
        if (
            name == "skill_discovery"
            and "autotem" in text.lower()
            and re.search(
                r"skill.{0,140}(?:invoked|loaded|used|activated)|(?:invoked|loaded|using).{0,140}skill|已加载.{0,140}技能|已调用.{0,140}技能|使用了.{0,140}技能",
                text,
                re.I | re.S,
            )
        ):
            checks["session_skill_discovery"] = "passed"
    if any(t["skill_opened"] for t in turns):
        checks["session_skill_discovery"] = "passed"


def verify(spec_path, *, bundle=None, root=None, project=None):
    base = spec_path.resolve().parent
    spec = unique_json(spec_path.read_text("utf-8-sig"))
    require(
        spec["schema_version"] == "1.0.0" and spec["host"] in HOSTS and spec["case_id"] in CASES,
        "unsupported_capture",
    )
    result = {
        "schema_version": "1.0.0",
        "execution_layer": "real_host_capture",
        "host": spec["host"],
        "case_id": spec["case_id"],
        "case_verdict": "not_evaluated",
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
        "host_delegation_mode": spec.get("delegation_mode", "sequential"),
        "evidence": {
            "case_record_sha256": digest(spec_path),
            "verifier_sha256": digest(Path(__file__)),
            "matrix_sha256": digest(Path(__file__).with_name("matrix.json")),
        },
        "checks": {},
        "tool_trace": [],
        "runs": [],
        "claims": [],
        "checked_utc": datetime.now(timezone.utc).isoformat(),
    }
    checks = result["checks"]
    try:
        if "variant" in spec:
            require(spec["variant"] in case_variants(spec["case_id"]), "unknown_case_variant")
            result["variant"] = spec["variant"]
        if bundle:
            manifest = bundle / "release-manifest.json"
            observed(manifest.is_file(), "release_manifest_missing")
        if spec.get("activation_file"):
            observed(bundle is not None, "activation_release_manifest_missing")
            activation_file = capture_file(base, spec["activation_file"])
            activation = unique_json(activation_file.read_text("utf-8-sig"))
            attempted = digest(manifest)
            require(
                activation.get("manifest_sha256") == attempted
                and activation.get("runtime_id") == f"rt-{attempted[:20]}"
                and spec["host"] in activation.get("host_installations", {}),
                "attempted_project_release_mismatch",
            )
            result["attempted_release_manifest_sha256"] = attempted
            result["evidence"]["activation_sha256"] = digest(activation_file)
        result["capture_stage"] = spec.get(
            "capture_stage", "host_turn" if spec["observation"] == "recorded" else "not_observed"
        )
        require(
            result["capture_stage"]
            in {"not_observed", "local_discovery", "host_preflight", "host_turn"},
            "unknown_capture_stage",
        )
        observed(spec["observation"] != "not_evaluated", "journey_not_executed")
        if spec.get("started_at") and spec.get("ended_at"):
            start, end = stamp(spec["started_at"]), stamp(spec["ended_at"])
            require(0 < start <= end, "invalid_observation_window")
            result.update(started_at=spec["started_at"], ended_at=spec["ended_at"])
        if spec.get("version_file"):
            version_file = capture_file(base, spec["version_file"])
            version = spec.get("host_version", "")
            require(
                bool(re.fullmatch(r"[0-9][0-9a-zA-Z.\-+]{1,79}", version))
                and re.search(
                    r"(?<![\w.+-])[vV]?" + re.escape(version) + r"(?![\w.+-])",
                    version_file.read_text("utf-8-sig"),
                ),
                "host_version_evidence_mismatch",
            )
            result.update(host_version=version)
            result["evidence"]["host_version_sha256"] = digest(version_file)
        if spec["observation"] == "blocked":
            blocker = capture_file(base, spec.get("blocker_file"))
            code = spec.get("blocker_code", "")
            require(bool(re.fullmatch(r"[A-Z][A-Z0-9_]{1,79}", code)), "missing_blocker_code")
            result.update(case_verdict="blocked", reason=code)
            result["evidence"]["blocker_sha256"] = digest(blocker)
            if spec.get("blocked_prompt_file"):
                result["evidence"]["prompt_sha256"] = digest(
                    capture_file(base, spec["blocked_prompt_file"])
                )
            return result
        require(spec["observation"] == "recorded", "invalid_observation_type")
        observed("host_version" in result, "host_version_not_observed")
        observed("ended_at" in result, "missing_capture_window")
        require(result["capture_stage"] == "host_turn", "scientific_journey_requires_host_turn")
        turns = [
            read_turn(base, t, spec["host"], spec["capture_format"], spec.get("server", "autotem"))
            for t in spec["turns"]
        ]
        observed(turns and len({t["session"] for t in turns}) == 1, "missing_single_host_session")
        children = native_turns(spec, base, turns)
        calls = [c for t in turns + children for c in t["calls"]]
        observed(calls, "no_real_autotem_host_call")
        checks["host_tool_call"] = "passed"
        result["evidence"]["turns"] = [
            {
                "session_sha256": text_digest(t["session"]),
                "prompt_sha256": t["prompt_sha256"],
                "trace_sha256": t["trace_sha256"],
                "answer_sha256": text_digest(t["answer"]),
                "attachments": t["attachments"],
            }
            for t in turns + children
        ]
        for c in calls:
            result["tool_trace"].append(
                {
                    "tool": c["tool"],
                    "line": c["line"],
                    "ok": c["response"]["ok"],
                    **(
                        {"error_code": c["response"]["error"].get("code")}
                        if not c["response"]["ok"]
                        else {}
                    ),
                    "argument_sha256": text_digest(
                        json.dumps(c["arguments"], sort_keys=True, ensure_ascii=False)
                    ),
                    "response_sha256": text_digest(
                        json.dumps(c["response"], sort_keys=True, ensure_ascii=False)
                    ),
                }
            )
        observed(root and root.is_dir(), "harness_root_missing")
        ids = {
            c["response"]["result"]["run_id"]
            for c in calls
            if c["tool"] in {"inspect_input", "submit_analysis"} and c["response"]["ok"]
        }
        runtime = None
        if bundle:
            from autotem.acceptance.driver import runtime_record

            runtime = runtime_record(True, bundle.resolve())
            result["runtime"] = runtime
        runs = {i: run_evidence(root.resolve(), i, (start, end), runtime) for i in sorted(ids)}
        result["runs"] = [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in runs.values()
        ]
        check_tool_runs(calls, runs)
        variant = spec.get("variant")
        if spec["case_id"] == "J14":
            mode = spec.get("delegation_mode", "sequential")
            require(variant is None or variant == mode, "delegation_variant_mismatch")
            variant = mode
        if variant:
            result["variant"] = variant
        case_evidence = check_case(
            spec["case_id"], calls, runs, turns, spec.get("followup_excerpts", {}), variant
        )
        if case_evidence and spec["case_id"] == "J08":
            result["gpa_followup"] = case_evidence
        checks["case_execution"] = "passed"
        all_artifacts = {i: a for run in runs.values() for i, a in run["artifacts"].items()}
        returned = set()
        reports = set()
        for c in calls:
            if not c["response"]["ok"]:
                continue
            data = c["response"]["result"]
            if c["tool"] == "export_report":
                reports.add(data["run_id"])
            if c["tool"] == "read_artifact":
                artifacts = [data["artifact"]]
                require(
                    c["arguments"]["artifact_id"] == data["artifact"]["artifact_id"],
                    "read_result_mismatch",
                )
            elif c["tool"] == "list_artifacts":
                artifacts = data
            elif c["tool"] in {"inspect_input", "export_report"}:
                artifacts = data["artifacts"]
            else:
                artifacts = []
            for a in artifacts:
                i = a["artifact_id"]
                require(
                    i in all_artifacts
                    and all_artifacts[i]["sha256"] == a["sha256"]
                    and all_artifacts[i]["role"] == a["role"],
                    "host_artifact_does_not_match_disk",
                )
                returned.add(i)
        diagnostic = (
            spec["case_id"] in {"J10", "J11", "J12", "J13"}
            or spec["case_id"] == "J08"
            and variant in {"unanswered", "invalid_reply", "gpa_unanswered", "gpa_invalid_reply"}
            or spec["case_id"] == "J15"
            and variant not in {None, "executed"}
        )
        if runs and not diagnostic:
            require(
                reports & {r["run_id"] for r in runs.values() if r["status"] == "completed"},
                "host_did_not_export_analysis_report",
            )
        final = turns[-1]["answer"]
        claims = spec.get("claims", [])
        if not claims:
            # Replies commonly cite the artifact IDs directly in each result paragraph.
            for line in final.splitlines():
                refs = sorted(i for i in returned if i in line)
                if refs:
                    claims.append({"quote": line.strip(), "artifact_ids": refs})
        if runs and not diagnostic:
            observed(claims, "final_answer_has_no_traceable_citations")
        if runs and diagnostic and not claims:
            observed(any(i in final for i in runs), "diagnostic_answer_missing_run_reference")
        for claim in claims:
            quote, refs = claim["quote"], claim["artifact_ids"]
            require(
                quote
                and quote in final
                and refs
                and all(i in returned and i in final for i in refs),
                "claim_not_grounded_in_observed_answer_and_evidence",
            )
            result["claims"].append({"quote_sha256": text_digest(quote), "artifact_ids": refs})
        if spec["case_id"] == "J01":
            result["spacing_answer"] = check_spacing_answer(final, claims, runs, returned)
            checks["quoted_spacing_consistency"] = "passed"
        elif spec["case_id"] == "J03":
            try:
                result["gpa_coverage_answer"] = check_gpa_coverage_answer(
                    final, claims, runs, returned, case_evidence
                )
            except NotEvaluated:
                checks["quoted_gpa_coverage_consistency"] = "not_evaluated"
                raise
            except EvidenceError:
                checks["quoted_gpa_coverage_consistency"] = "failed"
                raise
            checks["quoted_gpa_coverage_consistency"] = "passed"
        checks["answer_return"] = "passed"
        host_observations(spec, base, turns + children, calls, result)
        observed(runtime is not None, "no_verified_release_binding")
        if spec["case_id"] == "J15" and variant in {None, "executed"}:
            observed(
                runtime["release_identity"].get("profile") == "traditional-simulation",
                "simulation_runtime_profile_not_verified",
            )
        observed(project and project.is_dir(), "deployed_host_project_missing")
        skill = {
            "codex": ".agents/skills/autotem",
            "workbuddy": "autotem-host/workbuddy/skill",
            "openclaw": "skills/autotem",
        }[spec["host"]]
        for name, expected in runtime["skill_resource_sha256"].items():
            path = project / skill / name
            require(
                path.is_file() and digest(path) == expected, "deployed_skill_or_profile_mismatch"
            )
        checks["release_and_resources"] = "passed"
        adapter = project / "autotem-host" / spec["host"] / "adapter.json"
        observed(adapter.is_file(), "missing_project_host_adapter")
        adapter_data = unique_json(adapter.read_text("utf-8"))
        require(
            adapter_data.get("host") == spec["host"]
            and root.resolve() == (project / ".autotem" / "runs").resolve(),
            "host_project_binding_mismatch",
        )
        result["evidence"]["adapter_sha256"] = digest(adapter)
        from autotem.installation import host_entry

        control = project.resolve() / ".autotem"
        entry = adapter_data["mcp"]["mcpServers"]["autotem"]
        require(
            host_entry.active_python(control) == Path(sys.executable)
            and entry["args"] == ["-I", str(control / "host-entry.py"), "mcp"]
            and Path(entry["command"]).is_file(),
            "project_runtime_binding_mismatch",
        )
        if spec["host"] == "codex":
            from autotem.harness.hosts import _read_toml

            path = project / ".codex" / "config.toml"
            observed(path.is_file(), "missing_project_mcp_config")
            configured = (
                _read_toml(path.read_text("utf-8-sig"))
                .get("mcp_servers", {})
                .get(spec.get("server", "autotem"), {})
            )
            require(
                all(configured.get(k) == entry[k] for k in ("command", "args"))
                and Path(configured.get("cwd", "")).resolve() == project.resolve(),
                "active_host_mcp_entry_mismatch",
            )
            result["evidence"]["host_config_sha256"] = digest(path)
        require(
            digest(control / "host-entry.py") == digest(Path(host_entry.__file__)),
            "deployed_host_entry_mismatch",
        )
        observed(
            checks["session_skill_discovery"] == "passed", "skill_use_in_host_session_unobserved"
        )
        result["case_verdict"] = "passed"
    except NotEvaluated as exc:
        result["reason"] = str(exc)
    except (
        KeyError,
        IndexError,
        ValueError,
        TypeError,
        AttributeError,
        OSError,
        sqlite3.Error,
        ImportError,
        RuntimeError,
        AssertionError,
    ) as exc:
        result.update(
            case_verdict="failed",
            reason=str(exc) if type(exc) is EvidenceError else type(exc).__name__,
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    new = sub.add_parser("init")
    new.add_argument("directory", type=Path)
    new.add_argument("--host", required=True, choices=HOSTS)
    new.add_argument("--case", required=True, choices=CASES)
    new.add_argument("--variant", help="Optional matrix variant or supplementary J08 GPA case")
    check = sub.add_parser("verify")
    check.add_argument("record", type=Path)
    check.add_argument("--release", type=Path)
    check.add_argument("--harness-root", type=Path)
    check.add_argument("--project", type=Path)
    check.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.action == "init":
        if args.variant is not None and args.variant not in case_variants(args.case):
            parser.error("Unknown variant for this case")
        args.directory.mkdir(parents=True, exist_ok=True)
        template = {
            "schema_version": "1.0.0",
            "host": args.host,
            "case_id": args.case,
            "observation": "not_evaluated",
            "host_version": "",
            "started_at": "",
            "ended_at": "",
            "capture_format": "codex-exec-jsonl"
            if args.host == "codex"
            else "observed-host-export",
            "turns": [],
            "delegation_mode": "sequential",
            "claims": [],
            "host_observations": {},
        }
        if args.variant is not None:
            template["variant"] = args.variant
        with (args.directory / "case.json").open("x", encoding="utf-8") as f:
            json.dump(template, f, indent=2)
        return 0
    try:
        result = verify(
            args.record, bundle=args.release, root=args.harness_root, project=args.project
        )
        payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.out:
            with args.out.open("x", encoding="utf-8") as f:
                f.write(payload + "\n")
        else:
            print(payload)
        return 0 if result["case_verdict"] == "passed" else 1
    except (KeyError, ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"Invalid capture or output: {type(exc).__name__}\n")


if __name__ == "__main__":
    raise SystemExit(main())
