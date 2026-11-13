"""Exercise failures, protocols, multiple results and legacy entry points on installed code."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import sysconfig
import time
import venv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from autotem.acceptance.cases import write_fixture
from autotem.acceptance.evidence import digest, require
from autotem.harness import Harness, HarnessError

STATES = {"completed", "failed", "cancelled", "timed_out", "interrupted", "needs_input"}


def save(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


def sample(folder, case="J01"):
    return write_fixture(case, folder)[0]


def step(i, op, **p):
    return {"id": i, "operation_id": op, "parameters": p}


def plan(source, *steps):
    return {"input_path": str(source), "steps": [step("load", "input.load"), *steps]}


def terminal(h, run_id, limit=100):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        state = h.get_run_status(run_id)
        if state["status"] in STATES and state["jobs"][-1]["status"] in STATES:
            return state
        time.sleep(0.08)
    raise TimeoutError(f"Harness run {run_id} did not finalize")


def wait_for_worker(h, rid):
    from autotem.harness.worker import process_identity

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        record = h.store.get_run(rid)
        pid, identity = record["worker_pid"], record["worker_identity"]
        if record["status"] == "running" and identity and process_identity(pid) == identity:
            return pid, identity
        if record["status"] in STATES:
            raise ValueError(f"Run finished before owned worker observation: {record['status']}")
        time.sleep(0.025)
    raise TimeoutError("Worker did not acquire its run")


def record(folder, state):
    name = state["run_id"] + ".json"
    save(folder / name, state)
    return {
        "run_id": state["run_id"],
        "execution_status": state["status"],
        "error": (state["error"] or {}).get("code"),
        "state_file": name,
        "sha256": digest(folder / name),
    }


def expected(fn, code):
    try:
        fn()
    except HarnessError as exc:
        require(exc.code == code, f"Expected {code}, observed {exc.code}")
        return exc.code
    raise ValueError(f"Expected {code}; call succeeded")


def report(h, rid, formats=("json", "html")):
    export = h.export_report(rid, formats=list(formats))
    output = {a["role"]: a for a in export["artifacts"]}
    document = h.store.load_json(output["report_manifest"]["artifact_id"])
    if "report_html" in output:
        html = Path(output["report_html"]["path"]).read_text("utf-8")
        require(
            all('href="' + Path(a["path"]).name + '"' in html for a in document["artifacts"]),
            "HTML omits registered evidence",
        )
    return document, output


def inputs_and_science(folder, frames, base):
    h = Harness(folder / "harness")
    codes = {
        "missing": expected(lambda: h.inspect_input(str(folder / "missing.npy")), "INPUT_NOT_FOUND")
    }
    stack = np.arange(3 * 24 * 32, dtype=np.float32).reshape(3, 24, 32)
    path = folder / "stack.npy"
    np.save(path, stack)
    ambiguous = h.inspect_input(str(path))
    selected = h.inspect_input(str(path), frame=1, roi=[3, 20, 2, 28])
    require(
        ambiguous["status"] == "needs_input" and selected["status"] == "completed",
        "Ambiguous stack or resolved ROI",
    )
    np.testing.assert_array_equal(
        h.store.load_array(h.store.role(selected["run_id"], "raw")), stack[1, 3:20, 2:28]
    )
    source = sample(folder, "J04")
    common = [step("detect", "atomic.detect"), step("refine", "atomic.refine")]
    pending = h.submit_analysis(
        plan(source, *common, step("atom", "strain.atom")), asynchronous=False
    )
    require(
        pending["status"] == "needs_input"
        and pending["steps"][-1]["error"]["code"] == "NEEDS_INPUT",
        "Missing atom reference was silently accepted",
    )
    diagnostics = [a for a in h.list_artifacts(pending["run_id"]) if a["step_id"] == "atom"]
    require(
        diagnostics and all(not a["execution_usable"] for a in diagnostics),
        "Input-request evidence must be diagnostic",
    )
    attempt = {
        "input_path": str(source),
        "steps": [
            {**step("qa", "quality.assess"), "inputs": {"raw": diagnostics[0]["artifact_id"]}}
        ],
    }
    codes["diagnostic_replay"] = expected(
        lambda: h.validate_plan(attempt, run_id=pending["run_id"]), "ARTIFACT_NOT_EXECUTABLE"
    )
    bad = step(
        "collinear",
        "strain.gpa",
        g_vectors=[[0.125, 0], [0.25, 0]],
        reference_region=[4, 32, 4, 32],
    )
    failed = h.submit_analysis(plan(source, bad), asynchronous=False)
    require(
        failed["status"] == "failed" and failed["error"]["code"] == "INVALID_SCIENTIFIC_INPUT",
        "Collinear g vectors must fail",
    )
    skipped = h.submit_analysis(
        plan(source, {**bad, "optional": True}, step("valid", "quality.assess")), asynchronous=False
    )
    require(
        skipped["status"] == "completed", "Optional failure prevented an independent required step"
    )
    manifest, _ = report(h, skipped["run_id"])
    require(
        manifest["completeness"]["optional_skipped"] == ["collinear"]
        and manifest["completeness"]["required_complete"],
        "Optional failure entered the report as a result",
    )
    require("gpa" not in manifest["result_index"], "Optional failure published a scientific result")
    for state in (pending, failed):
        manifest, _ = report(h, state["run_id"])
        require(
            manifest["report_type"] == "diagnostic"
            and not manifest["completeness"]["required_complete"],
            "Failed analyses must export diagnostic evidence",
        )
    metric = step(
        "missing_metric",
        "validation.evaluate",
        acceptance_criteria=[
            {"metric": "atom_coordinate_count", "op": ">=", "threshold": 1, "unit": "count"}
        ],
    )
    state = h.submit_analysis(plan(source, metric), asynchronous=False)
    value = h.store.load_json(h.store.role(state["run_id"], "validation"))
    require(value["acceptance"]["all_passed"] is False, "Missing required metric passed")
    require(value["acceptance"]["results"][0].get("passed") is False, "No per-metric failure")
    return {
        "rejections": codes,
        "runs": [
            record(folder, terminal(h, v["run_id"]))
            for v in (ambiguous, selected, pending, failed, skipped, state)
        ],
    }


def corrupt_artifact(folder, frames, base):
    h = Harness(folder / "harness")
    source = sample(folder)
    state = h.submit_analysis(plan(source), asynchronous=False)
    raw_id = h.store.role(state["run_id"], "raw")
    registered = h.store.artifact(raw_id)
    path = Path(registered["path"])
    require(
        path.resolve().is_relative_to(folder.resolve()) and digest(path) == registered["sha256"],
        "Fixture outside test root",
    )
    with path.open("ab") as out:
        out.write(b"RG2 deliberate fixture corruption")
    observed = expected(lambda: h.store.load_array(raw_id), "ARTIFACT_CHANGED")
    follow = plan(source, step("qa", "quality.assess"))
    follow["steps"] = follow["steps"][1:]
    failed = h.submit_analysis(follow, run_id=state["run_id"], asynchronous=False)
    require(
        failed["status"] == "failed" and failed["error"]["code"] == "ARTIFACT_CHANGED",
        "Corrupt raw was reused",
    )
    return {
        "expected_error": observed,
        "registered_sha256": registered["sha256"],
        "corrupt_sha256": digest(path),
        "intentionally_corrupt_file": path.relative_to(folder).as_posix(),
        "artifact_id": raw_id,
        "job_status_at_failed_response": failed["jobs"][-1]["status"],
        "run": record(folder, terminal(h, failed["run_id"])),
    }


def stop_own_worker(h, rid, pid, identity, folder):
    """Bind the test-owned process and its creation time to the same NT process handle."""
    import ctypes
    from ctypes import wintypes

    require(
        os.name == "nt" and h.store.run_dir(rid).resolve().is_relative_to(folder.resolve()),
        "Interrupt only this test's private Windows run",
    )
    row = h.store.get_run(rid)
    require(
        (row["worker_pid"], row["worker_identity"], row["status"]) == (pid, identity, "running"),
        "Owned worker changed",
    )
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x1000 | 0x1, False, pid)
    require(handle, "Could not open owned worker")
    try:
        times = [ctypes.c_ulonglong() for _ in range(4)]
        require(
            kernel.GetProcessTimes(wintypes.HANDLE(handle), *(ctypes.byref(x) for x in times))
            and str(times[0].value) == identity,
            "Worker identity changed",
        )
        require(
            kernel.TerminateProcess(wintypes.HANDLE(handle), 83), "Owned worker interruption failed"
        )
    finally:
        kernel.CloseHandle(wintypes.HANDLE(handle))


def jobs(folder, frames, base):
    h = Harness(folder / "harness")
    source = sample(folder)
    request = plan(source, step("qa", "quality.assess"))
    active_request = plan(source, *(step(f"qa{i}", "quality.assess") for i in range(6)))
    run = h.submit_analysis(active_request, idempotency_key="active")
    pid, identity = wait_for_worker(h, run["run_id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        waiting, untouched = list(
            pool.map(
                lambda key: h.submit_analysis(request, idempotency_key=key), ("queue", "survivor")
            )
        )
    require(waiting["status"] == "queued", "No actual queued cancellation observed")
    queued = h.cancel_run(waiting["run_id"])
    require(queued["status"] == "cancelled", "Queue cancel failed")
    active_at_cancel = h.get_run_status(run["run_id"])["status"]
    require(active_at_cancel == "running", "No actual running cancellation observed")
    h.cancel_run(run["run_id"])
    cancel = terminal(h, run["run_id"])
    surviving = terminal(Harness(folder / "harness"), untouched["run_id"])
    require(
        cancel["status"] == "cancelled" and surviving["status"] == "completed", "Isolation/cancel"
    )
    retry = h.submit_analysis(request, idempotency_key="survivor")
    require(retry["run_id"] == surviving["run_id"], "Retry changed run")
    error = expected(
        lambda: h.submit_analysis({**request, "modality": "HRTEM"}, idempotency_key="survivor"),
        "IDEMPOTENCY_CONFLICT",
    )
    target = h.submit_analysis(request, idempotency_key="interrupt")
    p2, i2 = wait_for_worker(h, target["run_id"])
    stop_own_worker(h, target["run_id"], p2, i2, folder)
    interrupted = terminal(Harness(folder / "harness"), target["run_id"])
    require(
        interrupted["status"] == "interrupted"
        and interrupted["error"]["code"] == "WORKER_INTERRUPTED",
        "Worker exit lost",
    )
    timed = terminal(h, h.submit_analysis({**request, "budgets": {"run_seconds": 0.01}})["run_id"])
    require(timed["status"] == "timed_out", "Missing run timeout")
    independent = Harness(folder / "second-root")
    absent = expected(lambda: independent.get_run_status(surviving["run_id"]), "RUN_NOT_FOUND")
    return {
        "running_pid": pid,
        "running_creation_identity": identity,
        "interrupted_pid": p2,
        "interrupted_creation_identity": i2,
        "unrelated_root_error": absent,
        "idempotency_conflict": error,
        "runs": [record(folder, s) for s in (queued, cancel, surviving, interrupted, timed)],
    }


def cumulative(folder, frames, base):
    h = Harness(folder / "harness")
    source = sample(folder)
    budget = 12.0
    request = {
        "input_path": str(source),
        "steps": [step("load", "input.load")],
        "budgets": {"run_seconds": budget},
    }
    first = terminal(h, h.submit_analysis(request)["run_id"])
    require(
        first["status"] == "completed" and first["execution_seconds"] > 0,
        "Need a real completed first job",
    )
    last = first
    observations = [(first["job_id"], first["execution_seconds"], first["status"])]
    for i in range(48):
        follow = {**request, "steps": [step(f"qa{i}", "quality.assess")]}
        try:
            submitted = h.submit_analysis(follow, run_id=first["run_id"])
        except HarnessError as exc:
            require(
                exc.code == "RUN_BUDGET_EXHAUSTED" and last["execution_seconds"] >= budget,
                "Unexpected append rejection",
            )
            break
        current = terminal(h, submitted["run_id"])
        require(
            current["execution_seconds"] > last["execution_seconds"]
            and current["job_id"] != last["job_id"],
            "Cumulative budget reset or repeated job",
        )
        observations.append((current["job_id"], current["execution_seconds"], current["status"]))
        last = current
        if current["status"] == "timed_out":
            require(
                current["error"]["code"] == "TIMEOUT"
                and current["error"]["message"] == "Run budget exceeded",
                "Unexpected operation timeout",
            )
            break
        require(current["status"] == "completed", "Unexpected append result")
    require(
        len(observations) > 1 and last["execution_seconds"] >= budget,
        "Cumulative run budget never enforced",
    )
    return {"budget_seconds": budget, "job_measurements": observations, "run": record(folder, last)}


def multi_result(folder, frames, base):
    from autotem.core import StrainResult

    h = Harness(folder / "harness")
    source = sample(folder, "J04")
    raw = np.load(source)
    req = plan(
        source,
        step("detect", "atomic.detect"),
        step("refine", "atomic.refine"),
        step("atom", "strain.atom", reference_region=[8, 36, 8, 68]),
        step(
            "gpa",
            "strain.gpa",
            g_vectors=[[0.125, 0], [0, 0.125]],
            reference_region=[16, 48, 16, 48],
        ),
    )
    req["modality"] = "HAADF-STEM"
    state = h.submit_analysis(req, asynchronous=False)
    require(state["status"] == "completed", "Dual strain run failed")
    manifest, files = report(h, state["run_id"], ("json", "html", "pdf"))
    strain = manifest["strain_results"]
    require(
        [r["role"] for r in strain["all"]] == ["atom_strain", "gpa"]
        and strain["primary"]["role"] == "atom_strain",
        "Primary or alternate strain dropped",
    )
    for s in strain["all"]:
        obj = StrainResult.model_validate(h.store.load_json(s["artifact_id"]))
        require(
            obj.schema_version == "0.1.0"
            and obj.warnings
            and np.load(obj.strain_xx.uri, allow_pickle=False).shape == raw.shape,
            "Strict schema/strain fields",
        )
    return {
        "run": record(folder, state),
        "primary": strain["primary"],
        "all_strain": strain["all"],
        "reports": {k: v["sha256"] for k, v in files.items()},
    }


def legacy(folder, frames, base):
    import autotem.harness
    from autotem import TEMAnalysisSkill
    from autotem.core import AnalysisReport
    from autotem.harness.compiler import compile_legacy_recipe

    source = sample(folder)
    minimal = folder / "minimal.yaml"
    minimal.write_text(
        yaml.safe_dump(
            {
                "workflow": {"name": "read", "steps": ["load", "report"]},
                "outputs": {"formats": ["json"]},
            }
        ),
        "utf-8",
    )
    old = TEMAnalysisSkill().run(source, minimal, folder / "old")
    strict = AnalysisReport.model_validate_json(Path(old["report_path"]).read_bytes())
    require(
        strict.schema_version == "0.1.0"
        and tuple(strict.dataset.data.shape) == np.load(source).shape,
        "Legacy strict schema",
    )
    migrated = TEMAnalysisSkill().run_harness(source, minimal, folder / "migrated")
    h = Harness(folder / "migrated")
    minimal_report, _ = report(h, migrated["run_id"])
    require(
        [c["operation_id"] for c in minimal_report["claim_evidence"]] == ["input.load"],
        "Legacy recipe changed scope",
    )
    raw = np.load(source)
    np.testing.assert_array_equal(raw, h.store.load_array(h.store.role(migrated["run_id"], "raw")))
    resource = next(
        (Path(autotem.harness.__file__).parent / "resources").rglob("hrtem-diffraction.yaml")
    )
    spec = yaml.safe_load(resource.read_text("utf-8"))
    for s in spec["steps"]:
        if s["operation_id"] == "calibration.resolve":
            s["parameters"] = {"pixel_size_nm": 0.02}
        if s["operation_id"] == "preprocessing.apply":
            s["parameters"] = {"denoise_method": "none"}
    formal = folder / "formal.yaml"
    formal.write_text(yaml.safe_dump(spec), "utf-8")
    cmd = [
        sys.executable,
        "-I",
        "-m",
        "autotem.cli",
        "analyze",
        str(source),
        "--harness",
        "--workflow",
        str(formal),
        "--output",
        str(folder / "formal"),
        "--formats",
        "json",
    ]
    response = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=110)
    require(response.returncode == 0, "Registered CLI: " + response.stderr[-400:])
    value = json.loads(response.stdout.split("\n", 1)[1])
    formal_h = Harness(folder / "formal")
    formal_report, _ = report(formal_h, value["run_id"])
    observed = formal_h.store.load_array(formal_h.store.role(value["run_id"], "fft_complex"))
    np.testing.assert_array_equal(observed, frames["J01"].array("diffraction.fft", "fft_complex"))
    historical_recipe = folder / "historical-fft.yaml"
    historical_recipe.write_text(
        yaml.safe_dump(
            {
                "workflow": {
                    "name": "same-fft",
                    "steps": ["load", "assess_quality", "calibrate", "preprocess", "fft", "report"],
                    "parameters": {
                        "calibration": {"pixel_size_nm": 0.02},
                        "preprocessing": {"denoise_method": "none"},
                    },
                },
                "outputs": {"formats": ["json"]},
            }
        ),
        "utf-8",
    )
    historical = TEMAnalysisSkill().run(source, historical_recipe, folder / "old-fft")
    previous = AnalysisReport.model_validate_json(Path(historical["report_path"]).read_bytes())
    np.testing.assert_array_equal(
        np.load(previous.preprocessing.image.uri, allow_pickle=False),
        formal_h.store.load_array(formal_h.store.role(value["run_id"], "processed")),
    )
    np.testing.assert_array_equal(
        np.load(previous.diffraction.fft_image.uri, allow_pickle=False),
        np.log1p(np.abs(observed)).astype(np.float32),
    )
    require(
        previous.schema_version == "0.1.0"
        and (previous.calibration.pixel_size.value, previous.calibration.pixel_size.unit)
        == (0.02, "nm/pixel"),
        "Historical calibrated FFT schema/units",
    )
    require(
        [c["operation_id"] for c in formal_report["claim_evidence"]]
        == [s["operation_id"] for s in spec["steps"]],
        "Formal recipe omitted operations",
    )
    bad = folder / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({"workflow": {"name": "bad", "steps": ["load", "mystery_step", "report"]}}),
        "utf-8",
    )
    try:
        compile_legacy_recipe(str(source), bad)
    except HarnessError as exc:
        error = exc.code
    else:
        raise ValueError("Unknown legacy step silently accepted")
    require(error == "LEGACY_STEP_REQUIRES_MIGRATION", "Legacy migration code changed")
    return {
        "old_report_sha256": digest(Path(old["report_path"])),
        "old_fft_report_sha256": digest(Path(historical["report_path"])),
        "old_fft_image_sha256": digest(Path(previous.diffraction.fft_image.uri)),
        "old_fft_representation": "log1p(abs(FFT)) as float32",
        "minimal_run": migrated["run_id"],
        "formal_run": value["run_id"],
        "packaged_recipe_sha256": digest(resource),
        "unknown_step_code": error,
        "strict_schema_version": strict.schema_version,
        "numerical_comparison": "Complex rectangular FFT with equal input and calibration",
    }


def protocols(folder, frames, base):
    from autotem.harness.cli import TOOL_SCHEMAS
    from autotem.harness.readiness import _mcp_probe

    source = sample(folder)
    h = Harness(folder / "harness")
    done = h.submit_analysis(plan(source), asynchronous=False)
    run_id = done["run_id"]
    args = {
        "list_capabilities": {},
        "inspect_input": {"path": str(source)},
        "validate_plan": {"plan": plan(source)},
        "submit_analysis": {"plan": plan(source), "asynchronous": False},
        "get_run_status": {"run_id": run_id},
        "cancel_run": {"run_id": run_id},
        "list_artifacts": {"run_id": run_id},
        "read_artifact": {"artifact_id": h.store.role(run_id, "raw"), "limit": 4},
        "export_report": {"run_id": run_id, "formats": ["json"]},
    }
    require(args.keys() == TOOL_SCHEMAS.keys(), "Public nine tools changed")
    lines = ["{bad", '{"tool":"list_capabilities","tool":"list_capabilities"}']
    for tool, valid in args.items():
        lines += [
            json.dumps(
                {"request_id": f"{tool}-bad", "tool": tool, "arguments": {"unexpected": True}}
            ),
            json.dumps({"request_id": tool, "tool": tool, "arguments": valid}),
        ]
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "autotem.harness.cli",
            "--root",
            str(folder / "harness"),
            "stdio",
        ],
        input="\n".join(lines) + "\n",
        capture_output=True,
        encoding="utf-8",
        timeout=120,
    )
    require(result.returncode == 0, "Installed JSONL process failed")
    replies = [json.loads(s) for s in result.stdout.splitlines()]
    require(len(replies) == len(lines), "JSONL stdout did not have exactly one response per line")
    require(
        all(not v["ok"] and v["error"]["code"] == "REQUEST_INVALID" for v in replies[:2]),
        "Malformed or duplicate-key JSON accepted",
    )
    results = {}
    for i, tool in enumerate(args):
        no, yes = replies[2 + 2 * i : 4 + 2 * i]
        require(
            no["request_id"] == tool + "-bad"
            and not no["ok"]
            and no["error"]["code"] == "REQUEST_INVALID"
            and yes["request_id"] == tool
            and yes["ok"],
            f"Protocol argument verdicts for {tool}: {yes.get('error')}",
        )
        results[tool] = [no["error"]["code"], yes["ok"]]
    trace = [
        {"request_line": line, "response": resp} for line, resp in zip(lines, replies, strict=True)
    ]
    save(folder / "jsonl-trace.json", trace)
    mcp_started = time.monotonic()
    mcp = _mcp_probe(sys.executable, str(folder / "mcp"), 120)
    require(
        mcp["status"] == "ready" and mcp["tool_count"] == 9 and mcp["operation_count"] == 49,
        f"MCP call {mcp}",
    )
    return {
        "jsonl": results,
        "requests": len(trace),
        "trace_sha256": digest(folder / "jsonl-trace.json"),
        "mcp": mcp,
        "mcp_elapsed_seconds": time.monotonic() - mcp_started,
        "mcp_deadline_seconds": 120,
    }


PLUGIN_MODULE = """
from autotem.harness.contracts import OperationDefinition, ServiceResult

def count(ctx, params):
    return ServiceResult({'pixel_count': int(ctx.array('raw').size)})

def definitions():
    return [OperationDefinition('fixture.pixel_count', 'data-quality', count,
                                input_roles=('raw',), output_roles=('pixel_count',))]

def same_version():
    return [OperationDefinition('ai.denoise', 'reserved', count,
                                input_roles=('raw',), output_roles=('processed',))]

def wrong_roles():
    return [OperationDefinition('ai.denoise', 'reserved', count, version='2.0.0',
                                input_roles=('raw',), output_roles=('pixel_count',))]

def new_version():
    return [OperationDefinition('ai.denoise', 'reserved', count, version='2.0.0',
                                input_roles=('raw',), output_roles=('processed',))]
"""


def trusted_extension(folder, frames, base):
    suffix = Path("Scripts/python.exe") if os.name == "nt" else Path("bin/python")
    child = folder / "extension-venv"
    venv.EnvBuilder(with_pip=False).create(child)
    target = child / suffix
    lib = Path(
        subprocess.check_output(
            [str(target), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
        ).strip()
    )
    require(
        lib.resolve().is_relative_to(child.resolve()),
        "Supplementary plugin site must be disposable",
    )
    (lib / "release-libraries.pth").write_text(sysconfig.get_path("purelib") + "\n", "utf-8")
    plugin = lib / "qualification_pixel.py"
    plugin.write_text(PLUGIN_MODULE, "utf-8")
    (lib / "qualification_broken.py").write_text(
        "raise ImportError('qualification fixture has deliberately broken initialization')\n",
        "utf-8",
    )
    dist = lib / "qualification_pixel-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: qualification-pixel\nVersion: 1.0\nLicense: CC0-1.0\n",
        "utf-8",
    )
    (dist / "entry_points.txt").write_text(
        "[autotem.operations]\nopt_in = qualification_pixel:definitions\n"
        "same_version = qualification_pixel:same_version\n"
        "wrong_roles = qualification_pixel:wrong_roles\n"
        "new_version = qualification_pixel:new_version\n",
        "utf-8",
    )
    source = sample(folder)
    proc = subprocess.run(
        [
            str(target),
            "-I",
            str(Path(__file__).resolve()),
            "--trusted-probe",
            str(folder),
            str(source),
        ],
        capture_output=True,
        encoding="utf-8",
        timeout=120,
    )
    require(proc.returncode == 0, "Controlled trusted plugin: " + proc.stderr[-650:])
    result = json.loads(proc.stdout)
    require(result["actual_pixels"] == 64 * 80, "Installed plugin dropped raw input")
    result.update(
        plugin_source_sha256=digest(plugin),
        fixture_license="CC0-1.0",
        isolation="Separate plugin venv using the private wheel's site-packages",
        plugin_module=str(plugin.relative_to(folder)),
    )
    return result


def plugin_probe(folder, source):
    from autotem.harness.contracts import OperationDefinition, StrictParameters
    from autotem.harness.registry import default_registry, probe_dependency

    registry = default_registry()

    class OpenParameters(StrictParameters):
        model_config = {"extra": "allow"}

    def handler(ctx, params):
        return None

    error = {
        "disabled": expected(lambda: registry.get("fixture.pixel_count"), "UNKNOWN_OPERATION"),
        "duplicate": expected(
            lambda: registry.register(registry.get("input.load")), "DUPLICATE_OPERATION"
        ),
        "missing": expected(
            lambda: default_registry(["absent_qualification_plugin"]), "PLUGIN_NOT_FOUND"
        ),
        "same_version": expected(
            lambda: default_registry(["same_version"]), "PLUGIN_VERSION_REQUIRED"
        ),
        "roles": expected(
            lambda: default_registry(["wrong_roles"]), "INCOMPATIBLE_PLUGIN_CONTRACT"
        ),
        "no_implemented_handler": expected(
            lambda: registry.register(OperationDefinition("fixture.no_handler", "delivery", None)),
            "INVALID_OPERATION_CONTRACT",
        ),
        "reserved_handler": expected(
            lambda: registry.register(
                OperationDefinition(
                    "fixture.has_handler", "delivery", handler, implementation_status="reserved"
                )
            ),
            "INVALID_OPERATION_CONTRACT",
        ),
        "open_schema": expected(
            lambda: registry.register(
                OperationDefinition(
                    "fixture.open", "delivery", handler, parameter_model=OpenParameters
                )
            ),
            "INVALID_OPERATION_CONTRACT",
        ),
    }
    for name, state in (
        ("qualification_absent", "dependency_unavailable"),
        ("qualification_broken", "dependency_broken"),
    ):
        diagnosis = probe_dependency(name)
        require(diagnosis["state"] == state, "Dependency diagnostic changed")
        error[name] = state
    h = Harness(folder / "harness")
    error["pinned_version"] = expected(
        lambda: h.validate_plan(
            {
                "input_path": str(source),
                "steps": [{**step("read", "input.load"), "operation_version": "0.0.0"}],
            }
        ),
        "OPERATION_VERSION_MISMATCH",
    )
    replacement = default_registry(["new_version"]).get("ai.denoise")
    require(
        registry.get("ai.denoise").implementation_status == "reserved"
        and replacement.version == "2.0.0"
        and replacement.output_roles == ("processed",),
        "Explicit compatible reserved replacement not registered",
    )
    req = {**plan(source, step("plugin", "fixture.pixel_count")), "enabled_plugins": ["opt_in"]}
    accepted = h.submit_analysis(req, asynchronous=False)
    require(accepted["status"] == "completed", str(accepted["error"]))
    result = h.store.load_json(h.store.role(accepted["run_id"], "pixel_count"))
    provenance = default_registry(["opt_in"]).plugin_sources["fixture.pixel_count"]
    require(
        provenance["distribution"] == "qualification-pixel" and provenance["explicitly_enabled"],
        "Plugin provenance missing",
    )
    return {
        "actual_pixels": result,
        "controlled_reserved_replacement": replacement.version,
        "errors": error,
        "provenance": provenance,
        "run": record(folder, accepted),
    }


CHECKS = {
    "selection_science_and_optional_failures": inputs_and_science,
    "corrupt_artifact_fails_closed": corrupt_artifact,
    "owned_jobs_and_isolation": jobs,
    "cumulative_run_budget": cumulative,
    "multi_result_reports": multi_result,
    "legacy_api_cli_yaml": legacy,
    "local_nine_tool_protocol": protocols,
    "trusted_extension_contracts": trusted_extension,
}


def recheck(name, root, check, inventory):
    """Validate captured public responses against SQLite snapshots and the expected fault."""
    require(name in CHECKS and check["verdict"] == "passed", "Unverified local fault")
    folder = root / name
    snapshot = root / "sqlite-snapshots" / name
    dbs = sorted(snapshot.rglob("harness.sqlite3"))
    require(
        dbs and all(p.relative_to(root).as_posix() in inventory for p in dbs),
        "Missing SQLite snapshot " + name,
    )
    rows = {}
    for path in dbs:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as db:
            require(
                db.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
                "Invalid SQLite snapshot",
            )
            rows.update(
                {
                    r[0]: r[1:]
                    for r in db.execute("SELECT run_id,status,error,run_seconds_used FROM runs")
                }
            )
    for ref in [*check.get("runs", []), *([check["run"]] if "run" in check else [])]:
        state_path = (folder / ref["state_file"]).resolve()
        require(
            state_path.is_relative_to(folder)
            and state_path.relative_to(root).as_posix() in inventory
            and digest(state_path) == ref["sha256"],
            "Missing state response SHA",
        )
        state = json.loads(state_path.read_text("utf-8"))
        status, error, _ = rows[ref["run_id"]]
        require(
            ref["run_id"] == state["run_id"]
            and state["status"] == status == ref["execution_status"]
            and (json.loads(error) if error else None) == state["error"],
            "State and database diverged",
        )
        require(
            state["events"] and state["jobs"][-1]["status"] == status, "Run/job/event divergence"
        )
    if name == "selection_science_and_optional_failures":
        require(
            [r["execution_status"] for r in check["runs"]]
            == ["needs_input", "completed", "needs_input", "failed", "completed", "completed"]
            and check["runs"][3]["error"] == "INVALID_SCIENTIFIC_INPUT"
            and check["rejections"]["missing"] == "INPUT_NOT_FOUND",
            "Selection/science faults",
        )
    elif name == "corrupt_artifact_fails_closed":
        require(
            check["run"]["error"] == check["expected_error"] == "ARTIFACT_CHANGED"
            and check["registered_sha256"] != check["corrupt_sha256"]
            and inventory[name + "/" + check["intentionally_corrupt_file"]]
            == check["corrupt_sha256"],
            "Corruption proof",
        )
    elif name == "owned_jobs_and_isolation":
        require(
            [r["execution_status"] for r in check["runs"]]
            == ["cancelled", "cancelled", "completed", "interrupted", "timed_out"]
            and check["runs"][3]["error"] == "WORKER_INTERRUPTED"
            and check["running_creation_identity"] != check["interrupted_creation_identity"]
            and check["idempotency_conflict"] == "IDEMPOTENCY_CONFLICT",
            "Job proof",
        )
    elif name == "cumulative_run_budget":
        values = [s[1] for s in check["job_measurements"]]
        require(
            len(values) >= 2
            and all(a < b for a, b in zip(values, values[1:], strict=False))
            and values[-1] >= check["budget_seconds"]
            and rows[check["run"]["run_id"]][2] == values[-1],
            "Cumulative budget proof",
        )
    elif name == "multi_result_reports":
        require(
            check["primary"]["role"] == "atom_strain"
            and [s["role"] for s in check["all_strain"]] == ["atom_strain", "gpa"]
            and set(check["reports"]) == {"report_manifest", "report_html", "report_pdf"}
            and set(check["reports"].values()) <= set(inventory.values()),
            "Multi-result report digest",
        )
    elif name == "legacy_api_cli_yaml":
        require(
            rows[check["formal_run"]][0] == rows[check["minimal_run"]][0] == "completed"
            and check["old_report_sha256"] in inventory.values()
            and check["old_fft_report_sha256"] in inventory.values()
            and check["old_fft_image_sha256"] in inventory.values()
            and check["strict_schema_version"] == "0.1.0"
            and check["unknown_step_code"],
            "Legacy result evidence",
        )
    elif name == "local_nine_tool_protocol":
        from autotem.harness.cli import TOOL_SCHEMAS

        transcript = folder / "jsonl-trace.json"
        require(
            digest(transcript) == check["trace_sha256"]
            and len(json.loads(transcript.read_text("utf-8"))) == 20,
            "Protocol transcript",
        )
        require(
            set(check["jsonl"]) == TOOL_SCHEMAS.keys()
            and all(v == ["REQUEST_INVALID", True] for v in check["jsonl"].values())
            and check["mcp"]
            == {
                "status": "ready",
                "code": "LOCAL_MCP_TOOL_CALL_OK",
                "operation_count": 49,
                "tool_count": 9,
            },
            "Protocol contract",
        )
    elif name == "trusted_extension_contracts":
        require(
            check["actual_pixels"] == 5120
            and check["provenance"]["explicitly_enabled"]
            and inventory[name + "/" + Path(check["plugin_module"]).as_posix()]
            == check["plugin_source_sha256"]
            and set(check["errors"].values())
            == {
                "UNKNOWN_OPERATION",
                "DUPLICATE_OPERATION",
                "PLUGIN_NOT_FOUND",
                "PLUGIN_VERSION_REQUIRED",
                "INCOMPATIBLE_PLUGIN_CONTRACT",
                "INVALID_OPERATION_CONTRACT",
                "OPERATION_VERSION_MISMATCH",
                "dependency_unavailable",
                "dependency_broken",
            },
            "Trusted plugin result",
        )


if __name__ == "__main__" and sys.argv[1] == "--trusted-probe":
    print(json.dumps(plugin_probe(Path(sys.argv[2]), Path(sys.argv[3]))))
