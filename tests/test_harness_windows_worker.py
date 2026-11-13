"""On Windows a venv python.exe redirects to a different interpreter process."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows venv redirector regression")
def test_venv_job_owns_the_actual_scientific_worker(tmp_path, isolated_harness_venv):
    python, _ = isolated_harness_venv
    raw = tmp_path / "input.npy"
    np.save(raw, np.arange(30, dtype=float).reshape(5, 6), allow_pickle=False)
    # The source test shares only its already installed test dependencies. Its real
    # venv redirector starts the public Harness, scheduler and worker normally.
    code = """
import json, sys
from autotem.harness import Harness
harness = Harness(sys.argv[1])
result = harness.inspect_input(sys.argv[2], modality="HRTEM")
owner = harness.store.get_run(result["run_id"])
started = [e["payload"] for e in result["events"] if e["kind"] == "worker_started"]
print(json.dumps({"status": result["status"], "error": result["error"],
                  "worker_pid": owner["worker_pid"], "worker_identity": owner["worker_identity"],
                  "events": started,
                  "shape": result["dataset"]["data"]["shape"] if result["dataset"] else None,
                  "prefix": sys.prefix, "isolated": sys.flags.isolated}))
"""
    completed = subprocess.run(
        [str(python), "-I", "-c", code, str(tmp_path / "harness"), str(raw)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    result = json.loads(completed.stdout)
    assert Path(result["prefix"]).resolve() == python.parent.parent.resolve()
    assert result["isolated"] == 1
    assert result["status"] == "completed", result["error"]
    assert result["shape"] == [5, 6]
    assert result["events"] == [
        {"pid": result["worker_pid"], "process_identity": result["worker_identity"]}
    ]
    assert result["worker_identity"] and result["worker_pid"] != os.getpid()
