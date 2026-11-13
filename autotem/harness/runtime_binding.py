"""Bind managed project submissions to the runtime selected under the store write lock."""

import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from autotem.installation.host_entry import active_python

from .contracts import HarnessError


def selected_runtime(root: Path) -> str | None:
    control = root.parent
    if root.name != "runs" or control.name != ".autotem":
        return None
    pointer = control / "active-runtime.json"
    managed_process = Path(sys.prefix).parent == control / "runtimes"
    if not pointer.exists() and not managed_process:
        return None
    try:
        exe = active_python(control)
        if exe == Path(sys.executable):
            return Path(sys.prefix).name
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    raise HarnessError(
        "HOST_RECONNECT_REQUIRED", "Reconnect the host to the active project runtime."
    )


def record_job_runtime(db, root, run_id, job_id, plan, operations):
    runtime_id = selected_runtime(root)
    if runtime_id is None:
        return
    allowed = set(operations["enabled_plugins"])
    plugins = {
        ep.name: ep.dist.version
        for ep in importlib.metadata.entry_points(group="autotem.operations")
        if ep.name in allowed and ep.dist is not None
    }
    payload = {
        "job_id": job_id,
        "runtime_id": runtime_id,
        "package_version": importlib.metadata.version("autotem-insight"),
        "plugins": plugins,
        "plan_sha256": hashlib.sha256(plan.encode("utf-8")).hexdigest(),
    }
    db.execute(
        "INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
        (run_id, "runtime_bound", json.dumps(payload, sort_keys=True), time.time()),
    )
