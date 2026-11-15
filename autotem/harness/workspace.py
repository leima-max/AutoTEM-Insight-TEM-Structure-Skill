"""Project input boundary for jobs submitted by a workspace-limited host."""

import json
import os
import time
from pathlib import Path

from .contracts import HarnessError


def workspace_file(path: Path, workspace: Path | None) -> Path:
    if workspace is None:
        return path
    actual = os.path.realpath(path)
    if os.name == "nt":
        # The worker uses extended paths for long inputs. Compare ordinary spellings.
        if actual.startswith("\\\\?\\UNC\\"):
            actual = "\\\\" + actual[8:]
        elif actual.startswith("\\\\?\\"):
            actual = actual[4:]
    if not Path(actual).is_relative_to(workspace):
        raise HarnessError("WORKSPACE_INPUT_RESTRICTED",
                           "The input or a companion file resolves outside the project. "
                           "Use project-local files or a self-contained NPY.")
    return path


def record_workspace(db, run_id, job_id, workspace):
    if workspace is not None:
        payload = json.dumps({"job_id": job_id, "workspace": str(workspace)})
        db.execute(
            "INSERT INTO events(run_id,kind,payload,created) VALUES(?,?,?,?)",
            (run_id, "workspace_input_policy", payload, time.time()),
        )


def job_workspace(store, run_id, job_id):
    # One scheduler may serve both restricted and unrestricted jobs.
    with store.connect() as db:
        for row in db.execute(
            "SELECT payload FROM events WHERE run_id=? AND kind='workspace_input_policy' "
            "ORDER BY event_id DESC", (run_id,),
        ):
            event = json.loads(row[0])
            if event["job_id"] == job_id:
                return Path(event["workspace"])
    return None
