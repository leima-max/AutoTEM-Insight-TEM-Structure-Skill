"""J06, from an uncalibrated TIFF to an immutable, calibrated child run."""

from pathlib import Path

from .bridge import submit_and_wait
from .cases import PIXEL_NM
from .evidence import audit_artifact, digest, require


def answer_calibration(call, output: Path, source: Path, initial_args: dict, parent: dict):
    parent_id = parent["run_id"]
    require(
        parent["status"] == "needs_input"
        and parent["next_action"] == "provide_required_input_in_child_run",
        "Missing calibration did not request input",
    )
    (request,) = parent["input_requests"]
    require(request["run_id"] == parent_id and request["status"] == "open", "Invalid input request")
    field = next(f for f in request["fields"] if f["name"] == "pixel_size_nm")
    schema = field["answer_schema"]["pixel_size_nm"]
    require(
        schema["value"]["exclusiveMinimum"] == 0 and "pm/pixel" in schema["units"],
        "Missing calibration answer units or positive constraint",
    )
    diagnostic = call("export_report", {"run_id": parent_id, "formats": ["json"]})
    require(
        diagnostic["report_type"] == "diagnostic", "Uncalibrated parent received a science report"
    )
    before = [
        audit_artifact(output, parent_id, a) for a in call("list_artifacts", {"run_id": parent_id})
    ]
    raw_id = next(a["artifact_id"] for a in before if a["role"] == "raw")
    require(
        raw_id in request["evidence_artifact_ids"]
        and all(a["role"] != "diffraction" for a in before),
        "Request did not cite raw evidence or parent claims a measurement",
    )
    snapshot = output / "harness" / "runs" / parent_id / "raw" / source.name
    snapshot_hash = digest(snapshot)
    require(snapshot_hash == digest(source), "Initial snapshot differs from the fixture")
    basis = "Known spatial scale supplied by the synthetic fixture author"
    reply = {
        "request_id": request["request_id"],
        "source": "user",
        "basis": basis,
        "answers": {"pixel_size_nm": {"value": 20, "unit": "pm/pixel"}},
        "raw_reply": "The acquisition scale is 20 pm per pixel.",
        "evidence_artifact_ids": [raw_id],
    }
    wrong = {
        **reply,
        "raw_reply": "Use -1 pm per pixel.",
        "answers": {"pixel_size_nm": {"value": -1, "unit": "pm/pixel"}},
    }
    plan = initial_args["plan"]
    call(
        "submit_analysis",
        {
            **initial_args,
            "parent_run_id": parent_id,
            "idempotency_key": "J06-invalid",
            "plan": {**plan, "user_input": {"response": wrong}},
        },
        expect_error="INVALID_INPUT_RESPONSE",
    )
    still_waiting = call("get_run_status", {"run_id": parent_id})
    (rejected,) = still_waiting["input_responses"]
    require(
        still_waiting["status"] == "needs_input"
        and rejected["child_run_id"] is None
        and rejected["validation"]["status"] == "rejected",
        "Invalid response advanced the run",
    )
    answered = {**plan, "user_input": {"response": reply}}
    validated = call("validate_plan", {"plan": answered})
    adopted = validated["input_response_validation"]["adopted_values"]["pixel_size_nm"]
    require(adopted["parameters"]["pixel_size_nm"] == PIXEL_NM, "Wrong pm/pixel conversion")
    args = {
        **initial_args,
        "plan": answered,
        "parent_run_id": parent_id,
        "idempotency_key": "J06-accepted",
    }
    child = submit_and_wait(call, args)
    child_id = child["run_id"]
    require(
        child_id != parent_id
        and child["parent_run_id"] == parent_id
        and child["status"] == "completed",
        "Accepted input did not produce a completed child",
    )
    (accepted,) = child["input_responses"]
    require(
        accepted["child_run_id"] == child_id
        and accepted["request_id"] == request["request_id"]
        and accepted["validation"]["status"] == "accepted"
        and accepted["response"]["source"] == "user"
        and accepted["response"]["basis"] == basis
        and accepted["response"]["answers"] == reply["answers"],
        "Child lost reply provenance",
    )
    parent_after = call("get_run_status", {"run_id": parent_id})
    require(
        parent_after["status"] == "needs_input"
        and parent_after["latest_response_child_run_id"] == child_id,
        "Parent no longer points at the correct child",
    )
    unchanged = [
        audit_artifact(output, parent_id, a) for a in call("list_artifacts", {"run_id": parent_id})
    ]
    require(
        unchanged == before and digest(snapshot) == snapshot_hash,
        "Child execution changed parent evidence",
    )
    return (
        child,
        args,
        {
            "parent_run_id": parent_id,
            "parent_status": parent_after["status"],
            "request_id": request["request_id"],
            "request_fields": request["fields"],
            "parent_snapshot_sha256": snapshot_hash,
            "parent_artifacts": before,
            "invalid_response_id": rejected["response_id"],
            "invalid_reply_code": "INVALID_INPUT_RESPONSE",
            "accepted_response_id": accepted["response_id"],
            "accepted_reply": reply,
            "adopted": adopted,
            "diagnostic_report_id": diagnostic["artifacts"][0]["artifact_id"],
        },
    )
