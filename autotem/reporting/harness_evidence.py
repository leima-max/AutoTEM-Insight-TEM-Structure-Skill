"""Assemble report claims from final step records and their immutable artifacts."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

STRAIN_ROLES = {"strain.gpa": "gpa", "strain.atom": "atom_strain"}


def _measurement_examples(payload: dict, artifact_id: str) -> list[dict]:
    """Bound the displayed examples. Each path addresses the complete registered JSON."""
    examples = []
    for key, value in payload.items():
        if (
            not isinstance(value, dict)
            or not {"value", "unit", "method", "uncertainty"} <= value.keys()
        ):
            continue
        unit = value.get("unit")
        if not unit or not value.get("method") or value["value"] is None:
            continue
        pointer = "/" + key.replace("~", "~0").replace("/", "~1")
        examples.append(
            {
                "artifact_id": artifact_id,
                "pointer": pointer,
                "value": value["value"],
                "unit": unit,
                "method": value["method"],
                "parameters": value.get("parameters", {}),
                "uncertainty": value["uncertainty"],
                "uncertainty_status": "not_evaluated"
                if value["uncertainty"] is None
                else "reported",
                "provenance": value.get("provenance", {}),
                "warnings": value.get("warnings", []),
            }
        )
    return examples


def delivery_index(run: dict, state: dict, evidence: list[dict], load_json) -> dict:
    request = run["request"]
    steps = {step["step_id"]: step for step in state["steps"]}
    by_step = defaultdict(list)
    for artifact in evidence:
        by_step[artifact["step_id"]].append(artifact)
    goals = {}
    for event in state["events"]:
        if event["kind"] == "work_package_submitted":
            package = event["payload"]
            goals.update(
                (s["id"], package.get("user_request")) for s in package["request"]["steps"]
            )

    claims = []
    results = defaultdict(list)
    strain = []
    for i, planned in enumerate(request["steps"]):
        sid = planned["id"]
        op = planned["operation_id"]
        step = steps.get(sid, {})
        status = step.get("status", "not_executed")
        result = step.get("result") or {}
        refs = result.get("artifact_ids")
        by_role = defaultdict(list)
        diagnostic = []
        metrics = []
        for a in by_step[sid]:
            aid, role = a["artifact_id"], a["role"]
            if (
                status != "completed"
                or not a["execution_usable"]
                or (refs is not None and aid not in refs)
            ):
                diagnostic.append(aid)
                continue
            by_role[role].append(aid)
            results[role].append({"step_id": sid, "operation_id": op, "artifact_id": aid})
            if a["kind"] == "json" and a["size"] <= 256_000:
                payload = load_json(aid)
                if isinstance(payload, dict):
                    metrics.extend(_measurement_examples(payload, aid))
        present = {aid for ids in by_role.values() for aid in ids}
        missing = sorted(set(refs or ()) - present)
        delivered = status == "completed" and bool(present) and not missing
        row = {
            "user_request": goals.get(sid, run.get("user_request")),
            "step_id": sid,
            "operation_id": op,
            "operation_version": planned.get("operation_version"),
            "required": not planned.get("optional", False),
            "status": status,
            "delivered": delivered,
            "parameters": planned["parameters"],
            "parameter_source": f"/request/steps/{i}/parameters",
            "consumed_artifacts": result.get("consumed_artifacts", {}),
            "artifacts_by_role": dict(by_role),
            "missing_artifact_ids": missing,
            "diagnostic_artifact_ids": diagnostic,
            "measurement_examples": metrics,
            "scientific_level": result.get("scientific_level"),
            "warnings": result.get("warnings", []),
            "error": step.get("error"),
        }
        claims.append(row)
        strain_role = STRAIN_ROLES.get(op)
        if strain_role and delivered:
            for aid in by_role.get(strain_role, []):
                strain.append(
                    {
                        "step_id": sid,
                        "operation_id": op,
                        "role": strain_role,
                        "artifact_id": aid,
                        "required": row["required"],
                        "scientific_level": row["scientific_level"],
                        "warnings": row["warnings"],
                    }
                )
    required = [c for c in claims if c["required"]]
    unmet = [c["step_id"] for c in required if not c["delivered"]]
    primary = next((s for s in strain if s["required"]), strain[0] if strain else None)
    return {
        "user_request": run.get("user_request"),
        "claim_evidence": claims,
        "results_by_role": dict(results),
        "strain_results": {
            "primary": primary,
            "all": strain,
            "selection": "first delivered required strain in plan order, then optional",
        },
        "completeness": {
            "requested_steps": len(claims),
            "completed_steps": sum(c["status"] == "completed" for c in claims),
            "delivered_steps": sum(c["delivered"] for c in claims),
            "required_steps": len(required),
            "delivered_required_steps": len(required) - len(unmet),
            "unmet_required_steps": unmet,
            "required_complete": not unmet,
            "optional_skipped": [
                c["step_id"] for c in claims if not c["required"] and c["status"] == "skipped"
            ],
        },
    }


def claim_html(manifest: dict, evidence: list[dict]) -> str:
    artifacts = {a["artifact_id"]: Path(a["path"]).name for a in evidence}
    completeness = manifest["completeness"]
    parts = [
        f"<h2>Requested results</h2><p>Required steps delivered "
        f"{completeness['delivered_required_steps']} / {completeness['required_steps']}</p>"
    ]
    primary = manifest["strain_results"]["primary"]
    if primary:
        parts.append(
            f"<p>Primary strain result: {html.escape(primary['step_id'])} "
            f"({html.escape(primary['role'])})</p>"
        )
    for claim in manifest["claim_evidence"]:
        links = [
            f'<a href="{html.escape(artifacts[aid], quote=True)}">{html.escape(role)} ({aid})</a>'
            for role, ids in claim["artifacts_by_role"].items()
            for aid in ids
        ]
        parts.append(
            f"<section><h3>{html.escape(claim['step_id'])} "
            f"({html.escape(claim['operation_id'])})</h3><p>"
            + (" · ".join(links) if links else "No delivered evidence")
            + "</p><pre>"
            + html.escape(json.dumps(claim, ensure_ascii=False, indent=2))
            + "</pre></section>"
        )
    history = manifest.get("input_response_history", [])
    if history:
        parts.append("<section><h2>Scientific input history</h2><ol>")
        for entry in history:
            parts.append(
                f"<li><strong>{html.escape(entry['step_id'])} "
                f"({html.escape(entry['operation_id'])})</strong>"
                f"<p>{html.escape(entry['source'])} · {html.escape(entry['basis'])}</p><pre>"
                + html.escape(json.dumps(entry, ensure_ascii=False, indent=2))
                + "</pre></li>"
            )
        parts.append("</ol></section>")
    return "".join(parts)
