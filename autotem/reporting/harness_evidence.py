"""Assemble report claims from final step records and their immutable artifacts."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from itertools import islice
from pathlib import Path

from autotem.harness.gpa_evidence import gpa_result_mask

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
        gpa_mask_references = {}
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
                    if role == "gpa":
                        gpa_mask_references[aid] = payload.get("provenance", {}).get("quality_mask")
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
                entry = {
                    "step_id": sid,
                    "operation_id": op,
                    "role": strain_role,
                    "artifact_id": aid,
                    "required": row["required"],
                    "scientific_level": row["scientific_level"],
                    "warnings": row["warnings"],
                }
                if strain_role == "gpa":
                    mask_ids = by_role.get("gpa.quality_mask", [])
                    masks = [a for a in by_step[sid] if a["artifact_id"] in mask_ids]
                    entry["validity_mask"] = gpa_result_mask(gpa_mask_references.get(aid), masks)
                strain.append(entry)
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
    for result in manifest["strain_results"]["all"]:
        mask = result.get("validity_mask")
        if mask:
            label = f"GPA quality mask ({html.escape(result['step_id'])})"
            if mask["artifact_id"]:
                url = html.escape(artifacts[mask["artifact_id"]], quote=True)
                label = f'<a href="{url}">{label}</a>'
            parts.append(f"<p>{label}. {html.escape(mask['interpretation'])}</p>")
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
            + "</p><details><summary>Step parameters and provenance</summary><pre>"
            + html.escape(json.dumps(claim, ensure_ascii=False, indent=2))
            + "</pre></details></section>"
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


def _number(value) -> str:
    return format(value, ".6g") if type(value) in (int, float) else "—"


def _diffraction_html(result: dict, artifact: dict) -> str:
    """Display the chosen result's own measurements, directions and candidates."""
    spacings = result.get("d_spacings") or []
    candidates = result.get("phase_candidates") or []
    if not spacings and not candidates:
        return ""
    parts = ["<section><h2>Diffraction measurements</h2>"]
    if spacings:
        # A direct-beam or rejected peak can have no spacing. The compact
        # d_spacings list alone cannot identify its source indices.
        by_peak = result.get("provenance", {}).get("d_by_peak")
        indexed = isinstance(by_peak, list) and [d for d in by_peak if d is not None] == spacings
        angles = {
            g["source_peak_index"]: g.get("angle") or {}
            for g in result.get("g_vectors", [])
            if g.get("source_peak_index") is not None
        }
        rows = ((i, d) for i, d in enumerate(by_peak) if d is not None) if indexed else (
            (None, d) for d in spacings
        )
        preview = list(islice(rows, 6))
        methods = {d.get("method") for _, d in preview}
        shared_method = len(methods) == 1
        method_heading = "" if shared_method else "<th>Method</th>"
        parts.append('<div class="table-scroll"><table class="measurements"><thead><tr>'
                     "<th>Peak</th><th>Azimuth</th><th>Spacing</th>"
                     f"<th>Uncertainty</th>{method_heading}</tr></thead><tbody>")
        for peak, d in preview:
            angle = angles.get(peak, {})
            angle_value = angle.get("value")
            angle_text = _number(angle_value)
            if angle.get("unit") == "deg" and type(angle_value) in (int, float):
                angle_text = _number(round(angle_value, 3) or 0)
            azimuth = f"{angle_text} {html.escape(str(angle.get('unit') or ''))}"
            unit = html.escape(str(d.get("unit") or ""))
            uncertainty = d.get("uncertainty")
            uncertainty_text = (
                "Not evaluated" if uncertainty is None else f"{_number(uncertainty)} {unit}"
            )
            method = html.escape(d.get("method") or "—")
            method_cell = "" if shared_method else f"<td>{method}</td>"
            parts.append(
                f"<tr><td>{peak if peak is not None else '—'}</td>"
                f"<td>{azimuth}</td>"
                f"<td>{_number(d.get('value'))} {unit}</td>"
                f"<td>{uncertainty_text}</td>{method_cell}</tr>"
            )
        parts.append("</tbody></table></div>")
        if shared_method:
            parts.append(f"<p>Method · {html.escape(next(iter(methods)) or '—')}</p>")
        if len(spacings) > 6:
            parts.append(f"<p>Showing 6 of {len(spacings)} measured spacings.</p>")
    if candidates:
        parts.append("<h3>Phase candidates</h3><ul>")
        for candidate in candidates[:5]:
            name = html.escape(str(candidate.get("name") or "Unnamed candidate"))
            count = candidate.get("matched_spot_count")
            matched = f" · {count} matched spots" if type(count) is int else ""
            parts.append(f"<li>{name}{matched}</li>")
        parts.append("</ul><p>Candidate assignments depend on the structures and indexing "
                     "parameters used in this run.</p>")
        if len(candidates) > 5:
            parts.append(f"<p>Showing 5 of {len(candidates)} candidates.</p>")
    warnings = result.get("warnings") or []
    if warnings:
        parts.append("<ul>" + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings[:3])
                     + "</ul>")
        if len(warnings) > 3:
            parts.append("<details><summary>Additional warnings</summary><ul>"
                         + "".join(f"<li>{html.escape(str(w))}</li>" for w in warnings[3:])
                         + "</ul></details>")
    aid = html.escape(artifact["artifact_id"])
    url = html.escape(Path(artifact["path"]).name, quote=True)
    step = html.escape(artifact["step_id"])
    parts.append(f'<p>Source · {step} · <a href="{url}">{aid}</a></p></section>')
    return "".join(parts)


def render_harness_html(manifest: dict, load_json) -> str:
    """Render a portable report from the terminal manifest and registered artifacts."""
    evidence = manifest["artifacts"]
    by_id = {a["artifact_id"]: a for a in evidence}
    status = manifest["run"]
    report_type = manifest["report_type"]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>AutoTEM evidence report</title><style>"
        "body{font:1rem/1.5 Inter,system-ui,sans-serif;color:#15292c;max-width:72rem;"
        "margin:2rem auto;padding:0 1rem;overflow-wrap:anywhere}"
        "h1,h2,h3{line-height:1.2}h2{margin-top:1.8rem}a{color:#075d79}"
        "table{border-collapse:collapse;max-width:100%}th,td{padding:.35rem .55rem;"
        "text-align:left;border-bottom:1px solid #cad5d6;vertical-align:top}"
        "th{background:#eff5f4}.table-scroll{overflow-x:auto}.measurements{overflow-wrap:normal}"
        ".measurements td:nth-child(-n+3){white-space:nowrap}"
        "summary{cursor:pointer}pre{white-space:pre-wrap;"
        "overflow-wrap:anywhere}figure{margin:1rem 0}img{max-width:100%;height:auto}"
        "</style></head><body>",
        f"<h1>AutoTEM {html.escape(report_type)} report</h1>"
        f"<p>{html.escape(status['status'])} · scientific accuracy: "
        f"{html.escape(manifest['scientific_accuracy'])}</p>",
        f"<p>Run <code>{html.escape(status['run_id'])}</code></p>",
    ]
    missing = manifest["completeness"]["unmet_required_steps"]
    if missing:
        names = ", ".join(map(html.escape, missing))
        parts.append(f"<p>Undelivered required steps · {names}</p>")
    summaries = []
    for role in ("atom_distances", "diffraction", "atomic_intensity", "gpa", "atom_strain"):
        aid = manifest["result_index"].get(role)
        if not aid:
            continue
        value = load_json(aid)
        if role == "diffraction" and report_type == "analysis":
            parts.append(_diffraction_html(value, by_id[aid]))
        summaries.append(
            f"<details><summary>{html.escape(role)} result JSON</summary><pre>"
            + html.escape(json.dumps(value, ensure_ascii=False, indent=2)[:50000])
            + "</pre><p>Complete JSON is linked in the evidence table.</p></details>"
        )
    delivered = {c["step_id"]: c["artifacts_by_role"] for c in manifest["claim_evidence"]
                 if c["delivered"]}
    tables = [
        f'<li>{html.escape(a["step_id"])} · <a href="{html.escape(Path(a["path"]).name)}">'
        f'{html.escape(a["role"])} ({html.escape(a["artifact_id"])})</a></li>'
        for a in evidence if a["kind"] == "table"
        and a["artifact_id"] in delivered.get(a["step_id"], {}).get(a["role"], ())
    ]
    if tables:
        parts.append("<h2>Data tables</h2><ul>" + "".join(tables) + "</ul>")
    parts.append(claim_html(manifest, evidence))
    parts.append(
        "<details><summary>Run status and events</summary><pre>"
        + html.escape(json.dumps(status, ensure_ascii=False, indent=2)) + "</pre></details>"
    )
    parts.append("<h2>Registered evidence</h2><table><tr><th>Role</th><th>Type</th>"
                 "<th>Step status</th><th>Evidence</th><th>SHA256</th></tr>")
    for a in evidence:
        url = html.escape(Path(a["path"]).name, quote=True)
        parts.append(
            f"<tr><td>{html.escape(a['role'])}</td><td>{html.escape(a['kind'])}</td>"
            f"<td>{html.escape(a['operation_status'])}</td>"
            f'<td><a href="{url}">{html.escape(a["artifact_id"])}</a></td>'
            f"<td>{html.escape(a['sha256'])}</td></tr>"
        )
    parts.append("</table>")
    parts.extend(summaries)
    for a in evidence:
        if a["kind"] == "image":
            url = html.escape(Path(a["path"]).name, quote=True)
            role = html.escape(a["role"])
            parts.append(f'<figure><img width="600" src="{url}" alt="{role}">'
                         f"<figcaption>{role}</figcaption></figure>")
    return "".join(parts) + "</body></html>"
