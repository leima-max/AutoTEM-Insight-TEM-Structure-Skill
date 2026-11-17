"""Bind the Windows cold-install and disk evidence to one immutable release candidate."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def operation_coverage(rows, evidence, install, identity):
    results = {
        name: {
            "operation_version": row["version"],
            "verdict": row["disk_execution"],
            "evidence": "science/operations.json",
        }
        for name, row in rows.items()
        if row["implementation_status"] == "implemented"
    }
    if any(row["negative_contract"]["verdict"] != "passed" for row in rows.values()):
        raise ValueError("At least one operation did not pass its negative contract.")
    hashes = {}
    if "installed_operation_variants" in install["checks"]:
        variant_path = evidence / "variants/operations.json"
        hashes["variants/operations.json"] = digest(variant_path)
        record = read(variant_path)
        if (
            hashes["variants/operations.json"] != install["variant_evidence_sha256"]
            or record["case_verdict"] != "passed"
            or record["runtime"]["release_identity"] != identity
            or record["driver"]["source_archive_match"] is not True
        ):
            raise ValueError("Variant evidence belongs to a different install or verdict.")
        for name, row in record["operations"].items():
            binding = row["release_binding"]
            if (
                name not in results
                or binding["operation_version"] != rows[name]["version"]
                or any(
                    binding[key] != identity[key]
                    for key in ("manifest_sha256", "wheel_sha256", "lock_sha256")
                )
            ):
                raise ValueError(f"Variant release or operation version mismatch: {name}")
            if row["verdict"] == "passed":
                path = (variant_path.parent / row["case_evidence"]).resolve()
                if (
                    not path.is_relative_to(variant_path.parent.resolve())
                    or digest(path) != row["case_evidence_sha256"]
                    or not row["run_id"]
                    or not row["artifacts"]
                ):
                    raise ValueError(f"Variant run evidence mismatch: {name}")
                hashes[path.relative_to(evidence.resolve()).as_posix()] = row[
                    "case_evidence_sha256"
                ]
            results[name].update(verdict=row["verdict"], evidence="variants/operations.json")
    return {
        "implemented": len(results),
        "disk_execution_passed": sum(r["verdict"] == "passed" for r in results.values()),
        "reserved_rejected": sum(r["implementation_status"] == "reserved" for r in rows.values()),
        "by_operation": results,
    }, hashes


def summarize(bundle, evidence):
    manifest_path = bundle / "release-manifest.json"
    manifest = read(manifest_path)
    install_path = evidence / "installation.json"
    science_path = evidence / "science/evidence.json"
    matrix_path = evidence / "science/operations.json"
    install, science = read(install_path), read(science_path)
    if install["case_verdict"] != "passed" or science["case_verdict"] != "passed":
        raise ValueError("Installation and installed disk acceptance must both have passed.")
    fingerprint = digest(manifest_path)
    profile = manifest["profiles"]["traditional-core-mcp"]
    runtime = science["runtime"]
    identity = runtime["release_identity"]
    app_wheels = [
        row
        for row in manifest["artifacts"]
        if Path(row["path"]).parent.as_posix() == profile["wheelhouse"]
        and Path(row["path"]).name.startswith("autotem_insight-")
        and row["path"].endswith(".whl")
    ]
    if (
        install["release_manifest_sha256"] != fingerprint
        or len(app_wheels) != 1
        or identity["profile"] != "traditional-core-mcp"
        or identity["wheel_sha256"] != app_wheels[0]["sha256"]
        or identity["wheel_sha256"] != digest(bundle / app_wheels[0]["path"])
        or identity["manifest_sha256"] != fingerprint
        or identity["release_id"] != manifest["release_id"]
        or identity["source_archive_sha256"] != manifest["source"]["archive_sha256"]
        or identity["lock_sha256"] != profile["lock_sha256"]
        or identity["lock_sha256"] != digest(bundle / profile["lock"])
        or identity["source_archive_sha256"] != digest(bundle / manifest["source"]["archive"])
        or install["science_evidence_sha256"] != digest(science_path)
        or science["operations"]["sha256"] != digest(matrix_path)
    ):
        raise ValueError("Evidence or release fingerprint mismatch.")
    if not runtime["private_install_verified"] or not runtime["isolated"] or runtime["editable"]:
        raise ValueError("The recorded interpreter must be an isolated installed wheel.")
    cases = {case["case_id"]: case for case in science["cases"]}
    if not {"J01", "J02", "J06"} <= cases.keys() or any(
        case["case_verdict"] != "passed" for case in cases.values()
    ):
        raise ValueError("Required disk journeys are incomplete.")
    checks = {
        "read_only_preview",
        "cold_installed_private_cpython",
        "activated_host_entry",
        "installed_disk_journeys",
        "project_bound_job_and_artifacts",
        "idempotent_setup",
    }
    if not checks <= set(install["checks"]):
        raise ValueError("The cold installer and stable entry checks are required.")
    operations, extra_hashes = operation_coverage(
        read(matrix_path)["operations"], evidence, install, identity
    )
    return {
        "schema_version": "1.0.0",
        "release_id": manifest["release_id"],
        "release_manifest_sha256": fingerprint,
        "target": manifest["target"],
        "profile": identity["profile"],
        "wheel_sha256": identity["wheel_sha256"],
        "lock_sha256": identity["lock_sha256"],
        "source_archive_sha256": identity["source_archive_sha256"],
        "private_install_and_disk_verdict": "passed",
        "codex_user_journey": "not_evaluated",
        "first_slice_verdict": "not_evaluated",
        "disk_cases": {
            k: {"case_verdict": v["case_verdict"], "run_id": v["run_id"]} for k, v in cases.items()
        },
        "operations": operations,
        "evidence_sha256": {
            "installation.json": digest(install_path),
            "science/evidence.json": digest(science_path),
            "science/operations.json": digest(matrix_path),
            **extra_hashes,
        },
        "gates": {
            "RG0": "not_evaluated",
            "RG1": "passed",
            "RG2": "not_evaluated",
            "RG3": "not_evaluated",
            "RG4": "not_evaluated",
            "RG5": "not_evaluated",
        },
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New detached summary file")
    args = parser.parse_args()
    try:
        result = summarize(args.bundle, args.evidence)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Release evidence verification failed: {exc}\n")
