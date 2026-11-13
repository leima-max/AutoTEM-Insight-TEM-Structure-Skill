"""Index private-verifier host summaries for exactly one release and host version per host."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MATRIX_FILE = Path(__file__).with_name("matrix.json")
MATRIX = json.loads(MATRIX_FILE.read_text("utf-8"))
VERDICTS = ("failed", "blocked", "not_evaluated", "passed")


def read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    return json.loads(path.read_text("utf-8-sig"), object_pairs_hook=unique)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(states):
    return next((state for state in VERDICTS if state in states), "not_evaluated")


def build(
    release, reports=(), *, versions=None, native=(), discovery=None, historical=(), observations=()
):
    manifest_file = release / "release-manifest.json"
    manifest = read_json(manifest_file)
    identity = file_hash(manifest_file)
    versions = dict(versions or {})
    native = set(native)
    discovery = dict(discovery or {})
    if (native | set(versions) | set(discovery)) - set(MATRIX["hosts"]):
        raise ValueError("Unknown host")
    inventory = {}
    fingerprints = {}
    matrix_hash = file_hash(MATRIX_FILE)
    simulation = "traditional-simulation" in manifest["profiles"]
    records = [(read_json(path), file_hash(path)) for path in reports]
    for report in observations:
        encoded = json.dumps(report, sort_keys=True, allow_nan=False).encode("utf-8")
        records.append((report, hashlib.sha256(encoded).hexdigest()))
    for report, summary_hash in records:
        host, case = report["host"], report["case_id"]
        variant, verdict = report.get("variant"), report["case_verdict"]
        if (
            report.get("execution_layer") != "real_host_capture"
            or report.get("schema_version") != "1.0.0"
            or host not in MATRIX["hosts"]
            or case not in MATRIX["cases"]
            or verdict not in VERDICTS
            or report.get("release_ready") is not False
            or variant is not None
            and variant not in MATRIX["cases"][case]
        ):
            raise ValueError("Invalid host verifier summary")
        evidence = report["evidence"]
        if evidence.get("matrix_sha256") != matrix_hash or any(
            not re.fullmatch(r"[a-f0-9]{64}", evidence.get(k, ""))
            for k in ("case_record_sha256", "verifier_sha256")
        ):
            raise ValueError("Missing capture or verifier fingerprint, or incompatible case matrix")
        runtime = report.get("runtime", {})
        observed = runtime.get("release_identity", {})
        if (
            observed
            and observed.get("manifest_sha256") != identity
            or report.get("attempted_release_manifest_sha256", identity) != identity
        ):
            raise ValueError("Mixed release identities")
        version = report.get("host_version")
        if version:
            if host in versions and versions[host] != version:
                raise ValueError("Mixed host versions")
            versions[host] = version
        if verdict == "blocked" and not evidence.get("blocker_sha256"):
            raise ValueError("Blocked observation has no source")
        if verdict == "passed":
            profile = observed.get("profile")
            wheel_hashes = {
                a["sha256"]
                for a in manifest["artifacts"]
                if Path(a["path"]).name.startswith("autotem_insight-")
                and a["path"].endswith(".whl")
            }
            if (
                not version
                or not observed
                or profile not in manifest["profiles"]
                or observed.get("release_id") != manifest["release_id"]
                or observed.get("source_archive_sha256") != manifest["source"]["archive_sha256"]
                or observed.get("package_version") != manifest["package_version"]
                or observed.get("wheel_sha256") not in wheel_hashes
                or observed.get("lock_sha256") != manifest["profiles"][profile]["lock_sha256"]
                or report.get("scientific_accuracy") != "not_evaluated"
                or not runtime.get("private_install_verified")
                or not report.get("tool_trace")
                or not evidence.get("turns")
                or any(
                    report["checks"].get(k) != "passed"
                    for k in (
                        "host_tool_call",
                        "case_execution",
                        "answer_return",
                        "release_and_resources",
                        "effective_tool_permission",
                        "session_skill_discovery",
                    )
                )
            ):
                raise ValueError("Passed observation lacks verified release or host checks")
            if host in fingerprints and fingerprints[host] != runtime["skill_resource_sha256"]:
                raise ValueError("Mixed deployed Skill or Profile identities")
            fingerprints[host] = runtime["skill_resource_sha256"]
        key = (host, case, variant)
        summary = {
            "verdict": verdict,
            "summary_sha256": summary_hash,
            "case_record_sha256": evidence["case_record_sha256"],
            "verifier_sha256": evidence["verifier_sha256"],
            "host_version": version,
            "capture_stage": report["capture_stage"],
            "reason": report.get("reason"),
            "release_binding": "verified"
            if observed
            else "attempted"
            if report.get("attempted_release_manifest_sha256")
            else "unbound",
        }
        if key in inventory and inventory[key] != summary:
            raise ValueError("Conflicting summaries for one host, case and variant")
        inventory[key] = summary
    hosts = {}
    totals = Counter()
    for host in MATRIX["hosts"]:
        cases = {}
        for case, variants in MATRIX["cases"].items():
            rows = []
            for variant in variants or [None]:
                required = not (
                    case == "J14"
                    and variant == "native"
                    and host not in native
                    or case == "J15"
                    and not simulation
                    and variant != "missing_dependency"
                )
                item = inventory.get((host, case, variant), {"verdict": "not_evaluated"})
                rows.append({"variant": variant, "required": required, **item})
                if required:
                    totals[item["verdict"]] += 1
            states = [r["verdict"] for r in rows if r["required"]]
            unspecified = inventory.get((host, case, None)) if variants else None
            if unspecified and unspecified["verdict"] in {"blocked", "failed"}:
                states.append(unspecified["verdict"])
            cases[case] = {"verdict": aggregate(states), "variants": rows}
            if unspecified:
                cases[case]["unspecified_variant"] = unspecified
        hosts[host] = {
            "host_version": versions.get(host),
            "cases": cases,
            "journey_verdict": aggregate([c["verdict"] for c in cases.values()]),
        }
        if host in discovery:
            artifact = discovery[host]
            if not artifact.is_file() or not artifact.stat().st_size:
                raise ValueError("Missing local host discovery evidence")
            hosts[host]["local_discovery"] = {
                "source_sha256": file_hash(artifact),
                "host_session": "not_evaluated",
                "release_binding": "unbound",
            }
    previous = []
    for path in historical:
        old = read_json(path)
        if old.get("execution_layer") != "host_evidence_summary_index" or not re.fullmatch(
            r"[0-9a-f]{64}", old.get("release_manifest_sha256", "")
        ):
            raise ValueError("Invalid historical host index")
        previous.append(
            {
                "index_sha256": file_hash(path),
                "release_id": old["release_id"],
                "release_manifest_sha256": old["release_manifest_sha256"],
                "host_versions": old["host_versions"],
                "previous_matrix_verdict": old["host_journey_matrix"],
            }
        )
    return {
        "schema_version": "1.0.0",
        "execution_layer": "host_evidence_summary_index",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_id": manifest["release_id"],
        "release_manifest_sha256": identity,
        "matrix_sha256": matrix_hash,
        "declared_profiles": sorted(manifest["profiles"]),
        "host_versions": versions,
        "hosts": hosts,
        "historical_context_only": previous,
        "required_variant_counts": {k: totals[k] for k in VERDICTS},
        "host_journey_matrix": aggregate([h["journey_verdict"] for h in hosts.values()]),
        "scientific_accuracy": "not_evaluated",
        "release_ready": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--report", action="append", type=Path, default=[])
    parser.add_argument("--host-version", action="append", default=[], metavar="HOST=VERSION")
    parser.add_argument("--native", action="append", choices=MATRIX["hosts"], default=[])
    parser.add_argument("--discovery", action="append", default=[], metavar="HOST=LOCAL_CAPTURE")
    parser.add_argument("--historical-index", action="append", type=Path, default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        versions = {}
        for value in args.host_version:
            host, version = value.split("=", 1)
            if host in versions or not re.fullmatch(r"[0-9][0-9a-zA-Z.\-+]{1,79}", version):
                raise ValueError("Invalid host version filter")
            versions[host] = version
        discovery = {}
        for value in args.discovery:
            host, file = value.split("=", 1)
            if host in discovery:
                raise ValueError("Duplicate host discovery capture")
            discovery[host] = Path(file)
        report = build(
            args.release,
            args.report,
            versions=versions,
            native=args.native,
            discovery=discovery,
            historical=args.historical_index,
        )
        content = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
        if args.out:
            with args.out.open("x", encoding="utf-8") as stream:
                stream.write(content + "\n")
        else:
            print(content)
        return 0 if report["host_journey_matrix"] == "passed" else 1
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
        parser.exit(2, f"Invalid acceptance index: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
