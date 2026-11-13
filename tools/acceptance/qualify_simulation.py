"""Execute and recheck the two conditional operations in a managed simulation edition."""

import argparse
import hashlib
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from autotem.acceptance.driver import runtime_record
from autotem.acceptance.evidence import digest, require
from autotem.core import DiffractionResult, SimulationResult
from autotem.diffraction.phase_matching import match_diffsims_patterns_to_diffraction
from autotem.installation.host_entry import active_python
from autotem.installation.release import verify_release, wheel_identity

PROFILE = "traditional-simulation"
OPERATIONS = ("simulation.diffraction", "diffraction.match")
BACKENDS = {"numpy", "diffsims", "orix", "diffpy-structure", "numba", "llvmlite", "pycifrw"}


def read(path):
    return json.loads(path.read_bytes())


def sibling(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def installed_backends(verified, python):
    site = python.parents[1] / "Lib/site-packages"
    versions = {}
    for wheel in verified.wheelhouse.glob("*.whl"):
        name, version = wheel_identity(wheel)
        if name in BACKENDS:
            with ZipFile(wheel) as archive:
                (metadata,) = [p for p in archive.namelist() if p.endswith(".dist-info/METADATA")]
                require(
                    (site / metadata).read_bytes() == archive.read(metadata),
                    f"Installed {name} metadata differs from the locked wheel",
                )
            versions[name] = version
    require(set(versions) == BACKENDS, "Missing locked simulation backend")
    return versions


def numerical_evidence(frame):
    simulation = frame.value(OPERATIONS[0], "simulation")
    (pattern,) = simulation["patterns"]
    image = frame.array(OPERATIONS[0], "simulated_pattern_0")
    require(
        simulation["provenance"]["adapter"]["active"] is True
        and pattern["simulation_parameters"]["structure_source"] == "candidate_atom_sites"
        and pattern["simulation_parameters"]["atom_site_count"] == 1
        and image.shape == (64, 64)
        and np.isfinite(image).all()
        and image.max() > 0,
        "Missing full explicit-site simulation",
    )
    spots = pattern["spots"]
    require(
        spots
        and all(s[k]["unit"] == "1/nm" for s in spots for k in ("gx", "gy", "magnitude"))
        and all(s["d_spacing"]["unit"] == "nm" for s in spots),
        "Invalid reciprocal units",
    )
    vectors = np.array([[s["gx"]["value"], s["gy"]["value"]] for s in spots])
    lengths = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose([s["magnitude"]["value"] for s in spots], lengths, rtol=1e-12)
    np.testing.assert_allclose([s["d_spacing"]["value"] for s in spots], 1 / lengths, rtol=1e-12)
    match = frame.value(OPERATIONS[1], "diffraction")
    direct = match_diffsims_patterns_to_diffraction(
        DiffractionResult.model_validate(frame.value("diffraction.index", "diffraction")),
        SimulationResult.model_validate(simulation).patterns,
        min_matched_spots=3,
        d_tolerance_rel=0.06,
        previous_score_weight=0.0,
    )
    best = match["phase_candidates"][0]
    require(
        match["provenance"]["phase_matching"]["active"]
        and direct.provenance["active"]
        and best == direct.candidates[0].model_dump(mode="json")
        and best["name"] == "synthetic-cubic"
        and best["matched_spot_count"] == 4
        and best["mean_d_error"]["unit"] == "relative"
        and best["mean_d_error"]["value"] < 0.06
        and best["mean_angle_error"]["unit"] == "deg"
        and best["mean_angle_error"]["value"] < 1e-9,
        "Registered candidate differs from the direct match or analytic fixture",
    )
    return {
        "shape": list(image.shape),
        "simulated_spots": len(spots),
        "matched_spots": 4,
        "candidate": best["name"],
        "mean_d_error": best["mean_d_error"],
        "mean_angle_error": best["mean_angle_error"],
    }


def check_fixture_and_plan(raw, request, fixture):
    yy, xx = np.indices(fixture["shape"])
    expected = np.cos(2 * np.pi * xx / 8) + np.cos(2 * np.pi * yy / 8)
    np.testing.assert_allclose(np.load(raw, allow_pickle=False), expected, rtol=0, atol=1e-14)
    params = {step["operation_id"]: step["parameters"] for step in request["steps"]}
    (cell,) = params["diffraction.index"]["candidate_phase_params"]
    (site,) = cell["atom_sites"]
    settings = fixture["runtime_settings"]
    require(
        request["modality"] == fixture["modality"]
        and request["budgets"]["simulation_seconds"] == settings["simulation_seconds"]
        and params[OPERATIONS[0]]["diffsims_timeout_s"] == settings["diffsims_timeout_s"]
        and params["calibration.resolve"]["pixel_size_nm"] == fixture["pixel_size_nm"]
        and all(cell[axis] == fixture["cell_edge_nm"] for axis in ("a", "b", "c"))
        and cell["space_group_number"] == 221
        and site["element"] == "Si"
        and site["abc"] == [0, 0, 0]
        and site["occupancy"] == 1.0,
        "Original simulated structure, calibration or bounded computation differs",
    )


def summary(bundle, project, case_path):
    verified = verify_release(bundle, profile=PROFILE)
    manifest = read(bundle / "release-manifest.json")
    root = project / ".autotem/runs"
    python = active_python(root.parent)
    pointer = read(root.parent / "active-runtime.json")
    marker = read(python.parents[1] / ".autotem-runtime.json")
    require(
        pointer["manifest_sha256"] == verified.manifest_sha256
        and marker["wheel_sha256"] == digest(verified.wheel),
        "Wrong managed simulation runtime",
    )
    variants = sibling("verify_variants")
    source = bundle / manifest["source"]["archive"]
    with ZipFile(source) as archive:
        for name in (
            Path(__file__).name,
            "qualify_local.py",
            "verify_variants.py",
            "operation_variants.json",
        ):
            require(
                archive.read("tools/acceptance/" + name)
                == Path(__file__).with_name(name).read_bytes(),
                "Simulation verifier differs from the archived source",
            )
    case = read(case_path)
    runtime = case["runtime"]
    identity = runtime["release_identity"]
    require(
        case["case"] == "cubic-simulation"
        and case["verdict"] == "passed"
        and case["fixture"] == variants.FIXTURES["cases"]["cubic-simulation"]
        and case["fixture_manifest_sha256"] == digest(variants.FIXTURE_MANIFEST)
        and runtime["private_install_verified"]
        and runtime["isolated"]
        and not runtime["editable"]
        and identity["profile"] == PROFILE
        and identity["manifest_sha256"] == verified.manifest_sha256
        and identity["source_archive_sha256"] == digest(source)
        and identity["wheel_sha256"] == marker["wheel_sha256"]
        and identity["lock_sha256"] == digest(verified.lock),
        "Case, fixture or installed edition is inconsistent",
    )
    run_id = case["run_id"]
    frame = sibling("qualify_local").RecordedRun(
        project, "cubic-simulation", run_id, case["report_artifact_id"], store_root=root
    )
    with sqlite3.connect((root / "harness.sqlite3").as_uri() + "?mode=ro", uri=True) as db:
        run = db.execute(
            "SELECT r.input_path,r.source_hash,r.request,r.job_id,r.status,j.status,j.ended "
            "FROM runs r JOIN jobs j ON r.job_id=j.job_id AND r.run_id=j.run_id "
            "WHERE r.run_id=?",
            (run_id,),
        ).fetchone()
        events = db.execute(
            "SELECT payload FROM events WHERE run_id=? AND kind='runtime_bound'", (run_id,)
        ).fetchall()
    require(
        run and run[4:6] == ("completed", "completed") and run[6] is not None and len(events) == 1,
        "Unsettled or unbound simulation job",
    )
    raw = Path(run[0]).resolve()
    fixture = case_path.parent / "J01 合成 HRTEM.npy"
    bound = json.loads(events[0][0])
    require(
        raw.is_relative_to(root / "runs" / run_id / "raw")
        and digest(raw) == run[1] == digest(fixture) == case["source_sha256"]
        and bound["runtime_id"] == pointer["runtime_id"]
        and bound["job_id"] == run[3]
        and bound["package_version"] == manifest["package_version"]
        and bound["plugins"] == {}
        and bound["plan_sha256"] == hashlib.sha256(run[2].encode()).hexdigest(),
        "Changed simulation snapshot, plan or job binding",
    )
    check_fixture_and_plan(raw, json.loads(run[2]), case["fixture"])
    require(
        frame.report["report_type"] == "analysis"
        and frame.report["completeness"]["required_complete"]
        and frame.report["scientific_accuracy"] == "not_evaluated"
        and all(
            frame.claims[op]["status"] == "completed" and frame.claims[op]["delivered"]
            for op in OPERATIONS
        ),
        "Missing requested simulation result",
    )
    return {
        "schema_version": "1.0.0",
        "case_verdict": "passed",
        "profile": PROFILE,
        "verifier_sha256": digest(Path(__file__)),
        "installed_backends": installed_backends(verified, python),
        "release_manifest_sha256": verified.manifest_sha256,
        "wheel_sha256": identity["wheel_sha256"],
        "source_archive_sha256": identity["source_archive_sha256"],
        "lock_sha256": identity["lock_sha256"],
        "case_sha256": digest(case_path),
        "operations": list(OPERATIONS),
        "run_id": run_id,
        "job_id": run[3],
        "report_artifact_id": case["report_artifact_id"],
        "artifacts": len(frame.audit),
        "claim_artifacts": {
            op: {
                role: {aid: frame.audit[aid]["sha256"] for aid in ids}
                for role, ids in frame.claims[op]["artifacts_by_role"].items()
            }
            for op in OPERATIONS
        },
        "numerical_evidence": numerical_evidence(frame),
        "scientific_accuracy": "not_evaluated",
    }


def recheck(report, bundle, project):
    result = summary(bundle, project, report.parent / "variant-evidence.json")
    require(result == read(report), "Simulation certificate differs from the managed project")
    return result


def collect(bundle, project, out):
    runtime = runtime_record(True, bundle)
    require(
        runtime["release_identity"]["profile"] == PROFILE
        and Path(sys.executable) == active_python(project / ".autotem"),
        "Use this managed simulation project's private interpreter",
    )
    out.mkdir(parents=True, exist_ok=False)
    sibling("verify_variants").cubic_simulation(out, runtime, project=project)
    result = summary(bundle, project, out / "variant-evidence.json")
    with (out / "simulation.json").open("x", encoding="utf-8") as stream:
        print(json.dumps(result, indent=2, allow_nan=False), file=stream)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("--bundle", "--project", "--out"):
        parser.add_argument(option, type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.bundle.resolve(), args.project.resolve(), args.out.resolve())
    print(json.dumps(result["numerical_evidence"]))
