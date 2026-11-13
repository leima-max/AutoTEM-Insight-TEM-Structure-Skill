"""Full-file evidence for traditional operation variants outside the first release slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

from autotem.acceptance.cases import ATOM_REFERENCE, plan_for, write_fixture
from autotem.acceptance.driver import runtime_record, write_json
from autotem.acceptance.evidence import RunEvidence, audit_artifact, digest, require
from autotem.harness import Harness

FIXTURE_MANIFEST = Path(__file__).with_name("operation_variants.json")
FIXTURES = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))


class OperationCase:
    def __init__(self, root, source, modality, *, project=None):
        self.root = root
        self.source = source
        self.modality = modality
        self.artifact_base = project if project is not None else root
        self.harness = Harness(
            project / ".autotem/runs" if project is not None else root / "harness"
        )
        self.state = None
        self.artifacts = []
        self.checked = {}

    def submit(self, operations, *, budgets=None):
        previous = self.state["steps"][-1]["step_id"] if self.state else None
        steps = []
        for name, params in operations:
            steps.append(
                {
                    "id": name,
                    "operation_id": name,
                    "depends_on": [previous] if previous else [],
                    "parameters": params,
                }
            )
            previous = name
        plan = {"input_path": str(self.source), "modality": self.modality, "steps": steps}
        if budgets is not None:
            plan["budgets"] = budgets
        run_id = self.state["run_id"] if self.state else None
        self.state = self.harness.submit_analysis(plan, run_id=run_id, asynchronous=False)
        self.artifacts = self.harness.list_artifacts(self.state["run_id"])
        assert self.state["status"] == "completed", self.state["error"]

    def artifact(self, step, role):
        return next(a for a in self.artifacts if a["step_id"] == step and a["role"] == role)

    def audit(self, artifact):
        return audit_artifact(
            self.artifact_base, self.state["run_id"], artifact, store_root=self.harness.root
        )

    def value(self, step, role):
        artifact = self.artifact(step, role)
        self.audit(artifact)
        return json.loads(Path(artifact["path"]).read_text("utf-8"))

    def array(self, step, role):
        artifact = self.artifact(step, role)
        self.audit(artifact)
        return np.load(artifact["path"], allow_pickle=False)

    def verified(self, operation, check, *, observed=None):
        step = next(s for s in self.state["steps"] if s["step_id"] == operation)
        assert step["status"] == "completed"
        event = next(
            item["payload"]
            for item in self.state["events"]
            if item["kind"] == "operation_started" and item["payload"]["step_id"] == operation
        )
        artifacts = [self.audit(a) for a in self.artifacts if a["step_id"] == operation]
        assert artifacts and all(a["execution_usable"] for a in artifacts)
        self.checked[operation] = {
            "check": check,
            "artifacts": artifacts,
            "verdict": "passed",
            "operation_version": event["operation_version"],
            "parameters": event["parameters"],
            "scientific_level": step["result"]["scientific_level"],
            "consumed_artifacts": step["result"]["consumed_artifacts"],
        }
        if observed is not None:
            self.checked[operation]["observed"] = observed


@contextmanager
def disk_case(root, source, fixture_id, runtime, *, runtime_options=None, project=None):
    require(__debug__, "Numeric acceptance checks require a non-optimized interpreter")
    fixture = FIXTURES["cases"][fixture_id]
    case = OperationCase(root, source, fixture["modality"], project=project)
    info = {
        "schema_version": "1.0.0",
        "scientific_accuracy": "not_evaluated",
        "execution_layer": "installed_worker"
        if runtime["private_install_verified"]
        else "local_development_worker",
        "case": fixture_id,
        "fixture": fixture,
        "fixture_manifest_sha256": digest(FIXTURE_MANIFEST),
        "source_sha256": digest(source),
        "source_format": source.suffix,
        "runtime": runtime,
        "runtime_options": runtime_options or {},
        "verdict": "failed",
        "operations": case.checked,
    }
    try:
        raw = np.load(source, allow_pickle=False)
        assert list(raw.shape) == fixture["shape"] and str(raw.dtype) == FIXTURES["dtype"]
        yield case
        assert digest(source) == info["source_sha256"]
        run_id = case.state["run_id"]
        report = case.harness.export_report(run_id)
        artifacts = case.harness.list_artifacts(run_id)
        if fixture.get("expected_report_type") == "diagnostic":
            assert report["report_type"] == "diagnostic"
            reports = {a["role"]: a for a in report["artifacts"]}
            manifest = json.loads(Path(reports["report_manifest"]["path"]).read_text("utf-8"))
            assert manifest["run"]["acceptance_status"] == "blocked"
            assert manifest["scientific_accuracy"] == "not_evaluated"
            assert manifest["run"]["acceptance"]["benchmark"]["all_passed"] is False
            assert manifest["run"]["acceptance"]["validation"]["all_passed"] is True
            validated = [case.audit(item) for item in artifacts]
            linked = {a["artifact_id"]: a["sha256"] for a in validated if a["step_id"] != "_report"}
            assert {a["artifact_id"]: a["sha256"] for a in manifest["artifacts"]} == linked
            assert set(manifest["result_index"].values()) <= linked.keys()
            html = Path(reports["report_html"]["path"]).read_text("utf-8")
            assert "diagnostic report" in html
            assert all(f'href="{Path(a["path"]).name}"' in html for a in manifest["artifacts"])
            manifest_id = reports["report_manifest"]["artifact_id"]
        else:
            manifest_id = RunEvidence(
                case.artifact_base, run_id, artifacts, report, store_root=case.harness.root
            ).manifest_id
        info.update(
            verdict="passed", report_type=report["report_type"], report_artifact_id=manifest_id
        )
    except Exception as exc:
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        info["error"] = str(exc).replace(str(root), ".")[:2000] or frame.line or type(exc).__name__
        info["check_line"] = frame.lineno
        raise
    finally:
        if case.state:
            info.update(run_id=case.state["run_id"], execution_status=case.state["status"])
            info["steps"] = [
                {
                    "id": step["step_id"],
                    "operation_id": step["operation_id"],
                    "status": step["status"],
                    "error_code": (step.get("error") or {}).get("code"),
                }
                for step in case.state["steps"]
            ]
            if case.state.get("error"):
                info["execution_error_code"] = case.state["error"]["code"]
        write_json(root / "variant-evidence.json", info)


def atomic_grid(tmp_path, runtime):
    from autotem.atomic.displacement import compute_atomic_displacement_from_basis
    from autotem.atomic.sublattice import classify_sublattices_auto

    source, fixture = write_fixture("J04", tmp_path)
    with disk_case(tmp_path, source, "atomic-grid", runtime) as case:
        case.submit(
            [
                ("input.load", {}),
                ("atomic.detect", {}),
                ("atomic.refine", {}),
                (
                    "atomic.intensity",
                    {
                        "integration_radius": 1.5,
                        "background_inner_radius": 2.0,
                        "background_outer_radius": 3.0,
                    },
                ),
            ]
        )
        refined = case.value("atomic.refine", "refined_atoms")
        coords = np.array([[a["y"], a["x"]] for a in refined["atoms"]])
        centers = np.array(fixture["expected"]["centers_yx"])
        assert len(coords) == len(centers) == 42
        err, indices = cKDTree(coords).query(centers[[0, 1, 7]])
        assert max(err) < 0.5 and len(set(indices)) == 3
        basis = dict(zip(("base_index", "a_index", "b_index"), indices.tolist(), strict=True))
        seed_hash = case.artifact("atomic.refine", "refined_atoms")["sha256"]
        first_index = int(indices[0])
        count_criterion = {"metric": "atom_coordinate_count", "op": "==", "unit": "count"}
        case.submit(
            [
                ("atomic.regularize", {}),
                ("atomic.displacement", {**basis, "reference_region": ATOM_REFERENCE}),
                ("atomic.spacing.indexed", {}),
                ("atomic.spacing.basis", basis),
                ("atomic.spacing.lattice_angle", {"lattice_angle_deg": 90.0}),
                ("atomic.spacing.angle", {"angle_deg": 0.0}),
                (
                    "atomic.spacing.picked",
                    {"start_index": first_index, "end_index": int(indices[1])},
                ),
                ("atomic.sublattice.auto", {**basis, "use_intensity": False}),
                (
                    "atomic.sublattice.model",
                    {**basis, "basis_sites": [{"u": 0, "v": 0, "label": "A"}]},
                ),
                ("atomic.sublattice.manual", {"manual_labels": {str(first_index): "reviewed-A"}}),
                ("atomic.intensity_groups", {"grouping": "lattice", "mode": "sum_parity"}),
                (
                    "validation.evaluate",
                    {"acceptance_criteria": [{**count_criterion, "threshold": 42}]},
                ),
                (
                    "benchmark.evaluate",
                    {"acceptance_criteria": [{**count_criterion, "threshold": 43}]},
                ),
            ]
        )
        regular = case.value("atomic.regularize", "regularized_atoms")
        np.testing.assert_array_equal([[a["y"], a["x"]] for a in regular["atoms"]], coords)
        assert (
            case.artifact("atomic.regularize", "regularized_atoms")["artifact_id"]
            != (case.artifact("atomic.refine", "refined_atoms")["artifact_id"])
        )
        assert digest(Path(case.artifact("atomic.refine", "refined_atoms")["path"])) == seed_hash
        case.verified(
            "atomic.regularize",
            "Default regularization retains distinct artifacts and original refined bytes",
        )
        y0, y1, x0, x1 = ATOM_REFERENCE
        mask = (
            (coords[:, 0] >= y0) & (coords[:, 0] < y1) & (coords[:, 1] >= x0) & (coords[:, 1] < x1)
        )
        direct = compute_atomic_displacement_from_basis(
            coords, **basis, reference_mask=mask, allow_reference_fallback=False
        )
        d = "atomic.displacement"
        np.testing.assert_array_equal(
            case.array(d, "atomic_displacement.reference_atom_mask"), mask
        )
        np.testing.assert_array_equal(
            case.array(d, "atomic_displacement.valid_atom_mask"), direct.valid_atom_mask
        )
        np.testing.assert_allclose(
            case.array(d, "atomic_displacement.displacement_yx"), direct.displacement_yx, atol=1e-7
        )
        case.verified(d, "Direct API displacement vectors and exact selected reference/valid masks")
        for variant in ("indexed", "basis", "lattice_angle", "angle", "picked"):
            name = f"atomic.spacing.{variant}"
            start = case.array(name, "atomic_spacing.start_yx")
            end = case.array(name, "atomic_spacing.end_yx")
            length = case.array(name, "atomic_spacing.length_px")
            assert len(length) > 5 and abs(np.median(length) - 8) <= 0.01
            np.testing.assert_allclose(length, np.linalg.norm(end - start, axis=1), atol=1e-5)
            # The Gaussian fits can shift by a small fraction of a pixel at the boundaries.
            np.testing.assert_allclose(length / 8, np.rint(length / 8), atol=0.002)
            assert case.value(name, "atomic_spacing")["metadata"]["unit"] == "pixel"
            case.verified(
                name, "Every projected segment length agrees with endpoints and grid pitch"
            )
        auto = "atomic.sublattice.auto"
        intensity = np.array([a["intensity"]["value"] for a in refined["atoms"]])
        expected_labels = classify_sublattices_auto(
            coords, **basis, use_intensity=False, intensities=intensity
        ).labels
        np.testing.assert_array_equal(case.array(auto, "sublattices.labels"), expected_labels)
        case.verified(auto, "All per-site labels agree with direct classification")
        model = "atomic.sublattice.model"
        model_labels = case.array(model, "sublattices.labels")
        assert len(model_labels) == len(coords) and set(model_labels) == {"A"}
        case.verified(model, "All occupied analytic sites match the supplied A basis site")
        manual = "atomic.sublattice.manual"
        labels = model_labels.astype(object)
        labels[first_index] = "reviewed-A"
        np.testing.assert_array_equal(case.array(manual, "sublattices.labels"), labels)
        manual_mask = case.array(manual, "sublattices.manual_override_mask")
        assert np.flatnonzero(manual_mask).tolist() == [first_index]
        case.verified(manual, "Only the selected site changes label and manual override state")
        grouped = case.value("atomic.intensity_groups", "intensity_groups")
        expected_parity = np.sum(direct.lattice_indices_ij, axis=1) % 2
        trusted = case.array("atomic.intensity", "atomic_intensity.trusted_atom_mask")
        valid = trusted & direct.valid_atom_mask
        actual = case.array("atomic.intensity_groups", "intensity_groups.labels")
        np.testing.assert_array_equal(actual[valid], expected_parity[valid])
        assert len(grouped["summaries"]) == 2
        relative = case.array("atomic.intensity", "atomic_intensity.normalized_intensity")
        for summary in grouped["summaries"]:
            group = relative[valid & (actual == summary["group_id"])]
            assert len(group) == summary["count"] and np.isclose(
                np.mean(group), summary["mean"], rtol=1e-6, atol=1e-12
            )
        case.verified(
            "atomic.intensity_groups", "Parity and trusted raw intensity group means/counts"
        )
        for name, role, passed, threshold in (
            ("validation.evaluate", "validation", True, 42),
            ("benchmark.evaluate", "benchmark", False, 43),
        ):
            result = case.value(name, role)
            acceptance = result["acceptance"]
            assert acceptance["all_passed"] is passed and acceptance["evaluated"] == 1
            (criterion,) = acceptance["results"]
            assert criterion["observed"] == len(coords) and criterion["unit"] == "count"
            assert criterion["threshold"] == threshold and criterion["passed"] is passed
            metric = next(m for m in result["metrics"] if m["name"] == "atom_coordinate_count")
            assert metric["value"]["unit"] == "count" and metric["value"]["value"] == len(coords)
            case.verified(
                name,
                "Measured count of 42 independently refined sites against the requested threshold",
            )


def scale_bar(tmp_path, runtime):
    spec = FIXTURES["cases"]["scale-bar"]
    raw = np.random.default_rng(spec["seed"]).normal(0.1, 0.002, size=spec["shape"])
    y0, y1, x0, x1 = spec["bar_roi_y0_y1_x0_x1"]
    raw[y0:y1, x0:x1] = 1
    source = tmp_path / "scale-bar.npy"
    np.save(source, raw)
    with disk_case(tmp_path, source, "scale-bar", runtime) as case:
        case.submit(
            [
                ("input.load", {}),
                ("quality.assess", {}),
                (
                    "calibration.scalebar_candidate",
                    {"reference_length": spec["reference_length_nm"]},
                ),
                ("phase.lookup", {"source": "builtin", "formula": "Si"}),
                ("maintenance.dependencies", {}),
            ]
        )
        name = "calibration.scalebar_candidate"
        result = case.value(name, "scalebar_candidate")
        assert (
            result["requires_review"]
            and abs(result["pixel_size_nm"] - spec["expected_pixel_nm"]) <= 0.02
        )
        assert case.value("input.load", "dataset")["pixel_size"] is None
        np.testing.assert_array_equal(case.array(name, "scalebar_roi"), raw)
        case.verified(
            name, "Seeded 50-pixel bar at 5 nm yields a candidate before calibration is confirmed"
        )
        (phase,) = case.value("phase.lookup", "phase_candidates")
        assert abs(phase["a"] - 0.5431) < 1e-6 and len(phase["atom_sites"]) == 8
        assert case.value("phase.lookup", "phase_lookup")["network_used"] is False
        case.verified("phase.lookup", "Constrained offline Si cell in nm and eight explicit sites")
        diagnostics = case.value("maintenance.dependencies", "dependency_diagnostics")
        available = {row["operation_id"]: row["availability"]["state"] for row in diagnostics}
        current = case.harness.list_capabilities()["operations"]
        assert available == {row["operation_id"]: row["availability"]["state"] for row in current}
        case.verified(
            "maintenance.dependencies", "Recorded availability agrees with the live registry"
        )


def cubic_simulation(tmp_path, runtime, *, project=None):
    from autotem.core import DiffractionResult, SimulationResult
    from autotem.diffraction.phase_matching import match_diffsims_patterns_to_diffraction

    runtime_options = FIXTURES["cases"]["cubic-simulation"]["runtime_settings"]
    source, _ = write_fixture("J01", tmp_path)
    with disk_case(
        tmp_path,
        source,
        "cubic-simulation",
        runtime,
        runtime_options=runtime_options,
        project=project,
    ) as case:
        spec = FIXTURES["cases"]["cubic-simulation"]
        plan = plan_for("J01", source)
        index = plan["steps"][-1]["parameters"]
        (candidate,) = index["candidate_phase_params"]
        # A single *explicit* site is the defined primitive test structure, not a fallback.
        calibration = next(s for s in plan["steps"] if s["operation_id"] == "calibration.resolve")
        calibration["parameters"]["pixel_size_nm"] = spec["pixel_size_nm"]
        candidate.update(
            a=spec["cell_edge_nm"],
            b=spec["cell_edge_nm"],
            c=spec["cell_edge_nm"],
            formula="Si",
            space_group="Pm-3m",
            space_group_number=221,
            atom_sites=[{"element": "Si", "abc": [0, 0, 0], "occupancy": 1.0}],
        )
        ops = [(s["operation_id"], s["parameters"]) for s in plan["steps"]]
        case.submit(
            [
                *ops,
                (
                    "simulation.diffraction",
                    {
                        "pattern_shape": [64, 64],
                        "calibration": 0.025,
                        "max_candidates": 1,
                        "reciprocal_radius_inv_angstrom": 1.5,
                        "max_simulated_spots": 96,
                        "diffsims_timeout_s": runtime_options["diffsims_timeout_s"],
                    },
                ),
                ("diffraction.match", {"min_matched_spots": 3, "d_tolerance_rel": 0.06}),
            ],
            budgets={"simulation_seconds": runtime_options["simulation_seconds"]},
        )
        simulation = case.value("simulation.diffraction", "simulation")
        assert simulation["provenance"]["adapter"]["active"] is True
        (pattern,) = simulation["patterns"]
        array = case.array("simulation.diffraction", "simulated_pattern_0")
        assert array.shape == (64, 64) and np.isfinite(array).all() and array.max() > 0
        assert pattern["simulation_parameters"]["structure_source"] == "candidate_atom_sites"
        assert pattern["simulation_parameters"]["atom_site_count"] == 1
        assert pattern["spots"]
        for spot in pattern["spots"]:
            assert all(spot[key]["unit"] == "1/nm" for key in ("gx", "gy", "magnitude"))
            assert spot["d_spacing"]["unit"] == "nm"
            length = np.hypot(spot["gx"]["value"], spot["gy"]["value"])
            np.testing.assert_allclose(spot["magnitude"]["value"], length, rtol=1e-12)
            np.testing.assert_allclose(spot["d_spacing"]["value"], 1 / length, rtol=1e-12)
        case.verified(
            "simulation.diffraction",
            "Explicit sites, full image and reciprocal-length geometry",
            observed={"shape": list(array.shape), "simulated_spots": len(pattern["spots"])},
        )
        compared = case.value("diffraction.match", "simulation")
        match = case.value("diffraction.match", "diffraction")
        assert match["provenance"]["phase_matching"]["active"] and compared["comparison_metrics"]
        best = match["phase_candidates"][0]
        assert best["name"] == candidate["name"] and best["matched_spot_count"] == 4
        assert best["mean_d_error"]["unit"] == "relative" and best["mean_d_error"]["value"] < 0.06
        assert (
            best["mean_angle_error"]["unit"] == "deg" and best["mean_angle_error"]["value"] < 1e-9
        )
        direct = match_diffsims_patterns_to_diffraction(
            DiffractionResult.model_validate(case.value("diffraction.index", "diffraction")),
            SimulationResult.model_validate(simulation).patterns,
            min_matched_spots=3,
            d_tolerance_rel=0.06,
            previous_score_weight=0.0,
        )
        assert direct.provenance["active"]
        assert best == direct.candidates[0].model_dump(mode="json")
        case.verified(
            "diffraction.match",
            "Four analytic measured peaks and a direct API candidate comparison",
            observed={
                "candidate": best["name"],
                "matched_spots": best["matched_spot_count"],
                "mean_d_error": best["mean_d_error"],
                "mean_angle_error": best["mean_angle_error"],
            },
        )


CASES = {
    "atomic-grid": (
        atomic_grid,
        (
            "atomic.regularize",
            "atomic.displacement",
            "atomic.spacing.indexed",
            "atomic.spacing.basis",
            "atomic.spacing.lattice_angle",
            "atomic.spacing.angle",
            "atomic.spacing.picked",
            "atomic.sublattice.auto",
            "atomic.sublattice.model",
            "atomic.sublattice.manual",
            "atomic.intensity_groups",
            "validation.evaluate",
            "benchmark.evaluate",
        ),
    ),
    "scale-bar": (
        scale_bar,
        ("calibration.scalebar_candidate", "phase.lookup", "maintenance.dependencies"),
    ),
    "cubic-simulation": (cubic_simulation, ("simulation.diffraction", "diffraction.match")),
}


def driver_record(bundle):
    files = (Path(__file__), FIXTURE_MANIFEST)
    result = {"files": {p.name: digest(p) for p in files}, "source_archive_match": None}
    if bundle is not None:
        manifest = json.loads((bundle / "release-manifest.json").read_text("utf-8"))
        with ZipFile(bundle / manifest["source"]["archive"]) as archive:
            archived = {}
            for p in files:
                member = "tools/acceptance/" + p.name
                if member in archive.namelist():
                    archived[p.name] = hashlib.sha256(archive.read(member)).hexdigest()
        result["source_archive_files"] = archived
        result["source_archive_match"] = archived == result["files"]
    return result


def run_variants(output, *, require_installed=False, release_bundle=None, simulation=False):
    """Run independent numeric checks and record the exact installed identity on each row."""
    require(__debug__, "Numeric acceptance checks require a non-optimized interpreter")
    require(
        not require_installed or release_bundle is not None,
        "--require-installed needs --release-bundle",
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    require(not any(output.iterdir()), "Evidence output must be new or empty")
    result = {
        "schema_version": "1.0.0",
        "case_verdict": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "operations": {},
        "cases": {},
        "scientific_accuracy": "not_evaluated",
        "host_llm_e2e": "not_evaluated",
        "release_ready": False,
    }
    try:
        runtime = runtime_record(require_installed, release_bundle)
        result.update(runtime=runtime, driver=driver_record(release_bundle))
        capabilities = Harness(output / "capabilities").list_capabilities()["operations"]
        registered = {item["operation_id"]: item for item in capabilities}
        rows = result["operations"]
        for name, (_, expected) in CASES.items():
            for op in expected:
                definition = registered[op]
                require(
                    definition["implementation_status"] == "implemented",
                    f"{op} is reserved or absent",
                )
                rows[op] = {
                    "operation_id": op,
                    "operation_version": definition["version"],
                    "verdict": "not_evaluated",
                    "case": name,
                    "availability": {
                        k: v
                        for k, v in definition["availability"].items()
                        if k in {"state", "dependency", "code"}
                    },
                    "release_binding": {
                        **runtime["release_identity"],
                        "operation_version": definition["version"],
                    },
                    "artifacts": [],
                }
        for name, (check_case, expected) in CASES.items():
            if name == "cubic-simulation" and not simulation:
                state = rows["simulation.diffraction"]["availability"]["state"]
                if state != "available":
                    rows["simulation.diffraction"].update(
                        verdict=state, reason="Runtime dependency probe"
                    )
                    rows["diffraction.match"].update(
                        verdict="blocked",
                        upstream="simulation.diffraction",
                        reason="No real simulated-pattern artifact is available in this profile",
                    )
                else:
                    for op in expected:
                        rows[op]["reason"] = "Optional simulation was not requested"
                continue
            directory = output / name
            directory.mkdir()
            try:
                check_case(directory, runtime)
                error = None
            except Exception as exc:
                error = str(exc).replace(str(output), ".")[:2000] or type(exc).__name__
            evidence_path = directory / "variant-evidence.json"
            if not evidence_path.is_file():
                raise ValueError(f"Missing {name} evidence: {error}")
            evidence = json.loads(evidence_path.read_text("utf-8"))
            reference = {
                "case_evidence": evidence_path.relative_to(output).as_posix(),
                "case_evidence_sha256": digest(evidence_path),
                "run_id": evidence.get("run_id"),
            }
            case_ok = (
                error is None
                and evidence["verdict"] == "passed"
                and set(evidence["operations"]) == set(expected)
            )
            result["cases"][name] = {**reference, "verdict": "passed" if case_ok else "failed"}
            if not case_ok:
                result["cases"][name]["error"] = (
                    evidence.get("error") or error or "Incomplete operation evidence"
                )
            for op in expected:
                row = rows[op]
                record = evidence["operations"].get(op)
                if case_ok:
                    require(
                        record["operation_version"] == row["operation_version"],
                        f"Version changed for {op}",
                    )
                    row.update(record, **reference)
                else:
                    row.update(verdict="failed", **reference)
        require(
            all(c["verdict"] == "passed" for c in result["cases"].values()),
            "One or more variant cases failed",
        )
        result["case_verdict"] = "passed"
    except Exception as exc:
        result["error"] = str(exc).replace(str(output), ".")[:2000] or type(exc).__name__
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "operations.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--release-bundle", type=Path)
    parser.add_argument(
        "--simulation", action="store_true", help="Run real optional diffsims and matching"
    )
    args = parser.parse_args()
    try:
        result = run_variants(
            args.output,
            require_installed=args.require_installed,
            release_bundle=args.release_bundle,
            simulation=args.simulation,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"case_verdict": "failed", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "case_verdict": result["case_verdict"],
                "evidence": "operations.json",
                "error": result.get("error"),
            }
        )
    )
    return 0 if result["case_verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
