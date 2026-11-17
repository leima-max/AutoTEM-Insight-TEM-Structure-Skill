"""Independent reads of registered artifact bytes and numerical acceptance checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .cases import PERIOD_PX, PIXEL_NM


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_artifact(
    output: Path, run_id: str, artifact: dict, *, store_root: Path | None = None
) -> dict:
    """Independently verify file and array digests. Publish only relative evidence paths."""
    path = Path(artifact["path"]).resolve()
    role = artifact["role"]
    require(artifact["run_id"] == run_id and artifact["committed"], f"Uncommitted {role}")
    store = store_root if store_root is not None else output / "harness"
    require(path.is_relative_to(store / "runs" / run_id), f"Wrong run for {role}")
    require(path.stat().st_size == artifact["size"], f"Size mismatch for {role}")
    require(digest(path) == artifact["sha256"], f"SHA256 mismatch for {role}")
    entry = {
        key: artifact[key]
        for key in ("artifact_id", "role", "kind", "step_id", "sha256", "size", "execution_usable")
    }
    entry["path"] = path.relative_to(output).as_posix()
    kind = artifact["kind"]
    meta = artifact["metadata"]
    for field in ("unit", "coordinate_unit", "image_role", "display_only", "scientific_level"):
        if field in meta:
            entry[field] = meta[field]
    if kind == "array":
        array = np.load(path, allow_pickle=False)
        array_hash = hashlib.sha256(array.tobytes(order="C")).hexdigest()
        require(list(array.shape) == meta["shape"], f"Shape mismatch for {role}")
        require(str(array.dtype) == meta["dtype"], f"Dtype mismatch for {role}")
        require(array_hash == meta["array_sha256"], f"Array digest mismatch for {role}")
        entry.update(shape=list(array.shape), dtype=str(array.dtype), array_sha256=array_hash)
    elif kind == "table":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            require((reader.fieldnames or []) == meta["columns"], f"CSV columns for {role}")
            count = sum(1 for _ in reader)
            require(count == meta["rows"], f"CSV row count for {role}")
            entry["rows"] = count
    elif kind == "image":
        with Image.open(path) as image:
            image.load()
            entry["size_xy"] = list(image.size)
    elif kind == "json":
        json.loads(path.read_text(encoding="utf-8"))
    return entry


class RunEvidence:
    def __init__(
        self, output: Path, run_id: str, artifacts: list[dict], exported: dict, *, store_root=None
    ):
        self.entries = [audit_artifact(output, run_id, a, store_root=store_root) for a in artifacts]
        self.by_id = {a["artifact_id"]: a for a in artifacts}
        require(len(self.by_id) == len(artifacts), "Duplicate artifact IDs")
        reports = {a["role"]: a for a in exported["artifacts"]}
        require(
            all(a["artifact_id"] in self.by_id for a in reports.values()),
            "Report output missing from registry",
        )
        self.report = json.loads(Path(reports["report_manifest"]["path"]).read_text("utf-8"))
        self.manifest_id = reports["report_manifest"]["artifact_id"]
        html = Path(reports["report_html"]["path"]).read_text("utf-8")
        self.index = self.report["result_index"]
        self.run = self.report["run"]
        require(
            self.run["run_id"] == run_id and self.run["status"] == "completed",
            "Report belongs to another run or a non-completed analysis",
        )
        require(self.report["report_type"] == "analysis", "Expected an analysis report")
        require(self.report["scientific_accuracy"] == "not_evaluated", "Missing science limit")
        before_export = {a["artifact_id"] for a in artifacts if a["step_id"] != "_report"}
        require(
            {a["artifact_id"] for a in self.report["artifacts"]} == before_export,
            "Report omitted evidence or cited another run",
        )
        for artifact in self.report["artifacts"]:
            registered = self.by_id[artifact["artifact_id"]]
            require(registered["sha256"] == artifact["sha256"], "Report digest mismatch")
            require(f'href="{Path(registered["path"]).name}"' in html, "Missing HTML evidence link")
        for role, artifact_id in self.index.items():
            registered = self.by_id[artifact_id]
            require(
                registered["role"] == role and registered["execution_usable"],
                f"Report index points to unusable {role}",
            )

    def path(self, role):
        return Path(self.by_id[self.index[role]]["path"])

    def array(self, role):
        return np.load(self.path(role), allow_pickle=False)

    def value(self, role):
        return json.loads(self.path(role).read_text("utf-8"))


def check_diffraction(data: RunEvidence, *, with_candidate: bool = True) -> list[dict]:
    from autotem.diffraction.fft import compute_fft_analysis

    processed = data.array("processed")
    direct, _ = compute_fft_analysis(processed, center_mode="geometric")
    # Identical deterministic FFT and dtype on the same installation.
    np.testing.assert_array_equal(data.array("fft_complex"), direct.fft_complex)
    np.testing.assert_array_equal(data.array("reciprocal_image"), direct.magnitude)
    result = data.value("diffraction")
    rows, columns = processed.shape
    center = np.array([columns // 2, rows // 2])
    np.testing.assert_array_equal(result["center"], center)
    observations = []
    for vector in result["g_vectors"]:
        index = vector["source_peak_index"]
        peak = result["peaks"][index]
        gx = (peak["x"] - center[0]) / (columns * PIXEL_NM)
        gy = (peak["y"] - center[1]) / (rows * PIXEL_NM)
        components = [vector["gx"], vector["gy"]]
        require(all(c["unit"] == "1/nm" for c in components), "Reciprocal unit mismatch")
        spacing = result["provenance"]["d_by_peak"][index]
        require(spacing["unit"] == "nm", "Missing physical d-spacing")
        # Only binary64 arithmetic/serialization error is expected for this conversion.
        np.testing.assert_allclose(
            [c["value"] for c in components], [gx, gy], rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(spacing["value"], 1 / np.hypot(gx, gy), rtol=1e-12)
        observations.append((peak["x"], peak["y"], spacing["value"]))
    points = np.array(observations)
    require(len(points) >= 4, "Missing analytic lattice peaks")
    spacings = []
    # Analytic Bragg maxima lie on integer bins. Allow half a bin for subpixel localization.
    for offset in (
        (columns / PERIOD_PX, 0),
        (-columns / PERIOD_PX, 0),
        (0, rows / PERIOD_PX),
        (0, -rows / PERIOD_PX),
    ):
        errors = np.linalg.norm(points[:, :2] - (center + offset), axis=1)
        closest = int(np.argmin(errors))
        require(errors[closest] <= 0.5, "Missing analytic Bragg maximum")
        spacings.append(float(points[closest, 2]))
    claims = [
        {
            "claim": "Spacing at the four analytic Bragg maxima",
            "value": spacings,
            "unit": "nm",
            "method": "per-axis calibrated FFT",
            "artifact_id": data.index["diffraction"],
            "uncertainty": "Total measurement uncertainty was not evaluated",
        }
    ]
    if not with_candidate:
        return claims
    candidates = result["phase_candidates"]
    require(
        any(c["name"] == "synthetic-cubic" and c["matched_spot_count"] for c in candidates),
        "No matched constrained candidate",
    )
    require(
        data.by_id[data.index["diffraction"]]["metadata"]["scientific_level"] == "candidate",
        "Indexed candidate lost its scientific level",
    )
    claims.append(
        {
            "claim": "Constrained phase candidate",
            "value": "synthetic-cubic",
            "artifact_id": data.index["diffraction"],
            "limitation": "Candidate identification",
        }
    )
    return claims


def check_atoms(data: RunEvidence, fixture: dict) -> list[dict]:
    from scipy.spatial import cKDTree

    from autotem.atomic.atom_detection import detect_atoms
    from autotem.atomic.gaussian_refinement import refine_atom_coordinates
    from autotem.atomic.intensity_quantification import quantify_atomic_column_intensity

    raw, processed = data.array("raw"), data.array("processed")
    require(not np.array_equal(raw, processed), "Fixture failed to distinguish image sources")
    atoms = data.value("atoms")
    require(atoms["provenance"]["image_role"] == "processed", "Wrong candidate image")
    candidates = np.array([[a["y"], a["x"]] for a in atoms["atoms"]])
    np.testing.assert_array_equal(candidates, detect_atoms(processed).coordinates)
    refined = data.value("refined_atoms")
    require(refined["coordinate_unit"] == "pixel", "Wrong coordinate unit")
    coordinates = np.array([[a["y"], a["x"]] for a in refined["atoms"]])
    direct = refine_atom_coordinates(raw, candidates)
    # Same fit on the same runtime. A nanometre-scale calibration is not part of the fit.
    np.testing.assert_allclose(coordinates, direct.coordinates_refined, rtol=0, atol=1e-8)
    point_status = refined["provenance"]["raw_refinement"]["point_status"]
    require(
        point_status == direct.provenance["point_status"]
        and len(point_status) == len(candidates)
        and "accepted" in point_status,
        "Missing per-point refinement status",
    )
    expected = np.array(fixture["expected"]["centers_yx"])
    errors, indices = cKDTree(expected).query(coordinates)
    require(
        len(coordinates) == len(expected)
        and len(set(indices)) == len(expected)
        and np.max(errors) < 0.5,
        "Refined coordinates did not recover the fixture grid",
    )
    distances = data.array("atom_distances.nearest_distances")
    neighbors, _ = cKDTree(coordinates).query(coordinates, k=2)
    np.testing.assert_allclose(distances, neighbors[:, 1] * PIXEL_NM, rtol=1e-12)
    require(data.value("atom_distances")["unit"] == "nm", "Nearest-neighbor units")
    direct_intensity = quantify_atomic_column_intensity(
        raw,
        coordinates,
        integration_radius=1.5,
        background_inner_radius=2.0,
        background_outer_radius=3.0,
    )
    np.testing.assert_allclose(data.array("atomic_intensity.raw_sum"), direct_intensity.raw_sum)
    np.testing.assert_allclose(
        data.array("atomic_intensity.normalized_intensity"), direct_intensity.normalized_intensity
    )
    trusted = data.array("atomic_intensity.trusted_atom_mask")
    np.testing.assert_array_equal(
        trusted,
        direct_intensity.valid_atom_mask
        & ~direct_intensity.edge_flag
        & ~direct_intensity.overlap_flag
        & ~direct_intensity.saturation_flag,
    )
    require(trusted.any(), "No trusted relative intensities")
    return [
        {
            "claim": "Refined column count",
            "value": len(coordinates),
            "unit": "columns",
            "method": "Gaussian refinement on raw",
            "artifact_id": data.index["refined_atoms"],
        },
        {
            "claim": "Median nearest-neighbor distance",
            "value": float(np.median(distances)),
            "unit": "nm",
            "method": "calibrated nearest neighbors",
            "artifact_id": data.index["atom_distances.nearest_distances"],
        },
        {
            "claim": "Columns with trusted relative intensities",
            "value": int(trusted.sum()),
            "unit": "columns",
            "artifact_id": data.index["atomic_intensity.trusted_atom_mask"],
            "limitation": "Relative intensities only; quantitative composition was not evaluated",
        },
    ]
