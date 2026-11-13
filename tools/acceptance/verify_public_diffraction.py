"""Qualify two CAU-36 disk inputs with an installed core runtime.

Run the managed python.exe with -I and this script, supplying --archive cau-36.zip,
--output NEW_DIRECTORY and --expected-manifest RELEASE_MANIFEST_SHA256. The source
archive stays external. HDF5 is decoded only by the acceptance driver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

DATASET = "https://doi.org/10.5281/zenodo.1158421"
PAPER = "https://doi.org/10.1107/S1600576718009500"
MD5 = "050f3832e12031afa4e60c615a2dc34a"
ZIP_SHA = "5f52aa135795509fe163eee9c8c9910dbeec342b255534f884218792b9231b33"
FRAME = "cau-36/data/image_0020_0000.h5"
TIFF = "cau-36/calib/calib_db_DiffShift_0000.tiff"
FRAME_SHA = "869dba8ec1881f13919d2770cdeb6d2dd1a05f2c0a0b4d1f9a53224e7dfe9d75"
TIFF_SHA = "a8bb53d613b2d206e795acd6d89c12b5a144316bd3a4a42be0eeadda7bf8d5a6"
FRAME_PIXELS = "09ea7f15a848fa99ee7fae03aa52976bc86166a7a640ecb7df2edaa2369b7bc4"
TIFF_PIXELS = "08fa338d5d4867a1cc8e8ff408b40bd450bf9a96e833a86812e1e240cb648a1b"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def hash_bytes(data):
    return hashlib.sha256(data).hexdigest()


def pixels(array, dtype, expected):
    digest = hash_bytes(array.tobytes(order="C"))
    require(array.dtype == np.dtype(dtype) and array.shape == (516, 516), "Wrong detector layout")
    require(digest == expected, "Detector pixels differ from the pinned frame")
    return {
        "dtype": array.dtype.str,
        "axes": ["y", "x"],
        "shape": list(array.shape),
        "minimum": array.min().item(),
        "maximum": array.max().item(),
        "pixels_sha256": digest,
    }


def inputs(archive, output):
    # Zenodo publishes MD5. The independently pinned SHA256 is also mandatory.
    md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with archive.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
        require((md5.hexdigest(), sha.hexdigest()) == (MD5, ZIP_SHA), "Wrong CAU-36 archive")
        with zipfile.ZipFile(source) as bundle:
            frame_bytes, tiff_bytes = bundle.read(FRAME), bundle.read(TIFF)
    require(
        hash_bytes(frame_bytes) == FRAME_SHA and hash_bytes(tiff_bytes) == TIFF_SHA,
        "Source member checksum mismatch",
    )
    import h5py
    import tifffile

    with h5py.File(io.BytesIO(frame_bytes), "r") as h5:
        frame = h5["/data"][()]
        attrs = h5["/data"].attrs
        original = {
            k: str(attrs[k])
            for k in (
                "ImageCameraName",
                "ImageExposureTime",
                "ImagePixelsize",
                "exp_diff_cameralength",
            )
        }
    frame_info = pixels(frame, "<i4", FRAME_PIXELS)
    npy = output / "image_0020_0000.npy"
    np.save(npy, frame, allow_pickle=False)
    saved = np.load(npy, allow_pickle=False)
    require(saved.dtype == frame.dtype and np.array_equal(saved, frame), "Lossy HDF5 conversion")
    native = output / Path(TIFF).name
    native.write_bytes(tiff_bytes)
    require(hash_bytes(native.read_bytes()) == TIFF_SHA, "Native TIFF copy changed")
    native_info = pixels(tifffile.imread(io.BytesIO(tiff_bytes)), "<f4", TIFF_PIXELS)
    return (
        npy,
        native,
        {
            "archive_md5": MD5,
            "archive_sha256": ZIP_SHA,
            "hdf5": {
                "member": FRAME,
                "member_sha256": FRAME_SHA,
                "dataset": "/data",
                "file": npy.name,
                "file_sha256": hash_bytes(npy.read_bytes()),
                "source_attributes_as_stored": original,
                **frame_info,
            },
            "tiff": {"member": TIFF, "member_sha256": TIFF_SHA, "file": native.name, **native_info},
        },
    )


def step(sid, operation, depends=(), **parameters):
    return {
        "id": sid,
        "operation_id": operation,
        "depends_on": list(depends),
        "parameters": parameters,
    }


def detector_geometry(measured, coords, center):
    vectors = measured["g_vectors"]
    require(measured["provenance"]["physical"] is False, "Unexpected physical result")
    require(
        all(v[field]["unit"] == "pixel" for v in vectors for field in ("gx", "gy", "magnitude")),
        "Uncalibrated detector offsets must use detector pixels",
    )
    require(
        measured["provenance"].get("coordinate_space") == "diffraction_detector",
        "Detector coordinates lack explicit provenance",
    )
    require(
        measured["d_spacings"] == []
        and measured["provenance"]["d_by_peak"] == [None] * len(coords),
        "Uncalibrated detector radii cannot yield d-spacings",
    )
    indices = [v["source_peak_index"] for v in vectors]
    require(sorted(indices) == list(range(len(coords))), "Detector vector/peak binding differs")
    offsets = coords[indices] - [center["x"], center["y"]]
    radii = np.linalg.norm(offsets, axis=1)
    components = [[v["gx"]["value"], v["gy"]["value"]] for v in vectors]
    require(
        np.allclose(components, offsets, atol=1e-8, rtol=0)
        and np.allclose([v["magnitude"]["value"] for v in vectors], radii, atol=1e-8, rtol=0),
        "Detector offsets differ from the recorded center and peak positions",
    )
    angles = np.array([v["angle"]["value"] for v in vectors])
    expected = np.degrees(np.arctan2(offsets[:, 1], offsets[:, 0]))
    require(
        all(v["angle"]["unit"] == "deg" for v in vectors)
        and np.allclose((angles - expected + 180) % 360 - 180, 0, atol=1e-7, rtol=0),
        "Detector angle differs from the observed axis convention",
    )
    return [float(radii.min()), float(radii.max())]


def sample_run(h, source):
    steps = [
        step("load", "input.load"),
        step("pattern", "diffraction.pattern", ["load"]),
        step("center", "diffraction.center", ["pattern"]),
        step("peaks", "diffraction.peaks", ["center"]),
        step("cal", "calibration.resolve", ["load"]),
        step("pixel", "diffraction.metrics", ["peaks", "cal"]),
        step("physical", "diffraction.metrics", ["pixel"], require_physical=True),
    ]
    status = h.submit_analysis(
        {"input_path": str(source), "modality": "NBED", "steps": steps}, asynchronous=False
    )
    rid = status["run_id"]
    require(
        status["status"] == "needs_input" and status["jobs"][-1]["status"] == "needs_input",
        f"Physical scale was not requested: {rid} {status['status']} {status['error']}",
    )
    by_step = {s["step_id"]: s for s in status["steps"]}
    require(
        all(by_step[s["id"]]["status"] == "completed" for s in steps[:-1])
        and by_step["physical"]["status"] == "needs_input",
        "Unexpected work before scale request",
    )
    pending = status["input_requests"][-1]
    fields = pending["fields"]
    scale = "diffraction_scale_inv_nm_per_px"
    require(
        [f["name"] for f in fields] == [scale]
        and fields[0]["answer_schema"][scale]["units"] == ["1/nm/pixel"],
        "Missing inverse-length-per-detector-pixel field",
    )

    exported = h.export_report(rid, ["json", "html"])
    manifest_art = next(a for a in exported["artifacts"] if a["role"] == "report_manifest")
    manifest = h.store.load_json(manifest_art["artifact_id"])
    require(
        manifest["report_type"] == "diagnostic"
        and manifest["completeness"]["unmet_required_steps"] == ["physical"],
        "Blocked physical result was promoted in the report",
    )
    claims = {c["step_id"]: c for c in manifest["claim_evidence"]}
    require(
        not claims["physical"]["delivered"] and not claims["physical"]["artifacts_by_role"],
        "Uncalibrated physical results were published",
    )
    evidence = {a["artifact_id"]: a for a in manifest["artifacts"]}
    require(
        all(a["role"] != "fft_complex" for a in evidence.values())
        and all(s["operation_id"] != "diffraction.fft" for s in by_step.values()),
        "Unexpected FFT",
    )

    def aid(sid, role):
        (result,) = claims[sid]["artifacts_by_role"][role]
        require(evidence[result]["step_id"] == sid, "Wrong producing step")
        return result

    for sid, role in [
        ("load", "raw"),
        ("load", "reciprocal_image"),
        ("pattern", "reciprocal_image"),
    ]:
        pixels(h.store.load_array(aid(sid, role)), "<i4", FRAME_PIXELS)
    for sid in ("load", "pattern"):
        require(
            h.store.load_json(aid(sid, "reciprocal_context"))["kind"] == "direct_diffraction",
            "Reciprocal image did not come directly from the detector",
        )
    measured = h.store.load_json(aid("pixel", "diffraction"))
    calibration = h.store.load_json(aid("cal", "calibration"))
    require(
        calibration["diffraction_scale"] is None and calibration["pixel_size"] is None,
        "Unitless source metadata was promoted to calibration",
    )
    prior = {
        aid("load", "raw"),
        aid("pattern", "reciprocal_image"),
        aid("center", "center"),
        aid("peaks", "peak_table"),
        aid("cal", "calibration"),
    }
    require(
        prior.issubset(pending["evidence_artifact_ids"]), "Scale request lost detector evidence"
    )
    peaks = h.store.load_json(aid("peaks", "peaks"))
    table = evidence[aid("peaks", "peak_table")]
    with Path(table["path"]).open(newline="", encoding="utf-8-sig") as stream:
        peak_rows = list(csv.DictReader(stream))
    require(
        peaks and len(peak_rows) == len(measured["g_vectors"]) == len(peaks),
        "Peak/vector counts do not match",
    )
    coords = np.array([[p["x"], p["y"]] for p in peaks])
    require(
        np.isfinite(coords).all()
        and ((coords >= 0) & (coords < 516)).all()
        and np.array_equal(coords, [[float(p["x"]), float(p["y"])] for p in peak_rows]),
        "Peak positions do not match the detector or CSV",
    )
    center = h.store.load_json(aid("center", "center"))
    radius_range = detector_geometry(measured, coords, center)
    for a in [*evidence.values(), *exported["artifacts"]]:
        require(
            hash_bytes(Path(a["path"]).read_bytes()) == a["sha256"], "Changed registered evidence"
        )
    return {
        "run_id": rid,
        "job_id": status["job_id"],
        "state": status["status"],
        "report_type": manifest["report_type"],
        "operation_status": {k: v["status"] for k, v in by_step.items()},
        "physical_scale_fields": fields,
        "input_request_id": pending["request_id"],
        "input_request_evidence": pending["evidence_artifact_ids"],
        "peak_count": len(peaks),
        "center": center,
        "detector_radius_px_range": radius_range,
        "detector_offset_unit": "pixel",
        "coordinate_space": measured["provenance"]["coordinate_space"],
        "first_peak_xy": coords[0].tolist(),
        "first_offset_xy_px": (coords[0] - [center["x"], center["y"]]).tolist(),
        "d_spacing_count": len(measured["d_spacings"]),
        "reports": [
            {k: a[k] for k in ("role", "artifact_id", "sha256")} for a in exported["artifacts"]
        ],
        "artifacts": [
            {k: a[k] for k in ("step_id", "role", "artifact_id", "sha256")}
            for a in evidence.values()
        ],
    }


def run(archive, output, expected_manifest):
    import autotem
    from autotem.harness.service import Harness

    prefix = Path(sys.prefix).resolve()
    verifier_hash = hash_bytes(Path(__file__).read_bytes())
    marker = json.loads((prefix / ".autotem-runtime.json").read_text(encoding="utf-8"))
    require(
        sys.flags.isolated
        and Path(autotem.__file__).resolve().is_relative_to(prefix)
        and Path(marker["runtime"]).resolve() == prefix
        and marker["state"] == "ready"
        and marker["manifest_sha256"] == expected_manifest,
        "Run with the isolated, ready private interpreter for the expected release",
    )
    output = output.resolve()
    require(
        not output.is_relative_to(Path(marker["project"]).resolve() / ".autotem"),
        "Use an independent output directory outside the existing installation",
    )
    output.mkdir(parents=True, exist_ok=False)
    npy, native, provenance = inputs(archive, output)
    harness = Harness(output / "harness")
    sample = sample_run(harness, npy)
    control = harness.submit_analysis(
        {"input_path": str(native), "modality": "NBED", "steps": [step("load", "input.load")]},
        asynchronous=False,
    )
    require(control["status"] == "completed", "Native calibration TIFF reader failed")
    raw_id = harness.store.role(control["run_id"], "raw")
    pixels(harness.store.load_array(raw_id), "<f4", TIFF_PIXELS)
    result = {
        "schema_version": "1.0.0",
        "case_verdict": "passed",
        "release_manifest_sha256": expected_manifest,
        "own_wheel_sha256": marker["wheel_sha256"],
        "python": sys.version.split()[0],
        "verifier_sha256": verifier_hash,
        "attribution": {
            "creator": "Stef Smeets",
            "title": "serial electron diffraction data",
            "year": 2018,
            "doi": DATASET,
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "metadata": "https://api.datacite.org/dois/10.5281/zenodo.1158421",
            "changes": (
                "One HDF5 /data frame repackaged losslessly as NPY. "
                "Original calibration TIFF copied verbatim."
            ),
        },
        "acquisition": {
            "paper": PAPER,
            "instrument": "Timepix",
            "sample_collection": "Co-CAU-36",
            "routing": "NBED",
            "basis": (
                "SerialED direct patterns from a quasi-parallel 200–500 nm probe "
                "(paper sections 2.3–2.4)."
            ),
            "scale_basis": (
                "No documented reciprocal-space scale supplied for this frame. "
                "Nominal camera length and the unitless ImagePixelsize were kept as metadata."
            ),
        },
        "inputs": provenance,
        "sample": sample,
        "native_tiff_reader": {
            "run_id": control["run_id"],
            "job_id": control["job_id"],
            "raw_artifact_id": raw_id,
            "sha256": harness.store.artifact(raw_id)["sha256"],
        },
        "scientific_accuracy": "not_evaluated",
        "ground_truth": "No indexed peak or phase labels for this frame were used.",
    }
    require(hash_bytes(Path(__file__).read_bytes()) == verifier_hash, "Acceptance driver changed")
    (output / "public-diffraction.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest", required=True)
    args = parser.parse_args()
    result = run(args.archive, args.output, args.expected_manifest)
    print(f"{result['case_verdict']} {args.output / 'public-diffraction.json'}")
