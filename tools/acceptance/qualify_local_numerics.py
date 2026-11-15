"""Compare persisted operation outputs with the underlying B/C APIs or an independent formula."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np

from autotem.acceptance.cases import read_fixture
from autotem.acceptance.evidence import check_atoms, check_diffraction, require
from autotem.acceptance.strain_pattern import check_atom_strain, check_direct_pattern, check_gpa
from autotem.core import (
    AnalysisReport,
    ArrayReference,
    AtomLattice,
    DiffractionResult,
    Measurement,
    TEMDataset,
)
from autotem.services.atomic import clean_json


class FinalRoles:
    """Expose the registered final index to immutable, wheel-verified acceptance functions."""

    def __init__(self, frame):
        self.index = frame.report["result_index"]
        self.by_id = frame.index
        self.report = frame.report

    def path(self, role):
        return Path(self.by_id[self.index[role]]["path"])

    def array(self, role):
        return np.load(self.path(role), allow_pickle=False)

    def value(self, role):
        import json

        return json.loads(self.path(role).read_text("utf-8"))


def equal(got, want):
    if isinstance(want, dict):
        require(
            isinstance(got, dict) and set(want) == set(got),
            f"Different mapping keys: {set(want) ^ set(got)}",
        )
        for k in want:
            equal(got[k], want[k])
    elif isinstance(want, (list, tuple)):
        require(isinstance(got, (list, tuple)) and len(got) == len(want), "Sequence length or type")
        for a, b in zip(got, want, strict=True):
            equal(a, b)
    elif isinstance(want, (float, np.floating)):
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)
    else:
        require(got == want, f"Value differs: {str(got)[:150]} / {str(want)[:150]}")


def arrays_equal(got, want):
    require(got.shape == want.shape, "Direct API array shape differs")
    if want.dtype.kind in "fci" and got.dtype.kind in "fc" and want.dtype.kind != "i":
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)
    else:
        np.testing.assert_array_equal(got, want)


def coordinates(f):
    atoms = f.value("atomic.refine", "refined_atoms")["atoms"]
    return np.array([[a["y"], a["x"]] for a in atoms]), np.array(
        [a["intensity"]["value"] for a in atoms]
    )


def dataclass_outputs(f, op, role, direct):
    payload = f.value(op, role)
    for field in fields(direct):
        want = getattr(direct, field.name)
        if isinstance(want, np.ndarray):
            want = want.astype(str) if want.dtype.kind == "O" else want
            arrays_equal(f.array(op, role + "." + field.name), want)
        else:
            equal(payload[field.name], clean_json(want))
    return len(fields(direct))


def atomic_variant(f, op, p):
    from autotem.atomic import plane_spacing as spacing
    from autotem.atomic import sublattice
    from autotem.atomic.displacement import compute_atomic_displacement_from_basis
    from autotem.atomic.distance_statistics import compute_distance_statistics
    from autotem.atomic.intensity_quantification import (
        SATURATION_THRESHOLD_MISSING,
        group_atomic_column_intensities_by_lattice,
        quantify_atomic_column_intensity,
    )
    from autotem.atomic.lattice_constraint import apply_lattice_constraint

    coords, intensity = coordinates(f)
    p = dict(p)
    p.pop("coordinate_role", None)
    if op == "atomic.distances":
        pixel = f.value("calibration.resolve", "calibration")["pixel_size"]
        require(
            pixel["unit"].startswith("nm") and p["output_unit"] == "nm",
            "Distance calibration units",
        )
        result = compute_distance_statistics(coords, pixel_size=pixel["value"], output_unit="nm")
        role = "atom_distances"
    elif op == "atomic.regularize":
        result = apply_lattice_constraint(coords, **p)
        role = "atomic_regularization"
    elif op == "atomic.displacement":
        y0, y1, x0, x1 = p.pop("reference_region")
        p.pop("reference_atom_indices")
        mask = (
            (coords[:, 0] >= y0) & (coords[:, 0] < y1) & (coords[:, 1] >= x0) & (coords[:, 1] < x1)
        )
        result = compute_atomic_displacement_from_basis(
            coords, **p, reference_mask=mask, allow_reference_fallback=False
        )
        role = "atomic_displacement"
    elif op.startswith("atomic.spacing."):
        role = "atomic_spacing"
        variant = op.split(".")[-1]
        if "directions" in p:
            p["directions"] = tuple(p["directions"])
        if variant == "indexed":
            src = p.pop("lattice_role")
            result = spacing.compute_atomic_plane_spacing_segments(
                coords,
                f.array("atomic.displacement", src + ".lattice_indices_ij"),
                valid_atom_mask=f.array("atomic.displacement", src + ".valid_atom_mask"),
                **p,
            )
        else:
            fn = {
                "basis": spacing.compute_atomic_plane_spacing_segments_from_basis,
                "lattice_angle": spacing.compute_atomic_plane_spacing_segments_from_lattice_angle,
                "angle": spacing.compute_atomic_plane_spacing_segments_by_angle,
                "picked": spacing.compute_atomic_plane_spacing_segments_from_picked_direction,
            }[variant]
            result = fn(coords, **p)
    elif op.startswith("atomic.sublattice."):
        role = "sublattices"
        if op.endswith(".auto"):
            result = sublattice.classify_sublattices_auto(coords, intensities=intensity, **p)
        else:
            model = dict(f.claims["atomic.sublattice.model"]["parameters"])
            model.pop("coordinate_role")
            result = sublattice.classify_sublattices_from_model(
                coords, intensities=intensity, **model
            )
            if op.endswith(".manual"):
                result = sublattice.apply_manual_sublattice_labels(
                    result, {int(k): v for k, v in p["manual_labels"].items()}
                )
    elif op == "atomic.intensity_groups":
        role = "intensity_groups"
        kwargs = dict(f.claims["atomic.intensity"]["parameters"])
        kwargs.pop("coordinate_role")
        intensities = quantify_atomic_column_intensity(
            f.array("input.load", "raw"), coords, **kwargs
        )
        intensities.valid_atom_mask &= (
            ~intensities.edge_flag & ~intensities.overlap_flag & ~intensities.saturation_flag
        )
        result = group_atomic_column_intensities_by_lattice(
            intensities,
            f.array("atomic.displacement", "atomic_displacement.lattice_indices_ij"),
            metric=p["metric"],
            mode=p["mode"],
            valid_lattice_mask=f.array(
                "atomic.displacement", "atomic_displacement.valid_atom_mask"
            ),
        )
        # The service adds the parent intensity step's detector assessment to its grouped artifact.
        if kwargs.get("saturation_threshold") is None:
            result.warnings.append(SATURATION_THRESHOLD_MISSING)
    else:
        raise ValueError(f"No direct comparison for {op}")
    return {
        "direct_fields": dataclass_outputs(f, op, role, result),
        "method": "B/C direct dataclass fields, registered full arrays and scalar payload",
    }


def data_operation(f, op, p, output):
    from autotem.calibration.basic import calibrate_dataset
    from autotem.calibration.pixel_calibration import infer_pixel_size_from_scalebar
    from autotem.preprocessing.basic import assess_quality, preprocess_dataset
    from autotem.preprocessing.quality import load_dataset_array
    from autotem.services.data_quality import QualityParameters

    data = TEMDataset.model_validate(f.value("input.load", "dataset"))
    raw = f.array("input.load", "raw")
    data.data.uri = str(f.ref("input.load", "raw"))
    if op == "input.load":
        fixture = next((f.root / "fixtures").glob(f.case_id + " *"))
        arrays_equal(raw, read_fixture(fixture))
        require(
            tuple(raw.shape) == tuple(data.data.shape) and str(raw.dtype) == data.data.dtype,
            "Wrong input dtype or shape",
        )
        return {
            "method": "Independent disk decoder and selected raw y,x, shape and dtype",
            "equivalence": "analytic",
        }
    if op == "quality.assess":
        result = assess_quality(
            data, parameters=QualityParameters.model_validate(p).model_dump(exclude_none=True)
        )
        equal(f.value(op, "quality"), result.model_dump(mode="json"))
    elif op == "calibration.resolve":
        values = {k: v for k, v in p.items() if k != "require_physical" and v is not None}
        reference = {
            k: values[k]
            for k in ("reference_length", "reference_pixels", "reference_unit")
            if k in values
        }
        direct = calibrate_dataset(
            data.model_copy(
                update={
                    "data": ArrayReference(shape=data.data.shape, dtype=data.data.dtype),
                    "source_path": "",
                },
                deep=True,
            ),
            reference=reference,
            parameters=values,
        )
        observed = f.value(op, "calibration")
        key = "diffraction_scale" if data.modality in {"SAED", "NBED"} else "pixel_size"
        expected = getattr(direct, key)
        for field in ("value", "unit", "method"):
            equal(observed[key][field], getattr(expected, field))
        if key == "pixel_size":
            scale = expected.value
            equal(
                observed["transform"]["fft_per_axis_inv_nm_per_px"],
                {"x": 1 / (raw.shape[1] * scale), "y": 1 / (raw.shape[0] * scale)},
            )
    elif op == "preprocessing.apply":
        v = {k: x for k, x in p.items() if k != "recipe" and x is not None}
        scratch = output / ("direct-preprocess-" + f.case_id)
        scratch.mkdir(exist_ok=True)
        result = preprocess_dataset(data, recipe=p["recipe"], parameters=v, output_dir=scratch)
        arr = load_dataset_array(data.model_copy(update={"data": result.image}))
        arrays_equal(f.array(op, "processed"), arr)
    elif op == "calibration.scalebar_candidate":
        crop = f.array(op, "scalebar_roi")
        arrays_equal(crop, raw)
        value, source, _ = infer_pixel_size_from_scalebar(
            crop, metadata={"explicit_reference": f"{p['reference_length']} {p['reference_unit']}"}
        )
        actual = f.value(op, "scalebar_candidate")
        equal(
            [actual["pixel_size_nm"], actual["source"], actual["requires_review"]],
            [value, source, True],
        )
    return {"method": "Direct API with registered measurements, parameters and full input"}


def diffraction(f, op, p):
    from autotem.diffraction.b4_indexing import run_b4_indexing
    from autotem.diffraction.fft import locate_direct_beam
    from autotem.diffraction.peak_detection import (
        detect_bragg_peaks,
        suggest_peak_parameter_defaults,
    )
    from autotem.diffraction.phase_lookup import lookup_phase_candidates

    if op == "phase.lookup":
        answer = lookup_phase_candidates(
            formula=p["formula"],
            elements=p["elements"],
            material_names=p["material_names"],
            sources=["local_builtin"],
            limit_per_source=p["limit_per_source"],
            timeout_s=p["timeout_s"],
            materials_project_ids=p["materials_project_ids"],
            materials_project_api_isolate=False,
        )
        wire = f.value(op, "phase_lookup")
        equal(wire["provider_candidates"], answer.candidates)
        require(
            wire["network_used"] is False
            and len(f.value(op, "phase_candidates")) == len(answer.candidates),
            "Offline candidates",
        )
        return {"method": "Direct offline provider candidate cell, atom sites and provenance"}
    if op == "diffraction.index":
        d = DiffractionResult.model_validate(f.value("diffraction.metrics", "diffraction"))
        result = run_b4_indexing(
            peaks=d.peaks,
            g_vectors=d.g_vectors,
            d_by_peak=[
                Measurement.model_validate(v) if v else None for v in d.provenance["d_by_peak"]
            ],
            initial_indexed_spots=[],
            candidate_phase_params=[
                {k: v for k, v in row.items() if v is not None}
                for row in p["candidate_phase_params"]
            ],
            allowed_phase_names=None,
            **{k: v for k, v in p.items() if k != "candidate_phase_params"},
        )
        wire = f.value(op, "diffraction")
        for field in ("phase_candidates", "indexed_spots"):
            equal(wire[field], [v.model_dump(mode="json") for v in getattr(result, field)])
        return {
            "method": "Direct B4 with per-peak measurements, indexed spots and phase candidates"
        }
    direct = f.case_id.startswith("J05")
    source = "diffraction.pattern" if direct else "diffraction.fft"
    if op in {"diffraction.metrics", "diffraction.fft"}:
        if direct:
            check_direct_pattern(FinalRoles(f))
        else:
            check_diffraction(FinalRoles(f), with_candidate="diffraction.index" in f.claims)
        return {
            "method": "Complex FFT, magnitude, calibrated axes and inverse vector norm"
            if not direct
            else "Detector reciprocal scale, every source peak, axis and inverse norm",
            "equivalence": "analytic" if op == "diffraction.metrics" else "direct_api",
        }
    arr = f.array(source, "reciprocal_image")
    if op == "diffraction.pattern":
        arrays_equal(arr, f.array("input.load", "raw"))
        require(
            f.value(op, "reciprocal_context")["kind"] == "direct_diffraction",
            "Wrong reciprocal context",
        )
        return {
            "method": "Full detector array equals the registered reciprocal intensity",
            "equivalence": "analytic",
        }
    center = f.value("diffraction.center", "center")
    xy = (center["x"], center["y"])
    if op == "diffraction.center":
        expected, _, _ = locate_direct_beam(
            arr, **{k: p[k] for k in ("center_mode", "search_radius_px", "refine_window")}
        )
        equal(xy, expected)
    elif op == "diffraction.peaks.suggest":
        result = suggest_peak_parameter_defaults(np.log1p(arr.astype(float)))
        result["top_k"] = result.pop("top_k_peaks")
        result["min_distance"] = result.pop("min_peak_distance")
        equal(f.value(op, "peak_parameter_suggestions"), result)
    elif op == "diffraction.peaks":
        direct_peaks, _, _ = detect_bragg_peaks(np.log1p(arr.astype(float)), center=xy, **p)
        equal(f.value(op, "peaks"), [v.model_dump(mode="json") for v in direct_peaks])
    elif op == "diffraction.peaks.manual":
        rows = f.value(op, "peaks")
        require(len(rows) == len(p["peaks"]), "Manual peak count")
        for row, coord in zip(rows, p["peaks"], strict=True):
            x, y = coord["x"], coord["y"]
            equal([row["x"], row["y"], row["intensity"]], [x, y, float(arr[round(y), round(x)])])
            require(
                any("uncertainty" in w for w in row["warnings"]), "Missing manual point limitation"
            )
        return {
            "method": "All supplied x,y, detector intensities and uncertainty limits",
            "equivalence": "analytic",
        }
    else:
        raise ValueError(f"Unknown comparison: {op}")
    return {"method": "Direct detector or estimator with full reciprocal image and recorded center"}


def delivered(f, op, p):
    if op == "maintenance.dependencies":
        from autotem.harness.registry import default_registry

        equal(f.value(op, "dependency_diagnostics"), default_registry().capabilities())
        return {
            "method": "Full current registry compared with every persisted availability/schema row",
            "equivalence": "registry",
        }
    from autotem.validation.benchmarks import run_benchmark_summary, run_validation_summary

    report = AnalysisReport(
        summary="Registered headless results; scientific accuracy not_evaluated",
        dataset=TEMDataset.model_validate(f.value("input.load", "dataset")),
        atom_lattice=AtomLattice.model_validate(f.value("atomic.refine", "refined_atoms")),
        provenance={
            "run_id": f.run_id,
            "primary_strain": p["strain_role"],
            "scientific_accuracy": "not_evaluated",
        },
    )
    role = op.split(".")[0]
    fn = run_benchmark_summary if role == "benchmark" else run_validation_summary
    _, payload, _ = fn(report, {"acceptance_criteria": p["acceptance_criteria"]})
    wire = f.value(op, role)
    equal(wire["acceptance"], payload["acceptance"])
    equal(wire["metrics"], payload["metrics"])
    return {"method": "Direct validation/benchmark metrics and acceptance on 42 measured sites"}


def expected_cases(op, frames):
    names = [n for n, f in frames.items() if op in f.claims]
    if op in (
        "input.load",
        "quality.assess",
        "calibration.resolve",
        "preprocessing.apply",
        "diffraction.metrics",
        "diffraction.pattern",
    ):
        names = [n for n in names if n in {"J01", "J02", "J05-SAED", "J05-NBED"}]
    elif op in ("diffraction.center", "diffraction.peaks", "diffraction.peaks.suggest"):
        names = ["J05-SAED"]
    elif op in ("atomic.detect", "atomic.refine", "atomic.intensity", "atomic.distances"):
        names = ["J02"]
    else:
        names = names[:1]
    require(names, f"Missing recorded case for {op}")
    return names


def verify(op, frames, matrix, extra, output):
    names = expected_cases(op, frames)
    results = []
    for name in names:
        f = frames[name]
        claim = f.claims[op]
        p = claim["parameters"]
        if op in {
            "quality.assess",
            "input.load",
            "calibration.resolve",
            "preprocessing.apply",
            "calibration.scalebar_candidate",
        }:
            result = data_operation(f, op, p, output)
        elif op.startswith(("diffraction.", "phase.")):
            result = diffraction(f, op, p)
        elif op in {"atomic.detect", "atomic.refine", "atomic.intensity"}:
            import json

            metadata = json.loads((f.root / "evidence.json").read_text("utf-8"))
            fixture = next(c for c in metadata["fixtures"] if c["case_id"] == name)
            check_atoms(FinalRoles(f), fixture)
            result = {
                "method": "Installed direct detector, raw fit/status, relative intensity and masks"
            }
        elif op.startswith("atomic."):
            result = atomic_variant(f, op, p)
        elif op == "strain.gpa" or op == "strain.atom":
            (check_gpa if op == "strain.gpa" else check_atom_strain)(FinalRoles(f))
            result = {"method": "Direct strain arrays, reference/valid masks and coordinate axes"}
        elif op == "strain.gpa_compare":
            from autotem.strain.gpa_alignment import GPAAlignmentConfig, compare_gpa_configs

            configs = []
            for k in ("reference_config", "candidate_config"):
                conf = {
                    k: v
                    for k, v in p[k].items()
                    if k not in ("image_role", "absolute_reference", "reference_regions")
                }
                conf["reference_region"] = tuple(conf["reference_region"])
                configs.append(GPAAlignmentConfig(**conf))
            result = compare_gpa_configs(
                f.array("input.load", "raw"), *configs, sample_step=p["sample_step"]
            )
            wire = f.value(op, "gpa_comparison")
            equal(
                {k: v for k, v in wire.items() if k != "scientific_limitation"}, clean_json(result)
            )
            require(
                "not independent" in wire["scientific_limitation"], "Missing GPA comparison limit"
            )
            result = {
                "method": "Direct comparison metrics for two identical analytic GPA configurations"
            }
        else:
            result = delivered(f, op, p)
        result.setdefault("equivalence", "direct_api")
        result.update(
            case=name,
            run_id=f.run_id,
            outputs={
                r: {i: f.audit[i]["sha256"] for i in ids}
                for r, ids in claim["artifacts_by_role"].items()
            },
            consumed_artifacts=claim["consumed_artifacts"],
        )
        results.append(result)
    return {
        "comparisons": results,
        "tolerance": (
            "Exact shape, discrete values and labels. Same-runtime binary64 rtol=atol=1e-12 "
            "except the documented installed FFT, raw fits/intensity and strain checks."
        ),
    }
