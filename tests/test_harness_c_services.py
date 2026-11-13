"""Execute every traditional C handler, checking numerical and invalid-input contracts."""

from __future__ import annotations

import os
import time
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from autotem.atomic.atom_detection import detect_atoms
from autotem.atomic.displacement import compute_atomic_displacement_from_basis
from autotem.atomic.gaussian_refinement import refine_atom_coordinates
from autotem.atomic.intensity_quantification import quantify_atomic_column_intensity
from autotem.atomic.sublattice import classify_sublattices_auto
from autotem.core.models import (
    ArrayReference,
    AtomLattice,
    AtomSite,
    CalibrationResult,
    DiffractionResult,
    GVector,
    Measurement,
)
from autotem.harness.contracts import (
    HarnessError,
    NeedsInput,
    OperationContext,
    OperationDefinition,
)
from autotem.services import atomic, strain
from autotem.strain.atom_strain import compute_atom_strain_fields
from autotem.strain.gpa import compute_gpa_fields


class Context:
    """Small file-backed context; numerical algorithms execute without mocks."""

    def __init__(self, directory: Path):
        self.output_dir = directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = "c-service-test"
        self.input_path = directory / "raw.npy"
        self.request = {}
        self.values = {}
        self.arrays = {}
        self.images = {}
        self.tables = {}
        self.array_metadata = {}

    def array(self, role):
        if role not in self.arrays:
            raise HarnessError("MISSING_ARTIFACT", role)
        return self.arrays[role]

    def value(self, role):
        if role not in self.values:
            raise HarnessError("MISSING_ARTIFACT", role)
        return self.values[role]

    def model(self, role, cls):
        value = self.value(role)
        return cls.model_validate(value.model_dump() if hasattr(value, "model_dump") else value)

    def emit_array(self, role, array, metadata=None):
        array = np.asarray(array)
        path = self.output_dir / (role.replace(".", "_") + ".npy")
        np.save(path, array, allow_pickle=False)
        self.arrays[role] = array.copy()
        self.array_metadata[role] = metadata or {}
        return ArrayReference(shape=array.shape, dtype=str(array.dtype), uri=str(path))

    def emit_json(self, role, value, metadata=None):
        self.values[role] = value
        return role

    def resolve_array(self, reference):
        return np.load(reference["uri"], allow_pickle=False)

    def emit_table(self, role, rows, metadata=None):
        self.tables[role] = rows
        return role

    def emit_image(self, role, image, metadata=None):
        self.images[role] = image
        return role

    def accept(self, result):
        self.values.update(result.outputs)
        return result


@pytest.fixture
def context(tmp_path):
    ctx = Context(tmp_path)
    coords = np.asarray([(y, x) for y in range(12, 61, 8) for x in range(12, 61, 8)], float)
    yy, xx = np.indices((80, 80), dtype=float)
    raw = np.full((80, 80), 10.0)
    atoms = []
    for i, (y, x) in enumerate(coords):
        amplitude = 30.0 + 20.0 * (i % 2)
        raw += amplitude * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.1**2))
        atoms.append(AtomSite(y=y, x=x))
    lattice = AtomLattice(atoms=atoms, coordinate_unit="pixel")
    ctx.arrays.update(raw=raw, processed=raw.copy())
    np.save(ctx.input_path, raw)
    ctx.values.update(atoms=lattice, refined_atoms=lattice)
    reference = np.arange(21)
    mask = np.zeros(len(coords), bool)
    mask[reference] = True
    displacement = compute_atomic_displacement_from_basis(
        coords,
        base_index=0,
        a_index=1,
        b_index=7,
        reference_mask=mask,
        allow_reference_fallback=False,
    )
    ctx.values["atomic_displacement"] = atomic.save_dataclass(
        ctx, "atomic_displacement", displacement
    )
    sublattices = classify_sublattices_auto(
        coords, base_index=0, a_index=1, b_index=7, use_intensity=False
    )
    ctx.values["sublattices"] = atomic.save_dataclass(ctx, "sublattices", sublattices)
    ctx.accept(
        atomic.intensity(
            ctx,
            {
                "integration_radius": 1.5,
                "background_inner_radius": 2.0,
                "background_outer_radius": 3.0,
            },
        )
    )
    return ctx


OPERATIONS = {d.operation_id: d for d in [*atomic.definitions(), *strain.definitions()]}
BASE = {"base_index": 0, "a_index": 1, "b_index": 7}
GPA = {"g_vectors": [[0.125, 0.0], [0.0, 0.125]], "reference_region": [16, 48, 16, 48]}
CASES = [
    ("atomic.detect", {}, "atoms"),
    ("atomic.refine", {}, "refined_atoms"),
    ("atomic.distances", {}, "atom_distances"),
    ("atomic.regularize", {}, "regularized_atoms"),
    (
        "atomic.displacement",
        {**BASE, "reference_atom_indices": list(range(21))},
        "atomic_displacement",
    ),
    ("atomic.spacing.indexed", {}, "atomic_spacing"),
    ("atomic.spacing.basis", BASE, "atomic_spacing"),
    ("atomic.spacing.lattice_angle", {"lattice_angle_deg": 90.0}, "atomic_spacing"),
    ("atomic.spacing.angle", {"angle_deg": 0.0}, "atomic_spacing"),
    ("atomic.spacing.picked", {"start_index": 0, "end_index": 1}, "atomic_spacing"),
    ("atomic.sublattice.auto", {**BASE, "use_intensity": False}, "sublattices"),
    (
        "atomic.sublattice.model",
        {**BASE, "basis_sites": [{"u": 0.0, "v": 0.0, "label": "A"}]},
        "sublattices",
    ),
    ("atomic.sublattice.manual", {"manual_labels": {"0": "reviewed-A"}}, "sublattices"),
    (
        "atomic.intensity",
        {"integration_radius": 1.5, "background_inner_radius": 2.0, "background_outer_radius": 3.0},
        "atomic_intensity",
    ),
    ("atomic.intensity_groups", {}, "intensity_groups"),
    ("strain.gpa", GPA, "gpa"),
    ("strain.gpa_compare", {"reference_config": GPA, "candidate_config": GPA}, "gpa_comparison"),
    ("strain.atom", {"reference_region": [8, 36, 8, 68]}, "atom_strain"),
]


def gpa_image(ctx):
    yy, xx = np.indices(ctx.array("raw").shape, dtype=float)
    ctx.arrays["raw"] = np.cos(2 * np.pi * xx / 8) + np.cos(2 * np.pi * yy / 8)


@pytest.mark.parametrize("operation_id,params,output_role", CASES)
def test_every_operation_executes_real_algorithm(context, operation_id, params, output_role):
    if operation_id.startswith("strain.gpa"):
        gpa_image(context)
    result = OPERATIONS[operation_id].handler(context, params)
    assert output_role in result.outputs
    assert OPERATIONS[operation_id].profile_id == "atomic-strain"


@pytest.mark.parametrize("operation_id,params,output_role", CASES)
def test_every_operation_rejects_unknown_parameters(context, operation_id, params, output_role):
    with pytest.raises(HarnessError, match="unexpected_field") as error:
        OPERATIONS[operation_id].handler(context, {**params, "unexpected_field": True})
    assert error.value.code == "INVALID_PARAMETERS"


def test_detection_and_raw_refinement_match_direct_api(context):
    from PIL import Image

    direct = detect_atoms(context.array("raw"))
    output = context.accept(atomic.detect(context, {})).outputs["atoms"]
    observed = np.asarray([[a.y, a.x] for a in output.atoms])
    np.testing.assert_array_equal(observed, direct.coordinates)
    assert len(observed) > 20
    direct_refined = refine_atom_coordinates(context.array("raw"), observed)
    refined = atomic.refine(context, {}).outputs["refined_atoms"]
    np.testing.assert_allclose(
        [[a.y, a.x] for a in refined.atoms], direct_refined.coordinates_refined
    )
    assert refined.provenance["raw_refinement"]["success_count"] > 20
    assert output.provenance["overlay_artifact_id"] == "atom_candidates_overlay"
    assert refined.provenance["overlay_artifact_id"] == "refined_atoms_overlay"
    for role in ("atom_candidates_overlay", "refined_atoms_overlay"):
        with Image.open(context.images[role]) as overlay:
            overlay.verify()


def test_distance_units_convert_actual_numbers_and_keep_pixel_coordinates(context):
    nm = atomic.distances(context, {"pixel_size_nm": 0.02, "output_unit": "nm"}).outputs[
        "atom_distances"
    ]
    pm = atomic.distances(context, {"pixel_size_nm": 0.02, "output_unit": "pm"}).outputs[
        "atom_distances"
    ]
    np.testing.assert_allclose(nm["nearest_distances"], 0.16)
    np.testing.assert_allclose(pm["nearest_distances"], 160.0)
    np.testing.assert_array_equal(nm["coordinates"], pm["coordinates"])
    assert nm["metadata"]["coordinate_unit"] == "pixel"
    assert nm["metadata"]["coordinates_sha256"]


@pytest.mark.parametrize(
    "params",
    [
        {"pixel_size_nm": float("nan")},
        {"pixel_size_nm": float("inf")},
        {"pixel_size_nm": True},
        {"pixel_size_nm": -1.0},
        {"pixel_size_nm": 0.02},
        {"output_unit": "unknown"},
    ],
)
def test_distance_rejects_invalid_calibration(context, params):
    with pytest.raises(HarnessError):
        atomic.distances(context, params)


def test_missing_physical_calibration_is_needs_input(context):
    with pytest.raises(NeedsInput) as error:
        atomic.distances(context, {"output_unit": "nm"})
    assert error.value.details["fields"][0]["name"] == "pixel_size_nm"


@pytest.mark.parametrize("unit", ["nm", "unknown", "angstrom"])
def test_coordinate_unit_cannot_be_relabelled(context, unit):
    context.values["refined_atoms"] = context.values["refined_atoms"].model_copy(
        update={"coordinate_unit": unit}
    )
    with pytest.raises(HarnessError) as error:
        atomic.distances(context, {})
    assert error.value.code == "INVALID_UNIT"


@pytest.mark.parametrize("point", [(np.nan, 2.0), (2.0, np.inf), (-1.0, 2.0), (81.0, 2.0)])
def test_raw_based_operations_reject_bad_coordinates(context, point):
    lattice = AtomLattice(atoms=[AtomSite(y=point[0], x=point[1])], coordinate_unit="pixel")
    context.values["atoms"] = lattice
    with pytest.raises(HarnessError):
        atomic.refine(context, {})


def test_missing_basis_emits_numbered_selection_evidence(context):
    with pytest.raises(NeedsInput) as error:
        atomic.displacement(context, {})
    assert {f["name"] for f in error.value.details["fields"]} == set(BASE)
    assert len(context.tables["atom_selection"]) == 49
    assert "numbered_atoms" in context.images


def test_displacement_no_reference_fallback_and_numeric_equivalence(context):
    _, coords = atomic.coordinates(context)
    with pytest.raises(NeedsInput):
        atomic.displacement(context, BASE)
    with pytest.raises(HarnessError):
        atomic.displacement(context, {**BASE, "reference_atom_indices": [0, 1, 2]})
    result = atomic.displacement(context, {**BASE, "reference_atom_indices": list(range(21))})
    mask = np.arange(len(coords)) < 21
    direct = compute_atomic_displacement_from_basis(
        coords, **BASE, reference_mask=mask, allow_reference_fallback=False
    )
    payload = result.outputs["atomic_displacement"]
    np.testing.assert_array_equal(payload["valid_atom_mask"], direct.valid_atom_mask)
    np.testing.assert_allclose(payload["displacement_yx"], direct.displacement_yx)
    assert payload["provenance"]["parameters"]["allow_reference_fallback"] is False


def test_all_spacing_variants_measure_projected_grid_segments(context):
    for operation_id, params, _ in CASES:
        if operation_id.startswith("atomic.spacing"):
            result = OPERATIONS[operation_id].handler(context, params).outputs["atomic_spacing"]
            assert result["length_px"]
            lengths = np.asarray(result["length_px"])
            assert np.median(lengths) == pytest.approx(8.0)
            np.testing.assert_allclose(np.mod(lengths, 8.0), 0.0)


def test_indexed_spacing_rejects_mixed_coordinate_revision(context):
    context.values["refined_atoms"].atoms[0].y += 0.5
    with pytest.raises(HarnessError) as error:
        atomic.spacing(context, {}, "indexed")
    assert error.value.code == "MIXED_COORDINATE_REVISION"


def test_regularizer_default_is_identity_and_separate_role(context):
    result = atomic.regularize(context, {})
    assert "regularized_atoms" in result.outputs and "refined_atoms" not in result.outputs
    actual = result.outputs["regularized_atoms"]
    assert [(a.y, a.x) for a in actual.atoms] == [
        (a.y, a.x) for a in context.values["refined_atoms"].atoms
    ]


def test_manual_sublattice_uses_explicit_prior_result(context):
    result = atomic.sublattice(context, {"manual_labels": {"0": "Cu"}}, "manual")
    payload = result.outputs["sublattices"]
    assert payload["labels"][0] == "Cu"
    assert payload["manual_override_mask"][0]


def test_intensity_is_from_raw_and_rejects_unsafe_group_members(context):
    context.arrays["processed"] = np.zeros_like(context.array("raw"))
    p = {
        "integration_radius": 1.5,
        "background_inner_radius": 2.0,
        "background_outer_radius": 3.0,
        "saturation_threshold": 20.0,
    }
    direct = quantify_atomic_column_intensity(
        context.array("raw"), atomic.coordinates(context)[1], **p
    )
    output = context.accept(atomic.intensity(context, p)).outputs["atomic_intensity"]
    np.testing.assert_allclose(output["raw_sum"], direct.raw_sum)
    assert context.array_metadata["atomic_intensity.coordinates_yx"]["unit"] == "pixel"
    assert (
        context.array_metadata["atomic_intensity.normalized_intensity"]["unit"] == "dimensionless"
    )
    assert context.array_metadata["atomic_intensity.atom_area_px"]["unit"] == "pixel^2"
    assert context.array_metadata["atomic_intensity.saturation_flag"]["unit"] == "boolean"
    assert any(output["saturation_flag"])
    assert not any(output["trusted_atom_mask"])
    grouped = atomic.intensity_groups(context, {}).outputs["intensity_groups"]
    assert all(label == -1 for label in grouped["labels"])


def test_intensity_lattice_groups_execute(context):
    result = atomic.intensity_groups(context, {"grouping": "lattice", "mode": "sum_parity"})
    assert len(result.outputs["intensity_groups"]["summaries"]) == 2


def test_sublattice_lattice_groups_preserve_subpixel_coordinate_revision(context):
    for atom in context.values["refined_atoms"].atoms:
        atom.y += 0.123456789
        atom.x += 0.234567891
    context.accept(atomic.sublattice(context, {**BASE, "use_intensity": False}, "auto"))
    context.accept(
        atomic.intensity(
            context,
            {
                "integration_radius": 1.5,
                "background_inner_radius": 2.0,
                "background_outer_radius": 3.0,
            },
        )
    )
    result = atomic.intensity_groups(
        context, {"grouping": "lattice", "lattice_role": "sublattices"}
    )
    assert result.outputs["intensity_groups"]["summaries"]


def test_gpa_saves_full_arrays_mask_and_matches_direct_numeric_fields(context):
    from PIL import Image

    gpa_image(context)
    direct = compute_gpa_fields(
        context.array("raw"),
        g_vectors=GPA["g_vectors"],
        reference_region=tuple(GPA["reference_region"]),
    )
    result = strain.gpa(context, GPA).outputs["gpa"]
    assert result.strain_xx.inline is None
    assert result.strain_xx.shape == context.array("raw").shape
    np.testing.assert_allclose(np.load(result.strain_xx.uri), direct.strain_xx)
    mask = result.provenance["quality_mask"]
    np.testing.assert_array_equal(np.load(mask["uri"]), direct.quality_mask)
    assert (
        context.array_metadata["gpa.strain_xx"]["validity_sha256"]
        == sha256(np.load(mask["uri"]).tobytes()).hexdigest()
    )
    wrapped = result.provenance["phase_maps"]["g1"]["phase_wrapped"]
    assert wrapped["shape"] == [80, 80] and wrapped["inline"] is None
    assert context.array_metadata["gpa.g1.gaussian_mask"]["unit"] == "dimensionless"
    assert result.provenance["complete_arrays"]
    assert result.provenance["static_map_artifact_id"] == "gpa_maps"
    with Image.open(context.images["gpa_maps"]) as montage:
        assert min(montage.size) > 500
        montage.verify()


@pytest.mark.parametrize(
    "params",
    [
        {"g_vectors": [[0.125, 0.0], [0.25, 0.0]], "absolute_reference": True},
        {**GPA, "reference_region": [0, 1, 0, 1]},
        {**GPA, "reference_region": [0, 90, 0, 90]},
        {**GPA, "g_vector_unit": "pixel"},
        {**GPA, "g_vectors": [[float("nan"), 0.0], [0.0, 0.125]]},
    ],
)
def test_gpa_rejects_bad_reference_units_vectors(context, params):
    gpa_image(context)
    with pytest.raises(HarnessError):
        strain.gpa(context, params)


def test_gpa_missing_reference_is_needs_input(context):
    gpa_image(context)
    with pytest.raises(NeedsInput):
        strain.gpa(context, {"g_vectors": GPA["g_vectors"]})


def test_gpa_automatic_physical_vectors_use_real_calibration(context):
    gpa_image(context)
    context.values["diffraction"] = DiffractionResult(
        g_vectors=[
            GVector(
                gx=Measurement(value=6.25, unit="1/nm"), gy=Measurement(value=0.0, unit="1/nm")
            ),
            GVector(
                gx=Measurement(value=0.0, unit="1/nm"), gy=Measurement(value=6.25, unit="1/nm")
            ),
        ]
    )
    with pytest.raises(NeedsInput):
        strain.gpa(context, {"reference_region": GPA["reference_region"]})
    context.values["calibration"] = CalibrationResult(
        pixel_size=Measurement(value=0.02, unit="nm/pixel"), pixel_unit="nm"
    )
    actual = strain.gpa(context, {"reference_region": GPA["reference_region"]}).outputs["gpa"]
    direct = compute_gpa_fields(
        context.array("raw"),
        g_vectors=GPA["g_vectors"],
        reference_region=tuple(GPA["reference_region"]),
    )
    np.testing.assert_allclose(np.load(actual.strain_xx.uri), direct.strain_xx)


def test_gpa_auto_mixed_component_units_fail_and_collinear_candidates_need_input(context):
    gpa_image(context)
    vectors = [
        GVector(
            gx=Measurement(value=0.125, unit="1/pixel"), gy=Measurement(value=0.0, unit="1/pixel")
        )
        for _ in range(2)
    ]
    context.values["diffraction"] = DiffractionResult(g_vectors=vectors)
    with pytest.raises(NeedsInput):
        strain.gpa(context, {"reference_region": GPA["reference_region"]})
    vectors[0].gy.unit = "1/nm"
    with pytest.raises(HarnessError) as error:
        strain.gpa(context, {"reference_region": GPA["reference_region"]})
    assert error.value.code == "INVALID_UNIT"


def test_gpa_compare_identical_configs_and_no_false_independent_validation(context):
    gpa_image(context)
    output = strain.gpa_compare(context, {"reference_config": GPA, "candidate_config": GPA})
    assert output.outputs["gpa_comparison"]["diagnosis"]["max_strain_mae_percent"] == 0.0
    assert "not independent" in output.outputs["gpa_comparison"]["scientific_limitation"]


def test_atom_strain_full_arrays_and_per_atom_validity_match_direct(context):
    from PIL import Image

    region = [8, 36, 8, 68]
    coords = atomic.coordinates(context)[1]
    direct = compute_atom_strain_fields(
        coords, context.array("raw").shape, reference_region=tuple(region)
    )
    output = strain.atom_strain(context, {"reference_region": region}).outputs["atom_strain"]
    assert output.strain_xx.inline is None
    np.testing.assert_allclose(np.load(output.strain_xx.uri), direct.strain_xx_map)
    refs = output.provenance["array_references"]
    np.testing.assert_array_equal(np.load(refs["valid_atom_mask"]["uri"]), direct.valid_atom_mask)
    metadata = context.array_metadata["atom_strain.strain_xx_map"]
    assert (
        metadata["coordinates_sha256"]
        == sha256(
            np.asarray(np.load(refs["coordinates_yx"]["uri"]), dtype="<f8").tobytes()
        ).hexdigest()
    )
    assert (
        metadata["validity_sha256"]
        == sha256(np.load(refs["valid_atom_mask"]["uri"]).tobytes()).hexdigest()
    )
    assert output.provenance["map_semantics"].startswith("display_interpolation")
    assert (
        context.array_metadata["atom_strain.strain_xx_map"]["semantics"] == "display_interpolation"
    )
    assert output.provenance["static_map_artifact_id"] == "atom_strain_maps"
    with Image.open(context.images["atom_strain_maps"]) as montage:
        assert min(montage.size) > 500
        montage.verify()


def test_atom_strain_small_reference_does_not_fallback(context):
    with pytest.raises(NeedsInput):
        strain.atom_strain(context, {"reference_region": [10, 16, 10, 16]})
    assert "atom_strain.strain_xx_map" not in context.arrays


def test_atom_strain_explicit_reference_atoms(context):
    output = strain.atom_strain(context, {"reference_atom_indices": list(range(21))})
    assert output.outputs["atom_strain"].provenance["reference_atom_count"] == 21


def test_unbounded_nonfinite_image_cannot_enter_scientific_algorithms(context):
    context.arrays["raw"][0, 0] = np.nan
    with pytest.raises(HarnessError) as error:
        atomic.detect(context, {})
    assert error.value.code == "INVALID_ARRAY"


def test_harness_raw_to_atomic_and_both_strain_results(context, tmp_path):
    """The actual worker/Store must consume a source file and preserve full C artifacts."""
    from autotem.harness.service import Harness

    harness = Harness(tmp_path / "real-harness")
    plan = {
        "input_path": str(context.input_path),
        "modality": "HAADF-STEM",
        "budgets": {"run_seconds": 60.0, "operation_seconds": 20.0},
        "steps": [
            {"id": "load", "operation_id": "input.load"},
            {"id": "detect", "operation_id": "atomic.detect", "depends_on": ["load"]},
            {"id": "refine", "operation_id": "atomic.refine", "depends_on": ["detect"]},
            {"id": "distance", "operation_id": "atomic.distances", "depends_on": ["refine"]},
            {
                "id": "intensity",
                "operation_id": "atomic.intensity",
                "depends_on": ["refine"],
                "parameters": {
                    "integration_radius": 1.5,
                    "background_inner_radius": 2.0,
                    "background_outer_radius": 3.0,
                },
            },
            {"id": "gpa", "operation_id": "strain.gpa", "depends_on": ["load"], "parameters": GPA},
            {
                "id": "atom_strain",
                "operation_id": "strain.atom",
                "depends_on": ["refine"],
                "parameters": {"reference_region": [8, 36, 8, 68]},
            },
        ],
    }
    result = harness.submit_analysis(plan, asynchronous=True)
    deadline = time.monotonic() + 60
    run_id = result["run_id"]
    while result["status"] not in {"completed", "failed", "needs_input", "timed_out", "cancelled"}:
        if time.monotonic() > deadline:
            harness.cancel_run(run_id)
            pytest.fail(f"Bounded harness C closure exceeded 60 seconds: {result}")
        time.sleep(0.1)
        result = harness.get_run_status(run_id)
    assert result["status"] == "completed", result
    for role in (
        "atoms",
        "refined_atoms",
        "atom_distances",
        "atomic_intensity",
        "gpa",
        "atom_strain",
    ):
        assert harness.store.role(run_id, role), (role, result)
    for role, field in (("gpa", "strain_xx"), ("atom_strain", "strain_xx")):
        payload = harness.store.load_json(harness.store.role(run_id, role))
        assert payload[field]["shape"] == [80, 80]
        assert payload[field]["inline"] is None
        assert np.load(payload[field]["uri"], allow_pickle=False).shape == (80, 80)
        assert "Diagnostic strain only" in payload["provenance"]["scientific_limitation"]
    np.testing.assert_array_equal(np.load(context.input_path), context.array("raw"))


def test_worker_honors_dynamic_coordinate_role_without_refinement(context, tmp_path):
    from autotem.atomic.distance_statistics import compute_distance_statistics
    from autotem.harness.service import Harness

    harness = Harness(tmp_path / "dynamic-role-harness")
    state = harness.submit_analysis(
        {
            "input_path": str(context.input_path),
            "modality": "HAADF-STEM",
            "steps": [
                {"id": "load", "operation_id": "input.load"},
                {"id": "detect", "operation_id": "atomic.detect", "depends_on": ["load"]},
                {
                    "id": "distance",
                    "operation_id": "atomic.distances",
                    "depends_on": ["detect"],
                    "parameters": {"coordinate_role": "atoms"},
                },
            ],
        },
        asynchronous=False,
    )
    assert state["status"] == state["jobs"][-1]["status"] == "completed"
    run_id = state["run_id"]
    assert harness.store.role(run_id, "refined_atoms") is None
    atoms = harness.store.load_json(harness.store.role(run_id, "atoms"))["atoms"]
    coords = np.array([[atom["y"], atom["x"]] for atom in atoms])
    np.testing.assert_allclose(
        harness.store.load_array(harness.store.role(run_id, "atom_distances.nearest_distances")),
        compute_distance_statistics(coords).nearest_distances,
    )


def test_failed_step_evidence_is_readable_but_cannot_reenter_calculation(
    context, tmp_path, monkeypatch
):
    from autotem.harness import service, worker
    from autotem.harness.registry import default_registry

    registry = default_registry()

    def fail_after_array(ctx, params):
        ctx.emit_array("processed", ctx.array("raw"))
        raise HarnessError("EXPECTED_TEST_FAILURE", "Abort after publishing partial evidence")

    registry.register(
        OperationDefinition(
            "review.partial",
            "atomic-strain",
            fail_after_array,
            input_roles=("raw",),
            output_roles=("processed",),
        )
    )
    monkeypatch.setattr(service, "default_registry", lambda *args: registry)
    monkeypatch.setattr(worker, "default_registry", lambda *args: registry)
    harness = service.Harness(tmp_path / "failed-step-harness")
    monkeypatch.setattr(harness, "_start_scheduler", lambda: None)
    submitted = harness.submit_analysis(
        {
            "input_path": str(context.input_path),
            "steps": [
                {"id": "load", "operation_id": "input.load"},
                {"id": "partial", "operation_id": "review.partial", "depends_on": ["load"]},
            ],
        }
    )
    run_id = submitted["run_id"]
    harness.store.update_run(
        run_id,
        status="running",
        worker_pid=os.getpid(),
        worker_identity=worker.process_identity(os.getpid()),
    )
    worker.execute(str(harness.root), run_id)
    # The injected handler runs in this process. The scheduler still owns job settlement.
    state = harness.get_run_status(run_id)
    assert state["status"] == "finalizing"
    assert state["steps"][-1]["status"] == "failed"
    assert state["steps"][-1]["error"]["code"] == "EXPECTED_TEST_FAILURE"
    artifact = next(a for a in harness.store.artifacts(run_id) if a["role"] == "processed")
    assert not artifact["execution_usable"]
    np.testing.assert_array_equal(
        harness.store.load_array(artifact["artifact_id"]), context.array("raw")
    )
    explicit = OperationContext(
        harness.store, run_id, "followup", {}, inputs={"processed": artifact["artifact_id"]}
    )
    with pytest.raises(HarnessError) as by_id:
        explicit.array("processed")
    assert by_id.value.code == "ARTIFACT_NOT_EXECUTABLE"
    with pytest.raises(HarnessError) as by_path:
        explicit.resolve_array({"uri": artifact["path"]})
    assert by_path.value.code == "ARTIFACT_NOT_EXECUTABLE"


def test_append_work_packages_cannot_exceed_bounded_queue(context, tmp_path, monkeypatch):
    from autotem.harness import worker
    from autotem.harness.service import Harness

    harness = Harness(tmp_path / "bounded-append-harness", queue_limit=1)
    monkeypatch.setattr(harness, "_start_scheduler", lambda: None)
    plan = {
        "input_path": str(context.input_path),
        "steps": [{"id": "load", "operation_id": "input.load"}],
    }
    runs = []
    for _ in range(2):
        run_id = harness.submit_analysis(plan)["run_id"]
        # Drain each seed through the scheduler before testing two competing appends.
        worker.schedule(str(harness.root))
        state = harness.get_run_status(run_id)
        assert state["status"] == state["jobs"][-1]["status"] == "completed"
        runs.append(run_id)
    package = {
        "input_path": str(context.input_path),
        "steps": [{"id": "quality", "operation_id": "quality.assess"}],
    }
    assert harness.submit_analysis(package, run_id=runs[0])["status"] == "queued"
    with pytest.raises(HarnessError) as error:
        harness.submit_analysis(package, run_id=runs[1])
    assert error.value.code == "QUEUE_FULL"


def test_missing_unwrap_backend_does_not_disable_atom_strain(context, monkeypatch):
    from autotem.harness import registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "probe_dependency",
        lambda name: {
            "state": "dependency_unavailable" if name == "skimage" else "available",
            "dependency": name,
        },
    )
    registry = registry_module.default_registry()
    assert registry.availability(registry.get("strain.gpa"))["state"] == "dependency_unavailable"
    assert registry.availability(registry.get("strain.atom"))["state"] == "available"
    output = strain.atom_strain(context, {"reference_region": [8, 36, 8, 68]})
    assert output.outputs["atom_strain"].provenance["complete_arrays"] is True
