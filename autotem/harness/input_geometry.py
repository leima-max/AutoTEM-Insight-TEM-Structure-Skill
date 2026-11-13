"""Check replies against the image and numbered evidence that prompted the question."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .contracts import HarnessError
from .input_requests import invalid


def selected_g_vectors(store, field, indices):
    if (
        any(type(i) is not int or not 0 <= i < field["count"] for i in indices)
        or len(set(indices)) != 2
    ):
        raise invalid("Choose two distinct displayed peak indices", "g_peak_indices")
    table = store._check(field["evidence"]["peak_table_artifact_id"])
    with Path(table["path"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = {int(row["peak_index"]): row for row in csv.DictReader(stream)}
    return [
        [float(rows[i]["gx_cycles_per_pixel"]), float(rows[i]["gy_cycles_per_pixel"])]
        for i in indices
    ]


def source_region(value, spec, name):
    """Transform source 2-D coordinates to coordinates of the selected raw array."""
    if not isinstance(value, list) or len(value) != 4 or any(type(v) is not int for v in value):
        raise invalid("Use four integer bounds [y0,y1,x0,x1], upper bounds exclusive", name)
    oy, ox = spec["origin_yx"]
    h, w = spec["shape_yx"]
    y0, y1, x0, x1 = value
    if not (oy <= y0 < y1 <= oy + h and ox <= x0 < x1 <= ox + w):
        raise invalid("ROI is outside the selected image", name, coordinates=spec)
    return [y0 - oy, y1 - oy, x0 - ox, x1 - ox]


def _validate_numbered_atoms(params, table_id, store, operation):
    from autotem.services.atomic import region_mask

    table = store._check(table_id)
    with Path(table["path"]).open(encoding="utf-8-sig", newline="") as stream:
        coords = np.array(
            [[float(row["y_px"]), float(row["x_px"])] for row in csv.DictReader(stream)]
        )
    for keys in (("base_index", "a_index", "b_index"), ("start_index", "end_index")):
        if any(params.get(key) is None for key in keys):
            continue
        indices = [params[key] for key in keys]
        if any(i < 0 or i >= len(coords) for i in indices) or len(set(indices)) != len(indices):
            raise invalid("Choose distinct indices from the numbered atom table", keys[0])
        if len(keys) == 3:
            points = coords[indices]
            if abs(np.linalg.det((points[1:] - points[0]).T)) < 1e-9:
                raise invalid("Choose three non-collinear basis atoms", keys[0])
    indices = params.get("reference_atom_indices")
    region = params.get("reference_region")
    if not indices and region is None:
        return
    if indices is not None:
        if len(indices) != len(set(indices)) or any(i < 0 or i >= len(coords) for i in indices):
            raise invalid(
                "Choose distinct atoms within the numbered table", "reference_atom_indices"
            )
        selected = coords[indices]
    else:
        selected = coords[region_mask(coords, region)]
    minimum = 6 if operation == "strain.atom" else 3
    if len(selected) < minimum or np.linalg.matrix_rank(selected - selected.mean(axis=0)) < 2:
        raise invalid(
            "Select a reference spanning two dimensions",
            "reference_atom_indices",
            minimum_atom_count=minimum,
        )


def check_geometry(compiled, record, adopted, parent, store):
    fields = set(adopted)
    spatial = {
        "g_vectors",
        "g_peak_indices",
        "reference_region",
        "reference_regions",
        "roi",
        "base_index",
        "a_index",
        "b_index",
        "reference_atom_indices",
        "start_index",
        "end_index",
    }
    if not fields & spatial:
        return
    # Indices refer to the exact atoms displayed to the user. Coordinates refer to the
    # selected 2-D image. Validate those producer inputs even when operation IDs are renamed.
    table_id = next(
        (
            f.get("evidence", {}).get("table_artifact_id")
            for f in record["fields"]
            if f.get("evidence", {}).get("table_artifact_id")
        ),
        None,
    )
    producers = {"input.load"}
    if table_id:
        producers.update(
            {"atomic.detect", "atomic.refine", "atomic.regularize", "preprocessing.apply"}
        )
    if "g_peak_indices" in fields:
        producers.add("preprocessing.apply")
    child_steps = [step.model_dump(mode="json") for step in compiled.steps]

    def precursors(steps, stop_id):
        result = []
        for step in steps:
            if step["id"] == stop_id:
                break
            if step["operation_id"] in producers:
                result.append((step["operation_id"], step["parameters"]))
        return result

    first = next(iter(adopted.values()))
    if precursors(child_steps, first["step_id"]) != precursors(
        parent["request"]["steps"], record["step_id"]
    ):
        raise invalid(
            "Input selection or numbered atom evidence changed; inspect the revised image first",
            "steps",
        )
    target = next(s for s in compiled.steps if s.id == first["step_id"])
    params = target.parameters
    if group := first.get("parameter_group"):
        params = params[group]
    if "g_peak_indices" in fields:
        parent_params = next(
            step["parameters"]
            for step in parent["request"]["steps"]
            if step["id"] == record["step_id"]
        )
        if group:
            parent_params = parent_params[group]
        if params.get("image_role", "raw") != parent_params.get("image_role", "raw"):
            raise invalid(
                "Choose numbered peaks from the same image role as the displayed FFT", "image_role"
            )
    if table_id:
        _validate_numbered_atoms(params, table_id, store, target.operation_id)
    if target.operation_id.startswith("strain.gpa"):
        from autotem.services.strain import checked_region
        from autotem.strain.gpa import _convert_g_vector_units, _validate_g_vector_pair

        raw_id = store.role(record["run_id"], "raw")
        shape = tuple(store.artifact(raw_id)["metadata"]["shape"])
        quality = None
        mask_id = next(
            (
                f["evidence"]["quality_mask_artifact_id"]
                for f in record["fields"]
                if "quality_mask_artifact_id" in f.get("evidence", {})
            ),
            None,
        )
        if mask_id:
            old = next(
                s["parameters"] for s in parent["request"]["steps"] if s["id"] == record["step_id"]
            )
            references = {"reference_region", "reference_regions", "absolute_reference"}
            if all(old.get(k) == v for k, v in params.items() if k not in references):
                quality = store.load_array(mask_id)
        for region in params.get("reference_regions") or [params.get("reference_region")]:
            if region is not None:
                try:
                    y0, y1, x0, x1 = checked_region(region, shape)
                except HarnessError as exc:
                    raise invalid(str(exc), "reference_region") from exc
                if quality is not None and np.count_nonzero(quality[y0:y1, x0:x1]) < 9:
                    raise invalid(
                        "Choose at least nine valid pixels in the displayed quality mask",
                        "reference_region",
                    )
        if fields & {"g_vectors", "g_peak_indices"}:
            vectors = np.asarray(params["g_vectors"], dtype=float)
            if vectors.shape != (2, 2) or np.linalg.matrix_rank(vectors) != 2:
                raise invalid("Supply two nonzero, non-collinear (gx,gy) pairs", "g_vectors")
            scale = params.get("pixel_size_nm")
            if params["g_vector_unit"] == "1/nm" and scale is None:
                # The next computation can request a physical scale if its calibration
                # step does not provide one. Never infer a scale to pass this check.
                return
            try:
                converted = _convert_g_vector_units(
                    vectors, unit=params["g_vector_unit"], pixel_size_nm=scale, image_shape=shape
                )
                _validate_g_vector_pair(converted)
            except ValueError as exc:
                raise invalid(str(exc), "g_vectors") from exc
