"""Local atom-to-atom spacing segments along user-defined a/b lattice directions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(slots=True)
class AtomicPlaneSpacingResult:
    """Short line segments connecting adjacent atoms along a/b lattice directions."""

    start_yx: np.ndarray
    end_yx: np.ndarray
    length_px: np.ndarray
    direction: np.ndarray
    source_atom_index: np.ndarray
    target_atom_index: np.ndarray
    source_lattice_ij: np.ndarray
    target_lattice_ij: np.ndarray
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]


def compute_atomic_plane_spacing_segments(
    coordinates: np.ndarray,
    lattice_indices_ij: np.ndarray,
    *,
    valid_atom_mask: np.ndarray | None = None,
    directions: tuple[str, ...] = ("a", "b"),
) -> AtomicPlaneSpacingResult:
    """Connect atoms separated by one integer step along the custom a/b lattice."""
    coords = np.asarray(coordinates, dtype=np.float64)
    indices = np.asarray(lattice_indices_ij, dtype=np.int32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    if indices.ndim != 2 or indices.shape != coords.shape:
        raise ValueError("lattice_indices_ij must have shape (N, 2).")
    if not directions:
        raise ValueError("directions must include at least one of 'a' or 'b'.")
    invalid_directions = sorted(set(directions) - {"a", "b"})
    if invalid_directions:
        raise ValueError(f"Unsupported directions: {invalid_directions}")

    n_atoms = len(coords)
    if valid_atom_mask is None:
        valid = np.ones(n_atoms, dtype=bool)
    else:
        valid = np.asarray(valid_atom_mask, dtype=bool)
        if valid.shape != (n_atoms,):
            raise ValueError("valid_atom_mask must have shape (N,).")

    warnings: list[str] = []
    index_to_atom: dict[tuple[int, int], int] = {}
    duplicate_count = 0
    for atom_idx, (ij, is_valid) in enumerate(zip(indices, valid, strict=False)):
        if not bool(is_valid):
            continue
        key = (int(ij[0]), int(ij[1]))
        if key in index_to_atom:
            duplicate_count += 1
            continue
        index_to_atom[key] = int(atom_idx)
    if duplicate_count:
        warnings.append(
            "Duplicate lattice indices were found; only the first atom per index was used. "
            f"duplicate_count={duplicate_count}"
        )

    direction_steps = {"a": (1, 0), "b": (0, 1)}
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    lengths: list[float] = []
    direction_labels: list[str] = []
    source_atoms: list[int] = []
    target_atoms: list[int] = []
    source_indices: list[tuple[int, int]] = []
    target_indices: list[tuple[int, int]] = []

    for key, source_idx in index_to_atom.items():
        for direction in directions:
            di, dj = direction_steps[direction]
            target_key = (key[0] + di, key[1] + dj)
            target_idx = index_to_atom.get(target_key)
            if target_idx is None:
                continue
            start = coords[source_idx]
            end = coords[target_idx]
            length = float(np.linalg.norm(end - start))
            if not np.isfinite(length):
                continue
            starts.append(start)
            ends.append(end)
            lengths.append(length)
            direction_labels.append(direction)
            source_atoms.append(source_idx)
            target_atoms.append(target_idx)
            source_indices.append(key)
            target_indices.append(target_key)

    if not lengths:
        warnings.append(
            "No adjacent a/b lattice segments were found. Define a cleaner custom a/b basis "
            "or increase lattice-index tolerance."
        )

    valid_ratio = float(np.mean(valid)) if n_atoms else 0.0
    possible_count = max(int(np.sum(valid)) - 1, 1)
    segment_density = float(np.clip(len(lengths) / (2.0 * possible_count), 0.0, 1.0))
    confidence = float(np.clip(0.55 * valid_ratio + 0.45 * segment_density, 0.0, 1.0))

    provenance = {
        "source": "autotem.atomic.plane_spacing.compute_atomic_plane_spacing_segments",
        "method": "short_segments_between_adjacent_custom_ab_lattice_indices",
        "parameters": {
            "directions": list(directions),
            "valid_atom_count": int(np.sum(valid)),
            "segment_count": int(len(lengths)),
        },
        "caution": (
            "Segment lengths are local atom-to-atom spacing along the custom a/b "
            "directions. They are not perpendicular interplanar spacings unless "
            "the selected direction and lattice geometry make that interpretation valid."
        ),
    }

    return AtomicPlaneSpacingResult(
        start_yx=np.asarray(starts, dtype=np.float32).reshape((-1, 2)),
        end_yx=np.asarray(ends, dtype=np.float32).reshape((-1, 2)),
        length_px=np.asarray(lengths, dtype=np.float32),
        direction=np.asarray(direction_labels, dtype=object),
        source_atom_index=np.asarray(source_atoms, dtype=np.int32),
        target_atom_index=np.asarray(target_atoms, dtype=np.int32),
        source_lattice_ij=np.asarray(source_indices, dtype=np.int32).reshape((-1, 2)),
        target_lattice_ij=np.asarray(target_indices, dtype=np.int32).reshape((-1, 2)),
        confidence=confidence,
        warnings=warnings,
        provenance=provenance,
    )


def compute_atomic_plane_spacing_segments_from_basis(
    coordinates: np.ndarray,
    *,
    base_index: int,
    a_index: int,
    b_index: int,
    lattice_index_tolerance: float = 0.40,
    directions: tuple[str, ...] = ("a", "b"),
) -> AtomicPlaneSpacingResult:
    """Connect adjacent atoms after indexing coordinates with a custom base/a/b basis."""
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    n_atoms = len(coords)
    if n_atoms < 3:
        raise ValueError("Need at least 3 atoms to define custom a/b spacing.")
    for idx_name, idx in {
        "base_index": base_index,
        "a_index": a_index,
        "b_index": b_index,
    }.items():
        if idx < 0 or idx >= n_atoms:
            raise ValueError(f"{idx_name}={idx} is out of range for {n_atoms} atoms.")
    if len({int(base_index), int(a_index), int(b_index)}) != 3:
        raise ValueError("base_index, a_index, and b_index must reference three different atoms.")
    if not (0.05 <= lattice_index_tolerance <= 1.0):
        raise ValueError("lattice_index_tolerance must be in [0.05, 1.0].")

    base_yx = coords[int(base_index)]
    a_vec_yx = coords[int(a_index)] - base_yx
    b_vec_yx = coords[int(b_index)] - base_yx
    basis_yx = np.column_stack([a_vec_yx, b_vec_yx])
    det = float(np.linalg.det(basis_yx))
    if abs(det) <= 1e-9:
        raise ValueError("Custom basis vectors are singular or nearly collinear.")

    fractional = (np.linalg.inv(basis_yx) @ (coords - base_yx).T).T
    lattice_indices = np.rint(fractional).astype(np.int32)
    assignment_residual = np.linalg.norm(fractional - lattice_indices, axis=1)
    valid = assignment_residual <= float(lattice_index_tolerance)
    valid[int(base_index)] = True
    valid[int(a_index)] = True
    valid[int(b_index)] = True

    result = compute_atomic_plane_spacing_segments(
        coords,
        lattice_indices,
        valid_atom_mask=valid,
        directions=directions,
    )
    invalid_count = int(np.sum(~valid))
    if invalid_count:
        result.warnings.append(
            "Some atoms were not assigned to the custom a/b lattice for spacing lines. "
            f"invalid_count={invalid_count}"
        )
    result.provenance["source"] = (
        "autotem.atomic.plane_spacing.compute_atomic_plane_spacing_segments_from_basis"
    )
    result.provenance["method"] = "direct_short_segments_from_custom_ab_basis"
    result.provenance["basis_indices"] = {
        "base_index": int(base_index),
        "a_index": int(a_index),
        "b_index": int(b_index),
    }
    result.provenance["parameters"]["lattice_index_tolerance"] = float(
        lattice_index_tolerance
    )
    result.provenance["fit_statistics"] = {
        "valid_atom_count": int(np.sum(valid)),
        "invalid_atom_count": invalid_count,
        "assignment_residual_median": float(np.median(assignment_residual[valid]))
        if np.any(valid)
        else None,
    }
    return result


def compute_atomic_plane_spacing_segments_from_angle_constrained_basis(
    coordinates: np.ndarray,
    *,
    base_index: int,
    a_index: int,
    b_index: int,
    lattice_angle_deg: float = 90.0,
    lattice_index_tolerance: float = 0.40,
    directions: tuple[str, ...] = ("a", "b"),
) -> AtomicPlaneSpacingResult:
    """Connect adjacent atoms after constraining the a-b lattice angle.

    The selected a vector is kept as the primary direction. The selected b vector
    contributes its length and rough side, then is rotated to the requested
    a-b angle. This is useful when the user knows the projected lattice angle
    (for example 90 degrees) but manual b picking or local distortion makes
    direct integer indexing unstable.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    n_atoms = len(coords)
    if n_atoms < 3:
        raise ValueError("Need at least 3 atoms to define custom a/b spacing.")
    for idx_name, idx in {
        "base_index": base_index,
        "a_index": a_index,
        "b_index": b_index,
    }.items():
        if idx < 0 or idx >= n_atoms:
            raise ValueError(f"{idx_name}={idx} is out of range for {n_atoms} atoms.")
    if len({int(base_index), int(a_index), int(b_index)}) != 3:
        raise ValueError("base_index, a_index, and b_index must reference three different atoms.")
    if not (5.0 <= float(lattice_angle_deg) <= 175.0):
        raise ValueError("lattice_angle_deg must be in [5, 175].")
    if not (0.05 <= lattice_index_tolerance <= 1.0):
        raise ValueError("lattice_index_tolerance must be in [0.05, 1.0].")

    base_yx = coords[int(base_index)]
    a_vec_yx = coords[int(a_index)] - base_yx
    raw_b_vec_yx = coords[int(b_index)] - base_yx
    a_len = float(np.linalg.norm(a_vec_yx))
    b_len = float(np.linalg.norm(raw_b_vec_yx))
    if a_len <= 1e-9 or b_len <= 1e-9:
        raise ValueError("Custom basis vectors must have non-zero length.")

    constrained_b_vec_yx = _constrain_b_vector_to_lattice_angle(
        a_vec_yx,
        raw_b_vec_yx,
        lattice_angle_deg=float(lattice_angle_deg),
    )
    basis_yx = np.column_stack([a_vec_yx, constrained_b_vec_yx])
    det = float(np.linalg.det(basis_yx))
    if abs(det) <= 1e-9:
        raise ValueError("Angle-constrained basis vectors are singular or nearly collinear.")

    fractional = (np.linalg.inv(basis_yx) @ (coords - base_yx).T).T
    lattice_indices = np.rint(fractional).astype(np.int32)
    assignment_residual = np.linalg.norm(fractional - lattice_indices, axis=1)
    valid = assignment_residual <= float(lattice_index_tolerance)
    valid[int(base_index)] = True
    valid[int(a_index)] = True
    valid[int(b_index)] = True

    result = compute_atomic_plane_spacing_segments(
        coords,
        lattice_indices,
        valid_atom_mask=valid,
        directions=directions,
    )
    invalid_count = int(np.sum(~valid))
    if invalid_count:
        result.warnings.append(
            "Some atoms were not assigned to the angle-constrained a/b lattice. "
            f"invalid_count={invalid_count}"
        )
    original_angle = _basis_angle_deg(a_vec_yx, raw_b_vec_yx)
    constrained_angle = _basis_angle_deg(a_vec_yx, constrained_b_vec_yx)
    result.provenance["source"] = (
        "autotem.atomic.plane_spacing."
        "compute_atomic_plane_spacing_segments_from_angle_constrained_basis"
    )
    result.provenance["method"] = "direct_short_segments_from_angle_constrained_ab_basis"
    result.provenance["basis_indices"] = {
        "base_index": int(base_index),
        "a_index": int(a_index),
        "b_index": int(b_index),
    }
    result.provenance["parameters"]["lattice_index_tolerance"] = float(
        lattice_index_tolerance
    )
    result.provenance["parameters"]["lattice_angle_deg"] = float(lattice_angle_deg)
    result.provenance["fit_statistics"] = {
        "valid_atom_count": int(np.sum(valid)),
        "invalid_atom_count": invalid_count,
        "assignment_residual_median": float(np.median(assignment_residual[valid]))
        if np.any(valid)
        else None,
        "original_ab_angle_deg": original_angle,
        "constrained_ab_angle_deg": constrained_angle,
        "a_length_px": a_len,
        "b_length_px": b_len,
    }
    return result


def compute_atomic_plane_spacing_segments_from_lattice_angle(
    coordinates: np.ndarray,
    *,
    lattice_angle_deg: float = 90.0,
    angle_tolerance_deg: float = 10.0,
    min_spacing_px: float = 1.0,
    max_spacing_px: float = 30.0,
    valid_atom_mask: np.ndarray | None = None,
) -> AtomicPlaneSpacingResult:
    """Infer two atom-row directions from a lattice angle, without picking a/b.

    Only the angle between the two lattice axes is supplied by the user. The
    absolute orientation is estimated from nearby atom-pair direction statistics,
    then spacing segments are drawn along the two strongest compatible axes.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    n_atoms = len(coords)
    if n_atoms < 3:
        raise ValueError("Need at least 3 atoms for lattice-angle-only spacing.")
    if not (5.0 <= float(lattice_angle_deg) <= 175.0):
        raise ValueError("lattice_angle_deg must be in [5, 175].")
    if not (0.1 <= float(angle_tolerance_deg) <= 45.0):
        raise ValueError("angle_tolerance_deg must be in [0.1, 45.0].")
    if not (float(min_spacing_px) > 0.0):
        raise ValueError("min_spacing_px must be positive.")
    if not (float(max_spacing_px) > float(min_spacing_px)):
        raise ValueError("max_spacing_px must be larger than min_spacing_px.")

    if valid_atom_mask is None:
        valid = np.ones(n_atoms, dtype=bool)
    else:
        valid = np.asarray(valid_atom_mask, dtype=bool)
        if valid.shape != (n_atoms,):
            raise ValueError("valid_atom_mask must have shape (N,).")

    tree = cKDTree(coords)
    min_r = float(min_spacing_px)
    max_r = float(max_spacing_px)
    pair_angles: list[float] = []
    pair_weights: list[float] = []
    for source_idx, target_idx in tree.query_pairs(r=max_r):
        if not bool(valid[int(source_idx)]) or not bool(valid[int(target_idx)]):
            continue
        vector_yx = coords[int(target_idx)] - coords[int(source_idx)]
        length = float(np.linalg.norm(vector_yx))
        if not (min_r <= length <= max_r):
            continue
        pair_angles.append(_axis_angle_deg(vector_yx))
        # Short nearest-neighbor links should dominate over longer harmonics.
        pair_weights.append(1.0 / max(length, 1e-9))

    warnings: list[str] = []
    requested_angle = float(lattice_angle_deg)
    effective_axis_angle = min(requested_angle, 180.0 - requested_angle)
    if not np.isclose(requested_angle, effective_axis_angle):
        warnings.append(
            "Lattice-angle-only spacing treats atom rows as undirected axes; "
            f"{requested_angle:.2f} deg is equivalent to {effective_axis_angle:.2f} deg."
        )

    if not pair_angles:
        warnings.append(
            "No nearby atom pairs were available for lattice-angle direction inference. "
            "Increase the distance range or check atom detection."
        )
        return _empty_plane_spacing_result(
            confidence=0.0,
            warnings=warnings,
            provenance={
                "source": (
                    "autotem.atomic.plane_spacing."
                    "compute_atomic_plane_spacing_segments_from_lattice_angle"
                ),
                "method": "auto_direction_segments_from_lattice_angle",
                "parameters": {
                    "lattice_angle_deg": requested_angle,
                    "effective_axis_angle_deg": effective_axis_angle,
                    "angle_tolerance_deg": float(angle_tolerance_deg),
                    "min_spacing_px": min_r,
                    "max_spacing_px": max_r,
                    "valid_atom_count": int(np.sum(valid)),
                    "segment_count": 0,
                },
            },
        )

    angles = np.asarray(pair_angles, dtype=np.float64)
    weights = np.asarray(pair_weights, dtype=np.float64)
    tolerance = float(angle_tolerance_deg)

    def _score_axis(center_deg: float) -> float:
        errors = np.asarray(
            [_axis_angle_difference_deg(angle, center_deg) for angle in angles],
            dtype=np.float64,
        )
        return float(np.sum(weights[errors <= tolerance]))

    best_axis_1 = 0.0
    best_axis_2 = effective_axis_angle
    best_score = -np.inf
    best_component_scores = (0.0, 0.0)
    for axis_1 in np.arange(0.0, 180.0, 1.0):
        score_1 = _score_axis(axis_1)
        axis_2_plus = (axis_1 + effective_axis_angle) % 180.0
        axis_2_minus = (axis_1 - effective_axis_angle) % 180.0
        score_plus = _score_axis(axis_2_plus)
        score_minus = _score_axis(axis_2_minus)
        if score_plus >= score_minus:
            axis_2 = axis_2_plus
            score_2 = score_plus
        else:
            axis_2 = axis_2_minus
            score_2 = score_minus
        pair_score = 2.0 * min(score_1, score_2) + 0.25 * (score_1 + score_2)
        if pair_score > best_score:
            best_score = pair_score
            best_axis_1 = float(axis_1)
            best_axis_2 = float(axis_2)
            best_component_scores = (float(score_1), float(score_2))

    result_1 = compute_atomic_plane_spacing_segments_by_angle(
        coords,
        angle_deg=best_axis_1,
        angle_tolerance_deg=tolerance,
        min_spacing_px=min_r,
        max_spacing_px=max_r,
        valid_atom_mask=valid,
        direction_label="angle_only_axis_1",
    )
    result_2 = compute_atomic_plane_spacing_segments_by_angle(
        coords,
        angle_deg=best_axis_2,
        angle_tolerance_deg=tolerance,
        min_spacing_px=min_r,
        max_spacing_px=max_r,
        valid_atom_mask=valid,
        direction_label="angle_only_axis_2",
    )

    combined_warnings = warnings + result_1.warnings + result_2.warnings
    starts = _stack_spacing_arrays(result_1.start_yx, result_2.start_yx, columns=2)
    ends = _stack_spacing_arrays(result_1.end_yx, result_2.end_yx, columns=2)
    lengths = np.concatenate([result_1.length_px, result_2.length_px]).astype(np.float32)
    directions = np.concatenate([result_1.direction, result_2.direction]).astype(object)
    source_atoms = np.concatenate(
        [result_1.source_atom_index, result_2.source_atom_index]
    ).astype(np.int32)
    target_atoms = np.concatenate(
        [result_1.target_atom_index, result_2.target_atom_index]
    ).astype(np.int32)
    source_indices = _stack_spacing_arrays(
        result_1.source_lattice_ij, result_2.source_lattice_ij, columns=2
    ).astype(np.int32)
    target_indices = _stack_spacing_arrays(
        result_1.target_lattice_ij, result_2.target_lattice_ij, columns=2
    ).astype(np.int32)

    segment_count = int(len(lengths))
    valid_count = int(np.sum(valid))
    segment_density = float(np.clip(segment_count / max(2 * valid_count, 1), 0.0, 1.0))
    confidence = float(
        np.clip(
            0.25 + 0.45 * segment_density + 0.30 * min(result_1.confidence, result_2.confidence),
            0.0,
            1.0,
        )
    )
    if segment_count == 0:
        combined_warnings.append(
            "No segments were drawn after lattice-angle direction inference. "
            "Increase angle tolerance or distance range."
        )

    provenance = {
        "source": (
            "autotem.atomic.plane_spacing."
            "compute_atomic_plane_spacing_segments_from_lattice_angle"
        ),
        "method": "auto_direction_segments_from_lattice_angle",
        "parameters": {
            "lattice_angle_deg": requested_angle,
            "effective_axis_angle_deg": effective_axis_angle,
            "angle_tolerance_deg": tolerance,
            "min_spacing_px": min_r,
            "max_spacing_px": max_r,
            "valid_atom_count": valid_count,
            "segment_count": segment_count,
        },
        "fit_statistics": {
            "inferred_axis_1_deg": best_axis_1,
            "inferred_axis_2_deg": best_axis_2,
            "axis_separation_deg": _axis_angle_difference_deg(best_axis_1, best_axis_2),
            "direction_score_axis_1": best_component_scores[0],
            "direction_score_axis_2": best_component_scores[1],
            "candidate_pair_count": int(len(angles)),
        },
        "angle_convention": (
            "image-coordinate row axes: 0 deg horizontal, 90 deg vertical; "
            "opposite directions are treated as the same axis"
        ),
        "caution": (
            "This mode infers lattice orientation statistically from atom pairs. "
            "For multi-grain or strongly defective regions, use a local ROI or "
            "picked-direction mode for more controlled measurements."
        ),
    }
    return AtomicPlaneSpacingResult(
        start_yx=starts,
        end_yx=ends,
        length_px=lengths,
        direction=directions,
        source_atom_index=source_atoms,
        target_atom_index=target_atoms,
        source_lattice_ij=source_indices,
        target_lattice_ij=target_indices,
        confidence=confidence,
        warnings=combined_warnings,
        provenance=provenance,
    )


def compute_atomic_plane_spacing_segments_by_angle(
    coordinates: np.ndarray,
    *,
    angle_deg: float,
    angle_tolerance_deg: float = 10.0,
    min_spacing_px: float = 1.0,
    max_spacing_px: float = 30.0,
    valid_atom_mask: np.ndarray | None = None,
    direction_label: str | None = None,
) -> AtomicPlaneSpacingResult:
    """Connect nearest atom pairs along an image-coordinate axis angle.

    The angle uses image coordinates: 0 degrees points right (+x) and
    90 degrees points down (+y). The matching is axis-based, so angle and
    angle + 180 degrees are treated as the same atom-row direction.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    n_atoms = len(coords)
    if n_atoms < 2:
        raise ValueError("Need at least 2 atoms for angle-based spacing.")
    if not np.isfinite(angle_deg):
        raise ValueError("angle_deg must be finite.")
    if not (0.1 <= float(angle_tolerance_deg) <= 45.0):
        raise ValueError("angle_tolerance_deg must be in [0.1, 45.0].")
    if not (float(min_spacing_px) > 0.0):
        raise ValueError("min_spacing_px must be positive.")
    if not (float(max_spacing_px) > float(min_spacing_px)):
        raise ValueError("max_spacing_px must be larger than min_spacing_px.")

    if valid_atom_mask is None:
        valid = np.ones(n_atoms, dtype=bool)
    else:
        valid = np.asarray(valid_atom_mask, dtype=bool)
        if valid.shape != (n_atoms,):
            raise ValueError("valid_atom_mask must have shape (N,).")

    theta = np.deg2rad(float(angle_deg) % 360.0)
    direction_yx = np.array([np.sin(theta), np.cos(theta)], dtype=np.float64)
    tree = cKDTree(coords)
    max_r = float(max_spacing_px)
    min_r = float(min_spacing_px)
    tolerance_rad = np.deg2rad(float(angle_tolerance_deg))

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    lengths: list[float] = []
    direction_labels: list[str] = []
    source_atoms: list[int] = []
    target_atoms: list[int] = []
    source_indices: list[tuple[int, int]] = []
    target_indices: list[tuple[int, int]] = []
    label = direction_label or f"angle_{float(angle_deg) % 360.0:.2f}deg"
    seen_pairs: set[tuple[int, int]] = set()

    for source_idx, coord in enumerate(coords):
        if not bool(valid[source_idx]):
            continue
        candidate_idx = tree.query_ball_point(coord, r=max_r)
        accepted_candidates: list[tuple[float, int]] = []
        for target_idx in candidate_idx:
            target_idx = int(target_idx)
            if target_idx == source_idx or not bool(valid[target_idx]):
                continue
            vector_yx = coords[target_idx] - coord
            length = float(np.linalg.norm(vector_yx))
            if not (min_r <= length <= max_r):
                continue
            dot_abs = abs(float(np.dot(vector_yx, direction_yx))) / length
            angle_error = float(np.arccos(np.clip(dot_abs, -1.0, 1.0)))
            if angle_error > tolerance_rad:
                continue
            accepted_candidates.append((length, target_idx))

        for best_length, best_idx in sorted(accepted_candidates, key=lambda item: item[0])[:2]:
            pair = tuple(sorted((int(source_idx), int(best_idx))))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            starts.append(coord)
            ends.append(coords[best_idx])
            lengths.append(float(best_length))
            direction_labels.append(label)
            source_atoms.append(int(source_idx))
            target_atoms.append(int(best_idx))
            source_indices.append((-1, -1))
            target_indices.append((-1, -1))

    warnings: list[str] = []
    if not lengths:
        warnings.append(
            "No angle-based spacing segments were found. Increase angle tolerance, "
            "expand distance range, or choose a direction closer to the atom rows."
        )
    valid_count = int(np.sum(valid))
    segment_density = float(np.clip(len(lengths) / max(valid_count, 1), 0.0, 1.0))
    confidence = float(np.clip(0.40 + 0.60 * segment_density, 0.0, 1.0))
    provenance = {
        "source": "autotem.atomic.plane_spacing.compute_atomic_plane_spacing_segments_by_angle",
        "method": "nearest_neighbor_segments_along_image_angle",
        "parameters": {
            "angle_deg": float(angle_deg) % 360.0,
            "angle_tolerance_deg": float(angle_tolerance_deg),
            "min_spacing_px": float(min_spacing_px),
            "max_spacing_px": float(max_spacing_px),
            "valid_atom_count": valid_count,
            "segment_count": int(len(lengths)),
            "direction_label": label,
        },
        "angle_convention": "image coordinates: 0 deg right (+x), 90 deg down (+y)",
        "caution": (
            "Angle mode measures nearest atom-to-atom segments along a selected "
            "row axis and does not require global integer a/b lattice indexing."
        ),
    }
    return AtomicPlaneSpacingResult(
        start_yx=np.asarray(starts, dtype=np.float32).reshape((-1, 2)),
        end_yx=np.asarray(ends, dtype=np.float32).reshape((-1, 2)),
        length_px=np.asarray(lengths, dtype=np.float32),
        direction=np.asarray(direction_labels, dtype=object),
        source_atom_index=np.asarray(source_atoms, dtype=np.int32),
        target_atom_index=np.asarray(target_atoms, dtype=np.int32),
        source_lattice_ij=np.asarray(source_indices, dtype=np.int32).reshape((-1, 2)),
        target_lattice_ij=np.asarray(target_indices, dtype=np.int32).reshape((-1, 2)),
        confidence=confidence,
        warnings=warnings,
        provenance=provenance,
    )


def compute_atomic_plane_spacing_segments_from_picked_direction(
    coordinates: np.ndarray,
    *,
    start_index: int,
    end_index: int,
    angle_tolerance_deg: float = 10.0,
    min_spacing_px: float = 1.0,
    max_spacing_px: float = 30.0,
    valid_atom_mask: np.ndarray | None = None,
) -> AtomicPlaneSpacingResult:
    """Connect nearest atom pairs along the direction defined by two picked atoms."""
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    n_atoms = len(coords)
    for idx_name, idx in {"start_index": start_index, "end_index": end_index}.items():
        if idx < 0 or idx >= n_atoms:
            raise ValueError(f"{idx_name}={idx} is out of range for {n_atoms} atoms.")
    if int(start_index) == int(end_index):
        raise ValueError("start_index and end_index must be different atoms.")

    vector_yx = coords[int(end_index)] - coords[int(start_index)]
    if float(np.linalg.norm(vector_yx)) <= 1e-9:
        raise ValueError("Picked direction atoms are coincident or too close.")
    angle_deg = float(np.degrees(np.arctan2(vector_yx[0], vector_yx[1])) % 360.0)
    result = compute_atomic_plane_spacing_segments_by_angle(
        coords,
        angle_deg=angle_deg,
        angle_tolerance_deg=angle_tolerance_deg,
        min_spacing_px=min_spacing_px,
        max_spacing_px=max_spacing_px,
        valid_atom_mask=valid_atom_mask,
        direction_label=f"picked_{int(start_index)}_{int(end_index)}",
    )
    result.provenance["source"] = (
        "autotem.atomic.plane_spacing.compute_atomic_plane_spacing_segments_from_picked_direction"
    )
    result.provenance["method"] = "nearest_neighbor_segments_from_two_picked_atoms"
    result.provenance["picked_direction"] = {
        "start_index": int(start_index),
        "end_index": int(end_index),
        "angle_deg": angle_deg,
    }
    return result


def _empty_plane_spacing_result(
    *,
    confidence: float,
    warnings: list[str],
    provenance: dict[str, Any],
) -> AtomicPlaneSpacingResult:
    return AtomicPlaneSpacingResult(
        start_yx=np.empty((0, 2), dtype=np.float32),
        end_yx=np.empty((0, 2), dtype=np.float32),
        length_px=np.empty((0,), dtype=np.float32),
        direction=np.empty((0,), dtype=object),
        source_atom_index=np.empty((0,), dtype=np.int32),
        target_atom_index=np.empty((0,), dtype=np.int32),
        source_lattice_ij=np.empty((0, 2), dtype=np.int32),
        target_lattice_ij=np.empty((0, 2), dtype=np.int32),
        confidence=float(confidence),
        warnings=warnings,
        provenance=provenance,
    )


def _stack_spacing_arrays(
    first: np.ndarray,
    second: np.ndarray,
    *,
    columns: int,
) -> np.ndarray:
    first_arr = np.asarray(first)
    second_arr = np.asarray(second)
    if first_arr.size == 0 and second_arr.size == 0:
        return np.empty((0, columns), dtype=np.result_type(first_arr, second_arr, np.float32))
    if first_arr.size == 0:
        return second_arr.reshape((-1, columns))
    if second_arr.size == 0:
        return first_arr.reshape((-1, columns))
    return np.vstack([first_arr.reshape((-1, columns)), second_arr.reshape((-1, columns))])


def _axis_angle_deg(vector_yx: np.ndarray) -> float:
    vector = np.asarray(vector_yx, dtype=np.float64)
    angle = float(np.degrees(np.arctan2(vector[0], vector[1])) % 180.0)
    return angle


def _axis_angle_difference_deg(angle_a: float, angle_b: float) -> float:
    return float(abs((float(angle_a) - float(angle_b) + 90.0) % 180.0 - 90.0))


def _basis_angle_deg(a_vec_yx: np.ndarray, b_vec_yx: np.ndarray) -> float:
    a = np.asarray(a_vec_yx, dtype=np.float64)
    b = np.asarray(b_vec_yx, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    cosang = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


def _constrain_b_vector_to_lattice_angle(
    a_vec_yx: np.ndarray,
    raw_b_vec_yx: np.ndarray,
    *,
    lattice_angle_deg: float,
) -> np.ndarray:
    a = np.asarray(a_vec_yx, dtype=np.float64)
    raw_b = np.asarray(raw_b_vec_yx, dtype=np.float64)
    a_len = float(np.linalg.norm(a))
    b_len = float(np.linalg.norm(raw_b))
    if a_len <= 1e-12 or b_len <= 1e-12:
        raise ValueError("Custom basis vectors must have non-zero length.")

    a_xy = np.array([a[1], a[0]], dtype=np.float64) / a_len
    raw_b_xy = np.array([raw_b[1], raw_b[0]], dtype=np.float64) / b_len
    theta = np.deg2rad(float(lattice_angle_deg))
    candidates: list[np.ndarray] = []
    for sign in (1.0, -1.0):
        angle = sign * theta
        rot = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ],
            dtype=np.float64,
        )
        candidates.append(rot @ a_xy)
    best_xy = max(candidates, key=lambda cand: float(np.dot(cand, raw_b_xy)))
    return np.array([best_xy[1], best_xy[0]], dtype=np.float64) * b_len
