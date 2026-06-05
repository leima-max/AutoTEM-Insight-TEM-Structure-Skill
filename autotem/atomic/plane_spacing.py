"""Local atom-to-atom spacing segments along user-defined a/b lattice directions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
