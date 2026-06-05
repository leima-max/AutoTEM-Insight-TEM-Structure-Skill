"""Ideal-lattice atomic displacement from user-defined 2D basis vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class AtomicDisplacementResult:
    """Observed-vs-ideal lattice displacement derived from a custom 2D basis."""

    coordinates_observed_yx: np.ndarray
    coordinates_ideal_yx: np.ndarray
    displacement_yx: np.ndarray
    displacement_magnitude: np.ndarray
    lattice_indices_ij: np.ndarray
    fractional_indices_ij: np.ndarray
    assignment_residual: np.ndarray
    valid_atom_mask: np.ndarray
    reference_atom_mask: np.ndarray
    fit_atom_mask: np.ndarray
    base_index: int
    a_index: int
    b_index: int
    initial_basis_yx: np.ndarray
    fitted_origin_yx: np.ndarray
    fitted_basis_yx: np.ndarray
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]


def compute_atomic_displacement_from_basis(
    coordinates: np.ndarray,
    *,
    base_index: int,
    a_index: int,
    b_index: int,
    reference_mask: np.ndarray | None = None,
    lattice_index_tolerance: float = 0.40,
    manual_lattice_indices: dict[int, tuple[int, int]] | None = None,
    allow_reference_fallback: bool = True,
) -> AtomicDisplacementResult:
    """Fit an ideal 2D lattice from custom basis picks and compute atom displacements."""
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    n_atoms = len(coords)
    if n_atoms < 3:
        raise ValueError("Need at least 3 atoms to fit an ideal lattice.")
    for idx_name, idx in {"base_index": base_index, "a_index": a_index, "b_index": b_index}.items():
        if idx < 0 or idx >= n_atoms:
            raise ValueError(f"{idx_name}={idx} is out of range for {n_atoms} atoms.")
    if len({int(base_index), int(a_index), int(b_index)}) != 3:
        raise ValueError("base_index, a_index, and b_index must reference three different atoms.")
    if not (0.05 <= lattice_index_tolerance <= 1.0):
        raise ValueError("lattice_index_tolerance must be in [0.05, 1.0].")

    warnings: list[str] = []
    base_yx = coords[int(base_index)]
    a_vec_yx = coords[int(a_index)] - base_yx
    b_vec_yx = coords[int(b_index)] - base_yx
    basis_yx = np.column_stack([a_vec_yx, b_vec_yx])
    det = float(np.linalg.det(basis_yx))
    if abs(det) <= 1e-9:
        raise ValueError("Custom basis vectors are singular or nearly collinear.")

    delta = (coords - base_yx).T
    inv_basis = np.linalg.inv(basis_yx)
    fractional = (inv_basis @ delta).T
    lattice_indices = np.rint(fractional).astype(np.int32)
    assignment_residual = np.linalg.norm(fractional - lattice_indices, axis=1)
    valid_mask = assignment_residual <= float(lattice_index_tolerance)
    valid_mask[int(base_index)] = True
    valid_mask[int(a_index)] = True
    valid_mask[int(b_index)] = True
    manual_mask = np.zeros(n_atoms, dtype=bool)
    if manual_lattice_indices:
        for atom_index, lattice_ij in manual_lattice_indices.items():
            atom_idx = int(atom_index)
            if atom_idx < 0 or atom_idx >= n_atoms:
                raise ValueError(
                    f"manual_lattice_indices contains atom index {atom_idx}, "
                    f"but valid range is [0, {n_atoms - 1}]."
                )
            ij = np.asarray(lattice_ij, dtype=np.int32)
            if ij.shape != (2,):
                raise ValueError("Each manual lattice index must be a pair (i, j).")
            lattice_indices[atom_idx] = ij
            assignment_residual[atom_idx] = 0.0
            valid_mask[atom_idx] = True
            manual_mask[atom_idx] = True

    if np.sum(valid_mask) < 3:
        raise ValueError(
            "Too few atoms could be assigned to the custom ideal lattice. "
            "Try picking a cleaner basis or increasing tolerance slightly."
        )

    ref_mask = _normalize_reference_mask(reference_mask, n_atoms)
    fit_mask = (valid_mask & ref_mask) | manual_mask
    fit_mask = _ensure_fit_mask(
        coords=coords,
        lattice_indices=lattice_indices,
        fit_mask=fit_mask,
        valid_mask=valid_mask,
        warnings=warnings,
        allow_reference_fallback=allow_reference_fallback,
    )

    fitted_origin_y, fitted_a_y, fitted_b_y = _fit_axis(
        lattice_indices=lattice_indices,
        values=coords[:, 0],
        fit_mask=fit_mask,
    )
    fitted_origin_x, fitted_a_x, fitted_b_x = _fit_axis(
        lattice_indices=lattice_indices,
        values=coords[:, 1],
        fit_mask=fit_mask,
    )
    fitted_origin_yx = np.array([fitted_origin_y, fitted_origin_x], dtype=np.float64)
    fitted_basis_yx = np.array(
        [[fitted_a_y, fitted_b_y], [fitted_a_x, fitted_b_x]],
        dtype=np.float64,
    )

    ideal_y = (
        fitted_origin_y
        + lattice_indices[:, 0] * fitted_a_y
        + lattice_indices[:, 1] * fitted_b_y
    )
    ideal_x = (
        fitted_origin_x
        + lattice_indices[:, 0] * fitted_a_x
        + lattice_indices[:, 1] * fitted_b_x
    )
    ideal_coords = np.column_stack([ideal_y, ideal_x]).astype(np.float64)
    displacement_yx = coords - ideal_coords
    displacement_mag = np.linalg.norm(displacement_yx, axis=1)

    basis_scale = 0.5 * (
        np.linalg.norm(fitted_basis_yx[:, 0]) + np.linalg.norm(fitted_basis_yx[:, 1])
    )
    fit_rms = float(np.sqrt(np.mean(displacement_mag[fit_mask] ** 2))) if np.any(fit_mask) else 0.0
    assign_med = float(np.median(assignment_residual[valid_mask])) if np.any(valid_mask) else 1.0
    valid_ratio = float(np.mean(valid_mask))
    fit_score = float(np.clip(1.0 - fit_rms / max(0.25 * basis_scale, 1e-6), 0.0, 1.0))
    assign_score = float(
        np.clip(1.0 - assign_med / max(float(lattice_index_tolerance), 1e-6), 0.0, 1.0)
    )
    confidence = float(
        np.clip(0.45 * fit_score + 0.35 * valid_ratio + 0.20 * assign_score, 0.0, 1.0)
    )

    if valid_ratio < 0.85:
        warnings.append(
            "Some atoms could not be assigned cleanly to the ideal lattice. "
            f"valid_ratio={valid_ratio:.3f}"
        )
    if fit_rms > 0.15 * basis_scale:
        warnings.append(
            "Ideal-lattice fit residual is relatively large. "
            f"rms={fit_rms:.4f}px, basis_scale={basis_scale:.4f}px"
        )

    provenance = {
        "source": "autotem.atomic.displacement.compute_atomic_displacement_from_basis",
        "parameters": {
            "lattice_index_tolerance": float(lattice_index_tolerance),
            "manual_lattice_index_count": int(np.sum(manual_mask)),
            "allow_reference_fallback": bool(allow_reference_fallback),
        },
        "basis_indices": {
            "base_index": int(base_index),
            "a_index": int(a_index),
            "b_index": int(b_index),
        },
        "fit_statistics": {
            "valid_ratio": valid_ratio,
            "fit_rms_px": fit_rms,
            "assignment_residual_median": assign_med,
            "reference_atom_count": int(np.sum(ref_mask)),
            "fit_atom_count": int(np.sum(fit_mask)),
            "manual_anchor_count": int(np.sum(manual_mask)),
        },
        "manual_lattice_indices": {
            str(int(atom_idx)): [
                int(lattice_indices[int(atom_idx), 0]),
                int(lattice_indices[int(atom_idx), 1]),
            ]
            for atom_idx in np.flatnonzero(manual_mask)
        },
    }

    return AtomicDisplacementResult(
        coordinates_observed_yx=coords,
        coordinates_ideal_yx=ideal_coords,
        displacement_yx=displacement_yx.astype(np.float32),
        displacement_magnitude=displacement_mag.astype(np.float32),
        lattice_indices_ij=lattice_indices.astype(np.int32),
        fractional_indices_ij=fractional.astype(np.float32),
        assignment_residual=assignment_residual.astype(np.float32),
        valid_atom_mask=valid_mask,
        reference_atom_mask=ref_mask,
        fit_atom_mask=fit_mask,
        base_index=int(base_index),
        a_index=int(a_index),
        b_index=int(b_index),
        initial_basis_yx=basis_yx.astype(np.float32),
        fitted_origin_yx=fitted_origin_yx.astype(np.float32),
        fitted_basis_yx=fitted_basis_yx.astype(np.float32),
        confidence=confidence,
        warnings=warnings,
        provenance=provenance,
    )


def _normalize_reference_mask(reference_mask: np.ndarray | None, n_atoms: int) -> np.ndarray:
    if reference_mask is None:
        return np.ones(n_atoms, dtype=bool)
    mask = np.asarray(reference_mask, dtype=bool)
    if mask.shape != (n_atoms,):
        raise ValueError("reference_mask must have shape (N,) matching the atom coordinates.")
    if int(np.sum(mask)) < 3:
        raise ValueError("reference_mask must include at least 3 atoms.")
    return mask


def _ensure_fit_mask(
    *,
    coords: np.ndarray,
    lattice_indices: np.ndarray,
    fit_mask: np.ndarray,
    valid_mask: np.ndarray,
    warnings: list[str],
    allow_reference_fallback: bool,
) -> np.ndarray:
    fit_mask = np.asarray(fit_mask, dtype=bool).copy()
    design = _design_matrix(lattice_indices[fit_mask])
    if np.sum(fit_mask) >= 3 and np.linalg.matrix_rank(design) == 3:
        return fit_mask

    if not allow_reference_fallback:
        raise ValueError(
            "Reference ROI atoms are insufficient for a stable ideal-lattice fit. "
            "Select a larger, cleaner ROI with at least three non-collinear assigned atoms."
        )

    warnings.append(
        "Reference atoms were insufficient for a stable ideal-lattice fit; "
        "falling back to all valid assigned atoms."
    )
    design_valid = _design_matrix(lattice_indices[valid_mask])
    if np.sum(valid_mask) < 3 or np.linalg.matrix_rank(design_valid) < 3:
        raise ValueError(
            "The chosen custom basis does not generate enough non-collinear lattice assignments."
        )
    return valid_mask.copy()


def _design_matrix(indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.column_stack(
        [
            np.ones(len(indices), dtype=np.float64),
            np.asarray(indices[:, 0], dtype=np.float64),
            np.asarray(indices[:, 1], dtype=np.float64),
        ]
    )


def _fit_axis(
    *,
    lattice_indices: np.ndarray,
    values: np.ndarray,
    fit_mask: np.ndarray,
) -> tuple[float, float, float]:
    design = _design_matrix(lattice_indices[fit_mask])
    target = np.asarray(values[fit_mask], dtype=np.float64)
    params, *_ = np.linalg.lstsq(design, target, rcond=None)
    return float(params[0]), float(params[1]), float(params[2])
