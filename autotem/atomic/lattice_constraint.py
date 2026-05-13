"""Optional lattice-constraint regularization for refined atom coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class LatticeConstraintResult:
    coordinates_input: np.ndarray
    coordinates_regularized: np.ndarray
    displacements: np.ndarray
    enabled: bool
    parameters: dict[str, Any]
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]


def apply_lattice_constraint(
    coordinates: np.ndarray,
    enabled: bool = False,
    k_neighbors: int = 6,
    strength: float = 0.2,
    iterations: int = 1,
    max_shift_per_iter: float = 0.6,
) -> LatticeConstraintResult:
    """
    Optional local-lattice regularization.

    This step smooths local coordinate outliers toward nearby neighborhood geometry.
    Keep it disabled for defect-sensitive analysis (dislocations, defects, GPA strain).
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    if k_neighbors < 2:
        raise ValueError("k_neighbors must be >= 2.")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1].")
    if iterations < 1:
        raise ValueError("iterations must be >= 1.")
    if max_shift_per_iter <= 0:
        raise ValueError("max_shift_per_iter must be > 0.")

    warnings: list[str] = []
    if not enabled or len(coords) < (k_neighbors + 1):
        if enabled and len(coords) < (k_neighbors + 1):
            warnings.append(
                "Lattice constraint skipped: insufficient points for selected k_neighbors."
            )
        return LatticeConstraintResult(
            coordinates_input=coords,
            coordinates_regularized=coords.copy(),
            displacements=np.zeros_like(coords),
            enabled=False,
            parameters={
                "k_neighbors": int(k_neighbors),
                "strength": float(strength),
                "iterations": int(iterations),
                "max_shift_per_iter": float(max_shift_per_iter),
            },
            confidence=1.0,
            warnings=warnings,
            provenance={
                "source": "autotem.atomic.lattice_constraint.apply_lattice_constraint",
                "applied": False,
            },
        )

    current = coords.copy()
    total_clamped = 0

    for _ in range(iterations):
        tree = cKDTree(current)
        _, idxs = tree.query(current, k=min(k_neighbors + 1, len(current)))
        updated = current.copy()

        for i in range(len(current)):
            nn = np.atleast_1d(idxs[i])
            nn = nn[nn != i]
            if len(nn) == 0:
                continue
            local_center = np.mean(current[nn], axis=0)
            delta = (local_center - current[i]) * strength
            shift = float(np.hypot(delta[0], delta[1]))
            if shift > max_shift_per_iter:
                delta *= max_shift_per_iter / shift
                total_clamped += 1
            updated[i] = current[i] + delta
        current = updated

    displacements = current - coords
    mean_shift = float(np.mean(np.linalg.norm(displacements, axis=1))) if len(coords) else 0.0
    confidence = float(
        np.clip(
            1.0 - mean_shift / (max_shift_per_iter * max(iterations, 1)),
            0.0,
            1.0,
        )
    )

    warnings.append(
        "Lattice constraint may suppress true defect/dislocation signals; "
        "disable for defect/GPA studies."
    )
    if total_clamped > 0:
        warnings.append(f"{total_clamped} updates were clamped by max_shift_per_iter.")

    return LatticeConstraintResult(
        coordinates_input=coords,
        coordinates_regularized=current,
        displacements=displacements,
        enabled=True,
        parameters={
            "k_neighbors": int(k_neighbors),
            "strength": float(strength),
            "iterations": int(iterations),
            "max_shift_per_iter": float(max_shift_per_iter),
        },
        confidence=confidence,
        warnings=warnings,
        provenance={
            "source": "autotem.atomic.lattice_constraint.apply_lattice_constraint",
            "applied": True,
            "clamped_updates": total_clamped,
        },
    )
