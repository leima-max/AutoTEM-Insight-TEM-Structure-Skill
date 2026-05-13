"""Atom-distance statistics utilities (C1 baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from autotem.core.models import AtomLattice


@dataclass
class DistanceStatisticsResult:
    coordinates: np.ndarray
    nearest_distances: np.ndarray
    next_nearest_distances: np.ndarray
    nearest_summary: dict[str, float | int]
    next_nearest_summary: dict[str, float | int]
    unit: str
    method: str
    parameters: dict[str, Any]
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]

    def to_rows(self) -> list[dict[str, float | int]]:
        """Export per-atom rows for CSV writers."""
        rows: list[dict[str, float | int]] = []
        for idx, (coord, nn, nnn) in enumerate(
            zip(
                self.coordinates,
                self.nearest_distances,
                self.next_nearest_distances,
                strict=False,
            )
        ):
            y, x = float(coord[0]), float(coord[1])
            rows.append(
                {
                    "atom_index": idx,
                    "y": y,
                    "x": x,
                    "nearest_distance": float(nn) if np.isfinite(nn) else float("nan"),
                    "next_nearest_distance": float(nnn) if np.isfinite(nnn) else float("nan"),
                }
            )
        return rows


def compute_distance_statistics(
    coordinates: np.ndarray,
    *,
    pixel_size: float | None = None,
    output_unit: str | None = None,
) -> DistanceStatisticsResult:
    """
    Compute nearest-neighbor and next-nearest-neighbor distances.

    Parameters
    ----------
    coordinates:
        Atom coordinates in `[y, x]` convention with shape `(N, 2)`.
    pixel_size:
        Optional pixel-to-physical scale factor. If provided, output distances
        are scaled by `pixel_size`.
    output_unit:
        Unit name for scaled output. Required when `pixel_size` is provided.
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")

    warnings: list[str] = []
    n_atoms = int(len(coords))

    if pixel_size is None:
        scale = 1.0
        unit = "pixel"
    else:
        if pixel_size <= 0:
            raise ValueError("pixel_size must be > 0 when provided.")
        if output_unit is None or not output_unit.strip():
            raise ValueError("output_unit is required when pixel_size is provided.")
        scale = float(pixel_size)
        unit = output_unit.strip()

    nearest = np.full((n_atoms,), np.nan, dtype=np.float64)
    next_nearest = np.full((n_atoms,), np.nan, dtype=np.float64)

    if n_atoms < 2:
        warnings.append("At least 2 atoms are required for nearest-neighbor statistics.")
    else:
        k = min(3, n_atoms)
        tree = cKDTree(coords)
        distances, _ = tree.query(coords, k=k)
        distances = np.asarray(distances, dtype=np.float64)
        if distances.ndim == 1:
            distances = distances[:, np.newaxis]

        # cKDTree query returns self distance at index 0.
        nearest = distances[:, 1] * scale
        if k >= 3:
            next_nearest = distances[:, 2] * scale
        else:
            warnings.append("Need at least 3 atoms for next-nearest-neighbor statistics.")

    nearest_summary = _summarize(nearest)
    next_nearest_summary = _summarize(next_nearest)
    confidence = _distance_confidence(
        n_atoms=n_atoms,
        nearest=nearest,
        next_nearest=next_nearest,
        warning_count=len(warnings),
    )

    return DistanceStatisticsResult(
        coordinates=coords,
        nearest_distances=nearest,
        next_nearest_distances=next_nearest,
        nearest_summary=nearest_summary,
        next_nearest_summary=next_nearest_summary,
        unit=unit,
        method="cKDTree kNN",
        parameters={
            "pixel_size": pixel_size,
            "output_unit": unit,
            "n_atoms": n_atoms,
        },
        confidence=confidence,
        warnings=warnings,
        provenance={
            "source": "autotem.atomic.distance_statistics.compute_distance_statistics",
            "coordinate_convention": "yx",
            "scaling_factor": scale,
        },
    )


def compute_distance_statistics_from_lattice(
    atom_lattice: AtomLattice,
    *,
    pixel_size: float | None = None,
    output_unit: str | None = None,
) -> DistanceStatisticsResult:
    """
    Compute distance statistics directly from an `AtomLattice`.

    `AtomLattice` stores atom positions as x/y fields. This helper converts
    them to internal `[y, x]` order before distance calculation.
    """
    coords = np.array([[atom.y, atom.x] for atom in atom_lattice.atoms], dtype=np.float64)
    return compute_distance_statistics(
        coords,
        pixel_size=pixel_size,
        output_unit=output_unit,
    )


def _summarize(values: np.ndarray) -> dict[str, float | int]:
    finite = np.isfinite(values)
    valid = values[finite]
    if len(valid) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "sem": float("nan"),
        }
    return {
        "count": int(len(valid)),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "std": float(np.std(valid, ddof=0)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "sem": float(np.std(valid, ddof=0) / np.sqrt(len(valid))),
    }


def _distance_confidence(
    *,
    n_atoms: int,
    nearest: np.ndarray,
    next_nearest: np.ndarray,
    warning_count: int,
) -> float:
    if n_atoms <= 1:
        return 0.0

    finite_nn = nearest[np.isfinite(nearest)]
    finite_nnn = next_nearest[np.isfinite(next_nearest)]
    coverage = (len(finite_nn) + len(finite_nnn)) / max(2 * n_atoms, 1)

    score = 0.45 + 0.45 * min(coverage, 1.0)
    if len(finite_nn) > 1:
        mean_nn = float(np.mean(finite_nn))
        std_nn = float(np.std(finite_nn, ddof=0))
        cv = std_nn / max(mean_nn, 1e-12)
        score += float(np.clip(0.2 * (1.0 - cv), -0.1, 0.15))

    score -= min(0.12 * warning_count, 0.36)
    return float(np.clip(score, 0.0, 1.0))
