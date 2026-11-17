"""Atomic-column intensity quantification from detected atom coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

SATURATION_THRESHOLD_MISSING = "Detector saturation was not evaluated. No threshold was supplied."


@dataclass(slots=True)
class AtomicColumnIntensityResult:
    """Local disk-integrated atomic-column intensity table."""

    coordinates_yx: np.ndarray
    peak_intensity: np.ndarray
    raw_sum: np.ndarray
    raw_mean: np.ndarray
    background_level: np.ndarray
    background_mean: np.ndarray
    background_median: np.ndarray
    background_std: np.ndarray
    background_subtracted_sum: np.ndarray
    normalized_intensity: np.ndarray
    snr: np.ndarray
    atom_area_px: np.ndarray
    background_area_px: np.ndarray
    nearest_neighbor_distance_px: np.ndarray
    valid_atom_mask: np.ndarray
    edge_flag: np.ndarray
    overlap_flag: np.ndarray
    saturation_flag: np.ndarray
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]

    @property
    def saturation_status(self) -> list[str]:
        """Threshold status of finite integration-disk pixels in each measured column."""
        if self.provenance.get("parameters", {}).get("saturation_threshold") is None:
            return ["not_evaluated"] * len(self.saturation_flag)
        measured = np.where(self.saturation_flag, "at_or_above_threshold", "below_threshold")
        return np.where(self.valid_atom_mask, measured, "not_evaluated").tolist()


@dataclass(slots=True)
class AtomicIntensityGroupResult:
    """Grouped atomic-column intensity statistics."""

    labels: np.ndarray
    summaries: list[dict[str, Any]]
    metric: str
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]


def quantify_atomic_column_intensity(
    image: np.ndarray,
    coordinates: np.ndarray,
    *,
    integration_radius: float = 3.0,
    background_inner_radius: float = 5.0,
    background_outer_radius: float = 8.0,
    background_statistic: str = "median",
    normalization: str = "median",
    saturation_threshold: float | None = None,
) -> AtomicColumnIntensityResult:
    """Measure local atom-column intensity with annulus background subtraction.

    Coordinates are expected in ``[y, x]`` pixel order. The method integrates raw
    image intensity in a disk around each atom and subtracts a local background
    estimated from an annulus around the same atom.
    """
    arr = np.asarray(image, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("image must be a 2D array.")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    if not (integration_radius > 0):
        raise ValueError("integration_radius must be positive.")
    if not (background_inner_radius > integration_radius):
        raise ValueError("background_inner_radius must be larger than integration_radius.")
    if not (background_outer_radius > background_inner_radius):
        raise ValueError("background_outer_radius must be larger than background_inner_radius.")
    if background_statistic not in {"median", "mean", "percentile_10", "percentile_25"}:
        raise ValueError(
            "background_statistic must be median, mean, percentile_10, or percentile_25."
        )
    if normalization not in {"none", "median", "mean"}:
        raise ValueError("normalization must be none, median, or mean.")
    if saturation_threshold is not None:
        saturation_threshold = float(saturation_threshold)
        if not np.isfinite(saturation_threshold):
            raise ValueError("saturation_threshold must be finite.")

    n_atoms = len(coords)
    shape = arr.shape
    warnings: list[str] = []
    if saturation_threshold is None:
        warnings.append(SATURATION_THRESHOLD_MISSING)

    peak = np.full(n_atoms, np.nan, dtype=np.float64)
    raw_sum = np.full(n_atoms, np.nan, dtype=np.float64)
    raw_mean = np.full(n_atoms, np.nan, dtype=np.float64)
    bg_level = np.full(n_atoms, np.nan, dtype=np.float64)
    bg_mean = np.full(n_atoms, np.nan, dtype=np.float64)
    bg_median = np.full(n_atoms, np.nan, dtype=np.float64)
    bg_std = np.full(n_atoms, np.nan, dtype=np.float64)
    bg_sub_sum = np.full(n_atoms, np.nan, dtype=np.float64)
    snr = np.full(n_atoms, np.nan, dtype=np.float64)
    atom_area = np.zeros(n_atoms, dtype=np.int32)
    bg_area = np.zeros(n_atoms, dtype=np.int32)
    edge_flag = np.zeros(n_atoms, dtype=bool)
    saturation_flag = np.zeros(n_atoms, dtype=bool)
    valid = np.zeros(n_atoms, dtype=bool)

    nearest = _nearest_neighbor_distances(coords)
    overlap_flag = nearest < (2.0 * float(integration_radius))

    patch_radius = int(np.ceil(background_outer_radius))
    for idx, (y, x) in enumerate(coords):
        if not np.isfinite(y) or not np.isfinite(x):
            continue
        y0 = max(0, int(np.floor(y - patch_radius)))
        y1 = min(shape[0], int(np.ceil(y + patch_radius)) + 1)
        x0 = max(0, int(np.floor(x - patch_radius)))
        x1 = min(shape[1], int(np.ceil(x + patch_radius)) + 1)
        edge_flag[idx] = (
            y - background_outer_radius < 0
            or y + background_outer_radius >= shape[0]
            or x - background_outer_radius < 0
            or x + background_outer_radius >= shape[1]
        )
        if y0 >= y1 or x0 >= x1:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        rr = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
        atom_mask = rr <= float(integration_radius)
        bg_mask = (rr >= float(background_inner_radius)) & (
            rr <= float(background_outer_radius)
        )
        atom_values = arr[y0:y1, x0:x1][atom_mask]
        bg_values = arr[y0:y1, x0:x1][bg_mask]
        atom_values = atom_values[np.isfinite(atom_values)]
        bg_values = bg_values[np.isfinite(bg_values)]
        atom_area[idx] = int(atom_values.size)
        bg_area[idx] = int(bg_values.size)
        if atom_values.size == 0 or bg_values.size < 3:
            continue

        peak[idx] = float(np.max(atom_values))
        raw_sum[idx] = float(np.sum(atom_values))
        raw_mean[idx] = float(np.mean(atom_values))
        bg_mean[idx] = float(np.mean(bg_values))
        bg_median[idx] = float(np.median(bg_values))
        bg_std[idx] = float(np.std(bg_values))
        bg_level[idx] = _background_level(bg_values, background_statistic)
        bg_sub_sum[idx] = raw_sum[idx] - bg_level[idx] * float(atom_values.size)
        snr[idx] = (raw_mean[idx] - bg_level[idx]) / max(bg_std[idx], 1e-12)
        if saturation_threshold is not None:
            saturation_flag[idx] = bool(np.any(atom_values >= saturation_threshold))
        valid[idx] = True

    normalized = _normalize_intensity(bg_sub_sum, valid, normalization, warnings)
    valid_ratio = float(np.mean(valid)) if n_atoms else 0.0
    edge_ratio = float(np.mean(edge_flag)) if n_atoms else 0.0
    overlap_ratio = float(np.mean(overlap_flag)) if n_atoms else 0.0
    median_snr = _safe_nanmedian(snr[valid])
    snr_score = float(np.clip(median_snr / 5.0, 0.0, 1.0)) if np.isfinite(median_snr) else 0.0
    confidence = float(
        np.clip(
            0.45 * valid_ratio
            + 0.25 * (1.0 - edge_ratio)
            + 0.15 * (1.0 - overlap_ratio)
            + 0.15 * snr_score,
            0.0,
            1.0,
        )
    )

    if valid_ratio < 0.95:
        warnings.append(
            "Some atom-column intensities could not be measured. "
            f"valid_ratio={valid_ratio:.3f}"
        )
    if edge_ratio > 0.0:
        warnings.append(
            "Some atoms are close to image edges; their background annulus is truncated. "
            f"edge_count={int(np.sum(edge_flag))}"
        )
    if overlap_ratio > 0.0:
        warnings.append(
            "Some integration disks overlap neighboring atoms. "
            "Reduce integration radius for dense columns. "
            f"overlap_count={int(np.sum(overlap_flag))}"
        )
    if np.isfinite(median_snr) and median_snr < 1.0:
        warnings.append(
            "Median local SNR is low; intensity quantification may be unstable. "
            f"median_snr={median_snr:.3f}"
        )

    provenance = {
        "source": "autotem.atomic.intensity_quantification.quantify_atomic_column_intensity",
        "parameters": {
            "integration_radius": float(integration_radius),
            "background_inner_radius": float(background_inner_radius),
            "background_outer_radius": float(background_outer_radius),
            "background_statistic": background_statistic,
            "normalization": normalization,
            "saturation_threshold": saturation_threshold,
        },
        "method": "disk_integration_with_local_annulus_background",
        "fit_statistics": {
            "atom_count": int(n_atoms),
            "valid_ratio": valid_ratio,
            "edge_count": int(np.sum(edge_flag)),
            "overlap_count": int(np.sum(overlap_flag)),
            "median_snr": None if not np.isfinite(median_snr) else float(median_snr),
        },
        "caution": (
            "Values are relative atomic-column intensities. Interpreting them as "
            "Z-contrast requires appropriate HAADF-STEM conditions and calibration."
        ),
    }

    return AtomicColumnIntensityResult(
        coordinates_yx=coords,
        peak_intensity=peak.astype(np.float32),
        raw_sum=raw_sum.astype(np.float32),
        raw_mean=raw_mean.astype(np.float32),
        background_level=bg_level.astype(np.float32),
        background_mean=bg_mean.astype(np.float32),
        background_median=bg_median.astype(np.float32),
        background_std=bg_std.astype(np.float32),
        background_subtracted_sum=bg_sub_sum.astype(np.float32),
        normalized_intensity=normalized.astype(np.float32),
        snr=snr.astype(np.float32),
        atom_area_px=atom_area,
        background_area_px=bg_area,
        nearest_neighbor_distance_px=nearest.astype(np.float32),
        valid_atom_mask=valid,
        edge_flag=edge_flag,
        overlap_flag=overlap_flag,
        saturation_flag=saturation_flag,
        confidence=confidence,
        warnings=warnings,
        provenance=provenance,
    )


def cluster_atomic_column_intensities(
    result: AtomicColumnIntensityResult,
    *,
    metric: str = "normalized_intensity",
    n_groups: int = 2,
) -> AtomicIntensityGroupResult:
    """Group atom columns by 1D intensity clustering."""
    values = atomic_intensity_values(result, metric)
    valid = np.asarray(result.valid_atom_mask, dtype=bool) & np.isfinite(values)
    labels, warnings = _cluster_1d(values, valid, n_groups=n_groups)
    grouped = summarize_atomic_intensity_groups(
        values,
        labels,
        valid_mask=valid,
        metric=metric,
        method="intensity_1d_kmeans",
    )
    grouped.warnings[:0] = warnings
    grouped.provenance["parameters"]["n_groups"] = int(n_groups)
    return grouped


def group_atomic_column_intensities_by_lattice(
    result: AtomicColumnIntensityResult,
    lattice_indices_ij: np.ndarray,
    *,
    metric: str = "normalized_intensity",
    mode: str = "ij_parity",
    valid_lattice_mask: np.ndarray | None = None,
) -> AtomicIntensityGroupResult:
    """Group atom columns by user-defined a/b lattice-index parity."""
    values = atomic_intensity_values(result, metric)
    labels = lattice_parity_labels(lattice_indices_ij, mode=mode)
    valid = np.asarray(result.valid_atom_mask, dtype=bool) & np.isfinite(values) & (labels >= 0)
    if valid_lattice_mask is not None:
        mask = np.asarray(valid_lattice_mask, dtype=bool)
        if mask.shape != valid.shape:
            raise ValueError("valid_lattice_mask must match intensity result length.")
        valid &= mask

    grouped = summarize_atomic_intensity_groups(
        values,
        labels,
        valid_mask=valid,
        metric=metric,
        method=f"lattice_{mode}",
        group_names=_lattice_group_names(mode),
    )
    grouped.provenance["parameters"]["lattice_group_mode"] = mode
    return grouped


def summarize_atomic_intensity_groups(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    metric: str = "intensity",
    method: str = "provided_labels",
    group_names: dict[int, str] | None = None,
) -> AtomicIntensityGroupResult:
    """Summarize intensity values for precomputed group labels."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    if values.shape != labels.shape:
        raise ValueError("values and labels must have matching shape.")
    if valid_mask is None:
        valid = np.isfinite(values) & (labels >= 0)
    else:
        valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values) & (labels >= 0)
        if valid.shape != values.shape:
            raise ValueError("valid_mask must match values shape.")

    warnings: list[str] = []
    summaries: list[dict[str, Any]] = []
    for group_id in sorted(int(v) for v in np.unique(labels[valid])):
        group_values = values[valid & (labels == group_id)]
        if group_values.size == 0:
            continue
        summaries.append(
            {
                "group_id": group_id,
                "group_name": (
                    group_names.get(group_id, f"group_{group_id}")
                    if group_names is not None
                    else f"group_{group_id}"
                ),
                "count": int(group_values.size),
                "mean": float(np.mean(group_values)),
                "std": float(np.std(group_values)),
                "median": float(np.median(group_values)),
                "min": float(np.min(group_values)),
                "max": float(np.max(group_values)),
            }
        )

    if not summaries:
        warnings.append("No valid atom columns are available for intensity grouping.")
    elif len(summaries) == 1:
        warnings.append("Only one non-empty intensity group was found.")

    grouped_count = int(np.sum(valid))
    confidence = float(grouped_count / max(len(values), 1))
    provenance = {
        "source": "autotem.atomic.intensity_quantification.summarize_atomic_intensity_groups",
        "method": method,
        "parameters": {
            "metric": metric,
            "valid_count": grouped_count,
            "group_count": len(summaries),
        },
    }
    return AtomicIntensityGroupResult(
        labels=labels.astype(np.int32),
        summaries=summaries,
        metric=metric,
        confidence=confidence,
        warnings=warnings,
        provenance=provenance,
    )


def atomic_intensity_values(result: AtomicColumnIntensityResult, metric: str) -> np.ndarray:
    """Return a selected intensity metric from an intensity result."""
    mapping = {
        "background_subtracted_sum": result.background_subtracted_sum,
        "normalized_intensity": result.normalized_intensity,
        "peak_intensity": result.peak_intensity,
        "snr": result.snr,
    }
    if metric not in mapping:
        raise ValueError(
            "metric must be background_subtracted_sum, normalized_intensity, "
            "peak_intensity, or snr."
        )
    return np.asarray(mapping[metric], dtype=np.float64)


def lattice_parity_labels(lattice_indices_ij: np.ndarray, *, mode: str = "ij_parity") -> np.ndarray:
    """Create deterministic sublattice labels from integer a/b lattice indices."""
    indices = np.asarray(lattice_indices_ij, dtype=np.int32)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError("lattice_indices_ij must have shape (N, 2).")
    i_parity = np.mod(indices[:, 0], 2)
    j_parity = np.mod(indices[:, 1], 2)
    if mode == "i_parity":
        return i_parity.astype(np.int32)
    if mode == "j_parity":
        return j_parity.astype(np.int32)
    if mode == "sum_parity":
        return np.mod(i_parity + j_parity, 2).astype(np.int32)
    if mode == "ij_parity":
        return (2 * i_parity + j_parity).astype(np.int32)
    raise ValueError("mode must be i_parity, j_parity, sum_parity, or ij_parity.")


def _background_level(values: np.ndarray, statistic: str) -> float:
    if statistic == "median":
        return float(np.median(values))
    if statistic == "mean":
        return float(np.mean(values))
    if statistic == "percentile_10":
        return float(np.percentile(values, 10))
    if statistic == "percentile_25":
        return float(np.percentile(values, 25))
    raise ValueError(f"Unsupported background statistic: {statistic}")


def _normalize_intensity(
    values: np.ndarray,
    valid: np.ndarray,
    normalization: str,
    warnings: list[str],
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if normalization == "none":
        return out
    finite = out[np.asarray(valid, dtype=bool) & np.isfinite(out)]
    if finite.size == 0:
        warnings.append("No finite valid intensities are available for normalization.")
        return np.full_like(out, np.nan, dtype=np.float64)
    if normalization == "median":
        denom = float(np.median(finite))
    elif normalization == "mean":
        denom = float(np.mean(finite))
    else:
        raise ValueError(f"Unsupported normalization: {normalization}")
    if not np.isfinite(denom) or abs(denom) <= 1e-12:
        warnings.append("Intensity normalization denominator is near zero; normalization skipped.")
        return out
    return out / denom


def _cluster_1d(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    n_groups: int,
    max_iter: int = 100,
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if not (2 <= int(n_groups) <= 8):
        raise ValueError("n_groups must be in [2, 8].")
    finite_values = values[valid]
    labels = np.full(len(values), -1, dtype=np.int32)
    warnings: list[str] = []
    if finite_values.size < int(n_groups):
        warnings.append(
            "Too few valid atoms for requested intensity clusters; "
            f"valid={finite_values.size}, requested={int(n_groups)}."
        )
        return labels, warnings

    percentiles = np.linspace(0, 100, int(n_groups) + 2)[1:-1]
    centers = np.percentile(finite_values, percentiles).astype(np.float64)
    centers = np.unique(centers)
    if centers.size < int(n_groups):
        centers = np.linspace(
            float(np.min(finite_values)),
            float(np.max(finite_values)),
            int(n_groups),
        )
    if np.allclose(centers, centers[0]):
        warnings.append("Intensity values are nearly identical; clustering is not meaningful.")
        labels[valid] = 0
        return labels, warnings

    centers = centers[: int(n_groups)]
    working = finite_values
    assigned = np.zeros(working.size, dtype=np.int32)
    for _ in range(max_iter):
        distances = np.abs(working[:, None] - centers[None, :])
        new_assigned = np.argmin(distances, axis=1).astype(np.int32)
        new_centers = centers.copy()
        for group_id in range(len(centers)):
            group_values = working[new_assigned == group_id]
            if group_values.size:
                new_centers[group_id] = float(np.mean(group_values))
        if np.array_equal(new_assigned, assigned) and np.allclose(new_centers, centers):
            assigned = new_assigned
            centers = new_centers
            break
        assigned = new_assigned
        centers = new_centers

    order = np.argsort(centers)
    remap = {int(old): int(new) for new, old in enumerate(order)}
    sorted_assigned = np.asarray([remap[int(v)] for v in assigned], dtype=np.int32)
    labels[valid] = sorted_assigned
    return labels, warnings


def _lattice_group_names(mode: str) -> dict[int, str]:
    if mode == "i_parity":
        return {0: "i even", 1: "i odd"}
    if mode == "j_parity":
        return {0: "j even", 1: "j odd"}
    if mode == "sum_parity":
        return {0: "i+j even", 1: "i+j odd"}
    if mode == "ij_parity":
        return {
            0: "i even / j even",
            1: "i even / j odd",
            2: "i odd / j even",
            3: "i odd / j odd",
        }
    return {}


def _nearest_neighbor_distances(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) <= 1:
        return np.full(len(coords), np.inf, dtype=np.float64)
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    return np.asarray(distances[:, 1], dtype=np.float64)


def _safe_nanmedian(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))
