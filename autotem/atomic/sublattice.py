"""Sublattice classification and projected-structure summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class SublatticeClassificationResult:
    """Per-atom sublattice labels in a user-defined projected a/b cell."""

    coordinates_yx: np.ndarray
    labels: np.ndarray
    label_ids: np.ndarray
    lattice_indices_ij: np.ndarray
    fractional_uv: np.ndarray
    unit_cell_uv: np.ndarray
    intensities: np.ndarray
    valid_atom_mask: np.ndarray
    manual_override_mask: np.ndarray
    label_summaries: list[dict[str, Any]]
    projected_structure: dict[str, Any]
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]


def classify_sublattices_auto(
    coordinates: np.ndarray,
    *,
    base_index: int,
    a_index: int,
    b_index: int,
    intensities: np.ndarray | None = None,
    n_sublattices: int = 2,
    use_intensity: bool = True,
    intensity_weight: float = 0.6,
    manual_labels: dict[int, str] | None = None,
) -> SublatticeClassificationResult:
    """Explore sublattices by clustering unit-cell position and optional intensity."""
    coords, fractional, unit_uv, lattice_indices = _fractional_coordinates_from_basis(
        coordinates,
        base_index=base_index,
        a_index=a_index,
        b_index=b_index,
    )
    if not (1 <= int(n_sublattices) <= 8):
        raise ValueError("n_sublattices must be in [1, 8].")

    intensity_arr = _normalize_intensity_input(intensities, len(coords))
    valid = np.all(np.isfinite(unit_uv), axis=1)
    features = _unit_cell_features(unit_uv)
    if use_intensity and np.any(np.isfinite(intensity_arr)):
        intensity_feature = _robust_zscore(intensity_arr)
        intensity_feature[~np.isfinite(intensity_feature)] = 0.0
        features = np.column_stack([features, float(intensity_weight) * intensity_feature])

    label_ids = _kmeans_features(features[valid], n_clusters=int(n_sublattices))
    full_label_ids = np.full(len(coords), -1, dtype=np.int32)
    full_label_ids[valid] = label_ids
    full_label_ids = _sort_cluster_ids(full_label_ids, intensity_arr, unit_uv)
    labels = np.asarray(
        [f"sublattice_{idx}" if idx >= 0 else "unknown" for idx in full_label_ids],
        dtype=object,
    )
    manual_mask = _apply_manual_labels(labels, manual_labels)
    result = _build_result(
        coords=coords,
        labels=labels,
        fractional=fractional,
        unit_uv=unit_uv,
        lattice_indices=lattice_indices,
        intensities=intensity_arr,
        valid=valid,
        manual_mask=manual_mask,
        method="auto_exploration",
        parameters={
            "base_index": int(base_index),
            "a_index": int(a_index),
            "b_index": int(b_index),
            "n_sublattices": int(n_sublattices),
            "use_intensity": bool(use_intensity),
            "intensity_weight": float(intensity_weight),
        },
    )
    if not use_intensity:
        result.warnings.append("Auto sublattice classification used geometry only.")
    return result


def classify_sublattices_from_model(
    coordinates: np.ndarray,
    *,
    base_index: int,
    a_index: int,
    b_index: int,
    basis_sites: list[dict[str, Any]],
    intensities: np.ndarray | None = None,
    match_tolerance: float = 0.20,
    manual_labels: dict[int, str] | None = None,
) -> SublatticeClassificationResult:
    """Assign atoms to user-provided projected 2D basis sites."""
    coords, fractional, unit_uv, lattice_indices = _fractional_coordinates_from_basis(
        coordinates,
        base_index=base_index,
        a_index=a_index,
        b_index=b_index,
    )
    sites = _normalize_basis_sites(basis_sites)
    if not (0.01 <= float(match_tolerance) <= 0.75):
        raise ValueError("match_tolerance must be in [0.01, 0.75].")

    site_uv = np.asarray([[site["u"], site["v"]] for site in sites], dtype=np.float64)
    site_labels = [str(site["label"]) for site in sites]
    distances = np.zeros((len(coords), len(sites)), dtype=np.float64)
    for idx, uv in enumerate(site_uv):
        diff = np.abs(unit_uv - uv)
        diff = np.minimum(diff, 1.0 - diff)
        distances[:, idx] = np.linalg.norm(diff, axis=1)
    nearest = np.argmin(distances, axis=1)
    nearest_dist = distances[np.arange(len(coords)), nearest]
    valid = np.isfinite(nearest_dist) & (nearest_dist <= float(match_tolerance))
    labels = np.asarray(
        [
            site_labels[int(idx)] if bool(ok) else "unknown"
            for idx, ok in zip(nearest, valid, strict=False)
        ],
        dtype=object,
    )
    manual_mask = _apply_manual_labels(labels, manual_labels)
    result = _build_result(
        coords=coords,
        labels=labels,
        fractional=fractional,
        unit_uv=unit_uv,
        lattice_indices=lattice_indices,
        intensities=_normalize_intensity_input(intensities, len(coords)),
        valid=valid,
        manual_mask=manual_mask,
        method="user_projected_model",
        parameters={
            "base_index": int(base_index),
            "a_index": int(a_index),
            "b_index": int(b_index),
            "match_tolerance": float(match_tolerance),
            "basis_sites": sites,
        },
    )
    unknown_count = int(np.sum(labels == "unknown"))
    if unknown_count:
        result.warnings.append(
            "Some atoms could not be matched to the projected model. "
            f"unknown_count={unknown_count}"
        )
    return result


def apply_manual_sublattice_labels(
    result: SublatticeClassificationResult,
    manual_labels: dict[int, str],
) -> SublatticeClassificationResult:
    """Return a copy of an existing classification with manual label overrides."""
    labels = np.asarray(result.labels, dtype=object).copy()
    manual_mask = np.asarray(result.manual_override_mask, dtype=bool).copy()
    for atom_idx, label in manual_labels.items():
        idx = int(atom_idx)
        if idx < 0 or idx >= len(labels):
            raise ValueError(f"Manual label atom index {idx} is out of range.")
        labels[idx] = str(label).strip() or "unknown"
        manual_mask[idx] = True
    return _build_result(
        coords=np.asarray(result.coordinates_yx, dtype=np.float64),
        labels=labels,
        fractional=np.asarray(result.fractional_uv, dtype=np.float64),
        unit_uv=np.asarray(result.unit_cell_uv, dtype=np.float64),
        lattice_indices=np.asarray(result.lattice_indices_ij, dtype=np.int32),
        intensities=np.asarray(result.intensities, dtype=np.float64),
        valid=np.asarray(result.valid_atom_mask, dtype=bool),
        manual_mask=manual_mask,
        method=str(result.provenance.get("method", "manual_relabel")),
        parameters={
            **dict(result.provenance.get("parameters", {})),
            "manual_label_count": int(np.sum(manual_mask)),
        },
    )


def parse_projected_model_csv(text: str) -> list[dict[str, Any]]:
    """Parse ``label,u,v`` projected model text from the UI."""
    rows: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().replace(" ", "") in {"label,u,v", "label,frac_u,frac_v"}:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Model line {line_no} must contain label,u,v.")
        label, u_text, v_text = parts
        rows.append({"label": label, "u": float(u_text), "v": float(v_text)})
    if not rows:
        raise ValueError("Projected model must include at least one basis site.")
    return _normalize_basis_sites(rows)


def _fractional_coordinates_from_basis(
    coordinates: np.ndarray,
    *,
    base_index: int,
    a_index: int,
    b_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    n_atoms = len(coords)
    if n_atoms < 3:
        raise ValueError("Need at least 3 atoms to define a projected a/b cell.")
    for idx_name, idx in {
        "base_index": base_index,
        "a_index": a_index,
        "b_index": b_index,
    }.items():
        if idx < 0 or idx >= n_atoms:
            raise ValueError(f"{idx_name}={idx} is out of range for {n_atoms} atoms.")
    if len({int(base_index), int(a_index), int(b_index)}) != 3:
        raise ValueError("base_index, a_index, and b_index must reference three different atoms.")

    base_yx = coords[int(base_index)]
    a_vec_yx = coords[int(a_index)] - base_yx
    b_vec_yx = coords[int(b_index)] - base_yx
    basis_yx = np.column_stack([a_vec_yx, b_vec_yx])
    det = float(np.linalg.det(basis_yx))
    if abs(det) <= 1e-9:
        raise ValueError("Custom basis vectors are singular or nearly collinear.")
    fractional = (np.linalg.inv(basis_yx) @ (coords - base_yx).T).T
    lattice_indices = np.floor(fractional).astype(np.int32)
    unit_uv = fractional - np.floor(fractional)
    unit_uv = np.mod(unit_uv, 1.0)
    return coords, fractional, unit_uv, lattice_indices


def _normalize_intensity_input(intensities: np.ndarray | None, n_atoms: int) -> np.ndarray:
    if intensities is None:
        return np.full(n_atoms, np.nan, dtype=np.float64)
    arr = np.asarray(intensities, dtype=np.float64)
    if arr.shape != (n_atoms,):
        raise ValueError("intensities must have shape (N,) matching coordinates.")
    return arr


def _unit_cell_features(unit_uv: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi * np.asarray(unit_uv, dtype=np.float64)
    return np.column_stack(
        [
            np.cos(angle[:, 0]),
            np.sin(angle[:, 0]),
            np.cos(angle[:, 1]),
            np.sin(angle[:, 1]),
        ]
    )


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(arr)
    return (arr - med) / scale


def _kmeans_features(features: np.ndarray, *, n_clusters: int, max_iter: int = 100) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("features must be 2D.")
    n_samples = len(arr)
    if n_samples == 0:
        return np.zeros(0, dtype=np.int32)
    n_clusters = min(int(n_clusters), n_samples)
    if n_clusters <= 1:
        return np.zeros(n_samples, dtype=np.int32)

    centers = [arr[int(np.argmax(np.linalg.norm(arr - np.mean(arr, axis=0), axis=1)))]]
    while len(centers) < n_clusters:
        center_arr = np.asarray(centers)
        d2 = np.min(np.sum((arr[:, None, :] - center_arr[None, :, :]) ** 2, axis=2), axis=1)
        centers.append(arr[int(np.argmax(d2))])
    centers_arr = np.asarray(centers, dtype=np.float64)
    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(max_iter):
        distances = np.sum((arr[:, None, :] - centers_arr[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(distances, axis=1).astype(np.int32)
        new_centers = centers_arr.copy()
        for cluster_id in range(n_clusters):
            cluster_values = arr[new_labels == cluster_id]
            if len(cluster_values):
                new_centers[cluster_id] = np.mean(cluster_values, axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(centers_arr, new_centers):
            break
        labels = new_labels
        centers_arr = new_centers
    return labels


def _sort_cluster_ids(
    label_ids: np.ndarray,
    intensities: np.ndarray,
    unit_uv: np.ndarray,
) -> np.ndarray:
    out = np.asarray(label_ids, dtype=np.int32).copy()
    cluster_ids = [int(v) for v in np.unique(out[out >= 0])]
    if not cluster_ids:
        return out
    if np.any(np.isfinite(intensities)):
        sort_key = {
            cid: -float(np.nanmean(intensities[out == cid]))
            if np.any(np.isfinite(intensities[out == cid]))
            else 0.0
            for cid in cluster_ids
        }
    else:
        sort_key = {
            cid: (
                float(np.mean(unit_uv[out == cid, 0])),
                float(np.mean(unit_uv[out == cid, 1])),
            )
            for cid in cluster_ids
        }
    ordered = sorted(cluster_ids, key=lambda cid: sort_key[cid])
    remap = {old: new for new, old in enumerate(ordered)}
    for old, new in remap.items():
        out[label_ids == old] = int(new)
    return out


def _normalize_basis_sites(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for site in sites:
        label = str(site.get("label", "")).strip()
        if not label:
            raise ValueError("Each projected model basis site needs a non-empty label.")
        u = float(site.get("u"))
        v = float(site.get("v"))
        if not np.isfinite(u) or not np.isfinite(v):
            raise ValueError(f"Projected model site {label} has non-finite coordinates.")
        normalized.append({"label": label, "u": float(np.mod(u, 1.0)), "v": float(np.mod(v, 1.0))})
    return normalized


def _apply_manual_labels(labels: np.ndarray, manual_labels: dict[int, str] | None) -> np.ndarray:
    manual_mask = np.zeros(len(labels), dtype=bool)
    if not manual_labels:
        return manual_mask
    for atom_idx, label in manual_labels.items():
        idx = int(atom_idx)
        if idx < 0 or idx >= len(labels):
            raise ValueError(f"Manual label atom index {idx} is out of range.")
        labels[idx] = str(label).strip() or "unknown"
        manual_mask[idx] = True
    return manual_mask


def _build_result(
    *,
    coords: np.ndarray,
    labels: np.ndarray,
    fractional: np.ndarray,
    unit_uv: np.ndarray,
    lattice_indices: np.ndarray,
    intensities: np.ndarray,
    valid: np.ndarray,
    manual_mask: np.ndarray,
    method: str,
    parameters: dict[str, Any],
) -> SublatticeClassificationResult:
    labels = np.asarray(labels, dtype=object)
    unique_labels = sorted(str(v) for v in np.unique(labels))
    id_lookup = {label: idx for idx, label in enumerate(unique_labels)}
    label_ids = np.asarray([id_lookup[str(label)] for label in labels], dtype=np.int32)
    summaries = _label_summaries(labels, unit_uv, intensities, valid, manual_mask)
    projected = {
        "basis_sites": [
            {
                "label": row["label"],
                "u": row["mean_u"],
                "v": row["mean_v"],
                "count": row["count"],
                "mean_intensity": row["mean_intensity"],
            }
            for row in summaries
        ],
        "coordinate_frame": "fractional coordinates in the user-defined projected a/b cell",
    }
    valid_ratio = float(np.mean(valid)) if len(valid) else 0.0
    confidence = float(
        np.clip(
            valid_ratio * (1.0 - 0.03 * max(len(unique_labels) - 1, 0)),
            0.0,
            1.0,
        )
    )
    warnings: list[str] = []
    if "unknown" in unique_labels:
        warnings.append(f"Unknown/unmatched atoms: {int(np.sum(labels == 'unknown'))}")
    if np.sum(valid) < max(3, len(unique_labels)):
        warnings.append("Too few valid atoms for robust sublattice classification.")
    provenance = {
        "source": "autotem.atomic.sublattice",
        "method": method,
        "parameters": parameters,
        "manual_label_count": int(np.sum(manual_mask)),
    }
    return SublatticeClassificationResult(
        coordinates_yx=np.asarray(coords, dtype=np.float32),
        labels=labels,
        label_ids=label_ids,
        lattice_indices_ij=np.asarray(lattice_indices, dtype=np.int32),
        fractional_uv=np.asarray(fractional, dtype=np.float32),
        unit_cell_uv=np.asarray(unit_uv, dtype=np.float32),
        intensities=np.asarray(intensities, dtype=np.float32),
        valid_atom_mask=np.asarray(valid, dtype=bool),
        manual_override_mask=np.asarray(manual_mask, dtype=bool),
        label_summaries=summaries,
        projected_structure=projected,
        confidence=confidence,
        warnings=warnings,
        provenance=provenance,
    )


def _label_summaries(
    labels: np.ndarray,
    unit_uv: np.ndarray,
    intensities: np.ndarray,
    valid: np.ndarray,
    manual_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in sorted(str(v) for v in np.unique(labels)):
        mask = labels == label
        values = intensities[mask]
        finite_values = values[np.isfinite(values)]
        rows.append(
            {
                "label": label,
                "count": int(np.sum(mask)),
                "valid_count": int(np.sum(mask & valid)),
                "manual_count": int(np.sum(mask & manual_mask)),
                "mean_u": float(np.mean(unit_uv[mask, 0])) if np.any(mask) else None,
                "mean_v": float(np.mean(unit_uv[mask, 1])) if np.any(mask) else None,
                "mean_intensity": float(np.mean(finite_values)) if finite_values.size else None,
                "std_intensity": float(np.std(finite_values)) if finite_values.size else None,
            }
        )
    return rows
