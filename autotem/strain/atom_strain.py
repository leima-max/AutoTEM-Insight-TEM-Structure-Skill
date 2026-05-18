"""Atom-position-based strain mapping inspired by AI-STEM/PPA-style workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from autotem.core import ArrayReference, StrainResult


@dataclass(slots=True)
class AtomStrainFieldResult:
    """Full atom-strain fields used by the UI and tests."""

    coordinates_yx: np.ndarray
    atom_displacement_xy: np.ndarray
    atom_strain_xx: np.ndarray
    atom_strain_yy: np.ndarray
    atom_strain_xy: np.ndarray
    atom_rotation_xy: np.ndarray
    atom_uncertainty: np.ndarray
    valid_atom_mask: np.ndarray
    displacement_x_map: np.ndarray
    displacement_y_map: np.ndarray
    strain_xx_map: np.ndarray
    strain_yy_map: np.ndarray
    strain_xy_map: np.ndarray
    rotation_xy_map: np.ndarray
    uncertainty_map: np.ndarray
    reference_vectors_xy: np.ndarray
    reference_center_xy: np.ndarray
    reference_atom_mask: np.ndarray
    reference_region: tuple[int, int, int, int] | None
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]

    def to_strain_result(self, *, map_inline_mode: str = "preview") -> StrainResult:
        """Convert to the cross-module StrainResult contract."""
        map_stats = {
            "mean_ux": float(np.nanmean(self.atom_displacement_xy[:, 0]))
            if len(self.atom_displacement_xy)
            else 0.0,
            "mean_uy": float(np.nanmean(self.atom_displacement_xy[:, 1]))
            if len(self.atom_displacement_xy)
            else 0.0,
            "mean_exx": float(np.nanmean(self.atom_strain_xx)) if len(self.atom_strain_xx) else 0.0,
            "mean_eyy": float(np.nanmean(self.atom_strain_yy)) if len(self.atom_strain_yy) else 0.0,
            "mean_exy": float(np.nanmean(self.atom_strain_xy)) if len(self.atom_strain_xy) else 0.0,
            "mean_rotation": float(np.nanmean(self.atom_rotation_xy))
            if len(self.atom_rotation_xy)
            else 0.0,
            "valid_atom_count": int(np.sum(self.valid_atom_mask)),
            "total_atom_count": int(len(self.coordinates_yx)),
        }
        provenance = dict(self.provenance)
        provenance["map_stats"] = map_stats
        return StrainResult(
            displacement_x=_to_array_ref(
                self.displacement_x_map,
                description="Atom-position ux displacement map.",
                inline_mode=map_inline_mode,
            ),
            displacement_y=_to_array_ref(
                self.displacement_y_map,
                description="Atom-position uy displacement map.",
                inline_mode=map_inline_mode,
            ),
            strain_xx=_to_array_ref(
                self.strain_xx_map,
                description="Atom-position exx strain map.",
                inline_mode=map_inline_mode,
            ),
            strain_yy=_to_array_ref(
                self.strain_yy_map,
                description="Atom-position eyy strain map.",
                inline_mode=map_inline_mode,
            ),
            strain_xy=_to_array_ref(
                self.strain_xy_map,
                description="Atom-position exy strain map.",
                inline_mode=map_inline_mode,
            ),
            rotation_xy=_to_array_ref(
                self.rotation_xy_map,
                description="Atom-position local lattice rotation map.",
                inline_mode=map_inline_mode,
            ),
            uncertainty=_to_array_ref(
                self.uncertainty_map,
                description="Heuristic uncertainty map from local affine residuals.",
                inline_mode=map_inline_mode,
            ),
            reference_region=self.reference_region,
            confidence=self.confidence,
            provenance=provenance,
            warnings=list(self.warnings),
        )


def run_atom_strain(
    coordinates: np.ndarray,
    image_shape: tuple[int, int],
    *,
    reference_region: tuple[int, int, int, int] | None = None,
    reference_mask: np.ndarray | None = None,
    neighbor_k: int = 10,
    direction_tolerance_deg: float = 18.0,
    interpolation_grid_step: int = 1,
    map_inline_mode: str = "preview",
) -> StrainResult:
    """Run atom-position-based strain mapping and return a StrainResult contract."""
    fields = compute_atom_strain_fields(
        coordinates=coordinates,
        image_shape=image_shape,
        reference_region=reference_region,
        reference_mask=reference_mask,
        neighbor_k=neighbor_k,
        direction_tolerance_deg=direction_tolerance_deg,
        interpolation_grid_step=interpolation_grid_step,
    )
    return fields.to_strain_result(map_inline_mode=map_inline_mode)


def compute_atom_strain_fields(
    coordinates: np.ndarray,
    image_shape: tuple[int, int],
    *,
    reference_region: tuple[int, int, int, int] | None = None,
    reference_mask: np.ndarray | None = None,
    neighbor_k: int = 10,
    direction_tolerance_deg: float = 18.0,
    interpolation_grid_step: int = 1,
) -> AtomStrainFieldResult:
    """Estimate local affine strain from atom-column coordinates."""
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")
    if len(image_shape) != 2:
        raise ValueError("image_shape must be a 2D (height, width) tuple.")
    if neighbor_k < 4:
        raise ValueError("neighbor_k must be >= 4.")
    if direction_tolerance_deg <= 0.0 or direction_tolerance_deg >= 90.0:
        raise ValueError("direction_tolerance_deg must be in (0, 90).")
    if interpolation_grid_step < 1:
        raise ValueError("interpolation_grid_step must be >= 1.")

    warnings: list[str] = []
    ref_region = _normalize_reference_region(reference_region, image_shape)
    ref_mask_atoms, ref_region = _select_reference_atoms(
        coords=coords,
        image_shape=image_shape,
        reference_region=ref_region,
        reference_mask=reference_mask,
        warnings=warnings,
    )

    if len(coords) < 6:
        warnings.append("Atom strain skipped: need at least 6 atom positions.")
        return _empty_result(coords, image_shape, ref_mask_atoms, ref_region, warnings)

    ref_vectors_xy, ref_meta = _estimate_reference_basis(
        coords=coords,
        ref_atom_mask=ref_mask_atoms,
        neighbor_k=neighbor_k,
        direction_tolerance_deg=direction_tolerance_deg,
        warnings=warnings,
    )
    if ref_vectors_xy is None:
        return _empty_result(coords, image_shape, ref_mask_atoms, ref_region, warnings)

    ref_center_xy = np.mean(coords[ref_mask_atoms][:, ::-1], axis=0)
    ref_basis = np.column_stack([ref_vectors_xy[0], ref_vectors_xy[1]])
    inv_ref_basis = np.linalg.inv(ref_basis)
    tol_rad = np.deg2rad(direction_tolerance_deg)
    ref_lengths = np.linalg.norm(ref_vectors_xy, axis=1)

    points_xy = coords[:, ::-1]
    tree = cKDTree(points_xy)

    ux_atom = np.full(len(coords), np.nan, dtype=np.float64)
    uy_atom = np.full(len(coords), np.nan, dtype=np.float64)
    exx_atom = np.full(len(coords), np.nan, dtype=np.float64)
    eyy_atom = np.full(len(coords), np.nan, dtype=np.float64)
    exy_atom = np.full(len(coords), np.nan, dtype=np.float64)
    rot_atom = np.full(len(coords), np.nan, dtype=np.float64)
    unc_atom = np.full(len(coords), np.nan, dtype=np.float64)
    valid_atom_mask = np.zeros(len(coords), dtype=bool)

    local_residuals: list[float] = []
    cond_values: list[float] = []

    for idx in range(len(coords)):
        obs_basis, quality = _estimate_local_basis(
            tree=tree,
            points_xy=points_xy,
            center_idx=idx,
            ref_vectors_xy=ref_vectors_xy,
            ref_lengths=ref_lengths,
            tolerance_rad=tol_rad,
            neighbor_k=neighbor_k,
        )
        if obs_basis is None:
            continue
        try:
            deformation = obs_basis @ inv_ref_basis
        except np.linalg.LinAlgError:
            continue

        strain = 0.5 * (deformation + deformation.T) - np.eye(2, dtype=np.float64)
        rel_xy = points_xy[idx] - ref_center_xy
        disp_xy = (deformation - np.eye(2, dtype=np.float64)) @ rel_xy

        ux_atom[idx] = float(disp_xy[0])
        uy_atom[idx] = float(disp_xy[1])
        exx_atom[idx] = float(strain[0, 0])
        eyy_atom[idx] = float(strain[1, 1])
        exy_atom[idx] = float(strain[0, 1])
        rot_atom[idx] = float(0.5 * (deformation[1, 0] - deformation[0, 1]))
        cond_penalty = max(float(np.linalg.cond(obs_basis)) - 1.0, 0.0)
        unc_atom[idx] = float(
            quality["residual"] / max(np.mean(ref_lengths), 1e-6) + 0.02 * cond_penalty
        )
        valid_atom_mask[idx] = True
        local_residuals.append(float(quality["residual"]))
        cond_values.append(float(np.linalg.cond(obs_basis)))

    if not np.any(valid_atom_mask):
        warnings.append(
            "No atoms produced a valid local affine fit. "
            "Check atom editing/reference ROI."
        )
        return _empty_result(coords, image_shape, ref_mask_atoms, ref_region, warnings)

    ux_fill = _fill_nan(ux_atom)
    uy_fill = _fill_nan(uy_atom)
    exx_fill = _fill_nan(exx_atom)
    eyy_fill = _fill_nan(eyy_atom)
    exy_fill = _fill_nan(exy_atom)
    rot_fill = _fill_nan(rot_atom)
    unc_fill = _fill_nan(unc_atom, fill_value=float(np.nanmedian(unc_atom[valid_atom_mask])))

    disp_x_map = _interpolate_atom_values(
        points_xy=points_xy,
        values=ux_fill,
        image_shape=image_shape,
        grid_step=interpolation_grid_step,
    )
    disp_y_map = _interpolate_atom_values(
        points_xy=points_xy,
        values=uy_fill,
        image_shape=image_shape,
        grid_step=interpolation_grid_step,
    )
    strain_xx_map = _interpolate_atom_values(
        points_xy=points_xy,
        values=exx_fill,
        image_shape=image_shape,
        grid_step=interpolation_grid_step,
    )
    strain_yy_map = _interpolate_atom_values(
        points_xy=points_xy,
        values=eyy_fill,
        image_shape=image_shape,
        grid_step=interpolation_grid_step,
    )
    strain_xy_map = _interpolate_atom_values(
        points_xy=points_xy,
        values=exy_fill,
        image_shape=image_shape,
        grid_step=interpolation_grid_step,
    )
    rotation_xy_map = _interpolate_atom_values(
        points_xy=points_xy,
        values=rot_fill,
        image_shape=image_shape,
        grid_step=interpolation_grid_step,
    )
    uncertainty_map = _interpolate_atom_values(
        points_xy=points_xy,
        values=unc_fill,
        image_shape=image_shape,
        grid_step=interpolation_grid_step,
    )

    valid_ratio = float(np.mean(valid_atom_mask))
    residual_median = float(np.median(local_residuals)) if local_residuals else 0.0
    cond_median = float(np.median(cond_values)) if cond_values else 1.0
    confidence = float(
        np.clip(
            0.55 * valid_ratio
            + 0.25 * np.clip(1.0 - residual_median / max(np.mean(ref_lengths), 1e-6), 0.0, 1.0)
            + 0.20 * np.clip(1.0 / max(cond_median, 1.0), 0.0, 1.0),
            0.0,
            1.0,
        )
    )

    provenance = {
        "source": "autotem.strain.atom_strain.compute_atom_strain_fields",
        "method": "atom_ai_stem",
        "parameters": {
            "neighbor_k": int(neighbor_k),
            "direction_tolerance_deg": float(direction_tolerance_deg),
            "interpolation_grid_step": int(interpolation_grid_step),
        },
        "reference_atom_count": int(np.sum(ref_mask_atoms)),
        "reference_vectors_xy": ref_vectors_xy.tolist(),
        "reference_center_xy": ref_center_xy.tolist(),
        "reference_region": ref_region,
        "reference_metadata": ref_meta,
        "fit_statistics": {
            "valid_atom_ratio": valid_ratio,
            "median_local_residual_px": residual_median,
            "median_condition_number": cond_median,
        },
    }

    return AtomStrainFieldResult(
        coordinates_yx=coords,
        atom_displacement_xy=np.column_stack([ux_fill, uy_fill]).astype(np.float32),
        atom_strain_xx=exx_fill.astype(np.float32),
        atom_strain_yy=eyy_fill.astype(np.float32),
        atom_strain_xy=exy_fill.astype(np.float32),
        atom_rotation_xy=rot_fill.astype(np.float32),
        atom_uncertainty=unc_fill.astype(np.float32),
        valid_atom_mask=valid_atom_mask,
        displacement_x_map=disp_x_map.astype(np.float32),
        displacement_y_map=disp_y_map.astype(np.float32),
        strain_xx_map=strain_xx_map.astype(np.float32),
        strain_yy_map=strain_yy_map.astype(np.float32),
        strain_xy_map=strain_xy_map.astype(np.float32),
        rotation_xy_map=rotation_xy_map.astype(np.float32),
        uncertainty_map=uncertainty_map.astype(np.float32),
        reference_vectors_xy=ref_vectors_xy.astype(np.float32),
        reference_center_xy=ref_center_xy.astype(np.float32),
        reference_atom_mask=ref_mask_atoms,
        reference_region=ref_region,
        confidence=confidence,
        warnings=warnings,
        provenance=provenance,
    )


def _empty_result(
    coords: np.ndarray,
    image_shape: tuple[int, int],
    ref_atom_mask: np.ndarray,
    ref_region: tuple[int, int, int, int] | None,
    warnings: list[str],
) -> AtomStrainFieldResult:
    h, w = int(image_shape[0]), int(image_shape[1])
    zeros_atom = np.zeros(len(coords), dtype=np.float32)
    zeros_map = np.zeros((h, w), dtype=np.float32)
    return AtomStrainFieldResult(
        coordinates_yx=coords,
        atom_displacement_xy=np.zeros((len(coords), 2), dtype=np.float32),
        atom_strain_xx=zeros_atom.copy(),
        atom_strain_yy=zeros_atom.copy(),
        atom_strain_xy=zeros_atom.copy(),
        atom_rotation_xy=zeros_atom.copy(),
        atom_uncertainty=zeros_atom.copy(),
        valid_atom_mask=np.zeros(len(coords), dtype=bool),
        displacement_x_map=zeros_map.copy(),
        displacement_y_map=zeros_map.copy(),
        strain_xx_map=zeros_map.copy(),
        strain_yy_map=zeros_map.copy(),
        strain_xy_map=zeros_map.copy(),
        rotation_xy_map=zeros_map.copy(),
        uncertainty_map=zeros_map.copy(),
        reference_vectors_xy=np.zeros((2, 2), dtype=np.float32),
        reference_center_xy=np.zeros(2, dtype=np.float32),
        reference_atom_mask=ref_atom_mask.copy(),
        reference_region=ref_region,
        confidence=0.0,
        warnings=list(warnings),
        provenance={
            "source": "autotem.strain.atom_strain.compute_atom_strain_fields",
            "method": "atom_ai_stem",
            "parameters": {},
            "fit_statistics": {"valid_atom_ratio": 0.0},
        },
    )


def _normalize_reference_region(
    reference_region: tuple[int, int, int, int] | None,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if reference_region is None:
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    y0, y1, x0, x1 = [int(v) for v in reference_region]
    y0 = int(np.clip(y0, 0, h))
    y1 = int(np.clip(y1, 0, h))
    x0 = int(np.clip(x0, 0, w))
    x1 = int(np.clip(x1, 0, w))
    if y1 <= y0 or x1 <= x0:
        return None
    return y0, y1, x0, x1


def _select_reference_atoms(
    coords: np.ndarray,
    image_shape: tuple[int, int],
    reference_region: tuple[int, int, int, int] | None,
    reference_mask: np.ndarray | None,
    warnings: list[str],
) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    if reference_mask is not None:
        mask = np.asarray(reference_mask, dtype=bool)
        if mask.shape != tuple(int(v) for v in image_shape):
            raise ValueError("reference_mask must have same shape as image_shape.")
        yi = np.clip(np.round(coords[:, 0]).astype(int), 0, mask.shape[0] - 1)
        xi = np.clip(np.round(coords[:, 1]).astype(int), 0, mask.shape[1] - 1)
        atom_mask = mask[yi, xi]
        bbox = _mask_bbox(mask)
        if int(np.sum(atom_mask)) < 6:
            warnings.append(
                "Reference ROI has too few atoms; "
                "falling back to full edited atom set."
            )
        else:
            return atom_mask, bbox

    if reference_region is not None:
        y0, y1, x0, x1 = reference_region
        atom_mask = (
            (coords[:, 0] >= y0)
            & (coords[:, 0] < y1)
            & (coords[:, 1] >= x0)
            & (coords[:, 1] < x1)
        )
        if int(np.sum(atom_mask)) < 6:
            warnings.append(
                "Reference region has too few atoms; "
                "falling back to full edited atom set."
            )
        else:
            return atom_mask, reference_region

    return np.ones(len(coords), dtype=bool), reference_region


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(np.min(ys)), int(np.max(ys)) + 1, int(np.min(xs)), int(np.max(xs)) + 1


def _estimate_reference_basis(
    coords: np.ndarray,
    ref_atom_mask: np.ndarray,
    neighbor_k: int,
    direction_tolerance_deg: float,
    warnings: list[str],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    points_xy = coords[:, ::-1]
    ref_points_xy = points_xy[ref_atom_mask]
    if len(ref_points_xy) < 6:
        warnings.append("Too few reference atoms for basis estimation.")
        return None, {}

    tree = cKDTree(ref_points_xy)
    query_k = min(len(ref_points_xy), max(6, neighbor_k + 1))
    dists, idxs = tree.query(ref_points_xy, k=query_k)
    dists = np.atleast_2d(dists)
    idxs = np.atleast_2d(idxs)

    vectors: list[np.ndarray] = []
    nearest_shell: list[float] = []
    for row_idx in range(len(ref_points_xy)):
        nn = np.atleast_1d(idxs[row_idx])
        dd = np.atleast_1d(dists[row_idx])
        for dist, col_idx in zip(dd[1:], nn[1:], strict=False):
            if not np.isfinite(dist) or dist <= 1e-8:
                continue
            vec = ref_points_xy[int(col_idx)] - ref_points_xy[row_idx]
            vectors.append(vec)
        finite_nn = dd[1:][np.isfinite(dd[1:]) & (dd[1:] > 1e-8)]
        if finite_nn.size:
            nearest_shell.append(float(np.min(finite_nn)))

    if not vectors or not nearest_shell:
        warnings.append("Unable to estimate lattice basis from reference atoms.")
        return None, {}

    vectors_arr = np.asarray(vectors, dtype=np.float64)
    lengths = np.linalg.norm(vectors_arr, axis=1)
    nn_median = float(np.median(nearest_shell))
    shell_mask = (lengths >= 0.55 * nn_median) & (lengths <= 1.8 * nn_median)
    if np.sum(shell_mask) < 12:
        shell_mask = lengths <= 2.2 * nn_median

    shell_vecs = vectors_arr[shell_mask]
    shell_lengths = lengths[shell_mask]
    canon = shell_vecs.copy()
    flip = (canon[:, 0] < 0.0) | ((np.abs(canon[:, 0]) < 1e-9) & (canon[:, 1] < 0.0))
    canon[flip] *= -1.0
    angles = np.mod(np.arctan2(canon[:, 1], canon[:, 0]), np.pi)
    weights = 1.0 / np.clip(shell_lengths, 1e-6, None)

    bins = 90
    hist, edges = np.histogram(angles, bins=bins, range=(0.0, np.pi), weights=weights)
    hist = np.convolve(hist, np.array([0.25, 0.5, 0.25], dtype=np.float64), mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])

    first_idx = int(np.argmax(hist))
    theta1 = float(centers[first_idx])
    delta = np.abs(((centers - theta1 + (np.pi / 2.0)) % np.pi) - (np.pi / 2.0))
    candidate_score = hist * np.sin(delta) ** 2
    candidate_score[delta < np.deg2rad(22.0)] = -np.inf
    second_idx = int(np.argmax(candidate_score))
    theta2 = float(centers[second_idx])

    tol = np.deg2rad(direction_tolerance_deg)
    vec1 = _robust_direction_vector(canon, shell_lengths, theta1, tol)
    vec2 = _robust_direction_vector(canon, shell_lengths, theta2, tol)
    if vec1 is None or vec2 is None:
        warnings.append("Failed to recover two non-collinear reference directions.")
        return None, {}

    basis = np.column_stack([vec1, vec2])
    det = float(np.linalg.det(basis))
    if abs(det) < 1e-6:
        warnings.append("Reference basis is nearly singular; atom strain not reliable.")
        return None, {}

    return (
        np.vstack([vec1, vec2]),
        {
            "nearest_shell_median_px": nn_median,
            "angle_theta1_deg": float(np.rad2deg(theta1)),
            "angle_theta2_deg": float(np.rad2deg(theta2)),
            "basis_determinant": det,
        },
    )


def _robust_direction_vector(
    canon_vectors_xy: np.ndarray,
    lengths: np.ndarray,
    theta: float,
    tolerance_rad: float,
) -> np.ndarray | None:
    if len(canon_vectors_xy) == 0:
        return None
    delta = _angular_distance_pi(
        np.mod(np.arctan2(canon_vectors_xy[:, 1], canon_vectors_xy[:, 0]), np.pi),
        theta,
    )
    mask = delta <= tolerance_rad
    if int(np.sum(mask)) < 3:
        return None
    selected = canon_vectors_xy[mask]
    selected_lengths = lengths[mask]
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    proj = selected @ u
    proj = np.abs(proj)
    length = (
        float(np.median(proj[proj > 1e-8]))
        if np.any(proj > 1e-8)
        else float(np.median(selected_lengths))
    )
    if length <= 1e-8:
        return None
    return length * u


def _estimate_local_basis(
    tree: cKDTree,
    points_xy: np.ndarray,
    center_idx: int,
    ref_vectors_xy: np.ndarray,
    ref_lengths: np.ndarray,
    tolerance_rad: float,
    neighbor_k: int,
) -> tuple[np.ndarray | None, dict[str, float]]:
    query_k = min(len(points_xy), max(6, neighbor_k + 1))
    dists, idxs = tree.query(points_xy[center_idx], k=query_k)
    idxs = np.atleast_1d(idxs)
    vecs_xy = points_xy[idxs[1:]] - points_xy[center_idx]
    if len(vecs_xy) < 4:
        return None, {}

    obs_vecs: list[np.ndarray] = []
    residuals: list[float] = []
    used_neighbor_ids: set[int] = set()

    for ref_vec, ref_len in zip(ref_vectors_xy, ref_lengths, strict=False):
        pos = _pick_direction_neighbor(
            vecs_xy=vecs_xy,
            ref_vec_xy=ref_vec,
            ref_length=ref_len,
            tolerance_rad=tolerance_rad,
            sign=+1,
            used_neighbor_ids=used_neighbor_ids,
        )
        neg = _pick_direction_neighbor(
            vecs_xy=vecs_xy,
            ref_vec_xy=ref_vec,
            ref_length=ref_len,
            tolerance_rad=tolerance_rad,
            sign=-1,
            used_neighbor_ids=used_neighbor_ids,
        )

        if pos is None and neg is None:
            return None, {}
        if pos is not None and neg is not None:
            obs_vec = 0.5 * (pos["vector"] - neg["vector"])
            residuals.append(0.5 * (pos["score"] + neg["score"]))
            used_neighbor_ids.add(int(pos["local_index"]))
            used_neighbor_ids.add(int(neg["local_index"]))
        elif pos is not None:
            obs_vec = pos["vector"]
            residuals.append(pos["score"])
            used_neighbor_ids.add(int(pos["local_index"]))
        else:
            obs_vec = -neg["vector"]
            residuals.append(neg["score"])
            used_neighbor_ids.add(int(neg["local_index"]))
        obs_vecs.append(obs_vec)

    if len(obs_vecs) != 2:
        return None, {}
    obs_basis = np.column_stack(obs_vecs)
    if abs(float(np.linalg.det(obs_basis))) < 1e-6:
        return None, {}
    return obs_basis, {"residual": float(np.mean(residuals)) if residuals else 0.0}


def _pick_direction_neighbor(
    vecs_xy: np.ndarray,
    ref_vec_xy: np.ndarray,
    ref_length: float,
    tolerance_rad: float,
    sign: int,
    used_neighbor_ids: set[int],
) -> dict[str, Any] | None:
    u = ref_vec_xy / max(np.linalg.norm(ref_vec_xy), 1e-8)
    proj = vecs_xy @ u
    norms = np.linalg.norm(vecs_xy, axis=1)
    perp = np.abs(vecs_xy[:, 0] * u[1] - vecs_xy[:, 1] * u[0])
    cosang = np.clip(np.abs(proj) / np.clip(norms, 1e-8, None), -1.0, 1.0)
    ang = np.arccos(cosang)
    if sign > 0:
        mask = proj > 0.0
    else:
        mask = proj < 0.0
    mask &= ang <= tolerance_rad
    if used_neighbor_ids:
        used_mask = np.ones(len(vecs_xy), dtype=bool)
        for idx in used_neighbor_ids:
            if 0 <= idx < len(used_mask):
                used_mask[idx] = False
        mask &= used_mask
    if not np.any(mask):
        return None
    score = perp + 0.35 * np.abs(np.abs(proj) - ref_length) + 0.15 * np.abs(norms - ref_length)
    masked_idx = np.where(mask)[0]
    local_best = int(masked_idx[np.argmin(score[mask])])
    return {
        "local_index": local_best,
        "vector": vecs_xy[local_best],
        "score": float(score[local_best]),
    }


def _fill_nan(values: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if not np.any(np.isfinite(arr)):
        return np.full_like(arr, fill_value, dtype=np.float64)
    fill = float(fill_value if np.isfinite(fill_value) else np.nanmedian(arr[np.isfinite(arr)]))
    arr[~np.isfinite(arr)] = fill
    return arr


def _interpolate_atom_values(
    points_xy: np.ndarray,
    values: np.ndarray,
    image_shape: tuple[int, int],
    grid_step: int,
) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    grid_y = np.arange(0, h, grid_step, dtype=np.float64)
    grid_x = np.arange(0, w, grid_step, dtype=np.float64)
    if grid_y[-1] != h - 1:
        grid_y = np.append(grid_y, h - 1.0)
    if grid_x[-1] != w - 1:
        grid_x = np.append(grid_x, w - 1.0)
    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    linear = griddata(points_xy, values, (mesh_x, mesh_y), method="linear")
    nearest = griddata(points_xy, values, (mesh_x, mesh_y), method="nearest")
    coarse = np.where(np.isfinite(linear), linear, nearest)
    if grid_step == 1:
        return np.asarray(coarse, dtype=np.float32)

    zoom_y = h / coarse.shape[0]
    zoom_x = w / coarse.shape[1]
    full = ndi.zoom(coarse, (zoom_y, zoom_x), order=1)
    full = full[:h, :w]
    if full.shape != (h, w):
        fixed = np.zeros((h, w), dtype=np.float64)
        fixed[: full.shape[0], : full.shape[1]] = full
        if full.shape[0] < h:
            fixed[full.shape[0] :, :] = fixed[full.shape[0] - 1 : full.shape[0], :]
        if full.shape[1] < w:
            fixed[:, full.shape[1] :] = fixed[:, full.shape[1] - 1 : full.shape[1]]
        full = fixed
    return np.asarray(full, dtype=np.float32)


def _to_array_ref(
    arr: np.ndarray,
    *,
    description: str,
    inline_mode: str,
) -> ArrayReference:
    arr = np.asarray(arr, dtype=np.float32)
    if inline_mode == "full":
        inline = arr.tolist()
    else:
        inline = arr[: min(8, arr.shape[0]), : min(8, arr.shape[1])].tolist()
    return ArrayReference(
        shape=tuple(int(v) for v in arr.shape),
        dtype=str(arr.dtype),
        uri=None,
        inline=inline,
        description=description,
    )


def _angular_distance_pi(theta: np.ndarray, center: float) -> np.ndarray:
    return np.abs(((theta - center + (np.pi / 2.0)) % np.pi) - (np.pi / 2.0))
