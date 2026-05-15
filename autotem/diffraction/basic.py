"""A2/B2 FFT, robust peak extraction, indexing, and phase matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import curve_fit
from scipy.signal import windows as sig_windows

from autotem.core import (
    ArrayReference,
    CalibrationResult,
    DiffractionPeak,
    DiffractionResult,
    GVector,
    IndexedSpot,
    Measurement,
    PhaseCandidate,
    PreprocessingResult,
)
from autotem.diffraction.b4_indexing import run_b4_indexing


@dataclass(slots=True)
class _PhaseReference:
    name: str
    formula: str
    identifier: str
    d_refs_nm: tuple[float, ...]


_PHASE_DB: tuple[_PhaseReference, ...] = (
    _PhaseReference("Silicon", "Si", "mp-149", (0.313, 0.192, 0.163, 0.136)),
    _PhaseReference("Gold", "Au", "mp-81", (0.235, 0.204, 0.144, 0.123)),
    _PhaseReference("Graphite", "C", "mp-48", (0.335, 0.213, 0.168, 0.142)),
    _PhaseReference("Titanium Dioxide", "TiO2", "mp-2657", (0.325, 0.249, 0.190, 0.169)),
)


def _load_preprocessed_array(preprocessing: PreprocessingResult) -> np.ndarray | None:
    uri = preprocessing.image.uri if preprocessing.image is not None else None
    if uri is None:
        return None
    path = Path(uri)
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float32)
    if path.suffix.lower() in {".txt", ".csv"}:
        return np.loadtxt(path).astype(np.float32)
    return None


def _make_index(dx: float, dy: float) -> tuple[int, int, int]:
    denom = max(abs(dx), abs(dy), 1.0)
    h = int(np.rint(dx / denom * 4))
    k = int(np.rint(dy / denom * 4))
    return h, k, 0


def _compute_peak_threshold(
    image: np.ndarray,
    *,
    strategy: str,
    value: float | None,
) -> float:
    flat = image.ravel().astype(np.float64)
    if flat.size == 0:
        return 0.0
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median))) * 1.4826
    mean = float(np.mean(flat))
    std = float(np.std(flat))
    max_val = float(np.max(flat))
    quantile_995 = float(np.quantile(flat, 0.995))

    strategy_key = strategy.lower().strip()
    if strategy_key == "quantile":
        q = 0.995 if value is None else float(value)
        q = min(0.999999, max(0.0, q))
        return float(np.quantile(flat, q))
    if strategy_key == "std":
        z = 3.0 if value is None else float(value)
        z = max(0.0, z)
        return mean + z * std
    if strategy_key == "relative_max":
        r = 0.2 if value is None else float(value)
        r = min(1.0, max(0.0, r))
        return r * max_val
    if strategy_key == "mad":
        k = 6.0 if value is None else float(value)
        k = max(0.0, k)
        return median + k * mad
    return min(max_val, max(quantile_995, median + 6.0 * mad))


def _select_basis_indices(
    gxy: list[tuple[float, float]],
    intensities: list[float],
) -> tuple[int, int] | None:
    if len(gxy) < 2:
        return None
    ranked = sorted(
        range(len(gxy)),
        key=lambda idx: float(intensities[idx]),
        reverse=True,
    )[: min(12, len(gxy))]

    best_pair: tuple[int, int] | None = None
    best_score = -np.inf
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            idx1 = ranked[i]
            idx2 = ranked[j]
            v1 = np.array(gxy[idx1], dtype=np.float64)
            v2 = np.array(gxy[idx2], dtype=np.float64)
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 <= 1e-9 or n2 <= 1e-9:
                continue
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            cosang = max(-1.0, min(1.0, cosang))
            angle = float(np.arccos(cosang))
            if angle < np.deg2rad(10.0) or angle > np.deg2rad(170.0):
                continue
            bmat = np.column_stack((v1, v2))
            cond = float(np.linalg.cond(bmat))
            if not np.isfinite(cond) or cond > 1000:
                continue
            score = float(np.sin(angle) * np.sqrt(n1 * n2) / (1.0 + 0.02 * cond))
            if score > best_score:
                best_score = score
                best_pair = (idx1, idx2)
    return best_pair


def _index_spots_with_lattice_fit(
    gxy: list[tuple[float, float]],
    intensities: list[float],
) -> tuple[list[tuple[int, int, int, float, float]], dict[str, Any]]:
    basis_pair = _select_basis_indices(gxy, intensities)
    if basis_pair is None:
        return [], {"indexing_method": "lattice_fit_fallback"}

    i1, i2 = basis_pair
    basis_1 = np.array(gxy[i1], dtype=np.float64)
    basis_2 = np.array(gxy[i2], dtype=np.float64)
    bmat = np.column_stack((basis_1, basis_2))
    basis_scale = max(float(np.linalg.norm(basis_1)), float(np.linalg.norm(basis_2)), 1e-9)

    indexed: list[tuple[int, int, int, float, float]] = []
    for gx, gy in gxy:
        gvec = np.array([gx, gy], dtype=np.float64)
        coeff = np.linalg.solve(bmat, gvec)
        rounded = np.rint(coeff)
        h = int(rounded[0])
        k = int(rounded[1])
        recon = bmat @ rounded
        residual = float(np.linalg.norm(gvec - recon))
        coeff_err = float(np.linalg.norm(coeff - rounded))
        confidence = float(
            np.exp(-residual / (0.12 * basis_scale + 1e-9)) * np.exp(-coeff_err / 0.5)
        )
        confidence = max(0.05, min(0.95, confidence))
        indexed.append((h, k, 0, residual, confidence))

    metadata = {
        "indexing_method": "lattice_fit",
        "basis_indices": [i1, i2],
        "basis_vectors_1_per_nm": {
            "g1": [float(basis_1[0]), float(basis_1[1])],
            "g2": [float(basis_2[0]), float(basis_2[1])],
        },
    }
    return indexed, metadata


def _phase_candidates_from_d_spacings(
    values_nm: list[float],
    *,
    allowed_phase_names: set[str] | None = None,
    top_n: int = 3,
) -> list[PhaseCandidate]:
    if not values_nm:
        return []
    candidates: list[PhaseCandidate] = []
    for ref in _PHASE_DB:
        if allowed_phase_names is not None and ref.name not in allowed_phase_names:
            continue
        errors = []
        for d in values_nm[: min(8, len(values_nm))]:
            nearest = min(ref.d_refs_nm, key=lambda r: abs(r - d))
            errors.append(abs(nearest - d) / nearest)
        mean_rel_error = float(np.mean(errors))
        score = float(np.exp(-mean_rel_error * 4.0))
        candidates.append(
            PhaseCandidate(
                name=ref.name,
                formula=ref.formula,
                source="A2 baseline reference",
                identifier=ref.identifier,
                score=max(0.0, min(1.0, score)),
                matched_spot_count=min(len(values_nm), len(ref.d_refs_nm)),
                mean_d_error=Measurement(
                    value=mean_rel_error,
                    unit="relative",
                    method="nearest_ref",
                ),
                zone_axis=(0, 0, 1),
            )
        )
    candidates.sort(key=lambda c: c.score or 0.0, reverse=True)
    return candidates[: max(1, top_n)]


def _drop_origin_like_spots(
    *,
    peaks: list[DiffractionPeak],
    g_vectors: list[GVector],
    indexed_spots: list[IndexedSpot],
    d_by_peak: list[Measurement | None],
    peak_intensities: list[float],
    peak_details: list[dict[str, Any]],
    g_norm_threshold: float,
    hkl_origin_g_max: float,
    center_xy: tuple[float, float],
    origin_filter_max_radius_px: float,
) -> tuple[
    list[DiffractionPeak],
    list[GVector],
    list[IndexedSpot],
    list[Measurement | None],
    list[float],
    list[dict[str, Any]],
    int,
]:
    idx_map = {int(item.peak_index): item for item in indexed_spots}
    keep_old_indices: list[int] = []
    removed = 0
    for i, _peak in enumerate(peaks):
        r_px = float(np.hypot(float(peaks[i].x) - center_xy[0], float(peaks[i].y) - center_xy[1]))
        g_mag = 0.0
        if i < len(g_vectors) and g_vectors[i].magnitude is not None:
            try:
                g_mag = float(g_vectors[i].magnitude.value or 0.0)
            except (TypeError, ValueError):
                g_mag = 0.0
        indexed = idx_map.get(i)
        is_origin_hkl = False
        if indexed is not None:
            is_origin_hkl = int(indexed.h) == 0 and int(indexed.k) == 0 and int(indexed.l) == 0
        near_center = r_px <= max(0.0, origin_filter_max_radius_px)
        remove_by_hkl_origin = near_center and is_origin_hkl and g_mag <= hkl_origin_g_max
        remove_by_small_g = near_center and g_mag <= g_norm_threshold
        if remove_by_hkl_origin or remove_by_small_g:
            removed += 1
            continue
        keep_old_indices.append(i)

    if removed <= 0:
        return peaks, g_vectors, indexed_spots, d_by_peak, peak_intensities, peak_details, 0

    remap = {old: new for new, old in enumerate(keep_old_indices)}
    new_peaks = [peaks[i] for i in keep_old_indices]
    new_g = []
    for old in keep_old_indices:
        vec = g_vectors[old]
        new_g.append(vec.model_copy(update={"source_peak_index": remap[old]}))
    new_idx: list[IndexedSpot] = []
    for old in keep_old_indices:
        row = idx_map.get(old)
        if row is None:
            continue
        new_idx.append(row.model_copy(update={"peak_index": remap[old]}))
    new_d = [d_by_peak[i] for i in keep_old_indices]
    new_intensities = [peak_intensities[i] for i in keep_old_indices]
    detail_by_old: dict[int, dict[str, Any]] = {}
    for row in peak_details:
        try:
            detail_by_old[int(row.get("peak_id", -1))] = dict(row)
        except (TypeError, ValueError):
            continue
    new_details: list[dict[str, Any]] = []
    for old in keep_old_indices:
        row = detail_by_old.get(old, {})
        row["peak_id"] = remap[old]
        new_details.append(row)
    return new_peaks, new_g, new_idx, new_d, new_intensities, new_details, removed


def _compress_fft(magnitude: np.ndarray, method: str) -> np.ndarray:
    key = str(method).strip().lower()
    if key == "asinh":
        return np.arcsinh(np.maximum(magnitude, 0.0)).astype(np.float32)
    return np.log1p(np.maximum(magnitude, 0.0)).astype(np.float32)


def _percentile_clip(
    image: np.ndarray,
    *,
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, float, float]:
    low = float(np.clip(low_percentile, 0.0, 99.9))
    high = float(np.clip(high_percentile, max(low + 1e-6, 0.001), 100.0))
    p_low = float(np.percentile(image, low))
    p_high = float(np.percentile(image, high))
    if p_high <= p_low:
        return image.astype(np.float32), p_low, p_high
    clipped = np.clip(image, p_low, p_high)
    return clipped.astype(np.float32), p_low, p_high


def _fft_window(shape: tuple[int, int], mode: str, *, tukey_alpha: float) -> np.ndarray:
    h, w = shape
    key = str(mode).strip().lower()
    if key in {"none", "", "off"}:
        return np.ones((h, w), dtype=np.float32)
    if key == "hann":
        wy = np.hanning(h)
        wx = np.hanning(w)
    elif key == "tukey":
        alpha = float(np.clip(tukey_alpha, 0.0, 1.0))
        wy = sig_windows.tukey(h, alpha=alpha)
        wx = sig_windows.tukey(w, alpha=alpha)
    else:
        wy = np.ones(h, dtype=np.float32)
        wx = np.ones(w, dtype=np.float32)
    return np.outer(wy, wx).astype(np.float32)


def _build_masks(
    shape: tuple[int, int],
    *,
    center_xy: tuple[float, float],
    center_mask_radius: float,
    axis_streak_mask_width: float,
    border_mask: float,
) -> dict[str, np.ndarray]:
    h, w = shape
    cy, cx = center_xy
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)

    center_mask = rr <= max(0.0, float(center_mask_radius))
    axis_w = max(0.0, float(axis_streak_mask_width))
    if axis_w <= 0:
        axis_mask = np.zeros_like(center_mask, dtype=bool)
    else:
        axis_mask = (np.abs(xx - cx) <= axis_w) | (np.abs(yy - cy) <= axis_w)
    border = max(0.0, float(border_mask))
    border_mask_arr = (
        (xx < border) | (xx >= (w - border)) | (yy < border) | (yy >= (h - border))
    )
    combined = center_mask | axis_mask | border_mask_arr
    return {
        "center_mask": center_mask,
        "axis_mask": axis_mask,
        "border_mask": border_mask_arr,
        "combined_mask": combined,
        "valid_mask": ~combined,
    }


def _radial_background_stats(
    image: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radial_bin_width: float,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image.shape
    cy, cx = center_xy
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    bw = max(0.5, float(radial_bin_width))
    bin_idx = np.floor(rr / bw).astype(np.int32)
    n_bins = int(np.max(bin_idx)) + 1

    if valid_mask is None:
        valid_mask = np.ones_like(image, dtype=bool)

    global_vals = image[valid_mask]
    if global_vals.size == 0:
        global_vals = image.ravel()
    global_median = float(np.median(global_vals))
    global_mad = float(np.median(np.abs(global_vals - global_median)))
    global_noise = max(global_mad * 1.4826, 1e-6)

    medians = np.full(n_bins, global_median, dtype=np.float32)
    mads = np.full(n_bins, global_mad, dtype=np.float32)
    for b in range(n_bins):
        mask = (bin_idx == b) & valid_mask
        if not np.any(mask):
            continue
        vals = image[mask]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        medians[b] = med
        mads[b] = max(mad, 1e-6)

    radial_median = medians[bin_idx]
    radial_noise = np.maximum(mads[bin_idx] * 1.4826, global_noise).astype(np.float32)
    z_map = ((image - radial_median) / radial_noise).astype(np.float32)
    return radial_median.astype(np.float32), radial_noise, z_map


def _local_maxima_candidates(
    score_map: np.ndarray,
    *,
    valid_mask: np.ndarray,
    threshold: float,
    min_distance: float,
    source: str,
) -> list[dict[str, float | str]]:
    radius = max(1, int(round(min_distance)))
    size = int(2 * radius + 1)
    local_max = score_map >= ndi.maximum_filter(score_map, size=size, mode="nearest")
    cand_mask = local_max & valid_mask & (score_map >= float(threshold))
    ys, xs = np.nonzero(cand_mask)
    out: list[dict[str, float | str]] = []
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        out.append(
            {
                "y": float(y),
                "x": float(x),
                "score": float(score_map[y, x]),
                "source": source,
            }
        )
    out.sort(key=lambda item: float(item["score"]), reverse=True)
    return out


def _log_blob_candidates(
    score_map: np.ndarray,
    *,
    valid_mask: np.ndarray,
    threshold: float | None,
    min_distance: float,
) -> list[dict[str, float | str]]:
    sigmas = np.linspace(1.0, 4.5, 6)
    resp_best = np.full(score_map.shape, -np.inf, dtype=np.float32)
    sigma_best = np.full(score_map.shape, sigmas[0], dtype=np.float32)
    for sigma in sigmas:
        resp = -ndi.gaussian_laplace(score_map, sigma=float(sigma))
        better = resp > resp_best
        resp_best[better] = resp[better]
        sigma_best[better] = float(sigma)
    if threshold is None:
        threshold = float(np.quantile(resp_best[valid_mask], 0.997))
    base = _local_maxima_candidates(
        resp_best,
        valid_mask=valid_mask,
        threshold=float(threshold),
        min_distance=min_distance,
        source="log_blob",
    )
    for row in base:
        y = int(round(float(row["y"])))
        x = int(round(float(row["x"])))
        row["sigma"] = float(sigma_best[y, x])
    return base


def _gaussian_2d(
    coords: tuple[np.ndarray, np.ndarray],
    amp: float,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    bg: float,
) -> np.ndarray:
    xx, yy = coords
    sx_safe = max(1e-6, sx)
    sy_safe = max(1e-6, sy)
    expo = ((xx - x0) ** 2) / (2.0 * sx_safe * sx_safe) + ((yy - y0) ** 2) / (
        2.0 * sy_safe * sy_safe
    )
    return (bg + amp * np.exp(-expo)).ravel()


def _subpixel_com(
    patch: np.ndarray,
    *,
    y0: int,
    x0: int,
) -> tuple[float, float]:
    weights = patch.astype(np.float64)
    baseline = float(np.percentile(weights, 10.0))
    weights = np.clip(weights - baseline, 0.0, None)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return float(y0 + patch.shape[0] // 2), float(x0 + patch.shape[1] // 2)
    yy, xx = np.mgrid[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
    y_refined = float(np.sum(yy * weights) / total)
    x_refined = float(np.sum(xx * weights) / total)
    return y_refined, x_refined


def _refine_peak_subpixel_v2(
    image: np.ndarray,
    *,
    y: float,
    x: float,
    window_radius: int,
    max_shift: float,
    prefer_gaussian: bool = True,
) -> tuple[float, float, str, float, str | None]:
    yi = int(round(y))
    xi = int(round(x))
    h, w = image.shape
    y0 = max(0, yi - window_radius)
    y1 = min(h, yi + window_radius + 1)
    x0 = max(0, xi - window_radius)
    x1 = min(w, xi + window_radius + 1)
    patch = image[y0:y1, x0:x1].astype(np.float64)
    if patch.size == 0:
        return float(y), float(x), "none", 0.0, "empty_patch"

    method = "gaussian_2d"
    warning: str | None = None
    y_refined = float(y)
    x_refined = float(x)

    try:
        if not prefer_gaussian:
            raise RuntimeError("skip_gaussian")
        yy, xx = np.mgrid[y0:y1, x0:x1]
        amp0 = max(float(np.max(patch) - np.min(patch)), 1e-3)
        bg0 = float(np.percentile(patch, 20.0))
        p0 = (
            amp0,
            float(x),
            float(y),
            max(0.8, window_radius / 1.5),
            max(0.8, window_radius / 1.5),
            bg0,
        )
        lower = (
            0.0,
            float(x0),
            float(y0),
            0.3,
            0.3,
            float(np.min(patch) - amp0),
        )
        upper = (
            float(np.max(patch) * 5.0 + 1e-6),
            float(x1 - 1),
            float(y1 - 1),
            float(window_radius * 2 + 2),
            float(window_radius * 2 + 2),
            float(np.max(patch) + amp0),
        )
        popt, _ = curve_fit(
            _gaussian_2d,
            (xx.astype(np.float64), yy.astype(np.float64)),
            patch.ravel(),
            p0=p0,
            bounds=(lower, upper),
            maxfev=3000,
        )
        x_refined = float(popt[1])
        y_refined = float(popt[2])
    except Exception:
        method = "center_of_mass"
        y_refined, x_refined = _subpixel_com(patch, y0=y0, x0=x0)

    shift = float(np.hypot(y_refined - y, x_refined - x))
    if shift > max_shift:
        warning = f"subpixel_shift_exceeds_max({shift:.3f}>{max_shift:.3f})"
        if shift > 1e-9 and max_shift > 0:
            scale = max_shift / shift
            y_refined = float(y + (y_refined - y) * scale)
            x_refined = float(x + (x_refined - x) * scale)
            shift = float(max_shift)
        else:
            y_refined = float(y)
            x_refined = float(x)
            shift = 0.0
    return y_refined, x_refined, method, shift, warning


def _greedy_distance_filter(
    candidates: list[dict[str, float | str]],
    *,
    min_distance: float,
    max_count: int,
) -> list[dict[str, float | str]]:
    selected: list[dict[str, float | str]] = []
    for cand in candidates:
        y = float(cand["y"])
        x = float(cand["x"])
        if any(
            np.hypot(y - float(row["y"]), x - float(row["x"])) < min_distance
            for row in selected
        ):
            continue
        selected.append(cand)
        if len(selected) >= max_count:
            break
    return selected


def _sample_nearest(image: np.ndarray, y: float, x: float) -> float:
    yi = int(np.clip(round(y), 0, image.shape[0] - 1))
    xi = int(np.clip(round(x), 0, image.shape[1] - 1))
    return float(image[yi, xi])


def _angular_diff_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return float(min(d, 360.0 - d))


def _dedup_peaks_spatial_polar(
    peaks: list[dict[str, Any]],
    *,
    center_xy: tuple[float, float],
    dedup_distance_px: float,
    dedup_radial_tol_px: float,
    dedup_angle_tol_deg: float,
    min_radius_px_hard: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not peaks:
        return [], []
    cx, cy = center_xy
    ordered = sorted(
        peaks,
        key=lambda p: (
            float(p.get("snr", 0.0)),
            float(p.get("intensity", 0.0)),
        ),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for peak in ordered:
        x = float(peak["refined_x"])
        y = float(peak["refined_y"])
        dx = x - cx
        dy = y - cy
        r = float(np.hypot(dx, dy))
        theta = float(np.degrees(np.arctan2(dy, dx)))

        if r < min_radius_px_hard:
            rejected.append(
                {
                    "x": x,
                    "y": y,
                    "reason": "below_min_radius_hard",
                    "r_px": r,
                    "snr": float(peak.get("snr", 0.0)),
                }
            )
            continue

        is_dup = False
        for prev in kept:
            px = float(prev["refined_x"])
            py = float(prev["refined_y"])
            pdx = px - cx
            pdy = py - cy
            pr = float(np.hypot(pdx, pdy))
            ptheta = float(np.degrees(np.arctan2(pdy, pdx)))
            spatial_close = float(np.hypot(x - px, y - py)) <= dedup_distance_px
            polar_close = (
                abs(r - pr) <= dedup_radial_tol_px
                and _angular_diff_deg(theta, ptheta) <= dedup_angle_tol_deg
            )
            if spatial_close or polar_close:
                is_dup = True
                rejected.append(
                    {
                        "x": x,
                        "y": y,
                        "reason": "duplicate_peak_cluster",
                        "snr": float(peak.get("snr", 0.0)),
                        "r_px": r,
                    }
                )
                break
        if not is_dup:
            kept.append(peak)
    return kept, rejected


def _auto_center_cluster_cutoff(radii_px: list[float]) -> float | None:
    valid = sorted(float(r) for r in radii_px if np.isfinite(r) and r > 0.0)
    if len(valid) < 8:
        return None
    r_max = valid[-1]
    if r_max <= 0:
        return None
    upper_i = min(len(valid) - 2, len(valid) // 2)
    best_i = -1
    best_gap = -np.inf
    for i in range(0, upper_i + 1):
        a = valid[i]
        b = valid[i + 1]
        if a > max(30.0, 0.25 * r_max):
            continue
        gap = b - a
        if gap > best_gap:
            best_gap = gap
            best_i = i
    if best_i < 0:
        cluster_band = max(20.0, 0.12 * r_max)
        center_cluster = [r for r in valid if r <= cluster_band]
        if len(center_cluster) >= max(4, len(valid) // 4):
            return float(np.percentile(center_cluster, 85))
        return None
    if best_gap < max(2.0, 0.08 * r_max):
        cluster_band = max(20.0, 0.12 * r_max)
        center_cluster = [r for r in valid if r <= cluster_band]
        if len(center_cluster) >= max(4, len(valid) // 4):
            return float(np.percentile(center_cluster, 85))
        return None
    return 0.5 * (valid[best_i] + valid[best_i + 1])


def _lattice_consistency_filter_px(
    peaks: list[dict[str, Any]],
    *,
    center_xy: tuple[float, float],
    coeff_tol: float,
    recon_tol_px: float,
    min_keep: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(peaks) < max(4, min_keep):
        return peaks, [], {"applied": False, "reason": "too_few_peaks"}
    cx, cy = center_xy
    vectors: list[tuple[float, float]] = []
    intensities: list[float] = []
    for peak in peaks:
        vx = float(peak["refined_x"]) - cx
        vy = float(peak["refined_y"]) - cy
        vectors.append((vx, vy))
        intensities.append(float(peak.get("intensity", 0.0)))
    basis_pair = _select_basis_indices(vectors, intensities)
    if basis_pair is None:
        return peaks, [], {"applied": False, "reason": "basis_not_found"}

    i1, i2 = basis_pair
    b1 = np.array(vectors[i1], dtype=np.float64)
    b2 = np.array(vectors[i2], dtype=np.float64)
    bmat = np.column_stack((b1, b2))
    try:
        cond = float(np.linalg.cond(bmat))
    except np.linalg.LinAlgError:
        return peaks, [], {"applied": False, "reason": "singular_basis"}
    if not np.isfinite(cond) or cond > 1e4:
        return peaks, [], {"applied": False, "reason": "ill_conditioned_basis"}

    kept_by_hk: dict[tuple[int, int], tuple[dict[str, Any], float, float]] = {}
    rejected: list[dict[str, Any]] = []
    for peak in peaks:
        vec = np.array(
            [
                float(peak["refined_x"]) - cx,
                float(peak["refined_y"]) - cy,
            ],
            dtype=np.float64,
        )
        try:
            coeff = np.linalg.solve(bmat, vec)
        except np.linalg.LinAlgError:
            rejected.append(
                {
                    "x": float(peak["refined_x"]),
                    "y": float(peak["refined_y"]),
                    "reason": "lattice_solve_failed",
                }
            )
            continue
        rounded = np.rint(coeff)
        h = int(rounded[0])
        k = int(rounded[1])
        recon = bmat @ rounded
        coeff_err = float(np.linalg.norm(coeff - rounded))
        recon_err = float(np.linalg.norm(vec - recon))
        if h == 0 and k == 0:
            rejected.append(
                {
                    "x": float(peak["refined_x"]),
                    "y": float(peak["refined_y"]),
                    "reason": "lattice_origin_forbidden",
                    "coeff_err": coeff_err,
                    "recon_err_px": recon_err,
                }
            )
            continue
        if coeff_err > coeff_tol and recon_err > recon_tol_px:
            rejected.append(
                {
                    "x": float(peak["refined_x"]),
                    "y": float(peak["refined_y"]),
                    "reason": "lattice_inconsistent",
                    "coeff_err": coeff_err,
                    "recon_err_px": recon_err,
                    "h": h,
                    "k": k,
                }
            )
            continue
        snr = float(peak.get("snr", 0.0))
        score = -coeff_err + 0.02 * max(snr, 0.0)
        hk_key = (h, k)
        prev = kept_by_hk.get(hk_key)
        if prev is None or score > prev[1]:
            kept_by_hk[hk_key] = (peak, score, recon_err)
        else:
            rejected.append(
                {
                    "x": float(peak["refined_x"]),
                    "y": float(peak["refined_y"]),
                    "reason": "duplicate_hk_cluster",
                    "h": h,
                    "k": k,
                }
            )

    kept = [item[0] for item in kept_by_hk.values()]
    if len(kept) < min_keep:
        return peaks, rejected, {"applied": False, "reason": "over_pruned"}
    kept.sort(key=lambda p: float(p.get("intensity", 0.0)), reverse=True)
    return kept, rejected, {"applied": True, "basis_indices": [i1, i2], "condition": cond}


def _pair_friedel(
    peaks: list[dict[str, Any]],
    *,
    center_xy: tuple[float, float],
    pair_tolerance_px: float,
) -> tuple[list[dict[str, Any]], int]:
    if not peaks:
        return peaks, 0
    cx, cy = center_xy
    unpaired = set(range(len(peaks)))
    pair_id = 0
    pair_count = 0

    for idx in list(unpaired):
        if idx not in unpaired:
            continue
        vx = float(peaks[idx]["refined_x"]) - cx
        vy = float(peaks[idx]["refined_y"]) - cy
        best_j = None
        best_err = np.inf
        for j in unpaired:
            if j == idx:
                continue
            ux = float(peaks[j]["refined_x"]) - cx
            uy = float(peaks[j]["refined_y"]) - cy
            err = float(np.hypot(vx + ux, vy + uy))
            if err < best_err:
                best_err = err
                best_j = j
        if best_j is None or best_err > pair_tolerance_px:
            continue
        unpaired.remove(idx)
        unpaired.remove(best_j)
        peaks[idx]["paired"] = True
        peaks[best_j]["paired"] = True
        peaks[idx]["pair_id"] = pair_id
        peaks[best_j]["pair_id"] = pair_id
        peaks[idx]["pair_error_px"] = best_err
        peaks[best_j]["pair_error_px"] = best_err
        pair_id += 1
        pair_count += 1

    for idx in range(len(peaks)):
        if peaks[idx].get("paired"):
            continue
        peaks[idx]["paired"] = False
        peaks[idx]["pair_id"] = None
        peaks[idx]["pair_error_px"] = None
    return peaks, pair_count


def _detect_peaks_v2(
    magnitude: np.ndarray,
    *,
    parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    if magnitude.size == 0:
        return [], {"peak_details": []}, warnings

    h, w = magnitude.shape
    center_xy = (w / 2.0, h / 2.0)
    strategy_raw = str(parameters.get("peak_threshold_strategy", "hybrid")).strip().lower()
    strategy = strategy_raw
    legacy_strategies = {"quantile", "std", "relative_max", "mad"}
    if strategy == "auto":
        strategy = "hybrid"

    max_peaks = int(parameters.get("max_peaks", parameters.get("top_k_peaks", 24)))
    min_distance = float(parameters.get("min_distance", parameters.get("min_peak_distance", 3.0)))
    min_snr = float(parameters.get("min_snr", 1.2))
    center_mask_radius = float(
        parameters.get("center_mask_radius", parameters.get("center_exclusion_radius", 5.0))
    )
    center_reject_margin = float(parameters.get("center_reject_margin", 0.0))
    center_reject_radius = max(0.0, center_mask_radius + center_reject_margin)
    center_keep_snr = float(parameters.get("center_keep_snr", max(min_snr + 0.8, 2.0)))
    min_radius_px_hard = float(
        parameters.get(
            "min_radius_px_hard",
            center_reject_radius + 0.5,
        )
    )
    axis_streak_mask_width = float(parameters.get("axis_streak_mask_width", 0.0))
    border_mask = float(parameters.get("border_mask", 2.0))
    radial_bin_width = float(parameters.get("radial_bin_width", 2.0))
    peak_threshold_value_raw = parameters.get("peak_threshold_value")
    peak_threshold_value = (
        None if peak_threshold_value_raw is None else float(peak_threshold_value_raw)
    )
    compression = str(parameters.get("fft_compression", "log1p")).strip().lower()
    clip_low = float(parameters.get("clip_low_percentile", 1.0))
    clip_high = float(parameters.get("clip_high_percentile", 99.8))
    bg_mode = str(parameters.get("background_subtraction_method", "radial_mad")).strip().lower()
    gaussian_sigma = float(parameters.get("gaussian_background_sigma", 12.0))
    pair_tolerance_px = float(parameters.get("pair_tolerance_px", max(2.0, min_distance * 1.5)))
    max_radius_ratio = float(parameters.get("max_radius_ratio", 0.45))
    max_candidates_for_refine = int(
        parameters.get("max_candidates_for_refine", max(max_peaks * 6, 120))
    )
    dedup_distance_px = float(parameters.get("dedup_distance_px", max(4.5, min_distance * 1.35)))
    dedup_radial_tol_px = float(
        parameters.get("dedup_radial_tol_px", max(2.0, min_distance * 0.9))
    )
    dedup_angle_tol_deg = float(parameters.get("dedup_angle_tol_deg", 2.2))
    subpixel_refine = bool(parameters.get("subpixel_refine", True))
    subpixel_window_radius = int(parameters.get("subpixel_window_radius", 2))
    subpixel_max_shift = float(parameters.get("subpixel_max_shift", 1.5))
    max_gaussian_refine_count = int(parameters.get("max_gaussian_refine_count", 80))
    rescue_on_empty = bool(parameters.get("rescue_on_empty", True))
    enable_inner_pair_recovery = bool(parameters.get("enable_inner_pair_recovery", True))
    inner_pair_min_snr = float(parameters.get("inner_pair_min_snr", max(0.5, min_snr * 0.5)))
    inner_pair_recover_radius_px = float(
        parameters.get(
            "inner_pair_recover_radius_px",
            max(20.0, 0.20 * min(h, w)),
        )
    )
    inner_pair_max_pairs = int(parameters.get("inner_pair_max_pairs", 8))
    allow_relaxed_fallback = bool(parameters.get("allow_relaxed_fallback", False))
    enable_lattice_consistency_filter = bool(
        parameters.get("enable_lattice_consistency_filter", False)
    )
    lattice_coeff_tol = float(parameters.get("lattice_coeff_tol", 0.38))
    lattice_recon_tol_px = float(parameters.get("lattice_recon_tol_px", max(4.0, min_distance)))
    lattice_filter_min_keep = int(parameters.get("lattice_filter_min_keep", 6))
    auto_center_cluster_filter = bool(parameters.get("auto_center_cluster_filter", False))

    compressed = _compress_fft(magnitude, compression)
    clipped, clip_low_used, clip_high_used = _percentile_clip(
        compressed,
        low_percentile=clip_low,
        high_percentile=clip_high,
    )
    masks = _build_masks(
        clipped.shape,
        center_xy=(center_xy[1], center_xy[0]),
        center_mask_radius=center_mask_radius,
        axis_streak_mask_width=axis_streak_mask_width,
        border_mask=border_mask,
    )
    valid_mask = masks["valid_mask"]

    radial_median, radial_noise, radial_z = _radial_background_stats(
        clipped,
        center_xy=(center_xy[1], center_xy[0]),
        radial_bin_width=radial_bin_width,
        valid_mask=valid_mask,
    )
    radial_sub = (clipped - radial_median).astype(np.float32)
    if bg_mode in {"gaussian", "hybrid"}:
        gaussian_bg = ndi.gaussian_filter(
            clipped,
            sigma=max(0.1, gaussian_sigma),
        ).astype(np.float32)
        gaussian_sub = (clipped - gaussian_bg).astype(np.float32)
    else:
        gaussian_sub = np.zeros_like(radial_sub, dtype=np.float32)
    if bg_mode == "gaussian":
        background_sub = gaussian_sub
    elif bg_mode == "hybrid":
        background_sub = (0.6 * radial_sub + 0.4 * gaussian_sub).astype(np.float32)
    else:
        background_sub = radial_sub

    candidates: list[dict[str, float | str]] = []
    if strategy_raw in legacy_strategies:
        threshold = _compute_peak_threshold(
            background_sub,
            strategy=strategy_raw,
            value=peak_threshold_value,
        )
        candidates = _local_maxima_candidates(
            background_sub,
            valid_mask=valid_mask,
            threshold=threshold,
            min_distance=min_distance,
            source=f"legacy_{strategy_raw}",
        )
    elif strategy == "radial_mad":
        threshold = 4.0 if peak_threshold_value is None else float(peak_threshold_value)
        candidates = _local_maxima_candidates(
            radial_z,
            valid_mask=valid_mask,
            threshold=threshold,
            min_distance=min_distance,
            source="radial_mad",
        )
    elif strategy == "local_max":
        threshold = _compute_peak_threshold(
            background_sub,
            strategy="mad",
            value=peak_threshold_value,
        )
        candidates = _local_maxima_candidates(
            background_sub,
            valid_mask=valid_mask,
            threshold=threshold,
            min_distance=min_distance,
            source="local_max",
        )
    elif strategy == "log_blob":
        threshold = None if peak_threshold_value is None else float(peak_threshold_value)
        candidates = _log_blob_candidates(
            background_sub,
            valid_mask=valid_mask,
            threshold=threshold,
            min_distance=min_distance,
        )
    else:
        threshold = 2.5 if peak_threshold_value is None else float(peak_threshold_value)
        radius = max(1, int(round(min_distance)))
        local_max = background_sub >= ndi.maximum_filter(
            background_sub,
            size=(2 * radius + 1),
            mode="nearest",
        )
        seed = (radial_z >= threshold) & valid_mask
        cand_mask = local_max & seed
        ys, xs = np.nonzero(cand_mask)
        for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
            candidates.append(
                {
                    "y": float(y),
                    "x": float(x),
                    "score": float(background_sub[y, x]),
                    "source": "hybrid",
                }
            )
        if len(candidates) < max(4, max_peaks // 4):
            fallback = _local_maxima_candidates(
                radial_z,
                valid_mask=valid_mask,
                threshold=max(0.8, threshold * 0.8),
                min_distance=min_distance,
                source="hybrid_fallback",
            )
            candidates.extend(fallback)
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)

    if not candidates:
        fallback_threshold = _compute_peak_threshold(
            background_sub,
            strategy="auto",
            value=None,
        )
        candidates = _local_maxima_candidates(
            background_sub,
            valid_mask=valid_mask,
            threshold=fallback_threshold,
            min_distance=min_distance,
            source="empty_fallback_auto",
        )
    if not candidates:
        relaxed_threshold = float(np.quantile(background_sub[valid_mask], 0.995))
        candidates = _local_maxima_candidates(
            background_sub,
            valid_mask=valid_mask,
            threshold=relaxed_threshold,
            min_distance=min_distance,
            source="empty_fallback_quantile",
        )
    if not candidates:
        relaxed_threshold = _compute_peak_threshold(
            magnitude,
            strategy="relative_max",
            value=0.1,
        )
        raw_max = magnitude >= ndi.maximum_filter(
            magnitude,
            size=(2 * max(1, int(round(min_distance))) + 1),
            mode="nearest",
        )
        raw_mask = raw_max & valid_mask & (magnitude >= float(relaxed_threshold))
        ys, xs = np.nonzero(raw_mask)
        candidates = [
            {
                "y": float(y),
                "x": float(x),
                "score": float(magnitude[y, x]),
                "source": "empty_fallback_raw_relative_max",
            }
            for y, x in zip(ys.tolist(), xs.tolist(), strict=False)
        ]
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)

    candidates = _greedy_distance_filter(
        candidates,
        min_distance=max(0.5, min_distance),
        max_count=max(max_peaks * 4, 64),
    )
    if max_candidates_for_refine > 0 and len(candidates) > max_candidates_for_refine:
        candidates = candidates[:max_candidates_for_refine]

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    radius_base = 0.5 * float(min(h, w))
    max_radius_px = max(1.0, max_radius_ratio * radius_base)
    outside_max_radius_rejects = 0
    for cand_idx, cand in enumerate(candidates):
        y = float(cand["y"])
        x = float(cand["x"])
        r_center = float(np.hypot(x - center_xy[0], y - center_xy[1]))
        if r_center > max_radius_px:
            outside_max_radius_rejects += 1
            rejected.append(
                {
                    "x": x,
                    "y": y,
                    "reason": "outside_max_radius",
                    "r_px": r_center,
                }
            )
            continue
        local_bg = _sample_nearest(radial_median, y, x)
        local_noise = max(_sample_nearest(radial_noise, y, x), 1e-6)
        signal = max(_sample_nearest(radial_sub, y, x), _sample_nearest(background_sub, y, x))
        snr = float(signal / local_noise)
        if r_center < center_reject_radius and snr < center_keep_snr:
            rejected.append(
                {
                    "x": x,
                    "y": y,
                    "reason": "near_center_transmission_artifact",
                    "snr": snr,
                    "r_px": r_center,
                }
            )
            continue
        if snr < min_snr:
            rejected.append(
                {
                    "x": x,
                    "y": y,
                    "reason": "snr_below_threshold",
                    "snr": snr,
                }
            )
            continue
        refined_y = y
        refined_x = x
        refine_method = "none"
        shift_px = 0.0
        warning: str | None = None
        if subpixel_refine:
            use_gaussian = cand_idx < max(0, max_gaussian_refine_count)
            refined_y, refined_x, refine_method, shift_px, warning = _refine_peak_subpixel_v2(
                clipped,
                y=y,
                x=x,
                window_radius=max(1, subpixel_window_radius),
                max_shift=max(0.0, subpixel_max_shift),
                prefer_gaussian=use_gaussian,
            )
        if any(
            np.hypot(refined_y - float(item["refined_y"]), refined_x - float(item["refined_x"]))
            < min_distance
            for item in accepted
        ):
            rejected.append(
                {
                    "x": x,
                    "y": y,
                    "reason": "too_close_to_existing_peak",
                    "snr": snr,
                }
            )
            continue
        intensity = _sample_nearest(magnitude, refined_y, refined_x)
        peak = {
            "x": x,
            "y": y,
            "refined_x": float(refined_x),
            "refined_y": float(refined_y),
            "shift_px": float(shift_px),
            "intensity": float(intensity),
            "snr": snr,
            "local_background": float(local_bg),
            "refine_method": refine_method,
            "warning": warning,
        }
        accepted.append(peak)
        if len(accepted) >= max_peaks:
            break

    if allow_relaxed_fallback and not accepted and candidates:
        relaxed = _greedy_distance_filter(
            candidates,
            min_distance=max(0.5, min_distance),
            max_count=max_peaks,
        )
        for cand in relaxed:
            y = float(cand["y"])
            x = float(cand["x"])
            refined_y = y
            refined_x = x
            refine_method = "none"
            shift_px = 0.0
            warning = "relaxed_acceptance_low_snr"
            if subpixel_refine:
                use_gaussian = len(accepted) < max(0, max_gaussian_refine_count)
                (
                    refined_y,
                    refined_x,
                    refine_method,
                    shift_px,
                    warning_fit,
                ) = _refine_peak_subpixel_v2(
                    clipped,
                    y=y,
                    x=x,
                    window_radius=max(1, subpixel_window_radius),
                    max_shift=max(0.0, subpixel_max_shift),
                    prefer_gaussian=use_gaussian,
                )
                if warning_fit:
                    warning = f"{warning};{warning_fit}"
            intensity = _sample_nearest(magnitude, refined_y, refined_x)
            accepted.append(
                {
                    "x": x,
                    "y": y,
                    "refined_x": float(refined_x),
                    "refined_y": float(refined_y),
                    "shift_px": float(shift_px),
                    "intensity": float(intensity),
                    "snr": 0.0,
                    "local_background": float(_sample_nearest(radial_median, y, x)),
                    "refine_method": refine_method,
                    "warning": warning,
                }
            )
            if len(accepted) >= max_peaks:
                break

    if enable_inner_pair_recovery and candidates and len(accepted) < max_peaks:
        accepted_xy = {
            (float(item["refined_x"]), float(item["refined_y"]))
            for item in accepted
        }
        pair_cap = max(0, inner_pair_max_pairs)
        pair_added = 0
        inner_pool: list[dict[str, float]] = []
        for cand in candidates:
            y = float(cand["y"])
            x = float(cand["x"])
            r = float(np.hypot(x - center_xy[0], y - center_xy[1]))
            if r < max(min_radius_px_hard, center_reject_radius):
                continue
            if r > max(0.0, inner_pair_recover_radius_px):
                continue
            local_noise = max(_sample_nearest(radial_noise, y, x), 1e-6)
            signal = max(_sample_nearest(radial_sub, y, x), _sample_nearest(background_sub, y, x))
            snr = float(signal / local_noise)
            if snr < inner_pair_min_snr:
                continue
            if any(
                np.hypot(y - yy, x - xx) < max(1.0, min_distance * 0.9)
                for xx, yy in accepted_xy
            ):
                continue
            inner_pool.append({"x": x, "y": y, "snr": snr})
        used: set[int] = set()
        for i, cand_i in enumerate(inner_pool):
            if i in used:
                continue
            vx = cand_i["x"] - center_xy[0]
            vy = cand_i["y"] - center_xy[1]
            best_j = None
            best_err = np.inf
            for j, cand_j in enumerate(inner_pool):
                if j == i or j in used:
                    continue
                ux = cand_j["x"] - center_xy[0]
                uy = cand_j["y"] - center_xy[1]
                err = float(np.hypot(vx + ux, vy + uy))
                if err < best_err:
                    best_err = err
                    best_j = j
            if best_j is None or best_err > max(1.5, pair_tolerance_px * 1.2):
                continue
            if pair_cap > 0 and pair_added >= pair_cap:
                break
            for idx in (i, best_j):
                cand = inner_pool[idx]
                accepted.append(
                    {
                        "x": float(cand["x"]),
                        "y": float(cand["y"]),
                        "refined_x": float(cand["x"]),
                        "refined_y": float(cand["y"]),
                        "shift_px": 0.0,
                        "intensity": float(_sample_nearest(magnitude, cand["y"], cand["x"])),
                        "snr": float(cand["snr"]),
                        "local_background": float(
                            _sample_nearest(radial_median, cand["y"], cand["x"])
                        ),
                        "refine_method": "inner_pair_recovery",
                        "warning": "inner_pair_recovery",
                    }
                )
                accepted_xy.add((float(cand["x"]), float(cand["y"])))
                used.add(idx)
                if len(accepted) >= max_peaks:
                    break
            used.add(i)
            used.add(best_j)
            pair_added += 1
            if len(accepted) >= max_peaks:
                break

    accepted, dedup_rejected = _dedup_peaks_spatial_polar(
        accepted,
        center_xy=center_xy,
        dedup_distance_px=max(0.5, dedup_distance_px),
        dedup_radial_tol_px=max(0.2, dedup_radial_tol_px),
        dedup_angle_tol_deg=max(0.1, dedup_angle_tol_deg),
        min_radius_px_hard=max(0.0, min_radius_px_hard),
    )
    rejected.extend(dedup_rejected)

    lattice_meta: dict[str, Any] = {"applied": False}
    if enable_lattice_consistency_filter:
        accepted, lattice_rejected, lattice_meta = _lattice_consistency_filter_px(
            accepted,
            center_xy=center_xy,
            coeff_tol=max(0.05, lattice_coeff_tol),
            recon_tol_px=max(0.5, lattice_recon_tol_px),
            min_keep=max(2, lattice_filter_min_keep),
        )
        rejected.extend(lattice_rejected)

    center_cluster_cutoff = None
    if auto_center_cluster_filter and accepted:
        radii = [
            float(
                np.hypot(
                    float(p["refined_x"]) - center_xy[0],
                    float(p["refined_y"]) - center_xy[1],
                )
            )
            for p in accepted
        ]
        center_cluster_cutoff = _auto_center_cluster_cutoff(radii)
        if center_cluster_cutoff is not None:
            kept_auto: list[dict[str, Any]] = []
            for peak in accepted:
                r = float(
                    np.hypot(
                        float(peak["refined_x"]) - center_xy[0],
                        float(peak["refined_y"]) - center_xy[1],
                    )
                )
                if r < center_cluster_cutoff:
                    rejected.append(
                        {
                            "x": float(peak["refined_x"]),
                            "y": float(peak["refined_y"]),
                            "reason": "auto_center_cluster_removed",
                            "r_px": r,
                            "cutoff_px": center_cluster_cutoff,
                        }
                    )
                    continue
                kept_auto.append(peak)
            if kept_auto:
                accepted = kept_auto

    rescue_min_snr = float(parameters.get("rescue_min_snr", max(0.2, min_snr * 0.4)))
    rescue_max_radius_px = max_radius_px
    if (
        not accepted
        and len(candidates) > 0
        and outside_max_radius_rejects >= max(6, int(0.4 * len(candidates)))
    ):
        rescue_max_radius_px = max(max_radius_px, 0.65 * radius_base)
    ensure_multiring_coverage = bool(parameters.get("ensure_multiring_coverage", True))
    inner_ring_recover_pairs = int(parameters.get("inner_ring_recover_pairs", 4))
    inner_ring_recover_min_snr = float(
        parameters.get("inner_ring_recover_min_snr", max(0.6, min_snr * 0.45))
    )
    if rescue_on_empty and not accepted and candidates:
        rescue_pool = sorted(
            candidates,
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )[: max(8, min(max_peaks, 20))]
        for cand in rescue_pool:
            y = float(cand["y"])
            x = float(cand["x"])
            r_center = float(np.hypot(x - center_xy[0], y - center_xy[1]))
            if r_center > rescue_max_radius_px:
                continue
            if r_center < max(min_radius_px_hard, center_reject_radius):
                continue
            local_bg = _sample_nearest(radial_median, y, x)
            local_noise = max(_sample_nearest(radial_noise, y, x), 1e-6)
            signal = max(_sample_nearest(radial_sub, y, x), _sample_nearest(background_sub, y, x))
            snr = float(signal / local_noise)
            if snr < rescue_min_snr:
                continue
            refined_y, refined_x, refine_method, shift_px, warning = _refine_peak_subpixel_v2(
                clipped,
                y=y,
                x=x,
                window_radius=max(1, subpixel_window_radius),
                max_shift=max(0.0, subpixel_max_shift),
                prefer_gaussian=False,
            )
            accepted.append(
                {
                    "x": x,
                    "y": y,
                    "refined_x": float(refined_x),
                    "refined_y": float(refined_y),
                    "shift_px": float(shift_px),
                    "intensity": float(_sample_nearest(magnitude, refined_y, refined_x)),
                    "snr": snr,
                    "local_background": float(local_bg),
                    "refine_method": refine_method,
                    "warning": (
                        "rescue_on_empty"
                        if warning is None
                        else f"rescue_on_empty;{warning}"
                    ),
                }
            )
            if len(accepted) >= max_peaks:
                break
        if not accepted:
            for cand in rescue_pool:
                y = float(cand["y"])
                x = float(cand["x"])
                r_center = float(np.hypot(x - center_xy[0], y - center_xy[1]))
                if r_center > rescue_max_radius_px or r_center < max(
                    min_radius_px_hard,
                    center_reject_radius,
                ):
                    continue
                if any(
                    np.hypot(
                        y - float(item["refined_y"]),
                        x - float(item["refined_x"]),
                    )
                    < min_distance
                    for item in accepted
                ):
                    continue
                accepted.append(
                    {
                        "x": x,
                        "y": y,
                        "refined_x": float(x),
                        "refined_y": float(y),
                        "shift_px": 0.0,
                        "intensity": float(_sample_nearest(magnitude, y, x)),
                        "snr": float(max(0.0, _sample_nearest(radial_z, y, x))),
                        "local_background": float(_sample_nearest(radial_median, y, x)),
                        "refine_method": "rescue_direct",
                        "warning": "rescue_on_empty;direct_accept",
                    }
                )
                if len(accepted) >= max_peaks:
                    break

    if ensure_multiring_coverage and accepted and len(accepted) < max_peaks and candidates:
        accepted_r = np.array(
            [
                float(
                    np.hypot(
                        float(item["refined_x"]) - center_xy[0],
                        float(item["refined_y"]) - center_xy[1],
                    )
                )
                for item in accepted
            ],
            dtype=np.float64,
        )
        narrow_band = False
        if accepted_r.size >= 8:
            r_span = float(np.percentile(accepted_r, 90) - np.percentile(accepted_r, 10))
            narrow_band = r_span <= max(3.0, 0.14 * float(np.max(accepted_r)))
        if narrow_band:
            ring_floor = float(np.percentile(accepted_r, 20))
            inner_pool: list[dict[str, float]] = []
            inner_low = max(center_reject_radius, min_radius_px_hard)
            inner_high = max(inner_low + 2.0, max(0.0, ring_floor - 1.0))
            for cand in candidates:
                y = float(cand["y"])
                x = float(cand["x"])
                r = float(np.hypot(x - center_xy[0], y - center_xy[1]))
                if r <= inner_low:
                    continue
                if r >= inner_high:
                    continue
                local_noise = max(_sample_nearest(radial_noise, y, x), 1e-6)
                signal = max(
                    _sample_nearest(radial_sub, y, x),
                    _sample_nearest(background_sub, y, x),
                )
                snr = float(signal / local_noise)
                if snr < inner_ring_recover_min_snr:
                    continue
                inner_pool.append({"x": x, "y": y, "snr": snr})

            # If candidate pool itself missed the first inner ring, bootstrap from
            # radial-z local maxima in the annulus between transmission and first ring.
            if len(inner_pool) < max(6, 2 * max(1, inner_ring_recover_pairs)):
                yy, xx = np.indices(clipped.shape, dtype=np.float32)
                rr = np.hypot(xx - center_xy[0], yy - center_xy[1])
                radius = max(1, int(round(min_distance * 0.6)))
                local_max_map = radial_z >= ndi.maximum_filter(
                    radial_z,
                    size=(2 * radius + 1),
                    mode="nearest",
                )
                annulus = (rr > inner_low) & (rr < inner_high)
                seed = (
                    local_max_map
                    & valid_mask
                    & annulus
                    & (radial_z >= float(inner_ring_recover_min_snr))
                )
                ys2, xs2 = np.nonzero(seed)
                for y2, x2 in zip(ys2.tolist(), xs2.tolist(), strict=False):
                    local_noise = max(_sample_nearest(radial_noise, y2, x2), 1e-6)
                    signal = max(
                        _sample_nearest(radial_sub, y2, x2),
                        _sample_nearest(background_sub, y2, x2),
                    )
                    snr2 = float(signal / local_noise)
                    if snr2 < inner_ring_recover_min_snr:
                        continue
                    inner_pool.append(
                        {
                            "x": float(x2),
                            "y": float(y2),
                            "snr": snr2,
                        }
                    )

            pair_added = 0
            accepted_xy = {
                (float(item["refined_x"]), float(item["refined_y"]))
                for item in accepted
            }
            inner_pool.sort(key=lambda item: float(item["snr"]), reverse=True)
            filtered_inner_pool: list[dict[str, float]] = []
            for cand in inner_pool:
                if any(
                    np.hypot(cand["y"] - float(p["y"]), cand["x"] - float(p["x"]))
                    < max(0.8, min_distance * 0.75)
                    for p in filtered_inner_pool
                ):
                    continue
                filtered_inner_pool.append(cand)
            inner_pool = filtered_inner_pool
            used: set[int] = set()
            for i, cand_i in enumerate(inner_pool):
                if i in used:
                    continue
                vx = cand_i["x"] - center_xy[0]
                vy = cand_i["y"] - center_xy[1]
                best_j = None
                best_err = np.inf
                for j, cand_j in enumerate(inner_pool):
                    if j == i or j in used:
                        continue
                    ux = cand_j["x"] - center_xy[0]
                    uy = cand_j["y"] - center_xy[1]
                    err = float(np.hypot(vx + ux, vy + uy))
                    ri = float(np.hypot(vx, vy))
                    rj = float(np.hypot(ux, uy))
                    if abs(ri - rj) > max(2.0, 0.8 * min_distance):
                        continue
                    if err < best_err:
                        best_err = err
                        best_j = j
                if best_j is None or best_err > max(1.4, pair_tolerance_px * 1.15):
                    continue
                if pair_added >= max(0, inner_ring_recover_pairs):
                    break
                for idx in (i, best_j):
                    cand = inner_pool[idx]
                    if any(
                        np.hypot(cand["y"] - yy, cand["x"] - xx)
                        < max(1.0, min_distance * 0.9)
                        for xx, yy in accepted_xy
                    ):
                        continue
                    accepted.append(
                        {
                            "x": float(cand["x"]),
                            "y": float(cand["y"]),
                            "refined_x": float(cand["x"]),
                            "refined_y": float(cand["y"]),
                            "shift_px": 0.0,
                            "intensity": float(_sample_nearest(magnitude, cand["y"], cand["x"])),
                            "snr": float(cand["snr"]),
                            "local_background": float(
                                _sample_nearest(radial_median, cand["y"], cand["x"])
                            ),
                            "refine_method": "inner_ring_recovery",
                            "warning": "inner_ring_recovery",
                        }
                    )
                    accepted_xy.add((float(cand["x"]), float(cand["y"])))
                    used.add(idx)
                    if len(accepted) >= max_peaks:
                        break
                used.add(i)
                used.add(best_j)
                pair_added += 1
                if len(accepted) >= max_peaks:
                    break

    # Force supplement the first inner ring when accepted peaks start too far
    # from center (common failure mode: outer rings found, inner ring missing).
    if ensure_multiring_coverage and accepted and len(accepted) < max_peaks:
        accepted_r = np.array(
            [
                float(
                    np.hypot(
                        float(item["refined_x"]) - center_xy[0],
                        float(item["refined_y"]) - center_xy[1],
                    )
                )
                for item in accepted
            ],
            dtype=np.float64,
        )
        if accepted_r.size >= 6:
            inner_low = max(center_reject_radius, min_radius_px_hard)
            r_min_acc = float(np.min(accepted_r))
            need_force_inner = r_min_acc > max(inner_low + 14.0, 0.12 * min(h, w))
            if need_force_inner:
                target_r = max(inner_low + 1.5, 0.60 * r_min_acc)
                band_low = max(inner_low + 0.8, 0.65 * target_r)
                band_high = min(r_min_acc - 0.8, 1.30 * target_r)
                if band_high > band_low + 2.0:
                    yy2, xx2 = np.indices(clipped.shape, dtype=np.float32)
                    rr2 = np.hypot(xx2 - center_xy[0], yy2 - center_xy[1])
                    annulus = (rr2 >= band_low) & (rr2 <= band_high) & valid_mask
                    inner_enh = (
                        ndi.gaussian_filter(clipped, sigma=1.2)
                        - ndi.gaussian_filter(clipped, sigma=4.0)
                    ).astype(np.float32)
                    radius2 = max(1, int(round(min_distance * 0.6)))
                    local_max2 = inner_enh >= ndi.maximum_filter(
                        inner_enh,
                        size=(2 * radius2 + 1),
                        mode="nearest",
                    )
                    vals = inner_enh[annulus]
                    if vals.size > 0:
                        thr2 = float(np.quantile(vals, 0.82))
                        seed2 = annulus & local_max2 & (inner_enh >= thr2)
                        ys2, xs2 = np.nonzero(seed2)
                        pool2: list[dict[str, float]] = []
                        for y2, x2 in zip(ys2.tolist(), xs2.tolist(), strict=False):
                            local_noise2 = max(_sample_nearest(radial_noise, y2, x2), 1e-6)
                            signal2 = max(
                                _sample_nearest(radial_sub, y2, x2),
                                _sample_nearest(background_sub, y2, x2),
                            )
                            snr2 = float(signal2 / local_noise2)
                            if snr2 < max(0.25, inner_ring_recover_min_snr * 0.55):
                                continue
                            pool2.append(
                                {
                                    "x": float(x2),
                                    "y": float(y2),
                                    "snr": snr2,
                                }
                            )
                        pool2.sort(key=lambda item: float(item["snr"]), reverse=True)
                        filtered2: list[dict[str, float]] = []
                        for cand in pool2:
                            if any(
                                np.hypot(cand["y"] - p["y"], cand["x"] - p["x"])
                                < max(0.9, min_distance * 0.75)
                                for p in filtered2
                            ):
                                continue
                            filtered2.append(cand)
                        pool2 = filtered2
                        accepted_xy2 = {
                            (float(item["refined_x"]), float(item["refined_y"]))
                            for item in accepted
                        }
                        pair_added2 = 0
                        used2: set[int] = set()
                        for i, cand_i in enumerate(pool2):
                            if i in used2:
                                continue
                            vxi = cand_i["x"] - center_xy[0]
                            vyi = cand_i["y"] - center_xy[1]
                            ri = float(np.hypot(vxi, vyi))
                            best_j = None
                            best_err = np.inf
                            for j, cand_j in enumerate(pool2):
                                if j == i or j in used2:
                                    continue
                                vxj = cand_j["x"] - center_xy[0]
                                vyj = cand_j["y"] - center_xy[1]
                                rj = float(np.hypot(vxj, vyj))
                                if abs(ri - rj) > max(4.0, 1.5 * min_distance):
                                    continue
                                err = float(np.hypot(vxi + vxj, vyi + vyj))
                                if err < best_err:
                                    best_err = err
                                    best_j = j
                            if best_j is None or best_err > max(2.2, pair_tolerance_px * 1.5):
                                continue
                            if pair_added2 >= max(0, inner_ring_recover_pairs):
                                break
                            for idx2 in (i, best_j):
                                cand2 = pool2[idx2]
                                if any(
                                    np.hypot(cand2["y"] - yyv, cand2["x"] - xxv)
                                    < max(1.0, min_distance * 0.9)
                                    for xxv, yyv in accepted_xy2
                                ):
                                    continue
                                accepted.append(
                                    {
                                        "x": float(cand2["x"]),
                                        "y": float(cand2["y"]),
                                        "refined_x": float(cand2["x"]),
                                        "refined_y": float(cand2["y"]),
                                        "shift_px": 0.0,
                                        "intensity": float(
                                            _sample_nearest(magnitude, cand2["y"], cand2["x"])
                                        ),
                                        "snr": float(cand2["snr"]),
                                        "local_background": float(
                                            _sample_nearest(radial_median, cand2["y"], cand2["x"])
                                        ),
                                        "refine_method": "inner_ring_force_recovery",
                                        "warning": "inner_ring_force_recovery",
                                    }
                                )
                                accepted_xy2.add((float(cand2["x"]), float(cand2["y"])))
                                used2.add(idx2)
                                if len(accepted) >= max_peaks:
                                    break
                            used2.add(i)
                            used2.add(best_j)
                            pair_added2 += 1
                            if len(accepted) >= max_peaks:
                                break

    # Transmission-first basis completion:
    # use two nearest non-collinear diffraction vectors to recover missing first-ring spots.
    idx_mode = str(parameters.get("indexing_mode", "auto")).strip().lower()
    enable_basis_completion = bool(parameters.get("enable_basis_completion", True))
    if idx_mode == "b4_strict":
        enable_basis_completion = False
    basis_completion_search_radius_px = float(
        parameters.get(
            "basis_completion_search_radius_px",
            max(3.0, 1.8 * min_distance),
        )
    )
    basis_completion_min_snr = float(
        parameters.get("basis_completion_min_snr", max(0.25, min_snr * 0.35))
    )
    if enable_basis_completion and accepted and len(accepted) < max_peaks:
        vectors: list[tuple[float, float, float]] = []
        for item in accepted:
            vx = float(item["refined_x"]) - center_xy[0]
            vy = float(item["refined_y"]) - center_xy[1]
            rr = float(np.hypot(vx, vy))
            if rr <= max(center_reject_radius, min_radius_px_hard):
                continue
            vectors.append((vx, vy, rr))
        vectors.sort(key=lambda row: row[2])
        b1 = None
        b2 = None
        for vx, vy, rr in vectors:
            if b1 is None:
                b1 = (vx, vy, rr)
                continue
            if b2 is None and b1 is not None:
                cosang = float(
                    (vx * b1[0] + vy * b1[1])
                    / max(1e-9, rr * b1[2])
                )
                # reject near-collinear or near-opposite choices.
                if abs(cosang) >= 0.92:
                    continue
                b2 = (vx, vy, rr)
                break
        if b1 is not None and b2 is not None:
            r_ref = min(b1[2], b2[2])
            probe_vecs = [
                (b1[0], b1[1]),
                (-b1[0], -b1[1]),
                (b2[0], b2[1]),
                (-b2[0], -b2[1]),
                (b1[0] + b2[0], b1[1] + b2[1]),
                (-(b1[0] + b2[0]), -(b1[1] + b2[1])),
                (b1[0] - b2[0], b1[1] - b2[1]),
                (-(b1[0] - b2[0]), -(b1[1] - b2[1])),
            ]
            probe_vecs = [
                (vx, vy)
                for vx, vy in probe_vecs
                if max(center_reject_radius, min_radius_px_hard) < np.hypot(vx, vy)
                <= max(r_ref * 1.35, r_ref + 10.0)
            ]
            if probe_vecs:
                inner_enh = (
                    ndi.gaussian_filter(clipped, sigma=1.0)
                    - ndi.gaussian_filter(clipped, sigma=3.4)
                ).astype(np.float32)
                accepted_xy = {
                    (float(item["refined_x"]), float(item["refined_y"]))
                    for item in accepted
                }
                win = max(2, int(round(basis_completion_search_radius_px)))
                for pvx, pvy in probe_vecs:
                    tx = float(center_xy[0] + pvx)
                    ty = float(center_xy[1] + pvy)
                    if tx < 1 or ty < 1 or tx >= w - 1 or ty >= h - 1:
                        continue
                    x0 = max(0, int(np.floor(tx - win)))
                    x1 = min(w, int(np.ceil(tx + win + 1)))
                    y0 = max(0, int(np.floor(ty - win)))
                    y1 = min(h, int(np.ceil(ty + win + 1)))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    patch = inner_enh[y0:y1, x0:x1]
                    if patch.size == 0:
                        continue
                    max_idx = int(np.argmax(patch))
                    py, px = np.unravel_index(max_idx, patch.shape)
                    y_peak = float(y0 + py)
                    x_peak = float(x0 + px)
                    if not bool(valid_mask[int(y_peak), int(x_peak)]):
                        continue
                    local_noise = max(_sample_nearest(radial_noise, y_peak, x_peak), 1e-6)
                    signal = max(
                        _sample_nearest(radial_sub, y_peak, x_peak),
                        _sample_nearest(background_sub, y_peak, x_peak),
                    )
                    snr = float(signal / local_noise)
                    if snr < basis_completion_min_snr:
                        continue
                    if any(
                        np.hypot(y_peak - yy, x_peak - xx) < max(1.2, min_distance * 0.95)
                        for xx, yy in accepted_xy
                    ):
                        continue
                    accepted.append(
                        {
                            "x": x_peak,
                            "y": y_peak,
                            "refined_x": x_peak,
                            "refined_y": y_peak,
                            "shift_px": 0.0,
                            "intensity": float(_sample_nearest(magnitude, y_peak, x_peak)),
                            "snr": snr,
                            "local_background": float(
                                _sample_nearest(radial_median, y_peak, x_peak)
                            ),
                            "refine_method": "basis_completion",
                            "warning": "basis_completion",
                        }
                    )
                    accepted_xy.add((x_peak, y_peak))
                    if len(accepted) >= max_peaks:
                        break

    # Final global dedup after all recovery stages (prevents duplicate ids like 45/47).
    accepted, late_dedup_rejected = _dedup_peaks_spatial_polar(
        accepted,
        center_xy=center_xy,
        dedup_distance_px=max(1.2, max(dedup_distance_px, 1.35 * min_distance)),
        dedup_radial_tol_px=max(0.2, dedup_radial_tol_px),
        dedup_angle_tol_deg=max(0.1, dedup_angle_tol_deg),
        min_radius_px_hard=max(0.0, min_radius_px_hard),
    )
    rejected.extend(late_dedup_rejected)

    accepted, pair_count = _pair_friedel(
        accepted,
        center_xy=center_xy,
        pair_tolerance_px=pair_tolerance_px,
    )
    for peak in accepted:
        if not bool(peak.get("paired", False)):
            prior = str(peak.get("warning") or "").strip()
            extra = "unpaired_friedel"
            peak["warning"] = extra if not prior else f"{prior};{extra}"

    details: list[dict[str, Any]] = []
    for idx, peak in enumerate(accepted):
        dx = float(peak["refined_x"]) - center_xy[0]
        dy = float(peak["refined_y"]) - center_xy[1]
        details.append(
            {
                "peak_id": idx,
                "x": float(peak["x"]),
                "y": float(peak["y"]),
                "refined_x": float(peak["refined_x"]),
                "refined_y": float(peak["refined_y"]),
                "dx": dx,
                "dy": dy,
                "r_px": float(np.hypot(dx, dy)),
                "theta_deg": float(np.degrees(np.arctan2(dy, dx))),
                "intensity": float(peak["intensity"]),
                "snr": float(peak["snr"]),
                "local_background": float(peak["local_background"]),
                "shift_px": float(peak["shift_px"]),
                "refine_method": str(peak["refine_method"]),
                "paired": bool(peak.get("paired", False)),
                "pair_id": peak.get("pair_id"),
                "pair_error_px": peak.get("pair_error_px"),
                "warning": str(peak.get("warning") or ""),
            }
        )

    debug = {
        "strategy": strategy,
        "strategy_raw": strategy_raw,
        "peak_details": details,
        "counts": {
            "candidate_count": len(candidates),
            "accepted_count": len(details),
            "rejected_count": len(rejected),
            "pair_count": int(pair_count),
        },
        "clip_low_used": clip_low_used,
        "clip_high_used": clip_high_used,
        "parameters_used": {
            "max_peaks": max_peaks,
            "min_distance": min_distance,
            "min_snr": min_snr,
            "center_mask_radius": center_mask_radius,
            "center_reject_margin": center_reject_margin,
            "center_reject_radius": center_reject_radius,
            "center_keep_snr": center_keep_snr,
            "min_radius_px_hard": min_radius_px_hard,
            "axis_streak_mask_width": axis_streak_mask_width,
            "border_mask": border_mask,
            "radial_bin_width": radial_bin_width,
            "pair_tolerance_px": pair_tolerance_px,
            "max_radius_ratio": max_radius_ratio,
            "max_radius_px": max_radius_px,
            "rescue_max_radius_px": rescue_max_radius_px,
            "outside_max_radius_rejects": outside_max_radius_rejects,
            "max_candidates_for_refine": max_candidates_for_refine,
            "dedup_distance_px": dedup_distance_px,
            "dedup_radial_tol_px": dedup_radial_tol_px,
            "dedup_angle_tol_deg": dedup_angle_tol_deg,
            "allow_relaxed_fallback": allow_relaxed_fallback,
            "enable_lattice_consistency_filter": enable_lattice_consistency_filter,
            "lattice_coeff_tol": lattice_coeff_tol,
            "lattice_recon_tol_px": lattice_recon_tol_px,
            "lattice_filter_min_keep": lattice_filter_min_keep,
            "auto_center_cluster_filter": auto_center_cluster_filter,
            "background_subtraction_method": bg_mode,
            "gaussian_background_sigma": gaussian_sigma,
            "fft_compression": compression,
            "subpixel_refine": subpixel_refine,
            "subpixel_window_radius": max(1, subpixel_window_radius),
            "subpixel_max_shift": max(0.0, subpixel_max_shift),
            "max_gaussian_refine_count": max(0, max_gaussian_refine_count),
            "rescue_on_empty": rescue_on_empty,
            "enable_inner_pair_recovery": enable_inner_pair_recovery,
            "inner_pair_min_snr": inner_pair_min_snr,
            "inner_pair_recover_radius_px": inner_pair_recover_radius_px,
            "inner_pair_max_pairs": max(0, inner_pair_max_pairs),
            "ensure_multiring_coverage": ensure_multiring_coverage,
            "inner_ring_recover_pairs": max(0, inner_ring_recover_pairs),
            "inner_ring_recover_min_snr": inner_ring_recover_min_snr,
            "enable_basis_completion": enable_basis_completion,
            "basis_completion_search_radius_px": basis_completion_search_radius_px,
            "basis_completion_min_snr": basis_completion_min_snr,
        },
        "lattice_consistency_meta": lattice_meta,
        "auto_center_cluster_cutoff_px": center_cluster_cutoff,
        "candidate_peaks_xy": [{"x": c["x"], "y": c["y"]} for c in candidates],
        "accepted_peaks_xy": [{"x": p["refined_x"], "y": p["refined_y"]} for p in accepted],
        "rejected_peaks_xy": rejected,
        "paired_segments": [
            {
                "x1": float(details[i]["refined_x"]),
                "y1": float(details[i]["refined_y"]),
                "x2": float(details[j]["refined_x"]),
                "y2": float(details[j]["refined_y"]),
                "pair_id": details[i].get("pair_id"),
            }
            for i in range(len(details))
            for j in range(i + 1, len(details))
            if details[i].get("pair_id") is not None
            and details[i].get("pair_id") == details[j].get("pair_id")
        ],
        "residual_distribution": [
            float(row.get("pair_error_px"))
            for row in details
            if row.get("pair_error_px") is not None
        ],
    }
    if bool(parameters.get("return_peak_debug", False)):
        debug["raw_fft_magnitude"] = magnitude.astype(np.float32).tolist()
        debug["log_fft"] = compressed.astype(np.float32).tolist()
        debug["background_subtracted_fft"] = background_sub.astype(np.float32).tolist()
        debug["radial_z_score_map"] = radial_z.astype(np.float32).tolist()
        debug["masks"] = {
            "center_mask": masks["center_mask"].astype(np.uint8).tolist(),
            "axis_mask": masks["axis_mask"].astype(np.uint8).tolist(),
            "border_mask": masks["border_mask"].astype(np.uint8).tolist(),
            "combined_mask": masks["combined_mask"].astype(np.uint8).tolist(),
        }
    if pair_count < 3:
        warnings.append(
            "Peak pairing insufficient: detected fewer than 3 Friedel pairs (±g)."
        )
    return details, debug, warnings


def run_diffraction_analysis(
    preprocessing: PreprocessingResult,
    calibration: CalibrationResult,
    *,
    parameters: dict[str, Any] | None = None,
) -> DiffractionResult:
    """Run robust FFT and diffraction analysis compatible with DiffractionResult."""
    parameters = parameters or {}
    warnings = list(preprocessing.warnings) + list(calibration.warnings)
    image = _load_preprocessed_array(preprocessing)
    if image is None or image.size == 0:
        warnings.append("Diffraction analysis skipped because image pixels were unavailable.")
        return DiffractionResult(
            confidence=0.2,
            warnings=warnings,
            provenance={"method": "a2_baseline_no_pixels"},
        )

    window_mode = str(parameters.get("fft_window", "none")).strip().lower()
    tukey_alpha = float(parameters.get("fft_window_tukey_alpha", 0.25))
    window = _fft_window(image.shape, window_mode, tukey_alpha=tukey_alpha)
    fft = np.fft.fftshift(np.fft.fft2(image * window))
    magnitude = np.abs(fft).astype(np.float32)

    top_k = int(parameters.get("top_k_peaks", 24))
    min_peak_distance = float(parameters.get("min_peak_distance", 3.0))
    center_exclusion_radius = float(parameters.get("center_exclusion_radius", 3.0))
    indexing_mode = str(parameters.get("indexing_mode", "auto")).strip().lower()
    peak_threshold_strategy = str(parameters.get("peak_threshold_strategy", "hybrid"))
    peak_threshold_value_raw = parameters.get("peak_threshold_value")
    peak_threshold_value = (
        None if peak_threshold_value_raw is None else float(peak_threshold_value_raw)
    )
    subpixel_refine = bool(parameters.get("subpixel_refine", True))
    subpixel_window_radius = int(parameters.get("subpixel_window_radius", 2))
    subpixel_max_shift = float(parameters.get("subpixel_max_shift", 1.5))
    max_hkl = int(parameters.get("max_hkl", 4))
    d_tolerance_rel = float(parameters.get("d_spacing_tolerance_rel", 0.08))
    angle_tolerance_deg = float(parameters.get("angle_tolerance_deg", 12.0))
    cif_candidates_dir = parameters.get("cif_candidates_dir")
    candidate_phase_params = parameters.get("candidate_phase_params")
    phase_top_n = int(parameters.get("phase_top_n", 3))
    candidate_phases_raw = parameters.get("candidate_phases")
    min_pair_count_for_strict = int(parameters.get("min_pair_count_for_strict", 3))
    origin_g_norm_threshold = float(parameters.get("origin_g_norm_threshold", 0.12))
    origin_hkl_g_max = float(parameters.get("origin_hkl_g_max", 0.30))
    origin_filter_max_radius_px = float(parameters.get("origin_filter_max_radius_px", 20.0))
    allowed_phase_names = None
    if isinstance(candidate_phases_raw, list):
        allowed_phase_names = {str(item) for item in candidate_phases_raw}

    detect_params = dict(parameters)
    detect_params.setdefault("max_peaks", max(1, top_k))
    detect_params.setdefault("min_distance", max(0.0, min_peak_distance))
    detect_params.setdefault("center_mask_radius", max(0.0, center_exclusion_radius))
    detect_params.setdefault("peak_threshold_strategy", peak_threshold_strategy)
    detect_params.setdefault("peak_threshold_value", peak_threshold_value)
    detect_params.setdefault("subpixel_refine", subpixel_refine)
    detect_params.setdefault("subpixel_window_radius", max(1, subpixel_window_radius))
    detect_params.setdefault("subpixel_max_shift", max(0.0, subpixel_max_shift))
    if "allow_relaxed_fallback" not in detect_params:
        legacy_relaxed = {"auto", "hybrid", "relative_max", "quantile", "std", "mad"}
        detect_params["allow_relaxed_fallback"] = (
            str(peak_threshold_strategy).strip().lower() in legacy_relaxed
        )

    peak_details, peak_debug, detect_warnings = _detect_peaks_v2(
        magnitude,
        parameters=detect_params,
    )
    warnings.extend(detect_warnings)

    h, w = magnitude.shape
    cy, cx = h / 2.0, w / 2.0
    fft_min_dim = float(min(h, w))
    pixel_nm = 0.1
    if calibration.pixel_size is not None and calibration.pixel_size.unit == "nm":
        pixel_nm = float(calibration.pixel_size.value)
    reciprocal_scale = 1.0 / (fft_min_dim * max(pixel_nm, 1e-9))
    reciprocal_scale_source = "derived_from_pixel_size"
    if calibration.diffraction_scale is not None:
        scale_value = calibration.diffraction_scale.value
        try:
            candidate_scale = float(scale_value)
        except (TypeError, ValueError):
            warnings.append(
                "Invalid diffraction_scale value in calibration; fallback to pixel-size scale."
            )
        else:
            if candidate_scale <= 0:
                warnings.append(
                    "Non-positive diffraction_scale in calibration; fallback to pixel-size scale."
                )
            else:
                scale_method = calibration.diffraction_scale.method or ""
                if scale_method == "reciprocal_pixel_size":
                    reciprocal_scale = candidate_scale / fft_min_dim
                    reciprocal_scale_source = (
                        "calibration.diffraction_scale.reciprocal_pixel_size_normalized"
                    )
                else:
                    reciprocal_scale = candidate_scale
                    reciprocal_scale_source = "calibration.diffraction_scale"

    reciprocal_scale_uncertainty = None
    if calibration.diffraction_scale is not None:
        try:
            if calibration.diffraction_scale.uncertainty is not None:
                reciprocal_scale_uncertainty = float(calibration.diffraction_scale.uncertainty)
        except (TypeError, ValueError):
            reciprocal_scale_uncertainty = None

    peaks: list[DiffractionPeak] = []
    g_vectors: list[GVector] = []
    d_spacings: list[Measurement] = []
    indexed_spots: list[IndexedSpot] = []
    gxy: list[tuple[float, float]] = []
    peak_intensities: list[float] = []
    d_by_peak: list[Measurement | None] = []
    for i, detail in enumerate(peak_details):
        x_refined = float(detail["refined_x"])
        y_refined = float(detail["refined_y"])
        snr = float(detail["snr"])
        paired = bool(detail["paired"])
        peak_warning = str(detail.get("warning") or "")
        peak_conf = 1.0 - np.exp(-max(0.0, snr) / 6.0)
        if not paired:
            peak_conf *= 0.75
        if peak_warning:
            peak_conf *= 0.8
        peak_conf = float(np.clip(peak_conf, 0.05, 0.99))
        peak_warnings = [w for w in peak_warning.split(";") if w]
        peaks.append(
            DiffractionPeak(
                x=x_refined,
                y=y_refined,
                intensity=float(detail["intensity"]),
                confidence=peak_conf,
                warnings=peak_warnings,
            )
        )
        peak_intensities.append(float(detail["intensity"]))
        dx = float(x_refined - cx)
        dy = float(y_refined - cy)
        gx_val = dx * reciprocal_scale
        gy_val = dy * reciprocal_scale
        detail["g_x"] = gx_val
        detail["g_y"] = gy_val
        gxy.append((gx_val, gy_val))
        g_mag = float(np.hypot(gx_val, gy_val))
        detail["g_norm"] = g_mag
        g_mag_uncertainty = None
        if reciprocal_scale_uncertainty is not None:
            pixel_radius = float(np.hypot(dx, dy))
            g_mag_uncertainty = pixel_radius * reciprocal_scale_uncertainty
        angle_deg = float(np.degrees(np.arctan2(gy_val, gx_val)))
        g_vectors.append(
            GVector(
                gx=Measurement(value=gx_val, unit="1/nm", method="fft_pixel_offset"),
                gy=Measurement(value=gy_val, unit="1/nm", method="fft_pixel_offset"),
                magnitude=Measurement(
                    value=g_mag,
                    unit="1/nm",
                    method="vector_norm",
                    uncertainty=g_mag_uncertainty,
                ),
                angle=Measurement(value=angle_deg, unit="deg", method="atan2"),
                source_peak_index=i,
            )
        )
        d_measurement: Measurement | None = None
        if g_mag > 1e-8:
            d_spacing_nm = 1.0 / g_mag
            detail["d_spacing"] = d_spacing_nm
            d_uncertainty = None
            if g_mag_uncertainty is not None:
                d_uncertainty = g_mag_uncertainty / max(g_mag * g_mag, 1e-16)
            d_measurement = Measurement(
                value=d_spacing_nm,
                unit="nm",
                method=f"inverse_g_magnitude[{reciprocal_scale_source}]",
                uncertainty=d_uncertainty,
            )
            d_spacings.append(d_measurement)
        else:
            detail["d_spacing"] = None
        d_by_peak.append(d_measurement)

    indexed_by_lattice: list[tuple[int, int, int, float, float]] = []
    indexing_meta: dict[str, Any] = {"indexing_method": "baseline"}
    if indexing_mode in {"auto", "lattice_fit"}:
        indexed_by_lattice, indexing_meta = _index_spots_with_lattice_fit(gxy, peak_intensities)
        if not indexed_by_lattice and indexing_mode == "lattice_fit":
            warnings.append(
                "Lattice-fit indexing requested but basis selection failed; fallback to baseline."
            )

    for i, peak in enumerate(peaks):
        d_measurement = d_by_peak[i] if i < len(d_by_peak) else None
        if indexed_by_lattice:
            h_idx, k_idx, ell, residual, conf = indexed_by_lattice[i]
            indexed_spots.append(
                IndexedSpot(
                    peak_index=i,
                    h=h_idx,
                    k=k_idx,
                    l=ell,
                    d_spacing=d_measurement,
                    error=Measurement(
                        value=residual,
                        unit="1/nm",
                        method="lattice_fit_residual",
                    ),
                    confidence=conf,
                )
            )
            peak_details[i]["h"] = h_idx
            peak_details[i]["k"] = k_idx
            peak_details[i]["l"] = ell
            peak_details[i]["residual_error"] = residual
            peak_details[i]["confidence"] = conf
        else:
            dx = float(peak.x - cx)
            dy = float(peak.y - cy)
            hkl = _make_index(dx, dy)
            indexed_spots.append(
                IndexedSpot(
                    peak_index=i,
                    h=hkl[0],
                    k=hkl[1],
                    l=hkl[2],
                    d_spacing=d_measurement,
                    confidence=0.4,
                )
            )
            peak_details[i]["h"] = hkl[0]
            peak_details[i]["k"] = hkl[1]
            peak_details[i]["l"] = hkl[2]
            peak_details[i]["residual_error"] = None
            peak_details[i]["confidence"] = 0.4

    (
        peaks,
        g_vectors,
        indexed_spots,
        d_by_peak,
        peak_intensities,
        peak_details,
        removed_origin_like_count,
    ) = _drop_origin_like_spots(
        peaks=peaks,
        g_vectors=g_vectors,
        indexed_spots=indexed_spots,
        d_by_peak=d_by_peak,
        peak_intensities=peak_intensities,
        peak_details=peak_details,
        g_norm_threshold=max(0.0, origin_g_norm_threshold),
        hkl_origin_g_max=max(0.0, origin_hkl_g_max),
        center_xy=(cx, cy),
        origin_filter_max_radius_px=max(0.0, origin_filter_max_radius_px),
    )
    if removed_origin_like_count > 0:
        warnings.append(
            "Removed "
            f"{removed_origin_like_count} origin-like spots "
            "(hkl=000 or very small g_norm)."
        )

    d_spacings = [item for item in d_by_peak if item is not None]

    phase_candidates = _phase_candidates_from_d_spacings(
        [float(d.value) for d in d_spacings],
        allowed_phase_names=allowed_phase_names,
        top_n=phase_top_n,
    )

    pair_count = int(peak_debug.get("counts", {}).get("pair_count", 0))
    if indexing_mode == "b4_strict" and pair_count < max(1, min_pair_count_for_strict):
        warnings.append(
            "peak_detection_insufficient: fewer than required ±g pairs; strict indexing skipped."
        )
        indexing_meta["b4"] = {
            "active": False,
            "reason": "peak_detection_insufficient",
            "required_pair_count": max(1, min_pair_count_for_strict),
            "detected_pair_count": pair_count,
        }
    else:
        b4_requested = indexing_mode in {"b4_mvp", "b4_strict"} or (
            candidate_phase_params is not None and indexing_mode != "baseline"
        )
        if b4_requested:
            b4_result = run_b4_indexing(
                peaks=peaks,
                g_vectors=g_vectors,
                d_by_peak=d_by_peak,
                initial_indexed_spots=indexed_spots,
                candidate_phase_params=candidate_phase_params,
                allowed_phase_names=allowed_phase_names,
                max_hkl=max(1, max_hkl),
                d_tolerance_rel=max(1e-4, d_tolerance_rel),
                angle_tolerance_deg=max(0.1, angle_tolerance_deg),
                strict_mode=indexing_mode == "b4_strict",
                cif_candidates_dir=None if cif_candidates_dir is None else str(cif_candidates_dir),
                top_n=phase_top_n,
            )
            warnings.extend(b4_result.warnings)
            if b4_result.phase_candidates:
                phase_candidates = b4_result.phase_candidates
            if b4_result.indexed_spots:
                indexed_spots = b4_result.indexed_spots
            indexing_meta["b4"] = b4_result.provenance
            if b4_result.provenance.get("active"):
                indexing_meta["indexing_method"] = str(b4_result.provenance.get("mode"))

    preview = _compress_fft(magnitude, "log1p")
    preview /= float(preview.max() + 1e-8)
    preview_ref = ArrayReference(
        shape=tuple(int(axis) for axis in preview.shape),
        dtype=str(preview.dtype),
        uri=preprocessing.image.uri if preprocessing.image is not None else None,
        inline=preview[: min(8, h), : min(8, w)].tolist(),
        description="Log-scaled FFT magnitude preview",
    )

    confidence = 0.2 + min(0.45, len(peaks) / 50.0)
    if pair_count >= 3:
        confidence += 0.15
    if phase_candidates:
        confidence += 0.1
    confidence = float(np.clip(confidence, 0.0, 1.0))

    return DiffractionResult(
        peaks=peaks,
        center=(cx, cy),
        g_vectors=g_vectors,
        d_spacings=d_spacings,
        indexed_spots=indexed_spots,
        phase_candidates=phase_candidates,
        fft_image=preview_ref,
        confidence=confidence,
        warnings=warnings,
        provenance={
            "method": "a2_b2_fft_peak_detection_v2",
            "used_parameters": {
                "top_k_peaks": max(1, top_k),
                "max_peaks": int(detect_params.get("max_peaks", max(1, top_k))),
                "min_peak_distance": max(0.0, min_peak_distance),
                "min_snr": float(detect_params.get("min_snr", 1.2)),
                "center_exclusion_radius": max(0.0, center_exclusion_radius),
                "center_mask_radius": float(detect_params.get("center_mask_radius", 0.0)),
                "axis_streak_mask_width": float(detect_params.get("axis_streak_mask_width", 0.0)),
                "border_mask": float(detect_params.get("border_mask", 2.0)),
                "radial_bin_width": float(detect_params.get("radial_bin_width", 2.0)),
                "indexing_mode": indexing_mode,
                "peak_threshold_strategy": peak_threshold_strategy,
                "peak_threshold_value": peak_threshold_value,
                "subpixel_refine": subpixel_refine,
                "subpixel_window_radius": max(1, subpixel_window_radius),
                "subpixel_max_shift": max(0.0, subpixel_max_shift),
                "fft_window": window_mode,
                "fft_window_tukey_alpha": float(np.clip(tukey_alpha, 0.0, 1.0)),
                "fft_compression": str(detect_params.get("fft_compression", "log1p")),
                "clip_low_percentile": float(detect_params.get("clip_low_percentile", 1.0)),
                "clip_high_percentile": float(detect_params.get("clip_high_percentile", 99.8)),
                "background_subtraction_method": str(
                    detect_params.get("background_subtraction_method", "radial_mad")
                ),
                "gaussian_background_sigma": float(
                    detect_params.get("gaussian_background_sigma", 12.0)
                ),
                "pair_tolerance_px": float(
                    detect_params.get("pair_tolerance_px", max(2.0, min_peak_distance * 1.5))
                ),
                "min_pair_count_for_strict": max(1, min_pair_count_for_strict),
                "origin_g_norm_threshold": max(0.0, origin_g_norm_threshold),
                "origin_hkl_g_max": max(0.0, origin_hkl_g_max),
                "origin_filter_max_radius_px": max(0.0, origin_filter_max_radius_px),
                "max_hkl": max(1, max_hkl),
                "d_spacing_tolerance_rel": max(1e-4, d_tolerance_rel),
                "angle_tolerance_deg": max(0.1, angle_tolerance_deg),
                "cif_candidates_dir": (
                    None if cif_candidates_dir is None else str(cif_candidates_dir)
                ),
                "phase_top_n": phase_top_n,
                "candidate_phases": (
                    sorted(allowed_phase_names) if allowed_phase_names is not None else None
                ),
                "candidate_phase_params_count": (
                    len(candidate_phase_params) if isinstance(candidate_phase_params, list) else 0
                ),
            },
            "calibration_scale_source": reciprocal_scale_source,
            "calibration_scale_inv_nm_per_px": reciprocal_scale,
            "indexing": indexing_meta,
            "peak_detection": peak_debug,
        },
    )
