"""A2 baseline FFT, peak extraction, indexing, and phase matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

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


def _detect_peaks(
    magnitude: np.ndarray,
    *,
    top_k: int = 24,
    min_distance: float = 3.0,
    center_exclusion_radius: float = 3.0,
    threshold: float = 0.0,
) -> list[tuple[float, float, float]]:
    if magnitude.size == 0:
        return []

    local_max = magnitude >= ndi.maximum_filter(magnitude, size=3, mode="nearest")
    candidate_mask = local_max & (magnitude >= float(threshold))
    candidate_idx = np.argwhere(candidate_mask)
    if candidate_idx.size == 0:
        return []

    candidate_values = magnitude[candidate_mask]
    sort_order = np.argsort(candidate_values)[::-1]
    sorted_idx = candidate_idx[sort_order]
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    selected: list[tuple[float, float, float]] = []
    for yx in sorted_idx:
        y = int(yx[0])
        x = int(yx[1])
        if np.hypot(y - cy, x - cx) < center_exclusion_radius:
            continue
        intensity = float(magnitude[y, x])
        if any(np.hypot(y - py, x - px) < min_distance for py, px, _ in selected):
            continue
        selected.append((float(y), float(x), intensity))
        if len(selected) >= top_k:
            break
    return selected


def _compute_peak_threshold(
    magnitude: np.ndarray,
    *,
    strategy: str,
    value: float | None,
) -> float:
    flat = magnitude.ravel().astype(np.float64)
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

    # "auto" and any unknown value fallback:
    # use the stricter of high quantile and robust MAD threshold.
    return max(quantile_995, median + 6.0 * mad)


def _refine_peak_subpixel(
    magnitude: np.ndarray,
    *,
    y: float,
    x: float,
    window_radius: int = 2,
    max_shift: float = 1.5,
) -> tuple[float, float]:
    yi = int(round(y))
    xi = int(round(x))
    h, w = magnitude.shape
    y0 = max(0, yi - window_radius)
    y1 = min(h, yi + window_radius + 1)
    x0 = max(0, xi - window_radius)
    x1 = min(w, xi + window_radius + 1)
    patch = magnitude[y0:y1, x0:x1]
    if patch.size == 0:
        return y, x

    weights = patch.astype(np.float64)
    min_val = float(np.min(weights))
    weights = weights - min_val
    if np.all(weights <= 0):
        return y, x
    weights += 1e-9

    yy, xx = np.mgrid[y0:y1, x0:x1]
    denom = float(np.sum(weights))
    y_refined = float(np.sum(yy * weights) / denom)
    x_refined = float(np.sum(xx * weights) / denom)

    dy = y_refined - y
    dx = x_refined - x
    shift = float(np.hypot(dy, dx))
    if shift > max_shift and shift > 1e-9:
        scale = max_shift / shift
        y_refined = y + dy * scale
        x_refined = x + dx * scale
    return y_refined, x_refined


def _make_index(dx: float, dy: float) -> tuple[int, int, int]:
    denom = max(abs(dx), abs(dy), 1.0)
    h = int(np.rint(dx / denom * 4))
    k = int(np.rint(dy / denom * 4))
    return h, k, 0


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


def run_diffraction_analysis(
    preprocessing: PreprocessingResult,
    calibration: CalibrationResult,
    *,
    parameters: dict[str, Any] | None = None,
) -> DiffractionResult:
    """Run a lightweight FFT and peak analysis compatible with DiffractionResult."""
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

    fft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.abs(fft).astype(np.float32)
    top_k = int(parameters.get("top_k_peaks", 24))
    min_peak_distance = float(parameters.get("min_peak_distance", 3.0))
    center_exclusion_radius = float(parameters.get("center_exclusion_radius", 3.0))
    indexing_mode = str(parameters.get("indexing_mode", "auto")).strip().lower()
    peak_threshold_strategy = str(parameters.get("peak_threshold_strategy", "auto"))
    peak_threshold_value_raw = parameters.get("peak_threshold_value")
    peak_threshold_value = (
        None if peak_threshold_value_raw is None else float(peak_threshold_value_raw)
    )
    subpixel_refine = bool(parameters.get("subpixel_refine", True))
    subpixel_window_radius = int(parameters.get("subpixel_window_radius", 2))
    subpixel_max_shift = float(parameters.get("subpixel_max_shift", 1.5))
    phase_top_n = int(parameters.get("phase_top_n", 3))
    candidate_phases_raw = parameters.get("candidate_phases")
    allowed_phase_names = None
    if isinstance(candidate_phases_raw, list):
        allowed_phase_names = {str(item) for item in candidate_phases_raw}
    peak_threshold = _compute_peak_threshold(
        magnitude,
        strategy=peak_threshold_strategy,
        value=peak_threshold_value,
    )
    peak_triplets = _detect_peaks(
        magnitude,
        top_k=max(1, top_k),
        min_distance=max(0.0, min_peak_distance),
        center_exclusion_radius=max(0.0, center_exclusion_radius),
        threshold=peak_threshold,
    )

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
                "Invalid diffraction_scale value in calibration; "
                "fallback to pixel-size-derived scale."
            )
        else:
            if candidate_scale <= 0:
                warnings.append(
                    "Non-positive diffraction_scale in calibration; "
                    "fallback to pixel-size-derived scale."
                )
            else:
                scale_method = calibration.diffraction_scale.method or ""
                # Legacy fallback calibration stores reciprocal pixel size (1/nm),
                # while diffraction analysis expects per-FFT-pixel scale (1/nm/pixel-bin).
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

    for i, (y, x, intensity) in enumerate(peak_triplets):
        y_refined = float(y)
        x_refined = float(x)
        if subpixel_refine:
            y_refined, x_refined = _refine_peak_subpixel(
                magnitude,
                y=y_refined,
                x=x_refined,
                window_radius=max(1, subpixel_window_radius),
                max_shift=max(0.0, subpixel_max_shift),
            )
        peak_conf = 0.75 if subpixel_refine else 0.65
        peaks.append(
            DiffractionPeak(
                x=float(x_refined),
                y=float(y_refined),
                intensity=intensity,
                confidence=peak_conf,
            )
        )
        peak_intensities.append(float(intensity))
        dx = float(x_refined - cx)
        dy = float(y_refined - cy)
        gx_val = dx * reciprocal_scale
        gy_val = dy * reciprocal_scale
        gxy.append((gx_val, gy_val))
        g_mag = float(np.hypot(gx_val, gy_val))
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
            h, k, ell, residual, conf = indexed_by_lattice[i]
            indexed_spots.append(
                IndexedSpot(
                    peak_index=i,
                    h=h,
                    k=k,
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

    phase_candidates = _phase_candidates_from_d_spacings(
        [float(d.value) for d in d_spacings],
        allowed_phase_names=allowed_phase_names,
        top_n=phase_top_n,
    )

    preview = np.log1p(magnitude)
    preview /= float(preview.max() + 1e-8)
    preview_ref = ArrayReference(
        shape=tuple(int(axis) for axis in preview.shape),
        dtype=str(preview.dtype),
        uri=preprocessing.image.uri if preprocessing.image is not None else None,
        inline=preview[: min(8, h), : min(8, w)].tolist(),
        description="Log-scaled FFT magnitude preview",
    )

    confidence = 0.25 + min(0.5, len(peaks) / 40.0)
    if phase_candidates:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence))

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
            "method": "a2_baseline_fft",
            "used_parameters": {
                "top_k_peaks": max(1, top_k),
                "min_peak_distance": max(0.0, min_peak_distance),
                "center_exclusion_radius": max(0.0, center_exclusion_radius),
                "indexing_mode": indexing_mode,
                "peak_threshold_strategy": peak_threshold_strategy,
                "peak_threshold_value": peak_threshold_value,
                "peak_threshold_used": peak_threshold,
                "subpixel_refine": subpixel_refine,
                "subpixel_window_radius": max(1, subpixel_window_radius),
                "subpixel_max_shift": max(0.0, subpixel_max_shift),
                "phase_top_n": phase_top_n,
                "candidate_phases": (
                    sorted(allowed_phase_names) if allowed_phase_names is not None else None
                ),
            },
            "calibration_scale_source": reciprocal_scale_source,
            "calibration_scale_inv_nm_per_px": reciprocal_scale,
            "indexing": indexing_meta,
        },
    )
