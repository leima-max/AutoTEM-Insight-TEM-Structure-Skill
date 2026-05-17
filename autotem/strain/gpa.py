"""Baseline strain mapping with selectable GPA and PPA-style methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import griddata

from autotem.core import (
    ArrayReference,
    DiffractionResult,
    GVector,
    Measurement,
    StrainResult,
)
from autotem.io import read_image

ImageInput = np.ndarray | str | Path
GVectorInput = list[tuple[float, float]]


def run_gpa(
    image: ImageInput,
    *,
    method: str = "gpa_b3",
    diffraction_result: DiffractionResult | None = None,
    g_vectors: GVectorInput | None = None,
    g_vector_unit: str = "1/nm",
    pixel_size_nm: float | None = None,
    reference_region: tuple[int, int, int, int] | None = None,
    gpa_mask_sigma_px: float = 8.0,
    ppa_window_radius: int = 8,
    ppa_grid_step: int = 6,
    map_inline_mode: str = "preview",
    include_debug_maps: bool = False,
) -> StrainResult:
    """
    Run strain analysis with two selectable methods.

    Methods
    -------
    - `gpa_b3`: GPA using B3-recommended g-vectors (or user-specified g-vectors).
    - `ppa_ai_stem`: PPA-style local phase-correlation baseline inspired by AI-STEM workflows.
    """
    method = method.lower().strip()
    if method not in {"gpa_b3", "ppa_ai_stem"}:
        raise ValueError("method must be one of {'gpa_b3', 'ppa_ai_stem'}.")
    map_inline_mode = map_inline_mode.lower().strip()
    if map_inline_mode not in {"preview", "full"}:
        raise ValueError("map_inline_mode must be one of {'preview', 'full'}.")

    warnings: list[str] = []
    arr = _load_image_2d(image, warnings=warnings)
    ref = _validate_reference_region(reference_region, arr.shape, warnings)

    if method == "gpa_b3":
        ux, uy, used_g_vectors, method_meta = _run_gpa_b3(
            arr,
            diffraction_result=diffraction_result,
            g_vectors=g_vectors,
            g_vector_unit=g_vector_unit,
            pixel_size_nm=pixel_size_nm,
            reference_region=ref,
            mask_sigma_px=gpa_mask_sigma_px,
            warnings=warnings,
            include_debug_maps=include_debug_maps,
        )
    else:
        ux, uy, used_g_vectors, method_meta = _run_ppa_ai_stem(
            arr,
            reference_region=ref,
            window_radius=ppa_window_radius,
            grid_step=ppa_grid_step,
            warnings=warnings,
        )

    strain_xx, strain_yy, strain_xy, rotation_xy = _strain_from_displacement(ux, uy)
    confidence = _estimate_confidence(
        ux=ux,
        uy=uy,
        warning_count=len(warnings),
        method=method,
    )

    map_stats = {
        "mean_ux": float(np.nanmean(ux)),
        "mean_uy": float(np.nanmean(uy)),
        "mean_exx": float(np.nanmean(strain_xx)),
        "mean_eyy": float(np.nanmean(strain_yy)),
        "mean_exy": float(np.nanmean(strain_xy)),
        "std_exx": float(np.nanstd(strain_xx)),
        "std_eyy": float(np.nanstd(strain_yy)),
        "std_exy": float(np.nanstd(strain_xy)),
    }

    return StrainResult(
        displacement_x=_to_array_ref(
            ux,
            description=f"Displacement ux map from method={method}.",
            inline_mode=map_inline_mode,
        ),
        displacement_y=_to_array_ref(
            uy,
            description=f"Displacement uy map from method={method}.",
            inline_mode=map_inline_mode,
        ),
        strain_xx=_to_array_ref(
            strain_xx,
            description=f"Strain exx map from method={method}.",
            inline_mode=map_inline_mode,
        ),
        strain_yy=_to_array_ref(
            strain_yy,
            description=f"Strain eyy map from method={method}.",
            inline_mode=map_inline_mode,
        ),
        strain_xy=_to_array_ref(
            strain_xy,
            description=f"Strain exy map from method={method}.",
            inline_mode=map_inline_mode,
        ),
        rotation_xy=_to_array_ref(
            rotation_xy,
            description=f"Rotation wxy map from method={method}.",
            inline_mode=map_inline_mode,
        ),
        uncertainty=_to_array_ref(
            np.sqrt(strain_xx**2 + strain_yy**2 + 2.0 * strain_xy**2).astype(np.float32),
            description="Heuristic strain-magnitude uncertainty proxy.",
            inline_mode=map_inline_mode,
        ),
        reference_region=ref,
        g_vectors=used_g_vectors,
        confidence=confidence,
        provenance={
            "source": "autotem.strain.gpa.run_gpa",
            "method": method,
            "parameters": {
                "g_vector_unit": g_vector_unit,
                "pixel_size_nm": pixel_size_nm,
                "gpa_mask_sigma_px": gpa_mask_sigma_px,
                "ppa_window_radius": ppa_window_radius,
                "ppa_grid_step": ppa_grid_step,
                "map_inline_mode": map_inline_mode,
                "include_debug_maps": include_debug_maps,
            },
            "method_metadata": method_meta,
            "map_stats": map_stats,
        },
        warnings=warnings,
    )


def _load_image_2d(image: ImageInput, warnings: list[str]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        arr = np.asarray(image, dtype=np.float64)
    else:
        path = Path(image)
        if path.suffix.lower() in {".npy"}:
            arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
        elif path.suffix.lower() in {".txt", ".csv"}:
            arr = np.asarray(np.loadtxt(path), dtype=np.float64)
        else:
            arr = np.asarray(read_image(path).array, dtype=np.float64)

    if arr.ndim < 2:
        raise ValueError(f"Input image must be at least 2D, got shape={arr.shape}.")
    if arr.ndim > 2:
        warnings.append(f"Input shape {arr.shape} is >2D; using first slice for strain analysis.")
        while arr.ndim > 2:
            arr = arr[0]

    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("Input image contains no finite values.")
    if not finite.all():
        fill = float(np.nanmedian(arr[finite]))
        arr = np.where(finite, arr, fill)

    arr = arr.astype(np.float64, copy=False)
    arr = arr - float(np.mean(arr))
    std = float(np.std(arr))
    if std > 1e-12:
        arr /= std
    return arr


def _validate_reference_region(
    reference_region: tuple[int, int, int, int] | None,
    shape: tuple[int, int],
    warnings: list[str],
) -> tuple[int, int, int, int]:
    h, w = shape
    if reference_region is None:
        y0, y1 = h // 4, (3 * h) // 4
        x0, x1 = w // 4, (3 * w) // 4
        return (y0, y1, x0, x1)

    y0, y1, x0, x1 = [int(v) for v in reference_region]
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, y0 + 1, h))
    x0 = int(np.clip(x0, 0, w - 1))
    x1 = int(np.clip(x1, x0 + 1, w))
    if (y0, y1, x0, x1) != reference_region:
        warnings.append("Reference region was clipped to image bounds.")
    return (y0, y1, x0, x1)


def _run_gpa_b3(
    image: np.ndarray,
    *,
    diffraction_result: DiffractionResult | None,
    g_vectors: GVectorInput | None,
    g_vector_unit: str,
    pixel_size_nm: float | None,
    reference_region: tuple[int, int, int, int],
    mask_sigma_px: float,
    warnings: list[str],
    include_debug_maps: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[GVector], dict[str, Any]]:
    if mask_sigma_px <= 0:
        raise ValueError("gpa_mask_sigma_px must be > 0.")

    g_vecs_px, used_g_objects = _select_two_g_vectors(
        diffraction_result=diffraction_result,
        g_vectors=g_vectors,
        g_vector_unit=g_vector_unit,
        pixel_size_nm=pixel_size_nm,
        warnings=warnings,
    )
    g1, g2 = g_vecs_px

    phi1_raw, mask1 = _extract_phase_map(image, g1, mask_sigma_px=mask_sigma_px)
    phi2_raw, mask2 = _extract_phase_map(image, g2, mask_sigma_px=mask_sigma_px)
    phi1 = phi1_raw.copy()
    phi2 = phi2_raw.copy()
    yy, xx = np.indices(image.shape, dtype=np.float64)
    phi1 = phi1 - (2.0 * np.pi * (g1[0] * xx + g1[1] * yy))
    phi2 = phi2 - (2.0 * np.pi * (g2[0] * xx + g2[1] * yy))
    phi1 = _unwrap_phase_2d(phi1)
    phi2 = _unwrap_phase_2d(phi2)

    y0, y1, x0, x1 = reference_region
    phi1 = phi1 - float(np.mean(phi1[y0:y1, x0:x1]))
    phi2 = phi2 - float(np.mean(phi2[y0:y1, x0:x1]))

    g_mat = np.array([[g1[0], g1[1]], [g2[0], g2[1]]], dtype=np.float64)
    det = float(np.linalg.det(g_mat))
    if abs(det) < 1e-8:
        raise ValueError("Selected g-vectors are near-collinear and cannot solve displacement.")

    inv_g = np.linalg.inv(g_mat)
    coef = 1.0 / (2.0 * np.pi)
    ux = coef * (inv_g[0, 0] * phi1 + inv_g[0, 1] * phi2)
    uy = coef * (inv_g[1, 0] * phi1 + inv_g[1, 1] * phi2)

    method_meta: dict[str, Any] = {
        "selected_g_vectors_pixel": [
            {"gx": float(g1[0]), "gy": float(g1[1])},
            {"gx": float(g2[0]), "gy": float(g2[1])},
        ],
        "determinant": det,
        "solver": "phase_unwrap + linear_inversion",
    }
    if include_debug_maps:
        method_meta["debug_maps"] = {
            "phase1_raw_rad": phi1_raw.astype(np.float32).tolist(),
            "phase2_raw_rad": phi2_raw.astype(np.float32).tolist(),
            "phase1_unwrapped_rad": phi1.astype(np.float32).tolist(),
            "phase2_unwrapped_rad": phi2.astype(np.float32).tolist(),
            "mask1": mask1.astype(np.float32).tolist(),
            "mask2": mask2.astype(np.float32).tolist(),
        }

    return (
        ux.astype(np.float32),
        uy.astype(np.float32),
        used_g_objects,
        method_meta,
    )


def _run_ppa_ai_stem(
    image: np.ndarray,
    *,
    reference_region: tuple[int, int, int, int],
    window_radius: int,
    grid_step: int,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray, list[GVector], dict[str, Any]]:
    if window_radius < 3:
        raise ValueError("ppa_window_radius must be >= 3.")
    if grid_step < 2:
        raise ValueError("ppa_grid_step must be >= 2.")

    h, w = image.shape
    y0, y1, x0, x1 = reference_region
    cy = int((y0 + y1) / 2)
    cx = int((x0 + x1) / 2)
    ref_patch = _extract_patch(image, cy, cx, window_radius)
    if ref_patch is None:
        raise ValueError("Reference patch for ppa_ai_stem is out of image bounds.")

    ys = np.arange(window_radius, h - window_radius, grid_step, dtype=int)
    xs = np.arange(window_radius, w - window_radius, grid_step, dtype=int)

    sample_points: list[tuple[float, float]] = []
    ux_samples: list[float] = []
    uy_samples: list[float] = []
    corr_scores: list[float] = []

    for y in ys:
        for x in xs:
            patch = _extract_patch(image, int(y), int(x), window_radius)
            if patch is None:
                continue
            dy, dx, score = _phase_correlation_shift(ref_patch, patch)
            sample_points.append((float(y), float(x)))
            uy_samples.append(float(dy))
            ux_samples.append(float(dx))
            corr_scores.append(float(score))

    if len(sample_points) < 12:
        warnings.append(
            "ppa_ai_stem sampled too few patches; using dense median-filter fallback displacement."
        )
        uy = ndi.median_filter(image, size=5) - image
        ux = np.zeros_like(uy)
        return (
            ux.astype(np.float32),
            uy.astype(np.float32),
            [],
            {"solver": "fallback_median", "sample_count": len(sample_points)},
        )

    points = np.asarray(sample_points, dtype=np.float64)
    yy, xx = np.indices(image.shape, dtype=np.float64)
    ux = griddata(points[:, ::-1], np.asarray(ux_samples), (xx, yy), method="linear")
    uy = griddata(points[:, ::-1], np.asarray(uy_samples), (xx, yy), method="linear")

    if np.isnan(ux).any() or np.isnan(uy).any():
        ux_nn = griddata(points[:, ::-1], np.asarray(ux_samples), (xx, yy), method="nearest")
        uy_nn = griddata(points[:, ::-1], np.asarray(uy_samples), (xx, yy), method="nearest")
        ux = np.where(np.isnan(ux), ux_nn, ux)
        uy = np.where(np.isnan(uy), uy_nn, uy)

    ux = ndi.gaussian_filter(np.asarray(ux, dtype=np.float64), sigma=1.0)
    uy = ndi.gaussian_filter(np.asarray(uy, dtype=np.float64), sigma=1.0)

    # Set reference mean displacement to zero.
    ref_ux = float(np.mean(ux[y0:y1, x0:x1]))
    ref_uy = float(np.mean(uy[y0:y1, x0:x1]))
    ux -= ref_ux
    uy -= ref_uy

    return (
        ux.astype(np.float32),
        uy.astype(np.float32),
        [],
        {
            "solver": "phase_correlation_grid",
            "sample_count": len(sample_points),
            "corr_score_mean": float(np.mean(corr_scores)),
            "window_radius": int(window_radius),
            "grid_step": int(grid_step),
        },
    )


def _extract_patch(image: np.ndarray, y: int, x: int, r: int) -> np.ndarray | None:
    y0, y1 = y - r, y + r + 1
    x0, x1 = x - r, x + r + 1
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        return None
    patch = image[y0:y1, x0:x1]
    return patch if patch.size > 0 else None


def _phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fb * np.conj(fa)
    cross /= np.abs(cross) + 1e-12
    corr = np.abs(np.fft.ifft2(cross))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    py, px = int(peak[0]), int(peak[1])
    h, w = corr.shape
    if py > h // 2:
        py -= h
    if px > w // 2:
        px -= w
    return float(py), float(px), float(corr.max())


def _select_two_g_vectors(
    *,
    diffraction_result: DiffractionResult | None,
    g_vectors: GVectorInput | None,
    g_vector_unit: str,
    pixel_size_nm: float | None,
    warnings: list[str],
) -> tuple[list[tuple[float, float]], list[GVector]]:
    candidates_px: list[tuple[float, float]] = []
    used_g_objects: list[GVector] = []

    if g_vectors is not None:
        for gx, gy in g_vectors:
            gx_px, gy_px = _to_cycles_per_pixel(
                float(gx),
                float(gy),
                unit=g_vector_unit,
                pixel_size_nm=pixel_size_nm,
                warnings=warnings,
            )
            candidates_px.append((gx_px, gy_px))
            used_g_objects.append(
                GVector(
                    gx=Measurement(value=float(gx), unit=g_vector_unit, method="user_input"),
                    gy=Measurement(value=float(gy), unit=g_vector_unit, method="user_input"),
                    magnitude=Measurement(
                        value=float(np.hypot(gx, gy)),
                        unit=g_vector_unit,
                        method="vector_norm",
                    ),
                )
            )
    elif diffraction_result is not None:
        for gv in diffraction_result.g_vectors:
            gx_val = float(gv.gx.value) if gv.gx.value is not None else 0.0
            gy_val = float(gv.gy.value) if gv.gy.value is not None else 0.0
            gx_px, gy_px = _to_cycles_per_pixel(
                gx_val,
                gy_val,
                unit=gv.gx.unit or "1/nm",
                pixel_size_nm=pixel_size_nm,
                warnings=warnings,
            )
            candidates_px.append((gx_px, gy_px))
            used_g_objects.append(gv)
    else:
        raise ValueError("gpa_b3 requires diffraction_result or g_vectors.")

    if len(candidates_px) < 2:
        raise ValueError("Need at least two g-vectors for GPA.")

    idx1, idx2 = _best_non_collinear_pair(candidates_px)
    g1 = candidates_px[idx1]
    g2 = candidates_px[idx2]
    selected = [g1, g2]
    selected_objs = [used_g_objects[idx1], used_g_objects[idx2]]

    for idx, g in enumerate(selected):
        mag = float(np.hypot(g[0], g[1]))
        if mag > 0.5:
            warnings.append(
                f"Selected g-vector #{idx} magnitude={mag:.3f} cycles/pixel is high; "
                "verify units/pixel_size."
            )
    return selected, selected_objs


def _best_non_collinear_pair(vectors: list[tuple[float, float]]) -> tuple[int, int]:
    norms = np.array([np.hypot(v[0], v[1]) for v in vectors], dtype=np.float64)
    order = np.argsort(norms)[::-1]
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            a = vectors[int(order[i])]
            b = vectors[int(order[j])]
            cross = abs(a[0] * b[1] - a[1] * b[0])
            if cross > 1e-6:
                return int(order[i]), int(order[j])
    return int(order[0]), int(order[1])


def _to_cycles_per_pixel(
    gx: float,
    gy: float,
    *,
    unit: str,
    pixel_size_nm: float | None,
    warnings: list[str],
) -> tuple[float, float]:
    unit_norm = unit.lower().replace(" ", "")
    if unit_norm in {"1/pixel", "pixel^-1", "1px", "1/px", "px^-1", "cycle/pixel"}:
        return gx, gy
    if unit_norm in {"1/nm", "nm^-1"}:
        if pixel_size_nm is None or pixel_size_nm <= 0:
            raise ValueError(
                "g-vectors are in 1/nm but pixel_size_nm is missing or invalid. "
                "Provide a positive pixel_size_nm for GPA."
            )
        return gx * pixel_size_nm, gy * pixel_size_nm
    warnings.append(f"Unsupported g-vector unit {unit!r}; treating as 1/pixel.")
    return gx, gy


def _extract_phase_map(
    image: np.ndarray,
    g_vec: tuple[float, float],
    mask_sigma_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape
    gy, gx = float(g_vec[1]), float(g_vec[0])
    fft = np.fft.fftshift(np.fft.fft2(image))
    yy, xx = np.indices((h, w), dtype=np.float64)
    py = (h / 2.0) + gy * h
    px = (w / 2.0) + gx * w
    mask = np.exp(-(((yy - py) ** 2 + (xx - px) ** 2) / (2.0 * mask_sigma_px**2)))
    filtered = np.fft.ifft2(np.fft.ifftshift(fft * mask))
    return np.angle(filtered), mask


def _unwrap_phase_2d(phase: np.ndarray) -> np.ndarray:
    return np.unwrap(np.unwrap(phase, axis=0), axis=1)


def _strain_from_displacement(
    ux: np.ndarray,
    uy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    duy_dy, duy_dx = np.gradient(uy)
    dux_dy, dux_dx = np.gradient(ux)
    strain_xx = dux_dx.astype(np.float32)
    strain_yy = duy_dy.astype(np.float32)
    strain_xy = (0.5 * (dux_dy + duy_dx)).astype(np.float32)
    rotation_xy = (0.5 * (duy_dx - dux_dy)).astype(np.float32)
    return strain_xx, strain_yy, strain_xy, rotation_xy


def _to_array_ref(
    arr: np.ndarray,
    *,
    description: str,
    inline_mode: str = "preview",
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


def _estimate_confidence(
    *,
    ux: np.ndarray,
    uy: np.ndarray,
    warning_count: int,
    method: str,
) -> float:
    finite_ratio = float(np.mean(np.isfinite(ux) & np.isfinite(uy)))
    disp_std = float(np.nanstd(ux) + np.nanstd(uy))
    score = 0.5 + 0.35 * finite_ratio
    score += 0.08 if method == "gpa_b3" else 0.03
    score -= min(0.1 * warning_count, 0.3)
    score -= 0.05 if disp_std > 20 else 0.0
    return float(np.clip(score, 0.0, 1.0))
