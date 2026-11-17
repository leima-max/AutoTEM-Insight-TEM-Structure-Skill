"""Subpixel atom-column refinement utilities (C1 baseline)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares


@dataclass
class RefinementResult:
    coordinates_input: np.ndarray
    coordinates_refined: np.ndarray
    displacements: np.ndarray
    method: str
    parameters: dict[str, Any]
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]


def refine_atom_coordinates(
    image: np.ndarray,
    coordinates: np.ndarray,
    method: str = "gaussian",
    window_radius: int = 3,
    max_shift: float = 1.5,
    max_iter: int = 200,
) -> RefinementResult:
    """
    Refine atom coordinates to subpixel precision.

    Parameters
    ----------
    image:
        2D microscopy image.
    coordinates:
        Input coordinates in `[y, x]` convention.
    method:
        `gaussian` or `com`.
    window_radius:
        Half-size of local patch used for refinement.
    max_shift:
        Reject/refuse updates larger than this pixel shift.
    max_iter:
        Maximum nonlinear fit iterations for `gaussian`.
    """
    method = method.lower()
    if method not in {"gaussian", "com"}:
        raise ValueError("method must be one of {'gaussian', 'com'}.")
    if window_radius < 1:
        raise ValueError("window_radius must be >= 1.")
    if max_shift <= 0:
        raise ValueError("max_shift must be > 0.")

    image = np.asarray(image, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) in [y, x] order.")

    refined = coords.copy()
    warnings: list[str] = []
    success = 0
    rejected_large_shift = 0
    fit_failures = 0
    out_of_bounds_idx: list[int] = []
    fit_failure_idx: list[int] = []
    large_shift_examples: list[tuple[int, float]] = []
    point_status: list[str] = []

    for i, (y0, x0) in enumerate(coords):
        patch_info = _extract_patch(image=image, y=y0, x=x0, radius=window_radius)
        if patch_info is None:
            fit_failures += 1
            out_of_bounds_idx.append(i)
            point_status.append("out_of_bounds")
            continue

        patch, y_min, x_min = patch_info
        if method == "com":
            result = _refine_com(patch=patch, y_min=y_min, x_min=x_min)
        else:
            result = _refine_gaussian(
                patch=patch,
                y_min=y_min,
                x_min=x_min,
                y_guess=y0,
                x_guess=x0,
                max_iter=max_iter,
            )

        if result is None:
            fit_failures += 1
            fit_failure_idx.append(i)
            point_status.append("fit_failed")
            continue

        y_ref, x_ref = result
        shift = float(np.hypot(y_ref - y0, x_ref - x0))
        if shift > max_shift:
            rejected_large_shift += 1
            point_status.append("max_shift_rejected")
            if len(large_shift_examples) < 8:
                large_shift_examples.append((i, shift))
            continue

        refined[i, 0] = y_ref
        refined[i, 1] = x_ref
        success += 1
        point_status.append("accepted")

    displacements = refined - coords

    if out_of_bounds_idx:
        warnings.append(
            "Patch out-of-bounds points kept original: "
            f"{len(out_of_bounds_idx)} (examples: {out_of_bounds_idx[:8]})."
        )
    if fit_failure_idx:
        warnings.append(
            "Refinement fit failures kept original: "
            f"{len(fit_failure_idx)} (examples: {fit_failure_idx[:8]})."
        )
    if rejected_large_shift > 0:
        examples = ", ".join(
            f"idx={idx}, shift={shift:.2f}px" for idx, shift in large_shift_examples
        )
        warnings.append(
            "Rejected by max_shift: "
            f"{rejected_large_shift} points (max_shift={max_shift:.2f}px). "
            f"Examples: {examples}."
        )

    confidence = _refinement_confidence(
        n_points=len(coords),
        n_success=success,
        mean_shift=float(np.mean(np.linalg.norm(displacements, axis=1))) if len(coords) else 0.0,
        max_shift=max_shift,
        fit_failures=fit_failures,
    )

    return RefinementResult(
        coordinates_input=coords,
        coordinates_refined=refined,
        displacements=displacements,
        method=method,
        parameters={
            "window_radius": int(window_radius),
            "max_shift": float(max_shift),
            "max_iter": int(max_iter),
        },
        confidence=confidence,
        warnings=warnings,
        provenance={
            "source": "autotem.atomic.gaussian_refinement.refine_atom_coordinates",
            "success_count": success,
            "fit_failures": fit_failures,
            "rejected_large_shift": rejected_large_shift,
            "out_of_bounds_idx": out_of_bounds_idx,
            "fit_failure_idx": fit_failure_idx,
            "point_status": point_status,
            "input_shape": tuple(int(x) for x in image.shape),
        },
    )


def _extract_patch(
    image: np.ndarray, y: float, x: float, radius: int
) -> tuple[np.ndarray, int, int] | None:
    h, w = image.shape
    yc = int(round(y))
    xc = int(round(x))
    y_min = yc - radius
    y_max = yc + radius + 1
    x_min = xc - radius
    x_max = xc + radius + 1
    if y_min < 0 or x_min < 0 or y_max > h or x_max > w:
        return None
    patch = image[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return None
    return patch, y_min, x_min


def _refine_com(patch: np.ndarray, y_min: int, x_min: int) -> tuple[float, float] | None:
    baseline = float(np.percentile(patch, 10.0))
    weights = np.maximum(patch - baseline, 0.0)
    denom = float(np.sum(weights))
    if denom <= 1e-12:
        return None

    yy, xx = np.indices(patch.shape, dtype=np.float64)
    y_ref = float(np.sum(weights * (yy + y_min)) / denom)
    x_ref = float(np.sum(weights * (xx + x_min)) / denom)
    return y_ref, x_ref


def _refine_gaussian(
    patch: np.ndarray,
    y_min: int,
    x_min: int,
    y_guess: float,
    x_guess: float,
    max_iter: int,
) -> tuple[float, float] | None:
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    yy_abs = yy + y_min
    xx_abs = xx + x_min

    com = _refine_com(patch=patch, y_min=y_min, x_min=x_min)
    if com is not None:
        y_init, x_init = com
    else:
        y_init, x_init = y_guess, x_guess

    p_low = np.array(
        [
            0.0,
            y_min,
            x_min,
            0.4,
            float(np.min(patch) - abs(np.min(patch))),
        ],
        dtype=np.float64,
    )
    p_high = np.array(
        [
            float(np.max(patch) * 4.0 + 1e-6),
            y_min + patch.shape[0] - 1,
            x_min + patch.shape[1] - 1,
            float(max(patch.shape) * 2.5),
            float(np.max(patch) + abs(np.max(patch))),
        ],
        dtype=np.float64,
    )
    p_init = np.array(
        [
            float(np.max(patch) - np.min(patch) + 1e-6),
            float(y_init),
            float(x_init),
            1.2,
            float(np.median(patch)),
        ],
        dtype=np.float64,
    )
    p_init = np.clip(p_init, p_low, p_high)

    def residuals(params: np.ndarray) -> np.ndarray:
        amp, y0, x0, sigma, bg = params
        g = amp * np.exp(-((yy_abs - y0) ** 2 + (xx_abs - x0) ** 2) / (2.0 * sigma**2)) + bg
        return (g - patch).ravel()

    try:
        fit = least_squares(
            residuals,
            p_init,
            bounds=(p_low, p_high),
            method="trf",
            loss="soft_l1",
            f_scale=0.1,
            max_nfev=max_iter,
        )
    except Exception:
        return None

    if not fit.success:
        return None
    _, y_ref, x_ref, _, _ = fit.x
    return float(y_ref), float(x_ref)


def _refinement_confidence(
    n_points: int,
    n_success: int,
    mean_shift: float,
    max_shift: float,
    fit_failures: int,
) -> float:
    if n_points <= 0:
        return 0.0
    success_ratio = n_success / n_points
    shift_score = float(np.clip(1.0 - mean_shift / max_shift, 0.0, 1.0))
    penalty = min(fit_failures / n_points, 1.0) * 0.2
    score = 0.6 * success_ratio + 0.4 * shift_score - penalty
    return float(np.clip(score, 0.0, 1.0))
