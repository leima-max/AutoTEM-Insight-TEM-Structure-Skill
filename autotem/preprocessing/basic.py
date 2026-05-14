"""A2 baseline image quality and preprocessing helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from autotem.core import (
    ArrayReference,
    PreprocessingResult,
    QualityAssessment,
    TEMDataset,
)
from autotem.preprocessing.quality import assess_image_quality, load_dataset_array


def _load_array(dataset: TEMDataset) -> np.ndarray | None:
    return load_dataset_array(dataset)


def assess_quality(
    dataset: TEMDataset,
    *,
    parameters: dict[str, Any] | None = None,
) -> QualityAssessment:
    """Return deterministic B1 quality metrics with optional threshold overrides."""
    return assess_image_quality(dataset, parameters=parameters)


def preprocess_dataset(
    dataset: TEMDataset,
    quality_after: QualityAssessment | None = None,
    *,
    recipe: str = "auto",
    parameters: dict[str, Any] | None = None,
) -> PreprocessingResult:
    """Normalize image intensities for downstream FFT and peak detection."""
    parameters = parameters or {}
    image = _load_array(dataset)
    warnings = list(dataset.warnings)
    if image is None or image.size == 0:
        warnings.append("Preprocessing skipped because pixel data was not loaded.")
        return PreprocessingResult(
            image=dataset.data,
            operations=["pass_through"],
            parameters={"recipe": recipe},
            quality_after=quality_after,
            warnings=warnings,
            provenance={"method": "a2_baseline_noop"},
        )

    clip_low = parameters.get("clip_percentile_low")
    clip_high = parameters.get("clip_percentile_high")
    working = image.astype(np.float32)
    operations: list[str] = []
    if clip_low is not None or clip_high is not None:
        lo_pct = float(clip_low if clip_low is not None else 0.0)
        hi_pct = float(clip_high if clip_high is not None else 100.0)
        if not (0.0 <= lo_pct < hi_pct <= 100.0):
            warnings.append("Invalid clip percentiles; clipping step skipped.")
        else:
            lo_clip = float(np.percentile(working, lo_pct))
            hi_clip = float(np.percentile(working, hi_pct))
            working = np.clip(working, lo_clip, hi_clip)
            operations.append(f"clip_percentile({lo_pct},{hi_pct})")

    if recipe not in {"auto", "min_max", "b2_fft"}:
        warnings.append(
            f"Unsupported preprocess recipe '{recipe}'; fallback to min-max normalization."
        )

    # B2 extension: deterministic denoise + background subtraction.
    apply_b2_ops = recipe == "b2_fft"
    denoise_sigma_raw = parameters.get("denoise_sigma")
    background_sigma_raw = parameters.get("background_sigma")
    if denoise_sigma_raw is not None or background_sigma_raw is not None:
        apply_b2_ops = True

    if apply_b2_ops:
        denoise_sigma = 1.0 if denoise_sigma_raw is None else float(denoise_sigma_raw)
        background_sigma = (
            12.0 if background_sigma_raw is None else float(background_sigma_raw)
        )

        if denoise_sigma > 0:
            working = ndi.gaussian_filter(working, sigma=denoise_sigma).astype(np.float32)
            operations.append(f"gaussian_denoise(sigma={denoise_sigma:.3f})")
        else:
            warnings.append("denoise_sigma <= 0; denoise step skipped.")

        if background_sigma > 0:
            background = ndi.gaussian_filter(working, sigma=background_sigma).astype(np.float32)
            working = (working - background).astype(np.float32)
            operations.append(f"background_subtract_gaussian(sigma={background_sigma:.3f})")
        else:
            warnings.append("background_sigma <= 0; background subtraction skipped.")

    lo = float(np.min(working))
    hi = float(np.max(working))
    if np.isclose(hi, lo):
        normalized = np.zeros_like(working, dtype=np.float32)
        warnings.append("Input image has near-constant intensity.")
    else:
        normalized = ((working - lo) / (hi - lo)).astype(np.float32)
    operations.append("min_max_normalize")

    preview = normalized[: min(8, normalized.shape[0]), : min(8, normalized.shape[1])]
    data_ref = ArrayReference(
        shape=tuple(int(axis) for axis in normalized.shape),
        dtype=str(normalized.dtype),
        uri=dataset.data.uri,
        inline=preview.tolist(),
        description="Min-max normalized image preview",
    )

    return PreprocessingResult(
        image=data_ref,
        operations=operations,
        parameters={
            "input_min": lo,
            "input_max": hi,
            "recipe": recipe,
            "requested_parameters": parameters,
        },
        quality_after=quality_after,
        warnings=warnings,
        provenance={"method": "a2_baseline", "recipe": recipe},
    )
