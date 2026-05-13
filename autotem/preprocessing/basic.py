"""A2 baseline image quality and preprocessing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from autotem.core import (
    ArrayReference,
    Measurement,
    PreprocessingResult,
    QualityAssessment,
    TEMDataset,
)


def _load_array(dataset: TEMDataset) -> np.ndarray | None:
    uri = dataset.data.uri
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


def assess_quality(dataset: TEMDataset) -> QualityAssessment:
    """Return simple but deterministic image-quality metrics for A2 workflows."""
    image = _load_array(dataset)
    warnings = list(dataset.warnings)
    if image is None or image.size == 0:
        warnings.append("Quality metrics are approximate because pixel data was not loaded.")
        return QualityAssessment(
            overall_score=0.3,
            metadata_completeness=0.4 if dataset.metadata else 0.2,
            confidence=0.35,
            warnings=warnings,
            provenance={"method": "a2_baseline_no_pixels"},
        )

    finite = np.isfinite(image)
    valid = image[finite]
    if valid.size == 0:
        warnings.append("Image contains no finite values.")
        return QualityAssessment(
            overall_score=0.1,
            confidence=0.1,
            warnings=warnings,
            provenance={"method": "a2_baseline_invalid_pixels"},
        )

    std = float(np.std(valid))
    mean = float(np.mean(valid))
    snr_value = abs(mean) / (std + 1e-8)
    grad_y, grad_x = np.gradient(valid.reshape(image.shape))
    blur_value = float(np.var(grad_x) + np.var(grad_y))
    metadata_completeness = 0.7 if dataset.metadata else 0.3
    quality_score = max(
        0.0,
        min(1.0, 0.3 * np.tanh(snr_value / 3.0) + 0.5 + 0.2 * np.tanh(blur_value)),
    )

    return QualityAssessment(
        overall_score=quality_score,
        snr=Measurement(value=snr_value, unit="a.u.", method="mean_over_std"),
        blur=Measurement(value=blur_value, unit="a.u.", method="gradient_variance"),
        metadata_completeness=metadata_completeness,
        confidence=0.6,
        warnings=warnings,
        provenance={"method": "a2_baseline"},
    )


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

    if recipe not in {"auto", "min_max"}:
        warnings.append(
            f"Unsupported preprocess recipe '{recipe}'; fallback to min-max normalization."
        )

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
