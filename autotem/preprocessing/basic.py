"""Baseline preprocessing and quality wrappers for A2/B1 workflow steps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from autotem._paths import io_path
from autotem.core import ArrayReference, PreprocessingResult, QualityAssessment, TEMDataset
from autotem.preprocessing.background import subtract_background
from autotem.preprocessing.denoise import apply_denoise
from autotem.preprocessing.quality import assess_image_quality, load_dataset_array


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re

        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value.strip())
        if match:
            return float(match.group(0))
    return None


def _to_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _preprocessing_enabled(params: dict[str, Any]) -> bool:
    for key in ("enabled", "apply_preprocessing", "denoise_enabled"):
        if key in params:
            return _to_bool(params.get(key), default=True)
    return True


def _raw_passthrough_result(
    dataset: TEMDataset,
    *,
    quality_after: QualityAssessment | None,
    recipe: str,
    parameters: dict[str, Any],
) -> PreprocessingResult:
    image = load_dataset_array(dataset)
    if image is None:
        return PreprocessingResult(
            image=None,
            operations=[],
            parameters={**parameters, "recipe": recipe, "enabled": False},
            quality_after=quality_after,
            provenance={
                "method": "preprocess_dataset",
                "recipe": recipe,
                "mode": "raw_passthrough",
            },
            warnings=["Input image is unavailable; raw passthrough returned empty result."],
        )

    image_ref = dataset.data.model_copy(deep=True)
    image_ref.description = "Raw input image array; B2 preprocessing and denoise disabled."
    return PreprocessingResult(
        image=image_ref,
        operations=["raw_passthrough"],
        parameters={
            **parameters,
            "recipe": recipe,
            "enabled": False,
            "image_source": "raw_input",
        },
        quality_after=quality_after,
        provenance={
            "method": "preprocess_dataset",
            "recipe": recipe,
            "mode": "raw_passthrough",
            "source_path": dataset.source_path,
        },
        warnings=[],
    )


def assess_quality(
    dataset: TEMDataset,
    *,
    parameters: dict[str, Any] | None = None,
) -> QualityAssessment:
    """Workflow wrapper for B1 quality assessment."""
    return assess_image_quality(dataset, parameters=parameters)


def _safe_percentile_clip(
    image: np.ndarray,
    *,
    low: float,
    high: float,
) -> tuple[np.ndarray, float, float]:
    p_low = float(np.percentile(image, low))
    p_high = float(np.percentile(image, high))
    if p_high <= p_low + 1e-12:
        p_high = p_low + 1e-12
    clipped = np.clip(image, p_low, p_high)
    return clipped, p_low, p_high


def _normalize_unit_interval(image: np.ndarray) -> np.ndarray:
    min_v = float(np.min(image))
    max_v = float(np.max(image))
    if max_v < min_v + 1e-12:
        return np.zeros_like(image, dtype=np.float32)
    return np.asarray((image - min_v) / (max_v - min_v), dtype=np.float32)


def _build_output_uri(
    dataset: TEMDataset, output_dir: str | Path | None = None,
) -> Path | None:
    if output_dir is not None:
        target = Path(output_dir) / "preprocessed_image.npy"
        if dataset.source_path and target.resolve() == Path(dataset.source_path).resolve():
            raise ValueError("Preprocessing output must not overwrite its raw input.")
        return target
    source_path = dataset.source_path.strip()
    if not source_path:
        return None
    source = Path(source_path)
    if io_path(source).is_file():
        return source.with_name(f"{source.stem}_preprocessed.npy")
    return None


def preprocess_dataset(
    dataset: TEMDataset,
    *,
    quality_after: QualityAssessment | None = None,
    recipe: str = "auto",
    parameters: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
) -> PreprocessingResult:
    """Preprocess and optionally persist inside a run-specific artifact directory.

    Omitting ``output_dir`` retains the legacy source-adjacent storage convention.
    Workflow callers should provide an isolated run directory.
    """
    params = dict(parameters or {})
    warnings: list[str] = []
    operations: list[str] = []

    if not _preprocessing_enabled(params):
        return _raw_passthrough_result(
            dataset,
            quality_after=quality_after,
            recipe=recipe,
            parameters=params,
        )

    image = load_dataset_array(dataset)
    if image is None:
        warnings.append("Input image is unavailable; preprocessing returned empty result.")
        return PreprocessingResult(
            image=None,
            operations=[],
            parameters=params,
            quality_after=quality_after,
            provenance={"method": "preprocess_dataset", "recipe": recipe},
            warnings=warnings,
        )

    arr = np.asarray(image, dtype=np.float64)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, copy=False)
        warnings.append("Input contains non-finite pixels; converted via nan_to_num.")

    low = _to_float(params.get("clip_percentile_low"))
    high = _to_float(params.get("clip_percentile_high"))
    if low is None:
        low = 1.0
    if high is None:
        high = 99.0
    if not (0.0 <= low < high <= 100.0):
        warnings.append(
            "Invalid clip percentiles in preprocessing parameters; fallback to [1.0, 99.0]."
        )
        low, high = 1.0, 99.0

    clipped, p_low, p_high = _safe_percentile_clip(arr, low=low, high=high)
    processed = _normalize_unit_interval(clipped)
    operations.extend(["clip_percentile", "normalize_unit_interval"])

    denoise_sigma = _to_float(params.get("denoise_sigma"))
    if denoise_sigma is None:
        denoise_sigma = 0.8 if str(recipe).strip().lower() == "auto" else 0.0
    denoise_method = str(params.get("denoise_method", "auto"))
    median_size = int(_to_float(params.get("median_size")) or 3)
    wiener_window = int(_to_float(params.get("wiener_window")) or 5)
    wiener_noise = _to_float(params.get("wiener_noise"))
    nlm_h = _to_float(params.get("nlm_h"))
    nlm_patch_size = int(_to_float(params.get("nlm_patch_size")) or 5)
    nlm_patch_distance = int(_to_float(params.get("nlm_patch_distance")) or 6)

    processed, denoise_effective, denoise_diag, denoise_warnings = apply_denoise(
        processed,
        method=denoise_method,
        sigma=denoise_sigma,
        median_size=median_size,
        wiener_window=wiener_window,
        wiener_noise=wiener_noise,
        nlm_h=nlm_h,
        nlm_patch_size=nlm_patch_size,
        nlm_patch_distance=nlm_patch_distance,
    )
    operations.append(f"denoise:{denoise_effective}")
    warnings.extend(denoise_warnings)

    background_sigma = _to_float(params.get("background_sigma"))
    if background_sigma is None:
        background_sigma = 0.0
    background_method_raw = params.get("background_method")
    if background_method_raw is None and background_sigma > 0:
        background_method = "gaussian"
    else:
        background_method = str(background_method_raw or "none")
    background_opening_size = int(_to_float(params.get("background_opening_size")) or 31)
    background_clip_zero = bool(params.get("background_clip_zero", False))
    processed, background, background_effective, background_diag, background_warnings = (
        subtract_background(
            processed,
            method=background_method,
            gaussian_sigma=background_sigma if background_sigma > 0 else 12.0,
            opening_size=background_opening_size,
            clip_to_zero=background_clip_zero,
        )
    )
    warnings.extend(background_warnings)
    if background_effective != "none":
        operations.append(f"background:{background_effective}")
    processed = _normalize_unit_interval(processed)

    background_ref: ArrayReference | None = None
    if background_effective != "none":
        background_ref = ArrayReference(
            shape=tuple(int(v) for v in background.shape),
            dtype=str(background.dtype),
            inline=(background.tolist() if background.size <= 4096 else None),
            description="Estimated low-frequency background.",
        )

    output_uri = _build_output_uri(dataset, output_dir)
    if output_uri is not None:
        try:
            target = io_path(output_uri)
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, processed)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Failed to persist preprocessed array to '{output_uri}': {exc}")
            output_uri = None
    else:
        warnings.append("Preprocessed array URI was not generated; using inline payload only.")

    image_ref = ArrayReference(
        shape=tuple(int(v) for v in processed.shape),
        dtype=str(processed.dtype),
        uri=(str(output_uri) if output_uri is not None else None),
        inline=(processed.tolist() if processed.size <= 4096 else None),
        description="Preprocessed image array.",
    )

    quality_out = quality_after
    if quality_out is None:
        temp_dataset = TEMDataset(
            data=image_ref,
            dataset_id=dataset.dataset_id,
            source_path=(str(output_uri) if output_uri is not None else dataset.source_path),
            modality=dataset.modality,
            pixel_size=dataset.pixel_size,
            pixel_unit=dataset.pixel_unit,
            metadata=dataset.metadata,
            provenance=dataset.provenance,
            warnings=dataset.warnings,
        )
        quality_out = assess_image_quality(temp_dataset, parameters={"thresholds": params})
        operations.append("reassess_quality")

    return PreprocessingResult(
        image=image_ref,
        background=background_ref,
        operations=operations,
        parameters={
            **params,
            "recipe": recipe,
            "clip_percentile_low_effective": low,
            "clip_percentile_high_effective": high,
            "clip_value_low": p_low,
            "clip_value_high": p_high,
            "denoise_sigma_effective": denoise_sigma,
            "denoise_method_effective": denoise_effective,
            "denoise_diagnostics": denoise_diag,
            "background_sigma_effective": background_sigma,
            "background_method_effective": background_effective,
            "background_diagnostics": background_diag,
        },
        quality_after=quality_out,
        provenance={"method": "preprocess_dataset", "recipe": recipe},
        warnings=warnings,
    )
