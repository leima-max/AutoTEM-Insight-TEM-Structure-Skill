"""Diagnostics for C2 GPA parameter drift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from autotem.strain.gpa import DEFAULT_GPA_MASK_SIGMA_PX, GPAFieldResult, compute_gpa_fields

_STRAIN_FIELDS = {"exx", "exy", "eyx", "eyy", "shear_exy"}
_COMPARE_FIELDS = ("exx", "exy", "eyx", "eyy", "shear_exy", "wxy", "ux", "uy")


@dataclass(frozen=True, slots=True)
class GPAAlignmentConfig:
    """All inputs that should be held constant for a GPA alignment check."""

    g_vectors: tuple[tuple[float, float], tuple[float, float]]
    g_vector_unit: str = "1/pixel"
    pixel_size_nm: float | None = None
    reference_region: tuple[int, int, int, int] | None = None
    mask_sigma_px: float = DEFAULT_GPA_MASK_SIGMA_PX
    hann_window: bool = False
    refine_reference: bool = True
    min_quality_fraction: float = 0.08
    quality_percentile: float = 10.0
    apply_quality_mask: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "g_vectors": [[float(x), float(y)] for x, y in self.g_vectors],
            "g_vector_unit": self.g_vector_unit,
            "pixel_size_nm": self.pixel_size_nm,
            "reference_region": (
                list(self.reference_region) if self.reference_region is not None else None
            ),
            "mask_sigma_px": float(self.mask_sigma_px),
            "hann_window": bool(self.hann_window),
            "refine_reference": bool(self.refine_reference),
            "min_quality_fraction": float(self.min_quality_fraction),
            "quality_percentile": float(self.quality_percentile),
            "apply_quality_mask": bool(self.apply_quality_mask),
        }


def compare_gpa_configs(
    image: np.ndarray,
    reference_config: GPAAlignmentConfig,
    candidate_config: GPAAlignmentConfig,
    *,
    backend: str = "gpa",
    sample_step: int = 1,
) -> dict[str, Any]:
    """Run one backend with two parameter sets and compare the result fields."""

    reference = _run_backend(image, reference_config, backend)
    candidate = _run_backend(image, candidate_config, backend)
    return _compare_field_results(
        reference,
        candidate,
        label_a=f"{backend}:reference_config",
        label_b=f"{backend}:candidate_config",
        config_a=reference_config.as_dict(),
        config_b=candidate_config.as_dict(),
        sample_step=sample_step,
    )


def _run_backend(image: np.ndarray, config: GPAAlignmentConfig, backend: str) -> GPAFieldResult:
    method = _backend_to_provenance_method(backend)
    return compute_gpa_fields(
        np.asarray(image, dtype=np.float32),
        g_vectors=[(float(x), float(y)) for x, y in config.g_vectors],
        g_vector_unit=config.g_vector_unit,
        pixel_size_nm=config.pixel_size_nm,
        reference_region=config.reference_region,
        mask_sigma_px=float(config.mask_sigma_px),
        hann_window=bool(config.hann_window),
        refine_reference=bool(config.refine_reference),
        min_quality_fraction=float(config.min_quality_fraction),
        quality_percentile=float(config.quality_percentile),
        apply_quality_mask=bool(config.apply_quality_mask),
        warnings=[],
        provenance_method=method,
    )


def _backend_to_provenance_method(backend: str) -> str:
    key = str(backend or "gpa").strip().casefold()
    if key in {"autotem", "autotem c2", "gpa"}:
        return "gpa"
    if key == "gpa_b3":
        return "gpa_b3"
    raise ValueError("backend must be 'gpa', 'gpa_b3', or 'AutoTEM C2'.")


def _compare_field_results(
    fields_a: GPAFieldResult,
    fields_b: GPAFieldResult,
    *,
    label_a: str,
    label_b: str,
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    sample_step: int,
) -> dict[str, Any]:
    step = max(1, int(sample_step))
    valid_a = np.asarray(fields_a.quality_mask, dtype=bool)[::step, ::step]
    valid_b = np.asarray(fields_b.quality_mask, dtype=bool)[::step, ::step]
    compare_mask = valid_a & valid_b
    if not np.any(compare_mask):
        compare_mask = np.ones_like(valid_a, dtype=bool)

    arrays_a = _field_arrays(fields_a)
    arrays_b = _field_arrays(fields_b)
    field_stats: dict[str, Any] = {}
    for name in _COMPARE_FIELDS:
        scale = 100.0 if name in _STRAIN_FIELDS else 1.0
        unit = "%" if name in _STRAIN_FIELDS else ("rad" if name == "wxy" else "px")
        a = np.asarray(arrays_a[name], dtype=np.float64)[::step, ::step]
        b = np.asarray(arrays_b[name], dtype=np.float64)[::step, ::step]
        field_stats[name] = _compare_arrays(a, b, compare_mask, scale=scale, unit=unit)

    strain_maes = [
        float(field_stats[name]["diff"]["mae"])
        for name in ("exx", "exy", "eyx", "eyy", "shear_exy")
    ]
    strain_corrs = [
        float(field_stats[name]["corr"])
        for name in ("exx", "exy", "eyx", "eyy", "shear_exy")
        if np.isfinite(float(field_stats[name]["corr"]))
    ]
    max_strain_mae = max(strain_maes) if strain_maes else float("nan")
    min_strain_corr = min(strain_corrs) if strain_corrs else float("nan")
    return {
        "label_a": label_a,
        "label_b": label_b,
        "config_a": config_a,
        "config_b": config_b,
        "sample_step": step,
        "quality": {
            "valid_fraction_a": float(np.mean(valid_a)),
            "valid_fraction_b": float(np.mean(valid_b)),
            "intersection_fraction": float(np.mean(compare_mask)),
            "iou": _mask_iou(valid_a, valid_b),
        },
        "phase_masks": {
            "a": _phase_mask_summary(fields_a, step=step),
            "b": _phase_mask_summary(fields_b, step=step),
        },
        "fields": field_stats,
        "diagnosis": {
            "max_strain_mae_percent": float(max_strain_mae),
            "min_strain_corr": float(min_strain_corr),
            "configs_aligned": bool(
                np.isfinite(max_strain_mae)
                and max_strain_mae <= 0.05
                and (not np.isfinite(min_strain_corr) or min_strain_corr >= 0.98)
            ),
        },
    }


def _field_arrays(fields: GPAFieldResult) -> dict[str, np.ndarray]:
    exy = np.asarray(fields.strain_xy, dtype=np.float64)
    wxy = np.asarray(fields.rotation_xy, dtype=np.float64)
    return {
        "ux": np.asarray(fields.displacement_x, dtype=np.float64),
        "uy": np.asarray(fields.displacement_y, dtype=np.float64),
        "exx": np.asarray(fields.strain_xx, dtype=np.float64),
        "exy": exy - wxy,
        "eyx": exy + wxy,
        "eyy": np.asarray(fields.strain_yy, dtype=np.float64),
        "shear_exy": exy,
        "wxy": wxy,
    }


def _compare_arrays(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray,
    *,
    scale: float,
    unit: str,
) -> dict[str, Any]:
    aa = np.asarray(a, dtype=np.float64)[mask]
    bb = np.asarray(b, dtype=np.float64)[mask]
    diff = bb - aa
    return {
        "unit": unit,
        "a": _summary(aa, scale=scale),
        "b": _summary(bb, scale=scale),
        "diff": _diff_summary(diff, scale=scale),
        "corr": _corr(aa, bb),
    }


def _summary(values: np.ndarray, *, scale: float) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {key: float("nan") for key in ("mean", "std", "p01", "p50", "p99")}
    return {
        "mean": float(np.mean(vals) * scale),
        "std": float(np.std(vals) * scale),
        "p01": float(np.percentile(vals, 1) * scale),
        "p50": float(np.percentile(vals, 50) * scale),
        "p99": float(np.percentile(vals, 99) * scale),
    }


def _diff_summary(values: np.ndarray, *, scale: float) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {key: float("nan") for key in ("mean", "mae", "rmse", "p99_abs")}
    return {
        "mean": float(np.mean(vals) * scale),
        "mae": float(np.mean(np.abs(vals)) * scale),
        "rmse": float(np.sqrt(np.mean(vals * vals)) * scale),
        "p99_abs": float(np.percentile(np.abs(vals), 99) * scale),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(aa) & np.isfinite(bb)
    aa = aa[finite]
    bb = bb[finite]
    if aa.size < 2 or float(np.std(aa)) == 0.0 or float(np.std(bb)) == 0.0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = a | b
    if not np.any(union):
        return float("nan")
    return float(np.mean(a & b) / np.mean(union))


def _phase_mask_summary(fields: GPAFieldResult, *, step: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for phase in fields.phase_maps:
        mask = np.asarray(phase.gaussian_mask, dtype=np.float64)[::step, ::step]
        center_x, center_y = _weighted_center(mask)
        out.append(
            {
                "g_vector_fft_pixels": [float(v) for v in phase.g_vector_pixels],
                "sampled_mask_center_xy": [center_x, center_y],
                "mask_max": float(np.nanmax(mask)) if mask.size else float("nan"),
            }
        )
    return out


def _weighted_center(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - np.nanmin(arr)
    max_value = float(np.nanmax(arr)) if arr.size else 0.0
    if max_value <= 0:
        return float("nan"), float("nan")
    yy, xx = np.indices(arr.shape, dtype=np.float64)
    total = float(arr.sum())
    return float((xx * arr).sum() / total), float((yy * arr).sum() / total)
