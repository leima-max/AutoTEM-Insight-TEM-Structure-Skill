"""B3 diffraction orchestration built on modular B2/B3 helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from autotem.core import (
    ArrayReference,
    CalibrationResult,
    DiffractionResult,
    IndexedSpot,
    Measurement,
    PhaseCandidate,
    PreprocessingResult,
)
from autotem.diffraction.b4_indexing import run_b4_indexing
from autotem.diffraction.fft import (
    build_fft_reference,
    compute_fft_analysis,
    persist_fft_array_near_source,
)
from autotem.diffraction.gvector import compute_gvectors_and_dspacing
from autotem.diffraction.open_source_adapters import detect_peaks_with_adapter
from autotem.diffraction.peak_detection import (
    detect_bragg_peaks,
    suggest_peak_parameter_defaults,
)
from autotem.diffraction.phase_lookup import PhaseLookupResult, lookup_phase_candidates

_SUPPORTED_IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re

        token = value.strip()
        if not token:
            return None
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", token)
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


def _optional_float(value: Any) -> float | None:
    parsed = _to_float(value)
    return parsed


def _optional_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _array_from_reference(reference: ArrayReference) -> np.ndarray | None:
    expected_shape = tuple(int(v) for v in reference.shape)
    if reference.inline is not None:
        inline = np.asarray(reference.inline, dtype=np.float64)
        if expected_shape and inline.size == int(np.prod(expected_shape)):
            inline = inline.reshape(expected_shape)
        if inline.ndim == 2:
            return inline
    if not reference.uri:
        return None
    path = Path(reference.uri)
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path, allow_pickle=False)
        return np.asarray(data, dtype=np.float64) if data.ndim == 2 else None
    if suffix in {".txt", ".csv"}:
        data = np.loadtxt(path)
        return np.asarray(data, dtype=np.float64) if data.ndim == 2 else None
    if suffix in _SUPPORTED_IMAGE_SUFFIXES:
        with Image.open(path) as handle:
            image = np.asarray(handle)
        if image.ndim == 3:
            image = image.mean(axis=2)
        return np.asarray(image, dtype=np.float64) if image.ndim == 2 else None
    return None


def _load_preprocessed_array(preprocessing: PreprocessingResult) -> np.ndarray | None:
    if preprocessing.image is None:
        return None
    array = _array_from_reference(preprocessing.image)
    if array is None:
        return None
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, copy=False)
    return np.asarray(array, dtype=np.float64)


def _baseline_indexed_spots(d_spacings: list[Measurement | None]) -> list[IndexedSpot]:
    indexed: list[IndexedSpot] = []
    for idx, d_spacing in enumerate(d_spacings):
        indexed.append(
            IndexedSpot(
                peak_index=idx,
                h=0,
                k=0,
                l=0,
                d_spacing=d_spacing,
                error=Measurement(
                    value=1.0,
                    unit="relative",
                    method="baseline_unindexed",
                ),
                confidence=0.1,
            )
        )
    return indexed


def _phase_placeholders(names: list[str], *, top_n: int) -> list[PhaseCandidate]:
    if not names:
        return []
    chosen = names[: max(1, int(top_n))]
    return [
        PhaseCandidate(
            name=name,
            source="candidate_phases",
            score=max(0.1, 0.35 - 0.04 * idx),
            matched_spot_count=0,
            mean_d_error=None,
            mean_angle_error=None,
            zone_axis=(0, 0, 1),
            lattice_parameters={},
            warnings=["Placeholder phase candidate; lattice parameters not supplied."],
        )
        for idx, name in enumerate(chosen)
    ]


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _resolve_phase_lookup_candidates(
    params: dict[str, Any],
    *,
    candidate_phases: list[str],
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    lookup_requested = _to_bool(params.get("phase_lookup_enabled"), default=False)
    lookup_materials = _as_string_list(params.get("phase_lookup_materials"))
    lookup_formula = params.get("phase_lookup_formula")
    lookup_elements = _as_string_list(params.get("phase_lookup_elements"))
    lookup_sources = _as_string_list(params.get("phase_lookup_sources"))
    lookup_mp_ids = _as_string_list(params.get("materials_project_ids"))
    lookup_mp_config_path = params.get("materials_project_config_path")
    lookup_mp_api_key = params.get("materials_project_api_key")
    mp_sort_by = str(params.get("materials_project_sort_by", "stability")).strip().lower()
    mp_include_deprecated = _to_bool(
        params.get("materials_project_include_deprecated"),
        default=False,
    )
    mp_filter_stable = _to_bool(
        params.get("materials_project_filter_stable"),
        default=False,
    )
    mp_max_energy = _optional_float(params.get("materials_project_max_energy_above_hull"))
    mp_prefer_experimental = _to_bool(
        params.get("materials_project_prefer_experimental"),
        default=True,
    )
    mp_candidate_pool_size = _optional_int(
        params.get("materials_project_candidate_pool_size")
    )

    if not lookup_requested:
        return [], [], {"enabled": False}
    if not lookup_materials and candidate_phases:
        lookup_materials = list(candidate_phases)
    formula = str(lookup_formula).strip() if isinstance(lookup_formula, str) else None
    limit_per_source = int(_to_float(params.get("phase_lookup_limit_per_source")) or 5)
    timeout_s = int(_to_float(params.get("phase_lookup_timeout_s")) or 8)

    try:
        result = lookup_phase_candidates(
            formula=formula,
            elements=lookup_elements,
            material_names=lookup_materials,
            sources=lookup_sources or None,
            limit_per_source=max(1, limit_per_source),
            timeout_s=max(1, timeout_s),
            materials_project_api_key=(
                lookup_mp_api_key.strip()
                if isinstance(lookup_mp_api_key, str) and lookup_mp_api_key.strip()
                else None
            ),
            materials_project_config_path=(
                str(lookup_mp_config_path)
                if isinstance(lookup_mp_config_path, str) and lookup_mp_config_path.strip()
                else None
            ),
            materials_project_ids=lookup_mp_ids,
            materials_project_sort_by=mp_sort_by,
            materials_project_include_deprecated=mp_include_deprecated,
            materials_project_filter_stable=mp_filter_stable,
            materials_project_max_energy_above_hull=mp_max_energy,
            materials_project_prefer_experimental=mp_prefer_experimental,
            materials_project_candidate_pool_size=mp_candidate_pool_size,
        )
    except Exception as exc:  # noqa: BLE001
        return [], [f"B4 phase lookup failed: {exc}"], {"enabled": True, "failed": True}
    if not isinstance(result, PhaseLookupResult):
        return [], ["B4 phase lookup returned an unexpected result type."], {
            "enabled": True,
            "failed": True,
        }
    provenance = {
        "enabled": True,
        "candidate_count": len(result.candidates),
        **result.provenance,
    }
    warnings = [f"B4 phase lookup: {warning}" for warning in result.warnings]
    if not result.candidates:
        warnings.append("B4 phase lookup returned no candidates.")
    return result.candidates, warnings, provenance


def _grade_fft_warnings(
    *,
    image_shape: tuple[int, int],
    log_magnitude: np.ndarray,
    peak_count: int,
    center_diagnostics: dict[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []
    contrast_std = float(np.std(log_magnitude))
    low_contrast_threshold = 0.08
    if contrast_std < low_contrast_threshold:
        warnings.append(
            "WARNING: Low FFT contrast detected; peak detection may be unstable "
            f"(std={contrast_std:.4f} < {low_contrast_threshold:.4f})."
        )

    if peak_count == 0:
        warnings.append("BLOCKING: No diffraction peaks detected after thresholding.")
    elif peak_count < 4:
        warnings.append(
            "WARNING: Few diffraction peaks detected "
            f"({peak_count}); indexing and phase matching confidence may be low."
        )

    if isinstance(center_diagnostics, dict):
        offset_px = _to_float(center_diagnostics.get("offset_px"))
        if offset_px is not None:
            h, w = image_shape
            severe = max(6.0, 0.20 * min(h, w))
            mild = max(3.0, 0.10 * min(h, w))
            if offset_px > severe:
                warnings.append(
                    "BLOCKING: Direct-beam center deviates strongly from geometric center "
                    f"(offset={offset_px:.2f}px > {severe:.2f}px)."
                )
            elif offset_px > mild:
                warnings.append(
                    "WARNING: Direct-beam center offset is large "
                    f"(offset={offset_px:.2f}px > {mild:.2f}px)."
                )
    return warnings


def _resolve_peak_params(
    *,
    image: np.ndarray,
    params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    auto_peak_tuning = _to_bool(params.get("auto_peak_tuning"), default=True)
    auto_defaults = suggest_peak_parameter_defaults(image) if auto_peak_tuning else {}

    raw_top_k = _to_float(params.get("top_k_peaks"))
    raw_min_distance = _to_float(params.get("min_peak_distance"))
    raw_center_exclusion_radius = _to_float(params.get("center_exclusion_radius"))
    raw_max_peak_radius = _to_float(params.get("max_peak_radius_px"))
    raw_detector_sigma = _to_float(params.get("detector_sigma"))
    raw_dog_sigma_ratio = _to_float(params.get("dog_sigma_ratio"))
    raw_threshold_value = _to_float(params.get("peak_threshold_value"))
    raw_subpixel_window_radius = _to_float(params.get("subpixel_window_radius"))
    raw_subpixel_max_shift = _to_float(params.get("subpixel_max_shift"))
    raw_radius_outlier_mad_factor = _to_float(params.get("radius_outlier_mad_factor"))
    raw_radius_outlier_filter = params.get("radius_outlier_filter")

    peak_params = {
        "top_k": (
            max(1, int(raw_top_k))
            if raw_top_k is not None
            else int(auto_defaults.get("top_k_peaks", 32))
        ),
        "min_distance": (
            max(1, int(raw_min_distance))
            if raw_min_distance is not None
            else int(auto_defaults.get("min_peak_distance", 3))
        ),
        "center_exclusion_radius": (
            max(0.0, float(raw_center_exclusion_radius))
            if raw_center_exclusion_radius is not None
            else float(
                auto_defaults.get(
                    "center_exclusion_radius",
                    max(2.0, min(image.shape) * 0.05),
                )
            )
        ),
        "max_peak_radius_px": (
            max(0.0, float(raw_max_peak_radius))
            if raw_max_peak_radius is not None
            else float(
                auto_defaults.get(
                    "max_peak_radius_px",
                    max(24.0, min(image.shape) * 0.35),
                )
            )
        ),
        "threshold_strategy": str(params.get("peak_threshold_strategy", "auto")),
        "threshold_value": raw_threshold_value,
        "subpixel_window_radius": (
            max(1, int(raw_subpixel_window_radius))
            if raw_subpixel_window_radius is not None
            else 2
        ),
        "subpixel_max_shift": (
            max(0.0, float(raw_subpixel_max_shift))
            if raw_subpixel_max_shift is not None
            else 1.5
        ),
        "peak_detector": str(params.get("peak_detector", "local_max")),
        "peak_adapter_engine": str(params.get("peak_adapter_engine", "none")),
        "detector_sigma": (
            max(0.2, float(raw_detector_sigma))
            if raw_detector_sigma is not None
            else float(auto_defaults.get("detector_sigma", 1.2))
        ),
        "dog_sigma_ratio": (
            max(1.2, float(raw_dog_sigma_ratio))
            if raw_dog_sigma_ratio is not None
            else float(auto_defaults.get("dog_sigma_ratio", 1.8))
        ),
        "radius_outlier_filter": _to_bool(raw_radius_outlier_filter, default=True),
        "radius_outlier_mad_factor": (
            max(2.0, float(raw_radius_outlier_mad_factor))
            if raw_radius_outlier_mad_factor is not None
            else 4.0
        ),
    }
    diagnostics = {
        "enabled": auto_peak_tuning,
        "auto_defaults": auto_defaults,
        "used_defaults": {
            "top_k_peaks": raw_top_k is None,
            "min_peak_distance": raw_min_distance is None,
            "center_exclusion_radius": raw_center_exclusion_radius is None,
            "max_peak_radius_px": raw_max_peak_radius is None,
            "detector_sigma": raw_detector_sigma is None,
            "dog_sigma_ratio": raw_dog_sigma_ratio is None,
            "radius_outlier_filter": raw_radius_outlier_filter is None,
            "radius_outlier_mad_factor": raw_radius_outlier_mad_factor is None,
        },
    }
    return peak_params, diagnostics


def _evaluate_b3_b4_consistency(
    *,
    peaks: list[Any],
    indexed_spots: list[IndexedSpot],
    phase_candidates: list[PhaseCandidate],
) -> tuple[list[str], dict[str, Any]]:
    warnings: list[str] = []
    peak_count = len(peaks)
    indexed_nonzero = sum(
        1
        for spot in indexed_spots
        if (int(spot.h), int(spot.k), int(spot.l)) != (0, 0, 0)
    )
    indexed_fraction = (indexed_nonzero / peak_count) if peak_count > 0 else 0.0
    if peak_count >= 6 and indexed_fraction < 0.20:
        warnings.append(
            "WARNING: B3/B4 consistency is weak; few peaks received non-zero hkl labels "
            f"({indexed_nonzero}/{peak_count})."
        )
    top_phase_score = max((float(item.score) for item in phase_candidates), default=0.0)
    if peak_count >= 6 and phase_candidates and top_phase_score < 0.30:
        warnings.append(
            "WARNING: Phase candidate confidence is low despite sufficient peaks; "
            "consider tighter d/angle tolerances or better candidate constraints."
        )
    return warnings, {
        "peak_count": peak_count,
        "indexed_nonzero_count": indexed_nonzero,
        "indexed_nonzero_fraction": indexed_fraction,
        "phase_candidate_count": len(phase_candidates),
        "top_phase_score": top_phase_score,
    }


def run_diffraction_analysis(
    preprocessing: PreprocessingResult,
    calibration: CalibrationResult,
    *,
    parameters: dict[str, Any] | None = None,
) -> DiffractionResult:
    """Run B2/B3 FFT-peak-gvector-dspacing pipeline with optional B4 indexing."""
    params = dict(parameters or {})
    warnings: list[str] = []

    image = _load_preprocessed_array(preprocessing)
    if image is None:
        warnings.append("Preprocessed image array is unavailable; diffraction result is empty.")
        return DiffractionResult(
            confidence=0.0,
            provenance={"method": "run_diffraction_analysis", "reason": "missing_preprocessed"},
            warnings=warnings,
        )

    fft_center_mode = str(params.get("fft_center_mode", "auto"))
    fft_search_radius = _to_float(params.get("direct_beam_search_radius"))
    fft_refine_window = int(_to_float(params.get("direct_beam_refine_window")) or 3)
    fft_analysis, fft_warnings = compute_fft_analysis(
        image,
        center_mode=fft_center_mode,
        search_radius_px=fft_search_radius,
        refine_window=max(1, fft_refine_window),
    )
    warnings.extend(fft_warnings)

    fft_uri = None
    try:
        fft_uri = persist_fft_array_near_source(
            fft_analysis.log_magnitude,
            source_uri=preprocessing.image.uri if preprocessing.image is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to persist FFT magnitude array near source: {exc}")
        fft_uri = None
    fft_image_ref = build_fft_reference(fft_analysis.log_magnitude, uri=fft_uri)

    peak_params, auto_peak_diag = _resolve_peak_params(image=image, params=params)
    top_k = int(peak_params["top_k"])
    min_distance = int(peak_params["min_distance"])
    center_exclusion_radius = float(peak_params["center_exclusion_radius"])
    max_peak_radius_px = _to_float(peak_params.get("max_peak_radius_px"))
    if max_peak_radius_px is not None and max_peak_radius_px <= 0:
        max_peak_radius_px = None
    threshold_strategy = str(peak_params["threshold_strategy"])
    threshold_value = _to_float(peak_params["threshold_value"])
    subpixel_window_radius = int(peak_params["subpixel_window_radius"])
    subpixel_max_shift = float(peak_params["subpixel_max_shift"])
    peak_detector = str(peak_params["peak_detector"])
    peak_adapter_engine = str(peak_params["peak_adapter_engine"])
    detector_sigma = float(peak_params["detector_sigma"])
    dog_sigma_ratio = float(peak_params["dog_sigma_ratio"])
    radius_outlier_filter = bool(peak_params["radius_outlier_filter"])
    radius_outlier_mad_factor = float(peak_params["radius_outlier_mad_factor"])
    peak_adapter_status = "internal"

    if peak_adapter_engine.strip().lower() != "none":
        try:
            peaks, peak_diagnostics, peak_warnings = detect_peaks_with_adapter(
                engine=peak_adapter_engine,
                log_magnitude=fft_analysis.log_magnitude,
                center=fft_analysis.center,
                top_k=max(1, top_k),
                min_distance=max(1, min_distance),
                center_exclusion_radius=max(0.0, center_exclusion_radius),
                max_peak_radius_px=max_peak_radius_px,
                threshold_strategy=threshold_strategy,
                threshold_value=threshold_value,
                subpixel_window_radius=max(1, subpixel_window_radius),
                subpixel_max_shift=max(0.0, subpixel_max_shift),
                detector=peak_detector,
                detector_sigma=max(0.2, detector_sigma),
                dog_sigma_ratio=max(1.2, dog_sigma_ratio),
                radius_outlier_filter=radius_outlier_filter,
                radius_outlier_mad_factor=max(2.0, radius_outlier_mad_factor),
            )
            peak_adapter_status = f"adapter.{peak_adapter_engine.strip().lower()}"
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                "Open-source peak adapter failed "
                f"(engine={peak_adapter_engine}): {exc}; "
                "fallback to internal peak detection."
            )
            peaks, peak_diagnostics, peak_warnings = detect_bragg_peaks(
                fft_analysis.log_magnitude,
                center=fft_analysis.center,
                top_k=max(1, top_k),
                min_distance=max(1, min_distance),
                center_exclusion_radius=max(0.0, center_exclusion_radius),
                max_peak_radius_px=max_peak_radius_px,
                threshold_strategy=threshold_strategy,
                threshold_value=threshold_value,
                subpixel_window_radius=max(1, subpixel_window_radius),
                subpixel_max_shift=max(0.0, subpixel_max_shift),
                detector=peak_detector,
                detector_sigma=max(0.2, detector_sigma),
                dog_sigma_ratio=max(1.2, dog_sigma_ratio),
                radius_outlier_filter=radius_outlier_filter,
                radius_outlier_mad_factor=max(2.0, radius_outlier_mad_factor),
            )
            peak_adapter_status = "internal_fallback"
    else:
        peaks, peak_diagnostics, peak_warnings = detect_bragg_peaks(
            fft_analysis.log_magnitude,
            center=fft_analysis.center,
            top_k=max(1, top_k),
            min_distance=max(1, min_distance),
            center_exclusion_radius=max(0.0, center_exclusion_radius),
            max_peak_radius_px=max_peak_radius_px,
            threshold_strategy=threshold_strategy,
            threshold_value=threshold_value,
            subpixel_window_radius=max(1, subpixel_window_radius),
            subpixel_max_shift=max(0.0, subpixel_max_shift),
            detector=peak_detector,
            detector_sigma=max(0.2, detector_sigma),
            dog_sigma_ratio=max(1.2, dog_sigma_ratio),
            radius_outlier_filter=radius_outlier_filter,
            radius_outlier_mad_factor=max(2.0, radius_outlier_mad_factor),
        )
    warnings.extend(peak_warnings)

    scale_value = None
    scale_uncertainty = None
    scale_source = "unresolved"
    if calibration.diffraction_scale is not None:
        scale_value = _to_float(calibration.diffraction_scale.value)
        scale_uncertainty = _to_float(calibration.diffraction_scale.uncertainty)
        scale_source = calibration.diffraction_scale.method or "calibration.diffraction_scale"
    center_uncertainty_px = _to_float(params.get("direct_beam_center_uncertainty_px"))
    if center_uncertainty_px is None:
        center_uncertainty_px = 0.35 if fft_center_mode == "auto" else 0.0

    reciprocal = compute_gvectors_and_dspacing(
        peaks,
        center=fft_analysis.center,
        diffraction_scale_value=scale_value,
        diffraction_scale_uncertainty=scale_uncertainty,
        diffraction_scale_source=scale_source,
        center_uncertainty_px=center_uncertainty_px,
    )
    warnings.extend(reciprocal.warnings)
    g_vectors = reciprocal.g_vectors
    d_by_peak = reciprocal.d_by_peak
    center_diag = fft_analysis.diagnostics.get("center")
    warnings.extend(
        _grade_fft_warnings(
            image_shape=tuple(int(v) for v in image.shape),
            log_magnitude=fft_analysis.log_magnitude,
            peak_count=len(peaks),
            center_diagnostics=center_diag if isinstance(center_diag, dict) else None,
        )
    )

    indexed_spots = _baseline_indexed_spots(d_by_peak)
    phase_candidates: list[PhaseCandidate] = []

    raw_candidate_phase_params = params.get("candidate_phase_params")
    allowed_names_raw = params.get("candidate_phases")
    candidate_phase_names = _as_string_list(allowed_names_raw)
    allowed_names = (
        {str(item).strip() for item in candidate_phase_names if str(item).strip()}
        if candidate_phase_names
        else None
    )
    candidate_phase_params = (
        list(raw_candidate_phase_params) if isinstance(raw_candidate_phase_params, list) else []
    )
    phase_lookup_candidates, phase_lookup_warnings, phase_lookup_provenance = (
        _resolve_phase_lookup_candidates(params, candidate_phases=candidate_phase_names)
    )
    if phase_lookup_candidates:
        candidate_phase_params.extend(phase_lookup_candidates)
        lookup_names = {
            str(row.get("name", "")).strip()
            for row in phase_lookup_candidates
            if isinstance(row, dict) and str(row.get("name", "")).strip()
        }
        allowed_names = (set(allowed_names or set()) | lookup_names) or None
    warnings.extend(phase_lookup_warnings)
    phase_top_n = int(_to_float(params.get("phase_top_n")) or 3)
    indexing_mode = str(params.get("indexing_mode", "auto")).strip().lower()
    enable_b4 = indexing_mode in {"auto", "lattice_fit"}
    if enable_b4 and peaks:
        b4_result = run_b4_indexing(
            peaks=peaks,
            g_vectors=g_vectors,
            d_by_peak=d_by_peak,
            initial_indexed_spots=indexed_spots,
            candidate_phase_params=candidate_phase_params,
            allowed_phase_names=allowed_names,
            max_hkl=int(_to_float(params.get("max_hkl")) or 4),
            d_tolerance_rel=float(_to_float(params.get("d_tolerance_rel")) or 0.10),
            angle_tolerance_deg=float(_to_float(params.get("angle_tolerance_deg")) or 12.0),
            strict_mode=bool(params.get("b4_strict_mode", False)),
            cif_candidates_dir=(
                str(params["cif_candidates_dir"]) if "cif_candidates_dir" in params else None
            ),
            cif_candidate_paths=params.get("cif_candidate_paths"),
            top_n=max(1, phase_top_n),
        )
        indexed_spots = b4_result.indexed_spots
        phase_candidates = b4_result.phase_candidates
        warnings.extend(b4_result.warnings)
        b4_provenance = b4_result.provenance
    else:
        b4_provenance = {"active": False, "reason": "disabled_or_no_peaks"}

    if not phase_candidates and candidate_phase_names:
        phase_candidates = _phase_placeholders(candidate_phase_names, top_n=phase_top_n)

    consistency_warnings, consistency_diag = _evaluate_b3_b4_consistency(
        peaks=peaks,
        indexed_spots=indexed_spots,
        phase_candidates=phase_candidates,
    )
    warnings.extend(consistency_warnings)

    d_spacings = [entry for entry in d_by_peak if entry is not None]
    confidence = 0.2
    confidence += min(0.45, 0.03 * len(peaks))
    confidence += 0.15 if scale_value is not None and scale_value > 0 else 0.0
    confidence += 0.1 if phase_candidates else 0.0
    confidence -= min(0.25, 0.03 * len(set(warnings)))
    confidence = max(0.0, min(1.0, confidence))

    provenance = {
        "method": "run_diffraction_analysis",
        "fft": fft_analysis.diagnostics,
        "peak_detection": peak_diagnostics,
        "auto_peak_tuning": auto_peak_diag,
        "peak_adapter_engine_requested": peak_adapter_engine,
        "peak_adapter_engine_status": peak_adapter_status,
        "reciprocal_metrics": reciprocal.diagnostics,
        "diffraction_scale_inv_nm_per_px": scale_value,
        "diffraction_scale_uncertainty": scale_uncertainty,
        "diffraction_scale_source": scale_source,
        "direct_beam_center_uncertainty_px": center_uncertainty_px,
        "indexing_mode": indexing_mode,
        "b4_indexing": b4_provenance,
        "b4_phase_lookup": phase_lookup_provenance,
        "b3_b4_consistency": consistency_diag,
        "parameters": params,
    }

    return DiffractionResult(
        peaks=peaks,
        center=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
        g_vectors=g_vectors,
        d_spacings=d_spacings,
        indexed_spots=indexed_spots,
        phase_candidates=phase_candidates,
        fft_image=fft_image_ref,
        confidence=confidence,
        provenance=provenance,
        warnings=list(dict.fromkeys(warnings)),
    )
