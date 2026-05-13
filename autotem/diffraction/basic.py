"""A2 baseline FFT, peak extraction, indexing, and phase matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
) -> list[tuple[int, int, float]]:
    if magnitude.size == 0:
        return []
    flat = magnitude.ravel()
    k = min(top_k * 4, flat.size)
    idx = np.argpartition(flat, -k)[-k:]
    sorted_idx = idx[np.argsort(flat[idx])[::-1]]
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    selected: list[tuple[int, int, float]] = []
    for flat_index in sorted_idx:
        y, x = np.unravel_index(int(flat_index), magnitude.shape)
        if np.hypot(y - cy, x - cx) < center_exclusion_radius:
            continue
        intensity = float(magnitude[y, x])
        if any(np.hypot(y - py, x - px) < min_distance for py, px, _ in selected):
            continue
        selected.append((int(y), int(x), intensity))
        if len(selected) >= top_k:
            break
    return selected


def _make_index(dx: float, dy: float) -> tuple[int, int, int]:
    denom = max(abs(dx), abs(dy), 1.0)
    h = int(np.rint(dx / denom * 4))
    k = int(np.rint(dy / denom * 4))
    return h, k, 0


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
    phase_top_n = int(parameters.get("phase_top_n", 3))
    candidate_phases_raw = parameters.get("candidate_phases")
    allowed_phase_names = None
    if isinstance(candidate_phases_raw, list):
        allowed_phase_names = {str(item) for item in candidate_phases_raw}
    peak_triplets = _detect_peaks(
        magnitude,
        top_k=max(1, top_k),
        min_distance=max(0.0, min_peak_distance),
        center_exclusion_radius=max(0.0, center_exclusion_radius),
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

    for i, (y, x, intensity) in enumerate(peak_triplets):
        peaks.append(DiffractionPeak(x=float(x), y=float(y), intensity=intensity, confidence=0.65))
        dx = float(x - cx)
        dy = float(y - cy)
        gx_val = dx * reciprocal_scale
        gy_val = dy * reciprocal_scale
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
                "phase_top_n": phase_top_n,
                "candidate_phases": (
                    sorted(allowed_phase_names) if allowed_phase_names is not None else None
                ),
            },
            "calibration_scale_source": reciprocal_scale_source,
            "calibration_scale_inv_nm_per_px": reciprocal_scale,
        },
    )
