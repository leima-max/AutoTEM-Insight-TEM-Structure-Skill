"""Geometric phase analysis (GPA) strain mapping for the C2 workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from autotem.core import (
    ArrayReference,
    DiffractionPeak,
    DiffractionResult,
    GVector,
    Measurement,
    StrainResult,
)
from autotem.io import read_image

ImageInput = np.ndarray | str | Path
GVectorInput = list[tuple[float, float]]
ReferenceRegion = tuple[int, int, int, int]
ReferenceRegions = tuple[ReferenceRegion | None, ReferenceRegion | None]

_TWO_PI = 2.0 * np.pi
DEFAULT_GPA_MASK_SIGMA_PX = 2.5
_SUPPORTED_GPA_METHODS = {"gpa", "gpa_b3"}


@dataclass(slots=True)
class GPAPhaseMap:
    """Intermediate phase and sideband information for one g-vector."""

    phase_wrapped: np.ndarray
    phase_unwrapped: np.ndarray
    bragg_amplitude: np.ndarray
    gaussian_mask: np.ndarray
    g_vector_cycles: tuple[float, float]
    g_vector_pixels: tuple[float, float]
    reference_region: ReferenceRegion | None
    reference_plane: tuple[float, float, float] | None
    reference_phase_offset: float


@dataclass(slots=True)
class GPAFieldResult:
    """Full GPA output fields before conversion to the project contract."""

    displacement_x: np.ndarray
    displacement_y: np.ndarray
    strain_xx: np.ndarray
    strain_yy: np.ndarray
    strain_xy: np.ndarray
    rotation_xy: np.ndarray
    uncertainty: np.ndarray
    quality_mask: np.ndarray
    phase_maps: list[GPAPhaseMap]
    g_vectors: list[GVector]
    reference_region: ReferenceRegion | None
    reference_regions: ReferenceRegions
    confidence: float
    warnings: list[str]
    provenance: dict[str, Any]

    def to_strain_result(self, *, map_inline_mode: str = "preview") -> StrainResult:
        provenance = dict(self.provenance)
        provenance["map_stats"] = _strain_map_stats(
            displacement_x=self.displacement_x,
            displacement_y=self.displacement_y,
            strain_xx=self.strain_xx,
            strain_yy=self.strain_yy,
            strain_xy=self.strain_xy,
            rotation_xy=self.rotation_xy,
            valid_mask=self.quality_mask,
        )
        provenance["phase_maps"] = {
            f"g{i + 1}": {
                "phase_wrapped": _to_array_ref(
                    phase.phase_wrapped,
                    description=f"GPA wrapped phase map for g{i + 1}.",
                    inline_mode=map_inline_mode,
                ).model_dump(mode="json"),
                "bragg_amplitude": _to_array_ref(
                    phase.bragg_amplitude,
                    description=f"GPA Bragg sideband amplitude for g{i + 1}.",
                    inline_mode=map_inline_mode,
                ).model_dump(mode="json"),
                "g_vector_cycles_per_pixel": list(phase.g_vector_cycles),
                "g_vector_fft_pixels": list(phase.g_vector_pixels),
                "reference_region": (
                    list(phase.reference_region) if phase.reference_region is not None else None
                ),
                "reference_plane": (
                    list(phase.reference_plane) if phase.reference_plane is not None else None
                ),
                "reference_phase_offset": phase.reference_phase_offset,
            }
            for i, phase in enumerate(self.phase_maps)
        }
        provenance["reference_regions"] = _reference_regions_to_json(self.reference_regions)

        return StrainResult(
            displacement_x=_to_array_ref(
                self.displacement_x,
                description="GPA ux displacement map in image pixels.",
                inline_mode=map_inline_mode,
            ),
            displacement_y=_to_array_ref(
                self.displacement_y,
                description="GPA uy displacement map in image pixels.",
                inline_mode=map_inline_mode,
            ),
            strain_xx=_to_array_ref(
                self.strain_xx,
                description="GPA exx strain map.",
                inline_mode=map_inline_mode,
            ),
            strain_yy=_to_array_ref(
                self.strain_yy,
                description="GPA eyy strain map.",
                inline_mode=map_inline_mode,
            ),
            strain_xy=_to_array_ref(
                self.strain_xy,
                description="GPA engineering shear strain component exy.",
                inline_mode=map_inline_mode,
            ),
            rotation_xy=_to_array_ref(
                self.rotation_xy,
                description="GPA local lattice rotation map in radians.",
                inline_mode=map_inline_mode,
            ),
            uncertainty=_to_array_ref(
                self.uncertainty,
                description="GPA heuristic uncertainty from Bragg sideband quality.",
                inline_mode=map_inline_mode,
            ),
            reference_region=self.reference_region,
            g_vectors=list(self.g_vectors),
            confidence=self.confidence,
            provenance=provenance,
            warnings=list(self.warnings),
        )


def run_gpa(
    image: ImageInput,
    *,
    method: str = "gpa",
    diffraction_result: DiffractionResult | None = None,
    g_vectors: GVectorInput | None = None,
    g_vector_unit: str = "1/pixel",
    pixel_size_nm: float | None = None,
    reference_region: ReferenceRegion | None = None,
    reference_regions: ReferenceRegions | None = None,
    gpa_mask_sigma_px: float | None = None,
    mask_sigma_px: float | None = None,
    hann_window: bool = True,
    refine_reference: bool = True,
    min_quality_fraction: float = 0.08,
    quality_percentile: float = 10.0,
    apply_quality_mask: bool = True,
    map_inline_mode: str = "preview",
    include_debug_maps: bool = True,
    ppa_window_radius: int = 8,
    ppa_grid_step: int = 6,
) -> StrainResult:
    """Run C2 geometric phase analysis and return a ``StrainResult``.

    The implementation follows a Hytch-style GPA workflow: isolate two Bragg
    sidebands, compute geometric phase maps, flatten an optional reference
    region, and recover the displacement-gradient tensor from the two phase
    gradients.  The strain maps are computed from wrapped phase gradients in the
    complex plane, which avoids the row-wise artifacts caused by differentiating
    naively unwrapped displacement maps.
    """

    del include_debug_maps, ppa_window_radius, ppa_grid_step
    method = method.lower().strip()
    if method == "gpa_b3":
        provenance_method = "gpa_b3"
    elif method in _SUPPORTED_GPA_METHODS:
        provenance_method = method
    else:
        raise ValueError("method must be one of {'gpa', 'gpa_b3'} for C2 GPA.")
    map_inline_mode = map_inline_mode.lower().strip()
    if map_inline_mode not in {"preview", "full"}:
        raise ValueError("map_inline_mode must be one of {'preview', 'full'}.")

    warnings: list[str] = []
    arr = _load_image_2d(image, warnings=warnings)
    sigma_px = _resolve_mask_sigma(
        gpa_mask_sigma_px=gpa_mask_sigma_px,
        mask_sigma_px=mask_sigma_px,
        image_shape=arr.shape,
    )

    fields = compute_gpa_fields(
        arr,
        diffraction_result=diffraction_result,
        g_vectors=g_vectors,
        g_vector_unit=g_vector_unit,
        pixel_size_nm=pixel_size_nm,
        reference_region=reference_region,
        reference_regions=reference_regions,
        mask_sigma_px=sigma_px,
        hann_window=hann_window,
        refine_reference=refine_reference,
        min_quality_fraction=min_quality_fraction,
        quality_percentile=quality_percentile,
        apply_quality_mask=apply_quality_mask,
        warnings=warnings,
        provenance_method=provenance_method,
    )
    return fields.to_strain_result(map_inline_mode=map_inline_mode)


def compute_gpa_fields(
    image: np.ndarray,
    *,
    diffraction_result: DiffractionResult | None = None,
    g_vectors: GVectorInput | None = None,
    g_vector_unit: str = "1/pixel",
    pixel_size_nm: float | None = None,
    reference_region: ReferenceRegion | None = None,
    reference_regions: ReferenceRegions | None = None,
    mask_sigma_px: float = DEFAULT_GPA_MASK_SIGMA_PX,
    hann_window: bool = True,
    refine_reference: bool = True,
    min_quality_fraction: float = 0.08,
    quality_percentile: float = 10.0,
    apply_quality_mask: bool = True,
    warnings: list[str] | None = None,
    provenance_method: str = "gpa",
) -> GPAFieldResult:
    """Compute C2 GPA fields from a 2D image array."""

    warnings = warnings if warnings is not None else []
    arr = _prepare_image(image)
    h, w = arr.shape
    g_cycles, g_source = _resolve_g_vectors(
        arr.shape,
        diffraction_result=diffraction_result,
        g_vectors=g_vectors,
        g_vector_unit=g_vector_unit,
        pixel_size_nm=pixel_size_nm,
        warnings=warnings,
    )
    _validate_g_vector_pair(g_cycles)
    refs = _normalize_reference_regions(
        reference_regions,
        reference_region,
        arr.shape,
        warnings,
    )

    fft_img = _fft_for_gpa(arr, hann_window=hann_window)
    phases: list[GPAPhaseMap] = []
    for index, g in enumerate(g_cycles):
        phases.append(
            _extract_phase_map(
                fft_img=fft_img,
                image_shape=(h, w),
                g_vector_cycles=g,
                mask_sigma_px=mask_sigma_px,
                reference_region=refs[index],
                refine_reference=refine_reference,
                warnings=warnings,
            )
        )
    g_refined = np.asarray([phase.g_vector_cycles for phase in phases], dtype=np.float64)
    _validate_g_vector_pair(g_refined)

    quality_mask, uncertainty = _quality_from_amplitudes(
        [phase.bragg_amplitude for phase in phases],
        min_quality_fraction=min_quality_fraction,
        quality_percentile=quality_percentile,
        warnings=warnings,
    )
    computation_mask = quality_mask if apply_quality_mask else None
    displacement_x, displacement_y = _displacement_from_phases(
        [phase.phase_wrapped for phase in phases],
        g_refined,
        quality_mask=computation_mask,
    )
    strain_xx, strain_yy, strain_xy, rotation_xy = _strain_from_phase_gradients(
        [phase.phase_wrapped for phase in phases],
        g_refined,
        quality_mask=computation_mask,
    )

    g_vector_models = [
        _gvector_model(g[0], g[1], source_peak_index=i if g_source == "diffraction_peaks" else None)
        for i, g in enumerate(g_refined)
    ]
    _append_field_sanity_warnings(
        image_shape=arr.shape,
        displacement_x=displacement_x,
        displacement_y=displacement_y,
        strain_xx=strain_xx,
        strain_yy=strain_yy,
        strain_xy=strain_xy,
        rotation_xy=rotation_xy,
        valid_mask=quality_mask,
        warnings=warnings,
    )
    confidence = _estimate_confidence(quality_mask, warnings)
    provenance = {
        "source": "autotem.strain.gpa.compute_gpa_fields",
        "method": provenance_method,
        "algorithm_reference": "Hytch-style GPA implemented in AutoTEM C2.",
        "g_vector_source": g_source,
        "parameters": {
            "g_vector_unit": g_vector_unit,
            "pixel_size_nm": pixel_size_nm,
            "mask_sigma_px": float(mask_sigma_px),
            "hann_window": bool(hann_window),
            "refine_reference": bool(refine_reference),
            "min_quality_fraction": float(min_quality_fraction),
            "quality_percentile": float(quality_percentile),
            "apply_quality_mask": bool(apply_quality_mask),
            "reference_region": (
                list(_common_reference_region(refs))
                if _common_reference_region(refs) is not None
                else None
            ),
            "reference_regions": _reference_regions_to_json(refs),
        },
        "quality": {
            "valid_pixel_fraction": float(np.mean(quality_mask)),
            "uncertainty_mean": float(np.mean(uncertainty)),
        },
    }
    return GPAFieldResult(
        displacement_x=displacement_x.astype(np.float32),
        displacement_y=displacement_y.astype(np.float32),
        strain_xx=strain_xx.astype(np.float32),
        strain_yy=strain_yy.astype(np.float32),
        strain_xy=strain_xy.astype(np.float32),
        rotation_xy=rotation_xy.astype(np.float32),
        uncertainty=uncertainty.astype(np.float32),
        quality_mask=quality_mask,
        phase_maps=phases,
        g_vectors=g_vector_models,
        reference_region=_common_reference_region(refs),
        reference_regions=refs,
        confidence=confidence,
        warnings=warnings,
        provenance=provenance,
    )


def _load_image_2d(image: ImageInput, *, warnings: list[str]) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return _prepare_image(image)
    result = read_image(image)
    warnings.extend(str(item) for item in result.warnings)
    return _prepare_image(result.array)


def _prepare_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError("GPA image input must be a 2D array after squeezing.")
    if min(arr.shape) < 16:
        raise ValueError("GPA image input must be at least 16x16 pixels.")
    arr = np.nan_to_num(arr, copy=False)
    arr = arr - float(np.mean(arr))
    std = float(np.std(arr))
    if std > 0:
        arr = arr / std
    return arr


def _resolve_mask_sigma(
    *,
    gpa_mask_sigma_px: float | None,
    mask_sigma_px: float | None,
    image_shape: tuple[int, int],
) -> float:
    sigma = mask_sigma_px if mask_sigma_px is not None else gpa_mask_sigma_px
    if sigma is None:
        sigma = DEFAULT_GPA_MASK_SIGMA_PX
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("mask sigma must be a positive finite value.")
    return sigma


def _fft_for_gpa(image: np.ndarray, *, hann_window: bool) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if hann_window:
        wy = np.hanning(arr.shape[0])
        wx = np.hanning(arr.shape[1])
        arr = arr * np.outer(wy, wx)
    return np.fft.fftshift(np.fft.fft2(arr))


def _resolve_g_vectors(
    image_shape: tuple[int, int],
    *,
    diffraction_result: DiffractionResult | None,
    g_vectors: GVectorInput | None,
    g_vector_unit: str,
    pixel_size_nm: float | None,
    warnings: list[str],
) -> tuple[np.ndarray, str]:
    if g_vectors is not None:
        raw = np.asarray(g_vectors, dtype=np.float64)
        if raw.shape != (2, 2):
            raise ValueError("g_vectors must contain exactly two (gx, gy) entries.")
        return _convert_g_vector_units(
            raw,
            unit=g_vector_unit,
            pixel_size_nm=pixel_size_nm,
            image_shape=image_shape,
        ), "manual"

    if diffraction_result is not None and len(diffraction_result.g_vectors) >= 2:
        raw_units = [
            (
                float(gv.gx.value),
                float(gv.gy.value),
                _diffraction_g_vector_unit_for_gpa(gv),
            )
            for gv in diffraction_result.g_vectors[:2]
        ]
        converted = [
            _convert_g_vector_units(
                np.asarray([[gx, gy]], dtype=np.float64),
                unit=unit,
                pixel_size_nm=pixel_size_nm,
                image_shape=image_shape,
            )[0]
            for gx, gy, unit in raw_units
        ]
        return np.asarray(converted, dtype=np.float64), "diffraction_g_vectors"

    if diffraction_result is not None and len(diffraction_result.peaks) >= 2:
        center = diffraction_result.center
        if center is None:
            center = (image_shape[1] / 2.0, image_shape[0] / 2.0)
            warnings.append(
                "DiffractionResult.center missing; using image center for GPA peak vectors."
            )
        return _select_g_vectors_from_peaks(
            peaks=diffraction_result.peaks,
            center_xy=(float(center[0]), float(center[1])),
            image_shape=image_shape,
        ), "diffraction_peaks"

    raise ValueError(
        "GPA requires two manual g_vectors or a DiffractionResult with g-vectors/peaks."
    )


def _convert_g_vector_units(
    vectors: np.ndarray,
    *,
    unit: str,
    pixel_size_nm: float | None,
    image_shape: tuple[int, int],
) -> np.ndarray:
    unit_norm = unit.lower().replace(" ", "")
    vectors = np.asarray(vectors, dtype=np.float64)
    if unit_norm in {"1/pixel", "1/px", "cycles/pixel", "cycles/px", "pixel^-1", "px^-1"}:
        return vectors
    if unit_norm in {"fft_pixel", "fft_pixels", "pixel_offset", "px", "pixels"}:
        scale = np.asarray([image_shape[1], image_shape[0]], dtype=np.float64)
        return vectors / scale
    if unit_norm in {"1/nm", "nm^-1", "cycles/nm"}:
        if pixel_size_nm is None or not np.isfinite(pixel_size_nm) or pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm is required for g_vector_unit='1/nm'.")
        return vectors * float(pixel_size_nm)
    raise ValueError(
        "g_vector_unit must be one of {'1/pixel', 'fft_pixel', '1/nm'} or equivalent."
    )


def _diffraction_g_vector_unit_for_gpa(g_vector: GVector) -> str:
    unit = str(g_vector.gx.unit or g_vector.gy.unit or "1/pixel")
    unit_norm = unit.lower().replace(" ", "")
    method = f"{g_vector.gx.method or ''} {g_vector.gy.method or ''}".lower()
    if unit_norm in {"1/pixel", "1/px", "pixel^-1", "px^-1"} and "peak_offset" in method:
        return "fft_pixel"
    return unit


def _select_g_vectors_from_peaks(
    *,
    peaks: list[DiffractionPeak],
    center_xy: tuple[float, float],
    image_shape: tuple[int, int],
) -> np.ndarray:
    candidates: list[tuple[float, float, float]] = []
    cx, cy = center_xy
    for peak in peaks:
        dx = float(peak.x) - cx
        dy = float(peak.y) - cy
        radius = float(np.hypot(dx, dy))
        if radius < 2.0:
            continue
        score = float(peak.intensity) if peak.intensity is not None else radius
        candidates.append((score, dx / image_shape[1], dy / image_shape[0]))
    if len(candidates) < 2:
        raise ValueError("Diffraction peaks do not contain two usable non-center g-vectors.")
    candidates.sort(reverse=True, key=lambda item: item[0])
    first = np.asarray(candidates[0][1:], dtype=np.float64)
    for _, gx, gy in candidates[1:]:
        second = np.asarray([gx, gy], dtype=np.float64)
        if abs(float(np.linalg.det(np.vstack([first, second])))) > 1e-5:
            return np.vstack([first, second])
    raise ValueError("Could not select two non-collinear g-vectors from diffraction peaks.")


def _validate_g_vector_pair(g_vectors: np.ndarray) -> None:
    g = np.asarray(g_vectors, dtype=np.float64)
    if g.shape != (2, 2):
        raise ValueError("GPA requires exactly two g-vectors.")
    if not np.all(np.isfinite(g)):
        raise ValueError("GPA g-vectors must be finite.")
    if np.any(np.linalg.norm(g, axis=1) <= 1e-8):
        raise ValueError("GPA g-vectors must be non-zero.")
    if np.any(np.abs(g) >= 0.5):
        raise ValueError(
            "GPA g-vectors must be FFT frequencies in cycles/pixel with each "
            "component inside the Nyquist range (-0.5, 0.5). Check whether "
            "diffraction-calibrated 1/nm vectors were passed instead of FFT peak offsets."
        )
    if abs(float(np.linalg.det(g))) <= 1e-6:
        raise ValueError("GPA g-vectors must be non-collinear.")


def _extract_phase_map(
    *,
    fft_img: np.ndarray,
    image_shape: tuple[int, int],
    g_vector_cycles: np.ndarray,
    mask_sigma_px: float,
    reference_region: tuple[int, int, int, int] | None,
    refine_reference: bool,
    warnings: list[str],
) -> GPAPhaseMap:
    g = np.asarray(g_vector_cycles, dtype=np.float64)
    phase, amplitude, mask = _phase_for_g_vector(
        fft_img=fft_img,
        image_shape=image_shape,
        g_vector_cycles=g,
        mask_sigma_px=mask_sigma_px,
    )
    plane: tuple[float, float, float] | None = None
    if reference_region is not None and refine_reference:
        plane = _fit_reference_plane(phase, reference_region, warnings)
        if plane is not None:
            _, slope_x, slope_y = plane
            g = g + np.asarray([slope_x, slope_y], dtype=np.float64) / _TWO_PI
            phase, amplitude, mask = _phase_for_g_vector(
                fft_img=fft_img,
                image_shape=image_shape,
                g_vector_cycles=g,
                mask_sigma_px=mask_sigma_px,
            )
    phase_offset = _reference_phase_offset(phase, reference_region)
    phase = _wrap_phase(phase - phase_offset)
    phase_unwrapped = _unwrap_phase_for_displacement(phase)
    return GPAPhaseMap(
        phase_wrapped=phase.astype(np.float32),
        phase_unwrapped=phase_unwrapped.astype(np.float32),
        bragg_amplitude=amplitude.astype(np.float32),
        gaussian_mask=mask.astype(np.float32),
        g_vector_cycles=(float(g[0]), float(g[1])),
        g_vector_pixels=(float(g[0] * image_shape[1]), float(g[1] * image_shape[0])),
        reference_region=reference_region,
        reference_plane=plane,
        reference_phase_offset=float(phase_offset),
    )


def _phase_for_g_vector(
    *,
    fft_img: np.ndarray,
    image_shape: tuple[int, int],
    g_vector_cycles: np.ndarray,
    mask_sigma_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    gx, gy = float(g_vector_cycles[0]), float(g_vector_cycles[1])
    cx = (w / 2.0) + gx * w
    cy = (h / 2.0) + gy * h
    mask = np.exp(-0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (mask_sigma_px**2)))
    sideband = np.fft.ifft2(np.fft.ifftshift(fft_img * mask))
    phase = np.angle(sideband) - _TWO_PI * (gx * xx + gy * yy)
    amplitude = np.abs(sideband)
    return _wrap_phase(phase), amplitude, mask


def _fit_reference_plane(
    phase: np.ndarray,
    reference_region: tuple[int, int, int, int],
    warnings: list[str],
) -> tuple[float, float, float] | None:
    y0, y1, x0, x1 = reference_region
    roi = np.asarray(phase[y0:y1, x0:x1], dtype=np.float64)
    if roi.size < 16:
        warnings.append("GPA reference region too small for phase-plane refinement.")
        return None
    roi_unwrapped = _unwrap_phase_for_displacement(roi)
    yy, xx = np.indices(roi.shape, dtype=np.float64)
    x_abs = xx + float(x0)
    y_abs = yy + float(y0)
    design = np.column_stack([np.ones(roi.size), x_abs.ravel(), y_abs.ravel()])
    coeffs, *_ = np.linalg.lstsq(design, roi_unwrapped.ravel(), rcond=None)
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def _reference_phase_offset(
    phase: np.ndarray,
    reference_region: tuple[int, int, int, int] | None,
) -> float:
    if reference_region is None:
        sample = phase
    else:
        y0, y1, x0, x1 = reference_region
        sample = phase[y0:y1, x0:x1]
        if sample.size == 0:
            sample = phase
    return float(np.angle(np.mean(np.exp(1j * sample))))


def _wrap_phase(phase: np.ndarray) -> np.ndarray:
    return (np.asarray(phase, dtype=np.float64) + np.pi) % _TWO_PI - np.pi


def _unwrap_phase_for_displacement(
    phase: np.ndarray,
    quality_mask: np.ndarray | None = None,
) -> np.ndarray:
    phase_arr = np.asarray(phase, dtype=np.float64)
    try:
        from skimage.restoration import unwrap_phase
    except Exception:
        unwrap_phase = None

    def _unwrap_2d(crop_phase: np.ndarray, crop_mask: np.ndarray | None = None) -> np.ndarray:
        crop_arr = np.asarray(crop_phase, dtype=np.float64)
        if unwrap_phase is None:
            if crop_mask is not None and crop_mask.shape == crop_arr.shape:
                crop_arr = np.where(crop_mask, crop_arr, 0.0)
            return np.unwrap(np.unwrap(crop_arr, axis=0), axis=1)
        if crop_mask is not None and crop_mask.shape == crop_arr.shape and not bool(np.all(crop_mask)):
            unwrapped = unwrap_phase(np.ma.array(crop_arr, mask=~crop_mask))
        else:
            unwrapped = unwrap_phase(crop_arr)
        if np.ma.isMaskedArray(unwrapped):
            return np.asarray(unwrapped.filled(0.0), dtype=np.float64)
        return np.asarray(unwrapped, dtype=np.float64)

    if quality_mask is not None and quality_mask.shape == phase_arr.shape and np.any(quality_mask):
        mask = np.asarray(quality_mask, dtype=bool)
        y_idx, x_idx = np.nonzero(mask)
        y0, y1 = int(np.min(y_idx)), int(np.max(y_idx)) + 1
        x0, x1 = int(np.min(x_idx)), int(np.max(x_idx)) + 1
        crop_phase = phase_arr[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        out = np.zeros_like(phase_arr, dtype=np.float64)
        out[y0:y1, x0:x1] = _unwrap_2d(crop_phase, crop_mask)
        return out
    return _unwrap_2d(phase_arr)


def _quality_from_amplitudes(
    amplitudes: list[np.ndarray],
    *,
    min_quality_fraction: float,
    quality_percentile: float,
    warnings: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    normalized: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for amp in amplitudes:
        arr = np.asarray(amp, dtype=np.float64)
        max_amp = float(np.max(arr)) if arr.size else 0.0
        if max_amp <= 0:
            norm = np.zeros_like(arr)
        else:
            norm = arr / max_amp
        threshold = max(
            float(min_quality_fraction),
            float(np.percentile(norm, np.clip(quality_percentile, 0.0, 100.0))),
        )
        normalized.append(norm)
        masks.append(norm >= threshold)
    quality = np.logical_and.reduce(masks) if masks else np.zeros((0, 0), dtype=bool)
    if quality.size and float(np.mean(quality)) < 0.2:
        warnings.append(
            "GPA Bragg sideband quality mask is sparse; strain maps may be unreliable."
        )
    min_quality = (
        np.minimum.reduce(normalized) if normalized else np.zeros_like(quality, dtype=float)
    )
    uncertainty = np.clip(1.0 - min_quality, 0.0, 1.0)
    return quality.astype(bool), uncertainty.astype(np.float64)


def _displacement_from_phases(
    phase_maps: list[np.ndarray],
    g_vectors: np.ndarray,
    *,
    quality_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    g_inv = np.linalg.inv(np.asarray(g_vectors, dtype=np.float64))
    p1, p2 = [
        _center_unwrapped_phase(
            _unwrap_phase_for_displacement(np.asarray(phase, dtype=np.float64), quality_mask),
            quality_mask,
        )
        for phase in phase_maps
    ]
    factor = -1.0 / _TWO_PI
    ux = factor * (g_inv[0, 0] * p1 + g_inv[0, 1] * p2)
    uy = factor * (g_inv[1, 0] * p1 + g_inv[1, 1] * p2)
    return _mask_fill(ux, quality_mask), _mask_fill(uy, quality_mask)


def _center_unwrapped_phase(
    phase: np.ndarray,
    quality_mask: np.ndarray | None,
) -> np.ndarray:
    arr = np.asarray(phase, dtype=np.float64)
    if quality_mask is None or quality_mask.shape != arr.shape or not np.any(quality_mask):
        return arr
    samples = arr[np.asarray(quality_mask, dtype=bool)]
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return arr
    return arr - float(np.median(samples))


def _strain_from_phase_gradients(
    phase_maps: list[np.ndarray],
    g_vectors: np.ndarray,
    *,
    quality_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gradients = [
        _complex_phase_gradient(np.asarray(phase, dtype=np.float64)) for phase in phase_maps
    ]
    d1dx, d1dy = gradients[0]
    d2dx, d2dy = gradients[1]
    g_inv = np.linalg.inv(np.asarray(g_vectors, dtype=np.float64))
    factor = -1.0 / _TWO_PI
    ux_x = factor * (g_inv[0, 0] * d1dx + g_inv[0, 1] * d2dx)
    ux_y = factor * (g_inv[0, 0] * d1dy + g_inv[0, 1] * d2dy)
    uy_x = factor * (g_inv[1, 0] * d1dx + g_inv[1, 1] * d2dx)
    uy_y = factor * (g_inv[1, 0] * d1dy + g_inv[1, 1] * d2dy)
    strain_xx = ux_x
    strain_yy = uy_y
    strain_xy = 0.5 * (ux_y + uy_x)
    rotation_xy = 0.5 * (uy_x - ux_y)
    return (
        _mask_fill(strain_xx, quality_mask),
        _mask_fill(strain_yy, quality_mask),
        _mask_fill(strain_xy, quality_mask),
        _mask_fill(rotation_xy, quality_mask),
    )


def _complex_phase_gradient(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = np.exp(1j * np.asarray(phase, dtype=np.float64))
    derivative_kernel = np.asarray([0.5, 0.0, -0.5], dtype=np.float64)
    dz_dy = ndi.convolve1d(z, weights=derivative_kernel, axis=0, mode="nearest")
    dz_dx = ndi.convolve1d(z, weights=derivative_kernel, axis=1, mode="nearest")
    dphi_dx = np.imag(np.conj(z) * dz_dx)
    dphi_dy = np.imag(np.conj(z) * dz_dy)
    return dphi_dx, dphi_dy


def _mask_fill(values: np.ndarray, quality_mask: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if quality_mask is None or quality_mask.shape != arr.shape:
        return np.nan_to_num(arr, copy=False)
    filled = np.nan_to_num(arr, copy=True)
    filled[~quality_mask] = 0.0
    return filled


def _append_field_sanity_warnings(
    *,
    image_shape: tuple[int, int],
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    strain_xx: np.ndarray,
    strain_yy: np.ndarray,
    strain_xy: np.ndarray,
    rotation_xy: np.ndarray,
    valid_mask: np.ndarray,
    warnings: list[str],
) -> None:
    if valid_mask.shape != tuple(image_shape) or not np.any(valid_mask):
        warnings.append("GPA quality mask has no valid pixels; output maps are not reliable.")
        return
    valid_fraction = float(np.mean(valid_mask))
    if valid_fraction < 0.2:
        warnings.append(
            f"GPA valid-pixel fraction is only {valid_fraction:.3f}; "
            "maps should be treated as diagnostic only."
        )
    max_reasonable_displacement = max(float(min(image_shape) * 0.25), 8.0)
    disp_abs = np.concatenate(
        [
            np.abs(np.asarray(displacement_x, dtype=np.float64)[valid_mask]).ravel(),
            np.abs(np.asarray(displacement_y, dtype=np.float64)[valid_mask]).ravel(),
        ]
    )
    if disp_abs.size and float(np.nanpercentile(disp_abs, 95.0)) > max_reasonable_displacement:
        warnings.append(
            "GPA displacement magnitude is far above a plausible local lattice shift; "
            "check g-vector units, Bragg peak selection, and whether the input contains "
            "continuous lattice fringes."
        )
    strain_abs = np.concatenate(
        [
            np.abs(np.asarray(strain_xx, dtype=np.float64)[valid_mask]).ravel(),
            np.abs(np.asarray(strain_yy, dtype=np.float64)[valid_mask]).ravel(),
            np.abs(np.asarray(strain_xy, dtype=np.float64)[valid_mask]).ravel(),
        ]
    )
    if strain_abs.size and float(np.nanpercentile(strain_abs, 95.0)) > 0.2:
        warnings.append(
            "GPA strain exceeds 20% over much of the valid region; this usually indicates "
            "bad basis vectors, a non-lattice image, or an overly broad/noisy Bragg mask."
        )
    rotation_abs = np.abs(np.asarray(rotation_xy, dtype=np.float64)[valid_mask]).ravel()
    if rotation_abs.size and float(np.nanpercentile(rotation_abs, 95.0)) > 0.2:
        warnings.append(
            "GPA rotation exceeds 0.2 rad over much of the valid region; verify the "
            "selected Bragg sidebands and reference ROI."
        )


def _strain_map_stats(
    *,
    displacement_x: np.ndarray,
    displacement_y: np.ndarray,
    strain_xx: np.ndarray,
    strain_yy: np.ndarray,
    strain_xy: np.ndarray,
    rotation_xy: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    return {
        "mean_ux": _masked_mean(displacement_x, valid_mask),
        "mean_uy": _masked_mean(displacement_y, valid_mask),
        "mean_exx": _masked_mean(strain_xx, valid_mask),
        "mean_eyy": _masked_mean(strain_yy, valid_mask),
        "mean_exy": _masked_mean(strain_xy, valid_mask),
        "mean_rotation": _masked_mean(rotation_xy, valid_mask),
        "std_exx": _masked_std(strain_xx, valid_mask),
        "std_eyy": _masked_std(strain_yy, valid_mask),
        "std_exy": _masked_std(strain_xy, valid_mask),
        "std_rotation": _masked_std(rotation_xy, valid_mask),
    }


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(arr, dtype=np.float64)[mask]
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _masked_std(arr: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(arr, dtype=np.float64)[mask]
    if values.size == 0:
        return 0.0
    return float(np.std(values))


def _estimate_confidence(quality_mask: np.ndarray, warnings: list[str]) -> float:
    valid_fraction = float(np.mean(quality_mask)) if quality_mask.size else 0.0
    penalty = min(0.4, 0.06 * len(warnings))
    return float(np.clip(0.35 + 0.65 * valid_fraction - penalty, 0.0, 1.0))


def _normalize_reference_regions(
    reference_regions: ReferenceRegions | None,
    reference_region: ReferenceRegion | None,
    image_shape: tuple[int, int],
    warnings: list[str],
) -> ReferenceRegions:
    if reference_regions is None:
        ref = _normalize_reference_region(reference_region, image_shape, warnings)
        return ref, ref
    if len(reference_regions) != 2:
        raise ValueError("GPA reference_regions must contain exactly two entries for g1 and g2.")
    if reference_region is not None:
        warnings.append(
            "Both GPA reference_region and reference_regions were provided; using per-g reference_regions."
        )
    refs = tuple(
        _normalize_reference_region(
            region,
            image_shape,
            warnings,
            label=f"reference_regions[{index}]",
            warn_if_missing=False,
        )
        for index, region in enumerate(reference_regions)
    )
    if refs[0] is None and refs[1] is None:
        warnings.append(
            "GPA reference_regions not provided; using supplied g-vectors as the absolute reference."
        )
    return refs  # type: ignore[return-value]


def _common_reference_region(refs: ReferenceRegions) -> ReferenceRegion | None:
    if refs[0] is not None and refs[0] == refs[1]:
        return refs[0]
    return None


def _reference_regions_to_json(refs: ReferenceRegions) -> dict[str, list[int] | None]:
    return {
        f"g{index + 1}": list(region) if region is not None else None
        for index, region in enumerate(refs)
    }


def _normalize_reference_region(
    reference_region: ReferenceRegion | None,
    image_shape: tuple[int, int],
    warnings: list[str],
    *,
    label: str = "reference_region",
    warn_if_missing: bool = True,
) -> ReferenceRegion | None:
    if reference_region is None:
        if warn_if_missing:
            warnings.append(
                f"GPA {label} not provided; using supplied g-vectors as the absolute reference."
            )
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    y0, y1, x0, x1 = [int(v) for v in reference_region]
    y0 = int(np.clip(y0, 0, h))
    y1 = int(np.clip(y1, 0, h))
    x0 = int(np.clip(x0, 0, w))
    x1 = int(np.clip(x1, 0, w))
    if y1 <= y0 or x1 <= x0:
        warnings.append(
            f"GPA {label} is empty after clipping; reference refinement disabled."
        )
        return None
    return y0, y1, x0, x1


def _gvector_model(gx: float, gy: float, *, source_peak_index: int | None = None) -> GVector:
    magnitude = float(np.hypot(gx, gy))
    angle = float(np.degrees(np.arctan2(gy, gx)))
    return GVector(
        gx=Measurement(value=float(gx), unit="1/pixel", method="gpa_refined"),
        gy=Measurement(value=float(gy), unit="1/pixel", method="gpa_refined"),
        magnitude=Measurement(value=magnitude, unit="1/pixel", method="gpa_refined"),
        angle=Measurement(value=angle, unit="deg", method="atan2"),
        source_peak_index=source_peak_index,
    )


def _to_array_ref(
    arr: np.ndarray,
    *,
    description: str,
    inline_mode: str,
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


def _strain_from_displacement(
    ux: np.ndarray,
    uy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return strain/rotation from displacement maps for tests and fallback workflows."""

    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    dux_dy, dux_dx = np.gradient(ux)
    duy_dy, duy_dx = np.gradient(uy)
    exx = dux_dx
    eyy = duy_dy
    exy = 0.5 * (dux_dy + duy_dx)
    rotation = 0.5 * (duy_dx - dux_dy)
    return exx, eyy, exy, rotation
