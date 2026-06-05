"""Export helpers for C2 GPA field results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from autotem.strain.gpa import GPAFieldResult

try:  # pragma: no cover - exercised through whichever writer is installed.
    import tifffile
except Exception:  # noqa: BLE001
    tifffile = None

SUPPORTED_GPA_EXPORT_COLORMAPS = [
    "Greyscale",
    "Turbo",
    "Matplotlib Turbo",
    "Thermal",
    "Polar",
    "BlOr",
]

GPA_COLORBAR_HEIGHT_PX = 512
GPA_COLORBAR_WIDTH_PX = 100
GPA_COLORBAR_STRIP_WIDTH_PX = 20

GPA_BLOR_STOPS = [
    (0.0, "#075afe"),
    (0.5, "#ebffeb"),
    (1.0, "#ff5500"),
]

GPA_TURBO_STOPS = [
    (0.0, "#331b3d"),
    (0.125, "#4d6edf"),
    (0.25, "#3db9e9"),
    (0.375, "#44ee9a"),
    (0.5, "#a4fa50"),
    (0.625, "#ebce4c"),
    (0.75, "#f78137"),
    (0.875, "#ce3a20"),
    (1.0, "#771513"),
]

GPA_THERMAL_STOPS = [
    (0.0, "#000032"),
    (0.15, "#140078"),
    (0.33, "#c81e8c"),
    (0.6, "#ff6400"),
    (0.85, "#ffff28"),
    (1.0, "#ffffff"),
]

GPA_POLAR_STOPS = [
    (0.0, "#32ffff"),
    (0.18, "#0a46ff"),
    (0.28, "#0a0abe"),
    (0.5, "#000000"),
    (0.72, "#be0a0a"),
    (0.82, "#ff460a"),
    (1.0, "#ffff32"),
]


def get_gpa_colormap(name: str) -> Colormap:
    """Return a GPA display colormap for C2 GPA maps."""

    key = str(name or "BlOr").strip().casefold()
    if key in {"greyscale", "grayscale", "grey", "gray"}:
        return plt.get_cmap("gray")
    if key in {"turbo", "jetlike", "jet-like"}:
        return LinearSegmentedColormap.from_list(
            "autotem_gpa_turbo",
            GPA_TURBO_STOPS,
            N=1024,
        )
    if key in {"matplotlib turbo", "matplotlib_turbo", "mpl turbo", "mpl_turbo"}:
        return plt.get_cmap("turbo")
    if key == "thermal":
        return LinearSegmentedColormap.from_list(
            "autotem_gpa_thermal",
            GPA_THERMAL_STOPS,
            N=1024,
        )
    if key == "polar":
        return LinearSegmentedColormap.from_list(
            "autotem_gpa_polar",
            GPA_POLAR_STOPS,
            N=1024,
        )
    return LinearSegmentedColormap.from_list(
        "autotem_gpa_blor",
        GPA_BLOR_STOPS,
        N=1024,
    )


def export_gpa_maps(
    fields: GPAFieldResult,
    output_dir: str | Path,
    *,
    image: np.ndarray | None = None,
    fft_log_magnitude: np.ndarray | None = None,
    fft_center_xy: tuple[float, float] | None = None,
    selected_points_xy: list[tuple[float, float]] | None = None,
    colormap_name: str = "BlOr",
    strain_scale: float = 100.0,
    strain_unit: str = "%",
    displacement_abs_limit: float = 0.0,
    strain_abs_limit: float = 0.0,
    rotation_abs_limit: float = 0.0,
    display_smoothing_sigma_px: float = 0.0,
) -> dict[str, Any]:
    """Export GPA arrays, RGB TIFFs, FFT overlays, and phase intermediates."""

    out_dir = Path(output_dir)
    data_dir = out_dir / "data"
    rgb_dir = out_dir / f"rgb_{_safe_token(colormap_name)}"
    data_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = np.asarray(fields.quality_mask, dtype=bool)
    scalar_data = _field_data(fields)
    display_fields, limits = _display_fields(
        fields,
        valid_mask=valid_mask,
        strain_scale=float(strain_scale),
        displacement_abs_limit=float(displacement_abs_limit),
        strain_abs_limit=float(strain_abs_limit),
        rotation_abs_limit=float(rotation_abs_limit),
    )
    cmap = get_gpa_colormap(colormap_name)
    artifacts: list[dict[str, Any]] = []

    for name, arr in scalar_data.items():
        data_path = data_dir / f"{name}.tif"
        npy_path = data_dir / f"{name}.npy"
        _write_tiff(data_path, np.asarray(arr, dtype=np.float32))
        np.save(npy_path, np.asarray(arr, dtype=np.float32))
        artifacts.extend(
            [
                {"kind": "data_tiff", "field": name, "path": str(data_path)},
                {"kind": "data_npy", "field": name, "path": str(npy_path)},
            ]
        )

    quality_path = data_dir / "quality_mask.tif"
    _write_tiff(quality_path, valid_mask.astype(np.uint8) * 255)
    artifacts.append({"kind": "mask", "field": "quality_mask", "path": str(quality_path)})

    for name, arr in display_fields.items():
        limit = limits.get(name, 1.0)
        rgb_path = rgb_dir / f"{name}.tif"
        display_arr = smooth_gpa_display_values(
            arr,
            sigma_px=float(display_smoothing_sigma_px),
            valid_mask=None,
        )
        rgb = render_gpa_rgb(display_arr, colormap_name=colormap_name, vmin=-limit, vmax=limit)
        _write_tiff(rgb_path, rgb)
        artifacts.append({"kind": "rgb_tiff", "field": name, "path": str(rgb_path)})

    colorbar_path = rgb_dir / "ColorBar.tif"
    _write_tiff(colorbar_path, _render_colorbar_rgba(cmap))
    artifacts.append({"kind": "rgb_tiff", "field": "ColorBar", "path": str(colorbar_path)})

    if image is not None:
        image_data_path = data_dir / "image.tif"
        image_rgb_path = rgb_dir / "image.tif"
        _write_tiff(image_data_path, np.asarray(image, dtype=np.float32))
        _write_tiff(image_rgb_path, _gray_rgb(image))
        artifacts.extend(
            [
                {"kind": "data_tiff", "field": "image", "path": str(image_data_path)},
                {"kind": "rgb_tiff", "field": "image", "path": str(image_rgb_path)},
            ]
        )

    fft_rgb = _fft_rgb(
        image=image,
        fft_log_magnitude=fft_log_magnitude,
        fft_center_xy=fft_center_xy,
        selected_points_xy=selected_points_xy,
    )
    if fft_rgb is not None:
        fft_path = rgb_dir / "FFT.tif"
        _write_tiff(fft_path, fft_rgb)
        artifacts.append({"kind": "rgb_tiff", "field": "FFT", "path": str(fft_path)})

    artifacts.extend(_export_phase_maps(fields, data_dir=data_dir, rgb_dir=rgb_dir))

    manifest = {
        "format": "autotem_c2_gpa_export",
        "colormap": str(colormap_name),
        "strain_display_unit": str(strain_unit),
        "strain_scale": float(strain_scale),
        "display_smoothing_sigma_px": float(display_smoothing_sigma_px),
        "limits": limits,
        "quality": {
            "valid_pixel_fraction": float(np.mean(valid_mask)) if valid_mask.size else 0.0,
        },
        "g_vectors": [
            {
                "g_vector_cycles_per_pixel": list(phase.g_vector_cycles),
                "g_vector_fft_pixels": list(phase.g_vector_pixels),
            }
            for phase in fields.phase_maps
        ],
        "artifacts": artifacts,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def render_gpa_rgb(
    values: np.ndarray,
    *,
    colormap_name: str,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Render a scalar field as uint8 RGB using the C2 GPA colormap choices."""

    arr = np.nan_to_num(np.asarray(values, dtype=np.float64), copy=True)
    span = max(float(vmax) - float(vmin), 1e-12)
    norm = np.clip((arr - float(vmin)) / span, 0.0, 1.0)
    rgba = get_gpa_colormap(colormap_name)(norm)
    return np.asarray(np.round(rgba[..., :3] * 255.0), dtype=np.uint8)


def _field_data(fields: GPAFieldResult) -> dict[str, np.ndarray]:
    exy_dist = np.asarray(fields.strain_xy, dtype=np.float64) - np.asarray(
        fields.rotation_xy,
        dtype=np.float64,
    )
    eyx_dist = np.asarray(fields.strain_xy, dtype=np.float64) + np.asarray(
        fields.rotation_xy,
        dtype=np.float64,
    )
    return {
        "ux": np.asarray(fields.displacement_x, dtype=np.float64),
        "uy": np.asarray(fields.displacement_y, dtype=np.float64),
        "exx": np.asarray(fields.strain_xx, dtype=np.float64),
        "exy": exy_dist,
        "eyx": eyx_dist,
        "eyy": np.asarray(fields.strain_yy, dtype=np.float64),
        "shear_exy": np.asarray(fields.strain_xy, dtype=np.float64),
        "wxy": np.asarray(fields.rotation_xy, dtype=np.float64),
        "uncertainty": np.asarray(fields.uncertainty, dtype=np.float64),
    }


def _display_fields(
    fields: GPAFieldResult,
    *,
    valid_mask: np.ndarray,
    strain_scale: float,
    displacement_abs_limit: float,
    strain_abs_limit: float,
    rotation_abs_limit: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    data = _field_data(fields)
    display = {
        "ux": data["ux"],
        "uy": data["uy"],
        "exx": data["exx"] * strain_scale,
        "exy": data["exy"] * strain_scale,
        "eyx": data["eyx"] * strain_scale,
        "eyy": data["eyy"] * strain_scale,
        "wxy": data["wxy"],
    }
    disp_lim = (
        max(
            _robust_abs_limit(display["ux"], valid_mask=valid_mask, fallback=0.1),
            _robust_abs_limit(display["uy"], valid_mask=valid_mask, fallback=0.1),
        )
        if displacement_abs_limit <= 0
        else float(displacement_abs_limit)
    )
    strain_lim = (
        max(
            _robust_abs_limit(display["exx"], valid_mask=valid_mask, fallback=0.01 * strain_scale),
            _robust_abs_limit(display["exy"], valid_mask=valid_mask, fallback=0.01 * strain_scale),
            _robust_abs_limit(display["eyx"], valid_mask=valid_mask, fallback=0.01 * strain_scale),
            _robust_abs_limit(display["eyy"], valid_mask=valid_mask, fallback=0.01 * strain_scale),
        )
        if strain_abs_limit <= 0
        else float(strain_abs_limit)
    )
    rot_lim = (
        _robust_abs_limit(display["wxy"], valid_mask=valid_mask, fallback=0.01)
        if rotation_abs_limit <= 0
        else float(rotation_abs_limit)
    )
    return display, {
        "ux": disp_lim,
        "uy": disp_lim,
        "exx": strain_lim,
        "exy": strain_lim,
        "eyx": strain_lim,
        "eyy": strain_lim,
        "wxy": rot_lim,
    }


def smooth_gpa_display_values(
    values: np.ndarray,
    *,
    sigma_px: float,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Apply display-only normalized Gaussian smoothing without changing data exports."""

    sigma = float(sigma_px)
    arr = np.asarray(values, dtype=np.float64)
    if sigma <= 0:
        return arr
    finite_mask = np.isfinite(arr)
    if valid_mask is not None and valid_mask.shape == arr.shape:
        finite_mask &= np.asarray(valid_mask, dtype=bool)
    if not np.any(finite_mask):
        return arr
    weights = finite_mask.astype(np.float64)
    numerator = gaussian_filter(np.where(finite_mask, arr, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(weights, sigma=sigma, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = numerator / np.maximum(denominator, 1e-12)
    return np.where(denominator > 1e-6, smoothed, np.nan)


def _export_phase_maps(
    fields: GPAFieldResult,
    *,
    data_dir: Path,
    rgb_dir: Path,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for idx, phase in enumerate(fields.phase_maps, start=1):
        prefix = f"phase_{idx}"
        phase_arrays = {
            f"{prefix}_raw_phase": np.asarray(phase.phase_wrapped, dtype=np.float64),
            f"{prefix}_phase": np.asarray(phase.phase_unwrapped, dtype=np.float64),
            f"{prefix}_bragg_image": np.asarray(phase.bragg_amplitude, dtype=np.float64),
            f"{prefix}_mask": np.asarray(phase.gaussian_mask, dtype=np.float64),
        }
        dphase_dy, dphase_dx = np.gradient(np.asarray(phase.phase_wrapped, dtype=np.float64))
        phase_arrays[f"{prefix}_dphasedx"] = dphase_dx
        phase_arrays[f"{prefix}_dphasedy"] = dphase_dy
        for name, arr in phase_arrays.items():
            path = data_dir / f"{name}.tif"
            _write_tiff(path, np.asarray(arr, dtype=np.float32))
            np.save(data_dir / f"{name}.npy", np.asarray(arr, dtype=np.float32))
            artifacts.append({"kind": "phase_data", "field": name, "path": str(path)})

        display_specs = {
            f"Phase {idx} Bragg image.tif": phase_arrays[f"{prefix}_bragg_image"],
            f"Phase {idx} Raw phase.tif": phase_arrays[f"{prefix}_raw_phase"],
            f"Phase {idx} Phase.tif": phase_arrays[f"{prefix}_phase"],
            f"Phase {idx} Normalised phase.tif": phase_arrays[f"{prefix}_raw_phase"],
            f"Phase {idx} dPhasedx.tif": phase_arrays[f"{prefix}_dphasedx"],
            f"Phase {idx} dPhasedy.tif": phase_arrays[f"{prefix}_dphasedy"],
            f"Phase {idx} Mask.tif": phase_arrays[f"{prefix}_mask"],
            f"Phase {idx} Masked FFT.tif": phase_arrays[f"{prefix}_mask"],
        }
        for filename, arr in display_specs.items():
            path = rgb_dir / filename
            _write_tiff(path, _gray_rgb(arr))
            artifacts.append({"kind": "phase_rgb", "field": filename[:-4], "path": str(path)})
    return artifacts


def _fft_rgb(
    *,
    image: np.ndarray | None,
    fft_log_magnitude: np.ndarray | None,
    fft_center_xy: tuple[float, float] | None,
    selected_points_xy: list[tuple[float, float]] | None,
) -> np.ndarray | None:
    if fft_log_magnitude is None and image is not None:
        arr = np.asarray(image, dtype=np.float64)
        fft_log_magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(arr))))
    if fft_log_magnitude is None:
        return None
    rgb = _gray_rgb(fft_log_magnitude)
    if selected_points_xy:
        pil = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil)
        colors = [(255, 0, 0), (0, 80, 255)]
        if fft_center_xy is not None:
            cx, cy = float(fft_center_xy[0]), float(fft_center_xy[1])
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), outline=(255, 255, 0), width=1)
        for idx, (x, y) in enumerate(selected_points_xy):
            color = colors[idx % len(colors)]
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=color, width=3)
        rgb = np.asarray(pil, dtype=np.uint8)
    return rgb


def _render_colorbar_rgba(
    cmap: Colormap,
    *,
    height: int = GPA_COLORBAR_HEIGHT_PX,
    width: int = GPA_COLORBAR_WIDTH_PX,
    strip_width: int = GPA_COLORBAR_STRIP_WIDTH_PX,
) -> np.ndarray:
    grad = np.linspace(1.0, 0.0, height, dtype=np.float64)[:, None]
    strip = np.asarray(np.round(cmap(np.repeat(grad, strip_width, axis=1)) * 255.0), dtype=np.uint8)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :strip_width, :] = strip
    rgba[:, :strip_width, 3] = 255
    return rgba


def _gray_rgb(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64), copy=True)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        norm = np.zeros_like(arr, dtype=np.float64)
    else:
        lo, hi = np.percentile(finite, [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(np.min(finite)), float(np.max(finite))
        span = max(float(hi - lo), 1e-12)
        norm = np.clip((arr - lo) / span, 0.0, 1.0)
    gray = np.asarray(np.round(norm * 255.0), dtype=np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _robust_abs_limit(
    values: np.ndarray,
    *,
    valid_mask: np.ndarray,
    fallback: float,
) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if valid_mask.shape == arr.shape and np.any(valid_mask):
        arr = arr[valid_mask]
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(fallback)
    return max(float(np.percentile(np.abs(finite), 98.0)), float(fallback), 1e-9)


def _write_tiff(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if tifffile is not None:
        kwargs: dict[str, Any] = {}
        if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
            kwargs["photometric"] = "rgb"
        tifffile.imwrite(str(path), arr, **kwargs)
        return
    Image.fromarray(arr).save(path, format="TIFF")


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))
    return token or "BlOr"
