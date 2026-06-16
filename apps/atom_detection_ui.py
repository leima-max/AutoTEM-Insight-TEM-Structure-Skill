from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from typing import Any

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.elements.image as st_image
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from streamlit_drawable_canvas import st_canvas

from autotem.atomic.atom_detection import _load_image, _normalize_image, detect_atoms
from autotem.atomic.displacement import (
    AtomicDisplacementResult,
    compute_atomic_displacement_from_basis,
)
from autotem.atomic.distance_statistics import compute_distance_statistics
from autotem.atomic.gaussian_refinement import refine_atom_coordinates
from autotem.atomic.intensity_quantification import (
    AtomicColumnIntensityResult,
    AtomicIntensityGroupResult,
    cluster_atomic_column_intensities,
    group_atomic_column_intensities_by_lattice,
    quantify_atomic_column_intensity,
)
from autotem.atomic.lattice_constraint import apply_lattice_constraint
from autotem.atomic.plane_spacing import (
    AtomicPlaneSpacingResult,
    compute_atomic_plane_spacing_segments_by_angle,
    compute_atomic_plane_spacing_segments_from_angle_constrained_basis,
    compute_atomic_plane_spacing_segments_from_basis,
    compute_atomic_plane_spacing_segments_from_lattice_angle,
    compute_atomic_plane_spacing_segments_from_picked_direction,
)
from autotem.atomic.sublattice import (
    SublatticeClassificationResult,
    apply_manual_sublattice_labels,
    classify_sublattices_auto,
    classify_sublattices_from_model,
    parse_projected_model_csv,
)
from autotem.io import load_tem_dataset
from autotem.io.metadata import normalize_metadata
from autotem.strain.atom_strain import AtomStrainFieldResult, compute_atom_strain_fields

# Compatibility patch:
# streamlit-drawable-canvas currently expects `streamlit.elements.image.image_to_url`
# which was moved in newer Streamlit versions.
if not hasattr(st_image, "image_to_url"):
    from types import SimpleNamespace

    from streamlit.elements.lib import image_utils as _st_image_utils

    def _image_to_url_compat(image, width, clamp, channels, output_format, image_id):
        layout_config = SimpleNamespace(width=width)
        return _st_image_utils.image_to_url(
            image=image,
            layout_config=layout_config,
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    st_image.image_to_url = _image_to_url_compat


st.set_page_config(page_title="AutoTEM Atom Detection (C0)", layout="wide")
st.title("AutoTEM-Insight: Atom-Column Detection (C0)")
st.caption("Support: dm/dm3/dm4/tif/tiff + LoG/DoG + optional Gaussian filter")

DISPLACEMENT_METHOD_MANUAL_BASIS = "Method 1 - manual base/a/b"
DISPLACEMENT_METHOD_REFERENCE_ROI = "Method 2 - reference ROI fit + custom a/b direction"
DISPLACEMENT_METHOD_OPTIONS = [
    DISPLACEMENT_METHOD_MANUAL_BASIS,
    DISPLACEMENT_METHOD_REFERENCE_ROI,
]


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_load_image(
    path: str, slice_index: int, file_size: int
) -> tuple[np.ndarray, dict[str, Any], list[str]]:
    # `file_size` is included so cache invalidates on changed uploads.
    _ = file_size
    warnings: list[str] = []
    arr, source_info = _load_image(path, warnings=warnings, slice_index=slice_index)
    return arr, source_info, warnings


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_infer_pixel_size_nm(path: str, file_size: int) -> tuple[float | None, str, list[str]]:
    # `file_size` is included so cache invalidates on changed uploads.
    _ = file_size
    suffix = Path(path).suffix.lower()
    messages: list[str] = []

    if suffix in {".dm", ".dm3", ".dm4"}:
        try:
            from ncempy.io import dm as dm_io

            payload = dm_io.dmReader(str(path))
            raw = dict(payload) if isinstance(payload, dict) else {"payload": payload}
            raw.pop("data", None)
            normalized, warnings = normalize_metadata(raw, source_format=suffix)
            pixel_size_nm = normalized.get("pixel_size_nm")
            if pixel_size_nm is not None:
                return float(pixel_size_nm), "DM metadata (ncempy + normalize_metadata)", messages
            for w in warnings:
                if w.get("code") in {"META_MISSING_PIXEL_SIZE", "META_INVALID_PIXEL_SIZE"}:
                    messages.append(f"{w.get('code')}: {w.get('message')}")
        except Exception as exc:
            messages.append(f"DM metadata read failed: {exc}")

    try:
        dataset = load_tem_dataset(path)
        if dataset.pixel_size is not None and dataset.pixel_size.value is not None:
            unit = str(dataset.pixel_size.unit or "").lower()
            value = float(dataset.pixel_size.value)
            if unit.startswith("nm"):
                return value, "TEMDataset pixel_size", messages
            if unit.startswith("pm"):
                return value * 0.001, "TEMDataset pixel_size (pm->nm)", messages
            if "angstrom" in unit or unit in {"a", "a/pixel"}:
                return value * 0.1, "TEMDataset pixel_size (A->nm)", messages
            messages.append(f"Unsupported pixel_size unit from dataset: {dataset.pixel_size.unit}")

        normalized = (
            dataset.metadata.get("normalized", {})
            if isinstance(dataset.metadata, dict)
            else {}
        )
        norm_val = normalized.get("pixel_size_nm")
        if norm_val is not None:
            return float(norm_val), "TEMDataset normalized metadata", messages
    except Exception as exc:
        messages.append(f"Generic metadata read failed: {exc}")

    return None, "unavailable", messages


def _save_upload_to_temp(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".dat"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def _signature(file_sig: str, params: dict[str, Any]) -> tuple[Any, ...]:
    return (
        file_sig,
        params["mode"],
        params["gaussian_sigma"],
        params["min_distance"],
        params["threshold_rel"],
        params["max_atoms"],
        params["slice_index"],
        params["refinement_method"],
        params["refinement_window_radius"],
        params["refinement_max_shift"],
        params["lattice_constraint_enable"],
        params["lattice_k_neighbors"],
        params["lattice_strength"],
        params["lattice_iterations"],
        params["lattice_max_shift_per_iter"],
        params["fast_mode_realtime"],
        params["fast_gaussian_max_points"],
    )


def _warning_summary_lines(
    warnings: list[str], max_items: int = 3, max_chars: int = 180
) -> list[str]:
    if not warnings:
        return []
    lines: list[str] = []
    for w in warnings[:max_items]:
        text = w.strip()
        if len(text) > max_chars:
            text = text[: max_chars - 3] + "..."
        lines.append(text)
    if len(warnings) > max_items:
        lines.append(f"... and {len(warnings) - max_items} more warnings")
    return lines


def _sample_intensity(image: np.ndarray, y: float, x: float) -> float:
    yi = int(np.clip(round(y), 0, image.shape[0] - 1))
    xi = int(np.clip(round(x), 0, image.shape[1] - 1))
    return float(image[yi, xi])


def _render_overlay(
    image_norm: np.ndarray,
    coords: np.ndarray,
    roi_mask: np.ndarray | None,
    reference_atom_mask: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=160)
    ax.imshow(image_norm, cmap="gray")
    if roi_mask is not None:
        roi_show = np.ma.masked_where(~roi_mask, roi_mask)
        ax.imshow(roi_show, cmap="spring", alpha=0.28, interpolation="none")
    if len(coords) > 0:
        ax.scatter(
            coords[:, 1],
            coords[:, 0],
            s=14,
            c="red",
            marker="o",
            linewidths=0.35,
            label="edited atoms",
        )
        if reference_atom_mask is not None and np.any(reference_atom_mask):
            ref_mask = np.asarray(reference_atom_mask, dtype=bool)
            ax.scatter(
                coords[ref_mask, 1],
                coords[ref_mask, 0],
                s=28,
                facecolors="none",
                edgecolors="#00ffff",
                linewidths=0.8,
                label="ideal lattice reference",
            )
            ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.set_title(f"Editable atoms: {len(coords)}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _coordinates_csv(coords: np.ndarray, intensities: np.ndarray) -> str:
    buffer = io.StringIO()
    buffer.write("y,x,intensity\n")
    for (y, x), v in zip(coords, intensities, strict=False):
        buffer.write(f"{float(y):.6f},{float(x):.6f},{float(v):.6f}\n")
    return buffer.getvalue()


def _distance_csv(
    coords: np.ndarray,
    nearest: np.ndarray,
    next_nearest: np.ndarray,
    unit: str,
) -> str:
    buffer = io.StringIO()
    buffer.write(f"unit,{unit}\n")
    buffer.write("atom_index,y,x,nearest_distance,next_nearest_distance\n")
    for idx, ((y, x), nn, nnn) in enumerate(
        zip(coords, nearest, next_nearest, strict=False)
    ):
        nn_text = "" if not np.isfinite(nn) else f"{float(nn):.6f}"
        nnn_text = "" if not np.isfinite(nnn) else f"{float(nnn):.6f}"
        buffer.write(f"{idx},{float(y):.6f},{float(x):.6f},{nn_text},{nnn_text}\n")
    return buffer.getvalue()


def _atom_strain_csv(
    coords: np.ndarray,
    fields: AtomStrainFieldResult,
    displacement_scale: float,
    displacement_unit: str,
    strain_scale: float,
    strain_unit: str,
    custom_components: dict[str, np.ndarray] | None = None,
) -> str:
    buffer = io.StringIO()
    buffer.write(f"displacement_unit,{displacement_unit}\n")
    buffer.write(f"strain_unit,{strain_unit}\n")
    header = ["atom_index", "y", "x", "ux", "uy", "exx", "eyy", "exy"]
    if custom_components is not None:
        header.extend(["eaa", "ebb", "eab"])
    header.extend(["rotation", "uncertainty", "valid_fit"])
    buffer.write(",".join(header) + "\n")
    for idx, (coord, disp_xy, exx, eyy, exy, rot, unc, valid) in enumerate(
        zip(
            coords,
            fields.atom_displacement_xy,
            fields.atom_strain_xx,
            fields.atom_strain_yy,
            fields.atom_strain_xy,
            fields.atom_rotation_xy,
            fields.atom_uncertainty,
            fields.valid_atom_mask,
            strict=False,
        )
    ):
        y, x = coord
        ux = float(disp_xy[0]) * displacement_scale
        uy = float(disp_xy[1]) * displacement_scale
        exx_scaled = float(exx) * strain_scale
        eyy_scaled = float(eyy) * strain_scale
        exy_scaled = float(exy) * strain_scale
        row = [
            f"{idx}",
            f"{float(y):.6f}",
            f"{float(x):.6f}",
            f"{ux:.6f}",
            f"{uy:.6f}",
            f"{exx_scaled:.6f}",
            f"{eyy_scaled:.6f}",
            f"{exy_scaled:.6f}",
        ]
        if custom_components is not None:
            row.extend(
                [
                    f"{float(custom_components['eaa'][idx]) * strain_scale:.6f}",
                    f"{float(custom_components['ebb'][idx]) * strain_scale:.6f}",
                    f"{float(custom_components['eab'][idx]) * strain_scale:.6f}",
                ]
            )
        row.extend([f"{float(rot):.6f}", f"{float(unc):.6f}", f"{int(bool(valid))}"])
        buffer.write(",".join(row) + "\n")
    return buffer.getvalue()


def _atomic_displacement_csv(
    coords: np.ndarray,
    result: AtomicDisplacementResult,
    displacement_scale: float,
    displacement_unit: str,
) -> str:
    buffer = io.StringIO()
    buffer.write(f"displacement_unit,{displacement_unit}\n")
    buffer.write(
        "atom_index,y,x,ideal_y,ideal_x,uy,ux,u_mag,lattice_i,lattice_j,"
        "valid_assignment,reference_atom,fit_atom\n"
    )
    for idx in range(len(coords)):
        obs_y, obs_x = coords[idx]
        ideal_y, ideal_x = result.coordinates_ideal_yx[idx]
        disp_y, disp_x = result.displacement_yx[idx]
        row = [
            f"{idx}",
            f"{float(obs_y):.6f}",
            f"{float(obs_x):.6f}",
            f"{float(ideal_y):.6f}",
            f"{float(ideal_x):.6f}",
            f"{float(disp_y) * displacement_scale:.6f}",
            f"{float(disp_x) * displacement_scale:.6f}",
            f"{float(result.displacement_magnitude[idx]) * displacement_scale:.6f}",
            f"{int(result.lattice_indices_ij[idx, 0])}",
            f"{int(result.lattice_indices_ij[idx, 1])}",
            f"{int(bool(result.valid_atom_mask[idx]))}",
            f"{int(bool(result.reference_atom_mask[idx]))}",
            f"{int(bool(result.fit_atom_mask[idx]))}",
        ]
        buffer.write(",".join(row) + "\n")
    return buffer.getvalue()


def _atomic_intensity_csv(coords: np.ndarray, result: AtomicColumnIntensityResult) -> str:
    buffer = io.StringIO()
    buffer.write("intensity_unit,a.u.\n")
    buffer.write(
        "atom_index,y,x,peak_intensity,raw_sum,raw_mean,background_level,"
        "background_mean,background_median,background_std,"
        "background_subtracted_sum,normalized_intensity,snr,atom_area_px,"
        "background_area_px,nearest_neighbor_distance_px,valid_intensity,"
        "edge_flag,overlap_flag,saturation_flag\n"
    )
    for idx in range(len(coords)):
        y, x = coords[idx]
        row = [
            str(idx),
            f"{float(y):.6f}",
            f"{float(x):.6f}",
            _csv_float(result.peak_intensity[idx]),
            _csv_float(result.raw_sum[idx]),
            _csv_float(result.raw_mean[idx]),
            _csv_float(result.background_level[idx]),
            _csv_float(result.background_mean[idx]),
            _csv_float(result.background_median[idx]),
            _csv_float(result.background_std[idx]),
            _csv_float(result.background_subtracted_sum[idx]),
            _csv_float(result.normalized_intensity[idx]),
            _csv_float(result.snr[idx]),
            str(int(result.atom_area_px[idx])),
            str(int(result.background_area_px[idx])),
            _csv_float(result.nearest_neighbor_distance_px[idx]),
            str(int(bool(result.valid_atom_mask[idx]))),
            str(int(bool(result.edge_flag[idx]))),
            str(int(bool(result.overlap_flag[idx]))),
            str(int(bool(result.saturation_flag[idx]))),
        ]
        buffer.write(",".join(row) + "\n")
    return buffer.getvalue()


def _csv_float(value: float | None) -> str:
    if value is None:
        return ""
    value = float(value)
    if not np.isfinite(value):
        return ""
    return f"{value:.6f}"


def _symmetric_limit(arr: np.ndarray, percentile: float = 98.0, fallback: float = 1.0) -> float:
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    lim = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(lim) or lim <= 1e-12:
        lim = fallback
    return lim


def _atomic_intensity_metric(
    result: AtomicColumnIntensityResult,
    metric: str,
) -> tuple[np.ndarray, str]:
    mapping = {
        "background_subtracted_sum": (
            result.background_subtracted_sum,
            "background-subtracted intensity [a.u.]",
        ),
        "normalized_intensity": (
            result.normalized_intensity,
            "normalized intensity [relative]",
        ),
        "peak_intensity": (result.peak_intensity, "peak intensity [a.u.]"),
        "snr": (result.snr, "local SNR"),
    }
    if metric not in mapping:
        raise ValueError(f"Unsupported intensity metric: {metric}")
    values, label = mapping[metric]
    return np.asarray(values, dtype=np.float64), label


def _auto_color_limits(values: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    vmin = float(np.percentile(finite, 2))
    vmax = float(np.percentile(finite, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) <= 1e-12:
        return fallback
    return vmin, vmax


def _render_atomic_intensity_overlay(
    image_norm: np.ndarray,
    coords: np.ndarray,
    result: AtomicColumnIntensityResult,
    *,
    metric: str,
    auto_color_range: bool,
    vmin: float,
    vmax: float,
    marker_size: float,
) -> plt.Figure:
    values, label = _atomic_intensity_metric(result, metric)
    valid = np.asarray(result.valid_atom_mask, dtype=bool) & np.isfinite(values)
    if auto_color_range:
        vmin, vmax = _auto_color_limits(values[valid], (0.0, 1.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=160)
    ax.imshow(image_norm, cmap="gray")
    if np.any(valid):
        scatter = ax.scatter(
            coords[valid, 1],
            coords[valid, 0],
            c=values[valid],
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
            s=float(marker_size),
            edgecolors="white",
            linewidths=0.35,
            alpha=0.92,
            label="valid intensity",
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.84)
        cbar.set_label(label, rotation=90)
    invalid = ~valid
    if np.any(invalid):
        ax.scatter(
            coords[invalid, 1],
            coords[invalid, 0],
            s=max(float(marker_size) * 0.85, 8.0),
            c="none",
            edgecolors="#ff3d3d",
            marker="x",
            linewidths=0.8,
            label="invalid intensity",
        )
    ax.set_title(f"Atomic-column intensity map: {metric}")
    ax.axis("off")
    if np.any(invalid):
        ax.legend(loc="upper right", fontsize=8, framealpha=0.82)
    fig.tight_layout()
    return fig


def _render_atomic_intensity_histogram(
    result: AtomicColumnIntensityResult,
    *,
    metric: str,
) -> plt.Figure:
    values, label = _atomic_intensity_metric(result, metric)
    valid = np.asarray(result.valid_atom_mask, dtype=bool) & np.isfinite(values)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=160)
    if np.any(valid):
        ax.hist(values[valid], bins=36, color="#147c8c", alpha=0.82, edgecolor="white")
        ax.axvline(float(np.median(values[valid])), color="#ff5a3d", lw=1.6, label="median")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No valid intensity values", ha="center", va="center")
    ax.set_xlabel(label)
    ax.set_ylabel("atom count")
    ax.set_title("Atomic-column intensity histogram")
    fig.tight_layout()
    return fig


def _render_atomic_intensity_group_histogram(
    result: AtomicColumnIntensityResult,
    group_result: AtomicIntensityGroupResult,
    *,
    metric: str,
) -> plt.Figure:
    values, label = _atomic_intensity_metric(result, metric)
    labels = np.asarray(group_result.labels, dtype=np.int32)
    valid = np.asarray(result.valid_atom_mask, dtype=bool) & np.isfinite(values) & (labels >= 0)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    if not np.any(valid):
        ax.text(0.5, 0.5, "No valid grouped intensity values", ha="center", va="center")
    else:
        summary_by_id = {int(row["group_id"]): row for row in group_result.summaries}
        group_ids = sorted(int(v) for v in np.unique(labels[valid]))
        for group_id in group_ids:
            group_values = values[valid & (labels == group_id)]
            group_name = str(summary_by_id.get(group_id, {}).get("group_name", group_id))
            ax.hist(
                group_values,
                bins=28,
                alpha=0.55,
                label=f"{group_name} (n={len(group_values)})",
            )
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel("atom count")
    ax.set_title("Grouped atomic-column intensity histogram")
    fig.tight_layout()
    return fig


def _atomic_intensity_group_csv(group_result: AtomicIntensityGroupResult) -> str:
    buffer = io.StringIO()
    buffer.write(f"metric,{group_result.metric}\n")
    buffer.write(
        "group_id,group_name,count,mean,std,median,min,max,method,confidence\n"
    )
    method = str(group_result.provenance.get("method", ""))
    for row in group_result.summaries:
        buffer.write(
            ",".join(
                [
                    str(int(row["group_id"])),
                    str(row["group_name"]),
                    str(int(row["count"])),
                    _csv_float(row["mean"]),
                    _csv_float(row["std"]),
                    _csv_float(row["median"]),
                    _csv_float(row["min"]),
                    _csv_float(row["max"]),
                    method,
                    _csv_float(group_result.confidence),
                ]
            )
            + "\n"
        )
    return buffer.getvalue()


def _sublattice_palette(labels: np.ndarray) -> dict[str, str]:
    unique_labels = sorted(str(v) for v in np.unique(labels))
    colors = [
        "#ff5252",
        "#26c6da",
        "#ffd54f",
        "#66bb6a",
        "#ab47bc",
        "#ff8a65",
        "#5c6bc0",
        "#8d6e63",
        "#ec407a",
        "#78909c",
    ]
    palette: dict[str, str] = {}
    for idx, label in enumerate(unique_labels):
        palette[label] = "#9e9e9e" if label == "unknown" else colors[idx % len(colors)]
    return palette


def _sublattice_csv(
    coords: np.ndarray,
    result: SublatticeClassificationResult,
) -> str:
    buffer = io.StringIO()
    buffer.write("coordinate_order,y,x\n")
    buffer.write(
        "atom_index,y,x,label,label_id,lattice_i,lattice_j,frac_u,frac_v,"
        "cell_u,cell_v,intensity,valid_sublattice,manual_override\n"
    )
    for idx in range(len(coords)):
        y, x = coords[idx]
        frac_u, frac_v = result.fractional_uv[idx]
        cell_u, cell_v = result.unit_cell_uv[idx]
        lattice_i, lattice_j = result.lattice_indices_ij[idx]
        row = [
            str(idx),
            f"{float(y):.6f}",
            f"{float(x):.6f}",
            str(result.labels[idx]),
            str(int(result.label_ids[idx])),
            str(int(lattice_i)),
            str(int(lattice_j)),
            f"{float(frac_u):.6f}",
            f"{float(frac_v):.6f}",
            f"{float(cell_u):.6f}",
            f"{float(cell_v):.6f}",
            _csv_float(result.intensities[idx]),
            str(int(bool(result.valid_atom_mask[idx]))),
            str(int(bool(result.manual_override_mask[idx]))),
        ]
        buffer.write(",".join(row) + "\n")
    return buffer.getvalue()


def _sublattice_summary_csv(result: SublatticeClassificationResult) -> str:
    buffer = io.StringIO()
    buffer.write(
        "label,count,valid_count,manual_count,mean_u,mean_v,"
        "mean_intensity,std_intensity\n"
    )
    for row in result.label_summaries:
        buffer.write(
            ",".join(
                [
                    str(row["label"]),
                    str(int(row["count"])),
                    str(int(row["valid_count"])),
                    str(int(row["manual_count"])),
                    _csv_float(row["mean_u"]),
                    _csv_float(row["mean_v"]),
                    _csv_float(row["mean_intensity"]),
                    _csv_float(row["std_intensity"]),
                ]
            )
            + "\n"
        )
    return buffer.getvalue()


def _render_sublattice_overlay(
    image_norm: np.ndarray,
    coords: np.ndarray,
    result: SublatticeClassificationResult,
    *,
    marker_size: float,
    show_label_text: bool,
    show_atom_index: bool,
) -> plt.Figure:
    labels = np.asarray(result.labels, dtype=object)
    palette = _sublattice_palette(labels)
    manual = np.asarray(result.manual_override_mask, dtype=bool)
    valid = np.asarray(result.valid_atom_mask, dtype=bool)

    fig, ax = plt.subplots(figsize=(7.2, 6.8), dpi=170)
    ax.imshow(image_norm, cmap="gray")
    for label in sorted(palette):
        mask = labels == label
        if not np.any(mask):
            continue
        alpha = 0.52 if label == "unknown" else 0.9
        marker = "x" if label == "unknown" else "o"
        ax.scatter(
            coords[mask, 1],
            coords[mask, 0],
            s=float(marker_size),
            c=palette[label],
            marker=marker,
            edgecolors="white",
            linewidths=0.35,
            alpha=alpha,
            label=f"{label} (n={int(np.sum(mask))})",
        )
    if np.any(manual):
        ax.scatter(
            coords[manual, 1],
            coords[manual, 0],
            s=max(float(marker_size) * 1.7, 18.0),
            facecolors="none",
            edgecolors="#fff176",
            linewidths=1.05,
            label="manual override",
        )
    invalid = ~valid
    if np.any(invalid) and "unknown" not in palette:
        ax.scatter(
            coords[invalid, 1],
            coords[invalid, 0],
            s=max(float(marker_size) * 0.72, 10.0),
            c="#9e9e9e",
            marker="x",
            linewidths=0.75,
            label="invalid/unmatched",
        )

    if show_label_text or show_atom_index:
        max_text = 800
        for idx in range(min(len(coords), max_text)):
            pieces: list[str] = []
            if show_atom_index:
                pieces.append(str(idx))
            if show_label_text:
                pieces.append(str(labels[idx]))
            ax.text(
                float(coords[idx, 1]) + 2.0,
                float(coords[idx, 0]) - 2.0,
                "/".join(pieces),
                color="#ffffff",
                fontsize=5.7,
                alpha=0.86,
            )
    ax.set_title("Sublattice classification overlay")
    ax.axis("off")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.82)
    fig.tight_layout()
    return fig


def _render_sublattice_unit_cell(result: SublatticeClassificationResult) -> plt.Figure:
    labels = np.asarray(result.labels, dtype=object)
    unit_uv = np.asarray(result.unit_cell_uv, dtype=np.float64)
    valid = np.asarray(result.valid_atom_mask, dtype=bool)
    palette = _sublattice_palette(labels)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=170)
    for label in sorted(palette):
        mask = valid & (labels == label)
        if not np.any(mask):
            continue
        ax.scatter(
            unit_uv[mask, 0],
            unit_uv[mask, 1],
            s=28,
            c=palette[label],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.86,
            label=label,
        )
    for row in result.projected_structure.get("basis_sites", []):
        ax.scatter(
            float(row["u"]),
            float(row["v"]),
            s=130,
            marker="*",
            c=palette.get(str(row["label"]), "#ffffff"),
            edgecolors="#263238",
            linewidths=0.8,
        )
        ax.text(
            float(row["u"]) + 0.025,
            float(row["v"]) + 0.025,
            str(row["label"]),
            fontsize=10,
            weight="bold",
        )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("u along custom a")
    ax.set_ylabel("v along custom b")
    ax.set_title("Projected unit-cell label positions")
    ax.set_aspect("equal")
    ax.grid(alpha=0.22)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.82)
    fig.tight_layout()
    return fig


def _render_sublattice_picker_chart(
    image_norm: np.ndarray,
    coords: np.ndarray,
    result: SublatticeClassificationResult,
    *,
    chart_max_side: int = 700,
):
    image_only, sx, sy = _to_display_image(
        image_norm,
        coords=np.empty((0, 2), dtype=np.float64),
        roi_mask=None,
        basis_markers=None,
        max_side=900,
    )
    disp_w, disp_h = image_only.size
    chart_scale = min(1.0, float(chart_max_side) / max(disp_w, disp_h))
    chart_w = max(320, int(round(disp_w * chart_scale)))
    chart_h = max(320, int(round(disp_h * chart_scale)))
    labels = np.asarray(result.labels, dtype=object)
    palette = _sublattice_palette(labels)
    manual = np.asarray(result.manual_override_mask, dtype=bool)
    point_df = pd.DataFrame(
        {
            "atom_index": np.arange(len(coords), dtype=int),
            "x": np.asarray(coords[:, 1] / sx, dtype=np.float64),
            "y": np.asarray(coords[:, 0] / sy, dtype=np.float64),
            "label": [str(v) for v in labels],
            "manual": manual.astype(np.int32),
        }
    )
    point_df["color"] = point_df["label"].map(palette).fillna("#9e9e9e")
    point_df["size"] = np.where(manual, 120, 70)
    picker = alt.selection_point(
        name="sublattice_pick",
        fields=["atom_index"],
        nearest=True,
        on="click",
        clear=False,
    )
    return (
        alt.Chart(point_df)
        .mark_circle(stroke="#263238", strokeWidth=0.45, opacity=0.9)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, disp_w])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[disp_h, 0])),
            color=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("size:Q", legend=None),
            tooltip=[
                alt.Tooltip("atom_index:Q", title="atom index"),
                alt.Tooltip("label:N", title="label"),
                alt.Tooltip("manual:Q", title="manual override"),
            ],
            fillOpacity=alt.condition(picker, alt.value(1.0), alt.value(0.84)),
        )
        .add_params(picker)
        .properties(width=chart_w, height=chart_h)
        .configure_view(stroke=None)
    )


def _parse_atom_index_text(text: str, atom_count: int) -> list[int]:
    normalized = (
        str(text)
        .replace("\uff0c", ",")
        .replace(";", ",")
        .replace("\n", ",")
        .replace(" ", ",")
    )
    out: list[int] = []
    seen: set[int] = set()
    for raw_token in normalized.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                start, end = end, start
            candidates = range(start, end + 1)
        else:
            candidates = [int(token)]
        for idx in candidates:
            if idx < 0 or idx >= atom_count:
                raise ValueError(f"Atom index {idx} is outside [0, {atom_count - 1}].")
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    text = str(hex_color).lstrip("#")
    if len(text) != 6:
        return (158, 158, 158)
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))


def _render_sublattice_index_reference_image(
    image_norm: np.ndarray,
    coords: np.ndarray,
    result: SublatticeClassificationResult,
    *,
    max_side: int = 1400,
) -> Image.Image:
    img, sx, sy = _to_display_image(
        image_norm,
        coords=np.empty((0, 2), dtype=np.float64),
        roi_mask=None,
        basis_markers=None,
        max_side=max_side,
    )
    draw = ImageDraw.Draw(img)
    labels = np.asarray(result.labels, dtype=object)
    palette = _sublattice_palette(labels)
    manual = np.asarray(result.manual_override_mask, dtype=bool)
    radius = 5 if max(img.size) >= 1000 else 4
    for idx, (y, x) in enumerate(np.asarray(coords, dtype=np.float64)):
        cx = float(x / sx)
        cy = float(y / sy)
        color = _hex_to_rgb(palette.get(str(labels[idx]), "#9e9e9e"))
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=color,
            outline=(255, 255, 255),
            width=1,
        )
        if bool(manual[idx]):
            rr = radius + 3
            draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=(255, 241, 118), width=2)
        text = str(idx)
        tx = cx + radius + 2
        ty = cy - radius - 2
        draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0))
        draw.text((tx, ty), text, fill=(255, 255, 255))
    return img


def _render_atom_index_reference_image(
    image_norm: np.ndarray,
    coords: np.ndarray,
    *,
    highlighted_indices: set[int] | None = None,
    max_side: int = 1200,
) -> Image.Image:
    img, sx, sy = _to_display_image(
        image_norm,
        coords=np.empty((0, 2), dtype=np.float64),
        roi_mask=None,
        basis_markers=None,
        max_side=max_side,
    )
    draw = ImageDraw.Draw(img)
    highlighted = highlighted_indices or set()
    radius = 5 if max(img.size) >= 900 else 4
    for idx, (y, x) in enumerate(np.asarray(coords, dtype=np.float64)):
        cx = float(x / sx)
        cy = float(y / sy)
        is_highlighted = idx in highlighted
        fill = (255, 213, 79) if is_highlighted else (255, 82, 82)
        outline = (0, 255, 255) if is_highlighted else (255, 255, 255)
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=fill,
            outline=outline,
            width=2 if is_highlighted else 1,
        )
        text = str(idx)
        tx = cx + radius + 2
        ty = cy - radius - 2
        draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0))
        draw.text((tx, ty), text, fill=(255, 255, 255))
    return img


def _plane_spacing_csv(
    result: AtomicPlaneSpacingResult,
    *,
    spacing_scale: float,
    spacing_unit: str,
) -> str:
    buffer = io.StringIO()
    buffer.write(f"spacing_unit,{spacing_unit}\n")
    buffer.write(
        "segment_index,direction,source_atom,target_atom,source_i,source_j,"
        "target_i,target_j,start_y,start_x,end_y,end_x,length\n"
    )
    for idx in range(len(result.length_px)):
        start_y, start_x = result.start_yx[idx]
        end_y, end_x = result.end_yx[idx]
        source_i, source_j = result.source_lattice_ij[idx]
        target_i, target_j = result.target_lattice_ij[idx]
        row = [
            str(idx),
            str(result.direction[idx]),
            str(int(result.source_atom_index[idx])),
            str(int(result.target_atom_index[idx])),
            str(int(source_i)),
            str(int(source_j)),
            str(int(target_i)),
            str(int(target_j)),
            f"{float(start_y):.6f}",
            f"{float(start_x):.6f}",
            f"{float(end_y):.6f}",
            f"{float(end_x):.6f}",
            f"{float(result.length_px[idx]) * spacing_scale:.6f}",
        ]
        buffer.write(",".join(row) + "\n")
    return buffer.getvalue()


def _plane_spacing_direction_mask(
    result: AtomicPlaneSpacingResult,
    direction_mode: str,
) -> np.ndarray:
    directions = np.asarray(result.direction, dtype=object)
    if direction_mode in {"a + b", "selected direction"}:
        return np.ones(len(directions), dtype=bool)
    if direction_mode == "a only":
        return directions == "a"
    if direction_mode == "b only":
        return directions == "b"
    raise ValueError(f"Unsupported spacing direction mode: {direction_mode}")


def _render_plane_spacing_segments(
    image_norm: np.ndarray,
    coords: np.ndarray,
    result: AtomicPlaneSpacingResult,
    *,
    direction_mode: str,
    spacing_scale: float,
    spacing_unit: str,
    show_image: bool,
    show_atoms: bool,
    auto_color_range: bool,
    vmin: float,
    vmax: float,
    line_width: float,
) -> plt.Figure:
    mask = _plane_spacing_direction_mask(result, direction_mode)
    lengths = np.asarray(result.length_px, dtype=np.float64) * spacing_scale
    finite_mask = mask & np.isfinite(lengths)
    if auto_color_range:
        vmin, vmax = _auto_color_limits(lengths[finite_mask], (0.0, 1.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(7.3, 6.8), dpi=160)
    if show_image:
        ax.imshow(image_norm, cmap="gray")
    else:
        ax.set_facecolor("white")
        ax.set_xlim(0, image_norm.shape[1])
        ax.set_ylim(image_norm.shape[0], 0)
    if show_atoms and len(coords) > 0:
        ax.scatter(
            coords[:, 1],
            coords[:, 0],
            s=8,
            c="#e8e8e8" if show_image else "#333333",
            edgecolors="none",
            alpha=0.72,
            label="atoms",
        )
    if np.any(finite_mask):
        start = result.start_yx[finite_mask]
        end = result.end_yx[finite_mask]
        segments = [
            [(float(sx), float(sy)), (float(ex), float(ey))]
            for (sy, sx), (ey, ex) in zip(start, end, strict=False)
        ]
        line_collection = LineCollection(
            segments,
            cmap="turbo",
            array=lengths[finite_mask],
            linewidths=float(line_width),
            alpha=0.96,
        )
        line_collection.set_clim(vmin, vmax)
        ax.add_collection(line_collection)
        cbar = fig.colorbar(line_collection, ax=ax, shrink=0.84)
        cbar.set_label(f"spacing [{spacing_unit}]", rotation=90)
    ax.set_title(f"Local atom-to-atom spacing segments ({direction_mode})")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _render_plane_spacing_histogram(
    result: AtomicPlaneSpacingResult,
    *,
    direction_mode: str,
    spacing_scale: float,
    spacing_unit: str,
) -> plt.Figure:
    mask = _plane_spacing_direction_mask(result, direction_mode)
    lengths = np.asarray(result.length_px, dtype=np.float64) * spacing_scale
    valid = mask & np.isfinite(lengths)
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=160)
    if np.any(valid):
        directions = np.asarray(result.direction, dtype=object)
        unique_directions = [str(v) for v in np.unique(directions[valid])]
        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for direction, color in zip(unique_directions, color_cycle, strict=False):
            direction_mask = valid & (np.asarray(result.direction, dtype=object) == direction)
            if np.any(direction_mask):
                ax.hist(
                    lengths[direction_mask],
                    bins=32,
                    alpha=0.58,
                    color=color,
                    edgecolor="white",
                    label=f"{direction} direction",
                )
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No valid spacing segments", ha="center", va="center")
    ax.set_xlabel(f"spacing [{spacing_unit}]")
    ax.set_ylabel("segment count")
    ax.set_title("Local spacing histogram")
    fig.tight_layout()
    return fig


def _render_atom_strain_maps(
    fields: AtomStrainFieldResult,
    *,
    displacement_scale: float,
    displacement_unit: str,
    strain_scale: float,
    strain_unit: str,
    auto_color_range: bool,
    displacement_abs_limit: float,
    strain_abs_limit: float,
    rotation_abs_limit: float,
) -> plt.Figure:
    ux = np.asarray(fields.displacement_x_map, dtype=np.float64) * displacement_scale
    uy = np.asarray(fields.displacement_y_map, dtype=np.float64) * displacement_scale
    exx = np.asarray(fields.strain_xx_map, dtype=np.float64) * strain_scale
    eyy = np.asarray(fields.strain_yy_map, dtype=np.float64) * strain_scale
    exy = np.asarray(fields.strain_xy_map, dtype=np.float64) * strain_scale
    rot = np.asarray(fields.rotation_xy_map, dtype=np.float64)

    if auto_color_range:
        disp_lim = max(_symmetric_limit(ux), _symmetric_limit(uy), 1e-6)
        strain_lim = max(
            _symmetric_limit(exx),
            _symmetric_limit(eyy),
            _symmetric_limit(exy),
            1e-6,
        )
        rot_lim = max(_symmetric_limit(rot), 1e-6)
    else:
        disp_lim = max(float(displacement_abs_limit), 1e-6)
        strain_lim = max(float(strain_abs_limit), 1e-6)
        rot_lim = max(float(rotation_abs_limit), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.9), dpi=150)
    panels = [
        (ux, f"ux displacement [{displacement_unit}]", displacement_unit, disp_lim),
        (uy, f"uy displacement [{displacement_unit}]", displacement_unit, disp_lim),
        (exx, f"exx strain [{strain_unit}]", strain_unit, strain_lim),
        (eyy, f"eyy strain [{strain_unit}]", strain_unit, strain_lim),
        (exy, f"exy strain [{strain_unit}]", strain_unit, strain_lim),
        (rot, "rotation [rad]", "rad", rot_lim),
    ]

    for ax, (arr, title, cbar_label, lim) in zip(axes.flat, panels, strict=False):
        im = ax.imshow(arr, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(cbar_label, rotation=90)

    fig.tight_layout()
    return fig


def _render_custom_basis_strain_maps(
    eaa_map: np.ndarray,
    ebb_map: np.ndarray,
    eab_map: np.ndarray,
    *,
    strain_unit: str,
    auto_color_range: bool,
    strain_abs_limit: float,
) -> plt.Figure:
    eaa = np.asarray(eaa_map, dtype=np.float64)
    ebb = np.asarray(ebb_map, dtype=np.float64)
    eab = np.asarray(eab_map, dtype=np.float64)
    if auto_color_range:
        lim = max(
            _symmetric_limit(eaa),
            _symmetric_limit(ebb),
            _symmetric_limit(eab),
            1e-6,
        )
    else:
        lim = max(float(strain_abs_limit), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.2), dpi=150)
    panels = [
        (eaa, f"eaa [{strain_unit}]"),
        (ebb, f"ebb [{strain_unit}]"),
        (eab, f"eab [{strain_unit}]"),
    ]
    for ax, (arr, title) in zip(axes.flat, panels, strict=False):
        im = ax.imshow(arr, cmap="coolwarm", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(strain_unit, rotation=90)
    fig.tight_layout()
    return fig


def _render_atom_strain_overlay(
    image_norm: np.ndarray,
    coords: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    colorbar_label: str,
    vmin: float,
    vmax: float,
    cmap: str = "jet",
    marker_size: float = 36.0,
    invalid_mask: np.ndarray | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180)
    ax.imshow(image_norm, cmap="gray")
    valid = np.isfinite(values)
    if invalid_mask is not None:
        valid &= ~invalid_mask
    if np.any(valid):
        sc = ax.scatter(
            coords[valid, 1],
            coords[valid, 0],
            c=values[valid],
            s=marker_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            edgecolors="white",
            linewidths=0.25,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(colorbar_label, rotation=90)
    if invalid_mask is not None and np.any(invalid_mask):
        ax.scatter(
            coords[invalid_mask, 1],
            coords[invalid_mask, 0],
            s=max(marker_size * 0.55, 10.0),
            c="#9e9e9e",
            marker="x",
            linewidths=0.7,
        )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _render_atomic_displacement_overlay(
    image_norm: np.ndarray,
    result: AtomicDisplacementResult,
    *,
    displacement_scale: float,
    displacement_unit: str,
    marker_size: float = 24.0,
    max_arrows: int = 1500,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180)
    ax.imshow(image_norm, cmap="gray")

    valid = np.asarray(result.valid_atom_mask, dtype=bool)
    obs = np.asarray(result.coordinates_observed_yx, dtype=np.float64)
    ideal = np.asarray(result.coordinates_ideal_yx, dtype=np.float64)
    disp = np.asarray(result.displacement_yx, dtype=np.float64)
    mag = np.asarray(result.displacement_magnitude, dtype=np.float64) * displacement_scale

    vmax = max(_symmetric_limit(mag[valid], percentile=98.0, fallback=1.0), 1e-6)
    if np.any(valid):
        sc = ax.scatter(
            obs[valid, 1],
            obs[valid, 0],
            c=mag[valid],
            s=marker_size,
            cmap="jet",
            vmin=0.0,
            vmax=vmax,
            edgecolors="white",
            linewidths=0.25,
            label="observed atoms",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(f"|u| [{displacement_unit}]", rotation=90)

        if int(np.sum(valid)) > max_arrows:
            stride = max(int(np.ceil(np.sum(valid) / max_arrows)), 1)
            arrow_mask = np.zeros(np.sum(valid), dtype=bool)
            arrow_mask[::stride] = True
            ideal_arrow = ideal[valid][arrow_mask]
            disp_arrow = disp[valid][arrow_mask]
        else:
            ideal_arrow = ideal[valid]
            disp_arrow = disp[valid]

        ax.scatter(
            ideal[valid, 1],
            ideal[valid, 0],
            s=max(marker_size * 0.7, 12.0),
            c="#00ffff",
            marker="x",
            linewidths=0.8,
            label="ideal lattice sites",
        )
        ax.quiver(
            ideal_arrow[:, 1],
            ideal_arrow[:, 0],
            disp_arrow[:, 1],
            disp_arrow[:, 0],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="#fff176",
            width=0.0022,
        )

    if np.any(result.fit_atom_mask):
        ax.scatter(
            obs[result.fit_atom_mask, 1],
            obs[result.fit_atom_mask, 0],
            s=max(marker_size * 1.15, 16.0),
            facecolors="none",
            edgecolors="#00ff66",
            linewidths=0.75,
            label="fit/reference atoms",
        )
    invalid = ~valid
    if np.any(invalid):
        ax.scatter(
            obs[invalid, 1],
            obs[invalid, 0],
            s=max(marker_size * 0.6, 10.0),
            c="#9e9e9e",
            marker="x",
            linewidths=0.7,
            label="invalid assignment",
        )
    ax.set_title("Atomic displacement relative to fitted ideal lattice")
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.82)
    fig.tight_layout()
    return fig


def _render_atomic_displacement_vector_map(
    image_norm: np.ndarray,
    result: AtomicDisplacementResult,
    *,
    displacement_scale: float,
    displacement_unit: str,
    show_image: bool,
    marker_size: float = 26.0,
    arrow_display_scale: float = 1.0,
    max_arrows: int = 1200,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180)
    h, w = image_norm.shape
    if show_image:
        ax.imshow(image_norm, cmap="gray")
    else:
        ax.set_facecolor("white")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")

    valid = np.asarray(result.valid_atom_mask, dtype=bool)
    obs = np.asarray(result.coordinates_observed_yx, dtype=np.float64)
    ideal = np.asarray(result.coordinates_ideal_yx, dtype=np.float64)
    disp = np.asarray(result.displacement_yx, dtype=np.float64)
    mag = np.asarray(result.displacement_magnitude, dtype=np.float64) * displacement_scale

    if np.any(valid):
        vmax = max(_symmetric_limit(mag[valid], percentile=98.0, fallback=1.0), 1e-6)
        sc = ax.scatter(
            obs[valid, 1],
            obs[valid, 0],
            c=mag[valid],
            s=marker_size,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            edgecolors="white" if show_image else "#212121",
            linewidths=0.25,
            zorder=3,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label(f"|u| [{displacement_unit}]", rotation=90)

        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) > max_arrows:
            stride = max(int(np.ceil(len(valid_idx) / max_arrows)), 1)
            valid_idx = valid_idx[::stride]
        ax.quiver(
            ideal[valid_idx, 1],
            ideal[valid_idx, 0],
            disp[valid_idx, 1] * float(arrow_display_scale),
            disp[valid_idx, 0] * float(arrow_display_scale),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color="#ffca28" if show_image else "#263238",
            width=0.0024,
            zorder=4,
        )

    ax.set_title("Atomic displacement vector map")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _render_atomic_displacement_component_maps(
    image_norm: np.ndarray,
    result: AtomicDisplacementResult,
    *,
    displacement_scale: float,
    displacement_unit: str,
    marker_size: float = 24.0,
) -> plt.Figure:
    h, w = image_norm.shape
    valid = np.asarray(result.valid_atom_mask, dtype=bool)
    obs = np.asarray(result.coordinates_observed_yx, dtype=np.float64)
    uy = np.asarray(result.displacement_yx[:, 0], dtype=np.float64) * displacement_scale
    ux = np.asarray(result.displacement_yx[:, 1], dtype=np.float64) * displacement_scale
    mag = np.asarray(result.displacement_magnitude, dtype=np.float64) * displacement_scale

    disp_lim = max(_symmetric_limit(ux[valid]), _symmetric_limit(uy[valid]), 1e-6)
    mag_lim = max(_symmetric_limit(mag[valid], percentile=98.0, fallback=1.0), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), dpi=170)
    panels = [
        (ux, f"ux [{displacement_unit}]", "coolwarm", -disp_lim, disp_lim),
        (uy, f"uy [{displacement_unit}]", "coolwarm", -disp_lim, disp_lim),
        (mag, f"|u| [{displacement_unit}]", "viridis", 0.0, mag_lim),
    ]
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, panels, strict=False):
        ax.set_facecolor("white")
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        if np.any(valid):
            sc = ax.scatter(
                obs[valid, 1],
                obs[valid, 0],
                c=values[valid],
                s=marker_size,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                edgecolors="#212121",
                linewidths=0.2,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label(title, rotation=90)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    return fig


def _render_atomic_displacement_invalid_map(
    image_norm: np.ndarray,
    result: AtomicDisplacementResult,
    *,
    marker_size: float = 28.0,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=180)
    ax.imshow(image_norm, cmap="gray")

    obs = np.asarray(result.coordinates_observed_yx, dtype=np.float64)
    valid = np.asarray(result.valid_atom_mask, dtype=bool)
    residual = np.asarray(result.assignment_residual, dtype=np.float64)
    invalid = ~valid

    if np.any(valid):
        ax.scatter(
            obs[valid, 1],
            obs[valid, 0],
            s=max(marker_size * 0.55, 10.0),
            c="#64b5f6",
            alpha=0.65,
            edgecolors="white",
            linewidths=0.2,
            label="valid assignment",
        )
    if np.any(invalid):
        vmax = max(float(np.nanmax(residual[invalid])), 1e-6)
        sc = ax.scatter(
            obs[invalid, 1],
            obs[invalid, 0],
            c=residual[invalid],
            s=marker_size,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            marker="x",
            linewidths=0.95,
            label="invalid assignment",
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("assignment residual", rotation=90)
    if np.any(result.fit_atom_mask):
        ax.scatter(
            obs[result.fit_atom_mask, 1],
            obs[result.fit_atom_mask, 0],
            s=max(marker_size * 1.1, 16.0),
            facecolors="none",
            edgecolors="#00ff66",
            linewidths=0.75,
            label="fit/reference atoms",
        )
    ax.set_title("Invalid assignment QA map")
    ax.axis("off")
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.82)
    fig.tight_layout()
    return fig


def _render_custom_basis_context(
    image_norm: np.ndarray,
    coords: np.ndarray,
    custom_basis: dict[str, Any],
) -> plt.Figure:
    basis_points = np.asarray(
        [
            custom_basis["base_yx"],
            custom_basis["base_yx"] + custom_basis["a_vec_yx"],
            custom_basis["base_yx"] + custom_basis["b_vec_yx"],
        ],
        dtype=np.float64,
    )
    pad = max(
        24.0,
        0.8
        * max(
            float(np.linalg.norm(custom_basis["a_vec_yx"])),
            float(np.linalg.norm(custom_basis["b_vec_yx"])),
        ),
    )
    y0 = max(int(np.floor(np.min(basis_points[:, 0]) - pad)), 0)
    y1 = min(int(np.ceil(np.max(basis_points[:, 0]) + pad)), image_norm.shape[0])
    x0 = max(int(np.floor(np.min(basis_points[:, 1]) - pad)), 0)
    x1 = min(int(np.ceil(np.max(basis_points[:, 1]) + pad)), image_norm.shape[1])

    fig, ax = plt.subplots(figsize=(4.3, 4.3), dpi=180)
    ax.imshow(image_norm[y0:y1, x0:x1], cmap="gray", origin="upper")

    in_crop = (
        (coords[:, 0] >= y0)
        & (coords[:, 0] <= y1)
        & (coords[:, 1] >= x0)
        & (coords[:, 1] <= x1)
    )
    crop_coords = coords[in_crop]
    if len(crop_coords) > 0:
        ax.scatter(
            crop_coords[:, 1] - x0,
            crop_coords[:, 0] - y0,
            s=18,
            facecolors="none",
            edgecolors="#ffffff",
            linewidths=0.45,
            alpha=0.85,
            label="atom positions",
        )

    base = np.asarray(custom_basis["base_yx"], dtype=np.float64)
    a_tip = base + np.asarray(custom_basis["a_vec_yx"], dtype=np.float64)
    b_tip = base + np.asarray(custom_basis["b_vec_yx"], dtype=np.float64)
    base_xy = (float(base[1] - x0), float(base[0] - y0))
    a_xy = (float(a_tip[1] - x0), float(a_tip[0] - y0))
    b_xy = (float(b_tip[1] - x0), float(b_tip[0] - y0))

    ax.scatter([base_xy[0]], [base_xy[1]], s=48, c="#00ffff", label="base")
    ax.scatter([a_xy[0]], [a_xy[1]], s=44, c="#ffd54f", label="a atom")
    ax.scatter([b_xy[0]], [b_xy[1]], s=44, c="#ff66cc", label="b atom")
    ax.annotate(
        "",
        xy=a_xy,
        xytext=base_xy,
        arrowprops=dict(arrowstyle="->", color="#ffd54f", lw=2.0),
    )
    ax.annotate(
        "",
        xy=b_xy,
        xytext=base_xy,
        arrowprops=dict(arrowstyle="->", color="#ff66cc", lw=2.0),
    )
    ax.text(a_xy[0] + 3, a_xy[1] - 3, "a", color="#ffd54f", fontsize=10, weight="bold")
    ax.text(b_xy[0] + 3, b_xy[1] - 3, "b", color="#ff66cc", fontsize=10, weight="bold")
    ax.text(
        base_xy[0] + 3,
        base_xy[1] - 3,
        "base",
        color="#00ffff",
        fontsize=9,
        weight="bold",
    )
    ax.set_title("Custom a-b basis on atom positions")
    ax.axis("off")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.82)
    fig.tight_layout()
    return fig


def _figure_png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=180)
    buffer.seek(0)
    return buffer.getvalue()


def _pil_png_bytes(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def _invalidate_sublattice_state(*, clear_manual_labels: bool = True) -> None:
    st.session_state.sublattice_result = None
    st.session_state.sublattice_meta = None
    st.session_state.sublattice_selected_atom = None
    st.session_state.sublattice_picker_nonce = int(
        st.session_state.get("sublattice_picker_nonce", 0)
    ) + 1
    if clear_manual_labels:
        st.session_state.sublattice_manual_labels = {}


def _invalidate_atomic_intensity_state() -> None:
    st.session_state.atomic_intensity_result = None
    st.session_state.atomic_intensity_meta = None
    _invalidate_sublattice_state(clear_manual_labels=False)


def _invalidate_plane_spacing_state() -> None:
    st.session_state.plane_spacing_result = None
    st.session_state.plane_spacing_meta = None


def _invalidate_atom_strain_state() -> None:
    st.session_state.atom_strain_fields = None
    st.session_state.atom_strain_meta = None
    _invalidate_custom_basis_state()
    _invalidate_atomic_intensity_state()


def _invalidate_custom_basis_state() -> None:
    st.session_state.custom_basis_indices = {"base": None, "a": None, "b": None}
    st.session_state.custom_basis_stage = 0
    st.session_state.custom_basis_message = ""
    st.session_state.custom_basis_last_picked_atom = None
    st.session_state.custom_basis_picker_nonce = int(
        st.session_state.get("custom_basis_picker_nonce", 0)
    ) + 1
    st.session_state.atomic_displacement_manual_pairs = {}
    st.session_state.atomic_displacement_manual_atom = None
    st.session_state.atomic_displacement_manual_site = None
    _invalidate_sublattice_state(clear_manual_labels=True)
    _invalidate_atomic_displacement_state()


def _disable_custom_basis_state() -> None:
    st.session_state.custom_basis_enabled = False
    _invalidate_custom_basis_state()


def _invalidate_atomic_displacement_state() -> None:
    st.session_state.atomic_displacement_result = None
    st.session_state.atomic_displacement_meta = None
    _invalidate_plane_spacing_state()


def _coords_inside_roi_mask(coords: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray | None:
    if roi_mask is None or len(coords) == 0:
        return None
    mask = np.asarray(roi_mask, dtype=bool)
    yi = np.clip(np.round(coords[:, 0]).astype(int), 0, mask.shape[0] - 1)
    xi = np.clip(np.round(coords[:, 1]).astype(int), 0, mask.shape[1] - 1)
    return mask[yi, xi]


def _current_basis_markers(coords: np.ndarray) -> dict[str, tuple[float, float]]:
    markers: dict[str, tuple[float, float]] = {}
    basis_indices = dict(st.session_state.custom_basis_indices)
    for key_name in ("base", "a", "b"):
        idx = basis_indices.get(key_name)
        if idx is None or not (0 <= int(idx) < len(coords)):
            continue
        yx = np.asarray(coords[int(idx)], dtype=np.float64)
        markers[key_name] = (float(yx[0]), float(yx[1]))
    return markers


def _apply_custom_basis_clicks(
    coords: np.ndarray,
    raw_points: np.ndarray,
) -> tuple[bool, str | None]:
    if not bool(st.session_state.custom_basis_enabled):
        return False, "Enable the custom a-b basis tool first."
    if len(coords) == 0 or len(raw_points) == 0:
        return False, None

    basis_indices = dict(st.session_state.custom_basis_indices)
    basis_stage = int(st.session_state.custom_basis_stage)
    changed = False
    message: str | None = None

    for y_click, x_click in raw_points:
        if basis_stage >= 3:
            break
        d2 = (coords[:, 0] - y_click) ** 2 + (coords[:, 1] - x_click) ** 2
        nearest_idx = int(np.argmin(d2))
        target_name = ("base", "a", "b")[basis_stage]
        if target_name == "a" and nearest_idx == basis_indices.get("base"):
            message = "The a-vector atom must be different from the base atom."
            break
        if target_name == "b" and nearest_idx in {
            basis_indices.get("base"),
            basis_indices.get("a"),
        }:
            message = "The b-vector atom must be different from both base and a atoms."
            break

        basis_indices[target_name] = nearest_idx
        basis_stage += 1
        changed = True
        if basis_stage == 1:
            message = "Base selected. Now click the atom that defines the a vector."
        elif basis_stage == 2:
            message = "a-vector atom selected. Now click the atom that defines the b vector."
        elif basis_stage >= 3:
            message = "Custom a-b basis is ready."

    if changed:
        st.session_state.custom_basis_indices = basis_indices
        st.session_state.custom_basis_stage = basis_stage
    if message is not None:
        st.session_state.custom_basis_message = message
    return changed, message


def _extract_altair_pick_value(
    event_state: Any,
    *,
    selection_name: str,
    field_name: str,
) -> Any | None:
    if event_state is None:
        return None
    selection = getattr(event_state, "selection", None)
    if selection is None:
        return None
    selection_state = selection.get(selection_name, {})
    if isinstance(selection_state, list) and selection_state:
        first = selection_state[0]
        if isinstance(first, dict) and field_name in first:
            value = first.get(field_name)
            if isinstance(value, list) and value:
                return value[0]
            return value
    if not isinstance(selection_state, dict):
        return None
    value = selection_state.get(field_name)
    if isinstance(value, list) and value:
        return value[0]
    return value


def _extract_altair_atom_pick(event_state: Any, selection_name: str = "basis_pick") -> int | None:
    atom_index = _extract_altair_pick_value(
        event_state,
        selection_name=selection_name,
        field_name="atom_index",
    )
    if isinstance(atom_index, list) and atom_index:
        return int(atom_index[0])
    if isinstance(atom_index, (int, float)):
        return int(atom_index)
    return None


def _extract_altair_site_pick(
    event_state: Any, selection_name: str = "manual_site_pick"
) -> tuple[int, int] | None:
    site_key = _extract_altair_pick_value(
        event_state,
        selection_name=selection_name,
        field_name="site_key",
    )
    if not isinstance(site_key, str) or "," not in site_key:
        return None
    i_text, j_text = site_key.split(",", maxsplit=1)
    try:
        return int(i_text), int(j_text)
    except ValueError:
        return None


def _render_custom_basis_picker_chart(
    image_norm: np.ndarray,
    coords: np.ndarray,
    basis_indices: dict[str, int | None],
    *,
    max_side: int = 900,
    chart_max_side: int = 620,
):
    image_only, sx, sy = _to_display_image(
        image_norm,
        coords=np.empty((0, 2), dtype=np.float64),
        roi_mask=None,
        basis_markers=None,
        max_side=max_side,
    )
    disp_w, disp_h = image_only.size
    chart_scale = min(1.0, float(chart_max_side) / max(disp_w, disp_h))
    chart_w = max(260, int(round(disp_w * chart_scale)))
    chart_h = max(260, int(round(disp_h * chart_scale)))

    role_lookup = {int(v): k for k, v in basis_indices.items() if v is not None}
    point_df = pd.DataFrame(
        {
            "atom_index": np.arange(len(coords), dtype=int),
            "x": np.asarray(coords[:, 1] / sx, dtype=np.float64),
            "y": np.asarray(coords[:, 0] / sy, dtype=np.float64),
            "role": [role_lookup.get(i, "atom") for i in range(len(coords))],
        }
    )
    point_df["color"] = point_df["role"].map(
        {"base": "#00ffff", "a": "#ffd54f", "b": "#ff66cc", "atom": "#ff4d4d"}
    )
    point_df["size"] = point_df["role"].map({"base": 120, "a": 110, "b": 110, "atom": 55})

    picker = alt.selection_point(
        name="basis_pick",
        fields=["atom_index"],
        nearest=True,
        on="click",
        clear=False,
    )

    points = (
        alt.Chart(point_df)
        .mark_circle(stroke="white", strokeWidth=0.45, opacity=0.9)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, disp_w])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[disp_h, 0])),
            color=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("size:Q", legend=None),
            tooltip=[
                alt.Tooltip("atom_index:Q", title="atom index"),
                alt.Tooltip("role:N", title="role"),
            ],
            fillOpacity=alt.condition(picker, alt.value(1.0), alt.value(0.85)),
        )
        .add_params(picker)
    )
    chart = (
        alt.layer(points)
        .properties(width=chart_w, height=chart_h)
        .configure_view(stroke=None)
    )
    return chart


def _render_manual_observed_picker_chart(
    image_norm: np.ndarray,
    result: AtomicDisplacementResult,
    manual_pairs: dict[int, tuple[int, int]],
    *,
    max_side: int = 900,
    chart_max_side: int = 520,
):
    image_only, sx, sy = _to_display_image(
        image_norm,
        coords=np.empty((0, 2), dtype=np.float64),
        roi_mask=None,
        basis_markers=None,
        max_side=max_side,
    )
    disp_w, disp_h = image_only.size
    chart_scale = min(1.0, float(chart_max_side) / max(disp_w, disp_h))
    chart_w = max(240, int(round(disp_w * chart_scale)))
    chart_h = max(240, int(round(disp_h * chart_scale)))
    obs = np.asarray(result.coordinates_observed_yx, dtype=np.float64)
    valid = np.asarray(result.valid_atom_mask, dtype=bool)
    manual_set = {int(k) for k in manual_pairs}
    roles = [
        "manual" if idx in manual_set else ("valid" if bool(valid[idx]) else "invalid")
        for idx in range(len(obs))
    ]
    point_df = pd.DataFrame(
        {
            "atom_index": np.arange(len(obs), dtype=int),
            "x": np.asarray(obs[:, 1] / sx, dtype=np.float64),
            "y": np.asarray(obs[:, 0] / sy, dtype=np.float64),
            "role": roles,
        }
    )
    point_df["color"] = point_df["role"].map(
        {"manual": "#ffd54f", "invalid": "#ff5252", "valid": "#64b5f6"}
    )
    point_df["size"] = point_df["role"].map({"manual": 95, "invalid": 75, "valid": 45})
    picker = alt.selection_point(
        name="manual_atom_pick",
        fields=["atom_index"],
        nearest=True,
        on="click",
        clear=False,
    )
    return (
        alt.Chart(point_df)
        .mark_circle(stroke="white", strokeWidth=0.45, opacity=0.9)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, disp_w])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[disp_h, 0])),
            color=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("size:Q", legend=None),
            tooltip=[
                alt.Tooltip("atom_index:Q", title="atom index"),
                alt.Tooltip("role:N", title="role"),
            ],
            fillOpacity=alt.condition(picker, alt.value(1.0), alt.value(0.82)),
        )
        .add_params(picker)
        .properties(width=chart_w, height=chart_h)
        .configure_view(stroke=None)
    )


def _render_manual_ideal_site_picker_chart(
    image_norm: np.ndarray,
    result: AtomicDisplacementResult,
    manual_pairs: dict[int, tuple[int, int]],
    *,
    max_side: int = 900,
    chart_max_side: int = 520,
):
    image_only, sx, sy = _to_display_image(
        image_norm,
        coords=np.empty((0, 2), dtype=np.float64),
        roi_mask=None,
        basis_markers=None,
        max_side=max_side,
    )
    disp_w, disp_h = image_only.size
    chart_scale = min(1.0, float(chart_max_side) / max(disp_w, disp_h))
    chart_w = max(240, int(round(disp_w * chart_scale)))
    chart_h = max(240, int(round(disp_h * chart_scale)))
    h, w = image_norm.shape
    origin = np.asarray(result.fitted_origin_yx, dtype=np.float64)
    basis = np.asarray(result.fitted_basis_yx, dtype=np.float64)
    indices = np.asarray(result.lattice_indices_ij, dtype=np.int32)
    i_min, j_min = np.min(indices, axis=0) - 2
    i_max, j_max = np.max(indices, axis=0) + 2
    manual_sites = {tuple(map(int, pair)) for pair in manual_pairs.values()}

    rows: list[dict[str, Any]] = []
    for i in range(int(i_min), int(i_max) + 1):
        for j in range(int(j_min), int(j_max) + 1):
            yx = origin + i * basis[:, 0] + j * basis[:, 1]
            y, x = float(yx[0]), float(yx[1])
            if y < -20 or y > h + 20 or x < -20 or x > w + 20:
                continue
            rows.append(
                {
                    "site_key": f"{i},{j}",
                    "lattice_i": i,
                    "lattice_j": j,
                    "x": x / sx,
                    "y": y / sy,
                    "role": "manual target" if (i, j) in manual_sites else "ideal site",
                }
            )
    site_df = pd.DataFrame(rows)
    if site_df.empty:
        site_df = pd.DataFrame(
            {
                "site_key": [],
                "lattice_i": [],
                "lattice_j": [],
                "x": [],
                "y": [],
                "role": [],
            }
        )
    site_df["color"] = site_df["role"].map(
        {"manual target": "#ffd54f", "ideal site": "#00bcd4"}
    )
    site_df["size"] = site_df["role"].map({"manual target": 95, "ideal site": 48})
    picker = alt.selection_point(
        name="manual_site_pick",
        fields=["site_key"],
        nearest=True,
        on="click",
        clear=False,
    )
    return (
        alt.Chart(site_df)
        .mark_circle(stroke="#263238", strokeWidth=0.45, opacity=0.9)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, disp_w])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[disp_h, 0])),
            color=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("size:Q", legend=None),
            tooltip=[
                alt.Tooltip("site_key:N", title="site"),
                alt.Tooltip("role:N", title="role"),
            ],
            fillOpacity=alt.condition(picker, alt.value(1.0), alt.value(0.82)),
        )
        .add_params(picker)
        .properties(width=chart_w, height=chart_h)
        .configure_view(stroke=None)
    )


def _ideal_site_rows_for_result(
    image_norm: np.ndarray,
    result: AtomicDisplacementResult,
    manual_pairs: dict[int, tuple[int, int]],
    *,
    sx: float,
    sy: float,
) -> list[dict[str, Any]]:
    h, w = image_norm.shape
    origin = np.asarray(result.fitted_origin_yx, dtype=np.float64)
    basis = np.asarray(result.fitted_basis_yx, dtype=np.float64)
    indices = np.asarray(result.lattice_indices_ij, dtype=np.int32)
    i_min, j_min = np.min(indices, axis=0) - 2
    i_max, j_max = np.max(indices, axis=0) + 2
    manual_sites = {tuple(map(int, pair)) for pair in manual_pairs.values()}
    rows: list[dict[str, Any]] = []
    for i in range(int(i_min), int(i_max) + 1):
        for j in range(int(j_min), int(j_max) + 1):
            yx = origin + i * basis[:, 0] + j * basis[:, 1]
            y, x = float(yx[0]), float(yx[1])
            if y < -20 or y > h + 20 or x < -20 or x > w + 20:
                continue
            rows.append(
                {
                    "site_key": f"{i},{j}",
                    "lattice_i": i,
                    "lattice_j": j,
                    "x": x / sx,
                    "y": y / sy,
                    "role": "manual target" if (i, j) in manual_sites else "ideal site",
                }
            )
    return rows


def _render_manual_pair_overlay_chart(
    image_norm: np.ndarray,
    result: AtomicDisplacementResult,
    manual_pairs: dict[int, tuple[int, int]],
    *,
    pick_mode: str,
    max_side: int = 900,
    chart_max_side: int = 760,
):
    image_only, sx, sy = _to_display_image(
        image_norm,
        coords=np.empty((0, 2), dtype=np.float64),
        roi_mask=None,
        basis_markers=None,
        max_side=max_side,
    )
    disp_w, disp_h = image_only.size
    chart_scale = min(1.0, float(chart_max_side) / max(disp_w, disp_h))
    chart_w = max(320, int(round(disp_w * chart_scale)))
    chart_h = max(320, int(round(disp_h * chart_scale)))

    obs = np.asarray(result.coordinates_observed_yx, dtype=np.float64)
    valid = np.asarray(result.valid_atom_mask, dtype=bool)
    manual_set = {int(k) for k in manual_pairs}
    atom_df = pd.DataFrame(
        {
            "atom_index": np.arange(len(obs), dtype=int),
            "x": np.asarray(obs[:, 1] / sx, dtype=np.float64),
            "y": np.asarray(obs[:, 0] / sy, dtype=np.float64),
            "role": [
                "manual atom"
                if idx in manual_set
                else ("valid atom" if bool(valid[idx]) else "invalid atom")
                for idx in range(len(obs))
            ],
        }
    )
    atom_df["color"] = atom_df["role"].map(
        {"manual atom": "#ffd54f", "invalid atom": "#ff5252", "valid atom": "#64b5f6"}
    )
    atom_df["size"] = atom_df["role"].map(
        {"manual atom": 115, "invalid atom": 92, "valid atom": 62}
    )

    site_rows = _ideal_site_rows_for_result(
        image_norm,
        result,
        manual_pairs,
        sx=sx,
        sy=sy,
    )
    site_df = pd.DataFrame(site_rows)
    if site_df.empty:
        site_df = pd.DataFrame(
            {
                "site_key": [],
                "lattice_i": [],
                "lattice_j": [],
                "x": [],
                "y": [],
                "role": [],
            }
        )
    site_df["color"] = site_df["role"].map(
        {"manual target": "#ffd54f", "ideal site": "#00bcd4"}
    )
    site_df["size"] = site_df["role"].map({"manual target": 110, "ideal site": 70})

    atom_picker = alt.selection_point(
        name="manual_atom_pick",
        fields=["atom_index"],
        nearest=True,
        on="click",
        clear=False,
    )
    site_picker = alt.selection_point(
        name="manual_site_pick",
        fields=["site_key"],
        nearest=True,
        on="click",
        clear=False,
    )
    atom_points = (
        alt.Chart(atom_df)
        .mark_circle(stroke="white", strokeWidth=0.45, opacity=0.82)
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, disp_w])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[disp_h, 0])),
            color=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("size:Q", legend=None),
            tooltip=[
                alt.Tooltip("atom_index:Q", title="atom index"),
                alt.Tooltip("role:N", title="role"),
            ],
        )
    )
    site_points = (
        alt.Chart(site_df)
        .mark_point(
            shape="cross",
            strokeWidth=1.25,
            filled=False,
            opacity=0.88,
        )
        .encode(
            x=alt.X("x:Q", axis=None, scale=alt.Scale(domain=[0, disp_w])),
            y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[disp_h, 0])),
            color=alt.Color("color:N", scale=None, legend=None),
            size=alt.Size("size:Q", legend=None),
            tooltip=[
                alt.Tooltip("site_key:N", title="site"),
                alt.Tooltip("role:N", title="role"),
            ],
        )
    )
    if pick_mode == "Observed atom":
        atom_points = atom_points.add_params(atom_picker).encode(
            fillOpacity=alt.condition(atom_picker, alt.value(1.0), alt.value(0.82))
        )
    else:
        site_points = site_points.add_params(site_picker).encode(
            opacity=alt.condition(site_picker, alt.value(1.0), alt.value(0.88))
        )

    return (
        alt.layer(site_points, atom_points)
        .properties(width=chart_w, height=chart_h)
        .configure_view(stroke=None)
    )


def _project_strain_to_custom_basis(
    exx: np.ndarray,
    eyy: np.ndarray,
    exy: np.ndarray,
    *,
    a_vec_yx: np.ndarray,
    b_vec_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_xy = np.array([float(a_vec_yx[1]), float(a_vec_yx[0])], dtype=np.float64)
    b_xy = np.array([float(b_vec_yx[1]), float(b_vec_yx[0])], dtype=np.float64)
    a_norm = float(np.linalg.norm(a_xy))
    b_norm = float(np.linalg.norm(b_xy))
    if a_norm <= 1e-9 or b_norm <= 1e-9:
        raise ValueError("Custom basis vectors must have non-zero length.")
    a_hat = a_xy / a_norm
    b_hat = b_xy / b_norm
    exx = np.asarray(exx, dtype=np.float64)
    eyy = np.asarray(eyy, dtype=np.float64)
    exy = np.asarray(exy, dtype=np.float64)
    eaa = (a_hat[0] ** 2) * exx + 2.0 * a_hat[0] * a_hat[1] * exy + (a_hat[1] ** 2) * eyy
    ebb = (b_hat[0] ** 2) * exx + 2.0 * b_hat[0] * b_hat[1] * exy + (b_hat[1] ** 2) * eyy
    eab = (
        a_hat[0] * b_hat[0] * exx
        + (a_hat[0] * b_hat[1] + a_hat[1] * b_hat[0]) * exy
        + a_hat[1] * b_hat[1] * eyy
    )
    return eaa, ebb, eab


def _custom_basis_info(
    coords: np.ndarray,
    basis_indices: dict[str, int | None],
) -> dict[str, Any] | None:
    base_idx = basis_indices.get("base")
    a_idx = basis_indices.get("a")
    b_idx = basis_indices.get("b")
    if base_idx is None or a_idx is None or b_idx is None:
        return None
    if any(idx < 0 or idx >= len(coords) for idx in (base_idx, a_idx, b_idx)):
        return None
    base = np.asarray(coords[base_idx], dtype=np.float64)
    a_vec = np.asarray(coords[a_idx], dtype=np.float64) - base
    b_vec = np.asarray(coords[b_idx], dtype=np.float64) - base
    a_xy = np.array([a_vec[1], a_vec[0]], dtype=np.float64)
    b_xy = np.array([b_vec[1], b_vec[0]], dtype=np.float64)
    a_norm = float(np.linalg.norm(a_xy))
    b_norm = float(np.linalg.norm(b_xy))
    if a_norm <= 1e-9 or b_norm <= 1e-9:
        return None
    cosang = float(np.clip(np.dot(a_xy, b_xy) / (a_norm * b_norm), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosang)))
    return {
        "base_idx": int(base_idx),
        "a_idx": int(a_idx),
        "b_idx": int(b_idx),
        "base_yx": base,
        "a_vec_yx": a_vec,
        "b_vec_yx": b_vec,
        "a_length_px": float(np.linalg.norm(a_vec)),
        "b_length_px": float(np.linalg.norm(b_vec)),
        "angle_deg": angle_deg,
    }


def _to_display_image(
    image_norm: np.ndarray,
    coords: np.ndarray,
    roi_mask: np.ndarray | None = None,
    basis_markers: dict[str, tuple[float, float]] | None = None,
    max_side: int = 900,
    max_draw_atoms: int | None = None,
) -> tuple[Image.Image, float, float]:
    h, w = image_norm.shape
    scale = min(max_side / max(h, w), 1.0)
    disp_h = max(1, int(round(h * scale)))
    disp_w = max(1, int(round(w * scale)))
    arr = (np.clip(image_norm, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB").resize(
        (disp_w, disp_h), resample=Image.BILINEAR
    )

    draw = ImageDraw.Draw(img)
    if roi_mask is not None:
        roi_disp = Image.fromarray((roi_mask.astype(np.uint8) * 255), mode="L").resize(
            (disp_w, disp_h), resample=Image.NEAREST
        )
        roi_arr = np.asarray(roi_disp)
        overlay = np.asarray(img).copy()
        overlay[roi_arr > 0] = (255, 160, 0)
        blend = (0.75 * np.asarray(img) + 0.25 * overlay).astype(np.uint8)
        img = Image.fromarray(blend, mode="RGB")
        draw = ImageDraw.Draw(img)

    draw_coords = np.asarray(coords, dtype=np.float64)
    if max_draw_atoms is not None and len(draw_coords) > int(max_draw_atoms):
        stride = max(int(np.ceil(len(draw_coords) / int(max_draw_atoms))), 1)
        draw_coords = draw_coords[::stride]

    if len(draw_coords) > 0:
        px_r = max(2, int(round(2.6 * scale)))
        for y, x in draw_coords:
            cx = float(x / (w / disp_w))
            cy = float(y / (h / disp_h))
            draw.ellipse(
                (cx - px_r, cy - px_r, cx + px_r, cy + px_r),
                outline=(255, 48, 48),
                width=1,
            )

    if basis_markers:
        color_map = {
            "base": (0, 255, 255),
            "a": (255, 220, 0),
            "b": (255, 0, 255),
        }
        disp_points: dict[str, tuple[float, float]] = {}
        for label, (y, x) in basis_markers.items():
            cx = float(x / (w / disp_w))
            cy = float(y / (h / disp_h))
            disp_points[label] = (cx, cy)
            rr = max(4, int(round(5.5 * scale)))
            color = color_map.get(label, (0, 255, 255))
            draw.ellipse(
                (cx - rr, cy - rr, cx + rr, cy + rr),
                outline=color,
                width=2,
            )
            draw.text((cx + rr + 2, cy - rr - 2), label, fill=color)
        if "base" in disp_points and "a" in disp_points:
            draw.line([disp_points["base"], disp_points["a"]], fill=color_map["a"], width=2)
        if "base" in disp_points and "b" in disp_points:
            draw.line([disp_points["base"], disp_points["b"]], fill=color_map["b"], width=2)

    sx = w / disp_w
    sy = h / disp_h
    return img, sx, sy


def _map_display_to_raw(points_yx: np.ndarray, sx: float, sy: float) -> np.ndarray:
    if len(points_yx) == 0:
        return points_yx
    mapped = points_yx.copy()
    mapped[:, 0] *= sy
    mapped[:, 1] *= sx
    return mapped


def _nearest_atom_index(coords: np.ndarray, y: float, x: float) -> int | None:
    if len(coords) == 0:
        return None
    d2 = (coords[:, 0] - float(y)) ** 2 + (coords[:, 1] - float(x)) ** 2
    return int(np.argmin(d2))


def _canvas_object_points(canvas_result) -> np.ndarray:
    json_data = getattr(canvas_result, "json_data", None)
    if not isinstance(json_data, dict):
        return np.empty((0, 2), dtype=np.float64)
    rows: list[tuple[float, float]] = []
    for obj in json_data.get("objects", []):
        if not isinstance(obj, dict):
            continue
        if obj.get("type") != "circle":
            continue
        left = obj.get("left")
        top = obj.get("top")
        radius = float(obj.get("radius", 0.0) or 0.0)
        if left is None or top is None:
            continue
        x = float(left)
        y = float(top)
        if obj.get("originX") != "center":
            x += radius * float(obj.get("scaleX", 1.0) or 1.0)
        if obj.get("originY") != "center":
            y += radius * float(obj.get("scaleY", 1.0) or 1.0)
        rows.append((y, x))
    if not rows:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _canvas_draw_mask(canvas_result) -> np.ndarray | None:
    if canvas_result is None or getattr(canvas_result, "image_data", None) is None:
        return None
    arr = np.asarray(canvas_result.image_data)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return None
    alpha = arr[:, :, 3]
    return alpha > 0


def _mask_centroids(mask: np.ndarray, min_pixels: int = 3) -> np.ndarray:
    labeled, n_obj = ndi.label(mask)
    if n_obj <= 0:
        return np.empty((0, 2), dtype=np.float64)
    out: list[tuple[float, float]] = []
    for idx in range(1, n_obj + 1):
        comp = labeled == idx
        if int(np.sum(comp)) < min_pixels:
            continue
        cy, cx = ndi.center_of_mass(comp)
        out.append((float(cy), float(cx)))
    if not out:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def _resize_mask_to_raw(mask_disp: np.ndarray, raw_shape: tuple[int, int]) -> np.ndarray:
    h, w = raw_shape
    im = Image.fromarray((mask_disp.astype(np.uint8) * 255), mode="L").resize(
        (w, h), resample=Image.NEAREST
    )
    return np.asarray(im) > 0


for key, default in [
    ("uploaded_sig", None),
    ("uploaded_temp_path", None),
    ("last_detection_signature", None),
    ("base_result", None),
    ("base_image_raw", None),
    ("base_image_norm", None),
    ("edited_coords", None),
    ("edited_intensities", None),
    ("auto_coords", None),
    ("auto_intensities", None),
    ("roi_mask", None),
    ("canvas_nonce", 0),
    ("refinement_info", None),
    ("lattice_info", None),
    ("last_run_summary", None),
    ("atom_strain_fields", None),
    ("atom_strain_meta", None),
    ("custom_basis_indices", {"base": None, "a": None, "b": None}),
    ("custom_basis_stage", 0),
    ("custom_basis_message", ""),
    ("custom_basis_enabled", False),
    ("custom_basis_picker_nonce", 0),
    ("custom_basis_last_picked_atom", None),
    ("atomic_displacement_result", None),
    ("atomic_displacement_meta", None),
    ("atomic_displacement_method", DISPLACEMENT_METHOD_MANUAL_BASIS),
    ("atomic_displacement_tolerance", 0.40),
    ("atomic_displacement_manual_pairs", {}),
    ("atomic_displacement_manual_atom", None),
    ("atomic_displacement_manual_site", None),
    ("plane_spacing_result", None),
    ("plane_spacing_meta", None),
    ("atomic_intensity_result", None),
    ("atomic_intensity_meta", None),
    ("sublattice_result", None),
    ("sublattice_meta", None),
    ("sublattice_manual_labels", {}),
    ("sublattice_selected_atom", None),
    ("sublattice_picker_nonce", 0),
    ("distance_pixel_size", 0.02),
    ("distance_unit", "nm"),
    ("pixel_size_auto_nm", None),
    ("pixel_size_auto_source", None),
    ("pixel_size_auto_messages", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

uploaded = st.file_uploader("Upload TEM image", type=["dm", "dm3", "dm4", "tif", "tiff"])
manual_run = st.button("Run Detection", type="primary", disabled=uploaded is None)

if uploaded is None:
    st.info("Upload a TEM image to begin.")
    st.stop()

current_file_sig = f"{uploaded.name}:{uploaded.size}"
if st.session_state.uploaded_sig != current_file_sig:
    st.session_state.uploaded_sig = current_file_sig
    st.session_state.uploaded_temp_path = _save_upload_to_temp(uploaded)
    st.session_state.last_detection_signature = None
    st.session_state.base_result = None
    st.session_state.base_image_raw = None
    st.session_state.base_image_norm = None
    st.session_state.edited_coords = None
    st.session_state.edited_intensities = None
    st.session_state.auto_coords = None
    st.session_state.auto_intensities = None
    st.session_state.roi_mask = None
    st.session_state.canvas_nonce = 0
    st.session_state.refinement_info = None
    st.session_state.lattice_info = None
    _invalidate_atom_strain_state()
    auto_nm, auto_source, auto_messages = _cached_infer_pixel_size_nm(
        st.session_state.uploaded_temp_path,
        file_size=uploaded.size,
    )
    st.session_state.pixel_size_auto_nm = auto_nm
    st.session_state.pixel_size_auto_source = auto_source
    st.session_state.pixel_size_auto_messages = auto_messages
    if auto_nm is not None:
        st.session_state.distance_pixel_size = float(auto_nm)


with st.sidebar:
    st.header("Parameters")
    auto_run = st.toggle(
        "Realtime update",
        value=True,
        help="Rerun automatically when parameters change.",
    )
    st.caption("Realtime ON: parameter changes trigger rerun automatically.")
    fast_mode_realtime = st.toggle(
        "Fast mode (realtime)",
        value=True,
        help="Use cached image loading and faster refinement path while adjusting sliders.",
    )
    fast_gaussian_max_points = st.number_input(
        "Fast mode Gaussian max points",
        min_value=50,
        max_value=5000,
        value=800,
        step=50,
        help=(
            "If detected points exceed this count in realtime, "
            "Gaussian refinement auto-falls back to CoM."
        ),
    )
    mode = st.selectbox("Mode", ["auto", "log", "dog"], index=0)
    gaussian_sigma = st.slider(
        "Gaussian sigma (0 = off)",
        min_value=0.0,
        max_value=3.0,
        value=0.8,
        step=0.1,
    )
    min_distance = st.slider("Min distance (pixels)", min_value=1, max_value=20, value=4, step=1)
    threshold_rel = st.slider(
        "Threshold rel",
        min_value=0.05,
        max_value=0.95,
        value=0.22,
        step=0.01,
    )
    max_atoms = st.number_input("Max atoms", min_value=10, max_value=200000, value=5000, step=100)
    slice_index = st.number_input("Slice index (for stack-like DM)", min_value=0, value=0, step=1)

    st.divider()
    st.subheader("Subpixel Refinement")
    refinement_method = st.selectbox(
        "Refinement method",
        ["none", "gaussian", "com"],
        index=1,
        help="Use subpixel refinement after baseline detection.",
    )
    refinement_window_radius = st.slider(
        "Refinement window radius", min_value=2, max_value=10, value=4, step=1
    )
    refinement_max_shift = st.slider(
        "Refinement max shift (px)", min_value=0.2, max_value=3.0, value=1.5, step=0.1
    )

    st.divider()
    st.subheader("Lattice Constraint (Optional)")
    lattice_constraint_enable = st.checkbox(
        "Enable lattice constraint correction",
        value=False,
        help=(
            "Optional local smoothing. Keep OFF for defect/dislocation analysis "
            "and many GPA use-cases."
        ),
    )
    lattice_k_neighbors = st.slider(
        "Lattice neighbors", min_value=3, max_value=16, value=6, step=1
    )
    lattice_strength = st.slider(
        "Lattice strength", min_value=0.05, max_value=0.8, value=0.2, step=0.05
    )
    lattice_iterations = st.slider("Lattice iterations", min_value=1, max_value=5, value=1, step=1)
    lattice_max_shift_per_iter = st.slider(
        "Lattice max shift/iter (px)", min_value=0.1, max_value=2.0, value=0.6, step=0.1
    )

    st.divider()
    st.subheader("Distance Statistics")
    use_physical_unit = st.checkbox(
        "Convert distance to physical unit",
        value=False,
        help="If enabled, distance output will be scaled by pixel size.",
    )
    pixel_size = st.number_input(
        "Pixel size",
        min_value=1e-6,
        key="distance_pixel_size",
        step=0.001,
        format="%.6f",
        disabled=not use_physical_unit,
    )
    distance_unit = st.selectbox(
        "Distance unit",
        ["nm", "angstrom", "pm", "um"],
        key="distance_unit",
        index=0,
        disabled=not use_physical_unit,
    )
    if st.session_state.pixel_size_auto_nm is not None:
        st.caption(
            "Auto pixel size: "
            f"{float(st.session_state.pixel_size_auto_nm):.6f} nm/px "
            f"({st.session_state.pixel_size_auto_source})"
        )
    else:
        st.caption("Auto pixel size unavailable. Please input manually or calibrate by scale bar.")
        if st.session_state.pixel_size_auto_messages:
            with st.expander("Auto-read details"):
                st.code("\n".join(st.session_state.pixel_size_auto_messages), language="text")

    st.divider()
    st.subheader("Atom-Position Strain")
    atom_strain_use_roi_reference = st.checkbox(
        "Use current ROI as ideal lattice region",
        value=False,
        help=(
            "If enabled, the ROI drawn on the image will define the ideal/reference lattice "
            "for atom-position strain, without forcing you to keep/remove atoms."
        ),
    )
    st.caption(
        "Draw ROI in the image panel when you want a custom ideal-lattice reference area. "
        "Leave this OFF to use the whole edited atom set as the reference."
    )
    atom_strain_neighbor_k = st.slider(
        "Atom strain neighbors", min_value=6, max_value=20, value=10, step=1
    )
    atom_strain_tolerance_deg = st.slider(
        "Direction tolerance (deg)",
        min_value=10.0,
        max_value=35.0,
        value=18.0,
        step=1.0,
    )
    atom_strain_grid_step = st.slider(
        "Interpolation grid step", min_value=1, max_value=6, value=2, step=1
    )
    atom_strain_display_unit = st.selectbox(
        "Strain display unit",
        ["fraction", "%"],
        index=0,
        help="Controls how exx / eyy / exy are shown in maps, summary, and CSV export.",
    )
    atom_strain_auto_color_range = st.checkbox(
        "Atom-strain auto color range",
        value=True,
        help="If OFF, use the manual symmetric ranges below for the atom-strain maps.",
    )
    atom_strain_displacement_abs = st.number_input(
        "Displacement range abs",
        min_value=0.001,
        value=5.0,
        step=0.1,
        disabled=atom_strain_auto_color_range,
    )
    atom_strain_strain_abs = st.number_input(
        "Strain range abs",
        min_value=0.001,
        value=1.0,
        step=0.01,
        format="%.3f",
        disabled=atom_strain_auto_color_range,
        help="Applies to exx / eyy / exy together, shown as symmetric [-range, +range].",
    )
    atom_strain_rotation_abs = st.number_input(
        "Rotation range abs (rad)",
        min_value=0.001,
        value=0.5,
        step=0.01,
        format="%.3f",
        disabled=atom_strain_auto_color_range,
    )
    st.caption("Overlay strain map on original image")
    atom_strain_overlay_metric = st.selectbox(
        "Overlay metric",
        ["exx", "eyy", "exy"],
        index=0,
        help="Color atom-column positions on the original image by the selected strain component.",
    )
    atom_strain_overlay_auto_range = st.checkbox(
        "Overlay auto range",
        value=True,
        help="If OFF, use manual min/max for the atom-position strain overlay.",
    )
    atom_strain_overlay_min = st.number_input(
        "Overlay min",
        value=-1.0,
        step=0.01,
        format="%.3f",
        disabled=atom_strain_overlay_auto_range,
    )
    atom_strain_overlay_max = st.number_input(
        "Overlay max",
        value=1.0,
        step=0.01,
        format="%.3f",
        disabled=atom_strain_overlay_auto_range,
    )
    atom_strain_overlay_marker_size = st.slider(
        "Overlay marker size", min_value=8.0, max_value=80.0, value=28.0, step=2.0
    )

    st.divider()
    st.subheader("Atomic Displacement")
    st.caption(
        "Standalone ideal-lattice displacement module. "
        "Methods 1 and 2 both use the custom a-b direction picker."
    )
    current_displacement_method = st.session_state.atomic_displacement_method
    if current_displacement_method not in DISPLACEMENT_METHOD_OPTIONS:
        current_displacement_method = DISPLACEMENT_METHOD_MANUAL_BASIS
    atomic_displacement_method = st.selectbox(
        "Displacement method",
        DISPLACEMENT_METHOD_OPTIONS,
        index=DISPLACEMENT_METHOD_OPTIONS.index(current_displacement_method),
        key="atomic_displacement_method",
        help=(
            "Method 1 uses the custom base/a/b basis directly. "
            "Method 2 uses your custom a/b direction, but fits origin/a/b from the current ROI."
        ),
    )
    atomic_displacement_tolerance = st.slider(
        "Lattice index tolerance",
        min_value=0.10,
        max_value=0.80,
        value=float(st.session_state.atomic_displacement_tolerance),
        step=0.01,
        key="atomic_displacement_tolerance",
        help=(
            "Atoms whose fractional lattice index is farther than this threshold "
            "from the nearest integer lattice site are marked as invalid assignment."
        ),
    )
    atomic_displacement_vector_show_image = st.checkbox(
        "Vector map overlay original image",
        value=True,
        help="If OFF, show the displacement vector map on a clean white background.",
    )
    atomic_displacement_arrow_scale = st.slider(
        "Vector arrow display scale",
        min_value=0.2,
        max_value=8.0,
        value=1.0,
        step=0.2,
        help="Only changes arrow length in the figure; it does not change measured values.",
    )
    atomic_displacement_marker_size = st.slider(
        "Displacement marker size",
        min_value=8.0,
        max_value=80.0,
        value=26.0,
        step=2.0,
    )
    atomic_displacement_max_arrows = st.slider(
        "Max arrows to draw",
        min_value=100,
        max_value=3000,
        value=1200,
        step=100,
        help="Large atom maps are downsampled only for arrow display speed.",
    )

    st.divider()
    st.subheader("Atomic Spacing Segments")
    plane_spacing_method = st.selectbox(
        "Spacing method",
        [
            "a/b lattice index",
            "a/b lattice index + fixed angle",
            "lattice angle only (auto directions)",
            "manual angle",
            "picked direction from two atoms",
        ],
        index=0,
        help=(
            "a/b lattice index uses integer custom-basis indexing. "
            "lattice angle only skips base/a/b picking and infers two row directions "
            "from atom-pair statistics."
        ),
    )
    plane_spacing_uses_lattice_controls = plane_spacing_method in {
        "a/b lattice index",
        "a/b lattice index + fixed angle",
    }
    plane_spacing_uses_lattice_angle = plane_spacing_method in {
        "a/b lattice index + fixed angle",
        "lattice angle only (auto directions)",
    }
    plane_spacing_uses_angle_controls = plane_spacing_method in {
        "lattice angle only (auto directions)",
        "manual angle",
        "picked direction from two atoms",
    }
    plane_spacing_direction = st.selectbox(
        "Spacing segment direction",
        ["a + b", "a only", "b only"],
        index=0,
        disabled=not plane_spacing_uses_lattice_controls,
        help="Connect adjacent atoms according to the current custom a/b lattice indices.",
    )
    plane_spacing_lattice_tolerance = st.slider(
        "Spacing lattice index tolerance",
        min_value=0.10,
        max_value=0.80,
        value=0.40,
        step=0.01,
        disabled=not plane_spacing_uses_lattice_controls,
        help=(
            "Used only inside spacing quantification to decide whether an atom belongs "
            "to the custom a/b lattice."
        ),
    )
    plane_spacing_lattice_angle = st.number_input(
        "Lattice angle a-b (deg)",
        min_value=5.0,
        max_value=175.0,
        value=90.0,
        step=1.0,
        format="%.2f",
        disabled=not plane_spacing_uses_lattice_angle,
        help=(
            "Angle between two lattice row axes. In angle-only mode, the absolute "
            "orientation is inferred automatically from atom positions."
        ),
    )
    plane_spacing_angle_deg = st.number_input(
        "Spacing row direction angle (deg)",
        min_value=0.0,
        max_value=360.0,
        value=0.0,
        step=1.0,
        format="%.2f",
        disabled=plane_spacing_method != "manual angle",
        help=(
            "Image-coordinate axis angle: 0 deg is horizontal, 90 deg is vertical. "
            "Angle + 180 deg is treated as the same atom-row direction."
        ),
    )
    plane_spacing_angle_tolerance = st.slider(
        "Spacing angle tolerance (deg)",
        min_value=1.0,
        max_value=45.0,
        value=10.0,
        step=1.0,
        disabled=not plane_spacing_uses_angle_controls,
    )
    plane_spacing_min_px = st.number_input(
        "Spacing min distance (px)",
        min_value=0.1,
        value=3.0,
        step=0.5,
        format="%.2f",
        disabled=not plane_spacing_uses_angle_controls,
    )
    plane_spacing_max_px = st.number_input(
        "Spacing max distance (px)",
        min_value=0.2,
        value=80.0,
        step=0.5,
        format="%.2f",
        disabled=not plane_spacing_uses_angle_controls,
    )
    plane_spacing_pick_start = st.number_input(
        "Picked direction atom 1 index",
        min_value=0,
        value=0,
        step=1,
        disabled=plane_spacing_method != "picked direction from two atoms",
    )
    plane_spacing_pick_end = st.number_input(
        "Picked direction atom 2 index",
        min_value=0,
        value=1,
        step=1,
        disabled=plane_spacing_method != "picked direction from two atoms",
    )
    plane_spacing_show_image = st.checkbox(
        "Spacing map overlay original image",
        value=True,
    )
    plane_spacing_show_atoms = st.checkbox(
        "Show atom points on spacing map",
        value=True,
    )
    plane_spacing_auto_color_range = st.checkbox(
        "Spacing auto color range",
        value=True,
    )
    plane_spacing_color_min = st.number_input(
        "Spacing color min",
        value=0.0,
        step=0.1,
        format="%.3f",
        disabled=plane_spacing_auto_color_range,
    )
    plane_spacing_color_max = st.number_input(
        "Spacing color max",
        value=10.0,
        step=0.1,
        format="%.3f",
        disabled=plane_spacing_auto_color_range,
    )
    plane_spacing_line_width = st.slider(
        "Spacing line width",
        min_value=0.5,
        max_value=8.0,
        value=2.2,
        step=0.1,
    )

    st.divider()
    st.subheader("Atomic Column Intensity")
    intensity_method = st.selectbox(
        "Intensity method",
        ["Disk integration + annulus background"],
        index=0,
        help="Baseline quantitative intensity from C0/C1 atom coordinates.",
    )
    intensity_integration_radius = st.slider(
        "Integration radius (px)",
        min_value=0.5,
        max_value=10.0,
        value=2.5,
        step=0.5,
    )
    intensity_background_inner_radius = st.slider(
        "Background inner radius (px)",
        min_value=1.0,
        max_value=20.0,
        value=4.0,
        step=0.5,
    )
    intensity_background_outer_radius = st.slider(
        "Background outer radius (px)",
        min_value=2.0,
        max_value=30.0,
        value=7.0,
        step=0.5,
    )
    intensity_background_statistic = st.selectbox(
        "Background statistic",
        ["median", "mean", "percentile_25", "percentile_10"],
        index=0,
        help="Median is usually robust against neighboring weak contrast and noise.",
    )
    intensity_normalization = st.selectbox(
        "Normalize intensity",
        ["median", "mean", "none"],
        index=0,
        help="Median normalization makes the map easier to compare across atom columns.",
    )
    intensity_overlay_metric = st.selectbox(
        "Intensity map metric",
        [
            "normalized_intensity",
            "background_subtracted_sum",
            "peak_intensity",
            "snr",
        ],
        index=0,
    )
    intensity_auto_color_range = st.checkbox(
        "Intensity auto color range",
        value=True,
        help="If OFF, use manual min/max for the intensity overlay map.",
    )
    intensity_color_min = st.number_input(
        "Intensity color min",
        value=0.0,
        step=0.1,
        format="%.3f",
        disabled=intensity_auto_color_range,
    )
    intensity_color_max = st.number_input(
        "Intensity color max",
        value=2.0,
        step=0.1,
        format="%.3f",
        disabled=intensity_auto_color_range,
    )
    intensity_marker_size = st.slider(
        "Intensity marker size",
        min_value=8.0,
        max_value=90.0,
        value=30.0,
        step=2.0,
    )
    intensity_grouping_enabled = st.checkbox(
        "Enable intensity group analysis",
        value=False,
        help=(
            "Optional. Group atom columns by intensity clusters or by current "
            "user-defined a/b lattice sublattice."
        ),
    )
    intensity_grouping_method = "Intensity clustering"
    intensity_group_metric = "normalized_intensity"
    intensity_cluster_count = 2
    intensity_lattice_group_mode = "ij_parity"
    if intensity_grouping_enabled:
        intensity_grouping_method = st.selectbox(
            "Intensity grouping method",
            ["Intensity clustering", "Current a/b lattice sublattice"],
            index=0,
        )
        intensity_group_metric = st.selectbox(
            "Grouping metric",
            [
                "normalized_intensity",
                "background_subtracted_sum",
                "peak_intensity",
                "snr",
            ],
            index=0,
        )
        if intensity_grouping_method == "Intensity clustering":
            intensity_cluster_count = st.slider(
                "Intensity cluster count",
                min_value=2,
                max_value=6,
                value=2,
                step=1,
            )
        else:
            intensity_lattice_group_mode = st.selectbox(
                "Sublattice grouping",
                [
                    "ij_parity",
                    "sum_parity",
                    "i_parity",
                    "j_parity",
                ],
                index=0,
                help=(
                    "Uses lattice indices from the current atomic displacement result. "
                    "The a/b direction is defined by your selected custom basis."
                ),
            )

    st.divider()
    st.subheader("Sublattice Classification")
    sublattice_method = st.selectbox(
        "Sublattice method",
        ["Auto exploration", "User 2D projected model"],
        index=0,
        help=(
            "Auto exploration clusters atoms by projected unit-cell position and optional "
            "intensity. User model assigns labels from label,u,v sites."
        ),
    )
    sublattice_n_labels = st.slider(
        "Auto sublattice count",
        min_value=1,
        max_value=8,
        value=2,
        step=1,
        disabled=sublattice_method != "Auto exploration",
    )
    sublattice_use_intensity = st.checkbox(
        "Use intensity feature",
        value=True,
        help=(
            "Use current atom intensity or quantified normalized intensity as a "
            "clustering feature."
        ),
    )
    sublattice_intensity_source = st.selectbox(
        "Sublattice intensity source",
        ["Current atom peak intensity", "Atomic-column normalized intensity"],
        index=0,
        disabled=not sublattice_use_intensity,
    )
    sublattice_intensity_weight = st.slider(
        "Sublattice intensity weight",
        min_value=0.0,
        max_value=2.0,
        value=0.6,
        step=0.05,
        disabled=not sublattice_use_intensity
        or sublattice_method != "Auto exploration",
    )
    sublattice_model_tolerance = st.slider(
        "Model match tolerance",
        min_value=0.01,
        max_value=0.75,
        value=0.20,
        step=0.01,
        disabled=sublattice_method != "User 2D projected model",
        help="Distance tolerance inside the fractional a/b unit cell.",
    )
    sublattice_marker_size = st.slider(
        "Sublattice marker size",
        min_value=8.0,
        max_value=90.0,
        value=34.0,
        step=2.0,
    )
    sublattice_show_label_text = st.checkbox(
        "Show sublattice label text",
        value=False,
    )
    sublattice_show_atom_index = st.checkbox(
        "Show atom index text",
        value=False,
    )

params = {
    "mode": mode,
    "gaussian_sigma": None if gaussian_sigma == 0 else float(gaussian_sigma),
    "min_distance": int(min_distance),
    "threshold_rel": float(threshold_rel),
    "max_atoms": int(max_atoms),
    "slice_index": int(slice_index),
    "refinement_method": str(refinement_method),
    "refinement_window_radius": int(refinement_window_radius),
    "refinement_max_shift": float(refinement_max_shift),
    "lattice_constraint_enable": bool(lattice_constraint_enable),
    "lattice_k_neighbors": int(lattice_k_neighbors),
    "lattice_strength": float(lattice_strength),
    "lattice_iterations": int(lattice_iterations),
    "lattice_max_shift_per_iter": float(lattice_max_shift_per_iter),
    "fast_mode_realtime": bool(fast_mode_realtime),
    "fast_gaussian_max_points": int(fast_gaussian_max_points),
    "use_physical_unit": bool(use_physical_unit),
    "pixel_size": float(pixel_size),
    "distance_unit": str(distance_unit),
    "atom_strain_reference_mode": (
        "Current ROI mask"
        if bool(atom_strain_use_roi_reference)
        else "Whole edited atom set"
    ),
    "atom_strain_use_roi_reference": bool(atom_strain_use_roi_reference),
    "atom_strain_neighbor_k": int(atom_strain_neighbor_k),
    "atom_strain_tolerance_deg": float(atom_strain_tolerance_deg),
    "atom_strain_grid_step": int(atom_strain_grid_step),
    "atom_strain_display_unit": str(atom_strain_display_unit),
    "atom_strain_auto_color_range": bool(atom_strain_auto_color_range),
    "atom_strain_displacement_abs": float(atom_strain_displacement_abs),
    "atom_strain_strain_abs": float(atom_strain_strain_abs),
    "atom_strain_rotation_abs": float(atom_strain_rotation_abs),
    "atom_strain_overlay_metric": str(atom_strain_overlay_metric),
    "atom_strain_overlay_auto_range": bool(atom_strain_overlay_auto_range),
    "atom_strain_overlay_min": float(atom_strain_overlay_min),
    "atom_strain_overlay_max": float(atom_strain_overlay_max),
    "atom_strain_overlay_marker_size": float(atom_strain_overlay_marker_size),
    "atomic_displacement_method": str(atomic_displacement_method),
    "atomic_displacement_tolerance": float(atomic_displacement_tolerance),
    "atomic_displacement_vector_show_image": bool(atomic_displacement_vector_show_image),
    "atomic_displacement_arrow_scale": float(atomic_displacement_arrow_scale),
    "atomic_displacement_marker_size": float(atomic_displacement_marker_size),
    "atomic_displacement_max_arrows": int(atomic_displacement_max_arrows),
    "plane_spacing_method": str(plane_spacing_method),
    "plane_spacing_direction": str(plane_spacing_direction),
    "plane_spacing_lattice_tolerance": float(plane_spacing_lattice_tolerance),
    "plane_spacing_lattice_angle": float(plane_spacing_lattice_angle),
    "plane_spacing_angle_deg": float(plane_spacing_angle_deg),
    "plane_spacing_angle_tolerance": float(plane_spacing_angle_tolerance),
    "plane_spacing_min_px": float(plane_spacing_min_px),
    "plane_spacing_max_px": float(plane_spacing_max_px),
    "plane_spacing_pick_start": int(plane_spacing_pick_start),
    "plane_spacing_pick_end": int(plane_spacing_pick_end),
    "plane_spacing_show_image": bool(plane_spacing_show_image),
    "plane_spacing_show_atoms": bool(plane_spacing_show_atoms),
    "plane_spacing_auto_color_range": bool(plane_spacing_auto_color_range),
    "plane_spacing_color_min": float(plane_spacing_color_min),
    "plane_spacing_color_max": float(plane_spacing_color_max),
    "plane_spacing_line_width": float(plane_spacing_line_width),
    "intensity_method": str(intensity_method),
    "intensity_integration_radius": float(intensity_integration_radius),
    "intensity_background_inner_radius": float(intensity_background_inner_radius),
    "intensity_background_outer_radius": float(intensity_background_outer_radius),
    "intensity_background_statistic": str(intensity_background_statistic),
    "intensity_normalization": str(intensity_normalization),
    "intensity_overlay_metric": str(intensity_overlay_metric),
    "intensity_auto_color_range": bool(intensity_auto_color_range),
    "intensity_color_min": float(intensity_color_min),
    "intensity_color_max": float(intensity_color_max),
    "intensity_marker_size": float(intensity_marker_size),
    "intensity_grouping_enabled": bool(intensity_grouping_enabled),
    "intensity_grouping_method": str(intensity_grouping_method),
    "intensity_group_metric": str(intensity_group_metric),
    "intensity_cluster_count": int(intensity_cluster_count),
    "intensity_lattice_group_mode": str(intensity_lattice_group_mode),
    "sublattice_method": str(sublattice_method),
    "sublattice_n_labels": int(sublattice_n_labels),
    "sublattice_use_intensity": bool(sublattice_use_intensity),
    "sublattice_intensity_source": str(sublattice_intensity_source),
    "sublattice_intensity_weight": float(sublattice_intensity_weight),
    "sublattice_model_tolerance": float(sublattice_model_tolerance),
    "sublattice_marker_size": float(sublattice_marker_size),
    "sublattice_show_label_text": bool(sublattice_show_label_text),
    "sublattice_show_atom_index": bool(sublattice_show_atom_index),
}

sig = _signature(current_file_sig, params)
need_rerun = st.session_state.last_detection_signature != sig
should_run = manual_run or (auto_run and need_rerun)

if should_run:
    overlay_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    overlay_temp.close()
    with st.spinner("Running detection + refinement..."):
        try:
            base_raw, source_info, load_warnings = _cached_load_image(
                st.session_state.uploaded_temp_path,
                slice_index=params["slice_index"],
                file_size=uploaded.size,
            )
            result = detect_atoms(
                image=base_raw,
                mode=params["mode"],
                gaussian_sigma=params["gaussian_sigma"],
                min_distance=params["min_distance"],
                threshold_rel=params["threshold_rel"],
                max_atoms=params["max_atoms"],
                slice_index=params["slice_index"],
                overlay_path=overlay_temp.name,
            )
            # Restore file provenance when detect_atoms is fed with ndarray.
            result.provenance["source"] = source_info
            if load_warnings:
                result.warnings = list(load_warnings) + list(result.warnings)
            base_norm = _normalize_image(base_raw)

            # Step 2: subpixel refinement.
            refined_coords = result.coordinates.copy()
            refinement_info = None
            effective_refinement_method = params["refinement_method"]
            if (
                auto_run
                and params["fast_mode_realtime"]
                and effective_refinement_method == "gaussian"
                and len(result.coordinates) > params["fast_gaussian_max_points"]
            ):
                effective_refinement_method = "com"

            if effective_refinement_method != "none":
                refinement = refine_atom_coordinates(
                    image=base_raw,
                    coordinates=result.coordinates,
                    method=effective_refinement_method,
                    window_radius=params["refinement_window_radius"],
                    max_shift=params["refinement_max_shift"],
                )
                refined_coords = refinement.coordinates_refined
                warn = list(refinement.warnings)
                if effective_refinement_method != params["refinement_method"]:
                    warn = [
                        (
                            f"Fast mode switched refinement from {params['refinement_method']} "
                            f"to {effective_refinement_method} for speed "
                            f"(points={len(result.coordinates)})."
                        )
                    ] + warn
                refinement_info = {
                    "method": effective_refinement_method,
                    "confidence": refinement.confidence,
                    "warnings": warn,
                    "mean_displacement": float(
                        np.mean(np.linalg.norm(refinement.displacements, axis=1))
                    )
                    if len(refinement.displacements)
                    else 0.0,
                    "provenance": refinement.provenance,
                }

            # Step 3: optional lattice-constraint correction.
            constrained_coords = refined_coords
            lattice_info = None
            if params["lattice_constraint_enable"]:
                lattice_result = apply_lattice_constraint(
                    coordinates=refined_coords,
                    enabled=True,
                    k_neighbors=params["lattice_k_neighbors"],
                    strength=params["lattice_strength"],
                    iterations=params["lattice_iterations"],
                    max_shift_per_iter=params["lattice_max_shift_per_iter"],
                )
                constrained_coords = lattice_result.coordinates_regularized
                lattice_info = {
                    "enabled": lattice_result.enabled,
                    "confidence": lattice_result.confidence,
                    "warnings": lattice_result.warnings,
                    "mean_displacement": float(
                        np.mean(np.linalg.norm(lattice_result.displacements, axis=1))
                    )
                    if len(lattice_result.displacements)
                    else 0.0,
                    "provenance": lattice_result.provenance,
                }
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
        else:
            st.session_state.base_result = result
            st.session_state.base_image_raw = base_raw
            st.session_state.base_image_norm = base_norm
            st.session_state.edited_coords = constrained_coords.copy()
            st.session_state.edited_intensities = np.array(
                [_sample_intensity(base_raw, y, x) for y, x in constrained_coords],
                dtype=np.float64,
            )
            st.session_state.auto_coords = st.session_state.edited_coords.copy()
            st.session_state.auto_intensities = st.session_state.edited_intensities.copy()
            st.session_state.last_detection_signature = sig
            st.session_state.roi_mask = None
            st.session_state.refinement_info = refinement_info
            st.session_state.lattice_info = lattice_info
            _invalidate_atom_strain_state()
            st.session_state.last_run_summary = {
                "atom_count": int(len(constrained_coords)),
                "mode": params["mode"],
                "refinement": params["refinement_method"],
                "lattice_constraint": bool(params["lattice_constraint_enable"]),
            }

if st.session_state.base_result is None:
    st.warning("No detection result yet. Click Run Detection.")
    st.stop()

if st.session_state.last_run_summary is not None:
    st.caption(f"Last run: {st.session_state.last_run_summary}")

result = st.session_state.base_result
coords = np.asarray(st.session_state.edited_coords, dtype=np.float64)
intensities = np.asarray(st.session_state.edited_intensities, dtype=np.float64)
raw = np.asarray(st.session_state.base_image_raw, dtype=np.float64)
norm = np.asarray(st.session_state.base_image_norm, dtype=np.float64)
roi_mask = st.session_state.roi_mask

distance_stats = compute_distance_statistics(
    coords,
    pixel_size=params["pixel_size"] if params["use_physical_unit"] else None,
    output_unit=params["distance_unit"] if params["use_physical_unit"] else None,
)

st.subheader("Direct Image Interaction")
enable_canvas_interaction = st.checkbox(
    "Enable canvas editing / ROI drawing",
    value=False,
    help=(
        "Turn this on only when you need manual add/delete/ROI editing. "
        "Keeping it off avoids the drawable-canvas frontend remount issue."
    ),
)

reset_auto = st.button("Reset to Auto Detection", use_container_width=False)
if reset_auto:
    if st.session_state.auto_coords is not None and st.session_state.auto_intensities is not None:
        st.session_state.edited_coords = np.asarray(st.session_state.auto_coords).copy()
        st.session_state.edited_intensities = np.asarray(st.session_state.auto_intensities).copy()
    else:
        st.session_state.edited_coords = result.coordinates.copy()
        st.session_state.edited_intensities = result.intensities.copy()
    st.session_state.roi_mask = None
    _invalidate_atom_strain_state()
    coords = st.session_state.edited_coords
    intensities = st.session_state.edited_intensities
    roi_mask = None
    st.success("Reset to automatic result.")

if enable_canvas_interaction:
    tool = st.radio(
        "Tool",
        [
            "Add atom by click",
            "Delete atom by click",
            "Draw irregular ROI (right-click to close)",
        ],
        horizontal=True,
    )
    delete_radius = st.slider(
        "Delete radius (px)", min_value=0.5, max_value=20.0, value=3.0, step=0.5
    )
    st.caption(
        "Usage: click/draw on image. "
        "For ROI mode, `Keep/Remove` can auto-apply current ROI drawing. "
        "The same ROI can also be reused as the ideal-lattice reference area in atom strain."
    )

    roi_action_col1, roi_action_col2 = st.columns(2)
    keep_inside = roi_action_col1.button("Keep Atoms Inside ROI", use_container_width=True)
    remove_inside = roi_action_col2.button("Remove Atoms Inside ROI", use_container_width=True)

    canvas_image, sx, sy = _to_display_image(norm, coords=coords, roi_mask=roi_mask, max_side=900)
    draw_mode = "point" if "Draw irregular ROI" not in tool else "polygon"
    canvas_key = f"canvas-direct-interaction-{st.session_state.canvas_nonce}"
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.25)",
        stroke_width=2,
        stroke_color="#00ffaa",
        background_image=canvas_image,
        drawing_mode=draw_mode,
        point_display_radius=4,
        display_toolbar=True,
        update_streamlit=True,
        height=canvas_image.height,
        width=canvas_image.width,
        key=canvas_key,
    )

    act_col1, act_col2 = st.columns([1, 1])
    apply_action = act_col1.button("Apply Canvas Action", type="primary")
    clear_canvas = act_col2.button("Clear Canvas")

    if clear_canvas:
        st.session_state.canvas_nonce = int(st.session_state.canvas_nonce) + 1
        st.rerun()

    if apply_action:
        draw_mask_disp = _canvas_draw_mask(canvas_result)
        if draw_mask_disp is None or not np.any(draw_mask_disp):
            st.warning("No drawing detected on canvas.")
        else:
            if tool == "Add atom by click":
                disp_points = _mask_centroids(draw_mask_disp, min_pixels=2)
                raw_points = _map_display_to_raw(disp_points, sx=sx, sy=sy)
                if len(raw_points) == 0:
                    st.warning("No clicked points found.")
                else:
                    for y, x in raw_points:
                        coords = (
                            np.vstack([coords, np.array([[y, x]], dtype=np.float64)])
                            if len(coords)
                            else np.array([[y, x]])
                        )
                        intensities = np.append(intensities, _sample_intensity(raw, y, x))
                    st.session_state.edited_coords = coords
                    st.session_state.edited_intensities = intensities
                    _invalidate_atom_strain_state()
                    st.success(f"Added {len(raw_points)} points.")

            elif tool == "Delete atom by click":
                disp_points = _mask_centroids(draw_mask_disp, min_pixels=2)
                raw_points = _map_display_to_raw(disp_points, sx=sx, sy=sy)
                if len(raw_points) == 0:
                    st.warning("No clicked points found.")
                elif len(coords) == 0:
                    st.warning("No atom points to delete.")
                else:
                    removed = 0
                    for y, x in raw_points:
                        if len(coords) == 0:
                            break
                        d2 = (coords[:, 0] - y) ** 2 + (coords[:, 1] - x) ** 2
                        idx = int(np.argmin(d2))
                        d = float(np.sqrt(d2[idx]))
                        if d <= float(delete_radius):
                            coords = np.delete(coords, idx, axis=0)
                            intensities = np.delete(intensities, idx, axis=0)
                            removed += 1
                    st.session_state.edited_coords = coords
                    st.session_state.edited_intensities = intensities
                    _invalidate_atom_strain_state()
                    st.success(f"Removed {removed} points.")

            else:
                # Polygon drawing may return border + fill; fill holes to get stable ROI area.
                roi_disp = ndi.binary_fill_holes(draw_mask_disp)
                mask = _resize_mask_to_raw(roi_disp, raw.shape)
                st.session_state.roi_mask = mask
                _invalidate_atom_strain_state()
                roi_mask = mask
                st.success("ROI updated.")

            st.session_state.canvas_nonce = int(st.session_state.canvas_nonce) + 1
            st.rerun()

    if keep_inside or remove_inside:
        effective_roi_mask = roi_mask
        if effective_roi_mask is None and "Draw irregular ROI" in tool:
            draw_mask_disp = _canvas_draw_mask(canvas_result)
            if draw_mask_disp is not None and np.any(draw_mask_disp):
                roi_disp = ndi.binary_fill_holes(draw_mask_disp)
                effective_roi_mask = _resize_mask_to_raw(roi_disp, raw.shape)
                st.session_state.roi_mask = effective_roi_mask
                _invalidate_atom_strain_state()
                roi_mask = effective_roi_mask
                st.info("ROI auto-applied from current drawing.")

        if effective_roi_mask is None:
            st.warning("No ROI available. Draw ROI first.")
        elif len(coords) == 0:
            st.warning("No atom points to filter.")
        else:
            yi = np.clip(
                np.round(coords[:, 0]).astype(int), 0, effective_roi_mask.shape[0] - 1
            )
            xi = np.clip(
                np.round(coords[:, 1]).astype(int), 0, effective_roi_mask.shape[1] - 1
            )
            inside_mask = effective_roi_mask[yi, xi]
            if keep_inside:
                select_mask = inside_mask
                msg = "Kept ROI-inside atoms."
            else:
                select_mask = ~inside_mask
                msg = "Removed ROI-inside atoms."

            st.session_state.edited_coords = coords[select_mask]
            st.session_state.edited_intensities = intensities[select_mask]
            _invalidate_atom_strain_state()
            st.success(msg)
            st.rerun()
else:
    st.caption("Canvas editor is off. Existing atom coordinates and ROI remain unchanged.")

if roi_mask is not None:
    inside = 0
    if len(coords) > 0:
        yi = np.clip(np.round(coords[:, 0]).astype(int), 0, roi_mask.shape[0] - 1)
        xi = np.clip(np.round(coords[:, 1]).astype(int), 0, roi_mask.shape[1] - 1)
        inside = int(np.sum(roi_mask[yi, xi]))
    st.info(f"ROI pixels: {int(np.sum(roi_mask))} | Atoms inside ROI: {inside}")

prospective_reference_atom_mask = None
if params["atom_strain_use_roi_reference"]:
    prospective_reference_atom_mask = _coords_inside_roi_mask(coords, roi_mask)

atomic_displacement_reference_atom_mask = None
if params["atomic_displacement_method"] == DISPLACEMENT_METHOD_REFERENCE_ROI:
    atomic_displacement_reference_atom_mask = _coords_inside_roi_mask(coords, roi_mask)

left, right = st.columns([1.45, 1.0])
with left:
    st.subheader("Overlay")
    fig = _render_overlay(
        norm,
        coords,
        roi_mask=roi_mask,
        reference_atom_mask=prospective_reference_atom_mask,
    )
    st.pyplot(fig)

with right:
    st.subheader("Summary")
    st.metric("Auto detected atoms", len(result.coordinates))
    st.metric("Edited atoms", len(coords))
    st.metric("Confidence (auto)", f"{result.confidence:.3f}")
    refinement_info = st.session_state.refinement_info
    lattice_info = st.session_state.lattice_info
    if refinement_info is not None:
        st.write(
            "Refinement:",
            {
                "method": refinement_info["method"],
                "confidence": round(float(refinement_info["confidence"]), 3),
                "mean_displacement_px": round(float(refinement_info["mean_displacement"]), 4),
            },
        )
        if refinement_info["warnings"]:
            summary = _warning_summary_lines(refinement_info["warnings"])
            st.warning("\n".join(summary))
            with st.expander("Show full refinement warnings"):
                st.code("\n".join(refinement_info["warnings"]), language="text")
    else:
        st.write("Refinement:", {"method": "none"})

    if lattice_info is not None:
        st.write(
            "Lattice constraint:",
            {
                "enabled": bool(lattice_info["enabled"]),
                "confidence": round(float(lattice_info["confidence"]), 3),
                "mean_displacement_px": round(float(lattice_info["mean_displacement"]), 4),
            },
        )
        if lattice_info["warnings"]:
            summary = _warning_summary_lines(lattice_info["warnings"])
            st.warning("\n".join(summary))
            with st.expander("Show full lattice warnings"):
                st.code("\n".join(lattice_info["warnings"]), language="text")
    else:
        st.write("Lattice constraint:", {"enabled": False})

    st.write("Method:", result.method)
    st.write("Parameters:", result.parameters)
    st.write("Provenance:", result.provenance)
    st.write("Distance unit:", distance_stats.unit)
    st.write(
        "Nearest-neighbor summary:",
        {
            "count": int(distance_stats.nearest_summary["count"]),
            "mean": distance_stats.nearest_summary["mean"],
            "median": distance_stats.nearest_summary["median"],
            "std": distance_stats.nearest_summary["std"],
        },
    )
    st.write(
        "Next-nearest summary:",
        {
            "count": int(distance_stats.next_nearest_summary["count"]),
            "mean": distance_stats.next_nearest_summary["mean"],
            "median": distance_stats.next_nearest_summary["median"],
            "std": distance_stats.next_nearest_summary["std"],
        },
    )
    if distance_stats.warnings:
        summary = _warning_summary_lines(distance_stats.warnings)
        st.warning("\n".join(summary))
    if result.warnings:
        summary = _warning_summary_lines(result.warnings)
        st.warning("\n".join(summary))
        with st.expander("Show full detection warnings"):
            st.code("\n".join(result.warnings), language="text")
    else:
        st.success("No warnings.")
    if params["atom_strain_use_roi_reference"]:
        if roi_mask is None:
            st.info("Ideal lattice reference: ROI mode is ON. Draw/apply ROI to define it.")
        elif prospective_reference_atom_mask is None or not np.any(prospective_reference_atom_mask):
            st.warning(
                "Current ROI will be used as ideal lattice region, "
                "but no edited atoms are inside it yet."
            )
        else:
            st.success(
                "Current ROI will be used as ideal lattice region. "
                f"Reference atoms: {int(np.sum(prospective_reference_atom_mask))}"
            )

st.subheader("Atom-Position Strain (AI-STEM inspired)")
st.caption(
    "Alternative strain path from the current edited atom-column coordinates. "
    "This workflow is independent from external g-vector pipelines and is intended "
    "for C0/C1-based testing."
)
atom_strain_button_col, atom_strain_info_col = st.columns([0.34, 0.66])
run_atom_strain_button = atom_strain_button_col.button(
    "Run Atom-Position Strain",
    type="primary",
    use_container_width=True,
)
atom_strain_info_col.caption(
    "Reference source: "
    f"{params['atom_strain_reference_mode']} | "
    f"k={params['atom_strain_neighbor_k']} | "
    f"tol={params['atom_strain_tolerance_deg']:.1f} deg | "
    f"grid step={params['atom_strain_grid_step']}"
)
if params["atom_strain_use_roi_reference"]:
    if roi_mask is None:
        st.info("Current ROI will be used as ideal lattice region after you draw/apply ROI.")
    else:
        ref_count = (
            int(np.sum(prospective_reference_atom_mask))
            if prospective_reference_atom_mask is not None
            else 0
        )
        st.info(
            "Current ROI will be used as ideal lattice region. "
            f"Atoms participating in reference fit: {ref_count}"
        )

if run_atom_strain_button:
    if len(coords) < 6:
        st.warning("Need at least 6 edited atom points before running atom-position strain.")
    elif (
        params["atom_strain_reference_mode"] == "Current ROI mask"
        and roi_mask is None
    ):
        st.warning("Current ROI mask is not available yet. Draw/apply ROI first.")
    else:
        reference_mask = (
            roi_mask if params["atom_strain_reference_mode"] == "Current ROI mask" else None
        )
        with st.spinner("Running atom-position strain..."):
            try:
                fields = compute_atom_strain_fields(
                    coordinates=coords,
                    image_shape=raw.shape,
                    reference_mask=reference_mask,
                    neighbor_k=params["atom_strain_neighbor_k"],
                    direction_tolerance_deg=params["atom_strain_tolerance_deg"],
                    interpolation_grid_step=params["atom_strain_grid_step"],
                )
            except Exception as exc:
                st.error(f"Atom-position strain failed: {exc}")
            else:
                st.session_state.atom_strain_fields = fields
                st.session_state.atom_strain_meta = {
                    "reference_source": params["atom_strain_reference_mode"],
                    "neighbor_k": params["atom_strain_neighbor_k"],
                    "direction_tolerance_deg": params["atom_strain_tolerance_deg"],
                    "grid_step": params["atom_strain_grid_step"],
                }
                st.success("Atom-position strain finished.")

atom_strain_fields = st.session_state.atom_strain_fields
atom_strain_meta = st.session_state.atom_strain_meta
displacement_result: AtomicDisplacementResult | None = st.session_state.atomic_displacement_result
displacement_meta = st.session_state.atomic_displacement_meta
plane_spacing_result: AtomicPlaneSpacingResult | None = st.session_state.plane_spacing_result
plane_spacing_meta = st.session_state.plane_spacing_meta
atomic_intensity_result: AtomicColumnIntensityResult | None = (
    st.session_state.atomic_intensity_result
)
atomic_intensity_meta = st.session_state.atomic_intensity_meta
sublattice_result: SublatticeClassificationResult | None = st.session_state.sublattice_result
sublattice_meta = st.session_state.sublattice_meta
displacement_scale = params["pixel_size"] if params["use_physical_unit"] else 1.0
displacement_unit = params["distance_unit"] if params["use_physical_unit"] else "px"
strain_scale = 100.0 if params["atom_strain_display_unit"] == "%" else 1.0
strain_unit = "%" if params["atom_strain_display_unit"] == "%" else "fraction"
custom_components: dict[str, np.ndarray] | None = None
custom_basis: dict[str, Any] | None = None
custom_basis_ready = False

if atom_strain_fields is None:
    st.info("Run atom-position strain to visualize displacement/strain from edited atom columns.")
else:
    valid_count = int(np.sum(atom_strain_fields.valid_atom_mask))
    fit_stats = atom_strain_fields.provenance.get("fit_statistics", {})
    strain_left, strain_right = st.columns([1.55, 1.0])
    with strain_left:
        strain_fig = _render_atom_strain_maps(
            atom_strain_fields,
            displacement_scale=displacement_scale,
            displacement_unit=displacement_unit,
            strain_scale=strain_scale,
            strain_unit=strain_unit,
            auto_color_range=params["atom_strain_auto_color_range"],
            displacement_abs_limit=params["atom_strain_displacement_abs"],
            strain_abs_limit=params["atom_strain_strain_abs"],
            rotation_abs_limit=params["atom_strain_rotation_abs"],
        )
        st.pyplot(strain_fig)
    with strain_right:
        st.metric("Atom-strain confidence", f"{atom_strain_fields.confidence:.3f}")
        st.metric("Valid local fits", f"{valid_count}/{len(coords)}")
        if atom_strain_meta is not None:
            st.write("Run settings:", atom_strain_meta)
        st.write(
            "Reference region:",
            (
                atom_strain_fields.reference_region
                if atom_strain_fields.reference_region
                else "full set"
            ),
        )
        st.write(
            "Reference vectors (x, y):",
            np.round(atom_strain_fields.reference_vectors_xy, 4).tolist(),
        )
        st.write(
            "Fit statistics:",
            {
                "valid_atom_ratio": fit_stats.get("valid_atom_ratio"),
                "median_local_residual_px": fit_stats.get("median_local_residual_px"),
                "median_condition_number": fit_stats.get("median_condition_number"),
            },
        )
        atom_summary = {
            "mean_ux": float(np.mean(atom_strain_fields.atom_displacement_xy[:, 0]))
            * displacement_scale,
            "mean_uy": float(np.mean(atom_strain_fields.atom_displacement_xy[:, 1]))
            * displacement_scale,
            "mean_exx": float(np.mean(atom_strain_fields.atom_strain_xx)) * strain_scale,
            "mean_eyy": float(np.mean(atom_strain_fields.atom_strain_yy)) * strain_scale,
            "mean_exy": float(np.mean(atom_strain_fields.atom_strain_xy)) * strain_scale,
            "mean_rotation": float(np.mean(atom_strain_fields.atom_rotation_xy)),
            "strain_unit": strain_unit,
        }
        st.write("Atom summary:", atom_summary)
        if params["lattice_constraint_enable"]:
            st.warning(
                "Lattice constraint is ON. For defect-sensitive atom strain, "
                "OFF is often safer."
            )
        if atom_strain_fields.warnings:
            summary = _warning_summary_lines(atom_strain_fields.warnings)
            st.warning("\n".join(summary))
            with st.expander("Show full atom-strain warnings"):
                st.code("\n".join(atom_strain_fields.warnings), language="text")
        else:
            st.success("No atom-strain warnings.")

    overlay_metric = params["atom_strain_overlay_metric"]
    overlay_values_lookup = {
        "exx": atom_strain_fields.atom_strain_xx * strain_scale,
        "eyy": atom_strain_fields.atom_strain_yy * strain_scale,
        "exy": atom_strain_fields.atom_strain_xy * strain_scale,
    }
    overlay_values = np.asarray(overlay_values_lookup[overlay_metric], dtype=np.float64)
    if params["atom_strain_overlay_auto_range"]:
        overlay_vmin = -_symmetric_limit(overlay_values, fallback=1.0)
        overlay_vmax = _symmetric_limit(overlay_values, fallback=1.0)
    else:
        overlay_vmin = float(params["atom_strain_overlay_min"])
        overlay_vmax = float(params["atom_strain_overlay_max"])
        if overlay_vmax <= overlay_vmin:
            overlay_vmax = overlay_vmin + 1e-6
            st.warning("Overlay max must be larger than min; adjusted slightly for display.")

    st.subheader("Atom-Position Strain Overlay on Original Image")
    st.caption(
        "Jet-style colormap on the original image. "
        "Points with failed local fits are marked as gray x."
    )
    overlay_left, overlay_right = st.columns([1.45, 1.0])
    with overlay_left:
        overlay_fig = _render_atom_strain_overlay(
            norm,
            coords,
            overlay_values,
            title=f"{overlay_metric} overlay on original image [{strain_unit}]",
            colorbar_label=f"{overlay_metric} [{strain_unit}]",
            vmin=overlay_vmin,
            vmax=overlay_vmax,
            cmap="jet",
            marker_size=params["atom_strain_overlay_marker_size"],
            invalid_mask=~atom_strain_fields.valid_atom_mask,
        )
        st.pyplot(overlay_fig)
        overlay_png = _figure_png_bytes(overlay_fig)
        st.download_button(
            label="Download atom-position strain overlay PNG",
            data=overlay_png,
            file_name=(
                f"{Path(uploaded.name).stem}_atom_position_{overlay_metric}_overlay.png"
            ),
            mime="image/png",
        )
        plt.close(overlay_fig)
    with overlay_right:
        overlay_valid = overlay_values[np.asarray(atom_strain_fields.valid_atom_mask, dtype=bool)]
        st.write(
            "Overlay settings:",
            {
                "metric": overlay_metric,
                "unit": strain_unit,
                "range_min": overlay_vmin,
                "range_max": overlay_vmax,
                "marker_size": params["atom_strain_overlay_marker_size"],
                "colormap": "jet",
            },
        )
        st.write(
            "Valid-point value summary:",
            {
                "min": float(np.nanmin(overlay_valid[np.isfinite(overlay_valid)]))
                if np.any(np.isfinite(overlay_valid))
                else None,
                "max": float(np.nanmax(overlay_valid[np.isfinite(overlay_valid)]))
                if np.any(np.isfinite(overlay_valid))
                else None,
                "mean": float(np.nanmean(overlay_valid[np.isfinite(overlay_valid)]))
                if np.any(np.isfinite(overlay_valid))
                else None,
            },
        )

st.subheader("Custom a-b Basis Tool")
st.caption(
    "Independent basis picker shared by atom-based strain projection and atomic "
    "displacement method 1. You do not need to run atom-position strain before using it."
)
if not bool(st.session_state.custom_basis_enabled):
    if st.button("Enable a-b Basis Picking", use_container_width=False):
        st.session_state.custom_basis_enabled = True
        _invalidate_custom_basis_state()
        st.session_state.custom_basis_message = "Click the base atom for the custom a-b basis."
        st.rerun()
    st.info("Custom a-b basis tool is currently off.")
else:
    basis_button_col1, basis_button_col2 = st.columns([0.34, 0.34])
    if basis_button_col1.button("Start / Reset a-b Picking", use_container_width=True):
        _invalidate_custom_basis_state()
        st.session_state.custom_basis_message = "Click the base atom for the custom a-b basis."
        st.rerun()
    if basis_button_col2.button("Turn Off a-b Basis", use_container_width=True):
        _disable_custom_basis_state()
        st.rerun()
    stage_prompts = {
        0: "Click one atom in the interactive atom-point picker below to choose the base atom.",
        1: "Base selected. Click another atom in the picker to define the a vector.",
        2: "a-vector atom selected. Click one more atom in the picker to define the b vector.",
        3: "Custom a-b basis is ready.",
    }
    basis_stage = int(st.session_state.custom_basis_stage)
    basis_preview_col, basis_picker_col = st.columns([1.15, 1.0])
    with basis_picker_col:
        if st.session_state.custom_basis_message:
            st.info(st.session_state.custom_basis_message)
        else:
            st.caption(stage_prompts.get(basis_stage, ""))
        if basis_stage >= 3:
            st.success(
                "Custom a-b basis is ready. The interactive picker is unloaded "
                "to keep downstream Run buttons responsive."
            )
        else:
            basis_picker_mode = st.radio(
                "a/b picker mode",
                ["Fast image click (snap nearest atom)", "Atom-point picker (slower)"],
                index=0 if len(coords) > 2500 else 1,
                horizontal=True,
            )
            if basis_picker_mode == "Fast image click (snap nearest atom)":
                st.caption(
                    "Click the image; the click is snapped to the nearest detected atom. "
                    "No Apply button is needed."
                )
                if len(coords) > 6000:
                    st.caption(
                        "For speed, only a subset of red atom rings is drawn, but snapping "
                        "still uses all detected atoms."
                    )
                basis_canvas_img, basis_sx, basis_sy = _to_display_image(
                    norm,
                    coords,
                    roi_mask=None,
                    basis_markers=_current_basis_markers(coords),
                    max_side=760,
                    max_draw_atoms=6000,
                )
                basis_canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 0, 0.55)",
                    stroke_width=1,
                    stroke_color="#ffff00",
                    background_image=basis_canvas_img,
                    drawing_mode="point",
                    point_display_radius=4,
                    display_toolbar=False,
                    update_streamlit=True,
                    height=basis_canvas_img.height,
                    width=basis_canvas_img.width,
                    key=f"custom-basis-canvas-{st.session_state.custom_basis_picker_nonce}",
                )
                basis_click_disp = _canvas_object_points(basis_canvas_result)
                if len(basis_click_disp) > 0:
                    raw_points = _map_display_to_raw(
                        basis_click_disp[-1:],
                        sx=basis_sx,
                        sy=basis_sy,
                    )
                    nearest_idx = _nearest_atom_index(
                        coords,
                        float(raw_points[-1, 0]),
                        float(raw_points[-1, 1]),
                    )
                    if nearest_idx is not None:
                        st.session_state.custom_basis_last_picked_atom = int(nearest_idx)
                    changed, message = _apply_custom_basis_clicks(coords, raw_points)
                    if message is not None:
                        st.session_state.custom_basis_message = message
                    if changed:
                        st.session_state.custom_basis_picker_nonce = int(
                            st.session_state.custom_basis_picker_nonce
                        ) + 1
                        st.rerun()
            else:
                render_dense_picker = True
                if len(coords) > 5000:
                    st.warning(
                        "Atom-point picker renders every atom as an interactive point and "
                        "can freeze the browser for large maps. Fast image click is safer."
                    )
                    render_dense_picker = st.checkbox(
                        "Render dense atom-point picker anyway",
                        value=False,
                    )
                if render_dense_picker:
                    basis_picker_chart = _render_custom_basis_picker_chart(
                        norm,
                        coords,
                        dict(st.session_state.custom_basis_indices),
                        max_side=900,
                    )
                    st.caption(
                        "Atom-point picker only. Click one point here to choose base, "
                        "then a, then b."
                    )
                    st.caption(
                        "No Apply button is needed here. One single click on the "
                        "right-side point is enough."
                    )
                    basis_event = st.altair_chart(
                        basis_picker_chart,
                        key=f"custom-basis-picker-{st.session_state.custom_basis_picker_nonce}",
                        on_select="rerun",
                        selection_mode="basis_pick",
                        use_container_width=False,
                    )
                    picked_atom = _extract_altair_atom_pick(
                        basis_event,
                        selection_name="basis_pick",
                    )
                    if picked_atom is not None and 0 <= int(picked_atom) < len(coords):
                        st.session_state.custom_basis_last_picked_atom = int(picked_atom)
                        raw_point = np.asarray([coords[int(picked_atom)]], dtype=np.float64)
                        _, message = _apply_custom_basis_clicks(coords, raw_point)
                        if message is not None:
                            st.session_state.custom_basis_message = message
        picks = dict(st.session_state.custom_basis_indices)
        st.write(
            "Current picks:",
            {
                "base_idx": picks.get("base"),
                "a_idx": picks.get("a"),
                "b_idx": picks.get("b"),
                "last_clicked_atom": st.session_state.custom_basis_last_picked_atom,
                "stage": ("pick base", "pick a", "pick b", "ready")[
                    min(int(st.session_state.custom_basis_stage), 3)
                ],
            },
        )
    basis_preview_img, _, _ = _to_display_image(
        norm,
        coords,
        roi_mask=None,
        basis_markers=_current_basis_markers(coords),
        max_side=900,
        max_draw_atoms=6000,
    )
    with basis_preview_col:
        st.caption("Original image with detected atoms and current base / a / b markers.")
        st.image(basis_preview_img, use_container_width=False)

custom_basis = _custom_basis_info(coords, dict(st.session_state.custom_basis_indices))
if custom_basis is not None:
    st.write(
        "Custom basis atoms:",
        {
            "base_idx": custom_basis["base_idx"],
            "a_idx": custom_basis["a_idx"],
            "b_idx": custom_basis["b_idx"],
            "a_length_px": round(float(custom_basis["a_length_px"]), 4),
            "b_length_px": round(float(custom_basis["b_length_px"]), 4),
            "angle_deg": round(float(custom_basis["angle_deg"]), 3),
        },
    )
    if not (5.0 < float(custom_basis["angle_deg"]) < 175.0):
        st.warning("Custom a and b vectors are nearly collinear. Please re-pick the basis.")
    else:
        custom_basis_ready = True
        if atom_strain_fields is not None:
            st.subheader("Custom a-b Basis Strain Projection")
            st.caption(
                "Reproject the current atom-position strain result into your custom a-b basis."
            )
            eaa_atom, ebb_atom, eab_atom = _project_strain_to_custom_basis(
                atom_strain_fields.atom_strain_xx,
                atom_strain_fields.atom_strain_yy,
                atom_strain_fields.atom_strain_xy,
                a_vec_yx=np.asarray(custom_basis["a_vec_yx"], dtype=np.float64),
                b_vec_yx=np.asarray(custom_basis["b_vec_yx"], dtype=np.float64),
            )
            eaa_map, ebb_map, eab_map = _project_strain_to_custom_basis(
                atom_strain_fields.strain_xx_map,
                atom_strain_fields.strain_yy_map,
                atom_strain_fields.strain_xy_map,
                a_vec_yx=np.asarray(custom_basis["a_vec_yx"], dtype=np.float64),
                b_vec_yx=np.asarray(custom_basis["b_vec_yx"], dtype=np.float64),
            )
            custom_components = {
                "eaa": np.asarray(eaa_atom, dtype=np.float64),
                "ebb": np.asarray(ebb_atom, dtype=np.float64),
                "eab": np.asarray(eab_atom, dtype=np.float64),
            }
            custom_fig = _render_custom_basis_strain_maps(
                eaa_map * strain_scale,
                ebb_map * strain_scale,
                eab_map * strain_scale,
                strain_unit=strain_unit,
                auto_color_range=params["atom_strain_auto_color_range"],
                strain_abs_limit=params["atom_strain_strain_abs"],
            )
            st.pyplot(custom_fig)
            plt.close(custom_fig)
        else:
            st.info(
                "Custom a-b basis is ready. You can now run atomic displacement method 1 below. "
                "If you later run atom-position strain, this same basis will also support "
                "eaa / ebb / eab."
            )

st.subheader("Sublattice Classification / Projected Structure")
st.caption(
    "First C4 module. It classifies edited atom columns into sublattice labels using "
    "the current custom a/b unit cell, then lets you manually correct wrong labels."
)
projected_model_text = ""
if params["sublattice_method"] == "User 2D projected model":
    projected_model_text = st.text_area(
        "Projected 2D model sites (`label,u,v` in the custom a/b unit cell)",
        value="label,u,v\nA,0,0\nB,0.5,0.5",
        height=120,
        help=(
            "u and v are fractional positions inside the custom a/b unit cell. "
            "Example labels can be A/B/O or material-specific names."
        ),
    )

manual_sublattice_labels = {
    int(atom_idx): str(label)
    for atom_idx, label in dict(st.session_state.sublattice_manual_labels).items()
    if 0 <= int(atom_idx) < len(coords)
}
if len(manual_sublattice_labels) != len(st.session_state.sublattice_manual_labels):
    st.session_state.sublattice_manual_labels = manual_sublattice_labels

sublattice_intensity_values = np.asarray(intensities, dtype=np.float64)
sublattice_intensity_note = "current atom peak intensity"
if (
    params["sublattice_use_intensity"]
    and params["sublattice_intensity_source"] == "Atomic-column normalized intensity"
):
    intensity_result_matches = (
        atomic_intensity_result is not None
        and len(atomic_intensity_result.normalized_intensity) == len(coords)
    )
    if intensity_result_matches:
        sublattice_intensity_values = np.asarray(
            atomic_intensity_result.normalized_intensity,
            dtype=np.float64,
        )
        sublattice_intensity_note = "atomic-column normalized intensity"
    else:
        st.warning(
            "Atomic-column normalized intensity is not available for the current atoms. "
            "Sublattice classification will fall back to current atom peak intensity."
        )

sublattice_ready = custom_basis_ready and custom_basis is not None and len(coords) >= 3
sublattice_settings = {
    "method": params["sublattice_method"],
    "atom_count": int(len(coords)),
    "basis_indices": {
        "base_idx": int(custom_basis["base_idx"]) if custom_basis is not None else None,
        "a_idx": int(custom_basis["a_idx"]) if custom_basis is not None else None,
        "b_idx": int(custom_basis["b_idx"]) if custom_basis is not None else None,
    },
    "n_labels": int(params["sublattice_n_labels"]),
    "use_intensity": bool(params["sublattice_use_intensity"]),
    "intensity_source": sublattice_intensity_note,
    "intensity_weight": float(params["sublattice_intensity_weight"]),
    "model_tolerance": float(params["sublattice_model_tolerance"]),
    "projected_model_text": projected_model_text.strip()
    if params["sublattice_method"] == "User 2D projected model"
    else "",
}
if (
    sublattice_meta is not None
    and sublattice_meta.get("settings") != sublattice_settings
):
    _invalidate_sublattice_state(clear_manual_labels=False)
    sublattice_result = None
    sublattice_meta = None

sublattice_button_col, sublattice_info_col = st.columns([0.34, 0.66])
run_sublattice = sublattice_button_col.button(
    "Run Sublattice Classification",
    type="primary",
    use_container_width=False,
    disabled=not sublattice_ready,
)
sublattice_info_col.caption(
    f"method={params['sublattice_method']} | "
    f"basis ready={bool(custom_basis_ready)} | "
    f"intensity={sublattice_intensity_note} | "
    f"manual labels={len(manual_sublattice_labels)}"
)
if not custom_basis_ready:
    st.info("Define the custom base/a/b basis first, then run sublattice classification.")
elif len(coords) < 3:
    st.warning("Need at least 3 atom coordinates before classifying sublattices.")

if run_sublattice and sublattice_ready and custom_basis is not None:
    with st.spinner("Running sublattice classification..."):
        try:
            if params["sublattice_method"] == "Auto exploration":
                computed_sublattice = classify_sublattices_auto(
                    coords,
                    base_index=int(custom_basis["base_idx"]),
                    a_index=int(custom_basis["a_idx"]),
                    b_index=int(custom_basis["b_idx"]),
                    intensities=sublattice_intensity_values,
                    n_sublattices=int(params["sublattice_n_labels"]),
                    use_intensity=bool(params["sublattice_use_intensity"]),
                    intensity_weight=float(params["sublattice_intensity_weight"]),
                    manual_labels=manual_sublattice_labels,
                )
            else:
                model_sites = parse_projected_model_csv(projected_model_text)
                computed_sublattice = classify_sublattices_from_model(
                    coords,
                    base_index=int(custom_basis["base_idx"]),
                    a_index=int(custom_basis["a_idx"]),
                    b_index=int(custom_basis["b_idx"]),
                    basis_sites=model_sites,
                    intensities=sublattice_intensity_values,
                    match_tolerance=float(params["sublattice_model_tolerance"]),
                    manual_labels=manual_sublattice_labels,
                )
        except Exception as exc:
            _invalidate_sublattice_state(clear_manual_labels=False)
            st.error(f"Sublattice classification failed: {exc}")
            sublattice_result = None
            sublattice_meta = None
        else:
            st.session_state.sublattice_result = computed_sublattice
            st.session_state.sublattice_meta = {
                "settings": sublattice_settings,
                "method": computed_sublattice.provenance.get("method"),
                "manual_label_count": len(manual_sublattice_labels),
            }
            sublattice_result = computed_sublattice
            sublattice_meta = st.session_state.sublattice_meta
            st.success("Sublattice classification finished.")

if sublattice_meta is not None:
    st.write("Sublattice settings:", sublattice_meta)

if sublattice_result is not None:
    if len(sublattice_result.labels) != len(coords):
        st.warning(
            "Current sublattice result does not match the edited atom table. "
            "Run sublattice classification again."
        )
    else:
        sublattice_view = st.radio(
            "Sublattice view",
            ["Overlay", "Unit-cell model", "Manual correction", "Summary / Table"],
            horizontal=True,
            key="sublattice_view",
        )
        st.caption(
            "Only the selected sublattice view is rendered for speed. "
            "This avoids loading the large manual-correction point picker unless needed."
        )
        if sublattice_view == "Overlay":
            sub_overlay_fig = _render_sublattice_overlay(
                norm,
                coords,
                sublattice_result,
                marker_size=params["sublattice_marker_size"],
                show_label_text=params["sublattice_show_label_text"],
                show_atom_index=params["sublattice_show_atom_index"],
            )
            st.pyplot(sub_overlay_fig)
            sub_overlay_png = _figure_png_bytes(sub_overlay_fig)
            st.download_button(
                label="Download sublattice overlay PNG",
                data=sub_overlay_png,
                file_name=f"{Path(uploaded.name).stem}_sublattice_overlay.png",
                mime="image/png",
            )
            plt.close(sub_overlay_fig)

        elif sublattice_view == "Unit-cell model":
            sub_cell_fig = _render_sublattice_unit_cell(sublattice_result)
            st.pyplot(sub_cell_fig)
            sub_cell_png = _figure_png_bytes(sub_cell_fig)
            st.download_button(
                label="Download projected unit-cell PNG",
                data=sub_cell_png,
                file_name=f"{Path(uploaded.name).stem}_sublattice_unit_cell.png",
                mime="image/png",
            )
            plt.close(sub_cell_fig)
            st.json(sublattice_result.projected_structure)

        elif sublattice_view == "Manual correction":
            st.caption(
                "Static index map only. Read atom indices from the image, type index "
                "values below, then assign them to A/B/C or another label."
            )
            index_map_img = _render_sublattice_index_reference_image(
                norm,
                coords,
                sublattice_result,
                max_side=1400,
            )
            st.image(
                index_map_img,
                caption=(
                    "Atom index reference map. Color = current sublattice label; "
                    "yellow ring = manual override."
                ),
                use_container_width=False,
            )
            st.download_button(
                label="Download atom-index reference PNG",
                data=_pil_png_bytes(index_map_img),
                file_name=f"{Path(uploaded.name).stem}_sublattice_atom_indices.png",
                mime="image/png",
            )

            index_text = st.text_input(
                "Atom index / indices to correct",
                value="",
                placeholder="Example: 12 or 12,18,20-25",
                help="Use comma/space separated indices. Ranges like 20-25 are supported.",
            )
            selected_indices: list[int] = []
            index_parse_error = ""
            if index_text.strip():
                try:
                    selected_indices = _parse_atom_index_text(index_text, len(coords))
                except Exception as exc:
                    index_parse_error = str(exc)
                    st.warning(f"Invalid atom index input: {index_parse_error}")
            label_options = sorted(
                str(v)
                for v in set(np.asarray(sublattice_result.labels, dtype=object).tolist())
                | set(manual_sublattice_labels.values())
            )
            if "unknown" not in label_options:
                label_options.append("unknown")
            label_options.append("custom label")
            selected_target_label = st.selectbox(
                "Correct selected atom to label",
                label_options,
                index=0,
            )
            custom_target_label = ""
            if selected_target_label == "custom label":
                custom_target_label = st.text_input(
                    "Custom label",
                    value="",
                )
            final_target_label = (
                custom_target_label.strip()
                if selected_target_label == "custom label"
                else selected_target_label
            )
            selected_preview = []
            if selected_indices:
                selected_preview = [
                    {
                        "atom_index": idx,
                        "current_label": str(sublattice_result.labels[idx]),
                        "target_label": final_target_label,
                    }
                    for idx in selected_indices[:50]
                ]
            st.write(
                "Current manual correction:",
                {
                    "selected_atoms": selected_indices[:20],
                    "selected_count": len(selected_indices),
                    "target_label": final_target_label,
                    "manual_override_count": len(manual_sublattice_labels),
                },
            )
            if selected_preview:
                st.dataframe(pd.DataFrame(selected_preview), use_container_width=True)
                if len(selected_indices) > len(selected_preview):
                    st.caption(
                        f"Showing first {len(selected_preview)} of "
                        f"{len(selected_indices)} selected atoms."
                    )
            sub_manual_col1, sub_manual_col2 = st.columns([1, 1])
            apply_sub_label = sub_manual_col1.button(
                "Apply manual label",
                type="primary",
                disabled=not selected_indices
                or not str(final_target_label).strip(),
            )
            clear_sub_labels = sub_manual_col2.button(
                "Clear manual labels",
                disabled=len(manual_sublattice_labels) == 0,
            )
            if apply_sub_label and selected_indices:
                for atom_idx in selected_indices:
                    manual_sublattice_labels[int(atom_idx)] = str(final_target_label).strip()
                st.session_state.sublattice_manual_labels = manual_sublattice_labels
                corrected_sublattice = apply_manual_sublattice_labels(
                    sublattice_result,
                    manual_sublattice_labels,
                )
                st.session_state.sublattice_result = corrected_sublattice
                if st.session_state.sublattice_meta is not None:
                    st.session_state.sublattice_meta["manual_label_count"] = len(
                        manual_sublattice_labels
                    )
                st.success(
                    f"Relabeled {len(selected_indices)} atom(s) to "
                    f"{str(final_target_label).strip()}."
                )
                st.rerun()
            if clear_sub_labels:
                _invalidate_sublattice_state(clear_manual_labels=True)
                st.success("Manual sublattice labels cleared. Run classification again.")
                st.rerun()
            if manual_sublattice_labels:
                manual_preview = pd.DataFrame(
                    [
                        {"atom_index": atom_idx, "manual_label": label}
                        for atom_idx, label in sorted(manual_sublattice_labels.items())
                    ]
                )
                st.dataframe(manual_preview, use_container_width=True)

        else:
            st.metric("Sublattice confidence", f"{sublattice_result.confidence:.3f}")
            summary_df = pd.DataFrame(sublattice_result.label_summaries)
            st.dataframe(summary_df, use_container_width=True)
            if sublattice_result.warnings:
                summary = _warning_summary_lines(sublattice_result.warnings)
                st.warning("\n".join(summary))
                with st.expander("Show full sublattice warnings"):
                    st.code("\n".join(sublattice_result.warnings), language="text")
            else:
                st.success("No sublattice warnings.")

            sub_preview_count = min(len(coords), 500)
            if sub_preview_count > 0:
                sub_preview = pd.DataFrame(
                    {
                        "atom_index": np.arange(sub_preview_count, dtype=int),
                        "y": coords[:sub_preview_count, 0],
                        "x": coords[:sub_preview_count, 1],
                        "label": sublattice_result.labels[:sub_preview_count],
                        "lattice_i": sublattice_result.lattice_indices_ij[
                            :sub_preview_count,
                            0,
                        ],
                        "lattice_j": sublattice_result.lattice_indices_ij[
                            :sub_preview_count,
                            1,
                        ],
                        "cell_u": sublattice_result.unit_cell_uv[:sub_preview_count, 0],
                        "cell_v": sublattice_result.unit_cell_uv[:sub_preview_count, 1],
                        "intensity": sublattice_result.intensities[:sub_preview_count],
                        "manual_override": sublattice_result.manual_override_mask[
                            :sub_preview_count
                        ].astype(np.int32),
                    }
                )
                st.dataframe(sub_preview, use_container_width=True)
            st.download_button(
                label="Download sublattice atom table CSV",
                data=_sublattice_csv(coords, sublattice_result).encode("utf-8"),
                file_name=f"{Path(uploaded.name).stem}_sublattice_atoms.csv",
                mime="text/csv",
            )
            st.download_button(
                label="Download sublattice summary CSV",
                data=_sublattice_summary_csv(sublattice_result).encode("utf-8"),
                file_name=f"{Path(uploaded.name).stem}_sublattice_summary.csv",
                mime="text/csv",
            )
            structure_json = json.dumps(
                sublattice_result.projected_structure,
                ensure_ascii=False,
                indent=2,
            )
            st.download_button(
                label="Download projected structure JSON",
                data=structure_json.encode("utf-8"),
                file_name=f"{Path(uploaded.name).stem}_projected_structure.json",
                mime="application/json",
            )
elif sublattice_ready:
    st.info(
        "Sublattice inputs are ready. Click `Run Sublattice Classification` to "
        "color atoms by sublattice and start manual correction."
    )

st.subheader("Atomic Displacement")
st.caption(
    "Standalone module for atom displacement. Method 1 uses the custom base/a/b picks. "
    "Method 2 still uses your custom a/b direction, but fits the ideal lattice from "
    "a regular reference ROI."
)
displacement_method = params["atomic_displacement_method"]
displacement_uses_reference_roi = displacement_method == DISPLACEMENT_METHOD_REFERENCE_ROI
displacement_method_slug = (
    "ideal_lattice_from_reference_roi_with_custom_ab_direction"
    if displacement_uses_reference_roi
    else "ideal_lattice_from_manual_ab_basis"
)
if displacement_meta is not None and displacement_meta.get("method") != displacement_method_slug:
    _invalidate_atomic_displacement_state()
    displacement_result = None
    displacement_meta = None
reference_atom_mask_for_fit = (
    atomic_displacement_reference_atom_mask if displacement_uses_reference_roi else None
)
reference_atom_count = (
    int(np.sum(reference_atom_mask_for_fit))
    if reference_atom_mask_for_fit is not None
    else 0
)
displacement_ready = custom_basis_ready and (
    not displacement_uses_reference_roi or reference_atom_count >= 3
)
st.write("Method:", displacement_method)
disp_button_col, disp_info_col = st.columns([0.34, 0.66])
run_atomic_displacement = disp_button_col.button(
    "Run Atomic Displacement",
    type="primary",
    use_container_width=False,
    disabled=not displacement_ready,
)
manual_displacement_pairs = {
    int(atom_idx): (int(pair[0]), int(pair[1]))
    for atom_idx, pair in dict(st.session_state.atomic_displacement_manual_pairs).items()
}
if custom_basis_ready and displacement_uses_reference_roi:
    disp_info_col.caption(
        "Reference source: current ROI fit | "
        f"ROI atoms={reference_atom_count} | "
        f"lattice index tolerance={params['atomic_displacement_tolerance']:.2f} | "
        f"manual pairs={len(manual_displacement_pairs)}"
    )
    if roi_mask is None:
        st.info(
            "Method 2 needs a reference ROI. Enable canvas editing / ROI drawing above, "
            "draw a regular lattice region, then run displacement."
        )
    elif reference_atom_count < 3:
        st.warning(
            "Method 2 needs at least 3 atoms inside the current ROI. "
            "Select a larger regular lattice area before running displacement."
        )
    else:
        st.success(
            "Method 2 will fit origin, a_vec, and b_vec from the current ROI while "
            "using your custom a/b direction for lattice indexing."
        )
elif custom_basis_ready:
    disp_info_col.caption(
        "Reference source: all valid atoms from the custom base/a/b lattice | "
        f"lattice index tolerance={params['atomic_displacement_tolerance']:.2f} | "
        f"manual pairs={len(manual_displacement_pairs)}"
    )
elif bool(st.session_state.custom_basis_enabled):
    disp_info_col.caption(
        "Finish the base-a-b picking above first, then run atomic displacement."
    )
else:
    disp_info_col.caption(
        "Enable the custom a-b basis tool first. This method will then call that basis directly."
    )

if run_atomic_displacement and displacement_ready and custom_basis is not None:
    try:
        displacement_result = compute_atomic_displacement_from_basis(
            coords,
            base_index=int(custom_basis["base_idx"]),
            a_index=int(custom_basis["a_idx"]),
            b_index=int(custom_basis["b_idx"]),
            reference_mask=reference_atom_mask_for_fit,
            lattice_index_tolerance=float(params["atomic_displacement_tolerance"]),
            manual_lattice_indices=manual_displacement_pairs or None,
            allow_reference_fallback=not displacement_uses_reference_roi,
        )
    except Exception as exc:
        _invalidate_atomic_displacement_state()
        st.error(f"Atomic displacement fit failed: {exc}")
        displacement_result = None
        displacement_meta = None
    else:
        st.session_state.atomic_displacement_result = displacement_result
        st.session_state.atomic_displacement_meta = {
            "lattice_index_tolerance": float(params["atomic_displacement_tolerance"]),
            "basis_indices": {
                "base_idx": int(custom_basis["base_idx"]),
                "a_idx": int(custom_basis["a_idx"]),
                "b_idx": int(custom_basis["b_idx"]),
            },
            "reference_source": (
                "Current ROI mask"
                if displacement_uses_reference_roi
                else "All valid assigned atoms"
            ),
            "reference_roi_atom_count": reference_atom_count,
            "method": displacement_method_slug,
            "manual_lattice_pair_count": len(manual_displacement_pairs),
        }
        _invalidate_plane_spacing_state()
        displacement_meta = st.session_state.atomic_displacement_meta
        st.success("Atomic displacement finished.")

if displacement_meta is not None:
    st.write("Atomic displacement settings:", displacement_meta)

if displacement_result is not None:
    vector_tab, component_tab, qa_tab, debug_tab = st.tabs(
        ["Vector map", "Component maps", "Invalid QA", "Debug"]
    )

    with vector_tab:
        vector_fig = _render_atomic_displacement_vector_map(
            norm,
            displacement_result,
            displacement_scale=displacement_scale,
            displacement_unit=displacement_unit,
            show_image=params["atomic_displacement_vector_show_image"],
            marker_size=params["atomic_displacement_marker_size"],
            arrow_display_scale=params["atomic_displacement_arrow_scale"],
            max_arrows=params["atomic_displacement_max_arrows"],
        )
        st.pyplot(vector_fig)
        vector_png = _figure_png_bytes(vector_fig)
        st.download_button(
            label="Download displacement vector map PNG",
            data=vector_png,
            file_name=f"{Path(uploaded.name).stem}_atomic_displacement_vector_map.png",
            mime="image/png",
        )
        plt.close(vector_fig)

    with component_tab:
        component_fig = _render_atomic_displacement_component_maps(
            norm,
            displacement_result,
            displacement_scale=displacement_scale,
            displacement_unit=displacement_unit,
            marker_size=params["atomic_displacement_marker_size"],
        )
        st.pyplot(component_fig)
        component_png = _figure_png_bytes(component_fig)
        st.download_button(
            label="Download displacement component maps PNG",
            data=component_png,
            file_name=f"{Path(uploaded.name).stem}_atomic_displacement_components.png",
            mime="image/png",
        )
        plt.close(component_fig)

    with qa_tab:
        qa_left, qa_right = st.columns([1.35, 1.0])
        with qa_left:
            qa_fig = _render_atomic_displacement_invalid_map(
                norm,
                displacement_result,
                marker_size=params["atomic_displacement_marker_size"],
            )
            st.pyplot(qa_fig)
            qa_png = _figure_png_bytes(qa_fig)
            st.download_button(
                label="Download invalid-assignment QA PNG",
                data=qa_png,
                file_name=f"{Path(uploaded.name).stem}_atomic_displacement_invalid_qa.png",
                mime="image/png",
            )
            plt.close(qa_fig)
            enable_manual_adjustment = st.checkbox(
                "Enable manual ideal-lattice adjustment",
                value=False,
                help=(
                    "Use this when invalid atoms need manual observed-atom to ideal-site "
                    "matching before recalculating displacement."
                ),
            )
            if enable_manual_adjustment:
                manual_pairs = {
                    int(atom_idx): (int(pair[0]), int(pair[1]))
                    for atom_idx, pair in dict(
                        st.session_state.atomic_displacement_manual_pairs
                    ).items()
                }
                manual_pick_mode = st.radio(
                    "Manual pick mode",
                    ["Observed atom", "Ideal lattice site"],
                    horizontal=True,
                )
                manual_overlay_chart = _render_manual_pair_overlay_chart(
                    norm,
                    displacement_result,
                    manual_pairs,
                    pick_mode=manual_pick_mode,
                )
                manual_event = st.altair_chart(
                    manual_overlay_chart,
                    key=f"manual-pair-overlay-{manual_pick_mode}",
                    on_select="rerun",
                    selection_mode=(
                        "manual_atom_pick"
                        if manual_pick_mode == "Observed atom"
                        else "manual_site_pick"
                    ),
                    use_container_width=False,
                )
                if manual_pick_mode == "Observed atom":
                    observed_pick = _extract_altair_atom_pick(
                        manual_event,
                        selection_name="manual_atom_pick",
                    )
                    if observed_pick is not None:
                        st.session_state.atomic_displacement_manual_atom = int(
                            observed_pick
                        )
                else:
                    ideal_pick = _extract_altair_site_pick(
                        manual_event,
                        selection_name="manual_site_pick",
                    )
                    if ideal_pick is not None:
                        st.session_state.atomic_displacement_manual_site = ideal_pick

                selected_atom = st.session_state.atomic_displacement_manual_atom
                selected_site = st.session_state.atomic_displacement_manual_site
                st.write(
                    "Manual selection:",
                    {
                        "observed_atom": selected_atom,
                        "ideal_site": selected_site,
                        "manual_pair_count": len(manual_pairs),
                    },
                )
                pair_col1, pair_col2, pair_col3 = st.columns([1, 1, 1])
                add_pair = pair_col1.button(
                    "Add manual pair",
                    disabled=selected_atom is None or selected_site is None,
                )
                clear_pairs = pair_col2.button(
                    "Clear manual pairs",
                    disabled=len(manual_pairs) == 0,
                )
                rerun_manual = pair_col3.button(
                    "Re-run with manual pairs",
                    type="primary",
                    disabled=len(manual_pairs) == 0
                    or custom_basis is None
                    or (displacement_uses_reference_roi and reference_atom_count < 3),
                )
                if add_pair and selected_atom is not None and selected_site is not None:
                    manual_pairs[int(selected_atom)] = (
                        int(selected_site[0]),
                        int(selected_site[1]),
                    )
                    st.session_state.atomic_displacement_manual_pairs = manual_pairs
                    st.success(
                        f"Added atom {int(selected_atom)} -> site "
                        f"({int(selected_site[0])}, {int(selected_site[1])})."
                    )
                    st.rerun()
                if clear_pairs:
                    st.session_state.atomic_displacement_manual_pairs = {}
                    st.session_state.atomic_displacement_manual_atom = None
                    st.session_state.atomic_displacement_manual_site = None
                    st.success("Manual lattice pairs cleared.")
                    st.rerun()
                if manual_pairs:
                    pair_preview = pd.DataFrame(
                        [
                            {"atom_index": atom_idx, "lattice_i": ij[0], "lattice_j": ij[1]}
                            for atom_idx, ij in sorted(manual_pairs.items())
                        ]
                    )
                    st.dataframe(pair_preview, use_container_width=True)
                if rerun_manual and custom_basis is not None:
                    try:
                        corrected = compute_atomic_displacement_from_basis(
                            coords,
                            base_index=int(custom_basis["base_idx"]),
                            a_index=int(custom_basis["a_idx"]),
                            b_index=int(custom_basis["b_idx"]),
                            reference_mask=reference_atom_mask_for_fit,
                            lattice_index_tolerance=float(
                                params["atomic_displacement_tolerance"]
                            ),
                            manual_lattice_indices=manual_pairs,
                            allow_reference_fallback=not displacement_uses_reference_roi,
                        )
                    except Exception as exc:
                        st.error(f"Manual displacement correction failed: {exc}")
                    else:
                        st.session_state.atomic_displacement_result = corrected
                        st.session_state.atomic_displacement_meta = {
                            "lattice_index_tolerance": float(
                                params["atomic_displacement_tolerance"]
                            ),
                            "basis_indices": {
                                "base_idx": int(custom_basis["base_idx"]),
                                "a_idx": int(custom_basis["a_idx"]),
                                "b_idx": int(custom_basis["b_idx"]),
                            },
                            "reference_source": (
                                "Current ROI mask"
                                if displacement_uses_reference_roi
                                else "All valid assigned atoms"
                            ),
                            "reference_roi_atom_count": reference_atom_count,
                            "method": displacement_method_slug,
                            "manual_lattice_pair_count": len(manual_pairs),
                        }
                        _invalidate_plane_spacing_state()
                        st.success("Manual displacement correction finished.")
                        st.rerun()
        with qa_right:
            fit_stats = displacement_result.provenance.get("fit_statistics", {})
            disp_valid = np.asarray(displacement_result.valid_atom_mask, dtype=bool)
            invalid_mask = ~disp_valid
            invalid_count = int(np.sum(invalid_mask))
            invalid_ratio = float(invalid_count / max(len(coords), 1))
            invalid_residual = displacement_result.assignment_residual[invalid_mask]
            valid_mag = displacement_result.displacement_magnitude[disp_valid]
            st.write(
                "Atomic displacement summary:",
                {
                    "confidence": round(float(displacement_result.confidence), 3),
                    "mean_|u|": float(np.mean(valid_mag) * displacement_scale)
                    if valid_mag.size
                    else None,
                    "max_|u|": float(np.max(valid_mag) * displacement_scale)
                    if valid_mag.size
                    else None,
                    "invalid_assignment_count": invalid_count,
                    "invalid_assignment_ratio": invalid_ratio,
                    "valid_atom_ratio": fit_stats.get("valid_ratio"),
                    "fit_rms_px": fit_stats.get("fit_rms_px"),
                    "fit_atom_count": fit_stats.get("fit_atom_count"),
                },
            )
            st.info(
                "Invalid assignment means an atom is too far from the nearest integer "
                "site in the current base-a-b lattice coordinates."
            )
            st.write(
                "Invalid-assignment diagnostics:",
                {
                    "lattice_index_tolerance": float(
                        params["atomic_displacement_tolerance"]
                    ),
                    "assignment_residual_median": fit_stats.get(
                        "assignment_residual_median"
                    ),
                    "invalid_residual_mean": (
                        float(np.mean(invalid_residual)) if invalid_count > 0 else None
                    ),
                    "invalid_residual_max": (
                        float(np.max(invalid_residual)) if invalid_count > 0 else None
                    ),
                },
            )
            if invalid_count > 0:
                invalid_idx = np.flatnonzero(invalid_mask)
                top_order = np.argsort(displacement_result.assignment_residual[invalid_mask])[
                    ::-1
                ]
                top_invalid = invalid_idx[top_order[: min(10, invalid_count)]]
                invalid_preview = {
                    "atom_index": top_invalid.astype(int),
                    "y": coords[top_invalid, 0],
                    "x": coords[top_invalid, 1],
                    "assignment_residual": displacement_result.assignment_residual[
                        top_invalid
                    ],
                    "lattice_i": displacement_result.lattice_indices_ij[top_invalid, 0],
                    "lattice_j": displacement_result.lattice_indices_ij[top_invalid, 1],
                }
                with st.expander("Show top invalid-assignment atoms"):
                    st.dataframe(invalid_preview, use_container_width=True)
            if displacement_result.warnings:
                summary = _warning_summary_lines(displacement_result.warnings)
                st.warning("\n".join(summary))
                with st.expander("Show full atomic-displacement warnings"):
                    st.code("\n".join(displacement_result.warnings), language="text")
            else:
                st.success("No atomic-displacement warnings.")

    with debug_tab:
        debug_left, debug_right = st.columns([1.35, 1.0])
        with debug_left:
            disp_fig = _render_atomic_displacement_overlay(
                norm,
                displacement_result,
                displacement_scale=displacement_scale,
                displacement_unit=displacement_unit,
                marker_size=params["atomic_displacement_marker_size"],
                max_arrows=params["atomic_displacement_max_arrows"],
            )
            st.pyplot(disp_fig)
            disp_png = _figure_png_bytes(disp_fig)
            st.download_button(
                label="Download atomic displacement debug PNG",
                data=disp_png,
                file_name=f"{Path(uploaded.name).stem}_atomic_displacement_debug.png",
                mime="image/png",
            )
            plt.close(disp_fig)
        with debug_right:
            st.caption(
                "This view keeps observed atoms, ideal lattice sites, reference atoms, "
                "invalid assignments, and arrows together for algorithm diagnosis."
            )
            if custom_basis is not None:
                basis_fig = _render_custom_basis_context(norm, coords, custom_basis)
                st.pyplot(basis_fig)
                plt.close(basis_fig)
                st.caption(
                    "Current base / a / b lattice definition used by atomic displacement."
                )

    st.subheader("Atomic Displacement Preview")
    disp_preview_count = min(len(coords), 500)
    if disp_preview_count > 0:
        disp_preview = {
            "y": coords[:disp_preview_count, 0],
            "x": coords[:disp_preview_count, 1],
            "ideal_y [px]": displacement_result.coordinates_ideal_yx[:disp_preview_count, 0],
            "ideal_x [px]": displacement_result.coordinates_ideal_yx[:disp_preview_count, 1],
            f"ideal_lattice_uy [{displacement_unit}]": (
                displacement_result.displacement_yx[:disp_preview_count, 0] * displacement_scale
            ),
            f"ideal_lattice_ux [{displacement_unit}]": (
                displacement_result.displacement_yx[:disp_preview_count, 1] * displacement_scale
            ),
            f"ideal_lattice_|u| [{displacement_unit}]": (
                displacement_result.displacement_magnitude[:disp_preview_count] * displacement_scale
            ),
            "lattice_i": displacement_result.lattice_indices_ij[:disp_preview_count, 0],
            "lattice_j": displacement_result.lattice_indices_ij[:disp_preview_count, 1],
            "valid_assignment": displacement_result.valid_atom_mask[
                :disp_preview_count
            ].astype(np.int32),
        }
        st.dataframe(disp_preview, use_container_width=True)
    atomic_displacement_csv_text = _atomic_displacement_csv(
        coords=coords,
        result=displacement_result,
        displacement_scale=displacement_scale,
        displacement_unit=displacement_unit,
    )
    st.download_button(
        label="Download atomic displacement CSV",
        data=atomic_displacement_csv_text.encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_atomic_displacement.csv",
        mime="text/csv",
    )
elif displacement_ready:
    st.info(
        "Atomic displacement inputs are ready. Click `Run Atomic Displacement` to fit "
        "the ideal lattice and compute atom displacements."
    )
elif custom_basis_ready and displacement_uses_reference_roi:
    st.info(
        "Custom a/b direction is ready. For method 2, draw a reference ROI with at least "
        "3 regular atoms before running displacement."
    )
else:
    st.info(
        "Atomic displacement is fully standalone now. "
        "Choose a displacement method, define the custom a-b direction above, "
        "then run this module directly."
    )

st.subheader("Atomic Spacing Segments")
st.caption(
    "Connect adjacent atoms by custom a/b lattice indexing, lattice-angle-only "
    "auto direction inference, manual row direction, or a direction inferred from "
    "two atom indices. Line color represents local atom-to-atom spacing."
)
spacing_method = params["plane_spacing_method"]
spacing_uses_lattice = spacing_method in {
    "a/b lattice index",
    "a/b lattice index + fixed angle",
}
spacing_uses_fixed_lattice_angle = spacing_method == "a/b lattice index + fixed angle"
spacing_uses_angle_only = spacing_method == "lattice angle only (auto directions)"
spacing_uses_picked_direction = spacing_method == "picked direction from two atoms"
spacing_pick_valid = (
    0 <= int(params["plane_spacing_pick_start"]) < len(coords)
    and 0 <= int(params["plane_spacing_pick_end"]) < len(coords)
    and int(params["plane_spacing_pick_start"]) != int(params["plane_spacing_pick_end"])
)
spacing_ready = len(coords) >= 2 and (
    (spacing_uses_lattice and custom_basis_ready and custom_basis is not None)
    or (spacing_uses_angle_only and len(coords) >= 3)
    or spacing_method == "manual angle"
    or (spacing_uses_picked_direction and spacing_pick_valid)
)
picked_spacing_angle = None
if spacing_uses_picked_direction and spacing_pick_valid:
    picked_vec = (
        coords[int(params["plane_spacing_pick_end"])]
        - coords[int(params["plane_spacing_pick_start"])]
    )
    picked_spacing_angle = float(np.degrees(np.arctan2(picked_vec[0], picked_vec[1])) % 360.0)
spacing_direction_display = (
    params["plane_spacing_direction"] if spacing_uses_lattice else "selected direction"
)
current_spacing_settings = {
    "method": spacing_method,
    "atom_count": int(len(coords)),
    "basis_indices": {
        "base_idx": int(custom_basis["base_idx"]) if custom_basis is not None else None,
        "a_idx": int(custom_basis["a_idx"]) if custom_basis is not None else None,
        "b_idx": int(custom_basis["b_idx"]) if custom_basis is not None else None,
    },
    "lattice_index_tolerance": params["plane_spacing_lattice_tolerance"],
    "lattice_angle_deg": params["plane_spacing_lattice_angle"],
    "angle_deg": params["plane_spacing_angle_deg"],
    "angle_tolerance_deg": params["plane_spacing_angle_tolerance"],
    "min_spacing_px": params["plane_spacing_min_px"],
    "max_spacing_px": params["plane_spacing_max_px"],
    "pick_start": params["plane_spacing_pick_start"],
    "pick_end": params["plane_spacing_pick_end"],
}
if (
    plane_spacing_meta is not None
    and plane_spacing_meta.get("settings") != current_spacing_settings
):
    _invalidate_plane_spacing_state()
    plane_spacing_result = None
    plane_spacing_meta = None

spacing_button_col, spacing_info_col = st.columns([0.34, 0.66])
run_plane_spacing = spacing_button_col.button(
    "Run Atomic Spacing Segments",
    type="primary",
    use_container_width=False,
    disabled=not spacing_ready,
)
if spacing_ready:
    if spacing_uses_lattice:
        if spacing_uses_fixed_lattice_angle:
            spacing_info_col.caption(
                "Source: custom base/a/b with fixed lattice angle | "
                f"a-b angle={params['plane_spacing_lattice_angle']:.2f} deg | "
                f"index tolerance={params['plane_spacing_lattice_tolerance']:.2f} | "
                f"display={params['plane_spacing_direction']} | unit={displacement_unit}"
            )
        else:
            spacing_info_col.caption(
                "Source: current custom base/a/b lattice index | "
                f"index tolerance={params['plane_spacing_lattice_tolerance']:.2f} | "
                f"display={params['plane_spacing_direction']} | "
                f"unit={displacement_unit}"
            )
    elif spacing_uses_angle_only:
        spacing_info_col.caption(
            "Source: lattice angle only, auto-inferred row directions | "
            f"lattice angle={params['plane_spacing_lattice_angle']:.2f} deg | "
            f"direction tolerance={params['plane_spacing_angle_tolerance']:.1f} deg | "
            f"range={params['plane_spacing_min_px']:.1f}-"
            f"{params['plane_spacing_max_px']:.1f} px | unit={displacement_unit}"
        )
    elif spacing_method == "manual angle":
        spacing_info_col.caption(
            f"Source: manual angle {params['plane_spacing_angle_deg']:.2f} deg | "
            f"tolerance={params['plane_spacing_angle_tolerance']:.1f} deg | "
            f"range={params['plane_spacing_min_px']:.1f}-"
            f"{params['plane_spacing_max_px']:.1f} px | unit={displacement_unit}"
        )
    else:
        spacing_info_col.caption(
            "Source: picked atom direction | "
            f"atoms={params['plane_spacing_pick_start']} -> "
            f"{params['plane_spacing_pick_end']} | "
            f"angle={picked_spacing_angle:.2f} deg | "
            f"tolerance={params['plane_spacing_angle_tolerance']:.1f} deg | "
            f"range={params['plane_spacing_min_px']:.1f}-"
            f"{params['plane_spacing_max_px']:.1f} px | unit={displacement_unit}"
        )
else:
    if spacing_uses_lattice:
        spacing_info_col.caption(
            "Define the custom base/a/b basis first. This module directly measures "
            "atom-to-atom distances and does not require running atomic displacement."
        )
    elif spacing_uses_angle_only:
        spacing_info_col.caption(
            "Need at least 3 atom points. This mode skips custom base/a/b selection "
            "and infers two row directions from the requested lattice angle."
        )
    elif spacing_uses_picked_direction and not spacing_pick_valid:
        spacing_info_col.caption(
            "Pick two different valid atom indices for the direction. Use the atom-index "
            "reference maps or coordinate table if needed."
        )
    else:
        spacing_info_col.caption("Need at least 2 atom points before measuring spacing.")

if spacing_uses_picked_direction and len(coords) > 0:
    highlighted_spacing_indices = {
        idx
        for idx in (
            int(params["plane_spacing_pick_start"]),
            int(params["plane_spacing_pick_end"]),
        )
        if 0 <= idx < len(coords)
    }
    with st.expander("Show atom-index reference map for picked direction", expanded=False):
        spacing_index_img = _render_atom_index_reference_image(
            norm,
            coords,
            highlighted_indices=highlighted_spacing_indices,
            max_side=1200,
        )
        st.image(
            spacing_index_img,
            caption="Use these atom indices for `Picked direction atom 1/2 index`.",
            use_container_width=False,
        )
        st.download_button(
            label="Download spacing atom-index reference PNG",
            data=_pil_png_bytes(spacing_index_img),
            file_name=f"{Path(uploaded.name).stem}_spacing_atom_indices.png",
            mime="image/png",
        )

if run_plane_spacing and spacing_ready:
    try:
        if spacing_uses_fixed_lattice_angle and custom_basis is not None:
            measured_spacing = compute_atomic_plane_spacing_segments_from_angle_constrained_basis(
                coords,
                base_index=int(custom_basis["base_idx"]),
                a_index=int(custom_basis["a_idx"]),
                b_index=int(custom_basis["b_idx"]),
                lattice_angle_deg=params["plane_spacing_lattice_angle"],
                lattice_index_tolerance=params["plane_spacing_lattice_tolerance"],
                directions=("a", "b"),
            )
        elif spacing_uses_lattice and custom_basis is not None:
            measured_spacing = compute_atomic_plane_spacing_segments_from_basis(
                coords,
                base_index=int(custom_basis["base_idx"]),
                a_index=int(custom_basis["a_idx"]),
                b_index=int(custom_basis["b_idx"]),
                lattice_index_tolerance=params["plane_spacing_lattice_tolerance"],
                directions=("a", "b"),
            )
        elif spacing_uses_angle_only:
            measured_spacing = compute_atomic_plane_spacing_segments_from_lattice_angle(
                coords,
                lattice_angle_deg=params["plane_spacing_lattice_angle"],
                angle_tolerance_deg=params["plane_spacing_angle_tolerance"],
                min_spacing_px=params["plane_spacing_min_px"],
                max_spacing_px=params["plane_spacing_max_px"],
            )
        elif spacing_method == "manual angle":
            measured_spacing = compute_atomic_plane_spacing_segments_by_angle(
                coords,
                angle_deg=params["plane_spacing_angle_deg"],
                angle_tolerance_deg=params["plane_spacing_angle_tolerance"],
                min_spacing_px=params["plane_spacing_min_px"],
                max_spacing_px=params["plane_spacing_max_px"],
            )
        else:
            measured_spacing = compute_atomic_plane_spacing_segments_from_picked_direction(
                coords,
                start_index=params["plane_spacing_pick_start"],
                end_index=params["plane_spacing_pick_end"],
                angle_tolerance_deg=params["plane_spacing_angle_tolerance"],
                min_spacing_px=params["plane_spacing_min_px"],
                max_spacing_px=params["plane_spacing_max_px"],
            )
    except Exception as exc:
        _invalidate_plane_spacing_state()
        st.error(f"Atomic spacing segment quantification failed: {exc}")
        plane_spacing_result = None
        plane_spacing_meta = None
    else:
        st.session_state.plane_spacing_result = measured_spacing
        st.session_state.plane_spacing_meta = {
            "settings": current_spacing_settings,
            "method": measured_spacing.provenance.get("method", spacing_method),
            "segment_count": int(len(measured_spacing.length_px)),
        }
        plane_spacing_result = measured_spacing
        plane_spacing_meta = st.session_state.plane_spacing_meta
        st.success("Atomic spacing segments finished.")

if plane_spacing_meta is not None:
    st.write("Atomic spacing settings:", plane_spacing_meta)

if plane_spacing_result is not None:
    spacing_map_tab, spacing_hist_tab, spacing_table_tab = st.tabs(
        ["Spacing map", "Histogram", "Summary / Table"]
    )
    with spacing_map_tab:
        spacing_fig = _render_plane_spacing_segments(
            norm,
            coords,
            plane_spacing_result,
            direction_mode=spacing_direction_display,
            spacing_scale=displacement_scale,
            spacing_unit=displacement_unit,
            show_image=params["plane_spacing_show_image"],
            show_atoms=params["plane_spacing_show_atoms"],
            auto_color_range=params["plane_spacing_auto_color_range"],
            vmin=params["plane_spacing_color_min"],
            vmax=params["plane_spacing_color_max"],
            line_width=params["plane_spacing_line_width"],
        )
        st.pyplot(spacing_fig)
        spacing_png = _figure_png_bytes(spacing_fig)
        st.download_button(
            label="Download atomic spacing segment map PNG",
            data=spacing_png,
            file_name=f"{Path(uploaded.name).stem}_atomic_spacing_segments.png",
            mime="image/png",
        )
        plt.close(spacing_fig)
    with spacing_hist_tab:
        spacing_hist_fig = _render_plane_spacing_histogram(
            plane_spacing_result,
            direction_mode=spacing_direction_display,
            spacing_scale=displacement_scale,
            spacing_unit=displacement_unit,
        )
        st.pyplot(spacing_hist_fig)
        spacing_hist_png = _figure_png_bytes(spacing_hist_fig)
        st.download_button(
            label="Download atomic spacing histogram PNG",
            data=spacing_hist_png,
            file_name=f"{Path(uploaded.name).stem}_atomic_spacing_histogram.png",
            mime="image/png",
        )
        plt.close(spacing_hist_fig)
    with spacing_table_tab:
        direction_mask = _plane_spacing_direction_mask(
            plane_spacing_result,
            spacing_direction_display,
        )
        spacing_lengths = (
            np.asarray(plane_spacing_result.length_px, dtype=np.float64) * displacement_scale
        )
        summary_rows = []
        unique_spacing_directions = sorted(
            str(v)
            for v in np.unique(
                np.asarray(plane_spacing_result.direction, dtype=object)[direction_mask]
            )
        )
        for direction in unique_spacing_directions:
            mask = (
                direction_mask
                & (np.asarray(plane_spacing_result.direction, dtype=object) == direction)
                & np.isfinite(spacing_lengths)
            )
            values = spacing_lengths[mask]
            if values.size:
                summary_rows.append(
                    {
                        "direction": direction,
                        "count": int(values.size),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "median": float(np.median(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                )
        st.write(
            "Atomic spacing summary:",
            {
                "confidence": round(float(plane_spacing_result.confidence), 3),
                "method": plane_spacing_result.provenance.get("method"),
                "display_direction": spacing_direction_display,
                "unit": displacement_unit,
                "segment_count": int(np.sum(direction_mask)),
            },
        )
        spacing_fit_stats = plane_spacing_result.provenance.get("fit_statistics", {})
        if spacing_fit_stats:
            st.write(
                "Spacing fit statistics:",
                {
                    "original_ab_angle_deg": spacing_fit_stats.get(
                        "original_ab_angle_deg"
                    ),
                    "constrained_ab_angle_deg": spacing_fit_stats.get(
                        "constrained_ab_angle_deg"
                    ),
                    "valid_atom_count": spacing_fit_stats.get("valid_atom_count"),
                    "invalid_atom_count": spacing_fit_stats.get("invalid_atom_count"),
                    "assignment_residual_median": spacing_fit_stats.get(
                        "assignment_residual_median"
                    ),
                },
            )
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
        st.info(
            "These short-line values are local atom-to-atom distances along custom a/b. "
            "They are different from perpendicular crystallographic plane spacing."
        )
        if plane_spacing_result.warnings:
            summary = _warning_summary_lines(plane_spacing_result.warnings)
            st.warning("\n".join(summary))
            with st.expander("Show full atomic-spacing warnings"):
                st.code("\n".join(plane_spacing_result.warnings), language="text")
        spacing_preview_count = min(len(plane_spacing_result.length_px), 500)
        if spacing_preview_count > 0:
            spacing_preview = {
                "direction": plane_spacing_result.direction[:spacing_preview_count],
                "source_atom": plane_spacing_result.source_atom_index[:spacing_preview_count],
                "target_atom": plane_spacing_result.target_atom_index[:spacing_preview_count],
                "source_i": plane_spacing_result.source_lattice_ij[:spacing_preview_count, 0],
                "source_j": plane_spacing_result.source_lattice_ij[:spacing_preview_count, 1],
                "target_i": plane_spacing_result.target_lattice_ij[:spacing_preview_count, 0],
                "target_j": plane_spacing_result.target_lattice_ij[:spacing_preview_count, 1],
                f"length [{displacement_unit}]": (
                    plane_spacing_result.length_px[:spacing_preview_count] * displacement_scale
                ),
            }
            st.dataframe(spacing_preview, use_container_width=True)
        spacing_csv_text = _plane_spacing_csv(
            plane_spacing_result,
            spacing_scale=displacement_scale,
            spacing_unit=displacement_unit,
        )
        st.download_button(
            label="Download atomic spacing segments CSV",
            data=spacing_csv_text.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_atomic_spacing_segments.csv",
            mime="text/csv",
        )
else:
    st.info(
        "Click `Run Atomic Spacing Segments` after choosing a spacing method. "
        "For angle modes, tune angle tolerance and distance range if lines connect "
        "the wrong neighbors."
    )

st.subheader("Atomic Column Intensity Quantification")
st.caption(
    "Standalone intensity module from current atom coordinates. First version uses "
    "disk integration plus local annulus background subtraction; output is relative "
    "atomic-column intensity, not absolute Z unless HAADF calibration is available."
)
current_intensity_settings = {
    "method": params["intensity_method"],
    "integration_radius": params["intensity_integration_radius"],
    "background_inner_radius": params["intensity_background_inner_radius"],
    "background_outer_radius": params["intensity_background_outer_radius"],
    "background_statistic": params["intensity_background_statistic"],
    "normalization": params["intensity_normalization"],
}
if (
    atomic_intensity_meta is not None
    and atomic_intensity_meta.get("settings") != current_intensity_settings
):
    _invalidate_atomic_intensity_state()
    atomic_intensity_result = None
    atomic_intensity_meta = None

intensity_ready = (
    len(coords) > 0
    and params["intensity_integration_radius"] > 0
    and params["intensity_background_inner_radius"]
    > params["intensity_integration_radius"]
    and params["intensity_background_outer_radius"]
    > params["intensity_background_inner_radius"]
)
intensity_button_col, intensity_info_col = st.columns([0.34, 0.66])
run_atomic_intensity = intensity_button_col.button(
    "Run Atomic Column Intensity",
    type="primary",
    use_container_width=False,
    disabled=not intensity_ready,
)
intensity_info_col.caption(
    f"method={params['intensity_method']} | "
    f"r={params['intensity_integration_radius']:.1f}px | "
    f"bg={params['intensity_background_inner_radius']:.1f}-"
    f"{params['intensity_background_outer_radius']:.1f}px | "
    f"background={params['intensity_background_statistic']} | "
    f"normalize={params['intensity_normalization']}"
)
if len(coords) == 0:
    st.warning("Need atom coordinates before running intensity quantification.")
elif not intensity_ready:
    st.warning(
        "Intensity radii are invalid. Use: integration radius < background inner "
        "radius < background outer radius."
    )

if run_atomic_intensity and intensity_ready:
    with st.spinner("Running atomic-column intensity quantification..."):
        try:
            measured_intensity = quantify_atomic_column_intensity(
                raw,
                coords,
                integration_radius=params["intensity_integration_radius"],
                background_inner_radius=params["intensity_background_inner_radius"],
                background_outer_radius=params["intensity_background_outer_radius"],
                background_statistic=params["intensity_background_statistic"],
                normalization=params["intensity_normalization"],
            )
        except Exception as exc:
            _invalidate_atomic_intensity_state()
            st.error(f"Atomic-column intensity quantification failed: {exc}")
            atomic_intensity_result = None
            atomic_intensity_meta = None
        else:
            st.session_state.atomic_intensity_result = measured_intensity
            st.session_state.atomic_intensity_meta = {
                "settings": current_intensity_settings,
                "method": "disk_integration_with_local_annulus_background",
                "atom_count": int(len(coords)),
            }
            atomic_intensity_result = measured_intensity
            atomic_intensity_meta = st.session_state.atomic_intensity_meta
            st.success("Atomic-column intensity quantification finished.")

if atomic_intensity_meta is not None:
    st.write("Atomic intensity settings:", atomic_intensity_meta)

if atomic_intensity_result is not None:
    intensity_map_tab, intensity_hist_tab, intensity_table_tab = st.tabs(
        ["Intensity map", "Histogram", "Summary / Table"]
    )
    with intensity_map_tab:
        intensity_fig = _render_atomic_intensity_overlay(
            norm,
            coords,
            atomic_intensity_result,
            metric=params["intensity_overlay_metric"],
            auto_color_range=params["intensity_auto_color_range"],
            vmin=params["intensity_color_min"],
            vmax=params["intensity_color_max"],
            marker_size=params["intensity_marker_size"],
        )
        st.pyplot(intensity_fig)
        intensity_png = _figure_png_bytes(intensity_fig)
        st.download_button(
            label="Download atomic-column intensity map PNG",
            data=intensity_png,
            file_name=f"{Path(uploaded.name).stem}_atomic_column_intensity_map.png",
            mime="image/png",
        )
        plt.close(intensity_fig)
    with intensity_hist_tab:
        hist_fig = _render_atomic_intensity_histogram(
            atomic_intensity_result,
            metric=params["intensity_overlay_metric"],
        )
        st.pyplot(hist_fig)
        hist_png = _figure_png_bytes(hist_fig)
        st.download_button(
            label="Download atomic-column intensity histogram PNG",
            data=hist_png,
            file_name=f"{Path(uploaded.name).stem}_atomic_column_intensity_histogram.png",
            mime="image/png",
        )
        plt.close(hist_fig)
    with intensity_table_tab:
        intensity_values, intensity_label = _atomic_intensity_metric(
            atomic_intensity_result,
            params["intensity_overlay_metric"],
        )
        valid_intensity = (
            np.asarray(atomic_intensity_result.valid_atom_mask, dtype=bool)
            & np.isfinite(intensity_values)
        )
        finite_values = intensity_values[valid_intensity]
        fit_stats = atomic_intensity_result.provenance.get("fit_statistics", {})
        st.write(
            "Atomic-column intensity summary:",
            {
                "confidence": round(float(atomic_intensity_result.confidence), 3),
                "metric": intensity_label,
                "valid_atom_count": int(np.sum(valid_intensity)),
                "valid_ratio": fit_stats.get("valid_ratio"),
                "median": float(np.median(finite_values)) if finite_values.size else None,
                "mean": float(np.mean(finite_values)) if finite_values.size else None,
                "min": float(np.min(finite_values)) if finite_values.size else None,
                "max": float(np.max(finite_values)) if finite_values.size else None,
                "edge_count": fit_stats.get("edge_count"),
                "overlap_count": fit_stats.get("overlap_count"),
                "median_snr": fit_stats.get("median_snr"),
            },
        )
        st.info(
            "Interpretation note: these are relative atomic-column intensities. "
            "For HAADF-STEM they can support Z-contrast analysis, but thickness, "
            "tilt, detector angle, drift, and calibration still matter."
        )
        if atomic_intensity_result.warnings:
            summary = _warning_summary_lines(atomic_intensity_result.warnings)
            st.warning("\n".join(summary))
            with st.expander("Show full atomic-intensity warnings"):
                st.code("\n".join(atomic_intensity_result.warnings), language="text")
        else:
            st.success("No atomic-intensity warnings.")

        intensity_preview_count = min(len(coords), 500)
        if intensity_preview_count > 0:
            intensity_preview = {
                "y": coords[:intensity_preview_count, 0],
                "x": coords[:intensity_preview_count, 1],
                "peak_intensity": atomic_intensity_result.peak_intensity[
                    :intensity_preview_count
                ],
                "background_subtracted_sum": (
                    atomic_intensity_result.background_subtracted_sum[
                        :intensity_preview_count
                    ]
                ),
                "normalized_intensity": atomic_intensity_result.normalized_intensity[
                    :intensity_preview_count
                ],
                "snr": atomic_intensity_result.snr[:intensity_preview_count],
                "valid_intensity": atomic_intensity_result.valid_atom_mask[
                    :intensity_preview_count
                ].astype(np.int32),
                "edge_flag": atomic_intensity_result.edge_flag[
                    :intensity_preview_count
                ].astype(np.int32),
                "overlap_flag": atomic_intensity_result.overlap_flag[
                    :intensity_preview_count
                ].astype(np.int32),
            }
            st.dataframe(intensity_preview, use_container_width=True)
        atomic_intensity_csv_text = _atomic_intensity_csv(
            coords,
            atomic_intensity_result,
        )
        st.download_button(
            label="Download atomic-column intensity CSV",
            data=atomic_intensity_csv_text.encode("utf-8"),
            file_name=f"{Path(uploaded.name).stem}_atomic_column_intensity.csv",
            mime="text/csv",
        )

    if params["intensity_grouping_enabled"]:
        st.subheader("Optional Intensity Group Analysis")
        st.caption(
            "This optional section is OFF by default. It summarizes intensity by "
            "automatic intensity clusters or by the current user-defined a/b sublattice."
        )
        group_result: AtomicIntensityGroupResult | None = None
        try:
            if params["intensity_grouping_method"] == "Intensity clustering":
                group_result = cluster_atomic_column_intensities(
                    atomic_intensity_result,
                    metric=params["intensity_group_metric"],
                    n_groups=params["intensity_cluster_count"],
                )
            elif displacement_result is None:
                st.warning(
                    "Current a/b lattice sublattice grouping needs an atomic displacement "
                    "result first, because it uses the fitted lattice indices."
                )
            elif len(displacement_result.lattice_indices_ij) != len(coords):
                st.warning(
                    "The current atomic displacement result does not match the current "
                    "atom-coordinate table. Re-run atomic displacement first."
                )
            else:
                group_result = group_atomic_column_intensities_by_lattice(
                    atomic_intensity_result,
                    displacement_result.lattice_indices_ij,
                    metric=params["intensity_group_metric"],
                    mode=params["intensity_lattice_group_mode"],
                    valid_lattice_mask=displacement_result.valid_atom_mask,
                )
        except Exception as exc:
            st.error(f"Intensity group analysis failed: {exc}")

        if group_result is not None:
            group_summary = pd.DataFrame(group_result.summaries)
            if group_summary.empty:
                st.warning("No non-empty intensity groups were found.")
            else:
                st.write(
                    "Group settings:",
                    {
                        "method": group_result.provenance.get("method"),
                        "metric": group_result.metric,
                        "confidence": round(float(group_result.confidence), 3),
                    },
                )
                st.dataframe(group_summary, use_container_width=True)
                group_hist_fig = _render_atomic_intensity_group_histogram(
                    atomic_intensity_result,
                    group_result,
                    metric=params["intensity_group_metric"],
                )
                st.pyplot(group_hist_fig)
                group_hist_png = _figure_png_bytes(group_hist_fig)
                st.download_button(
                    label="Download grouped intensity histogram PNG",
                    data=group_hist_png,
                    file_name=(
                        f"{Path(uploaded.name).stem}_atomic_column_intensity_groups.png"
                    ),
                    mime="image/png",
                )
                plt.close(group_hist_fig)
                group_csv = _atomic_intensity_group_csv(group_result)
                st.download_button(
                    label="Download grouped intensity statistics CSV",
                    data=group_csv.encode("utf-8"),
                    file_name=(
                        f"{Path(uploaded.name).stem}_atomic_column_intensity_groups.csv"
                    ),
                    mime="text/csv",
                )
            if group_result.warnings:
                summary = _warning_summary_lines(group_result.warnings)
                st.warning("\n".join(summary))
else:
    st.info(
        "Click `Run Atomic Column Intensity` to quantify peak, integrated, "
        "background-subtracted, normalized intensity, and local SNR for each atom column."
    )

if atom_strain_fields is not None:
    st.subheader("Atom-Position Strain Preview")
    atom_preview_count = min(len(coords), 500)
    if atom_preview_count > 0:
        atom_preview = {
            "y": coords[:atom_preview_count, 0],
            "x": coords[:atom_preview_count, 1],
            f"ux [{displacement_unit}]": (
                atom_strain_fields.atom_displacement_xy[:atom_preview_count, 0]
                * displacement_scale
            ),
            f"uy [{displacement_unit}]": (
                atom_strain_fields.atom_displacement_xy[:atom_preview_count, 1]
                * displacement_scale
            ),
            f"exx [{strain_unit}]": (
                atom_strain_fields.atom_strain_xx[:atom_preview_count] * strain_scale
            ),
            f"eyy [{strain_unit}]": (
                atom_strain_fields.atom_strain_yy[:atom_preview_count] * strain_scale
            ),
            f"exy [{strain_unit}]": (
                atom_strain_fields.atom_strain_xy[:atom_preview_count] * strain_scale
            ),
            "rotation [rad]": atom_strain_fields.atom_rotation_xy[:atom_preview_count],
            "uncertainty": atom_strain_fields.atom_uncertainty[:atom_preview_count],
            "valid_fit": atom_strain_fields.valid_atom_mask[:atom_preview_count].astype(np.int32),
        }
        if custom_components is not None:
            atom_preview[f"eaa [{strain_unit}]"] = (
                custom_components["eaa"][:atom_preview_count] * strain_scale
            )
            atom_preview[f"ebb [{strain_unit}]"] = (
                custom_components["ebb"][:atom_preview_count] * strain_scale
            )
            atom_preview[f"eab [{strain_unit}]"] = (
                custom_components["eab"][:atom_preview_count] * strain_scale
            )
        st.dataframe(atom_preview, use_container_width=True)
    atom_strain_csv_text = _atom_strain_csv(
        coords=coords,
        fields=atom_strain_fields,
        displacement_scale=displacement_scale,
        displacement_unit=displacement_unit,
        strain_scale=strain_scale,
        strain_unit=strain_unit,
        custom_components=custom_components,
    )
    st.download_button(
        label="Download atom-position strain CSV",
        data=atom_strain_csv_text.encode("utf-8"),
        file_name=f"{Path(uploaded.name).stem}_atom_position_strain.csv",
        mime="text/csv",
    )

st.subheader("Edited Coordinates Preview")
preview_count = min(len(coords), 500)
if preview_count > 0:
    preview = np.column_stack([coords[:preview_count], intensities[:preview_count]])
    st.dataframe(preview, use_container_width=True)
else:
    st.info("No atoms available.")

csv_text = _coordinates_csv(coords, intensities)
st.download_button(
    label="Download edited coordinates CSV",
    data=csv_text.encode("utf-8"),
    file_name=f"{Path(uploaded.name).stem}_atom_coordinates_edited.csv",
    mime="text/csv",
)

distance_csv_text = _distance_csv(
    coords=coords,
    nearest=distance_stats.nearest_distances,
    next_nearest=distance_stats.next_nearest_distances,
    unit=distance_stats.unit,
)
st.download_button(
    label="Download distance statistics CSV",
    data=distance_csv_text.encode("utf-8"),
    file_name=f"{Path(uploaded.name).stem}_atom_distance_statistics.csv",
    mime="text/csv",
)
