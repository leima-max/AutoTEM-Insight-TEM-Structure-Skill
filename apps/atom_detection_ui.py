from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from streamlit_drawable_canvas import st_canvas

from autotem.atomic.atom_detection import _load_image, _normalize_image, detect_atoms
from autotem.atomic.distance_statistics import compute_distance_statistics
from autotem.atomic.gaussian_refinement import refine_atom_coordinates
from autotem.atomic.lattice_constraint import apply_lattice_constraint
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


def _symmetric_limit(arr: np.ndarray, percentile: float = 98.0, fallback: float = 1.0) -> float:
    finite = np.asarray(arr, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return fallback
    lim = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(lim) or lim <= 1e-12:
        lim = fallback
    return lim


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


def _figure_png_bytes(fig: plt.Figure) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=180)
    buffer.seek(0)
    return buffer.getvalue()


def _invalidate_atom_strain_state() -> None:
    st.session_state.atom_strain_fields = None
    st.session_state.atom_strain_meta = None
    _invalidate_custom_basis_state()


def _invalidate_custom_basis_state() -> None:
    st.session_state.custom_basis_indices = {"base": None, "a": None, "b": None}
    st.session_state.custom_basis_stage = 0
    st.session_state.custom_basis_canvas_nonce = int(
        st.session_state.get("custom_basis_canvas_nonce", 0)
    ) + 1
    st.session_state.custom_basis_message = ""


def _disable_custom_basis_state() -> None:
    st.session_state.custom_basis_enabled = False
    _invalidate_custom_basis_state()


def _coords_inside_roi_mask(coords: np.ndarray, roi_mask: np.ndarray | None) -> np.ndarray | None:
    if roi_mask is None or len(coords) == 0:
        return None
    mask = np.asarray(roi_mask, dtype=bool)
    yi = np.clip(np.round(coords[:, 0]).astype(int), 0, mask.shape[0] - 1)
    xi = np.clip(np.round(coords[:, 1]).astype(int), 0, mask.shape[1] - 1)
    return mask[yi, xi]


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

    if len(coords) > 0:
        px_r = max(2, int(round(2.6 * scale)))
        for y, x in coords:
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
    ("custom_basis_canvas_nonce", 0),
    ("custom_basis_message", ""),
    ("custom_basis_enabled", False),
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
tool = st.radio(
    "Tool",
    [
        "Add atom by click",
        "Delete atom by click",
        "Draw irregular ROI (right-click to close)",
    ],
    horizontal=True,
)
delete_radius = st.slider("Delete radius (px)", min_value=0.5, max_value=20.0, value=3.0, step=0.5)
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
canvas_key = f"canvas-{tool}-{st.session_state.canvas_nonce}"
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

act_col1, act_col2, act_col3 = st.columns([1, 1, 1])
apply_action = act_col1.button("Apply Canvas Action", type="primary")
clear_canvas = act_col2.button("Clear Canvas")
reset_auto = act_col3.button("Reset to Auto Detection")

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

if roi_mask is not None:
    inside = 0
    if len(coords) > 0:
        yi = np.clip(np.round(coords[:, 0]).astype(int), 0, roi_mask.shape[0] - 1)
        xi = np.clip(np.round(coords[:, 1]).astype(int), 0, roi_mask.shape[1] - 1)
        inside = int(np.sum(roi_mask[yi, xi]))
    st.info(
        f"ROI pixels: {int(np.sum(roi_mask))} | Atoms inside ROI: {inside}"
    )

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
        yi = np.clip(np.round(coords[:, 0]).astype(int), 0, effective_roi_mask.shape[0] - 1)
        xi = np.clip(np.round(coords[:, 1]).astype(int), 0, effective_roi_mask.shape[1] - 1)
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

prospective_reference_atom_mask = None
if params["atom_strain_use_roi_reference"]:
    prospective_reference_atom_mask = _coords_inside_roi_mask(coords, roi_mask)

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
if atom_strain_fields is None:
    st.info("Run atom-position strain to visualize displacement/strain from edited atom columns.")
else:
    displacement_scale = params["pixel_size"] if params["use_physical_unit"] else 1.0
    displacement_unit = params["distance_unit"] if params["use_physical_unit"] else "px"
    strain_scale = 100.0 if params["atom_strain_display_unit"] == "%" else 1.0
    strain_unit = "%" if params["atom_strain_display_unit"] == "%" else "fraction"
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

    st.subheader("Custom a-b Basis Strain")
    st.caption(
        "Optional. Click to enable only when you want to define a custom "
        "base-a-b strain basis from atom picks."
    )
    custom_components: dict[str, np.ndarray] | None = None
    if not bool(st.session_state.custom_basis_enabled):
        if st.button("Enable a-b Basis Picking", use_container_width=False):
            st.session_state.custom_basis_enabled = True
            _invalidate_custom_basis_state()
            st.session_state.custom_basis_message = (
                "Click the base atom for the custom a-b basis."
            )
            st.rerun()
        st.info("Custom a-b basis strain is currently off.")
    else:
        basis_button_col1, basis_button_col2 = st.columns([0.34, 0.34])
        if basis_button_col1.button("Start / Reset a-b Picking", use_container_width=True):
            _invalidate_custom_basis_state()
            st.session_state.custom_basis_message = (
                "Click the base atom for the custom a-b basis."
            )
            st.rerun()
        if basis_button_col2.button("Turn Off a-b Basis", use_container_width=True):
            _disable_custom_basis_state()
            st.rerun()

        basis_indices = dict(st.session_state.custom_basis_indices)
        basis_stage = int(st.session_state.custom_basis_stage)
        stage_prompts = {
            0: "Waiting for base atom click.",
            1: "Base selected. Click the atom that defines the a vector.",
            2: "a-vector atom selected. Click the atom that defines the b vector.",
            3: "Custom a-b basis is ready.",
        }
        if st.session_state.custom_basis_message:
            st.info(st.session_state.custom_basis_message)
        else:
            st.caption(stage_prompts.get(basis_stage, ""))

        basis_markers: dict[str, tuple[float, float]] = {}
        for key_name in ("base", "a", "b"):
            idx = basis_indices.get(key_name)
            if idx is not None and 0 <= int(idx) < len(coords):
                yx = np.asarray(coords[int(idx)], dtype=np.float64)
                basis_markers[key_name] = (float(yx[0]), float(yx[1]))

        basis_canvas_image, basis_sx, basis_sy = _to_display_image(
            norm,
            coords=coords,
            roi_mask=roi_mask,
            basis_markers=basis_markers,
            max_side=900,
        )
        basis_canvas = st_canvas(
            fill_color="rgba(255, 255, 0, 0.25)",
            stroke_width=2,
            stroke_color="#00ffff",
            background_image=basis_canvas_image,
            drawing_mode="point",
            point_display_radius=5,
            display_toolbar=True,
            update_streamlit=True,
            height=basis_canvas_image.height,
            width=basis_canvas_image.width,
            key=f"basis-canvas-{st.session_state.custom_basis_canvas_nonce}",
        )

        if basis_stage < 3:
            basis_draw_mask = _canvas_draw_mask(basis_canvas)
            if basis_draw_mask is not None and np.any(basis_draw_mask):
                disp_points = _mask_centroids(basis_draw_mask, min_pixels=2)
                raw_points = _map_display_to_raw(disp_points, sx=basis_sx, sy=basis_sy)
                if len(raw_points) > 0:
                    y_click, x_click = raw_points[0]
                    d2 = (coords[:, 0] - y_click) ** 2 + (coords[:, 1] - x_click) ** 2
                    nearest_idx = int(np.argmin(d2))
                    target_name = ("base", "a", "b")[basis_stage]
                    if target_name == "a" and nearest_idx == basis_indices.get("base"):
                        st.session_state.custom_basis_message = (
                            "The a-vector atom must be different from the base atom."
                        )
                    elif target_name == "b" and nearest_idx in {
                        basis_indices.get("base"),
                        basis_indices.get("a"),
                    }:
                        st.session_state.custom_basis_message = (
                            "The b-vector atom must be different from both base and a atoms."
                        )
                    else:
                        basis_indices[target_name] = nearest_idx
                        st.session_state.custom_basis_indices = basis_indices
                        st.session_state.custom_basis_stage = basis_stage + 1
                        if basis_stage == 0:
                            st.session_state.custom_basis_message = (
                                "Base selected. Now click the atom that defines the a vector."
                            )
                        elif basis_stage == 1:
                            st.session_state.custom_basis_message = (
                                "a-vector atom selected. "
                                "Now click the atom that defines the b vector."
                            )
                        else:
                            st.session_state.custom_basis_message = "Custom a-b basis is ready."
                    st.session_state.custom_basis_canvas_nonce = (
                        int(st.session_state.custom_basis_canvas_nonce) + 1
                    )
                    st.rerun()

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
                st.warning(
                    "Custom a and b vectors are nearly collinear. Please re-pick the basis."
                )
            else:
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
