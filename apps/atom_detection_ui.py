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


def _render_overlay(image_norm: np.ndarray, coords: np.ndarray, roi_mask: np.ndarray | None):
    fig, ax = plt.subplots(figsize=(6.8, 6.8), dpi=160)
    ax.imshow(image_norm, cmap="gray")
    if roi_mask is not None:
        roi_show = np.ma.masked_where(~roi_mask, roi_mask)
        ax.imshow(roi_show, cmap="spring", alpha=0.28, interpolation="none")
    if len(coords) > 0:
        ax.scatter(coords[:, 1], coords[:, 0], s=14, c="red", marker="o", linewidths=0.35)
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


def _to_display_image(
    image_norm: np.ndarray,
    coords: np.ndarray,
    roi_mask: np.ndarray | None = None,
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
]:
    if key not in st.session_state:
        st.session_state[key] = default


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
        value=0.02,
        step=0.001,
        format="%.6f",
        disabled=not use_physical_unit,
    )
    distance_unit = st.selectbox(
        "Distance unit",
        ["nm", "angstrom", "pm", "um"],
        index=0,
        disabled=not use_physical_unit,
    )

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
    "For ROI mode, `Keep/Remove` can auto-apply current ROI drawing."
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
                st.success(f"Removed {removed} points.")

        else:
            # Polygon drawing may return border + fill; fill holes to get stable ROI area.
            roi_disp = ndi.binary_fill_holes(draw_mask_disp)
            mask = _resize_mask_to_raw(roi_disp, raw.shape)
            st.session_state.roi_mask = mask
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
        st.success(msg)
        st.rerun()

left, right = st.columns([1.45, 1.0])
with left:
    st.subheader("Overlay")
    fig = _render_overlay(norm, coords, roi_mask=roi_mask)
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
