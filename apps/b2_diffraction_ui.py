from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.elements.image as st_image
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from autotem.calibration import calibrate_dataset
from autotem.diffraction import run_diffraction_analysis
from autotem.io import load_tem_dataset, read_image
from autotem.preprocessing import assess_quality, preprocess_dataset

# Compatibility patch for streamlit-drawable-canvas on newer Streamlit.
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

PHASE_OPTIONS = ["Silicon", "Gold", "Graphite", "Titanium Dioxide"]
PHASE_REFERENCE_DB: list[tuple[str, str, str, tuple[float, ...]]] = [
    ("Silicon", "Si", "mp-149", (0.313, 0.192, 0.163, 0.136)),
    ("Gold", "Au", "mp-81", (0.235, 0.204, 0.144, 0.123)),
    ("Graphite", "C", "mp-48", (0.335, 0.213, 0.168, 0.142)),
    ("Titanium Dioxide", "TiO2", "mp-2657", (0.325, 0.249, 0.190, 0.169)),
]


def _measurement_value(measurement: Any) -> float | None:
    if measurement is None:
        return None
    value = getattr(measurement, "value", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _materialize_upload(name: str, payload: bytes) -> Path:
    suffix = Path(name).suffix or ".dat"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(payload)
        return Path(tmp.name)


def _write_npy_array(array: np.ndarray, *, prefix: str) -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy", prefix=prefix) as tmp:
        np.save(tmp, array.astype(np.float32), allow_pickle=False)
        return Path(tmp.name)


def _build_dataset_for_b2(input_path: Path):
    read_result = read_image(input_path)
    array = np.asarray(read_result.array).astype(np.float32)
    npy_path = _write_npy_array(array, prefix="autotem_b2_")

    dataset = load_tem_dataset(input_path)
    dataset_for_b2 = dataset.model_copy(deep=True)
    preview = array[: min(8, array.shape[0]), : min(8, array.shape[1])].tolist()
    dataset_for_b2.data = dataset_for_b2.data.model_copy(
        update={
            "shape": tuple(int(v) for v in array.shape),
            "dtype": str(array.dtype),
            "uri": str(npy_path),
            "inline": preview,
            "description": "B2 UI converted array for deterministic FFT pipeline.",
        }
    )
    if input_path.suffix.lower() != ".npy":
        dataset_for_b2.warnings.append(
            f"Input {input_path.suffix.lower()} converted to temporary .npy for B2 FFT analysis."
        )
    return dataset_for_b2, array, npy_path


def _build_demo_input() -> Path:
    rng = np.random.default_rng(20260514)
    h, w = 256, 256
    image = rng.normal(loc=0.05, scale=0.01, size=(h, w)).astype(np.float32)
    center_y, center_x = h // 2, w // 2
    offsets = [(0, 32), (0, -32), (26, 20), (-26, -20), (22, -18), (-22, 18)]
    for dy, dx in offsets:
        y = center_y + dy
        x = center_x + dx
        if 2 <= y < h - 2 and 2 <= x < w - 2:
            image[y - 1 : y + 2, x - 1 : x + 2] += 0.9
    return _write_npy_array(image, prefix="autotem_b2_demo_")


def _fft_preview(image: np.ndarray) -> np.ndarray:
    fft = np.fft.fftshift(np.fft.fft2(image))
    mag = np.log1p(np.abs(fft))
    max_val = float(np.max(mag))
    if max_val > 0:
        mag = mag / max_val
    return mag


def _to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    buffer = io.StringIO()
    buffer.write(",".join(keys) + "\n")
    for row in rows:
        values = [str(row.get(key, "")) for key in keys]
        buffer.write(",".join(values) + "\n")
    return buffer.getvalue()


def _extract_auto_pixel_size_nm(dataset) -> tuple[float | None, str]:
    # Prefer normalized metadata scale first, then dataset.pixel_size.
    metadata = dataset.metadata if isinstance(dataset.metadata, dict) else {}
    normalized = metadata.get("normalized", {}) if isinstance(metadata, dict) else {}
    if isinstance(normalized, dict):
        value = normalized.get("pixel_size_nm")
        try:
            if value is not None and float(value) > 0:
                source_fmt = normalized.get("source_format") or "metadata"
                return float(value), f"normalized metadata ({source_fmt})"
        except (TypeError, ValueError):
            pass

    if dataset.pixel_size is not None and dataset.pixel_size.value is not None:
        try:
            unit = str(dataset.pixel_size.unit or "").lower()
            value = float(dataset.pixel_size.value)
            if value > 0 and "nm" in unit:
                return value, f"dataset.pixel_size ({dataset.pixel_size.unit})"
            if value > 0 and unit.startswith("pm"):
                return value * 0.001, f"dataset.pixel_size ({dataset.pixel_size.unit} -> nm)"
            if value > 0 and ("angstrom" in unit or unit == "a"):
                return value * 0.1, f"dataset.pixel_size ({dataset.pixel_size.unit} -> nm)"
        except (TypeError, ValueError):
            pass

    return None, "not found"


def _apply_fft_display_adjustments(
    fft_image: np.ndarray,
    *,
    brightness: float,
    contrast: float,
    gamma: float,
) -> np.ndarray:
    img = np.asarray(fft_image, dtype=np.float32)
    img = img * max(contrast, 0.0) + brightness
    img = np.clip(img, 0.0, 1.0)
    safe_gamma = max(gamma, 1e-3)
    img = np.power(img, safe_gamma)
    return np.clip(img, 0.0, 1.0)


def _deduplicate_points(
    points_xy: list[tuple[float, float]],
    *,
    min_distance: float = 1.0,
) -> list[tuple[float, float]]:
    selected: list[tuple[float, float]] = []
    for x, y in points_xy:
        if any(np.hypot(x - sx, y - sy) < min_distance for sx, sy in selected):
            continue
        selected.append((x, y))
    return selected


def _make_index(dx: float, dy: float) -> tuple[int, int, int]:
    denom = max(abs(dx), abs(dy), 1.0)
    h = int(np.rint(dx / denom * 4))
    k = int(np.rint(dy / denom * 4))
    return h, k, 0


def _phase_candidates_from_d_values(
    d_values_nm: list[float],
    *,
    allowed_phase_names: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    if not d_values_nm:
        return []
    allowed = set(allowed_phase_names)
    rows: list[dict[str, Any]] = []
    for name, formula, identifier, d_refs in PHASE_REFERENCE_DB:
        if allowed and name not in allowed:
            continue
        errors = []
        for d in d_values_nm[: min(8, len(d_values_nm))]:
            nearest = min(d_refs, key=lambda ref: abs(ref - d))
            errors.append(abs(nearest - d) / nearest)
        mean_rel_error = float(np.mean(errors))
        score = float(np.exp(-mean_rel_error * 4.0))
        rows.append(
            {
                "name": name,
                "formula": formula,
                "score": round(max(0.0, min(1.0, score)), 6),
                "matched_spot_count": min(len(d_values_nm), len(d_refs)),
                "zone_axis": (0, 0, 1),
                "mean_d_error": round(mean_rel_error, 6),
                "source": "A2 baseline reference",
                "identifier": identifier,
            }
        )
    rows.sort(key=lambda item: float(item["score"]), reverse=True)
    return rows[: max(1, int(top_n))]


def _select_basis_indices_ui(
    gxy: list[tuple[float, float]],
    intensities: list[float],
) -> tuple[int, int] | None:
    if len(gxy) < 2:
        return None
    ranked = sorted(
        range(len(gxy)),
        key=lambda idx: float(intensities[idx]),
        reverse=True,
    )[: min(12, len(gxy))]
    best_pair: tuple[int, int] | None = None
    best_score = -np.inf
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            idx1 = ranked[i]
            idx2 = ranked[j]
            v1 = np.array(gxy[idx1], dtype=np.float64)
            v2 = np.array(gxy[idx2], dtype=np.float64)
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 <= 1e-9 or n2 <= 1e-9:
                continue
            cosang = float(np.dot(v1, v2) / (n1 * n2))
            cosang = max(-1.0, min(1.0, cosang))
            angle = float(np.arccos(cosang))
            if angle < np.deg2rad(10.0) or angle > np.deg2rad(170.0):
                continue
            bmat = np.column_stack((v1, v2))
            cond = float(np.linalg.cond(bmat))
            if not np.isfinite(cond) or cond > 1000:
                continue
            score = float(np.sin(angle) * np.sqrt(n1 * n2) / (1.0 + 0.02 * cond))
            if score > best_score:
                best_score = score
                best_pair = (idx1, idx2)
    return best_pair


def _index_points_lattice_ui(
    gxy: list[tuple[float, float]],
    intensities: list[float],
) -> tuple[list[tuple[int, int, int, float, float]], dict[str, Any]]:
    basis_pair = _select_basis_indices_ui(gxy, intensities)
    if basis_pair is None:
        return [], {"indexing_method": "lattice_fit_fallback"}
    i1, i2 = basis_pair
    basis_1 = np.array(gxy[i1], dtype=np.float64)
    basis_2 = np.array(gxy[i2], dtype=np.float64)
    bmat = np.column_stack((basis_1, basis_2))
    basis_scale = max(float(np.linalg.norm(basis_1)), float(np.linalg.norm(basis_2)), 1e-9)

    indexed: list[tuple[int, int, int, float, float]] = []
    for gx, gy in gxy:
        gvec = np.array([gx, gy], dtype=np.float64)
        coeff = np.linalg.solve(bmat, gvec)
        rounded = np.rint(coeff)
        h = int(rounded[0])
        k = int(rounded[1])
        recon = bmat @ rounded
        residual = float(np.linalg.norm(gvec - recon))
        coeff_err = float(np.linalg.norm(coeff - rounded))
        confidence = float(
            np.exp(-residual / (0.12 * basis_scale + 1e-9)) * np.exp(-coeff_err / 0.5)
        )
        confidence = max(0.05, min(0.95, confidence))
        indexed.append((h, k, 0, residual, confidence))
    meta = {
        "indexing_method": "lattice_fit",
        "basis_indices": [i1, i2],
    }
    return indexed, meta


def _analyze_selected_spots(
    *,
    points_xy: list[tuple[float, float]],
    fft_magnitude: np.ndarray,
    pixel_size_nm: float,
    phase_top_n: int,
    candidate_phases: list[str],
    indexing_mode: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
    list[float],
    dict[str, Any],
]:
    h, w = fft_magnitude.shape
    cy, cx = h / 2.0, w / 2.0
    reciprocal_scale = 1.0 / (min(h, w) * max(pixel_size_nm, 1e-9))

    base_rows: list[dict[str, Any]] = []
    d_values_nm: list[float] = []
    gxy: list[tuple[float, float]] = []
    intensities: list[float] = []

    for _, (x, y) in enumerate(points_xy):
        xi = int(np.clip(round(x), 0, w - 1))
        yi = int(np.clip(round(y), 0, h - 1))
        intensity = float(fft_magnitude[yi, xi])
        intensities.append(intensity)

        dx = float(x - cx)
        dy = float(y - cy)
        gx = dx * reciprocal_scale
        gy = dy * reciprocal_scale
        gxy.append((gx, gy))
        g_mag = float(np.hypot(gx, gy))
        d_nm = 0.0 if g_mag <= 1e-8 else 1.0 / g_mag
        if d_nm > 0:
            d_values_nm.append(d_nm)
        hkl = _make_index(dx, dy)

        base_rows.append(
            {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "intensity": round(intensity, 6),
                "gx_1_per_nm": round(gx, 6),
                "gy_1_per_nm": round(gy, 6),
                "g_mag_1_per_nm": round(g_mag, 6),
                "d_nm": round(d_nm, 6),
                "h_baseline": hkl[0],
                "k_baseline": hkl[1],
                "l_baseline": hkl[2],
            }
        )

    lattice_rows, lattice_meta = _index_points_lattice_ui(gxy, intensities)
    residuals_lattice: list[float] = (
        [float(item[3]) for item in lattice_rows] if lattice_rows else []
    )

    compare_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    mode = indexing_mode.strip().lower()
    for idx, row in enumerate(base_rows):
        h_l = ""
        k_l = ""
        l_l = ""
        residual = None
        conf_l = None
        if idx < len(lattice_rows):
            h_l, k_l, l_l, residual, conf_l = lattice_rows[idx]

        compare_rows.append(
            {
                "peak_index": idx,
                "h_baseline": row["h_baseline"],
                "k_baseline": row["k_baseline"],
                "h_lattice_fit": h_l,
                "k_lattice_fit": k_l,
                "residual_lattice_1_per_nm": (
                    round(float(residual), 6) if residual is not None else ""
                ),
                "confidence_lattice": (
                    round(float(conf_l), 6) if conf_l is not None else ""
                ),
            }
        )

        if mode == "lattice_fit" and idx < len(lattice_rows):
            h_p, k_p, l_p, residual_p, conf_p = lattice_rows[idx]
            peak_rows.append(
                {
                    "peak_index": idx,
                    "x": row["x"],
                    "y": row["y"],
                    "intensity": row["intensity"],
                    "gx_1_per_nm": row["gx_1_per_nm"],
                    "gy_1_per_nm": row["gy_1_per_nm"],
                    "g_mag_1_per_nm": row["g_mag_1_per_nm"],
                    "d_nm": row["d_nm"],
                    "h": h_p,
                    "k": k_p,
                    "l": l_p,
                    "residual_error_1_per_nm": round(float(residual_p), 6),
                    "confidence": round(float(conf_p), 6),
                }
            )
        else:
            peak_rows.append(
                {
                    "peak_index": idx,
                    "x": row["x"],
                    "y": row["y"],
                    "intensity": row["intensity"],
                    "gx_1_per_nm": row["gx_1_per_nm"],
                    "gy_1_per_nm": row["gy_1_per_nm"],
                    "g_mag_1_per_nm": row["g_mag_1_per_nm"],
                    "d_nm": row["d_nm"],
                    "h": row["h_baseline"],
                    "k": row["k_baseline"],
                    "l": row["l_baseline"],
                    "residual_error_1_per_nm": (
                        round(float(residual), 6) if residual is not None and mode == "auto" else ""
                    ),
                    "confidence": (
                        round(float(conf_l), 6) if conf_l is not None and mode == "auto" else 0.4
                    ),
                }
            )

    phase_rows = _phase_candidates_from_d_values(
        d_values_nm,
        allowed_phase_names=list(candidate_phases),
        top_n=phase_top_n,
    )
    confidence = 0.25 + min(0.5, len(peak_rows) / 40.0)
    if phase_rows:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence))
    return peak_rows, compare_rows, phase_rows, confidence, residuals_lattice, lattice_meta


def _to_fft_display_image(
    fft_img_display: np.ndarray,
    *,
    max_side: int = 900,
) -> tuple[Image.Image, float, float]:
    h, w = fft_img_display.shape
    scale = min(max_side / max(h, w), 1.0)
    disp_h = max(1, int(round(h * scale)))
    disp_w = max(1, int(round(w * scale)))
    arr = (np.clip(fft_img_display, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB").resize(
        (disp_w, disp_h),
        resample=Image.BILINEAR,
    )
    sx = w / disp_w
    sy = h / disp_h
    return img, sx, sy


def _extract_canvas_points_xy(
    canvas_result: Any,
    *,
    sx: float,
    sy: float,
) -> list[tuple[float, float]]:
    json_data = getattr(canvas_result, "json_data", None)
    if not isinstance(json_data, dict):
        return []
    objects = json_data.get("objects")
    if not isinstance(objects, list):
        return []

    points: list[tuple[float, float]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        left = obj.get("left")
        top = obj.get("top")
        radius = obj.get("radius")
        width = obj.get("width")
        height = obj.get("height")
        try:
            lx = float(left)
            ty = float(top)
        except (TypeError, ValueError):
            continue

        # point mode is rendered as circles; use center for robustness.
        if radius is not None:
            try:
                r = float(radius)
            except (TypeError, ValueError):
                r = 0.0
            disp_x = lx + r
            disp_y = ty + r
        elif width is not None and height is not None:
            try:
                w = float(width)
                h = float(height)
            except (TypeError, ValueError):
                w = 0.0
                h = 0.0
            disp_x = lx + 0.5 * w
            disp_y = ty + 0.5 * h
        else:
            disp_x = lx
            disp_y = ty

        points.append((disp_x * sx, disp_y * sy))
    return points


def _generate_lattice_points_from_two_basis(
    *,
    center_xy: tuple[float, float],
    basis1_xy: tuple[float, float],
    basis2_xy: tuple[float, float],
    shape_hw: tuple[int, int],
    max_order: int,
    min_distance: float,
) -> list[tuple[float, float]]:
    cy, cx = center_xy[1], center_xy[0]
    p1x, p1y = basis1_xy
    p2x, p2y = basis2_xy
    v1x, v1y = p1x - cx, p1y - cy
    v2x, v2y = p2x - cx, p2y - cy

    h, w = shape_hw
    points: list[tuple[float, float]] = []
    for m in range(-max_order, max_order + 1):
        for n in range(-max_order, max_order + 1):
            if m == 0 and n == 0:
                continue
            x = cx + m * v1x + n * v2x
            y = cy + m * v1y + n * v2y
            if 0.0 <= x < float(w) and 0.0 <= y < float(h):
                points.append((x, y))

    points = _deduplicate_points(points, min_distance=min_distance)
    points.sort(key=lambda p: float(np.hypot(p[0] - cx, p[1] - cy)))
    return points


def main() -> None:
    st.set_page_config(page_title="AutoTEM B2 Diffraction UI", layout="wide")
    st.title("AutoTEM-Insight: B2 FFT / Diffraction Test UI")
    st.caption(
        "Upload a TEM image, tune B2 parameters, and inspect FFT peaks, "
        "d-spacing, indexing, and phase candidates."
    )

    with st.sidebar:
        st.subheader("Input")
        uploaded = st.file_uploader(
            "Upload image/array",
            type=[
                "npy",
                "txt",
                "csv",
                "tif",
                "tiff",
                "png",
                "jpg",
                "jpeg",
                "dm",
                "dm3",
                "dm4",
                "mrc",
                "h5",
                "hdf5",
                "emd",
            ],
        )
        use_demo = st.button("Use Built-In Demo Sample")

        st.subheader("Calibration")
        auto_scale = st.checkbox("Auto scale from DM/TIFF metadata", value=True)
        pixel_size_nm_manual = st.number_input(
            "Manual pixel size (nm/pixel, fallback)",
            min_value=1e-6,
            value=0.1,
            step=0.01,
            format="%.6f",
        )
        voltage_kv = st.number_input("Voltage (kV)", min_value=1.0, value=200.0, step=10.0)

        st.subheader("Preprocess")
        recipe = st.selectbox("Recipe", ["auto", "min_max"], index=0)
        use_clip = st.checkbox("Enable percentile clipping", value=False)
        clip_low = st.number_input(
            "Clip low percentile",
            min_value=0.0,
            max_value=99.0,
            value=1.0,
            step=0.5,
            disabled=not use_clip,
        )
        clip_high = st.number_input(
            "Clip high percentile",
            min_value=1.0,
            max_value=100.0,
            value=99.0,
            step=0.5,
            disabled=not use_clip,
        )

        st.subheader("Diffraction")
        top_k_peaks = st.slider("Top-k peaks", min_value=1, max_value=128, value=24, step=1)
        min_peak_distance = st.slider(
            "Min peak distance (px)",
            min_value=0.0,
            max_value=30.0,
            value=3.0,
            step=0.5,
        )
        center_exclusion_radius = st.slider(
            "Center exclusion radius (px)",
            min_value=0.0,
            max_value=30.0,
            value=3.0,
            step=0.5,
        )
        indexing_mode = st.selectbox(
            "indexing_mode",
            options=["auto", "lattice_fit", "baseline"],
            index=0,
        )
        phase_top_n = st.slider("Phase top-n", min_value=1, max_value=8, value=3, step=1)
        candidate_phases = st.multiselect(
            "Candidate phases",
            options=PHASE_OPTIONS,
            default=PHASE_OPTIONS[:3],
        )

        st.subheader("FFT Display (Gray)")
        fft_brightness = st.slider(
            "Brightness",
            min_value=-1.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )
        fft_contrast = st.slider("Contrast", min_value=0.1, max_value=3.0, value=1.0, step=0.05)
        fft_gamma = st.slider("Gamma", min_value=0.2, max_value=3.0, value=1.0, step=0.05)

    input_path: Path | None = None
    if use_demo:
        input_path = _build_demo_input()
    elif uploaded is not None:
        input_path = _materialize_upload(uploaded.name, uploaded.getvalue())

    if input_path is None:
        st.info("Upload a file or click 'Use Built-In Demo Sample' to start B2 testing.")
        return

    run_clicked = st.button("Run B2 Analysis", type="primary")
    if run_clicked:
        try:
            dataset, original_image, npy_path = _build_dataset_for_b2(input_path)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load input: {exc}")
            return

        auto_pixel_size_nm, auto_scale_source = _extract_auto_pixel_size_nm(dataset)
        if auto_scale and auto_pixel_size_nm is not None:
            effective_pixel_size_nm = auto_pixel_size_nm
            scale_note = (
                f"Scale auto-calibrated: {effective_pixel_size_nm:.6f} nm/pixel "
                f"(source: {auto_scale_source})."
            )
        else:
            effective_pixel_size_nm = float(pixel_size_nm_manual)
            if auto_scale:
                scale_note = (
                    "Auto scale was requested but metadata scale was not found; "
                    f"fallback to manual {effective_pixel_size_nm:.6f} nm/pixel."
                )
            else:
                scale_note = f"Using manual scale: {effective_pixel_size_nm:.6f} nm/pixel."

        try:
            quality = assess_quality(dataset)
            calibration_params = {
                "pixel_size_nm": effective_pixel_size_nm,
                "voltage_kv": float(voltage_kv),
            }
            calibration = calibrate_dataset(dataset, parameters=calibration_params)

            preprocess_params: dict[str, Any] = {}
            if use_clip:
                preprocess_params = {
                    "clip_percentile_low": float(clip_low),
                    "clip_percentile_high": float(clip_high),
                }
            preprocessing = preprocess_dataset(
                dataset,
                quality_after=quality,
                recipe=recipe,
                parameters=preprocess_params,
            )

            diffraction_params = {
                "top_k_peaks": int(top_k_peaks),
                "min_peak_distance": float(min_peak_distance),
                "center_exclusion_radius": float(center_exclusion_radius),
                "indexing_mode": str(indexing_mode),
                "phase_top_n": int(phase_top_n),
                "candidate_phases": list(candidate_phases),
            }
            diffraction = run_diffraction_analysis(
                preprocessing,
                calibration,
                parameters=diffraction_params,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"B2 analysis failed: {exc}")
            return

        processed_image = np.load(npy_path, allow_pickle=False).astype(np.float32)
        fft_complex = np.fft.fftshift(np.fft.fft2(processed_image))
        fft_magnitude = np.abs(fft_complex).astype(np.float32)
        fft_img = _fft_preview(processed_image)

        auto_points_xy = [(float(peak.x), float(peak.y)) for peak in diffraction.peaks]
        if "b2_canvas_nonce" not in st.session_state:
            st.session_state["b2_canvas_nonce"] = 0
        st.session_state["b2_context"] = {
            "original_image": original_image,
            "processed_image": processed_image,
            "fft_img": fft_img,
            "fft_magnitude": fft_magnitude,
            "auto_points_xy": auto_points_xy,
            "effective_pixel_size_nm": float(effective_pixel_size_nm),
            "phase_top_n": int(phase_top_n),
            "candidate_phases": list(candidate_phases),
            "dataset_warnings": list(dataset.warnings),
            "quality_warnings": list(quality.warnings),
            "calibration_warnings": list(calibration.warnings),
            "preprocessing_warnings": list(preprocessing.warnings),
            "diffraction_warnings": list(diffraction.warnings),
            "scale_note": scale_note,
        }

    context = st.session_state.get("b2_context")
    if context is None:
        st.write(f"Current input: `{input_path}`")
        return

    scale_note = str(context.get("scale_note", ""))
    if scale_note.startswith("Scale auto-calibrated"):
        st.info(scale_note)
    else:
        st.warning(scale_note)

    processed_image = np.asarray(context["processed_image"], dtype=np.float32)
    fft_img = np.asarray(context["fft_img"], dtype=np.float32)
    fft_img_display = _apply_fft_display_adjustments(
        fft_img,
        brightness=float(fft_brightness),
        contrast=float(fft_contrast),
        gamma=float(fft_gamma),
    )
    fft_magnitude = np.asarray(context["fft_magnitude"], dtype=np.float32)
    auto_points_xy = list(context["auto_points_xy"])

    st.subheader("Manual Basis Selection On FFT")
    st.caption(
        "直接在 FFT 图上点击两个基点 g1/g2。其余斑点将按矢量叠加自动生成。"
    )
    selection_mode = st.radio(
        "Spot source mode",
        options=["Auto peaks", "Two-point vector synthesis"],
        horizontal=True,
    )
    lattice_order = st.slider("Vector expansion order", 1, 8, 3, 1)
    lattice_min_distance = st.slider(
        "Lattice point min distance (px)",
        0.5,
        8.0,
        1.5,
        0.1,
    )

    fft_canvas_img, sx, sy = _to_fft_display_image(fft_img_display, max_side=900)
    canvas_col1, canvas_col2 = st.columns([1, 1])
    with canvas_col1:
        st.write("Click two basis points on FFT:")
        canvas_key = f"b2-fft-canvas-{int(st.session_state.get('b2_canvas_nonce', 0))}"
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.25)",
            stroke_width=2,
            stroke_color="#00ffaa",
            background_image=fft_canvas_img,
            drawing_mode="point",
            point_display_radius=4,
            display_toolbar=True,
            update_streamlit=True,
            height=fft_canvas_img.height,
            width=fft_canvas_img.width,
            key=canvas_key,
        )
        clear_canvas = st.button("Clear Basis Points")
        if clear_canvas:
            st.session_state["b2_canvas_nonce"] = int(
                st.session_state.get("b2_canvas_nonce", 0)
            ) + 1
            st.rerun()

    clicked_points_xy = _extract_canvas_points_xy(canvas_result, sx=sx, sy=sy)
    with canvas_col2:
        st.write(f"Clicked points: {len(clicked_points_xy)}")
        if clicked_points_xy:
            clicked_text = "\n".join(
                f"{idx+1}: x={x:.2f}, y={y:.2f}"
                for idx, (x, y) in enumerate(clicked_points_xy)
            )
            st.code(
                clicked_text,
                language="text",
            )

    selected_points_xy: list[tuple[float, float]]
    if selection_mode == "Auto peaks":
        selected_points_xy = list(auto_points_xy)
    else:
        if len(clicked_points_xy) < 2:
            st.warning("请在 FFT 图上至少点击两个点作为基矢 g1/g2。")
            selected_points_xy = []
        else:
            if len(clicked_points_xy) > 2:
                st.warning("检测到超过两个点，当前仅使用前两个点作为基矢。")
            basis1 = clicked_points_xy[0]
            basis2 = clicked_points_xy[1]
            h, w = fft_magnitude.shape
            center_xy = (w / 2.0, h / 2.0)
            selected_points_xy = _generate_lattice_points_from_two_basis(
                center_xy=center_xy,
                basis1_xy=basis1,
                basis2_xy=basis2,
                shape_hw=(h, w),
                max_order=int(lattice_order),
                min_distance=float(lattice_min_distance),
            )
            # Keep the two manually selected basis points explicitly.
            selected_points_xy = _deduplicate_points(
                [basis1, basis2] + selected_points_xy,
                min_distance=float(lattice_min_distance),
            )

    (
        peak_rows,
        compare_rows,
        phase_rows,
        selected_confidence,
        residuals_lattice,
        lattice_meta,
    ) = _analyze_selected_spots(
        points_xy=selected_points_xy,
        fft_magnitude=fft_magnitude,
        pixel_size_nm=float(context["effective_pixel_size_nm"]),
        phase_top_n=int(context["phase_top_n"]),
        candidate_phases=list(context["candidate_phases"]),
        indexing_mode=str(indexing_mode),
    )

    st.success("B2 analysis finished.")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Detected Peaks", len(peak_rows))
    metric_col2.metric("Indexed Spots", sum(1 for row in peak_rows if float(row["d_nm"]) > 0))
    metric_col3.metric("Phase Candidates", len(phase_rows))
    metric_col4.metric("Confidence", f"{selected_confidence:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=150)
    ax1.imshow(np.asarray(context["original_image"], dtype=np.float32), cmap="gray")
    ax1.set_title("Input Image")
    ax1.axis("off")

    ax2.imshow(fft_img_display, cmap="gray")
    if selected_points_xy:
        peak_xy = np.array(selected_points_xy, dtype=float)
        ax2.scatter(
            peak_xy[:, 0],
            peak_xy[:, 1],
            s=22,
            c="cyan",
            linewidths=0.3,
            edgecolors="black",
        )
    ax2.set_title("FFT + Detected Peaks")
    ax2.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.subheader("Peaks / g-vectors / d-spacing / hkl")
    st.dataframe(peak_rows, use_container_width=True)

    st.subheader("Baseline vs Lattice-Fit Comparison")
    st.caption(f"Lattice-fit status: {lattice_meta.get('indexing_method', 'unknown')}")
    st.dataframe(compare_rows, use_container_width=True)

    st.subheader("Residual Distribution (Lattice-Fit)")
    if residuals_lattice:
        fig_res, ax_res = plt.subplots(figsize=(6, 3), dpi=150)
        ax_res.hist(residuals_lattice, bins=min(20, max(5, len(residuals_lattice) // 2)))
        ax_res.set_xlabel("Residual (1/nm)")
        ax_res.set_ylabel("Count")
        ax_res.set_title("Lattice-Fit Residual Histogram")
        fig_res.tight_layout()
        st.pyplot(fig_res, clear_figure=True)
    else:
        st.info("No lattice-fit residuals available for the current selection.")

    st.subheader("Phase Candidates")
    st.dataframe(phase_rows, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download peaks CSV",
            data=_to_csv(peak_rows),
            file_name="diffraction_peaks_ui.csv",
            mime="text/csv",
        )
    with col_b:
        st.download_button(
            "Download phase candidates CSV",
            data=_to_csv(phase_rows),
            file_name="phase_candidates_ui.csv",
            mime="text/csv",
        )

    warnings = []
    warnings.extend(list(context.get("dataset_warnings", [])))
    warnings.extend(list(context.get("quality_warnings", [])))
    warnings.extend(list(context.get("calibration_warnings", [])))
    warnings.extend(list(context.get("preprocessing_warnings", [])))
    warnings.extend(list(context.get("diffraction_warnings", [])))
    if not selected_points_xy:
        warnings.append("No diffraction spot selected after manual filter.")
    if warnings:
        st.subheader("Warnings")
        for warning in dict.fromkeys(warnings):
            st.warning(warning)


if __name__ == "__main__":
    main()
