from __future__ import annotations

import io
import json
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
from autotem.diffraction.phase_lookup import lookup_phase_candidates
from autotem.diffraction.phase_mvp import (
    PhaseEntry,
    load_cif_phase_entries,
    match_candidates,
    plot_annotated_fft,
    plot_atom_projection,
    plot_experimental_vs_simulated_overlay,
    plot_simulated_diffraction,
)
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
FFT_DETECTOR_BUILD_TAG = "b2_fft_peak_v2_2026-05-15"
PHASE_REFERENCE_DB: list[tuple[str, str, str, tuple[float, ...]]] = [
    ("Silicon", "Si", "mp-149", (0.313, 0.192, 0.163, 0.136)),
    ("Gold", "Au", "mp-81", (0.235, 0.204, 0.144, 0.123)),
    ("Graphite", "C", "mp-48", (0.335, 0.213, 0.168, 0.142)),
    ("Titanium Dioxide", "TiO2", "mp-2657", (0.325, 0.249, 0.190, 0.169)),
]

INDEXING_MODE_LABELS: dict[str, str] = {
    "auto": "auto (auto select)",
    "lattice_fit": "lattice_fit (vector lattice fit)",
    "baseline": "baseline (basic indexing)",
    "b4_mvp": "b4_mvp (lattice-parameter calibration)",
    "b4_strict": "b4_strict (CIF-model calibration)",
}

INDEXING_MODE_HINTS: dict[str, str] = {
    "auto": "Automatically choose baseline or lattice-fit for quick exploration.",
    "lattice_fit": "Fit two major vectors and index spots on the fitted lattice.",
    "baseline": "Fast baseline h,k,l assignment with lower reliability.",
    "b4_mvp": (
        "B4 MVP: generate theoretical d_hkl from lattice parameters "
        "(a,b,c,alpha,beta,gamma) and match measured spots."
    ),
    "b4_strict": (
        "B4 Strict: CIF-model based calibration with extinction checks, "
        "Friedel-pair consistency, and zone-axis consistency scoring."
    ),
}

DEFAULT_B4_CANDIDATE_JSON = (
    '[{"name":"Silicon_like","a":0.543,"b":0.543,"c":0.543,'
    '"alpha":90,"beta":90,"gamma":90,"space_group":"Fd-3m"}]'
)

FALLBACK_PHASE_LATTICE: dict[str, dict[str, Any]] = {
    "Silicon": {
        "formula": "Si",
        "a": 0.5431,
        "b": 0.5431,
        "c": 0.5431,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fd-3m",
    },
    "Gold": {
        "formula": "Au",
        "a": 0.4078,
        "b": 0.4078,
        "c": 0.4078,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fm-3m",
    },
    "Graphite": {
        "formula": "C",
        "a": 0.2460,
        "b": 0.2460,
        "c": 0.6710,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 120.0,
        "space_group": "P63/mmc",
    },
    "Titanium Dioxide": {
        "formula": "TiO2",
        "a": 0.4594,
        "b": 0.4594,
        "c": 0.2958,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "P42/mnm",
    },
}


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


def _parse_elements_text(raw: str) -> list[str]:
    tokens = [token.strip() for token in str(raw).replace(";", ",").split(",")]
    return [token for token in tokens if token]


def _parse_zone_axis_text(raw: str) -> tuple[int, int, int]:
    parts = [token.strip() for token in str(raw).replace(";", ",").split(",")]
    vals: list[int] = []
    for token in parts:
        if not token:
            continue
        try:
            vals.append(int(token))
        except ValueError:
            continue
    if len(vals) >= 3:
        return (vals[0], vals[1], vals[2])
    return (0, 0, 1)


def _phase_entries_from_candidate_params(
    candidate_phase_params: list[dict[str, Any]],
    *,
    selected_names: set[str] | None = None,
) -> tuple[list[PhaseEntry], list[str]]:
    warnings: list[str] = []
    rows: list[PhaseEntry] = []
    for item in candidate_phase_params:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        if selected_names is not None and name not in selected_names:
            continue
        try:
            a_val = float(item.get("a"))
            b_val = float(item.get("b"))
            c_val = float(item.get("c"))
            # Backward compatibility: older auto-fetch values were in Angstrom.
            if max(a_val, b_val, c_val) > 3.0:
                a_val *= 0.1
                b_val *= 0.1
                c_val *= 0.1
                warnings.append(
                    f"{name}: lattice lengths look like Angstrom, auto-converted to nm."
                )
            entry = PhaseEntry(
                name=name,
                path="candidate_phase_params",
                a=a_val,
                b=b_val,
                c=c_val,
                alpha_deg=float(item.get("alpha")),
                beta_deg=float(item.get("beta")),
                gamma_deg=float(item.get("gamma")),
                space_group=str(item.get("space_group", "P1")),
                formula=(
                    None
                    if item.get("formula") is None
                    else str(item.get("formula"))
                ),
                atoms=[],
            )
            rows.append(entry)
        except (TypeError, ValueError):
            warnings.append(f"Invalid lattice params in candidate_phase_params: {name}")
    if not rows:
        warnings.append("No valid lattice candidates in candidate_phase_params.")
    return rows, warnings


def _fallback_phase_entries(
    *,
    selected_names: set[str] | None = None,
) -> list[PhaseEntry]:
    out: list[PhaseEntry] = []
    for name, payload in FALLBACK_PHASE_LATTICE.items():
        if selected_names is not None and name not in selected_names:
            continue
        out.append(
            PhaseEntry(
                name=name,
                path="builtin_fallback",
                a=float(payload["a"]),
                b=float(payload["b"]),
                c=float(payload["c"]),
                alpha_deg=float(payload["alpha"]),
                beta_deg=float(payload["beta"]),
                gamma_deg=float(payload["gamma"]),
                space_group=str(payload["space_group"]),
                formula=str(payload["formula"]),
                atoms=[],
            )
        )
    return out


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


def _fig_to_bytes(fig: plt.Figure, *, fmt: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    return buf.getvalue()


def _as_array_from_debug(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    return arr


def _build_core_peak_rows(
    diffraction_obj: Any,
    *,
    pixel_size_nm: float,
) -> tuple[list[dict[str, Any]], list[float]]:
    if diffraction_obj is None:
        return [], []
    center = getattr(diffraction_obj, "center", None)
    if not isinstance(center, (list, tuple)) or len(center) != 2:
        return [], []
    cx, cy = float(center[0]), float(center[1])
    peaks = list(getattr(diffraction_obj, "peaks", []))
    g_vectors = list(getattr(diffraction_obj, "g_vectors", []))
    indexed_spots = list(getattr(diffraction_obj, "indexed_spots", []))
    idx_map = {int(getattr(row, "peak_index", -1)): row for row in indexed_spots}
    prov = getattr(diffraction_obj, "provenance", {})
    pd = prov.get("peak_detection", {}) if isinstance(prov, dict) else {}
    details = pd.get("peak_details", []) if isinstance(pd, dict) else []
    detail_map: dict[int, dict[str, Any]] = {}
    for row in details:
        if not isinstance(row, dict):
            continue
        try:
            detail_map[int(row.get("peak_id", -1))] = row
        except (TypeError, ValueError):
            continue

    rows: list[dict[str, Any]] = []
    residuals: list[float] = []
    for i, peak in enumerate(peaks):
        x = float(peak.x)
        y = float(peak.y)
        detail = detail_map.get(i, {})
        g_row = g_vectors[i] if i < len(g_vectors) else None
        if g_row is None:
            gx = 0.0
            gy = 0.0
        else:
            gx = float(getattr(g_row.gx, "value", 0.0))
            gy = float(getattr(g_row.gy, "value", 0.0))
        g_norm = float(np.hypot(gx, gy))
        d_nm = 0.0 if g_norm <= 1e-9 else 1.0 / g_norm

        indexed = idx_map.get(i)
        if indexed is None:
            h_val = int(detail.get("h", 0))
            k_val = int(detail.get("k", 0))
            l_val = int(detail.get("l", 0))
            residual_obj = None
            conf_val = getattr(peak, "confidence", None)
        else:
            h_val = int(indexed.h)
            k_val = int(indexed.k)
            l_val = int(indexed.l)
            residual_obj = indexed.error
            conf_val = indexed.confidence
        residual_val = None if residual_obj is None else getattr(residual_obj, "value", None)
        if residual_val is not None:
            try:
                residuals.append(float(residual_val))
            except (TypeError, ValueError):
                pass
        theta_fallback = np.degrees(np.arctan2(y - cy, x - cx))
        rows.append(
            {
                "peak_id": i,
                "x": round(float(detail.get("x", x)), 4),
                "y": round(float(detail.get("y", y)), 4),
                "refined_x": round(float(detail.get("refined_x", x)), 4),
                "refined_y": round(float(detail.get("refined_y", y)), 4),
                "dx": round(float(detail.get("dx", x - cx)), 4),
                "dy": round(float(detail.get("dy", y - cy)), 4),
                "r_px": round(float(detail.get("r_px", np.hypot(x - cx, y - cy))), 4),
                "theta_deg": round(float(detail.get("theta_deg", theta_fallback)), 3),
                "g_x": round(gx, 6),
                "g_y": round(gy, 6),
                "g_norm": round(g_norm, 6),
                "d_spacing": round(d_nm, 6),
                "gx_1_per_nm": round(gx, 6),
                "gy_1_per_nm": round(gy, 6),
                "g_mag_1_per_nm": round(g_norm, 6),
                "d_nm": round(d_nm, 6),
                "h": h_val,
                "k": k_val,
                "l": l_val,
                "intensity": round(float(getattr(peak, "intensity", 0.0) or 0.0), 6),
                "snr": round(float(detail.get("snr", 0.0)), 4),
                "local_background": round(float(detail.get("local_background", 0.0)), 6),
                "subpixel_shift_px": round(float(detail.get("shift_px", 0.0)), 4),
                "residual_error": (
                    "" if residual_val is None else round(float(residual_val), 6)
                ),
                "residual_error_1_per_nm": (
                    "" if residual_val is None else round(float(residual_val), 6)
                ),
                "confidence": (
                    "" if conf_val is None else round(float(conf_val), 6)
                ),
                "paired": bool(detail.get("paired", False)),
                "pair_error_px": (
                    ""
                    if detail.get("pair_error_px") is None
                    else round(float(detail.get("pair_error_px")), 4)
                ),
                "warning": str(detail.get("warning", "")),
                "algo_build_tag": FFT_DETECTOR_BUILD_TAG,
            }
        )
    return rows, residuals


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


def _extract_zone_axis_text(
    phase_rows: list[dict[str, Any]],
    *,
    fallback: tuple[int, int, int] = (0, 0, 1),
) -> str:
    if phase_rows:
        raw = phase_rows[0].get("zone_axis")
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            try:
                u, v, w = int(raw[0]), int(raw[1]), int(raw[2])
                return f"[{u} {v} {w}]"
            except (TypeError, ValueError):
                pass
    return f"[{fallback[0]} {fallback[1]} {fallback[2]}]"


def _build_simulated_reciprocal_rows(
    peak_rows: list[dict[str, Any]],
    *,
    max_index: int = 4,
) -> list[dict[str, Any]]:
    valid: list[tuple[int, int, int, float, float]] = []
    for row in peak_rows:
        try:
            h = int(row.get("h", 0))
            k = int(row.get("k", 0))
            ell = int(row.get("l", 0))
            gx = float(row.get("gx_1_per_nm", row.get("g_x", 0.0)))
            gy = float(row.get("gy_1_per_nm", row.get("g_y", 0.0)))
        except (TypeError, ValueError):
            continue
        if h == 0 and k == 0 and ell == 0:
            continue
        valid.append((h, k, ell, gx, gy))
    if len(valid) < 3:
        return []

    a_mat = np.array([[h, k, ell] for h, k, ell, _, _ in valid], dtype=np.float64)
    g_mat = np.array([[gx, gy] for _, _, _, gx, gy in valid], dtype=np.float64)
    try:
        coeff, _, _, _ = np.linalg.lstsq(a_mat, g_mat, rcond=None)
    except np.linalg.LinAlgError:
        return []

    rows: list[dict[str, Any]] = []
    for h in range(-max_index, max_index + 1):
        for k in range(-max_index, max_index + 1):
            for ell in range(-max_index, max_index + 1):
                if h == 0 and k == 0 and ell == 0:
                    continue
                vec = np.array([h, k, ell], dtype=np.float64)
                gx, gy = (vec @ coeff).tolist()
                g_mag = float(np.hypot(gx, gy))
                if g_mag <= 1e-8 or g_mag > 40.0:
                    continue
                rows.append(
                    {
                        "h": h,
                        "k": k,
                        "l": ell,
                        "gx_1_per_nm": float(gx),
                        "gy_1_per_nm": float(gy),
                        "g_mag_1_per_nm": g_mag,
                    }
                )
    rows.sort(key=lambda item: float(item["g_mag_1_per_nm"]))
    return rows[:180]


def _phase_rows_from_candidate_params(
    candidate_phase_params: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidate_phase_params:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "name": str(item.get("name", "")),
                "formula": item.get("formula"),
                "score": "",
                "matched_spot_count": 0,
                "zone_axis": "",
                "mean_d_error": "",
                "source": str(item.get("source", "candidate_phase_params")),
                "identifier": item.get("identifier"),
                "status": "candidate_loaded_not_matched",
            }
        )
    return rows


def _build_b4_diagnostic_rows(
    *,
    diffraction_provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    indexing_meta = diffraction_provenance.get("indexing", {})
    if not isinstance(indexing_meta, dict):
        return []
    b4_meta = indexing_meta.get("b4", {})
    if not isinstance(b4_meta, dict):
        return []
    per_phase = b4_meta.get("per_phase_diagnostics", [])
    if not isinstance(per_phase, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in per_phase:
        if not isinstance(item, dict):
            continue
        total_spot_count = int(item.get("total_spot_count", 0))
        valid_d_count = int(item.get("valid_d_spot_count", 0))
        within_d_count = int(item.get("within_d_count", 0))
        within_angle_count = int(item.get("within_angle_count", 0))
        hit_count = int(item.get("matched_within_all_constraints_count", 0))
        rows.append(
            {
                "phase_name": str(item.get("phase_name", "")),
                "anchor_pair_found": bool(item.get("anchor_pair_found", False)),
                "total_spots": total_spot_count,
                "valid_d_spots": valid_d_count,
                "d_pass_count": within_d_count,
                "angle_pass_count": within_angle_count,
                "all_constraints_pass_count": hit_count,
                "d_pass_ratio": (
                    round(within_d_count / max(1, valid_d_count), 3)
                    if valid_d_count > 0
                    else 0.0
                ),
                "angle_pass_ratio": (
                    round(within_angle_count / max(1, total_spot_count), 3)
                    if total_spot_count > 0
                    else 0.0
                ),
                "mean_best_d_error": item.get("mean_best_d_error"),
                "mean_best_angle_error_deg": item.get("mean_best_angle_error_deg"),
            }
        )
    return rows


def _build_b4_diagnostic_summary(
    *,
    diffraction_provenance: dict[str, Any],
) -> list[str]:
    indexing_meta = diffraction_provenance.get("indexing", {})
    if not isinstance(indexing_meta, dict):
        return ["b4_not_executed: indexing metadata missing."]
    if "b4" not in indexing_meta:
        return ["b4_not_executed: B4 branch was not entered in this run."]
    b4_meta = indexing_meta.get("b4", {})
    if not isinstance(b4_meta, dict):
        return ["b4_not_executed: invalid B4 metadata payload."]

    summary: list[str] = []
    reason_map = {
        "no_phase_candidates": (
            "No valid phase candidates loaded "
            "(check candidate_phase_params/CIF query)."
        ),
        "no_g_vectors": "No usable g-vectors detected (effective spots are missing).",
        "all_matches_failed": (
            "All candidates failed constraints "
            "(check d/angle tolerance, spot selection, scale)."
        ),
    }
    if not bool(b4_meta.get("active", False)):
        raw_reason = str(b4_meta.get("reason", "unknown"))
        summary.append(f"b4_active=False reason={raw_reason}")
        if raw_reason in reason_map:
            summary.append(f"reason_explained: {reason_map[raw_reason]}")
    diagnostics = b4_meta.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        if "valid_d_count" in diagnostics and "g_vector_count" in diagnostics:
            summary.append(
                "valid_d_count/g_vector_count="
                f"{diagnostics.get('valid_d_count')}/{diagnostics.get('g_vector_count')}"
            )
        if "phase_candidate_count" in diagnostics:
            summary.append(f"phase_candidate_count={diagnostics.get('phase_candidate_count')}")
        selected_diag = diagnostics.get("selected_phase_diagnostics")
        if isinstance(selected_diag, dict):
            summary.append(
                "selected_phase_all_constraints_pass_count="
                f"{selected_diag.get('matched_within_all_constraints_count', 0)}"
            )
            summary.append(
                "selected_phase_anchor_pair_found="
                f"{selected_diag.get('anchor_pair_found', False)}"
            )
    used = diffraction_provenance.get("used_parameters", {})
    if isinstance(used, dict):
        if "candidate_phase_params_count" in used:
            summary.append(
                "candidate_phase_params_count="
                f"{used.get('candidate_phase_params_count')}"
            )
    return summary


def main() -> None:
    st.set_page_config(page_title="AutoTEM B2 Diffraction UI", layout="wide")
    st.title("AutoTEM-Insight: B2 FFT / Diffraction Test UI")
    st.caption(
        "Upload a TEM image, tune B2 parameters, and inspect FFT peaks, "
        "d-spacing, indexing, and phase candidates."
    )
    st.caption(f"FFT detector build: `{FFT_DETECTOR_BUILD_TAG}`")
    if "b4_candidate_params_text" not in st.session_state:
        st.session_state["b4_candidate_params_text"] = DEFAULT_B4_CANDIDATE_JSON
    if "b2_candidate_phases_selected" not in st.session_state:
        st.session_state["b2_candidate_phases_selected"] = list(PHASE_OPTIONS[:3])
    if "b4_candidate_params_text_pending" in st.session_state:
        st.session_state["b4_candidate_params_text"] = st.session_state.pop(
            "b4_candidate_params_text_pending"
        )
    if "b2_candidate_phases_selected_pending" in st.session_state:
        st.session_state["b2_candidate_phases_selected"] = st.session_state.pop(
            "b2_candidate_phases_selected_pending"
        )
    if "b4_lookup_message" in st.session_state:
        st.success(str(st.session_state.pop("b4_lookup_message")))

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
        high_recall_mode = st.checkbox(
            "High recall mode (select as many diffraction spots as possible)",
            value=True,
        )
        top_k_peaks = st.slider("Top-k peaks", min_value=1, max_value=128, value=24, step=1)
        max_peaks = st.slider("Max accepted peaks", min_value=1, max_value=256, value=48, step=1)
        min_peak_distance = st.slider(
            "Min peak distance (px)",
            min_value=0.0,
            max_value=30.0,
            value=3.0,
            step=0.5,
        )
        min_snr = st.slider("Min SNR", min_value=0.0, max_value=12.0, value=1.2, step=0.1)
        center_exclusion_radius = st.slider(
            "Center mask radius (px)",
            min_value=0.0,
            max_value=30.0,
            value=5.5,
            step=0.5,
        )
        center_reject_margin = st.slider(
            "Center reject margin (px)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
        )
        center_keep_snr = st.slider(
            "Center keep SNR (anti-duplicate)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
        )
        min_radius_px_hard = st.slider(
            "Min radius hard cutoff (px)",
            min_value=0.0,
            max_value=80.0,
            value=9.0,
            step=0.5,
        )
        axis_streak_mask_width = st.slider(
            "Axis streak mask width (px)",
            min_value=0.0,
            max_value=20.0,
            value=1.5,
            step=0.5,
        )
        dedup_distance_px = st.slider(
            "Dedup distance (px)",
            min_value=0.5,
            max_value=30.0,
            value=6.0,
            step=0.5,
        )
        dedup_radial_tol_px = st.slider(
            "Dedup radial tol (px)",
            min_value=0.2,
            max_value=20.0,
            value=3.0,
            step=0.2,
        )
        dedup_angle_tol_deg = st.slider(
            "Dedup angle tol (deg)",
            min_value=0.1,
            max_value=15.0,
            value=2.2,
            step=0.1,
        )
        max_radius_ratio = st.slider(
            "Max radius ratio (vs Nyquist radius)",
            min_value=0.05,
            max_value=1.0,
            value=0.45,
            step=0.01,
        )
        max_candidates_for_refine = st.slider(
            "Max candidates for refine",
            min_value=20,
            max_value=2000,
            value=140,
            step=10,
        )
        max_gaussian_refine_count = st.slider(
            "Max Gaussian subpixel fits (speed)",
            min_value=0,
            max_value=400,
            value=24,
            step=10,
        )
        return_peak_debug = st.checkbox(
            "Enable full debug maps (slow)",
            value=False,
        )
        rescue_on_empty = st.checkbox(
            "Rescue when no accepted peak",
            value=True,
        )
        rescue_min_snr = st.slider(
            "Rescue min SNR",
            min_value=0.0,
            max_value=6.0,
            value=0.6,
            step=0.1,
            disabled=not rescue_on_empty,
        )
        allow_relaxed_fallback = st.checkbox(
            "Allow relaxed fallback (not recommended for real FFT)",
            value=False,
        )
        enable_inner_pair_recovery = st.checkbox(
            "Enable inner ±g pair recovery",
            value=True,
        )
        inner_pair_min_snr = st.slider(
            "Inner pair min SNR",
            min_value=0.0,
            max_value=6.0,
            value=0.8,
            step=0.1,
            disabled=not enable_inner_pair_recovery,
        )
        inner_pair_recover_radius_px = st.slider(
            "Inner pair recover radius (px)",
            min_value=8.0,
            max_value=120.0,
            value=44.0,
            step=1.0,
            disabled=not enable_inner_pair_recovery,
        )
        inner_pair_max_pairs = st.slider(
            "Inner pair max pairs",
            min_value=0,
            max_value=20,
            value=8,
            step=1,
            disabled=not enable_inner_pair_recovery,
        )
        ensure_multiring_coverage = st.checkbox(
            "Ensure multi-ring coverage",
            value=True,
        )
        inner_ring_recover_pairs = st.slider(
            "Inner ring recover pairs",
            min_value=0,
            max_value=20,
            value=4,
            step=1,
            disabled=not ensure_multiring_coverage,
        )
        inner_ring_recover_min_snr = st.slider(
            "Inner ring recover min SNR",
            min_value=0.0,
            max_value=6.0,
            value=0.7,
            step=0.1,
            disabled=not ensure_multiring_coverage,
        )
        enable_lattice_consistency_filter = st.checkbox(
            "Enable lattice consistency filter (parallelogram rule)",
            value=False,
        )
        auto_center_cluster_filter = st.checkbox(
            "Auto remove center clutter cluster",
            value=False,
        )
        lattice_coeff_tol = st.slider(
            "Lattice coeff tol",
            min_value=0.05,
            max_value=1.5,
            value=0.38,
            step=0.01,
            disabled=not enable_lattice_consistency_filter,
        )
        lattice_recon_tol_px = st.slider(
            "Lattice recon tol (px)",
            min_value=0.5,
            max_value=20.0,
            value=4.0,
            step=0.1,
            disabled=not enable_lattice_consistency_filter,
        )
        lattice_filter_min_keep = st.slider(
            "Lattice filter min keep",
            min_value=2,
            max_value=40,
            value=6,
            step=1,
            disabled=not enable_lattice_consistency_filter,
        )
        origin_g_norm_threshold = st.slider(
            "Origin g-norm cutoff (1/nm)",
            min_value=0.0,
            max_value=2.0,
            value=0.12,
            step=0.01,
        )
        origin_hkl_g_max = st.slider(
            "Origin hkl(000) max g-norm (1/nm)",
            min_value=0.0,
            max_value=2.0,
            value=0.30,
            step=0.01,
        )
        origin_filter_max_radius_px = st.slider(
            "Origin filter max radius (px)",
            min_value=0.0,
            max_value=80.0,
            value=20.0,
            step=0.5,
        )
        border_mask = st.slider(
            "Border mask (px)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.5,
        )
        radial_bin_width = st.slider(
            "Radial bin width (px)",
            min_value=0.5,
            max_value=20.0,
            value=2.0,
            step=0.5,
        )
        peak_threshold_strategy = st.selectbox(
            "peak_threshold_strategy",
            options=[
                "hybrid",
                "radial_mad",
                "local_max",
                "log_blob",
                "auto",
                "relative_max",
                "mad",
            ],
            index=0,
        )
        peak_threshold_value = st.number_input(
            "peak_threshold_value (empty uses default)",
            value=4.0,
            step=0.1,
            format="%.3f",
        )
        fft_window = st.selectbox("FFT window", options=["none", "hann", "tukey"], index=1)
        fft_window_tukey_alpha = st.slider(
            "Tukey alpha",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            disabled=fft_window != "tukey",
        )
        fft_compression = st.selectbox("FFT compression", options=["log1p", "asinh"], index=0)
        bg_sub_method = st.selectbox(
            "Background subtraction",
            options=["radial_mad", "gaussian", "hybrid"],
            index=2,
        )
        gaussian_background_sigma = st.slider(
            "Gaussian background sigma",
            min_value=1.0,
            max_value=80.0,
            value=12.0,
            step=1.0,
            disabled=bg_sub_method == "radial_mad",
        )
        clip_low_percentile = st.slider(
            "Clip low percentile",
            min_value=0.0,
            max_value=20.0,
            value=1.0,
            step=0.5,
        )
        clip_high_percentile = st.slider(
            "Clip high percentile",
            min_value=80.0,
            max_value=100.0,
            value=99.8,
            step=0.1,
        )
        pair_tolerance_px = st.slider(
            "±g pair tolerance (px)",
            min_value=0.5,
            max_value=20.0,
            value=4.0,
            step=0.5,
        )
        min_pair_count_for_strict = st.slider(
            "Min ±g pairs for strict indexing",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )
        if high_recall_mode:
            top_k_peaks = max(int(top_k_peaks), 96)
            max_peaks = max(int(max_peaks), 128)
            min_peak_distance = min(float(min_peak_distance), 2.0)
            min_snr = min(float(min_snr), 0.7)
            center_exclusion_radius = min(float(center_exclusion_radius), 4.0)
            center_reject_margin = min(float(center_reject_margin), 2.0)
            center_keep_snr = min(float(center_keep_snr), 1.2)
            min_radius_px_hard = min(float(min_radius_px_hard), 6.0)
            max_radius_ratio = max(float(max_radius_ratio), 0.62)
            max_candidates_for_refine = max(int(max_candidates_for_refine), 500)
            allow_relaxed_fallback = True
            rescue_on_empty = True
            rescue_min_snr = min(float(rescue_min_snr), 0.3)
            enable_inner_pair_recovery = True
            inner_pair_min_snr = min(float(inner_pair_min_snr), 0.4)
            inner_pair_recover_radius_px = max(float(inner_pair_recover_radius_px), 70.0)
            inner_pair_max_pairs = max(int(inner_pair_max_pairs), 10)
            ensure_multiring_coverage = True
            inner_ring_recover_pairs = max(int(inner_ring_recover_pairs), 8)
            inner_ring_recover_min_snr = min(float(inner_ring_recover_min_snr), 0.4)
            enable_lattice_consistency_filter = False
            auto_center_cluster_filter = False
            min_pair_count_for_strict = max(1, int(min_pair_count_for_strict))
            st.caption(
                "High recall mode applied: relaxed center/radius thresholds, "
                "larger candidate budget, and multi-ring recovery enabled."
            )
        indexing_mode = st.selectbox(
            "indexing_mode",
            options=["auto", "lattice_fit", "baseline", "b4_mvp", "b4_strict"],
            index=0,
            format_func=lambda key: INDEXING_MODE_LABELS.get(str(key), str(key)),
        )
        st.caption(INDEXING_MODE_HINTS.get(str(indexing_mode), ""))
        max_hkl = st.slider("B4 max_hkl", min_value=1, max_value=8, value=4, step=1)
        d_spacing_tolerance_rel = st.slider(
            "B4 d-spacing tolerance (relative)",
            min_value=0.01,
            max_value=0.4,
            value=0.08,
            step=0.01,
        )
        angle_tolerance_deg = st.slider(
            "B4 angle tolerance (deg)",
            min_value=1.0,
            max_value=45.0,
            value=12.0,
            step=0.5,
        )
        candidate_phase_params_text = st.text_area(
            "B4 candidate_phase_params (JSON list)",
            key="b4_candidate_params_text",
            height=120,
        )
        cif_candidates_dir = st.text_input(
            "B4 cif_candidates_dir (for b4_strict)",
            value="",
            placeholder="./cif_candidates",
        )
        with st.expander("Auto Lookup: formula/elements", expanded=False):
            phase_formula = st.text_input(
                "Chemical formula (e.g., SiO2, Al2O3)",
                value="",
                key="b4_lookup_formula",
            )
            phase_elements_text = st.text_input(
                "Possible elements (comma-separated, optional)",
                value="",
                key="b4_lookup_elements",
                placeholder="Si, O",
            )
            lookup_sources = st.multiselect(
                "Lookup sources",
                options=["materials_project_optimade", "cod"],
                default=["materials_project_optimade", "cod"],
                key="b4_lookup_sources",
            )
            lookup_limit = st.slider(
                "Max candidates per source",
                min_value=1,
                max_value=30,
                value=10,
                step=1,
                key="b4_lookup_limit",
            )
            do_lookup = st.button(
                "Auto Fetch Candidate Lattice Parameters",
                key="b4_lookup_btn",
            )
            st.caption(
                "Materials Project is queried via public OPTIMADE API; "
                "COD is queried by formula. ICDD PDF card is licensed and not queried directly."
            )
            if do_lookup:
                lookup_result = lookup_phase_candidates(
                    formula=str(phase_formula).strip() or None,
                    elements=_parse_elements_text(phase_elements_text),
                    sources=[str(item) for item in lookup_sources],
                    limit_per_source=int(lookup_limit),
                )
                if lookup_result.candidates:
                    st.session_state["b4_candidate_params_text_pending"] = json.dumps(
                        lookup_result.candidates,
                        ensure_ascii=False,
                        indent=2,
                    )
                    fetched_names = [
                        str(item.get("name", "")).strip()
                        for item in lookup_result.candidates
                        if str(item.get("name", "")).strip()
                    ]
                    st.session_state["b4_candidate_phase_names"] = fetched_names
                    st.session_state["b2_candidate_phases_selected_pending"] = list(fetched_names)
                    st.session_state["b4_lookup_message"] = (
                        f"Fetched {len(lookup_result.candidates)} candidates and "
                        "auto-filled candidate_phase_params."
                    )
                    st.rerun()
                else:
                    st.warning("No candidates found for current query.")
                for warning in lookup_result.warnings:
                    st.warning(warning)
        phase_top_n = st.slider("Phase top-n", min_value=1, max_value=8, value=3, step=1)
        phase_options = list(PHASE_OPTIONS)
        for name in st.session_state.get("b4_candidate_phase_names", []):
            text = str(name).strip()
            if text and text not in phase_options:
                phase_options.append(text)
        candidate_phases = st.multiselect(
            "Candidate phases",
            options=phase_options,
            key="b2_candidate_phases_selected",
        )
        st.subheader("MVP Phase Match")
        mvp_zone_axis_text = st.text_input(
            "Zone axis (u,v,w)",
            value="0,0,1",
            help="Used for CIF simulated diffraction and atomic projection.",
        )
        mvp_max_hkl = st.slider(
            "MVP max_hkl",
            min_value=2,
            max_value=10,
            value=6,
            step=1,
        )
        mvp_d_tol_rel = st.slider(
            "MVP d-spacing tolerance (relative)",
            min_value=0.01,
            max_value=0.35,
            value=0.10,
            step=0.01,
        )
        mvp_angle_tol_deg = st.slider(
            "MVP angle tolerance (deg)",
            min_value=2.0,
            max_value=45.0,
            value=14.0,
            step=0.5,
        )
        run_phase_match_clicked = st.button(
            "Run Phase Match",
            key="run_phase_match_mvp",
        )

        st.subheader("FFT Display (Gray)")
        show_peak_indices = st.checkbox(
            "Show peak indices on FFT",
            value=True,
        )
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

    auto_run_analysis = st.checkbox(
        "Auto-run after parameter changes",
        value=True,
        help="When enabled, B2/B4 reruns automatically after you change analysis parameters.",
    )
    run_button_label = (
        "Run B4 Analysis"
        if str(indexing_mode) in {"b4_mvp", "b4_strict"}
        else "Run B2 Analysis"
    )
    run_clicked = st.button(run_button_label, type="primary")
    auto_run_b4 = str(indexing_mode) in {"b4_mvp", "b4_strict"}
    if auto_run_b4:
        st.info("B4 auto-run is enabled. No click needed after switching to B4 mode.")

    if use_demo:
        input_fingerprint = "demo_builtin"
    elif uploaded is not None:
        input_fingerprint = f"upload::{uploaded.name}::{getattr(uploaded, 'size', 0)}"
    else:
        input_fingerprint = f"path::{input_path}"
    run_signature_payload = {
        "input_fingerprint": input_fingerprint,
        "auto_scale": bool(auto_scale),
        "pixel_size_nm_manual": float(pixel_size_nm_manual),
        "voltage_kv": float(voltage_kv),
        "recipe": str(recipe),
        "use_clip": bool(use_clip),
        "clip_low": float(clip_low),
        "clip_high": float(clip_high),
        "top_k_peaks": int(top_k_peaks),
        "max_peaks": int(max_peaks),
        "min_peak_distance": float(min_peak_distance),
        "high_recall_mode": bool(high_recall_mode),
        "min_snr": float(min_snr),
        "center_exclusion_radius": float(center_exclusion_radius),
        "center_reject_margin": float(center_reject_margin),
        "center_keep_snr": float(center_keep_snr),
        "min_radius_px_hard": float(min_radius_px_hard),
        "axis_streak_mask_width": float(axis_streak_mask_width),
        "dedup_distance_px": float(dedup_distance_px),
        "dedup_radial_tol_px": float(dedup_radial_tol_px),
        "dedup_angle_tol_deg": float(dedup_angle_tol_deg),
        "max_radius_ratio": float(max_radius_ratio),
        "max_candidates_for_refine": int(max_candidates_for_refine),
        "max_gaussian_refine_count": int(max_gaussian_refine_count),
        "return_peak_debug": bool(return_peak_debug),
        "rescue_on_empty": bool(rescue_on_empty),
        "rescue_min_snr": float(rescue_min_snr),
        "allow_relaxed_fallback": bool(allow_relaxed_fallback),
        "enable_inner_pair_recovery": bool(enable_inner_pair_recovery),
        "inner_pair_min_snr": float(inner_pair_min_snr),
        "inner_pair_recover_radius_px": float(inner_pair_recover_radius_px),
        "inner_pair_max_pairs": int(inner_pair_max_pairs),
        "ensure_multiring_coverage": bool(ensure_multiring_coverage),
        "inner_ring_recover_pairs": int(inner_ring_recover_pairs),
        "inner_ring_recover_min_snr": float(inner_ring_recover_min_snr),
        "enable_lattice_consistency_filter": bool(enable_lattice_consistency_filter),
        "auto_center_cluster_filter": bool(auto_center_cluster_filter),
        "lattice_coeff_tol": float(lattice_coeff_tol),
        "lattice_recon_tol_px": float(lattice_recon_tol_px),
        "lattice_filter_min_keep": int(lattice_filter_min_keep),
        "origin_g_norm_threshold": float(origin_g_norm_threshold),
        "origin_hkl_g_max": float(origin_hkl_g_max),
        "origin_filter_max_radius_px": float(origin_filter_max_radius_px),
        "border_mask": float(border_mask),
        "radial_bin_width": float(radial_bin_width),
        "peak_threshold_strategy": str(peak_threshold_strategy),
        "peak_threshold_value": float(peak_threshold_value),
        "fft_window": str(fft_window),
        "fft_window_tukey_alpha": float(fft_window_tukey_alpha),
        "fft_compression": str(fft_compression),
        "background_subtraction_method": str(bg_sub_method),
        "gaussian_background_sigma": float(gaussian_background_sigma),
        "clip_low_percentile": float(clip_low_percentile),
        "clip_high_percentile": float(clip_high_percentile),
        "pair_tolerance_px": float(pair_tolerance_px),
        "min_pair_count_for_strict": int(min_pair_count_for_strict),
        "indexing_mode": str(indexing_mode),
        "max_hkl": int(max_hkl),
        "d_spacing_tolerance_rel": float(d_spacing_tolerance_rel),
        "angle_tolerance_deg": float(angle_tolerance_deg),
        "phase_top_n": int(phase_top_n),
        "candidate_phases": sorted([str(v) for v in list(candidate_phases)]),
        "candidate_phase_params_text": str(candidate_phase_params_text),
        "cif_candidates_dir": str(cif_candidates_dir).strip(),
        "mvp_zone_axis_text": str(mvp_zone_axis_text),
        "mvp_max_hkl": int(mvp_max_hkl),
        "mvp_d_tol_rel": float(mvp_d_tol_rel),
        "mvp_angle_tol_deg": float(mvp_angle_tol_deg),
        "show_peak_indices": bool(show_peak_indices),
    }
    run_signature = json.dumps(run_signature_payload, sort_keys=True, ensure_ascii=False)
    prev_run_signature = str(st.session_state.get("b2_last_run_signature", ""))
    has_existing_context = "b2_context" in st.session_state
    auto_run_by_change = bool(auto_run_analysis) and (
        (not has_existing_context) or run_signature != prev_run_signature
    )
    if auto_run_by_change and not run_clicked:
        st.caption("Parameters changed; auto-running analysis.")
    run_requested = run_clicked or auto_run_b4 or auto_run_by_change
    if run_requested:
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

            candidate_phase_params_payload: list[dict[str, Any]] = []
            raw_phase_text = str(candidate_phase_params_text).strip()
            if raw_phase_text:
                decoded = json.loads(raw_phase_text)
                if not isinstance(decoded, list):
                    raise ValueError("candidate_phase_params JSON must be a list.")
                candidate_phase_params_payload = [row for row in decoded if isinstance(row, dict)]
            b4_phase_names = [
                str(row.get("name")).strip()
                for row in candidate_phase_params_payload
                if str(row.get("name", "")).strip()
            ]
            if not b4_phase_names:
                b4_phase_names = [
                    str(name).strip()
                    for name in st.session_state.get("b4_candidate_phase_names", [])
                    if str(name).strip()
                ]
            effective_candidate_phases = list(candidate_phases)
            if indexing_mode in {"b4_mvp", "b4_strict"} and b4_phase_names:
                effective_candidate_phases = b4_phase_names

            diffraction_params = {
                "top_k_peaks": int(top_k_peaks),
                "max_peaks": int(max_peaks),
                "min_peak_distance": float(min_peak_distance),
                "high_recall_mode": bool(high_recall_mode),
                "center_exclusion_radius": float(center_exclusion_radius),
                "center_mask_radius": float(center_exclusion_radius),
                "center_reject_margin": float(center_reject_margin),
                "center_keep_snr": float(center_keep_snr),
                "min_radius_px_hard": float(min_radius_px_hard),
                "axis_streak_mask_width": float(axis_streak_mask_width),
                "border_mask": float(border_mask),
                "radial_bin_width": float(radial_bin_width),
                "dedup_distance_px": float(dedup_distance_px),
                "dedup_radial_tol_px": float(dedup_radial_tol_px),
                "dedup_angle_tol_deg": float(dedup_angle_tol_deg),
                "max_radius_ratio": float(max_radius_ratio),
                "max_candidates_for_refine": int(max_candidates_for_refine),
                "max_gaussian_refine_count": int(max_gaussian_refine_count),
                "rescue_on_empty": bool(rescue_on_empty),
                "rescue_min_snr": float(rescue_min_snr),
                "allow_relaxed_fallback": bool(allow_relaxed_fallback),
                "enable_inner_pair_recovery": bool(enable_inner_pair_recovery),
                "inner_pair_min_snr": float(inner_pair_min_snr),
                "inner_pair_recover_radius_px": float(inner_pair_recover_radius_px),
                "inner_pair_max_pairs": int(inner_pair_max_pairs),
                "ensure_multiring_coverage": bool(ensure_multiring_coverage),
                "inner_ring_recover_pairs": int(inner_ring_recover_pairs),
                "inner_ring_recover_min_snr": float(inner_ring_recover_min_snr),
                "enable_lattice_consistency_filter": bool(enable_lattice_consistency_filter),
                "auto_center_cluster_filter": bool(auto_center_cluster_filter),
                "lattice_coeff_tol": float(lattice_coeff_tol),
                "lattice_recon_tol_px": float(lattice_recon_tol_px),
                "lattice_filter_min_keep": int(lattice_filter_min_keep),
                "origin_g_norm_threshold": float(origin_g_norm_threshold),
                "origin_hkl_g_max": float(origin_hkl_g_max),
                "origin_filter_max_radius_px": float(origin_filter_max_radius_px),
                "min_snr": float(min_snr),
                "peak_threshold_strategy": str(peak_threshold_strategy),
                "peak_threshold_value": float(peak_threshold_value),
                "fft_window": str(fft_window),
                "fft_window_tukey_alpha": float(fft_window_tukey_alpha),
                "fft_compression": str(fft_compression),
                "background_subtraction_method": str(bg_sub_method),
                "gaussian_background_sigma": float(gaussian_background_sigma),
                "clip_low_percentile": float(clip_low_percentile),
                "clip_high_percentile": float(clip_high_percentile),
                "pair_tolerance_px": float(pair_tolerance_px),
                "min_pair_count_for_strict": int(min_pair_count_for_strict),
                "return_peak_debug": bool(return_peak_debug),
                "indexing_mode": str(indexing_mode),
                "max_hkl": int(max_hkl),
                "d_spacing_tolerance_rel": float(d_spacing_tolerance_rel),
                "angle_tolerance_deg": float(angle_tolerance_deg),
                "phase_top_n": int(phase_top_n),
                "candidate_phases": effective_candidate_phases,
                "candidate_phase_params": candidate_phase_params_payload,
                "cif_candidates_dir": str(cif_candidates_dir).strip() or None,
            }
            diffraction = run_diffraction_analysis(
                preprocessing,
                calibration,
                parameters=diffraction_params,
            )

            core_phase_rows: list[dict[str, Any]] = []
            for candidate in list(diffraction.phase_candidates):
                mean_d_err = getattr(candidate, "mean_d_error", None)
                mean_d_value = None if mean_d_err is None else getattr(mean_d_err, "value", None)
                core_phase_rows.append(
                    {
                        "name": str(candidate.name),
                        "formula": candidate.formula,
                        "score": (
                            None if candidate.score is None else round(float(candidate.score), 6)
                        ),
                        "matched_spot_count": candidate.matched_spot_count,
                        "mean_d_error": (
                            None if mean_d_value is None else round(float(mean_d_value), 6)
                        ),
                        "zone_axis": candidate.zone_axis,
                        "source": candidate.source,
                        "identifier": candidate.identifier,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            st.error(f"B2 analysis failed: {exc}")
            return

        processed_image = np.load(npy_path, allow_pickle=False).astype(np.float32)
        fft_complex = np.fft.fftshift(np.fft.fft2(processed_image))
        fft_magnitude = np.abs(fft_complex).astype(np.float32)
        fft_img = _fft_preview(processed_image)

        auto_points_xy = [(float(peak.x), float(peak.y)) for peak in diffraction.peaks]
        core_peak_rows, core_residuals = _build_core_peak_rows(
            diffraction,
            pixel_size_nm=float(effective_pixel_size_nm),
        )
        peak_detection_debug = {}
        if isinstance(diffraction.provenance, dict):
            maybe_debug = diffraction.provenance.get("peak_detection", {})
            if isinstance(maybe_debug, dict):
                peak_detection_debug = maybe_debug
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
            "candidate_phases": list(effective_candidate_phases),
            "core_phase_rows": core_phase_rows,
            "candidate_phase_params": list(candidate_phase_params_payload),
            "diffraction_provenance": diffraction.provenance,
            "core_peak_rows": core_peak_rows,
            "core_residuals": core_residuals,
            "peak_detection_debug": peak_detection_debug,
            "peak_debug_enabled": bool(return_peak_debug),
            "dataset_warnings": list(dataset.warnings),
            "quality_warnings": list(quality.warnings),
            "calibration_warnings": list(calibration.warnings),
            "preprocessing_warnings": list(preprocessing.warnings),
            "diffraction_warnings": list(diffraction.warnings),
            "scale_note": scale_note,
        }
        st.session_state.pop("b2_phase_mvp", None)
        st.session_state["b2_last_run_signature"] = run_signature

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
        "Click two basis points directly on the FFT image (g1/g2). "
        "Other lattice spots will be generated from vector synthesis."
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
            st.session_state.pop("b2_last_clicked_points_xy", None)
            st.rerun()

    clicked_points_xy = _extract_canvas_points_xy(canvas_result, sx=sx, sy=sy)
    if clicked_points_xy:
        st.session_state["b2_last_clicked_points_xy"] = list(clicked_points_xy)
    cached_clicked_points_xy = list(st.session_state.get("b2_last_clicked_points_xy", []))
    with canvas_col2:
        show_points = clicked_points_xy if clicked_points_xy else cached_clicked_points_xy
        st.write(f"Clicked points: {len(show_points)}")
        if show_points:
            clicked_text = "\n".join(
                f"{idx+1}: x={x:.2f}, y={y:.2f}"
                for idx, (x, y) in enumerate(show_points)
            )
            st.code(
                clicked_text,
                language="text",
            )

    selected_points_xy: list[tuple[float, float]]
    if selection_mode == "Auto peaks":
        selected_points_xy = list(auto_points_xy)
    else:
        points_for_synthesis = (
            clicked_points_xy if len(clicked_points_xy) >= 2 else cached_clicked_points_xy
        )
        if len(points_for_synthesis) < 2:
            st.warning("Please click at least two FFT points as basis vectors g1/g2.")
            selected_points_xy = []
        else:
            if len(points_for_synthesis) > 2:
                st.warning("More than two points detected; only the first two are used as basis.")
            basis1 = points_for_synthesis[0]
            basis2 = points_for_synthesis[1]
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
    if selection_mode == "Auto peaks":
        core_peak_rows = list(context.get("core_peak_rows", []))
        core_residuals = list(context.get("core_residuals", []))
        if core_peak_rows:
            peak_rows = core_peak_rows
            if core_residuals:
                residuals_lattice = [float(v) for v in core_residuals]
    core_phase_rows_for_display = list(context.get("core_phase_rows", []))
    if str(indexing_mode) in {"b4_mvp", "b4_strict"} and core_phase_rows_for_display:
        phase_rows = core_phase_rows_for_display
    elif str(indexing_mode) in {"b4_mvp", "b4_strict"}:
        fallback_rows = _phase_rows_from_candidate_params(
            list(context.get("candidate_phase_params", []))
        )
        if fallback_rows:
            phase_rows = fallback_rows

    st.success("B2 analysis finished.")
    provenance_obj = context.get("diffraction_provenance", {})
    provenance_dict = provenance_obj if isinstance(provenance_obj, dict) else {}
    method_text = str(provenance_dict.get("method", "unknown"))
    st.caption(f"Current core method: `{method_text}`")
    if selection_mode == "Auto peaks" and peak_rows:
        snr_vals: list[float] = []
        near_center_000 = 0
        for row in peak_rows:
            try:
                snr_vals.append(float(row.get("snr", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
            try:
                r_val = float(row.get("r_px", 0.0) or 0.0)
                h_val = int(row.get("h", 0))
                k_val = int(row.get("k", 0))
                l_val = int(row.get("l", 0))
            except (TypeError, ValueError):
                continue
            if r_val < 20.0 and h_val == 0 and k_val == 0 and l_val == 0:
                near_center_000 += 1
        if snr_vals and max(snr_vals) <= 0.0:
            st.error(
                "Detected SNR values are all zero. This usually means stale/legacy "
                "results or fallback-only output; please hard refresh and rerun."
            )
        if near_center_000 > 0:
            st.warning(
                f"Found {near_center_000} near-center 000-like spots (<20 px). "
                "Please tighten center cutoff or verify the run is using the latest settings."
            )
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Detected Peaks", len(peak_rows))
    metric_col2.metric(
        "Indexed Spots",
        sum(
            1
            for row in peak_rows
            if float(row.get("d_nm", row.get("d_spacing", 0.0)) or 0.0) > 0
        ),
    )
    metric_col3.metric("Phase Candidates", len(phase_rows))
    metric_col4.metric("Confidence", f"{selected_confidence:.3f}")
    st.info(
        "How to read: Detected Peaks = number of detected diffraction points; "
        "Indexed Spots = spots with assigned h,k,l; "
        "Phase Candidates = candidate phases passing current criteria; "
        "Confidence = overall reliability score (0~1, higher is better)."
    )
    st.caption(
        "Rule of thumb: mean_d_error < 0.08 is usually good; 0.08~0.15 needs caution; "
        "> 0.15 suggests re-checking scale, selected spots, or phase candidates."
    )

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
        show_idx = bool(show_peak_indices)
        if show_idx:
            for idx, (px, py) in enumerate(selected_points_xy):
                ax2.text(
                    float(px) + 4.0,
                    float(py) + 3.0,
                    str(idx),
                    fontsize=6,
                    color="yellow",
                    ha="left",
                    va="bottom",
                    bbox={
                        "facecolor": "black",
                        "alpha": 0.35,
                        "pad": 0.6,
                        "edgecolor": "none",
                    },
                )
    ax2.set_title("FFT + Detected Peaks")
    ax2.axis("off")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    peak_debug_enabled = bool(context.get("peak_debug_enabled", False))
    peak_debug = context.get("peak_detection_debug", {})
    if peak_debug_enabled:
        st.subheader("FFT Peak Detection Debug (v2)")
    if peak_debug_enabled and isinstance(peak_debug, dict):
        raw_fft = _as_array_from_debug(peak_debug.get("raw_fft_magnitude"))
        log_fft = _as_array_from_debug(peak_debug.get("log_fft"))
        bg_sub = _as_array_from_debug(peak_debug.get("background_subtracted_fft"))
        radial_z = _as_array_from_debug(peak_debug.get("radial_z_score_map"))
        masks_obj = peak_debug.get("masks")
        masks_debug = masks_obj if isinstance(masks_obj, dict) else {}
        center_mask_arr = _as_array_from_debug(masks_debug.get("center_mask"))
        axis_mask_arr = _as_array_from_debug(masks_debug.get("axis_mask"))
        border_mask_arr = _as_array_from_debug(masks_debug.get("border_mask"))

        cands_xy = peak_debug.get("candidate_peaks_xy", [])
        accepts_xy = peak_debug.get("accepted_peaks_xy", [])
        rejects_xy = peak_debug.get("rejected_peaks_xy", [])
        paired_segments = peak_debug.get("paired_segments", [])
        residual_values = peak_debug.get("residual_distribution", [])

        fig_dbg, axes = plt.subplots(3, 3, figsize=(13, 12), dpi=140)
        ax = axes.ravel()
        ax[0].imshow(raw_fft if raw_fft is not None else fft_magnitude, cmap="gray")
        ax[0].set_title("1) Raw FFT Magnitude")
        ax[0].axis("off")
        ax[1].imshow(log_fft if log_fft is not None else fft_img_display, cmap="gray")
        ax[1].set_title("2) Log FFT")
        ax[1].axis("off")
        ax[2].imshow(bg_sub if bg_sub is not None else fft_img_display, cmap="gray")
        ax[2].set_title("3) Background-Subtracted FFT")
        ax[2].axis("off")
        ax[3].imshow(radial_z if radial_z is not None else fft_img_display, cmap="gray")
        ax[3].set_title("4) Radial Z-Score Map")
        ax[3].axis("off")

        if (
            center_mask_arr is not None
            and axis_mask_arr is not None
            and border_mask_arr is not None
        ):
            merged_mask = np.clip(center_mask_arr + axis_mask_arr + border_mask_arr, 0, 1)
        else:
            merged_mask = np.zeros_like(fft_img_display)
        ax[4].imshow(merged_mask, cmap="gray")
        ax[4].set_title("5) Masks (Center/Axis/Border)")
        ax[4].axis("off")

        ax[5].imshow(fft_img_display, cmap="gray")
        if isinstance(cands_xy, list) and cands_xy:
            xs = [float(item.get("x", 0.0)) for item in cands_xy if isinstance(item, dict)]
            ys = [float(item.get("y", 0.0)) for item in cands_xy if isinstance(item, dict)]
            ax[5].scatter(xs, ys, s=10, c="yellow")
        ax[5].set_title("6) Candidate Peaks")
        ax[5].axis("off")

        ax[6].imshow(fft_img_display, cmap="gray")
        if isinstance(accepts_xy, list) and accepts_xy:
            xs = [float(item.get("x", 0.0)) for item in accepts_xy if isinstance(item, dict)]
            ys = [float(item.get("y", 0.0)) for item in accepts_xy if isinstance(item, dict)]
            ax[6].scatter(xs, ys, s=18, c="cyan", edgecolors="black", linewidths=0.3)
        ax[6].set_title("7) Accepted Peaks")
        ax[6].axis("off")

        ax[7].imshow(fft_img_display, cmap="gray")
        if isinstance(rejects_xy, list) and rejects_xy:
            xs = [float(item.get("x", 0.0)) for item in rejects_xy if isinstance(item, dict)]
            ys = [float(item.get("y", 0.0)) for item in rejects_xy if isinstance(item, dict)]
            ax[7].scatter(xs, ys, s=14, c="red", marker="x")
        if isinstance(paired_segments, list):
            for seg in paired_segments:
                if not isinstance(seg, dict):
                    continue
                ax[7].plot(
                    [float(seg.get("x1", 0.0)), float(seg.get("x2", 0.0))],
                    [float(seg.get("y1", 0.0)), float(seg.get("y2", 0.0))],
                    c="lime",
                    linewidth=0.8,
                )
        ax[7].set_title("8/9) Rejected + ±g Paired Peaks")
        ax[7].axis("off")

        if isinstance(residual_values, list) and residual_values:
            vals = [float(v) for v in residual_values]
            ax[8].hist(vals, bins=min(20, max(5, len(vals) // 2)))
        ax[8].set_title("10) Residual Distribution")
        ax[8].set_xlabel("pair error (px)")
        ax[8].set_ylabel("count")
        fig_dbg.tight_layout()
        st.pyplot(fig_dbg, clear_figure=True)
    elif not peak_debug_enabled:
        st.caption("FFT peak debug maps are hidden for speed. Enable them from sidebar if needed.")

    st.subheader("Peaks / g-vectors / d-spacing / hkl")
    st.caption(
        "Key fields: h,k,l are indexing results; d_spacing is measured spacing; "
        "residual_error lower is better; confidence higher is better."
    )
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
    st.caption(
        "Ranking rule: higher score is better; more matched_spot_count is better; "
        "lower mean_d_error is better. Strict mode also adds consistency checks."
    )
    if str(indexing_mode) in {"b4_mvp", "b4_strict"} and not core_phase_rows_for_display:
        st.warning(
            "B4 candidates were loaded, but current diffraction points "
            "did not produce a valid match yet. "
            "Check scale, selected spots, and tolerance settings."
        )
    st.dataframe(phase_rows, use_container_width=True)
    if str(indexing_mode) in {"b4_mvp", "b4_strict"}:
        st.subheader("B4 Match Diagnostics")
        provenance_obj = context.get("diffraction_provenance", {})
        provenance_dict = provenance_obj if isinstance(provenance_obj, dict) else {}
        diag_rows = _build_b4_diagnostic_rows(
            diffraction_provenance=provenance_dict
        )
        diag_summary = _build_b4_diagnostic_summary(
            diffraction_provenance=provenance_dict
        )
        if diag_rows:
            st.caption(
                "Why not matched: check d_pass_count and angle_pass_count. "
                "If d_pass is low, d error exceeds threshold; "
                "if angle_pass is low, angle constraint failed."
            )
            st.dataframe(diag_rows, use_container_width=True)
        else:
            st.info("No per-phase diagnostics returned in this run.")
        for text in diag_summary:
            st.caption(f"- {text}")

    st.subheader("Run Phase Match (MVP)")
    mvp_zone_axis = _parse_zone_axis_text(mvp_zone_axis_text)
    cif_dir_effective = str(cif_candidates_dir).strip()
    mvp_mode_enabled = str(indexing_mode) in {"b4_mvp", "b4_strict"}
    core_has_ranked_phase = any(
        float(item.get("score", 0.0) or 0.0) > 0.0
        for item in phase_rows
        if isinstance(item, dict)
    )
    mvp_signature = json.dumps(
        {
            "run_signature": run_signature,
            "zone_axis": list(mvp_zone_axis),
            "mvp_max_hkl": int(mvp_max_hkl),
            "mvp_d_tol_rel": float(mvp_d_tol_rel),
            "mvp_angle_tol_deg": float(mvp_angle_tol_deg),
            "cif_dir": cif_dir_effective,
            "candidate_phases": [str(v) for v in candidate_phases],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    prev_mvp_state = st.session_state.get("b2_phase_mvp", {})
    prev_mvp_signature = (
        str(prev_mvp_state.get("signature", ""))
        if isinstance(prev_mvp_state, dict)
        else ""
    )
    auto_run_mvp = (
        mvp_mode_enabled
        and core_has_ranked_phase
        and bool(peak_rows)
        and not run_phase_match_clicked
        and mvp_signature != prev_mvp_signature
    )
    if run_phase_match_clicked or auto_run_mvp:
        mvp_warnings: list[str] = []
        selected_names = {str(v) for v in candidate_phases} if candidate_phases else None
        entries: list[PhaseEntry] = []

        if cif_dir_effective:
            cif_entries, load_warnings = load_cif_phase_entries(
                cif_dir_effective,
                selected_names=selected_names,
            )
            entries.extend(cif_entries)
            mvp_warnings.extend(load_warnings)

        if not entries:
            candidate_params = list(context.get("candidate_phase_params", []))
            param_entries, param_warnings = _phase_entries_from_candidate_params(
                candidate_params,
                selected_names=selected_names,
            )
            entries.extend(param_entries)
            mvp_warnings.extend(param_warnings)
            if param_entries:
                mvp_warnings.append(
                    "No CIF loaded: using candidate_phase_params for phase match."
                )

        if not entries:
            fallback_entries = _fallback_phase_entries(selected_names=selected_names)
            entries.extend(fallback_entries)
            if fallback_entries:
                mvp_warnings.append(
                    "No CIF/candidate params available: using built-in fallback phases."
                )

        ranking_rows, detail_map, match_warnings = match_candidates(
            peak_rows,
            entries,
            max_hkl=int(mvp_max_hkl),
            zone_axis=mvp_zone_axis,
            d_tol_rel=float(mvp_d_tol_rel),
            angle_tol_deg=float(mvp_angle_tol_deg),
        )
        mvp_warnings.extend(match_warnings)
        st.session_state["b2_phase_mvp"] = {
            "zone_axis": mvp_zone_axis,
            "ranking_rows": ranking_rows,
            "detail_map": detail_map,
            "warnings": mvp_warnings,
            "cif_dir": cif_dir_effective,
            "signature": mvp_signature,
        }

    mvp_state = st.session_state.get("b2_phase_mvp", {})
    if isinstance(mvp_state, dict) and mvp_state.get("ranking_rows"):
        ranking_rows = list(mvp_state.get("ranking_rows", []))
        detail_map = mvp_state.get("detail_map", {})
        warnings_mvp = list(mvp_state.get("warnings", []))
        phase_names = [str(row.get("name", "")) for row in ranking_rows if str(row.get("name", ""))]
        default_name = phase_names[0] if phase_names else ""
        prev_selected = str(st.session_state.get("b2_mvp_selected_phase", default_name))
        if prev_selected not in phase_names:
            prev_selected = default_name
        selected_name = st.selectbox(
            "Choose candidate phase result",
            options=phase_names,
            index=(phase_names.index(prev_selected) if prev_selected in phase_names else 0),
            key="b2_mvp_selected_phase",
        ) if phase_names else ""
        top_name = str(selected_name or default_name)
        top_detail = detail_map.get(top_name, {})
        assignments = list(top_detail.get("assignments", []))
        sim_rows = list(top_detail.get("simulated_rows", []))
        entry = top_detail.get("entry")
        st.success(f"Current phase candidate: {top_name}")
        st.dataframe(ranking_rows, use_container_width=True)

        fig_ann = plot_annotated_fft(fft_img_display, peak_rows, assignments)
        ann_png = _fig_to_bytes(fig_ann, fmt="png")
        ann_svg = _fig_to_bytes(fig_ann, fmt="svg")
        st.subheader("Annotated FFT")
        st.pyplot(fig_ann, clear_figure=False)

        fig_sim = plot_simulated_diffraction(
            sim_rows,
            title=(
                f"Simulated Diffraction: {top_name} "
                f"[{mvp_zone_axis[0]} {mvp_zone_axis[1]} {mvp_zone_axis[2]}]"
            ),
        )
        sim_png = _fig_to_bytes(fig_sim, fmt="png")
        st.subheader("Simulated Diffraction Pattern")
        st.pyplot(fig_sim, clear_figure=False)

        fig_overlay = plot_experimental_vs_simulated_overlay(peak_rows, assignments, sim_rows)
        overlay_png = _fig_to_bytes(fig_overlay, fmt="png")
        st.subheader("Experimental vs Simulated Overlay")
        st.pyplot(fig_overlay, clear_figure=False)

        if entry is not None:
            fig_atoms = plot_atom_projection(entry, zone_axis=mvp_zone_axis)
            atoms_png = _fig_to_bytes(fig_atoms, fmt="png")
            st.subheader("Atomic Projection")
            st.pyplot(fig_atoms, clear_figure=False)
        else:
            fig_atoms = None
            atoms_png = b""
            st.info("Atomic projection unavailable for top phase entry.")

        assign_map = {int(row.get("peak_id", -1)): row for row in assignments}
        peaks_out: list[dict[str, Any]] = []
        for row in peak_rows:
            pid = int(row.get("peak_id", -1))
            merged = dict(row)
            rec = assign_map.get(pid, {})
            merged["matched"] = bool(rec.get("matched", False))
            merged["matched_h"] = int(rec.get("h", 0))
            merged["matched_k"] = int(rec.get("k", 0))
            merged["matched_l"] = int(rec.get("l", 0))
            merged["matched_d_nm"] = rec.get("d_nm")
            merged["match_d_error"] = rec.get("d_error")
            merged["match_angle_error_deg"] = rec.get("angle_error_deg")
            peaks_out.append(merged)

        report_json = {
            "cif_dir": str(mvp_state.get("cif_dir", "")),
            "zone_axis": list(mvp_zone_axis),
            "top_phase": ranking_rows[0],
            "ranking": ranking_rows,
            "warnings": warnings_mvp,
            "match_count": {
                "matched": int(
                    sum(1 for row in assignments if bool(row.get("matched", False)))
                ),
                "unmatched": int(
                    sum(1 for row in assignments if not bool(row.get("matched", False)))
                ),
            },
        }

        st.subheader("Export")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "annotated_fft.png",
                data=ann_png,
                file_name="annotated_fft.png",
                mime="image/png",
            )
            st.download_button(
                "simulated_diffraction.png",
                data=sim_png,
                file_name="simulated_diffraction.png",
                mime="image/png",
            )
        with c2:
            st.download_button(
                "experimental_vs_simulated_overlay.png",
                data=overlay_png,
                file_name="experimental_vs_simulated_overlay.png",
                mime="image/png",
            )
            if fig_atoms is not None:
                st.download_button(
                    "atom_projection.png",
                    data=atoms_png,
                    file_name="atom_projection.png",
                    mime="image/png",
                )
        with c3:
            st.download_button(
                "annotated_fft.svg",
                data=ann_svg,
                file_name="annotated_fft.svg",
                mime="image/svg+xml",
            )
            st.download_button(
                "peaks_table.csv",
                data=_to_csv(peaks_out),
                file_name="peaks_table.csv",
                mime="text/csv",
            )
            st.download_button(
                "phase_match_report.json",
                data=json.dumps(report_json, ensure_ascii=False, indent=2),
                file_name="phase_match_report.json",
                mime="application/json",
            )

        if warnings_mvp:
            st.subheader("MVP Match Warnings")
            for text in dict.fromkeys(warnings_mvp):
                st.warning(str(text))
    else:
        if mvp_mode_enabled and core_has_ranked_phase:
            st.info(
                "Core B4 ranking is available. If plots are not shown yet, "
                "click `Run Phase Match` once to refresh MVP visuals."
            )
        else:
            st.info(
                "Select CIF folder and click `Run Phase Match` to generate "
                "phase ranking and visual outputs."
            )

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

