from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.elements.image as st_image
import yaml
from PIL import Image, ImageDraw
from scipy.optimize import least_squares
from streamlit_drawable_canvas import st_canvas

from autotem.calibration import calibrate_dataset
from autotem.core import (
    DiffractionPeak,
    DiffractionResult,
    IndexedSpot,
    Measurement,
    PhaseCandidate,
    SimulatedPattern,
    SimulationResult,
)
from autotem.diffraction.b4_indexing import run_b4_indexing
from autotem.diffraction.basic import run_diffraction_analysis
from autotem.diffraction.fft import compute_fft_analysis
from autotem.diffraction.gvector import compute_gvectors_and_dspacing
from autotem.diffraction.open_source_adapters import SUPPORTED_PEAK_ADAPTER_ENGINES
from autotem.diffraction.peak_detection import (
    SUPPORTED_PEAK_DETECTORS,
    SUPPORTED_THRESHOLD_STRATEGIES,
)
from autotem.diffraction.phase_lookup import (
    materials_project_api_setup_guidance,
    supported_phase_lookup_sources,
)
from autotem.io import load_tem_dataset, read_image
from autotem.preprocessing import SUPPORTED_DENOISE_METHODS, assess_quality, preprocess_dataset
from autotem.simulation.basic import simulate_from_diffraction
from autotem.simulation.process_utils import collect_process_queue_payload

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


_APP_TITLE = "AutoTEM B1-B5 Diffraction Debug UI"
_UI_BUILD_TAG = "b1_b5_debug_ui_2026-05-30"
_WORKFLOW_TEMPLATE = "configs/hrtem_fft_phase.yaml"
_OUTPUT_ROOT_ENV = "AUTOTEM_UI_OUTPUT_ROOT"

PHASE_OPTIONS = ["Silicon", "Gold", "Graphite", "Titanium Dioxide"]
# Keep mode labels ASCII because these strings become Streamlit widget keys.
MODE_AUTO_EDIT = "Auto peaks"
MODE_TWO_VECTOR = "Two-point vector synthesis"
INDEXING_MODE_LABELS: dict[str, str] = {
    "auto": "auto (B3+B4 auto)",
    "baseline": "baseline (B3 baseline hkl)",
    "lattice_fit": "lattice_fit (B4 lattice-fit)",
}

SUPPORTED_BACKGROUND_METHODS = ["none", "gaussian", "rolling_ball"]
SUPPORTED_FFT_CENTER_MODES = ["auto", "geometric"]
SUPPORTED_INDEXING_MODES = ["auto", "baseline", "lattice_fit"]
SUPPORTED_SIMULATION_ENGINES = ["diffsims"]
SUPPORTED_MP_SORT_MODES = ["stability", "energy_above_hull", "material_id", "none"]
DEFAULT_PHASE_LOOKUP_SOURCES = [
    "local_builtin",
    "materials_project_optimade",
    "cod",
    "materials_project_api",
]

DEFAULT_B4_CANDIDATES_JSON = json.dumps(
    [
        {
            "name": "Silicon",
            "a": 0.5431,
            "b": 0.5431,
            "c": 0.5431,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "space_group": "Fd-3m",
            "formula": "Si",
        },
        {
            "name": "Gold",
            "a": 0.4078,
            "b": 0.4078,
            "c": 0.4078,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
            "space_group": "Fm-3m",
            "formula": "Au",
        },
        {
            "name": "Graphite",
            "a": 0.2460,
            "b": 0.2460,
            "c": 0.6710,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 120.0,
            "space_group": "P63/mmc",
            "formula": "C",
        },
    ],
    ensure_ascii=False,
    indent=2,
)

_FALLBACK_DEFAULTS: dict[str, Any] = {
    "quality": {
        "thresholds": {
            "snr_min": 6.0,
            "drift_px_max": 2.0,
            "overall_score_min": 0.35,
        }
    },
    "calibration": {
        "method": "auto",
        "diffraction_calibration_engine": "none",
        "voltage_kv": 200.0,
        "pixel_size_nm": 0.02,
        "pixel_size_uncertainty_nm": 0.001,
        "known_d_spacing": 0.235,
        "known_d_spacing_unit": "nm",
        "known_d_uncertainty_nm": 0.002,
        "measured_radius_px": 8.5,
        "measured_radius_uncertainty_px": 0.3,
        "detector_pixel_size_um": 14.0,
    },
    "preprocessing": {
        "recipe": "auto",
        "clip_percentile_low": 1.0,
        "clip_percentile_high": 99.0,
        "denoise_method": "wiener",
        "denoise_sigma": 0.8,
        "median_size": 3,
        "wiener_window": 5,
        "wiener_noise": 0.0,
        "nlm_h": 0.05,
        "nlm_patch_size": 5,
        "nlm_patch_distance": 6,
        "background_method": "gaussian",
        "background_sigma": 12.0,
        "background_opening_size": 31,
        "background_clip_zero": False,
    },
    "diffraction": {
        "auto_peak_tuning": True,
        "top_k_peaks": 24,
        "min_peak_distance": 3,
        "center_exclusion_radius": 4.0,
        "max_peak_radius_px": 0.0,
        "radius_outlier_filter": True,
        "radius_outlier_mad_factor": 4.0,
        "peak_threshold_strategy": "auto",
        "peak_detector": "log",
        "peak_adapter_engine": "none",
        "detector_sigma": 1.2,
        "dog_sigma_ratio": 1.8,
        "subpixel_window_radius": 2,
        "subpixel_max_shift": 1.5,
        "fft_center_mode": "auto",
        "direct_beam_refine_window": 3,
        "direct_beam_center_uncertainty_px": 0.35,
        "indexing_mode": "auto",
        "phase_top_n": 3,
        "phase_candidate_materials": ["Silicon", "Gold", "Graphite"],
        "max_hkl": 4,
        "d_tolerance_rel": 0.10,
        "angle_tolerance_deg": 12.0,
        "b4_strict_mode": False,
        "phase_lookup_sources": list(DEFAULT_PHASE_LOOKUP_SOURCES),
        "phase_lookup_limit_per_source": 3,
        "phase_lookup_timeout_s": 10.0,
        "materials_project_sort_by": "stability",
        "materials_project_include_deprecated": False,
        "materials_project_filter_stable": False,
        "materials_project_max_energy_above_hull": 0.0,
        "materials_project_prefer_experimental": True,
        "materials_project_candidate_pool_size": 20,
    },
    "simulation": {
        "enabled": True,
        "simulation_engine": "diffsims",
        "reciprocal_radius_inv_angstrom": 1.0,
        "max_simulated_spots": 128,
        "minimum_simulated_g_inv_nm": 0.001,
        "with_direct_beam": True,
        "max_excitation_error": 0.01,
        "fallback_single_atom": True,
        "zone_axis_mode": "inherit_b4_locked",
        "zone_axis_candidates": [[0, 0, 1]],
        "manual_zone_axis": [0, 0, 1],
        "show_unmatched_simulated_spots": False,
        "show_match_residual_lines": False,
        "rotation_axis": [0, 0, 1],
        "rotation_angles_deg": [0.0],
        "phase_match_enabled": True,
        "phase_match_min_spots": 2,
        "phase_match_previous_score_weight": 0.35,
        "phase_match_max_rotation_candidates": 64,
        "timeout_s": 20.0,
        "auto_rerun": False,
    },
}

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_name(raw_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name.strip())
    return token or "input"


def _json_signature(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _default_output_root() -> Path:
    from_env = os.environ.get(_OUTPUT_ROOT_ENV)
    if from_env:
        return Path(from_env).expanduser().resolve()
    return _repo_root() / "outputs" / "b1_b3_preview"


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _read_template_defaults() -> dict[str, Any]:
    path = _repo_root() / _WORKFLOW_TEMPLATE
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    workflow = payload.get("workflow", {})
    if not isinstance(workflow, dict):
        return {}
    params = workflow.get("parameters", {})
    return dict(params) if isinstance(params, dict) else {}


def _num(value: Any, fallback: float) -> float:
    if value is None or isinstance(value, bool):
        return float(fallback)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return float(fallback)
    return float(fallback)


def _choice(options: list[str], value: Any, fallback: str) -> str:
    text = str(value)
    return text if text in options else fallback


def _comma_list(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    text = str(raw_value or "")
    return [item.strip() for item in re.split(r"[,;\n]+", text) if item.strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        text = str(item).strip()
        key = text.casefold()
        if text and key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _phase_material_defaults(diff_defaults: dict[str, Any]) -> list[str]:
    raw = _dedupe_strings(
        [
            *_comma_list(diff_defaults.get("phase_candidate_materials", [])),
            *_comma_list(diff_defaults.get("material_phase_candidates", [])),
            *_comma_list(diff_defaults.get("candidate_phases", [])),
            *_comma_list(diff_defaults.get("phase_lookup_materials", [])),
            *_comma_list(diff_defaults.get("phase_lookup_formula", [])),
            *_comma_list(diff_defaults.get("materials_project_ids", [])),
        ]
    )
    return raw or PHASE_OPTIONS[:3]


def _json_or_default(raw_text: str, fallback: Any) -> tuple[Any, list[str]]:
    text = str(raw_text or "").strip()
    if not text:
        return fallback, []
    try:
        return json.loads(text), []
    except Exception as exc:  # noqa: BLE001
        return fallback, [f"JSON parse failed: {exc}"]


def _zone_axis_tuple(raw_axis: Any) -> tuple[int, int, int] | None:
    if not isinstance(raw_axis, (list, tuple)) or len(raw_axis) < 3:
        return None
    try:
        axis = tuple(int(round(float(raw_axis[idx]))) for idx in range(3))
    except (TypeError, ValueError):
        return None
    return None if axis == (0, 0, 0) else axis


def _parse_zone_axis_candidates(raw_text: str) -> tuple[list[tuple[int, int, int]], list[str]]:
    payload, warnings = _json_or_default(raw_text, [[0, 0, 1]])
    axes: list[tuple[int, int, int]] = []
    single_axis = _zone_axis_tuple(payload)
    if single_axis is not None:
        axes.append(single_axis)
    elif isinstance(payload, list):
        for raw_axis in payload:
            axis = _zone_axis_tuple(raw_axis)
            if axis is not None:
                axes.append(axis)
    if not axes:
        warnings.append("zone_axis_candidates has no valid non-zero [h,k,l] rows.")
        axes = [(0, 0, 1)]
    unique: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for axis in axes:
        if axis in seen:
            continue
        seen.add(axis)
        unique.append(axis)
    return unique, warnings


def _parse_float_vector(raw_text: str, fallback: list[float]) -> tuple[list[float], list[str]]:
    payload, warnings = _json_or_default(raw_text, fallback)
    if not isinstance(payload, list):
        warnings.append("Expected a JSON list of numbers.")
        return list(fallback), warnings
    values: list[float] = []
    for item in payload:
        value = _to_float(item)
        if value is not None:
            values.append(float(value))
    if not values:
        warnings.append("No valid numeric values found.")
        values = list(fallback)
    return values, warnings


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _decode_uploaded_text(uploaded_file: Any) -> str:
    if uploaded_file is None:
        return ""
    raw = uploaded_file.getvalue()
    if not raw:
        return ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return raw.decode(encoding)
        except Exception:  # noqa: BLE001
            continue
    return ""


def _wrap_angle_deg(value: float) -> float:
    out = (float(value) + 180.0) % 360.0 - 180.0
    return float(out)


def _angle_error_deg(angle_a: float, angle_b: float) -> float:
    return abs(_wrap_angle_deg(float(angle_a) - float(angle_b)))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), q))


def _parse_manual_spot_table(raw_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    text = str(raw_text or "")
    if not text.strip():
        return [], []
    number_pat = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip().replace("\ufeff", "")
        if not stripped:
            continue
        lowered = stripped.lower()
        if (
            "spot#" in lowered
            or "d-spacing" in lowered
            or "rec. pos" in lowered
            or "degrees to" in lowered
            or "amplitude" in lowered
        ):
            continue
        tokens = number_pat.findall(stripped)
        if len(tokens) < 2:
            continue
        values: list[float] = []
        for token in tokens:
            try:
                values.append(float(token))
            except ValueError:
                continue
        if len(values) < 2:
            continue

        spot_id = len(rows) + 1
        d_spacing_nm: float | None = None
        g_magnitude: float | None = None
        angle_x_deg: float | None = None
        amplitude: float | None = None

        has_spot_id = len(values) >= 3 and abs(values[0] - round(values[0])) < 1e-6
        if has_spot_id:
            spot_id = int(round(values[0]))

        if len(values) >= 6 and has_spot_id:
            d_spacing_nm = float(values[1])
            g_magnitude = float(values[2])
            angle_x_deg = float(values[4])
            amplitude = float(values[5])
        elif len(values) >= 5 and has_spot_id:
            d_spacing_nm = float(values[1])
            g_magnitude = float(values[2])
            angle_x_deg = float(values[4])
        elif len(values) >= 5:
            d_spacing_nm = float(values[0])
            g_magnitude = float(values[1])
            angle_x_deg = float(values[3])
            amplitude = float(values[4])
        elif len(values) >= 4:
            d_spacing_nm = float(values[0])
            g_magnitude = float(values[1])
            angle_x_deg = float(values[2])
            amplitude = float(values[3])
        else:
            d_spacing_nm = float(values[0])
            g_magnitude = float(values[1])
            angle_x_deg = None

        if d_spacing_nm is not None and g_magnitude is not None:
            if d_spacing_nm > 1.0 and g_magnitude < 1.0:
                d_spacing_nm, g_magnitude = g_magnitude, d_spacing_nm

        if (d_spacing_nm is None or d_spacing_nm <= 0) and g_magnitude and g_magnitude > 0:
            d_spacing_nm = 1.0 / float(g_magnitude)
        if (g_magnitude is None or g_magnitude <= 0) and d_spacing_nm and d_spacing_nm > 0:
            g_magnitude = 1.0 / float(d_spacing_nm)

        if d_spacing_nm is None or d_spacing_nm <= 0 or g_magnitude is None or g_magnitude <= 0:
            warnings.append(f"Line {line_no}: skipped due to invalid d/g values.")
            continue

        rows.append(
            {
                "spot_id": int(spot_id),
                "d_spacing_nm": round(float(d_spacing_nm), 6),
                "g_magnitude": round(float(g_magnitude), 6),
                "angle_x_deg": (
                    None if angle_x_deg is None else round(_wrap_angle_deg(float(angle_x_deg)), 4)
                ),
                "amplitude": None if amplitude is None else round(float(amplitude), 6),
                "source_line": int(line_no),
            }
        )

    if not rows:
        warnings.append("No valid manual spot rows parsed from text.")
    return rows, list(dict.fromkeys(warnings))


def _build_align_candidates(
    peak_rows: list[dict[str, Any]],
    *,
    id_key: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in peak_rows:
        g_mag = _to_float(row.get("g_magnitude"))
        d_nm = _to_float(row.get("d_spacing"))
        gx = _to_float(row.get("gx"))
        gy = _to_float(row.get("gy"))
        if g_mag is None or d_nm is None or g_mag <= 0 or d_nm <= 0:
            continue
        angle_x_deg = None
        if gx is not None and gy is not None:
            angle_x_deg = float(np.degrees(np.arctan2(float(gy), float(gx))))
            angle_x_deg = _wrap_angle_deg(angle_x_deg)
        out.append(
            {
                "id": int(_to_float(row.get(id_key)) or len(out)),
                "g_magnitude": float(g_mag),
                "d_spacing_nm": float(d_nm),
                "angle_x_deg": angle_x_deg,
                "raw": row,
            }
        )
    return out


def _match_with_transform(
    manual_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    *,
    sign: int,
    angle_offset_deg: float,
    max_g_rel: float,
    max_d_rel: float,
    max_angle_deg: float,
    weight_g: float,
    weight_d: float,
    weight_angle: float,
) -> list[dict[str, Any]]:
    pair_candidates: list[dict[str, Any]] = []
    for m_idx, manual in enumerate(manual_rows):
        mg = float(manual["g_magnitude"])
        md = float(manual["d_spacing_nm"])
        ma = manual.get("angle_x_deg")
        for t_idx, target in enumerate(target_rows):
            tg = float(target["g_magnitude"])
            td = float(target["d_spacing_nm"])
            ta = target.get("angle_x_deg")
            g_rel = abs(tg - mg) / max(abs(mg), 1e-9)
            d_rel = abs(td - md) / max(abs(md), 1e-9)
            if g_rel > max_g_rel or d_rel > max_d_rel:
                continue
            angle_err = None
            if ma is not None and ta is not None:
                predicted = _wrap_angle_deg(float(sign) * float(ma) + float(angle_offset_deg))
                angle_err = _angle_error_deg(float(ta), predicted)
                if angle_err > max_angle_deg:
                    continue
            angle_term = 0.0 if angle_err is None else float(angle_err) / 90.0
            cost = (
                float(weight_g) * float(g_rel)
                + float(weight_d) * float(d_rel)
                + float(weight_angle) * float(angle_term)
            )
            pair_candidates.append(
                {
                    "manual_idx": int(m_idx),
                    "target_idx": int(t_idx),
                    "g_rel": float(g_rel),
                    "d_rel": float(d_rel),
                    "angle_err_deg": angle_err,
                    "cost": float(cost),
                }
            )
    pair_candidates.sort(key=lambda item: float(item["cost"]))

    used_manual: set[int] = set()
    used_target: set[int] = set()
    matches: list[dict[str, Any]] = []
    for item in pair_candidates:
        m_idx = int(item["manual_idx"])
        t_idx = int(item["target_idx"])
        if m_idx in used_manual or t_idx in used_target:
            continue
        used_manual.add(m_idx)
        used_target.add(t_idx)
        merged = dict(item)
        merged["manual"] = manual_rows[m_idx]
        merged["target"] = target_rows[t_idx]
        matches.append(merged)
    return matches


def _align_manual_spot_table(
    *,
    manual_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    max_g_rel: float,
    max_d_rel: float,
    max_angle_deg: float,
    weight_g: float,
    weight_d: float,
    weight_angle: float,
    angle_step_deg: float,
) -> dict[str, Any]:
    if not manual_rows:
        return {"warnings": ["Manual spot table is empty."]}
    if not target_rows:
        return {"warnings": ["Target peak table has no valid rows for alignment."]}
    step = max(0.25, float(angle_step_deg))
    offsets = np.arange(-180.0, 180.0, step, dtype=float)
    best: dict[str, Any] | None = None
    best_key: tuple[int, float] | None = None
    for sign in (1, -1):
        for offset in offsets:
            matches = _match_with_transform(
                manual_rows,
                target_rows,
                sign=sign,
                angle_offset_deg=float(offset),
                max_g_rel=float(max_g_rel),
                max_d_rel=float(max_d_rel),
                max_angle_deg=float(max_angle_deg),
                weight_g=float(weight_g),
                weight_d=float(weight_d),
                weight_angle=float(weight_angle),
            )
            matched = len(matches)
            if matched == 0:
                continue
            mean_cost = float(np.mean([float(item["cost"]) for item in matches]))
            key = (matched, -mean_cost)
            if best_key is None or key > best_key:
                best_key = key
                best = {
                    "sign": int(sign),
                    "angle_offset_deg": round(float(offset), 4),
                    "matches": matches,
                    "mean_cost": float(mean_cost),
                }

    if best is None:
        return {
            "warnings": [
                "No matches found under current alignment thresholds. Relax d/g/angle thresholds."
            ]
        }

    matches = list(best["matches"])
    used_manual = {int(item["manual_idx"]) for item in matches}
    used_target = {int(item["target_idx"]) for item in matches}
    manual_unmatched = [
        row for idx, row in enumerate(manual_rows) if int(idx) not in used_manual
    ]
    target_unmatched = [
        row for idx, row in enumerate(target_rows) if int(idx) not in used_target
    ]

    g_rel_vals = [float(item["g_rel"]) for item in matches]
    d_rel_vals = [float(item["d_rel"]) for item in matches]
    angle_vals = [
        float(item["angle_err_deg"])
        for item in matches
        if item.get("angle_err_deg") is not None
    ]

    match_rows: list[dict[str, Any]] = []
    for item in matches:
        manual = item["manual"]
        target = item["target"]
        match_rows.append(
            {
                "manual_spot_id": int(manual["spot_id"]),
                "target_peak_id": int(target["id"]),
                "manual_d_spacing_nm": round(float(manual["d_spacing_nm"]), 6),
                "target_d_spacing_nm": round(float(target["d_spacing_nm"]), 6),
                "d_rel_error_pct": round(float(item["d_rel"]) * 100.0, 4),
                "manual_g_1_per_nm": round(float(manual["g_magnitude"]), 6),
                "target_g_1_per_nm": round(float(target["g_magnitude"]), 6),
                "g_rel_error_pct": round(float(item["g_rel"]) * 100.0, 4),
                "manual_angle_x_deg": manual.get("angle_x_deg"),
                "target_angle_x_deg": target.get("angle_x_deg"),
                "angle_error_deg": (
                    None
                    if item.get("angle_err_deg") is None
                    else round(float(item["angle_err_deg"]), 4)
                ),
                "match_cost": round(float(item["cost"]), 6),
            }
        )
    match_rows.sort(key=lambda row: (int(row["manual_spot_id"]), int(row["target_peak_id"])))

    return {
        "warnings": [],
        "sign": int(best["sign"]),
        "angle_offset_deg": float(best["angle_offset_deg"]),
        "matched_count": int(len(matches)),
        "manual_count": int(len(manual_rows)),
        "target_count": int(len(target_rows)),
        "mean_cost": round(float(best["mean_cost"]), 6),
        "summary": {
            "g_rel_mean_pct": (
                round(float(np.mean(g_rel_vals)) * 100.0, 4) if g_rel_vals else None
            ),
            "g_rel_median_pct": (
                round(float(np.median(g_rel_vals)) * 100.0, 4)
                if g_rel_vals
                else None
            ),
            "g_rel_p90_pct": (
                None
                if _percentile(g_rel_vals, 90.0) is None
                else round(float(_percentile(g_rel_vals, 90.0)) * 100.0, 4)
            ),
            "d_rel_mean_pct": (
                round(float(np.mean(d_rel_vals)) * 100.0, 4) if d_rel_vals else None
            ),
            "d_rel_median_pct": (
                round(float(np.median(d_rel_vals)) * 100.0, 4)
                if d_rel_vals
                else None
            ),
            "d_rel_p90_pct": (
                None
                if _percentile(d_rel_vals, 90.0) is None
                else round(float(_percentile(d_rel_vals, 90.0)) * 100.0, 4)
            ),
            "angle_mean_deg": round(float(np.mean(angle_vals)), 4) if angle_vals else None,
            "angle_median_deg": round(float(np.median(angle_vals)), 4) if angle_vals else None,
            "angle_p90_deg": (
                None
                if _percentile(angle_vals, 90.0) is None
                else round(float(_percentile(angle_vals, 90.0)), 4)
            ),
        },
        "matches": match_rows,
        "manual_unmatched": [
            {
                "manual_spot_id": int(row.get("spot_id", 0)),
                "manual_d_spacing_nm": row.get("d_spacing_nm"),
                "manual_g_1_per_nm": row.get("g_magnitude"),
                "manual_angle_x_deg": row.get("angle_x_deg"),
                "amplitude": row.get("amplitude"),
            }
            for row in manual_unmatched
        ],
        "target_unmatched": [
            {
                "target_peak_id": int(row.get("id", 0)),
                "target_d_spacing_nm": round(float(row.get("d_spacing_nm", 0.0)), 6),
                "target_g_1_per_nm": round(float(row.get("g_magnitude", 0.0)), 6),
                "target_angle_x_deg": row.get("angle_x_deg"),
            }
            for row in target_unmatched
        ],
    }


def _measurement_value(measurement: Measurement | None) -> float | None:
    if measurement is None:
        return None
    if isinstance(measurement.value, (int, float)):
        return float(measurement.value)
    return None


def _measurement_uncertainty(measurement: Measurement | None) -> float | None:
    if measurement is None:
        return None
    if isinstance(measurement.uncertainty, (int, float)):
        return float(measurement.uncertainty)
    return None


def _normalize_for_display(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    if not finite.all():
        fill = float(np.nanmedian(arr[finite]))
        arr = np.where(finite, arr, fill)
    p1, p99 = np.percentile(arr, [1.0, 99.0])
    span = max(float(p99 - p1), 1e-12)
    out = (arr - float(p1)) / span
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _to_fft_display_image(
    fft_image_norm: np.ndarray,
    *,
    max_side: int = 900,
    zoom: float = 1.0,
    center_xy: tuple[float, float] | None = None,
) -> tuple[Image.Image, float, float, float, float]:
    h, w = fft_image_norm.shape
    zoom = max(1.0, float(zoom))
    if center_xy is None or zoom <= 1.0:
        x0, y0, x1, y1 = 0, 0, int(w), int(h)
    else:
        crop_w = max(1, min(int(w), int(round(float(w) / zoom))))
        crop_h = max(1, min(int(h), int(round(float(h) / zoom))))
        cx = float(np.clip(float(center_xy[0]), 0.0, max(0.0, float(w - 1))))
        cy = float(np.clip(float(center_xy[1]), 0.0, max(0.0, float(h - 1))))
        x0 = int(round(cx - 0.5 * crop_w))
        y0 = int(round(cy - 0.5 * crop_h))
        x0 = max(0, min(x0, int(w) - crop_w))
        y0 = max(0, min(y0, int(h) - crop_h))
        x1 = int(x0 + crop_w)
        y1 = int(y0 + crop_h)

    crop = np.asarray(fft_image_norm[y0:y1, x0:x1], dtype=np.float32)
    crop_h_actual, crop_w_actual = crop.shape
    scale = max(0.05, float(max_side) / max(crop_h_actual, crop_w_actual))
    disp_h = max(1, int(round(crop_h_actual * scale)))
    disp_w = max(1, int(round(crop_w_actual * scale)))
    arr = (np.clip(crop, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB").resize(
        (disp_w, disp_h),
        resample=Image.BILINEAR,
    )
    sx = crop_w_actual / disp_w
    sy = crop_h_actual / disp_h
    return img, sx, sy, float(x0), float(y0)


def _extract_canvas_points_xy(
    canvas_result: Any,
    *,
    sx: float,
    sy: float,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
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
        try:
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
        except (TypeError, ValueError):
            continue

        radius = obj.get("radius")
        width = obj.get("width")
        height = obj.get("height")
        if radius is not None:
            try:
                r = float(radius)
            except (TypeError, ValueError):
                r = 0.0
            disp_x = left + r
            disp_y = top + r
        elif width is not None and height is not None:
            try:
                w = float(width)
                h = float(height)
            except (TypeError, ValueError):
                w = 0.0
                h = 0.0
            disp_x = left + 0.5 * w
            disp_y = top + 0.5 * h
        else:
            disp_x = left
            disp_y = top
        points.append((float(x_offset) + disp_x * sx, float(y_offset) + disp_y * sy))
    return points


def _draw_auto_points_on_canvas_image(
    image: Image.Image,
    *,
    auto_points_xy: list[tuple[float, float]],
    center_xy: tuple[float, float],
    sx: float,
    sy: float,
    x_offset: float,
    y_offset: float,
    removed_auto_peak_ids: set[int] | None = None,
    manual_points_xy: list[tuple[float, float]] | None = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    removed_ids = set(removed_auto_peak_ids or set())
    for idx, (x, y) in enumerate(auto_points_xy):
        dx = (float(x) - float(x_offset)) / max(float(sx), 1e-12)
        dy = (float(y) - float(y_offset)) / max(float(sy), 1e-12)
        if not (-8.0 <= dx <= float(width + 8) and -8.0 <= dy <= float(height + 8)):
            continue
        if int(idx) in removed_ids:
            draw.ellipse((dx - 5, dy - 5, dx + 5, dy + 5), outline="red", width=2)
            draw.line((dx - 5, dy - 5, dx + 5, dy + 5), fill="red", width=2)
            draw.line((dx - 5, dy + 5, dx + 5, dy - 5), fill="red", width=2)
            draw.text((dx + 6, dy - 11), str(idx), fill="red")
        else:
            draw.ellipse((dx - 4, dy - 4, dx + 4, dy + 4), outline="cyan", width=2)
            draw.text((dx + 5, dy - 10), str(idx), fill="cyan")

    for idx, (x, y) in enumerate(manual_points_xy or []):
        dx = (float(x) - float(x_offset)) / max(float(sx), 1e-12)
        dy = (float(y) - float(y_offset)) / max(float(sy), 1e-12)
        if not (-8.0 <= dx <= float(width + 8) and -8.0 <= dy <= float(height + 8)):
            continue
        draw.rectangle((dx - 5, dy - 5, dx + 5, dy + 5), outline="magenta", width=2)
        draw.text((dx + 6, dy - 11), f"M{idx}", fill="magenta")

    cx = (float(center_xy[0]) - float(x_offset)) / max(float(sx), 1e-12)
    cy = (float(center_xy[1]) - float(y_offset)) / max(float(sy), 1e-12)
    if -8.0 <= cx <= float(width + 8) and -8.0 <= cy <= float(height + 8):
        draw.line((cx - 6, cy, cx + 6, cy), fill="yellow", width=2)
        draw.line((cx, cy - 6, cx, cy + 6), fill="yellow", width=2)
    return out


def _normalize_points_xy(raw_points: Any) -> list[tuple[float, float]]:
    if not isinstance(raw_points, list):
        return []
    points: list[tuple[float, float]] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue
        points.append((x, y))
    return points


def _nearest_point_index(
    points_xy: list[tuple[float, float]],
    target_xy: tuple[float, float],
    *,
    hit_radius_px: float,
) -> tuple[int, float | None]:
    radius = max(0.1, float(hit_radius_px))
    best_idx = -1
    best_dist = radius + 1e-9
    tx, ty = target_xy
    for idx, (px, py) in enumerate(points_xy):
        dist = float(np.hypot(float(tx) - float(px), float(ty) - float(py)))
        if dist <= radius and dist < best_dist:
            best_idx = int(idx)
            best_dist = dist
    return best_idx, (None if best_idx < 0 else float(best_dist))


def _append_manual_canvas_points(
    existing_points_xy: list[tuple[float, float]],
    clicks_xy: list[tuple[float, float]],
    *,
    manual_min_distance_px: float,
    max_points: int,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    selected = list(existing_points_xy)
    added_count = 0
    duplicate_count = 0
    capacity_count = 0
    min_distance = max(0.1, float(manual_min_distance_px))
    limit = max(1, int(max_points))

    for x, y in clicks_xy:
        point = (float(x), float(y))
        if any(
            float(np.hypot(point[0] - old_x, point[1] - old_y)) < min_distance
            for old_x, old_y in selected
        ):
            duplicate_count += 1
            continue
        if len(selected) >= limit:
            capacity_count += 1
            continue
        selected.append(point)
        added_count += 1

    return selected, {
        "mode": "add",
        "click_count": int(len(clicks_xy)),
        "manual_add_count": int(added_count),
        "duplicate_click_count": int(duplicate_count),
        "capacity_skip_count": int(capacity_count),
    }


def _resolve_delete_canvas_clicks(
    clicks_xy: list[tuple[float, float]],
    *,
    auto_points_xy: list[tuple[float, float]],
    manual_points_xy: list[tuple[float, float]],
    hit_radius_px: float,
) -> tuple[list[int], list[int], dict[str, Any]]:
    remove_auto_ids: set[int] = set()
    remove_manual_ids: set[int] = set()
    missed_count = 0

    for x, y in clicks_xy:
        point = (float(x), float(y))
        auto_idx, auto_dist = _nearest_point_index(
            auto_points_xy,
            point,
            hit_radius_px=float(hit_radius_px),
        )
        manual_idx, manual_dist = _nearest_point_index(
            manual_points_xy,
            point,
            hit_radius_px=float(hit_radius_px),
        )
        if manual_idx >= 0 and (
            auto_idx < 0
            or (
                manual_dist is not None
                and auto_dist is not None
                and float(manual_dist) <= float(auto_dist)
            )
        ):
            remove_manual_ids.add(int(manual_idx))
        elif auto_idx >= 0:
            remove_auto_ids.add(int(auto_idx))
        else:
            missed_count += 1

    return sorted(remove_auto_ids), sorted(remove_manual_ids), {
        "mode": "delete",
        "click_count": int(len(clicks_xy)),
        "removed_auto_peak_ids": sorted(int(item) for item in remove_auto_ids),
        "removed_manual_point_ids": sorted(int(item) for item in remove_manual_ids),
        "missed_click_count": int(missed_count),
        "hit_radius_px": float(max(0.1, float(hit_radius_px))),
    }


def _deduplicate_points(
    points_xy: list[tuple[float, float]],
    *,
    min_distance_px: float,
) -> list[tuple[float, float]]:
    selected: list[tuple[float, float]] = []
    for x, y in points_xy:
        if any(np.hypot(x - sx, y - sy) < min_distance_px for sx, sy in selected):
            continue
        selected.append((x, y))
    return selected


def _round_points_for_signature(
    points_xy: list[tuple[float, float]],
    *,
    digits: int = 3,
) -> list[list[float]]:
    return [
        [round(float(x), digits), round(float(y), digits)]
        for x, y in points_xy
    ]


def _sort_points_by_fft_intensity(
    points_xy: list[tuple[float, float]],
    fft_log_magnitude: np.ndarray,
) -> list[tuple[float, float]]:
    h, w = fft_log_magnitude.shape

    def _score(point: tuple[float, float]) -> float:
        x, y = point
        xx = int(np.clip(round(float(x)), 0, w - 1))
        yy = int(np.clip(round(float(y)), 0, h - 1))
        return float(fft_log_magnitude[yy, xx])

    return sorted(
        [(float(x), float(y)) for x, y in points_xy],
        key=_score,
        reverse=True,
    )


def _generate_lattice_points_from_two_basis(
    *,
    center_xy: tuple[float, float],
    basis1_xy: tuple[float, float],
    basis2_xy: tuple[float, float],
    shape_hw: tuple[int, int],
    max_order: int,
    min_distance: float,
) -> list[tuple[float, float]]:
    cx, cy = float(center_xy[0]), float(center_xy[1])
    p1x, p1y = float(basis1_xy[0]), float(basis1_xy[1])
    p2x, p2y = float(basis2_xy[0]), float(basis2_xy[1])
    v1x, v1y = p1x - cx, p1y - cy
    v2x, v2y = p2x - cx, p2y - cy
    h, w = shape_hw
    points: list[tuple[float, float]] = []
    radius_ref = float(max(np.hypot(v1x, v1y), np.hypot(v2x, v2y), 1e-9))

    for m in range(-int(max_order), int(max_order) + 1):
        for n in range(-int(max_order), int(max_order) + 1):
            if m == 0 and n == 0:
                continue
            # Skip very high-order points far from the basis shell to reduce noise picks.
            if np.hypot(float(m), float(n)) > float(max_order) + 0.35:
                continue
            x = cx + float(m) * v1x + float(n) * v2x
            y = cy + float(m) * v1y + float(n) * v2y
            if not (0.0 <= x < float(w) and 0.0 <= y < float(h)):
                continue
            if np.hypot(x - cx, y - cy) < 0.25 * radius_ref:
                continue
            points.append((float(x), float(y)))

    points = _deduplicate_points(points, min_distance_px=max(0.1, float(min_distance)))
    points.sort(key=lambda p: float(np.hypot(float(p[0]) - cx, float(p[1]) - cy)))
    return points


def _nearest_point_within_radius(
    point_xy: tuple[float, float],
    candidates_xy: list[tuple[float, float]],
    *,
    max_radius_px: float,
) -> tuple[tuple[float, float], float] | None:
    if not candidates_xy:
        return None
    px, py = float(point_xy[0]), float(point_xy[1])
    best_idx = -1
    best_dist = float(max_radius_px) + 1e-9
    for idx, (cx, cy) in enumerate(candidates_xy):
        dist = float(np.hypot(float(cx) - px, float(cy) - py))
        if dist <= float(max_radius_px) and dist < best_dist:
            best_dist = dist
            best_idx = int(idx)
    if best_idx < 0:
        return None
    candidate = candidates_xy[best_idx]
    return (float(candidate[0]), float(candidate[1])), float(best_dist)


def _find_local_max_near_point(
    image: np.ndarray,
    *,
    x: float,
    y: float,
    radius_px: float,
) -> tuple[tuple[float, float], float, float] | None:
    h, w = image.shape
    xi = float(np.clip(float(x), 0.0, max(0.0, float(w - 1))))
    yi = float(np.clip(float(y), 0.0, max(0.0, float(h - 1))))
    r = max(1, int(np.ceil(float(radius_px))))
    x0 = max(0, int(np.floor(xi)) - r)
    x1 = min(w, int(np.floor(xi)) + r + 1)
    y0 = max(0, int(np.floor(yi)) - r)
    y1 = min(h, int(np.floor(yi)) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    if patch.size == 0 or not np.isfinite(patch).any():
        return None
    finite = np.isfinite(patch)
    patch_safe = np.where(finite, patch, -np.inf)
    max_idx = int(np.argmax(patch_safe))
    yy, xx = np.unravel_index(max_idx, patch_safe.shape)
    xf = float(x0 + xx)
    yf = float(y0 + yy)
    intensity = float(patch_safe[yy, xx])
    dist = float(np.hypot(xf - xi, yf - yi))
    return (xf, yf), intensity, dist


def _synthesize_points_from_two_basis(
    *,
    center_xy: tuple[float, float],
    basis_points_xy: list[tuple[float, float]],
    shape_hw: tuple[int, int],
    fft_log_magnitude: np.ndarray,
    max_order: int,
    min_distance_px: float,
    snap_mode: str,
    snap_radius_px: float,
    local_intensity_percentile: float,
    max_points: int,
) -> tuple[list[tuple[float, float]], dict[str, Any], list[str]]:
    warnings: list[str] = []
    if len(basis_points_xy) < 2:
        return [], {"enabled": True, "reason": "insufficient_basis_points"}, [
            "Need at least two basis points for vector synthesis."
        ]
    basis1 = (float(basis_points_xy[0][0]), float(basis_points_xy[0][1]))
    basis2 = (float(basis_points_xy[1][0]), float(basis_points_xy[1][1]))
    cx, cy = float(center_xy[0]), float(center_xy[1])
    v1 = np.array([basis1[0] - cx, basis1[1] - cy], dtype=np.float64)
    v2 = np.array([basis2[0] - cx, basis2[1] - cy], dtype=np.float64)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 <= 1.0 or n2 <= 1.0:
        warnings.append("Basis vectors are too short. Please click farther diffraction spots.")
    cosang = float(np.dot(v1, v2) / max(1e-12, n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    angle_deg = float(np.degrees(np.arccos(cosang)))
    if angle_deg < 10.0 or angle_deg > 170.0:
        warnings.append(
            "Basis vectors are nearly collinear "
            f"(angle={angle_deg:.2f} deg); synthesis may be unstable."
        )

    generated = _generate_lattice_points_from_two_basis(
        center_xy=(cx, cy),
        basis1_xy=basis1,
        basis2_xy=basis2,
        shape_hw=shape_hw,
        max_order=max(1, int(max_order)),
        min_distance=max(0.1, float(min_distance_px)),
    )
    seed_points = _deduplicate_points(
        [basis1, basis2] + list(generated),
        min_distance_px=max(0.1, float(min_distance_px)),
    )
    if int(max_points) > 0:
        seed_points = seed_points[: int(max_points)]

    image = np.asarray(fft_log_magnitude, dtype=np.float64)
    finite_vals = image[np.isfinite(image)]
    if finite_vals.size:
        q = float(np.clip(float(local_intensity_percentile), 50.0, 99.99))
        local_threshold = float(np.percentile(finite_vals, q))
    else:
        local_threshold = float("-inf")

    mode = str(snap_mode).strip().lower()
    if mode not in {"none", "local_max"}:
        mode = "local_max"
    snapped_points: list[tuple[float, float]] = []
    source_counts: dict[str, int] = {
        "basis_keep": 0,
        "local_max": 0,
        "lattice_seed": 0,
    }
    dropped_non_basis = 0
    pairs: list[dict[str, Any]] = []

    for idx, (px, py) in enumerate(seed_points):
        is_basis = idx < 2
        best_pt: tuple[float, float] | None = None
        best_dist = float("inf")
        best_src = ""

        if mode == "local_max":
            local_hit = _find_local_max_near_point(
                image,
                x=float(px),
                y=float(py),
                radius_px=float(snap_radius_px),
            )
            if local_hit is not None:
                (lx, ly), lv, ld = local_hit
                if lv >= float(local_threshold) or is_basis:
                    if best_pt is None or float(ld) < float(best_dist):
                        best_pt = (float(lx), float(ly))
                        best_dist = float(ld)
                        best_src = "local_max"
        elif mode == "none":
            best_pt = (float(px), float(py))
            best_dist = 0.0
            best_src = "basis_keep" if is_basis else "lattice_seed"

        if best_pt is None:
            if is_basis:
                best_pt = (float(px), float(py))
                best_src = "basis_keep"
                best_dist = 0.0
            else:
                dropped_non_basis += 1
                continue

        source_counts[best_src] = int(source_counts.get(best_src, 0)) + 1
        snapped_points.append(best_pt)
        pairs.append(
            {
                "x_seed": round(float(px), 3),
                "y_seed": round(float(py), 3),
                "x_selected": round(float(best_pt[0]), 3),
                "y_selected": round(float(best_pt[1]), 3),
                "shift_px": round(float(best_dist), 4),
                "source": best_src,
                "is_basis": bool(is_basis),
            }
        )

    snapped_points = _deduplicate_points(
        snapped_points,
        min_distance_px=max(0.1, float(min_distance_px)),
    )
    snapped_points.sort(key=lambda p: float(np.hypot(float(p[0]) - cx, float(p[1]) - cy)))
    if int(max_points) > 0 and len(snapped_points) > int(max_points):
        snapped_points = snapped_points[: int(max_points)]

    diagnostics = {
        "enabled": True,
        "mode": mode,
        "basis_angle_deg": round(float(angle_deg), 4),
        "basis_norm_px": [round(float(n1), 4), round(float(n2), 4)],
        "generated_count": int(len(generated)),
        "seed_count": int(len(seed_points)),
        "selected_count": int(len(snapped_points)),
        "dropped_non_basis_count": int(dropped_non_basis),
        "snap_radius_px": float(snap_radius_px),
        "local_intensity_percentile": float(local_intensity_percentile),
        "local_intensity_threshold": float(local_threshold),
        "source_counts": source_counts,
        "seed_to_selected_pairs": pairs,
    }
    if dropped_non_basis > 0:
        warnings.append(
            f"Vector synthesis dropped {dropped_non_basis} predicted point(s) "
            "with no reliable local match."
        )
    return snapped_points, diagnostics, warnings


def _refine_point_centroid(
    image: np.ndarray,
    *,
    x: float,
    y: float,
    window_radius: int,
    max_shift_px: float,
) -> tuple[float, float, str]:
    h, w = image.shape
    xi = float(np.clip(x, 0.0, max(0.0, w - 1)))
    yi = float(np.clip(y, 0.0, max(0.0, h - 1)))
    x0 = max(0, int(np.floor(xi)) - int(window_radius))
    x1 = min(w, int(np.floor(xi)) + int(window_radius) + 1)
    y0 = max(0, int(np.floor(yi)) - int(window_radius))
    y1 = min(h, int(np.floor(yi)) + int(window_radius) + 1)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    if patch.size == 0:
        return xi, yi, "centroid_empty_patch"
    if not np.isfinite(patch).any():
        return xi, yi, "centroid_invalid_patch"
    finite_patch = patch[np.isfinite(patch)]
    floor = float(np.min(finite_patch))
    weights = np.maximum(patch - floor, 0.0)
    total = float(np.sum(weights))
    if total <= 1e-12:
        return xi, yi, "centroid_low_contrast"
    yy, xx = np.indices(patch.shape, dtype=np.float64)
    abs_x = xx + float(x0)
    abs_y = yy + float(y0)
    xf = float(np.sum(weights * abs_x) / total)
    yf = float(np.sum(weights * abs_y) / total)
    shift = float(np.hypot(xf - xi, yf - yi))
    if shift > float(max_shift_px) and shift > 1e-12:
        ratio = float(max_shift_px) / shift
        xf = xi + (xf - xi) * ratio
        yf = yi + (yf - yi) * ratio
        return float(xf), float(yf), "centroid_clamped"
    return float(xf), float(yf), "centroid_ok"


def _refine_point_gaussian(
    image: np.ndarray,
    *,
    x: float,
    y: float,
    window_radius: int,
    max_shift_px: float,
) -> tuple[float, float, str]:
    h, w = image.shape
    xi = float(np.clip(x, 0.0, max(0.0, w - 1)))
    yi = float(np.clip(y, 0.0, max(0.0, h - 1)))
    x0 = max(0, int(np.floor(xi)) - int(window_radius))
    x1 = min(w, int(np.floor(xi)) + int(window_radius) + 1)
    y0 = max(0, int(np.floor(yi)) - int(window_radius))
    y1 = min(h, int(np.floor(yi)) + int(window_radius) + 1)
    patch = np.asarray(image[y0:y1, x0:x1], dtype=np.float64)
    if patch.size < 9 or not np.isfinite(patch).any():
        xf, yf, tag = _refine_point_centroid(
            image,
            x=xi,
            y=yi,
            window_radius=window_radius,
            max_shift_px=max_shift_px,
        )
        return xf, yf, f"gaussian_fallback_{tag}"

    yy, xx = np.indices(patch.shape, dtype=np.float64)
    abs_x = xx + float(x0)
    abs_y = yy + float(y0)
    finite_patch = patch[np.isfinite(patch)]
    bg0 = float(np.percentile(finite_patch, 10.0))
    signal = np.maximum(patch - bg0, 0.0)
    total = float(np.sum(signal))
    if total <= 1e-12:
        xf, yf, tag = _refine_point_centroid(
            image,
            x=xi,
            y=yi,
            window_radius=window_radius,
            max_shift_px=max_shift_px,
        )
        return xf, yf, f"gaussian_fallback_{tag}"

    cx0 = float(np.sum(signal * abs_x) / total)
    cy0 = float(np.sum(signal * abs_y) / total)
    amp0 = float(np.max(signal))
    std_patch = float(np.std(finite_patch))
    bg_low = float(np.min(finite_patch) - max(1e-6, 2.0 * std_patch))
    bg_high = float(np.max(finite_patch) + max(1e-6, 2.0 * std_patch))
    amp_high = max(1e-6, 5.0 * max(amp0, std_patch))
    sigma_high = max(0.8, 2.5 * float(window_radius))
    cx_low = max(float(x0), xi - float(max_shift_px))
    cx_high = min(float(x1 - 1), xi + float(max_shift_px))
    cy_low = max(float(y0), yi - float(max_shift_px))
    cy_high = min(float(y1 - 1), yi + float(max_shift_px))
    if cx_low >= cx_high or cy_low >= cy_high:
        xf, yf, tag = _refine_point_centroid(
            image,
            x=xi,
            y=yi,
            window_radius=window_radius,
            max_shift_px=max_shift_px,
        )
        return xf, yf, f"gaussian_fallback_{tag}"

    p0 = np.array(
        [
            bg0,
            max(1e-6, amp0),
            float(np.clip(cx0, cx_low, cx_high)),
            float(np.clip(cy0, cy_low, cy_high)),
            max(0.6, float(window_radius) * 0.8),
            max(0.6, float(window_radius) * 0.8),
        ],
        dtype=np.float64,
    )
    lower = np.array([bg_low, 0.0, cx_low, cy_low, 0.35, 0.35], dtype=np.float64)
    upper = np.array(
        [bg_high, amp_high, cx_high, cy_high, sigma_high, sigma_high],
        dtype=np.float64,
    )

    def _residual(params: np.ndarray) -> np.ndarray:
        bg, amp, cx, cy, sx, sy = [float(v) for v in params]
        model = bg + amp * np.exp(
            -0.5 * (((abs_x - cx) / max(0.2, sx)) ** 2 + ((abs_y - cy) / max(0.2, sy)) ** 2)
        )
        return (model - patch).ravel()

    try:
        result = least_squares(
            _residual,
            p0,
            bounds=(lower, upper),
            method="trf",
            loss="soft_l1",
            max_nfev=80,
        )
        if not result.success:
            xf, yf, tag = _refine_point_centroid(
                image,
                x=xi,
                y=yi,
                window_radius=window_radius,
                max_shift_px=max_shift_px,
            )
            return xf, yf, f"gaussian_fallback_{tag}"
        cx_fit = float(result.x[2])
        cy_fit = float(result.x[3])
        return cx_fit, cy_fit, "gaussian_ok"
    except Exception:  # noqa: BLE001
        xf, yf, tag = _refine_point_centroid(
            image,
            x=xi,
            y=yi,
            window_radius=window_radius,
            max_shift_px=max_shift_px,
        )
        return xf, yf, f"gaussian_fallback_{tag}"


def _refine_manual_points(
    points_xy: list[tuple[float, float]],
    *,
    fft_log_magnitude: np.ndarray,
    method: str,
    window_radius: int,
    max_shift_px: float,
) -> tuple[list[tuple[float, float]], dict[str, Any], list[str]]:
    mode = str(method or "gaussian").strip().lower()
    if mode not in {"none", "centroid", "gaussian"}:
        mode = "gaussian"
    if not points_xy or mode == "none":
        return (
            list(points_xy),
            {
                "enabled": False,
                "method": mode,
                "point_count": int(len(points_xy)),
                "refined_count": int(len(points_xy)),
                "mean_shift_px": 0.0,
                "max_shift_px": 0.0,
                "fallback_count": 0,
            },
            [],
        )

    image = np.asarray(fft_log_magnitude, dtype=np.float64)
    refined: list[tuple[float, float]] = []
    shifts: list[float] = []
    fallback_count = 0
    tags: list[str] = []
    point_pairs: list[dict[str, Any]] = []
    for x, y in points_xy:
        x0 = float(x)
        y0 = float(y)
        if mode == "centroid":
            xf, yf, tag = _refine_point_centroid(
                image,
                x=x0,
                y=y0,
                window_radius=max(1, int(window_radius)),
                max_shift_px=max(0.0, float(max_shift_px)),
            )
        else:
            xf, yf, tag = _refine_point_gaussian(
                image,
                x=x0,
                y=y0,
                window_radius=max(1, int(window_radius)),
                max_shift_px=max(0.0, float(max_shift_px)),
            )
        if "fallback" in tag:
            fallback_count += 1
        tags.append(tag)
        refined.append((float(xf), float(yf)))
        shift_px = float(np.hypot(float(xf) - x0, float(yf) - y0))
        shifts.append(shift_px)
        point_pairs.append(
            {
                "x_raw": round(x0, 3),
                "y_raw": round(y0, 3),
                "x_refined": round(float(xf), 3),
                "y_refined": round(float(yf), 3),
                "shift_px": round(float(shift_px), 4),
                "tag": tag,
            }
        )

    mean_shift = float(np.mean(shifts)) if shifts else 0.0
    max_shift = float(np.max(shifts)) if shifts else 0.0
    diagnostics = {
        "enabled": True,
        "method": mode,
        "window_radius_px": int(max(1, int(window_radius))),
        "max_shift_limit_px": float(max(0.0, float(max_shift_px))),
        "point_count": int(len(points_xy)),
        "refined_count": int(len(refined)),
        "mean_shift_px": float(mean_shift),
        "max_shift_px": float(max_shift),
        "fallback_count": int(fallback_count),
        "tags": list(tags),
        "point_pairs": point_pairs,
    }
    warnings: list[str] = []
    if fallback_count > 0:
        warnings.append(
            f"Manual refine fallback occurred for {fallback_count}/{len(points_xy)} point(s)."
        )
    return refined, diagnostics, warnings


def _array_from_reference(array_ref: Any) -> np.ndarray | None:
    if array_ref is None:
        return None
    inline = getattr(array_ref, "inline", None)
    shape = tuple(int(v) for v in getattr(array_ref, "shape", ()))
    if inline is not None:
        try:
            arr = np.asarray(inline, dtype=np.float32)
            if shape and arr.size == int(np.prod(shape)):
                arr = arr.reshape(shape)
            if arr.ndim == 2:
                return arr
        except Exception:  # noqa: BLE001
            pass

    uri = getattr(array_ref, "uri", None)
    if not isinstance(uri, str) or not uri:
        return None
    path = Path(uri)
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            return np.asarray(arr, dtype=np.float32) if arr.ndim == 2 else None
        if suffix in {".txt", ".csv"}:
            arr = np.loadtxt(path)
            return np.asarray(arr, dtype=np.float32) if arr.ndim == 2 else None
        if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            with Image.open(path) as handle:
                image = np.asarray(handle)
            if image.ndim == 3:
                image = image.mean(axis=2)
            return np.asarray(image, dtype=np.float32) if image.ndim == 2 else None
    except Exception:  # noqa: BLE001
        return None
    return None


def _build_demo_input(run_dir: Path) -> Path:
    rng = np.random.default_rng(20260524)
    h, w = 256, 256
    image = rng.normal(loc=0.03, scale=0.008, size=(h, w)).astype(np.float32)
    cy, cx = h // 2, w // 2
    points = [(0, 34), (0, -34), (22, 18), (-22, -18), (26, -20), (-26, 20)]
    for dy, dx in points:
        yy = cy + dy
        xx = cx + dx
        if 2 <= yy < h - 2 and 2 <= xx < w - 2:
            image[yy - 1 : yy + 2, xx - 1 : xx + 2] += 1.0
    in_dir = run_dir / "inputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_path = in_dir / "demo_b1_b3.npy"
    np.save(out_path, image)
    return out_path


def _save_uploaded(uploaded_file, run_dir: Path) -> Path:
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    path = input_dir / _safe_name(uploaded_file.name)
    path.write_bytes(uploaded_file.getbuffer())
    return path


def _parse_candidate_phase_params(raw_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    text = str(raw_text).strip()
    if not text:
        return [], []
    try:
        payload = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        return [], [f"candidate_phase_params JSON parse failed: {exc}"]
    if not isinstance(payload, list):
        return [], ["candidate_phase_params must be a JSON list."]
    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        return [], ["candidate_phase_params has no valid mapping rows."]
    return rows, []


def _baseline_indexed_spots(d_spacings: list[Measurement | None]) -> list[IndexedSpot]:
    out: list[IndexedSpot] = []
    for idx, d_spacing in enumerate(d_spacings):
        out.append(
            IndexedSpot(
                peak_index=idx,
                h=0,
                k=0,
                l=0,
                d_spacing=d_spacing,
                error=Measurement(value=1.0, unit="relative", method="manual_unindexed"),
                confidence=0.1,
            )
        )
    return out


def _phase_placeholders(names: list[str], top_n: int) -> list[PhaseCandidate]:
    chosen = [name for name in names if name][: max(1, int(top_n))]
    if not chosen:
        return []
    return [
        PhaseCandidate(
            name=name,
            source="candidate_phases",
            score=max(0.1, 0.35 - 0.04 * idx),
            matched_spot_count=0,
            warnings=["Placeholder candidate; candidate_phase_params not used in this mode."],
        )
        for idx, name in enumerate(chosen)
    ]


def _phase_candidate_param_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("name", "")).strip().casefold(),
        str(row.get("identifier", "")).strip().casefold(),
        str(row.get("source", "")).strip().casefold(),
    )


def _phase_candidate_to_param(candidate: PhaseCandidate) -> dict[str, Any] | None:
    lattice = dict(candidate.lattice_parameters or {})
    required = {"a", "b", "c"}
    if not required.issubset(lattice):
        return None
    row = dict(lattice)
    row.update(
        {
            "name": candidate.name,
            "formula": candidate.formula,
            "source": candidate.source or "diffraction_phase_candidates",
            "identifier": candidate.identifier,
        }
    )
    return {key: value for key, value in row.items() if value is not None}


def _merge_candidate_phase_params(
    base_rows: list[dict[str, Any]],
    phase_candidates: list[PhaseCandidate],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in base_rows if isinstance(row, dict)]
    existing = {_phase_candidate_param_key(row) for row in rows}
    for candidate in phase_candidates:
        row = _phase_candidate_to_param(candidate)
        if row is None:
            continue
        key = _phase_candidate_param_key(row)
        if key not in existing:
            rows.append(row)
            existing.add(key)
    return rows


def _build_peak_rows(
    *,
    peaks: list[DiffractionPeak],
    g_vectors: list[Any],
    d_by_peak: list[Measurement | None],
    indexed_spots: list[IndexedSpot],
) -> list[dict[str, Any]]:
    indexed_map = {int(spot.peak_index): spot for spot in indexed_spots}
    rows: list[dict[str, Any]] = []
    for idx, peak in enumerate(peaks):
        g_item = g_vectors[idx] if idx < len(g_vectors) else None
        d_item = d_by_peak[idx] if idx < len(d_by_peak) else None
        spot = indexed_map.get(idx)

        gx = _measurement_value(getattr(g_item, "gx", None))
        gy = _measurement_value(getattr(g_item, "gy", None))
        gm = _measurement_value(getattr(g_item, "magnitude", None))
        dval = _measurement_value(d_item)
        dunc = _measurement_uncertainty(d_item)

        rows.append(
            {
                "peak_id": idx,
                "x": round(float(peak.x), 4),
                "y": round(float(peak.y), 4),
                "intensity": (
                    None if peak.intensity is None else round(float(peak.intensity), 6)
                ),
                "peak_confidence": (
                    None if peak.confidence is None else round(float(peak.confidence), 4)
                ),
                "gx": None if gx is None else round(float(gx), 6),
                "gy": None if gy is None else round(float(gy), 6),
                "g_magnitude": None if gm is None else round(float(gm), 6),
                "g_unit": (
                    "-"
                    if g_item is None
                    else str(getattr(getattr(g_item, "gx", None), "unit", "-"))
                ),
                "d_spacing": None if dval is None else round(float(dval), 6),
                "d_uncertainty": None if dunc is None else round(float(dunc), 6),
                "d_unit": "-" if d_item is None else str(d_item.unit or "-"),
                "h": 0 if spot is None else int(spot.h),
                "k": 0 if spot is None else int(spot.k),
                "l": 0 if spot is None else int(spot.l),
                "index_confidence": (
                    None
                    if spot is None or spot.confidence is None
                    else round(float(spot.confidence), 4)
                ),
            }
        )
    return rows


def _build_phase_rows(phase_candidates: list[PhaseCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in phase_candidates:
        lattice = item.lattice_parameters if isinstance(item.lattice_parameters, dict) else {}
        atom_sites = lattice.get("atom_sites") if isinstance(lattice, dict) else None
        rows.append(
            {
                "name": item.name,
                "formula": item.formula,
                "source": item.source,
                "identifier": item.identifier,
                "score": item.score,
                "matched_spot_count": item.matched_spot_count,
                "zone_axis": item.zone_axis,
                "mean_d_error": (
                    None
                    if item.mean_d_error is None or item.mean_d_error.value is None
                    else item.mean_d_error.value
                ),
                "mean_angle_error": (
                    None
                    if item.mean_angle_error is None or item.mean_angle_error.value is None
                    else item.mean_angle_error.value
                ),
                "space_group": lattice.get("space_group"),
                "space_group_number": lattice.get("space_group_number"),
                "atom_site_count": len(atom_sites) if isinstance(atom_sites, list) else None,
                "structure_source": lattice.get("structure_source"),
            }
        )
    return rows


def _diffraction_from_manual_result(manual_result: dict[str, Any]) -> DiffractionResult:
    d_by_peak = [item for item in manual_result.get("d_by_peak", []) if item is not None]
    return DiffractionResult(
        peaks=list(manual_result.get("peaks", [])),
        g_vectors=list(manual_result.get("g_vectors", [])),
        d_spacings=list(d_by_peak),
        indexed_spots=list(manual_result.get("indexed_spots", [])),
        phase_candidates=list(manual_result.get("phase_candidates", [])),
        provenance=dict(manual_result.get("provenance", {})),
        warnings=list(manual_result.get("warnings", [])),
    )


def _simulation_worker(
    diffraction_payload: dict[str, Any],
    parameters: dict[str, Any],
    queue: Any,
) -> None:
    try:
        diffraction = DiffractionResult.model_validate(diffraction_payload)
        simulation = simulate_from_diffraction(diffraction, parameters=parameters)
        queue.put(
            {
                "ok": True,
                "simulation": simulation.model_dump(mode="json"),
                "diffraction": diffraction.model_dump(mode="json"),
            }
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


def _run_b5_simulation_with_timeout(
    diffraction: DiffractionResult,
    *,
    parameters: dict[str, Any],
    timeout_s: float,
) -> tuple[SimulationResult, DiffractionResult, list[str]]:
    engine = str(parameters.get("simulation_engine", "diffsims")).strip().lower()
    if engine != "diffsims":
        simulation = simulate_from_diffraction(diffraction, parameters=parameters)
        return simulation, diffraction, list(simulation.warnings)

    timeout = max(1.0, float(timeout_s))
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_simulation_worker,
        args=(diffraction.model_dump(mode="json"), dict(parameters), queue),
    )
    process.start()
    status, payload = collect_process_queue_payload(
        process,
        queue,
        timeout_s=timeout,
        settle_s=2.0,
    )
    if status == "timeout":
        warning = (
            "B5 diffsims simulation timed out in the UI subprocess "
            f"after {timeout:.1f}s; reduce candidates/zone axes or increase timeout."
        )
        return (
            SimulationResult(
                matched_phase_candidates=list(diffraction.phase_candidates),
                warnings=[warning],
                provenance={
                    "method": "diffsims",
                    "ui_timeout": True,
                    "timeout_s": float(timeout),
                },
            ),
            diffraction,
            [warning],
        )

    if payload is None:
        warning = "B5 diffsims subprocess exited without returning a result."
        return (
            SimulationResult(
                matched_phase_candidates=list(diffraction.phase_candidates),
                warnings=[warning],
                provenance={
                    "method": "diffsims",
                    "ui_worker_empty": True,
                    "ui_worker_status": status,
                },
            ),
            diffraction,
            [warning],
        )

    if not isinstance(payload, dict) or not payload.get("ok"):
        error = (
            str(payload.get("error", "unknown diffsims subprocess error"))
            if isinstance(payload, dict)
            else f"unexpected worker payload status={status}"
        )
        warning = f"B5 diffsims subprocess failed: {error}"
        return (
            SimulationResult(
                matched_phase_candidates=list(diffraction.phase_candidates),
                warnings=[warning],
                provenance={
                    "method": "diffsims",
                    "ui_worker_failed": True,
                    "ui_worker_status": status,
                },
            ),
            diffraction,
            [warning],
        )

    simulation = SimulationResult.model_validate(payload["simulation"])
    updated_diffraction = DiffractionResult.model_validate(payload["diffraction"])
    return simulation, updated_diffraction, list(simulation.warnings)


def _build_simulation_phase_rows(simulation: SimulationResult | None) -> list[dict[str, Any]]:
    if simulation is None:
        return []
    rows: list[dict[str, Any]] = []
    phase_matching = _as_mapping(simulation.provenance.get("phase_matching", {}))
    per_phase = phase_matching.get("per_phase", [])
    per_phase_rows = {
        str(row.get("phase_name")): dict(row)
        for row in per_phase
        if isinstance(row, dict) and str(row.get("phase_name", "")).strip()
    } if isinstance(per_phase, list) else {}
    for candidate in simulation.matched_phase_candidates:
        row = per_phase_rows.get(candidate.name, {})
        rows.append(
            {
                "name": candidate.name,
                "formula": candidate.formula,
                "source": candidate.source,
                "identifier": candidate.identifier,
                "score": candidate.score,
                "matched_spot_count": candidate.matched_spot_count,
                "mean_d_error": (
                    None
                    if candidate.mean_d_error is None
                    else candidate.mean_d_error.value
                ),
                "mean_angle_error": (
                    None
                    if candidate.mean_angle_error is None
                    else candidate.mean_angle_error.value
                ),
                "zone_axis": candidate.zone_axis,
                "best_pattern_index": row.get("best_pattern_index"),
                "simulated_spot_count": row.get("simulated_spot_count"),
                "phase_match_rotation_deg": row.get("rotation_deg"),
            }
        )
    return rows


def _build_simulated_spot_rows(simulation: SimulationResult | None) -> list[dict[str, Any]]:
    if simulation is None:
        return []
    rows: list[dict[str, Any]] = []
    for pattern_idx, pattern in enumerate(simulation.patterns):
        original_pattern_index = pattern.simulation_parameters.get(
            "display_source_pattern_index",
            pattern_idx,
        )
        for spot_idx, spot in enumerate(pattern.spots):
            rows.append(
                {
                    "pattern_index": pattern_idx,
                    "source_pattern_index": original_pattern_index,
                    "structure_id": pattern.structure_id,
                    "zone_axis": pattern.zone_axis,
                    "rotation_angle_deg": pattern.simulation_parameters.get(
                        "rotation_angle_deg"
                    ),
                    "spot_index": spot_idx,
                    "gx": spot.gx.value,
                    "gy": spot.gy.value,
                    "magnitude": None if spot.magnitude is None else spot.magnitude.value,
                    "intensity": spot.intensity,
                    "h": spot.h,
                    "k": spot.k,
                    "l": spot.l,
                }
            )
    return rows


def _zone_axis_payload(axis: Any) -> list[int] | None:
    parsed = _zone_axis_tuple(axis)
    return list(parsed) if parsed is not None else None


def _zone_axis_equal(left: Any, right: Any) -> bool:
    parsed_left = _zone_axis_tuple(left)
    parsed_right = _zone_axis_tuple(right)
    return parsed_left is not None and parsed_left == parsed_right


def _pattern_matches_phase_key(pattern: SimulatedPattern, key: str) -> bool:
    if not key:
        return True
    values = [pattern.structure_id]
    for name in ("phase_name", "candidate_name", "candidate_identifier"):
        value = pattern.simulation_parameters.get(name)
        if value is not None:
            values.append(str(value))
    return any(str(value).strip().casefold() == key for value in values)


def _filter_b5_simulation_for_zone_axis(
    simulation: SimulationResult,
    *,
    zone_axis_mode: str,
) -> SimulationResult:
    mode = str(zone_axis_mode or "auto_matched").strip().lower()
    if mode == "scan_all":
        return simulation
    if not simulation.patterns:
        return simulation

    phase_matching = _as_mapping(simulation.provenance.get("phase_matching", {}))
    selected_axis = _zone_axis_payload(phase_matching.get("selected_zone_axis"))
    selected_phase = str(phase_matching.get("selected_phase") or "").strip().casefold()
    per_phase = phase_matching.get("per_phase", [])
    selected_indices: list[int] = []

    if isinstance(per_phase, list):
        for row in per_phase:
            if not isinstance(row, dict):
                continue
            row_phase = str(row.get("phase_name") or "").strip().casefold()
            if selected_phase and row_phase and row_phase != selected_phase:
                continue
            idx = row.get("best_pattern_index")
            if isinstance(idx, int) and 0 <= idx < len(simulation.patterns):
                selected_indices.append(idx)
                if selected_axis is None:
                    selected_axis = _zone_axis_payload(row.get("zone_axis"))

    if selected_axis is not None:
        for idx, pattern in enumerate(simulation.patterns):
            if not _zone_axis_equal(pattern.zone_axis, selected_axis):
                continue
            if not _pattern_matches_phase_key(pattern, selected_phase):
                continue
            selected_indices.append(idx)

    if not selected_indices:
        for idx, pattern in enumerate(simulation.patterns):
            if _as_mapping(pattern.simulation_parameters.get("phase_match", {})):
                selected_indices.append(idx)
                selected_axis = _zone_axis_payload(pattern.zone_axis)
                break

    if not selected_indices:
        return simulation

    unique_indices: list[int] = []
    seen: set[int] = set()
    for idx in selected_indices:
        if idx in seen:
            continue
        seen.add(idx)
        unique_indices.append(idx)

    filtered_patterns = []
    for idx in unique_indices:
        pattern = simulation.patterns[idx]
        filtered_patterns.append(
            pattern.model_copy(
                update={
                    "simulation_parameters": {
                        **pattern.simulation_parameters,
                        "display_source_pattern_index": idx,
                    }
                }
            )
        )

    return simulation.model_copy(
        update={
            "patterns": filtered_patterns,
            "provenance": {
                **simulation.provenance,
                "display_filter": {
                    "mode": mode,
                    "source_pattern_count": len(simulation.patterns),
                    "display_pattern_count": len(filtered_patterns),
                    "selected_zone_axis": selected_axis,
                    "selected_phase": (
                        phase_matching.get("selected_phase") if phase_matching else None
                    ),
                    "kept_source_pattern_indices": unique_indices,
                },
            },
        }
    )


def _best_simulation_pattern(simulation: SimulationResult | None) -> SimulatedPattern | None:
    if simulation is None or not simulation.patterns:
        return None
    phase_matching = _as_mapping(simulation.provenance.get("phase_matching", {}))
    per_phase = phase_matching.get("per_phase", [])
    selected_phase = str(phase_matching.get("selected_phase", ""))
    if isinstance(per_phase, list):
        for row in per_phase:
            if not isinstance(row, dict):
                continue
            if selected_phase and str(row.get("phase_name", "")) != selected_phase:
                continue
            idx = row.get("best_pattern_index")
            if isinstance(idx, int) and 0 <= idx < len(simulation.patterns):
                return simulation.patterns[idx]
    for pattern in simulation.patterns:
        if _as_mapping(pattern.simulation_parameters.get("phase_match", {})):
            return pattern
    return simulation.patterns[0]


def _b5_pattern_title(pattern: SimulatedPattern, phase_matching: dict[str, Any]) -> str:
    phase_name = str(phase_matching.get("selected_phase") or pattern.structure_id or "").strip()
    axis = _zone_axis_payload(pattern.zone_axis)
    axis_text = "" if axis is None else f" [{axis[0]} {axis[1]} {axis[2]}]"
    return f"Simulated Diffraction: {phase_name or 'B5 candidate'}{axis_text}"


def _b5_hkl_label(spot: Any) -> str:
    values = (getattr(spot, "h", None), getattr(spot, "k", None), getattr(spot, "l", None))
    if any(value is None for value in values):
        return ""
    return "(" + "".join(str(int(value)) for value in values) + ")"


def _b5_spot_d_spacing(spot: Any, gx: float, gy: float) -> float | None:
    d_value = _measurement_value(getattr(spot, "d_spacing", None))
    if d_value is not None and d_value > 0:
        return float(d_value)
    magnitude = _measurement_value(getattr(spot, "magnitude", None))
    if magnitude is None:
        magnitude = float(np.hypot(gx, gy))
    if magnitude <= 1e-12:
        return None
    return float(1.0 / magnitude)


def _b5_simulated_spots_xy(
    pattern: SimulatedPattern,
    *,
    rotation_deg: float = 0.0,
) -> list[dict[str, Any]]:
    theta = np.deg2rad(float(rotation_deg))
    cr = float(np.cos(theta))
    sr = float(np.sin(theta))
    rows: list[dict[str, Any]] = []
    for idx, spot in enumerate(pattern.spots):
        gx = _measurement_value(spot.gx)
        gy = _measurement_value(spot.gy)
        if gx is None or gy is None:
            continue
        gx_rot = float(gx) * cr - float(gy) * sr
        gy_rot = float(gx) * sr + float(gy) * cr
        d_spacing = _b5_spot_d_spacing(spot, gx_rot, gy_rot)
        rows.append(
            {
                "index": idx,
                "gx": gx_rot,
                "gy": gy_rot,
                "magnitude": float(np.hypot(gx_rot, gy_rot)),
                "intensity": (
                    None
                    if getattr(spot, "intensity", None) is None
                    else float(spot.intensity)
                ),
                "h": getattr(spot, "h", None),
                "k": getattr(spot, "k", None),
                "l": getattr(spot, "l", None),
                "label": _b5_hkl_label(spot),
                "d_spacing": d_spacing,
            }
        )
    return rows


def _b5_spot_sizes(rows: list[dict[str, Any]], *, base: float = 28.0) -> np.ndarray:
    intensities = np.asarray(
        [
            1.0 if row.get("intensity") is None else max(0.0, float(row["intensity"]))
            for row in rows
        ],
        dtype=float,
    )
    if intensities.size == 0:
        return np.asarray([], dtype=float)
    max_intensity = float(np.nanmax(intensities))
    if not np.isfinite(max_intensity) or max_intensity <= 0:
        return np.full(intensities.shape, base, dtype=float)
    scaled = np.sqrt(intensities / max_intensity)
    return base + 26.0 * scaled


def _b5_matched_simulated_indices(phase_match: dict[str, Any]) -> set[int]:
    pairs = phase_match.get("matched_pairs", [])
    matched: set[int] = set()
    if not isinstance(pairs, list):
        return matched
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        s_idx = pair.get("simulated_spot_index")
        if isinstance(s_idx, int):
            matched.add(s_idx)
    return matched


def _b5_row_intensity(row: dict[str, Any]) -> float:
    value = row.get("intensity")
    if value is None:
        return 1.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if np.isfinite(number) else 0.0


def _b5_row_hkl_order(row: dict[str, Any]) -> int | None:
    values = []
    for key in ("h", "k", "l"):
        value = row.get(key)
        if value is None:
            return None
        try:
            values.append(abs(int(value)))
        except (TypeError, ValueError):
            return None
    return max(values) if values else None


def _b5_display_rows(
    rows: list[dict[str, Any]],
    phase_match: dict[str, Any],
    *,
    max_points: int = 96,
    max_hkl_order: int = 12,
) -> list[dict[str, Any]]:
    if len(rows) <= max_points:
        return rows
    matched = _b5_matched_simulated_indices(phase_match)
    filtered = [
        row
        for row in rows
        if int(row.get("index", -1)) in matched
        or (_b5_row_hkl_order(row) is not None and _b5_row_hkl_order(row) <= max_hkl_order)
    ]
    if len(filtered) < min(max_points, max(24, len(matched))):
        filtered = rows
    ranked = sorted(
        filtered,
        key=lambda row: (
            int(row.get("index", -1)) not in matched,
            -_b5_row_intensity(row),
            float(row.get("magnitude") or np.hypot(row["gx"], row["gy"])),
        ),
    )
    selected = ranked[:max_points]
    return sorted(selected, key=lambda row: int(row.get("index", 0)))


def _b5_label_indices(
    rows: list[dict[str, Any]],
    phase_match: dict[str, Any],
    *,
    max_labels: int = 18,
) -> set[int]:
    matched = _b5_matched_simulated_indices(phase_match)
    ranked = sorted(
        rows,
        key=lambda row: (
            int(row.get("index", -1)) not in matched,
            -_b5_row_intensity(row),
            float(row.get("magnitude") or np.hypot(row["gx"], row["gy"])),
        ),
    )
    return {int(row["index"]) for row in ranked[:max(0, int(max_labels))]}


def _style_b5_reciprocal_axes(ax: Any, points: np.ndarray, title: str) -> None:
    limit = 1.0
    if points.size:
        max_abs = float(np.nanmax(np.abs(points)))
        if np.isfinite(max_abs):
            limit = max(1.0, max_abs * 1.12)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("gx (1/nm)")
    ax.set_ylabel("gy (1/nm)")
    ax.set_title(title, fontsize=13, pad=10)
    ax.grid(True, color="#d6dbe3", linewidth=0.7, alpha=0.55)
    ax.axhline(0.0, color="#9aa4b2", linewidth=0.75, alpha=0.55)
    ax.axvline(0.0, color="#9aa4b2", linewidth=0.75, alpha=0.55)
    ax.tick_params(colors="#1f2937")
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
        spine.set_linewidth(0.8)


def _plot_b5_simulated_pattern(simulation: SimulationResult | None) -> Any | None:
    pattern = _best_simulation_pattern(simulation)
    if pattern is None or not pattern.spots:
        return None
    phase_matching = (
        {}
        if simulation is None
        else _as_mapping(simulation.provenance.get("phase_matching", {}))
    )
    phase_match = _as_mapping(pattern.simulation_parameters.get("phase_match", {}))
    rotation_deg = _num(phase_match.get("rotation_deg"), 0.0)
    rows = _b5_simulated_spots_xy(pattern, rotation_deg=float(rotation_deg))
    if not rows:
        return None

    display_rows = _b5_display_rows(rows, phase_match)
    xy = np.asarray([(row["gx"], row["gy"]) for row in display_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.7, 6.7), dpi=150)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=_b5_spot_sizes(display_rows),
        c="#ff8a3d",
        edgecolors="white",
        linewidths=0.6,
        alpha=0.92,
    )

    label_indices = _b5_label_indices(display_rows, phase_match)
    for row in display_rows:
        if int(row["index"]) not in label_indices:
            continue
        label = str(row.get("label", ""))
        d_spacing = row.get("d_spacing")
        if not label and d_spacing is None:
            continue
        label_text = label
        if d_spacing is not None:
            label_text = f"{label}\n{float(d_spacing):.3f}" if label else f"{float(d_spacing):.3f}"
        ax.text(
            float(row["gx"]),
            float(row["gy"]),
            label_text,
            fontsize=5.5,
            ha="center",
            va="bottom",
            color="#111827",
        )

    _style_b5_reciprocal_axes(
        ax,
        xy,
        (
            _b5_pattern_title(pattern, phase_matching)
            if len(display_rows) == len(rows)
            else f"{_b5_pattern_title(pattern, phase_matching)} "
            f"(showing {len(display_rows)}/{len(rows)} spots)"
        ),
    )
    fig.tight_layout()
    return fig


def _plot_b5_simulation_overlay(
    diffraction: DiffractionResult,
    simulation: SimulationResult | None,
    *,
    show_unmatched_simulated_spots: bool = False,
    show_match_residual_lines: bool = False,
) -> Any | None:
    pattern = _best_simulation_pattern(simulation)
    if pattern is None or not pattern.spots or not diffraction.g_vectors:
        return None

    phase_match = _as_mapping(pattern.simulation_parameters.get("phase_match", {}))
    rotation_deg = _num(phase_match.get("rotation_deg"), 0.0)

    measured: dict[int, tuple[float, float]] = {}
    for idx, g_vector in enumerate(diffraction.g_vectors):
        gx = _measurement_value(g_vector.gx)
        gy = _measurement_value(g_vector.gy)
        if gx is not None and gy is not None:
            measured[idx] = (float(gx), float(gy))

    rows = _b5_simulated_spots_xy(pattern, rotation_deg=float(rotation_deg))
    simulated = {int(row["index"]): (float(row["gx"]), float(row["gy"])) for row in rows}

    pairs = phase_match.get("matched_pairs", [])
    matched_simulated_indices: set[int] = set()
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            s_idx = pair.get("simulated_spot_index")
            if isinstance(s_idx, int):
                matched_simulated_indices.add(s_idx)

    visible_simulated = simulated
    if matched_simulated_indices and not show_unmatched_simulated_spots:
        visible_simulated = {
            idx: xy for idx, xy in simulated.items() if idx in matched_simulated_indices
        }

    if not measured or not visible_simulated:
        return None

    fig, ax = plt.subplots(figsize=(6.7, 6.7), dpi=150)
    measured_xy = np.asarray(list(measured.values()), dtype=float)
    visible_simulated_xy = np.asarray(list(visible_simulated.values()), dtype=float)
    ax.scatter(
        measured_xy[:, 0],
        measured_xy[:, 1],
        s=36,
        c="#2f80c2",
        edgecolors="white",
        linewidths=0.5,
        alpha=0.82,
        label="Experimental",
    )
    if show_unmatched_simulated_spots and matched_simulated_indices:
        unmatched = {
            idx: xy for idx, xy in simulated.items() if idx not in matched_simulated_indices
        }
        if unmatched:
            unmatched_xy = np.asarray(list(unmatched.values()), dtype=float)
            ax.scatter(
                unmatched_xy[:, 0],
                unmatched_xy[:, 1],
                s=24,
                c="#ff8a3d",
                edgecolors="none",
                alpha=0.22,
                label="Simulated unmatched",
            )
    ax.scatter(
        visible_simulated_xy[:, 0],
        visible_simulated_xy[:, 1],
        s=30,
        c="#ff8a3d",
        edgecolors="white",
        linewidths=0.45,
        alpha=0.84,
        label="Simulated",
    )
    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            m_idx = pair.get("measured_g_index")
            s_idx = pair.get("simulated_spot_index")
            if not isinstance(m_idx, int) or not isinstance(s_idx, int):
                continue
            if m_idx in measured and s_idx in simulated:
                mx, my = measured[m_idx]
                sx, sy = simulated[s_idx]
                if show_match_residual_lines:
                    ax.plot(
                        [mx, sx],
                        [my, sy],
                        color="#64748b",
                        linewidth=0.75,
                        alpha=0.45,
                    )
                spot = pattern.spots[s_idx] if 0 <= s_idx < len(pattern.spots) else None
                label = _b5_hkl_label(spot) if spot is not None else ""
                if label:
                    ax.text(
                        sx,
                        sy,
                        label,
                        fontsize=5.3,
                        ha="center",
                        va="bottom",
                        color="#111827",
                    )
    stacked = np.vstack([measured_xy, visible_simulated_xy])
    _style_b5_reciprocal_axes(ax, stacked, "Experimental vs Simulated Overlay")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    fig.tight_layout()
    return fig


def _build_manual_diffraction(
    *,
    selected_points_xy: list[tuple[float, float]],
    fft_log_magnitude: np.ndarray,
    fft_center: tuple[float, float],
    calibration: Any,
    diff_params: dict[str, Any],
    candidate_phase_params: list[dict[str, Any]],
    candidate_phases: list[str],
) -> dict[str, Any]:
    warnings: list[str] = []
    if not selected_points_xy:
        return {
            "peaks": [],
            "g_vectors": [],
            "d_by_peak": [],
            "indexed_spots": [],
            "phase_candidates": [],
            "warnings": ["No manual points selected."],
            "provenance": {"manual_selection": True, "selected_point_count": 0},
        }

    h, w = fft_log_magnitude.shape
    peaks: list[DiffractionPeak] = []
    for x, y in selected_points_xy:
        xx = int(np.clip(round(x), 0, w - 1))
        yy = int(np.clip(round(y), 0, h - 1))
        intensity = float(fft_log_magnitude[yy, xx])
        peaks.append(
            DiffractionPeak(
                x=float(x),
                y=float(y),
                intensity=intensity,
                confidence=0.9,
            )
        )

    scale_value = _measurement_value(getattr(calibration, "diffraction_scale", None))
    scale_unc = _measurement_uncertainty(getattr(calibration, "diffraction_scale", None))
    scale_source = (
        "unresolved"
        if getattr(calibration, "diffraction_scale", None) is None
        else str(getattr(calibration.diffraction_scale, "method", "unknown"))
    )

    fft_center_mode = str(diff_params.get("fft_center_mode", "auto")).strip().lower()
    center_unc = diff_params.get("direct_beam_center_uncertainty_px")
    center_unc_val = _num(center_unc, 0.35 if fft_center_mode == "auto" else 0.0)

    reciprocal = compute_gvectors_and_dspacing(
        peaks,
        center=fft_center,
        diffraction_scale_value=scale_value,
        diffraction_scale_uncertainty=scale_unc,
        diffraction_scale_source=scale_source,
        center_uncertainty_px=center_unc_val,
    )
    warnings.extend(reciprocal.warnings)

    indexed_spots = _baseline_indexed_spots(reciprocal.d_by_peak)
    phase_candidates: list[PhaseCandidate] = []
    indexing_mode = str(diff_params.get("indexing_mode", "auto")).strip().lower()
    if indexing_mode in {"auto", "lattice_fit"} and peaks:
        b4_result = run_b4_indexing(
            peaks=peaks,
            g_vectors=reciprocal.g_vectors,
            d_by_peak=reciprocal.d_by_peak,
            initial_indexed_spots=indexed_spots,
            candidate_phase_params=candidate_phase_params,
            allowed_phase_names=set(candidate_phases) if candidate_phases else None,
            max_hkl=max(1, int(_num(diff_params.get("max_hkl"), 4))),
            d_tolerance_rel=max(1e-4, _num(diff_params.get("d_tolerance_rel"), 0.10)),
            angle_tolerance_deg=max(0.1, _num(diff_params.get("angle_tolerance_deg"), 12.0)),
            strict_mode=bool(diff_params.get("b4_strict_mode", False)),
            cif_candidates_dir=(
                str(diff_params.get("cif_candidates_dir"))
                if diff_params.get("cif_candidates_dir")
                else None
            ),
            top_n=max(1, int(_num(diff_params.get("phase_top_n"), 3))),
        )
        indexed_spots = b4_result.indexed_spots
        phase_candidates = b4_result.phase_candidates
        warnings.extend(b4_result.warnings)
        b4_provenance = b4_result.provenance
    else:
        b4_provenance = {"active": False, "reason": "indexing_mode_baseline_or_no_peaks"}

    if not phase_candidates and candidate_phases:
        phase_candidates = _phase_placeholders(
            list(candidate_phases),
            max(1, int(_num(diff_params.get("phase_top_n"), 3))),
        )

    return {
        "peaks": peaks,
        "g_vectors": reciprocal.g_vectors,
        "d_by_peak": reciprocal.d_by_peak,
        "indexed_spots": indexed_spots,
        "phase_candidates": phase_candidates,
        "warnings": list(dict.fromkeys(warnings)),
        "provenance": {
            "manual_selection": True,
            "selected_point_count": len(peaks),
            "fft_center": [float(fft_center[0]), float(fft_center[1])],
            "reciprocal_metrics": reciprocal.diagnostics,
            "b4_indexing": b4_provenance,
        },
    }


def _with_manual_context(
    manual_result: dict[str, Any],
    *,
    selection_mode: str,
    selected_points_xy: list[tuple[float, float]],
    manual_refine_warnings: list[str],
    vector_synth_warnings: list[str],
    manual_refine_diagnostics: dict[str, Any],
    vector_synth_diagnostics: dict[str, Any],
    auto_remove_peak_ids: list[int],
    manual_points_refined_xy: list[tuple[float, float]],
) -> dict[str, Any]:
    result = dict(manual_result)
    result["warnings"] = list(
        dict.fromkeys(
            list(result.get("warnings", []))
            + list(manual_refine_warnings)
            + list(vector_synth_warnings)
        )
    )

    provenance = dict(result.get("provenance", {}))
    provenance["manual_point_refine"] = dict(manual_refine_diagnostics)
    if selection_mode == MODE_TWO_VECTOR:
        provenance["vector_synthesis"] = dict(vector_synth_diagnostics)
    if selection_mode == MODE_AUTO_EDIT:
        provenance["auto_edit"] = {
            "removed_auto_peak_ids": [int(i) for i in auto_remove_peak_ids],
            "removed_count": int(len(auto_remove_peak_ids)),
            "manual_add_count": int(len(manual_points_refined_xy)),
            "selected_point_count": int(len(selected_points_xy)),
        }
    result["provenance"] = provenance
    return result


def _plot_peak_overlay(
    fft_image_norm: np.ndarray,
    *,
    center_xy: tuple[float, float],
    auto_points_xy: list[tuple[float, float]],
    selected_points_xy: list[tuple[float, float]],
    manual_raw_points_xy: list[tuple[float, float]] | None = None,
    manual_refined_points_xy: list[tuple[float, float]] | None = None,
    basis_points_xy: list[tuple[float, float]] | None = None,
    annotate_auto_ids: bool = True,
    annotate_selected_ids: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 7.0), dpi=150)
    ax.imshow(fft_image_norm, cmap="gray")
    if auto_points_xy:
        auto_x = [float(item[0]) for item in auto_points_xy]
        auto_y = [float(item[1]) for item in auto_points_xy]
        ax.scatter(auto_x, auto_y, s=16, c="cyan", marker="o", label="auto")
        if annotate_auto_ids:
            for idx, (xv, yv) in enumerate(auto_points_xy):
                ax.annotate(
                    str(idx),
                    (float(xv), float(yv)),
                    xytext=(4, 4),
                    textcoords="offset points",
                    color="cyan",
                    fontsize=7,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "fc": (0.0, 0.0, 0.0, 0.35),
                        "ec": (0.0, 0.0, 0.0, 0.0),
                    },
                )
    if selected_points_xy:
        sel_x = [float(item[0]) for item in selected_points_xy]
        sel_y = [float(item[1]) for item in selected_points_xy]
        ax.scatter(sel_x, sel_y, s=28, facecolors="none", edgecolors="red", label="selected")
        if annotate_selected_ids:
            for idx, (xv, yv) in enumerate(selected_points_xy):
                ax.annotate(
                    str(idx),
                    (float(xv), float(yv)),
                    xytext=(4, -10),
                    textcoords="offset points",
                    color="red",
                    fontsize=7,
                    fontweight="bold",
                    bbox={
                        "boxstyle": "round,pad=0.15",
                        "fc": (0.0, 0.0, 0.0, 0.35),
                        "ec": (0.0, 0.0, 0.0, 0.0),
                    },
                )
    if manual_raw_points_xy:
        raw_x = [float(item[0]) for item in manual_raw_points_xy]
        raw_y = [float(item[1]) for item in manual_raw_points_xy]
        ax.scatter(raw_x, raw_y, s=26, c="orange", marker="x", label="manual_raw")
    if manual_refined_points_xy:
        ref_x = [float(item[0]) for item in manual_refined_points_xy]
        ref_y = [float(item[1]) for item in manual_refined_points_xy]
        ax.scatter(ref_x, ref_y, s=22, c="lime", marker="+", label="manual_refined")
    if basis_points_xy:
        bx = [float(item[0]) for item in basis_points_xy]
        by = [float(item[1]) for item in basis_points_xy]
        ax.scatter(bx, by, s=44, c="magenta", marker="*", label="basis(g1/g2)")
    ax.scatter([center_xy[0]], [center_xy[1]], s=40, c="yellow", marker="+", label="center")
    ax.set_title("FFT Peaks Overlay")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _run_b1b3_pipeline(
    *,
    input_path: Path,
    quality_params: dict[str, Any],
    calibration_method: str,
    calibration_params: dict[str, Any],
    recipe: str,
    preprocessing_params: dict[str, Any],
    diffraction_params: dict[str, Any],
    fft_center_mode: str,
    direct_beam_search_radius: float,
    direct_beam_refine_window: int,
) -> dict[str, Any]:
    dataset = load_tem_dataset(input_path)
    quality = assess_quality(dataset, parameters=quality_params)
    calibration = calibrate_dataset(
        dataset,
        method=str(calibration_method),
        parameters=calibration_params,
    )
    preprocessing = preprocess_dataset(
        dataset,
        quality_after=quality,
        recipe=str(recipe),
        parameters=preprocessing_params,
    )
    diffraction_auto = run_diffraction_analysis(
        preprocessing,
        calibration,
        parameters=diffraction_params,
    )

    original_image = _array_from_reference(dataset.data)
    if original_image is None:
        read_result = read_image(input_path)
        original_image = np.asarray(read_result.array, dtype=np.float32)
    processed_image = _array_from_reference(preprocessing.image)
    if processed_image is None:
        processed_image = np.asarray(original_image, dtype=np.float32)

    fft_analysis, fft_warnings = compute_fft_analysis(
        np.asarray(processed_image, dtype=np.float64),
        center_mode=str(fft_center_mode),
        search_radius_px=(
            float(direct_beam_search_radius)
            if float(direct_beam_search_radius) > 0
            else None
        ),
        refine_window=max(1, int(direct_beam_refine_window)),
    )

    return {
        "dataset": dataset,
        "quality": quality,
        "calibration": calibration,
        "preprocessing": preprocessing,
        "diffraction_auto": diffraction_auto,
        "original_image": np.asarray(original_image, dtype=np.float32),
        "processed_image": np.asarray(processed_image, dtype=np.float32),
        "fft_analysis": fft_analysis,
        "fft_warnings": list(fft_warnings),
    }


def _render_mode_selector(mode_options: list[str]) -> str:
    return str(
        st.radio(
            "Spot source mode",
            options=mode_options,
            horizontal=True,
            key="b1b3_selection_mode_main",
        )
    )


def _render_preview_metrics(
    *,
    quality: Any,
    calibration: Any,
    diffraction_auto: Any,
) -> None:
    st.subheader("B1/B2/B3 Preview")
    metrics_cols = st.columns(6)
    metrics_cols[0].metric("Quality Score", f"{float(quality.overall_score or 0.0):.3f}")
    metrics_cols[1].metric("Auto Peaks", str(len(diffraction_auto.peaks)))
    metrics_cols[2].metric("Auto d-spacing", str(len(diffraction_auto.d_spacings)))
    metrics_cols[3].metric("Auto Phase Top-N", str(len(diffraction_auto.phase_candidates)))
    metrics_cols[4].metric(
        "Diffraction Confidence",
        f"{float(diffraction_auto.confidence or 0.0):.3f}",
    )
    metrics_cols[5].metric(
        "Calibration Confidence",
        f"{float(calibration.confidence or 0.0):.3f}",
    )


def _render_preview_images(
    *,
    original_image: np.ndarray,
    processed_image: np.ndarray,
    fft_image_norm: np.ndarray,
) -> None:
    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.image(
            _normalize_for_display(original_image),
            caption="Input image",
            use_container_width=True,
        )
    with img_col2:
        st.image(
            _normalize_for_display(processed_image),
            caption="Preprocessed",
            use_container_width=True,
        )
    with img_col3:
        st.image(fft_image_norm, caption="FFT log-magnitude", use_container_width=True)


def main() -> None:
    st.set_page_config(page_title=_APP_TITLE, layout="wide")
    st.title(_APP_TITLE)
    st.caption(
        "B1/B2/B3/B4/B5 interactive debugging for quality, calibration, "
        "preprocessing, FFT peaks, indexing, phase lookup, and simulation matching."
    )
    st.caption(f"UI build: `{_UI_BUILD_TAG}`")

    merged_defaults = _deep_merge(_FALLBACK_DEFAULTS, _read_template_defaults())
    quality_defaults = _as_mapping(_as_mapping(merged_defaults.get("quality")).get("thresholds"))
    cal_defaults = _as_mapping(merged_defaults.get("calibration"))
    prep_defaults = _as_mapping(merged_defaults.get("preprocessing"))
    diff_defaults = _as_mapping(merged_defaults.get("diffraction"))
    sim_defaults = _as_mapping(merged_defaults.get("simulation"))

    st.sidebar.header("Run Setup")
    input_mode = st.sidebar.radio(
        "Input source",
        ["Upload file", "Built-in demo", "Local file path"],
        index=0,
    )
    uploaded = None
    local_file_path = ""
    if input_mode == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Input TEM image/data",
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
    elif input_mode == "Local file path":
        local_file_path = st.sidebar.text_input(
            "Local data file path",
            value=str(st.session_state.get("b1b3_local_file_path", "")),
            help="Example: D:/Users/.../1.0017.dm4",
        ).strip()
        if local_file_path:
            st.session_state["b1b3_local_file_path"] = local_file_path

    if "b1b3_output_root_value" not in st.session_state:
        st.session_state["b1b3_output_root_value"] = str(_default_output_root())
    if "b1b3_run_name_value" not in st.session_state:
        st.session_state["b1b3_run_name_value"] = (
            f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    output_root_raw = st.sidebar.text_input(
        "Output root",
        key="b1b3_output_root_value",
    )
    run_name = st.sidebar.text_input(
        "Run folder name",
        key="b1b3_run_name_value",
    )
    auto_rerun = st.sidebar.checkbox("Auto rerun on parameter change", value=True)

    with st.sidebar.expander("B1 Quality", expanded=False):
        snr_min = st.number_input(
            "snr_min",
            min_value=0.0,
            value=_num(quality_defaults.get("snr_min"), 6.0),
            step=0.5,
        )
        drift_px_max = st.number_input(
            "drift_px_max",
            min_value=0.0,
            value=_num(quality_defaults.get("drift_px_max"), 2.0),
            step=0.2,
        )
        overall_score_min = st.number_input(
            "overall_score_min",
            min_value=0.0,
            max_value=1.0,
            value=_num(quality_defaults.get("overall_score_min"), 0.35),
            step=0.05,
        )

    with st.sidebar.expander("B1 Calibration", expanded=True):
        manual_calibration_override = st.checkbox(
            "manual_calibration_override",
            value=bool(cal_defaults.get("manual_calibration_override", False)),
            help=(
                "Off: DM uses file metadata auto calibration; non-DM prioritizes "
                "scale-bar auto sizing. "
                "On: apply manual values below."
            ),
        )
        cal_method_default = str(cal_defaults.get("method", "auto"))
        if cal_method_default not in {"auto", "reference", "manual", "metadata"}:
            cal_method_default = "auto"
        cal_method = st.selectbox(
            "method",
            options=["auto", "reference", "manual", "metadata"],
            index=["auto", "reference", "manual", "metadata"].index(cal_method_default),
            disabled=not manual_calibration_override,
        )
        diffraction_engine_default = str(
            cal_defaults.get("diffraction_calibration_engine", "none")
        )
        if diffraction_engine_default not in {"none", "pyxem", "py4dstem"}:
            diffraction_engine_default = "none"
        diffraction_engine = st.selectbox(
            "diffraction_calibration_engine",
            options=["none", "pyxem", "py4dstem"],
            index=["none", "pyxem", "py4dstem"].index(diffraction_engine_default),
        )
        voltage_kv = st.number_input(
            "voltage_kv",
            min_value=1.0,
            value=_num(cal_defaults.get("voltage_kv"), 200.0),
            step=10.0,
            disabled=not manual_calibration_override,
        )
        pixel_size_nm = st.number_input(
            "pixel_size_nm",
            min_value=1e-6,
            value=_num(cal_defaults.get("pixel_size_nm"), 0.02),
            step=0.001,
            format="%.6f",
            disabled=not manual_calibration_override,
        )
        pixel_size_unc_nm = st.number_input(
            "pixel_size_uncertainty_nm",
            min_value=0.0,
            value=_num(cal_defaults.get("pixel_size_uncertainty_nm"), 0.001),
            step=0.0005,
            format="%.6f",
            disabled=not manual_calibration_override,
        )
        known_d_spacing = st.number_input(
            "known_d_spacing",
            min_value=1e-6,
            value=_num(cal_defaults.get("known_d_spacing"), 0.235),
            step=0.001,
            format="%.6f",
            disabled=not manual_calibration_override,
        )
        known_d_unit = st.selectbox(
            "known_d_spacing_unit",
            options=["nm", "angstrom", "pm"],
            index=["nm", "angstrom", "pm"].index(
                str(cal_defaults.get("known_d_spacing_unit", "nm"))
                if str(cal_defaults.get("known_d_spacing_unit", "nm")) in {"nm", "angstrom", "pm"}
                else "nm"
            ),
            disabled=not manual_calibration_override,
        )
        measured_radius_px = st.number_input(
            "measured_radius_px",
            min_value=0.1,
            value=_num(cal_defaults.get("measured_radius_px"), 8.5),
            step=0.1,
            disabled=not manual_calibration_override,
        )
        known_d_unc_nm = st.number_input(
            "known_d_uncertainty_nm",
            min_value=0.0,
            value=_num(cal_defaults.get("known_d_uncertainty_nm"), 0.002),
            step=0.0005,
            format="%.6f",
            disabled=not manual_calibration_override,
        )
        radius_unc_px = st.number_input(
            "measured_radius_uncertainty_px",
            min_value=0.0,
            value=_num(cal_defaults.get("measured_radius_uncertainty_px"), 0.3),
            step=0.05,
            disabled=not manual_calibration_override,
        )
        detector_pixel_size_um = st.number_input(
            "detector_pixel_size_um",
            min_value=0.1,
            value=_num(cal_defaults.get("detector_pixel_size_um"), 14.0),
            step=0.5,
            disabled=not manual_calibration_override,
        )
        multi_ref_json = st.text_area(
            "known_d_spacing_refs (optional JSON list)",
            value="",
            height=96,
            disabled=not manual_calibration_override,
            help=(
                "Optional list rows like "
                '{"known_d_spacing":0.213,"known_d_spacing_unit":"nm","measured_radius_px":46.5}.'
            ),
        )

    with st.sidebar.expander("B2 Preprocessing", expanded=False):
        recipe = st.selectbox(
            "recipe",
            options=["auto", "min_max"],
            index=0 if str(prep_defaults.get("recipe", "auto")) == "auto" else 1,
        )
        clip_low = st.number_input(
            "clip_percentile_low",
            min_value=0.0,
            max_value=99.9,
            value=_num(prep_defaults.get("clip_percentile_low"), 1.0),
            step=0.5,
        )
        clip_high = st.number_input(
            "clip_percentile_high",
            min_value=0.1,
            max_value=100.0,
            value=_num(prep_defaults.get("clip_percentile_high"), 99.0),
            step=0.5,
        )
        denoise_method = st.selectbox(
            "denoise_method",
            options=sorted(SUPPORTED_DENOISE_METHODS),
            index=sorted(SUPPORTED_DENOISE_METHODS).index(
                str(prep_defaults.get("denoise_method", "wiener"))
                if str(prep_defaults.get("denoise_method", "wiener")) in SUPPORTED_DENOISE_METHODS
                else "auto"
            ),
        )
        denoise_sigma = st.number_input(
            "denoise_sigma",
            min_value=0.0,
            value=_num(prep_defaults.get("denoise_sigma"), 0.8),
            step=0.1,
        )
        median_size = st.number_input(
            "median_size",
            min_value=1,
            value=int(_num(prep_defaults.get("median_size"), 3)),
            step=1,
        )
        wiener_window = st.number_input(
            "wiener_window",
            min_value=3,
            value=int(_num(prep_defaults.get("wiener_window"), 5)),
            step=1,
        )
        wiener_noise = st.number_input(
            "wiener_noise",
            min_value=0.0,
            value=_num(prep_defaults.get("wiener_noise"), 0.0),
            step=0.01,
        )
        nlm_h = st.number_input(
            "nlm_h",
            min_value=0.0,
            value=_num(prep_defaults.get("nlm_h"), 0.05),
            step=0.01,
        )
        nlm_patch_size = st.number_input(
            "nlm_patch_size",
            min_value=3,
            value=int(_num(prep_defaults.get("nlm_patch_size"), 5)),
            step=1,
        )
        nlm_patch_distance = st.number_input(
            "nlm_patch_distance",
            min_value=1,
            value=int(_num(prep_defaults.get("nlm_patch_distance"), 6)),
            step=1,
        )
        background_method_default = _choice(
            SUPPORTED_BACKGROUND_METHODS,
            prep_defaults.get("background_method", "gaussian"),
            "none",
        )
        background_method = st.selectbox(
            "background_method",
            options=SUPPORTED_BACKGROUND_METHODS,
            index=SUPPORTED_BACKGROUND_METHODS.index(background_method_default),
        )
        background_sigma = st.number_input(
            "background_sigma",
            min_value=0.0,
            value=_num(prep_defaults.get("background_sigma"), 12.0),
            step=0.5,
        )
        background_opening_size = st.number_input(
            "background_opening_size",
            min_value=3,
            value=int(_num(prep_defaults.get("background_opening_size"), 31)),
            step=2,
        )
        background_clip_zero = st.checkbox(
            "background_clip_zero",
            value=bool(prep_defaults.get("background_clip_zero", False)),
        )

    with st.sidebar.expander("B3/B4 Diffraction + Indexing", expanded=True):
        auto_peak_tuning = st.checkbox(
            "auto_peak_tuning",
            value=bool(diff_defaults.get("auto_peak_tuning", True)),
        )
        peak_detector = st.selectbox(
            "peak_detector",
            options=sorted(SUPPORTED_PEAK_DETECTORS),
            index=sorted(SUPPORTED_PEAK_DETECTORS).index(
                str(diff_defaults.get("peak_detector", "log"))
                if str(diff_defaults.get("peak_detector", "log")) in SUPPORTED_PEAK_DETECTORS
                else "local_max"
            ),
        )
        peak_adapter_options = sorted(SUPPORTED_PEAK_ADAPTER_ENGINES)
        peak_adapter_default = _choice(
            peak_adapter_options,
            diff_defaults.get("peak_adapter_engine", "none"),
            "none",
        )
        peak_adapter_engine = st.selectbox(
            "peak_adapter_engine",
            options=peak_adapter_options,
            index=peak_adapter_options.index(peak_adapter_default),
        )
        threshold_strategy_options = sorted(SUPPORTED_THRESHOLD_STRATEGIES)
        threshold_strategy_default = _choice(
            threshold_strategy_options,
            diff_defaults.get("peak_threshold_strategy", "auto"),
            "auto",
        )
        peak_threshold_strategy = st.selectbox(
            "peak_threshold_strategy",
            options=threshold_strategy_options,
            index=threshold_strategy_options.index(threshold_strategy_default),
        )
        peak_threshold_value = st.number_input(
            "peak_threshold_value",
            value=_num(diff_defaults.get("peak_threshold_value"), 0.15),
            step=0.01,
            disabled=peak_threshold_strategy == "auto",
        )
        top_k_peaks = st.number_input(
            "top_k_peaks",
            min_value=1,
            value=int(_num(diff_defaults.get("top_k_peaks"), 24)),
            step=1,
        )
        min_peak_distance = st.number_input(
            "min_peak_distance",
            min_value=1,
            value=int(_num(diff_defaults.get("min_peak_distance"), 3)),
            step=1,
        )
        max_peak_radius_px = st.number_input(
            "max_peak_radius_px (0=auto)",
            min_value=0.0,
            value=_num(diff_defaults.get("max_peak_radius_px"), 0.0),
            step=10.0,
        )
        radius_outlier_filter = st.checkbox(
            "radius_outlier_filter",
            value=bool(diff_defaults.get("radius_outlier_filter", True)),
            help="Automatically remove far-radius outlier peaks and refill from inlier candidates.",
        )
        radius_outlier_mad_factor = st.number_input(
            "radius_outlier_mad_factor",
            min_value=2.0,
            value=_num(diff_defaults.get("radius_outlier_mad_factor"), 4.0),
            step=0.2,
            format="%.2f",
            disabled=not radius_outlier_filter,
        )
        center_exclusion_radius = st.number_input(
            "center_exclusion_radius",
            min_value=0.0,
            value=_num(diff_defaults.get("center_exclusion_radius"), 4.0),
            step=0.5,
        )
        detector_sigma = st.number_input(
            "detector_sigma",
            min_value=0.2,
            value=_num(diff_defaults.get("detector_sigma"), 1.2),
            step=0.1,
        )
        dog_sigma_ratio = st.number_input(
            "dog_sigma_ratio",
            min_value=1.2,
            value=_num(diff_defaults.get("dog_sigma_ratio"), 1.8),
            step=0.1,
        )
        subpixel_window_radius = st.number_input(
            "subpixel_window_radius",
            min_value=1,
            value=int(_num(diff_defaults.get("subpixel_window_radius"), 2)),
            step=1,
        )
        subpixel_max_shift = st.number_input(
            "subpixel_max_shift",
            min_value=0.0,
            value=_num(diff_defaults.get("subpixel_max_shift"), 1.5),
            step=0.1,
        )
        fft_center_mode = st.selectbox(
            "fft_center_mode",
            options=SUPPORTED_FFT_CENTER_MODES,
            index=SUPPORTED_FFT_CENTER_MODES.index(
                str(diff_defaults.get("fft_center_mode", "auto"))
                if str(diff_defaults.get("fft_center_mode", "auto")) in SUPPORTED_FFT_CENTER_MODES
                else "auto"
            ),
        )
        direct_beam_search_radius = st.number_input(
            "direct_beam_search_radius",
            min_value=0.0,
            value=_num(diff_defaults.get("direct_beam_search_radius"), 0.0),
            step=0.5,
        )
        direct_beam_refine_window = st.number_input(
            "direct_beam_refine_window",
            min_value=1,
            value=int(_num(diff_defaults.get("direct_beam_refine_window"), 3)),
            step=1,
        )
        direct_beam_center_uncertainty_px = st.number_input(
            "direct_beam_center_uncertainty_px",
            min_value=0.0,
            value=_num(diff_defaults.get("direct_beam_center_uncertainty_px"), 0.35),
            step=0.05,
        )
        indexing_mode = st.selectbox(
            "indexing_mode",
            options=SUPPORTED_INDEXING_MODES,
            index=SUPPORTED_INDEXING_MODES.index(
                str(diff_defaults.get("indexing_mode", "auto"))
                if str(diff_defaults.get("indexing_mode", "auto")) in SUPPORTED_INDEXING_MODES
                else "auto"
            ),
            format_func=lambda key: INDEXING_MODE_LABELS.get(str(key), str(key)),
        )
        phase_top_n = st.number_input(
            "phase_top_n",
            min_value=1,
            value=int(_num(diff_defaults.get("phase_top_n"), 3)),
            step=1,
        )
        phase_materials_default = _phase_material_defaults(diff_defaults)
        phase_materials_text = st.text_area(
            "materials / phases",
            value=", ".join(phase_materials_default),
            height=80,
            placeholder="PbS, TiO2, quartz, mp-149",
            help=(
                "Comma-, semicolon-, or newline-separated names. Letter case is preserved "
                "for chemical formulas, so PbS stays PbS."
            ),
        )
        candidate_phase_names = _dedupe_strings(_comma_list(phase_materials_text))
        b4_strict_mode = st.checkbox(
            "b4_strict_mode",
            value=bool(diff_defaults.get("b4_strict_mode", False)),
            disabled=indexing_mode == "baseline",
        )
        max_hkl = st.number_input(
            "max_hkl",
            min_value=1,
            value=int(_num(diff_defaults.get("max_hkl"), 4)),
            step=1,
        )
        d_tolerance_rel = st.number_input(
            "d_tolerance_rel",
            min_value=0.001,
            value=_num(diff_defaults.get("d_tolerance_rel"), 0.10),
            step=0.01,
            format="%.3f",
        )
        angle_tolerance_deg = st.number_input(
            "angle_tolerance_deg",
            min_value=0.1,
            value=_num(diff_defaults.get("angle_tolerance_deg"), 12.0),
            step=0.5,
        )
        cif_candidates_dir = st.text_input("cif_candidates_dir (optional)", value="")
        candidate_phase_params_text = st.text_area(
            "candidate_phase_params (JSON list)",
            value=DEFAULT_B4_CANDIDATES_JSON,
            height=190,
        )

        st.markdown("**Crystal sources**")
        st.caption(
            "B4 lookup runs automatically when B4 indexing is active and at least "
            "one crystal source is selected."
        )
        lookup_sources_all = sorted(supported_phase_lookup_sources())
        lookup_sources_config = [
            item
            for item in diff_defaults.get("phase_lookup_sources", DEFAULT_PHASE_LOOKUP_SOURCES)
            if item in lookup_sources_all
        ]
        if diff_defaults.get("phase_lookup_enabled") is False:
            lookup_sources_default = ["local_builtin"]
        else:
            lookup_sources_default = lookup_sources_config or [
                item for item in DEFAULT_PHASE_LOOKUP_SOURCES if item in lookup_sources_all
            ]
        phase_lookup_sources = st.multiselect(
            "phase_lookup_sources",
            options=lookup_sources_all,
            default=lookup_sources_default or ["local_builtin"],
            disabled=indexing_mode == "baseline",
        )
        phase_lookup_active = indexing_mode != "baseline" and bool(phase_lookup_sources)
        phase_lookup_elements_text = st.text_input(
            "phase_lookup_elements",
            value=", ".join(_comma_list(diff_defaults.get("phase_lookup_elements", []))),
            disabled=not phase_lookup_active,
        )
        materials_project_config_path = st.text_input(
            "materials_project_config_path",
            value=str(diff_defaults.get("materials_project_config_path", "")),
            disabled=not phase_lookup_active,
        )
        phase_lookup_limit_per_source = st.number_input(
            "phase_lookup_limit_per_source",
            min_value=1,
            value=int(_num(diff_defaults.get("phase_lookup_limit_per_source"), 3)),
            step=1,
            disabled=not phase_lookup_active,
        )
        phase_lookup_timeout_s = st.number_input(
            "phase_lookup_timeout_s",
            min_value=0.1,
            value=_num(diff_defaults.get("phase_lookup_timeout_s"), 10.0),
            step=0.5,
            disabled=not phase_lookup_active,
        )
        mp_sort_by = st.selectbox(
            "materials_project_sort_by",
            options=SUPPORTED_MP_SORT_MODES,
            index=SUPPORTED_MP_SORT_MODES.index(
                _choice(
                    SUPPORTED_MP_SORT_MODES,
                    diff_defaults.get("materials_project_sort_by", "stability"),
                    "stability",
                )
            ),
            disabled=not phase_lookup_active,
        )
        mp_include_deprecated = st.checkbox(
            "materials_project_include_deprecated",
            value=bool(diff_defaults.get("materials_project_include_deprecated", False)),
            disabled=not phase_lookup_active,
        )
        mp_filter_stable = st.checkbox(
            "materials_project_filter_stable",
            value=bool(diff_defaults.get("materials_project_filter_stable", False)),
            disabled=not phase_lookup_active,
        )
        mp_max_energy = st.number_input(
            "materials_project_max_energy_above_hull (0 disables)",
            min_value=0.0,
            value=_num(diff_defaults.get("materials_project_max_energy_above_hull"), 0.0),
            step=0.01,
            format="%.4f",
            disabled=not phase_lookup_active,
        )
        mp_prefer_experimental = st.checkbox(
            "materials_project_prefer_experimental",
            value=bool(diff_defaults.get("materials_project_prefer_experimental", True)),
            disabled=not phase_lookup_active,
        )
        mp_candidate_pool_size = st.number_input(
            "materials_project_candidate_pool_size",
            min_value=1,
            value=int(_num(diff_defaults.get("materials_project_candidate_pool_size"), 20)),
            step=1,
            disabled=not phase_lookup_active,
        )
        if phase_lookup_active and any(
            source == "materials_project_api" for source in phase_lookup_sources
        ):
            st.info(materials_project_api_setup_guidance(materials_project_config_path or None))
            st.caption(
                "UI runs Materials Project API lookup in an isolated child process so "
                "native crashes or timeouts return B4 warnings instead of closing Streamlit."
            )

    with st.sidebar.expander("B5 Simulation / Phase Matching", expanded=False):
        b5_enabled = st.checkbox(
            "Run B5 simulation step",
            value=bool(sim_defaults.get("enabled", True)),
        )
        simulation_engine = st.selectbox(
            "simulation_engine",
            options=SUPPORTED_SIMULATION_ENGINES,
            index=SUPPORTED_SIMULATION_ENGINES.index(
                _choice(
                    SUPPORTED_SIMULATION_ENGINES,
                    sim_defaults.get("simulation_engine", "diffsims"),
                    "diffsims",
                )
            ),
            disabled=not b5_enabled,
            help=(
                "B5 simulated diffraction uses diffsims in a timeout-limited subprocess."
            ),
        )
        st.caption(
            "B5 uses the current active peak set, B1 voltage, B3/B4 phase candidates, "
            "B4 d/angle tolerances, and by default locks to each candidate's B4 zone axis."
        )
        b5_auto_rerun = st.checkbox(
            "Auto rerun B5 on parameter change",
            value=bool(sim_defaults.get("auto_rerun", False)),
            disabled=not b5_enabled,
        )
        b5_timeout_s = st.number_input(
            "diffsims timeout_s",
            min_value=1.0,
            value=_num(sim_defaults.get("timeout_s"), 20.0),
            step=1.0,
            disabled=not b5_enabled or simulation_engine != "diffsims",
        )
        reciprocal_radius = st.number_input(
            "reciprocal_radius_inv_angstrom",
            min_value=0.01,
            value=_num(sim_defaults.get("reciprocal_radius_inv_angstrom"), 1.0),
            step=0.1,
            disabled=not b5_enabled,
        )
        max_simulated_spots = st.number_input(
            "max_simulated_spots",
            min_value=1,
            value=int(_num(sim_defaults.get("max_simulated_spots"), 128)),
            step=16,
            disabled=not b5_enabled,
        )
        min_simulated_g = st.number_input(
            "minimum_simulated_g_inv_nm",
            min_value=0.0,
            value=_num(sim_defaults.get("minimum_simulated_g_inv_nm"), 0.001),
            step=0.001,
            format="%.6f",
            disabled=not b5_enabled,
        )
        sim_with_direct_beam = st.checkbox(
            "with_direct_beam",
            value=bool(sim_defaults.get("with_direct_beam", True)),
            disabled=not b5_enabled,
        )
        sim_max_excitation_error = st.number_input(
            "max_excitation_error",
            min_value=1e-6,
            value=_num(sim_defaults.get("max_excitation_error"), 0.01),
            step=0.005,
            format="%.6f",
            disabled=not b5_enabled,
        )
        sim_fallback_single_atom = st.checkbox(
            "fallback_single_atom",
            value=bool(sim_defaults.get("fallback_single_atom", True)),
            disabled=not b5_enabled,
        )
        zone_axis_mode_options = [
            "inherit_b4_locked",
            "manual_fixed",
            "auto_matched",
            "scan_all",
        ]
        zone_axis_mode_labels = {
            "inherit_b4_locked": "Inherit B4 zone axis (locked)",
            "manual_fixed": "Manual fixed zone axis",
            "auto_matched": "Debug: rescan and show best axis",
            "scan_all": "Debug: scan/show all axes",
        }
        zone_axis_mode = st.selectbox(
            "zone_axis_mode",
            options=zone_axis_mode_options,
            index=zone_axis_mode_options.index(
                _choice(
                    zone_axis_mode_options,
                    sim_defaults.get("zone_axis_mode", "inherit_b4_locked"),
                    "inherit_b4_locked",
                )
            ),
            format_func=lambda value: zone_axis_mode_labels.get(str(value), str(value)),
            disabled=not b5_enabled,
            help=(
                "Locked mode simulates each phase candidate using the B4 zone axis. "
                "Debug modes can rescan candidate axes and should not be used as the "
                "default production interpretation."
            ),
        )
        manual_zone_axis_text = st.text_input(
            "manual_zone_axis JSON",
            value=json.dumps(sim_defaults.get("manual_zone_axis", [0, 0, 1])),
            disabled=not b5_enabled or zone_axis_mode != "manual_fixed",
            help="Single [h,k,l] zone axis used when zone_axis_mode is manual_fixed.",
        )
        zone_axis_text = st.text_area(
            "zone_axis_candidates JSON",
            value=json.dumps(
                sim_defaults.get("zone_axis_candidates", [[0, 0, 1]]),
                ensure_ascii=False,
            ),
            height=80,
            disabled=not b5_enabled
            or zone_axis_mode in {"inherit_b4_locked", "manual_fixed"},
            help="Candidate axes scanned only in debug rescan modes.",
        )
        show_unmatched_simulated_spots = st.checkbox(
            "show_unmatched_simulated_spots",
            value=bool(sim_defaults.get("show_unmatched_simulated_spots", False)),
            disabled=not b5_enabled,
            help="Show all simulated spots in the overlay; matched spots are always shown.",
        )
        show_match_residual_lines = st.checkbox(
            "show_match_residual_lines",
            value=bool(sim_defaults.get("show_match_residual_lines", False)),
            disabled=not b5_enabled,
            help=(
                "Draw experimental-to-simulated match residual lines in the overlay. "
                "Off by default because Friedel-pair matches can cross the center."
            ),
        )
        rotation_axis_text = st.text_input(
            "rotation_axis JSON",
            value=json.dumps(sim_defaults.get("rotation_axis", [0, 0, 1])),
            disabled=not b5_enabled,
        )
        rotation_angles_text = st.text_input(
            "rotation_angles_deg JSON",
            value=json.dumps(sim_defaults.get("rotation_angles_deg", [0.0])),
            disabled=not b5_enabled,
        )
        phase_match_enabled = st.checkbox(
            "phase_match_enabled",
            value=bool(sim_defaults.get("phase_match_enabled", True)),
            disabled=not b5_enabled,
        )
        phase_match_min_spots = st.number_input(
            "phase_match_min_spots",
            min_value=1,
            value=int(_num(sim_defaults.get("phase_match_min_spots"), 2)),
            step=1,
            disabled=not b5_enabled or not phase_match_enabled,
        )
        phase_match_previous_score_weight = st.number_input(
            "phase_match_previous_score_weight",
            min_value=0.0,
            max_value=1.0,
            value=_num(sim_defaults.get("phase_match_previous_score_weight"), 0.35),
            step=0.05,
            disabled=not b5_enabled or not phase_match_enabled,
        )
        phase_match_max_rotation_candidates = st.number_input(
            "phase_match_max_rotation_candidates",
            min_value=1,
            value=int(_num(sim_defaults.get("phase_match_max_rotation_candidates"), 64)),
            step=1,
            disabled=not b5_enabled or not phase_match_enabled,
        )
        if b5_enabled and simulation_engine == "diffsims":
            st.warning(
                "Real diffsims can hang in this environment; the UI uses a subprocess timeout."
            )

    st.sidebar.subheader("Manual Point Selection Params")
    overlay_label_mode = st.sidebar.selectbox(
        "Overlay numbering",
        options=["Auto peaks", "Selected points", "Both", "Off"],
        index=1,
    )
    manual_min_distance = st.sidebar.slider("Manual dedup distance (px)", 0.5, 12.0, 2.0, 0.1)
    manual_max_points = st.sidebar.slider("Manual max points", 2, 256, 64, 1)
    with st.sidebar.expander("Manual mode: two-point vector synthesis", expanded=False):
        vector_expand_order = st.slider(
            "Vector expansion order",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
        )
        vector_snap_mode = st.selectbox(
            "Vector snap method",
            options=["none", "local_max"],
            index=1,
            help=(
                "`none` keeps strict lattice coordinates from the two basis vectors. "
                "`local_max` optionally shifts synthesized points to nearby FFT maxima. "
                "Pipeline auto peaks are never used in vector mode."
            ),
        )
        vector_snap_radius_px = st.slider(
            "Vector snap radius (px)",
            min_value=0.5,
            max_value=20.0,
            value=7.5,
            step=0.5,
        )
        vector_local_intensity_percentile = st.slider(
            "Local max intensity percentile",
            min_value=90.0,
            max_value=99.99,
            value=97.5,
            step=0.1,
        )
        vector_max_points = st.slider(
            "Vector synthesized max points",
            min_value=8,
            max_value=256,
            value=96,
            step=1,
        )
    with st.sidebar.expander("Manual point refine", expanded=False):
        manual_refine_enabled = st.checkbox(
            "Enable refine",
            value=True,
            help=(
                "On refines clicked basis points with Gaussian/centroid fitting before "
                "vector synthesis. With Vector snap method = none, generated lattice "
                "spots remain strict vector combinations after basis refinement."
            ),
        )
        manual_refine_method = st.selectbox(
            "Refine method",
            options=["gaussian", "centroid", "none"],
            index=0,
            disabled=not manual_refine_enabled,
        )
        manual_refine_window_radius = st.slider(
            "Refine window radius (px)",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            disabled=not manual_refine_enabled or manual_refine_method == "none",
        )
        manual_refine_max_shift_px = st.slider(
            "Refine max shift (px)",
            min_value=0.2,
            max_value=6.0,
            value=3.0,
            step=0.1,
            disabled=not manual_refine_enabled or manual_refine_method == "none",
        )
    with st.sidebar.expander("Manual Spot Table Import + Align", expanded=False):
        manual_table_upload = st.file_uploader(
            "Upload manual spot table (.txt/.csv)",
            type=["txt", "csv", "tsv"],
            key="b1b3_manual_table_upload",
        )
        manual_table_text = st.text_area(
            "Or paste manual spot table text",
            value="",
            height=150,
            key="b1b3_manual_table_text",
            help=(
                "Supported: DM-like rows such as "
                "'Spot# d-Spacing Rec. Pos.(1/nm) ... Degrees to x-axis Amplitude'."
            ),
        )
        align_target_mode = st.selectbox(
            "Align target",
            options=["Auto (pipeline)", "Selected points"],
            index=0,
            key="b1b3_align_target_mode",
        )
        align_max_d_rel = st.slider(
            "Max d relative error",
            min_value=0.01,
            max_value=0.80,
            value=0.25,
            step=0.01,
            key="b1b3_align_max_d_rel",
        )
        align_max_g_rel = st.slider(
            "Max g relative error",
            min_value=0.01,
            max_value=0.80,
            value=0.25,
            step=0.01,
            key="b1b3_align_max_g_rel",
        )
        align_max_angle_deg = st.slider(
            "Max angle error (deg)",
            min_value=2.0,
            max_value=90.0,
            value=35.0,
            step=1.0,
            key="b1b3_align_max_angle_deg",
        )
        align_angle_step_deg = st.slider(
            "Angle search step (deg)",
            min_value=0.25,
            max_value=5.0,
            value=1.0,
            step=0.25,
            key="b1b3_align_angle_step_deg",
        )
        align_weight_d = st.slider(
            "Weight: d",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            key="b1b3_align_weight_d",
        )
        align_weight_g = st.slider(
            "Weight: g",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.1,
            key="b1b3_align_weight_g",
        )
        align_weight_angle = st.slider(
            "Weight: angle",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key="b1b3_align_weight_angle",
        )

    quality_params = {
        "thresholds": {
            "snr_min": float(snr_min),
            "drift_px_max": float(drift_px_max),
            "overall_score_min": float(overall_score_min),
        }
    }
    manual_calibration_params: dict[str, Any] = {
        "diffraction_calibration_engine": str(diffraction_engine),
        "voltage_kv": float(voltage_kv),
        "pixel_size_nm": float(pixel_size_nm),
        "pixel_size_uncertainty_nm": float(pixel_size_unc_nm),
        "known_d_spacing": float(known_d_spacing),
        "known_d_spacing_unit": str(known_d_unit),
        "known_d_uncertainty_nm": float(known_d_unc_nm),
        "measured_radius_px": float(measured_radius_px),
        "measured_radius_uncertainty_px": float(radius_unc_px),
        "detector_pixel_size_um": float(detector_pixel_size_um),
    }
    if multi_ref_json.strip():
        try:
            refs_payload = json.loads(multi_ref_json)
            if isinstance(refs_payload, list):
                manual_calibration_params["known_d_spacing_refs"] = refs_payload
        except Exception:  # noqa: BLE001
            pass

    preprocessing_params: dict[str, Any] = {
        "clip_percentile_low": float(clip_low),
        "clip_percentile_high": float(clip_high),
        "denoise_method": str(denoise_method),
        "denoise_sigma": float(denoise_sigma),
        "median_size": int(median_size),
        "wiener_window": int(wiener_window),
        "wiener_noise": float(wiener_noise),
        "nlm_h": float(nlm_h),
        "nlm_patch_size": int(nlm_patch_size),
        "nlm_patch_distance": int(nlm_patch_distance),
        "background_method": str(background_method),
        "background_sigma": float(background_sigma),
        "background_opening_size": int(background_opening_size),
        "background_clip_zero": bool(background_clip_zero),
    }

    candidate_phase_params, candidate_phase_warnings = _parse_candidate_phase_params(
        candidate_phase_params_text
    )
    phase_lookup_elements = _comma_list(phase_lookup_elements_text)
    zone_axis_candidates, zone_axis_warnings = _parse_zone_axis_candidates(zone_axis_text)
    manual_zone_axis_candidates, manual_zone_axis_warnings = _parse_zone_axis_candidates(
        manual_zone_axis_text
    )
    manual_zone_axis = manual_zone_axis_candidates[0]
    zone_axis_mode_value = str(zone_axis_mode)
    if zone_axis_mode_value == "inherit_b4_locked":
        effective_zone_axis_candidates = None
    elif zone_axis_mode_value == "manual_fixed":
        effective_zone_axis_candidates = [manual_zone_axis]
    else:
        effective_zone_axis_candidates = list(zone_axis_candidates)
    rotation_axis_values, rotation_axis_warnings = _parse_float_vector(
        rotation_axis_text,
        [0.0, 0.0, 1.0],
    )
    if len(rotation_axis_values) < 3:
        rotation_axis_warnings.append("rotation_axis must contain at least three numbers.")
        rotation_axis_values = [0.0, 0.0, 1.0]
    rotation_axis_values = rotation_axis_values[:3]
    rotation_angles, rotation_angle_warnings = _parse_float_vector(
        rotation_angles_text,
        [0.0],
    )
    diffraction_params: dict[str, Any] = {
        "auto_peak_tuning": bool(auto_peak_tuning),
        "peak_detector": str(peak_detector),
        "peak_adapter_engine": str(peak_adapter_engine),
        "peak_threshold_strategy": str(peak_threshold_strategy),
        "top_k_peaks": int(top_k_peaks),
        "min_peak_distance": int(min_peak_distance),
        "center_exclusion_radius": float(center_exclusion_radius),
        "max_peak_radius_px": (
            float(max_peak_radius_px) if float(max_peak_radius_px) > 0 else None
        ),
        "radius_outlier_filter": bool(radius_outlier_filter),
        "radius_outlier_mad_factor": float(radius_outlier_mad_factor),
        "detector_sigma": float(detector_sigma),
        "dog_sigma_ratio": float(dog_sigma_ratio),
        "subpixel_window_radius": int(subpixel_window_radius),
        "subpixel_max_shift": float(subpixel_max_shift),
        "fft_center_mode": str(fft_center_mode),
        "direct_beam_refine_window": int(direct_beam_refine_window),
        "direct_beam_center_uncertainty_px": float(direct_beam_center_uncertainty_px),
        "indexing_mode": str(indexing_mode),
        "phase_top_n": int(phase_top_n),
        "phase_candidate_materials": list(candidate_phase_names),
        "candidate_phases": list(candidate_phase_names),
        "candidate_phase_params": list(candidate_phase_params),
        "max_hkl": int(max_hkl),
        "d_tolerance_rel": float(d_tolerance_rel),
        "angle_tolerance_deg": float(angle_tolerance_deg),
        "b4_strict_mode": bool(b4_strict_mode),
    }
    if peak_threshold_strategy != "auto":
        diffraction_params["peak_threshold_value"] = float(peak_threshold_value)
    if float(direct_beam_search_radius) > 0:
        diffraction_params["direct_beam_search_radius"] = float(direct_beam_search_radius)
    if cif_candidates_dir.strip():
        diffraction_params["cif_candidates_dir"] = str(cif_candidates_dir.strip())
    if phase_lookup_active:
        diffraction_params.update(
            {
                "phase_lookup_sources": list(phase_lookup_sources),
                "phase_lookup_limit_per_source": int(phase_lookup_limit_per_source),
                "phase_lookup_timeout_s": float(phase_lookup_timeout_s),
                "materials_project_sort_by": str(mp_sort_by),
                "materials_project_include_deprecated": bool(mp_include_deprecated),
                "materials_project_filter_stable": bool(mp_filter_stable),
                "materials_project_prefer_experimental": bool(mp_prefer_experimental),
                "materials_project_candidate_pool_size": int(mp_candidate_pool_size),
            }
        )
        if "materials_project_api" in phase_lookup_sources:
            diffraction_params["materials_project_api_isolate"] = True
        if phase_lookup_elements:
            diffraction_params["phase_lookup_elements"] = phase_lookup_elements
        if materials_project_config_path.strip():
            diffraction_params["materials_project_config_path"] = (
                materials_project_config_path.strip()
            )
        if float(mp_max_energy) > 0:
            diffraction_params["materials_project_max_energy_above_hull"] = float(mp_max_energy)

    b5_params: dict[str, Any] = {
        "simulation_engine": str(simulation_engine),
        "voltage_kv": float(voltage_kv),
        "reciprocal_radius_inv_angstrom": float(reciprocal_radius),
        "max_simulated_spots": int(max_simulated_spots),
        "minimum_simulated_g_inv_nm": float(min_simulated_g),
        "with_direct_beam": bool(sim_with_direct_beam),
        "max_excitation_error": float(sim_max_excitation_error),
        "fallback_single_atom": bool(sim_fallback_single_atom),
        "zone_axis_mode": zone_axis_mode_value,
        "zone_axis_candidates": (
            None
            if effective_zone_axis_candidates is None
            else [list(axis) for axis in effective_zone_axis_candidates]
        ),
        "zone_axis_scan_candidates": [list(axis) for axis in zone_axis_candidates],
        "manual_zone_axis": list(manual_zone_axis),
        "preserve_candidate_zone_axis": zone_axis_mode_value == "inherit_b4_locked",
        "preserve_candidate_ranking": zone_axis_mode_value == "inherit_b4_locked",
        "rotation_axis": [float(item) for item in rotation_axis_values],
        "rotation_angles_deg": [float(item) for item in rotation_angles],
        "phase_match_enabled": bool(phase_match_enabled),
        "phase_match_d_tolerance_rel": float(d_tolerance_rel),
        "phase_match_angle_tolerance_deg": float(angle_tolerance_deg),
        "phase_match_min_spots": int(phase_match_min_spots),
        "phase_match_previous_score_weight": float(phase_match_previous_score_weight),
        "phase_match_max_rotation_candidates": int(phase_match_max_rotation_candidates),
    }

    manual_table_from_upload = _decode_uploaded_text(manual_table_upload).strip()
    manual_table_from_text = str(manual_table_text or "").strip()
    if manual_table_from_upload:
        manual_table_payload = manual_table_from_upload
        manual_table_source = "upload"
    else:
        manual_table_payload = manual_table_from_text
        manual_table_source = "paste"
    manual_spot_rows, manual_table_parse_warnings = _parse_manual_spot_table(manual_table_payload)

    output_root = Path(output_root_raw).expanduser()
    run_dir = output_root / _safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    if input_mode == "Upload file":
        if uploaded is None:
            st.info("Upload one file to start preview.")
            return
        input_path = _save_uploaded(uploaded, run_dir)
    elif input_mode == "Local file path":
        if not local_file_path:
            st.info("Enter a valid local file path to start preview.")
            return
        path_obj = Path(local_file_path).expanduser()
        if not path_obj.exists() or not path_obj.is_file():
            st.error(f"Local file not found: {path_obj}")
            return
        input_path = path_obj
    else:
        input_path = _build_demo_input(run_dir)

    input_ext = input_path.suffix.lower().lstrip(".")
    input_is_dm = input_ext in {"dm", "dm3", "dm4"}

    calibration_params: dict[str, Any] = {
        "diffraction_calibration_engine": str(diffraction_engine),
    }
    if manual_calibration_override:
        calibration_method_effective = str(cal_method)
        calibration_params.update(manual_calibration_params)
        size_policy_label = "Manual override enabled"
    else:
        calibration_method_effective = "auto"
        calibration_params["auto_scalebar_for_non_dm"] = not input_is_dm
        calibration_params["prefer_scalebar"] = not input_is_dm
        if input_is_dm:
            size_policy_label = "DM auto metadata calibration"
        else:
            size_policy_label = "Non-DM auto scale-bar calibration"

    st.caption(
        "Size policy: "
        f"`{size_policy_label}` | input=`.{input_ext or 'unknown'}` | "
        f"method=`{calibration_method_effective}`"
    )
    mode_options = [MODE_AUTO_EDIT, MODE_TWO_VECTOR]
    if "b1b3_selection_mode_main" not in st.session_state:
        st.session_state["b1b3_selection_mode_main"] = mode_options[0]
    if st.session_state.get("b1b3_selection_mode_main") not in mode_options:
        st.session_state["b1b3_selection_mode_main"] = mode_options[0]

    st.subheader("Spot Selection Mode")
    selection_mode = _render_mode_selector(mode_options)
    if selection_mode == MODE_AUTO_EDIT:
        st.caption(
            "Mode 1 mirrors the backup UI: run the pipeline detector first, then review "
            "or exclude auto peaks before reading the tables."
        )
    else:
        st.caption(
            "Mode 2 mirrors the backup UI: click two FFT basis spots and synthesize the "
            "reciprocal lattice from those vectors."
        )

    signature_payload = {
        "input_path": str(input_path),
        "input_mode": input_mode,
        "input_extension": input_ext,
        "manual_calibration_override": bool(manual_calibration_override),
        "calibration_method_effective": calibration_method_effective,
        "quality": quality_params,
        "calibration": calibration_params,
        "preprocessing": preprocessing_params,
        "recipe": recipe,
        "diffraction": diffraction_params,
    }
    signature = _json_signature(signature_payload)
    run_button_label = (
        "Run Auto-Peak Pipeline"
        if selection_mode == MODE_AUTO_EDIT
        else "Run Vector-Mode Base Analysis"
    )
    run_clicked = st.button(run_button_label, type="primary")
    has_cache = isinstance(st.session_state.get("b1b3_analysis_cache"), dict)
    last_signature = st.session_state.get("b1b3_last_signature")
    if auto_rerun:
        need_run = run_clicked or (not has_cache) or (last_signature != signature)
    else:
        need_run = run_clicked or (not has_cache)

    if need_run:
        with st.spinner("Running B1/B2/B3 analysis with current parameters..."):
            try:
                st.session_state["b1b3_analysis_cache"] = _run_b1b3_pipeline(
                    input_path=input_path,
                    quality_params=quality_params,
                    calibration_method=str(calibration_method_effective),
                    calibration_params=calibration_params,
                    recipe=str(recipe),
                    preprocessing_params=preprocessing_params,
                    diffraction_params=diffraction_params,
                    fft_center_mode=str(fft_center_mode),
                    direct_beam_search_radius=float(direct_beam_search_radius),
                    direct_beam_refine_window=int(direct_beam_refine_window),
                )
            except Exception as exc:  # noqa: BLE001 - UI should surface pipeline failures.
                st.error(f"B1/B2/B3 pipeline failed: {exc}")
                if has_cache:
                    st.warning("Keeping the previous successful analysis result on screen.")
                else:
                    return
            else:
                st.session_state["b1b3_last_signature"] = signature
                st.session_state["b1b3_auto_manual_points_xy"] = []
                st.session_state["b1b3_auto_removed_peak_ids"] = []
                st.session_state["b1b3_auto_remove_peak_ids_main"] = []
                st.session_state["b1b3_auto_canvas_action_signature"] = ""
                st.session_state["b1b3_vector_aux_points_xy"] = []
                st.session_state["b1b3_vector_removed_point_ids"] = []
                st.session_state["b1b3_vector_synth_signature"] = ""
                st.session_state["b1b3_vector_edit_action_signature"] = ""
                st.session_state["b1b3_vector_edit_canvas_nonce"] = int(
                    st.session_state.get("b1b3_vector_edit_canvas_nonce", 0)
                ) + 1
                st.session_state["b1b3_auto_manual_delete_nonce"] = int(
                    st.session_state.get("b1b3_auto_manual_delete_nonce", 0)
                ) + 1

    cache = st.session_state.get("b1b3_analysis_cache")
    if not isinstance(cache, dict):
        st.info(f"Click `{run_button_label}` to render previews.")
        return

    dataset = cache["dataset"]
    quality = cache["quality"]
    calibration = cache["calibration"]
    preprocessing = cache["preprocessing"]
    diffraction_auto = cache["diffraction_auto"]
    manual_candidate_phase_params = _merge_candidate_phase_params(
        candidate_phase_params,
        list(diffraction_auto.phase_candidates),
    )
    original_image = np.asarray(cache["original_image"], dtype=np.float32)
    processed_image = np.asarray(cache["processed_image"], dtype=np.float32)
    fft_analysis = cache["fft_analysis"]
    fft_warnings = list(cache.get("fft_warnings", []))

    fft_image_norm = _normalize_for_display(
        np.asarray(fft_analysis.log_magnitude, dtype=np.float32)
    )
    auto_points_xy = [(float(p.x), float(p.y)) for p in diffraction_auto.peaks]

    if "b1b3_canvas_nonce" not in st.session_state:
        st.session_state["b1b3_canvas_nonce"] = 0
    if "b1b3_last_clicked_points_xy" not in st.session_state:
        st.session_state["b1b3_last_clicked_points_xy"] = []
    if "b1b3_last_canvas_points_xy" not in st.session_state:
        st.session_state["b1b3_last_canvas_points_xy"] = []
    if "b1b3_manual_points_bank_xy" not in st.session_state:
        st.session_state["b1b3_manual_points_bank_xy"] = []
    if "b1b3_auto_manual_points_xy" not in st.session_state:
        st.session_state["b1b3_auto_manual_points_xy"] = []
    if "b1b3_auto_removed_peak_ids" not in st.session_state:
        st.session_state["b1b3_auto_removed_peak_ids"] = []
    if "b1b3_last_canvas_zoom_signature" not in st.session_state:
        st.session_state["b1b3_last_canvas_zoom_signature"] = ""
    if "b1b3_auto_canvas_nonce" not in st.session_state:
        st.session_state["b1b3_auto_canvas_nonce"] = 0
    if "b1b3_auto_last_canvas_zoom_signature" not in st.session_state:
        st.session_state["b1b3_auto_last_canvas_zoom_signature"] = ""
    if "b1b3_auto_canvas_action_signature" not in st.session_state:
        st.session_state["b1b3_auto_canvas_action_signature"] = ""
    if "b1b3_auto_manual_delete_nonce" not in st.session_state:
        st.session_state["b1b3_auto_manual_delete_nonce"] = 0
    if "b1b3_vector_aux_points_xy" not in st.session_state:
        st.session_state["b1b3_vector_aux_points_xy"] = []
    if "b1b3_vector_removed_point_ids" not in st.session_state:
        st.session_state["b1b3_vector_removed_point_ids"] = []
    if "b1b3_vector_synth_signature" not in st.session_state:
        st.session_state["b1b3_vector_synth_signature"] = ""
    if "b1b3_vector_edit_canvas_nonce" not in st.session_state:
        st.session_state["b1b3_vector_edit_canvas_nonce"] = 0
    if "b1b3_vector_edit_action_signature" not in st.session_state:
        st.session_state["b1b3_vector_edit_action_signature"] = ""
    if "b1b3_vector_aux_delete_nonce" not in st.session_state:
        st.session_state["b1b3_vector_aux_delete_nonce"] = 0
    if st.session_state.get("b1b3_last_selection_mode") != selection_mode:
        st.session_state["b1b3_last_selection_mode"] = selection_mode
        st.session_state["b1b3_canvas_nonce"] = int(st.session_state["b1b3_canvas_nonce"]) + 1
        st.session_state["b1b3_auto_canvas_nonce"] = int(
            st.session_state["b1b3_auto_canvas_nonce"]
        ) + 1
        st.session_state["b1b3_last_clicked_points_xy"] = []
        st.session_state["b1b3_last_canvas_points_xy"] = []
        st.session_state["b1b3_manual_points_bank_xy"] = []
        st.session_state["b1b3_auto_manual_points_xy"] = []
        st.session_state["b1b3_auto_removed_peak_ids"] = []
        st.session_state["b1b3_auto_remove_peak_ids_main"] = []
        st.session_state["b1b3_vector_aux_points_xy"] = []
        st.session_state["b1b3_vector_removed_point_ids"] = []
        st.session_state["b1b3_vector_synth_signature"] = ""
        st.session_state["b1b3_last_canvas_zoom_signature"] = ""
        st.session_state["b1b3_auto_last_canvas_zoom_signature"] = ""
        st.session_state["b1b3_auto_canvas_action_signature"] = ""
        st.session_state["b1b3_vector_edit_action_signature"] = ""

    _render_preview_metrics(
        quality=quality,
        calibration=calibration,
        diffraction_auto=diffraction_auto,
    )
    _render_preview_images(
        original_image=original_image,
        processed_image=processed_image,
        fft_image_norm=fft_image_norm,
    )

    manual_points_xy: list[tuple[float, float]] = []
    manual_points_refined_xy: list[tuple[float, float]] = []
    manual_refine_warnings: list[str] = []
    manual_refine_diagnostics: dict[str, Any] = {
        "enabled": False,
        "method": "none",
        "point_count": 0,
        "refined_count": 0,
        "mean_shift_px": 0.0,
        "max_shift_px": 0.0,
        "fallback_count": 0,
    }
    vector_basis_points_xy: list[tuple[float, float]] = []
    vector_synth_points_xy: list[tuple[float, float]] = []
    vector_synth_warnings: list[str] = []
    vector_synth_diagnostics: dict[str, Any] = {"enabled": False, "mode": "none"}
    auto_remove_peak_ids: list[int] = []
    auto_canvas_diagnostics: dict[str, Any] = {"enabled": False}
    if selection_mode == MODE_AUTO_EDIT:
        st.subheader("Auto Peak Review")
        st.caption(
            "Choose a canvas action first: add mode records missed spots, delete mode "
            "removes the nearest auto or manual point."
        )

        manual_points_state = _normalize_points_xy(
            st.session_state.get("b1b3_auto_manual_points_xy", [])
        )
        auto_remove_state = sorted(
            {
                int(item)
                for item in st.session_state.get("b1b3_auto_removed_peak_ids", [])
                if isinstance(item, int | float)
            }
        )
        auto_canvas_col, auto_status_col = st.columns([1.05, 0.95])
        with auto_status_col:
            st.subheader("Auto Edit")
            action_options = ["Add point", "Delete point"]
            current_action = st.session_state.get("b1b3_auto_canvas_action_mode")
            action_index = (
                action_options.index(str(current_action))
                if str(current_action) in action_options
                else 0
            )
            auto_canvas_action = st.radio(
                "Canvas click action",
                options=action_options,
                index=action_index,
                horizontal=True,
                key="b1b3_auto_canvas_action_mode",
            )
            auto_canvas_zoom = st.slider(
                "Auto edit zoom",
                min_value=1.0,
                max_value=6.0,
                value=1.0,
                step=0.1,
                help=(
                    "Zoom into the FFT center while keeping the edit canvas visible. "
                    "Changing zoom clears current canvas clicks."
                ),
                key="b1b3_auto_canvas_zoom_inline",
            )
            auto_hit_radius_px = st.slider(
                "Delete hit radius (px)",
                min_value=1.0,
                max_value=20.0,
                value=6.0,
                step=0.5,
                help="Click within this radius of an existing point in delete mode.",
                key="b1b3_auto_hit_radius_px",
            )

        auto_action_signature = f"{auto_canvas_action}|{float(auto_canvas_zoom):.2f}"
        if (
            st.session_state.get("b1b3_auto_canvas_action_signature")
            != auto_action_signature
        ):
            st.session_state["b1b3_auto_canvas_action_signature"] = auto_action_signature
            st.session_state["b1b3_auto_canvas_nonce"] = int(
                st.session_state.get("b1b3_auto_canvas_nonce", 0)
            ) + 1

        auto_canvas_bg, auto_sx, auto_sy, auto_x_offset, auto_y_offset = _to_fft_display_image(
            fft_image_norm,
            max_side=900,
            zoom=float(auto_canvas_zoom),
            center_xy=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
        )
        auto_canvas_bg = _draw_auto_points_on_canvas_image(
            auto_canvas_bg,
            auto_points_xy=list(auto_points_xy),
            center_xy=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
            sx=float(auto_sx),
            sy=float(auto_sy),
            x_offset=float(auto_x_offset),
            y_offset=float(auto_y_offset),
            removed_auto_peak_ids=set(auto_remove_state),
            manual_points_xy=list(manual_points_state),
        )
        with auto_canvas_col:
            st.write(
                f"{auto_canvas_action} on FFT (zoom x{float(auto_canvas_zoom):.1f}):"
            )
            auto_canvas_key = (
                f"b1b3-auto-edit-canvas-"
                f"{int(st.session_state.get('b1b3_auto_canvas_nonce', 0))}"
            )
            auto_canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0.25)",
                stroke_width=2,
                stroke_color=(
                    "#00ffaa" if str(auto_canvas_action) == "Add point" else "#ff3333"
                ),
                background_image=auto_canvas_bg,
                drawing_mode="point",
                point_display_radius=4,
                display_toolbar=True,
                update_streamlit=True,
                height=auto_canvas_bg.height,
                width=auto_canvas_bg.width,
                key=auto_canvas_key,
            )
            if st.button("Clear Auto Canvas Edits", key="b1b3_clear_auto_canvas_edits_btn"):
                st.session_state["b1b3_auto_manual_points_xy"] = []
                st.session_state["b1b3_auto_removed_peak_ids"] = []
                st.session_state["b1b3_auto_remove_peak_ids_main"] = []
                st.session_state["b1b3_auto_canvas_nonce"] = int(
                    st.session_state.get("b1b3_auto_canvas_nonce", 0)
                ) + 1
                st.session_state["b1b3_auto_manual_delete_nonce"] = int(
                    st.session_state.get("b1b3_auto_manual_delete_nonce", 0)
                ) + 1
                st.rerun()

        auto_canvas_clicks_xy = _extract_canvas_points_xy(
            auto_canvas_result,
            sx=float(auto_sx),
            sy=float(auto_sy),
            x_offset=float(auto_x_offset),
            y_offset=float(auto_y_offset),
        )
        if auto_canvas_clicks_xy:
            if str(auto_canvas_action) == "Add point":
                manual_points_state, auto_canvas_diagnostics = _append_manual_canvas_points(
                    manual_points_state,
                    auto_canvas_clicks_xy,
                    manual_min_distance_px=float(manual_min_distance),
                    max_points=int(manual_max_points),
                )
            else:
                (
                    canvas_remove_peak_ids,
                    canvas_remove_manual_ids,
                    auto_canvas_diagnostics,
                ) = _resolve_delete_canvas_clicks(
                    auto_canvas_clicks_xy,
                    auto_points_xy=list(auto_points_xy),
                    manual_points_xy=list(manual_points_state),
                    hit_radius_px=float(auto_hit_radius_px),
                )
                auto_remove_state = sorted(
                    {int(item) for item in list(auto_remove_state) + list(canvas_remove_peak_ids)}
                )
                remove_manual_set = {int(item) for item in canvas_remove_manual_ids}
                manual_points_state = [
                    point
                    for idx, point in enumerate(manual_points_state)
                    if int(idx) not in remove_manual_set
                ]
            st.session_state["b1b3_auto_manual_points_xy"] = list(manual_points_state)
            st.session_state["b1b3_auto_removed_peak_ids"] = list(auto_remove_state)
            st.session_state["b1b3_auto_remove_peak_ids_main"] = list(auto_remove_state)
            st.session_state["b1b3_auto_canvas_nonce"] = int(
                st.session_state.get("b1b3_auto_canvas_nonce", 0)
            ) + 1
            st.rerun()

        manual_points_state = _normalize_points_xy(
            st.session_state.get("b1b3_auto_manual_points_xy", [])
        )
        auto_remove_state = sorted(
            {
                int(item)
                for item in st.session_state.get("b1b3_auto_removed_peak_ids", [])
                if isinstance(item, int | float)
            }
        )

        auto_options = list(range(len(auto_points_xy)))
        auto_remove_key = "b1b3_auto_remove_peak_ids_main"
        if st.session_state.get(auto_remove_key) != auto_remove_state:
            st.session_state[auto_remove_key] = list(auto_remove_state)
        with auto_status_col.expander("Auto peak deletion list", expanded=False):
            list_remove_peak_ids = st.multiselect(
                "Delete auto peak ids",
                options=auto_options,
                format_func=lambda idx: (
                    f"{int(idx)}: x={float(auto_points_xy[int(idx)][0]):.1f}, "
                    f"y={float(auto_points_xy[int(idx)][1]):.1f}"
                ),
                key=auto_remove_key,
            )
        auto_remove_peak_ids = sorted({int(item) for item in list_remove_peak_ids})
        st.session_state["b1b3_auto_removed_peak_ids"] = list(auto_remove_peak_ids)

        manual_rows = [
            {
                "manual_id": f"M{idx}",
                "x": round(float(point[0]), 4),
                "y": round(float(point[1]), 4),
            }
            for idx, point in enumerate(manual_points_state)
        ]
        with auto_status_col.expander("Manual added point list", expanded=bool(manual_rows)):
            if manual_rows:
                st.dataframe(manual_rows, use_container_width=True, hide_index=True, height=160)
                manual_delete_key = (
                    "b1b3_auto_manual_delete_ids_"
                    f"{int(st.session_state.get('b1b3_auto_manual_delete_nonce', 0))}"
                )
                manual_delete_ids = st.multiselect(
                    "Delete manual point ids",
                    options=list(range(len(manual_points_state))),
                    format_func=lambda idx: (
                        f"M{int(idx)}: x={float(manual_points_state[int(idx)][0]):.1f}, "
                        f"y={float(manual_points_state[int(idx)][1]):.1f}"
                    ),
                    key=manual_delete_key,
                )
                if st.button(
                    "Delete selected manual points",
                    disabled=not manual_delete_ids,
                    key="b1b3_delete_selected_manual_points_btn",
                ):
                    delete_set = {int(item) for item in manual_delete_ids}
                    st.session_state["b1b3_auto_manual_points_xy"] = [
                        point
                        for idx, point in enumerate(manual_points_state)
                        if int(idx) not in delete_set
                    ]
                    st.session_state["b1b3_auto_manual_delete_nonce"] = int(
                        st.session_state.get("b1b3_auto_manual_delete_nonce", 0)
                    ) + 1
                    st.session_state["b1b3_auto_canvas_nonce"] = int(
                        st.session_state.get("b1b3_auto_canvas_nonce", 0)
                    ) + 1
                    st.rerun()
            else:
                st.info("No manual points added.")

        manual_points_state = _normalize_points_xy(
            st.session_state.get("b1b3_auto_manual_points_xy", [])
        )

        manual_points_xy = _deduplicate_points(
            list(manual_points_state),
            min_distance_px=float(manual_min_distance),
        )
        if len(manual_points_xy) > int(manual_max_points):
            manual_points_xy = manual_points_xy[: int(manual_max_points)]
        st.session_state["b1b3_auto_manual_points_xy"] = list(manual_points_xy)
        manual_points_refined_xy = list(manual_points_xy)
        refine_active = bool(manual_refine_enabled) and str(manual_refine_method) != "none"
        if refine_active and manual_points_xy:
            (
                manual_points_refined_xy,
                manual_refine_diagnostics,
                manual_refine_warnings,
            ) = _refine_manual_points(
                manual_points_xy,
                fft_log_magnitude=np.asarray(fft_analysis.log_magnitude, dtype=np.float64),
                method=str(manual_refine_method),
                window_radius=int(manual_refine_window_radius),
                max_shift_px=float(manual_refine_max_shift_px),
            )
            manual_points_refined_xy = _deduplicate_points(
                manual_points_refined_xy,
                min_distance_px=float(manual_min_distance),
            )
        else:
            manual_refine_diagnostics = {
                "enabled": bool(refine_active),
                "method": str(manual_refine_method),
                "point_count": int(len(manual_points_xy)),
                "refined_count": int(len(manual_points_refined_xy)),
                "mean_shift_px": 0.0,
                "max_shift_px": 0.0,
                "fallback_count": 0,
            }

        with auto_status_col:
            review_cols = st.columns(4)
            review_cols[0].metric("Auto peaks", str(len(auto_points_xy)))
            review_cols[1].metric("Removed", str(len(auto_remove_peak_ids)))
            review_cols[2].metric("Added", str(len(manual_points_refined_xy)))
            review_cols[3].metric("Mode", "Auto edit")
            if auto_remove_peak_ids:
                st.caption(
                    f"Removed auto peak id(s): {', '.join(str(i) for i in auto_remove_peak_ids)}"
                )
            if manual_points_refined_xy:
                status_rows = []
                for idx, raw_point in enumerate(manual_points_xy):
                    refined_point = (
                        manual_points_refined_xy[idx]
                        if idx < len(manual_points_refined_xy)
                        else raw_point
                    )
                    status_rows.append(
                        {
                            "manual_id": f"M{idx}",
                            "raw_x": round(float(raw_point[0]), 3),
                            "raw_y": round(float(raw_point[1]), 3),
                            "refined_x": round(float(refined_point[0]), 3),
                            "refined_y": round(float(refined_point[1]), 3),
                        }
                    )
                st.dataframe(status_rows, use_container_width=True, hide_index=True, height=170)
            st.caption(
                "Canvas auto ids are cyan/red; manual ids are magenta labels M0, M1, ..."
            )

    if selection_mode == MODE_TWO_VECTOR:
        st.subheader("Manual Basis Selection On FFT")
        st.caption(
            "Click two basis points directly on the FFT image. The app immediately "
            "synthesizes lattice spots from those two vectors, matching the backup UI flow."
        )
        if str(vector_snap_mode) == "none" and not bool(manual_refine_enabled):
            st.caption(
                "Strict mode is active: clicked basis points stay fixed and synthesized "
                "spots are exact integer combinations of g1/g2."
            )
        elif str(vector_snap_mode) == "none" and bool(manual_refine_enabled):
            st.caption(
                "Gaussian basis calibration is active: clicked g1/g2 are refined first, "
                "then synthesized spots stay on strict integer combinations of refined g1/g2."
            )
        elif str(vector_snap_mode) == "local_max":
            st.caption(
                "Local-max snap is active: synthesized spots may shift to nearby FFT maxima."
            )

        canvas_col, status_col = st.columns([1.05, 0.95])
        with status_col:
            st.subheader("Canvas View")
            fft_canvas_zoom = st.slider(
                "FFT canvas zoom",
                min_value=1.0,
                max_value=6.0,
                value=1.0,
                step=0.1,
                help=(
                    "Zoom into a center-cropped FFT view for more precise clicks while "
                    "keeping the canvas fully visible. Changing this value resets the "
                    "current canvas marks."
                ),
                key="b1b3_fft_canvas_zoom_inline",
            )
            st.caption(
                "Zoom crops around the FFT center and keeps the canvas fixed on screen. "
                "Changing zoom clears current marks so coordinates stay aligned."
            )

        canvas_zoom_signature = f"{float(fft_canvas_zoom):.2f}"
        if st.session_state.get("b1b3_last_canvas_zoom_signature") != canvas_zoom_signature:
            st.session_state["b1b3_last_canvas_zoom_signature"] = canvas_zoom_signature
            st.session_state["b1b3_canvas_nonce"] = int(
                st.session_state.get("b1b3_canvas_nonce", 0)
            ) + 1
            st.session_state["b1b3_last_clicked_points_xy"] = []
            st.session_state["b1b3_last_canvas_points_xy"] = []
            st.session_state["b1b3_manual_points_bank_xy"] = []

        canvas_bg, sx, sy, canvas_x_offset, canvas_y_offset = _to_fft_display_image(
            fft_image_norm,
            max_side=900,
            zoom=float(fft_canvas_zoom),
            center_xy=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
        )
        with canvas_col:
            st.write(f"Click two FFT basis points (zoom x{float(fft_canvas_zoom):.1f}):")
            canvas_key = (
                f"b1b3-fft-basis-canvas-"
                f"{int(st.session_state.get('b1b3_canvas_nonce', 0))}"
            )
            canvas_result = st_canvas(
                fill_color="rgba(255,255,255,0.25)",
                stroke_width=2,
                stroke_color="#00ffaa",
                background_image=canvas_bg,
                drawing_mode="point",
                point_display_radius=4,
                display_toolbar=True,
                update_streamlit=True,
                height=canvas_bg.height,
                width=canvas_bg.width,
                key=canvas_key,
            )
            if st.button("Clear Basis Points", key="b1b3_clear_basis_points_btn"):
                st.session_state["b1b3_canvas_nonce"] = int(
                    st.session_state.get("b1b3_canvas_nonce", 0)
                ) + 1
                st.session_state["b1b3_last_clicked_points_xy"] = []
                st.session_state["b1b3_last_canvas_points_xy"] = []
                st.session_state["b1b3_manual_points_bank_xy"] = []
                st.session_state["b1b3_vector_aux_points_xy"] = []
                st.session_state["b1b3_vector_removed_point_ids"] = []
                st.session_state["b1b3_vector_synth_signature"] = ""
                st.session_state["b1b3_vector_edit_canvas_nonce"] = int(
                    st.session_state.get("b1b3_vector_edit_canvas_nonce", 0)
                ) + 1
                st.rerun()

        clicked_points_xy = _extract_canvas_points_xy(
            canvas_result,
            sx=sx,
            sy=sy,
            x_offset=float(canvas_x_offset),
            y_offset=float(canvas_y_offset),
        )
        if clicked_points_xy:
            st.session_state["b1b3_last_clicked_points_xy"] = list(clicked_points_xy)
            st.session_state["b1b3_last_canvas_points_xy"] = list(clicked_points_xy)

        manual_points_xy = _deduplicate_points(
            list(st.session_state.get("b1b3_last_canvas_points_xy", [])),
            min_distance_px=float(manual_min_distance),
        )
        if len(manual_points_xy) > int(manual_max_points):
            manual_points_xy = manual_points_xy[: int(manual_max_points)]
        st.session_state["b1b3_manual_points_bank_xy"] = list(manual_points_xy)
        manual_points_refined_xy = list(manual_points_xy)

        refine_active = bool(manual_refine_enabled) and str(manual_refine_method) != "none"
        if refine_active and manual_points_xy:
            (
                manual_points_refined_xy,
                manual_refine_diagnostics,
                manual_refine_warnings,
            ) = _refine_manual_points(
                manual_points_xy,
                fft_log_magnitude=np.asarray(fft_analysis.log_magnitude, dtype=np.float64),
                method=str(manual_refine_method),
                window_radius=int(manual_refine_window_radius),
                max_shift_px=float(manual_refine_max_shift_px),
            )
            manual_points_refined_xy = _deduplicate_points(
                manual_points_refined_xy,
                min_distance_px=float(manual_min_distance),
            )
            if len(manual_points_refined_xy) > int(manual_max_points):
                manual_points_refined_xy = manual_points_refined_xy[
                    : int(manual_max_points)
                ]
            manual_refine_diagnostics["refined_count"] = int(len(manual_points_refined_xy))
        else:
            manual_refine_diagnostics = {
                "enabled": bool(refine_active),
                "method": str(manual_refine_method),
                "point_count": int(len(manual_points_xy)),
                "refined_count": int(len(manual_points_refined_xy)),
                "mean_shift_px": 0.0,
                "max_shift_px": 0.0,
                "fallback_count": 0,
            }

        with status_col:
            st.write(f"Clicked points: {len(manual_points_xy)}")
            if manual_points_refined_xy:
                st.code(
                    "\n".join(
                        f"{idx + 1}: x={pt[0]:.2f}, y={pt[1]:.2f}"
                        for idx, pt in enumerate(manual_points_refined_xy)
                    ),
                    language="text",
                )
            else:
                st.info("Click two FFT spots to define g1 and g2.")
            if len(manual_points_refined_xy) >= 2:
                st.success("Two basis spots are ready for vector synthesis.")
            if bool(manual_refine_diagnostics.get("enabled")) and manual_points_refined_xy:
                st.caption(
                    "Refine "
                    f"{manual_refine_diagnostics.get('method')} | "
                    "mean shift="
                    f"{float(manual_refine_diagnostics.get('mean_shift_px', 0.0)):.3f}px | "
                    "max shift="
                    f"{float(manual_refine_diagnostics.get('max_shift_px', 0.0)):.3f}px | "
                    f"fallback={int(manual_refine_diagnostics.get('fallback_count', 0))}"
                )
                point_pairs = manual_refine_diagnostics.get("point_pairs")
                if isinstance(point_pairs, list) and point_pairs:
                    st.dataframe(point_pairs, use_container_width=True, height=180)

    if selection_mode == MODE_AUTO_EDIT:
        remove_id_set = {int(item) for item in auto_remove_peak_ids}
        auto_kept_points_xy = [
            (float(pt[0]), float(pt[1]))
            for idx, pt in enumerate(auto_points_xy)
            if int(idx) not in remove_id_set
        ]
        selected_points_xy = _deduplicate_points(
            list(auto_kept_points_xy) + list(manual_points_refined_xy),
            min_distance_px=float(manual_min_distance),
        )
        selected_points_xy = _sort_points_by_fft_intensity(
            selected_points_xy,
            np.asarray(fft_analysis.log_magnitude, dtype=np.float64),
        )
    elif selection_mode == MODE_TWO_VECTOR:
        basis_candidates = (
            list(manual_points_refined_xy)
            if manual_points_refined_xy
            else list(manual_points_xy)
        )
        if len(basis_candidates) < 2:
            st.warning("Please click at least two FFT spots as basis vectors g1/g2.")
            selected_points_xy = []
        else:
            if len(basis_candidates) > 2:
                st.warning(
                    "More than two points were clicked. "
                    "Only the first two are used as basis vectors."
                )
            vector_basis_points_xy = [basis_candidates[0], basis_candidates[1]]
            h, w = fft_image_norm.shape
            (
                vector_synth_points_xy,
                vector_synth_diagnostics,
                vector_synth_warnings,
            ) = _synthesize_points_from_two_basis(
                center_xy=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
                basis_points_xy=vector_basis_points_xy,
                shape_hw=(int(h), int(w)),
                fft_log_magnitude=np.asarray(fft_analysis.log_magnitude, dtype=np.float64),
                max_order=int(vector_expand_order),
                min_distance_px=float(manual_min_distance),
                snap_mode=str(vector_snap_mode),
                snap_radius_px=float(vector_snap_radius_px),
                local_intensity_percentile=float(vector_local_intensity_percentile),
                max_points=int(vector_max_points),
            )
            vector_synth_signature = _json_signature(
                {
                    "basis": _round_points_for_signature(vector_basis_points_xy),
                    "synth": _round_points_for_signature(vector_synth_points_xy),
                    "snap_mode": str(vector_snap_mode),
                    "expand_order": int(vector_expand_order),
                    "manual_refine": manual_refine_diagnostics,
                }
            )
            if st.session_state.get("b1b3_vector_synth_signature") != vector_synth_signature:
                st.session_state["b1b3_vector_synth_signature"] = vector_synth_signature
                st.session_state["b1b3_vector_aux_points_xy"] = []
                st.session_state["b1b3_vector_removed_point_ids"] = []
                st.session_state["b1b3_vector_edit_action_signature"] = ""
                st.session_state["b1b3_vector_edit_canvas_nonce"] = int(
                    st.session_state.get("b1b3_vector_edit_canvas_nonce", 0)
                ) + 1

            vector_aux_points_xy = _normalize_points_xy(
                st.session_state.get("b1b3_vector_aux_points_xy", [])
            )
            vector_removed_ids = sorted(
                {
                    int(item)
                    for item in st.session_state.get("b1b3_vector_removed_point_ids", [])
                    if isinstance(item, int | float)
                }
            )
            if vector_synth_diagnostics.get("enabled"):
                st.caption(
                    "Vector synthesis: "
                    f"selected={int(vector_synth_diagnostics.get('selected_count', 0))}, "
                    f"dropped={int(vector_synth_diagnostics.get('dropped_non_basis_count', 0))}, "
                    "basis angle="
                    f"{float(vector_synth_diagnostics.get('basis_angle_deg', 0.0)):.2f} deg"
                )
                pair_rows = vector_synth_diagnostics.get("seed_to_selected_pairs")
                if isinstance(pair_rows, list) and pair_rows:
                    st.dataframe(
                        pair_rows[: min(120, len(pair_rows))],
                        use_container_width=True,
                        height=200,
                    )

            st.subheader("Vector Point Review")
            st.caption(
                "After two-point synthesis, add missed spots or delete synthesized/auxiliary "
                "spots. The reviewed points are merged and sorted before B3/B4 metrics."
            )
            vector_edit_canvas_col, vector_edit_status_col = st.columns([1.05, 0.95])
            with vector_edit_status_col:
                vector_action_options = ["Add point", "Delete point"]
                current_vector_action = st.session_state.get("b1b3_vector_edit_action_mode")
                vector_action_index = (
                    vector_action_options.index(str(current_vector_action))
                    if str(current_vector_action) in vector_action_options
                    else 0
                )
                vector_edit_action = st.radio(
                    "Vector canvas action",
                    options=vector_action_options,
                    index=vector_action_index,
                    horizontal=True,
                    key="b1b3_vector_edit_action_mode",
                )
                vector_edit_zoom = st.slider(
                    "Vector edit zoom",
                    min_value=1.0,
                    max_value=6.0,
                    value=float(fft_canvas_zoom),
                    step=0.1,
                    key="b1b3_vector_edit_zoom",
                    help="Zoomed center crop for reviewing synthesized vector points.",
                )
                vector_hit_radius_px = st.slider(
                    "Vector delete hit radius (px)",
                    min_value=1.0,
                    max_value=20.0,
                    value=6.0,
                    step=0.5,
                    key="b1b3_vector_hit_radius_px",
                )

            vector_action_signature = (
                f"{vector_edit_action}|{float(vector_edit_zoom):.2f}|"
                f"{float(vector_hit_radius_px):.2f}"
            )
            if (
                st.session_state.get("b1b3_vector_edit_action_signature")
                != vector_action_signature
            ):
                st.session_state["b1b3_vector_edit_action_signature"] = vector_action_signature
                st.session_state["b1b3_vector_edit_canvas_nonce"] = int(
                    st.session_state.get("b1b3_vector_edit_canvas_nonce", 0)
                ) + 1

            (
                vector_edit_bg,
                vector_edit_sx,
                vector_edit_sy,
                vector_edit_x_offset,
                vector_edit_y_offset,
            ) = _to_fft_display_image(
                fft_image_norm,
                max_side=900,
                zoom=float(vector_edit_zoom),
                center_xy=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
            )
            vector_edit_bg = _draw_auto_points_on_canvas_image(
                vector_edit_bg,
                auto_points_xy=list(vector_synth_points_xy),
                center_xy=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
                sx=float(vector_edit_sx),
                sy=float(vector_edit_sy),
                x_offset=float(vector_edit_x_offset),
                y_offset=float(vector_edit_y_offset),
                removed_auto_peak_ids=set(vector_removed_ids),
                manual_points_xy=list(vector_aux_points_xy),
            )
            with vector_edit_canvas_col:
                st.write(
                    f"{vector_edit_action} synthesized points "
                    f"(zoom x{float(vector_edit_zoom):.1f}):"
                )
                vector_edit_canvas_result = st_canvas(
                    fill_color="rgba(255,255,255,0.25)",
                    stroke_width=2,
                    stroke_color=(
                        "#00ffaa" if str(vector_edit_action) == "Add point" else "#ff3333"
                    ),
                    background_image=vector_edit_bg,
                    drawing_mode="point",
                    point_display_radius=4,
                    display_toolbar=True,
                    update_streamlit=True,
                    height=vector_edit_bg.height,
                    width=vector_edit_bg.width,
                    key=(
                        "b1b3-vector-edit-canvas-"
                        f"{int(st.session_state.get('b1b3_vector_edit_canvas_nonce', 0))}"
                    ),
                )
                if st.button("Clear Vector Point Edits", key="b1b3_clear_vector_edits_btn"):
                    st.session_state["b1b3_vector_aux_points_xy"] = []
                    st.session_state["b1b3_vector_removed_point_ids"] = []
                    st.session_state["b1b3_vector_aux_delete_nonce"] = int(
                        st.session_state.get("b1b3_vector_aux_delete_nonce", 0)
                    ) + 1
                    st.session_state["b1b3_vector_edit_canvas_nonce"] = int(
                        st.session_state.get("b1b3_vector_edit_canvas_nonce", 0)
                    ) + 1
                    st.rerun()

            vector_edit_clicks_xy = _extract_canvas_points_xy(
                vector_edit_canvas_result,
                sx=float(vector_edit_sx),
                sy=float(vector_edit_sy),
                x_offset=float(vector_edit_x_offset),
                y_offset=float(vector_edit_y_offset),
            )
            if vector_edit_clicks_xy:
                if str(vector_edit_action) == "Add point":
                    vector_aux_points_xy, _vector_edit_diag = _append_manual_canvas_points(
                        vector_aux_points_xy,
                        vector_edit_clicks_xy,
                        manual_min_distance_px=float(manual_min_distance),
                        max_points=int(manual_max_points),
                    )
                else:
                    (
                        canvas_remove_synth_ids,
                        canvas_remove_aux_ids,
                        _vector_edit_diag,
                    ) = _resolve_delete_canvas_clicks(
                        vector_edit_clicks_xy,
                        auto_points_xy=list(vector_synth_points_xy),
                        manual_points_xy=list(vector_aux_points_xy),
                        hit_radius_px=float(vector_hit_radius_px),
                    )
                    vector_removed_ids = sorted(
                        {
                            int(item)
                            for item in list(vector_removed_ids)
                            + list(canvas_remove_synth_ids)
                        }
                    )
                    remove_aux_set = {int(item) for item in canvas_remove_aux_ids}
                    vector_aux_points_xy = [
                        point
                        for idx, point in enumerate(vector_aux_points_xy)
                        if int(idx) not in remove_aux_set
                    ]
                st.session_state["b1b3_vector_aux_points_xy"] = list(vector_aux_points_xy)
                st.session_state["b1b3_vector_removed_point_ids"] = list(vector_removed_ids)
                st.session_state["b1b3_vector_edit_canvas_nonce"] = int(
                    st.session_state.get("b1b3_vector_edit_canvas_nonce", 0)
                ) + 1
                st.rerun()

            vector_aux_points_xy = _normalize_points_xy(
                st.session_state.get("b1b3_vector_aux_points_xy", [])
            )
            vector_removed_ids = sorted(
                {
                    int(item)
                    for item in st.session_state.get("b1b3_vector_removed_point_ids", [])
                    if isinstance(item, int | float)
                }
            )
            with vector_edit_status_col:
                synth_delete_key = "b1b3_vector_removed_point_ids_widget"
                if st.session_state.get(synth_delete_key) != vector_removed_ids:
                    st.session_state[synth_delete_key] = list(vector_removed_ids)
                synth_delete_ids = st.multiselect(
                    "Delete synthesized point ids",
                    options=list(range(len(vector_synth_points_xy))),
                    format_func=lambda idx: (
                        f"{int(idx)}: x={float(vector_synth_points_xy[int(idx)][0]):.1f}, "
                        f"y={float(vector_synth_points_xy[int(idx)][1]):.1f}"
                    ),
                    key=synth_delete_key,
                )
                vector_removed_ids = sorted({int(item) for item in synth_delete_ids})
                st.session_state["b1b3_vector_removed_point_ids"] = list(vector_removed_ids)

                aux_rows = [
                    {
                        "aux_id": f"M{idx}",
                        "x": round(float(point[0]), 4),
                        "y": round(float(point[1]), 4),
                    }
                    for idx, point in enumerate(vector_aux_points_xy)
                ]
                if aux_rows:
                    st.dataframe(aux_rows, use_container_width=True, hide_index=True, height=150)
                    aux_delete_key = (
                        "b1b3_vector_aux_delete_ids_"
                        f"{int(st.session_state.get('b1b3_vector_aux_delete_nonce', 0))}"
                    )
                    aux_delete_ids = st.multiselect(
                        "Delete auxiliary point ids",
                        options=list(range(len(vector_aux_points_xy))),
                        format_func=lambda idx: (
                            f"M{int(idx)}: x={float(vector_aux_points_xy[int(idx)][0]):.1f}, "
                            f"y={float(vector_aux_points_xy[int(idx)][1]):.1f}"
                        ),
                        key=aux_delete_key,
                    )
                    if st.button(
                        "Delete selected auxiliary points",
                        disabled=not aux_delete_ids,
                        key="b1b3_delete_vector_aux_points_btn",
                    ):
                        aux_delete_set = {int(item) for item in aux_delete_ids}
                        st.session_state["b1b3_vector_aux_points_xy"] = [
                            point
                            for idx, point in enumerate(vector_aux_points_xy)
                            if int(idx) not in aux_delete_set
                        ]
                        st.session_state["b1b3_vector_aux_delete_nonce"] = int(
                            st.session_state.get("b1b3_vector_aux_delete_nonce", 0)
                        ) + 1
                        st.session_state["b1b3_vector_edit_canvas_nonce"] = int(
                            st.session_state.get("b1b3_vector_edit_canvas_nonce", 0)
                        ) + 1
                        st.rerun()
                else:
                    st.caption("No auxiliary points added.")

            synth_kept_points_xy = [
                (float(point[0]), float(point[1]))
                for idx, point in enumerate(vector_synth_points_xy)
                if int(idx) not in set(vector_removed_ids)
            ]
            selected_points_xy = _deduplicate_points(
                list(synth_kept_points_xy) + list(vector_aux_points_xy),
                min_distance_px=float(manual_min_distance),
            )
            selected_points_xy = _sort_points_by_fft_intensity(
                selected_points_xy,
                np.asarray(fft_analysis.log_magnitude, dtype=np.float64),
            )
    else:
        selected_points_xy = list(auto_points_xy)

    manual_signature = _json_signature(
        {
            "mode": selection_mode,
            "selected_points_xy": _round_points_for_signature(selected_points_xy),
            "fft_center": _round_points_for_signature(
                [(float(fft_analysis.center[0]), float(fft_analysis.center[1]))]
            ),
            "calibration_scale": _measurement_value(
                getattr(calibration, "diffraction_scale", None)
            ),
            "calibration_scale_unc": _measurement_uncertainty(
                getattr(calibration, "diffraction_scale", None)
            ),
            "diffraction": diffraction_params,
            "candidate_phase_params": manual_candidate_phase_params,
            "candidate_phases": list(candidate_phase_names),
            "manual_refine": manual_refine_diagnostics,
            "vector_synthesis": vector_synth_diagnostics,
            "auto_remove_peak_ids": [int(i) for i in auto_remove_peak_ids],
        }
    )
    manual_cache = st.session_state.get("b1b3_manual_result_cache")
    if isinstance(manual_cache, dict) and manual_cache.get("signature") == manual_signature:
        manual_result = manual_cache["result"]
    else:
        manual_result = _build_manual_diffraction(
            selected_points_xy=selected_points_xy,
            fft_log_magnitude=np.asarray(fft_analysis.log_magnitude, dtype=np.float64),
            fft_center=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
            calibration=calibration,
            diff_params=diffraction_params,
            candidate_phase_params=manual_candidate_phase_params,
            candidate_phases=list(candidate_phase_names),
        )
        manual_result = _with_manual_context(
            manual_result,
            selection_mode=selection_mode,
            selected_points_xy=selected_points_xy,
            manual_refine_warnings=manual_refine_warnings,
            vector_synth_warnings=vector_synth_warnings,
            manual_refine_diagnostics=manual_refine_diagnostics,
            vector_synth_diagnostics=vector_synth_diagnostics,
            auto_remove_peak_ids=auto_remove_peak_ids,
            manual_points_refined_xy=manual_points_refined_xy,
        )
        st.session_state["b1b3_manual_result_cache"] = {
            "signature": manual_signature,
            "result": manual_result,
        }

    if selection_mode == MODE_AUTO_EDIT:
        annotate_auto_ids = overlay_label_mode == "Both"
        annotate_selected_ids = overlay_label_mode != "Off"
    elif selection_mode == MODE_TWO_VECTOR:
        annotate_auto_ids = False
        annotate_selected_ids = overlay_label_mode != "Off"
    else:
        annotate_auto_ids = False
        annotate_selected_ids = overlay_label_mode in {"Selected points", "Both"}

    overlay_fig = _plot_peak_overlay(
        fft_image_norm,
        center_xy=(float(fft_analysis.center[0]), float(fft_analysis.center[1])),
        auto_points_xy=(auto_points_xy if selection_mode == MODE_AUTO_EDIT else []),
        selected_points_xy=selected_points_xy,
        manual_raw_points_xy=(
            manual_points_xy if selection_mode in {MODE_AUTO_EDIT, MODE_TWO_VECTOR} else []
        ),
        manual_refined_points_xy=(
            manual_points_refined_xy if selection_mode in {MODE_AUTO_EDIT, MODE_TWO_VECTOR} else []
        ),
        basis_points_xy=(vector_basis_points_xy if selection_mode == MODE_TWO_VECTOR else []),
        annotate_auto_ids=annotate_auto_ids,
        annotate_selected_ids=annotate_selected_ids,
    )
    st.pyplot(overlay_fig, clear_figure=True)
    if selection_mode == MODE_AUTO_EDIT:
        st.caption(
            "Red `selected` labels are the active table `peak_id` after canvas edits "
            "and intensity sorting."
        )
    else:
        st.caption(
            "Vector-mode overlay hides pipeline auto peaks; selected markers come only "
            "from two-point vector synthesis."
        )
    st.caption(
        "The red `selected` markers are the active spot set for the selected mode."
    )

    auto_rows = _build_peak_rows(
        peaks=list(diffraction_auto.peaks),
        g_vectors=list(diffraction_auto.g_vectors),
        d_by_peak=list(diffraction_auto.d_spacings),
        indexed_spots=list(diffraction_auto.indexed_spots),
    )
    selected_rows = _build_peak_rows(
        peaks=list(manual_result["peaks"]),
        g_vectors=list(manual_result["g_vectors"]),
        d_by_peak=list(manual_result["d_by_peak"]),
        indexed_spots=list(manual_result["indexed_spots"]),
    )
    active_peak_source_label = (
        "Review set (active)"
        if selection_mode == MODE_AUTO_EDIT
        else "Vector synthesis set (active)"
    )
    align_target_rows = auto_rows if align_target_mode == "Auto (pipeline)" else selected_rows
    align_target_candidates = _build_align_candidates(align_target_rows, id_key="peak_id")
    align_signature = _json_signature(
        {
            "manual_rows": manual_spot_rows,
            "target_rows": align_target_candidates,
            "max_g_rel": float(align_max_g_rel),
            "max_d_rel": float(align_max_d_rel),
            "max_angle_deg": float(align_max_angle_deg),
            "weight_g": float(align_weight_g),
            "weight_d": float(align_weight_d),
            "weight_angle": float(align_weight_angle),
            "angle_step_deg": float(align_angle_step_deg),
        }
    )
    align_cache = st.session_state.get("b1b3_manual_align_cache")
    if isinstance(align_cache, dict) and align_cache.get("signature") == align_signature:
        align_result = align_cache["result"]
    else:
        align_result = _align_manual_spot_table(
            manual_rows=manual_spot_rows,
            target_rows=align_target_candidates,
            max_g_rel=float(align_max_g_rel),
            max_d_rel=float(align_max_d_rel),
            max_angle_deg=float(align_max_angle_deg),
            weight_g=float(align_weight_g),
            weight_d=float(align_weight_d),
            weight_angle=float(align_weight_angle),
            angle_step_deg=float(align_angle_step_deg),
        )
        st.session_state["b1b3_manual_align_cache"] = {
            "signature": align_signature,
            "result": align_result,
        }

    b5_source_mode = active_peak_source_label
    b5_source_diffraction = _diffraction_from_manual_result(manual_result)
    b5_signature = _json_signature(
        {
            "enabled": bool(b5_enabled),
            "source_mode": str(b5_source_mode),
            "diffraction": b5_source_diffraction.model_dump(mode="json"),
            "parameters": b5_params,
        }
    )
    b5_cache = st.session_state.get("b1b5_simulation_cache")
    b5_cached = isinstance(b5_cache, dict) and b5_cache.get("signature") == b5_signature
    b5_col1, b5_col2 = st.columns([0.22, 0.78])
    with b5_col1:
        b5_run_clicked = st.button(
            "Run / refresh B5",
            type="secondary",
            disabled=not b5_enabled,
            help="Runs simulation and experimental-vs-simulated phase matching.",
        )
    with b5_col2:
        if b5_enabled:
            st.caption(
                f"B5 source: `{b5_source_mode}` | engine: `{simulation_engine}` | "
                f"zone axis mode: `{zone_axis_mode}` | cached: `{b5_cached}`"
            )
        else:
            st.caption("B5 simulation is disabled in the sidebar.")
    should_auto_run_b5 = bool(b5_enabled) and bool(b5_auto_rerun)
    need_b5_run = bool(b5_enabled) and (
        b5_run_clicked or (should_auto_run_b5 and not b5_cached)
    )
    if need_b5_run:
        with st.spinner("Running B5 simulation / phase matching..."):
            simulation_result, b5_updated_diffraction, b5_warnings = (
                _run_b5_simulation_with_timeout(
                    b5_source_diffraction.model_copy(deep=True),
                    parameters=b5_params,
                    timeout_s=float(b5_timeout_s),
                )
            )
            simulation_result = _filter_b5_simulation_for_zone_axis(
                simulation_result,
                zone_axis_mode=str(zone_axis_mode),
            )
        st.session_state["b1b5_simulation_cache"] = {
            "signature": b5_signature,
            "simulation": simulation_result,
            "diffraction": b5_updated_diffraction,
            "warnings": b5_warnings,
        }
        b5_cache = st.session_state.get("b1b5_simulation_cache")
        b5_cached = True

    if (
        bool(b5_enabled)
        and isinstance(b5_cache, dict)
        and b5_cache.get("signature") == b5_signature
    ):
        b5_simulation = b5_cache.get("simulation")
        b5_diffraction_used = b5_cache.get("diffraction")
        b5_run_warnings = list(b5_cache.get("warnings", []))
    else:
        b5_simulation = None
        b5_diffraction_used = b5_source_diffraction
        b5_run_warnings = []

    tab_peak_sets, tab_b5, tab_manual_align, tab_b1, tab_diag = st.tabs(
        [
            "Peak sets / Phase candidates",
            "B5 simulation",
            "Manual table align",
            "B1 details",
            "Warnings / Provenance",
        ]
    )

    with tab_peak_sets:
        peak_rows = selected_rows
        phase_candidates = list(manual_result["phase_candidates"])
        if selection_mode == MODE_AUTO_EDIT:
            source_caption = (
                "Active reviewed set after canvas edits, merge, and intensity sorting."
            )
        else:
            source_caption = "Active vector-mode set synthesized from the two basis spots."

        source_cols = st.columns(3)
        source_cols[0].metric("Peak rows", str(len(peak_rows)))
        source_cols[1].metric("Phase candidates", str(len(phase_candidates)))
        source_cols[2].metric("Pipeline original", str(len(auto_rows)))
        st.caption(f"Source: `{active_peak_source_label}`")
        st.caption(source_caption)
        st.dataframe(peak_rows, use_container_width=True, height=300)
        st.subheader("Phase Candidates")
        st.dataframe(
            _build_phase_rows(phase_candidates),
            use_container_width=True,
        )

    with tab_b5:
        st.caption(
            "B5 debug panel for simulated diffraction, zone-axis candidates, and "
            "experimental-vs-simulated phase matching."
        )
        if not b5_enabled:
            st.info("Enable `Run B5 simulation step` in the sidebar to debug B5.")
        elif b5_simulation is None:
            st.info("Click `Run / refresh B5` to run B5 simulation for the current peak set.")
        else:
            phase_matching = _as_mapping(b5_simulation.provenance.get("phase_matching", {}))
            adapter = _as_mapping(b5_simulation.provenance.get("adapter", {}))
            metric_cols = st.columns(6)
            metric_cols[0].metric(
                "Engine",
                str(b5_simulation.provenance.get("method", b5_params.get("simulation_engine"))),
            )
            metric_cols[1].metric(
                "Patterns",
                str(len(b5_simulation.patterns)),
            )
            metric_cols[2].metric(
                "Matched phases",
                str(len(b5_simulation.matched_phase_candidates)),
            )
            metric_cols[3].metric(
                "Selected phase",
                str(phase_matching.get("selected_phase", "-")),
            )
            metric_cols[4].metric(
                "Selected zone axis",
                str(phase_matching.get("selected_zone_axis", "-")),
            )
            metric_cols[5].metric(
                "Adapter active",
                str(adapter.get("active", "")),
            )
            if bool(phase_matching.get("zone_axis_locked")):
                st.success(
                    "B5 zone axis is locked to the B4 phase-candidate zone axis."
                )
            if bool(phase_matching.get("candidate_ranking_locked")):
                st.success(
                    "B5 keeps the B4 phase-candidate ranking and only adds simulation "
                    "validation metrics."
                )
            elif str(b5_params.get("zone_axis_mode")) in {"auto_matched", "scan_all"}:
                st.warning(
                    "Debug zone-axis rescan is active; B5 may differ from the B4 "
                    "phase-candidate zone axis."
                )
            suggested_axes = []
            per_phase = phase_matching.get("per_phase", [])
            if isinstance(per_phase, list):
                for row in per_phase:
                    if not isinstance(row, dict):
                        continue
                    suggested_axis = row.get("b5_suggested_zone_axis")
                    if suggested_axis:
                        suggested_axes.append(suggested_axis)
            if suggested_axes:
                st.warning(
                    "B5 suggested different zone axes during matching, but the "
                    f"B4 axes were preserved: {suggested_axes}"
                )

            simulated_fig_b5 = _plot_b5_simulated_pattern(b5_simulation)
            if simulated_fig_b5 is not None:
                st.subheader("Simulated Diffraction Pattern")
                st.pyplot(simulated_fig_b5, clear_figure=True)

            overlay_fig_b5 = _plot_b5_simulation_overlay(
                b5_diffraction_used
                if isinstance(b5_diffraction_used, DiffractionResult)
                else b5_source_diffraction,
                b5_simulation,
                show_unmatched_simulated_spots=bool(show_unmatched_simulated_spots),
                show_match_residual_lines=bool(show_match_residual_lines),
            )
            if overlay_fig_b5 is not None:
                st.subheader("Experimental vs Simulated Overlay")
                st.pyplot(overlay_fig_b5, clear_figure=True)
            else:
                st.info("No B5 reciprocal-space overlay is available for this result.")

            st.subheader("B5 Phase Match Candidates")
            phase_rows = _build_simulation_phase_rows(b5_simulation)
            if phase_rows:
                st.dataframe(phase_rows, use_container_width=True, height=260)
            else:
                st.info("No B5 matched phase candidates recorded.")

            st.subheader("Simulated Spots")
            display_filter = _as_mapping(b5_simulation.provenance.get("display_filter", {}))
            if display_filter:
                st.caption(
                    "Displayed patterns: "
                    f"{display_filter.get('display_pattern_count')} of "
                    f"{display_filter.get('source_pattern_count')} | "
                    f"mode: `{display_filter.get('mode')}` | "
                    f"zone axis: `{display_filter.get('selected_zone_axis')}`"
                )
            spot_rows = _build_simulated_spot_rows(b5_simulation)
            if spot_rows:
                st.dataframe(spot_rows, use_container_width=True, height=260)
            else:
                st.info("No simulated spots recorded.")

            st.subheader("B5 Parameters")
            st.json(
                {
                    "source_mode": b5_source_mode,
                    "parameters": b5_params,
                    "display": {
                        "show_unmatched_simulated_spots": bool(
                            show_unmatched_simulated_spots
                        ),
                        "show_match_residual_lines": bool(show_match_residual_lines),
                    },
                    "phase_matching": phase_matching,
                    "adapter": adapter,
                    "warnings": b5_run_warnings,
                }
            )

    with tab_manual_align:
        st.caption(
            "Import manual spot table and auto-align with current peak table for error statistics."
        )
        st.write(f"Table source: `{manual_table_source}` | Align target: `{align_target_mode}`")
        if manual_spot_rows:
            st.subheader("Imported Manual Spot Table")
            st.dataframe(manual_spot_rows, use_container_width=True, height=220)
        else:
            st.info("Upload or paste manual spot table to enable alignment stats.")

        align_warnings = list(manual_table_parse_warnings)
        align_warnings.extend(list(align_result.get("warnings", [])))
        if align_warnings:
            for text in dict.fromkeys(align_warnings):
                st.warning(str(text))

        if not align_result.get("matches"):
            st.info("No valid alignment result yet.")
        else:
            summary = align_result.get("summary", {})
            metric_cols = st.columns(6)
            metric_cols[0].metric(
                "Matched",
                f"{align_result.get('matched_count', 0)} / {align_result.get('manual_count', 0)}",
            )
            metric_cols[1].metric(
                "Angle offset (deg)",
                f"{float(align_result.get('angle_offset_deg', 0.0)):.2f}",
            )
            metric_cols[2].metric("Mirror sign", str(int(align_result.get("sign", 1))))
            metric_cols[3].metric(
                "d mean err (%)",
                f"{float(summary.get('d_rel_mean_pct', 0.0)):.3f}",
            )
            metric_cols[4].metric(
                "g mean err (%)",
                f"{float(summary.get('g_rel_mean_pct', 0.0)):.3f}",
            )
            angle_mean = summary.get("angle_mean_deg")
            metric_cols[5].metric(
                "angle mean err (deg)",
                "-" if angle_mean is None else f"{float(angle_mean):.3f}",
            )
            st.subheader("Matched Spot Error Table")
            st.dataframe(
                list(align_result.get("matches", [])),
                use_container_width=True,
                height=300,
            )
            unmatched_manual = list(align_result.get("manual_unmatched", []))
            unmatched_target = list(align_result.get("target_unmatched", []))
            if unmatched_manual:
                st.subheader("Unmatched Manual Spots")
                st.dataframe(unmatched_manual, use_container_width=True, height=180)
            if unmatched_target:
                st.subheader("Unmatched Target Peaks")
                st.dataframe(unmatched_target, use_container_width=True, height=180)

    with tab_b1:
        cal = calibration
        cal_rows = [
            {
                "item": "pixel_size",
                "value": (
                    "-"
                    if cal.pixel_size is None
                    else f"{cal.pixel_size.value} {cal.pixel_size.unit or ''}"
                ),
                "uncertainty": None if cal.pixel_size is None else cal.pixel_size.uncertainty,
                "method": None if cal.pixel_size is None else cal.pixel_size.method,
            },
            {
                "item": "diffraction_scale",
                "value": (
                    "-"
                    if cal.diffraction_scale is None
                    else f"{cal.diffraction_scale.value} {cal.diffraction_scale.unit or ''}"
                ),
                "uncertainty": (
                    None if cal.diffraction_scale is None else cal.diffraction_scale.uncertainty
                ),
                "method": None if cal.diffraction_scale is None else cal.diffraction_scale.method,
            },
            {
                "item": "camera_length",
                "value": (
                    "-"
                    if cal.camera_length is None
                    else f"{cal.camera_length.value} {cal.camera_length.unit or ''}"
                ),
                "uncertainty": None if cal.camera_length is None else cal.camera_length.uncertainty,
                "method": None if cal.camera_length is None else cal.camera_length.method,
            },
            {
                "item": "voltage",
                "value": (
                    "-"
                    if cal.voltage is None
                    else f"{cal.voltage.value} {cal.voltage.unit or ''}"
                ),
                "uncertainty": None if cal.voltage is None else cal.voltage.uncertainty,
                "method": None if cal.voltage is None else cal.voltage.method,
            },
        ]
        st.dataframe(cal_rows, use_container_width=True, hide_index=True)
        transform = cal.transform if isinstance(cal.transform, dict) else {}
        source_rows = [
            {"source_item": "pixel_source", "value": transform.get("pixel_source")},
            {"source_item": "voltage_source", "value": transform.get("voltage_source")},
            {"source_item": "diffraction_source", "value": transform.get("diffraction_source")},
            {"source_item": "method", "value": transform.get("method")},
        ]
        st.subheader("Calibration Sources")
        st.dataframe(source_rows, use_container_width=True, hide_index=True)

        metadata_norm = {}
        if isinstance(dataset.metadata, dict):
            metadata_norm = dataset.metadata.get("normalized", {})
            if not isinstance(metadata_norm, dict):
                metadata_norm = {}
        st.subheader("Parsed Input Metadata")
        st.json(
            {
                "source_format": metadata_norm.get("source_format"),
                "pixel_size_nm": metadata_norm.get("pixel_size_nm"),
                "accelerating_voltage_kv": metadata_norm.get("accelerating_voltage_kv"),
                "camera_length_mm": metadata_norm.get("camera_length_mm"),
                "magnification": metadata_norm.get("magnification"),
                "metadata_completeness": metadata_norm.get("metadata_completeness"),
                "metadata_confidence": metadata_norm.get("metadata_confidence"),
            }
        )
        st.subheader("Quality Provenance")
        st.json(quality.provenance)

    with tab_diag:
        warnings = []
        warnings.extend(list(candidate_phase_warnings))
        warnings.extend(list(zone_axis_warnings))
        warnings.extend(list(manual_zone_axis_warnings))
        warnings.extend(list(rotation_axis_warnings))
        warnings.extend(list(rotation_angle_warnings))
        warnings.extend(list(manual_table_parse_warnings))
        warnings.extend(list(align_result.get("warnings", [])))
        warnings.extend(list(b5_run_warnings))
        warnings.extend(list(dataset.warnings))
        warnings.extend(list(quality.warnings))
        warnings.extend(list(calibration.warnings))
        warnings.extend(list(preprocessing.warnings))
        warnings.extend(list(diffraction_auto.warnings))
        warnings.extend(list(fft_warnings))
        warnings.extend(list(manual_result["warnings"]))
        if warnings:
            for warning in dict.fromkeys(warnings):
                if str(warning).startswith("BLOCKING:"):
                    st.error(str(warning))
                elif str(warning).startswith("WARNING:"):
                    st.warning(str(warning))
                else:
                    st.info(str(warning))
        else:
            st.success("No warnings.")

        st.subheader("Provenance")
        st.markdown("**Auto diffraction provenance**")
        st.json(diffraction_auto.provenance)
        st.markdown("**Selected-point provenance**")
        st.json(manual_result["provenance"])
        st.markdown("**B5 simulation provenance**")
        if isinstance(b5_simulation, SimulationResult):
            st.json(b5_simulation.provenance)
        else:
            st.info("B5 simulation has not been run for the current signature.")
        st.markdown("**Manual-table alignment summary**")
        st.json(
            {
                "table_source": manual_table_source,
                "align_target": align_target_mode,
                "manual_row_count": len(manual_spot_rows),
                "target_row_count": len(align_target_candidates),
                "matched_count": align_result.get("matched_count"),
                "angle_offset_deg": align_result.get("angle_offset_deg"),
                "mirror_sign": align_result.get("sign"),
                "summary": align_result.get("summary"),
            }
        )

        st.subheader("Run context")
        st.code(
            yaml.safe_dump(
                {
                    "input_path": str(input_path),
                    "output_dir": str(run_dir),
                    "signature_len": len(signature),
                    "diffraction": diffraction_params,
                    "simulation": b5_params,
                },
                sort_keys=False,
                allow_unicode=False,
            )
        )


if __name__ == "__main__":
    main()
