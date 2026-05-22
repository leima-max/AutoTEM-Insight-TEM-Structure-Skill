from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import streamlit as st
import yaml

from autotem.core import AnalysisReport, Measurement, read_json
from autotem.workflows.pipeline import TEMAnalysisSkill

_APP_TITLE = "AutoTEM B1-B3 Preview"
_WORKFLOW_TEMPLATE = "configs/hrtem_b1_b3.yaml"
_OUTPUT_ROOT_ENV = "AUTOTEM_UI_OUTPUT_ROOT"
_UI_BUILD_TAG = "b1_b3_preview_v2_2026-05-22"

PHASE_OPTIONS = ["Silicon", "Gold", "Graphite", "Titanium Dioxide"]
INDEXING_MODE_LABELS: dict[str, str] = {
    "auto": "auto (auto select)",
    "lattice_fit": "lattice_fit (vector lattice fit)",
    "baseline": "baseline (basic indexing)",
}

_FALLBACK_DEFAULTS: dict[str, Any] = {
    "quality": {
        "thresholds": {
            "snr_min": 6.0,
            "drift_px_max": 2.0,
            "overall_score_min": 0.35,
        }
    },
    "calibration": {
        "method": "reference",
        "voltage_kv": 200.0,
        "pixel_size_nm": 0.02,
        "known_d_spacing": 0.213,
        "known_d_spacing_unit": "nm",
        "known_d_uncertainty_nm": 0.002,
        "measured_radius_px": 46.5,
        "measured_radius_uncertainty_px": 0.25,
        "detector_pixel_size_um": 15.0,
    },
    "preprocessing": {
        "recipe": "min_max",
        "clip_percentile_low": 1.0,
        "clip_percentile_high": 99.0,
        "denoise_method": "wiener",
        "denoise_sigma": 1.0,
        "background_method": "gaussian",
        "background_sigma": 12.0,
    },
    "diffraction": {
        "auto_peak_tuning": True,
        "peak_detector": "log",
        "peak_adapter_engine": "none",
        "peak_threshold_strategy": "auto",
        "peak_threshold_value": 0.15,
        "top_k_peaks": 24,
        "min_peak_distance": 3,
        "center_exclusion_radius": 3.0,
        "detector_sigma": 1.2,
        "dog_sigma_ratio": 1.8,
        "fft_center_mode": "auto",
        "direct_beam_refine_window": 3,
        "indexing_mode": "auto",
        "phase_top_n": 3,
        "candidate_phases": ["Silicon", "Gold", "Graphite"],
    },
    "reporting": {
        "diffraction_gates": {
            "min_peak_count": 4,
            "min_uncertainty_coverage": 0.8,
            "max_mean_relative_uncertainty": 0.35,
            "min_indexed_nonzero_fraction": 0.2,
        }
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_name(raw_name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name.strip())
    return token or "input.npy"


def _default_output_root() -> Path:
    from_env = os.environ.get(_OUTPUT_ROOT_ENV)
    if from_env:
        return Path(from_env).expanduser().resolve()
    return _repo_root() / "outputs" / "b1_b3_preview"


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


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


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _num(value: Any, fallback: float) -> float:
    if isinstance(value, bool) or value is None:
        return float(fallback)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return float(fallback)
    return float(fallback)


def _measurement_text(measurement: Measurement | None) -> str:
    if measurement is None:
        return "-"
    value = measurement.value
    unc = measurement.uncertainty
    unit = measurement.unit or ""
    if isinstance(value, (int, float)):
        if isinstance(unc, (int, float)):
            return f"{float(value):.6g} +/- {float(unc):.3g} {unit}".strip()
        return f"{float(value):.6g} {unit}".strip()
    return str(value)


def _build_demo_input(run_dir: Path) -> Path:
    rng = np.random.default_rng(20260522)
    h, w = 256, 256
    image = rng.normal(loc=0.02, scale=0.008, size=(h, w)).astype(np.float32)
    cy, cx = h // 2, w // 2
    peaks = [(0, 34), (0, -34), (26, 20), (-26, -20), (18, -14), (-18, 14)]
    for dy, dx in peaks:
        yy = cy + dy
        xx = cx + dx
        if 2 <= yy < h - 2 and 2 <= xx < w - 2:
            image[yy - 1 : yy + 2, xx - 1 : xx + 2] += 1.1
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    out_path = input_dir / "b1_b3_demo.npy"
    np.save(out_path, image)
    return out_path


def _save_uploaded(uploaded_file, run_dir: Path) -> Path:
    input_dir = run_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / _safe_name(uploaded_file.name)
    input_path.write_bytes(uploaded_file.getbuffer())
    return input_path


def _build_workflow_payload(values: dict[str, Any]) -> dict[str, Any]:
    preprocessing_payload: dict[str, Any] = {
        "recipe": str(values["recipe"]),
        "denoise_method": str(values["denoise_method"]),
        "denoise_sigma": float(values["denoise_sigma"]),
        "background_method": str(values["background_method"]),
        "background_sigma": float(values["background_sigma"]),
    }
    if values["use_clip"]:
        preprocessing_payload["clip_percentile_low"] = float(values["clip_low"])
        preprocessing_payload["clip_percentile_high"] = float(values["clip_high"])

    threshold_strategy = str(values["peak_threshold_strategy"])
    diffraction_payload: dict[str, Any] = {
        "auto_peak_tuning": bool(values["auto_peak_tuning"]),
        "peak_detector": str(values["peak_detector"]),
        "peak_adapter_engine": str(values["peak_adapter_engine"]),
        "peak_threshold_strategy": threshold_strategy,
        "top_k_peaks": int(values["top_k_peaks"]),
        "min_peak_distance": int(values["min_peak_distance"]),
        "center_exclusion_radius": float(values["center_exclusion_radius"]),
        "detector_sigma": float(values["detector_sigma"]),
        "dog_sigma_ratio": float(values["dog_sigma_ratio"]),
        "fft_center_mode": str(values["fft_center_mode"]),
        "direct_beam_refine_window": int(values["direct_beam_refine_window"]),
        "indexing_mode": str(values["indexing_mode"]),
        "phase_top_n": int(values["phase_top_n"]),
        "candidate_phases": list(values["candidate_phases"]),
    }
    if threshold_strategy != "auto":
        diffraction_payload["peak_threshold_value"] = float(values["peak_threshold_value"])

    return {
        "workflow": {
            "name": "b1_b3_preview_ui",
            "description": "Interactive B1-B3 preview workflow from Streamlit UI.",
            "steps": [
                "load",
                "assess_quality",
                "calibrate",
                "preprocess",
                "analyze",
                "report",
            ],
            "parameters": {
                "quality": {
                    "thresholds": {
                        "snr_min": float(values["snr_min"]),
                        "drift_px_max": float(values["drift_px_max"]),
                        "overall_score_min": float(values["overall_score_min"]),
                    }
                },
                "calibration": {
                    "method": str(values["calibration_method"]),
                    "voltage_kv": float(values["voltage_kv"]),
                    "pixel_size_nm": float(values["pixel_size_nm"]),
                    "known_d_spacing": float(values["known_d_spacing"]),
                    "known_d_spacing_unit": "nm",
                    "known_d_uncertainty_nm": float(values["known_d_uncertainty_nm"]),
                    "measured_radius_px": float(values["measured_radius_px"]),
                    "measured_radius_uncertainty_px": float(
                        values["measured_radius_uncertainty_px"]
                    ),
                    "detector_pixel_size_um": float(values["detector_pixel_size_um"]),
                },
                "preprocessing": preprocessing_payload,
                "diffraction": diffraction_payload,
                "reporting": {
                    "diffraction_gates": {
                        "min_peak_count": int(values["gate_min_peak_count"]),
                        "min_uncertainty_coverage": float(values["gate_min_unc_coverage"]),
                        "max_mean_relative_uncertainty": float(
                            values["gate_max_mean_rel_unc"]
                        ),
                        "min_indexed_nonzero_fraction": float(
                            values["gate_min_indexed_fraction"]
                        ),
                    }
                },
            },
        },
        "outputs": {
            "formats": ["json", "csv", "html"],
            "save_intermediates": True,
        },
    }


def _download_button_for_path(path: Path, *, label: str, mime: str) -> None:
    if not path.exists() or not path.is_file():
        return
    st.download_button(
        label=label,
        data=path.read_bytes(),
        file_name=path.name,
        mime=mime,
        use_container_width=True,
    )


def _display_report(report: AnalysisReport, output_dir: Path) -> None:
    tabs = st.tabs(["Summary", "Tables", "Provenance", "Figures", "Warnings"])
    quality = report.quality
    calibration = report.calibration
    diffraction = report.diffraction

    with tabs[0]:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Overall Quality",
            (
                f"{float(quality.overall_score):.3f}"
                if quality and quality.overall_score is not None
                else "-"
            ),
        )
        col2.metric("Diffraction Peaks", str(len(diffraction.peaks)) if diffraction else "0")
        col3.metric("d-spacing Count", str(len(diffraction.d_spacings)) if diffraction else "0")
        col4.metric(
            "Phase Candidates",
            str(len(diffraction.phase_candidates)) if diffraction else "0",
        )

        st.markdown("**Calibration (B1)**")
        cal_rows = [
            {
                "item": "pixel_size",
                "value": _measurement_text(calibration.pixel_size if calibration else None),
            },
            {
                "item": "diffraction_scale",
                "value": _measurement_text(
                    calibration.diffraction_scale if calibration else None
                ),
            },
            {
                "item": "camera_length",
                "value": _measurement_text(calibration.camera_length if calibration else None),
            },
            {
                "item": "voltage",
                "value": _measurement_text(calibration.voltage if calibration else None),
            },
        ]
        st.dataframe(cal_rows, use_container_width=True, hide_index=True)

    with tabs[1]:
        if diffraction is None:
            st.warning("DiffractionResult is missing.")
        else:
            if diffraction.peaks:
                peak_rows = [
                    {
                        "index": idx,
                        "x": round(float(peak.x), 4),
                        "y": round(float(peak.y), 4),
                        "intensity": (
                            None if peak.intensity is None else round(float(peak.intensity), 6)
                        ),
                        "confidence": (
                            None
                            if peak.confidence is None
                            else round(float(peak.confidence), 4)
                        ),
                    }
                    for idx, peak in enumerate(diffraction.peaks)
                ]
                st.markdown("**Peaks**")
                st.dataframe(peak_rows, use_container_width=True, height=220)
            if diffraction.d_spacings:
                d_rows = [
                    {
                        "index": idx,
                        "value": item.value,
                        "unit": item.unit,
                        "uncertainty": item.uncertainty,
                        "method": item.method,
                    }
                    for idx, item in enumerate(diffraction.d_spacings)
                ]
                st.markdown("**d-spacings**")
                st.dataframe(d_rows, use_container_width=True, height=220)
            if diffraction.phase_candidates:
                phase_rows = [
                    {
                        "name": item.name,
                        "score": item.score,
                        "matched_spot_count": item.matched_spot_count,
                        "source": item.source,
                    }
                    for item in diffraction.phase_candidates
                ]
                st.markdown("**Phase candidates**")
                st.dataframe(phase_rows, use_container_width=True, height=180)

    with tabs[2]:
        st.markdown("**Report Provenance**")
        st.json(report.provenance)
        if diffraction is not None:
            st.markdown("**Diffraction Provenance**")
            st.json(diffraction.provenance)

    with tabs[3]:
        fig_dir = output_dir / "figures"
        for name in ("input_image.png", "fft_magnitude.png", "fft_peaks_overlay.png"):
            path = fig_dir / name
            if path.exists():
                st.image(str(path), caption=name, use_container_width=True)

    with tabs[4]:
        all_warnings = list(report.warnings)
        if not all_warnings:
            st.success("No warnings.")
        else:
            for warning in all_warnings:
                if warning.startswith("BLOCKING:"):
                    st.error(warning)
                elif warning.startswith("WARNING:"):
                    st.warning(warning)
                else:
                    st.info(warning)

    st.markdown("**Artifacts**")
    artifact_col1, artifact_col2, artifact_col3 = st.columns(3)
    report_json = output_dir / "analysis_report.json"
    report_html = output_dir / "report.html"
    workflow_yaml = output_dir / "workflow_used.yaml"
    with artifact_col1:
        _download_button_for_path(
            report_json,
            label="Download analysis_report.json",
            mime="application/json",
        )
    with artifact_col2:
        _download_button_for_path(
            report_html,
            label="Download report.html",
            mime="text/html",
        )
    with artifact_col3:
        _download_button_for_path(
            workflow_yaml,
            label="Download workflow_used.yaml",
            mime="application/x-yaml",
        )

    st.code(f"output_dir: {output_dir}")


def main() -> None:
    st.set_page_config(page_title=_APP_TITLE, layout="wide")
    st.title(_APP_TITLE)
    st.caption(
        "Upload/generate sample input, run B1-B3, and review "
        "results in Summary/Tables/Provenance/Figures tabs."
    )
    st.caption(f"UI build: `{_UI_BUILD_TAG}`")

    defaults = _deep_merge(_FALLBACK_DEFAULTS, _read_template_defaults())
    quality_defaults = _as_mapping(_as_mapping(defaults.get("quality")).get("thresholds"))
    calib_defaults = _as_mapping(defaults.get("calibration"))
    prep_defaults = _as_mapping(defaults.get("preprocessing"))
    diff_defaults = _as_mapping(defaults.get("diffraction"))
    gate_defaults = _as_mapping(_as_mapping(defaults.get("reporting")).get("diffraction_gates"))

    st.sidebar.header("Run Setup")
    input_mode = st.sidebar.radio(
        "Input source",
        options=["Upload file", "Built-in demo"],
        index=0,
    )
    uploaded = None
    if input_mode == "Upload file":
        uploaded = st.sidebar.file_uploader(
            "Input TEM image/data",
            type=["npy", "tif", "tiff", "png", "jpg", "jpeg", "dm", "dm3", "dm4"],
        )
    else:
        st.sidebar.info("Will generate a synthetic diffraction demo input.")

    output_root_default = _default_output_root()
    output_root_raw = st.sidebar.text_input("Output root", value=str(output_root_default))
    run_name = st.sidebar.text_input(
        "Run folder name",
        value=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    use_template_workflow = st.sidebar.checkbox(
        f"Use template workflow ({_WORKFLOW_TEMPLATE})",
        value=True,
    )

    values: dict[str, Any] = {}
    if not use_template_workflow:
        with st.expander("B1 Quality / Calibration", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                values["snr_min"] = st.number_input(
                    "quality.snr_min",
                    min_value=0.0,
                    value=_num(quality_defaults.get("snr_min"), 6.0),
                    step=0.5,
                )
                values["drift_px_max"] = st.number_input(
                    "quality.drift_px_max",
                    min_value=0.0,
                    value=_num(quality_defaults.get("drift_px_max"), 2.0),
                    step=0.2,
                )
                values["overall_score_min"] = st.number_input(
                    "quality.overall_score_min",
                    min_value=0.0,
                    max_value=1.0,
                    value=_num(quality_defaults.get("overall_score_min"), 0.35),
                    step=0.05,
                )
                values["calibration_method"] = st.selectbox(
                    "calibration.method",
                    options=["reference", "manual", "auto", "metadata"],
                    index=0,
                )
                values["known_d_spacing"] = st.number_input(
                    "calibration.known_d_spacing (nm)",
                    min_value=0.0001,
                    value=_num(calib_defaults.get("known_d_spacing"), 0.213),
                    step=0.001,
                    format="%.6f",
                )
                values["measured_radius_px"] = st.number_input(
                    "calibration.measured_radius_px",
                    min_value=0.1,
                    value=_num(calib_defaults.get("measured_radius_px"), 46.5),
                    step=0.1,
                )
            with col2:
                values["known_d_uncertainty_nm"] = st.number_input(
                    "calibration.known_d_uncertainty_nm",
                    min_value=0.0,
                    value=_num(calib_defaults.get("known_d_uncertainty_nm"), 0.002),
                    step=0.001,
                )
                values["measured_radius_uncertainty_px"] = st.number_input(
                    "calibration.measured_radius_uncertainty_px",
                    min_value=0.0,
                    value=_num(calib_defaults.get("measured_radius_uncertainty_px"), 0.25),
                    step=0.05,
                )
                values["voltage_kv"] = st.number_input(
                    "calibration.voltage_kv",
                    min_value=1.0,
                    value=_num(calib_defaults.get("voltage_kv"), 200.0),
                    step=10.0,
                )
                values["pixel_size_nm"] = st.number_input(
                    "calibration.pixel_size_nm",
                    min_value=0.0001,
                    value=_num(calib_defaults.get("pixel_size_nm"), 0.02),
                    step=0.001,
                    format="%.6f",
                )
                values["detector_pixel_size_um"] = st.number_input(
                    "calibration.detector_pixel_size_um",
                    min_value=0.1,
                    value=_num(calib_defaults.get("detector_pixel_size_um"), 15.0),
                    step=0.5,
                )

        with st.expander("B2 Preprocessing", expanded=False):
            values["recipe"] = st.selectbox(
                "preprocessing.recipe",
                options=["auto", "min_max"],
                index=1 if str(prep_defaults.get("recipe", "min_max")) == "min_max" else 0,
            )
            values["use_clip"] = st.checkbox("Enable percentile clipping", value=True)
            clip_col1, clip_col2 = st.columns(2)
            with clip_col1:
                values["clip_low"] = st.number_input(
                    "preprocessing.clip_percentile_low",
                    min_value=0.0,
                    max_value=99.9,
                    value=_num(prep_defaults.get("clip_percentile_low"), 1.0),
                    step=0.5,
                    disabled=not values["use_clip"],
                )
            with clip_col2:
                values["clip_high"] = st.number_input(
                    "preprocessing.clip_percentile_high",
                    min_value=0.1,
                    max_value=100.0,
                    value=_num(prep_defaults.get("clip_percentile_high"), 99.0),
                    step=0.5,
                    disabled=not values["use_clip"],
                )
            denoise_options = ["auto", "none", "gaussian", "median", "wiener", "nlm"]
            bg_options = ["none", "gaussian", "rolling_ball"]
            default_denoise = str(prep_defaults.get("denoise_method", "wiener"))
            default_bg = str(prep_defaults.get("background_method", "gaussian"))
            values["denoise_method"] = st.selectbox(
                "preprocessing.denoise_method",
                options=denoise_options,
                index=(
                    denoise_options.index(default_denoise)
                    if default_denoise in denoise_options
                    else 0
                ),
            )
            values["denoise_sigma"] = st.number_input(
                "preprocessing.denoise_sigma",
                min_value=0.0,
                value=_num(prep_defaults.get("denoise_sigma"), 1.0),
                step=0.1,
            )
            values["background_method"] = st.selectbox(
                "preprocessing.background_method",
                options=bg_options,
                index=bg_options.index(default_bg) if default_bg in bg_options else 0,
            )
            values["background_sigma"] = st.number_input(
                "preprocessing.background_sigma",
                min_value=0.0,
                value=_num(prep_defaults.get("background_sigma"), 12.0),
                step=0.5,
            )

        with st.expander("B3 Diffraction", expanded=False):
            detector_options = ["local_max", "log", "dog"]
            strategy_options = ["auto", "quantile", "std", "relative_max", "mad"]
            fft_center_options = ["auto", "geometric"]
            indexing_options = ["auto", "baseline", "lattice_fit"]
            adapter_options = ["none", "pyxem", "py4dstem"]
            default_detector = str(diff_defaults.get("peak_detector", "log"))
            default_strategy = str(diff_defaults.get("peak_threshold_strategy", "auto"))
            default_fft = str(diff_defaults.get("fft_center_mode", "auto"))
            default_indexing = str(diff_defaults.get("indexing_mode", "auto"))
            default_adapter = str(diff_defaults.get("peak_adapter_engine", "none"))
            values["auto_peak_tuning"] = st.checkbox(
                "diffraction.auto_peak_tuning",
                value=bool(diff_defaults.get("auto_peak_tuning", True)),
            )
            values["peak_detector"] = st.selectbox(
                "diffraction.peak_detector",
                options=detector_options,
                index=(
                    detector_options.index(default_detector)
                    if default_detector in detector_options
                    else 0
                ),
            )
            values["peak_adapter_engine"] = st.selectbox(
                "diffraction.peak_adapter_engine",
                options=adapter_options,
                index=(
                    adapter_options.index(default_adapter)
                    if default_adapter in adapter_options
                    else 0
                ),
            )
            values["peak_threshold_strategy"] = st.selectbox(
                "diffraction.peak_threshold_strategy",
                options=strategy_options,
                index=(
                    strategy_options.index(default_strategy)
                    if default_strategy in strategy_options
                    else 0
                ),
            )
            values["peak_threshold_value"] = st.number_input(
                "diffraction.peak_threshold_value",
                min_value=0.0,
                value=_num(diff_defaults.get("peak_threshold_value"), 0.15),
                step=0.01,
                disabled=values["peak_threshold_strategy"] == "auto",
            )
            diff_col1, diff_col2 = st.columns(2)
            with diff_col1:
                values["top_k_peaks"] = st.number_input(
                    "diffraction.top_k_peaks",
                    min_value=1,
                    value=int(_num(diff_defaults.get("top_k_peaks"), 24)),
                    step=1,
                )
                values["min_peak_distance"] = st.number_input(
                    "diffraction.min_peak_distance",
                    min_value=1,
                    value=int(_num(diff_defaults.get("min_peak_distance"), 3)),
                    step=1,
                )
                values["center_exclusion_radius"] = st.number_input(
                    "diffraction.center_exclusion_radius",
                    min_value=0.0,
                    value=_num(diff_defaults.get("center_exclusion_radius"), 3.0),
                    step=0.5,
                )
                values["detector_sigma"] = st.number_input(
                    "diffraction.detector_sigma",
                    min_value=0.2,
                    value=_num(diff_defaults.get("detector_sigma"), 1.2),
                    step=0.1,
                )
                values["dog_sigma_ratio"] = st.number_input(
                    "diffraction.dog_sigma_ratio",
                    min_value=1.2,
                    value=_num(diff_defaults.get("dog_sigma_ratio"), 1.8),
                    step=0.1,
                )
            with diff_col2:
                values["fft_center_mode"] = st.selectbox(
                    "diffraction.fft_center_mode",
                    options=fft_center_options,
                    index=(
                        fft_center_options.index(default_fft)
                        if default_fft in fft_center_options
                        else 0
                    ),
                )
                values["direct_beam_refine_window"] = st.number_input(
                    "diffraction.direct_beam_refine_window",
                    min_value=1,
                    value=int(_num(diff_defaults.get("direct_beam_refine_window"), 3)),
                    step=1,
                )
                values["indexing_mode"] = st.selectbox(
                    "diffraction.indexing_mode",
                    options=indexing_options,
                    index=(
                        indexing_options.index(default_indexing)
                        if default_indexing in indexing_options
                        else 0
                    ),
                    format_func=lambda key: INDEXING_MODE_LABELS.get(str(key), str(key)),
                )
                values["phase_top_n"] = st.number_input(
                    "diffraction.phase_top_n",
                    min_value=1,
                    value=int(_num(diff_defaults.get("phase_top_n"), 3)),
                    step=1,
                )
            phase_default = diff_defaults.get("candidate_phases", PHASE_OPTIONS[:3])
            phase_default_list = [item for item in phase_default if item in PHASE_OPTIONS]
            values["candidate_phases"] = st.multiselect(
                "diffraction.candidate_phases",
                options=PHASE_OPTIONS,
                default=phase_default_list if phase_default_list else PHASE_OPTIONS[:3],
            )

        with st.expander("Report Diffraction Gates", expanded=False):
            gate_col1, gate_col2 = st.columns(2)
            with gate_col1:
                values["gate_min_peak_count"] = st.number_input(
                    "reporting.diffraction_gates.min_peak_count",
                    min_value=1,
                    value=int(_num(gate_defaults.get("min_peak_count"), 4)),
                    step=1,
                )
                values["gate_min_unc_coverage"] = st.number_input(
                    "reporting.diffraction_gates.min_uncertainty_coverage",
                    min_value=0.0,
                    max_value=1.0,
                    value=_num(gate_defaults.get("min_uncertainty_coverage"), 0.8),
                    step=0.05,
                )
            with gate_col2:
                values["gate_max_mean_rel_unc"] = st.number_input(
                    "reporting.diffraction_gates.max_mean_relative_uncertainty",
                    min_value=0.0,
                    value=_num(gate_defaults.get("max_mean_relative_uncertainty"), 0.35),
                    step=0.05,
                )
                values["gate_min_indexed_fraction"] = st.number_input(
                    "reporting.diffraction_gates.min_indexed_nonzero_fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=_num(gate_defaults.get("min_indexed_nonzero_fraction"), 0.2),
                    step=0.05,
                )

    run_clicked = st.button("Run B1-B3 Preview", type="primary", use_container_width=True)
    if not run_clicked:
        return

    output_root = Path(output_root_raw).expanduser()
    run_dir = output_root / _safe_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    if input_mode == "Upload file":
        if uploaded is None:
            st.error("Please upload one input file first.")
            return
        input_path = _save_uploaded(uploaded, run_dir)
    else:
        input_path = _build_demo_input(run_dir)

    if use_template_workflow:
        workflow_path = _repo_root() / _WORKFLOW_TEMPLATE
        if not workflow_path.exists():
            st.error(f"Template workflow not found: {workflow_path}")
            return
    else:
        workflow_payload = _build_workflow_payload(values)
        workflow_path = run_dir / "workflow_b1_b3_ui.yaml"
        workflow_path.write_text(
            yaml.safe_dump(workflow_payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )

    run_t0 = perf_counter()
    with st.spinner("Running B1-B3 pipeline..."):
        try:
            result = TEMAnalysisSkill().run(
                input_path=input_path,
                workflow_path=workflow_path,
                output_dir=run_dir,
            )
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            return
    run_elapsed = perf_counter() - run_t0

    st.success(f"Run completed in {run_elapsed:.2f}s.")
    st.json(result)
    report_path = Path(result["report_path"])
    if not report_path.exists():
        st.error(f"analysis_report.json missing: {report_path}")
        return
    report = read_json(report_path, AnalysisReport)
    _display_report(report, run_dir)

    st.markdown("**Run Context**")
    context_payload = {
        "input_path": str(input_path),
        "workflow_path": str(workflow_path),
        "output_dir": str(run_dir),
        "elapsed_s": round(run_elapsed, 3),
    }
    st.code(yaml.safe_dump(context_payload, sort_keys=False, allow_unicode=False))

    run_summary_path = run_dir / "run_summary.json"
    if run_summary_path.exists():
        _download_button_for_path(
            run_summary_path,
            label="Download run_summary.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()
