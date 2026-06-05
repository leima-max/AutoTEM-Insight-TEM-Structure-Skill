from __future__ import annotations

# ruff: noqa: I001

import importlib.util
import sys
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import pytest

if importlib.util.find_spec("streamlit") is None:
    streamlit_module = ModuleType("streamlit")
    streamlit_module.__path__ = []
    streamlit_module.session_state = {}
    elements_module = ModuleType("streamlit.elements")
    elements_module.__path__ = []
    image_module = ModuleType("streamlit.elements.image")
    image_module.image_to_url = lambda *args, **kwargs: ""
    streamlit_module.elements = elements_module
    elements_module.image = image_module
    sys.modules.setdefault("streamlit", streamlit_module)
    sys.modules.setdefault("streamlit.elements", elements_module)
    sys.modules.setdefault("streamlit.elements.image", image_module)

if importlib.util.find_spec("streamlit_drawable_canvas") is None:
    canvas_module = ModuleType("streamlit_drawable_canvas")
    canvas_module.st_canvas = lambda *args, **kwargs: None
    sys.modules.setdefault("streamlit_drawable_canvas", canvas_module)

import apps.b2_diffraction_ui as b2_ui

from apps.b2_diffraction_ui import (
    _FALLBACK_DEFAULTS,
    _build_simulated_spot_rows,
    _c2_gpa_blocking_warnings,
    _c2_gpa_pixel_size_nm_from_calibration,
    _compute_gpa_fields_compat,
    _default_c2_gpa_mask_sigma_px,
    _extract_c2_gpa_reference_roi_from_canvas,
    _normalize_c2_gpa_reference_region,
    _plot_c2_gpa_maps,
    _resolve_c2_gpa_input_image,
    _select_c2_gpa_auto_basis,
    _select_c2_gpa_manual_basis,
    _to_c2_gpa_reference_roi_canvas_image,
    _filter_b5_simulation_for_zone_axis,
    _plot_b5_simulated_pattern,
    _plot_b5_simulation_overlay,
    _save_uploaded,
    SUPPORTED_SIMULATION_ENGINES,
)
from autotem.core import (
    CalibrationResult,
    DiffractionResult,
    GVector,
    Measurement,
    PhaseCandidate,
    SimulatedPattern,
    SimulatedSpot,
    SimulationResult,
)


def _spot(gx: float, gy: float) -> SimulatedSpot:
    return SimulatedSpot(
        gx=Measurement(value=gx, unit="1/nm"),
        gy=Measurement(value=gy, unit="1/nm"),
        d_spacing=Measurement(value=1.0 / max((gx**2 + gy**2) ** 0.5, 1e-9), unit="nm"),
        h=int(round(gx)),
        k=int(round(gy)),
        l=0,
    )


def _gvector(gx: float, gy: float, peak_index: int) -> GVector:
    return GVector(
        gx=Measurement(value=gx, unit="1/nm"),
        gy=Measurement(value=gy, unit="1/nm"),
        source_peak_index=peak_index,
    )


def test_save_uploaded_reuses_existing_same_size_file(tmp_path) -> None:
    class Upload:
        def __init__(self, name: str, payload: bytes) -> None:
            self.name = name
            self._payload = payload

        def getbuffer(self) -> memoryview:
            return memoryview(self._payload)

    saved = _save_uploaded(Upload("1.0017.dm4", b"original"), tmp_path)
    saved_again = _save_uploaded(Upload("1.0017.dm4", b"changed!"), tmp_path)

    assert saved_again == saved
    assert saved.read_bytes() == b"original"


def test_compute_gpa_fields_compat_drops_new_keyword_for_stale_module(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def old_compute(image, *, g_vectors=None):
        calls["shape"] = tuple(image.shape)
        calls["g_vectors"] = g_vectors
        return "ok"

    monkeypatch.setattr(b2_ui.gpa_module, "compute_gpa_fields", old_compute)
    monkeypatch.setattr(b2_ui.importlib, "reload", lambda module: module)

    result = _compute_gpa_fields_compat(
        np.zeros((3, 4), dtype=float),
        g_vectors=[(1.0, 0.0), (0.0, 1.0)],
        apply_quality_mask=False,
    )

    assert result == "ok"
    assert calls == {"shape": (3, 4), "g_vectors": [(1.0, 0.0), (0.0, 1.0)]}


def test_b5_ui_auto_mode_keeps_selected_zone_axis_pattern_only() -> None:
    simulation = SimulationResult(
        patterns=[
            SimulatedPattern(
                kind="diffraction",
                structure_id="mp-557719",
                zone_axis=(0, 0, 1),
                spots=[_spot(1.0, 0.0)],
            ),
            SimulatedPattern(
                kind="diffraction",
                structure_id="mp-557719",
                zone_axis=(1, 1, 0),
                spots=[_spot(0.0, 2.0)],
                simulation_parameters={
                    "phase_match": {"matched_spot_count": 3, "matched_pairs": []}
                },
            ),
        ],
        matched_phase_candidates=[
            PhaseCandidate(name="mp-557719", zone_axis=(1, 1, 0), score=0.9)
        ],
        provenance={
            "phase_matching": {
                "active": True,
                "selected_phase": "mp-557719",
                "selected_zone_axis": [1, 1, 0],
                "per_phase": [
                    {
                        "phase_name": "mp-557719",
                        "best_pattern_index": 1,
                        "zone_axis": [1, 1, 0],
                    }
                ],
            }
        },
    )

    filtered = _filter_b5_simulation_for_zone_axis(
        simulation,
        zone_axis_mode="auto_matched",
    )

    assert len(filtered.patterns) == 1
    assert filtered.patterns[0].zone_axis == (1, 1, 0)
    assert (
        filtered.patterns[0].simulation_parameters["display_source_pattern_index"] == 1
    )
    assert filtered.provenance["display_filter"]["source_pattern_count"] == 2
    rows = _build_simulated_spot_rows(filtered)
    assert rows[0]["source_pattern_index"] == 1


def test_b5_ui_scan_all_mode_keeps_all_zone_axis_patterns() -> None:
    simulation = SimulationResult(
        patterns=[
            SimulatedPattern(
                kind="diffraction",
                structure_id="phase",
                zone_axis=(0, 0, 1),
                spots=[_spot(1.0, 0.0)],
            ),
            SimulatedPattern(
                kind="diffraction",
                structure_id="phase",
                zone_axis=(1, 1, 0),
                spots=[_spot(0.0, 2.0)],
            ),
        ]
    )

    filtered = _filter_b5_simulation_for_zone_axis(
        simulation,
        zone_axis_mode="scan_all",
    )

    assert len(filtered.patterns) == 2
    assert "display_filter" not in filtered.provenance


def test_b5_ui_simulated_pattern_uses_selected_phase_title() -> None:
    simulation = SimulationResult(
        patterns=[
            SimulatedPattern(
                kind="diffraction",
                structure_id="Silicon",
                zone_axis=(0, 0, 1),
                spots=[_spot(2.0, 0.0), _spot(0.0, 2.0)],
            )
        ],
        provenance={"phase_matching": {"selected_phase": "Silicon"}},
    )

    fig = _plot_b5_simulated_pattern(simulation)

    assert fig is not None
    assert fig.axes[0].get_title() == "Simulated Diffraction: Silicon [0 0 1]"
    assert fig.axes[0].get_xlabel() == "gx (1/nm)"
    assert fig.axes[0].get_ylabel() == "gy (1/nm)"
    plt.close(fig)


def test_b5_ui_overlay_uses_backup_style_titles_and_data() -> None:
    diffraction = DiffractionResult(
        g_vectors=[
            _gvector(2.0, 0.0, 0),
            _gvector(0.0, 2.0, 1),
        ]
    )
    simulation = SimulationResult(
        patterns=[
            SimulatedPattern(
                kind="diffraction",
                structure_id="Silicon",
                zone_axis=(0, 0, 1),
                spots=[_spot(2.0, 0.0), _spot(0.0, 2.0)],
                simulation_parameters={
                    "phase_match": {
                        "rotation_deg": 0.0,
                        "matched_pairs": [
                            {"measured_g_index": 0, "simulated_spot_index": 0},
                            {"measured_g_index": 1, "simulated_spot_index": 1},
                        ],
                    }
                },
            )
        ],
    )

    fig = _plot_b5_simulation_overlay(diffraction, simulation)

    assert fig is not None
    assert fig.axes[0].get_title() == "Experimental vs Simulated Overlay"
    assert fig.axes[0].get_xlabel() == "gx (1/nm)"
    assert fig.axes[0].get_ylabel() == "gy (1/nm)"
    assert len(fig.axes[0].collections) >= 2
    assert len(fig.axes[0].lines) == 2
    plt.close(fig)


def test_b5_ui_overlay_can_show_match_residual_lines_when_enabled() -> None:
    diffraction = DiffractionResult(
        g_vectors=[
            _gvector(2.0, 0.0, 0),
            _gvector(0.0, 2.0, 1),
        ]
    )
    simulation = SimulationResult(
        patterns=[
            SimulatedPattern(
                kind="diffraction",
                structure_id="Silicon",
                zone_axis=(0, 0, 1),
                spots=[_spot(2.1, 0.1), _spot(0.1, 2.1)],
                simulation_parameters={
                    "phase_match": {
                        "rotation_deg": 0.0,
                        "matched_pairs": [
                            {"measured_g_index": 0, "simulated_spot_index": 0},
                            {"measured_g_index": 1, "simulated_spot_index": 1},
                        ],
                    }
                },
            )
        ],
    )

    fig = _plot_b5_simulation_overlay(
        diffraction,
        simulation,
        show_match_residual_lines=True,
    )

    assert fig is not None
    assert len(fig.axes[0].lines) > 2
    plt.close(fig)


def test_b5_defaults_lock_to_b4_zone_axis() -> None:
    assert _FALLBACK_DEFAULTS["preprocessing"]["enabled"] is False
    assert _FALLBACK_DEFAULTS["simulation"]["zone_axis_mode"] == "inherit_b4_locked"
    assert _FALLBACK_DEFAULTS["simulation"]["simulation_engine"] == "diffsims"
    assert _FALLBACK_DEFAULTS["simulation"]["show_match_residual_lines"] is False
    assert SUPPORTED_SIMULATION_ENGINES == ["diffsims"]


def test_c2_gpa_auto_basis_uses_non_collinear_active_b_vectors() -> None:
    peak_rows = [
        {
            "peak_id": 0,
            "x": 152.0,
            "y": 128.0,
            "gx": 24.0,
            "gy": 0.0,
            "g_magnitude": 24.0,
            "g_unit": "1/pixel",
            "intensity": 100.0,
            "peak_confidence": 0.9,
        },
        {
            "peak_id": 1,
            "x": 104.0,
            "y": 128.0,
            "gx": -24.0,
            "gy": 0.0,
            "g_magnitude": 24.0,
            "g_unit": "1/pixel",
            "intensity": 98.0,
            "peak_confidence": 0.9,
        },
        {
            "peak_id": 2,
            "x": 128.0,
            "y": 146.0,
            "gx": 0.0,
            "gy": 18.0,
            "g_magnitude": 18.0,
            "g_unit": "1/pixel",
            "intensity": 70.0,
            "peak_confidence": 0.8,
        },
    ]

    basis = _select_c2_gpa_auto_basis(
        peak_rows,
        fft_center_xy=(128.0, 128.0),
        min_angle_deg=15.0,
    )

    assert [row["peak_id"] for row in basis["rows"]] == [0, 2]
    assert basis["g_vector_unit"] == "fft_pixel"
    assert basis["g_vectors"] == [(24.0, 0.0), (0.0, 18.0)]
    assert basis["angle_deg"] == pytest.approx(90.0)


def test_c2_gpa_manual_basis_rejects_collinear_peak_ids() -> None:
    peak_rows = [
        {"peak_id": 0, "x": 132.0, "y": 128.0, "gx": 4.0, "gy": 0.0, "g_unit": "1/nm"},
        {"peak_id": 1, "x": 136.0, "y": 128.0, "gx": 8.0, "gy": 0.0, "g_unit": "1/nm"},
    ]

    with pytest.raises(ValueError, match="collinear"):
        _select_c2_gpa_manual_basis(peak_rows, (0, 1), fft_center_xy=(128.0, 128.0))


def test_c2_gpa_manual_basis_uses_fft_offsets_not_calibrated_b_units() -> None:
    peak_rows = [
        {"peak_id": 3, "x": 149.0, "y": 128.0, "gx": 4.2, "gy": 0.0, "g_unit": "1/nm"},
        {"peak_id": 7, "x": 128.0, "y": 147.0, "gx": 0.0, "gy": 3.8, "g_unit": "1/nm"},
    ]

    basis = _select_c2_gpa_manual_basis(
        peak_rows,
        (7, 3),
        fft_center_xy=(128.0, 128.0),
    )

    assert basis["g_vectors"] == [(0.0, 19.0), (21.0, 0.0)]
    assert basis["g_vector_unit"] == "fft_pixel"
    assert basis["display_unit"] == "FFT pixel offset"
    assert basis["rows"][0]["b_g_unit"] == "1/nm"


def test_c2_gpa_pixel_size_from_calibration_converts_units() -> None:
    assert _c2_gpa_pixel_size_nm_from_calibration(
        CalibrationResult(pixel_size=Measurement(value=2.5, unit="angstrom"))
    ) == pytest.approx(0.25)
    assert _c2_gpa_pixel_size_nm_from_calibration(
        CalibrationResult(pixel_size=Measurement(value=35.0, unit="pm/pixel"))
    ) == pytest.approx(0.035)


def test_c2_gpa_default_mask_sigma_uses_broader_ui_mask() -> None:
    assert _default_c2_gpa_mask_sigma_px((4096, 4096)) == pytest.approx(20.0)
    assert _default_c2_gpa_mask_sigma_px((160, 160)) == pytest.approx(20.0)
    assert _default_c2_gpa_mask_sigma_px((48, 48)) == pytest.approx(20.0)


def test_c2_gpa_input_image_follows_b2_preprocessing_switch() -> None:
    raw = np.zeros((4, 4), dtype=np.float32)
    processed = np.ones((4, 4), dtype=np.float32)

    source, image, warnings = _resolve_c2_gpa_input_image(
        original_image=raw,
        processed_image=processed,
        preprocessing_enabled=False,
    )

    assert source == "Raw input image"
    assert np.array_equal(image, raw)
    assert warnings == []

    source, image, warnings = _resolve_c2_gpa_input_image(
        original_image=raw,
        processed_image=processed,
        preprocessing_enabled=True,
    )

    assert source == "B2 preprocessed image"
    assert np.array_equal(image, processed)
    assert warnings == []


def test_c2_gpa_input_image_falls_back_to_b2_when_shapes_differ() -> None:
    source, image, warnings = _resolve_c2_gpa_input_image(
        original_image=np.zeros((3, 4), dtype=np.float32),
        processed_image=np.ones((4, 4), dtype=np.float32),
        preprocessing_enabled=False,
    )

    assert source == "B2 preprocessed image"
    assert image.shape == (4, 4)
    assert warnings


def test_c2_gpa_reference_region_normalizes_reversed_and_clipped_bounds() -> None:
    region = _normalize_c2_gpa_reference_region(
        (80.2, 10.1, 95.7, -5.0),
        (64, 96),
    )

    assert region == (10, 64, 0, 96)
    assert _normalize_c2_gpa_reference_region(
        (10.0, 10.4, 20.0, 21.0),
        (64, 96),
    ) is None


def test_c2_gpa_reference_roi_canvas_image_preserves_original_pixel_scale() -> None:
    image = np.zeros((4096, 2048), dtype=np.float32)

    canvas, sx, sy = _to_c2_gpa_reference_roi_canvas_image(
        image,
        reference_region=(1024, 3072, 512, 1536),
        max_side=512,
    )

    assert canvas.size == (256, 512)
    assert sx == pytest.approx(8.0)
    assert sy == pytest.approx(8.0)


def test_c2_gpa_reference_roi_canvas_maps_latest_rect_to_image_pixels() -> None:
    class CanvasResult:
        json_data = {
            "objects": [
                {"type": "rect", "left": 1, "top": 1, "width": 2, "height": 2},
                {
                    "type": "rect",
                    "left": -5,
                    "top": 5,
                    "width": 25,
                    "height": 20,
                    "scaleX": 2,
                    "scaleY": 3,
                },
            ]
        }

    region = _extract_c2_gpa_reference_roi_from_canvas(
        CanvasResult(),
        image_shape=(64, 96),
        sx=2.0,
        sy=1.0,
    )

    assert region == (5, 64, 0, 90)


def test_c2_gpa_plot_defaults_to_full_field_display_and_can_mask_invalid_pixels() -> None:
    from types import SimpleNamespace

    valid_mask = np.zeros((12, 12), dtype=bool)
    valid_mask[3:9, 3:9] = True
    values = np.ones((12, 12), dtype=float)
    fields = SimpleNamespace(
        displacement_x=values,
        displacement_y=values * 0.5,
        strain_xx=values * 0.01,
        strain_yy=values * -0.01,
        strain_xy=values * 0.002,
        rotation_xy=values * 0.001,
        quality_mask=valid_mask,
    )

    full_fig = _plot_c2_gpa_maps(
        fields,
        colormap_name="BlOr",
        strain_scale=100.0,
        strain_unit="%",
        displacement_abs_limit=0.0,
        strain_abs_limit=0.0,
        rotation_abs_limit=0.0,
        display_smoothing_sigma_px=1.0,
    )

    full_array = full_fig.axes[0].images[0].get_array()
    assert int(np.ma.count_masked(full_array)) == 0
    assert np.all(np.isfinite(np.asarray(full_array)))
    assert sum(1 for ax in full_fig.axes if ax.images) == 6
    plt.close(full_fig)

    masked_fig = _plot_c2_gpa_maps(
        fields,
        colormap_name="Turbo",
        strain_scale=100.0,
        strain_unit="%",
        displacement_abs_limit=0.0,
        strain_abs_limit=0.0,
        rotation_abs_limit=0.0,
        mask_invalid=True,
    )

    assert np.ma.isMaskedArray(masked_fig.axes[0].images[0].get_array())
    assert int(np.ma.count_masked(masked_fig.axes[0].images[0].get_array())) == int(
        np.sum(~valid_mask)
    )
    plt.close(masked_fig)


def test_c2_gpa_blocking_warnings_classify_unreportable_maps() -> None:
    warnings = [
        "GPA strain exceeds 20% over much of the valid region; this usually indicates bad basis.",
        "minor display warning",
    ]

    blocking = _c2_gpa_blocking_warnings(warnings)

    assert blocking == [warnings[0]]


def test_b5_simulated_pattern_caps_dense_labels() -> None:
    spots = [
        SimulatedSpot(
            gx=Measurement(value=float(idx % 20), unit="1/nm"),
            gy=Measurement(value=float(idx // 20), unit="1/nm"),
            d_spacing=Measurement(value=0.1, unit="nm"),
            intensity=float(idx),
            h=idx,
            k=0,
            l=0,
        )
        for idx in range(180)
    ]
    simulation = SimulationResult(
        patterns=[
            SimulatedPattern(
                kind="diffraction",
                structure_id="dense",
                zone_axis=(0, 0, 1),
                spots=spots,
            )
        ]
    )

    fig = _plot_b5_simulated_pattern(simulation)

    assert fig is not None
    assert "showing" in fig.axes[0].get_title()
    assert len(fig.axes[0].texts) <= 18
    plt.close(fig)
