from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import matplotlib.pyplot as plt

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

from apps.b2_diffraction_ui import (
    _FALLBACK_DEFAULTS,
    _build_simulated_spot_rows,
    _filter_b5_simulation_for_zone_axis,
    _plot_b5_simulated_pattern,
    _plot_b5_simulation_overlay,
    SUPPORTED_SIMULATION_ENGINES,
)
from autotem.core import (
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
    assert _FALLBACK_DEFAULTS["simulation"]["zone_axis_mode"] == "inherit_b4_locked"
    assert _FALLBACK_DEFAULTS["simulation"]["simulation_engine"] == "diffsims"
    assert _FALLBACK_DEFAULTS["simulation"]["show_match_residual_lines"] is False
    assert SUPPORTED_SIMULATION_ENGINES == ["diffsims"]


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
