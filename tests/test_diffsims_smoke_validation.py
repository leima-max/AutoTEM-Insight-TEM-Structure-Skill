from __future__ import annotations

from autotem.core import Measurement, SimulatedPattern, SimulatedSpot
from autotem.simulation.diffsims_adapter import DiffsimsSimulationResult
from autotem.validation import diffsims_smoke
from autotem.validation.diffsims_smoke import run_diffsims_smoke_validation


def _spot(gx: float, gy: float, intensity: float) -> SimulatedSpot:
    magnitude = (gx * gx + gy * gy) ** 0.5
    return SimulatedSpot(
        gx=Measurement(value=gx, unit="1/nm"),
        gy=Measurement(value=gy, unit="1/nm"),
        magnitude=Measurement(value=magnitude, unit="1/nm"),
        d_spacing=Measurement(value=1.0 / magnitude, unit="nm"),
        intensity=intensity,
    )


def test_diffsims_smoke_validation_closes_loop_with_fake_simulator(monkeypatch) -> None:
    monkeypatch.setattr(
        diffsims_smoke,
        "_dependency_versions",
        lambda: {"diffsims": "0.7.0", "diffpy.structure": "3.2.3", "orix": "0.14.2"},
    )

    def fake_simulate(candidates, **kwargs):
        assert kwargs["pattern_shape"] == (64, 64)
        candidate = candidates[0]
        return DiffsimsSimulationResult(
            patterns=[
                SimulatedPattern(
                    kind="diffraction",
                    structure_id=candidate.identifier or candidate.name,
                    zone_axis=(0, 0, 1),
                    simulation_parameters={"engine": "diffsims"},
                    spots=[
                        _spot(2.5, 0.0, 100.0),
                        _spot(0.0, 2.5, 90.0),
                        _spot(2.5, 2.5, 70.0),
                        _spot(-2.5, 2.5, 60.0),
                    ],
                )
            ],
            warnings=[],
            provenance={"adapter": "diffsims", "active": True},
        )

    result = run_diffsims_smoke_validation(
        materials=["si"],
        min_matched_spots=3,
        simulate_func=fake_simulate,
    )

    assert result["ok"] is True
    assert result["materials"][0]["ok"] is True
    assert result["materials"][0]["matched_spot_count"] >= 3
    assert result["materials"][0]["source"] == "local_builtin_crystal_library"


def test_diffsims_smoke_validation_reports_simulation_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        diffsims_smoke,
        "_dependency_versions",
        lambda: {"diffsims": "0.7.0", "diffpy.structure": "3.2.3", "orix": "0.14.2"},
    )

    def fake_simulate(candidates, **kwargs):
        return DiffsimsSimulationResult(
            patterns=[],
            warnings=["fake diffsims unavailable"],
            provenance={"adapter": "diffsims", "active": False},
        )

    result = run_diffsims_smoke_validation(
        materials=["si"],
        simulate_func=fake_simulate,
    )

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert result["materials"][0]["stage"] == "diffsims_simulation"
    assert "fake diffsims unavailable" in result["warnings"]
