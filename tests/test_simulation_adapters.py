from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from autotem.core import (
    DiffractionPeak,
    DiffractionResult,
    GVector,
    Measurement,
    PhaseCandidate,
    SimulatedPattern,
    SimulatedSpot,
)
from autotem.simulation.basic import simulate_from_diffraction
from autotem.simulation.diffsims_adapter import simulate_diffraction_with_diffsims
from autotem.simulation.pymatgen_adapter import load_candidate_phase_from_cif


def test_pymatgen_adapter_reads_cif_with_fake_pymatgen(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeLattice:
        a = 4.0
        b = 4.0
        c = 4.0
        alpha = 90.0
        beta = 90.0
        gamma = 90.0

    class FakeComposition:
        reduced_formula = "X"

    class FakeStructure:
        lattice = FakeLattice()
        composition = FakeComposition()
        sites = []

        @classmethod
        def from_file(cls, path: str) -> FakeStructure:
            assert path.endswith("fake.cif")
            instance = cls()
            instance.sites = [
                types.SimpleNamespace(
                    specie=types.SimpleNamespace(symbol="X"),
                    frac_coords=[0.0, 0.0, 0.0],
                    species=None,
                )
            ]
            return instance

        def __iter__(self):
            return iter(self.sites)

        def get_space_group_info(self) -> tuple[str, int]:
            return "Pm-3m", 221

    pymatgen_pkg = types.ModuleType("pymatgen")
    pymatgen_core = types.ModuleType("pymatgen.core")
    pymatgen_core.Structure = FakeStructure
    monkeypatch.setitem(sys.modules, "pymatgen", pymatgen_pkg)
    monkeypatch.setitem(sys.modules, "pymatgen.core", pymatgen_core)

    cif_path = tmp_path / "fake.cif"
    cif_path.write_text("_cell_length_a 4.0", encoding="utf-8")
    result = load_candidate_phase_from_cif(cif_path)

    assert result.candidate is not None
    assert result.candidate["source"] == "pymatgen_cif"
    assert result.candidate["a"] == 0.4
    assert result.candidate["space_group"] == "Pm-3m"
    assert result.candidate["space_group_number"] == 221
    assert result.candidate["atom_sites"][0]["element"] == "X"
    assert result.provenance["active"] is True


def test_diffsims_adapter_runs_with_fake_diffsims_modules(monkeypatch) -> None:
    class FakeAtom:
        def __init__(self, element: str, xyz: list[float]) -> None:
            self.element = element
            self.xyz = xyz

    class FakeLattice:
        def __init__(self, *args: float) -> None:
            self.args = args

    class FakeStructure:
        def __init__(self, *, atoms: list[FakeAtom], lattice: FakeLattice) -> None:
            self.atoms = atoms
            self.lattice = lattice

    class FakePhase:
        def __init__(self, *, name: str, structure: FakeStructure) -> None:
            self.name = name
            self.structure = structure

    class FakeRotation:
        @classmethod
        def from_axes_angles(cls, axis, angle, *, degrees: bool):
            assert degrees is True
            return {"axis": axis, "angle": angle}

    class FakeSimulation:
        coordinates = [(0.0, 0.0), (1.0, 0.0)]
        intensities = [100.0, 80.0]
        indices = [(0, 0, 0), (1, 0, 0)]

        def get_diffraction_pattern(self, **kwargs) -> np.ndarray:
            assert kwargs["shape"] == (8, 8)
            return np.ones((8, 8), dtype=np.float32)

    class FakeSimulationGenerator:
        def __init__(self, *, accelerating_voltage: float) -> None:
            self.accelerating_voltage = accelerating_voltage

        def calculate_diffraction2d(self, **kwargs) -> FakeSimulation:
            assert kwargs["reciprocal_radius"] == 1.0
            return FakeSimulation()

    diffpy_pkg = types.ModuleType("diffpy")
    diffpy_structure = types.ModuleType("diffpy.structure")
    diffpy_structure.Atom = FakeAtom
    diffpy_structure.Lattice = FakeLattice
    diffpy_structure.Structure = FakeStructure

    diffsims_pkg = types.ModuleType("diffsims")
    diffsims_generators = types.ModuleType("diffsims.generators")
    diffsims_sim_generator = types.ModuleType("diffsims.generators.simulation_generator")
    diffsims_sim_generator.SimulationGenerator = FakeSimulationGenerator

    orix_pkg = types.ModuleType("orix")
    orix_crystal_map = types.ModuleType("orix.crystal_map")
    orix_crystal_map.Phase = FakePhase
    orix_quaternion = types.ModuleType("orix.quaternion")
    orix_quaternion.Rotation = FakeRotation

    monkeypatch.setitem(sys.modules, "diffpy", diffpy_pkg)
    monkeypatch.setitem(sys.modules, "diffpy.structure", diffpy_structure)
    monkeypatch.setitem(sys.modules, "diffsims", diffsims_pkg)
    monkeypatch.setitem(sys.modules, "diffsims.generators", diffsims_generators)
    monkeypatch.setitem(
        sys.modules,
        "diffsims.generators.simulation_generator",
        diffsims_sim_generator,
    )
    monkeypatch.setitem(sys.modules, "orix", orix_pkg)
    monkeypatch.setitem(sys.modules, "orix.crystal_map", orix_crystal_map)
    monkeypatch.setitem(sys.modules, "orix.quaternion", orix_quaternion)

    candidate = PhaseCandidate(
        name="fake",
        formula="Si",
        lattice_parameters={
            "a": 0.4,
            "b": 0.4,
            "c": 0.4,
            "alpha": 90,
            "beta": 90,
            "gamma": 90,
            "atom_sites": [
                {"element": "Si", "abc": [0.0, 0.0, 0.0], "occupancy": 1.0}
            ],
        },
    )
    result = simulate_diffraction_with_diffsims(
        [candidate],
        pattern_shape=(8, 8),
        rotation_angles_deg=[0.0, 15.0],
    )

    assert not result.warnings
    assert len(result.patterns) == 2
    assert result.patterns[0].simulation_parameters["engine"] == "diffsims"
    assert result.patterns[1].simulation_parameters["rotation_angle_deg"] == 15.0
    assert result.patterns[0].simulation_parameters["structure_source"] == "candidate_atom_sites"
    assert result.patterns[0].data is not None
    assert result.patterns[0].data.shape == (8, 8)
    assert len(result.patterns[0].spots) == 1
    assert result.patterns[0].spots[0].magnitude is not None
    assert result.patterns[0].spots[0].magnitude.value == 10.0
    assert result.provenance["active"] is True


def test_diffsims_adapter_scans_zone_axis_candidates(monkeypatch) -> None:
    class FakeAtom:
        def __init__(self, element: str, xyz: list[float]) -> None:
            self.element = element
            self.xyz = xyz

    class FakeLattice:
        def __init__(self, *args: float) -> None:
            self.args = args

    class FakeStructure:
        def __init__(self, *, atoms: list[FakeAtom], lattice: FakeLattice) -> None:
            self.atoms = atoms
            self.lattice = lattice

    class FakePhase:
        def __init__(self, *, name: str, structure: FakeStructure) -> None:
            self.name = name
            self.structure = structure

    class FakeRotation:
        @classmethod
        def from_axes_angles(cls, axis, angle, *, degrees: bool):
            assert degrees is True
            return {"axis": axis, "angle": angle}

    class FakeSimulation:
        coordinates = [(0.0, 0.0), (1.0, 0.0)]
        intensities = [100.0, 80.0]
        indices = [(0, 0, 0), (1, 0, 0)]

    class FakeSimulationGenerator:
        def __init__(self, *, accelerating_voltage: float) -> None:
            self.accelerating_voltage = accelerating_voltage

        def calculate_diffraction2d(self, **kwargs) -> FakeSimulation:
            assert kwargs["reciprocal_radius"] == 1.0
            return FakeSimulation()

    diffpy_pkg = types.ModuleType("diffpy")
    diffpy_structure = types.ModuleType("diffpy.structure")
    diffpy_structure.Atom = FakeAtom
    diffpy_structure.Lattice = FakeLattice
    diffpy_structure.Structure = FakeStructure

    diffsims_pkg = types.ModuleType("diffsims")
    diffsims_generators = types.ModuleType("diffsims.generators")
    diffsims_sim_generator = types.ModuleType("diffsims.generators.simulation_generator")
    diffsims_sim_generator.SimulationGenerator = FakeSimulationGenerator

    orix_pkg = types.ModuleType("orix")
    orix_crystal_map = types.ModuleType("orix.crystal_map")
    orix_crystal_map.Phase = FakePhase
    orix_quaternion = types.ModuleType("orix.quaternion")
    orix_quaternion.Rotation = FakeRotation

    monkeypatch.setitem(sys.modules, "diffpy", diffpy_pkg)
    monkeypatch.setitem(sys.modules, "diffpy.structure", diffpy_structure)
    monkeypatch.setitem(sys.modules, "diffsims", diffsims_pkg)
    monkeypatch.setitem(sys.modules, "diffsims.generators", diffsims_generators)
    monkeypatch.setitem(
        sys.modules,
        "diffsims.generators.simulation_generator",
        diffsims_sim_generator,
    )
    monkeypatch.setitem(sys.modules, "orix", orix_pkg)
    monkeypatch.setitem(sys.modules, "orix.crystal_map", orix_crystal_map)
    monkeypatch.setitem(sys.modules, "orix.quaternion", orix_quaternion)

    result = simulate_diffraction_with_diffsims(
        [
            PhaseCandidate(
                name="fake",
                formula="Si",
                lattice_parameters={
                    "a": 0.4,
                    "b": 0.4,
                    "c": 0.4,
                    "alpha": 90,
                    "beta": 90,
                    "gamma": 90,
                    "atom_sites": [
                        {"element": "Si", "abc": [0.0, 0.0, 0.0], "occupancy": 1.0}
                    ],
                },
            )
        ],
        zone_axis_candidates=[[0, 0, 1], [1, 1, 0]],
        rotation_angles_deg=[0.0],
    )

    assert len(result.patterns) == 2
    assert result.patterns[0].zone_axis == (0, 0, 1)
    assert result.patterns[1].zone_axis == (1, 1, 0)
    assert result.patterns[1].simulation_parameters["zone_axis_index"] == 1
    assert result.provenance["zone_axis_candidates"] == [[0, 0, 1], [1, 1, 0]]


def test_simulation_engine_diffsims_routes_through_optional_adapter(monkeypatch) -> None:
    def fake_simulate_diffraction_with_diffsims(candidates, **kwargs):
        from autotem.simulation.diffsims_adapter import DiffsimsSimulationResult

        return DiffsimsSimulationResult(
            patterns=[],
            warnings=["fake diffsims unavailable"],
            provenance={"adapter": "diffsims", "active": False},
        )

    monkeypatch.setattr(
        "autotem.simulation.basic.simulate_diffraction_with_diffsims",
        fake_simulate_diffraction_with_diffsims,
    )
    report = simulate_from_diffraction(
        DiffractionResult(phase_candidates=[PhaseCandidate(name="fake")]),
        parameters={"simulation_engine": "diffsims"},
    )

    assert report.provenance["method"] == "diffsims"
    assert "fake diffsims unavailable" in report.warnings


def test_simulation_engine_diffsims_closes_phase_matching_loop(monkeypatch) -> None:
    class FakeAtom:
        def __init__(self, element: str, xyz: list[float]) -> None:
            self.element = element
            self.xyz = xyz

    class FakeLattice:
        def __init__(self, *args: float) -> None:
            self.args = args

    class FakeStructure:
        def __init__(self, *, atoms: list[FakeAtom], lattice: FakeLattice) -> None:
            self.atoms = atoms
            self.lattice = lattice

    class FakePhase:
        def __init__(self, *, name: str, structure: FakeStructure) -> None:
            self.name = name
            self.structure = structure

    class FakeSimulation:
        coordinates = np.asarray(
            [
                (0.0, 0.0),
                (0.25, 0.0),
                (0.0, 0.25),
                (0.25, 0.25),
                (-0.25, 0.25),
            ],
            dtype=np.float64,
        )
        intensities = np.asarray([999.0, 100.0, 96.0, 80.0, 72.0], dtype=np.float64)
        indices = np.asarray(
            [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (-1, 1, 0),
            ],
            dtype=np.float64,
        )

    class FakeSimulationGenerator:
        def __init__(self, *, accelerating_voltage: float) -> None:
            self.accelerating_voltage = accelerating_voltage

        def calculate_diffraction2d(self, **kwargs) -> FakeSimulation:
            assert kwargs["reciprocal_radius"] == 1.0
            return FakeSimulation()

    diffpy_pkg = types.ModuleType("diffpy")
    diffpy_structure = types.ModuleType("diffpy.structure")
    diffpy_structure.Atom = FakeAtom
    diffpy_structure.Lattice = FakeLattice
    diffpy_structure.Structure = FakeStructure

    diffsims_pkg = types.ModuleType("diffsims")
    diffsims_generators = types.ModuleType("diffsims.generators")
    diffsims_sim_generator = types.ModuleType("diffsims.generators.simulation_generator")
    diffsims_sim_generator.SimulationGenerator = FakeSimulationGenerator

    orix_pkg = types.ModuleType("orix")
    orix_crystal_map = types.ModuleType("orix.crystal_map")
    orix_crystal_map.Phase = FakePhase

    monkeypatch.setitem(sys.modules, "diffpy", diffpy_pkg)
    monkeypatch.setitem(sys.modules, "diffpy.structure", diffpy_structure)
    monkeypatch.setitem(sys.modules, "diffsims", diffsims_pkg)
    monkeypatch.setitem(sys.modules, "diffsims.generators", diffsims_generators)
    monkeypatch.setitem(
        sys.modules,
        "diffsims.generators.simulation_generator",
        diffsims_sim_generator,
    )
    monkeypatch.setitem(sys.modules, "orix", orix_pkg)
    monkeypatch.setitem(sys.modules, "orix.crystal_map", orix_crystal_map)

    gxy = [(2.5, 0.0), (0.0, 2.5), (2.5, 2.5), (-2.5, 2.5)]
    diffraction = DiffractionResult(
        peaks=[
            DiffractionPeak(x=0.0, y=0.0, intensity=intensity, confidence=0.9)
            for intensity in [100.0, 96.0, 80.0, 72.0]
        ],
        g_vectors=[
            GVector(
                gx=Measurement(value=gx, unit="1/nm", method="test"),
                gy=Measurement(value=gy, unit="1/nm", method="test"),
                magnitude=Measurement(value=float(np.hypot(gx, gy)), unit="1/nm"),
                source_peak_index=idx,
            )
            for idx, (gx, gy) in enumerate(gxy)
        ],
        phase_candidates=[
            PhaseCandidate(
                name="fake",
                formula="Si",
                score=0.1,
            lattice_parameters={
                "a": 0.4,
                "b": 0.4,
                "c": 0.4,
                "alpha": 90,
                "beta": 90,
                "gamma": 90,
                "atom_sites": [
                    {"element": "Si", "abc": [0.0, 0.0, 0.0], "occupancy": 1.0}
                ],
            },
        )
        ],
    )

    report = simulate_from_diffraction(
        diffraction,
        parameters={
            "simulation_engine": "diffsims",
            "phase_match_previous_score_weight": 0.0,
            "phase_match_min_spots": 3,
            "phase_match_d_tolerance_rel": 0.02,
            "phase_match_angle_tolerance_deg": 2.0,
        },
    )

    assert report.provenance["phase_matching"]["active"] is True
    assert report.matched_phase_candidates[0].matched_spot_count == 4
    assert (report.matched_phase_candidates[0].score or 0.0) > 0.95
    assert report.matched_phase_candidates[0].mean_d_error is not None
    assert report.matched_phase_candidates[0].mean_d_error.method == "diffsims_phase_match"
    assert diffraction.phase_candidates[0].matched_spot_count == 4
    assert report.patterns[0].score is not None
    assert report.patterns[0].simulation_parameters["phase_match"]["matched_spot_count"] == 4


def test_simulation_engine_writes_best_zone_axis_from_matching(monkeypatch) -> None:
    from autotem.simulation.diffsims_adapter import DiffsimsSimulationResult

    def fake_simulate_diffraction_with_diffsims(candidates, **kwargs):
        assert kwargs["zone_axis_candidates"] == [(0, 0, 1), (1, 1, 0)]
        return DiffsimsSimulationResult(
            patterns=[
                SimulatedPattern(
                    kind="diffraction",
                    structure_id="fake",
                    zone_axis=(0, 0, 1),
                    simulation_parameters={"engine": "diffsims", "zone_axis_index": 0},
                    spots=[
                        SimulatedSpot(
                            gx=Measurement(value=7.0, unit="1/nm"),
                            gy=Measurement(value=7.0, unit="1/nm"),
                            magnitude=Measurement(value=float(np.hypot(7.0, 7.0)), unit="1/nm"),
                        )
                    ],
                ),
                SimulatedPattern(
                    kind="diffraction",
                    structure_id="fake",
                    zone_axis=(1, 1, 0),
                    simulation_parameters={"engine": "diffsims", "zone_axis_index": 1},
                    spots=[
                        SimulatedSpot(
                            gx=Measurement(value=2.5, unit="1/nm"),
                            gy=Measurement(value=0.0, unit="1/nm"),
                            magnitude=Measurement(value=2.5, unit="1/nm"),
                        ),
                        SimulatedSpot(
                            gx=Measurement(value=0.0, unit="1/nm"),
                            gy=Measurement(value=2.5, unit="1/nm"),
                            magnitude=Measurement(value=2.5, unit="1/nm"),
                        ),
                    ],
                ),
            ],
            warnings=[],
            provenance={"adapter": "diffsims", "active": True, "pattern_count": 2},
        )

    monkeypatch.setattr(
        "autotem.simulation.basic.simulate_diffraction_with_diffsims",
        fake_simulate_diffraction_with_diffsims,
    )

    diffraction = DiffractionResult(
        peaks=[
            DiffractionPeak(x=0.0, y=0.0, intensity=100.0),
            DiffractionPeak(x=0.0, y=0.0, intensity=90.0),
        ],
        g_vectors=[
            GVector(
                gx=Measurement(value=2.5, unit="1/nm"),
                gy=Measurement(value=0.0, unit="1/nm"),
                magnitude=Measurement(value=2.5, unit="1/nm"),
                source_peak_index=0,
            ),
            GVector(
                gx=Measurement(value=0.0, unit="1/nm"),
                gy=Measurement(value=2.5, unit="1/nm"),
                magnitude=Measurement(value=2.5, unit="1/nm"),
                source_peak_index=1,
            ),
        ],
        phase_candidates=[PhaseCandidate(name="fake", score=0.0)],
    )

    report = simulate_from_diffraction(
        diffraction,
        parameters={
            "simulation_engine": "diffsims",
            "zone_axis_candidates": [[0, 0, 1], [1, 1, 0]],
            "phase_match_previous_score_weight": 0.0,
            "phase_match_min_spots": 2,
            "phase_match_d_tolerance_rel": 0.02,
            "phase_match_angle_tolerance_deg": 2.0,
        },
    )

    assert report.provenance["phase_matching"]["selected_zone_axis"] == [1, 1, 0]
    assert report.matched_phase_candidates[0].zone_axis == (1, 1, 0)
    assert diffraction.phase_candidates[0].zone_axis == (1, 1, 0)
    phase_match = report.patterns[1].simulation_parameters["phase_match"]
    assert phase_match["zone_axis"] == [1, 1, 0]
    assert phase_match["matched_spot_count"] == 2
