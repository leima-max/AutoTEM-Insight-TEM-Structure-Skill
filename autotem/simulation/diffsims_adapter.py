"""Optional diffsims adapter for kinematic electron diffraction simulation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import acos, degrees
from typing import Any

import numpy as np

from autotem.core import (
    ArrayReference,
    Measurement,
    PhaseCandidate,
    SimulatedPattern,
    SimulatedSpot,
)


@dataclass(slots=True)
class DiffsimsSimulationResult:
    patterns: list[SimulatedPattern]
    warnings: list[str]
    provenance: dict[str, Any]


_ELEMENT_RE = re.compile(r"[A-Z][a-z]?")
_SPACE_GROUP_SYMBOL_ALIASES = {
    "P1": 1,
    "P-1": 2,
    "P2": 3,
    "P21": 4,
    "C2": 5,
    "PM": 6,
    "PC": 7,
    "CM": 8,
    "CC": 9,
    "P2/M": 10,
    "P21/M": 11,
    "C2/M": 12,
    "P2/C": 13,
    "P21/C": 14,
    "PNMA": 62,
    "PBNM": 62,
    "P42/MNM": 136,
    "R-3M": 166,
    "P63MC": 186,
    "P63/MMC": 194,
    "PM-3M": 221,
    "FM-3M": 225,
    "FD-3M": 227,
    "IM-3M": 229,
}


def _first_element(formula: str | None) -> str | None:
    if not formula:
        return None
    match = _ELEMENT_RE.search(str(formula))
    return match.group(0) if match else None


def _phase_lattice_nm(
    candidate: PhaseCandidate,
) -> tuple[float, float, float, float, float, float] | None:
    lattice = candidate.lattice_parameters
    if not isinstance(lattice, dict):
        return None
    try:
        return (
            float(lattice["a"]),
            float(lattice["b"]),
            float(lattice["c"]),
            float(lattice.get("alpha", 90.0)),
            float(lattice.get("beta", 90.0)),
            float(lattice.get("gamma", 90.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _phase_space_group_number(candidate: PhaseCandidate) -> int | None:
    lattice = candidate.lattice_parameters
    if not isinstance(lattice, dict):
        return None
    for key in ("space_group_number", "spacegroup_number", "sg_number", "sgno"):
        value = lattice.get(key)
        number = _coerce_space_group_number(value)
        if number is not None:
            return number
    return None


def _coerce_space_group_number(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    if 1 <= number <= 230:
        return number
    return None


def _phase_space_group_symbol(candidate: PhaseCandidate) -> str | None:
    lattice = candidate.lattice_parameters
    if not isinstance(lattice, dict):
        return None
    for key in (
        "space_group_symbol",
        "space_group",
        "spacegroup",
        "sg",
        "hm_symbol",
        "hall_symbol",
    ):
        value = lattice.get(key)
        if value is None:
            continue
        symbol = str(value).strip()
        if symbol and symbol.lower() not in {"none", "null", "nan", "unknown", "?"}:
            return symbol
    return None


def _normalize_space_group_symbol(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    normalized = (
        str(symbol)
        .strip()
        .replace("_", "")
        .replace(" ", "")
        .replace("\t", "")
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .upper()
    )
    return normalized or None


def _phase_space_group_metadata(
    candidate: PhaseCandidate,
) -> tuple[int, str | None, str, list[str]]:
    number = _phase_space_group_number(candidate)
    symbol = _phase_space_group_symbol(candidate)
    if number is not None:
        return number, symbol, "candidate_space_group_number", []

    normalized_symbol = _normalize_space_group_symbol(symbol)
    if normalized_symbol is not None:
        number = _SPACE_GROUP_SYMBOL_ALIASES.get(normalized_symbol)
        if number is not None:
            return number, symbol, "candidate_space_group_symbol", []

    warnings = [
        f"diffsims used P1 symmetry fallback for '{candidate.name}' because "
        "space_group_number was unavailable."
    ]
    if symbol:
        warnings[0] = (
            f"diffsims used P1 symmetry fallback for '{candidate.name}' because "
            f"space group symbol '{symbol}' could not be mapped to a number."
        )
    return 1, symbol, "p1_fallback", warnings


def _phase_atom_sites(candidate: PhaseCandidate) -> list[dict[str, Any]]:
    lattice = candidate.lattice_parameters
    if not isinstance(lattice, dict):
        return []
    raw_sites = lattice.get("atom_sites") or lattice.get("atoms") or lattice.get("sites")
    if not isinstance(raw_sites, list):
        return []
    sites: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_sites):
        if not isinstance(row, dict):
            continue
        element = row.get("element") or row.get("type_symbol") or row.get("species")
        if not isinstance(element, str) or not element.strip():
            continue
        coords = row.get("abc") or row.get("frac_coords") or row.get("fractional")
        if not isinstance(coords, (list, tuple)) or len(coords) < 3:
            continue
        try:
            abc = [float(coords[0]), float(coords[1]), float(coords[2])]
            occupancy = float(row.get("occupancy", 1.0))
        except (TypeError, ValueError):
            continue
        sites.append(
            {
                "element": element.strip(),
                "abc": abc,
                "occupancy": occupancy,
                "source_index": int(row.get("source_index", idx)),
            }
        )
    return sites


def _extract_coordinate_count(simulation: Any) -> int | None:
    coords = _array_or_none(getattr(simulation, "coordinates", None))
    if coords is None:
        return None
    try:
        return int(len(coords))
    except TypeError:
        return None


def _array_or_none(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if callable(value):
        try:
            value = value()
        except TypeError:
            return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        for attr in ("data", "xyz", "hkl", "coordinates"):
            nested = getattr(value, attr, None)
            if nested is value:
                continue
            array = _array_or_none(nested)
            if array is not None:
                return array
        return None
    if array.size == 0:
        return None
    return array


def _first_available_array(source: Any, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        array = _array_or_none(getattr(source, name, None))
        if array is not None:
            return array
    return None


def _array_from_xyz_tuple(value: Any) -> np.ndarray | None:
    if not isinstance(value, tuple) or len(value) < 2:
        return None
    columns = []
    for column in value[:3]:
        array = _array_or_none(column)
        if array is None:
            return None
        columns.append(np.ravel(array))
    try:
        stacked = np.column_stack(columns)
    except ValueError:
        return None
    return stacked if stacked.ndim == 2 and stacked.shape[1] >= 2 else None


def _extract_reciprocal_coordinates(simulation: Any) -> np.ndarray | None:
    coordinates = getattr(simulation, "coordinates", None)
    if coordinates is None:
        return None

    # diffsims exposes a DiffractingVector object where np.asarray(...) returns
    # hkl indices, while .data/.xyz contain reciprocal coordinates in 1/Angstrom.
    for name in ("data", "xyz"):
        raw = getattr(coordinates, name, None)
        array = _array_from_xyz_tuple(raw)
        if array is None:
            array = _array_or_none(raw)
        if array is not None and array.ndim == 2 and array.shape[1] >= 2:
            return array

    for name in ("data", "xyz", "reciprocal_coordinates", "cartesian_coordinates"):
        raw = getattr(simulation, name, None)
        array = _array_from_xyz_tuple(raw)
        if array is None:
            array = _array_or_none(raw)
        if array is not None and array.ndim == 2 and array.shape[1] >= 2:
            return array

    return _array_or_none(coordinates)


def _extract_indices(simulation: Any) -> np.ndarray | None:
    coordinates = getattr(simulation, "coordinates", None)
    nested = _first_available_array(coordinates, ("hkl", "indices", "coordinates"))
    if nested is not None and nested.ndim == 2 and nested.shape[1] >= 3:
        return nested
    return _first_available_array(
        simulation,
        (
            "indices",
            "hkl",
            "hkls",
            "miller_indices",
            "miller_indices_all",
        ),
    )


def _extract_intensities(simulation: Any) -> np.ndarray | None:
    coordinates = getattr(simulation, "coordinates", None)
    nested = _first_available_array(coordinates, ("intensity", "intensities"))
    if nested is not None:
        return nested
    return _first_available_array(
        simulation,
        (
            "intensities",
            "intensity",
            "coordinates_intensity",
        ),
    )


def _extract_simulated_spots(
    simulation: Any,
    *,
    minimum_g_inv_nm: float,
    max_spots: int,
) -> list[SimulatedSpot]:
    coords = _extract_reciprocal_coordinates(simulation)
    if coords is None:
        return []
    if coords.ndim == 1:
        if coords.size < 2:
            return []
        coords = coords.reshape(1, -1)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return []

    indices = _extract_indices(simulation)
    if indices is not None and indices.ndim == 1 and indices.size >= 3:
        indices = indices.reshape(1, -1)
    intensities = _extract_intensities(simulation)
    if intensities is not None:
        intensities = np.ravel(intensities)

    spots: list[SimulatedSpot] = []
    for idx, row in enumerate(coords):
        # diffsims reciprocal coordinates are reported in inverse Angstrom for
        # electron diffraction templates. AutoTEM stores reciprocal metrics in 1/nm.
        gx = float(row[0]) * 10.0
        gy = float(row[1]) * 10.0
        magnitude = float(np.hypot(gx, gy))
        if magnitude <= max(1e-12, float(minimum_g_inv_nm)):
            continue
        intensity = None
        if intensities is not None and idx < len(intensities):
            intensity = float(intensities[idx])

        hkl: tuple[int | None, int | None, int | None] = (None, None, None)
        if indices is not None and indices.ndim == 2 and idx < indices.shape[0]:
            hkl_row = indices[idx]
            if hkl_row.size >= 3:
                hkl = (int(round(hkl_row[0])), int(round(hkl_row[1])), int(round(hkl_row[2])))

        spots.append(
            SimulatedSpot(
                gx=Measurement(value=gx, unit="1/nm", method="diffsims_coordinate"),
                gy=Measurement(value=gy, unit="1/nm", method="diffsims_coordinate"),
                magnitude=Measurement(
                    value=magnitude,
                    unit="1/nm",
                    method="diffsims_coordinate_norm",
                ),
                d_spacing=Measurement(
                    value=1.0 / magnitude,
                    unit="nm",
                    method="diffsims_reciprocal_inverse",
                ),
                intensity=intensity,
                h=hkl[0],
                k=hkl[1],
                l=hkl[2],
                source_index=idx,
            )
        )
        if len(spots) >= max(1, int(max_spots)):
            break
    return spots


def _make_atom(Atom: Any, element: str, abc: list[float], lattice: Any) -> Any:
    constructors = (
        lambda: Atom(atype=element, xyz=abc, lattice=lattice),
        lambda: Atom(element=element, xyz=abc, lattice=lattice),
        lambda: Atom(element, abc, lattice=lattice),
        lambda: Atom(element, abc),
    )
    last_exc: Exception | None = None
    for constructor in constructors:
        try:
            return constructor()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise TypeError("Unable to construct diffpy Atom.")


def _make_structure(Structure: Any, *, atoms: list[Any], lattice: Any) -> Any:
    try:
        return Structure(atoms=atoms, lattice=lattice)
    except Exception:  # noqa: BLE001
        return Structure(atoms, lattice)


def _make_phase(
    Phase: Any,
    *,
    candidate: PhaseCandidate,
    structure: Any,
    space_group_number: int | None,
) -> Any:
    constructors = []
    if space_group_number is not None:
        constructors.extend(
            [
                lambda: Phase(
                    name=candidate.name,
                    structure=structure,
                    space_group=space_group_number,
                ),
                lambda: Phase(
                    structure=structure,
                    space_group=space_group_number,
                ),
            ]
        )
    constructors.extend(
        [
            lambda: Phase(name=candidate.name, structure=structure),
            lambda: Phase(structure=structure),
        ]
    )
    last_exc: Exception | None = None
    for constructor in constructors:
        try:
            return constructor()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise TypeError("Unable to construct orix Phase.")


def _build_structure_and_phase(
    *,
    candidate: PhaseCandidate,
    Atom: Any,
    Lattice: Any,
    Structure: Any,
    Phase: Any,
    fallback_single_atom: bool,
) -> tuple[Any | None, Any | None, dict[str, Any], list[str]]:
    warnings: list[str] = []
    lattice_nm = _phase_lattice_nm(candidate)
    if lattice_nm is None:
        warnings.append(
            f"diffsims skipped '{candidate.name}': lattice parameters are not available."
        )
        return None, None, {"structure_source": "missing_lattice"}, warnings

    a_nm, b_nm, c_nm, alpha, beta, gamma = lattice_nm
    lattice = Lattice(
        a_nm * 10.0,
        b_nm * 10.0,
        c_nm * 10.0,
        alpha,
        beta,
        gamma,
    )
    raw_sites = _phase_atom_sites(candidate)
    structure_source = "candidate_atom_sites"
    if not raw_sites:
        if not fallback_single_atom:
            warnings.append(
                f"diffsims skipped '{candidate.name}': atom_sites are not available."
            )
            return None, None, {"structure_source": "missing_atom_sites"}, warnings
        element = _first_element(candidate.formula) or "Si"
        raw_sites = [{"element": element, "abc": [0.0, 0.0, 0.0], "occupancy": 1.0}]
        structure_source = "single_atom_fallback"
        warnings.append(
            f"diffsims used a single-atom fallback motif for '{candidate.name}'."
        )

    atoms = [
        _make_atom(Atom, str(site["element"]), list(site["abc"]), lattice)
        for site in raw_sites
    ]
    structure = _make_structure(Structure, atoms=atoms, lattice=lattice)
    space_group_number, space_group_symbol, space_group_source, symmetry_warnings = (
        _phase_space_group_metadata(candidate)
    )
    warnings.extend(symmetry_warnings)
    phase = _make_phase(
        Phase,
        candidate=candidate,
        structure=structure,
        space_group_number=space_group_number,
    )
    provenance = {
        "structure_source": structure_source,
        "atom_site_count": len(raw_sites),
        "space_group_number": space_group_number,
        "space_group_symbol": space_group_symbol,
        "space_group_source": space_group_source,
    }
    return structure, phase, provenance, warnings


def _rotation_values(raw_angles: list[float] | tuple[float, ...] | None) -> list[float]:
    if raw_angles is None:
        return [0.0]
    values: list[float] = []
    for raw in raw_angles:
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values or [0.0]


def _zone_axis_tuple(raw_axis: object) -> tuple[int, int, int] | None:
    if not isinstance(raw_axis, (list, tuple)) or len(raw_axis) < 3:
        return None
    try:
        axis = tuple(int(round(float(raw_axis[idx]))) for idx in range(3))
    except (TypeError, ValueError):
        return None
    if axis == (0, 0, 0):
        return None
    return axis


def _zone_axis_values(
    raw_axes: object,
    *,
    default_axis: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    if raw_axes is None:
        return [default_axis]
    axes: list[tuple[int, int, int]] = []
    single_axis = _zone_axis_tuple(raw_axes)
    if single_axis is not None:
        axes.append(single_axis)
    elif isinstance(raw_axes, list):
        for raw_axis in raw_axes:
            axis = _zone_axis_tuple(raw_axis)
            if axis is not None:
                axes.append(axis)
    seen: set[tuple[int, int, int]] = set()
    unique: list[tuple[int, int, int]] = []
    for axis in axes or [default_axis]:
        if axis in seen:
            continue
        seen.add(axis)
        unique.append(axis)
    return unique


def _rotation_axis_values(raw_axis: tuple[float, float, float] | list[float] | None) -> list[float]:
    if raw_axis is None:
        return [0.0, 0.0, 1.0]
    if len(raw_axis) < 3:
        return [0.0, 0.0, 1.0]
    try:
        axis = [float(raw_axis[0]), float(raw_axis[1]), float(raw_axis[2])]
    except (TypeError, ValueError):
        return [0.0, 0.0, 1.0]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return [0.0, 0.0, 1.0]
    return [value / norm for value in axis]


def _rotation_for_angle(Rotation: Any, axis: list[float], angle_deg: float) -> Any | None:
    if Rotation is None or abs(float(angle_deg)) <= 1e-12:
        return None
    try:
        return Rotation.from_axes_angles(axis, float(angle_deg), degrees=True)
    except Exception:  # noqa: BLE001
        return None


def _zone_axis_rotation(Rotation: Any, zone_axis: tuple[int, int, int]) -> Any | None:
    if Rotation is None or zone_axis == (0, 0, 1):
        return None
    target = np.asarray(zone_axis, dtype=np.float64)
    norm = float(np.linalg.norm(target))
    if norm <= 1e-12:
        return None
    target = target / norm
    source = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    rotation_axis = np.cross(source, target)
    axis_norm = float(np.linalg.norm(rotation_axis))
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    angle_deg = degrees(acos(dot))
    if abs(angle_deg) <= 1e-9:
        return None
    if axis_norm <= 1e-12:
        rotation_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        rotation_axis = rotation_axis / axis_norm
    try:
        return Rotation.from_axes_angles(rotation_axis.tolist(), float(angle_deg), degrees=True)
    except Exception:  # noqa: BLE001
        return None


def _combine_rotations(primary: Any | None, secondary: Any | None) -> Any | None:
    if primary is None:
        return secondary
    if secondary is None:
        return primary
    for left, right in ((secondary, primary), (primary, secondary)):
        try:
            return left * right
        except Exception:  # noqa: BLE001
            continue
    return primary


def simulate_diffraction_with_diffsims(
    candidates: list[PhaseCandidate],
    *,
    voltage_kv: float = 200.0,
    reciprocal_radius_inv_angstrom: float = 1.0,
    max_candidates: int = 3,
    pattern_shape: tuple[int, int] | None = None,
    calibration: float = 0.01,
    minimum_simulated_g_inv_nm: float = 1e-6,
    max_simulated_spots: int = 256,
    with_direct_beam: bool = True,
    max_excitation_error: float = 0.01,
    shape_factor_width: float | None = None,
    zone_axis_candidates: object = None,
    rotation_axis: tuple[float, float, float] | list[float] | None = None,
    rotation_angles_deg: list[float] | tuple[float, ...] | None = None,
    fallback_single_atom: bool = True,
    pattern_sigma: float = 10.0,
    pattern_in_plane_angle_deg: float = 0.0,
    pattern_mirrored: bool = False,
    pattern_normalize: bool = True,
    pattern_clip_threshold: float = 1.0,
) -> DiffsimsSimulationResult:
    """Run diffsims for phase candidates when diffsims and support libs exist."""
    provenance: dict[str, Any] = {
        "adapter": "diffsims",
        "active": False,
        "requested_candidate_count": len(candidates),
    }
    try:
        from diffpy.structure import Atom, Lattice, Structure  # type: ignore[import-not-found]
        from diffsims.generators.simulation_generator import (  # type: ignore[import-not-found]
            SimulationGenerator,
        )
        from orix.crystal_map import Phase  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return DiffsimsSimulationResult(
            patterns=[],
            warnings=[f"diffsims adapter unavailable: {exc}"],
            provenance={**provenance, "reason": "import_failed"},
        )
    try:
        from orix.quaternion import Rotation  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        Rotation = None

    warnings: list[str] = []
    patterns: list[SimulatedPattern] = []
    scanned_zone_axes: list[tuple[int, int, int]] = []
    try:
        generator = SimulationGenerator(accelerating_voltage=float(voltage_kv))
    except Exception as exc:  # noqa: BLE001
        return DiffsimsSimulationResult(
            patterns=[],
            warnings=[f"diffsims SimulationGenerator failed: {exc}"],
            provenance={**provenance, "reason": "generator_failed"},
        )

    axis = _rotation_axis_values(rotation_axis)
    angles = _rotation_values(rotation_angles_deg)
    if Rotation is None and any(abs(angle) > 1e-12 for angle in angles):
        warnings.append(
            "diffsims rotation sweep requested but orix.quaternion.Rotation is unavailable; "
            "falling back to identity rotation."
        )
        angles = [0.0]

    for candidate in candidates[: max(1, int(max_candidates))]:
        try:
            _structure, phase, structure_provenance, structure_warnings = (
                _build_structure_and_phase(
                    candidate=candidate,
                    Atom=Atom,
                    Lattice=Lattice,
                    Structure=Structure,
                    Phase=Phase,
                    fallback_single_atom=fallback_single_atom,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"diffsims failed to build structure for '{candidate.name}': {exc}")
            continue
        warnings.extend(structure_warnings)
        if phase is None:
            continue

        default_zone_axis = candidate.zone_axis or (0, 0, 1)
        axes = _zone_axis_values(zone_axis_candidates, default_axis=default_zone_axis)
        for axis_value in axes:
            if axis_value not in scanned_zone_axes:
                scanned_zone_axes.append(axis_value)
        if Rotation is None and any(axis_value != default_zone_axis for axis_value in axes):
            warnings.append(
                "diffsims zone-axis sweep requested but orix.quaternion.Rotation is unavailable; "
                "falling back to the candidate/default zone axis."
            )
            axes = [default_zone_axis]

        for zone_axis_index, zone_axis in enumerate(axes):
            zone_rotation = _zone_axis_rotation(Rotation, zone_axis)
            for rotation_index, angle_deg in enumerate(angles):
                in_plane_rotation = _rotation_for_angle(Rotation, axis, angle_deg)
                rotation = _combine_rotations(zone_rotation, in_plane_rotation)
                try:
                    kwargs: dict[str, Any] = {
                        "phase": phase,
                        "reciprocal_radius": float(reciprocal_radius_inv_angstrom),
                        "with_direct_beam": bool(with_direct_beam),
                        "max_excitation_error": float(max_excitation_error),
                        "show_progressbar": False,
                    }
                    if rotation is not None:
                        kwargs["rotation"] = rotation
                    if shape_factor_width is not None:
                        kwargs["shape_factor_width"] = float(shape_factor_width)
                    simulation = generator.calculate_diffraction2d(**kwargs)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        f"diffsims failed for '{candidate.name}' at zone axis "
                        f"{zone_axis} and rotation {angle_deg:g} deg: {exc}"
                    )
                    continue

                simulated_spots = _extract_simulated_spots(
                    simulation,
                    minimum_g_inv_nm=max(0.0, float(minimum_simulated_g_inv_nm)),
                    max_spots=max(1, int(max_simulated_spots)),
                )

                data_ref = None
                if pattern_shape is not None:
                    try:
                        pattern = simulation.get_diffraction_pattern(
                            shape=tuple(int(v) for v in pattern_shape),
                            calibration=float(calibration),
                            sigma=float(pattern_sigma),
                            in_plane_angle=float(pattern_in_plane_angle_deg),
                            mirrored=bool(pattern_mirrored),
                            normalize=bool(pattern_normalize),
                            clip_threshold=float(pattern_clip_threshold),
                        )
                        pattern_array = np.asarray(pattern, dtype=np.float32)
                        data_ref = ArrayReference(
                            shape=tuple(int(v) for v in pattern_array.shape),
                            dtype=str(pattern_array.dtype),
                            inline=pattern_array.tolist(),
                            description="diffsims simulated diffraction pattern",
                        )
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(
                            f"diffsims pattern image failed for '{candidate.name}' at "
                            f"zone axis {zone_axis} and rotation {angle_deg:g} deg: {exc}"
                        )

                patterns.append(
                    SimulatedPattern(
                        kind="diffraction",
                        data=data_ref,
                        structure_id=candidate.identifier or candidate.name,
                        zone_axis=zone_axis,
                        microscope_parameters={"voltage_kv": float(voltage_kv)},
                        simulation_parameters={
                            "engine": "diffsims",
                            "reciprocal_radius_inv_angstrom": float(
                                reciprocal_radius_inv_angstrom
                            ),
                            "coordinate_count": _extract_coordinate_count(simulation),
                            "simulated_spot_count": len(simulated_spots),
                            "simulated_spot_unit": "1/nm",
                            "qualitative_pattern": data_ref is not None,
                            "with_direct_beam": bool(with_direct_beam),
                            "max_excitation_error": float(max_excitation_error),
                            "shape_factor_width": shape_factor_width,
                            "zone_axis_index": zone_axis_index,
                            "zone_axis": list(zone_axis),
                            "zone_axis_orientation": "approximate_from_001",
                            "rotation_index": rotation_index,
                            "rotation_axis": axis,
                            "rotation_angle_deg": float(angle_deg),
                            **structure_provenance,
                        },
                        spots=simulated_spots,
                        score=candidate.mean_d_error,
                        warnings=[],
                    )
                )

    return DiffsimsSimulationResult(
        patterns=patterns,
        warnings=warnings,
        provenance={
            **provenance,
            "active": bool(patterns),
            "pattern_count": len(patterns),
            "voltage_kv": float(voltage_kv),
            "rotation_axis": axis,
            "rotation_angles_deg": angles,
            "zone_axis_candidates": [list(axis_value) for axis_value in scanned_zone_axes],
            "with_direct_beam": bool(with_direct_beam),
        },
    )
