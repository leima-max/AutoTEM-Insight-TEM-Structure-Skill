"""Optional pymatgen helpers for crystal-structure ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PymatgenCifResult:
    candidate: dict[str, Any] | None
    warnings: list[str]
    provenance: dict[str, Any]


def _site_element(site: Any) -> str | None:
    specie = getattr(site, "specie", None)
    symbol = getattr(specie, "symbol", None)
    if symbol:
        return str(symbol)
    species = getattr(site, "species", None)
    try:
        items = list(species.items())
    except Exception:  # noqa: BLE001
        items = []
    if items:
        element, _occupancy = max(items, key=lambda item: float(item[1]))
        symbol = getattr(element, "symbol", None)
        return str(symbol or element)
    return None


def _extract_atom_sites(structure: Any) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    try:
        iterator = iter(structure)
    except TypeError:
        return sites
    for idx, site in enumerate(iterator):
        element = _site_element(site)
        frac_coords = getattr(site, "frac_coords", None)
        if element is None or frac_coords is None:
            continue
        try:
            coords = [round(float(value), 8) for value in list(frac_coords)[:3]]
        except (TypeError, ValueError):
            continue
        if len(coords) != 3:
            continue
        occupancy = 1.0
        species = getattr(site, "species", None)
        try:
            species_items = list(species.items())
        except Exception:  # noqa: BLE001
            species_items = []
        if species_items:
            try:
                occupancy = float(max(species_items, key=lambda item: float(item[1]))[1])
            except (TypeError, ValueError):
                occupancy = 1.0
        sites.append(
            {
                "element": element,
                "abc": coords,
                "occupancy": occupancy,
                "source_index": idx,
            }
        )
    return sites


def load_candidate_phase_from_cif(
    path: str | Path,
    *,
    name: str | None = None,
) -> PymatgenCifResult:
    """Read a CIF through pymatgen and return B4 candidate_phase_params."""
    cif_path = Path(path)
    provenance: dict[str, Any] = {
        "adapter": "pymatgen",
        "path": str(cif_path),
        "active": False,
    }
    try:
        from pymatgen.core import Structure  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return PymatgenCifResult(
            candidate=None,
            warnings=[f"pymatgen is not available for CIF parsing: {exc}"],
            provenance={**provenance, "reason": "import_failed"},
        )

    try:
        structure = Structure.from_file(str(cif_path))
    except Exception as exc:  # noqa: BLE001
        return PymatgenCifResult(
            candidate=None,
            warnings=[f"pymatgen failed to parse CIF '{cif_path}': {exc}"],
            provenance={**provenance, "reason": "parse_failed"},
        )

    lattice = structure.lattice
    formula = ""
    try:
        formula = str(structure.composition.reduced_formula)
    except Exception:  # noqa: BLE001
        formula = ""

    space_group = "P1"
    space_group_number = None
    try:
        sg_symbol, sg_number = structure.get_space_group_info()
        space_group = str(sg_symbol)
        space_group_number = int(sg_number)
    except Exception:  # noqa: BLE001
        pass
    atom_sites = _extract_atom_sites(structure)

    # pymatgen lattice lengths are Angstrom; B4 uses nm.
    candidate = {
        "name": name or cif_path.stem,
        "formula": formula or None,
        "a": round(float(lattice.a) * 0.1, 8),
        "b": round(float(lattice.b) * 0.1, 8),
        "c": round(float(lattice.c) * 0.1, 8),
        "alpha": round(float(lattice.alpha), 8),
        "beta": round(float(lattice.beta), 8),
        "gamma": round(float(lattice.gamma), 8),
        "space_group": space_group,
        "source": "pymatgen_cif",
        "identifier": cif_path.name,
    }
    if space_group_number is not None:
        candidate["space_group_number"] = space_group_number
    if atom_sites:
        candidate["atom_sites"] = atom_sites

    return PymatgenCifResult(
        candidate=candidate,
        warnings=[],
        provenance={**provenance, "active": True, "space_group_number": space_group_number},
    )
