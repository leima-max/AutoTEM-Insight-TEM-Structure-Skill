"""Phase candidate lookup helpers for B4 indexing.

The lookup layer intentionally has a deterministic local path for tests and
offline use, plus optional online sources for broader candidate discovery.
"""

from __future__ import annotations

import json
import os
import re
import typing
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class PhaseLookupResult:
    candidates: list[dict[str, Any]]
    warnings: list[str]
    provenance: dict[str, Any]


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")
_MP_API_DASHBOARD_URL = "https://next-gen.materialsproject.org/api"
_SUPPORTED_SOURCES = {
    "local_builtin",
    "materials_project_api",
    "materials_project_optimade",
    "cod",
}
_MP_API_SORT_MODES = {"stability", "energy_above_hull", "material_id", "none"}
_COMMON_MATERIAL_FORMULA_ALIASES: dict[str, str] = {
    "alumina": "Al2O3",
    "aluminum oxide": "Al2O3",
    "aluminium oxide": "Al2O3",
    "barium titanate": "BaTiO3",
    "ceria": "CeO2",
    "cerium dioxide": "CeO2",
    "diamond": "C",
    "gallium arsenide": "GaAs",
    "gallium nitride": "GaN",
    "graphite": "C",
    "hematite": "Fe2O3",
    "magnesia": "MgO",
    "magnesium oxide": "MgO",
    "magnetite": "Fe3O4",
    "molybdenum disulfide": "MoS2",
    "molybdenum disulphide": "MoS2",
    "quartz": "SiO2",
    "sapphire": "Al2O3",
    "silica": "SiO2",
    "silicon carbide": "SiC",
    "silicon dioxide": "SiO2",
    "sodium chloride": "NaCl",
    "strontium titanate": "SrTiO3",
    "tungsten disulfide": "WS2",
    "tungsten disulphide": "WS2",
    "wustite": "FeO",
    "zirconia": "ZrO2",
    "zirconium dioxide": "ZrO2",
}
_FCC_MONOATOMIC_SITES: list[dict[str, Any]] = [
    {"abc": [0.0, 0.0, 0.0], "occupancy": 1.0},
    {"abc": [0.0, 0.5, 0.5], "occupancy": 1.0},
    {"abc": [0.5, 0.0, 0.5], "occupancy": 1.0},
    {"abc": [0.5, 0.5, 0.0], "occupancy": 1.0},
]
_SI_DIAMOND_SITES: list[dict[str, Any]] = [
    {"element": "Si", "abc": [0.0, 0.0, 0.0], "occupancy": 1.0},
    {"element": "Si", "abc": [0.0, 0.5, 0.5], "occupancy": 1.0},
    {"element": "Si", "abc": [0.5, 0.0, 0.5], "occupancy": 1.0},
    {"element": "Si", "abc": [0.5, 0.5, 0.0], "occupancy": 1.0},
    {"element": "Si", "abc": [0.25, 0.25, 0.25], "occupancy": 1.0},
    {"element": "Si", "abc": [0.25, 0.75, 0.75], "occupancy": 1.0},
    {"element": "Si", "abc": [0.75, 0.25, 0.75], "occupancy": 1.0},
    {"element": "Si", "abc": [0.75, 0.75, 0.25], "occupancy": 1.0},
]
_RUTILE_TIO2_SITES: list[dict[str, Any]] = [
    {"element": "Ti", "abc": [0.0, 0.0, 0.0], "occupancy": 1.0},
    {"element": "Ti", "abc": [0.5, 0.5, 0.5], "occupancy": 1.0},
    {"element": "O", "abc": [0.305, 0.305, 0.0], "occupancy": 1.0},
    {"element": "O", "abc": [0.695, 0.695, 0.0], "occupancy": 1.0},
    {"element": "O", "abc": [0.805, 0.195, 0.5], "occupancy": 1.0},
    {"element": "O", "abc": [0.195, 0.805, 0.5], "occupancy": 1.0},
]
_LOCAL_BUILTIN_CANDIDATES: dict[str, dict[str, Any]] = {
    "silicon": {
        "name": "Silicon",
        "formula": "Si",
        "a": 0.5431,
        "b": 0.5431,
        "c": 0.5431,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fd-3m",
        "space_group_number": 227,
        "atom_sites": _SI_DIAMOND_SITES,
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Si:diamond",
    },
    "si": {
        "name": "Silicon",
        "formula": "Si",
        "a": 0.5431,
        "b": 0.5431,
        "c": 0.5431,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fd-3m",
        "space_group_number": 227,
        "atom_sites": _SI_DIAMOND_SITES,
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Si:diamond",
    },
    "gold": {
        "name": "Gold",
        "formula": "Au",
        "a": 0.4078,
        "b": 0.4078,
        "c": 0.4078,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fm-3m",
        "space_group_number": 225,
        "atom_sites": [
            {**site, "element": "Au"} for site in _FCC_MONOATOMIC_SITES
        ],
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Au:fcc",
    },
    "au": {
        "name": "Gold",
        "formula": "Au",
        "a": 0.4078,
        "b": 0.4078,
        "c": 0.4078,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fm-3m",
        "space_group_number": 225,
        "atom_sites": [
            {**site, "element": "Au"} for site in _FCC_MONOATOMIC_SITES
        ],
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Au:fcc",
    },
    "copper": {
        "name": "Copper",
        "formula": "Cu",
        "a": 0.3615,
        "b": 0.3615,
        "c": 0.3615,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fm-3m",
        "space_group_number": 225,
        "atom_sites": [
            {**site, "element": "Cu"} for site in _FCC_MONOATOMIC_SITES
        ],
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Cu:fcc",
    },
    "cu": {
        "name": "Copper",
        "formula": "Cu",
        "a": 0.3615,
        "b": 0.3615,
        "c": 0.3615,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fm-3m",
        "space_group_number": 225,
        "atom_sites": [
            {**site, "element": "Cu"} for site in _FCC_MONOATOMIC_SITES
        ],
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Cu:fcc",
    },
    "aluminum": {
        "name": "Aluminum",
        "formula": "Al",
        "a": 0.4050,
        "b": 0.4050,
        "c": 0.4050,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fm-3m",
        "space_group_number": 225,
        "atom_sites": [
            {**site, "element": "Al"} for site in _FCC_MONOATOMIC_SITES
        ],
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Al:fcc",
    },
    "aluminium": {
        "name": "Aluminum",
        "formula": "Al",
        "a": 0.4050,
        "b": 0.4050,
        "c": 0.4050,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fm-3m",
        "space_group_number": 225,
        "atom_sites": [
            {**site, "element": "Al"} for site in _FCC_MONOATOMIC_SITES
        ],
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Al:fcc",
    },
    "al": {
        "name": "Aluminum",
        "formula": "Al",
        "a": 0.4050,
        "b": 0.4050,
        "c": 0.4050,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "Fm-3m",
        "space_group_number": 225,
        "atom_sites": [
            {**site, "element": "Al"} for site in _FCC_MONOATOMIC_SITES
        ],
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:Al:fcc",
    },
    "titanium dioxide": {
        "name": "Titanium Dioxide (rutile)",
        "formula": "TiO2",
        "a": 0.4594,
        "b": 0.4594,
        "c": 0.2959,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "P42/mnm",
        "space_group_number": 136,
        "atom_sites": _RUTILE_TIO2_SITES,
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:TiO2:rutile",
    },
    "tio2": {
        "name": "Titanium Dioxide (rutile)",
        "formula": "TiO2",
        "a": 0.4594,
        "b": 0.4594,
        "c": 0.2959,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "P42/mnm",
        "space_group_number": 136,
        "atom_sites": _RUTILE_TIO2_SITES,
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:TiO2:rutile",
    },
    "rutile": {
        "name": "Titanium Dioxide (rutile)",
        "formula": "TiO2",
        "a": 0.4594,
        "b": 0.4594,
        "c": 0.2959,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "space_group": "P42/mnm",
        "space_group_number": 136,
        "atom_sites": _RUTILE_TIO2_SITES,
        "source": "local_builtin_crystal_library",
        "identifier": "builtin:TiO2:rutile",
    },
}


def supported_phase_lookup_sources() -> set[str]:
    """Return supported B4 lookup source identifiers."""
    return set(_SUPPORTED_SOURCES)


def materials_project_api_config_path() -> Path:
    """Return the default local Materials Project API config path."""
    override = os.environ.get("AUTOTEM_MP_API_CONFIG")
    if override:
        return Path(override).expanduser()
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "AutoTEM-Insight" / "materials_project.json"
    return Path.home() / ".config" / "autotem-insight" / "materials_project.json"


def _read_materials_project_api_config(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def resolve_materials_project_api_key(
    *,
    explicit_api_key: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Resolve the Materials Project API key without exposing it in provenance."""
    if explicit_api_key and explicit_api_key.strip():
        return explicit_api_key.strip(), {"source": "explicit", "configured": True}
    env_key = os.environ.get("MP_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip(), {"source": "MP_API_KEY", "configured": True}

    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else materials_project_api_config_path()
    )
    config = _read_materials_project_api_config(path)
    raw_key = config.get("api_key")
    if isinstance(raw_key, str) and raw_key.strip():
        return raw_key.strip(), {
            "source": "config_file",
            "configured": True,
            "config_path": str(path),
        }
    return None, {"source": "missing", "configured": False, "config_path": str(path)}


def materials_project_api_setup_guidance(config_path: str | Path | None = None) -> str:
    """Return first-use setup guidance for Materials Project API access."""
    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else materials_project_api_config_path()
    )
    return (
        "Materials Project API key is not configured. Get a key from "
        f"{_MP_API_DASHBOARD_URL}, then set environment variable MP_API_KEY or run "
        "`autotem configure-mp-api --api-key-stdin` to store it in the local AutoTEM "
        f"config file ({path})."
    )


def write_materials_project_api_config(
    api_key: str,
    *,
    config_path: str | Path | None = None,
) -> Path:
    """Write the Materials Project API key to the local AutoTEM config file."""
    if not api_key.strip():
        raise ValueError("Materials Project API key cannot be empty.")
    path = (
        Path(config_path).expanduser()
        if config_path is not None
        else materials_project_api_config_path()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"api_key": api_key.strip()}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _fetch_json(url: str, *, timeout_s: int) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AutoTEM-Insight/0.1 (+https://github.com/leima-max/AutoTEM-Insight-TEM-Structure-Skill)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
        payload = resp.read()
    return json.loads(payload.decode("utf-8"))


def _parse_formula_counts(formula: str) -> dict[str, float]:
    text = str(formula).strip().replace(" ", "")
    if not text:
        return {}
    parts = _FORMULA_RE.findall(text)
    joined = "".join(f"{sym}{num}" for sym, num in parts)
    if not parts or joined != text:
        return {}
    counts: dict[str, float] = {}
    for symbol, num_text in parts:
        count = 1.0 if num_text == "" else float(num_text)
        counts[symbol] = counts.get(symbol, 0.0) + count
    return counts


def _normalize_query_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def _compact_query_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _candidate_copy(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


def _lookup_local_builtin_candidates(
    *,
    formula: str | None,
    material_names: list[str],
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []

    queries: list[str] = []
    if formula:
        queries.append(str(formula))
    queries.extend(material_names)

    for raw in queries:
        normal = _normalize_query_text(raw)
        compact = _compact_query_text(raw)
        candidate = _LOCAL_BUILTIN_CANDIDATES.get(normal) or _LOCAL_BUILTIN_CANDIDATES.get(compact)
        if candidate is None:
            continue
        rows.append(_candidate_copy(candidate))
        if len(rows) >= max(1, int(limit)):
            break

    if not rows and queries:
        warnings.append("Local built-in crystal lookup returned no candidates.")
    return rows, warnings


def _material_name_to_formula(name: str) -> str | None:
    normal = _normalize_query_text(name)
    compact = _compact_query_text(name)
    candidate = _LOCAL_BUILTIN_CANDIDATES.get(normal)
    if candidate is None:
        candidate = _LOCAL_BUILTIN_CANDIDATES.get(compact)
    if candidate is not None and candidate.get("formula"):
        return str(candidate["formula"])
    alias = _COMMON_MATERIAL_FORMULA_ALIASES.get(normal)
    if alias is None:
        alias = _COMMON_MATERIAL_FORMULA_ALIASES.get(compact)
    if alias:
        return alias
    if _parse_formula_counts(name):
        return str(name).strip()
    return None


def _material_name_to_mp_id(name: str) -> str | None:
    token = str(name).strip().lower()
    match = re.fullmatch(r"mp-\d+", token)
    return token if match else None


def _append_unique(values: list[str], value: str | None) -> None:
    if value is None:
        return
    cleaned = str(value).strip()
    if cleaned and cleaned not in values:
        values.append(cleaned)


def _format_count(count: float) -> str:
    if abs(count - round(count)) < 1e-8:
        iv = int(round(count))
        return "" if iv == 1 else str(iv)
    return f"{count:g}"


def _to_reduced_formula(formula: str) -> str | None:
    counts = _parse_formula_counts(formula)
    if not counts:
        return None
    parts = []
    for symbol in sorted(counts):
        parts.append(f"{symbol}{_format_count(counts[symbol])}")
    return "".join(parts)


def _to_hill_formula_for_cod(formula: str) -> str | None:
    counts = _parse_formula_counts(formula)
    if not counts:
        return None

    symbols: list[str]
    if "C" in counts:
        symbols = ["C"]
        if "H" in counts:
            symbols.append("H")
        symbols.extend(sorted(sym for sym in counts if sym not in {"C", "H"}))
    else:
        symbols = sorted(counts)

    tokens = [f"{sym}{_format_count(counts[sym])}" for sym in symbols]
    return " ".join(tokens)


def _lattice_vectors_to_cell(
    vectors: list[list[float]],
) -> tuple[float, float, float, float, float, float] | None:
    try:
        mat = np.asarray(vectors, dtype=np.float64)
        if mat.shape != (3, 3):
            return None
    except (TypeError, ValueError):
        return None

    a_vec = mat[0]
    b_vec = mat[1]
    c_vec = mat[2]
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    c = float(np.linalg.norm(c_vec))
    if min(a, b, c) <= 1e-9:
        return None

    alpha = float(np.degrees(np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1.0, 1.0))))
    beta = float(np.degrees(np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1.0, 1.0))))
    gamma = float(np.degrees(np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1.0, 1.0))))
    return a, b, c, alpha, beta, gamma


def _cell_lengths_angstrom_to_nm(
    a: float,
    b: float,
    c: float,
) -> tuple[float, float, float]:
    return a * 0.1, b * 0.1, c * 0.1


def _fetch_mp_candidates(
    *,
    formula: str | None,
    elements: list[str],
    limit: int,
    timeout_s: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    candidates: list[dict[str, Any]] = []
    base_url = "https://optimade.materialsproject.org/v1/structures"

    filters: list[str] = []
    reduced = _to_reduced_formula(formula) if formula else None
    if reduced:
        filters.append(f'chemical_formula_reduced="{reduced}"')
    if elements:
        quoted = ",".join(f'"{elem}"' for elem in elements if elem)
        if quoted:
            filters.append(f"elements HAS ALL {quoted}")
    if not filters:
        return [], ["No formula/elements provided for Materials Project query."]

    query_filter = filters[0]
    encoded_filter = urllib.parse.quote(query_filter, safe='=,"')
    query_url = (
        f"{base_url}?filter={encoded_filter}"
        f"&page_limit={max(1, int(limit))}"
    )
    try:
        payload = _fetch_json(query_url, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        return [], [f"Materials Project query failed: {exc}"]

    data = payload.get("data", []) if isinstance(payload, dict) else []
    if not data and len(filters) > 1:
        fallback_filter = filters[1]
        encoded_fallback = urllib.parse.quote(fallback_filter, safe='=,"')
        fallback_url = (
            f"{base_url}?filter={encoded_fallback}"
            f"&page_limit={max(1, int(limit))}"
        )
        try:
            payload = _fetch_json(fallback_url, timeout_s=timeout_s)
            data = payload.get("data", []) if isinstance(payload, dict) else []
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Materials Project fallback query failed: {exc}")

    for row in data[: max(1, int(limit))]:
        attrs = row.get("attributes", {}) if isinstance(row, dict) else {}
        cell = _lattice_vectors_to_cell(attrs.get("lattice_vectors", []))
        if cell is None:
            continue
        a, b, c, alpha, beta, gamma = cell
        # OPTIMADE lattice vectors are in Angstrom; convert to nm.
        a, b, c = _cell_lengths_angstrom_to_nm(a, b, c)
        candidates.append(
            {
                "name": str(row.get("id", attrs.get("chemical_formula_reduced", "mp_candidate"))),
                "formula": attrs.get("chemical_formula_reduced"),
                "a": round(a, 6),
                "b": round(b, 6),
                "c": round(c, 6),
                "alpha": round(alpha, 6),
                "beta": round(beta, 6),
                "gamma": round(gamma, 6),
                "space_group": str(attrs.get("space_group_symbol", "P1")),
                "source": "materials_project_optimade",
                "identifier": str(row.get("id", "")),
            }
        )
    if not candidates:
        warnings.append("Materials Project returned no valid lattice candidates.")
    return candidates, warnings


def _doc_get(doc: Any, key: str, default: Any = None) -> Any:
    if isinstance(doc, dict):
        return doc.get(key, default)
    return getattr(doc, key, default)


def _doc_float(doc: Any, key: str) -> float | None:
    value = _doc_get(doc, key)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _doc_bool(doc: Any, key: str) -> bool | None:
    value = _doc_get(doc, key)
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y"}:
            return True
        if token in {"0", "false", "no", "n"}:
            return False
    return None


def _structure_site_count(structure: Any) -> int | None:
    try:
        return int(len(structure))
    except Exception:  # noqa: BLE001
        pass
    sites = getattr(structure, "sites", None)
    try:
        return int(len(sites))
    except Exception:  # noqa: BLE001
        return None


def _structure_to_candidate_cell(
    structure: Any,
) -> tuple[float, float, float, float, float, float] | None:
    lattice = getattr(structure, "lattice", None)
    if lattice is None:
        return None
    try:
        return (
            float(lattice.a) * 0.1,
            float(lattice.b) * 0.1,
            float(lattice.c) * 0.1,
            float(lattice.alpha),
            float(lattice.beta),
            float(lattice.gamma),
        )
    except (TypeError, ValueError, AttributeError):
        return None


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
    if not items:
        return None
    element, _occupancy = max(items, key=lambda item: float(item[1]))
    symbol = getattr(element, "symbol", None)
    return str(symbol or element)


def _structure_to_atom_sites(structure: Any) -> list[dict[str, Any]]:
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


def _symmetry_value(symmetry: Any, *names: str) -> Any:
    for name in names:
        if isinstance(symmetry, dict) and name in symmetry:
            return symmetry[name]
        value = getattr(symmetry, name, None)
        if value is not None:
            return value
    return None


def _patch_typing_extensions_for_mp_api() -> None:
    """Patch Python 3.10 typing gaps used by current mp-api transitive imports."""
    if hasattr(typing, "NotRequired"):
        return
    try:
        from typing_extensions import NotRequired, Required  # type: ignore[import-not-found]
    except Exception:
        return
    typing.NotRequired = NotRequired  # type: ignore[attr-defined]
    if not hasattr(typing, "Required"):
        typing.Required = Required  # type: ignore[attr-defined]


def _mp_id_rank(material_id: str, requested_ids: list[str]) -> int:
    normalized = str(material_id).strip().lower()
    for idx, requested_id in enumerate(requested_ids):
        if normalized == str(requested_id).strip().lower():
            return idx
    return len(requested_ids) + 1


def _mp_doc_sort_key(
    doc: Any,
    *,
    material_ids: list[str],
    sort_by: str,
    prefer_experimental: bool,
) -> tuple[object, ...]:
    material_id = str(_doc_get(doc, "material_id", ""))
    energy_above_hull = _doc_float(doc, "energy_above_hull")
    is_stable = _doc_bool(doc, "is_stable")
    deprecated = _doc_bool(doc, "deprecated")
    theoretical = _doc_bool(doc, "theoretical")
    structure = _doc_get(doc, "structure")
    site_count = _structure_site_count(structure)

    if sort_by == "none":
        return (0,)
    if sort_by == "material_id":
        return (_mp_id_rank(material_id, material_ids), material_id)

    stable_rank = 0 if is_stable is True else 1 if is_stable is False else 2
    deprecated_rank = 1 if deprecated is True else 0
    theoretical_rank = 1 if prefer_experimental and theoretical is True else 0
    energy_rank = energy_above_hull if energy_above_hull is not None else float("inf")
    site_rank = site_count if site_count is not None else 10**9

    if sort_by == "energy_above_hull":
        return (
            _mp_id_rank(material_id, material_ids),
            deprecated_rank,
            energy_rank,
            stable_rank,
            theoretical_rank,
            site_rank,
            material_id,
        )
    return (
        _mp_id_rank(material_id, material_ids),
        deprecated_rank,
        stable_rank,
        energy_rank,
        theoretical_rank,
        site_rank,
        material_id,
    )


def _filter_mp_docs(
    docs: list[Any],
    *,
    include_deprecated: bool,
    filter_stable: bool,
    max_energy_above_hull: float | None,
) -> tuple[list[Any], dict[str, Any]]:
    filtered: list[Any] = []
    stats = {
        "input_count": len(docs),
        "deprecated_removed_count": 0,
        "unstable_removed_count": 0,
        "energy_above_hull_removed_count": 0,
        "missing_energy_above_hull_count": 0,
    }
    for doc in docs:
        deprecated = _doc_bool(doc, "deprecated")
        if deprecated is True and not include_deprecated:
            stats["deprecated_removed_count"] += 1
            continue
        is_stable = _doc_bool(doc, "is_stable")
        if filter_stable and is_stable is not True:
            stats["unstable_removed_count"] += 1
            continue
        energy_above_hull = _doc_float(doc, "energy_above_hull")
        if max_energy_above_hull is not None:
            if energy_above_hull is None:
                stats["missing_energy_above_hull_count"] += 1
            elif energy_above_hull > max_energy_above_hull:
                stats["energy_above_hull_removed_count"] += 1
                continue
        filtered.append(doc)
    stats["output_count"] = len(filtered)
    return filtered, stats


def _mp_candidate_metadata(doc: Any, structure: Any, *, rank: int) -> dict[str, Any]:
    metadata = {
        "rank": rank,
        "energy_above_hull": _doc_float(doc, "energy_above_hull"),
        "is_stable": _doc_bool(doc, "is_stable"),
        "theoretical": _doc_bool(doc, "theoretical"),
        "deprecated": _doc_bool(doc, "deprecated"),
        "site_count": _structure_site_count(structure),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _fetch_mp_api_candidates(
    *,
    formula: str | None,
    elements: list[str],
    material_ids: list[str],
    limit: int,
    timeout_s: int,
    api_key: str | None,
    config_path: str | Path | None,
    sort_by: str = "stability",
    include_deprecated: bool = False,
    filter_stable: bool = False,
    max_energy_above_hull: float | None = None,
    prefer_experimental: bool = True,
    candidate_pool_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    warnings: list[str] = []
    provenance: dict[str, Any] = {
        "source": "materials_project_api",
        "api_key_configured": False,
    }
    try:
        _patch_typing_extensions_for_mp_api()
        from mp_api.client import MPRester  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return [], [
            "mp-api is not installed. Install it with "
            '`pip install "autotem-insight[crystallography]"` before using '
            f"materials_project_api. Import error: {exc}"
        ], {**provenance, "reason": "import_failed"}

    resolved_key, key_info = resolve_materials_project_api_key(
        explicit_api_key=api_key,
        config_path=config_path,
    )
    provenance["api_key_configured"] = bool(key_info.get("configured"))
    provenance["api_key_source"] = key_info.get("source")
    if key_info.get("config_path"):
        provenance["config_path"] = key_info["config_path"]
    if not resolved_key:
        return [], [materials_project_api_setup_guidance(config_path)], {
            **provenance,
            "reason": "missing_api_key",
        }

    clean_sort_by = str(sort_by).strip().lower()
    if clean_sort_by not in _MP_API_SORT_MODES:
        clean_sort_by = "stability"
    pool_size = max(1, int(candidate_pool_size or max(limit, limit * 5)))
    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "symmetry",
        "energy_above_hull",
        "is_stable",
        "theoretical",
        "deprecated",
    ]
    search_kwargs: dict[str, Any] = {
        "fields": fields,
        "all_fields": False,
        "num_chunks": 1,
        "chunk_size": pool_size,
    }
    if material_ids:
        search_kwargs["material_ids"] = material_ids[: max(1, int(limit))]
    elif formula:
        search_kwargs["formula"] = formula
    elif elements:
        search_kwargs["elements"] = elements
    else:
        return [], ["Materials Project API lookup requires formula, elements, or material IDs."], {
            **provenance,
            "reason": "missing_query",
        }

    try:
        with MPRester(
            api_key=resolved_key,
            mute_progress_bars=True,
            include_user_agent=True,
        ) as mpr:
            docs = mpr.materials.summary.search(**search_kwargs)
    except TypeError:
        search_kwargs.pop("num_chunks", None)
        search_kwargs.pop("chunk_size", None)
        try:
            with MPRester(
                api_key=resolved_key,
                mute_progress_bars=True,
                include_user_agent=True,
            ) as mpr:
                docs = mpr.materials.summary.search(**search_kwargs)
        except Exception as exc:  # noqa: BLE001
            return [], [f"Materials Project API query failed: {exc}"], {
                **provenance,
                "reason": "query_failed",
            }
    except Exception as exc:  # noqa: BLE001
        return [], [f"Materials Project API query failed: {exc}"], {
            **provenance,
            "reason": "query_failed",
        }

    raw_docs = list(docs)
    filtered_docs, filter_stats = _filter_mp_docs(
        raw_docs,
        include_deprecated=include_deprecated,
        filter_stable=filter_stable,
        max_energy_above_hull=max_energy_above_hull,
    )
    ranked_docs = sorted(
        filtered_docs,
        key=lambda doc: _mp_doc_sort_key(
            doc,
            material_ids=material_ids,
            sort_by=clean_sort_by,
            prefer_experimental=prefer_experimental,
        ),
    )

    candidates: list[dict[str, Any]] = []
    for rank, doc in enumerate(ranked_docs[: max(1, int(limit))]):
        structure = _doc_get(doc, "structure")
        cell = _structure_to_candidate_cell(structure)
        if cell is None:
            continue
        a, b, c, alpha, beta, gamma = cell
        atom_sites = _structure_to_atom_sites(structure)
        material_id = str(_doc_get(doc, "material_id", ""))
        formula_pretty = _doc_get(doc, "formula_pretty")
        symmetry = _doc_get(doc, "symmetry", {})
        space_group = _symmetry_value(symmetry, "symbol", "spacegroup_symbol") or "P1"
        space_group_number = _symmetry_value(symmetry, "number", "spacegroup_number")
        candidate = {
            "name": material_id or str(formula_pretty or "mp_api_candidate"),
            "formula": formula_pretty,
            "a": round(a, 6),
            "b": round(b, 6),
            "c": round(c, 6),
            "alpha": round(alpha, 6),
            "beta": round(beta, 6),
            "gamma": round(gamma, 6),
            "space_group": str(space_group),
            "source": "materials_project_api",
            "identifier": material_id,
        }
        if space_group_number is not None:
            candidate["space_group_number"] = space_group_number
        if atom_sites:
            candidate["atom_sites"] = atom_sites
        metadata = _mp_candidate_metadata(doc, structure, rank=rank)
        if metadata:
            candidate["materials_project"] = metadata
        candidates.append(candidate)
    if not candidates:
        warnings.append("Materials Project API returned no valid structure candidates.")
    return candidates, warnings, {
        **provenance,
        "candidate_count": len(candidates),
        "candidate_count_before_filter": len(raw_docs),
        "filter": {
            "include_deprecated": bool(include_deprecated),
            "filter_stable": bool(filter_stable),
            "max_energy_above_hull": max_energy_above_hull,
            **filter_stats,
        },
        "ranking": {
            "sort_by": clean_sort_by,
            "prefer_experimental": bool(prefer_experimental),
            "candidate_pool_size": pool_size,
        },
        "query": {key: value for key, value in search_kwargs.items() if key != "fields"},
    }


def _fetch_cod_candidates(
    *,
    formula: str | None,
    limit: int,
    timeout_s: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not formula:
        return [], ["COD lookup currently requires a formula input."]
    hill = _to_hill_formula_for_cod(formula)
    if not hill:
        return [], ["COD lookup: formula parse failed."]

    url = (
        "https://www.crystallography.net/cod/result"
        f"?formula={urllib.parse.quote(hill)}&format=json"
    )
    try:
        payload = _fetch_json(url, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        return [], [f"COD query failed: {exc}"]

    if not isinstance(payload, list):
        return [], ["COD query returned unexpected payload."]

    candidates: list[dict[str, Any]] = []
    for row in payload[: max(1, int(limit))]:
        if not isinstance(row, dict):
            continue
        try:
            a = float(row.get("a"))
            b = float(row.get("b"))
            c = float(row.get("c"))
            alpha = float(row.get("alpha"))
            beta = float(row.get("beta"))
            gamma = float(row.get("gamma"))
        except (TypeError, ValueError):
            continue
        # COD cell parameters are reported in Angstrom; convert to nm.
        a, b, c = _cell_lengths_angstrom_to_nm(a, b, c)
        cod_id = str(row.get("file", ""))
        if not cod_id:
            continue
        candidates.append(
            {
                "name": f"COD_{cod_id}",
                "formula": row.get("formula", hill),
                "a": round(a, 6),
                "b": round(b, 6),
                "c": round(c, 6),
                "alpha": round(alpha, 6),
                "beta": round(beta, 6),
                "gamma": round(gamma, 6),
                "space_group": str(row.get("sg", "P1")),
                "source": "crystallography_open_database",
                "identifier": cod_id,
                "cif_url": f"https://www.crystallography.net/cod/{cod_id}.cif",
            }
        )
    if not candidates:
        warnings.append("COD returned no valid lattice candidates.")
    return candidates, warnings


def lookup_phase_candidates(
    *,
    formula: str | None,
    elements: list[str] | None,
    material_names: list[str] | None = None,
    sources: list[str] | None = None,
    limit_per_source: int = 10,
    timeout_s: int = 20,
    materials_project_api_key: str | None = None,
    materials_project_config_path: str | Path | None = None,
    materials_project_ids: list[str] | None = None,
    materials_project_sort_by: str = "stability",
    materials_project_include_deprecated: bool = False,
    materials_project_filter_stable: bool = False,
    materials_project_max_energy_above_hull: float | None = None,
    materials_project_prefer_experimental: bool = True,
    materials_project_candidate_pool_size: int | None = None,
) -> PhaseLookupResult:
    """Return B4 candidate_phase_params entries from local and optional online sources."""
    active_sources = sources or ["local_builtin", "materials_project_optimade", "cod"]
    active_sources = [str(item).strip() for item in active_sources if str(item).strip()]
    clean_elements = [str(item).strip() for item in (elements or []) if str(item).strip()]
    clean_materials = [
        str(item).strip() for item in (material_names or []) if str(item).strip()
    ]
    clean_mp_ids = [
        str(item).strip() for item in (materials_project_ids or []) if str(item).strip()
    ]
    warnings: list[str] = []
    merged: list[dict[str, Any]] = []
    resolved_formulas: list[str] = []
    resolved_material_queries: list[dict[str, str]] = []
    provenance: dict[str, Any] = {
        "formula": None if formula is None else str(formula).strip(),
        "elements": clean_elements,
        "sources": list(active_sources),
        "material_names": clean_materials,
        "materials_project_ids": clean_mp_ids,
    }

    for source in active_sources:
        if source not in _SUPPORTED_SOURCES:
            warnings.append(f"Unsupported phase lookup source skipped: {source}")

    if "local_builtin" in active_sources:
        local_rows, local_warnings = _lookup_local_builtin_candidates(
            formula=formula,
            material_names=clean_materials,
            limit=max(1, int(limit_per_source)),
        )
        merged.extend(local_rows)
        warnings.extend(local_warnings)

    query_formulas: list[str] = []
    if formula:
        _append_unique(query_formulas, str(formula).strip())
    for material in clean_materials:
        mp_id = _material_name_to_mp_id(material)
        if mp_id:
            _append_unique(clean_mp_ids, mp_id)
            resolved_material_queries.append(
                {"material": material, "query_type": "materials_project_id", "value": mp_id}
            )
            continue
        resolved_formula = _material_name_to_formula(material)
        if resolved_formula:
            _append_unique(resolved_formulas, resolved_formula)
            _append_unique(query_formulas, resolved_formula)
            resolved_material_queries.append(
                {"material": material, "query_type": "formula", "value": resolved_formula}
            )
        elif any(source != "local_builtin" for source in active_sources):
            warnings.append(
                f"Material name '{material}' could not be resolved to a formula for online lookup."
            )
    provenance["resolved_formulas"] = resolved_formulas
    provenance["resolved_material_queries"] = resolved_material_queries
    provenance["materials_project_ids"] = clean_mp_ids
    online_query_formulas: list[str | None] = list(query_formulas) or [None]

    if "materials_project_optimade" in active_sources:
        for query_formula in online_query_formulas:
            mp_rows, mp_warnings = _fetch_mp_candidates(
                formula=query_formula,
                elements=clean_elements,
                limit=max(1, int(limit_per_source)),
                timeout_s=max(5, int(timeout_s)),
            )
            merged.extend(mp_rows)
            warnings.extend(mp_warnings)

    if "materials_project_api" in active_sources:
        if clean_mp_ids:
            mp_api_rows, mp_api_warnings, mp_api_provenance = _fetch_mp_api_candidates(
                formula=None,
                elements=[],
                material_ids=clean_mp_ids,
                limit=max(1, int(limit_per_source)),
                timeout_s=max(5, int(timeout_s)),
                api_key=materials_project_api_key,
                config_path=materials_project_config_path,
                sort_by=materials_project_sort_by,
                include_deprecated=materials_project_include_deprecated,
                filter_stable=materials_project_filter_stable,
                max_energy_above_hull=materials_project_max_energy_above_hull,
                prefer_experimental=materials_project_prefer_experimental,
                candidate_pool_size=materials_project_candidate_pool_size,
            )
            merged.extend(mp_api_rows)
            warnings.extend(mp_api_warnings)
            provenance.setdefault("materials_project_api", []).append(mp_api_provenance)
        for query_formula in query_formulas:
            mp_api_rows, mp_api_warnings, mp_api_provenance = _fetch_mp_api_candidates(
                formula=query_formula,
                elements=[],
                material_ids=[],
                limit=max(1, int(limit_per_source)),
                timeout_s=max(5, int(timeout_s)),
                api_key=materials_project_api_key,
                config_path=materials_project_config_path,
                sort_by=materials_project_sort_by,
                include_deprecated=materials_project_include_deprecated,
                filter_stable=materials_project_filter_stable,
                max_energy_above_hull=materials_project_max_energy_above_hull,
                prefer_experimental=materials_project_prefer_experimental,
                candidate_pool_size=materials_project_candidate_pool_size,
            )
            merged.extend(mp_api_rows)
            warnings.extend(mp_api_warnings)
            provenance.setdefault("materials_project_api", []).append(mp_api_provenance)
        if not clean_mp_ids and not query_formulas and clean_elements:
            mp_api_rows, mp_api_warnings, mp_api_provenance = _fetch_mp_api_candidates(
                formula=None,
                elements=clean_elements,
                material_ids=[],
                limit=max(1, int(limit_per_source)),
                timeout_s=max(5, int(timeout_s)),
                api_key=materials_project_api_key,
                config_path=materials_project_config_path,
                sort_by=materials_project_sort_by,
                include_deprecated=materials_project_include_deprecated,
                filter_stable=materials_project_filter_stable,
                max_energy_above_hull=materials_project_max_energy_above_hull,
                prefer_experimental=materials_project_prefer_experimental,
                candidate_pool_size=materials_project_candidate_pool_size,
            )
            merged.extend(mp_api_rows)
            warnings.extend(mp_api_warnings)
            provenance.setdefault("materials_project_api", []).append(mp_api_provenance)

    if "cod" in active_sources:
        for query_formula in online_query_formulas:
            cod_rows, cod_warnings = _fetch_cod_candidates(
                formula=query_formula,
                limit=max(1, int(limit_per_source)),
                timeout_s=max(5, int(timeout_s)),
            )
            merged.extend(cod_rows)
            warnings.extend(cod_warnings)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in merged:
        key = (
            str(row.get("source", "")),
            round(float(row.get("a", 0.0)), 5),
            round(float(row.get("b", 0.0)), 5),
            round(float(row.get("c", 0.0)), 5),
            round(float(row.get("alpha", 0.0)), 4),
            round(float(row.get("beta", 0.0)), 4),
            round(float(row.get("gamma", 0.0)), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return PhaseLookupResult(
        candidates=deduped,
        warnings=warnings,
        provenance=provenance,
    )
