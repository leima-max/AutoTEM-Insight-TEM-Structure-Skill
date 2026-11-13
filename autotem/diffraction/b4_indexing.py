"""B4 hkl indexing helpers (MVP + strict extensions)."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from autotem.core import DiffractionPeak, GVector, IndexedSpot, Measurement, PhaseCandidate
from autotem.simulation.pymatgen_adapter import load_candidate_phase_from_cif


@dataclass(slots=True)
class _LatticePhase:
    name: str
    a: float
    b: float
    c: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    space_group: str = "P1"
    source: str = "candidate_phase_params"
    identifier: str | None = None
    formula: str | None = None
    space_group_number: int | None = None
    atom_sites: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _Reflection:
    h: int
    k: int
    ell: int
    d_nm: float
    g_cart: np.ndarray


@dataclass(slots=True)
class B4IndexingResult:
    indexed_spots: list[IndexedSpot]
    phase_candidates: list[PhaseCandidate]
    warnings: list[str]
    provenance: dict[str, Any]


def _measurement_in_nm(
    measurement: Measurement | None, *, reciprocal: bool = False
) -> float | None:
    """Normalize explicit physical lengths/frequencies; never infer pixel calibration."""
    if measurement is None or isinstance(measurement.value, bool):
        return None
    unit = str(measurement.unit).strip().lower().replace(" ", "")
    unit = unit.replace("å", "angstrom").replace("µ", "u").replace("μ", "u")
    unit = unit.replace("{", "").replace("}", "").replace("⁻¹", "^-1")
    if reciprocal:
        if unit.startswith("1/"):
            unit = unit[2:]
        elif unit.startswith("cycles/"):
            unit = unit[7:]
        elif unit.endswith("^-1"):
            unit = unit[:-3]
        else:
            return None
    scales = {"nm": 1.0, "pm": 0.001, "angstrom": 0.1, "um": 1000.0}
    scale = scales.get(unit)
    if scale is None:
        return None
    try:
        value = float(measurement.value)
    except (TypeError, ValueError):
        return None
    value = value / scale if reciprocal else value * scale
    if not np.isfinite(value) or (not reciprocal and value <= 0.0):
        return None
    return value


def _as_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Support CIF style values like "3.567(2)".
        token = value.strip()
        if not token:
            raise ValueError(f"{field_name} must be numeric.")
        keep = []
        seen_dot = False
        seen_exp = False
        for idx, ch in enumerate(token):
            if ch in "+-" and idx == 0:
                keep.append(ch)
                continue
            if ch.isdigit():
                keep.append(ch)
                continue
            if ch == "." and not seen_dot and not seen_exp:
                seen_dot = True
                keep.append(ch)
                continue
            if ch in "eE" and not seen_exp and keep:
                seen_exp = True
                keep.append(ch)
                continue
            break
        try:
            return float("".join(keep))
        except ValueError as exc:  # pragma: no cover - defensive branch.
            raise ValueError(f"{field_name} must be numeric.") from exc
    raise ValueError(f"{field_name} must be numeric.")


def _angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    na = float(np.linalg.norm(vec_a))
    nb = float(np.linalg.norm(vec_b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    cosang = float(np.dot(vec_a, vec_b) / (na * nb))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _canonical_zone_axis(raw_vec: tuple[int, int, int]) -> tuple[int, int, int] | None:
    u, v, w = raw_vec
    if u == 0 and v == 0 and w == 0:
        return None
    scale = gcd(gcd(abs(u), abs(v)), abs(w))
    scale = max(1, scale)
    u //= scale
    v //= scale
    w //= scale
    if u < 0 or (u == 0 and v < 0) or (u == 0 and v == 0 and w < 0):
        u, v, w = -u, -v, -w
    return (u, v, w)


def _normalize_label(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def _formula_counts_key(value: str | None) -> tuple[tuple[str, float], ...] | None:
    if value is None:
        return None
    text = re.sub(r"[^A-Za-z0-9.]+", "", str(value).strip())
    if not text:
        return None
    parts = _FORMULA_TOKEN_RE.findall(text)
    joined = "".join(f"{symbol}{count}" for symbol, count in parts)
    if not parts or joined != text:
        return None
    counts: dict[str, float] = {}
    for symbol, count_text in parts:
        count = 1.0 if count_text == "" else float(count_text)
        counts[symbol] = counts.get(symbol, 0.0) + count
    return tuple(sorted(counts.items()))


def _phase_allowed(phase: _LatticePhase, allowed_names: set[str] | None) -> bool:
    if allowed_names is None:
        return True
    allowed = {_normalize_label(item) for item in allowed_names if _normalize_label(item)}
    if not allowed:
        return True
    allowed_formula_keys = {
        key for key in (_formula_counts_key(item) for item in allowed_names) if key
    }
    phase_formula_keys = {
        key
        for key in (
            _formula_counts_key(phase.formula),
            _formula_counts_key(phase.name),
            _formula_counts_key(phase.identifier),
        )
        if key
    }
    if phase_formula_keys.intersection(allowed_formula_keys):
        return True
    labels = {
        _normalize_label(phase.name),
        _normalize_label(phase.formula),
        _normalize_label(phase.identifier),
    }
    labels = {label for label in labels if label}
    if labels.intersection(allowed):
        return True
    for allowed_label in allowed:
        for label in labels:
            if (
                min(len(allowed_label), len(label)) >= 4
                and (allowed_label in label or label in allowed_label)
            ):
                return True
    return False


def _as_optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _clean_atom_sites(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    sites: list[dict[str, Any]] = []
    for idx, row in enumerate(value):
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


def _parse_candidate_phase_params(
    raw_candidates: Any,
    *,
    allowed_names: set[str] | None,
) -> tuple[list[_LatticePhase], list[str]]:
    warnings: list[str] = []
    if raw_candidates is None:
        return [], warnings
    if not isinstance(raw_candidates, list):
        warnings.append(
            "candidate_phase_params must be a list; "
            "B4 indexing skipped for table input."
        )
        return [], warnings

    phases: list[_LatticePhase] = []
    for idx, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            warnings.append(f"candidate_phase_params[{idx}] is not a mapping; skipped.")
            continue
        name = str(item.get("name", f"phase_{idx}")).strip() or f"phase_{idx}"
        try:
            phase = _LatticePhase(
                name=name,
                a=_as_float(item.get("a"), field_name=f"{name}.a"),
                b=_as_float(item.get("b"), field_name=f"{name}.b"),
                c=_as_float(item.get("c"), field_name=f"{name}.c"),
                alpha_deg=_as_float(item.get("alpha", 90.0), field_name=f"{name}.alpha"),
                beta_deg=_as_float(item.get("beta", 90.0), field_name=f"{name}.beta"),
                gamma_deg=_as_float(item.get("gamma", 90.0), field_name=f"{name}.gamma"),
                space_group=str(item.get("space_group", "P1")),
                source=str(item.get("source", "candidate_phase_params")),
                identifier=(
                    None if item.get("identifier") is None else str(item.get("identifier"))
                ),
                formula=None if item.get("formula") is None else str(item.get("formula")),
                space_group_number=_as_optional_int(item.get("space_group_number")),
                atom_sites=_clean_atom_sites(
                    item.get("atom_sites", item.get("atoms", item.get("sites")))
                ),
                metadata={
                    "materials_project": item.get("materials_project"),
                    "cif_url": item.get("cif_url"),
                    "structure_source": item.get("structure_source"),
                    "mp_rank": item.get("mp_rank"),
                    "mp_energy_above_hull": item.get("mp_energy_above_hull"),
                    "mp_is_stable": item.get("mp_is_stable"),
                    "mp_theoretical": item.get("mp_theoretical"),
                    "mp_deprecated": item.get("mp_deprecated"),
                    "mp_site_count": item.get("mp_site_count"),
                },
            )
        except ValueError as exc:
            warnings.append(f"candidate phase '{name}' is invalid: {exc}")
            continue
        if not _phase_allowed(phase, allowed_names):
            continue
        phases.append(phase)
    return phases, warnings


def _cif_length_to_nm(values: dict[str, Any]) -> tuple[float, float, float]:
    a = _as_float(values["a"], field_name="cif.a")
    b = _as_float(values["b"], field_name="cif.b")
    c = _as_float(values["c"], field_name="cif.c")
    # Standard CIF cell lengths are Angstrom. Existing repo fixtures may already
    # use nm; values below 1 Angstrom are physically implausible for cell lengths,
    # so keep sub-1 inputs as nm for backward compatibility.
    if max(a, b, c) >= 1.0:
        return a * 0.1, b * 0.1, c * 0.1
    return a, b, c


def _clean_cif_numeric(value: Any) -> float | None:
    try:
        return _as_float(value, field_name="cif_atom_site")
    except ValueError:
        return None


def _element_from_cif_label(value: str) -> str:
    match = "".join(ch for ch in str(value).strip() if ch.isalpha())
    if not match:
        return "X"
    return match[0].upper() + match[1:].lower()


def _parse_atom_sites_from_cif_lines(lines: list[str]) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].strip().lower() != "loop_":
            idx += 1
            continue
        idx += 1
        headers: list[str] = []
        while idx < len(lines):
            line = lines[idx].strip()
            if not line or line.startswith("#"):
                idx += 1
                continue
            if not line.startswith("_"):
                break
            headers.append(line.split()[0].lower())
            idx += 1
        if not headers or not any(header.startswith("_atom_site_") for header in headers):
            while (
                idx < len(lines)
                and lines[idx].strip()
                and not lines[idx].strip().startswith("_")
            ):
                if lines[idx].strip().lower() == "loop_":
                    break
                idx += 1
            continue
        header_index = {header: pos for pos, header in enumerate(headers)}
        required = {
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
        }
        if not required.issubset(header_index):
            continue
        while idx < len(lines):
            raw_line = lines[idx].strip()
            if (
                not raw_line
                or raw_line.startswith("#")
                or raw_line.lower() == "loop_"
                or raw_line.startswith("_")
            ):
                break
            try:
                tokens = shlex.split(raw_line)
            except ValueError:
                tokens = raw_line.split()
            if len(tokens) >= len(headers):
                symbol = None
                for key in ("_atom_site_type_symbol", "_atom_site_label"):
                    pos = header_index.get(key)
                    if pos is not None and pos < len(tokens):
                        symbol = str(tokens[pos])
                        break
                x = _clean_cif_numeric(tokens[header_index["_atom_site_fract_x"]])
                y = _clean_cif_numeric(tokens[header_index["_atom_site_fract_y"]])
                z = _clean_cif_numeric(tokens[header_index["_atom_site_fract_z"]])
                if symbol and x is not None and y is not None and z is not None:
                    occupancy = 1.0
                    occ_pos = header_index.get("_atom_site_occupancy")
                    if occ_pos is not None and occ_pos < len(tokens):
                        occupancy = _clean_cif_numeric(tokens[occ_pos]) or 1.0
                    sites.append(
                        {
                            "element": _element_from_cif_label(symbol),
                            "abc": [x, y, z],
                            "occupancy": occupancy,
                            "source_index": len(sites),
                        }
                    )
            idx += 1
    return sites


def _phase_from_cif(path: Path) -> _LatticePhase | None:
    pymatgen_result = load_candidate_phase_from_cif(path)
    if pymatgen_result.candidate is not None:
        phase, parse_warnings = _parse_candidate_phase_params(
            [pymatgen_result.candidate],
            allowed_names=None,
        )
        if phase and not parse_warnings:
            return phase[0]

    keymap = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
        "_symmetry_space_group_name_h-m": "space_group",
        "_space_group_name_h-m_alt": "space_group",
        "_space_group_it_number": "space_group_number",
        "_symmetry_int_tables_number": "space_group_number",
        "_chemical_formula_sum": "formula",
    }
    parsed: dict[str, Any] = {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        key = parts[0].strip().lower()
        if key not in keymap:
            continue
        value = parts[1].strip().strip("'").strip('"')
        parsed[keymap[key]] = value
    required = {"a", "b", "c", "alpha", "beta", "gamma"}
    if not required.issubset(parsed):
        return None
    try:
        a_nm, b_nm, c_nm = _cif_length_to_nm(parsed)
        return _LatticePhase(
            name=path.stem,
            a=a_nm,
            b=b_nm,
            c=c_nm,
            alpha_deg=_as_float(parsed["alpha"], field_name=f"{path.name}.alpha"),
            beta_deg=_as_float(parsed["beta"], field_name=f"{path.name}.beta"),
            gamma_deg=_as_float(parsed["gamma"], field_name=f"{path.name}.gamma"),
            space_group=str(parsed.get("space_group", "P1")),
            source="cif_candidates",
            identifier=path.name,
            formula=None if parsed.get("formula") is None else str(parsed["formula"]),
            space_group_number=_as_optional_int(parsed.get("space_group_number")),
            atom_sites=_parse_atom_sites_from_cif_lines(lines),
        )
    except ValueError:
        return None


def _load_cif_candidates(
    cif_candidates_dir: str | None,
    *,
    cif_candidate_paths: Any = None,
    allowed_names: set[str] | None,
) -> tuple[list[_LatticePhase], list[str]]:
    warnings: list[str] = []
    paths: list[Path] = []
    if cif_candidates_dir:
        root = Path(cif_candidates_dir)
        if not root.exists() or not root.is_dir():
            warnings.append(f"cif_candidates_dir does not exist: {cif_candidates_dir}")
        else:
            paths.extend(sorted(root.glob("*.cif")))

    raw_paths: list[Any] = []
    if isinstance(cif_candidate_paths, (str, Path)):
        raw_paths = [cif_candidate_paths]
    elif isinstance(cif_candidate_paths, list):
        raw_paths = list(cif_candidate_paths)
    elif cif_candidate_paths is not None:
        warnings.append("cif_candidate_paths must be a string path or list of paths; skipped.")

    for raw_path in raw_paths:
        path = Path(str(raw_path))
        if path.is_dir():
            paths.extend(sorted(path.glob("*.cif")))
        elif path.is_file():
            paths.append(path)
        else:
            warnings.append(f"CIF candidate path does not exist: {path}")

    phases: list[_LatticePhase] = []
    seen_paths: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        phase = _phase_from_cif(path)
        if phase is None:
            warnings.append(f"Failed to parse CIF candidate: {path.name}")
            continue
        if not _phase_allowed(phase, allowed_names):
            continue
        phases.append(phase)
    return phases, warnings


def _reciprocal_basis(phase: _LatticePhase) -> np.ndarray | None:
    a = float(phase.a)
    b = float(phase.b)
    c = float(phase.c)
    if min(a, b, c) <= 0:
        return None

    alpha = np.deg2rad(float(phase.alpha_deg))
    beta = np.deg2rad(float(phase.beta_deg))
    gamma = np.deg2rad(float(phase.gamma_deg))
    sin_gamma = float(np.sin(gamma))
    if abs(sin_gamma) <= 1e-8:
        return None

    ax = np.array([a, 0.0, 0.0], dtype=np.float64)
    bx = np.array([b * np.cos(gamma), b * sin_gamma, 0.0], dtype=np.float64)
    cx = np.array(
        [
            c * np.cos(beta),
            c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma,
            0.0,
        ],
        dtype=np.float64,
    )
    cz_sq = c * c - cx[0] * cx[0] - cx[1] * cx[1]
    if cz_sq <= 1e-10:
        return None
    cx[2] = float(np.sqrt(cz_sq))

    volume = float(np.dot(ax, np.cross(bx, cx)))
    if abs(volume) <= 1e-10:
        return None

    a_star = np.cross(bx, cx) / volume
    b_star = np.cross(cx, ax) / volume
    c_star = np.cross(ax, bx) / volume
    return np.column_stack((a_star, b_star, c_star))


def _centering_letter(space_group: str) -> str:
    token = (space_group or "P1").strip()
    for ch in token:
        if ch.isalpha():
            return ch.upper()
    return "P"


def _passes_centering_rule(h: int, k: int, ell: int, letter: str) -> bool:
    if letter == "P":
        return True
    if letter == "I":
        return (h + k + ell) % 2 == 0
    if letter == "F":
        parity = (h & 1, k & 1, ell & 1)
        return parity == (0, 0, 0) or parity == (1, 1, 1)
    if letter == "A":
        return (k + ell) % 2 == 0
    if letter == "B":
        return (h + ell) % 2 == 0
    if letter == "C":
        return (h + k) % 2 == 0
    if letter == "R":
        return (-h + k + ell) % 3 == 0
    return True


def _enumerate_reflections(
    phase: _LatticePhase,
    *,
    max_hkl: int,
    apply_centering_rules: bool,
) -> list[_Reflection]:
    basis = _reciprocal_basis(phase)
    if basis is None:
        return []
    center_letter = _centering_letter(phase.space_group)

    reflections: list[_Reflection] = []
    for h in range(-max_hkl, max_hkl + 1):
        for k in range(-max_hkl, max_hkl + 1):
            for ell in range(-max_hkl, max_hkl + 1):
                if h == 0 and k == 0 and ell == 0:
                    continue
                if apply_centering_rules and not _passes_centering_rule(
                    h, k, ell, center_letter
                ):
                    continue
                g_cart = basis @ np.array([h, k, ell], dtype=np.float64)
                g_norm = float(np.linalg.norm(g_cart))
                if g_norm <= 1e-10:
                    continue
                reflections.append(
                    _Reflection(h=h, k=k, ell=ell, d_nm=1.0 / g_norm, g_cart=g_cart)
                )
    reflections.sort(key=lambda ref: ref.d_nm, reverse=True)
    return reflections


def _select_measured_basis_pair(
    gxy: list[np.ndarray],
    intensities: list[float],
) -> tuple[int, int] | None:
    if len(gxy) < 2:
        return None
    ranked = sorted(range(len(gxy)), key=lambda idx: intensities[idx], reverse=True)
    ranked = ranked[: min(12, len(ranked))]
    best: tuple[int, int] | None = None
    best_score = -np.inf
    for i in range(len(ranked)):
        for j in range(i + 1, len(ranked)):
            ii = ranked[i]
            jj = ranked[j]
            angle = _angle_deg(gxy[ii], gxy[jj])
            if angle < 10.0 or angle > 170.0:
                continue
            score = float(np.sin(np.deg2rad(angle)) * np.sqrt(intensities[ii] * intensities[jj]))
            if score > best_score:
                best_score = score
                best = (ii, jj)
    return best


def _match_phase(
    *,
    phase: _LatticePhase,
    reflections: list[_Reflection],
    gxy: list[np.ndarray],
    d_per_peak: list[float | None],
    intensities: list[float],
    d_tolerance_rel: float,
    angle_tolerance_deg: float,
    strict_mode: bool,
) -> tuple[list[IndexedSpot], PhaseCandidate, dict[str, Any]]:
    basis_pair = _select_measured_basis_pair(gxy, intensities)
    anchor_ref_a: _Reflection | None = None
    anchor_ref_b: _Reflection | None = None
    anchor_peak_a = -1
    anchor_peak_b = -1

    if basis_pair is not None:
        i, j = basis_pair
        d_i = d_per_peak[i]
        d_j = d_per_peak[j]
        if d_i is not None and d_j is not None:
            angle_ij = _angle_deg(gxy[i], gxy[j])
            best_pair_score = np.inf
            for ref_a in reflections:
                err_i = abs(ref_a.d_nm - d_i) / max(d_i, 1e-12)
                if err_i > max(0.4, d_tolerance_rel * 3.0):
                    continue
                for ref_b in reflections:
                    if (
                        ref_a.h == ref_b.h
                        and ref_a.k == ref_b.k
                        and ref_a.ell == ref_b.ell
                    ):
                        continue
                    err_j = abs(ref_b.d_nm - d_j) / max(d_j, 1e-12)
                    if err_j > max(0.4, d_tolerance_rel * 3.0):
                        continue
                    angle_ref = _angle_deg(ref_a.g_cart, ref_b.g_cart)
                    angle_err = abs(angle_ref - angle_ij)
                    if angle_err > max(35.0, angle_tolerance_deg * 2.5):
                        continue
                    pair_score = err_i + err_j + angle_err / 90.0
                    if pair_score < best_pair_score:
                        best_pair_score = pair_score
                        anchor_ref_a = ref_a
                        anchor_ref_b = ref_b
                        anchor_peak_a = i
                        anchor_peak_b = j

    indexed: list[IndexedSpot] = []
    d_errors: list[float] = []
    angle_errors: list[float] = []
    hit_count = 0
    within_d_count = 0
    within_angle_count = 0
    best_d_errors_all: list[float] = []
    best_angle_errors_all: list[float] = []
    unmatched_count = 0

    for peak_idx, meas_vec in enumerate(gxy):
        d_meas = d_per_peak[peak_idx]
        if d_meas is None:
            indexed.append(
                IndexedSpot(
                    peak_index=peak_idx,
                    h=0,
                    k=0,
                    l=0,
                    d_spacing=None,
                    error=Measurement(value=1.0, unit="relative", method="b4_missing_d"),
                    confidence=0.05,
                )
            )
            continue

        best_ref: _Reflection | None = None
        best_d_err = np.inf
        best_angle_err = 180.0
        best_score = np.inf
        for ref in reflections:
            d_err = abs(ref.d_nm - d_meas) / max(d_meas, 1e-12)
            if d_err > max(0.7, d_tolerance_rel * 5.0):
                continue
            angle_err = 0.0
            if (
                anchor_ref_a is not None
                and anchor_ref_b is not None
                and anchor_peak_a >= 0
                and anchor_peak_b >= 0
            ):
                a_meas = _angle_deg(meas_vec, gxy[anchor_peak_a])
                b_meas = _angle_deg(meas_vec, gxy[anchor_peak_b])
                a_ref = _angle_deg(ref.g_cart, anchor_ref_a.g_cart)
                b_ref = _angle_deg(ref.g_cart, anchor_ref_b.g_cart)
                angle_err = 0.5 * (abs(a_meas - a_ref) + abs(b_meas - b_ref))
            score = d_err + angle_err / max(1.0, angle_tolerance_deg * 6.0)
            if score < best_score:
                best_score = score
                best_ref = ref
                best_d_err = d_err
                best_angle_err = angle_err

        if best_ref is None:
            unmatched_count += 1
            indexed.append(
                IndexedSpot(
                    peak_index=peak_idx,
                    h=0,
                    k=0,
                    l=0,
                    d_spacing=Measurement(
                        value=float(d_meas),
                        unit="nm",
                        method="measured_d",
                    ),
                    error=Measurement(value=1.0, unit="relative", method="b4_no_match"),
                    confidence=0.05,
                )
            )
            continue

        within_d = best_d_err <= d_tolerance_rel
        within_angle = best_angle_err <= angle_tolerance_deg if anchor_ref_a is not None else True
        if within_d:
            within_d_count += 1
        if within_angle:
            within_angle_count += 1
        best_d_errors_all.append(float(best_d_err))
        best_angle_errors_all.append(float(best_angle_err))
        if within_d and within_angle:
            hit_count += 1
            d_errors.append(best_d_err)
            if anchor_ref_a is not None:
                angle_errors.append(best_angle_err)

        conf = float(np.exp(-best_d_err / max(d_tolerance_rel, 1e-6)))
        if anchor_ref_a is not None:
            conf *= float(np.exp(-best_angle_err / max(angle_tolerance_deg, 1e-6)))
        conf = max(0.03, min(0.98, conf))

        indexed.append(
            IndexedSpot(
                peak_index=peak_idx,
                h=best_ref.h,
                k=best_ref.k,
                l=best_ref.ell,
                d_spacing=Measurement(
                    value=float(best_ref.d_nm),
                    unit="nm",
                    method=f"b4_calc_d[{phase.name}]",
                ),
                error=Measurement(
                    value=float(best_d_err),
                    unit="relative",
                    method="b4_d_spacing_relative_error",
                ),
                confidence=conf,
            )
        )

    mean_d_error = float(np.mean(d_errors)) if d_errors else 1.0
    mean_angle_error = float(np.mean(angle_errors)) if angle_errors else None
    score = float(np.exp(-mean_d_error / max(d_tolerance_rel, 1e-6)))
    score *= hit_count / max(1, len(gxy))

    friedel_score = None
    zone_axis_score = None
    zone_axis = (0, 0, 1)
    if strict_mode and indexed:
        hkl_to_indices: dict[tuple[int, int, int], list[int]] = {}
        for idx, spot in enumerate(indexed):
            hkl = (spot.h, spot.k, spot.l)
            hkl_to_indices.setdefault(hkl, []).append(idx)
        pair_total = 0
        pair_hit = 0
        for hkl in hkl_to_indices:
            inv = (-hkl[0], -hkl[1], -hkl[2])
            if hkl == (0, 0, 0):
                continue
            pair_total += 1
            if inv in hkl_to_indices:
                pair_hit += 1
        friedel_score = pair_hit / max(1, pair_total)

        zone_votes: dict[tuple[int, int, int], int] = {}
        top_spots = [spot for spot in indexed if (spot.h, spot.k, spot.l) != (0, 0, 0)][:16]
        for i in range(len(top_spots)):
            hi = top_spots[i]
            vec_i = np.array([hi.h, hi.k, hi.l], dtype=np.int64)
            for j in range(i + 1, len(top_spots)):
                hj = top_spots[j]
                vec_j = np.array([hj.h, hj.k, hj.l], dtype=np.int64)
                cross = np.cross(vec_i, vec_j)
                axis = _canonical_zone_axis((int(cross[0]), int(cross[1]), int(cross[2])))
                if axis is None:
                    continue
                zone_votes[axis] = zone_votes.get(axis, 0) + 1
        if zone_votes:
            zone_axis = max(zone_votes, key=zone_votes.get)
            votes = zone_votes[zone_axis]
            total_votes = sum(zone_votes.values())
            zone_axis_score = votes / max(1, total_votes)
        else:
            zone_axis_score = 0.0

        score *= 0.5 + 0.25 * (friedel_score or 0.0) + 0.25 * (zone_axis_score or 0.0)

    candidate = PhaseCandidate(
        name=phase.name,
        formula=phase.formula,
        source=phase.source,
        identifier=phase.identifier,
        score=max(0.0, min(1.0, score)),
        matched_spot_count=hit_count,
        mean_d_error=Measurement(
            value=mean_d_error,
            unit="relative",
            method="b4_phase_match",
        ),
        mean_angle_error=(
            None
            if mean_angle_error is None
            else Measurement(value=mean_angle_error, unit="deg", method="b4_angle_match")
        ),
        zone_axis=zone_axis,
        lattice_parameters={
            "a": float(phase.a),
            "b": float(phase.b),
            "c": float(phase.c),
            "alpha": float(phase.alpha_deg),
            "beta": float(phase.beta_deg),
            "gamma": float(phase.gamma_deg),
            "unit": "nm",
            "space_group": phase.space_group,
            "space_group_number": phase.space_group_number,
            "atom_sites": phase.atom_sites,
            **{
                key: value
                for key, value in phase.metadata.items()
                if value is not None
            },
        },
        warnings=[],
    )
    if strict_mode:
        candidate.warnings.extend(
            [
                f"friedel_consistency={friedel_score:.3f}" if friedel_score is not None else "",
                (
                    f"zone_axis_consistency={zone_axis_score:.3f}"
                    if zone_axis_score is not None
                    else ""
                ),
            ]
        )
        candidate.warnings = [text for text in candidate.warnings if text]
    diagnostics = {
        "phase_name": phase.name,
        "anchor_pair_found": bool(anchor_ref_a is not None and anchor_ref_b is not None),
        "total_spot_count": len(gxy),
        "valid_d_spot_count": int(sum(1 for item in d_per_peak if item is not None)),
        "unmatched_spot_count": unmatched_count,
        "within_d_count": within_d_count,
        "within_angle_count": within_angle_count,
        "matched_within_all_constraints_count": hit_count,
        "mean_best_d_error": (
            None if not best_d_errors_all else float(np.mean(best_d_errors_all))
        ),
        "mean_best_angle_error_deg": (
            None if not best_angle_errors_all else float(np.mean(best_angle_errors_all))
        ),
        "d_tolerance_rel": float(d_tolerance_rel),
        "angle_tolerance_deg": float(angle_tolerance_deg),
    }
    return indexed, candidate, diagnostics


def run_b4_indexing(
    *,
    peaks: list[DiffractionPeak],
    g_vectors: list[GVector],
    d_by_peak: list[Measurement | None],
    initial_indexed_spots: list[IndexedSpot],
    candidate_phase_params: Any,
    allowed_phase_names: set[str] | None,
    max_hkl: int,
    d_tolerance_rel: float,
    angle_tolerance_deg: float,
    strict_mode: bool,
    cif_candidates_dir: str | None = None,
    cif_candidate_paths: Any = None,
    top_n: int = 3,
) -> B4IndexingResult:
    warnings: list[str] = []
    phases, parse_warnings = _parse_candidate_phase_params(
        candidate_phase_params, allowed_names=allowed_phase_names
    )
    warnings.extend(parse_warnings)

    cif_phases, cif_warnings = _load_cif_candidates(
        cif_candidates_dir,
        cif_candidate_paths=cif_candidate_paths,
        allowed_names=allowed_phase_names,
    )
    warnings.extend(cif_warnings)
    existing = {
        (
            _normalize_label(phase.name),
            _normalize_label(phase.identifier),
            _normalize_label(phase.source),
        )
        for phase in phases
    }
    for phase in cif_phases:
        key = (
            _normalize_label(phase.name),
            _normalize_label(phase.identifier),
            _normalize_label(phase.source),
        )
        if key not in existing:
            phases.append(phase)
            existing.add(key)

    gxy: list[np.ndarray] = []
    d_per_peak: list[float | None] = []
    intensities: list[float] = []
    rejected_measurements: list[dict[str, Any]] = []
    for idx, peak in enumerate(peaks):
        if idx >= len(g_vectors):
            break
        vector = g_vectors[idx]
        gx = _measurement_in_nm(vector.gx, reciprocal=True)
        gy = _measurement_in_nm(vector.gy, reciprocal=True)
        d_measurement = d_by_peak[idx] if idx < len(d_by_peak) else None
        d_nm = _measurement_in_nm(d_measurement)
        norm = float(np.hypot(gx, gy)) if gx is not None and gy is not None else 0.0
        valid_vector = np.isfinite(norm) and norm > 1e-12
        if vector.magnitude is not None:
            magnitude = _measurement_in_nm(vector.magnitude, reciprocal=True)
            valid_vector = valid_vector and magnitude is not None and magnitude > 0.0
        if not valid_vector or d_nm is None:
            rejected_measurements.append(
                {
                    "peak_index": idx,
                    "reason": (
                        "invalid_physical_g_vector" if not valid_vector else "invalid_d_spacing"
                    ),
                    "gx_unit": vector.gx.unit,
                    "gy_unit": vector.gy.unit,
                    "d_unit": d_measurement.unit if d_measurement is not None else None,
                }
            )
            # Keep original peak indices; a missing d-spacing cannot receive a new hkl.
            gxy.append(np.zeros(2, dtype=np.float64))
            d_per_peak.append(None)
        else:
            gxy.append(np.array([gx, gy], dtype=np.float64))
            d_per_peak.append(d_nm)
        intensities.append(float(peak.intensity or 1.0))

    valid_d_count = int(sum(1 for value in d_per_peak if value is not None))
    common_diag = {
        "total_peak_count": len(peaks),
        "g_vector_count": len(gxy),
        "valid_d_count": valid_d_count,
        "phase_candidate_count": len(phases),
        "d_tolerance_rel": float(d_tolerance_rel),
        "angle_tolerance_deg": float(angle_tolerance_deg),
        "normalized_units": {"g_vector": "1/nm", "d_spacing": "nm"},
        "rejected_measurements": rejected_measurements,
    }
    if rejected_measurements:
        warnings.append(
            f"B4 excluded {len(rejected_measurements)} uncalibrated or invalid measurement(s): "
            "physical g-vectors and finite positive d-spacings are required."
        )

    if valid_d_count == 0:
        withheld_hkl = [
            {"peak_index": spot.peak_index, "hkl": [spot.h, spot.k, spot.l]}
            for spot in initial_indexed_spots
            if (spot.h, spot.k, spot.l) != (0, 0, 0)
        ]
        safe_initial = [
            spot.model_copy(
                update={
                    "h": 0, "k": 0, "l": 0, "confidence": 0.0,
                    "d_spacing": (
                        d_by_peak[spot.peak_index]
                        if 0 <= spot.peak_index < len(d_by_peak)
                        else None
                    ),
                    "error": Measurement(
                        value=1.0, unit="relative", method="b4_physical_unit_gate",
                        warnings=["Previous hkl withheld: current physical inputs are invalid."],
                    ),
                }
            )
            if (spot.h, spot.k, spot.l) != (0, 0, 0)
            else spot
            for spot in initial_indexed_spots
        ]
        return B4IndexingResult(
            indexed_spots=safe_initial,
            phase_candidates=[],
            warnings=warnings + [
                "B4 indexing skipped: no valid physical measurements; "
                "pixel/unknown units cannot be compared with theoretical nm lattice spacings."
            ],
            provenance={
                "mode": "b4_strict" if strict_mode else "b4_mvp",
                "active": False,
                "reason": "no_valid_physical_measurements" if gxy else "no_g_vectors",
                "diagnostics": {**common_diag, "withheld_initial_hkl": withheld_hkl},
            },
        )

    if not phases:
        return B4IndexingResult(
            indexed_spots=initial_indexed_spots,
            phase_candidates=[],
            warnings=warnings
            + ["No valid B4 phase candidates found; fallback to existing indexing."],
            provenance={
                "mode": "b4_strict" if strict_mode else "b4_mvp",
                "active": False,
                "reason": "no_phase_candidates",
                "diagnostics": common_diag,
            },
        )

    ranked: list[tuple[list[IndexedSpot], PhaseCandidate, dict[str, Any]]] = []
    per_phase_diagnostics: list[dict[str, Any]] = []
    for phase in phases:
        refs = _enumerate_reflections(
            phase,
            max_hkl=max(1, int(max_hkl)),
            apply_centering_rules=bool(strict_mode),
        )
        if not refs:
            warnings.append(f"B4 reflection list is empty for phase '{phase.name}'.")
            continue
        indexed, candidate, diag = _match_phase(
            phase=phase,
            reflections=refs,
            gxy=gxy,
            d_per_peak=d_per_peak,
            intensities=intensities,
            d_tolerance_rel=max(1e-4, float(d_tolerance_rel)),
            angle_tolerance_deg=max(0.1, float(angle_tolerance_deg)),
            strict_mode=strict_mode,
        )
        per_phase_diagnostics.append(diag)
        ranked.append((indexed, candidate, diag))

    if not ranked:
        no_match_warning = "B4 matching failed for all phases; fallback to existing indexing."
        return B4IndexingResult(
            indexed_spots=initial_indexed_spots,
            phase_candidates=[],
            warnings=warnings + [no_match_warning],
            provenance={
                "mode": "b4_strict" if strict_mode else "b4_mvp",
                "active": False,
                "reason": "all_matches_failed",
                "diagnostics": common_diag,
                "per_phase_diagnostics": per_phase_diagnostics,
            },
        )

    ranked.sort(key=lambda item: float(item[1].score or 0.0), reverse=True)
    best_indexed, _, best_diag = ranked[0]
    phase_candidates = [item[1] for item in ranked[: max(1, int(top_n))]]
    provenance = {
        "mode": "b4_strict" if strict_mode else "b4_mvp",
        "active": True,
        "phase_candidate_count": len(phases),
        "max_hkl": max(1, int(max_hkl)),
        "d_tolerance_rel": float(d_tolerance_rel),
        "angle_tolerance_deg": float(angle_tolerance_deg),
        "strict_checks": bool(strict_mode),
        "selected_phase": phase_candidates[0].name if phase_candidates else None,
        "diagnostics": {**common_diag, "selected_phase_diagnostics": best_diag},
        "per_phase_diagnostics": per_phase_diagnostics,
    }
    return B4IndexingResult(
        indexed_spots=best_indexed,
        phase_candidates=phase_candidates,
        warnings=warnings,
        provenance=provenance,
    )
