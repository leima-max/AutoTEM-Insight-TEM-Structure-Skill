"""Online phase candidate lookup helpers for B4 indexing."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class PhaseLookupResult:
    candidates: list[dict[str, Any]]
    warnings: list[str]
    provenance: dict[str, Any]


_FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


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
    sources: list[str] | None = None,
    limit_per_source: int = 10,
    timeout_s: int = 20,
) -> PhaseLookupResult:
    """Query online material databases and return B4 candidate_phase_params entries."""
    active_sources = sources or ["materials_project_optimade", "cod"]
    clean_elements = [str(item).strip() for item in (elements or []) if str(item).strip()]
    warnings: list[str] = []
    merged: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {
        "formula": None if formula is None else str(formula).strip(),
        "elements": clean_elements,
        "sources": list(active_sources),
    }

    if "materials_project_optimade" in active_sources:
        mp_rows, mp_warnings = _fetch_mp_candidates(
            formula=formula,
            elements=clean_elements,
            limit=max(1, int(limit_per_source)),
            timeout_s=max(5, int(timeout_s)),
        )
        merged.extend(mp_rows)
        warnings.extend(mp_warnings)

    if "cod" in active_sources:
        cod_rows, cod_warnings = _fetch_cod_candidates(
            formula=formula,
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
