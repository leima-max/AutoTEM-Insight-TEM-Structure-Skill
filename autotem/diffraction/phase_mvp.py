"""MVP phase-match and visualization helpers for B2/B4 UI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(slots=True)
class PhaseEntry:
    name: str
    path: str
    a: float
    b: float
    c: float
    alpha_deg: float
    beta_deg: float
    gamma_deg: float
    space_group: str
    formula: str | None
    atoms: list[dict[str, Any]]


@dataclass(slots=True)
class PhaseMatchSummary:
    phase_name: str
    score: float
    confidence: float
    matched_peaks: int
    unmatched_peaks: int
    mean_d_error: float
    mean_angular_error_deg: float
    warning: str
    rotation_shift_deg: float


def _as_float_token(value: str | float | int, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric.")
    if isinstance(value, (int, float)):
        return float(value)
    token = str(value).strip()
    if not token:
        raise ValueError(f"{field_name} must be numeric.")
    keep: list[str] = []
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
            keep.append(ch)
            seen_dot = True
            continue
        if ch in "eE" and not seen_exp and keep:
            keep.append(ch)
            seen_exp = True
            continue
        break
    try:
        return float("".join(keep))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be numeric.") from exc


def _split_cif_row(line: str) -> list[str]:
    out: list[str] = []
    token: list[str] = []
    quote: str | None = None
    for ch in line.strip():
        if quote is not None:
            token.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            token.append(ch)
            continue
        if ch.isspace():
            if token:
                out.append("".join(token).strip())
                token = []
            continue
        token.append(ch)
    if token:
        out.append("".join(token).strip())
    return out


@lru_cache(maxsize=512)
def _parse_cif_cached(path_str: str, mtime_ns: int) -> PhaseEntry | None:
    del mtime_ns
    path = Path(path_str)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    keymap = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
        "_symmetry_space_group_name_h-m": "space_group",
        "_space_group_name_h-m_alt": "space_group",
        "_chemical_formula_sum": "formula",
    }
    parsed: dict[str, Any] = {}
    atoms: list[dict[str, Any]] = []

    idx = 0
    n = len(lines)
    while idx < n:
        line = lines[idx].strip()
        if not line or line.startswith("#"):
            idx += 1
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            key = parts[0].strip().lower()
            if key in keymap:
                parsed[keymap[key]] = parts[1].strip().strip("'").strip('"')
        if line.lower() == "loop_":
            idx += 1
            headers: list[str] = []
            while idx < n:
                hline = lines[idx].strip()
                if hline.startswith("_"):
                    headers.append(hline.lower())
                    idx += 1
                    continue
                break
            if headers:
                hmap = {h: i for i, h in enumerate(headers)}
                has_atom = (
                    "_atom_site_fract_x" in hmap
                    and "_atom_site_fract_y" in hmap
                    and "_atom_site_fract_z" in hmap
                    and (
                        "_atom_site_type_symbol" in hmap
                        or "_atom_site_label" in hmap
                    )
                )
                while idx < n:
                    dline = lines[idx].strip()
                    if (
                        not dline
                        or dline.lower() == "loop_"
                        or dline.startswith("_")
                        or dline.lower().startswith("data_")
                    ):
                        break
                    tokens = _split_cif_row(dline)
                    if len(tokens) >= len(headers) and has_atom:
                        sym_idx = hmap.get(
                            "_atom_site_type_symbol",
                            hmap.get("_atom_site_label", -1),
                        )
                        try:
                            fx_idx = hmap["_atom_site_fract_x"]
                            fy_idx = hmap["_atom_site_fract_y"]
                            fz_idx = hmap["_atom_site_fract_z"]
                            atom = {
                                "symbol": str(tokens[sym_idx]).strip("'\""),
                                "fx": _as_float_token(
                                    tokens[fx_idx],
                                    field_name="fract_x",
                                ),
                                "fy": _as_float_token(
                                    tokens[fy_idx],
                                    field_name="fract_y",
                                ),
                                "fz": _as_float_token(
                                    tokens[fz_idx],
                                    field_name="fract_z",
                                ),
                            }
                            atoms.append(atom)
                        except (ValueError, IndexError):
                            pass
                    idx += 1
                continue
        idx += 1

    required = {"a", "b", "c", "alpha", "beta", "gamma"}
    if not required.issubset(parsed):
        return None
    try:
        a_val = _as_float_token(parsed["a"], field_name=f"{path.name}.a")
        b_val = _as_float_token(parsed["b"], field_name=f"{path.name}.b")
        c_val = _as_float_token(parsed["c"], field_name=f"{path.name}.c")
        # CIF cell lengths are typically in Angstrom. If lengths are too large
        # for nm convention, convert Angstrom -> nm.
        if max(a_val, b_val, c_val) > 3.0:
            a_val *= 0.1
            b_val *= 0.1
            c_val *= 0.1
        return PhaseEntry(
            name=path.stem,
            path=str(path),
            a=a_val,
            b=b_val,
            c=c_val,
            alpha_deg=_as_float_token(parsed["alpha"], field_name=f"{path.name}.alpha"),
            beta_deg=_as_float_token(parsed["beta"], field_name=f"{path.name}.beta"),
            gamma_deg=_as_float_token(parsed["gamma"], field_name=f"{path.name}.gamma"),
            space_group=str(parsed.get("space_group", "P1")),
            formula=None if parsed.get("formula") is None else str(parsed["formula"]),
            atoms=atoms,
        )
    except ValueError:
        return None


def load_cif_phase_entries(
    cif_dir: str,
    *,
    selected_names: set[str] | None = None,
) -> tuple[list[PhaseEntry], list[str]]:
    warnings: list[str] = []
    root = Path(cif_dir)
    if not root.exists() or not root.is_dir():
        return [], [f"CIF folder does not exist: {cif_dir}"]
    out: list[PhaseEntry] = []
    for path in sorted(root.glob("*.cif")):
        try:
            stat = path.stat()
        except OSError:
            continue
        entry = _parse_cif_cached(str(path), int(stat.st_mtime_ns))
        if entry is None:
            warnings.append(f"Failed to parse CIF: {path.name}")
            continue
        if selected_names is not None and entry.name not in selected_names:
            continue
        out.append(entry)
    if not out:
        warnings.append("No valid CIF candidates loaded.")
    return out, warnings


def _direct_lattice_vectors(entry: PhaseEntry) -> np.ndarray | None:
    a = float(entry.a)
    b = float(entry.b)
    c = float(entry.c)
    if min(a, b, c) <= 0:
        return None
    alpha = np.deg2rad(float(entry.alpha_deg))
    beta = np.deg2rad(float(entry.beta_deg))
    gamma = np.deg2rad(float(entry.gamma_deg))
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
    return np.column_stack((ax, bx, cx))


def _reciprocal_basis(entry: PhaseEntry) -> np.ndarray | None:
    direct = _direct_lattice_vectors(entry)
    if direct is None:
        return None
    ax = direct[:, 0]
    bx = direct[:, 1]
    cx = direct[:, 2]
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


def _phase_cache_key(entry: PhaseEntry) -> tuple[Any, ...]:
    return (
        entry.name,
        round(float(entry.a), 8),
        round(float(entry.b), 8),
        round(float(entry.c), 8),
        round(float(entry.alpha_deg), 8),
        round(float(entry.beta_deg), 8),
        round(float(entry.gamma_deg), 8),
        _centering_letter(entry.space_group),
    )


@lru_cache(maxsize=1024)
def _enumerate_reflections_cached(
    phase_key: tuple[Any, ...],
    max_hkl: int,
    zone_u: int,
    zone_v: int,
    zone_w: int,
) -> tuple[tuple[float, ...], ...]:
    name, a, b, c, alpha, beta, gamma, centering = phase_key
    entry = PhaseEntry(
        name=str(name),
        path="",
        a=float(a),
        b=float(b),
        c=float(c),
        alpha_deg=float(alpha),
        beta_deg=float(beta),
        gamma_deg=float(gamma),
        space_group=str(centering),
        formula=None,
        atoms=[],
    )
    basis = _reciprocal_basis(entry)
    if basis is None:
        return tuple()

    rows: list[tuple[float, ...]] = []
    for h in range(-max_hkl, max_hkl + 1):
        for k in range(-max_hkl, max_hkl + 1):
            for ell in range(-max_hkl, max_hkl + 1):
                if h == 0 and k == 0 and ell == 0:
                    continue
                if zone_u * h + zone_v * k + zone_w * ell != 0:
                    continue
                if not _passes_centering_rule(h, k, ell, str(centering)):
                    continue
                g = basis @ np.array([h, k, ell], dtype=np.float64)
                g_norm = float(np.linalg.norm(g))
                if g_norm <= 1e-10:
                    continue
                rows.append(
                    (
                        float(h),
                        float(k),
                        float(ell),
                        g[0],
                        g[1],
                        g[2],
                        g_norm,
                        1.0 / g_norm,
                    )
                )
    rows.sort(key=lambda row: row[6])
    return tuple(rows)


def simulate_phase_diffraction(
    entry: PhaseEntry,
    *,
    max_hkl: int = 6,
    zone_axis: tuple[int, int, int] = (0, 0, 1),
) -> list[dict[str, Any]]:
    zone = tuple(int(v) for v in zone_axis)
    if zone == (0, 0, 0):
        zone = (0, 0, 1)
    key = _phase_cache_key(entry)
    raw = _enumerate_reflections_cached(key, int(max_hkl), zone[0], zone[1], zone[2])
    if not raw:
        return []

    g_cart = np.array([[row[3], row[4], row[5]] for row in raw], dtype=np.float64)
    _, _, vh = np.linalg.svd(g_cart, full_matrices=False)
    e1 = vh[0]
    e2 = vh[1]

    out: list[dict[str, Any]] = []
    for row in raw:
        g = np.array([row[3], row[4], row[5]], dtype=np.float64)
        gx2 = float(np.dot(g, e1))
        gy2 = float(np.dot(g, e2))
        out.append(
            {
                "h": int(row[0]),
                "k": int(row[1]),
                "l": int(row[2]),
                "gx": gx2,
                "gy": gy2,
                "g_norm": float(np.hypot(gx2, gy2)),
                "d_nm": float(row[7]),
            }
        )
    return out


def _angle_diff_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b)) % 360.0
    return float(min(d, 360.0 - d))


def match_phase_to_peaks(
    peak_rows: list[dict[str, Any]],
    simulated_rows: list[dict[str, Any]],
    *,
    d_tol_rel: float = 0.08,
    angle_tol_deg: float = 12.0,
) -> tuple[list[dict[str, Any]], PhaseMatchSummary]:
    exp_rows: list[dict[str, Any]] = []
    used_peak_ids: set[int] = set()
    for row_idx, row in enumerate(peak_rows):
        gx = float(row.get("gx_1_per_nm", row.get("g_x", 0.0)) or 0.0)
        gy = float(row.get("gy_1_per_nm", row.get("g_y", 0.0)) or 0.0)
        gnorm = float(np.hypot(gx, gy))
        if gnorm <= 1e-10:
            continue
        # Keep the central transmitted beam as origin only.
        r_px = float(row.get("r_px", np.inf) or np.inf)
        if r_px <= 8.0 and gnorm < 0.5:
            continue
        theta = float(np.degrees(np.arctan2(gy, gx)))
        try:
            peak_id = int(row.get("peak_id", row_idx))
        except (TypeError, ValueError):
            peak_id = row_idx
        if peak_id in used_peak_ids:
            peak_id = row_idx
        used_peak_ids.add(peak_id)
        exp_rows.append(
            {
                "peak_id": peak_id,
                "source_row_index": int(row_idx),
                "gx": gx,
                "gy": gy,
                "d_nm": float(1.0 / gnorm),
                "theta": theta,
                "snr": float(row.get("snr", 0.0) or 0.0),
                "intensity": float(row.get("intensity", 0.0) or 0.0),
            }
        )

    if not exp_rows or not simulated_rows:
        summary = PhaseMatchSummary(
            phase_name="",
            score=0.0,
            confidence=0.0,
            matched_peaks=0,
            unmatched_peaks=len(exp_rows),
            mean_d_error=1.0,
            mean_angular_error_deg=180.0,
            warning="no_peaks_or_simulation",
            rotation_shift_deg=0.0,
        )
        return [], summary

    sim_d = np.asarray([float(row["d_nm"]) for row in simulated_rows], dtype=np.float64)
    sim_theta = np.asarray(
        [
            float(np.degrees(np.arctan2(float(row["gy"]), float(row["gx"]))))
            for row in simulated_rows
        ],
        dtype=np.float64,
    )

    def _best_shift_on_grid(
        grid: np.ndarray,
        *,
        d_gate: float,
        angle_gate: float,
    ) -> float:
        best_shift = 0.0
        best_count = -1
        best_mean = np.inf
        scale = max(1.0, angle_tol_deg * 6.0)
        for shift in grid:
            count = 0
            err_total = 0.0
            for exp in exp_rows:
                d_meas = float(exp["d_nm"])
                d_err = np.abs(sim_d - d_meas) / max(d_meas, 1e-12)
                valid = d_err <= d_gate
                if not np.any(valid):
                    continue
                th = (float(exp["theta"]) - (sim_theta + shift) + 180.0) % 360.0 - 180.0
                a_err = np.abs(th)
                valid = np.logical_and(valid, a_err <= angle_gate)
                if not np.any(valid):
                    continue
                score = d_err + a_err / scale
                idx = int(np.argmin(np.where(valid, score, np.inf)))
                count += 1
                err_total += float(score[idx])
            mean_err = err_total / max(1, count)
            if count > best_count or (count == best_count and mean_err < best_mean):
                best_count = count
                best_mean = mean_err
                best_shift = float(shift)
        return best_shift

    coarse = np.linspace(-180.0, 180.0, 181, endpoint=False, dtype=np.float64)
    shift_coarse = _best_shift_on_grid(
        coarse,
        d_gate=max(0.25, d_tol_rel * 3.0),
        angle_gate=max(35.0, angle_tol_deg * 2.0),
    )
    fine = np.arange(shift_coarse - 2.0, shift_coarse + 2.001, 0.25, dtype=np.float64)
    shift_deg = _best_shift_on_grid(
        fine,
        d_gate=max(0.22, d_tol_rel * 2.6),
        angle_gate=max(24.0, angle_tol_deg * 1.6),
    )

    # Soft nearest-neighbor metrics are used for ranking even when strict
    # thresholds fail, so the UI can still show the "closest" phase.
    soft_d_errors: list[float] = []
    soft_ang_errors: list[float] = []
    for exp in exp_rows:
        d_meas = float(exp["d_nm"])
        best_d = np.inf
        best_ang = 180.0
        best_score = np.inf
        for sim in simulated_rows:
            d_err = abs(float(sim["d_nm"]) - d_meas) / max(d_meas, 1e-12)
            theta_sim = float(
                np.degrees(np.arctan2(float(sim["gy"]), float(sim["gx"]))) + shift_deg
            )
            ang_err = _angle_diff_deg(exp["theta"], theta_sim)
            score = d_err + ang_err / 90.0
            if score < best_score:
                best_score = score
                best_d = d_err
                best_ang = ang_err
        if np.isfinite(best_d):
            soft_d_errors.append(float(best_d))
            soft_ang_errors.append(float(best_ang))

    pair_pool: list[tuple[float, int, int, float, float]] = []
    for exp_idx, exp in enumerate(exp_rows):
        d_meas = exp["d_nm"]
        d_gate = max(0.25, d_tol_rel * 3.0)
        for sim_idx, sim in enumerate(simulated_rows):
            d_err = abs(float(sim["d_nm"]) - d_meas) / max(d_meas, 1e-12)
            if d_err > d_gate:
                continue
            theta_sim = float(
                np.degrees(np.arctan2(float(sim["gy"]), float(sim["gx"]))) + shift_deg
            )
            ang_err = _angle_diff_deg(exp["theta"], theta_sim)
            if ang_err > max(60.0, angle_tol_deg * 3.0):
                continue
            score = d_err + ang_err / max(1.0, angle_tol_deg * 6.0)
            pair_pool.append((score, exp_idx, sim_idx, d_err, ang_err))

    pair_pool.sort(key=lambda item: item[0])
    used_exp: set[int] = set()
    used_sim: set[int] = set()
    chosen: dict[int, tuple[int, float, float]] = {}
    for _, exp_idx, sim_idx, d_err, ang_err in pair_pool:
        if exp_idx in used_exp or sim_idx in used_sim:
            continue
        used_exp.add(exp_idx)
        used_sim.add(sim_idx)
        chosen[exp_idx] = (sim_idx, d_err, ang_err)

    assignments: list[dict[str, Any]] = []
    d_errors: list[float] = []
    ang_errors: list[float] = []
    matched = 0
    for exp_idx, exp in enumerate(exp_rows):
        d_meas = exp["d_nm"]
        chosen_row = chosen.get(exp_idx)
        if chosen_row is None:
            assignments.append(
                {
                    "peak_id": int(exp["peak_id"]),
                    "source_row_index": int(exp["source_row_index"]),
                    "matched": False,
                    "h": 0,
                    "k": 0,
                    "l": 0,
                    "d_nm": d_meas,
                    "d_error": 1.0,
                    "angle_error_deg": 180.0,
                    "sim_gx": 0.0,
                    "sim_gy": 0.0,
                }
            )
            continue
        sim_idx, best_d, best_ang = chosen_row
        best = simulated_rows[sim_idx]
        is_match = bool(best_d <= d_tol_rel and best_ang <= angle_tol_deg)
        if is_match:
            matched += 1
            d_errors.append(best_d)
            ang_errors.append(best_ang)
        assignments.append(
            {
                "peak_id": int(exp["peak_id"]),
                "source_row_index": int(exp["source_row_index"]),
                "matched": is_match,
                "h": int(best["h"]),
                "k": int(best["k"]),
                "l": int(best["l"]),
                "d_nm": float(best["d_nm"]),
                "d_error": float(best_d),
                "angle_error_deg": float(best_ang),
                "sim_gx": float(best["gx"]),
                "sim_gy": float(best["gy"]),
            }
        )

    unmatched = max(0, len(exp_rows) - matched)
    matched_ratio = matched / max(1, len(exp_rows))
    strict_mean_d = float(np.mean(d_errors)) if d_errors else np.inf
    strict_mean_ang = float(np.mean(ang_errors)) if ang_errors else np.inf
    soft_mean_d = float(np.mean(soft_d_errors)) if soft_d_errors else 1.0
    soft_mean_ang = float(np.mean(soft_ang_errors)) if soft_ang_errors else 180.0
    mean_d = strict_mean_d if np.isfinite(strict_mean_d) else soft_mean_d
    mean_ang = strict_mean_ang if np.isfinite(strict_mean_ang) else soft_mean_ang
    soft_score = float(
        np.exp(-soft_mean_d / max(0.18, d_tol_rel * 2.0))
        * np.exp(-soft_mean_ang / 65.0)
    )
    score = float(np.clip(soft_score * (0.35 + 0.65 * matched_ratio), 0.0, 1.0))
    confidence = float(np.clip(score * (0.4 + 0.6 * matched_ratio), 0.0, 1.0))
    warning_items: list[str] = []
    if len(exp_rows) < 8:
        warning_items.append("too_few_peaks")
    if mean_d > max(0.1, d_tol_rel * 1.4):
        warning_items.append("high_d_error")
    if mean_ang > max(18.0, angle_tol_deg * 1.5):
        warning_items.append("high_angular_error")
    if matched == 0:
        warning_items.append("no_strict_match_use_soft_ranking")
    summary = PhaseMatchSummary(
        phase_name="",
        score=score,
        confidence=confidence,
        matched_peaks=matched,
        unmatched_peaks=unmatched,
        mean_d_error=mean_d,
        mean_angular_error_deg=mean_ang,
        warning=";".join(warning_items),
        rotation_shift_deg=shift_deg,
    )
    return assignments, summary


def match_candidates(
    peak_rows: list[dict[str, Any]],
    candidates: list[PhaseEntry],
    *,
    max_hkl: int,
    zone_axis: tuple[int, int, int],
    d_tol_rel: float,
    angle_tol_deg: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    ranking: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    warnings: list[str] = []

    for entry in candidates:
        sim_rows = simulate_phase_diffraction(entry, max_hkl=max_hkl, zone_axis=zone_axis)
        assignments, summary = match_phase_to_peaks(
            peak_rows,
            sim_rows,
            d_tol_rel=d_tol_rel,
            angle_tol_deg=angle_tol_deg,
        )
        summary.phase_name = entry.name
        ranking.append(
            {
                "name": entry.name,
                "formula": entry.formula,
                "score": round(float(summary.score), 6),
                "confidence": round(float(summary.confidence), 6),
                "matched_peaks": int(summary.matched_peaks),
                "unmatched_peaks": int(summary.unmatched_peaks),
                "mean_d_spacing_error": round(float(summary.mean_d_error), 6),
                "mean_angular_error_deg": round(float(summary.mean_angular_error_deg), 6),
                "warning": summary.warning,
            }
        )
        details[entry.name] = {
            "entry": entry,
            "simulated_rows": sim_rows,
            "assignments": assignments,
            "summary": summary,
        }

    ranking.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
    if ranking:
        if len(ranking) >= 2:
            top = float(ranking[0]["score"])
            second = float(ranking[1]["score"])
            if abs(top - second) <= 0.05:
                warnings.append("top1_top2_too_close")
        if float(ranking[0]["confidence"]) < 0.35:
            warnings.append("low_confidence_match")
    else:
        warnings.append("no_phase_match_result")
    return ranking, details, warnings


def plot_annotated_fft(
    fft_image: np.ndarray,
    peak_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(np.asarray(fft_image, dtype=np.float32), cmap="gray")
    assign_by_row = {
        int(row.get("source_row_index", -1)): row for row in assignments
    }
    assign_by_pid = {int(row["peak_id"]): row for row in assignments}
    for row_idx, row in enumerate(peak_rows):
        try:
            pid = int(row.get("peak_id", row_idx))
        except (TypeError, ValueError):
            pid = row_idx
        x = float(row.get("refined_x", row.get("x", 0.0)))
        y = float(row.get("refined_y", row.get("y", 0.0)))
        rec = assign_by_row.get(row_idx, assign_by_pid.get(pid))
        matched = bool(rec and rec.get("matched", False))
        color = "lime" if matched else "tab:red"
        marker = "o" if matched else "x"
        ax.scatter([x], [y], s=24, c=color, marker=marker, linewidths=0.6)
        if rec and matched:
            label = (
                f"{pid}:({int(rec['h'])}{int(rec['k'])}{int(rec['l'])}) "
                f"d={float(rec['d_nm']):.3f}"
            )
            ax.text(
                x + 3.0,
                y + 2.0,
                label,
                fontsize=5.5,
                color="yellow",
                bbox={"facecolor": "black", "alpha": 0.32, "pad": 0.4, "edgecolor": "none"},
            )
        else:
            ax.text(
                x + 2.0,
                y + 1.5,
                str(pid),
                fontsize=5.5,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.24, "pad": 0.3, "edgecolor": "none"},
            )
    ax.set_title("Annotated FFT (matched=green, unmatched=red)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_simulated_diffraction(simulated_rows: list[dict[str, Any]], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    if simulated_rows:
        gx = [float(row["gx"]) for row in simulated_rows]
        gy = [float(row["gy"]) for row in simulated_rows]
        ax.scatter(gx, gy, s=22, c="tab:orange", alpha=0.8)
        for row in simulated_rows[: min(140, len(simulated_rows))]:
            ax.text(
                float(row["gx"]),
                float(row["gy"]),
                f"({int(row['h'])}{int(row['k'])}{int(row['l'])})\n{float(row['d_nm']):.3f}",
                fontsize=5.2,
            )
    ax.set_xlabel("gx (1/nm)")
    ax.set_ylabel("gy (1/nm)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_experimental_vs_simulated_overlay(
    peak_rows: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    simulated_rows: list[dict[str, Any]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    gx_exp = [float(row.get("gx_1_per_nm", row.get("g_x", 0.0))) for row in peak_rows]
    gy_exp = [float(row.get("gy_1_per_nm", row.get("g_y", 0.0))) for row in peak_rows]
    ax.scatter(gx_exp, gy_exp, s=26, c="tab:blue", alpha=0.75, label="Experimental")
    if simulated_rows:
        gx_sim = [float(row["gx"]) for row in simulated_rows]
        gy_sim = [float(row["gy"]) for row in simulated_rows]
        ax.scatter(gx_sim, gy_sim, s=20, c="tab:orange", alpha=0.7, label="Simulated")
    for row in assignments:
        if not bool(row.get("matched", False)):
            continue
        src_idx = int(row.get("source_row_index", -1))
        p = peak_rows[src_idx] if 0 <= src_idx < len(peak_rows) else None
        if p is None:
            pid = int(row["peak_id"])
            p = next(
                (item for item in peak_rows if int(item.get("peak_id", -1)) == pid),
                None,
            )
        if p is None:
            continue
        gx = float(p.get("gx_1_per_nm", p.get("g_x", 0.0)))
        gy = float(p.get("gy_1_per_nm", p.get("g_y", 0.0)))
        ax.text(gx, gy, f"({int(row['h'])}{int(row['k'])}{int(row['l'])})", fontsize=5.5)
    ax.set_xlabel("gx (1/nm)")
    ax.set_ylabel("gy (1/nm)")
    ax.set_title("Experimental vs Simulated Overlay")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def _zone_basis_from_direct(
    direct: np.ndarray,
    zone_axis: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    z = (
        float(zone_axis[0]) * direct[:, 0]
        + float(zone_axis[1]) * direct[:, 1]
        + float(zone_axis[2]) * direct[:, 2]
    )
    zn = float(np.linalg.norm(z))
    if zn <= 1e-10:
        return None
    ez = z / zn
    trial = direct[:, 0]
    if abs(float(np.dot(trial, ez))) > 0.95:
        trial = direct[:, 1]
    ex = trial - float(np.dot(trial, ez)) * ez
    exn = float(np.linalg.norm(ex))
    if exn <= 1e-10:
        return None
    ex = ex / exn
    ey = np.cross(ez, ex)
    eyn = float(np.linalg.norm(ey))
    if eyn <= 1e-10:
        return None
    ey = ey / eyn
    return ex, ey, ez


def plot_atom_projection(
    entry: PhaseEntry,
    *,
    zone_axis: tuple[int, int, int] = (0, 0, 1),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    direct = _direct_lattice_vectors(entry)
    if direct is None or not entry.atoms:
        ax.set_title("Atomic projection unavailable (missing CIF atom loop)")
        ax.axis("off")
        return fig
    basis = _zone_basis_from_direct(direct, zone_axis)
    if basis is None:
        ax.set_title("Atomic projection unavailable (invalid zone axis)")
        ax.axis("off")
        return fig
    ex, ey, _ez = basis
    symbols = sorted({str(row["symbol"]) for row in entry.atoms})
    cmap = plt.cm.get_cmap("tab10", max(1, len(symbols)))
    color_map = {sym: cmap(i) for i, sym in enumerate(symbols)}
    size_map = {sym: float(30 + 10 * (i % 6)) for i, sym in enumerate(symbols)}

    pts: list[tuple[float, float, str]] = []
    for tx in (-1, 0, 1):
        for ty in (-1, 0, 1):
            for atom in entry.atoms:
                frac = np.array(
                    [
                        float(atom["fx"]) + float(tx),
                        float(atom["fy"]) + float(ty),
                        float(atom["fz"]),
                    ],
                    dtype=np.float64,
                )
                cart = direct @ frac
                px = float(np.dot(cart, ex))
                py = float(np.dot(cart, ey))
                pts.append((px, py, str(atom["symbol"])))
    for sym in symbols:
        sub = [(x, y) for x, y, s in pts if s == sym]
        if not sub:
            continue
        xs = [x for x, _ in sub]
        ys = [y for _, y in sub]
        ax.scatter(
            xs,
            ys,
            s=size_map[sym],
            c=[color_map[sym]],
            alpha=0.82,
            edgecolors="black",
            linewidths=0.2,
            label=sym,
        )
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    cxy = []
    for frac in corners:
        cart = direct @ frac
        cxy.append([float(np.dot(cart, ex)), float(np.dot(cart, ey))])
    cxy_arr = np.array(cxy, dtype=np.float64)
    ax.plot(cxy_arr[:, 0], cxy_arr[:, 1], c="black", linewidth=1.0, alpha=0.7)
    zone_text = f"[{zone_axis[0]} {zone_axis[1]} {zone_axis[2]}]"
    ax.set_title(f"Atom Projection: {entry.name} zone {zone_text}")
    ax.set_xlabel("Projection X (nm)")
    ax.set_ylabel("Projection Y (nm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig
