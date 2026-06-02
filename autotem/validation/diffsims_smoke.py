"""Standalone validation helper for the real diffsims adapter."""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from autotem.core import DiffractionPeak, DiffractionResult, GVector, Measurement, PhaseCandidate
from autotem.diffraction.phase_lookup import lookup_phase_candidates
from autotem.diffraction.phase_matching import match_diffsims_patterns_to_diffraction
from autotem.simulation.diffsims_adapter import DiffsimsSimulationResult
from autotem.simulation.diffsims_adapter import (
    simulate_diffraction_with_diffsims as _simulate_diffraction_with_diffsims,
)

_DEFAULT_MATERIALS = ("si", "cu")
_DEPENDENCY_NAMES = ("diffsims", "diffpy.structure", "orix")


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _DEPENDENCY_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    return versions


def _candidate_from_lookup(material: str) -> tuple[PhaseCandidate | None, list[str]]:
    lookup = lookup_phase_candidates(
        formula=None,
        elements=[],
        material_names=[material],
        sources=["local_builtin"],
        limit_per_source=1,
        timeout_s=1,
    )
    warnings = list(lookup.warnings)
    if not lookup.candidates:
        return None, warnings or [f"No local built-in candidate found for '{material}'."]

    row = lookup.candidates[0]
    name = str(row.get("name") or material)
    lattice = {
        "a": row.get("a"),
        "b": row.get("b"),
        "c": row.get("c"),
        "alpha": row.get("alpha", 90.0),
        "beta": row.get("beta", 90.0),
        "gamma": row.get("gamma", 90.0),
        "unit": "nm",
        "space_group": row.get("space_group", "P1"),
        "space_group_number": row.get("space_group_number"),
        "atom_sites": row.get("atom_sites", []),
    }
    return (
        PhaseCandidate(
            name=name,
            formula=None if row.get("formula") is None else str(row.get("formula")),
            source=None if row.get("source") is None else str(row.get("source")),
            identifier=None if row.get("identifier") is None else str(row.get("identifier")),
            score=0.1,
            lattice_parameters=lattice,
            zone_axis=(0, 0, 1),
        ),
        warnings,
    )


def _diffraction_from_simulated_spots(
    candidate: PhaseCandidate,
    simulation: DiffsimsSimulationResult,
    *,
    max_measured_spots: int,
) -> DiffractionResult:
    spots = [
        spot
        for pattern in simulation.patterns
        for spot in pattern.spots
        if spot.magnitude is not None
    ]
    spots.sort(
        key=lambda spot: (
            0.0 if spot.intensity is None else float(spot.intensity),
            0.0 if spot.magnitude is None else float(spot.magnitude.value or 0.0),
        ),
        reverse=True,
    )
    selected = spots[: max(1, int(max_measured_spots))]

    peaks: list[DiffractionPeak] = []
    g_vectors: list[GVector] = []
    for idx, spot in enumerate(selected):
        gx = float(spot.gx.value or 0.0)
        gy = float(spot.gy.value or 0.0)
        magnitude = float(
            spot.magnitude.value if spot.magnitude is not None else np.hypot(gx, gy)
        )
        intensity = 1.0 if spot.intensity is None else float(spot.intensity)
        peaks.append(
            DiffractionPeak(
                x=float(idx),
                y=0.0,
                intensity=intensity,
                confidence=1.0,
            )
        )
        g_vectors.append(
            GVector(
                gx=Measurement(value=gx, unit="1/nm", method="diffsims_smoke"),
                gy=Measurement(value=gy, unit="1/nm", method="diffsims_smoke"),
                magnitude=Measurement(
                    value=magnitude,
                    unit="1/nm",
                    method="diffsims_smoke",
                ),
                source_peak_index=idx,
            )
        )

    return DiffractionResult(
        peaks=peaks,
        g_vectors=g_vectors,
        phase_candidates=[candidate],
        provenance={"method": "diffsims_smoke_synthetic_measured_spots"},
    )


def _material_result(
    *,
    material: str,
    voltage_kv: float,
    reciprocal_radius_inv_angstrom: float,
    max_simulated_spots: int,
    max_measured_spots: int,
    min_matched_spots: int,
    simulate_func: Callable[..., DiffsimsSimulationResult],
    progress_func: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if progress_func is not None:
        progress_func("candidate_lookup_start")
    candidate, lookup_warnings = _candidate_from_lookup(material)
    if candidate is None:
        return {
            "material": material,
            "ok": False,
            "stage": "candidate_lookup",
            "warnings": lookup_warnings,
        }

    if progress_func is not None:
        progress_func("simulation_start")
    simulation = simulate_func(
        [candidate],
        voltage_kv=voltage_kv,
        reciprocal_radius_inv_angstrom=reciprocal_radius_inv_angstrom,
        max_candidates=1,
        max_simulated_spots=max(1, int(max_simulated_spots)),
        minimum_simulated_g_inv_nm=1e-6,
        with_direct_beam=True,
        max_excitation_error=0.01,
        rotation_angles_deg=[0.0],
        fallback_single_atom=False,
        pattern_shape=(64, 64),
    )
    if progress_func is not None:
        progress_func("simulation_done")
    warnings = [*lookup_warnings, *simulation.warnings]
    simulated_spot_count = sum(len(pattern.spots) for pattern in simulation.patterns)
    if not simulation.patterns or simulated_spot_count == 0:
        return {
            "material": material,
            "phase": candidate.name,
            "formula": candidate.formula,
            "source": candidate.source,
            "identifier": candidate.identifier,
            "ok": False,
            "stage": "diffsims_simulation",
            "pattern_count": len(simulation.patterns),
            "simulated_spot_count": simulated_spot_count,
            "warnings": warnings,
            "provenance": simulation.provenance,
        }

    if progress_func is not None:
        progress_func("phase_matching_start")
    diffraction = _diffraction_from_simulated_spots(
        candidate,
        simulation,
        max_measured_spots=max(1, int(max_measured_spots)),
    )
    if progress_func is not None:
        progress_func("phase_matching_done")
    match = match_diffsims_patterns_to_diffraction(
        diffraction,
        simulation.patterns,
        d_tolerance_rel=0.02,
        angle_tolerance_deg=2.0,
        min_matched_spots=max(1, int(min_matched_spots)),
        top_n=1,
        previous_score_weight=0.0,
        max_rotation_candidates=32,
    )
    best = match.candidates[0] if match.candidates else None
    matched_spots = int(best.matched_spot_count or 0) if best is not None else 0
    ok = bool(match.provenance.get("active")) and matched_spots >= max(1, min_matched_spots)

    return {
        "material": material,
        "phase": candidate.name,
        "formula": candidate.formula,
        "source": candidate.source,
        "identifier": candidate.identifier,
        "ok": ok,
        "stage": "closed_loop_phase_matching" if ok else "phase_matching",
        "pattern_count": len(simulation.patterns),
        "simulated_spot_count": simulated_spot_count,
        "measured_spot_count": len(diffraction.g_vectors),
        "matched_spot_count": matched_spots,
        "score": None if best is None else best.score,
        "mean_d_error": (
            None
            if best is None or best.mean_d_error is None
            else best.mean_d_error.value
        ),
        "mean_angle_error_deg": (
            None
            if best is None or best.mean_angle_error is None
            else best.mean_angle_error.value
        ),
        "warnings": [*warnings, *match.warnings],
        "simulation_provenance": simulation.provenance,
        "phase_matching_provenance": match.provenance,
    }


def _temporary_directory_root() -> str | None:
    root = os.environ.get("AUTOTEM_TMP_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        return root
    if os.name == "nt":
        windows_root = Path("C:/tmp")
        try:
            windows_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return str(windows_root)
    return None


def _tail_text(text: str, *, limit: int = 2000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _material_result_with_timeout(
    *,
    timeout_s: float,
    **kwargs: Any,
) -> dict[str, Any]:
    timeout = max(1.0, float(timeout_s))
    material = str(kwargs.get("material", "unknown"))
    with tempfile.TemporaryDirectory(
        prefix="autotem_diffsims_validation_",
        dir=_temporary_directory_root(),
    ) as tmp_dir:
        input_path = Path(tmp_dir) / "request.json"
        output_path = Path(tmp_dir) / "response.json"
        progress_path = Path(tmp_dir) / "progress.jsonl"
        stdout_path = Path(tmp_dir) / "stdout.txt"
        stderr_path = Path(tmp_dir) / "stderr.txt"
        request = {**dict(kwargs), "_progress_path": str(progress_path)}
        input_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle:
                with stderr_path.open("w", encoding="utf-8") as stderr_handle:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "autotem.validation.diffsims_smoke_worker",
                            "--input",
                            str(input_path),
                            "--output",
                            str(output_path),
                        ],
                        cwd=Path.cwd(),
                        stdout=stdout_handle,
                        stderr=stderr_handle,
                        text=True,
                        timeout=timeout,
                        check=False,
                    )
        except subprocess.TimeoutExpired:
            progress = (
                progress_path.read_text(encoding="utf-8").splitlines()
                if progress_path.exists()
                else []
            )
            return {
                "material": material,
                "ok": False,
                "stage": "diffsims_timeout",
                "warnings": [
                    f"diffsims validation timed out after {timeout:.1f}s for '{material}'."
                    + (f" Last progress: {progress[-1]}" if progress else "")
                ],
                "progress": progress,
            }
        if completed.returncode != 0:
            stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.exists() else ""
            stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
            detail = _tail_text((stderr or stdout).strip())
            return {
                "material": material,
                "ok": False,
                "stage": "diffsims_worker_failed",
                "warnings": [
                    "diffsims validation subprocess failed"
                    + (f": {detail}" if detail else f" with exit code {completed.returncode}")
                ],
            }
        if not output_path.exists():
            return {
                "material": material,
                "ok": False,
                "stage": "diffsims_worker_empty",
                "warnings": ["diffsims validation subprocess exited without a result."],
            }
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, dict)
            and payload.get("ok")
            and isinstance(payload.get("result"), dict)
        ):
            return payload["result"]
        error = (
            str(payload.get("error", "unknown error"))
            if isinstance(payload, dict)
            else "invalid validation subprocess payload"
        )
        return {
            "material": material,
            "ok": False,
            "stage": "diffsims_worker_failed",
            "warnings": ["diffsims validation subprocess failed: " + error],
        }
    return {
        "material": material,
        "ok": False,
        "stage": "diffsims_worker_invalid_result",
        "warnings": ["diffsims validation subprocess returned an invalid result."],
    }


def run_diffsims_smoke_validation(
    *,
    materials: list[str] | tuple[str, ...] | None = None,
    voltage_kv: float = 200.0,
    reciprocal_radius_inv_angstrom: float = 1.5,
    max_simulated_spots: int = 128,
    max_measured_spots: int = 6,
    min_matched_spots: int = 3,
    timeout_s: float | None = 30.0,
    simulate_func: Callable[..., DiffsimsSimulationResult] | None = None,
) -> dict[str, Any]:
    """Validate real diffsims simulation and template matching without pytest."""
    requested_materials = [
        str(item).strip().lower()
        for item in (materials or _DEFAULT_MATERIALS)
        if str(item).strip()
    ]
    if not requested_materials:
        requested_materials = list(_DEFAULT_MATERIALS)
    dependency_versions = _dependency_versions()
    material_results = []
    for material in requested_materials:
        kwargs = {
            "material": material,
            "voltage_kv": voltage_kv,
            "reciprocal_radius_inv_angstrom": reciprocal_radius_inv_angstrom,
            "max_simulated_spots": max_simulated_spots,
            "max_measured_spots": max_measured_spots,
            "min_matched_spots": min_matched_spots,
        }
        if simulate_func is None and timeout_s is not None and float(timeout_s) > 0:
            material_results.append(
                _material_result_with_timeout(timeout_s=float(timeout_s), **kwargs)
            )
        else:
            material_results.append(
                _material_result(
                    **kwargs,
                    simulate_func=simulate_func or _simulate_diffraction_with_diffsims,
                )
            )
    ok = all(bool(item.get("ok")) for item in material_results)
    missing = [
        name
        for name, version in dependency_versions.items()
        if version == "not_installed"
    ]
    status = "passed" if ok and not missing else "failed"
    warnings = [
        warning
        for item in material_results
        for warning in item.get("warnings", [])
        if isinstance(warning, str) and warning
    ]
    if missing:
        warnings.append(
            "Missing diffsims runtime dependency/dependencies: " + ", ".join(missing)
        )
    return {
        "status": status,
        "ok": status == "passed",
        "dependencies": dependency_versions,
        "materials": material_results,
        "parameters": {
            "voltage_kv": float(voltage_kv),
            "reciprocal_radius_inv_angstrom": float(reciprocal_radius_inv_angstrom),
            "max_simulated_spots": int(max_simulated_spots),
            "max_measured_spots": int(max_measured_spots),
            "min_matched_spots": int(min_matched_spots),
            "timeout_s": None if timeout_s is None else float(timeout_s),
        },
        "warnings": list(dict.fromkeys(warnings)),
    }
