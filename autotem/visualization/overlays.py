"""A3 overlay generation for diffraction peaks and atom coordinates."""

from __future__ import annotations

from math import cos, radians, sin
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from autotem.core import AnalysisReport, OutputArtifact, ReportFigure
from autotem.io.readers import read_numeric_file

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover
    iio = None

plt.switch_backend("Agg")


def _load_array(uri: str | None) -> np.ndarray | None:
    if not uri:
        return None
    path = Path(uri)
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix in {".npy", ".txt", ".csv"}:
        return read_numeric_file(path)
    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"} and iio is not None:
        image = np.asarray(iio.imread(path))
        if image.ndim == 3:
            return np.asarray(image[..., 0])
        return image
    return None


def _as_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _as_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _measurement_value(measurement: object) -> float | None:
    return _as_float(getattr(measurement, "value", None))


def _gvector_xy(g_vector: object) -> tuple[float, float] | None:
    gx = _measurement_value(getattr(g_vector, "gx", None))
    gy = _measurement_value(getattr(g_vector, "gy", None))
    if gx is None or gy is None:
        return None
    if not np.isfinite([gx, gy]).all():
        return None
    return gx, gy


def _spot_xy(spot: object, *, rotation_deg: float = 0.0) -> tuple[float, float] | None:
    gx = _measurement_value(getattr(spot, "gx", None))
    gy = _measurement_value(getattr(spot, "gy", None))
    if gx is None or gy is None:
        return None
    if not np.isfinite([gx, gy]).all():
        return None
    theta = radians(float(rotation_deg))
    cr = cos(theta)
    sr = sin(theta)
    return gx * cr - gy * sr, gx * sr + gy * cr


def _candidate_keys(candidate: object | None) -> set[str]:
    if candidate is None:
        return set()
    values = (getattr(candidate, "identifier", None), getattr(candidate, "name", None))
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _pattern_keys(pattern: object) -> set[str]:
    structure_id = getattr(pattern, "structure_id", None)
    return {str(structure_id).strip().lower()} if str(structure_id).strip() else set()


def _best_phase_row(report: AnalysisReport, phase_name: str) -> dict[str, Any]:
    if report.simulation is None:
        return {}
    phase_matching = _as_mapping(report.simulation.provenance.get("phase_matching", {}))
    per_phase = phase_matching.get("per_phase", [])
    if not isinstance(per_phase, list):
        return {}
    for row in per_phase:
        if not isinstance(row, dict):
            continue
        if str(row.get("phase_name", "")) == phase_name:
            return dict(row)
    return {}


def _best_simulation_pattern(report: AnalysisReport) -> object | None:
    if report.simulation is None or not report.simulation.patterns:
        return None
    best_candidate = (
        report.simulation.matched_phase_candidates[0]
        if report.simulation.matched_phase_candidates
        else None
    )
    phase_name = str(getattr(best_candidate, "name", "") or "")
    phase_row = _best_phase_row(report, phase_name)
    pattern_index = phase_row.get("best_pattern_index")
    if isinstance(pattern_index, int) and 0 <= pattern_index < len(report.simulation.patterns):
        return report.simulation.patterns[pattern_index]
    if isinstance(pattern_index, str) and pattern_index.strip().isdigit():
        idx = int(pattern_index)
        if 0 <= idx < len(report.simulation.patterns):
            return report.simulation.patterns[idx]

    candidate_keys = _candidate_keys(best_candidate)
    fallback = None
    for pattern in report.simulation.patterns:
        if fallback is None:
            fallback = pattern
        if candidate_keys and not candidate_keys.intersection(_pattern_keys(pattern)):
            continue
        if _as_mapping(pattern.simulation_parameters.get("phase_match", {})):
            return pattern
        fallback = pattern
    return fallback


def _save_scatter_overlay(
    base_image: np.ndarray,
    points_xy: np.ndarray,
    path: Path,
    title: str,
    color: str = "#ff5a3d",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.imshow(base_image, cmap="gray")
    if points_xy.size > 0:
        ax.scatter(
            points_xy[:, 0],
            points_xy[:, 1],
            s=18,
            c=color,
            linewidths=0.3,
            edgecolors="white",
        )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_simulation_diffraction_overlay(
    *,
    measured: dict[int, tuple[float, float]],
    simulated: dict[int, tuple[float, float]],
    matched_pairs: list[dict[str, Any]],
    path: Path,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    measured_xy = np.asarray(list(measured.values()), dtype=float)
    simulated_xy = np.asarray(list(simulated.values()), dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=180)
    ax.axhline(0.0, color="#8a8a8a", linewidth=0.5, alpha=0.7)
    ax.axvline(0.0, color="#8a8a8a", linewidth=0.5, alpha=0.7)
    if measured_xy.size:
        ax.scatter(
            measured_xy[:, 0],
            measured_xy[:, 1],
            s=36,
            facecolors="none",
            edgecolors="#1f77b4",
            linewidths=1.2,
            label="measured",
        )
    if simulated_xy.size:
        ax.scatter(
            simulated_xy[:, 0],
            simulated_xy[:, 1],
            s=32,
            marker="x",
            c="#d64b3a",
            linewidths=1.2,
            label="simulated",
        )
    for pair in matched_pairs:
        measured_idx = pair.get("measured_g_index")
        simulated_idx = pair.get("simulated_spot_index")
        if not isinstance(measured_idx, int) or not isinstance(simulated_idx, int):
            continue
        if measured_idx not in measured or simulated_idx not in simulated:
            continue
        mx, my = measured[measured_idx]
        sx, sy = simulated[simulated_idx]
        ax.plot([mx, sx], [my, sy], color="#4c566a", linewidth=0.8, alpha=0.55)

    all_points = []
    if measured_xy.size:
        all_points.append(measured_xy)
    if simulated_xy.size:
        all_points.append(simulated_xy)
    if all_points:
        stacked = np.vstack(all_points)
        max_radius = float(np.nanmax(np.abs(stacked)))
        max_radius = max(max_radius * 1.15, 1.0)
        ax.set_xlim(-max_radius, max_radius)
        ax.set_ylim(-max_radius, max_radius)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("gx (1/nm)")
    ax.set_ylabel("gy (1/nm)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _append_svg_artifact(
    artifacts: list[OutputArtifact],
    png_path: Path,
    label: str,
) -> None:
    svg_path = png_path.with_suffix(".svg")
    if not svg_path.exists():
        return
    artifacts.append(
        OutputArtifact(
            kind="image",
            uri=str(svg_path),
            label=f"{label} (SVG)",
            mime_type="image/svg+xml",
        )
    )


def _build_simulation_diffraction_overlay(
    report: AnalysisReport,
    figures_dir: Path,
    *,
    export_svg: bool,
) -> tuple[list[OutputArtifact], list[ReportFigure], list[str]]:
    artifacts: list[OutputArtifact] = []
    figures: list[ReportFigure] = []
    warnings: list[str] = []
    if report.diffraction is None or report.simulation is None:
        return artifacts, figures, warnings

    pattern = _best_simulation_pattern(report)
    if pattern is None or not getattr(pattern, "spots", []):
        return artifacts, figures, warnings

    measured: dict[int, tuple[float, float]] = {}
    for idx, g_vector in enumerate(report.diffraction.g_vectors):
        xy = _gvector_xy(g_vector)
        if xy is not None:
            measured[idx] = xy
    if not measured:
        warnings.append(
            "B5 simulation overlay skipped because no measured g-vectors were available."
        )
        return artifacts, figures, warnings

    phase_match = _as_mapping(pattern.simulation_parameters.get("phase_match", {}))
    rotation_deg = _as_float(phase_match.get("rotation_deg")) or 0.0
    simulated: dict[int, tuple[float, float]] = {}
    for idx, spot in enumerate(pattern.spots):
        xy = _spot_xy(spot, rotation_deg=rotation_deg)
        if xy is not None:
            simulated[idx] = xy
    if not simulated:
        warnings.append(
            "B5 simulation overlay skipped because no simulated reciprocal spots were available."
        )
        return artifacts, figures, warnings

    matched_pairs_raw = phase_match.get("matched_pairs", [])
    matched_pairs = [dict(row) for row in matched_pairs_raw if isinstance(row, dict)]
    zone_axis = getattr(pattern, "zone_axis", None)
    phase_label = getattr(pattern, "structure_id", "")
    title = "Measured vs simulated diffraction"
    if phase_label:
        title += f" ({phase_label})"
    overlay_path = figures_dir / "simulation_diffraction_overlay.png"
    _save_simulation_diffraction_overlay(
        measured=measured,
        simulated=simulated,
        matched_pairs=matched_pairs,
        path=overlay_path,
        title=title,
    )
    if export_svg:
        _save_simulation_diffraction_overlay(
            measured=measured,
            simulated=simulated,
            matched_pairs=matched_pairs,
            path=overlay_path.with_suffix(".svg"),
            title=title,
        )

    provenance = {
        "structure_id": phase_label,
        "zone_axis": list(zone_axis) if zone_axis is not None else None,
        "phase_match_rotation_deg": rotation_deg,
        "matched_pair_count": len(matched_pairs),
    }
    artifact = OutputArtifact(
        kind="image",
        uri=str(overlay_path),
        label="Simulation diffraction overlay",
        mime_type="image/png",
        description="Measured g-vectors and best simulated diffsims spots in reciprocal space.",
        provenance=provenance,
    )
    artifacts.append(artifact)
    if report.simulation is not None:
        report.simulation.overlays = [*report.simulation.overlays, artifact]
    if export_svg:
        _append_svg_artifact(artifacts, overlay_path, "Simulation diffraction overlay")
    figures.append(
        ReportFigure(
            label="Simulation diffraction overlay",
            uri=str(overlay_path),
            caption=(
                "Measured g-vectors overlaid with the best simulated diffraction spots."
            ),
            figure_type="simulation_diffraction_overlay",
            provenance=provenance,
        )
    )
    return artifacts, figures, warnings


def build_core_overlays(
    report: AnalysisReport,
    output_dir: str | Path,
    *,
    export_svg: bool = False,
) -> tuple[list[OutputArtifact], list[ReportFigure], list[str]]:
    """Generate baseline overlays from diffraction peaks and atom coordinates."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    artifacts: list[OutputArtifact] = []
    figures: list[ReportFigure] = []
    warnings: list[str] = []

    dataset_image = None
    if report.dataset is not None:
        dataset_image = _load_array(report.dataset.data.uri)
    has_image = dataset_image is not None and dataset_image.ndim == 2
    if not has_image and (
        (report.diffraction is not None and report.diffraction.peaks)
        or (report.atom_lattice is not None and report.atom_lattice.atoms)
    ):
        warnings.append("A3 overlay generation skipped because 2D image pixels were unavailable.")

    if has_image and report.diffraction is not None and report.diffraction.peaks:
        fft_magnitude = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(dataset_image))))
        peak_points = np.array([[peak.x, peak.y] for peak in report.diffraction.peaks], dtype=float)
        peak_path = figures_dir / "fft_peaks_overlay.png"
        _save_scatter_overlay(
            fft_magnitude,
            peak_points,
            peak_path,
            "FFT peaks overlay",
            color="#2ab7ca",
        )
        if export_svg:
            _save_scatter_overlay(
                fft_magnitude,
                peak_points,
                peak_path.with_suffix(".svg"),
                "FFT peaks overlay",
                color="#2ab7ca",
            )
        artifacts.append(
            OutputArtifact(
                kind="image",
                uri=str(peak_path),
                label="FFT peaks overlay",
                mime_type="image/png",
            )
        )
        if export_svg:
            _append_svg_artifact(artifacts, peak_path, "FFT peaks overlay")
        figures.append(
            ReportFigure(
                label="FFT peaks overlay",
                uri=str(peak_path),
                caption="Detected peaks overlaid on FFT magnitude.",
                figure_type="diffraction_overlay",
            )
        )

    if has_image and report.atom_lattice is not None and report.atom_lattice.atoms:
        atom_points = np.array(
            [[atom.x, atom.y] for atom in report.atom_lattice.atoms],
            dtype=float,
        )
        atom_path = figures_dir / "atom_overlay.png"
        _save_scatter_overlay(dataset_image, atom_points, atom_path, "Atom coordinates overlay")
        if export_svg:
            _save_scatter_overlay(
                dataset_image,
                atom_points,
                atom_path.with_suffix(".svg"),
                "Atom coordinates overlay",
            )
        artifacts.append(
            OutputArtifact(
                kind="image",
                uri=str(atom_path),
                label="Atom overlay",
                mime_type="image/png",
            )
        )
        if export_svg:
            _append_svg_artifact(artifacts, atom_path, "Atom overlay")
        figures.append(
            ReportFigure(
                label="Atom overlay",
                uri=str(atom_path),
                caption="Detected atom coordinates overlaid on input image.",
                figure_type="atom_overlay",
            )
        )
    elif report.atom_lattice is not None and report.atom_lattice.overlay is not None:
        overlay_uri = report.atom_lattice.overlay.uri
        if overlay_uri:
            artifacts.append(
                OutputArtifact(
                    kind="image",
                    uri=overlay_uri,
                    label="Atom overlay (external)",
                    mime_type="image/png",
                )
            )
            figures.append(
                ReportFigure(
                    label="Atom overlay (external)",
                    uri=overlay_uri,
                    caption="Atom overlay supplied by atom-detection module.",
                    figure_type="atom_overlay_external",
                )
            )

    simulation_artifacts, simulation_figures, simulation_warnings = (
        _build_simulation_diffraction_overlay(report, figures_dir, export_svg=export_svg)
    )
    artifacts.extend(simulation_artifacts)
    figures.extend(simulation_figures)
    warnings.extend(simulation_warnings)

    return artifacts, figures, warnings
