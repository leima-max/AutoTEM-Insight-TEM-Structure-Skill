"""A3 overlay generation for diffraction peaks and atom coordinates."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autotem.core import AnalysisReport, OutputArtifact, ReportFigure

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover
    iio = None


def _load_array(uri: str | None) -> np.ndarray | None:
    if not uri:
        return None
    path = Path(uri)
    if not path.exists() or not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix in {".txt", ".csv"}:
        return np.loadtxt(path)
    if suffix in {".tif", ".tiff", ".png", ".jpg", ".jpeg"} and iio is not None:
        image = np.asarray(iio.imread(path))
        if image.ndim == 3:
            return np.asarray(image[..., 0])
        return image
    return None


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
    if dataset_image is None or dataset_image.ndim != 2:
        warnings.append("A3 overlay generation skipped because 2D image pixels were unavailable.")
        return artifacts, figures, warnings

    if report.diffraction is not None and report.diffraction.peaks:
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

    if report.atom_lattice is not None and report.atom_lattice.atoms:
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

    return artifacts, figures, warnings
