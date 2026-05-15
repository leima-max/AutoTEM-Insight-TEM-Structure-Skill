"""A3 plot generation for baseline report figures."""

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


def _save_image_plot(image: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    ax.imshow(image, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_fft_plot(image: np.ndarray, path: Path, title: str) -> None:
    fft = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log1p(np.abs(fft))
    _save_image_plot(magnitude, path, title)


def _save_heatmap(image: np.ndarray, path: Path, title: str, cmap: str = "viridis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    handle = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _save_phase_candidate_plot(
    names: list[str],
    scores: list[float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=180)
    bars = ax.bar(names, scores, color="#2d9cdb")
    for bar, score in zip(bars, scores, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Phase candidate confidence")
    ax.grid(axis="y", alpha=0.2)
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


def build_core_plots(
    report: AnalysisReport,
    output_dir: str | Path,
    *,
    export_svg: bool = False,
) -> tuple[list[OutputArtifact], list[ReportFigure], list[str]]:
    """Generate baseline plot images for input, FFT magnitude, and strain maps."""
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    artifacts: list[OutputArtifact] = []
    figures: list[ReportFigure] = []
    warnings: list[str] = []

    dataset_image = None
    if report.dataset is not None:
        dataset_image = _load_array(report.dataset.data.uri)
    if dataset_image is not None and dataset_image.ndim == 2:
        input_plot_path = figures_dir / "input_image.png"
        _save_image_plot(dataset_image, input_plot_path, "Input image")
        if export_svg:
            _save_image_plot(dataset_image, input_plot_path.with_suffix(".svg"), "Input image")
        artifacts.append(
            OutputArtifact(
                kind="image",
                uri=str(input_plot_path),
                label="Input image",
                mime_type="image/png",
            )
        )
        figures.append(
            ReportFigure(
                label="Input image",
                uri=str(input_plot_path),
                caption="Input image preview.",
                figure_type="input",
            )
        )
        if export_svg:
            _append_svg_artifact(artifacts, input_plot_path, "Input image")

        fft_plot_path = figures_dir / "fft_magnitude.png"
        _save_fft_plot(dataset_image, fft_plot_path, "FFT magnitude")
        if export_svg:
            _save_fft_plot(dataset_image, fft_plot_path.with_suffix(".svg"), "FFT magnitude")
        artifacts.append(
            OutputArtifact(
                kind="image",
                uri=str(fft_plot_path),
                label="FFT magnitude",
                mime_type="image/png",
            )
        )
        figures.append(
            ReportFigure(
                label="FFT magnitude",
                uri=str(fft_plot_path),
                caption="Log-scaled FFT magnitude generated from input image.",
                figure_type="fft",
            )
        )
        if export_svg:
            _append_svg_artifact(artifacts, fft_plot_path, "FFT magnitude")
    else:
        warnings.append(
            "A3 plot generation skipped input/FFT figures because 2D pixels were unavailable."
        )

    if report.diffraction is not None and report.diffraction.phase_candidates:
        phase_names = [candidate.name for candidate in report.diffraction.phase_candidates]
        phase_scores = [
            float(candidate.score or 0.0) for candidate in report.diffraction.phase_candidates
        ]
        phase_path = figures_dir / "phase_candidates.png"
        _save_phase_candidate_plot(phase_names, phase_scores, phase_path)
        if export_svg:
            _save_phase_candidate_plot(phase_names, phase_scores, phase_path.with_suffix(".svg"))
        artifacts.append(
            OutputArtifact(
                kind="image",
                uri=str(phase_path),
                label="Phase candidates",
                mime_type="image/png",
            )
        )
        figures.append(
            ReportFigure(
                label="Phase candidates",
                uri=str(phase_path),
                caption="Top candidate phases with confidence scores.",
                figure_type="phase_candidates",
            )
        )
        if export_svg:
            _append_svg_artifact(artifacts, phase_path, "Phase candidates")

    if report.strain is not None:
        strain_refs = {
            "strain_xx": report.strain.strain_xx,
            "strain_yy": report.strain.strain_yy,
            "strain_xy": report.strain.strain_xy,
            "rotation_xy": report.strain.rotation_xy,
        }
        for field_name, ref in strain_refs.items():
            strain_array = _load_array(ref.uri) if ref is not None else None
            if strain_array is None or strain_array.ndim != 2:
                continue
            plot_path = figures_dir / f"{field_name}.png"
            _save_heatmap(strain_array, plot_path, field_name.replace("_", " "), cmap="coolwarm")
            if export_svg:
                _save_heatmap(
                    strain_array,
                    plot_path.with_suffix(".svg"),
                    field_name.replace("_", " "),
                    cmap="coolwarm",
                )
            artifacts.append(
                OutputArtifact(
                    kind="image",
                    uri=str(plot_path),
                    label=field_name,
                    mime_type="image/png",
                )
            )
            figures.append(
                ReportFigure(
                    label=field_name,
                    uri=str(plot_path),
                    caption=f"{field_name} map.",
                    figure_type="strain",
                )
            )
            if export_svg:
                _append_svg_artifact(artifacts, plot_path, field_name)

    return artifacts, figures, warnings
