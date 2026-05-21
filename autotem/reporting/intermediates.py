"""A3 intermediate artifact persistence with layered large-array storage."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from autotem.core import AnalysisReport, OutputArtifact

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

try:
    import imageio.v3 as iio
except ImportError:  # pragma: no cover
    iio = None

_LARGE_ARRAY_BYTES_THRESHOLD = 16 * 1024 * 1024


def _load_array_from_uri(uri: str | None) -> np.ndarray | None:
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


def _save_preview_image(array: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = np.asarray(array)
    if display.ndim > 2:
        display = np.asarray(display[0])
    fig, ax = plt.subplots(figsize=(4, 4), dpi=140)
    ax.imshow(display, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _persist_layered_array(
    *,
    array: np.ndarray,
    name: str,
    intermediates_dir: Path,
) -> tuple[OutputArtifact, list[OutputArtifact]]:
    aux_artifacts: list[OutputArtifact] = []
    bytes_size = int(array.nbytes)
    preview_path = intermediates_dir / f"{name}_preview.png"
    _save_preview_image(array, preview_path, f"{name} preview")
    aux_artifacts.append(
        OutputArtifact(
            kind="image",
            uri=str(preview_path),
            label=f"Intermediate {name} preview",
            mime_type="image/png",
        )
    )

    metadata = {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "bytes": bytes_size,
        "storage_tier": "small_npy",
        "threshold_bytes": _LARGE_ARRAY_BYTES_THRESHOLD,
    }
    if bytes_size >= _LARGE_ARRAY_BYTES_THRESHOLD and h5py is not None:
        h5_path = intermediates_dir / "large_arrays.h5"
        dataset_key = name
        with h5py.File(h5_path, "a") as handle:
            if dataset_key in handle:
                del handle[dataset_key]
            handle.create_dataset(dataset_key, data=array, compression="gzip")
        metadata["storage_tier"] = "large_hdf5"
        metadata["dataset"] = dataset_key
        main_artifact = OutputArtifact(
            kind="array",
            uri=f"{h5_path}::/{dataset_key}",
            label=f"Intermediate {name} array (HDF5)",
            mime_type="application/x-hdf5",
        )
    else:
        npy_path = intermediates_dir / f"{name}.npy"
        np.save(npy_path, array)
        if bytes_size >= _LARGE_ARRAY_BYTES_THRESHOLD and h5py is None:
            metadata["storage_tier"] = "large_npy_fallback"
            metadata["fallback_reason"] = "h5py_not_installed"
        main_artifact = OutputArtifact(
            kind="array",
            uri=str(npy_path),
            label=f"Intermediate {name} array",
            mime_type="application/octet-stream",
        )

    metadata_path = intermediates_dir / f"{name}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    aux_artifacts.append(
        OutputArtifact(
            kind="json",
            uri=str(metadata_path),
            label=f"Intermediate {name} metadata",
            mime_type="application/json",
        )
    )
    return main_artifact, aux_artifacts


def persist_intermediate_artifacts(
    report: AnalysisReport,
    output_dir: str | Path,
) -> list[OutputArtifact]:
    """Persist reproducibility-oriented intermediate arrays and metadata."""
    output_dir = Path(output_dir)
    intermediates_dir = output_dir / "intermediates"
    intermediates_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[OutputArtifact] = []

    dataset_uri = report.dataset.data.uri if report.dataset is not None else None
    dataset_array = _load_array_from_uri(dataset_uri)
    if dataset_array is not None:
        main_artifact, aux_artifacts = _persist_layered_array(
            array=np.asarray(dataset_array),
            name="input_image",
            intermediates_dir=intermediates_dir,
        )
        artifacts.append(main_artifact)
        artifacts.extend(aux_artifacts)

    preprocessing_uri = (
        report.preprocessing.image.uri
        if report.preprocessing is not None and report.preprocessing.image is not None
        else None
    )
    preprocessing_array = _load_array_from_uri(preprocessing_uri)
    if preprocessing_array is not None:
        main_artifact, aux_artifacts = _persist_layered_array(
            array=np.asarray(preprocessing_array),
            name="preprocessed_image",
            intermediates_dir=intermediates_dir,
        )
        artifacts.append(main_artifact)
        artifacts.extend(aux_artifacts)

        fft = np.fft.fftshift(np.fft.fft2(preprocessing_array))
        fft_mag = np.abs(fft).astype(np.float32)
        fft_artifact, fft_aux_artifacts = _persist_layered_array(
            array=fft_mag,
            name="fft_magnitude",
            intermediates_dir=intermediates_dir,
        )
        artifacts.append(fft_artifact)
        artifacts.extend(fft_aux_artifacts)

    if report.diffraction is not None:
        peaks_path = intermediates_dir / "diffraction_peaks.json"
        peaks_payload = [peak.model_dump(mode="json") for peak in report.diffraction.peaks]
        peaks_path.write_text(json.dumps(peaks_payload, indent=2), encoding="utf-8")
        artifacts.append(
            OutputArtifact(
                kind="json",
                uri=str(peaks_path),
                label="Intermediate diffraction peaks",
                mime_type="application/json",
            )
        )

    if report.atom_lattice is not None and report.atom_lattice.atoms:
        atom_array = np.array(
            [[atom.x, atom.y] for atom in report.atom_lattice.atoms],
            dtype=np.float32,
        )
        atom_artifact, atom_aux = _persist_layered_array(
            array=atom_array,
            name="atom_coordinates",
            intermediates_dir=intermediates_dir,
        )
        artifacts.append(atom_artifact)
        artifacts.extend(atom_aux)

    if report.workflow is not None:
        workflow_steps_path = intermediates_dir / "workflow_steps.json"
        payload: list[dict[str, Any]] = [
            step.model_dump(mode="json") for step in report.workflow.steps
        ]
        workflow_steps_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        artifacts.append(
            OutputArtifact(
                kind="json",
                uri=str(workflow_steps_path),
                label="Intermediate workflow step records",
                mime_type="application/json",
            )
        )

    # Preserve effective workflow copy in intermediates for stable reruns.
    effective_workflow_path = output_dir / "workflow_used.yaml"
    if effective_workflow_path.exists():
        target = intermediates_dir / "workflow_used.yaml"
        shutil.copy2(effective_workflow_path, target)
        artifacts.append(
            OutputArtifact(
                kind="other",
                uri=str(target),
                label="Intermediate effective workflow",
                mime_type="application/x-yaml",
            )
        )

    return artifacts
