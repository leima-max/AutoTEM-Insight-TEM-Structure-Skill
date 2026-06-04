"""Training-plan contracts for future AutoTEM deep-learning models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from autotem.core import AIModelInfo, Measurement, OutputArtifact
from autotem.core.models import AutoTEMModel, JsonDict

DatasetSplit = Literal["train", "validation", "test", "holdout"]
DatasetSource = Literal["synthetic", "experimental", "mixed", "unknown"]
TrainingStatus = Literal["planned", "running", "completed", "failed", "cancelled"]


class TrainingDatasetEntry(AutoTEMModel):
    """One dataset or patch collection used by a training plan."""

    uri: str
    split: DatasetSplit = "train"
    source: DatasetSource = "unknown"
    label_uri: str | None = None
    modality: str = "unknown"
    sample_id: str = ""
    instrument_id: str = ""
    task_tags: list[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TrainingDatasetManifest(AutoTEMModel):
    """Serializable manifest for synthetic and experimental training data."""

    dataset_id: str
    task: str
    entries: list[TrainingDatasetEntry] = Field(default_factory=list)
    label_schema: JsonDict = Field(default_factory=dict)
    generation_method: str = ""
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TrainingRunConfig(AutoTEMModel):
    """Configuration for a planned or executed training run."""

    run_id: str
    model: AIModelInfo
    dataset_manifest_uri: str
    task: str = ""
    architecture: str = ""
    framework: str = "pytorch"
    seed: int = 0
    hyperparameters: JsonDict = Field(default_factory=dict)
    metrics_to_track: list[str] = Field(default_factory=list)
    output_dir: str = ""
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TrainingRunRecord(AutoTEMModel):
    """Auditable record for a model training run."""

    config: TrainingRunConfig
    status: TrainingStatus = "planned"
    metrics: list[Measurement] = Field(default_factory=list)
    artifacts: list[OutputArtifact] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def validate_training_manifest(manifest: TrainingDatasetManifest) -> list[str]:
    """Return warnings for data-split and labeling risks before training."""

    warnings = list(manifest.warnings)
    if not manifest.entries:
        warnings.append("Training manifest has no dataset entries.")
        return _dedupe(warnings)

    splits = {entry.split for entry in manifest.entries}
    if "train" not in splits:
        warnings.append("Training manifest is missing a train split.")
    if "validation" not in splits:
        warnings.append("Training manifest is missing a validation split.")
    if "test" not in splits and "holdout" not in splits:
        warnings.append("Training manifest should include a test or holdout split.")

    sources = {entry.source for entry in manifest.entries}
    if "synthetic" in sources and "experimental" not in sources and "mixed" not in sources:
        warnings.append(
            "Synthetic-only training data requires experimental validation before reporting."
        )
    if "experimental" in sources and "synthetic" not in sources and "mixed" not in sources:
        warnings.append(
            "Experimental-only training data may under-cover simulated defocus, dose, "
            "and thickness."
        )

    labeled_entries = [entry for entry in manifest.entries if entry.label_uri]
    if not labeled_entries:
        warnings.append("No label_uri values were provided; supervised training is not ready.")

    grouping = {(entry.sample_id, entry.instrument_id, entry.split) for entry in manifest.entries}
    samples = {(sample_id, instrument_id) for sample_id, instrument_id, _split in grouping}
    for sample_id, instrument_id in samples:
        sample_splits = {
            split
            for group_sample, group_instrument, split in grouping
            if group_sample == sample_id and group_instrument == instrument_id
        }
        if len(sample_splits) > 1 and sample_id:
            warnings.append(
                "Dataset split contains the same sample/instrument across multiple splits; "
                "check for leakage."
            )
            break

    return _dedupe(warnings)


def planned_training_run(
    *,
    run_id: str,
    model: AIModelInfo,
    dataset_manifest_uri: str,
    task: str | None = None,
    architecture: str = "",
    output_dir: str = "",
    hyperparameters: JsonDict | None = None,
    metrics_to_track: list[str] | None = None,
) -> TrainingRunRecord:
    """Build a planned training run record without executing a framework."""

    resolved_task = task or model.task
    config = TrainingRunConfig(
        run_id=run_id,
        model=model,
        dataset_manifest_uri=dataset_manifest_uri,
        task=resolved_task,
        architecture=architecture,
        output_dir=output_dir,
        hyperparameters=dict(hyperparameters or {}),
        metrics_to_track=list(metrics_to_track or _default_metrics_for_task(resolved_task)),
        warnings=[
            "Record training_data, license, validation split, and low-confidence review policy "
            "before using this model in reports."
        ],
    )
    return TrainingRunRecord(config=config, warnings=list(config.warnings))


def _default_metrics_for_task(task: str) -> list[str]:
    if "atom" in task:
        return ["atom_rmse_px", "precision", "recall", "f1", "ece"]
    if "defect" in task:
        return ["dice", "precision", "recall", "novelty_auc", "ece"]
    if "spectroscopy" in task or "eels" in task or "eds" in task:
        return ["reconstruction_error", "spectral_angle", "abundance_rmse", "ece"]
    if "z_contrast" in task:
        return ["relative_intensity_mae", "rank_correlation", "uncertainty_coverage"]
    return ["loss", "validation_metric", "ece"]


def _dedupe(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item not in deduped:
            deduped.append(item)
    return deduped
