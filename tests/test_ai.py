from pathlib import Path

from autotem.ai import (
    ActiveLearningQueue,
    CallableAIAdapter,
    ModelRegistry,
    TrainingDatasetEntry,
    TrainingDatasetManifest,
    planned_training_run,
    recommended_member_c_registry,
    run_ai_inference,
    validate_training_manifest,
)
from autotem.core import AIModelInfo, ArrayReference, TEMDataset


def test_member_c_registry_seed_round_trips(tmp_path: Path) -> None:
    registry = recommended_member_c_registry()

    c_cards = registry.list_cards(owner="C")
    atom_cards = registry.find_by_task("atom_column_detection")

    assert len(c_cards) == 4
    assert atom_cards[0].info.name == "atom-column-heatmap"
    assert atom_cards[0].quantitative_use_allowed is False
    assert atom_cards[0].requires_human_review is True

    registry_path = tmp_path / "registry.json"
    registry.write_json(registry_path)
    restored = ModelRegistry.read_json(registry_path)

    assert restored.get("atom-column-heatmap", "0.1-planned").owner == "C"
    assert restored.get("z-contrast-column-estimator", "0.1-planned").risk_level == "high"


def test_callable_adapter_normalizes_mapping_payload() -> None:
    model = AIModelInfo(
        name="atom-column-heatmap",
        version="0.1-test",
        task="atom_column_detection",
    )

    def fake_predict(request):
        assert request.primary_array is not None
        assert request.primary_array.shape == (8, 8)
        return {
            "masks": [{"shape": (8, 8), "dtype": "float32", "uri": "ai/atom_heatmap.npy"}],
            "labels": ["atom_column"],
            "low_confidence_regions": [
                {"shape": (2, 2), "dtype": "uint8", "uri": "ai/review_region.npy"}
            ],
            "confidence": 0.64,
            "warnings": ["prototype model"],
        }

    dataset = TEMDataset(data=ArrayReference(shape=(8, 8), uri="raw.npy"))
    adapter = CallableAIAdapter(model, fake_predict)

    result = run_ai_inference(
        adapter,
        dataset=dataset,
        parameters={"threshold": 0.42},
    )

    assert result.model.name == "atom-column-heatmap"
    assert result.masks[0].uri == "ai/atom_heatmap.npy"
    assert result.labels == ["atom_column"]
    assert result.low_confidence_regions[0].shape == (2, 2)
    assert result.provenance["parameters"] == {"threshold": 0.42}
    assert result.provenance["primary_input"]["shape"] == [8, 8]


def test_training_manifest_validation_and_planned_record() -> None:
    manifest = TrainingDatasetManifest(
        dataset_id="atom-heatmap-v0",
        task="atom_column_detection",
        entries=[
            TrainingDatasetEntry(
                uri="patches/train.zarr",
                label_uri="labels/train.zarr",
                split="train",
                source="synthetic",
                modality="HAADF-STEM",
                sample_id="sim-001",
            ),
            TrainingDatasetEntry(
                uri="patches/validation.zarr",
                label_uri="labels/validation.zarr",
                split="validation",
                source="experimental",
                modality="HAADF-STEM",
                sample_id="exp-001",
            ),
            TrainingDatasetEntry(
                uri="patches/test.zarr",
                label_uri="labels/test.zarr",
                split="test",
                source="experimental",
                modality="HAADF-STEM",
                sample_id="exp-002",
            ),
        ],
        label_schema={"mask": "atom_column_probability"},
    )

    warnings = validate_training_manifest(manifest)
    assert warnings == []

    record = planned_training_run(
        run_id="c7-atom-heatmap-001",
        model=AIModelInfo(name="atom-column-heatmap", task="atom_column_detection"),
        dataset_manifest_uri="manifests/atom-heatmap-v0.json",
        architecture="small_unet_heatmap",
    )

    assert record.status == "planned"
    assert "atom_rmse_px" in record.config.metrics_to_track
    assert record.config.framework == "pytorch"


def test_training_manifest_reports_split_and_label_risks() -> None:
    manifest = TrainingDatasetManifest(
        dataset_id="bad-manifest",
        task="defect_detection",
        entries=[
            TrainingDatasetEntry(
                uri="patches/train.zarr",
                split="train",
                source="synthetic",
                sample_id="sim-001",
            )
        ],
    )

    warnings = validate_training_manifest(manifest)

    assert "Training manifest is missing a validation split." in warnings
    assert "Training manifest should include a test or holdout split." in warnings
    assert "No label_uri values were provided; supervised training is not ready." in warnings


def test_active_learning_queue_from_low_confidence_regions(tmp_path: Path) -> None:
    model = AIModelInfo(
        name="defect-novelty-detector",
        version="0.1-test",
        task="defect_detection",
    )
    result = run_ai_inference(
        CallableAIAdapter(
            model,
            lambda _request: {
                "labels": ["unknown", "needs_review"],
                "low_confidence_regions": [
                    {"shape": (4, 4), "dtype": "uint8", "uri": "regions/r0.npy"},
                    {"shape": (4, 4), "dtype": "uint8", "uri": "regions/r1.npy"},
                ],
                "confidence": 0.31,
            },
        ),
        dataset=TEMDataset(data=ArrayReference(shape=(16, 16), uri="raw.npy")),
    )

    queue = ActiveLearningQueue.from_inference(
        result,
        batch_id="review-c5-001",
        source_uri="raw.npy",
        confidence_threshold=0.5,
        max_items=1,
    )

    assert len(queue.batch.items) == 1
    assert queue.batch.items[0].priority == "high"
    assert queue.batch.items[0].reason == "low_confidence_region"
    assert queue.batch.items[0].suggested_labels == ["unknown", "needs_review"]

    queue_path = tmp_path / "review.jsonl"
    queue.write_jsonl(queue_path)
    restored = ActiveLearningQueue.read_jsonl(
        queue_path,
        batch_id="review-c5-001",
        task="defect_detection",
    )

    assert restored.batch.items[0].region is not None
    assert restored.batch.items[0].region.uri == "regions/r0.npy"
