"""Active-learning review queue utilities for AI-assisted analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from autotem.core import AIInferenceResult, AIModelInfo, ArrayReference
from autotem.core.models import AutoTEMModel, JsonDict

ReviewPriority = Literal["low", "medium", "high"]
ReviewStatus = Literal["queued", "reviewed", "accepted", "rejected", "exported"]


class ReviewItem(AutoTEMModel):
    """One low-confidence region or candidate sent for expert review."""

    review_id: str
    task: str
    source_uri: str = ""
    model: AIModelInfo
    region: ArrayReference | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty_score: float | None = Field(default=None, ge=0.0)
    reason: str = ""
    suggested_labels: list[str] = Field(default_factory=list)
    status: ReviewStatus = "queued"
    priority: ReviewPriority = "medium"
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ActiveLearningBatch(AutoTEMModel):
    """A review batch that can be exported for annotation."""

    batch_id: str
    task: str
    items: list[ReviewItem] = Field(default_factory=list)
    provenance: JsonDict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ActiveLearningQueue:
    """Mutable review queue wrapper with JSONL helpers."""

    def __init__(self, batch: ActiveLearningBatch) -> None:
        self.batch = batch

    @classmethod
    def from_inference(
        cls,
        result: AIInferenceResult,
        *,
        batch_id: str,
        task: str | None = None,
        source_uri: str = "",
        confidence_threshold: float = 0.5,
        max_items: int | None = None,
    ) -> ActiveLearningQueue:
        """Create review items from low-confidence AI inference regions."""

        resolved_task = task or result.model.task or "ai_review"
        priority = _priority_from_confidence(result.confidence, confidence_threshold)
        regions = list(result.low_confidence_regions)
        if max_items is not None:
            regions = regions[:max_items]

        items: list[ReviewItem] = []
        for index, region in enumerate(regions):
            items.append(
                ReviewItem(
                    review_id=f"{batch_id}-{index + 1:04d}",
                    task=resolved_task,
                    source_uri=source_uri,
                    model=result.model,
                    region=region,
                    confidence=result.confidence,
                    reason="low_confidence_region",
                    suggested_labels=list(result.labels),
                    priority=priority,
                    provenance={
                        "model_name": result.model.name,
                        "model_version": result.model.version,
                    },
                )
            )

        warnings: list[str] = []
        if not items:
            warnings.append(
                "No low-confidence regions were present; queue contains whole-result review item."
            )
            items.append(
                ReviewItem(
                    review_id=f"{batch_id}-0001",
                    task=resolved_task,
                    source_uri=source_uri,
                    model=result.model,
                    confidence=result.confidence,
                    reason="whole_result_low_confidence"
                    if _is_low_confidence(result.confidence, confidence_threshold)
                    else "manual_review_requested",
                    suggested_labels=list(result.labels),
                    priority=priority,
                    provenance={
                        "model_name": result.model.name,
                        "model_version": result.model.version,
                    },
                    warnings=warnings,
                )
            )

        batch = ActiveLearningBatch(
            batch_id=batch_id,
            task=resolved_task,
            items=items,
            provenance={"model": result.model.model_dump(mode="json")},
            warnings=warnings,
        )
        return cls(batch)

    def add_item(self, item: ReviewItem) -> ReviewItem:
        """Append one review item."""

        self.batch.items.append(item)
        return item

    def write_jsonl(self, path: str | Path) -> Path:
        """Write queue items as UTF-8 JSONL for annotation tooling."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            json.dumps(item.model_dump(mode="json"), sort_keys=True)
            for item in self.batch.items
        ]
        output_path.write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def read_jsonl(
        cls,
        path: str | Path,
        *,
        batch_id: str,
        task: str,
    ) -> ActiveLearningQueue:
        """Read review queue items from UTF-8 JSONL."""

        items: list[ReviewItem] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                items.append(ReviewItem.model_validate_json(line))
        return cls(ActiveLearningBatch(batch_id=batch_id, task=task, items=items))


def _is_low_confidence(confidence: float | None, threshold: float) -> bool:
    return confidence is not None and confidence < threshold


def _priority_from_confidence(confidence: float | None, threshold: float) -> ReviewPriority:
    if confidence is None:
        return "medium"
    if confidence < threshold:
        return "high"
    return "medium"
