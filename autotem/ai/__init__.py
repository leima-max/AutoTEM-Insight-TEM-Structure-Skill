"""AI model registry, inference, training, and active-learning utilities."""

from autotem.ai.active_learning import ActiveLearningBatch, ActiveLearningQueue, ReviewItem
from autotem.ai.inference import (
    CallableAIAdapter,
    InferenceRequest,
    build_pending_inference_result,
    coerce_ai_inference_result,
    run_ai_inference,
)
from autotem.ai.model_registry import (
    AIModelCard,
    ModelRegistry,
    model_registry_key,
    recommended_member_c_model_cards,
    recommended_member_c_registry,
)
from autotem.ai.training import (
    TrainingDatasetEntry,
    TrainingDatasetManifest,
    TrainingRunConfig,
    TrainingRunRecord,
    planned_training_run,
    validate_training_manifest,
)

__all__ = [
    "AIModelCard",
    "ActiveLearningBatch",
    "ActiveLearningQueue",
    "CallableAIAdapter",
    "InferenceRequest",
    "ModelRegistry",
    "ReviewItem",
    "TrainingDatasetEntry",
    "TrainingDatasetManifest",
    "TrainingRunConfig",
    "TrainingRunRecord",
    "build_pending_inference_result",
    "coerce_ai_inference_result",
    "model_registry_key",
    "planned_training_run",
    "recommended_member_c_model_cards",
    "recommended_member_c_registry",
    "run_ai_inference",
    "validate_training_manifest",
]
