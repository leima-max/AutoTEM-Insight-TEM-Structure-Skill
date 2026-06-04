"""Unified inference adapter contracts for AI-assisted modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from pydantic import Field

from autotem.core import (
    AIInferenceResult,
    AIModelInfo,
    ArrayReference,
    OutputArtifact,
    PreprocessingResult,
    TEMDataset,
)
from autotem.core.models import AutoTEMModel, JsonDict


class InferenceRequest(AutoTEMModel):
    """Input envelope passed to model adapters."""

    dataset: TEMDataset | None = None
    preprocessing: PreprocessingResult | None = None
    parameters: JsonDict = Field(default_factory=dict)
    provenance: JsonDict = Field(default_factory=dict)

    @property
    def primary_array(self) -> ArrayReference | None:
        """Return the preferred image reference for inference."""

        if self.preprocessing is not None and self.preprocessing.image is not None:
            return self.preprocessing.image
        if self.dataset is not None:
            return self.dataset.data
        return None


class AIModelAdapter(Protocol):
    """Protocol implemented by concrete inference adapters."""

    @property
    def model_info(self) -> AIModelInfo:
        """Return model metadata used for provenance."""

    def predict(self, request: InferenceRequest) -> AIInferenceResult | Mapping[str, Any]:
        """Run inference and return a structured result or mapping payload."""


class CallableAIAdapter:
    """Small adapter for wrapping local callables during prototyping."""

    def __init__(
        self,
        model_info: AIModelInfo,
        predict_fn: Callable[[InferenceRequest], AIInferenceResult | Mapping[str, Any]],
    ) -> None:
        self._model_info = model_info
        self._predict_fn = predict_fn

    @property
    def model_info(self) -> AIModelInfo:
        """Return model metadata."""

        return self._model_info

    def predict(self, request: InferenceRequest) -> AIInferenceResult | Mapping[str, Any]:
        """Delegate prediction to the wrapped callable."""

        return self._predict_fn(request)


def build_pending_inference_result(
    model_info: AIModelInfo,
    *,
    reason: str = "Model adapter is not available in this environment.",
    parameters: Mapping[str, Any] | None = None,
) -> AIInferenceResult:
    """Create an auditable placeholder result for a planned or unavailable model."""

    return AIInferenceResult(
        model=model_info,
        confidence=0.0,
        provenance={
            "status": "pending",
            "parameters": dict(parameters or {}),
        },
        warnings=[reason],
    )


def run_ai_inference(
    adapter: AIModelAdapter,
    *,
    dataset: TEMDataset | None = None,
    preprocessing: PreprocessingResult | None = None,
    parameters: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> AIInferenceResult:
    """Run an adapter and normalize its output into `AIInferenceResult`."""

    request = InferenceRequest(
        dataset=dataset,
        preprocessing=preprocessing,
        parameters=dict(parameters or {}),
        provenance=dict(provenance or {}),
    )
    payload = adapter.predict(request)
    result = coerce_ai_inference_result(
        payload,
        model_info=adapter.model_info,
        request=request,
    )
    return result


def coerce_ai_inference_result(
    payload: AIInferenceResult | Mapping[str, Any],
    *,
    model_info: AIModelInfo,
    request: InferenceRequest | None = None,
) -> AIInferenceResult:
    """Coerce adapter payloads into the core AI inference contract."""

    if isinstance(payload, AIInferenceResult):
        result = payload
    else:
        result = AIInferenceResult(
            model=model_info,
            predictions=[
                _coerce_output_artifact(item) for item in payload.get("predictions", [])
            ],
            masks=[_coerce_array_reference(item) for item in payload.get("masks", [])],
            labels=list(payload.get("labels", [])),
            uncertainty=_coerce_optional_array_reference(payload.get("uncertainty")),
            low_confidence_regions=[
                _coerce_array_reference(item)
                for item in payload.get("low_confidence_regions", [])
            ],
            confidence=payload.get("confidence"),
            provenance=dict(payload.get("provenance", {})),
            warnings=list(payload.get("warnings", [])),
        )

    provenance = dict(result.provenance)
    provenance.setdefault("model_name", result.model.name)
    provenance.setdefault("model_version", result.model.version)
    provenance.setdefault("task", result.model.task)
    if request is not None:
        provenance.setdefault("parameters", dict(request.parameters))
        primary = request.primary_array
        if primary is not None:
            provenance.setdefault(
                "primary_input",
                {
                    "shape": list(primary.shape),
                    "dtype": primary.dtype,
                    "uri": primary.uri,
                },
            )
    return result.model_copy(update={"provenance": provenance})


def _coerce_array_reference(value: ArrayReference | Mapping[str, Any]) -> ArrayReference:
    if isinstance(value, ArrayReference):
        return value
    return ArrayReference.model_validate(dict(value))


def _coerce_optional_array_reference(value: object) -> ArrayReference | None:
    if value is None:
        return None
    if isinstance(value, ArrayReference):
        return value
    if isinstance(value, Mapping):
        return ArrayReference.model_validate(dict(value))
    msg = "Optional array reference payload must be a mapping or ArrayReference."
    raise TypeError(msg)


def _coerce_output_artifact(
    value: OutputArtifact | Mapping[str, Any],
) -> OutputArtifact:
    if isinstance(value, OutputArtifact):
        return value
    return OutputArtifact.model_validate(dict(value))
