"""Model registry helpers for AI-assisted AutoTEM analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from autotem.core import AIModelInfo
from autotem.core.models import AutoTEMModel, JsonDict

ModelLifecycle = Literal["candidate", "prototype", "validated", "deprecated"]
ModelOwner = Literal["A", "B", "C", "shared", "unknown"]
RiskLevel = Literal["low", "medium", "high"]


class AIModelCard(AutoTEMModel):
    """Governance metadata for one registered AI model."""

    info: AIModelInfo
    owner: ModelOwner = "unknown"
    lifecycle: ModelLifecycle = "candidate"
    input_modalities: list[str] = Field(default_factory=list)
    output_contract: str = "AIInferenceResult"
    output_artifacts: list[str] = Field(default_factory=list)
    default_parameters: JsonDict = Field(default_factory=dict)
    training_manifest_uri: str | None = None
    model_card_uri: str | None = None
    risk_level: RiskLevel = "medium"
    quantitative_use_allowed: bool = False
    requires_human_review: bool = True
    report_warnings: list[str] = Field(default_factory=list)
    provenance: JsonDict = Field(default_factory=dict)

    @property
    def registry_key(self) -> str:
        """Return a stable key using model name plus optional version."""

        return model_registry_key(self.info.name, self.info.version)


def model_registry_key(name: str, version: str = "") -> str:
    """Build a deterministic registry key."""

    cleaned_name = name.strip()
    cleaned_version = version.strip()
    return f"{cleaned_name}@{cleaned_version}" if cleaned_version else cleaned_name


class ModelRegistry:
    """In-memory model registry with JSON round-trip helpers."""

    def __init__(self, cards: list[AIModelCard] | None = None) -> None:
        self._cards: dict[str, AIModelCard] = {}
        for card in cards or []:
            self.register(card)

    def register(self, card: AIModelCard, *, replace: bool = False) -> AIModelCard:
        """Register a model card and reject accidental duplicate keys."""

        key = card.registry_key
        if key in self._cards and not replace:
            msg = f"AI model '{key}' is already registered."
            raise ValueError(msg)
        self._cards[key] = card
        return card

    def register_info(
        self,
        info: AIModelInfo,
        *,
        owner: ModelOwner = "unknown",
        lifecycle: ModelLifecycle = "candidate",
        replace: bool = False,
        **card_kwargs: object,
    ) -> AIModelCard:
        """Create and register a model card from core AI model metadata."""

        card = AIModelCard(
            info=info,
            owner=owner,
            lifecycle=lifecycle,
            **card_kwargs,
        )
        return self.register(card, replace=replace)

    def get(self, name: str, version: str = "") -> AIModelCard:
        """Return an exact model card by name and optional version."""

        key = model_registry_key(name, version)
        try:
            return self._cards[key]
        except KeyError as exc:
            msg = f"AI model '{key}' is not registered."
            raise KeyError(msg) from exc

    def list_cards(
        self,
        *,
        task: str | None = None,
        owner: ModelOwner | None = None,
        lifecycle: ModelLifecycle | None = None,
    ) -> list[AIModelCard]:
        """List cards, optionally filtered by task, owner, or lifecycle."""

        cards = list(self._cards.values())
        if task is not None:
            cards = [card for card in cards if card.info.task == task]
        if owner is not None:
            cards = [card for card in cards if card.owner == owner]
        if lifecycle is not None:
            cards = [card for card in cards if card.lifecycle == lifecycle]
        return sorted(cards, key=lambda card: card.registry_key)

    def find_by_task(self, task: str) -> list[AIModelCard]:
        """Return all cards registered for a task."""

        return self.list_cards(task=task)

    def to_payload(self) -> dict[str, object]:
        """Serialize the registry to a plain JSON-compatible payload."""

        return {"models": [card.model_dump(mode="json") for card in self.list_cards()]}

    def write_json(self, path: str | Path) -> Path:
        """Write the registry to a UTF-8 JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(self.to_payload(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> ModelRegistry:
        """Build a registry from a JSON-compatible payload."""

        raw_cards = payload.get("models", [])
        if not isinstance(raw_cards, list):
            msg = "Model registry payload must contain a 'models' list."
            raise ValueError(msg)
        cards = [AIModelCard.model_validate(item) for item in raw_cards]
        return cls(cards)

    @classmethod
    def read_json(cls, path: str | Path) -> ModelRegistry:
        """Read a registry from a UTF-8 JSON file."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = "Model registry JSON must be an object."
            raise ValueError(msg)
        return cls.from_payload(payload)


def recommended_member_c_model_cards() -> list[AIModelCard]:
    """Return first-pass registry cards for Member C deep-learning expansion."""

    return [
        AIModelCard(
            info=AIModelInfo(
                name="atom-column-heatmap",
                version="0.1-planned",
                task="atom_column_detection",
                training_data="synthetic+expert-corrected atom-column heatmaps",
                license="TBD",
                warnings=["Coordinates must be refined by Gaussian or center-of-mass fitting."],
            ),
            owner="C",
            lifecycle="candidate",
            input_modalities=["HRTEM", "STEM", "HAADF-STEM"],
            output_artifacts=["atom_candidate_heatmap", "candidate_mask"],
            default_parameters={"threshold": 0.5, "min_distance_px": 3},
            risk_level="medium",
            report_warnings=[
                "DL heatmaps are candidate evidence only; final coordinates require fitting "
                "residuals."
            ],
            provenance={"source": "C7 initial integration plan"},
        ),
        AIModelCard(
            info=AIModelInfo(
                name="defect-novelty-detector",
                version="0.1-planned",
                task="defect_detection",
                training_data="simulated defects + reviewed experimental patches",
                license="TBD",
                warnings=["Unknown and needs-review classes must be preserved."],
            ),
            owner="C",
            lifecycle="candidate",
            input_modalities=["HRTEM", "STEM", "HAADF-STEM"],
            output_artifacts=["defect_candidates", "novelty_map"],
            default_parameters={"novelty_threshold": 0.7},
            risk_level="high",
            provenance={"source": "C7 initial integration plan"},
        ),
        AIModelCard(
            info=AIModelInfo(
                name="z-contrast-column-estimator",
                version="0.1-planned",
                task="z_contrast_estimation",
                training_data="multislice-calibrated atom column intensity tables",
                license="TBD",
                warnings=[
                    "Relative contrast only unless thickness, tilt, detector, and simulation "
                    "calibration are available."
                ],
            ),
            owner="C",
            lifecycle="candidate",
            input_modalities=["HAADF-STEM"],
            output_artifacts=["relative_z_table", "uncertainty_map"],
            default_parameters={"report_relative_only": True},
            risk_level="high",
            provenance={"source": "C7 initial integration plan"},
        ),
        AIModelCard(
            info=AIModelInfo(
                name="spectroscopy-autoencoder-unmixing",
                version="0.1-planned",
                task="spectroscopy_unmixing",
                training_data="EDS/EELS spectral-image patches with PCA/NMF baselines",
                license="TBD",
                warnings=["Autoencoder components must be compared with PCA/NMF baselines."],
            ),
            owner="C",
            lifecycle="candidate",
            input_modalities=["EDS", "EELS"],
            output_artifacts=["abundance_maps", "latent_spectra"],
            default_parameters={"latent_dim": 8},
            risk_level="medium",
            provenance={"source": "C7 initial integration plan"},
        ),
    ]


def recommended_member_c_registry() -> ModelRegistry:
    """Build the initial Member C registry seed."""

    return ModelRegistry(recommended_member_c_model_cards())
