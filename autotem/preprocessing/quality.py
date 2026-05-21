"""Quality metrics implementation removed from main branch."""

from __future__ import annotations

from typing import Any


_REMOVED_MESSAGE = "Requested quality-metrics task content has been removed from main branch."


def load_dataset_array(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def assess_image_quality(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)
