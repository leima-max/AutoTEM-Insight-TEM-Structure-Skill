"""Preprocessing helpers after requested task cleanup."""

from __future__ import annotations

from typing import Any


_REMOVED_QUALITY = "Requested quality task content has been removed from main branch."
_REMOVED_PREPROCESS = "Requested preprocessing task content has been removed from main branch."


def assess_quality(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_QUALITY)


def preprocess_dataset(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_PREPROCESS)
