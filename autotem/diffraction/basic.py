"""Diffraction implementation removed from main branch."""

from __future__ import annotations

from typing import Any


_REMOVED_MESSAGE = "Requested diffraction task content has been removed from main branch."


def _load_preprocessed_array(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def run_diffraction_analysis(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)
