"""Adapter-based calibration implementation removed from main branch."""

from __future__ import annotations

from typing import Any


SUPPORTED_DIFFRACTION_CALIBRATION_ENGINES: frozenset[str] = frozenset()
_REMOVED_MESSAGE = "Requested adapter calibration task content has been removed from main branch."


def resolve_diffraction_calibration_engine(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def scale_from_known_d_spacing_with_adapter(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)
