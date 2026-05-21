"""Pixel calibration implementation removed from main branch."""

from __future__ import annotations

from typing import Any


_REMOVED_MESSAGE = "Requested pixel-calibration task content has been removed from main branch."


def convert_length(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def coerce_length_to_nm(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def manual_pixel_size_from_reference(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def infer_pixel_size_from_scalebar(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def resolve_pixel_size_measurement(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def resolve_voltage_kv(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)
