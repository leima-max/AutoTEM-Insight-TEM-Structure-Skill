"""Diffraction calibration implementation removed from main branch."""

from __future__ import annotations

from typing import Any


_REMOVED_MESSAGE = "Requested diffraction-calibration task content has been removed from main branch."


def electron_wavelength_nm(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def reciprocal_scale_from_pixel_size(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def diffraction_scale_from_known_d_spacing(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def estimate_camera_length_mm(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)


def resolve_diffraction_calibration(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)
