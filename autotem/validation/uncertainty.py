"""Uncertainty and error helpers for A4 validation and benchmark metrics."""

from __future__ import annotations

import math


def mean_relative_error(
    observed: list[float],
    reference: list[float],
    *,
    eps: float = 1e-12,
) -> float:
    """Return mean relative error for paired values using |obs-ref|/max(|ref|, eps)."""
    if not observed or not reference:
        raise ValueError("observed and reference must both be non-empty.")
    if len(observed) != len(reference):
        raise ValueError("observed and reference must have equal length.")

    errors: list[float] = []
    for obs, ref in zip(observed, reference, strict=True):
        denom = max(abs(ref), eps)
        errors.append(abs(obs - ref) / denom)
    return float(sum(errors) / len(errors))


def coordinate_rmse(
    observed: list[tuple[float, float]],
    reference: list[tuple[float, float]],
) -> float:
    """Return RMSE for paired 2D coordinates after stable lexicographic sorting."""
    if not observed or not reference:
        raise ValueError("observed and reference coordinates must both be non-empty.")

    observed_sorted = sorted((float(y), float(x)) for x, y in observed)
    reference_sorted = sorted((float(y), float(x)) for x, y in reference)
    pair_count = min(len(observed_sorted), len(reference_sorted))
    if pair_count == 0:
        raise ValueError("No coordinate pairs available for RMSE.")

    squared_error_sum = 0.0
    for (obs_y, obs_x), (ref_y, ref_x) in zip(
        observed_sorted[:pair_count],
        reference_sorted[:pair_count],
        strict=True,
    ):
        dy = obs_y - ref_y
        dx = obs_x - ref_x
        squared_error_sum += dx * dx + dy * dy

    return float(math.sqrt(squared_error_sum / pair_count))
