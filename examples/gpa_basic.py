"""Minimal C2 GPA example on a synthetic HRTEM-like lattice image."""

from __future__ import annotations

import numpy as np

from autotem.strain import run_gpa


def synthetic_lattice(
    shape: tuple[int, int] = (160, 160),
    *,
    exx: float = 0.012,
    eyy: float = -0.006,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    cx = shape[1] / 2.0
    cy = shape[0] / 2.0
    ux = exx * (xx - cx)
    uy = eyy * (yy - cy)
    g1 = (0.085, 0.0)
    g2 = (0.0, 0.082)
    phase1 = 2.0 * np.pi * (g1[0] * (xx - ux) + g1[1] * (yy - uy))
    phase2 = 2.0 * np.pi * (g2[0] * (xx - ux) + g2[1] * (yy - uy))
    return (np.cos(phase1) + np.cos(phase2)).astype(np.float32)


if __name__ == "__main__":
    image = synthetic_lattice()
    result = run_gpa(
        image,
        g_vectors=[(0.085, 0.0), (0.0, 0.082)],
        g_vector_unit="1/pixel",
        reference_region=None,
    )
    print(result.provenance["map_stats"])
