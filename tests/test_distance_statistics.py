import numpy as np
import pytest

from autotem.atomic.distance_statistics import (
    compute_distance_statistics,
    compute_distance_statistics_from_lattice,
)
from autotem.core.models import AtomLattice, AtomSite


def test_distance_statistics_rectangular_lattice() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 3.0],
            [4.0, 0.0],
            [4.0, 3.0],
        ],
        dtype=np.float64,
    )
    out = compute_distance_statistics(coords)

    assert np.allclose(out.nearest_distances, 3.0)
    assert np.allclose(out.next_nearest_distances, 4.0)
    assert out.unit == "pixel"
    assert out.nearest_summary["count"] == 4
    assert out.nearest_summary["mean"] == pytest.approx(3.0)
    assert out.next_nearest_summary["mean"] == pytest.approx(4.0)
    assert 0.0 <= out.confidence <= 1.0


def test_distance_statistics_scaled_unit() -> None:
    coords = np.array(
        [
            [0.0, 0.0],
            [0.0, 3.0],
            [4.0, 0.0],
            [4.0, 3.0],
        ],
        dtype=np.float64,
    )
    out = compute_distance_statistics(coords, pixel_size=0.2, output_unit="nm")

    assert out.unit == "nm"
    assert out.nearest_summary["mean"] == pytest.approx(0.6)
    assert out.next_nearest_summary["mean"] == pytest.approx(0.8)


def test_distance_statistics_small_n_warns() -> None:
    coords = np.array([[1.5, 2.5]], dtype=np.float64)
    out = compute_distance_statistics(coords)

    assert np.isnan(out.nearest_distances[0])
    assert np.isnan(out.next_nearest_distances[0])
    assert any("At least 2 atoms are required" in w for w in out.warnings)
    assert out.confidence == pytest.approx(0.0)


def test_distance_statistics_from_atom_lattice() -> None:
    lattice = AtomLattice(
        atoms=[
            AtomSite(x=0.0, y=0.0),
            AtomSite(x=3.0, y=0.0),
            AtomSite(x=0.0, y=4.0),
            AtomSite(x=3.0, y=4.0),
        ],
        coordinate_unit="pixel",
    )
    out = compute_distance_statistics_from_lattice(lattice)

    assert out.nearest_summary["mean"] == pytest.approx(3.0)
    assert out.next_nearest_summary["mean"] == pytest.approx(4.0)


def test_distance_statistics_requires_unit_when_scaled() -> None:
    coords = np.array([[0.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    with pytest.raises(ValueError):
        compute_distance_statistics(coords, pixel_size=0.1)
