import numpy as np
import pytest

from autotem.atomic.intensity_quantification import (
    cluster_atomic_column_intensities,
    group_atomic_column_intensities_by_lattice,
    quantify_atomic_column_intensity,
)


def _disk_image() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((64, 64), 10.0, dtype=np.float64)
    coords = np.asarray([[20.0, 20.0], [20.0, 42.0], [42.0, 20.0]], dtype=np.float64)
    amplitudes = [50.0, 100.0, 150.0]
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    for (y, x), amp in zip(coords, amplitudes, strict=False):
        image[((yy - y) ** 2 + (xx - x) ** 2) <= 2.0**2] += amp
    return image, coords


def test_disk_intensity_recovers_background_subtracted_ratios() -> None:
    image, coords = _disk_image()

    out = quantify_atomic_column_intensity(
        image,
        coords,
        integration_radius=2.0,
        background_inner_radius=4.0,
        background_outer_radius=7.0,
        background_statistic="median",
        normalization="median",
    )

    assert out.confidence > 0.9
    assert np.all(out.valid_atom_mask)
    assert np.allclose(out.background_level, 10.0)
    assert out.normalized_intensity[0] == pytest.approx(0.5, abs=0.05)
    assert out.normalized_intensity[1] == pytest.approx(1.0, abs=0.05)
    assert out.normalized_intensity[2] == pytest.approx(1.5, abs=0.05)
    assert out.provenance["method"] == "disk_integration_with_local_annulus_background"


def test_intensity_quantification_flags_edges_and_overlaps() -> None:
    image = np.full((32, 32), 5.0, dtype=np.float64)
    coords = np.asarray([[2.0, 2.0], [16.0, 16.0], [17.0, 17.0]], dtype=np.float64)

    out = quantify_atomic_column_intensity(
        image,
        coords,
        integration_radius=2.0,
        background_inner_radius=3.0,
        background_outer_radius=6.0,
    )

    assert out.edge_flag[0]
    assert out.overlap_flag[1]
    assert out.overlap_flag[2]
    assert any("image edges" in warning for warning in out.warnings)
    assert any("overlap" in warning for warning in out.warnings)


def test_intensity_quantification_validates_radii() -> None:
    image, coords = _disk_image()

    with pytest.raises(ValueError, match="background_inner_radius"):
        quantify_atomic_column_intensity(
            image,
            coords,
            integration_radius=3.0,
            background_inner_radius=2.0,
            background_outer_radius=5.0,
        )


def test_saturation_status_uses_measured_disk_and_inclusive_threshold() -> None:
    image, coords = _disk_image()
    # Disk peaks are 60, 110 and 160. The last disk has no finite samples.
    image[39:46, 17:24] = np.nan
    unassessed = quantify_atomic_column_intensity(image, coords)
    assert unassessed.saturation_status == ["not_evaluated"] * 3
    at_limit = quantify_atomic_column_intensity(image, coords, saturation_threshold=110)
    assert at_limit.saturation_status == [
        "below_threshold", "at_or_above_threshold", "not_evaluated"
    ]
    assert at_limit.saturation_flag.tolist() == [False, True, False]
    exceeded = quantify_atomic_column_intensity(image, coords, saturation_threshold=109)
    assert exceeded.saturation_status[1] == "at_or_above_threshold"
    for value in (np.inf, -np.inf, np.nan):
        with pytest.raises(ValueError, match="saturation_threshold must be finite"):
            quantify_atomic_column_intensity(image, coords, saturation_threshold=value)


def test_intensity_clustering_summarizes_groups_by_brightness() -> None:
    image, coords = _disk_image()
    out = quantify_atomic_column_intensity(
        image,
        coords,
        integration_radius=2.0,
        background_inner_radius=4.0,
        background_outer_radius=7.0,
        normalization="median",
    )

    grouped = cluster_atomic_column_intensities(
        out,
        metric="normalized_intensity",
        n_groups=2,
    )

    assert len(grouped.summaries) == 2
    assert grouped.summaries[0]["mean"] < grouped.summaries[1]["mean"]
    assert grouped.provenance["method"] == "intensity_1d_kmeans"


def test_lattice_grouping_uses_user_defined_parity_indices() -> None:
    image, coords = _disk_image()
    out = quantify_atomic_column_intensity(
        image,
        coords,
        integration_radius=2.0,
        background_inner_radius=4.0,
        background_outer_radius=7.0,
    )
    lattice_indices = np.asarray([[0, 0], [0, 1], [1, 0]], dtype=np.int32)

    grouped = group_atomic_column_intensities_by_lattice(
        out,
        lattice_indices,
        metric="normalized_intensity",
        mode="ij_parity",
    )

    assert [row["group_id"] for row in grouped.summaries] == [0, 1, 2]
    assert grouped.summaries[0]["group_name"] == "i even / j even"
    assert grouped.provenance["method"] == "lattice_ij_parity"
