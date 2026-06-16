import numpy as np

from autotem.atomic.sublattice import (
    apply_manual_sublattice_labels,
    classify_sublattices_auto,
    classify_sublattices_from_model,
    parse_projected_model_csv,
)


def _two_site_cell(nx: int = 4, ny: int = 3):
    coords: list[list[float]] = []
    intensities: list[float] = []
    for j in range(ny):
        for i in range(nx):
            origin = np.array([20.0 + j * 12.0, 30.0 + i * 10.0], dtype=np.float64)
            coords.append(origin.tolist())
            intensities.append(100.0)
            coords.append((origin + np.array([6.0, 5.0])).tolist())
            intensities.append(45.0)
    return np.asarray(coords, dtype=np.float64), np.asarray(intensities, dtype=np.float64)


def test_auto_sublattice_classification_finds_two_site_structure() -> None:
    coords, intensities = _two_site_cell()

    out = classify_sublattices_auto(
        coords,
        base_index=0,
        a_index=2,
        b_index=8,
        intensities=intensities,
        n_sublattices=2,
        use_intensity=True,
    )

    assert len(set(out.labels.tolist())) == 2
    assert len(out.label_summaries) == 2
    assert out.projected_structure["basis_sites"]
    means = sorted(row["mean_intensity"] for row in out.label_summaries)
    assert means[0] < means[1]


def test_model_sublattice_classification_assigns_named_labels() -> None:
    coords, intensities = _two_site_cell()

    out = classify_sublattices_from_model(
        coords,
        base_index=0,
        a_index=2,
        b_index=8,
        intensities=intensities,
        basis_sites=[
            {"label": "A", "u": 0.0, "v": 0.0},
            {"label": "B", "u": 0.5, "v": 0.5},
        ],
        match_tolerance=0.10,
    )

    assert set(out.labels.tolist()) == {"A", "B"}
    assert np.sum(out.labels == "A") == np.sum(out.labels == "B")
    assert out.provenance["method"] == "user_projected_model"


def test_manual_sublattice_labels_override_existing_result() -> None:
    coords, intensities = _two_site_cell()
    out = classify_sublattices_auto(
        coords,
        base_index=0,
        a_index=2,
        b_index=8,
        intensities=intensities,
        n_sublattices=2,
    )

    corrected = apply_manual_sublattice_labels(out, {1: "A_corrected"})

    assert corrected.labels[1] == "A_corrected"
    assert corrected.manual_override_mask[1]
    assert corrected.provenance["manual_label_count"] == 1


def test_parse_projected_model_csv() -> None:
    sites = parse_projected_model_csv(
        """
        label,u,v
        A,0,0
        B,0.5,0.5
        """
    )

    assert sites == [{"label": "A", "u": 0.0, "v": 0.0}, {"label": "B", "u": 0.5, "v": 0.5}]


def test_projected_model_allows_multiple_sites_with_same_label() -> None:
    sites = parse_projected_model_csv(
        """
        A,0,0
        B,0.5,0.5
        C,0,0.5
        C,0.5,0
        """
    )

    assert [site["label"] for site in sites] == ["A", "B", "C", "C"]
