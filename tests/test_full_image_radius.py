"""Full-resolution radius masks keep their exact dense-grid boundary and axis conventions."""

import numpy as np
import pytest

from autotem.diffraction.fft import locate_direct_beam
from autotem.diffraction.peak_detection import detect_bragg_peaks
from autotem.preprocessing.quality import _estimate_high_freq_ratio

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("shape", [(17, 31), (32, 19), (16, 24)])
def test_radius_results_match_dense_grid_on_rectangles(monkeypatch, shape):
    h, w = shape
    image = 0.01 * np.arange(h * w).reshape(shape)
    for row, col, intensity in ((5, 8, 16), (9, 7, 24), (6, 11, 30), (11, 12, 18)):
        image[row, col] = intensity
    image = image[::-1]
    center = (7.0, float(h - 6))

    def results():
        return (
            _estimate_high_freq_ratio(image),
            locate_direct_beam(image, center_mode="auto", search_radius_px=6),
            detect_bragg_peaks(
                image,
                center=center,
                top_k=6,
                min_distance=1,
                center_exclusion_radius=1.0,
                max_peak_radius_px=5.0,
                radius_outlier_filter=False,
                threshold_strategy="relative_max",
                threshold_value=0.2,
            ),
        )

    actual = results()
    assert actual[2][0]
    indices = np.indices
    monkeypatch.setattr(
        np, "indices", lambda dimensions, dtype=int, sparse=False: indices(dimensions, dtype=dtype)
    )
    assert results() == actual
