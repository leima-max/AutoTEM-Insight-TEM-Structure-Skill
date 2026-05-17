from __future__ import annotations

import numpy as np

import autotem.calibration.pixel_calibration as pixel_calibration
from autotem.calibration.pixel_calibration import infer_pixel_size_from_scalebar


def _synthetic_scalebar_image(
    *,
    h: int = 220,
    w: int = 320,
    bar_x0: int = 40,
    bar_x1: int = 140,
    bar_y0: int = 194,
    bar_y1: int = 197,
) -> np.ndarray:
    img = np.full((h, w), 0.2, dtype=np.float32)
    img[bar_y0:bar_y1, bar_x0:bar_x1] = 0.95
    return img


def test_infer_pixel_size_from_scalebar_metadata_text() -> None:
    image = _synthetic_scalebar_image()
    metadata = {
        "ImageDescription": "TEM image, Scale bar: 50 nm",
    }
    pixel_nm, source, warnings = infer_pixel_size_from_scalebar(
        image,
        metadata=metadata,
        file_hint="sample.tiff",
    )
    assert pixel_nm is not None
    # bar length is ~100 px; expected pixel size around 0.5 nm/px
    assert abs(pixel_nm - 0.5) < 0.08
    assert "scalebar" in source.lower()
    assert warnings == []


def test_infer_pixel_size_from_scalebar_without_text_returns_none() -> None:
    image = _synthetic_scalebar_image()
    pixel_nm, source, warnings = infer_pixel_size_from_scalebar(
        image,
        metadata={"ImageDescription": "no length info"},
        file_hint="sample.tiff",
    )
    assert pixel_nm is None
    assert source in {"scalebar.text.unavailable", "scalebar.unavailable"}
    assert warnings


def test_infer_pixel_size_from_outlined_bottom_edge_bar() -> None:
    rng = np.random.default_rng(123)
    h, w = 260, 360
    image = (0.45 + 0.06 * rng.normal(size=(h, w))).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)

    # Mimic an anti-aliased/outlined scale bar touching lower border.
    x0, x1 = 8, 128
    y = h - 2
    image[y, x0:x1] = 0.95
    image[y - 1, x0:x1] = 0.08
    image[y, x0 + 40 : x0 + 43] = 0.45  # tiny gap to test relaxed closing

    pixel_nm, source, warnings = infer_pixel_size_from_scalebar(
        image,
        metadata={"ImageDescription": "Scale bar 5 nm"},
        file_hint="foo_5nm.tiff",
    )
    assert pixel_nm is not None
    assert abs(pixel_nm - (5.0 / (x1 - x0))) < 0.08
    assert "scalebar.bar" in source
    assert warnings == []


def test_infer_pixel_size_from_scalebar_ocr_fallback(monkeypatch) -> None:
    image = _synthetic_scalebar_image(bar_x0=20, bar_x1=120)

    def _fake_ocr_rows(_image):
        # text, confidence, x_center, y_center
        return [("5 nm", 0.95, 24.0, float(_image.shape[0] - 8))]

    monkeypatch.setattr(pixel_calibration, "_ocr_read_text_rows", _fake_ocr_rows)

    pixel_nm, source, warnings = infer_pixel_size_from_scalebar(
        image,
        metadata={"ImageDescription": "no metadata scale text"},
        file_hint="sample_without_hint.tiff",
    )
    assert pixel_nm is not None
    assert abs(pixel_nm - (5.0 / 100.0)) < 0.02
    assert "scalebar.text (ocr:" in source
    assert warnings == []
