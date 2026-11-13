import html
import json
from pathlib import Path

import numpy as np
import pytest

from autotem.acceptance.cases import write_fixture
from autotem.core import AnalysisReport, DiffractionPeak, DiffractionResult, GVector, Measurement
from autotem.reporting import export_html_report, export_json_bundle
from autotem.strain import run_gpa
from autotem.strain.gpa import (
    InsufficientGPAReference,
    _diffraction_g_vector_unit_for_gpa,
    _displacement_from_phases,
    _strain_from_displacement,
    compute_gpa_fields,
)
from autotem.strain.gpa_alignment import (
    GPAAlignmentConfig,
    compare_gpa_configs,
)
from autotem.strain.gpa_export import (
    SUPPORTED_GPA_EXPORT_COLORMAPS,
    export_gpa_maps,
    render_gpa_rgb,
)

pytestmark = pytest.mark.smoke


def _array(ref):
    assert ref is not None
    assert ref.inline is not None
    return np.asarray(ref.inline, dtype=np.float64)


def _synthetic_lattice(
    shape: tuple[int, int] = (160, 160),
    *,
    g1: tuple[float, float] = (0.085, 0.0),
    g2: tuple[float, float] = (0.0, 0.082),
    exx: float = 0.0,
    eyy: float = 0.0,
    rotation: float = 0.0,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    cx = shape[1] / 2.0
    cy = shape[0] / 2.0
    x0 = xx - cx
    y0 = yy - cy
    ux = exx * x0 + rotation * y0
    uy = -rotation * x0 + eyy * y0
    phase1 = 2.0 * np.pi * (g1[0] * (xx - ux) + g1[1] * (yy - uy))
    phase2 = 2.0 * np.pi * (g2[0] * (xx - ux) + g2[1] * (yy - uy))
    return (np.cos(phase1) + np.cos(phase2)).astype(np.float32)


def _piecewise_exx_lattice(
    shape: tuple[int, int] = (192, 192),
    *,
    exx_right: float = 0.018,
) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    mid = shape[1] // 2
    ux = np.zeros(shape, dtype=np.float64)
    ux[:, mid:] = exx_right * (xx[:, mid:] - mid)
    g1 = (0.085, 0.0)
    g2 = (0.0, 0.082)
    phase1 = 2.0 * np.pi * (g1[0] * (xx - ux) + g1[1] * yy)
    phase2 = 2.0 * np.pi * (g2[0] * (xx - ux) + g2[1] * yy)
    return (np.cos(phase1) + np.cos(phase2)).astype(np.float32)


def test_run_gpa_recovers_uniform_strain_with_manual_vectors() -> None:
    image = _synthetic_lattice(exx=0.012, eyy=-0.007)

    out = run_gpa(
        image,
        g_vectors=[(0.085, 0.0), (0.0, 0.082)],
        g_vector_unit="1/pixel",
        reference_region=None,
        map_inline_mode="full",
    )

    stats = out.provenance["map_stats"]
    assert stats["mean_exx"] == pytest.approx(0.012, abs=0.006)
    assert stats["mean_eyy"] == pytest.approx(-0.007, abs=0.006)
    assert abs(stats["mean_exy"]) < 0.006
    assert out.provenance["phase_maps"]["g1"]["phase_wrapped"]["shape"] == [160, 160]


def test_run_gpa_rejects_unsupported_backend() -> None:
    image = _synthetic_lattice(exx=0.012, eyy=-0.007)

    with pytest.raises(ValueError, match=r"\{'gpa', 'gpa_b3'\}"):
        run_gpa(
            image,
            method="unsupported_backend",
            g_vectors=[(0.085, 0.0), (0.0, 0.082)],
            g_vector_unit="1/pixel",
            reference_region=None,
            map_inline_mode="full",
        )


def test_gpa_alignment_diagnostic_detects_parameter_drift() -> None:
    image = _synthetic_lattice(exx=0.01, eyy=-0.004, rotation=0.001)
    config = GPAAlignmentConfig(
        g_vectors=((0.085, 0.0), (0.0, 0.082)),
        g_vector_unit="1/pixel",
        mask_sigma_px=5.0,
        hann_window=False,
        apply_quality_mask=False,
    )

    same = compare_gpa_configs(image, config, config, backend="gpa", sample_step=2)

    assert same["diagnosis"]["configs_aligned"] is True
    assert same["diagnosis"]["max_strain_mae_percent"] == pytest.approx(0.0)

    shifted = GPAAlignmentConfig(
        g_vectors=((0.085 + 0.004, 0.0), (0.0, 0.082 - 0.003)),
        g_vector_unit="1/pixel",
        mask_sigma_px=5.0,
        hann_window=False,
        apply_quality_mask=False,
    )
    parameter = compare_gpa_configs(image, config, shifted, backend="gpa", sample_step=2)

    assert parameter["diagnosis"]["configs_aligned"] is False
    assert parameter["diagnosis"]["max_strain_mae_percent"] > 0.1


def test_gpa_alignment_requires_joint_samples_and_usable_reference(tmp_path):
    path, _ = write_fixture("J03", tmp_path)
    image = np.load(path, allow_pickle=False)
    settings = dict(g_vectors=((0.125, 0), (0, 0.125)), hann_window=True, apply_quality_mask=True)
    good = GPAAlignmentConfig(**settings, reference_region=(16, 48, 16, 48))
    bad = GPAAlignmentConfig(**settings, reference_region=(0, 4, 0, 4))
    for first, second, group in (
        (bad, good, "reference_config"), (good, bad, "candidate_config")
    ):
        with pytest.raises(InsufficientGPAReference, match="0 valid phase pixels") as error:
            compare_gpa_configs(image, first, second)
        assert error.value.parameter_group == group

    coarse = compare_gpa_configs(image, good, good, sample_step=100)
    assert coarse["quality"]["intersection_fraction"] == 0
    assert coarse["quality"]["iou"] is None
    assert coarse["diagnosis"]["configs_aligned"] is None
    assert coarse["diagnosis"]["max_strain_mae_percent"] is None
    assert coarse["diagnosis"]["min_strain_corr"] is None
    assert coarse["fields"] == {}
    json.dumps(coarse, allow_nan=False)
    full = compare_gpa_configs(image, good, good)
    assert full["quality"]["intersection_fraction"] == 3176 / image.size
    assert full["diagnosis"]["configs_aligned"] is True
    assert full["fields"]["exx"]["diff"]["mae"] == 0


def test_gpa_peak_fallback_preserves_original_rows(tmp_path):
    path, _ = write_fixture("J03", tmp_path)
    diffraction = DiffractionResult(
        center=(40, 32),
        peaks=[
            DiffractionPeak(x=40, y=32, intensity=1000),
            DiffractionPeak(x=30, y=32, intensity=90),
            DiffractionPeak(x=40, y=40, intensity=70),
            DiffractionPeak(x=50, y=32, intensity=98),
        ],
    )
    result = run_gpa(
        path,
        diffraction_result=diffraction,
        reference_region=(16, 48, 16, 48),
        refine_reference=False,
    )
    assert result.provenance["g_vector_source"] == "diffraction_peaks"
    assert [g.source_peak_index for g in result.g_vectors] == [3, 2]
    for g in result.g_vectors:
        peak = diffraction.peaks[g.source_peak_index]
        assert (g.gx.value, g.gy.value) == ((peak.x - 40) / 80, (peak.y - 32) / 64)
    report = AnalysisReport(
        strain=result,
        diffraction=diffraction,
        workflow={"steps": [{"name": "run_gpa", "status": "done"}]},
    )
    export_json_bundle(report, tmp_path)
    rows = json.loads((tmp_path / "json" / "strain.json").read_bytes())["g_vectors"]
    assert [row["source_peak_index"] for row in rows] == [3, 2]
    page = Path(export_html_report(report, tmp_path).uri)
    displayed = html.unescape(page.read_text("utf-8"))
    for index in (3, 2):
        assert f'"source_peak_index": {index}' in displayed


def test_gpa_export_writes_data_and_rgb_tiffs(tmp_path) -> None:
    tifffile = pytest.importorskip("tifffile")
    image = _synthetic_lattice(shape=(80, 80), exx=0.004, eyy=-0.002)
    fields = compute_gpa_fields(
        image,
        g_vectors=[(0.085, 0.0), (0.0, 0.082)],
        g_vector_unit="1/pixel",
        mask_sigma_px=5.0,
        apply_quality_mask=False,
    )

    manifest = export_gpa_maps(
        fields,
        tmp_path / "c2_gpa",
        image=image,
        fft_center_xy=(40.0, 40.0),
        selected_points_xy=[(46.8, 40.0), (40.0, 46.6)],
        colormap_name="BlOr",
        strain_scale=100.0,
        strain_unit="%",
    )

    assert "BlOr" in SUPPORTED_GPA_EXPORT_COLORMAPS
    assert (tmp_path / "c2_gpa" / "manifest.json").exists()
    assert (tmp_path / "c2_gpa" / "data" / "exx.npy").exists()
    assert (tmp_path / "c2_gpa" / "rgb_BlOr" / "ColorBar.tif").exists()
    assert (tmp_path / "c2_gpa" / "rgb_BlOr" / "Phase 1 Raw phase.tif").exists()
    assert manifest["quality"]["valid_pixel_fraction"] > 0.0
    assert tifffile.imread(tmp_path / "c2_gpa" / "data" / "exx.tif").dtype.kind == "f"
    assert tifffile.imread(tmp_path / "c2_gpa" / "rgb_BlOr" / "exx.tif").shape == (
        80,
        80,
        3,
    )
    colorbar = tifffile.imread(tmp_path / "c2_gpa" / "rgb_BlOr" / "ColorBar.tif")
    assert colorbar.shape == (512, 100, 4)
    assert np.allclose(colorbar[0, 0, :3], np.array([255, 85, 0]), atol=4)
    assert np.allclose(colorbar[-1, 0, :3], np.array([7, 90, 254]), atol=4)
    assert colorbar[0, 0, 3] == 255
    assert colorbar[0, -1, 3] == 0
    default_manifest = export_gpa_maps(
        fields,
        tmp_path / "c2_gpa_default",
        image=image,
    )
    assert default_manifest["colormap"] == "BlOr"
    assert (tmp_path / "c2_gpa_default" / "rgb_BlOr" / "ColorBar.tif").exists()
    smooth_manifest = export_gpa_maps(
        fields,
        tmp_path / "c2_gpa_smooth_display",
        image=image,
        display_smoothing_sigma_px=1.0,
    )
    assert smooth_manifest["display_smoothing_sigma_px"] == pytest.approx(1.0)
    assert np.array_equal(
        tifffile.imread(tmp_path / "c2_gpa_default" / "data" / "exx.tif"),
        tifffile.imread(tmp_path / "c2_gpa_smooth_display" / "data" / "exx.tif"),
    )
    assert tifffile.imread(
        tmp_path / "c2_gpa_smooth_display" / "rgb_BlOr" / "exx.tif"
    ).shape == (80, 80, 3)
    assert np.all(
        np.isfinite(tifffile.imread(tmp_path / "c2_gpa_smooth_display" / "rgb_BlOr" / "exx.tif"))
    )
    assert render_gpa_rgb(
        np.zeros((2, 2)),
        colormap_name="Thermal",
        vmin=-1.0,
        vmax=1.0,
    ).shape == (2, 2, 3)


def test_turbo_multistop_style_keeps_matplotlib_turbo_available() -> None:
    gpa_turbo = render_gpa_rgb(
        np.array([[0.0, 0.5, 1.0]], dtype=np.float64),
        colormap_name="Turbo",
        vmin=0.0,
        vmax=1.0,
    )
    matplotlib_turbo = render_gpa_rgb(
        np.array([[0.0, 0.5, 1.0]], dtype=np.float64),
        colormap_name="Matplotlib Turbo",
        vmin=0.0,
        vmax=1.0,
    )

    assert "Matplotlib Turbo" in SUPPORTED_GPA_EXPORT_COLORMAPS
    assert np.allclose(gpa_turbo[0, 0], np.array([51, 27, 61]), atol=4)
    assert np.allclose(gpa_turbo[0, 1], np.array([164, 250, 80]), atol=4)
    assert np.allclose(gpa_turbo[0, 2], np.array([119, 21, 19]), atol=4)
    assert not np.array_equal(gpa_turbo, matplotlib_turbo)


def test_blor_uses_blue_orange_diverging_stops() -> None:
    rgb = render_gpa_rgb(
        np.array([[-1.0, 0.0, 1.0]], dtype=np.float64),
        colormap_name="BlOr",
        vmin=-1.0,
        vmax=1.0,
    )

    assert np.allclose(rgb[0, 0], np.array([7, 90, 254]), atol=4)
    assert np.allclose(rgb[0, 1], np.array([235, 255, 235]), atol=4)
    assert np.allclose(rgb[0, 2], np.array([255, 85, 0]), atol=4)


def test_thermal_and_polar_follow_qcustomplot_presets() -> None:
    thermal = render_gpa_rgb(
        np.array([[0.0, 0.33, 0.85, 1.0]], dtype=np.float64),
        colormap_name="Thermal",
        vmin=0.0,
        vmax=1.0,
    )
    polar = render_gpa_rgb(
        np.array([[0.0, 0.5, 1.0]], dtype=np.float64),
        colormap_name="Polar",
        vmin=0.0,
        vmax=1.0,
    )

    assert np.allclose(thermal[0, 0], np.array([0, 0, 50]), atol=4)
    assert np.allclose(thermal[0, 1], np.array([200, 30, 140]), atol=4)
    assert np.allclose(thermal[0, 2], np.array([255, 255, 40]), atol=4)
    assert np.allclose(thermal[0, 3], np.array([255, 255, 255]), atol=4)
    assert np.allclose(polar[0, 0], np.array([50, 255, 255]), atol=4)
    assert np.allclose(polar[0, 1], np.array([0, 0, 0]), atol=4)
    assert np.allclose(polar[0, 2], np.array([255, 255, 50]), atol=4)


def test_run_gpa_reference_region_keeps_left_baseline_near_zero() -> None:
    image = _piecewise_exx_lattice()

    out = run_gpa(
        image,
        g_vectors=[(0.085, 0.0), (0.0, 0.082)],
        g_vector_unit="1/pixel",
        reference_region=(48, 144, 16, 80),
        refine_reference=True,
        gpa_mask_sigma_px=5.0,
        map_inline_mode="full",
    )

    exx = _array(out.strain_xx)
    left = exx[70:130, 24:78]
    right = exx[70:130, 128:178]
    assert abs(float(np.mean(left))) < 0.006
    assert float(np.mean(right)) > 0.008
    assert out.reference_region == (48, 144, 16, 80)


def test_compute_gpa_fields_accepts_independent_reference_regions_per_g() -> None:
    image = _synthetic_lattice(shape=(96, 96), exx=0.003, eyy=-0.002)
    refs = ((12, 48, 10, 44), (48, 86, 44, 88))

    fields = compute_gpa_fields(
        image,
        g_vectors=[(0.085, 0.0), (0.0, 0.082)],
        g_vector_unit="1/pixel",
        reference_regions=refs,
        mask_sigma_px=5.0,
        apply_quality_mask=False,
    )
    result = fields.to_strain_result(map_inline_mode="preview")

    assert fields.reference_regions == refs
    assert fields.reference_region is None
    assert [phase.reference_region for phase in fields.phase_maps] == list(refs)
    assert result.reference_region is None
    assert result.provenance["reference_regions"] == {
        "g1": list(refs[0]),
        "g2": list(refs[1]),
    }
    assert result.provenance["phase_maps"]["g1"]["reference_region"] == list(refs[0])
    assert result.provenance["phase_maps"]["g2"]["reference_region"] == list(refs[1])


def test_run_gpa_accepts_diffraction_result_inverse_nm_vectors() -> None:
    pixel_size_nm = 0.02
    image = _synthetic_lattice(exx=0.01)
    diffraction = DiffractionResult(
        g_vectors=[
            GVector(
                gx=Measurement(value=0.085 / pixel_size_nm, unit="1/nm", method="test"),
                gy=Measurement(value=0.0, unit="1/nm", method="test"),
            ),
            GVector(
                gx=Measurement(value=0.0, unit="1/nm", method="test"),
                gy=Measurement(value=0.082 / pixel_size_nm, unit="1/nm", method="test"),
            ),
        ]
    )

    out = run_gpa(
        image,
        method="gpa_b3",
        diffraction_result=diffraction,
        pixel_size_nm=pixel_size_nm,
    )

    assert out.provenance["method"] == "gpa_b3"
    assert len(out.g_vectors) == 2
    assert out.provenance["map_stats"]["mean_exx"] == pytest.approx(0.01, abs=0.007)


def test_run_gpa_rejects_collinear_vectors() -> None:
    image = _synthetic_lattice()
    with pytest.raises(ValueError, match="non-collinear"):
        run_gpa(
            image,
            g_vectors=[(0.08, 0.0), (0.16, 0.0)],
            g_vector_unit="1/pixel",
        )


def test_run_gpa_rejects_out_of_nyquist_frequency_vectors() -> None:
    image = _synthetic_lattice()
    with pytest.raises(ValueError, match="Nyquist"):
        run_gpa(
            image,
            g_vectors=[(0.7, 0.1), (0.0, 0.08)],
            g_vector_unit="1/pixel",
        )


def test_run_gpa_rejects_diffraction_calibrated_vectors_outside_fft_grid() -> None:
    image = _synthetic_lattice()
    with pytest.raises(ValueError, match="Nyquist"):
        run_gpa(
            image,
            g_vectors=[(-43.667449, -23.954475), (-44.47289, 3.905658)],
            g_vector_unit="1/nm",
            pixel_size_nm=0.02,
        )


def test_displacement_unwrap_respects_quality_mask() -> None:
    rng = np.random.default_rng(4)
    shape = (64, 64)
    yy, xx = np.indices(shape, dtype=np.float64)
    quality = np.zeros(shape, dtype=bool)
    quality[16:48, 16:48] = True
    phase1 = rng.uniform(-np.pi, np.pi, shape)
    phase2 = rng.uniform(-np.pi, np.pi, shape)
    phase1[quality] = -2.0 * np.pi * 0.08 * (0.1 * xx[quality])
    phase2[quality] = -2.0 * np.pi * 0.08 * (0.1 * yy[quality])

    ux, uy = _displacement_from_phases(
        [phase1, phase2],
        np.asarray([(0.08, 0.0), (0.0, 0.08)], dtype=np.float64),
        quality_mask=quality,
    )

    assert np.allclose(ux[~quality], 0.0)
    assert np.allclose(uy[~quality], 0.0)
    assert float(np.nanpercentile(np.abs(ux[quality]), 95.0)) < 8.0
    assert float(np.nanpercentile(np.abs(uy[quality]), 95.0)) < 8.0


def test_displacement_unwrap_can_skip_quality_mask_for_full_field_display() -> None:
    shape = (64, 64)
    yy, xx = np.indices(shape, dtype=np.float64)
    quality = np.zeros(shape, dtype=bool)
    quality[16:48, 16:48] = True
    phase1 = -2.0 * np.pi * 0.08 * (0.1 * xx)
    phase2 = -2.0 * np.pi * 0.08 * (0.1 * yy)

    ux, uy = _displacement_from_phases(
        [phase1, phase2],
        np.asarray([(0.08, 0.0), (0.0, 0.08)], dtype=np.float64),
        quality_mask=None,
    )

    assert float(np.mean(np.abs(ux[~quality]))) > 0.1
    assert float(np.mean(np.abs(uy[~quality]))) > 0.1


def test_b3_diffraction_pixel_offset_unit_is_interpreted_for_gpa() -> None:
    g_vector = GVector(
        gx=Measurement(value=24.0, unit="1/pixel", method="peak_offset_x_px"),
        gy=Measurement(value=12.0, unit="1/pixel", method="peak_offset_y_px"),
    )

    assert _diffraction_g_vector_unit_for_gpa(g_vector) == "fft_pixel"


def test_strain_tensor_axis_convention_direct() -> None:
    yy, xx = np.indices((80, 96), dtype=np.float32)
    ux = 0.02 * xx + 0.01 * yy
    uy = -0.03 * xx + 0.04 * yy

    exx, eyy, exy, wxy = _strain_from_displacement(ux, uy)

    assert float(np.mean(exx)) == pytest.approx(0.02, abs=1e-6)
    assert float(np.mean(eyy)) == pytest.approx(0.04, abs=1e-6)
    assert float(np.mean(exy)) == pytest.approx(-0.01, abs=1e-6)
    assert float(np.mean(wxy)) == pytest.approx(-0.02, abs=1e-6)
