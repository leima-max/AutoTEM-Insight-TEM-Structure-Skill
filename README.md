# AutoTEM-Insight / Auto TEM Skill

AutoTEM-Insight provides a reusable TEM analysis workflow scaffold with shared schemas,
calibration, diffraction analysis, reporting, and validation hooks.

## What Is Available

- B1 calibration baseline: pixel calibration + known-d-spacing diffraction calibration.
- B3 diffraction baseline: FFT peak extraction, g-vectors, d-spacing table.
- B4 indexing adapter: optional lattice-fit indexing via `candidate_phase_params`.
- A2/A3 pipeline/reporting: CLI + JSON/CSV/HTML outputs + intermediate artifacts.

## Quick Run

```powershell
autotem analyze examples/sample.tif --output outputs/sample_run
```

## B1-B3 Preview UI

Launch an interactive UI for B1/B2/B3 preview testing:

```powershell
autotem preview-b1b3 --host 127.0.0.1 --port 8501
```

If UI dependencies are missing:

```powershell
pip install "autotem-insight[ui]"
```

## B1 Calibration Fill-In (Recommended)

When you know one standard ring/spot:

- `known_d_spacing` (nm/angstrom/pm)
- `measured_radius_px` (peak radius in FFT pixels)

Use the provided workflow example:

```powershell
autotem analyze examples/sample.tif --workflow configs/hrtem_b1_b3.yaml --output outputs/hrtem_b1_b3
```

This will calibrate `diffraction_scale` first, then propagate it into B3 g-vector/d-spacing.

## B3/B4 Optimization Controls

`configs/hrtem_b1_b3.yaml` now includes baseline controls for staged optimization:

- `diffraction.auto_peak_tuning`: size-aware auto defaults for `top_k_peaks`,
  `min_peak_distance`, `center_exclusion_radius`, and detector sigma.
- `diffraction.peak_threshold_strategy: auto`: adaptive auto quantile thresholding
  based on FFT response robust SNR proxy.
- `reporting.diffraction_gates`: report-time acceptance gates for:
  - minimum peak count
  - d-spacing uncertainty coverage
  - mean relative uncertainty
  - indexed non-zero hkl fraction (B3->B4 consistency)

These controls are optional and fully workflow-driven.

## Tests

Smoke suite (default):

```powershell
python -m pytest
```

Run extended tests (including non-smoke):

```powershell
python -m pytest -m "smoke or not smoke"
```
