# AutoTEM-Insight / Auto TEM Skill

AutoTEM-Insight provides a reusable TEM analysis workflow scaffold with shared schemas,
calibration, diffraction analysis, reporting, and validation hooks.

## Target Local Product

The final product is a local **AutoTEM Skill + domain subagent profiles + Local
Harness + scientific algorithms/models + visualization workbench** bundle.
The user's existing Codex, OpenClaw, WorkBuddy, or another compatible agent acts
as the main agent; AutoTEM does not rebuild chat, intent recognition, or generic
task planning. Host capabilities are verified per adapter, not assumed identical.

The Skill will instruct the host to start or reuse the local frontend and open
the current analysis run automatically. The workbench will browse registered
intermediate images/arrays/tables/model evidence, support manual parameter/ROI/
peak/atom adjustments, and create versioned, dependency-aware recomputations.
Raw data and earlier results remain immutable; display settings do not alter
scientific measurements.

The workbench and interactive revisions remain development targets. The headless
classical Harness below is executable. Existing Streamlit UIs support debugging.

- [Local product goals, architecture and acceptance](docs/autotem-local-product-spec.md)
- [Complete technical development guide](docs/PROJECT_TECHNICAL_DEVELOPMENT_GUIDE.md)
- [Headless Harness release and end-to-end development plan (2026-11-12)](docs/HARNESS_RELEASE_TECHNICAL_DEVELOPMENT_PLAN.md)
- [Product/technical review and implemented fixes (2026-10-05)](docs/PRODUCT_TECHNICAL_REVIEW_2026-10-05.md)
- [Requirement traceability and acceptance scenarios](docs/product-acceptance-matrix.md)
- [Development stages](docs/development.md)
- [AI/DL implementation plan](docs/ai-dl-model-complete-implementation-plan.md)
- [Per-model data, training and evaluation manual](docs/ai-dl-model-babysitter-operation-manual.md)

## Headless Classical Harness

The Windows x64 candidate provides an offline setup for the `traditional-core-mcp`
profile, including project-local CPython and a hash-locked wheelhouse.
[Windows installation and acceptance](docs/harness-windows-release.md)

The registered execution path provides 37 implemented operations and 12 reserved
extension slots, nine coarse tools, three domain profiles, immutable artifacts,
SQLite jobs/events, and optional MCP stdio. No frontend or deep-learning runtime
is required. See [implementation and acceptance status](docs/harness-implementation-status.md),
[host installation/rollback](docs/harness-host-adapters.md), and the
[Skill](skills/autotem/SKILL.md).

Importing the Skill does not install its scientific dependencies. The stdlib-only
`python -m autotem.bootstrap` installer prepares a project-private environment,
validates the actual tools and installs the host resources; offline wheels are
the default, with explicit opt-in for network installation. The maintenance
`doctor` command reports runtime, transport and host readiness separately.
See [installation experience and remaining distribution gates](docs/harness-installation-experience.md).
Configured MCP is preferred for routine host calls; host launch permissions still apply.

```powershell
python -m autotem.harness.cli --root outputs/harness capabilities
python -m autotem.harness.cli --root outputs/harness inspect path/to/image.tif
python -m autotem.harness.cli --root outputs/harness validate --plan configs/harness/haadf-atomic.yaml --input path/to/image.tif
python -m autotem.harness.cli --root outputs/harness submit --plan configs/harness/haadf-atomic.yaml --input path/to/image.tif
```

Replace the input with a real file. Fill scientific parameters from supplied
metadata or user answers; recipes contain no assumed physical calibration.
Submission returns a run/job ID. Query `status RUN_ID`, list `artifacts RUN_ID`,
then `export RUN_ID --formats json html pdf`. Missing conditions return structured
`needs_input`; answers or scientific changes create a child run with `--parent-run-id`.
The Python entry point is `TEMAnalysisSkill.run_harness(...)`; the compatible
CLI facade is `autotem analyze ... --harness --workflow ...`.

Current Codex Desktop native delegation has been exercised with all three domain
profiles. WorkBuddy and OpenClaw adapter resources are supplied; their real user
journeys remain unverified. Successful execution does not certify scientific
accuracy, and does not complete the long-term workbench product gate.

## What Is Available

- B1 calibration baseline: pixel calibration + known-d-spacing diffraction calibration.
- B3 diffraction baseline: FFT peak extraction, g-vectors, d-spacing table.
- B4 indexing adapter: optional lattice-fit indexing via `candidate_phase_params`.
- A2/A3 pipeline/reporting: CLI + JSON/CSV/HTML outputs + intermediate artifacts.

## Quick Run

```powershell
autotem analyze path/to/your_image.tif --output outputs/your_unique_run
```

Replace the input path with an existing file. Preserve unique output roots for
scientific runs. Add `--fail-on-blocking` when automation must return nonzero for
partial execution or failed quality/acceptance gates. The legacy completion text
and a zero default exit code may describe a diagnostic report; they do not prove
scientific accuracy or completed A5/A6 product integration.

## B1-B5 Diffraction Debug UI

Launch an interactive UI for the B-section workflow: B1 quality/calibration,
B2 preprocessing, B3 FFT peaks/g-vectors/d-spacing, B4 indexing/phase lookup,
and B5 simulation/phase matching.

```powershell
autotem preview-b1b3 --host 127.0.0.1 --port 8501
```

The shorter alias is also available:

```powershell
autotem preview-b --host 127.0.0.1 --port 8501
```

The installed preview writes under `outputs/b1_b3_preview` in the current working directory.
Use `--output-root PATH` to select another destination. A source checkout defaults to its own
`outputs/b1_b3_preview` directory.

If UI dependencies are missing:

```powershell
pip install "autotem-insight[ui]"
```

Optional crystallography adapters:

```powershell
pip install "autotem-insight[crystallography]"
```

Materials Project API access uses the official `mp-api` package and requires
an API key. On first use, AutoTEM will warn if the key is missing. Configure it
with either `MP_API_KEY` or the helper command:

```powershell
autotem configure-mp-api --api-key-stdin
```

Optional GPL-licensed simulation adapters, including `diffsims`:

```powershell
pip install "autotem-insight[gpl_adapters]"
```

Check a real diffsims installation outside pytest:

```powershell
autotem validate-diffsims
```

This standalone smoke check simulates the local built-in Si/Cu structures with
the real diffsims adapter, then runs a small template-matching closed loop.

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
- `diffraction.candidate_phase_params`: optional B4 lattice constraints
  (`a/b/c`, angles, space group) used to write `indexed_spots.csv` and
  ranked `phase_candidates.csv`.
- `diffraction.cif_candidate_paths` / `cif_candidates_dir`: import existing CIF
  files as B4 candidates.
- `diffraction.phase_candidate_materials`: unified B3/B4 material or phase
  list. The same names are used as the B4 candidate filter and, when lookup is
  enabled, as crystal-source lookup queries. Legacy `candidate_phases` and
  `phase_lookup_materials` remain accepted and are merged for compatibility.
- `diffraction.phase_lookup_sources`: selecting one or more sources resolves
  material names or formulas through the local built-in crystal library and
  optional online sources (`materials_project_api`, `materials_project_optimade`,
  `cod`) before B4 matching. Legacy `phase_lookup_enabled` is still accepted for
  older recipes. The interactive B4 UI selects all four sources by default; if
  `materials_project_api` has no API key configured, the UI shows the setup
  guidance while the other sources can still return candidates.
- material entries accept local names/formulas, common aliases such as
  `quartz`, and Materials Project IDs such as `mp-149`. MP IDs are routed
  directly to `materials_project_api`; common aliases are resolved to formulas
  before online lookup.
- `diffraction.materials_project_sort_by`: ranks MP API candidates before B4
  matching. The default `stability` priority prefers exact MP IDs, non-deprecated
  entries, stable phases, lower energy above hull, non-theoretical entries, and
  smaller structures. Optional controls include
  `materials_project_max_energy_above_hull`,
  `materials_project_filter_stable`, and
  `materials_project_candidate_pool_size`.
- `simulation.simulation_engine: diffsims`: optional B5 kinematic diffraction
  simulation path. It now supports a first closed loop: diffsims simulated spots
  are rotation-aligned against measured B3 g-vectors, then used to update
  `matched_spot_count`, mean d-error, mean angle-error, and phase scores. This
  requires the GPL-isolated `gpl_adapters` extra.
- `simulation.phase_match_*`: controls for the diffsims closed-loop matcher,
  including d/angle tolerances, minimum matched spots, top-N candidates, and the
  weight used when combining B4 lattice-fit confidence with diffsims evidence.
- `simulation.zone_axis_candidates`: optional B5 zone-axis sweep. AutoTEM
  simulates each requested zone axis, matches every simulated pattern against
  measured B3 g-vectors, then writes the best zone axis back to the selected
  `PhaseCandidate` and simulation provenance.
- `simulation.rotation_axis` / `rotation_angles_deg`: optional diffsims rotation
  sweep. AutoTEM simulates each requested rotation and keeps the best
  experimental-vs-simulated match per phase.
- `candidate_phase_params[].atom_sites`: optional fractional atom motif used by
  diffsims. CIF/pymatgen and Materials Project API candidates populate this when
  available; otherwise the adapter can fall back to a single atom and records a
  warning.

These controls are optional and fully workflow-driven.

## B5 diffsims Closed-Loop Example

Generate a tiny synthetic HRTEM-like input, then run the B3->B4->B5 closed loop:

```powershell
python examples/generate_diffsims_closed_loop_input.py
autotem analyze examples/diffsims_closed_loop_input.npy --workflow examples/diffsims_closed_loop.yaml --output outputs/diffsims_closed_loop
```

Expected key outputs:

- `outputs/diffsims_closed_loop/tables/phase_candidates.csv`
- `outputs/diffsims_closed_loop/tables/simulation_phase_matches.csv`
- `outputs/diffsims_closed_loop/tables/simulated_spots.csv`
- `outputs/diffsims_closed_loop/figures/simulation_diffraction_overlay.png`
- `outputs/diffsims_closed_loop/report.html`

The HTML report's `Simulation Phase Matching` section includes a candidate
comparison table that combines Materials Project ranking metadata with diffsims
matched spot counts, scores, inferred zone axis, and d/angle errors. The
figures directory also includes a reciprocal-space overlay of measured
g-vectors and the best simulated diffraction spots.

With a configured Materials Project API key, use the MP lookup example:

```powershell
autotem configure-mp-api --check
autotem analyze your_input.npy --workflow examples/materials_project_diffsims_lookup.yaml --output outputs/mp_diffsims_lookup
```

That workflow resolves the MP structure first, carries its atom sites into B4
candidate metadata, and then runs the diffsims phase-matching loop.

## Tests

Smoke suite (default):

```powershell
python -m pytest
```

Run extended tests (including non-smoke):

```powershell
python -m pytest -m "smoke or not smoke"
```
