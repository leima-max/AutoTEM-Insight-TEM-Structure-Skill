# Headless report delivery and legacy recipes

The active report contract uses a `1.0.0` envelope. The object stored in an artifact such as `dataset`, `refined_atoms`, `gpa` or `atom_strain` retains its original core schema. Individual core results can be passed to `TEMDataset.model_validate`, `AtomLattice.model_validate` or `StrainResult.model_validate` after loading the registered JSON.

## Read a report

Export a terminal run with `export_report`. Its response identifies the run, the producing job, report type, required-step completeness and exported artifacts. JSON and HTML link the evidence for each requested step. PDF contains a bounded summary of the status and evidence identifiers.

| Report field | Interpretation |
|---|---|
| `claim_evidence` | Plan order, user request for the work package, effective operation/version, parameter location, consumed and produced Artifact IDs, warnings and scientific level. Failed or unexecuted steps retain their terminal reason. |
| `measurement_examples` within each claim | Selected top-level structured measurements from JSON artifacts up to 256 kB. Each displayed value includes its unit, method, source Artifact ID and JSON pointer. Missing uncertainty has an explicit status. Read the original artifact for all measurements, arrays, tables or masks. |
| `completeness` | Counts required steps with registered, usable evidence and identifies unmet required or skipped optional steps. Only the submitted plan defines the required steps. Scientific accuracy and acceptance gates have separate fields. |
| `results_by_role` | All delivered artifacts per role, including repeated executions of the same operation. The existing `result_index` selects the latest delivered result per role. |
| `strain_results.all` | Every delivered GPA and atom-based strain result with step, method, artifact and limitations. Both results retain their own full fields, reference and validity information in their registered artifacts. |
| `strain_results.primary` | First delivered required strain method in the submitted plan. A delivered optional method is the fallback. The selection rule is recorded alongside the full list. |

Needs-input, failure, cancellation, incomplete evidence, and failed acceptance gates produce diagnostic reports. Their claim rows identify pending or blocked work. Previous exports stay registered as report artifacts on the run. A subsequent export indexes the scientific and provenance artifacts for the captured job. Check each claim's scientific level, unit, reference and warnings before interpreting a quantitative value.

## Select an entry point

| Input and interface | Execution | Supported result |
|---|---|---|
| Formal YAML with top-level `steps` through `TEMAnalysisSkill.run` or `autotem analyze --workflow` | Shared Harness and registered services | Versioned run/report envelope and immutable results. The input argument supplies the real file. |
| Formal YAML or explicit plan via `run_harness`, Harness CLI or MCP | Shared Harness and registered services | Same operation and report contracts. A Python caller selects one plan or recipe. |
| Legacy YAML under `workflow` via `run_harness` or `autotem analyze --harness` | Compile supported steps into a registered plan, then submit it to the same service | Envelope with the translated steps, effective parameters, source recipe name and report formats. |
| Existing legacy YAML or omitted recipe via `run` or default `autotem analyze` | Historical workflow composition. Ambiguous image inputs request a selection. Explicit `workflow.parameters.load` selections use the same two-dimensional reader as `input.load`. | Original exports retain their strict `0.1.0` AnalysisReport and RunSummary, presentation and output options. For explicit selections, `dataset.source_path` identifies the original input and `dataset.data.uri` identifies the persisted selected plane. |

Use the packaged HRTEM, HAADF, GPA or SAED recipes for explicit headless analysis. A legacy `load, report` recipe produces one `input.load` operation and a terminal report. This supports reading and inspecting a selected file with no extra scientific branch.

When running an analysis, a multipage TIFF or an image with several bands (TIFF, PNG, JPEG) raises `NEEDS_INPUT` with a `frame` or `channel` field and its count. A NumPy stack requests one `selection_indices` entry per leading dimension. A multi-array NPZ first lists available `dataset_path` values. Set the requested fields under `workflow.parameters.load` and resubmit the recipe. Use zero-based indices. For the blue band on a second RGB page, set `load: {frame: 1, channel: 2}`. For an NPZ member named `scan` with shape `[3, 4, 20, 30]`, set `load: {dataset_path: scan, selection_indices: [1, 2]}`. The final two axes are the image. An optional `roi` selects `[y_start, y_stop, x_start, x_stop]` within it with exclusive stops.

A unique grayscale plane uses the original image reader and preserves TIFF calibration tags. Ordinary two-dimensional NPY, TXT and previously supported CSV inputs retain their reader metadata. The low-level `autotem.io.read_image()` still provides its historical first-page grayscale preview and corresponding warnings for multipage RGB TIFFs.

## Translate a legacy YAML workflow

| Legacy step or parameter group | Registered operations | Scope |
|---|---|---|
| `load` | `input.load` | Input selection fields, including modality, frame, channel, dataset selection, ROI. |
| `assess_quality` / `quality` | `quality.assess` | Registered threshold and weight parameters. |
| `calibrate` / `calibration` | `calibration.resolve` | Registered physical calibration parameters. The legacy `auto` method selects the service default. Use explicit registered parameters for manual references. |
| `preprocess` / `preprocessing` | `preprocessing.apply` | Registered preprocessing options and accepted recipe values. |
| `fft`, `diffraction_peaks`, `index_diffraction`, `phase_match` | `diffraction.fft`, `diffraction.center`, `diffraction.peaks`, `diffraction.metrics`, `diffraction.index` as selected by the steps | The historical `analyze` alias expands into these steps. The last two legacy steps share the candidate indexing result. Choose `diffraction.match` explicitly to compare against simulated evidence. |
| `detect_atoms` / `atom_detection` | `atomic.detect` | Registered detector parameters. |
| `run_gpa` / `gpa` | `strain.gpa` | Registered g vectors, units, reference and mask parameters. |
| `atom_strain`, `classify_sublattices` | `atomic.detect`, `atomic.refine`, then `strain.atom` or `atomic.sublattice.auto` | The result retains its own role. Use the formal recipe to set separate refinement or alternative coordinate inputs. |
| `simulate` / `simulation` | `simulation.diffraction` | Requires the operation's backend and sufficient structure evidence. |
| `report` / `outputs.formats` | `export_report` | JSON, HTML, PDF. A caller can override the requested report formats. Full intermediate evidence is retained. |

For diffraction, the translator accepts a uniquely bound registered field or an explicit alias. Supported aliases include `top_k_peaks` → `top_k`, `min_peak_distance` → `min_distance`, `fft_center_mode` → `center_mode`, `direct_beam_refine_window` → `refine_window`, `peak_threshold_strategy` → `threshold_strategy` and `phase_top_n` → `top_n`. Registry type, value and dependency checks run on the translated plan. Unsupported, unused, duplicate or ambiguous scientific parameters require migration. Edit the affected group or use a formal plan to target its operation explicitly.

The regression suite compares the legacy and registered reading paths against the same disk array, reloads both strict strain results, and exercises repeated roles, diagnostic evidence, invalid options and terminal report formats. [Report and compatibility tests](../tests/test_harness_report_compatibility.py)
