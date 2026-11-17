# Traditional operation plans

Use the bundled plans as parameter-free route examples. `input_path` must be replaced with an authorized file, e.g. CLI `--input`. Discover operation parameter schemas before adding calibration, candidate structures or reference data. File placeholders are not sample data. Plans with absent physical prerequisites may yield pixel diagnostics or needs-input, never implicit nm/pm/phase success.

- [HRTEM diffraction](recipes/hrtem-diffraction.yaml): QA/calibration/preprocess/FFT/center/peaks/metrics. Add registered `diffraction.index` only with authorized local candidates; add bounded simulation/matching only when dependencies are available.
- [SAED diffraction](recipes/saed-diffraction.yaml): QA/calibration/direct pattern/center/peaks/metrics, without FFT. Supply a trusted reciprocal scale for physical d-spacing; NBED can use this route with its modality explicitly selected.
- [HAADF atomic](recipes/haadf-atomic.yaml): QA/calibration/detect/raw-refine/distances/relative intensity. Pixel output remains pixel until valid calibration is supplied.
- [GPA](recipes/hrtem-gpa.yaml): QA/calibration/GPA. Supply reliable `g_vectors`, `g_vector_unit`, `[y0,y1,x0,x1]` reference ROI and necessary physical scale after host clarification. Absence is an intentional needs-input path, not a successful strain result.

All plans default to no network. Supplemented answers form a new plan/new run linked with `parent_run_id`; preserve the old evidence. Standalone advanced operations are discoverable but may require additional lattice indices/basis/models or reference inputs; do not invent these to make a template execute.
