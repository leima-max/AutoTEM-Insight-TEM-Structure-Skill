# AutoTEM-Insight / Auto TEM Skill

AutoTEM-Insight is a planned full-pipeline TEM data-analysis skill for converting raw TEM/HRTEM/STEM/4D-STEM data into reproducible structural parameters, diffraction indexing, atom coordinates, strain maps, defect statistics, phase candidates, simulations, and reports.

See [AGENTS.md](AGENTS.md) for the project-level agent guide, architecture expectations, MVP priorities, validation targets, and development standards.

## Initial MVP

The first reliable workflow should support a single HRTEM or HAADF-STEM image and produce:

- image quality assessment
- pixel calibration
- FFT image and diffraction peak table
- d-spacing and g-vector measurements
- hkl / phase candidates
- atom coordinates and atom-distance statistics
- preliminary strain map
- automatic JSON/CSV/HTML report outputs

## Local UI (Atom Detection C0)

Run a simple UI for atom-column detection (supports `dm/dm3/dm4/tif/tiff`):

```powershell
.\.venv\Scripts\python.exe -m streamlit run apps\atom_detection_ui.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

UI supports:
- realtime parameter update (toggle `Realtime update`)
- direct click add/remove on image canvas
- irregular polygon ROI selection
- CSV export of edited atom coordinates

## C0 Contract Alignment

`detect_atoms(...)` returns `AtomDetectionResult`, and you can convert it to the shared
`AtomLattice` contract used by core workflows:

```python
from autotem.atomic.atom_detection import detect_atoms

result = detect_atoms("example.dm4", mode="auto")
atom_lattice = result.to_atom_lattice(coordinate_unit="pixel")
```

`TEMAnalysisSkill` now auto-runs atom detection when workflow steps include `detect_atoms`,
and returns `atom_lattice` in the run output.

## Dependency Notes

- Core install (includes C0 atom detection + DM/TIFF support):
  - `python -m pip install -e ".[dev]"`
- UI install (Streamlit tooling):
  - `python -m pip install -e ".[dev,ui]"`

