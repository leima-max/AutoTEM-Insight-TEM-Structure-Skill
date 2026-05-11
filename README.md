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

