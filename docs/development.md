# Development Guide

## Branches

- `main`: stable branch protected by review and CI.
- `dev`: integration branch for milestone work.
- `codex/*`, `feature/*`, and `fix/*`: short-lived development branches.

## Local Setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
```

## Module Ownership

- Member A owns package structure, shared models, workflow, CLI, reporting, CI, and validation.
- Member B owns input, calibration, FFT, diffraction, indexing, phase matching, and simulation.
- Member C owns atom detection, atomic refinement, GPA, strain, defects, Z contrast, and AI modules.

## Interface Rule

Cross-module outputs should move through stable core models instead of loose dictionaries once the
model layer is implemented.
