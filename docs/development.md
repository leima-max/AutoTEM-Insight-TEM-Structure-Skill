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

## Pre-commit / Pre-push Hooks

Install hooks:

```bash
pre-commit install --hook-type pre-commit --hook-type pre-push
```

Run hooks manually:

```bash
pre-commit run --all-files
pre-commit run --hook-stage pre-push --all-files
```

Hook behavior:

- `pre-commit`: `ruff-check` (with `--fix`) and `ruff-format` for fast local hygiene.
- `pre-push`: full `pytest` plus runtime contract validation
  (`python -m autotem.validation.runtime_contract`).

## Module Ownership

- Member A owns package structure, shared models, workflow, CLI, reporting, CI, and validation.
- Member B owns input, calibration, FFT, diffraction, indexing, phase matching, and simulation.
- Member C owns atom detection, atomic refinement, GPA, strain, defects, Z contrast, and AI modules.

## Interface Rule

Cross-module outputs should move through stable core models instead of loose dictionaries once the
model layer is implemented.

## Runtime Contracts

- `docs/runtime-contracts.md` defines the stable schema for `run_summary.json` and JSONL `run.log`.
- `docs/a2-reserved-step-contracts.md` defines A2 placeholder I/O contracts for reserved B/C integration steps.
