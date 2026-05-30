# Project Memory (Durable)

This file captures stable, long-lived team rules, ownership, milestones, and operational quirks for this repository.

## Repo

- Repository: `leima-max/AutoTEM-Insight-TEM-Structure-Skill`
- Default branch: `main`

## Ownership (A/B/C)

Source of truth: `docs/development.md`.

- Member A: package structure, shared models, workflow, CLI, reporting, CI, validation.
- Member B: input, calibration, FFT, diffraction, indexing, phase matching, simulation.
- Member C: atom detection, atomic refinement, GPA, strain, defects, Z contrast, AI modules.

## Operational Notes

- User rule added on 2026-05-13: task ownership is determined by the leading letter of task ID (`A*` -> Member A, `B*` -> Member B, `C*` -> Member C).
- User rule added on 2026-05-13: if task/module ownership is unclear, check `configs/three-person-module-distribution.md` first.
- B0-B4 baseline milestone on 2026-05-27: B4 is connected through `run_diffraction_analysis`, configurable via workflow `candidate_phase_params`/CIF/phase lookup, exported through `indexed_spots.csv`, and covered by the public B benchmark fixture `b_diffraction_public`.
- B5 adapter milestone on 2026-05-27: optional `pymatgen` CIF parsing and GPL-isolated `diffsims` diffraction simulation adapters are wired into B4/B5 without becoming default hard dependencies.
- B5 phase-matching milestone on 2026-05-27: `simulation.simulation_engine: diffsims` now supports a tested experimental-vs-simulated spot matching loop that rotation-aligns simulated spots to B3 g-vectors, updates top-N phase candidates, and records matched spot count, mean d-error, mean angle-error, and provenance.
- B5 diffsims integration milestone on 2026-05-27: diffsims candidates now preserve optional CIF/Materials Project atom motifs and space-group numbers, support rotation sweeps, export simulated spot tables, and keep `diffsims`, `diffpy.structure`, and `orix` behind the GPL adapter extra.
- B5 real-environment milestone on 2026-05-27: Windows/Miniconda environment has real `diffsims 0.7.0`, `diffpy.structure 3.2.3`, and `orix 0.14.2` installed; a real Si diffsims smoke produced a pattern, 48 simulated spots, hkl labels, and intensities.
- Materials Project API milestone on 2026-05-27: `materials_project_api` lookup uses optional `mp-api`/`MPRester`; first-use guidance points users to `MP_API_KEY` or `autotem configure-mp-api --api-key-stdin`.
- Materials Project lookup milestone on 2026-05-28: `phase_lookup_materials` now accepts common material aliases such as `quartz` and Materials Project IDs such as `mp-149`, resolves them into formula/MP-ID queries with provenance, and can feed MP API candidates with atom sites into the B4->B5 diffsims closed-loop path.
- Materials Project ranking milestone on 2026-05-29: MP API lookup now ranks and optionally filters candidate structures before B4/B5 matching, preferring exact MP IDs, non-deprecated entries, stable/low-energy phases, non-theoretical entries, and smaller structures while preserving ranking metadata in phase candidates and CSV exports.
- B4/B5 reporting milestone on 2026-05-29: HTML reports now include a candidate comparison table that merges B4/Materials Project top-N ranking metadata with diffsims phase-matching evidence, including selected status, scores, matched spots, d-error, and angle-error.
- B-section UI milestone on 2026-05-30: the Streamlit preview app was expanded from B1-B3 into a B1-B5 diffraction debug UI with B4 phase lookup/MP controls, B5 simulation/phase-matching controls, B5 result tables/overlay, and a timeout-guarded diffsims subprocess path.
- Automation safety rule for `GitHub Progress Memory Sync`:
  - Always inspect local Git state before any sync action.
  - If the worktree has uncommitted changes, untracked files, an in-progress merge/rebase/cherry-pick, or local branches not safely merged, only record the status and stop.
  - In that state, do not run `git pull`, `git checkout`, `git reset`, `git clean`, delete files/directories, or reclone the repository.
  - Any destructive or state-changing recovery must wait for explicit user approval.
- Git for Windows (MSYS2) may fail with `couldn't create signal pipe, Win32 error 5` during clone/fetch operations.
  - Workaround (PowerShell): set `TEMP`/`TMP` to a writable directory such as `C:\tmp` for that command/session.
