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
- Automation safety rule for `GitHub Progress Memory Sync`:
  - Always inspect local Git state before any sync action.
  - If the worktree has uncommitted changes, untracked files, an in-progress merge/rebase/cherry-pick, or local branches not safely merged, only record the status and stop.
  - In that state, do not run `git pull`, `git checkout`, `git reset`, `git clean`, delete files/directories, or reclone the repository.
  - Any destructive or state-changing recovery must wait for explicit user approval.
- Git for Windows (MSYS2) may fail with `couldn't create signal pipe, Win32 error 5` during clone/fetch operations.
  - Workaround (PowerShell): set `TEMP`/`TMP` to a writable directory such as `C:\tmp` for that command/session.
