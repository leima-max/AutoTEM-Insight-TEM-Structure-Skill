# Task 2: Reference-aware inspection
Work only in the isolated source worktree; baseline 3e822c9. Read plan constraints.

Own autotem/harness/service.py, autotem/harness/cli.py, new tests/test_harness_inspection_modes.py. Task 1 owns all shared Skill/contracts.

Implement inspect_input(self, path: str, *, input_mode: Literal['snapshot','reference']='snapshot', **selection) and forward top-level mode into the existing inspection plan. Preserve selection/modality, workspace restrictions and source-hash checks; no storage/worker/algorithm rewrite.

Expose optional enum/default in TOOL_SCHEMAS, dispatch and inspect --input-mode {snapshot,reference}. MCP reuses schema/dispatch and retains nine names.

Red/green tests: default snapshot copies input; reference creates neither source raw/ snapshot nor .snapshot temporary file while dataset/scientific raw-array/preview remain readable. Preserve selections; reject illegal modes and nested selection.input_mode policy override. Verify public CLI/JSONL consistency and workspace/source-change protection. Reuse existing regressions rather than duplicate large suites.

No commits, nested agents, models or old evidence edits. Root supplies interpreter and full regression. Report task-2-report.md in repair output root with commands, red/green and limits.
