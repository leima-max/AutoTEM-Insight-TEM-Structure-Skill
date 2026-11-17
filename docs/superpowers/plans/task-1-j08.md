# Task 1: J08 persisted reference closure
Work only in the isolated source worktree; baseline 3e822c9. Read global constraints in the plan.

Own skills/autotem/SKILL.md, references/tool-contracts.md, references/profiles/atomic-strain.yaml and references/hosts/codex/autotem_atomic_strain.toml under that Skill, their exact autotem/harness/resources/skill mirrors, descriptive metadata in autotem/services/strain.py, and new tests/test_harness_atomic_reference_closure.py. Do not edit service.py, cli.py, verify.py or other tests.

Include the user's requested reference-dependent strain operation in the complete plan (or append to a completed refined run), even when the reference is missing. The existing gate must produce real needs_input, numbered evidence and a request ID before the host asks and ends its turn. Do not invent a reference/pending state or add strain to coordinate-only requests.

Replies retain actual request ID, untrimmed values, source, basis and native raw_reply. Validate is a read-only preview. A response to an active scientific request must reach existing bounded submit_analysis with parent_run_id and stable idempotency key; rejection is persisted before child/worker creation. Do not stop at read-only validation if durable reply handling is needed. This is not permission to submit unrelated invalid workflows. Keep current safe database text/numeric policy.

Describe mutually exclusive reference forms, raw half-open [row_start,row_stop,col_start,col_stop], at least six two-dimensional reference atoms, and missing reference -> needs_input before computation. No scientific algorithm changes.

Common tool contract also documents task 2's API: inspect_input(path, *, input_mode='snapshot', **selection), public JSON path/selection/input_mode enum snapshot/reference, CLI inspect --input-mode reference. A no-copy request selects reference from inspection onward; scientific outputs remain registered.

Use TDD for discoverable descriptions plus real service tests: missing reference has refined numbered evidence/no strain; reopen pending; valid ROI makes linked child with adopted values and unchanged parent scientific plan/artifacts; [8,1000,8,68] rejected durably with no child/job and open request; coordinates-only completes. Root supplies interpreter.

No commits, nested agents, model calls or old evidence edits. Report task-1-report.md in the new repair output root, with files, exact commands, red/green, coverage and risks.
