# Codex startup and reference coordinate contract implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development; the user has explicitly authorized implementation, local regression and affected Codex retests.

**Goal:** Ship the proven Codex readiness wait by default and make reference coordinates unambiguous without changing scientific algorithms.

**Architecture:** Codex project MCP registration declares required startup with a bounded 180-second timeout for managed and direct installations. The existing zero-based, half-open numerical API remains unchanged; Skill text and persisted question metadata explicitly explain that contract and the source/local-frame boundary. The host clarifies ambiguous natural-language row/column references before adopting them, instead of silently choosing a base.

**Tech Stack:** Python, TOML, bundled Skill/Profile resources, pytest, offline Windows release tools and real Codex MCP sessions.

**Spec:** The current user request and the previous delivery's `product-review-final.md`, based on frozen `d02e001` (1,461 full tests and seven affected Codex cases passed).

## Constraints and decisions

- Work in the existing isolated worktree, branch `codex/codex-startup-coordinate-contract`; preserve old source, releases and field records.
- Only Codex receives required/startup defaults. Do not change model selection, trust policy, tool approvals, global configuration or other hosts' behavior.
- Do not overwrite user-modified MCP entries. Managed upgrade/reconnect/rollback must use existing ownership and conflict safeguards.
- Numeric ROI is zero-based `[y0,y1,x0,x1]`, upper bounds exclusive. Atom IDs are zero-based labels in the actual numbered evidence. Explicit user bases/endpoints are converted once; original text/values and basis remain traceable, with no clipping.
- Direct strain parameters use selected-array local coordinates. Source-frame replies follow the actual request, with origin subtraction performed once by the service. Add explicit `index_base: 0` and `upper_bound: exclusive` coordinate metadata; old requests remain compatible.
- A natural-language ordinal range without a base or endpoint convention must be clarified before strain computation; do not invent a scientific reference or fabricate a persisted request. Missing-reference workflows continue to create real needs_input evidence.
- No scientific algorithm changes or new dependencies. OpenClaw/WorkBuddy real sessions and GitHub pushes remain held.
- Native retest must use generated project defaults, without process `required` or startup-timeout overrides. Local full regression precedes any new model turn.

## Tasks

### Task 1: Default Codex readiness

Own `autotem/harness/hosts.py`, narrowly necessary installation helpers, new startup-specific tests, and `skills/autotem/references/installation.md` plus its resource mirror.

- [ ] Add failing tests for managed and direct Codex registration (`required=true`, timeout 180), idempotence, managed upgrade/reconnect/rollback, and preservation/conflict on modified user entries; other hosts unchanged.
- [ ] Implement minimal shared Codex entry policy using existing safe merges; document startup failure as a host connection failure, never scientific success.
- [ ] Verify focused host/installation tests. Record red/green logs and code review package; root owns commits.

### Task 2: Explicit reference coordinates

Own `autotem/harness/store.py`, descriptive parameter metadata in `autotem/services/strain.py`, Skill/tool-contracts/atomic profile/Codex atomic template and their mirrors, and new coordinate-specific tests.

- [ ] Add failing tests for explicit question metadata and end-to-end one-time source-to-local transformation, including a cropped nonzero origin and out-of-bounds rejection; retain old question compatibility.
- [ ] Document explicit zero/one-based and inclusive/half-open examples, numbered-ID semantics, and clarification for unspecified natural-language conventions. Keep runtime numeric meaning unchanged.
- [ ] Verify numerical/schema tests and all resource mirrors. Do not hard-code responses for acceptance cards.

### Task 3: Integrated release and field acceptance (root)

- [ ] Independent spec/quality review for each task, then integrate and freeze a clean candidate.
- [ ] Rebuild offline editions, run the complete frozen-source test selection (no hidden smoke selection/skips), and cold-install the new core package in a fresh project.
- [ ] Verify default Codex config and no-model cold readiness, failed-server/timeout diagnostics, safe upgrade and reconnect. Actual host probes must not use readiness overrides or broaden permissions.
- [ ] Retest original J08 answered/unanswered/invalid and J09 reference/snapshot plus J06; J04's ambiguous range now must clarify, then an explicit original-range interpretation may be supplied as a separately identified follow-up. Add a bounded coordinate case with a nonzero crop and explicit convention if original cards do not cover the new contract.
- [ ] Preserve original native turns, failures, lineage, source/child hashes and reports; independently audit artifacts and user-visible interpretations. Finish with product review and exact scope, without claiming the full 27+4/new release gates or scientific accuracy.

## Review focus / preflight

| Boundary | Decision and verification |
|---|---|
| Tasks 1/2 share packaged resources | Different source files; mirror each owned file, root verifies wheel/resource identity. |
| Task 1/3 generated host config | Required/default timeout must come from installed files; remove operator overrides in retest driver. |
| Task 2/3 original ambiguous J04 | Preserve original first prompt and collect actual clarification; label any new answer separately, never retrofit old card evidence. |
| Task 1 self-consistency | User edits cause existing conflict behavior; no silent rewrite of unrelated settings or other hosts. |
| Task 2 self-consistency | Add explanatory metadata, no numeric algorithm or existing-request semantics change. |
| Task 3 self-consistency | Stage scripts while tests run, but no model calls until the exact candidate's complete regression passes. |

Ruling: user authorization covers these previously identified changes and the necessary local project setup/retests; no repeated design or execution permission gate is needed. Startup failure tests must remain bounded and no-model wherever possible. Reuse the installed offline test environment; preserve the prior release as the control rather than rerunning its complete regression without a change.
