# Task 3: Acceptance closure
Work only in the isolated source worktree. Own tools/host_acceptance/verify.py, new tests/test_host_reference_inspection.py and tests/test_host_rejected_reference_reply.py. Coordinate changes to shared existing tests.

J09-reference: require every run created by this capture's successful inspect_input/submit_analysis calls (already loaded by run_evidence), including early/later inspection, to have actual reference mode. Stable error reference_journey_created_snapshot. Do not require inspection, mix unrelated project runs, or prohibit scientific raw arrays. Preserve snapshot variant.

J08-invalid_reply: retain observed actual rejection and additionally require a persisted rejected response tied to the matched parent/request, no child/job and still-open request. Use real store representation. Read-only validation rejection alone cannot prove repaired durable closure. Do not require unsafe raw text in DB or reject unrelated historic rows.

TDD with real Harness and recorded-call fixtures around check_case/run_evidence: snapshot preflight with reference parent/child fails; reference preflight and no-inspection pass; snapshot compatibility; later snapshot fails; unrelated runs excluded; actual stored mode governs. Validation-only rejection fails durable check; submit rejection persists and passes.

Replay old real J09-reference into a separate maintained diagnostic if feasible, never alter old passed certificate. No commits/nested agents/model calls/old evidence edits. Report task-3-report.md in repair output root with exact commands and red/green.
