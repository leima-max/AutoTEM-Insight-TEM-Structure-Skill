# Harness host contract, version 1.0.0

The local Harness root is a persistent evidence directory. Prepare the private runtime and host bridge using [installation readiness](installation.md); importing Skill text does not install scientific dependencies. Skill resources work from an installed wheel and do not rely on this checkout. Prefer registered MCP tools; the approved CLI fallback uses the adapter's absolute interpreter with `-I`.

The nine coarse tools are `list_capabilities`, `inspect_input`, `validate_plan`, `submit_analysis`, `get_run_status`, `cancel_run`, `list_artifacts`, `read_artifact`, `export_report`. Exact JSON schemas are discoverable through real MCP. A JSONL request has `{"tool":"list_capabilities","arguments":{},"request_id":"discover"}`. Its response contains `schema_version`, `request_id`, `tool`, `ok`, `result`, `error`. Unknown fields/tools, duplicate JSON keys, non-finite JSON, oversized requests and invalid pagination are rejected. This envelope wraps the actual service result; `ok` describes transport execution, not scientific acceptance.

`inspect_input` takes `path` and optional `selection` mapping. `validate_plan` takes a `plan` and optional `run_id`. `submit_analysis` takes `plan`, optional `run_id`, `idempotency_key`, `parent_run_id`, `user_request`, `asynchronous` (default true), and `execution_mode` (this adapter supports sequential). Status/cancel/list/export take `run_id`. `read_artifact` takes `artifact_id`, optional zero-based `offset` and `limit` (1–1000). Export formats are JSON/HTML/PDF; PDF is a bounded evidence summary, while JSON/HTML preserve full structured status. Use references/small previews rather than returning whole image arrays to the host.

CLI commands:

```powershell
python -m autotem.harness.cli --root outputs/harness capabilities
python -m autotem.harness.cli --root outputs/harness inspect path/to/image.tif
python -m autotem.harness.cli --root outputs/harness validate --plan plan.yaml
python -m autotem.harness.cli --root outputs/harness submit --plan plan.yaml --wait
python -m autotem.harness.cli --root outputs/harness status RUN_ID
python -m autotem.harness.cli --root outputs/harness artifacts RUN_ID
python -m autotem.harness.cli --root outputs/harness read ARTIFACT_ID --limit 100
python -m autotem.harness.cli --root outputs/harness export RUN_ID
```

Replace paths and returned IDs with actual authorized values; do not interpret RUN_ID as a predefined run. The CLI `--input` option overrides a recipe's input file. Calling the same saved plan from Python, CLI or a host reaches the same service. CLI single-response exit 0 means the call returned an envelope; the returned run may still be queued, needs-input, failed or scientifically not evaluated.

For repeated requests use `python -m autotem.harness.cli --root outputs/harness stdio`, sending one JSON object per line. For MCP install the optional `autotem-insight[mcp]` extra and use `python -m autotem.harness.cli --root outputs/harness mcp`; do not point an MCP client at JSONL `stdio`.

For sequential domain work, finish a bounded work package, then validate/submit only new steps with `run_id` (CLI `--run-id`). A completed run's committed roles seed the next package. Keep the same input, selection, execution mode and budgets; do not reuse old step IDs or alter old parameters. This lets the three profiles share one run without recomputing input. Only append to completed runs; concurrent or failed runs require recovery rather than an unsafe append.

When a run returns `needs_input`, read its `input_requests`. Show the relevant
`evidence_artifact_ids` and ask for one of its `answer_options` with the allowed unit
and measurement basis. Use the actual request ID and the user's values in the
child plan. For example, a user who reports 20 pm per pixel from the acquisition
record supplies the following `plan.user_input`.

```json
{"response":{"request_id":"<request_id from parent>","source":"user","basis":"Acquisition record","answers":{"pixel_size_nm":{"value":20,"unit":"pm/pixel"}}}}
```

Validate the complete replacement plan, then submit with the parent's
`parent_run_id` and a new `idempotency_key`. Reuse that key for retries of
the same answer. If validation rejects a value, show the named field, range
and unit. If the user has yet to reply, leave the request open. Use
`latest_response_child_run_id` on the parent to find the resulting analysis.
