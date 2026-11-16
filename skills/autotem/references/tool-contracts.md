# Harness host contract, version 1.0.0

The local Harness root is a persistent evidence directory. Prepare the private runtime and host bridge using [installation readiness](installation.md); importing Skill text does not install scientific dependencies. Skill resources work from an installed wheel and do not rely on this checkout. Prefer registered MCP tools; the approved CLI fallback uses the adapter's absolute interpreter with `-I`.

The nine coarse tools are `list_capabilities`, `inspect_input`, `validate_plan`, `submit_analysis`, `get_run_status`, `cancel_run`, `list_artifacts`, `read_artifact`, `export_report`. Exact JSON schemas are discoverable through real MCP. A JSONL request has `{"tool":"list_capabilities","arguments":{},"request_id":"discover"}`. Its response contains `schema_version`, `request_id`, `tool`, `ok`, `result`, `error`. Unknown fields/tools, duplicate JSON keys, non-finite JSON, oversized requests and invalid pagination are rejected. This envelope wraps the actual service result; `ok` describes transport execution, not scientific acceptance.

`inspect_input` takes `path`, optional `selection` mapping and `input_mode` (`snapshot` by default, or `reference`). The Python signature is `inspect_input(path, *, input_mode="snapshot", **selection)`. For a request not to copy the source input, choose `reference` from inspection onward and retain it in subsequent plans; registered scientific arrays, tables and previews remain normal outputs. JSON uses `{"path":"<authorized file>","selection":{},"input_mode":"reference"}`; the CLI uses `inspect <path> --input-mode reference`. `validate_plan` takes a `plan` and optional `run_id`. `submit_analysis` takes `plan`, optional `run_id`, `idempotency_key`, `parent_run_id`, `user_request`, `asynchronous` (default true), and `execution_mode` (this adapter supports sequential). Status/cancel/list/export take `run_id`. `read_artifact` takes `artifact_id`, optional zero-based `offset` and `limit` (1–1000). Export formats are JSON/HTML/PDF; PDF is a bounded evidence summary, while JSON/HTML preserve full structured status. Use references/small previews rather than returning whole image arrays to the host.

The compact catalog uses `{"view":"summary"}` for the operation index and `{"view":"parameters","operation_id":"diffraction.match"}` for one operation's exact parameter schema, roles, budget, result level and scientific limits. An empty arguments object still returns the full catalog, including input and output schemas. A host can request that full detail for one operation with `{"operation_id":"diffraction.match"}`.

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

For requested reference-dependent strain, keep the operation in the complete plan when
its reference is missing. In particular, raw refinement followed by `strain.atom` without
reference values produces numbered evidence and a persisted `needs_input` before strain
computation. A completed refinement package may instead append the requested strain step
with a new step ID and `run_id`. Do not add strain to coordinate-only requests. Do not end
with a reference question until the actual run has supplied its `input_requests`.

When a run returns `needs_input`, read its `input_requests`. Give the full parent
`run_id`, request ID and relevant `evidence_artifact_ids`. End the current turn by asking
for one of its `answer_options` with the allowed unit and measurement basis. For atom
strain choose either a half-open `[row_start,row_stop,col_start,col_stop]` ROI or zero-based
indices from that request's numbered evidence, mutually exclusive, selecting at least six
atoms spanning a two-dimensional lattice. Direct `strain.atom` operation parameters use
selected raw-array local pixels. In contrast, reference replies must follow the actual
request's `answer_schema.reference_region.coordinates`: for `frame=source_2d`, supply the
original-image bounds, preserving `origin_yx` for the service to convert. Do not pre-subtract
that origin or submit already-local bounds as source coordinates. For example, origin
`[8,12]` and a reply `[16,40,20,60]` are adopted as local `[8,32,8,48]` once. Never invent or
trim the reference.

The next user message supplies the input. Preserve its original `raw_reply`, actual request
ID, source, basis and untrimmed values in the complete replacement plan. Convert explicit
units or inclusive endpoints according to the documented contract while preserving the
original text; never clip out-of-bounds values. For example, a user who reports 20 pm per
pixel from the acquisition record supplies this `plan.user_input`:

```json
{"response":{"request_id":"<request_id from parent>","source":"user","basis":"Acquisition record","raw_reply":"20 pm per pixel from the acquisition record","answers":{"pixel_size_nm":{"value":20,"unit":"pm/pixel"}}}}
```

`validate_plan` is a read-only preview, including for invalid replies; it does not persist
response history. A reply to an active scientific input request must then reach
`submit_analysis` with the same response, the parent's `parent_run_id` and a new stable
`idempotency_key`, even if preview rejects the supplied value. Reuse the key for retries of
the same answer. The service validates again and persists either acceptance or a bounded
safe rejection before allocating a child/worker. This is the reply transaction, not permission
to submit unrelated invalid workflows. Keep credentials outside plans; do not bypass the
existing response length and sensitive-text policy. Native tool records retain the original
submitted reply; the database intentionally stores only safe typed values for rejections.

Read the parent's persisted `input_responses`. For a rejection, report the original error
code, full parent `run_id`, field, range and unit requiring correction, then end with the
corrected input request. The parent remains pending and no child is created. For acceptance,
use `latest_response_child_run_id` to follow the linked analysis. Preserve the parent's
scientific plan and artifacts; never treat a conversational answer alone as a completed
response transaction.
