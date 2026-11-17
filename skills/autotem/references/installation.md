# Installation and readiness

An imported Skill supplies instructions. Its Python runtime, host connector,
permission and scientific capabilities have separate readiness states.

When the host exposes the nine AutoTEM MCP tools, call `list_capabilities` first.
The host launches the installed local server; routine analyses do not need a new
Python shell command for each step. MCP still requires the host's permission to
launch and use that server. A local handshake is not proof of a real host journey.

Project-local Codex MCP installations set `required = true` and
`startup_timeout_sec = 180` in `.codex/config.toml`. Codex waits for AutoTEM
MCP initialization during session startup, with a 180-second startup timeout.
If required startup fails, retain the Codex diagnostic as a host connection
failure. After resolving it, reconnect and call `list_capabilities` to verify
the tool catalog before running an analysis.

When using an approved CLI fallback, read `autotem-host/<host>/adapter.json`
from the installation's project directory. Use its explicit `python_executable`
and `jsonl_command`, preserving `-I` and the absolute Harness root. Do not infer
the interpreter from the current directory or a `python` alias. If requested or
an environment error occurs, the adapter's `readiness_command` performs bounded
runtime, local transport and host checks. It does not start a gateway or change
host policy. Host LLM use/native delegation remain separately unverified.

For the Windows x64 offline candidate, open `AutoTEM-Setup.exe` beside its
`release-manifest.json` and select an existing project folder and host. The
installer verifies the files, deploys its private CPython, checks the pinned
environment and prepares the project connector. At `host_setup_required`,
reload the selected host, authorize the project connector and try
`list_capabilities`. An old MCP session reporting `HOST_RECONNECT_REQUIRED`
needs a reconnect. Keep the setup bundle to repair or roll back the installation.
The rollback action restores the previous project configuration and retains run data.
A managed upgrade can migrate the previous 60-second Codex configuration when
it matches the installation receipt. A conflicting project configuration
requires review through the installer's conflict-resolution instructions.

For the developer wheel route, use the stdlib-only installer:

```text
python -m autotem.bootstrap --project PROJECT --wheel AUTOTEM_WHEEL --host codex --wheelhouse COMPLETE_WHEELHOUSE --apply
```

This creates an owned environment inside the selected project, installs the
wheel and its dependencies, validates the runtime and then installs host assets.
Without `--apply`, it only previews. Offline is the default; a complete compatible
wheelhouse is required. Online setup needs explicit `--allow-network` instead of
`--wheelhouse`; an approved index may be supplied with `--index-url`. It only
changes the install child process, preserving the user's environment/config.
Codex's default installer route includes the optional MCP dependency/config;
`--without-mcp` selects the CLI-only route. Other hosts receive reviewed proposals
and must complete their supported connector/import flow. This installer needs a
Python 3.10+ bootstrap interpreter and uses the fixed-path developer environment.

Resolve installation failures before scientific analysis:

- Missing/incomplete environment or offline wheels: use the approved installer
  and complete distribution. Do not silently download packages during analysis.
- A stopped OpenClaw gateway: explain the gateway state and its standard startup
  action; obtain host service authorization when it is required. Do not restart
  a global service automatically or disable its authentication.
- Host command/server launch denied: request authorization for the specific
  installed AutoTEM entry through the host. Do not retry aliases, nested agents,
  alternate shells or policy overrides to evade the refusal.
- Runtime and local transport pass but tools absent from the host: complete the
  host's supported connector registration/reload; record discovery as unverified
  until the host actually exposes and calls the tools.

Return one actionable explanation and retain the structured error. Do not label
installation failure as missing scientific calibration or keep guessing parameters.
