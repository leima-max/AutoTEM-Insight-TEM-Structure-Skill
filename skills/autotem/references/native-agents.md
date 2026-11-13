# Optional native-agent resources

Default execution is sequential. Domain profiles remain platform-neutral responsibility
contracts; host agent names are separate adapter identifiers.

Codex native templates are [data quality](hosts/codex/autotem_data_quality.toml),
[diffraction](hosts/codex/autotem_diffraction.toml) and
[atomic strain](hosts/codex/autotem_atomic_strain.toml). They define only name,
description and developer instructions, and inherit parent model/permissions/tools.
The installer provides these in `autotem-host/codex/native-agent-templates/`; it does
not activate `.codex/agents/`, change `[agents]`, choose models or elevate permissions.

[Current official Codex documentation](https://learn.chatgpt.com/docs/agent-configuration/subagents)
describes project-scoped standalone agents under `.codex/agents/*.toml`. This does
not prove the installed CLI/app exposes these roles. Before moving templates there,
check that host version's parser and actual native spawn mechanism, verify the
inherited Harness tools and execute an evidence-producing test. Record adapter name,
version, runtime profile ID, tool call, run/artifacts and actual outcome. A parsed TOML
file alone is `native_not_verified`. Unavailable or failed native dispatch uses
sequential packages with the same validated Harness contracts.

Only one package writes a shared run at a time. The parent owns user conversation
and human-input requests, keeps retries bounded, and collects artifact-based returns.
Do not translate software profile IDs into human A/B/C ownership. OpenClaw and
WorkBuddy use their shipped profile/Skill resources and sequential fallback; their
native configuration schema and real dispatch are not verified here.
