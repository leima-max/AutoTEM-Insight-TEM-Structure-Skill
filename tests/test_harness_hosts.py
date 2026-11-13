"""Host resources, guarded installation and strict transport behavior."""

from __future__ import annotations

import io
import json
import queue
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import pytest
import yaml

from autotem.harness.cli import (
    TOOL_SCHEMAS,
    ProtocolError,
    dispatch_request,
    serve_stdio,
    strict_json,
)
from autotem.harness.hosts import _atomic_write, _read_toml, install_host, rollback_host


def _long_unicode_target(tmp_path):
    project = tmp_path / "中文 项目"
    project.mkdir()
    padding = max(1, 190 - len(str(project)) - 1)
    directory = project / ("p" * padding)
    directory.mkdir()
    basename_length = 250 - len(str(directory)) - 1
    if basename_length < 12:
        pytest.skip("The test base directory itself exceeds the bounded Windows fixture.")
    target = directory / ("config-" + "x" * (basename_length - 12) + ".toml")
    assert len(str(target)) == 250
    return target


def test_atomic_write_succeeds_near_windows_path_limit_with_unicode_and_spaces(tmp_path):
    target = _long_unicode_target(tmp_path)
    target.write_bytes(b"previous complete contents")
    _atomic_write(target, b"replacement complete contents")
    assert target.read_bytes() == b"replacement complete contents"
    assert list(target.parent.iterdir()) == [target]


def test_atomic_replace_failure_preserves_original_and_cleans_owned_temp(tmp_path, monkeypatch):
    target = _long_unicode_target(tmp_path)
    target.write_bytes(b"original user configuration")
    user_file = target.parent / ".a-user.tmp"
    user_file.write_bytes(b"unrelated user file")

    def fail_replace(_source, _target):
        raise OSError("Simulated atomic replace failure")

    monkeypatch.setattr("autotem.harness.hosts.os.replace", fail_replace)
    with pytest.raises(OSError, match="Simulated"):
        _atomic_write(target, b"new configuration")
    assert target.read_bytes() == b"original user configuration"
    assert user_file.read_bytes() == b"unrelated user file"
    assert set(target.parent.iterdir()) == {target, user_file}


class RecordingHarness:
    def __init__(self):
        self.calls = []

    def list_capabilities(self):
        self.calls.append("list_capabilities")
        print("library diagnostic belongs on stderr")
        return {"operations": [{"operation_id": "input.load", "availability": "available"}]}

    def read_artifact(self, **kwargs):
        self.calls.append(kwargs)
        return {"artifact_id": kwargs["artifact_id"], "data": []}


@pytest.mark.smoke
@pytest.mark.parametrize("host", ["codex", "openclaw", "workbuddy"])
def test_host_dry_run_is_write_free_and_reports_no_native_verification(tmp_path, host):
    project = tmp_path / "project"
    project.mkdir()
    result = install_host(project, host)
    assert result["ok"] and result["dry_run"]
    assert list(project.iterdir()) == []
    assert result["execution_mode"] == "sequential"
    assert result["native_subagents_verified"] is False
    assert result["host_llm_e2e_verified"] is False


def test_install_refuses_same_name_user_skill_without_any_mutation(tmp_path):
    user_skill = tmp_path / ".agents/skills/autotem/SKILL.md"
    user_skill.parent.mkdir(parents=True)
    user_skill.write_text("user's existing content", encoding="utf-8")
    result = install_host(tmp_path, "codex", dry_run=False)
    assert not result["ok"]
    assert user_skill.read_text(encoding="utf-8") == "user's existing content"
    assert not (tmp_path / ".autotem").exists()


def test_project_toml_merge_backup_and_rollback_preserve_unrelated_config(tmp_path):
    config = tmp_path / ".codex/config.toml"
    config.parent.mkdir()
    original = (
        b'# preserve comments\nmodel = "existing-model"\n[mcp_servers.other]\ncommand = "other"\n'
    )
    config.write_bytes(original)
    result = install_host(tmp_path, "codex", dry_run=False, with_mcp=True)
    assert result["ok"] and result["installation_id"]
    merged = config.read_text(encoding="utf-8")
    assert merged.startswith(original.decode().rstrip())
    assert "[mcp_servers.autotem]" in merged
    first_hash = config.read_bytes()
    repeat = install_host(tmp_path, "codex", dry_run=False, with_mcp=True)
    assert repeat["ok"] and repeat["installation_id"] is None
    assert repeat["next_action"] == result["next_action"]
    assert ".codex/config.toml" in result["next_action"] and "信任" in result["next_action"]
    assert config.read_bytes() == first_hash
    assert rollback_host(tmp_path, result["installation_id"])["ok"]
    assert config.read_bytes() == first_hash  # rollback default is also a dry run.
    restored = rollback_host(tmp_path, result["installation_id"], dry_run=False)
    assert restored["ok"] and config.read_bytes() == original
    assert not (tmp_path / ".agents/skills/autotem/SKILL.md").exists()


def test_codex_native_templates_are_bounded_and_not_activated(tmp_path):
    result = install_host(tmp_path, "codex", dry_run=False)
    assert result["ok"] and not result["native_subagents_verified"]
    assert ".codex/config.toml" not in result["next_action"]
    assert "autotem-host/codex/adapter.json" in result["next_action"]
    assert not (tmp_path / ".codex/config.toml").exists()
    assert not (tmp_path / ".codex/agents").exists()
    templates = list((tmp_path / "autotem-host/codex/native-agent-templates").glob("*.toml"))
    assert len(templates) == 3
    for path in templates:
        payload = _read_toml(path.read_text(encoding="utf-8"))
        assert set(payload) == {"name", "description", "developer_instructions"}
        assert "human_input_request" in payload["developer_instructions"]
        assert "not_evaluated" in payload["developer_instructions"]
        assert "run_id" in payload["developer_instructions"]


def test_different_existing_mcp_entry_is_not_overwritten(tmp_path):
    config = tmp_path / ".codex/config.toml"
    config.parent.mkdir()
    original = '[mcp_servers.autotem]\ncommand = "user-owned-server"\n'
    config.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="different autotem"):
        install_host(tmp_path, "codex", dry_run=False, with_mcp=True)
    assert config.read_text(encoding="utf-8") == original
    assert not (tmp_path / ".agents").exists()


def test_rollback_detects_user_changes_before_deleting_any_file(tmp_path):
    result = install_host(tmp_path, "openclaw", dry_run=False)
    skill = tmp_path / "skills/autotem/SKILL.md"
    skill.write_text("modified after installation", encoding="utf-8")
    adapter = tmp_path / "autotem-host/openclaw/adapter.json"
    before = adapter.read_bytes()
    rejected = rollback_host(tmp_path, result["installation_id"], dry_run=False)
    assert not rejected["ok"]
    assert skill.read_text(encoding="utf-8") == "modified after installation"
    assert adapter.read_bytes() == before


def test_install_rejects_symlink_target_outside_project(tmp_path):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    try:
        (project / ".agents").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Creating a directory symlink is unavailable on this Windows environment.")
    with pytest.raises(ValueError, match="escapes"):
        install_host(project, "codex", dry_run=False)
    assert list(outside.iterdir()) == []


def test_workbuddy_import_archive_has_valid_metadata_and_no_fake_agent_config(tmp_path):
    result = install_host(tmp_path, "workbuddy", dry_run=False)
    assert result["ok"]
    archive_path = tmp_path / "autotem-host/workbuddy/autotem-skill.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert "SKILL.md" in archive.namelist()
        text = archive.read("SKILL.md").decode()
        metadata = yaml.safe_load(text.split("---", 2)[1])
        assert all(
            metadata[key]
            for key in (
                "name",
                "description",
                "description_zh",
                "description_en",
                "version",
                "author",
            )
        )
        assert "references/profiles/atomic-strain.yaml" in archive.namelist()
    adapter = json.loads((tmp_path / "autotem-host/workbuddy/adapter.json").read_text())
    assert not adapter["host_llm_e2e_verified"]
    assert adapter["jsonl_command"][-2:] != ["mcp", "stdio"]
    assert adapter["jsonl_command"][-1] == "stdio"


@pytest.mark.smoke
def test_transport_whitelist_unknown_arguments_and_schema_boundary():
    harness = RecordingHarness()
    for request in (
        {"tool": "__class__", "arguments": {}},
        {"tool": "list_capabilities", "arguments": {"code": "print('x')"}},
        {"tool": "read_artifact", "arguments": {"artifact_id": "x", "limit": True}},
        {"tool": "read_artifact", "arguments": {"artifact_id": "x", "limit": 1001}},
        {"tool": "submit_analysis", "arguments": {"plan": {}, "execution_mode": "native_verified"}},
        {"tool": "list_capabilities", "schema_version": "0.1.0"},
    ):
        result = dispatch_request(harness, request)
        assert not result["ok"] and result["error"]["code"] == "REQUEST_INVALID"
    assert harness.calls == []
    assert len(TOOL_SCHEMAS) == 9


def test_jsonl_protocol_recovers_bad_lines_and_keeps_stdout_clean(capsys):
    harness = RecordingHarness()
    source = io.StringIO('not-json\n{"tool":"list_capabilities","request_id":"next"}\n')
    sink = io.StringIO()
    assert serve_stdio(harness, source, sink) == 0
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert len(responses) == 2 and not responses[0]["ok"]
    assert responses[1]["ok"] and responses[1]["request_id"] == "next"
    assert harness.calls == ["list_capabilities"]
    assert "library diagnostic" in capsys.readouterr().err


@pytest.mark.parametrize("text", ['{"tool":"a","tool":"b"}', '{"x": NaN}', '{"x": Infinity}'])
def test_strict_json_rejects_duplicate_keys_and_non_finite_numbers(text):
    with pytest.raises(ProtocolError):
        strict_json(text)


def test_mirrors_match_wheel_resources_and_profiles_use_registered_ids():
    package = Path(__file__).parents[1] / "autotem/harness/resources/skill"
    mirror = Path(__file__).parents[1] / "skills/autotem"
    for source in package.rglob("*"):
        if source.is_file():
            assert (mirror / source.relative_to(package)).read_bytes() == source.read_bytes()
    profiles = [
        yaml.safe_load(path.read_text())
        for path in (package / "references/profiles").glob("*.yaml")
    ]
    assert {profile["profile_id"] for profile in profiles} == {
        "data-quality",
        "diffraction",
        "atomic-strain",
    }
    assert all(profile["native_installation_verified"] is False for profile in profiles)


def test_append_and_pdf_arguments_reach_shared_service_without_new_tools():
    class AppendHarness:
        def validate_plan(self, **kwargs):
            return kwargs

        def submit_analysis(self, **kwargs):
            return kwargs

        def export_report(self, run_id, formats):
            return {"run_id": run_id, "formats": formats}

    harness = AppendHarness()
    for tool in ("validate_plan", "submit_analysis"):
        response = dispatch_request(
            harness, {"tool": tool, "arguments": {"plan": {}, "run_id": "existing"}}
        )
        assert response["ok"] and response["result"]["run_id"] == "existing"
    response = dispatch_request(
        harness,
        {"tool": "export_report", "arguments": {"run_id": "existing", "formats": ["pdf"]}},
    )
    assert response["ok"] and response["result"]["formats"] == ["pdf"]
    assert len(TOOL_SCHEMAS) == 9


def test_domain_failure_retains_repair_fields_and_redacts_credentials():
    from autotem.harness.contracts import HarnessError

    class BrokenHarness:
        def validate_plan(self, **kwargs):
            raise HarnessError(
                "MISSING_DEPENDENCY",
                "token=private-value missing roles",
                {"roles": ["raw"], "api_key": "private-value"},
            )

    result = dispatch_request(BrokenHarness(), {"tool": "validate_plan", "arguments": {"plan": {}}})
    assert result["error"]["code"] == "MISSING_DEPENDENCY"
    assert result["error"]["details"]["roles"] == ["raw"]
    assert "private-value" not in json.dumps(result)


def test_optional_real_mcp_stdio_handshake_and_tool_response(tmp_path):
    pytest.importorskip("mcp")
    # Ordinary anonymous pipes avoid sandbox-blocked asyncio Windows named pipes.
    # Do not pass or display an environment dictionary containing user credentials.
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "autotem-test", "version": "1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "list_capabilities", "arguments": {}},
        },
    ]
    process = subprocess.Popen(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "autotem.harness.cli",
            "--root",
            str(tmp_path / "runs"),
            "mcp",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    received = queue.Queue()

    def collect_stdout():
        for line in process.stdout:
            received.put(json.loads(line))

    threading.Thread(target=collect_stdout, daemon=True).start()
    responses = {}
    try:
        for message in messages:
            process.stdin.write(json.dumps(message) + "\n")
            process.stdin.flush()
            if "id" in message:
                response = received.get(timeout=30)
                assert response["id"] == message["id"]
                responses[response["id"]] = response
    except queue.Empty:
        pytest.fail("MCP stdio handshake timed out.")
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
        process.stderr.close()
    assert process.returncode == 0
    assert responses[1]["result"]["serverInfo"]["name"] == "autotem-harness"
    assert {tool["name"] for tool in responses[2]["result"]["tools"]} == set(TOOL_SCHEMAS)
    called = responses[3]["result"]
    envelope = json.loads(called["content"][0]["text"])
    assert not called.get("isError", False)
    assert envelope["schema_version"] == "1.0.0" and envelope["ok"]
    assert envelope["result"]["scientific_accuracy"] == "not_evaluated"
    assert called["structuredContent"] == envelope
