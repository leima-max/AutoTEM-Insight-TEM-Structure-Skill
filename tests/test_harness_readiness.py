"""Installation diagnostics must distinguish local execution from host acceptance."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autotem.harness import readiness


def runtime_payload(**overrides):
    payload = {
        "python_version": [3, 10, 9],
        "installation_type": "non_editable",
        "modules": {name: True for name in (*readiness.CORE_MODULES, "autotem")},
    }
    payload.update(overrides)
    return payload


def install_stubs(
    monkeypatch,
    *,
    runtime=None,
    capability=None,
    gateway=None,
    health=None,
    failure=None,
    host_present=True,
):
    calls = []
    runtime = runtime_payload() if runtime is None else runtime
    capability = (
        capability
        if capability is not None
        else {
            "ok": True,
            "result": {"operations": [{"operation_id": "input.load"}]},
        }
    )
    gateway = (
        gateway
        if gateway is not None
        else {
            "service": {"loaded": True, "runtime": {"status": "running"}},
        }
    )
    health = health if health is not None else {"ok": True}
    monkeypatch.setattr(
        readiness.shutil, "which", lambda name: f"/fake/{name}" if host_present else None
    )

    def run(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["shell"] is False
        assert 0 < kwargs["timeout"] <= 60
        if failure is not None:
            response = failure(arguments)
            if response is not None:
                return response
        if "-c" in arguments:
            output = runtime
        elif "capabilities" in arguments:
            output = capability
        elif "--version" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "codex-cli 0.143.0", "")
        elif "check" in arguments:
            return subprocess.CompletedProcess(arguments, 0, "No broken requirements found.", "")
        elif "status" in arguments:
            output = gateway
        elif "health" in arguments:
            output = health
        else:
            pytest.fail(f"Unexpected process: {arguments}")
        text = output if isinstance(output, str) else json.dumps(output)
        return subprocess.CompletedProcess(arguments, 0, text, "")

    monkeypatch.setattr(readiness.subprocess, "run", run)
    return calls


def test_ready_local_probes_do_not_claim_host_permissions_or_analysis(monkeypatch, tmp_path):
    monkeypatch.delenv("PIP_NO_INDEX", raising=False)
    calls = install_stubs(monkeypatch)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["ok"] and result["status"] == "ready_for_host_verification"
    assert result["runtime"]["capabilities"]["operation_count"] == 1
    assert result["runtime"]["installation_type"] == "non_editable"
    assert result["runtime"]["local_mcp"]["code"] == "MCP_EXTRA_NOT_INSTALLED"
    assert result["host_policy"]["status"] == "not_evaluated"
    assert result["host_llm_e2e"] == result["native_subagents"] == "not_evaluated"
    assert result["host_mcp_connection"] == "not_evaluated"
    assert all("-I" in arguments for arguments in calls if "--version" not in arguments)
    assert list(tmp_path.iterdir()) == []
    scratch = next(
        arguments[arguments.index("--root") + 1]
        for arguments in calls
        if "capabilities" in arguments
    )
    assert Path(scratch).parent == tmp_path.resolve()
    assert not Path(scratch).exists()


def test_denied_scratch_creation_fails_after_one_attempt_without_retry(monkeypatch, tmp_path):
    calls = install_stubs(monkeypatch)
    attempts = []
    original = Path.mkdir

    def denied(path, *args, **kwargs):
        if path.name.startswith(".autotem-readiness-"):
            attempts.append(path)
            raise PermissionError("policy denied token=secret")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", denied)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "SCRATCH_UNAVAILABLE"
    assert not result["ok"]
    assert len(attempts) == 1
    assert not any("capabilities" in arguments for arguments in calls)
    assert list(tmp_path.iterdir()) == []
    assert "secret" not in json.dumps(result)
    actions = {row["code"] for row in result["next_actions"]}
    assert "SCRATCH_UNAVAILABLE" in actions
    assert "INSTALL_COMPLETE_RUNTIME" not in actions
    assert "OFFLINE_BUNDLE_REQUIRED" not in actions


def test_capability_failure_cleans_owned_project_scratch(monkeypatch, tmp_path):
    existing = tmp_path / "user-file.txt"
    existing.write_text("preserve this")
    calls = install_stubs(monkeypatch, capability="not-json")
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "CAPABILITY_CONTRACT_INVALID"
    scratch = next(
        arguments[arguments.index("--root") + 1]
        for arguments in calls
        if "capabilities" in arguments
    )
    assert Path(scratch).parent == tmp_path.resolve()
    assert not Path(scratch).exists()
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_text() == "preserve this"


def test_scratch_collision_does_not_delete_unowned_directory(monkeypatch, tmp_path):
    install_stubs(monkeypatch)
    monkeypatch.setattr(readiness.uuid, "uuid4", lambda: type("UUID", (), {"hex": "fixed"})())
    existing = tmp_path / ".autotem-readiness-fixed"
    existing.mkdir()
    marker = existing / "user-file.txt"
    marker.write_text("preserve this")
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "SCRATCH_UNAVAILABLE"
    assert marker.read_text() == "preserve this"


def test_cleanup_error_is_structured_and_does_not_delete_other_project_files(monkeypatch, tmp_path):
    install_stubs(monkeypatch)
    seen = []

    def denied(path):
        seen.append(path)
        raise PermissionError("token=secret")

    monkeypatch.setattr(readiness.shutil, "rmtree", denied)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "SCRATCH_CLEANUP_FAILED"
    assert not result["ok"]
    assert len(seen) == 1 and seen[0].parent == tmp_path.resolve()
    assert "secret" not in json.dumps(result)


def test_cleanup_rejects_windows_reparse_points_even_with_unchanged_identity(monkeypatch, tmp_path):
    install_stubs(monkeypatch)
    original = Path.lstat
    seen = []
    deleted = []

    def reparse_at_cleanup(path, *args, **kwargs):
        observed = original(path, *args, **kwargs)
        if path.name.startswith(".autotem-readiness-"):
            seen.append(path)
            if len(seen) > 1:
                return SimpleNamespace(
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino,
                    st_mode=observed.st_mode,
                    st_file_attributes=0x400,
                )
        return observed

    monkeypatch.setattr(Path, "lstat", reparse_at_cleanup)
    monkeypatch.setattr(readiness.shutil, "rmtree", lambda path: deleted.append(path))
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "SCRATCH_CLEANUP_FAILED"
    assert not deleted


def test_cleanup_rejects_redirect_within_same_project(monkeypatch, tmp_path):
    install_stubs(monkeypatch)
    protected = tmp_path / "existing-run"
    protected.mkdir()
    marker = protected / "evidence.txt"
    marker.write_text("keep this evidence")
    original = Path.resolve
    seen = []
    deleted = []

    def redirect_at_cleanup(path, *args, **kwargs):
        if path.name.startswith(".autotem-readiness-"):
            seen.append(path)
            if len(seen) > 1:
                return protected
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", redirect_at_cleanup)
    monkeypatch.setattr(readiness.shutil, "rmtree", lambda path: deleted.append(path))
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "SCRATCH_CLEANUP_FAILED"
    assert not deleted
    assert marker.read_text() == "keep this evidence"


def test_missing_dependencies_and_offline_policy_are_actionable_and_preserved(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("PIP_NO_INDEX", "true")
    payload = runtime_payload()
    payload["modules"]["PIL"] = False
    calls = install_stubs(monkeypatch, runtime=payload)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert not result["ok"]
    assert result["runtime"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert result["runtime"]["missing_modules"] == ["PIL"]
    assert "capabilities" not in [item for arguments in calls for item in arguments]
    assert result["package_index"] == {"pip_no_index_active": True, "configuration_changed": False}
    assert readiness.os.environ["PIP_NO_INDEX"] == "true"
    assert "OFFLINE_BUNDLE_REQUIRED" in {row["code"] for row in result["next_actions"]}


def test_healthy_offline_runtime_does_not_require_another_install(monkeypatch, tmp_path):
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    install_stubs(monkeypatch)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["ok"]
    assert result["package_index"]["pip_no_index_active"] is True
    assert "OFFLINE_BUNDLE_REQUIRED" not in {row["code"] for row in result["next_actions"]}


def test_package_not_installed_is_not_a_dependency_success(monkeypatch, tmp_path):
    payload = runtime_payload()
    payload["modules"]["autotem"] = False
    install_stubs(monkeypatch, runtime=payload)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "AUTOTEM_NOT_INSTALLED"
    assert result["runtime"]["capabilities"]["status"] == "not_evaluated"


@pytest.mark.parametrize("bad", ["not-json", "[]", '{"modules": {}}', '{"x":NaN}'])
def test_runtime_invalid_json_is_structured_failure(monkeypatch, tmp_path, bad):
    install_stubs(monkeypatch, runtime=bad)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert not result["ok"]
    assert result["runtime"]["code"] == "RUNTIME_PROBE_INVALID_JSON"


@pytest.mark.parametrize(
    "bad",
    [
        "not-json",
        {"ok": True, "result": {"operations": []}},
        {"ok": False, "result": {"operations": [{}]}},
    ],
)
def test_capability_invalid_json_or_empty_contract_is_not_success(monkeypatch, tmp_path, bad):
    install_stubs(monkeypatch, capability=bad)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "CAPABILITY_CONTRACT_INVALID"
    assert not result["ok"]


def test_broken_import_is_distinct_and_does_not_expose_logs(monkeypatch, tmp_path):
    def failure(arguments):
        if "capabilities" in arguments:
            return subprocess.CompletedProcess(
                arguments, 1, "", "DLL load failed token=secret-value"
            )

    install_stubs(monkeypatch, failure=failure)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "RUNTIME_IMPORT_OR_EXECUTION_FAILED"
    assert "secret-value" not in json.dumps(result)


def test_missing_import_module_classification_is_bounded(monkeypatch, tmp_path):
    def failure(arguments):
        if "capabilities" in arguments:
            return subprocess.CompletedProcess(
                arguments, 1, "", "No module named 'numpy' token=secret-value"
            )

    install_stubs(monkeypatch, failure=failure)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "DEPENDENCY_UNAVAILABLE"
    assert "secret-value" not in json.dumps(result)


def test_present_but_broken_core_dependency_is_not_available(monkeypatch, tmp_path):
    install_stubs(monkeypatch, runtime=runtime_payload(broken_modules=["numpy"]))
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "DEPENDENCY_BROKEN"
    assert result["runtime"]["broken_modules"] == ["numpy"]
    assert not result["ok"]


def test_incompatible_installed_dependencies_fail_before_capabilities(monkeypatch, tmp_path):
    def failure(arguments):
        if "check" in arguments:
            return subprocess.CompletedProcess(arguments, 1, "token=secret-value conflict", "")

    calls = install_stubs(monkeypatch, failure=failure)
    result = readiness.inspect_readiness(tmp_path, "codex")
    assert result["runtime"]["code"] == "DEPENDENCY_CONFLICT_OR_CHECK_FAILED"
    assert not result["ok"]
    assert "secret-value" not in json.dumps(result)
    assert not any("capabilities" in arguments for arguments in calls)


def test_timeout_does_not_leak_partial_output(monkeypatch, tmp_path):
    def failure(arguments):
        if "-c" in arguments:
            raise subprocess.TimeoutExpired(arguments, 1, output="token=secret-value")

    install_stubs(monkeypatch, failure=failure)
    result = readiness.inspect_readiness(tmp_path, "codex", timeout=1)
    assert result["runtime"]["code"] == "PROBE_TIMEOUT"
    assert "secret-value" not in json.dumps(result)


def test_gateway_stopped_is_reported_without_automatic_service_start(monkeypatch, tmp_path):
    calls = install_stubs(
        monkeypatch,
        gateway={
            "service": {"loaded": True, "runtime": {"status": "stopped", "detail": "token=secret"}},
            "config": {"token": "secret", "api_key": "hidden"},
        },
        health={"ok": False, "error": {"type": "gateway_transport_error"}},
    )
    result = readiness.inspect_readiness(tmp_path, "openclaw")
    assert result["gateway"] == {
        "status": "blocked",
        "code": "GATEWAY_STOPPED",
        "service_registered": True,
        "runtime_state": "stopped",
    }
    assert not result["ok"]
    assert all("start" not in arguments and "install" not in arguments for arguments in calls)
    assert any("health" in arguments for arguments in calls)
    assert "secret" not in json.dumps(result)


def test_gateway_running_requires_successful_health(monkeypatch, tmp_path):
    install_stubs(
        monkeypatch,
        health={
            "ok": False,
            "error": {
                "type": "authentication_error",
                "token": "secret-value",
                "message": "Use token=secret-value",
            },
        },
    )
    result = readiness.inspect_readiness(tmp_path, "openclaw")
    assert result["gateway"]["code"] == "GATEWAY_AUTH_REQUIRED"
    assert "secret-value" not in json.dumps(result)
    assert not result["ok"]


def test_gateway_success_projects_no_credentials(monkeypatch, tmp_path):
    install_stubs(monkeypatch, health={"ok": True, "credentials": "secret-value"})
    result = readiness.inspect_readiness(tmp_path, "openclaw")
    assert result["gateway"]["code"] == "GATEWAY_HEALTHY"
    assert result["ok"]
    assert "secret-value" not in json.dumps(result)
    assert result["host_llm_e2e"] == "not_evaluated"


def test_foreground_gateway_health_wins_over_stopped_registered_service(monkeypatch, tmp_path):
    calls = install_stubs(
        monkeypatch,
        gateway={"service": {"loaded": True, "runtime": {"status": "stopped"}}},
        health={"ok": True},
    )
    result = readiness.inspect_readiness(tmp_path, "openclaw")
    assert result["gateway"] == {
        "status": "ready",
        "code": "GATEWAY_HEALTHY",
        "service_registered": True,
        "runtime_state": "stopped",
    }
    assert result["ok"]
    assert all("start" not in arguments for arguments in calls)


def test_gateway_auth_error_takes_precedence_over_stopped_service(monkeypatch, tmp_path):
    install_stubs(
        monkeypatch,
        gateway={"service": {"loaded": True, "runtime": {"status": "stopped"}}},
        health={"ok": False, "error": {"type": "pairing_required", "message": "token=secret"}},
    )
    result = readiness.inspect_readiness(tmp_path, "openclaw")
    assert result["gateway"]["code"] == "GATEWAY_AUTH_REQUIRED"
    assert not result["ok"]
    assert "secret" not in json.dumps(result)


def test_valid_gateway_health_wins_over_unavailable_status_probe(monkeypatch, tmp_path):
    install_stubs(monkeypatch, gateway="not-json", health={"ok": True})
    result = readiness.inspect_readiness(tmp_path, "openclaw")
    assert result["gateway"]["code"] == "GATEWAY_HEALTHY"
    assert result["gateway"]["runtime_state"] == "unknown"
    assert result["gateway"]["service_registered"] is None


@pytest.mark.parametrize("bad", ["not-json", "[]"])
def test_gateway_status_invalid_json_is_not_success(monkeypatch, tmp_path, bad):
    install_stubs(monkeypatch, gateway=bad, health={"ok": False})
    result = readiness.inspect_readiness(tmp_path, "openclaw")
    assert result["gateway"]["code"] == "GATEWAY_STATUS_INVALID_JSON"
    assert not result["ok"]


def test_malformed_gateway_types_do_not_crash_or_leak(monkeypatch, tmp_path):
    install_stubs(
        monkeypatch,
        gateway={"service": {"runtime": {"status": {"token": "secret"}}}},
        health={"ok": False, "error": {"type": {"token": "secret"}}},
    )
    result = readiness.inspect_readiness(tmp_path, "openclaw")
    assert result["gateway"]["code"] == "GATEWAY_HEALTH_FAILED"
    assert "secret" not in json.dumps(result)


def test_workbuddy_cli_absence_preserves_gui_unknown(monkeypatch, tmp_path):
    install_stubs(monkeypatch, host_present=False)
    result = readiness.inspect_readiness(tmp_path, "workbuddy")
    assert result["host_cli"]["status"] == "gui_not_evaluated"
    assert result["status"] == "ready_for_host_verification"
    assert result["host_mcp_connection"] == "not_evaluated"


def test_selected_venv_interpreter_is_not_resolved_to_base(monkeypatch, tmp_path):
    base = tmp_path / "base-python"
    base.write_bytes(b"test executable")
    link = tmp_path / "venv-python"
    try:
        link.symlink_to(base)
    except OSError:
        pytest.skip("Creating symlinks requires permissions on this host")
    calls = install_stubs(monkeypatch)
    readiness.inspect_readiness(tmp_path, "codex", python_executable=link)
    assert calls[0][0] == str(link.absolute())
    assert calls[0][0] != str(base)


@pytest.mark.parametrize("timeout", [0, -1, 61, float("nan"), True, "15"])
def test_invalid_probe_inputs_are_rejected_before_processes(tmp_path, timeout):
    with pytest.raises(ValueError):
        readiness.inspect_readiness(tmp_path, "codex", timeout=timeout)


def test_unknown_host_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        readiness.inspect_readiness(tmp_path, "shell")


class MCPProcess:
    def __init__(self, responses):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("\n".join(json.dumps(response) for response in responses) + "\n")
        self.killed = False

    def wait(self, timeout):
        return 0

    def kill(self):
        self.killed = True


def test_real_mcp_protocol_probe_requires_initialize_and_complete_tool_contract(monkeypatch):
    names = [
        "list_capabilities",
        "inspect_input",
        "validate_plan",
        "submit_analysis",
        "get_run_status",
        "cancel_run",
        "list_artifacts",
        "read_artifact",
        "export_report",
    ]
    process = MCPProcess(
        [
            {"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "autotem-harness"}}},
            {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": name} for name in names]}},
            mcp_call_response(),
        ]
    )
    calls = []

    def popen(arguments, **kwargs):
        calls.append(arguments)
        assert kwargs["shell"] is False
        return process

    monkeypatch.setattr(readiness.subprocess, "Popen", popen)
    result = readiness._mcp_probe("/fake/python", "/scratch", 1)
    assert result == {
        "status": "ready",
        "code": "LOCAL_MCP_TOOL_CALL_OK",
        "tool_count": 9,
        "operation_count": 1,
    }
    assert calls[0][1:4] == ["-I", "-m", "autotem.harness.cli"]
    assert process.stdin.closed and process.stdout.closed


def test_mcp_malformed_tool_contract_does_not_crash(monkeypatch):
    process = MCPProcess(
        [
            {"id": 1, "result": {"serverInfo": {}}},
            {"id": 2, "result": {"tools": [{"name": {"token": "secret"}}]}},
        ]
    )
    monkeypatch.setattr(readiness.subprocess, "Popen", lambda *args, **kwargs: process)
    result = readiness._mcp_probe("/fake/python", "/scratch", 1)
    assert result["code"] == "MCP_TOOL_CONTRACT_FAILED"
    assert "secret" not in json.dumps(result)


def test_mcp_server_error_is_projected_without_raw_error_details(monkeypatch):
    process = MCPProcess([{"id": 1, "error": {"message": "token=secret"}}])
    monkeypatch.setattr(readiness.subprocess, "Popen", lambda *args, **kwargs: process)
    result = readiness._mcp_probe("/fake/python", "/scratch", 1)
    assert result["code"] == "MCP_PROTOCOL_FAILED"
    assert "secret" not in json.dumps(result)


def test_mcp_timeout_closes_owned_process_and_streams(monkeypatch):
    process = MCPProcess([])

    class SilentQueue:
        def __init__(self, **kwargs):
            pass

        def put_nowait(self, item):
            pass

        def get(self, timeout):
            raise readiness.queue.Empty

    monkeypatch.setattr(readiness.queue, "Queue", SilentQueue)
    monkeypatch.setattr(readiness.subprocess, "Popen", lambda *args, **kwargs: process)
    result = readiness._mcp_probe("/fake/python", "/scratch", 1)
    assert result["code"] == "MCP_PROBE_TIMEOUT"
    assert process.stdin.closed and process.stdout.closed


def mcp_call_response(**overrides):
    envelope = {
        "schema_version": "1.0.0",
        "ok": True,
        "result": {"operations": [{"operation_id": "input.load"}]},
    }
    envelope.update(overrides)
    return {
        "id": 3,
        "result": {
            "structuredContent": envelope,
            "content": [{"type": "text", "text": json.dumps(envelope)}],
        },
    }


def mcp_initial_responses():
    names = [
        "list_capabilities",
        "inspect_input",
        "validate_plan",
        "submit_analysis",
        "get_run_status",
        "cancel_run",
        "list_artifacts",
        "read_artifact",
        "export_report",
    ]
    return [
        {"id": 1, "result": {"serverInfo": {"name": "autotem-harness"}}},
        {"id": 2, "result": {"tools": [{"name": name} for name in names]}},
    ]


@pytest.mark.parametrize(
    "response",
    [
        {"id": 3, "error": {"message": "token=secret"}},
        {"id": 3, "result": {"isError": True, "content": [{"text": "token=secret"}]}},
        {"id": 4, "result": {}},
        mcp_call_response(ok=False),
    ],
)
def test_mcp_registered_tool_execution_failures_block_transport(monkeypatch, response):
    process = MCPProcess([*mcp_initial_responses(), response])
    monkeypatch.setattr(readiness.subprocess, "Popen", lambda *args, **kwargs: process)
    result = readiness._mcp_probe("/fake/python", "/scratch", 1)
    assert result["status"] == "blocked"
    assert result["code"] == "MCP_TOOL_CALL_FAILED"
    assert "secret" not in json.dumps(result)


@pytest.mark.parametrize(
    "response",
    [
        mcp_call_response(result={"operations": []}),
        mcp_call_response(result={"operations": [{}]}),
        mcp_call_response(schema_version="0.1.0"),
        {"id": 3, "result": {"content": [{"type": "text", "text": "not-json"}]}},
        {"id": 3, "result": {"structuredContent": ["token=secret"]}},
        {
            "id": 3,
            "result": {
                "structuredContent": {"ok": True},
                "content": [{"type": "text", "text": "token=secret"}],
            },
        },
    ],
)
def test_mcp_capability_envelope_contract_failure_blocks_transport(monkeypatch, response):
    process = MCPProcess([*mcp_initial_responses(), response])
    monkeypatch.setattr(readiness.subprocess, "Popen", lambda *args, **kwargs: process)
    result = readiness._mcp_probe("/fake/python", "/scratch", 1)
    assert result["status"] == "blocked"
    assert result["code"] == "MCP_TOOL_CALL_CONTRACT_FAILED"
    assert "secret" not in json.dumps(result)


def test_mcp_text_only_capability_envelope_is_accepted(monkeypatch):
    response = mcp_call_response()
    del response["result"]["structuredContent"]
    process = MCPProcess([*mcp_initial_responses(), response])
    monkeypatch.setattr(readiness.subprocess, "Popen", lambda *args, **kwargs: process)
    result = readiness._mcp_probe("/fake/python", "/scratch", 1)
    assert result["code"] == "LOCAL_MCP_TOOL_CALL_OK"


def test_mcp_tool_call_timeout_is_not_successful_discovery(monkeypatch):
    process = MCPProcess([])

    class TimeoutAfterDiscovery:
        def __init__(self, **kwargs):
            self.responses = iter(mcp_initial_responses())

        def put_nowait(self, item):
            pass

        def get(self, timeout):
            try:
                return json.dumps(next(self.responses))
            except StopIteration:
                raise readiness.queue.Empty from None

    monkeypatch.setattr(readiness.queue, "Queue", TimeoutAfterDiscovery)
    monkeypatch.setattr(readiness.subprocess, "Popen", lambda *args, **kwargs: process)
    result = readiness._mcp_probe("/fake/python", "/scratch", 1)
    assert result["code"] == "MCP_PROBE_TIMEOUT"
    assert result["status"] == "blocked"
    assert process.stdin.closed and process.stdout.closed
