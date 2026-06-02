from pathlib import Path
from unittest.mock import patch

import numpy as np

from autotem.cli import EXIT_CONFIG_ERROR, EXIT_FILE_ERROR, EXIT_RUNTIME_ERROR, main


def test_cli_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Run AutoTEM analysis workflows." in captured.out


def test_cli_analyze_generates_report(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "cli_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))
    output_dir = tmp_path / "cli_outputs"

    exit_code = main(["analyze", str(input_path), "--output", str(output_dir)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "A2 baseline workflow completed." in captured.out
    assert (output_dir / "analysis_report.json").exists()
    assert (output_dir / "workflow_used.yaml").exists()


def test_cli_returns_config_error_code_for_invalid_workflow(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "cli_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))
    workflow_path = tmp_path / "invalid.yaml"
    workflow_path.write_text(
        "\n".join(
            [
                "workflow:",
                "  name: invalid",
                "  parameters:",
                "    preprocessing:",
                "      clip_percentile_low: 99",
                "      clip_percentile_high: 1",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(["analyze", str(input_path), "--workflow", str(workflow_path)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_CONFIG_ERROR
    assert "configuration error" in captured.err


def test_cli_returns_file_error_code_for_missing_workflow(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "cli_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))
    missing = tmp_path / "missing.yaml"

    exit_code = main(["analyze", str(input_path), "--workflow", str(missing)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_FILE_ERROR
    assert "file error" in captured.err


def test_cli_returns_runtime_error_code(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "cli_input.npy"
    np.save(input_path, np.zeros((16, 16), dtype=np.float32))

    with patch("autotem.cli.TEMAnalysisSkill.run", side_effect=RuntimeError("boom")):
        exit_code = main(["analyze", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_RUNTIME_ERROR
    assert "runtime error" in captured.err


def test_cli_analyze_detect_atoms_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    class DummySkill:
        def run(self, **kwargs):
            assert kwargs["detect_atoms_step"] is True
            return {"status": "ok", "atom_count": 3}

    monkeypatch.setattr("autotem.cli.TEMAnalysisSkill", DummySkill)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))

    exit_code = main(["analyze", str(input_path), "--detect-atoms"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ok" in captured.out
    assert "Detected atoms: 3" in captured.out


def test_cli_analyze_without_detect_atoms_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    class DummySkill:
        def run(self, **kwargs):
            assert kwargs["detect_atoms_step"] is None
            return {"status": "ok"}

    monkeypatch.setattr("autotem.cli.TEMAnalysisSkill", DummySkill)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))

    exit_code = main(["analyze", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ok" in captured.out


def test_cli_analyze_formats_and_report_title(monkeypatch, tmp_path: Path, capsys) -> None:
    class DummySkill:
        def run(self, **kwargs):
            assert kwargs["output_formats_override"] == ["json", "csv", "html"]
            assert kwargs["report_title"] == "A3 Custom Title"
            assert kwargs["report_sections_override"] is None
            assert kwargs["open_step_sections"] is None
            return {"status": "ok"}

    monkeypatch.setattr("autotem.cli.TEMAnalysisSkill", DummySkill)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))

    exit_code = main(
        [
            "analyze",
            str(input_path),
            "--formats",
            "json,csv,html",
            "--report-title",
            "A3 Custom Title",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ok" in captured.out


def test_cli_analyze_report_section_controls(monkeypatch, tmp_path: Path, capsys) -> None:
    class DummySkill:
        def run(self, **kwargs):
            assert kwargs["report_sections_override"] == ["overview", "interface_checks"]
            assert kwargs["open_step_sections"] is True
            return {"status": "ok"}

    monkeypatch.setattr("autotem.cli.TEMAnalysisSkill", DummySkill)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))

    exit_code = main(
        [
            "analyze",
            str(input_path),
            "--report-sections",
            "overview,interface_checks",
            "--open-step-sections",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ok" in captured.out


def test_cli_analyze_pdf_controls(monkeypatch, tmp_path: Path, capsys) -> None:
    class DummySkill:
        def run(self, **kwargs):
            assert kwargs["pdf_sections_override"] == ["cover", "figures"]
            assert kwargs["pdf_max_figure_pages"] == 2
            assert kwargs["pdf_cover_fields"] == {"sample_id": "S01", "operator": "member-a"}
            return {"status": "ok"}

    monkeypatch.setattr("autotem.cli.TEMAnalysisSkill", DummySkill)
    input_path = tmp_path / "sample.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))

    exit_code = main(
        [
            "analyze",
            str(input_path),
            "--pdf-sections",
            "cover,figures",
            "--pdf-max-figures",
            "2",
            "--pdf-cover-field",
            "sample_id=S01",
            "--pdf-cover-field",
            "operator=member-a",
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ok" in captured.out


def test_cli_analyze_invalid_pdf_cover_field(tmp_path: Path, capsys) -> None:
    input_path = tmp_path / "sample.npy"
    np.save(input_path, np.zeros((8, 8), dtype=np.float32))
    exit_code = main(["analyze", str(input_path), "--pdf-cover-field", "bad-field"])
    captured = capsys.readouterr()
    assert exit_code == EXIT_CONFIG_ERROR
    assert "configuration error" in captured.err


def test_cli_benchmark_success(monkeypatch, tmp_path: Path, capsys) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("cases: []", encoding="utf-8")

    def _fake_runner(*, suite_path, output_dir=None):
        assert output_dir is None
        return {
            "suite_id": "test_suite",
            "acceptance": {"all_passed": True},
            "cases_executed": 1,
            "trend_report_json": "out/benchmark_suite_trend.json",
        }

    monkeypatch.setattr("autotem.cli.run_benchmark_suite_from_file", _fake_runner)
    exit_code = main(["benchmark", "--suite", str(suite_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "A4 benchmark suite completed." in captured.out
    assert "test_suite" in captured.out
    assert "benchmark_suite_trend.json" in captured.out


def test_cli_benchmark_fail_on_failed(monkeypatch, tmp_path: Path, capsys) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("cases: []", encoding="utf-8")

    def _fake_runner(*, suite_path, output_dir=None):
        return {
            "suite_id": "test_suite",
            "acceptance": {"all_passed": False},
            "cases_executed": 1,
        }

    monkeypatch.setattr("autotem.cli.run_benchmark_suite_from_file", _fake_runner)
    exit_code = main(["benchmark", "--suite", str(suite_path), "--fail-on-failed"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_RUNTIME_ERROR
    assert "A4 benchmark suite completed." in captured.out


def test_cli_benchmark_config_error(monkeypatch, tmp_path: Path, capsys) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("cases: []", encoding="utf-8")

    def _fake_runner(*, suite_path, output_dir=None):
        raise ValueError("bad suite")

    monkeypatch.setattr("autotem.cli.run_benchmark_suite_from_file", _fake_runner)
    exit_code = main(["benchmark", "--suite", str(suite_path)])
    captured = capsys.readouterr()

    assert exit_code == EXIT_CONFIG_ERROR
    assert "benchmark configuration error" in captured.err


def test_cli_preview_b1b3_launches_streamlit(monkeypatch, tmp_path: Path, capsys) -> None:
    called: dict[str, object] = {}

    class _Completed:
        returncode = 0

    def _fake_find_spec(name: str):
        return object() if name == "streamlit" else None

    def _fake_run(cmd, check, env):
        called["cmd"] = cmd
        called["check"] = check
        called["env"] = env
        return _Completed()

    monkeypatch.setattr("autotem.cli.importlib.util.find_spec", _fake_find_spec)
    monkeypatch.setattr("autotem.cli.subprocess.run", _fake_run)

    output_root = tmp_path / "preview_outputs"
    exit_code = main(
        [
            "preview-b1b3",
            "--host",
            "127.0.0.1",
            "--port",
            "8601",
            "--output-root",
            str(output_root),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Launching B1-B5 diffraction debug UI" in captured.out
    cmd = called["cmd"]
    assert isinstance(cmd, list)
    assert "streamlit" in cmd
    assert "b1_b3_preview_ui.py" in " ".join(str(item) for item in cmd)
    env = called["env"]
    assert isinstance(env, dict)
    assert env.get("AUTOTEM_UI_OUTPUT_ROOT") == str(output_root)


def test_cli_preview_b1b3_missing_streamlit_returns_config_error(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr("autotem.cli.importlib.util.find_spec", lambda _: None)
    exit_code = main(["preview-b1b3"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_CONFIG_ERROR
    assert "streamlit is not installed" in captured.err


def test_cli_preview_b1b3_nonzero_streamlit_exit_returns_runtime_error(
    monkeypatch,
    capsys,
) -> None:
    class _Completed:
        returncode = 1

    monkeypatch.setattr("autotem.cli.importlib.util.find_spec", lambda _: object())
    monkeypatch.setattr("autotem.cli.subprocess.run", lambda *_, **__: _Completed())
    exit_code = main(["preview-b1b3"])
    captured = capsys.readouterr()

    assert "Launching B1-B5 diffraction debug UI" in captured.out
    assert exit_code == EXIT_RUNTIME_ERROR


def test_cli_configure_mp_api_show_guidance(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "mp.json"
    exit_code = main(["configure-mp-api", "--show", "--config", str(config_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Materials Project API key is not configured" in captured.out
    assert str(config_path) in captured.out


def test_cli_configure_mp_api_writes_key_from_stdin(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    config_path = tmp_path / "mp.json"

    class _FakeStdin:
        def read(self) -> str:
            return "secret-key\n"

    monkeypatch.setattr("autotem.cli.sys.stdin", _FakeStdin())
    exit_code = main(
        ["configure-mp-api", "--api-key-stdin", "--config", str(config_path)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "stored" in captured.out
    assert "secret-key" in config_path.read_text(encoding="utf-8")


def test_cli_configure_mp_api_check_missing_returns_config_error(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    monkeypatch.delenv("MP_API_KEY", raising=False)
    exit_code = main(
        ["configure-mp-api", "--check", "--config", str(tmp_path / "missing.json")]
    )
    captured = capsys.readouterr()

    assert exit_code == EXIT_CONFIG_ERROR
    assert "Materials Project API key is not configured" in captured.err


def test_cli_validate_diffsims_success(monkeypatch, capsys) -> None:
    called: dict[str, object] = {}

    def fake_validate(**kwargs):
        called.update(kwargs)
        return {
            "status": "passed",
            "ok": True,
            "dependencies": {"diffsims": "0.7.0"},
            "materials": [{"material": "si", "ok": True}],
            "warnings": [],
        }

    monkeypatch.setattr("autotem.cli.run_diffsims_smoke_validation", fake_validate)
    exit_code = main(
        [
            "validate-diffsims",
            "--materials",
            "si",
            "--voltage-kv",
            "300",
            "--reciprocal-radius",
            "1.2",
            "--max-simulated-spots",
            "64",
            "--max-measured-spots",
            "4",
            "--min-matched-spots",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "diffsims validation passed" in captured.out
    assert called["materials"] == ["si"]
    assert called["voltage_kv"] == 300.0
    assert called["reciprocal_radius_inv_angstrom"] == 1.2
    assert called["max_simulated_spots"] == 64
    assert called["max_measured_spots"] == 4
    assert called["min_matched_spots"] == 2


def test_cli_validate_diffsims_failure_returns_runtime_error(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        "autotem.cli.run_diffsims_smoke_validation",
        lambda **_: {"status": "failed", "ok": False, "warnings": ["missing diffsims"]},
    )

    exit_code = main(["validate-diffsims", "--materials", "si"])
    captured = capsys.readouterr()

    assert exit_code == EXIT_RUNTIME_ERROR
    assert "diffsims validation failed" in captured.out
