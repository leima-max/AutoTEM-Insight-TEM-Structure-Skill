from pathlib import Path

from autotem.cli import main


def test_cli_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Run AutoTEM analysis workflows." in captured.out


def test_cli_analyze_detect_atoms_flag(monkeypatch, tmp_path: Path, capsys) -> None:
    class DummySkill:
        def run(self, **kwargs):
            assert kwargs["detect_atoms_step"] is True
            return {"status": "ok", "atom_count": 3}

    monkeypatch.setattr("autotem.cli.TEMAnalysisSkill", DummySkill)
    input_path = tmp_path / "sample.tif"
    input_path.write_bytes(b"fake")

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
    input_path = tmp_path / "sample.tif"
    input_path.write_bytes(b"fake")

    exit_code = main(["analyze", str(input_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "ok" in captured.out
