from autotem.cli import main


def test_cli_help(capsys) -> None:
    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Run AutoTEM analysis workflows." in captured.out
