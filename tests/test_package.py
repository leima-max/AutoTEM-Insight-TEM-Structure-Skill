from autotem import TEMAnalysisSkill, __version__


def test_package_imports() -> None:
    assert __version__ == "0.1.0"
    assert TEMAnalysisSkill is not None


def test_workflow_scaffold_run() -> None:
    result = TEMAnalysisSkill().run("sample.tif")
    assert result["status"] == "AutoTEM workflow scaffold is ready."
    assert result["input_path"] == "sample.tif"
