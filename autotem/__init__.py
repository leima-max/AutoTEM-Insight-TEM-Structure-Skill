"""AutoTEM-Insight public package interface."""

__all__ = ["TEMAnalysisSkill"]
__version__ = "0.1.0"


def __getattr__(name):
    # Installation diagnostics must work before optional/scientific dependencies
    # are installed. Preserve the public facade, loading it only when requested.
    if name == "TEMAnalysisSkill":
        from autotem.workflows.pipeline import TEMAnalysisSkill

        globals()[name] = TEMAnalysisSkill
        return TEMAnalysisSkill
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
