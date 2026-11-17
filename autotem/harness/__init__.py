"""Headless execution and evidence interface for domain tools."""

__all__ = ["Harness", "HarnessError", "HarnessPlan", "NeedsInput", "OperationDefinition"]


def __getattr__(name):
    if name in {"HarnessError", "HarnessPlan", "NeedsInput", "OperationDefinition"}:
        from . import contracts

        value = getattr(contracts, name)
        globals()[name] = value
        return value
    if name == "Harness":
        from .service import Harness

        return Harness
    raise AttributeError(name)
