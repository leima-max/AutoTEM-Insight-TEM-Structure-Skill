"""Phase-matching helpers removed from main branch."""

from __future__ import annotations

from typing import Any

_REMOVED_MESSAGE = "Requested phase-matching task content has been removed from main branch."


def __getattr__(name: str) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)
