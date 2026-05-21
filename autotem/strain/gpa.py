"""GPA/strain implementation removed from main branch."""

from __future__ import annotations

from typing import Any


_REMOVED_MESSAGE = "Requested GPA/strain task content has been removed from main branch."


def run_gpa(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError(_REMOVED_MESSAGE)
