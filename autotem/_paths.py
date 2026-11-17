"""Filesystem spellings for Windows I/O; keep user-facing and persisted paths ordinary."""

import os
from pathlib import Path


def io_path(path: Path) -> Path:
    """Add the Win32 extended path prefix only when the absolute path needs it."""
    if os.name != "nt":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith(("\\\\?\\", "\\\\.\\")):
        return path
    # MAX_PATH includes the trailing NUL and counts UTF-16 code units.
    if len(absolute.encode("utf-16-le", "surrogatepass")) // 2 < 260:
        return path
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)
