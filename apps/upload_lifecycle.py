from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

ALLOWED_IMAGE_SUFFIXES = {".dm", ".dm3", ".dm4", ".tif", ".tiff"}


def upload_signature(filename: str, content: memoryview) -> str:
    digest = hashlib.sha256(content).hexdigest()
    return f"{filename}:{len(content)}:{digest}"


class TemporaryUpload:
    def __init__(self, filename: str, content: memoryview) -> None:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {suffix or '(none)'}")

        self._directory = tempfile.TemporaryDirectory(prefix="autotem-upload-")
        self.path = Path(self._directory.name) / f"input{suffix}"
        try:
            self.path.write_bytes(content)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        self._directory.cleanup()
