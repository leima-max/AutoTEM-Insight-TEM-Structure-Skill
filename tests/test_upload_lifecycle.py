from pathlib import Path

import pytest

from apps.upload_lifecycle import TemporaryUpload, upload_signature


def test_upload_signature_distinguishes_same_name_and_size() -> None:
    first = upload_signature("sample.tif", memoryview(b"abcd"))
    second = upload_signature("sample.tif", memoryview(b"wxyz"))
    assert first != second


def test_temporary_upload_is_removed_when_closed() -> None:
    upload = TemporaryUpload("sample.tif", memoryview(b"image"))
    path = upload.path
    assert path.read_bytes() == b"image"
    upload.close()
    assert not path.exists()
    assert not Path(path.parent).exists()


def test_temporary_upload_rejects_unknown_suffix() -> None:
    with pytest.raises(ValueError, match="Unsupported image format"):
        TemporaryUpload("sample.txt", memoryview(b"image"))
