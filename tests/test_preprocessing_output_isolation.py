"""Run-scoped preprocessing must preserve raw data and previous run evidence."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

from autotem.core import ArrayReference, TEMDataset
from autotem.preprocessing.basic import preprocess_dataset

pytestmark = pytest.mark.smoke


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_dataset(tmp_path: Path, *, filename: str = "raw.npy") -> TEMDataset:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_path = raw_dir / filename
    image = np.random.default_rng(17).normal(size=(96, 96)).astype(np.float32)
    np.save(raw_path, image)
    return TEMDataset(
        data=ArrayReference(shape=image.shape, dtype=str(image.dtype), uri=str(raw_path)),
        source_path=str(raw_path),
    )


def test_preprocessing_same_raw_two_runs_keep_distinct_arrays(tmp_path: Path) -> None:
    dataset = _raw_dataset(tmp_path)
    raw_path = Path(dataset.source_path)
    raw_hash = _hash(raw_path)
    first = preprocess_dataset(
        dataset,
        parameters={"denoise_method": "none"},
        output_dir=tmp_path / "run1" / "intermediates",
    )
    first_path = Path(first.image.uri)
    first_hash = _hash(first_path)
    second = preprocess_dataset(
        dataset,
        parameters={"denoise_method": "gaussian", "denoise_sigma": 2.0},
        output_dir=tmp_path / "run2" / "intermediates",
    )
    second_path = Path(second.image.uri)
    assert first_path != second_path
    assert first_hash != _hash(second_path)
    assert _hash(first_path) == first_hash
    assert _hash(raw_path) == raw_hash
    assert not raw_path.with_name("raw_preprocessed.npy").exists()
    assert list(raw_path.parent.iterdir()) == [raw_path]


def test_preprocessing_explicit_output_never_writes_raw_directory(
    tmp_path: Path, monkeypatch,
) -> None:
    dataset = _raw_dataset(tmp_path)
    raw_dir = Path(dataset.source_path).parent
    save_array = np.save

    def deny_raw_writes(path, array) -> None:
        if Path(path).parent == raw_dir:
            raise PermissionError("raw directory is read-only")
        save_array(path, array)

    monkeypatch.setattr(np, "save", deny_raw_writes)
    result = preprocess_dataset(dataset, output_dir=tmp_path / "run" / "intermediates")
    assert result.image.uri is not None
    assert Path(result.image.uri).exists()
    assert not any("Failed to persist" in warning for warning in result.warnings)


def test_preprocessing_inline_dataset_supports_explicit_output(tmp_path: Path) -> None:
    dataset = TEMDataset(data=ArrayReference(
        shape=(8, 8), dtype="float32", inline=np.arange(64).reshape(8, 8).tolist(),
    ))
    result = preprocess_dataset(dataset, output_dir=tmp_path / "run")
    assert Path(result.image.uri) == tmp_path / "run" / "preprocessed_image.npy"
    assert np.load(result.image.uri, allow_pickle=False).shape == (8, 8)


def test_preprocessing_output_cannot_alias_raw_input(tmp_path: Path) -> None:
    dataset = _raw_dataset(tmp_path, filename="preprocessed_image.npy")
    raw_path = Path(dataset.source_path)
    raw_hash = _hash(raw_path)
    with pytest.raises(ValueError, match="must not overwrite"):
        preprocess_dataset(dataset, output_dir=raw_path.parent)
    assert _hash(raw_path) == raw_hash


def test_preprocessing_legacy_direct_api_storage_remains_compatible(tmp_path: Path) -> None:
    dataset = _raw_dataset(tmp_path)
    result = preprocess_dataset(dataset)
    expected = Path(dataset.source_path).with_name("raw_preprocessed.npy")
    assert Path(result.image.uri) == expected
    assert expected.exists()
