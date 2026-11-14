"""Run-scoped preprocessing must preserve raw data and previous run evidence."""

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autotem.core import ArrayReference, TEMDataset
from autotem.harness.contracts import HarnessError
from autotem.io.readers import read_numeric_file
from autotem.preprocessing.basic import preprocess_dataset
from autotem.preprocessing.quality import load_dataset_array
from autotem.services.data_quality import preprocess

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


@pytest.mark.skipif(os.name != "nt", reason="Win32 MAX_PATH and extended path handling")
def test_long_processed_array_reloads_through_ordinary_uri(tmp_path: Path) -> None:
    dataset = _raw_dataset(tmp_path)
    raw_hash = _hash(Path(dataset.source_path))
    folder = tmp_path / "图像"
    # Cross the Win32 threshold even in a short CI scratch directory.
    while len(str(folder / "preprocessed_image.npy").encode("utf-16-le")) // 2 <= 260:
        folder /= "intermediates_0123456789"
    output = folder / "preprocessed_image.npy"
    result = preprocess_dataset(dataset, parameters={"denoise_method": "none"}, output_dir=folder)
    assert result.image.uri == str(output) and result.image.inline is None
    reference = dataset.model_copy(update={"data": result.image})
    image = read_numeric_file(output, mmap_mode="r")
    assert isinstance(image, np.memmap) and image.dtype == np.float32 and image.shape == (96, 96)
    np.testing.assert_array_equal(load_dataset_array(reference), image)
    assert float(image.min()) == 0 and float(image.max()) == 1
    assert not any("Failed to persist" in warning for warning in result.warnings)
    assert _hash(Path(dataset.source_path)) == raw_hash


def test_harness_preserves_failed_processed_array_write(tmp_path: Path, monkeypatch) -> None:
    dataset = _raw_dataset(tmp_path)
    raw = read_numeric_file(Path(dataset.source_path))
    context = SimpleNamespace(
        output_dir=tmp_path / "op", array=lambda role: raw, model=lambda role, model_type: dataset
    )

    def deny_write(*args, **kwargs):
        raise PermissionError("read-only destination for this run")

    monkeypatch.setattr(np, "save", deny_write)
    with pytest.raises(HarnessError) as error:
        preprocess(context, {"denoise_method": "none"})
    assert error.value.code == "PREPROCESSING_FAILED"
    assert any("read-only destination for this run" in message for message in error.value.details["warnings"])


@pytest.mark.parametrize("suffix", [".npy", ".txt", ".csv"])
def test_dataset_uri_requires_a_regular_numeric_file(tmp_path: Path, suffix: str) -> None:
    folder = tmp_path / ("folder" + suffix)
    folder.mkdir()
    dataset = TEMDataset(data=ArrayReference(shape=(2, 2), dtype="float32", uri=str(folder)))
    assert load_dataset_array(dataset) is None
