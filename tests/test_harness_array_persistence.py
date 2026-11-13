"""The streamed NPY path must preserve the previous file format and publication contracts."""

import hashlib
import io
import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from autotem.harness.contracts import HarnessError
from autotem.harness.store import Store

pytestmark = pytest.mark.smoke


@pytest.fixture
def run(tmp_path):
    store = Store(tmp_path)
    run_id = uuid.uuid4().hex
    with store.connect() as db:
        db.execute(
            "INSERT INTO runs(run_id,request_hash,request,status,created,updated) "
            "VALUES(?,'x','{}','running',0,0)",
            (run_id,),
        )
    return store, run_id


@pytest.mark.parametrize(
    "value",
    [
        np.arange(9, dtype=">i4").reshape(3, 3)[::-1, ::2],
        np.asfortranarray(np.arange(513 * 513, dtype=">f8").reshape(513, 513)),
        np.array("NaT", dtype="datetime64[ns]"),
        np.empty((0, 3), dtype="<c16"),
        np.empty((2, 3), dtype="V0"),
        np.broadcast_to(np.array([float("nan"), -0.0]), (2, 2)),
    ],
    ids=[
        "strided-endian",
        "fortran-multichunk",
        "scalar-time",
        "empty-shape",
        "zero-itemsize",
        "broadcast-nan",
    ],
)
def test_npy_bytes_hash_and_layout_match_previous_writer(run, value):
    store, run_id = run
    previous = io.BytesIO()
    np.save(previous, value, allow_pickle=False)
    result = store.write_array(run_id, "step", "sample", value, {"unit": "pixel"})
    data = previous.getvalue()
    assert Path(result["path"]).read_bytes() == data
    assert result["sha256"] == hashlib.sha256(data).hexdigest() and result["size"] == len(data)
    assert result["metadata"] == {
        "unit": "pixel",
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "array_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
        "preview_is_full_array": False,
    }
    restored = store.load_array(result["artifact_id"])
    assert restored.tobytes(order="C") == value.tobytes(order="C")


@pytest.mark.parametrize("layout", ["strided", "fortran", "fortran-aligned"])
def test_struct_padding_preserves_complete_backing_records(run, layout):
    store, run_id = run
    dtype = np.dtype(
        {"names": ["α", "v"], "formats": ["i1", ">i4"], "offsets": [0, 4], "itemsize": 8},
        align=layout == "fortran-aligned",
    )
    backing = bytes(range(64))
    records = np.frombuffer(backing, dtype=dtype)
    if layout == "strided":
        value = records[6::-2]
        in_c_order = (6, 4, 2, 0)
        on_disk = in_c_order
    else:
        value = records.reshape(4, 2).T
        in_c_order = (0, 2, 4, 6, 1, 3, 5, 7)
        on_disk = range(8)

    def original_bytes(order):
        return b"".join(backing[8 * i : 8 * (i + 1)] for i in order)

    result = store.write_array(run_id, "step", "padded", value)
    raw = Path(result["path"]).read_bytes()
    assert raw.endswith(original_bytes(on_disk))
    assert result["sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        result["metadata"]["array_sha256"] == hashlib.sha256(original_bytes(in_c_order)).hexdigest()
    )
    assert result["metadata"]["dtype"] == str(value.dtype)
    restored = store.load_array(result["artifact_id"])
    assert restored.shape == value.shape and restored.dtype == value.dtype
    assert restored.dtype.isalignedstruct == value.dtype.isalignedstruct
    assert restored.view("V8").tobytes(order="C") == original_bytes(in_c_order)
    for field in dtype.names:
        np.testing.assert_array_equal(restored[field], value[field])


@pytest.mark.parametrize(
    "outer, inner, width", [(False, True, 1), (True, False, 1), (True, True, 1), (False, True, 4)]
)
def test_nested_struct_alignment_uses_npy_offsets_for_new_and_old_artifacts(
    run, outer, inner, width
):
    store, run_id = run
    cell = np.dtype([("tag", f"u{width}"), ("count", ">i4")], align=inner)
    dtype = np.dtype([(("pairs", "cells"), cell, (2,)), ("last", "u1")], align=outer)
    backing = bytes(range(2 * dtype.itemsize))
    value = np.frombuffer(backing, dtype=dtype)[::-1]
    result = store.write_array(run_id, "step", "records", value)
    expected = backing[dtype.itemsize :] + backing[: dtype.itemsize]
    assert result["metadata"]["array_sha256"] == hashlib.sha256(expected).hexdigest()
    for legacy in (False, True):
        if legacy:
            metadata = dict(result["metadata"])
            del metadata["npy_aligned_records"]
            with store.connect() as db:
                db.execute(
                    "UPDATE artifacts SET metadata=? WHERE artifact_id=?",
                    (json.dumps(metadata), result["artifact_id"]),
                )
        restored = store.load_array(result["artifact_id"])
        assert str(restored.dtype) == str(dtype) and restored.dtype == dtype
        assert restored.dtype.isalignedstruct == outer
        # With no padding, historical NPY + dtype text cannot distinguish this nested flag.
        recoverable_alignment = inner and not (legacy and width == 4)
        assert restored.dtype["pairs"].subdtype[0].isalignedstruct == recoverable_alignment
        assert restored.view(f"V{dtype.itemsize}").tobytes() == expected


def test_changing_only_padding_aborts_publication(run, monkeypatch):
    store, run_id = run
    dtype = np.dtype({"names": ["value"], "formats": ["u1"], "offsets": [0], "itemsize": 4})
    value = np.frombuffer(bytes(range(32)), dtype=dtype)[::-1]
    save = np.save

    def change_padding(stream, array, **kwargs):
        save(stream, array, **kwargs)
        stream.seek(-1, 1)
        stream.write(b"\xff")

    monkeypatch.setattr(np, "save", change_padding)
    with pytest.raises(HarnessError) as error:
        store.write_array(run_id, "step", "padded", value)
    assert error.value.code == "ARRAY_VALIDATION_FAILED"
    assert store.artifacts(run_id) == []
    assert not list((store.run_dir(run_id) / "artifacts").iterdir())


def test_struct_metadata_must_match_actual_npy_layout(run):
    store, run_id = run
    dtype = np.dtype([("value", "u1"), ("count", ">i4")], align=True)
    value = np.zeros(4, dtype=dtype)
    result = store.write_array(run_id, "step", "records", value)
    metadata = dict(result["metadata"])
    metadata["dtype"] = str(np.dtype([("value", "u1"), ("changed", ">i4")], align=True))
    with store.connect() as db:
        db.execute(
            "UPDATE artifacts SET metadata=? WHERE artifact_id=?",
            (json.dumps(metadata), result["artifact_id"]),
        )
    with pytest.raises(HarnessError) as error:
        store.load_array(result["artifact_id"])
    assert error.value.code == "ARRAY_VALIDATION_FAILED"


@pytest.mark.parametrize("damage", ["truncated", "trailing", "shape", "order", "value"])
def test_rejects_bad_staged_file_and_removes_partial_publication(run, monkeypatch, damage):
    store, run_id = run
    save = np.save

    def damaged(stream, value, **kwargs):
        if damage == "shape":
            value = value.reshape(3, 4)
        if damage == "order":
            value = np.asfortranarray(value)
        if damage == "value":
            value = value.copy()
            value[0, 0] ^= 1
        save(stream, value, **kwargs)
        if damage == "truncated":
            stream.truncate(stream.tell() - 1)
        if damage == "trailing":
            stream.write(b"extra")

    monkeypatch.setattr(np, "save", damaged)
    with pytest.raises((HarnessError, ValueError, OSError)):
        store.write_array(run_id, "step", "bad", np.arange(12).reshape(4, 3))
    assert store.artifacts(run_id) == []
    assert not list((store.run_dir(run_id) / "artifacts").iterdir())
