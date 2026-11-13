import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from PIL import Image

from tools.host_acceptance import prepare_fieldkit as kit

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def release_sources(tmp_path, request):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    matrix = kit.HERE / "matrix.json"
    source = bundle / "sources.zip"
    original = {name: (ROOT / name).read_bytes() for name in kit.INPUT_SOURCES}
    if getattr(request, "param", None) == "gpa":
        original[kit.GPA_CARD_SOURCE] = (ROOT / kit.GPA_CARD_SOURCE).read_bytes()
    with ZipFile(source, "w") as archive:
        for name, data in original.items():
            archive.writestr(name, data)
        index_file = "source-inventory.json"
        archive.writestr(
            index_file, kit.as_json({"files": {n: kit.sha(d) for n, d in original.items()}})
        )
    wheel = {
        "path": "wheelhouse/" + kit.CORE + "/autotem_insight-test-py3-none-any.whl",
        "sha256": "c" * 64,
    }
    manifest = {
        "schema_version": "1.0.0",
        "release_id": "synthetic-fieldkit-test",
        "source": {
            "archive": source.name,
            "archive_sha256": kit.file_sha(source),
            "inventory_member": index_file,
        },
        "artifacts": [wheel],
        "profiles": {kit.CORE: {"lock_sha256": "d" * 64}},
    }
    manifest_path = bundle / "release-manifest.json"
    manifest_path.write_bytes(kit.as_json(manifest))
    qualification = {
        "schema_version": "1.0.0",
        "execution_layer": "release_qualification",
        "release_id": manifest["release_id"],
        "profile": kit.CORE,
        "release_manifest_sha256": kit.file_sha(manifest_path),
        "source_archive_sha256": kit.file_sha(source),
        "wheel_sha256": wheel["sha256"],
        "lock_sha256": "d" * 64,
        "certificates": {
            "simulation_edition": {"verdict": "passed", "release_manifest_sha256": "e" * 64}
        },
        "host_evidence": {
            "matrix_sha256": kit.file_sha(matrix),
            "simulation_edition": {"manifest_sha256": "e" * 64},
        },
    }
    index = tmp_path / "release-index.json"
    index.write_bytes(kit.as_json(qualification))
    return bundle, index, matrix


@pytest.mark.parametrize("release_sources", ["gpa"], indirect=True)
def test_optional_gpa_card_is_frozen_without_expanding_required_cards(tmp_path, release_sources):
    bundle, index, matrix = release_sources
    destination = tmp_path / "现场验收"
    spec = kit.prepare(destination, bundle, index, matrix)
    assert len(spec["cards"]) == 31
    card = destination / kit.GPA_CARD
    assert card.read_bytes() == (ROOT / kit.GPA_CARD_SOURCE).read_bytes()
    assert kit.GPA_CARD in (destination / "README.md").read_text("utf-8")
    with card.open("ab") as stream:
        stream.write(b"changed")
    spec["files_sha256"][kit.GPA_CARD] = kit.file_sha(card)
    (destination / "index.json").write_bytes(kit.as_json(spec))
    with pytest.raises(ValueError, match="GPA follow-up card differs from frozen source"):
        kit.verify(destination, bundle, index, matrix)


def test_portable_operator_cards_make_and_verify_only_synthetic_inputs(tmp_path, release_sources):
    bundle, index, matrix = release_sources
    root = tmp_path / "例子" / "现场验收"
    args = [
        "prepare",
        "--release",
        str(bundle),
        "--release-index",
        str(index),
        "--matrix",
        str(matrix),
        "--out",
        str(root),
    ]
    assert kit.main(args) == 0
    spec = kit.load(root / "index.json")
    assert spec["kind"] == "synthetic_operator_materials"
    assert len(spec["cards"]) == 31 and len(spec["inputs"]) == 13
    previous_index = (root / "index.json").read_bytes()
    assert kit.main(args) == 0 and (root / "index.json").read_bytes() == previous_index
    assert all("case_verdict" not in row for row in spec["cards"])
    assert spec["release"]["profiles"] == {
        kit.CORE: kit.file_sha(bundle / "release-manifest.json"),
        kit.SIM: "e" * 64,
    }
    templates = {r["case_id"] + "-" + str(r["variant"]): r for r in spec["cards"]}
    assert templates["J15-executed"]["profiles"] == [kit.SIM]
    assert set(templates["J15-missing_dependency"]["profiles"]) == {kit.CORE, kit.SIM}
    for row in spec["cards"]:
        text = (root / row["path"]).read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert all("现场验收/" + spec["inputs"][key]["path"] in text for key in row["inputs"])
    inputs = spec["inputs"]

    def read(key):
        return root / inputs[key]["path"]

    assert inputs["j01"]["shape"] == [64, 80] and inputs["j01"]["dtype"] == "float64"
    assert set(inputs["datasets"]["members"]) == {"first", "second"}
    stack = np.load(read("stack"), allow_pickle=False)
    with np.load(read("datasets"), allow_pickle=False) as data:
        for frame, name in enumerate(("first", "second")):
            np.testing.assert_array_equal(data[name], stack[frame])
            assert (
                kit.sha(data[name].tobytes()) == inputs["datasets"]["members"][name]["array_sha256"]
            )
    with Image.open(read("rgb")) as im:
        im.seek(1)
        expected = np.arange(702, dtype=np.uint16).reshape(2, 9, 13, 3).astype(np.uint8)
        np.testing.assert_array_equal(np.array(im)[:, :, 2], expected[1, :, :, 2])
    assert np.count_nonzero(np.load(read("flat"), allow_pickle=False)) == 0
    with pytest.raises(ValueError):
        np.load(read("invalid"), allow_pickle=False)
    sim = kit.load(read("simulation_structure"))
    [phase] = sim["diffraction.index"]["candidate_phase_params"]
    assert phase["a"] == phase["b"] == phase["c"] == 0.4 and phase["space_group_number"] == 221
    assert phase["atom_sites"] == [{"element": "Si", "abc": [0, 0, 0], "occupancy": 1.0}]
    assert sim["calibration.resolve"] == {"pixel_size_nm": 0.05}
    assert sim["simulation.diffraction"]["diffsims_timeout_s"] == 105
    assert sim["budgets"]["simulation_seconds"] == 120
    moved = tmp_path / "另一个项目" / root.name
    shutil.copytree(root, moved)
    assert kit.main(["verify", str(moved)]) == 0
    assert (
        kit.main(
            [
                "verify",
                str(moved),
                "--release",
                str(bundle),
                "--release-index",
                str(index),
                "--matrix",
                str(matrix),
            ]
        )
        == 0
    )


@pytest.mark.parametrize("fault", ["receipt", "matrix", "source"])
def test_preparation_refuses_different_release_inputs(tmp_path, release_sources, fault):
    release, index, matrix = release_sources
    out = tmp_path / "现场验收"
    if fault == "receipt":
        data = kit.load(index)
        data["certificates"]["simulation_edition"]["release_manifest_sha256"] = "a" * 64
        index.write_bytes(kit.as_json(data))
    elif fault == "matrix":
        changed = tmp_path / "changed-matrix.json"
        changed.write_bytes(matrix.read_bytes() + b"\n")
        matrix = changed
    else:
        with (release / "sources.zip").open("ab") as append:
            append.write(b"changed")
    with pytest.raises(ValueError):
        kit.prepare(out, release, index, matrix)
    assert not out.exists()


@pytest.mark.parametrize("fault", ["edited_card", "escaped_member", "different_existing_target"])
def test_verify_refuses_changed_or_unrelated_directory(tmp_path, release_sources, fault):
    release, index, matrix = release_sources
    out = tmp_path / "现场验收"
    if fault == "different_existing_target":
        out.mkdir()
        existing = out / "operator-notes.txt"
        existing.write_bytes(b"existing")
        with pytest.raises((OSError, ValueError)):
            kit.prepare(out, release, index, matrix)
        assert existing.read_bytes() == b"existing"
        return
    spec = kit.prepare(out, release, index, matrix)
    if fault == "edited_card":
        card = spec["cards"][0]["path"]
        (out / card).write_bytes(b"changed")
        # Even synchronized file hashes cannot change the text derived from the card definition.
        spec["files_sha256"][card] = kit.file_sha(out / card)
    else:
        (tmp_path / "elsewhere.npy").write_bytes(b"outside")
        spec["inputs"]["j01"]["path"] = "../elsewhere.npy"
    (out / "index.json").write_bytes(kit.as_json(spec))
    with pytest.raises(ValueError):
        kit.verify(out)
    if fault == "edited_card":
        with pytest.raises(ValueError):
            kit.prepare(out, release, index, matrix)
        assert (out / card).read_bytes() == b"changed"


@pytest.mark.parametrize("fault", ["duplicate", "missing", "profile", "format", "native"])
def test_matrix_cards_fail_closed_on_invalid_scope_or_prompts(fault):
    catalogs = [
        kit.load(kit.HERE / "field" / (name + ".zh-CN.json")) for name in ("science", "operations")
    ]
    sample = catalogs[0][0]
    if fault == "duplicate":
        catalogs[1].append(sample)
    elif fault == "missing":
        catalogs[0].pop(0)
    elif fault == "profile":
        sample["profiles"] = ["simulation-diffsims"]
    elif fault == "native":
        sample["condition"] = "native_only"
    else:
        sample["turns"][0]["text"] = "{j01.__class__}"
    with pytest.raises(ValueError):
        kit.render_rows(kit.load(kit.HERE / "matrix.json"), catalogs, "现场验收")
