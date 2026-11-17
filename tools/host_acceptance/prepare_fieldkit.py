"""Build portable Chinese operator cards and synthetic inputs, or verify a relocated field kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from string import Formatter
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import numpy as np
from PIL import Image, ImageSequence

from autotem.acceptance import cases

HERE = Path(__file__).resolve().parent
CORE = "traditional-core-mcp"
SIM = "traditional-simulation"
PROFILES = {CORE: CORE, SIM: SIM, "simulation-diffsims": SIM}
CONDITIONS = {
    "ready": "可在已连接的项目操作",
    "controlled": "须具备指定试验条件",
    "native_only": "须有宿主原生委派能力",
}
ANALYTIC = {
    "j01": "J01",
    "j02": "J02",
    "j03": "J03",
    "j04": "J04",
    "saed": "J05-SAED",
    "nbed": "J05-NBED",
    "j06": "J06",
}
FILES = {
    **{
        key: f"{case} 合成 {cases.CASE_MODALITIES[case]}"
        + (".tif" if key == "j02" else ".tiff" if key == "j06" else ".npy")
        for key, case in ANALYTIC.items()
    },
    "stack": "stack.npy",
    "rgb": "multichannel.tif",
    "datasets": "datasets.npz",
    "invalid": "invalid.npy",
    "flat": "flat.npy",
    "simulation_structure": "simulation-structure.json",
}
CASE_SOURCE = "autotem/acceptance/cases.py"
MATRIX_SOURCE = "tools/host_acceptance/matrix.json"
GPA_CARD_SOURCE = "tools/host_acceptance/field/J08-GPA.zh-CN.md"
GPA_CARD = "cards/J08-GPA.md"
SELECTION_SOURCE = "tools/release/contract_probe.py"
SIM_OPTIONS = "tools/acceptance/operation_variants.json"
SIM_SOURCE = "tools/acceptance/verify_variants.py"
INPUT_SOURCES = (CASE_SOURCE, MATRIX_SOURCE, SELECTION_SOURCE, SIM_OPTIONS, SIM_SOURCE)
GENERATED = "synthetic_operator_materials"


def require(value, reason):
    if not value:
        raise ValueError(reason)


def unique(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), "Duplicate JSON property")
    return result


def load(path):
    return json.loads(path.read_bytes().decode("utf-8-sig"), object_pairs_hook=unique)


def as_json(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stream_sha(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def file_sha(path):
    with path.open("rb") as stream:
        return stream_sha(stream)


def relative(value):
    name = PurePosixPath(value)
    require(
        isinstance(value, str)
        and name.parts
        and name.as_posix() == value
        and not name.is_absolute()
        and not PureWindowsPath(value).drive
        and not any(c in value for c in '\\:*?"<>|\x00')
        and ".." not in name.parts
        and all(p.rstrip(". ") == p and not PureWindowsPath(p).is_reserved() for p in name.parts),
        f"Unsafe relative path: {value!r}",
    )
    return name


def linked(path):
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def safe_file(root, name):
    path = root.joinpath(*relative(name).parts)
    for part in (path, *path.parents):
        require(not linked(part), f"Linked kit or release member: {part}")
        if part == root:
            break
    require(
        path.is_file() and path.resolve().is_relative_to(root.resolve()),
        f"Missing or escaped member: {name}",
    )
    return path


def check_sha(digest):
    require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "Invalid SHA-256")
    return digest


def release_identity(manifest_path, index_path, matrix_path):
    manifest, index = load(manifest_path), load(index_path)
    matrix_digest = file_sha(matrix_path)
    release_digest = file_sha(manifest_path)
    source = manifest["source"]
    [wheel] = [
        a
        for a in manifest["artifacts"]
        if a["path"].startswith("wheelhouse/" + CORE + "/autotem_insight-")
        and a["path"].endswith(".whl")
    ]
    require(
        manifest["schema_version"] == index["schema_version"] == "1.0.0"
        and index["execution_layer"] == "release_qualification"
        and index["profile"] == CORE
        and set(manifest["profiles"]) == {CORE}
        and index["release_id"] == manifest["release_id"]
        and index["release_manifest_sha256"] == release_digest
        and index["source_archive_sha256"] == check_sha(source["archive_sha256"])
        and index["wheel_sha256"] == check_sha(wheel["sha256"])
        and index["lock_sha256"] == check_sha(manifest["profiles"][CORE]["lock_sha256"])
        and index["host_evidence"]["matrix_sha256"] == matrix_digest,
        "Qualification index, core release or frozen matrix differ",
    )
    simulation = index["certificates"]["simulation_edition"]
    companion = check_sha(simulation["release_manifest_sha256"])
    require(
        companion != release_digest
        and simulation["verdict"] == "passed"
        and index["host_evidence"]["simulation_edition"]["manifest_sha256"] == companion,
        "Unbound simulation edition in the qualification index",
    )
    return {
        "release_id": index["release_id"],
        "release_manifest_sha256": release_digest,
        "release_index_sha256": file_sha(index_path),
        "matrix_sha256": matrix_digest,
        "source_archive_sha256": source["archive_sha256"],
        "wheel_sha256": wheel["sha256"],
        "profiles": {CORE: release_digest, SIM: companion},
    }


def archived_sources(release, matrix):
    manifest = load(safe_file(release, "release-manifest.json"))
    source = manifest["source"]
    path = safe_file(release, source["archive"])
    require(file_sha(path) == source["archive_sha256"], "Changed release source archive")
    with ZipFile(path) as archive:
        inventory = json.loads(
            archive.read(str(relative(source["inventory_member"]))), object_pairs_hook=unique
        )["files"]
        hashes = {}
        members = list(INPUT_SOURCES)
        if GPA_CARD_SOURCE in inventory:
            members.append(GPA_CARD_SOURCE)
        for name in members:
            hashes[name] = sha(archive.read(name))
            require(hashes[name] == inventory[name], f"Changed frozen source: {name}")
        options = json.loads(archive.read(SIM_OPTIONS))["cases"]["cubic-simulation"]
        card = archive.read(GPA_CARD_SOURCE) if GPA_CARD_SOURCE in hashes else None
    require(file_sha(matrix) == hashes[MATRIX_SOURCE], "Matrix differs from the release source")
    require(
        file_sha(Path(cases.__file__)) == hashes[CASE_SOURCE],
        "Input generator differs from the one shipped by the release",
    )
    return hashes, options, card


def matrix_variants(matrix):
    require(
        matrix["schema_version"] == "1.0.0"
        and set(matrix["hosts"]) == {"codex", "openclaw", "workbuddy"}
        and len(matrix["hosts"]) == 3,
        "Invalid host matrix",
    )
    variant_keys = []
    for case, variants in matrix["cases"].items():
        require(
            re.fullmatch(r"J(?:0[1-9]|1[0-5])", case)
            and isinstance(variants, list)
            and len(set(variants)) == len(variants)
            and all(re.fullmatch(r"[a-z][a-z0-9_]*", v) for v in variants),
            "Invalid matrix case or variant",
        )
        variant_keys.extend((case, variant) for variant in variants or [None])
    require(len(matrix["cases"]) == 15, "Incomplete J01–J15 matrix")
    return variant_keys


def render_rows(matrix, catalogs, project_dir):
    expected = matrix_variants(matrix)
    rows = {}
    for catalog in catalogs:
        require(isinstance(catalog, list), "Expected an array of operator cards")
        for item in catalog:
            key = item["case_id"], item["variant"]
            require(key in expected and key not in rows, f"Duplicate or unknown card: {key}")
            condition, inputs = item["condition"], item["inputs"]
            require(
                condition in CONDITIONS
                and isinstance(inputs, list)
                and len(inputs) == len(set(inputs))
                and set(inputs) <= FILES.keys(),
                f"Invalid input or condition for {key}",
            )
            profiles = [PROFILES[p] for p in item["profiles"]]
            required = {SIM} if key[0] == "J15" else {CORE}
            if key == ("J15", "missing_dependency"):
                required = {CORE, SIM}
            require(
                set(profiles) == required and len(profiles) == len(required),
                f"Incorrect release profile scope: {key}",
            )
            require(
                (condition == "native_only") == (key == ("J14", "native")),
                f"Incorrect native delegation context: {key}",
            )
            require(
                isinstance(item["setup"], str)
                and item["setup"].strip()
                and isinstance(item["observe"], list)
                and item["observe"]
                and all(isinstance(s, str) and s.strip() for s in item["observe"]),
                f"Missing setup or observations: {key}",
            )
            turns = item["turns"]
            require(
                isinstance(turns, list)
                and turns
                and turns[0]["when"] == "start"
                and all(t["when"] != "start" for t in turns[1:]),
                f"Invalid turn order: {key}",
            )
            values = {k: f"{project_dir}/inputs/{FILES[k]}" for k in inputs}
            used, rendered = set(), []
            for turn in turns:
                require(
                    all(isinstance(turn[k], str) and turn[k].strip() for k in ("when", "text")),
                    f"Invalid prompt: {key}",
                )
                for _, field, form, conversion in Formatter().parse(turn["text"]):
                    if field is not None:
                        require(
                            field in values and not form and not conversion,
                            f"Invalid input placeholder: {field}",
                        )
                        used.add(field)
                rendered.append({"when": turn["when"], "text": turn["text"].format_map(values)})
            require(used == set(inputs), f"Unused or undeclared card input: {key}")
            rows[key] = {**item, "profiles": profiles, "turns": rendered}
    require(set(expected) == rows.keys(), "Operator cards do not cover the frozen matrix")
    return [rows[key] for key in expected]


def simulation_payload(options):
    edge = options["cell_edge_nm"]
    settings = options["runtime_settings"]
    return {
        "modality": options["modality"],
        "calibration.resolve": {"pixel_size_nm": options["pixel_size_nm"]},
        "diffraction.index": {
            "candidate_phase_params": [
                {
                    "name": "synthetic-cubic",
                    "a": edge,
                    "b": edge,
                    "c": edge,
                    "formula": "Si",
                    "space_group": "Pm-3m",
                    "space_group_number": 221,
                    "atom_sites": [{"element": "Si", "abc": [0, 0, 0], "occupancy": 1.0}],
                }
            ]
        },
        "simulation.diffraction": {
            "pattern_shape": [64, 64],
            "calibration": 0.025,
            "max_candidates": 1,
            "reciprocal_radius_inv_angstrom": 1.5,
            "max_simulated_spots": 96,
            "diffsims_timeout_s": settings["diffsims_timeout_s"],
        },
        "diffraction.match": {"min_matched_spots": 3, "d_tolerance_rel": 0.06},
        "budgets": {"simulation_seconds": settings["simulation_seconds"]},
    }


def sample_properties(values):
    return {
        "shape": list(values.shape),
        "dtype": values.dtype.name,
        "array_sha256": sha(values.tobytes(order="C")),
    }


def make_inputs(folder, options):
    details = {}
    for key, name in ANALYTIC.items():
        path, description = cases.write_fixture(name, folder)
        require(path.name == FILES[key], f"Unexpected analytic input name: {key}")
        details[key] = {
            "source": CASE_SOURCE,
            "case_id": name,
            "formula": description["formula"],
            "calibration": description["calibration"],
        }
    stack = np.arange(2 * 9 * 13, dtype=np.int16).reshape(2, 9, 13)
    color = np.arange(2 * 9 * 13 * 3, dtype=np.uint16).reshape(2, 9, 13, 3).astype(np.uint8)
    np.save(folder / FILES["stack"], stack, allow_pickle=False)
    pages = [Image.fromarray(frame) for frame in color]
    pages[0].save(folder / FILES["rgb"], save_all=True, append_images=pages[1:])
    np.savez(folder / FILES["datasets"], first=stack[0], second=stack[1])
    np.save(folder / FILES["flat"], np.zeros((64, 80), dtype=np.float32), allow_pickle=False)
    # A truncated v1.0 NumPy header exercises the reader without any serialized Python object.
    (folder / FILES["invalid"]).write_bytes(b"\x93NUMPY\x01\x00\x40\x00")
    (folder / FILES["simulation_structure"]).write_bytes(as_json(simulation_payload(options)))
    details.update(
        {
            "stack": {"source": SELECTION_SOURCE, "formula": "arange(234, int16).reshape(2,9,13)"},
            "rgb": {
                "source": SELECTION_SOURCE,
                "formula": "arange(702).reshape(2,9,13,3).astype(uint8)",
            },
            "datasets": {"source": "fieldkit", "formula": "first=stack[0], second=stack[1]"},
            "flat": {"source": "fieldkit", "formula": "zeros((64,80), float32)"},
            "invalid": {
                "source": "fieldkit",
                "reason": "truncated NumPy header, no data or pickle",
            },
            "simulation_structure": {"source": SIM_SOURCE, "settings_source": SIM_OPTIONS},
        }
    )
    return details


def input_info(folder, key):
    path = safe_file(folder, FILES[key])
    info = {"path": "inputs/" + path.name, "sha256": file_sha(path), "bytes": path.stat().st_size}
    if key == "datasets":
        with np.load(path, allow_pickle=False) as dataset:
            require(dataset.files == ["first", "second"], "Unexpected NPZ dataset names")
            members = {name: sample_properties(dataset[name]) for name in dataset.files}
        require(
            members["first"]["shape"] == members["second"]["shape"] == [9, 13]
            and members["first"]["array_sha256"] != members["second"]["array_sha256"],
            "NPZ must have two different selectable 2D arrays",
        )
        return {**info, "kind": "named_arrays", "shape": None, "dtype": None, "members": members}
    if key in ("invalid", "simulation_structure"):
        if key == "invalid":
            require(path.read_bytes() == b"\x93NUMPY\x01\x00\x40\x00", "Unexpected invalid fixture")
        return {
            **info,
            "kind": "invalid_npy" if key == "invalid" else "structure_parameters",
            "shape": None,
            "dtype": None,
        }
    if path.suffix in {".tif", ".tiff"}:
        with Image.open(path) as image:
            frames = [np.asarray(frame.copy()) for frame in ImageSequence.Iterator(image)]
        values = frames[0] if len(frames) == 1 else np.stack(frames)
    else:
        values = np.load(path, allow_pickle=False)
    return {**info, "kind": "array", **sample_properties(values)}


def card_name(row):
    return row["case_id"] + ("-" + row["variant"] if row["variant"] else "") + ".md"


def card_text(row):
    lines = [
        f"# {card_name(row)[:-3]} 现场操作卡",
        "",
        "适用发行档 " + "、".join(row["profiles"]),
        "",
        "执行条件 " + CONDITIONS[row["condition"]],
        "",
        row["setup"],
        "",
        "## 在新宿主会话中发送",
        "",
    ]
    for turn in row["turns"]:
        lines.extend(
            [
                "首轮" if turn["when"] == "start" else turn["when"],
                "",
                *("> " + line for line in turn["text"].splitlines()),
                "",
            ]
        )
    lines.extend(["## 现场记录", "", *("- " + tail for tail in row["observe"]), ""])
    return "\n".join(lines).encode("utf-8")


def guide(rows, identity, *, gpa_card=False, version=2):
    location = (
        "将整个目录放在所选项目根目录，保留目录名。按卡片指定的发行档和条件在真实宿主中执行。"
        if version == 1 else
        "将完整卡片和核对材料放在受测项目外。按卡选择发行档、输入和执行条件。"
        "向宿主发送当前轮的请求，在受测项目中仅放入这一轮使用的输入文件。"
    )
    lines = [
        "# AutoTEM 宿主现场验收",
        "",
        f"核心发行 {identity['release_id']}，清单 SHA-256 为 "
        f"`{identity['release_manifest_sha256']}`。",
        "",
        location
        + "请现场决定项目信任和工具权限，记录实际宿主版本、时间、请求、工具调用及运行编号。",
        "",
        "这些卡片和样本供操作使用。结果与原始会话应保存到独立证据目录。",
        "",
        "| 卡片 | 发行档 | 前提 |",
        "|---|---|---|",
    ]
    for row in rows:
        name = card_name(row)
        lines.append(
            f"| [{name[:-3]}](cards/{name}) | {'、'.join(row['profiles'])} | "
            f"{CONDITIONS[row['condition']]} |"
        )
    if gpa_card:
        lines += ["", f"[J08 GPA 补充卡]({GPA_CARD})包含选峰和参考区的合法、未答及非法回复观察。"]
    return ("\n".join(lines) + "\n").encode("utf-8")


def card_entries(rows):
    return [
        {
            "case_id": row["case_id"],
            "variant": row["variant"],
            "path": "cards/" + card_name(row),
            "profiles": row["profiles"],
            "condition": row["condition"],
            "inputs": row["inputs"],
        }
        for row in rows
    ]


BINDINGS = (
    "release-manifest.json",
    "release-index.json",
    "matrix.json",
    "science.zh-CN.json",
    "operations.zh-CN.json",
    "source-inventory.json",
    "simulation-case.json",
)


def walk_files(root):
    require(root.is_dir() and not linked(root), "Expected an ordinary field kit directory")
    result = {}
    for directory, subdirectories, files in os.walk(root, followlinks=False):
        current = Path(directory)
        for child in subdirectories:
            require(not linked(current / child), "Linked directory in field kit")
        for child in files:
            name = (current / child).relative_to(root).as_posix()
            path = safe_file(root, name)
            if name != "index.json":
                result[name] = file_sha(path)
    return result


def verify(root, release=None, release_index=None, matrix=None):
    root = Path(root).absolute()
    require(all(not linked(p) for p in (root, *root.parents)), "Linked field kit path")
    spec = load(safe_file(root, "index.json"))
    require(
        spec["schema_version"] in (1, 2)
        and spec["kind"] == GENERATED
        and spec["project_directory"] == root.name,
        "Wrong field kit identity or directory name",
    )
    check_sha(spec["generator_sha256"])
    inventory = walk_files(root)
    require(inventory == spec["files_sha256"], "Field kit files differ from the index")
    bindings = {n: safe_file(root, "basis/" + n) for n in BINDINGS}
    identity = release_identity(
        bindings["release-manifest.json"], bindings["release-index.json"], bindings["matrix.json"]
    )
    require(identity == spec["release"], "Field kit release identity differs")
    for path, name in ((release_index, "release-index.json"), (matrix, "matrix.json")):
        if path is not None:
            require(file_sha(path) == file_sha(bindings[name]), f"Different external {name}")
    src = load(bindings["source-inventory.json"])
    require(
        set(src) in (set(INPUT_SOURCES), {*INPUT_SOURCES, GPA_CARD_SOURCE})
        and all(check_sha(v) for v in src.values())
        and src[MATRIX_SOURCE] == identity["matrix_sha256"],
        "Invalid source bindings",
    )
    options = load(bindings["simulation-case.json"])
    if release is not None:
        require(
            file_sha(safe_file(release, "release-manifest.json"))
            == identity["release_manifest_sha256"],
            "Different external release manifest",
        )
        originals, settings, _ = archived_sources(release, bindings["matrix.json"])
        require(originals == src and settings == options, "Different release input sources")
    rows = render_rows(
        load(bindings["matrix.json"]),
        [load(bindings[k + ".zh-CN.json"]) for k in ("science", "operations")],
        spec["project_directory"],
    )
    require(
        card_entries(rows) == spec["cards"] and set(spec["inputs"]) == FILES.keys(),
        "Field kit case coverage or inputs differ",
    )
    require(
        safe_file(root, "README.md").read_bytes()
        == guide(rows, identity, gpa_card=GPA_CARD_SOURCE in src, version=spec["schema_version"]),
        "Field kit guide differs from its cards",
    )
    expected = {"README.md", *("basis/" + n for n in BINDINGS)}
    if GPA_CARD_SOURCE in src:
        require(
            file_sha(safe_file(root, GPA_CARD)) == src[GPA_CARD_SOURCE],
            "GPA follow-up card differs from frozen source",
        )
        expected.add(GPA_CARD)
    for row in rows:
        name = "cards/" + card_name(row)
        require(
            safe_file(root, name).read_bytes() == card_text(row), f"Changed operator card: {name}"
        )
        expected.add(name)
    for key, descriptor in spec["inputs"].items():
        require(descriptor["path"] == "inputs/" + FILES[key], f"Wrong input path: {key}")
        expected.add(descriptor["path"])
        measured = input_info(root / "inputs", key)
        require(all(descriptor[k] == v for k, v in measured.items()), f"Different sample: {key}")
        source = descriptor["source"]
        if source == "fieldkit":
            require(
                key in {"flat", "datasets", "invalid"}
                and descriptor["source_sha256"] == spec["generator_sha256"],
                "Unbound kit fixture",
            )
        else:
            require(
                source
                == (
                    CASE_SOURCE
                    if key in ANALYTIC
                    else SIM_SOURCE
                    if key == "simulation_structure"
                    else SELECTION_SOURCE
                )
                and descriptor["source_sha256"] == src[source],
                "Wrong fixture provenance",
            )
    require(set(inventory) == expected, "Unexpected files in field kit")
    payload = load(safe_file(root, "inputs/" + FILES["simulation_structure"]))
    require(
        payload == simulation_payload(options), "Simulation input differs from the acceptance case"
    )
    return spec


def prepare(out, release, release_index, matrix, *, templates=HERE / "field"):
    out = Path(out).absolute()
    require(all(not linked(p) for p in out.parents if p.exists()), "Linked destination ancestor")
    require(len(relative(out.name).parts) == 1, "Expected a single project-relative kit directory")
    identity = release_identity(safe_file(release, "release-manifest.json"), release_index, matrix)
    hashes, options, gpa_card = archived_sources(release, matrix)
    sources = {
        "release-manifest.json": safe_file(release, "release-manifest.json").read_bytes(),
        "release-index.json": release_index.read_bytes(),
        "matrix.json": matrix.read_bytes(),
        **{
            name + ".zh-CN.json": (templates / (name + ".zh-CN.json")).read_bytes()
            for name in ("science", "operations")
        },
        "source-inventory.json": as_json(hashes),
        "simulation-case.json": as_json(options),
    }
    rows = render_rows(
        load(matrix),
        [
            json.loads(sources[k + ".zh-CN.json"], object_pairs_hook=unique)
            for k in ("science", "operations")
        ],
        out.name,
    )
    generator = file_sha(Path(__file__))

    def existing():
        index = verify(out, release, release_index, matrix)
        require(
            index["generator_sha256"] == generator
            and all(
                index["files_sha256"]["basis/" + name] == sha(data)
                for name, data in sources.items()
            ),
            "Destination belongs to a different kit; choose a new directory",
        )
        return index

    if out.exists() or out.is_symlink():
        return existing()
    out.parent.mkdir(parents=True, exist_ok=True)
    stage = out.parent / (".fieldkit-" + uuid4().hex)
    stage.mkdir()
    try:
        # The inner directory has its final name so verification before publication exercises
        # the same project-relative paths that an operator will use after moving it.
        root = stage / out.name
        root.mkdir()
        for name in ("basis", "cards", "inputs"):
            (root / name).mkdir()
        for name, data in sources.items():
            (root / "basis" / name).write_bytes(data)
        for row in rows:
            (root / "cards" / card_name(row)).write_bytes(card_text(row))
        if gpa_card is not None:
            (root / GPA_CARD).write_bytes(gpa_card)
        (root / "README.md").write_bytes(guide(rows, identity, gpa_card=gpa_card is not None))
        details = make_inputs(root / "inputs", options)
        inputs = {}
        for key, detail in details.items():
            source_sha = generator if detail["source"] == "fieldkit" else hashes[detail["source"]]
            inputs[key] = {
                **detail,
                "source_sha256": source_sha,
                **input_info(root / "inputs", key),
            }
        index = {
            "schema_version": 2,
            "kind": GENERATED,
            "project_directory": out.name,
            "release": identity,
            "generator_sha256": generator,
            "inputs": inputs,
            "cards": card_entries(rows),
            "files_sha256": walk_files(root),
        }
        (root / "index.json").write_bytes(as_json(index))
        verify(root, release, release_index, matrix)
        try:
            root.rename(out)
        except OSError:
            if out.exists():
                return existing()
            raise
        return index
    finally:
        shutil.rmtree(stage)


def pack_operator(kit, release, release_index, output):
    """Attach the Windows entry point beside the verified field kit."""
    require(kit.name == "现场验收", "Use the field kit prepared with the directory name 现场验收")
    index = verify(kit, release, release_index)
    index_bytes = safe_file(kit, "index.json").read_bytes()
    require(json.loads(index_bytes) == index, "Field index changed during packaging")
    files = {"现场验收/index.json": index_bytes}
    for name, expected in index["files_sha256"].items():
        contents = safe_file(kit, name).read_bytes()
        require(sha(contents) == expected, f"Changed field kit member: {name}")
        files["现场验收/" + name] = contents
    manifest = load(safe_file(release, "release-manifest.json"))
    source = safe_file(release, manifest["source"]["archive"])
    with source.open("rb") as frozen:
        require(
            stream_sha(frozen) == index["release"]["source_archive_sha256"],
            "Changed release source archive",
        )
        with ZipFile(frozen) as archive:
            helper = archive.read("tools/host_acceptance/field_capture.py")
            guide = archive.read("tools/host_acceptance/README.md").decode("utf-8")
    section = guide[guide.index("## 本地记录") : guide.index("## 验证与复审")]
    files["现场取证/取证格式.md"] = ("# 填写本次真实会话记录\n\n" + section).encode("utf-8")
    files["现场取证/field_capture.py"] = helper
    files["现场取证/binding.json"] = as_json(
        {
            "kit_index_sha256": sha(index_bytes),
            "release_index_sha256": index["release"]["release_index_sha256"],
        }
    )
    for name in ("Collect.ps1", "collect.py", "README.md"):
        files["现场取证/" + name] = (HERE / "operator" / name).read_bytes()
    require(
        not output.resolve().is_relative_to(kit.resolve()),
        "Place the ZIP outside the verified field kit",
    )
    with TemporaryDirectory(dir=output.parent) as folder:
        staged = Path(folder) / "operator.zip"
        with ZipFile(staged, "x", compression=ZIP_DEFLATED) as archive:
            for name, payload in sorted(files.items()):
                archive.writestr(name, payload)
        with ZipFile(staged) as archive:
            require(
                set(archive.namelist()) == files.keys()
                and all(archive.read(n) == b for n, b in files.items()),
                "Operator ZIP readback failed",
            )
        os.link(staged, output)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    build = actions.add_parser("prepare", help="Create or reuse an intact field kit")
    check = actions.add_parser("verify", help="Verify a field kit offline")
    pack = actions.add_parser(
        "pack-operator", help="Package a qualified Windows field kit with a capture entry point"
    )
    pack.add_argument("kit", type=Path)
    pack.add_argument("--out", required=True, type=Path)
    pack.add_argument("--release", required=True, type=Path)
    pack.add_argument("--release-index", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    check.add_argument("kit", type=Path)
    for action in (build, check):
        action.add_argument("--release", required=action is build, type=Path)
        action.add_argument("--release-index", required=action is build, type=Path)
        action.add_argument(
            "--matrix", type=Path, default=HERE / "matrix.json" if action is build else None
        )
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            index = prepare(args.out, args.release, args.release_index, args.matrix)
        elif args.action == "pack-operator":
            index = pack_operator(args.kit, args.release, args.release_index, args.out)
        else:
            index = verify(args.kit, args.release, args.release_index, args.matrix)
        print(
            json.dumps(
                {
                    "kind": index["kind"],
                    "release": index["release"]["release_id"],
                    "cards": len(index["cards"]),
                    "inputs": len(index["inputs"]),
                },
                ensure_ascii=True,
            )
        )
        return 0
    except (ValueError, OSError, KeyError, IndexError, TypeError, BadZipFile) as exc:
        parser.exit(2, f"Field kit error: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
