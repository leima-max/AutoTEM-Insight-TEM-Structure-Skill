"""Project-local comparison records for host configuration conflicts."""

from __future__ import annotations

import hashlib
import os
import re

from autotem import bootstrap as boot
from autotem.installation.release import portable_path, unique_json


def digest(data):
    return hashlib.sha256(data).hexdigest()


def host_target(host, name):
    portable_path(name)
    if not (
        name == ".autotem/host-entry.py"
        or name.startswith(f"autotem-host/{host}/")
        or host == "codex"
        and (name == ".codex/config.toml" or name.startswith(".agents/skills/autotem/"))
        or host == "openclaw"
        and name.startswith("skills/autotem/")
    ):
        raise ValueError("Invalid path in host installation receipt")
    return name


def copy_name(index):
    return f"a/{index:04d}"


def matches(project, relative, checksum):
    try:
        path = boot._inside(project, relative)
        return path.is_file() and digest(path.read_bytes()) == checksum
    except (OSError, boot.BootstrapError):
        return False


def artifact(project, installation, index, row):
    name = copy_name(index)
    if row.get("after_copy") == name:
        relative = f".autotem/host-installations/{installation}/{name}"
        if matches(project, relative, row["after_sha256"]):
            return relative
    return None


def proposed_path(checksum):
    # Keep the reference shorter than the host receipt, also for existing long Windows projects.
    return f".autotem/host-comparisons/{checksum[:32]}"


def proposed_copy(project, data):
    """Publish generated, project-specific reference bytes without replacing an existing file."""
    checksum = digest(data)
    relative = proposed_path(checksum)
    try:
        path = boot._inside(project, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        pass
    except (OSError, boot.BootstrapError):
        return None
    if matches(project, relative, checksum):
        return relative
    return None


def describe(project, name, condition, expected, *, receipt=None, proposed=None, save=False):
    target = boot._inside(project, name)
    state = "file" if target.is_file() else "directory" if target.is_dir() else "missing"
    current = digest(target.read_bytes()) if state == "file" else None
    comparison = None
    comparison_sha = None
    kind = None
    origin = None
    if receipt:
        key, index, row = receipt
        if not re.fullmatch("install-[a-f0-9]{32}", key):
            raise ValueError("Invalid host receipt identifier")
        origin = f".autotem/host-installations/{key}/manifest.json"
        comparison = artifact(project, key, index, row)
        if comparison:
            comparison_sha = expected
            kind = "installed_project_config" if name == ".codex/config.toml" else "installed_file"
    if proposed is not None and not comparison and save:
        candidate_sha = digest(proposed)
        if (
            condition != "recorded_file"
            or candidate_sha == expected
            and name != ".codex/config.toml"
        ):
            comparison = proposed_copy(project, proposed)
            if comparison:
                comparison_sha = candidate_sha
                kind = "managed_toml_section" if condition == "compatible_toml" else "release_file"
    return {
        "path": name,
        "current_state": state,
        "current_sha256": current,
        "condition": condition,
        "expected_sha256": expected,
        "receipt_path": origin,
        "comparison_path": comparison,
        "comparison_sha256": comparison_sha,
        "comparison_kind": kind,
    }


def resolution(operation, files):
    steps = []
    for row in files:
        name = row["path"]
        if row["condition"] == "recorded_file":
            if row["current_state"] == "directory":
                steps.append(f"先将占用 {name} 的目录整体移到你的私人备份位置。")
            if row["comparison_path"]:
                steps.append(f"用项目内已核验的 {row['comparison_path']} 恢复 {name}。")
            else:
                steps.append(
                    f"{name} 的受管版本没有可核验副本。只有持有 SHA-256 等于 "
                    f"{row['expected_sha256']} 的原文件时，才可将它放回原路径。"
                )
        elif row["condition"] == "compatible_toml" and row["current_state"] != "directory":
            steps.append(
                f"用文本编辑器检查 {name} 的 TOML 语法，保留其他设置。"
                "备份后移出冲突的 mcp_servers.autotem 项，或将该项改为安装器的配置。"
            )
            if row["comparison_path"]:
                steps.append(f"可对照单独的 AutoTEM 配置片段 {row['comparison_path']}。")
        else:
            steps.append(f"把占用 {name} 的文件或目录移到你选定的私人备份位置，使该路径可供安装。")
    retry = "重新执行回滚。" if operation == "rollback" else "重新执行安装。"
    choices = [
        {
            "id": "choose_new_project",
            "label": "保留本项目的编辑",
            "instruction": "保留当前文件和历史结果。要安装发行版，请另选一个空项目目录。",
        },
        {
            "id": "restore_then_retry",
            "label": "在当前项目核对后重试",
            "instruction": "先把自己的编辑和凭据保存到私人备份位置。" + "".join(steps) + retry,
        },
    ]
    info = {"schema_version": "1.0.0", "operation": operation, "files": files, "choices": choices}
    return info, "\n".join(c["instruction"] for c in choices)


def child_conflicts(project, host, previous, result):
    """Recheck receipt ownership, local file hashes and references across the process boundary."""
    names = result.get("conflicts")
    if not isinstance(names, list) or not names:
        return None
    receipts = {}
    if previous:
        if not re.fullmatch("install-[a-f0-9]{32}", previous):
            raise ValueError("Invalid prior host receipt identifier")
        path = boot._inside(project, f".autotem/host-installations/{previous}/manifest.json")
        manifest = unique_json(path.read_bytes())
        if manifest["installation_id"] != previous or manifest["host"] != host:
            raise ValueError("Mismatched prior host receipt")
        receipts = {
            r["path"]: (previous, i, r)
            for i, r in enumerate(manifest["files"])
            if not r.get("preexisting_unmanaged")
        }
    details = {r["path"]: r for r in result.get("conflict_resolution", {}).get("files", [])}
    plans = {r["path"]: r for r in result.get("files", [])}
    files = []
    for name in names:
        host_target(host, name)
        old = receipts.get(name)
        child = details.get(name, {})
        expected = old[2]["after_sha256"] if old else plans.get(name, {}).get("after_sha256")
        condition = "recorded_file" if old else "available_path"
        if not old and name == ".codex/config.toml" and child.get("condition") == "compatible_toml":
            condition, expected = "compatible_toml", None
        row = describe(project, name, condition, expected, receipt=old)
        h = child.get("comparison_sha256")
        if not row["comparison_path"] and isinstance(h, str) and re.fullmatch("[a-f0-9]{64}", h):
            reference = proposed_path(h)
            if child.get("comparison_path") == reference and (
                expected == h or condition == "compatible_toml"
            ):
                if matches(project, reference, h):
                    row.update(
                        comparison_path=reference,
                        comparison_sha256=h,
                        comparison_kind=(
                            "managed_toml_section"
                            if condition == "compatible_toml"
                            else "release_file"
                        ),
                    )
        files.append(row)
    info, text = resolution("upgrade" if previous else "install", files)
    return {"conflict_resolution": info, "next_action": text}


class HostFileConflict(boot.BootstrapError):
    def __init__(self, result):
        super().__init__("HOST_FILE_CONFLICT", result["next_action"])
        self.resolution = result.get("conflict_resolution")
