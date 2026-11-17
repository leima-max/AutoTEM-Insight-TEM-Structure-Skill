"""Project-local, receipt-driven release removal. Scientific data is outside its file set."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import marshal
import os
import re
import stat
import struct
import sys
import warnings
import zipfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

from autotem import bootstrap as boot
from autotem.harness.hosts import rollback_host
from autotem.installation.release import portable_path, sha256, unique_json

JOURNAL = "uninstall.json"
INVENTORY = "runtime-inventory.json"
BUSY_STATES = {"detaching", "cleaning"}


def _fail(code, message):
    raise boot.BootstrapError(code, message)


def _stream_hash(stream):
    digest = hashlib.sha256()
    for data in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(data)
    return digest.hexdigest()


def _expected_cache(source):
    """Register the exact ordinary CPython cache for an installed source, before first import."""
    content = source.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            code = compile(content, str(source), "exec", dont_inherit=True, optimize=0)
        except (SyntaxError, ValueError, OverflowError):
            # Some wheel test fixtures deliberately contain invalid Python syntax.
            return None
    stamp = int(source.stat().st_mtime) & 0xFFFFFFFF
    header = importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, stamp, len(content) & 0xFFFFFFFF)
    digest = hashlib.sha256(header)
    digest.update(marshal.dumps(code))
    return digest.hexdigest()


def _tree(root):
    """Do not follow links, including Windows directory junctions."""
    stack = [root]
    while stack:
        directory = stack.pop()
        boot._no_links(directory)
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                mode = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(mode.st_mode) or getattr(mode, "st_file_attributes", 0) & 0x400:
                    yield path, "link"
                elif stat.S_ISDIR(mode.st_mode):
                    yield path, "directory"
                    stack.append(path)
                elif stat.S_ISREG(mode.st_mode):
                    yield path, "file"
                else:
                    yield path, "special"


def _venv_files(root):
    # A cold install can claim its venv launcher/templates and files installed through pip.
    # Restrict eligibility even when a user creates an extra file during provisioning.
    allowed = {root / boot.MARKER_NAME, root / "pyvenv.cfg", root / ".gitignore"}
    for folder in ("bin", "Scripts"):
        for name in (
            "python",
            "python3",
            "python.exe",
            "pythonw.exe",
            "activate",
            "Activate.ps1",
            "activate.bat",
            "deactivate.bat",
            "activate.csh",
            "activate.fish",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
        ):
            allowed.add(root / folder / name)
    sites = [root / "Lib/site-packages", *(root / "lib").glob("python*/site-packages")]
    for site in sites:
        for record in site.glob("*.dist-info/RECORD"):
            boot._no_links(record)
            with record.open(encoding="utf-8", newline="") as stream:
                for row in csv.reader(stream):
                    name = row[0]
                    path = Path(os.path.abspath(site / name))
                    if not path.is_relative_to(root):
                        _fail(
                            "INVALID_RUNTIME_INVENTORY", "An installed RECORD escapes its runtime."
                        )
                    allowed.add(path)
    return allowed


def record_new_runtime(control, transaction_path, record, *, fresh):
    digest = record["manifest_sha256"]
    runtime = boot._inside(control, "runtimes/rt-" + digest[:20])
    marker = boot._inside(runtime, boot.MARKER_NAME)
    record["runtime_marker_sha256"] = sha256(marker)
    if not fresh:
        return
    allowed = _venv_files(runtime)
    files = {}
    directories = set()
    for path, kind in _tree(runtime):
        if kind != "file":
            continue
        if path not in allowed:
            try:
                source = Path(importlib.util.source_from_cache(str(path)))
            except (ValueError, NotImplementedError):
                continue
            if source not in allowed:
                continue
        relative = path.relative_to(runtime)
        files[relative.as_posix()] = sha256(path)
        directories.update(p.as_posix() for p in relative.parents if p != Path("."))
    # pip installs without eagerly compiling every module. Host and worker imports can
    # later create standard caches. Only their pre-registered, exact byte hashes qualify.
    for name in list(files):
        if not name.endswith(".py"):
            continue
        source = runtime / name
        cache = Path(importlib.util.cache_from_source(str(source)))
        relative = cache.relative_to(runtime)
        if relative.as_posix() in files:
            continue
        expected = _expected_cache(source)
        if expected:
            files[relative.as_posix()] = expected
            directories.add(relative.parent.as_posix())
    # venv creates this directory even without any development headers.
    for empty in (
        "Include",
        "include",
        f"include/python{sys.version_info.major}.{sys.version_info.minor}",
    ):
        if (runtime / empty).is_dir():
            directories.add(empty)
    payload = {
        "schema_version": "1.0.0",
        "project": str(control.parent),
        "installation_id": record["installation_id"],
        "manifest_sha256": digest,
        "files": files,
        "directories": sorted(directories),
    }
    receipt = transaction_path.with_name(INVENTORY)
    boot._atomic_json(receipt, payload)
    record["runtime_inventory_sha256"] = sha256(receipt)


def base_caches(control):
    # Setup runs this module with -B. Its child venv and host commands can still write
    # standard-library bytecode into the pinned interpreter during installation.
    root = Path(sys.base_prefix)
    if root.parent != control / "python" or not re.fullmatch(
        r"cpython-[0-9]+\.[0-9]+\.[0-9]+-[0-9a-f]{12}", root.name
    ):
        return {}
    boot._no_links(root)
    return {
        p.relative_to(root).as_posix(): p
        for p in (root / "Lib").rglob("*.pyc")
        if p.parent.name == "__pycache__"
    }


def record_new_base_caches(control, record, before, *, bundle, release, fresh):
    after = base_caches(control)
    created = {}
    for relative in sorted(after.keys() - before.keys()):
        path = after[relative]
        boot._no_links(path)
        created[relative] = sha256(path)
    root = Path(sys.base_prefix)
    if fresh and root.parent == control / "python":
        # Use the verified PSF input to bound bytecode produced by later worker imports.
        manifest = Path(bundle) / "release-manifest.json"
        if sha256(manifest) != release.manifest_sha256:
            _fail("RELEASE_CHANGED", "The private Python manifest changed during installation.")
        spec = unique_json(manifest.read_bytes())["runtime"]
        if spec["status"] != "nuget_archive_included":
            return
        archive = boot._inside(Path(bundle).resolve(), portable_path(spec["archive"]))
        if not root.name.endswith("-" + spec["sha256"][:12]) or sha256(archive) != spec["sha256"]:
            _fail(
                "RELEASE_CHANGED",
                "The private Python archive no longer matches the installed base.",
            )
        with zipfile.ZipFile(archive) as zipped:
            for data in zipped.infolist():
                if not data.filename.startswith("tools/Lib/") or not data.filename.endswith(".py"):
                    continue
                path = root / portable_path(data.filename[6:])
                cache = (
                    Path(importlib.util.cache_from_source(str(path))).relative_to(root).as_posix()
                )
                if cache in after:
                    continue
                boot._no_links(path)
                with zipped.open(data) as content:
                    expected_source = _stream_hash(content)
                if not path.is_file() or sha256(path) != expected_source:
                    continue
                expected = _expected_cache(path)
                if expected:
                    created[cache] = expected
    if created:
        record["base_cache_runtime"] = Path(sys.base_prefix).name
        record["base_cache_hashes"] = created


def _records(control):
    records = {}
    for path in boot._inside(control, "installations").glob("*/transaction.json"):
        boot._no_links(path)
        row = unique_json(path.read_bytes())
        key = row["installation_id"]
        if (
            not re.fullmatch("[a-f0-9]{32}", key)
            or path.parent.name != key
            or not re.fullmatch("[a-f0-9]{64}", row["manifest_sha256"])
        ):
            _fail("INVALID_INSTALL_RECEIPT", "Check the project's installation journals.")
        records[key] = row
    return records


def _chain(pointer, records):
    receipts = []
    seen = set()
    owners = {key for key in (pointer or {}).get("host_installations", {}).values() if key}
    while pointer:
        key = pointer["installation_id"]
        row = records.get(key)
        if not row or key in seen or row["manifest_sha256"] != pointer["manifest_sha256"]:
            _fail("INVALID_INSTALL_RECEIPT", "The activation history is incomplete.")
        seen.add(key)
        applied = row.get("applied_assets", row["attempted_assets"])
        if not set(applied) <= set(row["attempted_assets"]):
            _fail("INVALID_INSTALL_RECEIPT", "The recorded host installations disagree.")
        receipts.extend(reversed(applied))
        pointer = row["previous"]
    if len(set(receipts)) != len(receipts):
        _fail("INVALID_INSTALL_RECEIPT", "Duplicate host receipts in activation history.")
    if not owners <= set(receipts):
        _fail("HOST_RECEIPT_MISSING", "Restore the receipts referenced by the active hosts.")
    return receipts


def _host_receipt(control, key):
    if not re.fullmatch("install-[a-f0-9]{32}", key):
        _fail("INVALID_HOST_RECEIPT", "Check the recorded host receipt identifier.")
    path = boot._inside(control, f"host-installations/{key}/manifest.json")
    if not path.is_file():
        _fail("HOST_RECEIPT_MISSING", f"Restore the host receipt for {key} and retry.")
    record = unique_json(path.read_bytes())
    host = record.get("host")
    if (
        host not in boot.HOSTS
        or key != record.get("installation_id")
        or record["state"] not in {"applied", "rolled_back"}
    ):
        _fail("INVALID_HOST_RECEIPT", f"Check the host receipt for {key}.")
    return path.parent, record


def _preflight(control, keys):
    # Simulate the whole reverse sequence before touching the first host file.
    expected = {}
    for key in keys:
        directory, receipt = _host_receipt(control, key)
        host = receipt["host"]
        for row in receipt["files"]:
            if row["action"] == "unchanged":
                continue
            rel = portable_path(row["path"])
            allowed = (
                rel == ".autotem/host-entry.py"
                or rel.startswith(f"autotem-host/{host}/")
                or host == "codex"
                and (rel == ".codex/config.toml" or rel.startswith(".agents/skills/autotem/"))
                or host == "openclaw"
                and rel.startswith("skills/autotem/")
            )
            if not allowed:
                _fail(
                    "INVALID_HOST_RECEIPT", "Host receipt names a file outside its managed paths."
                )
            target = boot._inside(control.parent, rel)
            if rel not in expected:
                expected[rel] = sha256(target) if target.exists() else None
            before = row["before_sha256"]
            if before is not None:
                backup = row.get("backup", "")
                if not re.fullmatch(r"backups/[0-9]{4,}\.bin", backup):
                    _fail("HOST_BACKUP_MISSING", f"Restore the original host backup for {rel}.")
                path = boot._inside(directory, backup)
                if not path.is_file() or sha256(path) != before:
                    _fail("HOST_BACKUP_MISSING", f"Restore the original host backup for {rel}.")
            if expected[rel] not in (before, row["after_sha256"]):
                _fail(
                    "HOST_FILE_CONFLICT", f"Preserved a changed host file. Review {rel} and retry."
                )
            expected[rel] = before


def _detach(control, journal, pointer, records, switch_window, apply):
    journal_path = control / JOURNAL
    key = pointer["installation_id"] if pointer else None
    if journal and journal["state"] in BUSY_STATES:
        if journal["state"] == "cleaning":
            if pointer:
                _fail("ACTIVATION_CONFLICT", "Review the active runtime before resuming cleanup.")
            return journal
        keys = journal["receipts"]
        cursor = journal["cursor"]
        if (
            not isinstance(cursor, int)
            or not 0 <= cursor <= len(keys)
            or pointer
            and (key != journal["active"] or keys != _chain(pointer, records))
            or not pointer
            and cursor != len(keys)
        ):
            _fail("ACTIVATION_CONFLICT", "Review the interrupted host detachment journal.")
    else:
        keys = _chain(pointer, records)
        cursor = 0
        journal = {
            "schema_version": "1.0.0",
            "state": "detaching",
            "active": key,
            "receipts": keys,
            "cursor": 0,
            "retired_installations": (journal or {}).get("retired_installations", []),
        }
    _preflight(control, keys[cursor:])
    if not apply:
        return journal
    with switch_window(control):
        boot._atomic_json(journal_path, journal)
        for index in range(cursor, len(keys)):
            response = rollback_host(control.parent, keys[index], dry_run=False, recover=True)
            if not response["ok"]:
                _fail(
                    "HOST_FILE_CONFLICT",
                    "Preserved modified host files: " + ", ".join(response["conflicts"]),
                )
            journal["cursor"] = index + 1
            boot._atomic_json(journal_path, journal)
        if pointer:
            boot._inside(control, "active-runtime.json").unlink()
        journal["state"] = "cleaning"
        boot._atomic_json(journal_path, journal)
    return journal


def _windows_users(images):
    """Inspect executable images, plus mapped libraries when Restart Manager is available."""
    import ctypes
    from ctypes import wintypes as w

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class Entry(ctypes.Structure):
        _fields_ = [
            ("size", w.DWORD),
            ("usage", w.DWORD),
            ("pid", w.DWORD),
            ("heap", ctypes.c_size_t),
            ("module", w.DWORD),
            ("threads", w.DWORD),
            ("parent", w.DWORD),
            ("priority", w.LONG),
            ("flags", w.DWORD),
            ("name", w.WCHAR * 260),
        ]

    kernel.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = w.HANDLE
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [
        w.HANDLE,
        ctypes.POINTER(Entry),
    ]
    kernel.QueryFullProcessImageNameW.argtypes = [
        w.HANDLE,
        w.DWORD,
        w.LPWSTR,
        ctypes.POINTER(w.DWORD),
    ]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        _fail("PROCESS_CHECK_UNAVAILABLE", "Windows could not inspect the process list.")
    paths = set(images)
    names = {p.name.casefold() for p in images if p.suffix.casefold() == ".exe"}
    users = set()
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        available = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while available:
            if entry.name.casefold() in names:
                process = kernel.OpenProcess(0x1000, False, entry.pid)
                if process:
                    try:
                        length = w.DWORD(32768)
                        name = ctypes.create_unicode_buffer(length.value)
                        if kernel.QueryFullProcessImageNameW(
                            process, 0, name, ctypes.byref(length)
                        ):
                            if Path(name.value) in paths:
                                users.add(entry.pid)
                        elif ctypes.get_last_error() != 87:
                            users.add(entry.pid)
                    finally:
                        kernel.CloseHandle(process)
                elif ctypes.get_last_error() != 87:
                    _fail(
                        "PROCESS_CHECK_UNAVAILABLE",
                        f"Windows cannot inspect Python PID {entry.pid}. Review access and retry.",
                    )
            available = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() != 18:  # ERROR_NO_MORE_FILES
            _fail("PROCESS_CHECK_UNAVAILABLE", "Windows returned an incomplete process list.")
    finally:
        kernel.CloseHandle(snapshot)

    class UniqueProcess(ctypes.Structure):
        _fields_ = [("pid", w.DWORD), ("start", w.FILETIME)]

    class ProcessInfo(ctypes.Structure):
        _fields_ = [
            ("process", UniqueProcess),
            ("name", w.WCHAR * 256),
            ("service", w.WCHAR * 64),
            ("app_type", w.DWORD),
            ("status", w.ULONG),
            ("session", w.DWORD),
            ("restartable", w.BOOL),
        ]

    api = ctypes.WinDLL("rstrtmgr")
    session = w.DWORD()
    session_key = ctypes.create_unicode_buffer(33)
    error = api.RmStartSession(ctypes.byref(session), 0, session_key)
    if error:
        return sorted(users)
    try:
        names = (w.LPCWSTR * len(images))(*(str(path) for path in images))
        error = api.RmRegisterResources(session, len(images), names, 0, None, 0, None)
        if error:
            return sorted(users)
        needed, count, reboot = w.UINT(), w.UINT(), w.DWORD()
        for _ in range(4):
            rows = (ProcessInfo * count.value)() if count.value else None
            error = api.RmGetList(
                session, ctypes.byref(needed), ctypes.byref(count), rows, ctypes.byref(reboot)
            )
            if error == 0:
                users.update(row.process.pid for row in (rows or ())[: count.value])
                return sorted(users)
            if error != 234:
                break
            count.value = needed.value
        return sorted(users)
    finally:
        api.RmEndSession(session)


def _images(roots):
    images = set()
    for root in roots:
        for folder in (root, root / "Scripts", root / "bin"):
            if not folder.exists():
                continue
            boot._no_links(folder)
            for path in folder.glob("python*"):
                if path.is_file() and (os.name != "nt" or path.suffix.lower() in {".exe", ".dll"}):
                    boot._no_links(path)
                    images.add(path)
    return images


def _runtime_users(roots):
    images = _images(roots)
    if not images:
        return []
    if os.name == "nt":
        return _windows_users(sorted(images))
    proc = Path("/proc")
    if not proc.is_dir():
        _fail("PROCESS_CHECK_UNAVAILABLE", "Process inspection is unavailable on this platform.")
    pids = set()
    for entry in proc.glob("[0-9]*"):
        try:
            exe = Path(os.readlink(entry / "exe"))
            arg = os.fsdecode((entry / "cmdline").read_bytes().split(b"\0", 1)[0])
            if exe in images or Path(arg) in images:
                pids.add(int(entry.name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return sorted(pids)


def _exclusive_file(path):
    """Windows denies new opens while bytes and deletion refer to the same handle."""
    import ctypes
    import msvcrt
    from ctypes import wintypes as w

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, w.LPVOID, w.DWORD, w.DWORD, w.HANDLE]
    create.restype = w.HANDLE
    handle = create(str(path), 0x80000000 | 0x00010000, 0, None, 3, 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(w.HANDLE(handle))
        raise
    return os.fdopen(fd, "rb")


@contextmanager
def _guard_images(roots):
    with ExitStack() as stack:
        opened = {}
        if os.name == "nt":
            for path in sorted(_images(roots)):
                try:
                    opened[path] = stack.enter_context(_exclusive_file(path))
                except OSError as exc:
                    _fail(
                        "RUNTIME_IN_USE",
                        f"Close the program using {path} and retry ({exc.winerror}).",
                    )
        yield opened


def _remove_file(path, digest, opened):
    expected = {digest} if isinstance(digest, str) else digest
    if os.name != "nt":
        if sha256(path) not in expected:
            return "changed"
        path.unlink()
        return None
    import ctypes
    import msvcrt
    from ctypes import wintypes as w

    guarded = opened.get(path)
    stream = guarded or _exclusive_file(path)
    try:
        if _stream_hash(stream) not in expected:
            return "changed"
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        disposition = ctypes.c_ubyte(1)
        handle = w.HANDLE(msvcrt.get_osfhandle(stream.fileno()))
        if not kernel.SetFileInformationByHandle(handle, 4, ctypes.byref(disposition), 1):
            raise ctypes.WinError(ctypes.get_last_error())
        stream.close()
    finally:
        if guarded is None:
            stream.close()
    return None


def _references(control, roots):
    # Only report configuration paths, never their contents or environment credentials.
    project = control.parent
    names = {
        "mcp.json",
        ".mcp.json",
        "mcp_config.json",
        "mcp_servers.json",
        "openclaw.json",
        "pyvenv.cfg",
    }
    paths = {}
    for relative in (
        ".codex/config.toml",
        ".vscode/mcp.json",
        ".mcp.json",
        ".openclaw/openclaw.json",
        ".workbuddy/mcp.json",
        ".workbuddy/settings.json",
    ):
        paths[project / relative] = True
    for parent, dirs, files in os.walk(project):
        path = Path(parent)
        dirs[:] = [
            name
            for name in dirs
            if name not in {".git", ".autotem"}
            and not (path / name).is_symlink()
            and not _is_reparse(path / name)
        ]
        for name in files:
            if (
                name.lower() in names
                or name == "adapter.json"
                and "autotem-host" in path.parts
                or name == "config.toml"
                and path.name == ".codex"
                or name == "settings.json"
                and path.name in {".vscode", ".workbuddy"}
            ):
                paths[path / name] = True
    home = Path.home()
    for path in (
        Path(os.environ.get("CODEX_HOME", home / ".codex")) / "config.toml",
        home / ".openclaw/openclaw.json",
        home / ".workbuddy/mcp.json",
        home / ".workbuddy/settings.json",
    ):
        paths.setdefault(path, False)
    if (control / "runtime/pyvenv.cfg").exists():
        paths[control / "runtime/pyvenv.cfg"] = True
    if not roots:
        return []
    targets = [*roots, control / "host-entry.py"]
    matched = []
    for path, local in paths.items():
        if not path.exists():
            continue
        boot._no_links(path)
        if path.stat().st_size > 16 * 1024 * 1024:
            _fail("CONFIG_CHECK_UNAVAILABLE", f"Review the oversized host configuration at {path}.")
        data = path.read_bytes()
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        text = data.decode(encoding).replace("\\\\", "/").replace("\\", "/").casefold()
        for target in targets:
            absolute = target.as_posix().casefold()
            rel = target.relative_to(project).as_posix().casefold()
            if absolute in text or local and rel in text:
                matched.append(str(path))
                break
    return sorted(matched)


def _is_reparse(path):
    return bool(getattr(path.lstat(), "st_file_attributes", 0) & 0x400)


def _inventories(control, records):
    by_root = {}
    markers = {}
    for row in sorted(records.values(), key=lambda r: r.get("created", 0)):
        root = boot._inside(control, "runtimes/rt-" + row["manifest_sha256"][:20])
        if row.get("runtime_marker_sha256"):
            markers[root] = row["runtime_marker_sha256"]
        if not row.get("runtime_inventory_sha256"):
            continue
        path = boot._inside(control, f"installations/{row['installation_id']}/{INVENTORY}")
        if not path.exists() or sha256(path) != row["runtime_inventory_sha256"]:
            _fail("INVALID_RUNTIME_INVENTORY", "Restore the installation file inventory and retry.")
        item = unique_json(path.read_bytes())
        if (
            item["project"] != str(control.parent)
            or item["installation_id"] != row["installation_id"]
            or item["manifest_sha256"] != row["manifest_sha256"]
        ):
            _fail(
                "INVALID_RUNTIME_INVENTORY",
                "The runtime inventory belongs to another installation.",
            )
        for name, digest in item["files"].items():
            portable_path(name)
            if not re.fullmatch("[a-f0-9]{64}", digest):
                _fail("INVALID_RUNTIME_INVENTORY", "A runtime file digest is incomplete.")
        for name in item["directories"]:
            portable_path(name)
        by_root[root] = item
    for root, item in by_root.items():
        if root in markers:
            item["files"][boot.MARKER_NAME] = markers[root]
    return by_root


def _base_inventory(control, bundle, release, records):
    base = boot._inside(control, "python")
    manifest = Path(bundle) / "release-manifest.json"
    if not base.exists() or not manifest.exists():
        return None
    if sha256(manifest) != release.manifest_sha256:
        _fail("RELEASE_CHANGED", "The verified release manifest changed during cleanup.")
    metadata = unique_json(manifest.read_bytes())
    spec = metadata["runtime"]
    if spec["status"] != "nuget_archive_included":
        return None
    version = metadata["target"]["python_version"]
    target = boot._inside(base, f"cpython-{version}-{spec['sha256'][:12]}")
    if not target.exists():
        return None
    marker = boot._inside(target, ".autotem-python")
    if not marker.is_file() or marker.read_bytes() != spec["sha256"].encode("ascii"):
        return None
    archive = boot._inside(Path(bundle).resolve(), portable_path(spec["archive"]))
    if sha256(archive) != spec["sha256"]:
        _fail("RELEASE_CHANGED", "Restore the pinned private Python archive and retry.")
    files, dirs = {".autotem-python": sha256(marker)}, set()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if not member.filename.startswith("tools/") or member.is_dir():
                continue
            relative = portable_path(member.filename[6:])
            with source.open(member) as stream:
                files[relative] = _stream_hash(stream)
            dirs.update(p.as_posix() for p in Path(relative).parents if p != Path("."))
    for record in sorted(records.values(), key=lambda row: row.get("created", 0)):
        if record.get("base_cache_runtime") != target.name:
            continue
        for relative, digest in record.get("base_cache_hashes", {}).items():
            portable_path(relative)
            source_name = importlib.util.source_from_cache(relative)
            if Path(source_name).as_posix() in files and re.fullmatch("[0-9a-f]{64}", digest):
                previous = files.get(relative, frozenset())
                if isinstance(previous, str):
                    previous = frozenset([previous])
                files[relative] = previous | {digest}
                dirs.add(Path(relative).parent.as_posix())
    return target, {"files": files, "directories": dirs}


def _current_owner(root, item):
    marker = boot._inside(root, boot.MARKER_NAME)
    if not marker.is_file() or sha256(marker) != item["files"].get(boot.MARKER_NAME):
        return False
    state = unique_json(marker.read_bytes())
    return (
        state.get("managed_by") == "autotem.bootstrap"
        and state.get("manifest_sha256") == item["manifest_sha256"]
        and state.get("runtime") == str(root)
        and state.get("project") == item["project"]
    )


def _remove_tree(root, inventory, report, opened):
    if not root.exists():
        return
    boot._no_links(root)
    files, dirs = inventory["files"], set(inventory["directories"])
    known_dirs = []
    preserved = report["retained"]
    # Ownership markers remain until the other original files have been considered.
    markers = []

    def visit(path, kind):
        rel = path.relative_to(root).as_posix()
        if kind == "directory" and rel in dirs:
            known_dirs.append(path)
            return
        if kind == "directory":
            preserved[str(path)] = "unregistered_directory"
            return
        expected = files.get(rel)
        if kind != "file" or expected is None:
            preserved[str(path)] = "unregistered_" + kind
            return
        boot._no_links(path)
        try:
            reason = _remove_file(path, expected, opened)
            if reason:
                preserved[str(path)] = reason
            else:
                report["removed_file_count"] += 1
        except OSError:
            preserved[str(path)] = "in_use_or_access_denied"

    for path, kind in _tree(root):
        if path.parent == root and path.name in {".autotem-python", boot.MARKER_NAME}:
            markers.append((path, kind))
        else:
            visit(path, kind)
    if not any(Path(path).is_relative_to(root) for path in preserved):
        for path, kind in markers:
            visit(path, kind)
    else:
        for path, _ in markers:
            preserved[str(path)] = "ownership_marker_for_retained_files"
    for path in sorted(known_dirs, key=lambda p: len(p.parts), reverse=True) + [root]:
        try:
            boot._no_links(path)
            path.rmdir()
        except OSError:
            if path.exists() and not any(Path(name).is_relative_to(path) for name in preserved):
                preserved[str(path)] = "directory_in_use_or_not_empty"


def remove(control, bundle, release, *, apply, pointer, switch_window):
    report = {
        "action": "uninstall",
        "ok": False,
        "state": "uninstall_pending",
        "removed_file_count": 0,
        "retained": {},
        "host_assets": "not_evaluated",
        "scientific_data": "preserved",
        "project": str(control.parent),
    }
    records = _records(control)
    runtimes = boot._inside(control, "runtimes")
    base_dir = boot._inside(control, "python")
    installed_dirs = []
    for parent in (runtimes, base_dir):
        if parent.exists():
            installed_dirs.extend(p for p in parent.iterdir() if p.is_dir() or p.is_symlink())
    legacy = boot._inside(control, "runtime")
    if legacy.exists():
        installed_dirs.append(legacy)
    if not records and not pointer and not installed_dirs:
        return {
            **report,
            "ok": True,
            "state": "not_installed",
            "next_action": "No managed installation is present in this project.",
        }
    journal_path = boot._inside(control, JOURNAL)
    journal = unique_json(journal_path.read_bytes()) if journal_path.exists() else None
    if not pointer and not installed_dirs and journal and journal.get("state") == "complete":
        return {
            **report,
            "ok": True,
            "state": "uninstalled",
            "host_assets": "detached",
            "next_action": "This project has already been uninstalled.",
        }
    retired = set((journal or {}).get("retired_installations", []))
    records = {key: row for key, row in records.items() if key not in retired}
    inventories = _inventories(control, records)
    base = _base_inventory(control, bundle, release, records) if records or not retired else None
    if not pointer and not records and not base:
        report["retained"] = {str(p): "no_verified_file_inventory" for p in installed_dirs}
        message = "Review the unregistered environments. No installation file was removed."
        return {
            **report,
            "error": {"code": "UNINSTALL_RETAINED_FILES", "message": message},
            "next_action": message,
        }
    if not apply:
        journal = _detach(control, journal, pointer, records, switch_window, False)
        return {
            **report,
            "ok": True,
            "state": "planned",
            "host_receipt_count": len(journal["receipts"]),
            "installed_directories": [str(p) for p in installed_dirs],
            "inventoried_runtimes": len(inventories),
            "next_action": "Apply to detach project hosts and remove unused, unchanged files.",
        }
    try:
        journal = _detach(control, journal, pointer, records, switch_window, True)
        report["host_assets"] = "detached"
        owned = {}
        for root, item in inventories.items():
            if root.exists():
                if _current_owner(root, item):
                    owned[root] = item
                else:
                    report["retained"][str(root)] = "ownership_marker_missing_or_changed"
        roots = list(owned)
        if base:
            roots.append(base[0])
        for path in installed_dirs:
            if path not in roots:
                report["retained"].setdefault(str(path), "no_verified_file_inventory")
        # The pointer was removed under the submission lock. Old or directly launched
        # interpreters may still be open. Do not delete any file until these exit.
        with switch_window(control):
            refs = _references(control, roots)
            if refs:
                report["configuration_references"] = refs
                _fail(
                    "RUNTIME_REFERENCED",
                    "Disconnect the listed configuration references and retry.",
                )
            # A Windows venv forwards to its base executable. Inspect every project-private
            # base, including an older base supplied by a different bundle.
            guard_roots = set(roots) | {path for path in installed_dirs if path.parent == base_dir}
            pids = _runtime_users(guard_roots) if roots else []
            if pids:
                report["active_pids"] = pids
                _fail(
                    "RUNTIME_IN_USE", "Close the listed MCP, worker or Python processes and retry."
                )
            with _guard_images(guard_roots if roots else ()) as opened:
                for root, inventory in owned.items():
                    _remove_tree(root, inventory, report, opened)
                if base:
                    if runtimes.exists() and any(runtimes.iterdir()) or legacy.exists():
                        report["retained"][str(base[0])] = "other_or_retained_runtime"
                    else:
                        _remove_tree(base[0], base[1], report, opened)
        if report["retained"]:
            reason = "Review the retained paths. Move local additions elsewhere before retrying."
            _fail("UNINSTALL_RETAINED_FILES", reason)
        journal["state"] = "complete"
        journal["retired_installations"] = sorted(retired | records.keys())
        boot._atomic_json(journal_path, journal)
        return {
            **report,
            "ok": True,
            "state": "uninstalled",
            "next_action": "Hosts are detached. The run store and installation receipts remain.",
        }
    except (boot.BootstrapError, OSError, ValueError) as exc:
        code = getattr(exc, "code", "UNINSTALL_REVIEW_REQUIRED")
        return {**report, "error": {"code": code, "message": str(exc)}, "next_action": str(exc)}
