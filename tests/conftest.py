"""Private interpreters for tests that launch the Harness and its workers."""

import json
import os
import site
import subprocess
import sys
import venv
from pathlib import Path

import pytest


@pytest.fixture
def isolated_harness_venv(tmp_path_factory):
    # Keep the real child interpreter and its site-packages close to pytest's temp root.
    root = tmp_path_factory.mktemp("hv")
    venv.EnvBuilder(with_pip=False).create(root)
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    query = "import json, sysconfig; print(json.dumps(sysconfig.get_path('purelib')))"
    location = subprocess.check_output([str(python), "-I", "-c", query], encoding="ascii")
    purelib = Path(json.loads(location))
    assert purelib.resolve().is_relative_to(root.resolve())

    # A nested venv's base is the original Python, not the venv running pytest.
    # Share only this checkout and the enabled package directories of the test interpreter.
    sites = {Path(path).resolve() for path in site.getsitepackages()}
    if site.ENABLE_USER_SITE:
        sites.add(Path(site.getusersitepackages()).resolve())
    paths = [str(Path(__file__).resolve().parents[1])]
    for entry in sys.path:
        path = Path(entry).resolve()
        if path.name in {"site-packages", "dist-packages"} and path in sites:
            paths.append(str(path))
    # ASCII escapes allow Unicode paths even with the pre-3.13 Windows .pth decoder.
    (purelib / "autotem-test.pth").write_text(
        f"import sys; sys.path.extend({ascii(paths)})\n", encoding="ascii"
    )
    return python, purelib
