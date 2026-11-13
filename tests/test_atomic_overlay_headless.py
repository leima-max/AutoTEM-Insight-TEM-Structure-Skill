"""The disk-only overlay must work even if the user's interactive backend requires Qt."""

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops


def test_static_atom_overlays_without_gui_backend(tmp_path):
    code = """
import sys
from pathlib import Path
import numpy as np
from autotem.atomic.atom_detection import _save_overlay

def observe(event, args):
    gui = {'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6'}
    if event == 'import' and args[0].split('.')[0] in gui:
        attempts.append(args[0])

attempts = []
sys.addaudithook(observe)
image = np.arange(48).reshape(6, 8)
out = Path(sys.argv[1])
_save_overlay(image, np.empty((0, 2)), out / 'empty.png')
_save_overlay(image, np.array([[2.0, 4.0]]), out / 'detected.png')
assert not attempts, attempts
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "MPLBACKEND": "QtAgg", "MPLCONFIGDIR": str(tmp_path / "mpl")},
        check=True,
        timeout=60,
    )
    with (
        Image.open(tmp_path / "empty.png") as empty,
        Image.open(tmp_path / "detected.png") as atoms,
    ):
        assert atoms.format == empty.format == "PNG"
        assert atoms.size == empty.size
        assert ImageChops.difference(atoms.convert("RGB"), empty.convert("RGB")).getbbox()
