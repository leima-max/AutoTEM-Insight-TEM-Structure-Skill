"""Deployment checks, explicitly selected outside the default smoke test filter.

AppTest cannot drive file_uploader or a canvas browser frontend. These tests
inject real UploadedFile objects at the uploader boundary and execute the real
app, detector, download widgets, and canvas server-side component call.
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import tifffile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPOSITORY_ROOT / "apps" / "atom_detection_ui.py"


def _tiff_bytes(*, x_shift: int = 0) -> bytes:
    y, x = np.indices((48, 48))
    image = np.zeros((48, 48), dtype=np.float64)
    for cy, cx in ((12, 12), (12, 32), (32, 12), (32, 32)):
        image += 4000 * np.exp(-((y - cy) ** 2 + (x - cx - x_shift) ** 2) / 3)
    buffer = io.BytesIO()
    tifffile.imwrite(buffer, image.astype(np.uint16), metadata=None, compression=None)
    return buffer.getvalue()


@pytest.fixture
def app_driver(monkeypatch):
    st = pytest.importorskip("streamlit")
    pytest.importorskip("streamlit_drawable_canvas")
    from streamlit.proto.Common_pb2 import FileURLs
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
    from streamlit.testing.v1 import AppTest

    class AppDriver:
        def __init__(self):
            self.upload = None
            self.downloads = {}
            self.apps = []

        def set_upload(self, content: bytes | None, *, file_id: str = "first"):
            self.upload = (
                None
                if content is None
                else UploadedFile(
                    UploadedFileRec(file_id, "sample.tif", "image/tiff", content),
                    FileURLs(),
                )
            )

        def new_app(self):
            app = AppTest.from_file(APP_PATH, default_timeout=45)
            self.apps.append(app)
            return app

    driver = AppDriver()
    original_download = st.download_button

    def capture_download(*args, **kwargs):
        driver.downloads[kwargs["label"]] = kwargs
        return original_download(*args, **kwargs)

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: driver.upload)
    monkeypatch.setattr(st, "download_button", capture_download)
    try:
        yield driver
    finally:
        for app in driver.apps:
            try:
                upload = app.session_state["uploaded_temp"]
            except KeyError:
                continue
            if upload is not None:
                upload.close()


def _assert_success(app) -> None:
    assert not app.exception, [error.message for error in app.exception]
    assert not app.error, [error.value for error in app.error]


def test_upload_detection_replacement_removal_csv_and_canvas(app_driver) -> None:
    app = app_driver.new_app().run()
    _assert_success(app)
    assert app.session_state["uploaded_temp"] is None

    first = _tiff_bytes()
    second = _tiff_bytes(x_shift=4)
    assert first != second and len(first) == len(second)
    app_driver.set_upload(first)
    app.run()
    _assert_success(app)
    first_path = app.session_state["uploaded_temp"].path
    first_signature = app.session_state["uploaded_sig"]
    first_image = app.session_state["base_image_raw"].copy()
    assert len(app.session_state["edited_coords"]) == 4
    assert first_path.read_bytes() == first
    assert app.session_state["base_result"].provenance["source"]["path"] == "sample.tif"

    download = app_driver.downloads["Download edited coordinates CSV"]
    assert download["mime"] == "text/csv"
    assert download["file_name"] == "sample_atom_coordinates_edited.csv"
    rows = list(csv.reader(io.StringIO(download["data"].decode("utf-8"))))
    assert rows[0] == ["y", "x", "intensity"]
    expected = np.column_stack(
        [app.session_state["edited_coords"], app.session_state["edited_intensities"]]
    )
    np.testing.assert_allclose(np.asarray(rows[1:], dtype=float), expected, atol=1e-6)
    assert app_driver.downloads["Download distance statistics CSV"]["data"]
    assert len(app.get("download_button")) >= 2

    canvas_checkbox = next(
        item for item in app.checkbox if item.label == "Enable canvas editing / ROI drawing"
    )
    canvas_checkbox.check().run()
    _assert_success(app)
    assert len(app.get("component_instance")) == 1

    app.session_state["sublattice_manual_labels"] = {0: "manual"}
    app.session_state["atomic_displacement_manual_pairs"] = {0: (1, 1)}
    app_driver.set_upload(second, file_id="replacement")
    app.run()
    _assert_success(app)
    second_path = app.session_state["uploaded_temp"].path
    assert second_path != first_path
    assert not first_path.parent.exists()
    assert second_path.read_bytes() == second
    assert app.session_state["uploaded_sig"] != first_signature
    assert not np.array_equal(app.session_state["base_image_raw"], first_image)
    assert len(app.session_state["edited_coords"]) == 4
    assert app.session_state["sublattice_manual_labels"] == {}
    assert app.session_state["atomic_displacement_manual_pairs"] == {}

    app.session_state["sublattice_manual_labels"] = {0: "manual"}
    app_driver.set_upload(None)
    app.run()
    _assert_success(app)
    assert not second_path.parent.exists()
    for key in (
        "uploaded_temp",
        "uploaded_sig",
        "last_detection_signature",
        "last_run_summary",
        "base_result",
        "base_image_raw",
        "base_image_norm",
        "edited_coords",
        "edited_intensities",
        "auto_coords",
        "auto_intensities",
        "roi_mask",
        "atom_strain_fields",
        "atomic_displacement_result",
        "plane_spacing_result",
        "atomic_intensity_result",
        "sublattice_result",
    ):
        assert app.session_state[key] is None, key
    assert app.session_state["sublattice_manual_labels"] == {}
    assert not app.get("download_button")


def test_intensity_download_distinguishes_unassessed_saturation(app_driver) -> None:
    app_driver.set_upload(_tiff_bytes())
    app = app_driver.new_app().run()
    next(button for button in app.button if button.label == "Run Atomic Column Intensity").click()
    app.run()
    _assert_success(app)
    download = app_driver.downloads["Download atomic-column intensity CSV"]
    lines = download["data"].decode("utf-8").splitlines()
    assert lines[0] == "intensity_unit,a.u."
    rows = list(csv.DictReader(lines[1:]))
    assert len(rows) == len(app.session_state["edited_coords"]) == 4
    flags = {(r["saturation_flag"], r["saturation_status"]) for r in rows}
    assert flags == {("0", "not_evaluated")}


def test_upload_sessions_have_independent_temporary_files(app_driver) -> None:
    app_driver.set_upload(_tiff_bytes())
    first_app = app_driver.new_app().run()
    second_app = app_driver.new_app().run()
    _assert_success(first_app)
    _assert_success(second_app)
    first_path = first_app.session_state["uploaded_temp"].path
    second_path = second_app.session_state["uploaded_temp"].path
    assert first_path != second_path
    app_driver.set_upload(None)
    first_app.run()
    _assert_success(first_app)
    assert not first_path.exists()
    assert second_path.exists()
    assert len(second_app.session_state["edited_coords"]) == 4


@pytest.mark.skipif(
    os.environ.get("AUTOTEM_DEPLOYMENT_WHEEL") != "1",
    reason="Requires the deployment CI step to replace the editable package with its wheel.",
)
def test_installed_wheel_starts_entrypoint_without_repository_imports(tmp_path) -> None:
    entrypoint = tmp_path / "atom_detection_ui.py"
    shutil.copyfile(APP_PATH, entrypoint)
    script = textwrap.dedent(
        """
        import json
        import sys
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        from autotem.io import upload_lifecycle

        repository = Path(sys.argv[2]).resolve()
        helper_path = Path(upload_lifecycle.__file__).resolve()
        assert repository not in helper_path.parents, helper_path
        assert any(part in {"site-packages", "dist-packages"} for part in helper_path.parts)
        assert all(Path(path).resolve() != repository for path in sys.path if path)
        app = AppTest.from_file(sys.argv[1], default_timeout=45).run()
        assert not app.exception, [error.message for error in app.exception]
        assert not app.error, [error.value for error in app.error]
        assert any(info.value == "Upload a TEM image to begin." for info in app.info)
        print(json.dumps({"installed_helper": str(helper_path), "startup": "passed"}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(entrypoint), str(REPOSITORY_ROOT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"startup": "passed"' in result.stdout
