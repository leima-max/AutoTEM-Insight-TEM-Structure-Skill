"""Local execution avoids managed-policy tempfile retry loops."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from autotem.core import DiffractionResult
from autotem.simulation.basic import _simulation_scratch, simulate_from_diffraction


def test_simulation_scratch_is_local_and_cleans_only_its_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOTEM_TMP_DIR", raising=False)
    root = tmp_path / ".autotem" / "tmp"
    root.mkdir(parents=True)
    user_file = root / "keep.txt"
    user_file.write_text("user data", encoding="utf-8")
    with _simulation_scratch() as directory:
        owned = Path(directory)
        assert owned.parent == root
        (owned / "result.json").write_text("{}", encoding="utf-8")
    assert not owned.exists()
    assert user_file.read_text(encoding="utf-8") == "user data"


def test_simulation_scratch_permission_failure_is_bounded_and_not_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOTEM_TMP_DIR", raising=False)
    original = Path.mkdir
    attempts = []

    def deny_owned(self, *args, **kwargs):
        if self.name.startswith("autotem_diffsims_"):
            attempts.append(self)
            raise PermissionError("managed policy denied scratch creation")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_owned)
    result = simulate_from_diffraction(
        DiffractionResult(), parameters={"diffsims_timeout_s": 1},
    )
    assert len(attempts) == 1
    assert result.provenance["adapter"]["active"] is False
    assert result.provenance["adapter"]["reason"] == "temporary_directory_unavailable"
    assert any("BLOCKING" in warning for warning in result.warnings)


def test_simulation_scratch_explicit_unicode_root(tmp_path, monkeypatch):
    root = tmp_path / "中文 path" / "scratch"
    monkeypatch.setenv("AUTOTEM_TMP_DIR", str(root))
    with _simulation_scratch() as directory:
        assert Path(directory).parent == root
    assert list(root.iterdir()) == []


def test_simulation_scratch_collision_does_not_delete_existing_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOTEM_TMP_DIR", raising=False)
    monkeypatch.setattr(
        "autotem.simulation.basic.uuid.uuid4", lambda: SimpleNamespace(hex="collision")
    )
    existing = tmp_path / ".autotem" / "tmp" / "autotem_diffsims_collision"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(OSError):
        with _simulation_scratch():
            pytest.fail("Collision must stop before entering the worker")
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_simulation_scratch_replacement_is_retained(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOTEM_TMP_DIR", raising=False)
    with _simulation_scratch() as directory:
        owned = Path(directory)
        owned.rename(owned.with_name("original-preserved"))
        owned.mkdir()
        (owned / "keep.txt").write_text("replacement", encoding="utf-8")
    assert (owned / "keep.txt").read_text(encoding="utf-8") == "replacement"
    assert "B5_SCRATCH_CLEANUP_FAILED" in caplog.text


def test_simulation_scratch_cleanup_failure_preserves_original_error(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AUTOTEM_TMP_DIR", raising=False)

    def deny_cleanup(_path):
        raise PermissionError("cleanup denied")

    monkeypatch.setattr("autotem.simulation.basic.shutil.rmtree", deny_cleanup)
    with pytest.raises(ValueError, match="original scientific failure"):
        with _simulation_scratch():
            raise ValueError("original scientific failure")
    assert "B5_SCRATCH_CLEANUP_FAILED" in caplog.text
