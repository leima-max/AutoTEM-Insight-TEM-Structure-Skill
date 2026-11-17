"""Exercise the SQLite publication boundary with real on-disk stores and competing connections."""

import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest
import test_release_activation

from autotem.bootstrap import BootstrapError
from autotem.harness.store import STORE_SCHEMA_VERSION, Store
from autotem.installation import managed

pytestmark = pytest.mark.smoke
release_project = test_release_activation.release_project
ABORT = "AUTOTEM_QUALIFICATION_REJECT_LEASE_WRITE"


def image(project):
    return {
        name: (project / name).read_bytes()
        for name in (
            ".autotem/active-runtime.json",
            ".codex/config.toml",
            ".autotem/host-entry.py",
            "autotem-host/codex/adapter.json",
            "history.bin",
        )
    }


@pytest.mark.parametrize("action", ["upgrade", "rollback", "uninstall"])
@pytest.mark.parametrize("fault", ["update", "commit"])
def test_failed_lease_write_never_publishes_or_detaches_files(release_project, action, fault):  # noqa: F811
    project, _ = release_project
    first = managed.manage(project, "first", apply=True)
    assert first["ok"], first.get("error")
    if action == "rollback":
        assert managed.manage(project, "second", apply=True)["ok"]
    (project / "history.bin").write_bytes(b"historical project input")
    before = image(project)
    database = project / ".autotem/runs/harness.sqlite3"
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("PRAGMA journal_mode=DELETE").fetchone() == ("delete",)
        db.execute("UPDATE scheduler SET pid=42,identity='old idle lease' WHERE singleton=1")
        if fault == "update":
            db.execute(
                "CREATE TRIGGER refuse_lease BEFORE UPDATE OF pid ON scheduler "
                f"BEGIN SELECT RAISE(ABORT, '{ABORT}'); END"
            )
        db.commit()
    flags = {"rollback": action == "rollback", "uninstall": action == "uninstall"}
    bundle = "first" if action == "uninstall" else "second"
    with closing(sqlite3.connect(database)) as reader:
        if fault == "commit":
            # A rollback-journal reader permits BEGIN IMMEDIATE and UPDATE, but prevents COMMIT.
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM runs").fetchone()
        blocked = managed.manage(project, bundle, apply=True, **flags)
    assert not blocked["ok"]
    code = "PROJECT_DATABASE_BUSY" if fault == "commit" else "STORE_UNAVAILABLE"
    assert blocked["error"]["code"] == code
    assert ("database is locked" if fault == "commit" else ABORT) in blocked["error"]["message"]
    assert image(project) == before
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("SELECT pid,identity FROM scheduler").fetchone() == (42, "old idle lease")
        if fault == "update":
            db.execute("DROP TRIGGER refuse_lease")
            db.commit()
    assert managed.manage(project, bundle, apply=True, **flags)["ok"]
    assert (project / "history.bin").read_bytes() == before["history.bin"]


@pytest.mark.parametrize("has_jobs", [True, False])
def test_version_zero_without_scheduler_identity_remains_unmigrated(tmp_path, has_jobs):
    control = tmp_path / ".autotem"
    database = control / "runs/harness.sqlite3"
    database.parent.mkdir(parents=True)
    with closing(sqlite3.connect(database)) as db:
        db.executescript(
            "CREATE TABLE runs(run_id,status); CREATE TABLE steps(id); CREATE TABLE artifacts(id);"
            "CREATE TABLE scheduler(singleton INTEGER PRIMARY KEY,pid,heartbeat);"
            "INSERT INTO scheduler VALUES(1,721,0); INSERT INTO runs VALUES('old','completed');"
        )
        if has_jobs:
            db.execute("CREATE TABLE jobs(job_id,status)")
        db.commit()
    with managed.switch_window(control) as version:
        assert version == 0
        with closing(sqlite3.connect(database, timeout=0)) as writer:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("BEGIN IMMEDIATE")
    with closing(sqlite3.connect(database)) as db:
        assert db.execute("PRAGMA user_version").fetchone() == (0,)
        assert [r[1] for r in db.execute("PRAGMA table_info(scheduler)")] == [
            "singleton",
            "pid",
            "heartbeat",
        ]
        assert db.execute("SELECT pid FROM scheduler").fetchone() == (None,)
        assert db.execute("SELECT * FROM runs").fetchone() == ("old", "completed")


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("submit", "WAIT_ACTIVE_RUNS"),
        ("UPDATE jobs SET status='running' WHERE job_id='j'", "WAIT_ACTIVE_RUNS"),
        (
            "UPDATE scheduler SET pid=77,identity='replacement' WHERE singleton=1",
            "PROJECT_DATABASE_BUSY",
        ),
        ("PRAGMA user_version=0", "STORE_VERSION_CHANGED"),
        ("PRAGMA user_version=100", "STORE_MIGRATION_REQUIRED"),
    ],
)
def test_independent_writer_winning_between_reservations_blocks_publication(
    tmp_path, monkeypatch, change, error
):
    control = tmp_path / ".autotem"
    store = Store(control / "runs")
    with store.connect() as db:
        db.execute("UPDATE scheduler SET pid=42,identity='old' WHERE singleton=1")
        db.execute(
            "INSERT INTO runs(run_id,request_hash,request,status,created,updated) "
            "VALUES('r','hash','{}','completed',0,0)"
        )
        db.execute("INSERT INTO jobs(job_id,run_id,created,status) VALUES('j','r',0,'completed')")

    class Interleave(sqlite3.Connection):
        def commit(self):
            super().commit()
            # A real independent connection wins the gap before the installer takes its last lock.
            with closing(sqlite3.connect(store.db_path, timeout=0)) as competitor:
                competitor.execute("BEGIN IMMEDIATE")
                if change == "submit":
                    competitor.execute(
                        "INSERT INTO runs(run_id,request_hash,request,status,created,updated) "
                        "VALUES('new','hash','{}','preparing',1,1)"
                    )
                    competitor.execute(
                        "INSERT INTO jobs(job_id,run_id,created,status) "
                        "VALUES('n','new',1,'queued')"
                    )
                else:
                    competitor.execute(change)
                competitor.commit()

    def connect(*args, **kwargs):
        return sqlite3.connect(*args, **kwargs, factory=Interleave)

    monkeypatch.setattr(managed, "sqlite3", SimpleNamespace(connect=connect, Error=sqlite3.Error))
    with pytest.raises(BootstrapError) as caught, managed.switch_window(control):
        pytest.fail("The file-publication block must remain unreachable")
    assert caught.value.code == error
    if change == "submit":
        with closing(sqlite3.connect(store.db_path)) as db:
            assert db.execute("SELECT status FROM runs WHERE run_id='new'").fetchone() == (
                "preparing",
            )
            assert db.execute("SELECT status FROM jobs WHERE job_id='n'").fetchone() == ("queued",)


@pytest.mark.parametrize(
    ("schema", "error"),
    [
        ("PRAGMA user_version=87", "STORE_MIGRATION_REQUIRED"),
        ("DROP TABLE scheduler", "STORE_UNAVAILABLE"),
        ("ALTER TABLE scheduler RENAME COLUMN identity TO other", "STORE_UNAVAILABLE"),
        ("DELETE FROM scheduler WHERE singleton=1", "STORE_UNAVAILABLE"),
    ],
)
def test_unknown_or_damaged_schema_refuses_before_mutation(tmp_path, schema, error):
    control = tmp_path / ".autotem"
    store = Store(control / "runs")
    with store.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == STORE_SCHEMA_VERSION == 1
        db.execute(schema)
    with pytest.raises(BootstrapError) as caught, managed.switch_window(control):
        pytest.fail("A future or damaged store must not publish files")
    assert caught.value.code == error


def test_clean_new_store_excludes_writers_until_guard_exit(tmp_path):
    control = tmp_path / ".autotem"
    control.mkdir()
    path = control / "runs/harness.sqlite3"
    with managed.switch_window(control) as version, closing(sqlite3.connect(path, timeout=0)) as db:
        assert version == 0
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            db.execute("BEGIN IMMEDIATE")
    with closing(sqlite3.connect(path, timeout=0)) as db:
        db.execute("BEGIN IMMEDIATE")
        assert db.execute("PRAGMA user_version").fetchone() == (0,)
        assert db.execute("SELECT name FROM sqlite_master").fetchall() == []
        db.rollback()
