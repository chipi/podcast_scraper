"""``db upgrade`` as the prod migrate step runs it (ADR-169)."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from podcast_scraper.db import migrate
from podcast_scraper.db.settings import normalize_url
from tests._database import worker_database_url

REPO = Path(__file__).resolve().parents[4]
VERSIONS = REPO / "src" / "podcast_scraper" / "db" / "migrations" / "versions"


@pytest.fixture
def url() -> str:
    found = worker_database_url()
    assert found, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    return found


def test_a_migrating_role_that_owns_the_schema_but_cannot_create_in_the_database(url: str) -> None:
    """Prod: infra pre-creates ``player``/``operator``/``corpus`` owned by ``migrator``, which does
    not hold CREATE on the database. ``CREATE SCHEMA IF NOT EXISTS`` needs that privilege even
    when the schema exists, so ``upgrade`` must only create a schema that is missing."""
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    role = f"migrator_{os.getpid()}"
    database = make_url(normalize_url(url)).database
    migrator_url = make_url(normalize_url(url)).set(username=role, password="migrator")
    try:
        with admin.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
            conn.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD 'migrator'"))
            conn.execute(text("CREATE SCHEMA operator AUTHORIZATION " + role))
            can_create = conn.execute(
                text("SELECT has_database_privilege(:r, :d, 'CREATE')"), {"r": role, "d": database}
            ).scalar()
        assert can_create is False, "the scenario needs a role without CREATE on the database"

        rendered = migrator_url.render_as_string(hide_password=False)
        migrate.upgrade(rendered, "operator")

        current, head = migrate.status(rendered, "operator")
        assert current == head
    finally:
        with admin.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
            conn.execute(text(f"DROP ROLE IF EXISTS {role}"))
        admin.dispose()
        migrate.upgrade(url, "operator")


def test_a_missing_schema_is_still_created_for_local_and_ci(url: str) -> None:
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
        migrate.upgrade(url, "operator")
        assert migrate.status(url, "operator")[0] == migrate.head_revision()
    finally:
        admin.dispose()


def test_migrations_only_create_objects_in_the_plane() -> None:
    """Infra owns schemas and grants; a migration creates objects, on the schema it is given."""
    sources = {p.name: p.read_text() for p in VERSIONS.glob("*.py") if p.name != "__init__.py"}
    assert sources
    for name, source in sources.items():
        assert not re.search(r"create\s+schema|\bgrant\b", source, re.I), name
        # An object names the plane it was given and no other schema.
        assert not re.search(r"\bschema=(?!schema\b)", source), name


def _cli(*args: str, url: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PODCAST_DATABASE_URL": url,
        "PYTHONPATH": str(REPO / "src"),
    }
    return subprocess.run(
        [sys.executable, "-m", "podcast_scraper.cli", "db", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )


def test_the_cli_entry_point_upgrades_and_reports_status(url: str) -> None:
    """The exact commands the migrate step and the stack run, in a real subprocess."""
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
    admin.dispose()

    behind = _cli("status", "--plane", "operator", url=url)
    assert behind.returncode == 2 and "current none" in behind.stdout, behind.stderr

    up = _cli("upgrade", "--plane", "operator", url=url)
    assert (
        up.returncode == 0
        and "plane operator: 0002_schema_history (head 0002_schema_history)" in up.stdout
    )

    player = _cli("status", "--plane", "player", url=url)
    assert (
        player.returncode == 0
        and "current 0002_schema_history, head 0002_schema_history" in player.stdout
    )


def test_the_cli_fails_loudly_with_no_database_url() -> None:
    env = {k: v for k, v in os.environ.items() if k != "PODCAST_DATABASE_URL"}
    env["PYTHONPATH"] = str(REPO / "src")
    done = subprocess.run(
        [sys.executable, "-m", "podcast_scraper.cli", "db", "status", "--plane", "player"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )
    assert done.returncode == 1 and "PODCAST_DATABASE_URL" in done.stderr


# --- db upgrade --sql: the dry run -------------------------------------------------------------


def _sql(*args: str, url: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "PODCAST_DATABASE_URL"}
    env["PYTHONPATH"] = str(REPO / "src")
    if url:
        env["PODCAST_DATABASE_URL"] = url
    return subprocess.run(
        [sys.executable, "-m", "podcast_scraper.cli", "db", "upgrade", "--sql", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        check=False,
    )


def test_sql_for_a_fresh_plane_creates_the_plane_tables_and_nothing_in_public() -> None:
    done = _sql("--plane", "player", url=None)
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert re.search(r'CREATE TABLE "?player"?\.users \(', out)
    assert re.search(r'CREATE TABLE "?player"?\.alembic_version \(', out)
    assert re.search(r'INSERT INTO "?player"?\.alembic_version', out)
    assert "CREATE SCHEMA" not in out.upper()
    for statement in re.findall(r"CREATE TABLE ([^ (]+)", out):
        assert statement.strip('"').startswith("player."), statement
    assert "public" not in out


def test_sql_against_a_plane_at_head_emits_no_ddl(url: str) -> None:
    migrate.upgrade(url, "player")
    done = _sql("--plane", "player", url=url)
    assert done.returncode == 0, done.stderr
    assert "nothing to run" in done.stdout
    assert not re.search(r"CREATE|ALTER|INSERT|DROP", done.stdout)


def test_sql_starts_from_the_plane_current_revision_and_runs_nothing(url: str) -> None:
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
            conn.execute(text("CREATE SCHEMA operator"))
        done = _sql("--plane", "operator", url=url)
        assert done.returncode == 0, done.stderr
        assert "base -> " in done.stdout and "CREATE TABLE" in done.stdout
        with admin.connect() as conn:
            tables = conn.execute(
                text("SELECT count(*) FROM pg_tables WHERE schemaname = 'operator'")
            ).scalar()
        assert tables == 0, "--sql must execute nothing"
        # --from overrides the database: from head there is nothing to run.
        head = migrate.head_revision()
        again = _sql("--plane", "operator", "--from", head, url=url)
        assert "nothing to run" in again.stdout
    finally:
        with admin.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
        admin.dispose()
        migrate.upgrade(url, "operator")


def test_the_emitted_sql_builds_the_same_table_the_upgrade_does(url: str) -> None:
    """The offline render is applied to an empty database and compared with the online result."""
    done = _sql("--plane", "player", url=None)
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    name = f"sqlcheck_{os.getpid()}"
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
            conn.execute(text(f'CREATE DATABASE "{name}"'))
        target = make_url(normalize_url(url)).set(database=name)
        scratch = create_engine(target, isolation_level="AUTOCOMMIT")
        with scratch.connect() as conn:
            conn.execute(text("CREATE SCHEMA player"))
            raw = conn.connection.dbapi_connection
            assert raw is not None
            raw.execute(done.stdout)
            columns_sql = conn.execute(
                text(
                    "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                    "WHERE table_schema = 'player' AND table_name = 'users' ORDER BY column_name"
                )
            ).all()
        scratch.dispose()
        live = create_engine(normalize_url(url))
        with live.connect() as conn:
            columns_live = conn.execute(
                text(
                    "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                    "WHERE table_schema = 'player' AND table_name = 'users' ORDER BY column_name"
                )
            ).all()
        live.dispose()
        assert columns_sql == columns_live and columns_sql
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


# --- schema_history: what each upgrade applied --------------------------------------------------


def _fresh_operator(url: str) -> None:
    admin = create_engine(normalize_url(url), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
    finally:
        admin.dispose()


def test_an_upgrade_from_empty_writes_a_history_row_per_revision_and_a_second_writes_none(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CL_RELEASE", "b5a243d")
    monkeypatch.setenv("PODCAST_RELEASE", "sha-f493b65")
    _fresh_operator(url)
    try:
        applied = migrate.upgrade(url, "operator")
        assert applied == ["0001_users", "0002_schema_history"]
        rows = migrate.history_rows(url, "operator")
        assert [r["revision"] for r in rows] == ["0002_schema_history", "0001_users"]
        assert {r["release"] for r in rows} == {"b5a243d"}
        assert {r["release_source"] for r in rows} == {"CL_RELEASE"}
        assert all(isinstance(r["duration_ms"], int) and r["duration_ms"] >= 0 for r in rows)
        assert all(r["applied_at"] is not None for r in rows)

        assert migrate.upgrade(url, "operator") == []
        assert len(migrate.history_rows(url, "operator")) == 2
    finally:
        migrate.upgrade(url, "operator")


def test_the_history_falls_back_to_the_image_release_and_names_it(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CL_RELEASE", raising=False)
    monkeypatch.setenv("PODCAST_RELEASE", "sha-f493b65")
    _fresh_operator(url)
    migrate.upgrade(url, "operator")
    row, _ = migrate.history_rows(url, "operator", limit=2)
    assert (row["release"], row["release_source"]) == ("sha-f493b65", "PODCAST_RELEASE")


def test_an_upgrade_with_no_release_set_records_null(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CL_RELEASE", raising=False)
    monkeypatch.delenv("PODCAST_RELEASE", raising=False)
    _fresh_operator(url)
    migrate.upgrade(url, "operator")
    row, *_ = migrate.history_rows(url, "operator")
    assert (row["release"], row["release_source"]) == (None, None)


def test_an_upgrade_that_stops_before_the_history_table_writes_no_rows(url: str) -> None:
    _fresh_operator(url)
    try:
        assert migrate.upgrade(url, "operator", "0001_users") == ["0001_users"]
        assert migrate.history_rows(url, "operator") == []
    finally:
        migrate.upgrade(url, "operator")


def test_the_offline_sql_creates_the_history_table_and_writes_no_rows() -> None:
    done = _sql("--plane", "player", url=None)
    assert done.returncode == 0, done.stderr
    assert re.search(r'CREATE TABLE "?player"?\.schema_history \(', done.stdout)
    assert not re.search(r"INSERT INTO \"?player\"?\.schema_history", done.stdout)


def test_status_prints_the_last_history_rows(url: str) -> None:
    _fresh_operator(url)
    try:
        assert _cli("upgrade", "--plane", "operator", url=url).returncode == 0
        out = _cli("status", "--plane", "operator", url=url)
        assert out.returncode == 0, out.stderr
        assert "0002_schema_history" in out.stdout and "0001_users" in out.stdout
        assert " ms" in out.stdout
    finally:
        migrate.upgrade(url, "operator")


def test_status_on_a_plane_before_the_history_table_prints_no_rows(url: str) -> None:
    _fresh_operator(url)
    try:
        migrate.upgrade(url, "operator", "0001_users")
        out = _cli("status", "--plane", "operator", url=url)
        assert out.returncode == 2 and "current 0001_users" in out.stdout
        assert " ms" not in out.stdout
    finally:
        migrate.upgrade(url, "operator")
