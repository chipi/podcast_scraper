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
        assert not re.search(r"create\s+schema|\bgrant\b|\bschema\s*=", source, re.I), name


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
    assert up.returncode == 0 and "plane operator: 0001_users (head 0001_users)" in up.stdout

    player = _cli("status", "--plane", "player", url=url)
    assert player.returncode == 0 and "current 0001_users, head 0001_users" in player.stdout


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
