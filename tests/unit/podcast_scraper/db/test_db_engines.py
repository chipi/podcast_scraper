"""The engine registry and migrations, against the test database (ADR-169)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import exc as sa_exc, select, text

from podcast_scraper.db import (
    engines as engines_mod,
    init_engines,
    migrate,
    plane_schema,
    read_with_retry,
    reset_engines,
    ro_engine,
    rw_engine,
)
from podcast_scraper.db.engines import DatabaseNotReady
from podcast_scraper.db.schema import users
from podcast_scraper.db.settings import DatabaseNotConfigured, DatabaseSettings
from tests._database import worker_database_url


@pytest.fixture
def url() -> str:
    found = worker_database_url()
    assert found, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    return found


def test_engines_are_opened_once_and_carry_the_plane(url: str) -> None:
    assert rw_engine() is rw_engine()
    assert ro_engine() is ro_engine()
    assert rw_engine() is not ro_engine()
    assert plane_schema() == "player"
    with rw_engine().connect() as conn:
        # Kernel tables carry no schema; the engine maps them onto the plane.
        assert conn.execute(select(users.c.user_id)).all() == []
        assert conn.execute(text("SELECT to_regclass('player.users')")).scalar() is not None


def test_a_second_init_is_a_no_op(url: str) -> None:
    first = rw_engine()
    init_engines(DatabaseSettings(rw_url=url, ro_url=url, plane="operator"))
    assert rw_engine() is first
    assert plane_schema() == "player"


def test_the_ro_engine_refuses_writes(url: str) -> None:
    with pytest.raises(sa_exc.InternalError):
        with ro_engine().begin() as conn:
            conn.execute(text("CREATE TEMP TABLE should_not_exist (x int)"))


def test_no_url_fails_with_the_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_engines()
    monkeypatch.delenv("PODCAST_DATABASE_URL", raising=False)
    with pytest.raises(DatabaseNotConfigured):
        rw_engine()


def test_readiness_fails_fast_on_an_unreachable_database() -> None:
    reset_engines()
    dead = "postgresql://nobody:nobody@127.0.0.1:1/none"
    with pytest.raises(DatabaseNotReady, match="not reachable"):
        init_engines(DatabaseSettings.from_env({"PODCAST_DATABASE_URL": dead}))
    assert not engines_mod.engines_open()


def test_readiness_fails_fast_on_an_unmigrated_plane(url: str) -> None:
    reset_engines()
    with rw_engine_for(url).begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
    settings = DatabaseSettings(rw_url=url, ro_url=url, plane="operator")
    with pytest.raises(DatabaseNotReady, match="db upgrade --plane operator"):
        init_engines(settings)
    migrate.upgrade(url, "operator")
    init_engines(settings)
    assert plane_schema() == "operator"


def rw_engine_for(url: str):
    from sqlalchemy import create_engine

    from podcast_scraper.db.settings import normalize_url

    return create_engine(normalize_url(url))


def test_upgrade_is_idempotent_and_status_reports_head(url: str) -> None:
    migrate.upgrade(url, "player")
    migrate.upgrade(url, "player")
    current, head = migrate.status(url, "player")
    assert current == head


def test_status_of_a_plane_never_upgraded_is_none(url: str) -> None:
    eng = rw_engine_for(url)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
    try:
        current, head = migrate.status(url, "operator")
        assert current is None and head
    finally:
        eng.dispose()
        migrate.upgrade(url, "operator")


def test_read_with_retry_retries_a_dropped_connection_then_gives_up(url: str) -> None:
    calls: list[int] = []

    def flaky(_engine: object) -> str:
        calls.append(1)
        if len(calls) < 3:
            raise sa_exc.OperationalError("SELECT 1", {}, Exception("server closed the connection"))
        return "ok"

    assert read_with_retry(flaky, backoff_s=0) == "ok"
    assert len(calls) == 3

    def always(_engine: object) -> None:
        raise sa_exc.OperationalError("SELECT 1", {}, Exception("down"))

    with pytest.raises(sa_exc.OperationalError):
        read_with_retry(always, attempts=2, backoff_s=0)


def test_read_with_retry_does_not_retry_a_query_error(url: str) -> None:
    calls: list[int] = []

    def broken(_engine: object) -> None:
        calls.append(1)
        raise sa_exc.ProgrammingError("SELECT nope", {}, Exception("bad sql"))

    with pytest.raises(sa_exc.ProgrammingError):
        read_with_retry(broken, backoff_s=0)
    assert calls == [1]


# --- the users table's constraints ------------------------------------------------------------


def _insert_user(user_id: str, **fields: object) -> None:
    from sqlalchemy import insert

    values = {"provider": "test", "subject": user_id, **fields}
    with rw_engine().begin() as conn:
        conn.execute(insert(users).values(user_id=user_id, **values))


def test_a_role_outside_the_vocabulary_is_rejected(url: str) -> None:
    with pytest.raises(sa_exc.IntegrityError, match="users_role_check"):
        _insert_user("u_" + "1" * 24, role="wizard")
    for role in ("listener", "creator", "admin"):
        _insert_user("u_" + role[0] * 24, role=role)


def test_the_role_check_lists_exactly_the_roles_in_app_roles() -> None:
    from sqlalchemy import CheckConstraint

    from podcast_scraper.server import app_roles

    (check,) = [c for c in users.constraints if isinstance(c, CheckConstraint)]
    import re

    assert tuple(re.findall(r"'(\w+)'", str(check.sqltext))) == app_roles.ROLES
    migration = (
        Path(migrate.__file__).parent / "migrations" / "versions" / "0001_users.py"
    ).read_text()
    listed = re.search(r"role IN \(([^)]*)\)", migration)
    assert listed is not None
    assert tuple(re.findall(r"'(\w+)'", listed.group(1))) == app_roles.ROLES


def test_a_duplicate_analytics_id_is_rejected(url: str) -> None:
    _insert_user("u_" + "a" * 24, analytics_id="same")
    with pytest.raises(sa_exc.IntegrityError, match="users_analytics_id_key"):
        _insert_user("u_" + "b" * 24, analytics_id="same")


def test_two_null_analytics_ids_are_allowed(url: str) -> None:
    _insert_user("u_" + "a" * 24)
    _insert_user("u_" + "b" * 24)
    with rw_engine().connect() as conn:
        assert conn.execute(select(users.c.analytics_id)).scalars().all() == [None, None]


def test_one_account_is_one_provider_subject_pair(url: str) -> None:
    _insert_user("u_" + "a" * 24, provider="google", subject="s")
    with pytest.raises(sa_exc.IntegrityError, match="users_provider_subject_key"):
        _insert_user("u_" + "b" * 24, provider="google", subject="s")


def test_email_is_not_unique(url: str) -> None:
    _insert_user("u_" + "a" * 24, email="same@x.com")
    _insert_user("u_" + "b" * 24, email="same@x.com")
