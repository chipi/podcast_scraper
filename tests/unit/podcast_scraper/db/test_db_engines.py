"""The engine registry and migrations, against the test database (ADR-169)."""

from __future__ import annotations

import logging
import time
import uuid
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


def _set_operator_revision(url: str, revision: str, *, recorded: bool) -> None:
    with rw_engine_for(url).begin() as conn:
        conn.execute(text("UPDATE operator.alembic_version SET version_num = :r"), {"r": revision})
        if recorded:
            conn.execute(
                text("INSERT INTO operator.schema_history (revision) VALUES (:r)"), {"r": revision}
            )


@pytest.fixture
def fresh_operator(url: str):
    reset_engines()
    with rw_engine_for(url).begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
    migrate.upgrade(url, "operator")
    yield DatabaseSettings(rw_url=url, ro_url=url, plane="operator")
    reset_engines()
    with rw_engine_for(url).begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS operator CASCADE"))
    migrate.upgrade(url, "operator")


def test_readiness_accepts_a_database_ahead_of_the_code(fresh_operator: DatabaseSettings) -> None:
    # A later release migrated past this code's head (the rollback case, ADR-172).
    _set_operator_revision(fresh_operator.rw_url, "9999_from_a_later_release", recorded=True)
    init_engines(fresh_operator)
    assert plane_schema() == "operator"


def test_readiness_refuses_a_database_behind_the_code(fresh_operator: DatabaseSettings) -> None:
    _set_operator_revision(fresh_operator.rw_url, "0001_users", recorded=False)
    with pytest.raises(DatabaseNotReady, match="behind"):
        init_engines(fresh_operator)


def test_readiness_refuses_a_revision_not_descended_from_the_head(
    fresh_operator: DatabaseSettings,
) -> None:
    # Unknown to the code and the code's head never applied here: not a later release of ours.
    with rw_engine_for(fresh_operator.rw_url).begin() as conn:
        conn.execute(text("DELETE FROM operator.schema_history"))
    _set_operator_revision(fresh_operator.rw_url, "9999_somewhere_else", recorded=False)
    with pytest.raises(DatabaseNotReady, match="does not know"):
        init_engines(fresh_operator)


def test_readiness_states(fresh_operator: DatabaseSettings) -> None:
    eng = rw_engine_for(fresh_operator.rw_url)
    head = migrate.head_revision()
    with eng.connect() as conn:
        assert migrate.readiness(conn, "operator") == ("at_head", head, head)
    _set_operator_revision(fresh_operator.rw_url, "9999_later", recorded=True)
    with eng.connect() as conn:
        assert migrate.readiness(conn, "operator") == ("ahead", "9999_later", head)


def test_readiness_without_read_on_the_history_is_unknown_not_a_crash(
    fresh_operator: DatabaseSettings,
) -> None:
    # Prod's plane roles read shared.alembic_version but not shared.schema_history (roles.sql).
    _set_operator_revision(fresh_operator.rw_url, "9999_later", recorded=True)
    with rw_engine_for(fresh_operator.rw_url).connect() as conn:
        conn.execute(text("DROP ROLE IF EXISTS readiness_probe"))
        conn.execute(text("CREATE ROLE readiness_probe"))
        conn.execute(text("GRANT USAGE ON SCHEMA operator TO readiness_probe"))
        conn.execute(text("GRANT SELECT ON operator.alembic_version TO readiness_probe"))
        conn.execute(text("SET ROLE readiness_probe"))
        try:
            assert migrate.readiness(conn, "operator")[0] == "unknown"
        finally:
            conn.execute(text("RESET ROLE"))
            conn.execute(text("DROP OWNED BY readiness_probe"))
            conn.execute(text("DROP ROLE readiness_probe"))
            conn.commit()


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


def test_a_failed_statement_does_not_carry_its_parameters(url: str) -> None:
    bound_value = "token-hash-that-must-not-be-logged"
    with pytest.raises(sa_exc.DBAPIError) as caught:
        with rw_engine().begin() as conn:
            conn.execute(text("SELECT CAST(:v AS text), 1 / 0"), {"v": bound_value})
    assert bound_value not in str(caught.value)


def test_connect_args_combine_the_timeouts_with_read_only() -> None:
    cfg = DatabaseSettings(rw_url="x", ro_url="x", statement_timeout_ms=1234, idle_tx_timeout_ms=5)
    rw = engines_mod.app_connect_args(cfg, read_only=False)
    ro = engines_mod.app_connect_args(cfg, read_only=True)
    assert rw["options"] == "-c statement_timeout=1234 -c idle_in_transaction_session_timeout=5"
    assert ro["options"] == f"{rw['options']} -c default_transaction_read_only=on"
    assert rw["prepare_threshold"] == 5
    off = DatabaseSettings(rw_url="x", ro_url="x", prepare_threshold=None)
    assert engines_mod.app_connect_args(off, read_only=False)["prepare_threshold"] is None


def test_every_app_engine_carries_the_timeouts_and_the_migration_engine_none(url: str) -> None:
    for engine in (rw_engine(), ro_engine()):
        with engine.connect() as conn:
            assert conn.execute(text("SHOW statement_timeout")).scalar() == "15s"
            assert conn.execute(text("SHOW idle_in_transaction_session_timeout")).scalar() == "1min"
    with ro_engine().connect() as conn:
        assert conn.execute(text("SHOW default_transaction_read_only")).scalar() == "on"
    with rw_engine().connect() as conn:
        assert conn.execute(text("SHOW default_transaction_read_only")).scalar() == "off"
    migration = migrate._engine(url)
    try:
        with migration.connect() as conn:
            assert conn.execute(text("SHOW statement_timeout")).scalar() == "0"
            assert conn.execute(text("SHOW idle_in_transaction_session_timeout")).scalar() == "0"
    finally:
        migration.dispose()


def _tight_engine(url: str, **overrides):
    cfg = DatabaseSettings(rw_url=url, ro_url=url, **overrides)
    return engines_mod._make_engine(cfg, url, read_only=False)


def test_a_statement_over_the_timeout_is_a_query_canceled_the_error_grouping_keys_on(
    url: str,
) -> None:
    from podcast_scraper.utils.sentry_init import _group_database_error

    engine = _tight_engine(url, statement_timeout_ms=100)
    try:
        with pytest.raises(sa_exc.OperationalError) as caught:
            with engine.connect() as conn:
                conn.execute(text("SELECT pg_sleep(2)"))
        driver = caught.value.orig
        assert driver is not None
        assert getattr(driver, "sqlstate", None) == "57014"
        event: dict = {}
        _group_database_error(event, caught.value, driver)
        assert event["fingerprint"] == ["database", "QueryCanceled", "57014"]
    finally:
        engine.dispose()


def test_a_session_idle_in_a_transaction_past_the_timeout_is_terminated(url: str) -> None:
    engine = _tight_engine(url, idle_tx_timeout_ms=200)
    try:
        with pytest.raises(sa_exc.DBAPIError):
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                time.sleep(0.8)
                conn.execute(text("SELECT 1"))
    finally:
        engine.dispose()


def test_a_starved_pool_fails_within_the_pool_timeout(url: str) -> None:
    engine = _tight_engine(url, pool_size=1, max_overflow=0, pool_timeout_s=0.5)
    try:
        with engine.connect():
            started = time.monotonic()
            with pytest.raises(sa_exc.TimeoutError, match="timed out"):
                engine.connect()
            assert time.monotonic() - started < 3
    finally:
        engine.dispose()


def test_the_pool_size_comes_from_the_settings(url: str) -> None:
    engine = _tight_engine(url, pool_size=2, max_overflow=1)
    try:
        assert (engine.pool.size(), engine.pool._max_overflow) == (2, 1)
    finally:
        engine.dispose()


@pytest.fixture
def role_with_default_timeouts(url: str):
    """A login role carrying role-level timeouts, as infra sets on the plane roles."""
    from sqlalchemy.engine import make_url

    name = f"tmo_probe_{uuid.uuid4().hex[:10]}"
    admin = rw_engine_for(url)
    try:
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            try:
                conn.execute(text(f"CREATE ROLE {name} LOGIN PASSWORD 'probe'"))
            except sa_exc.ProgrammingError:
                pytest.skip("the test database user cannot create roles")
            conn.execute(text(f"ALTER ROLE {name} SET statement_timeout = '15s'"))
            conn.execute(
                text(f"ALTER ROLE {name} SET idle_in_transaction_session_timeout = '1min'")
            )
        yield make_url(url).set(username=name, password="probe").render_as_string(
            hide_password=False
        )
    finally:
        with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"DROP ROLE IF EXISTS {name}"))
        admin.dispose()


def _show_timeouts(engine) -> tuple[str, str]:
    with engine.connect() as conn:
        return (
            conn.execute(text("SHOW statement_timeout")).scalar(),
            conn.execute(text("SHOW idle_in_transaction_session_timeout")).scalar(),
        )


def test_the_role_default_applies_to_a_plain_connection(role_with_default_timeouts: str) -> None:
    from sqlalchemy import create_engine

    engine = create_engine(role_with_default_timeouts)
    try:
        assert _show_timeouts(engine) == ("15s", "1min")
    finally:
        engine.dispose()


def test_the_migration_engine_overrides_a_role_default(role_with_default_timeouts: str) -> None:
    engine = migrate._engine(role_with_default_timeouts)
    try:
        assert _show_timeouts(engine) == ("0", "0")
    finally:
        engine.dispose()


def test_the_importer_connection_overrides_a_role_default(
    role_with_default_timeouts: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse
    from types import SimpleNamespace

    from podcast_scraper.db import cli_handlers

    seen: list[tuple[str, str]] = []

    class _Importer:
        def run(self, data_dir, engine, *, dry_run):
            seen.append(_show_timeouts(engine))
            return SimpleNamespace(
                store="probe", read=0, inserted=0, already_present=0, mismatches=[]
            )

    monkeypatch.setattr(cli_handlers, "registry", lambda plane: {"probe": _Importer()})
    monkeypatch.setenv("PODCAST_DATABASE_URL", role_with_default_timeouts)
    args = argparse.Namespace(from_files=tmp_path, plane="player", only=None, dry_run=True)
    cli_handlers._import(args, logging.getLogger("test"))
    assert seen == [("0", "0")]
