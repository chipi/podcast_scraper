"""Reads never open a server transaction, so they never end in a ROLLBACK.

A read closed by ``engine.connect()`` without a commit rolls back, and on prod those reads were 98%
of ``pg_stat_database.xact_rollback`` (about 26 a minute, 0 errors in the server log): the counter
counted reads instead of failures. Every read goes through ``read_connection`` (autocommit).

SQLAlchemy's own ``begin`` / ``rollback`` events fire on an autocommit connection too, without
sending anything, so the guard asks the server side instead: after each statement on the ro engine,
psycopg reports whether the server holds an open transaction (``INTRANS``) or none (``IDLE``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from sqlalchemy import event

from podcast_scraper.db import ro_engine
from podcast_scraper.server import (
    app_access_store,
    app_account_deletion,
    app_audit,
    app_graph_telemetry,
    app_outbox_store,
    app_push_store,
    app_user_store,
)
from tests._database import worker_database_url

USER = "u_" + "0" * 24


@pytest.fixture
def in_transaction() -> Iterator[list[str]]:
    """The statements on the ro engine that ran inside an open server transaction."""
    assert worker_database_url(), "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    found: list[str] = []

    def after(conn: Any, _cursor: Any, statement: str, *_args: Any) -> None:
        status = conn.connection.dbapi_connection.info.transaction_status
        if status == psycopg.pq.TransactionStatus.INTRANS:
            found.append(statement)

    engine = ro_engine()
    event.listen(engine, "after_cursor_execute", after)
    try:
        yield found
    finally:
        event.remove(engine, "after_cursor_execute", after)


READS = {
    "load_policy": lambda d: app_access_store.load_policy(d),
    "read_audit": lambda d: app_audit.read_audit(None),
    "read_events": lambda d: app_graph_telemetry.read_events(d, USER),
    "read_all_events": lambda d: app_graph_telemetry.read_all_events(d),
    "users_with_events": lambda d: app_graph_telemetry.users_with_events(d),
    "apple_refresh_token": lambda d: app_account_deletion.apple_refresh_token(d, USER),
    "list_pending": lambda d: app_outbox_store.list_pending(d, channel="email"),
    "list_subscriptions": lambda d: app_push_store.list_subscriptions(d, USER),
    "list_users": lambda d: app_user_store.list_users(d),
    "list_user_ids": lambda d: app_user_store.list_user_ids(d),
}


@pytest.mark.parametrize("name", sorted(READS))
def test_a_read_opens_no_server_transaction(
    name: str, tmp_path: Path, in_transaction: list[str]
) -> None:
    executed: list[str] = []

    def record(*args: Any) -> None:
        executed.append(args[2])

    engine = ro_engine()
    event.listen(engine, "after_cursor_execute", record)
    try:
        READS[name](tmp_path)
    finally:
        event.remove(engine, "after_cursor_execute", record)
    assert executed, "the read never reached the database, so it tested nothing"
    assert in_transaction == []


def test_the_guard_sees_a_read_that_would_end_in_rollback(in_transaction: list[str]) -> None:
    """The pattern the reads moved off: a server transaction, closed by a ROLLBACK."""
    with ro_engine().connect() as conn:
        conn.exec_driver_sql("SELECT 1")
    assert in_transaction == ["SELECT 1"]
