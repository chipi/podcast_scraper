"""A lone SELECT on a hot path costs the database no BEGIN and no ROLLBACK.

Prod (2026-11-06): ``pg_stat_statements`` counted 6,194 BEGINs in 5.4 h, nearly all read-only
requests; psycopg sends BEGIN, the statement and ROLLBACK as three round trips. The per-request
reads (the session's user, preferences, comms, the app releases) are single statements, so they run
in autocommit; a write stays a transaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.db import read_connection, ro_engine
from podcast_scraper.server import (
    app_comms_store,
    app_release_store,
    app_user_preferences,
    app_user_store,
)
from tests._database import captured_statements

pytestmark = pytest.mark.unit

_NO_DIR = Path(".")


def test_single_statement_reads_run_in_autocommit() -> None:
    user = app_user_store.get_or_create_user(
        _NO_DIR, provider="stub", subject="s1", email="j@x.com", name="J"
    )
    app_release_store.clear_cache()
    with captured_statements() as seen:
        assert app_user_store.get_user(_NO_DIR, user.user_id) is not None
        app_user_preferences.get_preferences(_NO_DIR, user.user_id)
        app_comms_store.get_comms(_NO_DIR, user.user_id)
        app_release_store.load_released_versions(_NO_DIR)
    assert len(seen) == 4
    assert all(autocommit for _, autocommit in seen), seen


def test_a_write_is_still_a_transaction() -> None:
    user = app_user_store.get_or_create_user(
        _NO_DIR, provider="stub", subject="s1", email="j@x.com", name="J"
    )
    with captured_statements() as seen:
        app_user_preferences.replace_preferences(_NO_DIR, user.user_id, {"a": 1})
    assert [autocommit for _, autocommit in seen] == [False]


def test_autocommit_does_not_lift_the_read_only_guard() -> None:
    from sqlalchemy import text
    from sqlalchemy.exc import DBAPIError

    with read_connection(ro_engine()) as conn:
        assert conn.execute(text("SHOW default_transaction_read_only")).scalar() == "on"
        with pytest.raises(DBAPIError, match="read-only"):
            conn.execute(text("DELETE FROM shared.app_releases"))


def test_the_pooled_connection_goes_back_in_transaction_mode() -> None:
    from sqlalchemy import text

    with read_connection(ro_engine()) as conn:
        conn.execute(text("SELECT 1"))
    with captured_statements() as seen:
        with ro_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    assert [autocommit for _, autocommit in seen] == [False]
