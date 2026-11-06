"""What the database tells observability (ADR-169): connections named, errors grouped by kind.

Prod, 2026-11-05: two probe reads of a missing table from player-api landed in GlitchTip as two
issues (PLAYER-API-4, -5), because a database error's text carries its whole statement. And every
connection was anonymous in ``pg_stat_activity``.
"""

from __future__ import annotations

import sys

import pytest
from sqlalchemy import create_engine, exc as sa_exc, text

from podcast_scraper import build_info
from podcast_scraper.db import init_engines, reset_engines, rw_engine
from podcast_scraper.db.engines import application_name_args
from podcast_scraper.db.settings import DatabaseSettings, normalize_url
from podcast_scraper.utils.sentry_init import _before_send
from tests._database import worker_database_url


@pytest.fixture
def url() -> str:
    found = worker_database_url()
    assert found, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    return found


def _event_for(exc: BaseException) -> dict:
    """The event the SDK would build for ``exc``: its type and str as the exception value."""
    return {
        "level": "error",
        "exception": {"values": [{"type": type(exc).__name__, "value": str(exc)}]},
    }


def _raise(sql: str) -> sa_exc.DBAPIError:
    with pytest.raises(sa_exc.DBAPIError) as caught:
        with rw_engine().connect() as conn:
            conn.execute(text(sql), {"secret_param": "alice@example.com"})
    return caught.value


def test_one_kind_of_database_failure_is_one_issue(url: str) -> None:
    first = _raise("SELECT * FROM player.no_such_table WHERE email = :secret_param")
    second = _raise("SELECT count(*) FROM player.no_such_table")
    events = [
        _before_send(_event_for(exc), {"exc_info": (type(exc), exc, None)})
        for exc in (first, second)
    ]
    assert events[0]["fingerprint"] == events[1]["fingerprint"]
    assert events[0]["fingerprint"] == ["database", "UndefinedTable", "42P01"]


def test_no_sql_and_no_parameters_leave_the_app(url: str) -> None:
    exc = _raise("SELECT * FROM player.no_such_table WHERE email = :secret_param")
    # The engine hides parameters (hide_parameters); the SQL still reaches the exception text,
    # so the event scrubber is what keeps it in the app.
    assert "SELECT" in str(exc) and "alice@example.com" not in str(exc)
    event = _before_send(_event_for(exc), {"exc_info": (type(exc), exc, None)})
    value = event["exception"]["values"][0]["value"]
    assert value == '[42P01] relation "player.no_such_table" does not exist'


def test_different_kinds_stay_different(url: str) -> None:
    missing = _raise("SELECT * FROM player.no_such_table")
    syntax = _raise("SELEC 1")
    a, b = (
        _before_send(_event_for(e), {"exc_info": (type(e), e, None)})["fingerprint"]
        for e in (missing, syntax)
    )
    assert a != b


def test_a_non_database_error_keeps_the_message_grouping() -> None:
    err = ValueError("bad input 42")
    event = _before_send(_event_for(err), {"exc_info": (ValueError, err, None)})
    assert event["fingerprint"][0] == "ValueError"
    assert event["exception"]["values"][0]["value"] == "bad input 42"


def test_a_connection_is_named_for_its_container(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SERVICE_NAME", "podcast-pipeline")
    monkeypatch.setenv("PODCAST_DB_APPLICATION_NAME", "player-board-summarizer")
    reset_engines()
    init_engines(DatabaseSettings(rw_url=url, ro_url=url))
    with rw_engine().connect() as conn:
        name = conn.execute(text("SELECT current_setting('application_name')")).scalar()
    assert name == "player-board-summarizer"


def test_the_otel_name_is_the_fallback_and_nothing_is_sent_without_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PODCAST_DB_APPLICATION_NAME", raising=False)
    monkeypatch.setenv("OTEL_SERVICE_NAME", "player-api")
    assert application_name_args() == {"application_name": "player-api"}
    monkeypatch.delenv("OTEL_SERVICE_NAME")
    assert application_name_args() == {}


def test_the_migrate_engine_is_named_too(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODCAST_DB_APPLICATION_NAME", "db-migrate-player")
    engine = create_engine(normalize_url(url), connect_args=application_name_args())
    try:
        with engine.connect() as conn:
            assert (
                conn.execute(text("SELECT current_setting('application_name')")).scalar()
                == "db-migrate-player"
            )
    finally:
        engine.dispose()


def test_the_release_is_the_deploys_release_id(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentry_sdk as sdk

    seen: dict[str, object] = {}
    monkeypatch.setattr(sdk, "init", lambda **kw: seen.update(kw))
    monkeypatch.setenv("PODCAST_SENTRY_DSN_API", "https://k@example.invalid/1")
    monkeypatch.setenv("PODCAST_RELEASE", "sha-0c51613")
    monkeypatch.setenv("CL_RELEASE", "f5a22db")
    from podcast_scraper.utils.sentry_init import init_sentry

    init_sentry("api")
    assert seen["release"] == "f5a22db"
    monkeypatch.delenv("CL_RELEASE")
    init_sentry("api")
    assert seen["release"] == "sha-0c51613"
    assert build_info.release() == ("sha-0c51613", "PODCAST_RELEASE")
    assert "sentry_sdk" in sys.modules
