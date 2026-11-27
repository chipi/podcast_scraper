"""The news re-check's rows (CRFC-006 G7, n-w9): a source's window, an item's state, what is due."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import exc as sa_exc, insert, select
from sqlalchemy.engine import Connection

from podcast_scraper.db import news_recheck as rows
from podcast_scraper.db.schema import content_items, news_recheck
from podcast_scraper.news import recheck
from podcast_scraper.news.recheck_fetch import Validators
from tests.unit.podcast_scraper.db.test_db_corpus import (  # noqa: F401
    _document,
    _item,
    _source,
    conn,
    url,
)

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=timezone.utc)


def _article(
    conn: Connection, source_id: int, n: str, fetched: datetime, served: int = 1  # noqa: F811
) -> int:
    item_id = _item(conn, source_id, n, content_type="news_article", served=served)
    _document(conn, item_id, kind="article", variant="raw", fetched_at=fetched)
    return item_id


def _news_source(conn: Connection, natural_id: str = "source:news") -> int:  # noqa: F811
    return _source(conn, natural_id, content_type="news_article")


def test_a_source_with_no_window_row_is_never_rechecked(conn: Connection) -> None:  # noqa: F811
    source_id = _news_source(conn)
    _article(conn, source_id, "1", NOW - timedelta(days=1))
    assert rows.enrol(conn, NOW) == 0
    assert rows.due(conn, NOW + timedelta(days=30)) == []
    assert conn.execute(select(news_recheck)).first() is None


def test_a_window_of_zero_is_off(conn: Connection) -> None:  # noqa: F811
    source_id = _news_source(conn)
    _article(conn, source_id, "1", NOW - timedelta(days=1))
    rows.set_window(conn, source_id, 0)
    assert rows.enrol(conn, NOW) == 0


def test_the_schedule_is_the_windows_share_of_the_default_days() -> None:
    assert rows.schedule(0) == ()
    assert rows.schedule(3) == (1, 2)
    assert rows.schedule(7) == recheck.DEFAULT_DAYS
    assert rows.schedule(30) == recheck.DEFAULT_DAYS


def test_a_served_article_inside_its_window_is_enrolled_once_and_due_on_day_one(
    conn: Connection,  # noqa: F811
) -> None:
    source_id = _news_source(conn)
    fetched = NOW - timedelta(hours=30)
    item_id = _article(conn, source_id, "1", fetched)
    rows.set_window(conn, source_id, 7)
    assert rows.enrol(conn, NOW) == 1
    assert rows.enrol(conn, NOW) == 0
    (found,) = rows.due(conn, NOW)
    assert (found.item_id, found.source_id, found.natural_id) == (item_id, source_id, "item:1")
    assert found.state == recheck.RecheckState(
        first_fetched_at=fetched,
        next_due=fetched + timedelta(days=1),
        days=recheck.DEFAULT_DAYS,
    )
    assert found.validators == Validators()


def test_an_item_past_its_window_unserved_or_not_news_is_not_enrolled(
    conn: Connection,  # noqa: F811
) -> None:
    source_id = _news_source(conn)
    _article(conn, source_id, "old", NOW - timedelta(days=8))
    _article(conn, source_id, "hidden", NOW - timedelta(days=1), served=0)
    # A served item of another type is not re-checked, even with an article-kind document.
    podcast = _item(conn, source_id, "pod", served=1)
    _document(conn, podcast, kind="article", variant="raw", fetched_at=NOW)
    _item(conn, source_id, "unfetched", content_type="news_article", served=1)
    rows.set_window(conn, source_id, 7)
    assert rows.enrol(conn, NOW) == 0


def test_the_first_fetch_is_the_items_earliest_article_document(
    conn: Connection,  # noqa: F811
) -> None:
    source_id = _news_source(conn)
    first = NOW - timedelta(days=2)
    item_id = _article(conn, source_id, "1", first)
    _document(conn, item_id, kind="article", variant="cleaned", fetched_at=NOW)
    # Only article documents place the window.
    _document(conn, item_id, kind="transcript", variant="raw", fetched_at=NOW - timedelta(days=9))
    rows.set_window(conn, source_id, 7)
    rows.enrol(conn, NOW)
    (found,) = rows.due(conn, NOW)
    assert found.state.first_fetched_at == first


def test_a_recorded_check_moves_the_item_to_its_next_day_and_keeps_its_validators(
    conn: Connection,  # noqa: F811
) -> None:
    source_id = _news_source(conn)
    fetched = NOW - timedelta(days=1)
    item_id = _article(conn, source_id, "1", fetched)
    rows.set_window(conn, source_id, 7)
    rows.enrol(conn, NOW)
    (found,) = rows.due(conn, NOW)
    state, action = recheck.advance(found.state, recheck.SAME)
    assert action == recheck.RECORD
    sent = Validators(etag='"v1"', last_modified="Thu, 26 Nov 2026 12:00:00 GMT")
    rows.record(conn, item_id, state, sent, NOW)
    assert rows.due(conn, NOW) == []
    (again,) = rows.due(conn, fetched + timedelta(days=2))
    assert again.state == state
    assert again.validators == sent


def test_a_done_item_is_never_due(conn: Connection) -> None:  # noqa: F811
    source_id = _news_source(conn)
    item_id = _article(conn, source_id, "1", NOW - timedelta(days=1))
    rows.set_window(conn, source_id, 1)
    rows.enrol(conn, NOW)
    (found,) = rows.due(conn, NOW)
    state, _ = recheck.advance(found.state, recheck.SAME)
    assert state.done
    rows.record(conn, item_id, state, Validators(), NOW)
    assert rows.due(conn, NOW + timedelta(days=30)) == []


def test_a_window_set_back_to_zero_stops_its_items_at_once(conn: Connection) -> None:  # noqa: F811
    source_id = _news_source(conn)
    _article(conn, source_id, "1", NOW - timedelta(days=1))
    rows.set_window(conn, source_id, 7)
    rows.enrol(conn, NOW)
    assert len(rows.due(conn, NOW)) == 1
    rows.set_window(conn, source_id, 0)
    assert rows.due(conn, NOW) == []


def test_an_item_withdrawn_from_serving_is_not_due(conn: Connection) -> None:  # noqa: F811
    source_id = _news_source(conn)
    item_id = _article(conn, source_id, "1", NOW - timedelta(days=1))
    rows.set_window(conn, source_id, 7)
    rows.enrol(conn, NOW)
    conn.execute(content_items.update().where(content_items.c.item_id == item_id).values(served=0))
    assert rows.due(conn, NOW) == []


def test_done_and_a_null_next_due_go_together(conn: Connection) -> None:  # noqa: F811
    source_id = _news_source(conn)
    item_id = _article(conn, source_id, "1", NOW)
    with pytest.raises(sa_exc.IntegrityError, match="news_recheck_due_check"):
        conn.execute(insert(news_recheck).values(item_id=item_id, first_fetched_at=NOW, done=False))


def test_a_negative_window_is_refused(conn: Connection) -> None:  # noqa: F811
    source_id = _news_source(conn)
    with pytest.raises(sa_exc.IntegrityError, match="news_recheck_sources_window_check"):
        rows.set_window(conn, source_id, -1)
