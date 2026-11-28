"""The news re-check's rows (CRFC-006 G7, n-w9): a source's window, an item's state, what is due."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import exc as sa_exc, insert, select, text
from sqlalchemy.engine import Connection

from podcast_scraper.db import news_recheck as rows
from podcast_scraper.db.schema import content_items, news_recheck
from podcast_scraper.news import recheck
from podcast_scraper.news.recheck_fetch import Validators
from tests.unit.podcast_scraper.db.test_db_corpus import (  # noqa: F401
    _document,
    _entity,
    _insight,
    _item,
    _mention,
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


def _count(conn: Connection, table: str, item_id: int) -> int:  # noqa: F811
    from sqlalchemy import func

    from podcast_scraper.db.schema import corpus_metadata

    t = corpus_metadata.tables[f"corpus.{table}"]
    return int(
        conn.execute(select(func.count()).select_from(t).where(t.c.item_id == item_id)).scalar_one()
    )


def _revision(conn: Connection) -> int:  # noqa: F811
    return int(conn.execute(text("SELECT revision FROM corpus.corpus_revision")).scalar_one())


def test_a_withdrawn_item_is_served_by_nothing_and_keeps_its_insights(
    conn: Connection,  # noqa: F811
) -> None:
    source_id = _news_source(conn)
    item_id = _item(
        conn, source_id, "1", content_type="news_article", served=1, has_kg=True, has_gi=True
    )
    raw = _document(conn, item_id, kind="article", variant="raw", fetched_at=NOW)
    old = _document(conn, item_id, kind="article", variant="cleaned", fetched_at=NOW)
    cleaned = _document(
        conn, item_id, kind="article", variant="cleaned", version=2, supersedes_id=old
    )
    insight = _insight(conn, item_id, cleaned)
    topic = _entity(conn, "t", natural_id="topic:t", type="topic")
    _mention(conn, insight, topic, item_id)
    conn.execute(
        text(
            "INSERT INTO corpus.item_entities (item_id, entity_id, role, content_type) "
            "VALUES (:i, :e, 'mentions', 'news_article')"
        ),
        {"i": item_id, "e": topic},
    )
    conn.execute(
        text("INSERT INTO corpus.item_cards (item_id, title, source_title) VALUES (:i, 't', 's')"),
        {"i": item_id},
    )
    # Another item's rows, which the withdrawal must leave alone.
    other = _item(conn, source_id, "2", content_type="news_article", served=1, has_kg=True)
    other_doc = _document(conn, other, kind="article", variant="cleaned", fetched_at=NOW)
    _mention(conn, _insight(conn, other, other_doc, natural_id="insight:other"), topic, other)
    conn.execute(
        text(
            "INSERT INTO corpus.item_entities (item_id, entity_id, role, content_type) "
            "VALUES (:i, :e, 'mentions', 'news_article')"
        ),
        {"i": other, "e": topic},
    )
    conn.execute(
        text("INSERT INTO corpus.item_cards (item_id, title, source_title) VALUES (:i, 'o', 's')"),
        {"i": other},
    )
    before = _revision(conn)
    assert rows.withdraw(conn, item_id) is True
    written = conn.execute(
        text(
            "SELECT variant, version, supersedes_id FROM corpus.documents "
            "WHERE item_id = :i AND change_kind = 'withdraw' ORDER BY variant"
        ),
        {"i": item_id},
    ).all()
    # One per series, each on the series' newest version; the superseded cleaned v1 gets none.
    assert [tuple(r) for r in written] == [("cleaned", 3, cleaned), ("raw", 2, raw)]
    item = conn.execute(
        text("SELECT served, has_kg, has_gi FROM corpus.content_items WHERE item_id = :i"),
        {"i": item_id},
    ).one()
    assert tuple(item) == (0, False, False)
    for table in ("item_entities", "insight_mentions", "item_cards"):
        assert _count(conn, table, item_id) == 0, table
    assert _count(conn, "insights", item_id) == 1
    for table in ("item_entities", "insight_mentions", "item_cards"):
        assert _count(conn, table, other) == 1, table
    assert _count(conn, "documents", other) == 1
    other_row = conn.execute(
        text("SELECT served, has_kg FROM corpus.content_items WHERE item_id = :i"), {"i": other}
    ).one()
    assert tuple(other_row) == (1, True)
    assert _revision(conn) == before + 1
    # A second call finds nothing served and writes nothing.
    assert rows.withdraw(conn, item_id) is False
    assert _count(conn, "documents", item_id) == 5
    assert _revision(conn) == before + 1


def _source_columns(conn: Connection, source_id: int) -> tuple:  # noqa: F811
    return tuple(
        conn.execute(
            text("SELECT title, language FROM corpus.sources WHERE source_id = :s"),
            {"s": source_id},
        ).one()
    )


def test_a_withdrawal_merges_its_sources_columns_from_the_items_still_served(
    conn: Connection,  # noqa: F811
) -> None:
    source_id = _news_source(conn)
    _item(
        conn,
        source_id,
        "old",
        content_type="news_article",
        served=1,
        published_at=NOW - timedelta(days=2),
        source_attrs={"title": "Old desk", "language": "en"},
    )
    newest = _item(
        conn,
        source_id,
        "new",
        content_type="news_article",
        served=1,
        published_at=NOW,
        source_attrs={"title": "New desk", "language": "de"},
    )
    conn.execute(
        text("UPDATE corpus.sources SET title = 'New desk', language = 'de' WHERE source_id = :s"),
        {"s": source_id},
    )
    assert rows.withdraw(conn, newest) is True
    assert _source_columns(conn, source_id) == ("Old desk", "en")


def test_withdrawing_a_sources_last_served_item_keeps_its_columns(
    conn: Connection,  # noqa: F811
) -> None:
    source_id = _news_source(conn)
    only = _item(
        conn,
        source_id,
        "1",
        content_type="news_article",
        served=1,
        source_attrs={"title": "Desk", "language": "en"},
    )
    conn.execute(
        text("UPDATE corpus.sources SET title = 'Desk', language = 'en' WHERE source_id = :s"),
        {"s": source_id},
    )
    assert rows.withdraw(conn, only) is True
    assert _source_columns(conn, source_id) == ("Desk", "en")
