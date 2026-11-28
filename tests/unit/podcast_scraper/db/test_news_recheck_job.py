"""The news re-check job on rows (CRFC-006 B.2, n-w9): what a run does with each answer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.engine import Connection, Engine

from podcast_scraper import content_types, news_items
from podcast_scraper.db import leases, news_recheck as rows
from podcast_scraper.db.schema import content_items, news_recheck
from podcast_scraper.db.settings import normalize_url
from podcast_scraper.news import extract, guardian, recheck, recheck_fetch, recheck_job
from podcast_scraper.news.recheck_fetch import Validators
from podcast_scraper.workflow import text_in
from tests.unit.podcast_scraper.db.test_corpus_writer import _empty_corpus
from tests.unit.podcast_scraper.db.test_db_corpus import (  # noqa: F401
    _document,
    _item,
    _source,
    conn,
    url,
)

NOW = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)
FETCHED = NOW - timedelta(days=1, hours=1)
#: When an item checked at NOW is due again: day 2 (FETCHED + 2 days, 23 hours on) is sooner
#: than the one-day gap a check keeps.
NEXT = NOW + rows.MIN_GAP
SERVED = "The council voted on Tuesday to close the library. " * 20
EDITED = SERVED + "Correction: the vote was on Wednesday, not Tuesday. " * 3
PAGE = f"<html lang='en'><body><article><p>{EDITED}</p></article></body></html>"


@pytest.fixture
def engine(url: str) -> Iterator[Engine]:  # noqa: F811
    made = create_engine(normalize_url(url))
    yield made
    made.dispose()
    _empty_corpus(url)


def _run_for(transport: str = "rss") -> recheck_job.SourceRun:
    return recheck_job.SourceRun(
        cfg=SimpleNamespace(output_dir="/nowhere", rss_url=None),
        entry=SimpleNamespace(url="https://news.example.com/feed.xml", news=None),
        ctype=content_types.content_type("news_article"),
        transport=content_types.Transport(name=transport),
    )


def _stored(n: str, **metadata: Any) -> content_types.SourceItem:
    return content_types.SourceItem(
        item_id=f"rss-{n}",
        metadata={
            "title": f"Library {n}",
            "language": "en",
            "fetched_at": "2026-11-27T11:00:00Z",
            "extraction": extract.TRAFILATURA,
            "extractor_version": extract.extractor_version(extract.TRAFILATURA),
            **metadata,
        },
        text=SERVED,
        url=f"https://news.example.com/{n}",
        raw="<html><body><p>old</p></body></html>",
    )


class World:
    """A news source with a 7-day window and served articles, its stored copies, a fake check
    that answers per URL, and a fake writer that keeps what it was given."""

    def __init__(self, engine: Engine, monkeypatch: pytest.MonkeyPatch, transport: str = "rss"):
        self.engine = engine
        self.source = _run_for(transport)
        self.stored: Dict[str, content_types.SourceItem] = {}
        self.answers: Dict[str, List[recheck_fetch.Check]] = {}
        self.checked: List[str] = []
        self.methods: List[Any] = []
        self.written: List[List[content_types.SourceItem]] = []
        self.fail_write = False
        #: Whether the fake write's row sync lands a new article version (else: refused).
        self.sync = True
        self.stages: List[Any] = []
        self.ids: Dict[str, int] = {}
        with engine.begin() as c:
            self.source_id = _source(
                c, self.source.natural_id, content_type="news_article", title="News"
            )
            rows.set_window(c, self.source_id, 7)
        monkeypatch.setattr(text_in, "stored_items_by_id", lambda _dir, _ctype: self.stored)
        #: Each stored copy's metadata file, by natural id (:meth:`on_disk`).
        self.metadata: Dict[str, Path] = {}
        monkeypatch.setattr(text_in, "stored_metadata_by_id", lambda _dir: self.metadata)

    def article(self, n: str, **metadata: Any) -> int:
        with self.engine.begin() as c:
            item_id = _item(c, self.source_id, n, content_type="news_article", served=1)
            _document(c, item_id, kind="article", variant="cleaned", fetched_at=FETCHED)
        self.stored[f"item:{n}"] = _stored(n, **metadata)
        self.ids[f"https://news.example.com/{n}"] = item_id
        return item_id

    def on_disk(self, tmp: Path, n: str) -> Path:
        """Item ``n``'s stored metadata file, as the writer leaves it."""
        path = tmp / f"{n}.metadata.json"
        doc = {"content_type": "news_article", "item_id": f"item:{n}", "title": f"Library {n}"}
        path.write_text(json.dumps({**doc, "serveable": "summary"}), encoding="utf-8")
        self.metadata[f"item:{n}"] = path
        return path

    def answer(self, n: str, *checks: recheck_fetch.Check) -> None:
        self.answers[f"https://news.example.com/{n}"] = list(checks)

    def check(self, served: str, page: str, source: Any, validators: Validators, **kw: Any) -> Any:
        self.checked.append(served)
        self.methods.append(kw.get("served_method"))
        return self.answers[page].pop(0)

    def check_guardian(self, served: str, item_id: str, source: Any, validators: Any) -> Any:
        self.checked.append(f"guardian:{item_id}")
        return self.answers[item_id].pop(0)

    def write(self, cfg: Any, entry: Any, ctype: Any, transport: Any, items: Any) -> Any:
        self.stages.append(getattr(cfg, "pipeline_stage", None))
        if self.fail_write:
            raise RuntimeError("the pipeline failed")
        self.written.append(list(items))
        if self.sync:
            # As the row sync writes a changed article: its next version over the served one.
            with self.engine.begin() as c:
                for item in items:
                    item_id = self.ids[item.url]
                    served = c.execute(
                        text(
                            "SELECT document_id, version FROM corpus.documents d WHERE item_id = :i"
                            " AND NOT EXISTS (SELECT 1 FROM corpus.documents n"
                            " WHERE n.supersedes_id = d.document_id)"
                        ),
                        {"i": item_id},
                    ).one()
                    _document(
                        c,
                        item_id,
                        kind="article",
                        variant="cleaned",
                        version=served.version + 1,
                        supersedes_id=served.document_id,
                        change_kind="updated",
                    )
        return len(items), "ok"

    def run(self, at: datetime = NOW) -> Dict[str, int]:
        return dict(
            recheck_job.run(
                self.engine,
                [self.source],
                at,
                check=self.check,
                check_guardian=self.check_guardian,
                write=self.write,
            )
        )


def _check(status: str, html: str | None = None) -> recheck_fetch.Check:
    return recheck_fetch.Check(status, Validators(etag='"e1"'), 200, html=html)


def _state(conn: Connection, item_id: int) -> Any:  # noqa: F811
    return conn.execute(select(news_recheck).where(news_recheck.c.item_id == item_id)).one()


def test_the_same_page_is_recorded_and_due_again_on_its_next_day(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    item_id = w.article("1")
    w.answer("1", _check(recheck.SAME))
    assert w.run() == {"enrolled": 1, recheck.SAME: 1}
    assert w.checked == [SERVED]
    state = _state(conn, item_id)
    assert (state.checks_done, state.last_status, state.etag) == (1, recheck.SAME, '"e1"')
    assert state.next_due_at == NEXT
    # Not due again the same day: nothing checked.
    assert w.run() == {"enrolled": 0}
    assert w.written == []


def test_a_changed_page_is_written_as_the_stored_item_with_the_pages_text(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    one, two = w.article("1"), w.article("2")
    w.answer("1", _check(recheck.CHANGED, PAGE))
    w.answer("2", _check(recheck.CHANGED, PAGE))
    assert w.run() == {"enrolled": 2, recheck.CHANGED: 2}
    # One write for the source, with both items.
    (batch,) = w.written
    assert [i.url for i in batch] == ["https://news.example.com/1", "https://news.example.com/2"]
    version = batch[0]
    assert "Correction: the vote was on Wednesday" in (version.text or "")
    assert (version.raw, version.raw_suffix) == (PAGE, ".html")
    assert version.item_id == "rss-1" and version.metadata["title"] == "Library 1"
    assert version.metadata["fetched_at"] != "2026-11-27T11:00:00Z"
    assert version.metadata["extraction"] in recheck_job.PAGE_EXTRACTIONS
    for item_id in (one, two):
        assert _state(conn, item_id).last_status == recheck.CHANGED


def test_a_failed_write_leaves_the_changed_items_due(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    item_id = w.article("1")
    w.fail_write = True
    w.answer("1", _check(recheck.CHANGED, PAGE))
    assert w.run() == {"enrolled": 1, "write_failed": 1}
    assert _state(conn, item_id).checks_done == 0
    w.fail_write = False
    w.answer("1", _check(recheck.CHANGED, PAGE))
    assert w.run() == {"enrolled": 0, recheck.CHANGED: 1}
    assert len(w.written) == 1


def test_two_gone_answers_in_a_row_withdraw_the_item(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    item_id = w.article("1")
    meta = w.on_disk(tmp_path, "1")
    w.answer("1", _check(recheck.GONE), _check(recheck.GONE))
    assert w.run() == {"enrolled": 1, recheck.GONE: 1}
    assert json.loads(meta.read_text(encoding="utf-8"))["serveable"] == "summary"
    assert w.run(NEXT) == {"enrolled": 0, "withdrawn": 1}
    # The files are what the news search and the other surfaces read: the stored copy says so too.
    doc = json.loads(meta.read_text(encoding="utf-8"))
    assert doc == {
        "content_type": "news_article",
        "item_id": "item:1",
        "title": "Library 1",
        "serveable": news_items.SERVEABLE_NONE,
    }
    assert not news_items.is_shown(doc)
    served = conn.execute(
        select(content_items.c.served).where(content_items.c.item_id == item_id)
    ).scalar_one()
    state = _state(conn, item_id)
    assert (served, state.done, state.gone_in_a_row) == (0, True, 2)
    kinds = conn.execute(
        text("SELECT change_kind FROM corpus.documents WHERE item_id = :i ORDER BY version"),
        {"i": item_id},
    ).scalars()
    assert list(kinds) == [None, rows.WITHDRAW]


def test_a_withdrawal_whose_file_cannot_be_marked_withdraws_nothing_and_stays_due(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    item_id = w.article("1")
    meta = w.on_disk(tmp_path, "1")
    w.answer("1", _check(recheck.GONE), _check(recheck.GONE), _check(recheck.GONE))
    w.run()
    stored = meta.read_text(encoding="utf-8")
    meta.unlink()
    assert w.run(NEXT) == {"enrolled": 0, "withdraw_failed": 1}
    served = conn.execute(
        select(content_items.c.served).where(content_items.c.item_id == item_id)
    ).scalar_one()
    assert (served, _state(conn, item_id).gone_in_a_row) == (1, 1)
    meta.write_text(stored, encoding="utf-8")
    assert w.run(NEXT) == {"enrolled": 0, "withdrawn": 1}
    assert json.loads(meta.read_text(encoding="utf-8"))["serveable"] == news_items.SERVEABLE_NONE


def test_a_withdrawal_waits_while_the_source_is_ingested(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    item_id = w.article("1")
    w.answer("1", _check(recheck.GONE), _check(recheck.GONE), _check(recheck.GONE))
    w.run()
    resource = leases.source_resource(w.source.natural_id)
    leases.acquire(engine, [leases.LeaseRequest(resource, leases.EXCLUSIVE)], "pipeline:t")
    later = NEXT
    assert w.run(later) == {"enrolled": 0, "withdraw_deferred": 1}
    assert _state(conn, item_id).gone_in_a_row == 1
    leases.release(engine, "pipeline:t")
    assert w.run(later) == {"enrolled": 0, "withdrawn": 1}


def test_a_feed_text_is_recorded_unsupported_and_a_standfirst_is_checked(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    feed = w.article("1", extraction=extract.FEED, extractor_version="feed/1")
    w.article("2", extraction=extract.STANDFIRST, extractor_version="standfirst/1")
    w.answer("2", _check(recheck.SAME))
    assert w.run() == {"enrolled": 2, recheck_job.UNSUPPORTED: 1, recheck.SAME: 1}
    assert w.checked == [SERVED]
    state = _state(conn, feed)
    # Recorded as a check, so its schedule moves on like any other.
    assert (state.checks_done, state.last_status) == (1, recheck_job.UNSUPPORTED)
    assert state.next_due_at == NEXT


def test_a_guardian_item_is_reread_by_id_and_its_record_is_the_next_version(
    engine: Engine, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    w = World(engine, monkeypatch, transport=recheck_job.GUARDIAN)
    w.article("1", extraction="guardian", extractor_version="guardian/api")
    record = {"id": "world/2026/nov/27/library", "fields": {"body": "<p>edited</p>"}}
    w.answers["rss-1"] = [recheck_fetch.Check(recheck.CHANGED, Validators(), 200, record=record)]
    made = content_types.SourceItem(
        item_id="world/2026/nov/27/library", text="edited", url="https://news.example.com/1"
    )
    seen: List[Any] = []

    def from_record(got: Any, source: Any) -> content_types.SourceItem:
        seen.append(got)
        return made

    monkeypatch.setattr(guardian, "item_from_record", from_record)
    assert w.run() == {"enrolled": 1, recheck.CHANGED: 1}
    assert w.checked == ["guardian:rss-1"]
    assert seen == [record] and w.written == [[made]]


def test_a_page_check_is_given_the_served_texts_method(
    engine: Engine, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    w.article("1", extraction=extract.READABILITY)
    w.answer("1", _check(recheck.SAME))
    w.run()
    assert w.methods == [extract.READABILITY]


def test_a_run_takes_only_its_sources_items_and_records_a_missing_copy(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    item_id = w.article("1")
    found = recheck_job.run(engine, [], NOW, check=w.check, write=w.write)
    assert dict(found) == {"enrolled": 1}
    assert _state(conn, item_id).checks_done == 0
    del w.stored["item:1"]
    assert w.run() == {"enrolled": 0, recheck_job.NO_BODY: 1}
    state = _state(conn, item_id)
    assert (state.checks_done, state.last_status) == (1, recheck_job.NO_BODY)


def test_an_item_that_cannot_be_checked_never_holds_the_head_of_the_queue(
    engine: Engine, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    # public's review of #2641: with limit 1, an unrecorded skip would be taken on every run.
    w = World(engine, monkeypatch)
    w.article("1", extraction=extract.FEED, extractor_version="feed/1")
    w.article("2")
    w.answer("2", _check(recheck.SAME))
    run = lambda: dict(  # noqa: E731
        recheck_job.run(engine, [w.source], NOW, check=w.check, write=w.write, limit=1)
    )
    assert run() == {"enrolled": 2, recheck_job.UNSUPPORTED: 1}
    assert run() == {"enrolled": 0, recheck.SAME: 1}
    assert w.checked == [SERVED]


def test_a_text_from_another_extractor_is_compared_after_extracting_its_raw_page_again(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = World(engine, monkeypatch)
    w.article("1", extractor_version="trafilatura/0.1")
    w.article("2")
    again = extract.Extracted("re-extracted", extract.TRAFILATURA, "en")
    monkeypatch.setattr(extract, "main_text", lambda _html, _page, **kw: again)
    w.answer("1", _check(recheck.SAME))
    w.answer("2", _check(recheck.SAME))
    w.run()
    assert w.checked == ["re-extracted", SERVED]


def _modes(engine: Engine, resource: str) -> List[str]:
    from podcast_scraper.db.schema import job_leases

    with engine.connect() as c:
        return list(
            c.execute(select(job_leases.c.mode).where(job_leases.c.resource == resource)).scalars()
        )


def test_a_run_holds_its_lease_exclusive_and_a_withdrawal_its_source_exclusive(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    w = World(engine, monkeypatch)
    w.article("1")
    w.answer("1", _check(recheck.GONE), _check(recheck.GONE))
    seen: Dict[str, List[str]] = {}
    check = w.check

    def watched(*args: Any, **kw: Any) -> Any:
        seen["run"] = _modes(engine, recheck_job.RESOURCE)
        return check(*args, **kw)

    withdraw = rows.withdraw

    def watched_withdraw(c: Connection, item_id: int) -> bool:
        seen["source"] = _modes(engine, leases.source_resource(w.source.natural_id))
        return withdraw(c, item_id)

    monkeypatch.setattr(w, "check", watched)
    monkeypatch.setattr(rows, "withdraw", watched_withdraw)
    w.run()
    w.run(NEXT)
    assert seen == {"run": [leases.EXCLUSIVE], "source": [leases.EXCLUSIVE]}
    # Released when the run ends.
    assert _modes(engine, recheck_job.RESOURCE) == []


def test_one_run_at_a_time(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    w = World(engine, monkeypatch)
    w.article("1")
    leases.acquire(
        engine, [leases.LeaseRequest(recheck_job.RESOURCE, leases.EXCLUSIVE)], "news_recheck:t"
    )
    with pytest.raises(leases.LeaseRefused):
        w.run()
    assert w.checked == []


def test_a_change_the_row_sync_refused_is_not_recorded_and_stays_due(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    # The review of #2641 (database's): the pipeline run reports a refused row sync as an
    # outcome, never an exception, so the job checks the rows itself.
    w = World(engine, monkeypatch)
    item_id = w.article("1")
    w.sync = False
    w.answer("1", _check(recheck.CHANGED, PAGE))
    assert w.run() == {"enrolled": 1, "not_synced": 1}
    assert len(w.written) == 1
    assert _state(conn, item_id).checks_done == 0


def test_the_write_runs_as_a_rederive_of_the_source(
    engine: Engine, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    w.article("1")
    w.answer("1", _check(recheck.CHANGED, PAGE))
    w.run()
    assert w.stages == [text_in.REDERIVE_ONLY]
    assert getattr(w.source.cfg, "pipeline_stage", None) is None  # the caller's cfg is untouched


def test_an_item_enrolled_late_takes_its_checks_a_day_apart(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    # Fetched 5 days ago: days 1, 2 and 4 are past. Two gone answers must not come minutes apart.
    w = World(engine, monkeypatch)
    with engine.begin() as c:
        item_id = _item(c, w.source_id, "late", content_type="news_article", served=1)
        _document(c, item_id, kind="article", variant="cleaned", fetched_at=NOW - timedelta(days=5))
    w.stored["item:late"] = _stored("late")
    w.ids["https://news.example.com/late"] = item_id
    w.answer("late", _check(recheck.GONE), _check(recheck.GONE))
    assert w.run() == {"enrolled": 1, recheck.GONE: 1}
    assert w.run(NOW + timedelta(hours=1)) == {"enrolled": 0}
    assert _state(conn, item_id).next_due_at == NEXT
    assert w.run(NEXT) == {"enrolled": 0, "withdrawn": 1}


def test_a_check_that_raises_is_recorded_as_failed(
    engine: Engine, monkeypatch: pytest.MonkeyPatch, conn: Connection  # noqa: F811
) -> None:
    w = World(engine, monkeypatch)
    one, two = w.article("1"), w.article("2")

    def broken(*args: Any, **kw: Any) -> Any:
        if args[1].endswith("/1"):
            raise RuntimeError("a parser bug")
        return _check(recheck.SAME)

    monkeypatch.setattr(w, "check", broken)
    assert w.run() == {"enrolled": 2, recheck.FAILED: 1, recheck.SAME: 1}
    assert _state(conn, one).last_status == recheck.FAILED
    assert _state(conn, two).last_status == recheck.SAME


def test_a_long_write_keeps_the_runs_leases_and_a_lost_one_is_raised(
    engine: Engine, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    import time

    from podcast_scraper.db.schema import job_leases

    held = leases.acquire(engine, [leases.LeaseRequest("build:t", leases.EXCLUSIVE)], "h:t")

    def beats() -> List[Any]:
        with engine.connect() as c:
            return list(
                c.execute(
                    select(job_leases.c.heartbeat_at).where(job_leases.c.holder == "h:t")
                ).scalars()
            )

    first = beats()
    with recheck_job._kept_alive(engine, "h:t", held, every_s=0.05):
        time.sleep(0.3)
    assert beats() != first
    with pytest.raises(leases.LeaseLost):
        with recheck_job._kept_alive(engine, "h:t", held, every_s=0.05):
            leases.release(engine, "h:t")
            time.sleep(0.3)
