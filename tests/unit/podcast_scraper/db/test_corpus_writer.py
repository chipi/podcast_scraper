"""The pipeline's source and item rows, transcript documents and segments in the corpus database
(db.corpus_writer)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator

import pytest
from sqlalchemy import and_, create_engine, select, text
from sqlalchemy.engine import Connection

from podcast_scraper.db import corpus_writer, engines, leases
from podcast_scraper.db.corpus_rows import item_row, source_row, source_row_for
from podcast_scraper.db.corpus_writer import (
    ENV_STRICT,
    OUTCOMES,
    RETRANSCRIBED,
    sync_documents,
    sync_episode,
    SyncFailed,
)
from podcast_scraper.db.importers.corpus_documents import (
    joined_body,
    transcript_files,
    TranscriptFile,
)
from podcast_scraper.db.schema import (
    corpus_metadata,
    corpus_revision,
    documents,
    item_cards,
    segments,
    sources,
)
from podcast_scraper.db.settings import ENV_PIPELINE_RW_URL, normalize_url
from tests.unit.podcast_scraper.db.test_db_corpus import _item, _source, conn, url  # noqa: F401

BASE = "0001 - An episode_abc"


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_bytes(body.encode("utf-8"))
    return path


def _files(directory: Path) -> list[TranscriptFile]:
    return transcript_files(directory, BASE, "en")


def _versions(db: Connection, item: int, variant: str = "raw") -> list[tuple]:
    rows = db.execute(
        text(
            "SELECT version, change_kind, supersedes_id IS NOT NULL FROM corpus.documents"
            " WHERE item_id = :i AND variant = :v ORDER BY version"
        ),
        {"i": item, "v": variant},
    ).all()
    return [tuple(row) for row in rows]


def test_a_first_sync_writes_version_one_and_its_segments(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    body = "Welcome back.\r\nToday, trails.\n"
    _write(tmp_path, BASE + ".txt", body)
    item = _item(conn, _source(conn))
    [synced] = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    assert (synced.variant, synced.version, synced.segments, synced.written) == ("raw", 1, 2, True)
    assert joined_body(conn, item, synced.document_id) == body
    assert _versions(conn, item) == [(1, "created", False)]
    conn.rollback()


def test_the_same_body_writes_nothing(conn: Connection, tmp_path: Path) -> None:  # noqa: F811
    _write(tmp_path, BASE + ".txt", "One line.\n")
    item = _item(conn, _source(conn))
    first = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    again = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    assert [d.written for d in again] == [False]
    assert again[0].document_id == first[0].document_id
    assert _versions(conn, item) == [(1, "created", False)]
    conn.rollback()


def test_a_new_body_supersedes_and_the_old_segments_stay(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, BASE + ".txt", "Old words.\n")
    item = _item(conn, _source(conn))
    [old] = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    _write(tmp_path, BASE + ".txt", "New words.\nAnd more.\n")
    [new] = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    assert (new.version, new.written) == (2, True)
    assert _versions(conn, item) == [(1, "created", False), (2, "retranscribed", True)]
    assert joined_body(conn, item, old.document_id) == "Old words.\n"
    assert joined_body(conn, item, new.document_id) == "New words.\nAnd more.\n"
    conn.rollback()


def test_an_unknown_cause_is_refused(conn: Connection, tmp_path: Path) -> None:  # noqa: F811
    _write(tmp_path, BASE + ".txt", "x\n")
    with pytest.raises(ValueError, match="change_kind"):
        sync_documents(conn, 1, _files(tmp_path), change_kind="created")


FEED = "sha256:" + "ab" * 32


def _doc(episode_id: str = "item:1", **episode: object) -> dict:
    return {
        "feed": {"feed_id": FEED, "title": "A show", "language": "en"},
        "episode": {
            "episode_id": episode_id,
            "title": "E",
            "language": "en",
            "published_date": "2026-11-20T08:00:00+00:00",
            **episode,
        },
    }


def _metadata(directory: Path, episode_id: str = "item:1", doc: dict | None = None) -> Path:
    path = directory / f"{episode_id.replace(':', '_')}.metadata.json"
    path.write_text(json.dumps(doc or _doc(episode_id)), encoding="utf-8")
    return path


def _empty_corpus(url: str) -> None:  # noqa: F811
    engine = create_engine(normalize_url(url))
    with engine.begin() as db:
        for table in reversed(corpus_metadata.sorted_tables):
            if table is not corpus_revision:
                db.execute(table.delete())
    engine.dispose()


@pytest.fixture
def pipeline_db(url: str, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:  # noqa: F811
    monkeypatch.setenv(ENV_PIPELINE_RW_URL, url)
    engines.reset_engines()
    OUTCOMES.clear()
    corpus_writer.UNPLACED.clear()
    corpus_writer._DEFERRED.clear()
    yield
    corpus_writer._DEFERRED.clear()
    engines.reset_engines()
    _empty_corpus(url)


@pytest.fixture
def corpus_held(url: str) -> Iterator[Callable[[], object]]:  # noqa: F811
    """Hold ``corpus`` exclusive as another job would; call the yielded function to release."""
    other = create_engine(normalize_url(url))
    leases.acquire(other, [leases.LeaseRequest(leases.CORPUS, leases.EXCLUSIVE)], "migrate:t")
    yield lambda: leases.release(other, "migrate:t")
    leases.release(other, "migrate:t")
    other.dispose()


def test_a_refused_episode_is_retried_when_the_run_ends(
    pipeline_db: None,
    corpus_held: Callable[[], object],
    conn: Connection,  # noqa: F811
    tmp_path: Path,
) -> None:
    transcript = _write(tmp_path, BASE + ".txt", "Hello.\\n")
    assert sync_episode(_metadata(tmp_path, "item:r"), transcript) == "lease_refused"
    corpus_held()
    assert corpus_writer.flush_deferred(str(tmp_path)) == 0
    assert _row(conn, "content_items", "item:r")["has_transcript"] is True
    assert not (tmp_path / corpus_writer.DEFERRED_FILE).exists()
    assert corpus_writer._DEFERRED == {}


def test_a_still_refused_episode_is_kept_and_a_later_run_syncs_it(
    pipeline_db: None,
    corpus_held: Callable[[], object],
    conn: Connection,  # noqa: F811
    tmp_path: Path,
) -> None:
    run = tmp_path / "run_1"
    run.mkdir()
    meta = _metadata(run, "item:k")
    assert sync_episode(meta, None) == "lease_refused"
    assert corpus_writer.flush_deferred(str(run)) == 1
    kept = [
        json.loads(line) for line in (run / corpus_writer.DEFERRED_FILE).read_text().splitlines()
    ]
    assert kept == [{"metadata": str(meta), "transcript": None, "change_kind": RETRANSCRIBED}]
    assert corpus_writer.retry_deferred(str(tmp_path)) == 1
    corpus_held()
    assert corpus_writer.retry_deferred(str(tmp_path)) == 0
    assert not (run / corpus_writer.DEFERRED_FILE).exists()
    assert _row(conn, "content_items", "item:k")["served"] == 1


def test_a_strict_refusal_is_kept_for_the_retry_too(
    pipeline_db: None,
    corpus_held: Callable[[], object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_STRICT, "1")
    with pytest.raises(SyncFailed, match="lease_refused"):
        sync_episode(_metadata(tmp_path, "item:s"), None)
    assert corpus_writer.flush_deferred(str(tmp_path)) == 1


def test_the_runs_retry_hook_never_stops_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper.workflow import orchestration

    def boom(_directory: str, **_kw: object) -> int:
        raise RuntimeError("database gone")

    monkeypatch.setattr(corpus_writer, "flush_deferred", boom)
    orchestration._corpus_deferred("flush_deferred", "/out/run")
    orchestration._corpus_deferred("retry_deferred", None)


def test_a_run_that_raises_flushes_without_retrying(monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper.workflow import orchestration

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        corpus_writer, "flush_deferred", lambda directory, **kw: calls.append((directory, kw))
    )
    orchestration._corpus_deferred("flush_deferred", "/out/run", retry=False)
    assert calls == [("/out/run", {"retry": False})]


def _row(db: Connection, table: str, natural_id: str) -> dict:
    found = corpus_metadata.tables[f"corpus.{table}"]
    rows = db.execute(select(found).where(found.c.natural_id == natural_id)).mappings()
    return dict(rows.one())


def _source_of(doc: dict) -> str:
    return str(source_row(doc)["natural_id"])


def test_with_no_database_configured_nothing_happens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_PIPELINE_RW_URL, raising=False)
    transcript = _write(tmp_path, BASE + ".txt", "x\n")
    assert sync_episode(_metadata(tmp_path), transcript) == "not_configured"


def test_a_sync_writes_the_source_and_item_rows_then_its_documents(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    transcript = _write(tmp_path, BASE + ".txt", "Hello.\n")
    meta = _metadata(tmp_path, "item:w1")
    assert sync_episode(meta, transcript) == "written"
    doc = _doc("item:w1")
    source = _row(conn, "sources", _source_of(doc))
    item = _row(conn, "content_items", "item:w1")
    for column, value in source_row(doc).items():
        assert source[column] == value, column
    for column, value in item_row(doc, has_bridge=False).items():
        assert item[column] == value, column
    assert item["source_id"] == source["source_id"]
    assert _versions(conn, item["item_id"]) == [(1, "created", False)]
    conn.rollback()
    assert sync_episode(meta, transcript) == "unchanged"
    again = _row(conn, "content_items", "item:w1")
    assert (again["updated_at"], again["item_id"]) == (item["updated_at"], item["item_id"])
    held = conn.execute(text("SELECT count(*) FROM corpus.job_leases")).scalar_one()
    assert held == 0


def test_a_changed_item_overwrites_its_row(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    transcript = _write(tmp_path, BASE + ".txt", "Hello.\n")
    sync_episode(_metadata(tmp_path, "item:c"), transcript)
    renamed = _doc("item:c", title="Renamed", duration_seconds=60)
    assert sync_episode(_metadata(tmp_path, "item:c", renamed), transcript) == "unchanged"
    item = _row(conn, "content_items", "item:c")
    assert (item["title"], item["duration_s"]) == ("Renamed", 60)


def _revision(db: Connection) -> int:
    db.rollback()
    return int(db.execute(select(corpus_revision.c.revision)).scalar_one())


def test_a_sync_that_writes_the_item_row_bumps_the_revision_and_an_unchanged_one_does_not(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    # The catalog's token reads the revision: a sync that started before another committed and
    # commits after it carries an older now() as updated_at, so the newest updated_at alone
    # would not move (#2614).
    transcript = _write(tmp_path, BASE + ".txt", "Hello.\n")
    start = _revision(conn)
    sync_episode(_metadata(tmp_path, "item:v"), transcript)
    assert _revision(conn) == start + 1
    sync_episode(_metadata(tmp_path, "item:v"), transcript)
    assert _revision(conn) == start + 1
    renamed = _doc("item:v", title="Renamed")
    sync_episode(_metadata(tmp_path, "item:v", renamed), transcript)
    assert _revision(conn) == start + 2


def test_a_newer_served_item_decides_the_source_and_an_older_one_fills_only_nulls(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    missing = tmp_path / "none" / "x.txt"
    old = _doc("item:old", published_date="2026-01-01T00:00:00+00:00")
    old["feed"].update(title="Old name", category="Science")
    sync_episode(_metadata(tmp_path, "item:old", old), missing)
    new = _doc("item:new", published_date="2026-11-01T00:00:00+00:00")
    new["feed"]["title"] = "New name"
    sync_episode(_metadata(tmp_path, "item:new", new), missing)
    older = _doc("item:older", published_date="2025-01-01T00:00:00+00:00")
    older["feed"].update(title="Oldest name", licence={"status": "declared"})
    sync_episode(_metadata(tmp_path, "item:older", older), missing)
    source = _row(conn, "sources", _source_of(new))
    assert (source["title"], source["category"]) == ("New name", "Science")
    assert json.loads(source["licence"]) == {"status": "declared"}
    assert OUTCOMES["no_files"] == 3


def test_equal_instants_go_to_the_greater_natural_id_whatever_the_order(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    missing = tmp_path / "none" / "x.txt"
    for episode_id in ("item:b", "item:a"):
        doc = _doc(episode_id)
        doc["feed"]["title"] = f"Title {episode_id}"
        sync_episode(_metadata(tmp_path, episode_id, doc), missing)
    assert _row(conn, "sources", _source_of(_doc()))["title"] == "Title item:b"


def test_an_item_not_served_changes_nothing_on_its_source(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    missing = tmp_path / "none" / "x.txt"
    sync_episode(_metadata(tmp_path, "item:1"), missing)
    unusable = _doc("item:2", published_date="2026-11-25T00:00:00+00:00")
    unusable["feed"].update(title="Unserved name", category="Comedy")
    unusable["episode"]["unusable"] = True
    sync_episode(_metadata(tmp_path, "item:2", unusable), missing)
    source = _row(conn, "sources", _source_of(unusable))
    assert (source["title"], source["category"]) == ("A show", None)
    assert _row(conn, "content_items", "item:2")["served"] == 0


def _dated(episode_id: str, day: str, **feed: object) -> dict:
    doc = _doc(episode_id, published_date=f"{day}T00:00:00+00:00")
    doc["feed"].update(feed)
    return doc


@pytest.mark.parametrize("order", ["abc", "acb", "bac", "bca", "cab", "cba"])
def test_each_column_comes_from_the_newest_served_item_that_states_it_in_any_order(
    pipeline_db: None, conn: Connection, tmp_path: Path, order: str  # noqa: F811
) -> None:
    # The database session's case (#2583 review): the newest item states no licence, so the
    # middle one's wins over the oldest one's, whichever arrives first.
    docs = {
        "a": _dated("item:a", "2026-11-01"),
        "c": _dated("item:c", "2026-06-01", licence={"status": "middle"}, category="Science"),
        "b": _dated("item:b", "2026-01-01", licence={"status": "oldest"}, category="History"),
    }
    for key in order:
        sync_episode(_metadata(tmp_path, f"item:{key}", docs[key]), None)
    source = _row(conn, "sources", _source_of(docs["a"]))
    assert json.loads(source["licence"]) == {"status": "middle"}
    assert source["category"] == "Science"
    assert source == {**source, **source_row_for(list(docs.values()))}


def test_an_item_that_stops_being_served_stops_deciding_its_source(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    old = _dated("item:old", "2026-01-01", title="Old name")
    new = _dated("item:new", "2026-11-01", title="New name")
    for doc in (old, new):
        sync_episode(_metadata(tmp_path, doc["episode"]["episode_id"], doc), None)
    assert _row(conn, "sources", _source_of(new))["title"] == "New name"
    conn.rollback()
    new["episode"]["unusable"] = True
    sync_episode(_metadata(tmp_path, "item:new", new), None)
    assert _row(conn, "sources", _source_of(new))["title"] == "Old name"
    assert _row(conn, "content_items", "item:new")["source_attrs"]["title"] == "New name"


def test_has_transcript_follows_the_items_transcript_documents(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    meta = _metadata(tmp_path, "item:t")
    assert sync_episode(meta, None) == "no_files"
    assert _row(conn, "content_items", "item:t")["has_transcript"] is False
    conn.rollback()
    assert sync_episode(meta, _write(tmp_path, BASE + ".txt", "Words.\n")) == "written"
    assert _row(conn, "content_items", "item:t")["has_transcript"] is True


def test_an_unreadable_publish_time_is_written_null_and_counted(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    corpus_writer.UNREADABLE.clear()
    doc = _doc("item:u", published_date="Tue, 3 Nov 2026 08:00 GMT")
    assert sync_episode(_metadata(tmp_path, "item:u", doc), None) == "no_files"
    assert _row(conn, "content_items", "item:u")["published_at"] is None
    assert corpus_writer.UNREADABLE == {"published_at": 1}


def test_an_item_whose_id_in_its_source_is_another_items_is_refused(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    missing = tmp_path / "none" / "x.txt"
    sync_episode(_metadata(tmp_path, "item:1"), missing)
    clash = _doc("item:2", guid="item:1")
    assert sync_episode(_metadata(tmp_path, "item:2", clash), missing) == "refused"
    found = conn.execute(text("SELECT count(*) FROM corpus.content_items")).scalar_one()
    assert found == 1


def test_an_item_moved_to_another_source_is_refused(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    missing = tmp_path / "none" / "x.txt"
    sync_episode(_metadata(tmp_path, "item:1"), missing)
    moved = _doc("item:1")
    moved["feed"]["feed_id"] = "sha256:" + "cd" * 32
    assert sync_episode(_metadata(tmp_path, "item:1", moved), missing) == "refused"
    assert conn.execute(text("SELECT count(*) FROM corpus.sources")).scalar_one() == 1


def test_a_held_lease_refuses_the_sync_and_writes_nothing(
    pipeline_db: None,
    conn: Connection,  # noqa: F811
    url: str,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = _write(tmp_path, BASE + ".txt", "Hello.\n")
    meta = _metadata(tmp_path, "item:l")
    other = create_engine(normalize_url(url))
    try:
        leases.acquire(other, [leases.LeaseRequest(leases.CORPUS, leases.EXCLUSIVE)], "migrate:t")
        assert sync_episode(meta, transcript) == "lease_refused"
        leases.release(other, "migrate:t")
        source = leases.source_resource(_source_of(_doc("item:l")))
        leases.acquire(other, [leases.LeaseRequest(source, leases.EXCLUSIVE)], "ingest:t")
        assert sync_episode(meta, transcript) == "lease_refused"
        monkeypatch.setenv(ENV_STRICT, "1")
        with pytest.raises(SyncFailed, match="lease_refused"):
            sync_episode(meta, transcript)
        leases.release(other, "ingest:t")
    finally:
        other.dispose()
    assert OUTCOMES["lease_refused"] == 3
    assert conn.execute(text("SELECT count(*) FROM corpus.content_items")).scalar_one() == 0
    assert conn.execute(text("SELECT count(*) FROM corpus.job_leases")).scalar_one() == 0


def test_a_lost_lease_rolls_the_sync_back(
    pipeline_db: None,
    conn: Connection,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def lost(*_args: object, **_kwargs: object) -> None:
        raise leases.LeaseLost("expired")

    monkeypatch.setattr(leases, "heartbeat", lost)
    transcript = _write(tmp_path, BASE + ".txt", "Hello.\n")
    assert sync_episode(_metadata(tmp_path, "item:x"), transcript) == "lease_lost"
    assert conn.execute(text("SELECT count(*) FROM corpus.sources")).scalar_one() == 0
    assert conn.execute(text("SELECT count(*) FROM corpus.job_leases")).scalar_one() == 0


def test_a_refused_item_is_counted_and_fails_only_when_strict(
    pipeline_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transcript = _write(tmp_path, BASE + ".txt", "x\n")
    _write(tmp_path, BASE + ".cleaned.adfree.txt", "x\n")
    meta = _metadata(tmp_path)
    assert sync_episode(meta, transcript) == "refused"
    monkeypatch.setenv(ENV_STRICT, "1")
    with pytest.raises(SyncFailed, match="refused"):
        sync_episode(meta, transcript)
    assert OUTCOMES["refused"] == 2


NEWS_BODY = "First paragraph.\n\nSecond paragraph.\n"


def _news_item(root: Path, body: str = NEWS_BODY, **fields: object) -> tuple[Path, Path]:
    """A news item as the text-in path writes it: metadata and body under a source run."""
    import hashlib

    run = root / "sources" / "news_article" / "example" / "run_20261124-060000"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(exist_ok=True)
    body_path = run / "transcripts" / "0001 - A headline.txt"
    body_path.write_text(body, encoding="utf-8")
    doc = {
        "content_type": "news_article",
        "item_id": "item:news_article:0001",
        "source_id": "source:0001",
        "url": "https://www.example.com/a/1",
        "title": "A headline",
        "published_at": "2026-11-22T01:00:09Z",
        "ingested_at": "2026-11-24T06:00:13Z",
        "fetched_at": "2026-11-24T06:00:10Z",
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "language": "en",
        "serveable": "summary",
        "news_article": {
            "editorial_type": "analysis",
            "text_status": "full",
            "extraction": "trafilatura",
            "extractor_version": "trafilatura/2.0.0",
        },
        **fields,
    }
    meta = run / "metadata" / "0001 - A headline.metadata.json"
    meta.write_text(json.dumps(doc), encoding="utf-8")
    return meta, body_path


@pytest.fixture
def news_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    from podcast_scraper.publishers import Publisher, Registry

    publisher = Publisher(
        id="org:example-times", name="The Example Times", domains=("example.com",)
    )
    monkeypatch.setattr(corpus_writer, "_REGISTRY", Registry(publishers=(publisher,)))


def test_a_news_item_writes_its_source_item_and_article(
    pipeline_db: None, news_registry: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    meta, body = _news_item(tmp_path)
    assert sync_episode(meta, body) == "written"
    item = _row(conn, "content_items", "item:news_article:0001")
    source = (
        conn.execute(select(sources).where(sources.c.source_id == item["source_id"]))
        .mappings()
        .one()
    )
    assert source["title"] == "The Example Times"
    assert item["source_attrs"]["title"] == "The Example Times"
    doc = (
        conn.execute(select(documents).where(documents.c.item_id == item["item_id"]))
        .mappings()
        .one()
    )
    assert (doc["kind"], doc["variant"], doc["version"], doc["change_kind"]) == (
        "article",
        "raw",
        1,
        "created",
    )
    assert doc["fetched_at"].isoformat() == "2026-11-24T06:00:10+00:00"
    assert (doc["extraction"], doc["extractor_version"]) == ("trafilatura", "trafilatura/2.0.0")
    assert _segment_texts(conn, item["item_id"], doc["document_id"]) == [
        "First paragraph.\n\n",
        "Second paragraph.\n",
    ]
    assert item["has_transcript"] is False
    count = conn.execute(text("SELECT count(*) FROM corpus.item_cards")).scalar_one()
    assert count == 0


def test_a_changed_article_is_a_corrected_version(
    pipeline_db: None, news_registry: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    meta, body = _news_item(tmp_path)
    assert sync_episode(meta, body) == "written"
    meta, body = _news_item(tmp_path, body="First paragraph, corrected.\n")
    assert sync_episode(meta, body) == "superseded"
    item = _row(conn, "content_items", "item:news_article:0001")["item_id"]
    rows = conn.execute(
        select(documents.c.version, documents.c.change_kind)
        .where(documents.c.item_id == item)
        .order_by(documents.c.version)
    ).all()
    assert [tuple(r) for r in rows] == [(1, "created"), (2, "corrected")]


def test_an_article_that_does_not_hash_to_its_body_sha256_is_refused(
    pipeline_db: None, news_registry: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    meta, body = _news_item(tmp_path, body_sha256="0" * 64)
    assert sync_episode(meta, body) == "refused"
    assert conn.execute(text("SELECT count(*) FROM corpus.content_items")).scalar_one() == 0


def _corpus_episode(root: Path, episode_id: str = "item:c", **episode: object) -> Path:
    run = root / "feeds" / "show" / "run_20261101-000000" / "metadata"
    run.mkdir(parents=True, exist_ok=True)
    return _metadata(run, episode_id, _doc(episode_id, **episode))


def test_an_episode_in_a_corpus_gets_its_card_and_flags(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    from podcast_scraper.db.corpus_cards import card_row

    meta = _corpus_episode(tmp_path)
    assert sync_episode(meta, None) == "no_files"
    item = _row(conn, "content_items", "item:c")
    card = (
        conn.execute(select(item_cards).where(item_cards.c.item_id == item["item_id"]))
        .mappings()
        .one()
    )
    rel = "feeds/show/run_20261101-000000/metadata/item_c.metadata.json"
    wanted = card_row(tmp_path, rel, item_row(_doc("item:c"), metadata_relpath=rel))
    assert wanted is not None
    for column, value in wanted.items():
        assert card[column] == value, column
    for flag in ("has_transcript", "has_gi", "has_kg", "has_moments"):
        assert item[flag] == wanted[flag], flag


def test_an_episode_that_becomes_unusable_loses_its_card(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    meta = _corpus_episode(tmp_path)
    assert sync_episode(meta, None) == "no_files"
    doc = json.loads(meta.read_text())
    doc["processing"] = {"unusable": {"reason": "silence"}}
    from podcast_scraper.languages import UNUSABLE_FIELD

    doc["episode"][UNUSABLE_FIELD] = {"reason": "silence"}
    meta.write_text(json.dumps(doc), encoding="utf-8")
    assert sync_episode(meta, None) == "no_files"
    count = conn.execute(text("SELECT count(*) FROM corpus.item_cards")).scalar_one()
    assert count == 0


def test_the_artifact_hooks_run_after_a_kept_sync_and_not_after_a_refused_one(
    pipeline_db: None,
    corpus_held: Callable[[], object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from podcast_scraper import stage_hooks

    calls: list[tuple[Path, Path | None, Path | None]] = []

    def record(m: Path, g: Path | None, k: Path | None) -> list[str]:
        calls.append((m, g, k))
        return []

    monkeypatch.setattr(stage_hooks, "artifact_rows", record)
    meta = _corpus_episode(tmp_path)
    gi = meta.with_name("item_c.gi.json")
    gi.write_text("{}", encoding="utf-8")
    assert sync_episode(meta, None) == "lease_refused"
    assert calls == []
    corpus_held()
    assert corpus_writer.flush_deferred(str(tmp_path)) == 0
    assert calls == [(meta, gi, None)]


def test_an_episode_that_becomes_unusable_has_its_card_flags_cleared(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    # As the importer leaves an item it writes no card for (review of #2597, point 3).
    from podcast_scraper.languages import UNUSABLE_FIELD

    meta = _corpus_episode(tmp_path)
    meta.with_name("item_c.gi.json").write_text("{}", encoding="utf-8")
    assert sync_episode(meta, None) == "no_files"
    assert _row(conn, "content_items", "item:c")["has_gi"] is True
    conn.rollback()
    doc = json.loads(meta.read_text())
    doc["episode"][UNUSABLE_FIELD] = {"reason": "silence"}
    meta.write_text(json.dumps(doc), encoding="utf-8")
    assert sync_episode(meta, None) == "no_files"
    item = _row(conn, "content_items", "item:c")
    assert (item["has_gi"], item["has_kg"], item["has_moments"]) == (False, False, False)


def test_a_card_that_cannot_be_read_refuses_the_item_and_is_not_retried(
    pipeline_db: None,
    conn: Connection,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The importer refuses such an item; failing it would retry it on every run (point 4).
    def corrupt(*_: object) -> None:
        raise ValueError("gi.json: Expecting value")

    monkeypatch.setattr(corpus_writer, "card_row", corrupt)
    meta = _corpus_episode(tmp_path)
    assert sync_episode(meta, None) == "refused"
    assert str(meta) not in corpus_writer._DEFERRED
    assert conn.execute(text("SELECT count(*) FROM corpus.content_items")).scalar_one() == 0


def test_a_raising_artifact_hook_neither_stops_the_next_nor_escapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from podcast_scraper import stage_hooks

    seen: list[str] = []

    def broken(*_: object) -> str:
        raise RuntimeError("gi rows")

    def after(*_: object) -> str:
        seen.append("after")
        return "written"

    monkeypatch.setattr(
        stage_hooks, "_all", lambda: [SimpleNamespace(artifact_rows=[broken, after])]
    )
    meta = tmp_path / "x.metadata.json"
    assert stage_hooks.artifact_rows(meta, None, None) == ["failed", "written"]
    assert seen == ["after"]


def test_a_failed_artifact_hook_is_counted_and_raised_only_when_strict(
    pipeline_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper import stage_hooks

    monkeypatch.setattr(stage_hooks, "artifact_rows", lambda *_: ["failed", "written"])
    corpus_writer.OUTCOMES.clear()
    meta = _corpus_episode(tmp_path)
    assert sync_episode(meta, None) == "no_files"
    assert corpus_writer.OUTCOMES["hook_failed"] == 1
    assert corpus_writer._DEFERRED[str(meta)] == (None, corpus_writer.RETRANSCRIBED)
    monkeypatch.setattr(stage_hooks, "artifact_rows", lambda *_: ["written"])
    assert sync_episode(meta, None) == "no_files"
    assert str(meta) not in corpus_writer._DEFERRED
    monkeypatch.setattr(stage_hooks, "artifact_rows", lambda *_: ["failed", "written"])
    monkeypatch.setenv(corpus_writer.ENV_STRICT, "1")
    with pytest.raises(corpus_writer.SyncFailed, match="1 artifact_rows hooks failed"):
        sync_episode(meta, None)


def test_every_variant_is_written_with_its_own_cuts(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, BASE + ".txt", "Raw one.\nRaw two.\n")
    _write(tmp_path, BASE + ".cleaned.txt", "Clean one.\n")
    _write(tmp_path, BASE + ".adfree.txt", "Free one.\nFree two.\nFree three.\n")
    item = _item(conn, _source(conn))
    synced = sync_documents(conn, item, _files(tmp_path), change_kind=RETRANSCRIBED)
    assert sorted((d.variant, d.segments) for d in synced) == [
        ("adfree", 3),
        ("cleaned", 1),
        ("raw", 2),
    ]
    conn.rollback()


def test_a_translated_item_writes_its_source_raw_and_its_render_en(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, BASE + ".txt", "Hello.\n")
    _write(tmp_path, BASE + ".es.txt", "Hola.\n")
    _write(tmp_path, BASE + ".translation.json", "{}")
    for name, words in ((".segments.json", "Hello."), (".es.segments.json", "Hola.")):
        cue = [{"start": 0.0, "end": 1.0, "text": words, "char_start": 0, "char_end": 6}]
        _write(tmp_path, BASE + name, json.dumps(cue))
    item = _item(conn, _source(conn))
    files = transcript_files(tmp_path, BASE, "es")
    synced = {d.variant: d for d in sync_documents(conn, item, files, change_kind=RETRANSCRIBED)}
    assert set(synced) == {"en", "raw"}
    assert joined_body(conn, item, synced["raw"].document_id) == "Hola.\n"
    assert joined_body(conn, item, synced["en"].document_id) == "Hello.\n"
    conn.rollback()


def test_no_transcript_body_is_its_own_outcome(pipeline_db: None, tmp_path: Path) -> None:
    meta = _metadata(tmp_path)
    assert sync_episode(meta, tmp_path / "gone" / f"{BASE}.txt") == "no_files"
    assert sync_episode(meta, tmp_path / f"{BASE}.vtt") == "no_files"
    assert OUTCOMES["no_files"] == 2


def test_a_later_version_is_counted_superseded(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    meta = _metadata(tmp_path, "item:s1")
    transcript = _write(tmp_path, BASE + ".txt", "First.\n")
    assert sync_episode(meta, transcript) == "written"
    _write(tmp_path, BASE + ".txt", "Second.\n")
    assert sync_episode(meta, transcript) == "superseded"
    item = _row(conn, "content_items", "item:s1")["item_id"]
    assert _versions(conn, item) == [(1, "created", False), (2, "retranscribed", True)]


def test_anything_else_is_counted_failed(
    pipeline_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse() -> None:
        raise RuntimeError("database gone")

    monkeypatch.setattr(engines, "pipeline_rw_engine", refuse)
    transcript = _write(tmp_path, BASE + ".txt", "x\n")
    assert sync_episode(_metadata(tmp_path), transcript) == "failed"
    assert OUTCOMES["failed"] == 1


def test_a_failed_write_is_kept_and_a_later_retry_syncs_it(
    pipeline_db: None,
    conn: Connection,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = engines.pipeline_rw_engine

    def gone() -> None:
        raise RuntimeError("database gone")

    monkeypatch.setattr(engines, "pipeline_rw_engine", gone)
    meta = _metadata(tmp_path, "item:f")
    assert sync_episode(meta, None) == "failed"
    assert corpus_writer.flush_deferred(str(tmp_path)) == 1
    assert (tmp_path / corpus_writer.DEFERRED_FILE).exists()
    monkeypatch.setattr(engines, "pipeline_rw_engine", real)
    assert corpus_writer.retry_deferred(str(tmp_path)) == 0
    assert not (tmp_path / corpus_writer.DEFERRED_FILE).exists()
    assert _row(conn, "content_items", "item:f")["served"] == 1


def test_a_refused_item_is_not_kept_for_a_retry(pipeline_db: None, tmp_path: Path) -> None:
    transcript = _write(tmp_path, BASE + ".txt", "x\n")
    _write(tmp_path, BASE + ".cleaned.adfree.txt", "x\n")
    assert sync_episode(_metadata(tmp_path), transcript) == "refused"
    assert corpus_writer._DEFERRED == {}


def test_a_run_that_raises_keeps_its_deferred_episodes_without_retrying(
    pipeline_db: None,
    corpus_held: Callable[[], object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = _metadata(tmp_path, "item:z")
    assert sync_episode(meta, None) == "lease_refused"
    corpus_held()
    tried: list[str] = []
    monkeypatch.setattr(corpus_writer, "sync_episode", lambda path, *a, **k: tried.append(path))
    assert corpus_writer.flush_deferred(str(tmp_path), retry=False) == 1
    assert tried == []
    kept = (tmp_path / corpus_writer.DEFERRED_FILE).read_text().splitlines()
    assert [json.loads(line)["metadata"] for line in kept] == [str(meta)]


def test_the_metadata_hook_joins_a_relative_path_keeps_an_absolute_one_and_syncs_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from podcast_scraper.workflow import metadata_generation

    calls: list[tuple[Path, Path | None]] = []
    monkeypatch.setattr(
        corpus_writer,
        "sync_episode",
        lambda meta, transcript, change_kind: calls.append((meta, transcript)),
    )
    hook = metadata_generation._sync_corpus_rows
    hook("/out/metadata/e.metadata.json", "/out", "transcripts/e.txt")
    hook("/out/metadata/e.metadata.json", "/elsewhere", "/app/output/transcripts/e.txt")
    hook("/out/metadata/e.metadata.json", "/out", None)
    assert calls == [
        (Path("/out/metadata/e.metadata.json"), Path("/out/transcripts/e.txt")),
        (Path("/out/metadata/e.metadata.json"), Path("/app/output/transcripts/e.txt")),
        (Path("/out/metadata/e.metadata.json"), None),
    ]


@pytest.mark.parametrize(
    "stage, kind",
    [
        ("full", RETRANSCRIBED),
        (None, RETRANSCRIBED),
        ("retranscript_only", RETRANSCRIBED),
        ("translate_only", "translated"),
        ("repunctuate_only", "repunctuated"),
        ("relabel_only", "corrected"),
        ("rediarize_only", "corrected"),
        ("rederive_only", "corrected"),
    ],
)
def test_the_metadata_hook_records_the_change_its_runs_stage_makes(
    stage: str | None, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from podcast_scraper.workflow import metadata_generation

    kinds: list[str] = []
    monkeypatch.setattr(
        corpus_writer,
        "sync_episode",
        lambda meta, transcript, change_kind: kinds.append(change_kind),
    )
    metadata_generation._sync_corpus_rows("/out/metadata/e.metadata.json", "/out", None, stage)
    assert kinds == [kind]


def test_a_translate_only_rerun_records_its_new_version_as_translated(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    from podcast_scraper.workflow import metadata_generation

    meta = _metadata(tmp_path, "item:t")
    _write(tmp_path, BASE + ".txt", "First.\n")
    assert metadata_generation._sync_corpus_rows(str(meta), str(tmp_path), BASE + ".txt") == (
        "written"
    )
    _write(tmp_path, BASE + ".txt", "Second.\n")
    assert (
        metadata_generation._sync_corpus_rows(
            str(meta), str(tmp_path), BASE + ".txt", "translate_only"
        )
        == "superseded"
    )
    item = _row(conn, "content_items", "item:t")["item_id"]
    assert _versions(conn, item) == [(1, "created", False), (2, "translated", True)]


def test_an_episode_with_no_transcript_still_gets_its_rows(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    assert sync_episode(_metadata(tmp_path, "item:n"), None) == "no_files"
    assert _row(conn, "content_items", "item:n")["served"] == 1


def test_after_an_import_a_synced_item_keeps_the_columns_its_imported_items_state(
    pipeline_db: None, url: str, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    from podcast_scraper.db.importers.corpus_items import import_corpus
    from podcast_scraper.publishers import Registry

    run = tmp_path / "feeds" / "show" / "run_20261101-000000" / "metadata"
    run.mkdir(parents=True)
    old = _doc("item:old", published_date="2026-01-01T00:00:00+00:00")
    old["feed"]["category"] = "Science"
    _metadata(run, "item:old", old)
    engine = create_engine(normalize_url(url))
    with engine.begin() as db:
        assert import_corpus(db, tmp_path, registry=Registry(publishers=())).refused == []
    engine.dispose()
    new = _doc("item:new", published_date="2026-11-01T00:00:00+00:00")
    assert sync_episode(_metadata(run, "item:new", new), None) == "no_files"
    assert _row(conn, "sources", _source_of(new))["category"] == "Science"


@pytest.mark.parametrize(
    "path, rel",
    [
        ("/srv/c/feeds/f/run_1/metadata/e.metadata.json", "feeds/f/run_1/metadata/e.metadata.json"),
        (
            "/srv/c/sources/news/s/run_1/metadata/e.metadata.json",
            "sources/news/s/run_1/metadata/e.metadata.json",
        ),
        ("/srv/c/run_1/metadata/e.metadata.json", "run_1/metadata/e.metadata.json"),
        ("/srv/c/metadata/e.metadata.json", "metadata/e.metadata.json"),
        ("/srv/c/elsewhere/e.metadata.json", None),
    ],
)
def test_the_corpus_relpath_is_the_longest_end_in_a_corpus_layout(path: str, rel: object) -> None:
    assert corpus_writer.corpus_relpath(Path(path)) == rel


def test_a_sync_stores_its_corpus_path_and_bridge(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    run = tmp_path / "feeds" / "show" / "run_1" / "metadata"
    run.mkdir(parents=True)
    meta = _metadata(run, "item:p")
    meta.with_name("item_p.bridge.json").write_text("{}", encoding="utf-8")
    assert sync_episode(meta, None) == "no_files"
    attrs = _row(conn, "content_items", "item:p")["type_attrs"]
    assert attrs["metadata_relative_path"] == "feeds/show/run_1/metadata/item_p.metadata.json"
    assert attrs["has_bridge"] is True
    assert corpus_writer.UNPLACED["metadata_relative_path"] == 0


def test_a_metadata_file_in_no_corpus_layout_is_counted(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    assert sync_episode(_metadata(tmp_path, "item:u"), None) == "no_files"
    assert "metadata_relative_path" not in _row(conn, "content_items", "item:u")["type_attrs"]
    assert corpus_writer.UNPLACED["metadata_relative_path"] == 1


def test_the_sources_type_attrs_follow_its_newest_item(
    pipeline_db: None, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    old = _doc("item:o", published_date="2026-01-01T00:00:00+00:00")
    old["feed"]["description"] = "Old blurb."
    old["feed"]["url"] = "https://example/feed.xml"
    new = _doc("item:n", published_date="2026-11-01T00:00:00+00:00")
    new["feed"]["description"] = "New blurb."
    for doc in (new, old):
        assert sync_episode(_metadata(tmp_path, doc["episode"]["episode_id"], doc), None) == (
            "no_files"
        )
    attrs = _row(conn, "sources", _source_of(new))["type_attrs"]
    assert attrs == {"feed_description": "New blurb.", "feed_rss_url": "https://example/feed.xml"}


def test_a_reimport_after_a_sync_is_accepted(
    pipeline_db: None, url: str, tmp_path: Path  # noqa: F811
) -> None:
    from podcast_scraper.db.importers.corpus_items import import_corpus
    from podcast_scraper.publishers import Registry

    run = tmp_path / "feeds" / "show" / "run_20261101-000000" / "metadata"
    run.mkdir(parents=True)
    meta = _metadata(run, "item:b")
    meta.with_name("item_b.bridge.json").write_text("{}", encoding="utf-8")
    assert sync_episode(meta, None) == "no_files"
    engine = create_engine(normalize_url(url))
    with engine.begin() as db:
        assert import_corpus(db, tmp_path, registry=Registry(publishers=())).refused == []
    engine.dispose()


def _segment_texts(db: Connection, item_id: int, document_id: int) -> list[str]:
    return [
        r[0]
        for r in db.execute(
            select(segments.c.text)
            .where(and_(segments.c.item_id == item_id, segments.c.document_id == document_id))
            .order_by(segments.c.seq)
        )
    ]
