"""An enricher's outputs onto the rows after its run (db.enrichment_rows): a build is read from
every served item's metadata file, so an episode this run did not touch keeps its output, and the
outcome follows the row writer's failure rule."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection

from podcast_scraper.db import corpus_writer, enrichment_rows, leases, projections, settings
from podcast_scraper.db.importers.corpus_enrichments import publish_enrichments
from podcast_scraper.db.importers.corpus_items import import_corpus
from podcast_scraper.db.settings import normalize_url
from podcast_scraper.enrichment.executor import EnrichmentExecutor
from tests.unit.podcast_scraper.db.test_corpus_writer import pipeline_db  # noqa: F401
from tests.unit.podcast_scraper.db.test_db_corpus import conn, url  # noqa: F401
from tests.unit.podcast_scraper.db.test_db_import_corpus import _REGISTRY
from tests.unit.podcast_scraper.db.test_db_import_enrich import _corpus, _envelope, _RUN

_META = f"feeds/rss_example/{_RUN}/metadata/0001 - One.metadata.json"
_OUT = f"feeds/rss_example/{_RUN}/metadata/enrichments/0001 - One.episode_note.json"


def _episode_output(root: Path, value: int) -> None:
    path = root / _OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _envelope("episode_note", {"episode_id": "ep-1", "v": value})
    path.write_text(json.dumps(payload), encoding="utf-8")


def _published(conn: Connection) -> list[tuple[Any, ...]]:  # noqa: F811
    return [
        tuple(r)
        for r in conn.execute(
            text(
                "SELECT o.enricher_id, o.output->>'v' FROM corpus.enrichment_outputs o"
                " JOIN corpus.enrichment_builds b USING (build_id)"
                " WHERE b.state = 'published' AND o.item_id IS NOT NULL ORDER BY 1"
            )
        ).all()
    ]


def test_a_publish_over_every_served_metadata_file_keeps_an_untouched_episode(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _corpus(tmp_path)
    _episode_output(tmp_path, 1)
    import_corpus(conn, tmp_path, registry=_REGISTRY)
    relpaths = enrichment_rows.served_metadata_relpaths(conn)
    assert relpaths == [_META]
    # This run's episodes (none here) would drop the episode's output; every served one keeps it.
    _episode_output(tmp_path, 2)
    publish_enrichments(conn, tmp_path, relpaths, "episode_note")
    assert _published(conn) == [("episode_note", "2")]
    conn.rollback()


def test_publish_after_run_publishes_on_the_pipeline_engine_under_its_leases(
    pipeline_db: None, url: str, conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _corpus(tmp_path)
    _episode_output(tmp_path, 3)
    engine = create_engine(normalize_url(url))
    with engine.begin() as db:
        import_corpus(db, tmp_path, registry=_REGISTRY)
    other = enrichment_rows.enricher_resource("episode_note")
    leases.acquire(engine, [leases.LeaseRequest(other, leases.EXCLUSIVE)], "another:t")
    assert enrichment_rows.publish_after_run(tmp_path, "episode_note") == "lease_refused"
    leases.release(engine, "another:t")
    engine.dispose()
    assert enrichment_rows.publish_after_run(tmp_path, "episode_note") == "published"
    assert _published(conn) == [("episode_note", "3")]
    assert conn.execute(text("SELECT count(*) FROM corpus.job_leases")).scalar_one() == 0
    assert enrichment_rows.pending(tmp_path) == set()


def test_publish_after_run_counts_its_outcome_and_raises_only_when_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    enrichment_rows.OUTCOMES.clear()
    monkeypatch.setattr(settings, "pipeline_is_configured", lambda: False)
    assert enrichment_rows.publish_after_run(tmp_path, "x") == "not_configured"
    monkeypatch.setattr(settings, "pipeline_is_configured", lambda: True)
    after: list[str] = []

    def built() -> str:
        after.append("built")
        return "published"

    monkeypatch.setattr(projections, "after_batch", built)
    monkeypatch.setattr(enrichment_rows, "_publish", lambda root, enricher_id: None)
    assert enrichment_rows.publish_after_run(tmp_path, "x") == "published"
    assert after == []
    assert enrichment_rows.publish_after_run(tmp_path, "topic_theme_clusters") == "published"
    # The groupings' enricher is followed by the projections, so reach counts its new build.
    assert after == ["built"]

    def refused(root: Path, enricher_id: str) -> None:
        raise leases.LeaseRefused("held")

    def broken(root: Path, enricher_id: str) -> None:
        raise RuntimeError("database down")

    monkeypatch.setattr(enrichment_rows, "_publish", refused)
    assert enrichment_rows.publish_after_run(tmp_path, "x") == "lease_refused"
    monkeypatch.setattr(enrichment_rows, "_publish", broken)
    monkeypatch.delenv(corpus_writer.ENV_STRICT, raising=False)
    assert enrichment_rows.publish_after_run(tmp_path, "x") == "failed"
    monkeypatch.setenv(corpus_writer.ENV_STRICT, "1")
    with pytest.raises(corpus_writer.SyncFailed):
        enrichment_rows.publish_after_run(tmp_path, "x")
    assert enrichment_rows.OUTCOMES == {
        "not_configured": 1,
        "published": 2,
        "lease_refused": 1,
        "failed": 2,
    }


def test_the_executor_publishes_only_an_enricher_that_wrote_an_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    published: list[str] = []

    def publish(root: Path, enricher_id: str) -> str:
        published.append(enricher_id)
        return "published"

    monkeypatch.setattr(enrichment_rows, "publish_after_run", publish)
    executor = EnrichmentExecutor(
        corpus_root=tmp_path, registry=MagicMock(), enricher_set=MagicMock()
    )
    executor._wrote.add("wrote")
    asyncio.run(executor._publish_rows("wrote"))
    asyncio.run(executor._publish_rows("skipped"))
    assert published == ["wrote"]


def test_a_publish_that_did_not_land_is_retried_by_a_run_that_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The staleness gate skips an enricher whose inputs did not change: its next run writes no
    # envelope, so without the pending set its files would stay newer than its build.
    monkeypatch.setattr(settings, "pipeline_is_configured", lambda: True)
    attempts: list[str] = []

    def refused(root: Path, enricher_id: str) -> None:
        attempts.append(enricher_id)
        raise leases.LeaseRefused("held")

    monkeypatch.setattr(enrichment_rows, "_publish", refused)
    monkeypatch.delenv(corpus_writer.ENV_STRICT, raising=False)
    first = EnrichmentExecutor(corpus_root=tmp_path, registry=MagicMock(), enricher_set=MagicMock())
    first._wrote.add("note")
    asyncio.run(first._publish_rows("note"))
    assert enrichment_rows.pending(tmp_path) == {"note"}

    monkeypatch.setattr(enrichment_rows, "_publish", lambda root, enricher_id: None)
    gated = EnrichmentExecutor(corpus_root=tmp_path, registry=MagicMock(), enricher_set=MagicMock())
    asyncio.run(gated._publish_rows("note"))
    asyncio.run(gated._publish_rows("other"))
    assert attempts == ["note"]
    assert enrichment_rows.pending(tmp_path) == set()


@pytest.mark.parametrize("body", ["{not json", '{"note": 1}', "[1, 2]"])
def test_an_unreadable_pending_file_reads_as_empty_and_is_rewritten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> None:
    # Database's review of #2616: a corrupt file raised inside _publish_rows and failed the run.
    path = enrichment_rows.pending_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    assert enrichment_rows.pending(tmp_path) == set()
    executor = EnrichmentExecutor(
        corpus_root=tmp_path, registry=MagicMock(), enricher_set=MagicMock()
    )
    asyncio.run(executor._publish_rows("note"))
    monkeypatch.setattr(settings, "pipeline_is_configured", lambda: True)

    def refused(root: Path, enricher_id: str) -> None:
        raise leases.LeaseRefused("held")

    monkeypatch.setattr(enrichment_rows, "_publish", refused)
    assert enrichment_rows.publish_after_run(tmp_path, "note") == "lease_refused"
    assert enrichment_rows.pending(tmp_path) == {"note"}


def test_a_failed_publish_stays_pending_under_strict_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(settings, "pipeline_is_configured", lambda: True)

    def broken(root: Path, enricher_id: str) -> None:
        raise RuntimeError("database down")

    monkeypatch.setattr(enrichment_rows, "_publish", broken)
    monkeypatch.setenv(corpus_writer.ENV_STRICT, "1")
    with pytest.raises(corpus_writer.SyncFailed):
        enrichment_rows.publish_after_run(tmp_path, "note")
    assert enrichment_rows.pending(tmp_path) == {"note"}
