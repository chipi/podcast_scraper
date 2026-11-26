"""Corpus job leases (db.leases): modes, all-or-nothing, expiry, heartbeat, a race for one lease."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from podcast_scraper.db.leases import (
    acquire,
    CORPUS,
    EXCLUSIVE,
    heartbeat,
    LeaseRefused,
    LeaseRequest,
    release,
    SHARED,
    source_resource,
)
from podcast_scraper.db.settings import normalize_url
from tests._database import worker_database_url

EXPIRED = timedelta(seconds=-1)


@pytest.fixture
def engine() -> Iterator[Engine]:
    url = worker_database_url()
    assert url, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    made = create_engine(normalize_url(url))
    yield made
    with made.begin() as conn:
        conn.execute(text("DELETE FROM corpus.job_leases"))
    made.dispose()


def _held(engine: Engine) -> set[tuple[str, str, str]]:
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT resource, mode, holder FROM corpus.job_leases")).all()
    return {(r.resource, r.mode, r.holder) for r in rows}


def _ingest(source: str = "podcast:a") -> list[LeaseRequest]:
    return [LeaseRequest(CORPUS, SHARED), LeaseRequest(source_resource(source), EXCLUSIVE)]


def test_shared_leases_share_and_an_exclusive_one_excludes_both_ways(engine: Engine) -> None:
    acquire(engine, [LeaseRequest(CORPUS, SHARED)], "enrich:1")
    acquire(engine, [LeaseRequest(CORPUS, SHARED)], "enrich:2")
    with pytest.raises(LeaseRefused, match="held by enrich:"):
        acquire(engine, [LeaseRequest(CORPUS, EXCLUSIVE)], "upgrade:1")
    release(engine, "enrich:1")
    release(engine, "enrich:2")
    acquire(engine, [LeaseRequest(CORPUS, EXCLUSIVE)], "upgrade:1")
    with pytest.raises(LeaseRefused, match=r"upgrade:1 \(exclusive\)"):
        acquire(engine, [LeaseRequest(CORPUS, SHARED)], "enrich:3")
    assert _held(engine) == {(CORPUS, EXCLUSIVE, "upgrade:1")}


def test_a_migration_is_refused_during_an_ingest(engine: Engine) -> None:
    acquire(engine, _ingest(), "ingest:7")
    with pytest.raises(LeaseRefused):
        acquire(engine, [LeaseRequest(CORPUS, EXCLUSIVE)], "upgrade:1")
    assert {h for _, _, h in _held(engine)} == {"ingest:7"}


def test_two_ingests_of_one_source_serialise_and_of_two_sources_run_together(
    engine: Engine,
) -> None:
    acquire(engine, _ingest("podcast:a"), "ingest:1")
    with pytest.raises(LeaseRefused, match="source:podcast:a exclusive refused"):
        acquire(engine, _ingest("podcast:a"), "ingest:2")
    # All or nothing: the refused ingest did not keep its corpus lease either.
    assert {h for _, _, h in _held(engine)} == {"ingest:1"}
    acquire(engine, _ingest("podcast:b"), "ingest:2")
    assert len(_held(engine)) == 4


def test_an_expired_lease_is_free_and_its_heartbeat_extends_nothing(engine: Engine) -> None:
    acquire(engine, [LeaseRequest(CORPUS, EXCLUSIVE)], "upgrade:old", ttl=EXPIRED)
    with engine.begin() as conn:
        assert heartbeat(conn, "upgrade:old") == 0
    acquire(engine, [LeaseRequest(CORPUS, EXCLUSIVE)], "upgrade:new")
    assert _held(engine) == {(CORPUS, EXCLUSIVE, "upgrade:new")}
    with engine.begin() as conn:
        assert heartbeat(conn, "upgrade:new") == 1


def test_a_holder_retaking_its_own_lease_refreshes_it(engine: Engine) -> None:
    acquire(engine, _ingest(), "ingest:1")
    acquire(engine, _ingest(), "ingest:1")
    assert len(_held(engine)) == 2


def test_a_bad_mode_is_refused(engine: Engine) -> None:
    with pytest.raises(ValueError, match="not shared or exclusive"):
        acquire(engine, [LeaseRequest(CORPUS, "both")], "x:1")


def test_two_takers_racing_for_one_exclusive_lease_get_one_winner(engine: Engine) -> None:
    for _ in range(20):
        start = threading.Barrier(2)
        won: list[str] = []
        refused: list[str] = []

        def take(holder: str) -> None:
            start.wait()
            try:
                acquire(engine, [LeaseRequest(CORPUS, EXCLUSIVE)], holder)
                won.append(holder)
            except LeaseRefused:
                refused.append(holder)

        threads = [threading.Thread(target=take, args=(f"upgrade:{n}",)) for n in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(won) == 1 and len(refused) == 1
        release(engine, won[0])
