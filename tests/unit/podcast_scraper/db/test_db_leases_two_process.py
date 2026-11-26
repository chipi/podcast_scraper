"""DECISIONS section 4's three two-process scenarios, each side its own process (lease_holder)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest
from sqlalchemy import create_engine, text

from podcast_scraper.db.settings import normalize_url
from tests._database import worker_database_url
from tests.unit.podcast_scraper.db.test_db_migrate import REPO

INGEST = ["corpus=shared", "source:podcast:a=exclusive"]


@pytest.fixture
def url() -> Iterator[str]:
    found = worker_database_url()
    assert found, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    yield found
    engine = create_engine(normalize_url(found))
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM corpus.job_leases"))
        conn.execute(text("DELETE FROM corpus.corpus_meta WHERE key LIKE 'lease-test-%'"))
    engine.dispose()


class Holder:
    """One lease_holder process."""

    def __init__(self, url: str, holder: str, *specs: str, ttl: float = 600) -> None:
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tests.unit.podcast_scraper.db.lease_holder",
                url,
                holder,
                str(ttl),
                *specs,
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO}"},
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert self.proc.stdout is not None
        self.first = self.proc.stdout.readline().strip()

    def send(self, line: str) -> str:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.send("release")
        self.proc.wait(timeout=30)


def _committed(url: str, key: str) -> bool:
    engine = create_engine(normalize_url(url))
    with engine.connect() as conn:
        found = conn.execute(
            text("SELECT count(*) FROM corpus.corpus_meta WHERE key = :k"), {"k": key}
        ).scalar_one()
    engine.dispose()
    return bool(found)


def test_a_migration_is_refused_during_an_ingest(url: str) -> None:
    ingest = Holder(url, "ingest:1", *INGEST)
    try:
        assert ingest.first == "HELD 2"
        migration = Holder(url, "upgrade:1", "corpus=exclusive")
        assert migration.first.startswith("REFUSED corpus exclusive refused: held by ingest:1")
        assert migration.proc.wait(timeout=30) == 1
        assert ingest.send("chunk lease-test-ingest") == "COMMITTED"
    finally:
        ingest.close()
    assert _committed(url, "lease-test-ingest")


def test_two_ingests_of_one_source_serialise(url: str) -> None:
    first = Holder(url, "ingest:1", *INGEST)
    try:
        assert first.first == "HELD 2"
        second = Holder(url, "ingest:2", *INGEST)
        assert second.first.startswith("REFUSED source:podcast:a exclusive refused")
        second.proc.wait(timeout=30)
    finally:
        first.close()
    after = Holder(url, "ingest:2", *INGEST)
    try:
        assert after.first == "HELD 2"
    finally:
        after.close()


def test_an_expired_lease_cannot_commit(url: str) -> None:
    wedged = Holder(url, "ingest:1", *INGEST, ttl=1)
    try:
        assert wedged.first == "HELD 2"
        time.sleep(1.5)
        migration = Holder(url, "upgrade:1", "corpus=exclusive")
        assert migration.first == "HELD 1"
        assert wedged.send("chunk lease-test-wedged") == "LOST"
        migration.close()
    finally:
        wedged.close()
    assert not _committed(url, "lease-test-wedged")
