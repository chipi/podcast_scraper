"""A blocked upstream must ABORT the run and record NOTHING (#2163).

Two incidents, same shape. On 2026-09-16 Wikimedia 403-blocked the egress IP mid-run; because
``_get_json`` collapsed every failure to ``None``, 911 entities were written as authoritative
misses with a 30-day TTL — a transient block became a month of missing data that then read as
"nothing left to enrich". That specific collapse was fixed. What was still missing on 2026-09-27,
when it happened again, was DETECTION: 403 is not in the retry forcelist, so each call raised
``TransientFetchError``, which the per-entity loop swallows and skips. The run would walk its whole
budget into a wall at ~1 req/s and report ``status: ok`` with zero rows.

So there are two properties, and both are load-bearing:

1. the run STOPS — it does not issue a request per remaining entity
2. nothing is written — no miss, no empty-reason, no skip. A block is not evidence about any entity.

Provider is a fake; no network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.enrichers import person_web
from podcast_scraper.enrichment.enrichers.person_web import (
    PersonWebEnricher,
    set_web_block_breaker,
    TransientFetchError,
    WebBlockBreaker,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext
from podcast_scraper.enrichment.resilience import UpstreamBlockedError

pytestmark = pytest.mark.integration

_IDS = tuple(f"person:p{i:02d}" for i in range(25))
_GI = {
    "nodes": [
        {"id": pid, "type": "Person", "properties": {"name": pid.split(":", 1)[1]}} for pid in _IDS
    ]
}

BLOCK_BODY = "Please respect our robot policy https://w.wiki/4wJS when crawling us."


@pytest.fixture(autouse=True)
def _fresh_breaker():
    set_web_block_breaker(None)
    yield
    set_web_block_breaker(None)


def _bundle(tmp_path: Path) -> EpisodeArtifactBundle:
    gi = tmp_path / "a.gi.json"
    gi.write_text(json.dumps(_GI), encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=gi, gi_path=gi, kg_path=None, bridge_path=None, episode_id="a", stem="a"
    )


def _ctx(run_id: str = "r1") -> RunContext:
    return RunContext(
        run_id=run_id,
        parent_run_id=None,
        enricher_id="person_web",
        enricher_version="0.1.0",
        tier="web",
        attempt=1,
        job_id=run_id,
        cancel_event=asyncio.Event(),
    )


class _BlockedProvider:
    """Every fetch behaves like Wikimedia's edge block: 403 with the robot-policy body."""

    name = "fake-blocked"

    def __init__(self) -> None:
        self.attempts: list[str] = []

    def fetch_raw(self, person_id, display_name):
        self.attempts.append(person_id)
        breaker = person_web.web_block_breaker()
        breaker.raise_if_open()
        breaker.note_response(403, BLOCK_BODY, "https://www.wikidata.org/w/api.php")
        breaker.raise_if_open()
        raise TransientFetchError("HTTP 403")

    def derive(self, person_id, display_name, raw):  # pragma: no cover - never reached
        raise AssertionError("derive must not run when every fetch is blocked")


def _run(tmp_path: Path, provider) -> object:
    enricher = PersonWebEnricher(provider=provider)
    return asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=tmp_path,
            all_bundles=[_bundle(tmp_path)],
            config={"max_persons": len(_IDS)},
            ctx=_ctx(),
        )
    )


def test_a_block_stops_the_run_instead_of_walking_the_budget(tmp_path: Path) -> None:
    """Budget is 25 entities; a block must cost ONE request, not 25."""
    provider = _BlockedProvider()

    _run(tmp_path, provider)

    assert len(provider.attempts) == 1, (
        f"expected the run to abort after the first block, but it attempted "
        f"{len(provider.attempts)} entities: {provider.attempts}"
    )


def test_a_block_writes_no_misses(tmp_path: Path) -> None:
    """The 2026-09-16 damage: a block recorded as 911 authoritative 30-day misses.

    A block says nothing about whether an entity has an article, so nothing may be persisted.
    """
    _run(tmp_path, _BlockedProvider())

    raw_dir = tmp_path / "enrichments" / "person_web_raw"
    written = sorted(p.name for p in raw_dir.glob("*.json")) if raw_dir.is_dir() else []

    assert written == [], f"a blocked run must persist nothing, found {written}"


def test_the_run_reports_failure_not_ok(tmp_path: Path) -> None:
    """Reporting ``ok`` after fetching nothing is how this stayed invisible for two incidents."""
    result = _run(tmp_path, _BlockedProvider())

    assert getattr(result, "status", None) == "failed"
    assert "Blocked" in (getattr(result, "error_class", "") or "") or "blocked" in (
        (getattr(result, "error", "") or "").lower()
    )


def test_a_later_run_is_not_poisoned_by_an_earlier_block(tmp_path: Path) -> None:
    """The breaker is per-RUN. Latching per process would leave the long-lived API container
    failing every future enrichment until someone restarted it."""
    breaker = WebBlockBreaker()
    breaker.reset_for_run("older-run")
    breaker.note_response(403, BLOCK_BODY, "https://www.wikidata.org/w/api.php")
    set_web_block_breaker(breaker)
    assert breaker.is_open

    provider = _BlockedProvider()
    _run(tmp_path, provider)  # ctx.run_id == "r1", a different run

    # It re-detected the block on its own (one attempt), rather than inheriting the stale trip
    # (which would have been zero attempts).
    assert len(provider.attempts) == 1


def test_blocked_error_is_not_swallowed_by_the_entity_loop() -> None:
    """Guard the type relationship the abort depends on."""
    assert not issubclass(UpstreamBlockedError, TransientFetchError)
