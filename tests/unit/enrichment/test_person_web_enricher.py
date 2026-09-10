"""Unit tests for the person_web enricher (wave-G, EnricherTier.WEB) — fixtured, no network."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.enrichers import person_web
from podcast_scraper.enrichment.enrichers.person_web import (
    PersonWebEnricher,
    PersonWebInfo,
    WikipediaProvider,
)
from podcast_scraper.enrichment.protocol import (
    EnricherScope,
    EnricherTier,
    EpisodeArtifactBundle,
    RunContext,
)

pytestmark = pytest.mark.unit


def _bundle(stem: str) -> EpisodeArtifactBundle:
    p = Path(f"/unused/{stem}.gi.json")
    return EpisodeArtifactBundle(
        metadata_path=p, gi_path=p, kg_path=None, bridge_path=None, episode_id=stem, stem=stem
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id="r",
        parent_run_id=None,
        enricher_id="person_web",
        enricher_version="0.1.0",
        tier="web",
        attempt=1,
        job_id="r",
        cancel_event=asyncio.Event(),
    )


class _FakeProvider:
    name = "fake"

    def __init__(self, found: set[str]) -> None:
        self._found = found

    def fetch(self, person_id: str, display_name: str) -> PersonWebInfo | None:
        if person_id not in self._found:
            return None
        return PersonWebInfo(
            person_id=person_id,
            name=display_name,
            bio=f"{display_name} is a person.",
            image_url="https://img.example/x.jpg",
            source="fake",
            source_url="https://example/x",
            license="CC-BY-SA 4.0",
        )


_GI = {
    "nodes": [
        {"id": "person:jane", "type": "Person", "properties": {"name": "Jane Doe"}},
        {"id": "person:john", "type": "Person", "properties": {"name": "John Roe"}},
    ]
}


def test_manifest_is_web_tier_corpus_scope() -> None:
    m = PersonWebEnricher().manifest
    assert m.tier is EnricherTier.WEB and m.scope is EnricherScope.CORPUS
    assert m.writes == "person_web.json"
    assert m.requires_opt_in is True  # external fetch is opt-in, never in airgapped CI


def test_distinct_persons_dedups_across_bundles(monkeypatch) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    persons = person_web._distinct_persons([_bundle("a"), _bundle("b")])
    assert persons == [("person:jane", "Jane Doe"), ("person:john", "John Roe")]


def test_enrich_writes_rows_for_found_persons(monkeypatch) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    enricher = PersonWebEnricher(provider=_FakeProvider(found={"person:jane"}))
    result = asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=Path("/unused"),
            all_bundles=[_bundle("a")],
            config={},
            ctx=_ctx(),
        )
    )
    assert result.status == "ok"
    assert result.data is not None
    rows = result.data["persons"]
    # Only the person the provider found is written; the miss is dropped (not a null row).
    assert [r["person_id"] for r in rows] == ["person:jane"]
    assert rows[0]["bio"] == "Jane Doe is a person."
    assert result.records_written == 1


def test_enrich_respects_max_persons(monkeypatch) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    enricher = PersonWebEnricher(provider=_FakeProvider(found={"person:jane", "person:john"}))
    result = asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=Path("/unused"),
            all_bundles=[_bundle("a")],
            config={"max_persons": 1},
            ctx=_ctx(),
        )
    )
    assert result.records_written == 1  # capped before fetch


def _opener_returning(payload: dict):
    def _open(_req):
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    return _open


def test_wikipedia_provider_parses_a_summary() -> None:
    p = WikipediaProvider(
        opener=_opener_returning(
            {
                "type": "standard",
                "extract": "Jane Doe is a researcher.",
                "thumbnail": {"source": "https://img/jane.jpg"},
                "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/Jane_Doe"}},
            }
        )
    )
    info = p.fetch("person:jane", "Jane Doe")
    assert info is not None
    assert info.bio == "Jane Doe is a researcher."
    assert info.image_url == "https://img/jane.jpg"
    assert info.source == "wikipedia"
    assert info.source_url is not None and info.source_url.endswith("Jane_Doe")


def test_wikipedia_provider_returns_none_on_disambiguation() -> None:
    p = WikipediaProvider(
        opener=_opener_returning({"type": "disambiguation", "extract": "Could be many people."})
    )
    assert p.fetch("person:x", "X") is None


def test_wikipedia_provider_swallows_network_error() -> None:
    def _boom(_req):
        raise OSError("no network")

    assert WikipediaProvider(opener=_boom).fetch("person:x", "X") is None
