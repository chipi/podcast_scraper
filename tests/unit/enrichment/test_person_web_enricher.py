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
    """A raw payload per found person; counts fetches so cache-reuse is observable."""

    name = "fake"

    def __init__(self, found: set[str]) -> None:
        self._found = found
        self.fetch_calls = 0

    def fetch_raw(self, person_id, display_name):
        self.fetch_calls += 1
        if person_id not in self._found:
            return None
        return {"type": "standard", "extract": f"{display_name} is a person."}

    def derive(self, person_id, display_name, raw):
        extract = raw.get("extract")
        if not extract:
            return None
        return person_web.PersonWebInfo(
            person_id=person_id,
            name=display_name,
            bio=extract,
            image_url=None,
            source="fake",
            source_url=None,
            license="CC-BY-SA 4.0",
        )


_GI = {
    "nodes": [
        {"id": "person:jane", "type": "Person", "properties": {"name": "Jane Doe"}},
        {"id": "person:john", "type": "Person", "properties": {"name": "John Roe"}},
    ]
}


def _run(enricher, tmp_path, config=None):
    return asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=tmp_path,
            all_bundles=[_bundle("a")],
            config=config or {},
            ctx=_ctx(),
        )
    )


def test_manifest_is_web_tier_corpus_scope() -> None:
    m = PersonWebEnricher().manifest
    assert m.tier is EnricherTier.WEB and m.scope is EnricherScope.CORPUS
    assert m.writes == "person_web.json"
    assert m.requires_opt_in is False  # ON by default in cloud/prod; airgap held by profile


def test_distinct_persons_dedups_across_bundles(monkeypatch) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    persons = person_web._distinct_persons([_bundle("a"), _bundle("b")])
    assert persons == [("person:jane", "Jane Doe"), ("person:john", "John Roe")]


def test_enrich_writes_rows_and_persists_raw(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    enricher = PersonWebEnricher(provider=_FakeProvider(found={"person:jane"}))
    result = _run(enricher, tmp_path)
    assert result.status == "ok" and result.data is not None
    rows = result.data["persons"]
    assert [r["person_id"] for r in rows] == ["person:jane"]  # the miss is dropped, not nulled
    assert rows[0]["bio"] == "Jane Doe is a person."
    # RAW is persisted per person for later re-derivation / mining.
    raw = person_web._read_raw_cache(tmp_path, "person:jane")
    assert raw is not None and raw["extract"] == "Jane Doe is a person."


def test_cached_raw_is_reused_without_refetch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    first = _FakeProvider(found={"person:jane"})
    _run(PersonWebEnricher(provider=first), tmp_path)
    assert first.fetch_calls == 2  # jane + john both attempted on the first pass

    # A second run with a NEW provider that would fetch again — but cached raw means derive-only.
    second = _FakeProvider(found={"person:jane"})
    result = _run(PersonWebEnricher(provider=second), tmp_path)
    assert second.fetch_calls == 1  # only john (uncached miss) re-attempted; jane from cache
    assert [r["person_id"] for r in result.data["persons"]] == ["person:jane"]


def test_refresh_bypasses_the_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    _run(PersonWebEnricher(provider=_FakeProvider(found={"person:jane"})), tmp_path)
    refetch = _FakeProvider(found={"person:jane"})
    _run(PersonWebEnricher(provider=refetch), tmp_path, config={"refresh": True})
    assert refetch.fetch_calls == 2  # both re-fetched despite the cache


def test_enrich_respects_max_persons(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
    result = _run(
        PersonWebEnricher(provider=_FakeProvider(found={"person:jane", "person:john"})),
        tmp_path,
        config={"max_persons": 1},
    )
    assert result.records_written == 1  # capped before fetch


def _opener_returning(payload: dict):
    def _open(_req):
        return io.BytesIO(json.dumps(payload).encode("utf-8"))

    return _open


def test_wikipedia_provider_fetch_then_derive() -> None:
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
    raw = p.fetch_raw("person:jane", "Jane Doe")
    assert raw is not None and raw["extract"].startswith("Jane Doe")
    info = p.derive("person:jane", "Jane Doe", raw)
    assert info is not None
    assert info.bio == "Jane Doe is a researcher."
    assert info.image_url == "https://img/jane.jpg"
    assert info.source == "wikipedia"
    assert info.source_url is not None and info.source_url.endswith("Jane_Doe")


def test_wikipedia_derive_none_on_disambiguation() -> None:
    p = WikipediaProvider()
    assert p.derive("person:x", "X", {"type": "disambiguation", "extract": "many"}) is None


def test_wikipedia_fetch_raw_swallows_network_error() -> None:
    def _boom(_req):
        raise OSError("no network")

    assert WikipediaProvider(opener=_boom).fetch_raw("person:x", "X") is None


def test_web_wiring_registers_and_returns_ids() -> None:
    from podcast_scraper.enrichment.registry import EnricherRegistry
    from podcast_scraper.enrichment.web_wiring import register_web_enrichers

    reg = EnricherRegistry()
    ids = register_web_enrichers(reg)
    assert ids == ["person_web"]
    assert reg.get("person_web").manifest.id == "person_web"
