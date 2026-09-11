"""Unit tests for the org_web enricher (#2035, EnricherTier.WEB) — fixtured, no network."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from podcast_scraper.enrichment.enrichers import org_web
from podcast_scraper.enrichment.enrichers.org_web import (
    OrgWebEnricher,
    OrgWebInfo,
    WikidataProvider,
)
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext

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
        enricher_id="org_web",
        enricher_version="0.1.0",
        tier="web",
        attempt=1,
        job_id="r",
        cancel_event=asyncio.Event(),
    )


_GI = {
    "nodes": [
        {"id": "org:acme", "type": "Organization", "properties": {"name": "Acme Labs"}},
        {"id": "org:globex", "type": "Organization", "properties": {"name": "Globex"}},
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


class _FakeProvider:
    """A raw payload per found org; counts fetches so cache-reuse is observable."""

    name = "fake"

    def __init__(self, found: set[str]) -> None:
        self._found = found
        self.fetch_calls = 0

    def fetch_raw(self, org_id, display_name):
        self.fetch_calls += 1
        return {"desc": f"{display_name} is an org."} if org_id in self._found else None

    def derive(self, org_id, display_name, raw):
        return OrgWebInfo(
            org_id=org_id,
            name=display_name,
            description=raw["desc"],
            summary=None,
            source="fake",
            source_url=None,
            logo_url=None,
            founded="2015",
            industry=None,
            website=None,
        )


class _LogoProvider(_FakeProvider):
    """Derives a logo_url; fetch_image is scripted + counts calls (skip-cache observability)."""

    def __init__(self, found, image_result) -> None:
        super().__init__(found)
        self._image_result = image_result
        self.image_calls = 0

    def derive(self, org_id, display_name, raw):
        info = super().derive(org_id, display_name, raw)
        return OrgWebInfo(**{**info.__dict__, "logo_url": "https://commons.example/x.png"})

    def fetch_image(self, image_url):
        self.image_calls += 1
        return self._image_result


def test_enrich_writes_a_row_per_matched_org(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(org_web, "load_gi", lambda _b: _GI)
    result = _run(OrgWebEnricher(provider=_FakeProvider({"org:acme"})), tmp_path)
    orgs = result.data["orgs"]
    assert result.data["provider"] == "fake"
    assert [r["org_id"] for r in orgs] == ["org:acme"]  # globex not found → no row
    assert orgs[0]["description"] == "Acme Labs is an org." and orgs[0]["founded"] == "2015"


def test_raw_cache_is_reused_across_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(org_web, "load_gi", lambda _b: _GI)
    prov = _FakeProvider({"org:acme", "org:globex"})
    _run(OrgWebEnricher(provider=prov), tmp_path)
    assert prov.fetch_calls == 2
    prov2 = _FakeProvider({"org:acme", "org:globex"})
    _run(OrgWebEnricher(provider=prov2), tmp_path)
    assert prov2.fetch_calls == 0  # both served from org_web_raw/ cache


def test_logo_hosted_when_provider_returns_validated_image(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(org_web, "load_gi", lambda _b: _GI)
    img = org_web.FetchedImage(data=b"\x89PNG\r\n\x1a\n", ext="png", license="CC0", artist=None)
    result = _run(OrgWebEnricher(provider=_LogoProvider({"org:acme"}, img)), tmp_path)
    row = result.data["orgs"][0]
    assert row["logo_hosted"] is True and row["logo_ext"] == "png" and row["logo_license"] == "CC0"
    assert org_web.org_logo_path(tmp_path, "org:acme") is not None


def test_unhostable_logo_caches_a_skip_and_never_refetches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(org_web, "load_gi", lambda _b: _GI)
    prov = _LogoProvider({"org:acme"}, org_web.IMAGE_SKIP)
    _run(OrgWebEnricher(provider=prov), tmp_path)
    assert prov.image_calls == 1
    assert org_web.org_logo_path(tmp_path, "org:acme") is None
    assert org_web._read_logo_meta(tmp_path, "org:acme") == {"skip": True}
    prov2 = _LogoProvider({"org:acme"}, org_web.IMAGE_SKIP)
    _run(OrgWebEnricher(provider=prov2), tmp_path)
    assert prov2.image_calls == 0  # skip sidecar suppressed the re-fetch


# --------------------------------------------------------------------------- #
# WikidataProvider — fixtured over an httpx.MockTransport (no live call)
# --------------------------------------------------------------------------- #

_ENTITY = {
    "entities": {
        "Q1": {
            "descriptions": {"en": {"value": "AI safety research lab"}},
            "claims": {
                "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q4830453"}}}}],
                "P154": [{"mainsnak": {"datavalue": {"value": "Acme logo.png"}}}],
                "P571": [{"mainsnak": {"datavalue": {"value": {"time": "+2015-06-01T00:00:00Z"}}}}],
                "P856": [{"mainsnak": {"datavalue": {"value": "https://acme.example"}}}],
            },
        },
        "Q9": {  # a NON-org entity (P31 = a fruit) — the disambiguation guard must reject it
            "descriptions": {"en": {"value": "a fruit"}},
            "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q999"}}}}]},
        },
    }
}


def _wikidata_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _provider(handler) -> WikidataProvider:
    return WikidataProvider(
        client=_wikidata_client(handler),
        api_base="https://wd.test/w/api.php",
        commons_api_base="https://commons.test/w/api.php",
        commons_filepath_base="https://commons.test/wiki/Special:FilePath/",
    )


def test_wikidata_fetch_raw_searches_then_gets_entity() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "wbsearchentities" in req.url.query.decode():
            return httpx.Response(200, json={"search": [{"id": "Q1"}]})
        if "wbgetentities" in req.url.query.decode():
            return httpx.Response(200, json=_ENTITY)
        return httpx.Response(404)

    raw = _provider(handler).fetch_raw("org:acme", "Acme Labs")
    assert raw is not None and raw["qid"] == "Q1"
    q1 = raw["entity"]["entities"]["Q1"]
    assert q1["descriptions"]["en"]["value"] == "AI safety research lab"


def test_wikidata_derive_extracts_description_logo_founded_site() -> None:
    prov = _provider(lambda r: httpx.Response(404))
    info = prov.derive("org:acme", "Acme Labs", {"qid": "Q1", "entity": _ENTITY})
    assert info is not None
    assert info.description == "AI safety research lab"
    assert info.founded == "2015"
    assert info.website == "https://acme.example"
    assert info.logo_url == "https://commons.test/wiki/Special:FilePath/Acme%20logo.png"
    assert info.source_url == "https://www.wikidata.org/wiki/Q1"


def test_wikidata_derive_rejects_non_organization() -> None:
    """The disambiguation guard: an entity whose P31 isn't org-like resolves to None (#2031)."""
    prov = _provider(lambda r: httpx.Response(404))
    assert prov.derive("org:apple", "Apple", {"qid": "Q9", "entity": _ENTITY}) is None
