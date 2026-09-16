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


# KG, not GI. ``Organization`` is a KG-only node type — GI carries Episode / Insight / Quote /
# Person / Topic / Podcast. This fixture was GI-shaped with Organization nodes in it, a graph that
# never exists in production, so it made the enricher look correct while the real code path read
# an artifact that could never contain an org. Every run in prod would have derived 0/0.
_KG = {
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
    monkeypatch.setattr(org_web, "load_kg", lambda _b: _KG)
    result = _run(OrgWebEnricher(provider=_FakeProvider({"org:acme"})), tmp_path)
    orgs = result.data["orgs"]
    assert result.data["provider"] == "fake"
    assert [r["org_id"] for r in orgs] == ["org:acme"]  # globex not found → no row
    assert orgs[0]["description"] == "Acme Labs is an org." and orgs[0]["founded"] == "2015"


def test_raw_cache_is_reused_across_runs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(org_web, "load_kg", lambda _b: _KG)
    prov = _FakeProvider({"org:acme", "org:globex"})
    _run(OrgWebEnricher(provider=prov), tmp_path)
    assert prov.fetch_calls == 2
    prov2 = _FakeProvider({"org:acme", "org:globex"})
    _run(OrgWebEnricher(provider=prov2), tmp_path)
    assert prov2.fetch_calls == 0  # both served from org_web_raw/ cache


def test_a_miss_is_recorded_and_not_refetched_next_run(monkeypatch, tmp_path: Path) -> None:
    """An org the upstream has nothing for must not be re-asked on every subsequent run.

    Regression: a miss never enters ``known`` (only derived ROWS do), so without a recorded
    miss the new-entity budget re-selected the same unresolvable orgs forever. On prod this
    burned ~200 attempts and 23 minutes per run to add +1 org (#2084).
    """
    monkeypatch.setattr(org_web, "load_kg", lambda _b: _KG)
    prov = _FakeProvider(set())  # neither org resolves
    assert _run(OrgWebEnricher(provider=prov), tmp_path).data["orgs"] == []
    assert prov.fetch_calls == 2  # both attempted once

    prov2 = _FakeProvider(set())
    assert _run(OrgWebEnricher(provider=prov2), tmp_path).data["orgs"] == []
    assert prov2.fetch_calls == 0  # both suppressed by the recorded miss


def test_a_miss_does_not_consume_the_new_entity_budget(monkeypatch, tmp_path: Path) -> None:
    """The miss filter runs BEFORE the cap, so a missed org frees its slot for a new one.

    Filtering after the slice would still waste the budget on entities known to be absent —
    which is exactly how prod stalled at ~4 resolved orgs per 200-slot run.
    """
    monkeypatch.setattr(org_web, "load_kg", lambda _b: _KG)
    # Budget of 1: acme is attempted, misses, and is recorded.
    prov = _FakeProvider({"org:globex"})
    _run(OrgWebEnricher(provider=prov), tmp_path, config={"max_orgs": 1})
    assert prov.fetch_calls == 1

    # Next run, same budget of 1: acme is filtered out, so the slot goes to globex.
    prov2 = _FakeProvider({"org:globex"})
    rows = _run(OrgWebEnricher(provider=prov2), tmp_path, config={"max_orgs": 1}).data["orgs"]
    assert [r["org_id"] for r in rows] == ["org:globex"]
    assert prov2.fetch_calls == 1


def test_refresh_ignores_a_recorded_miss(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(org_web, "load_kg", lambda _b: _KG)
    _run(OrgWebEnricher(provider=_FakeProvider(set())), tmp_path)
    prov = _FakeProvider({"org:acme", "org:globex"})
    rows = _run(OrgWebEnricher(provider=prov), tmp_path, config={"refresh": True}).data["orgs"]
    assert prov.fetch_calls == 2  # refresh re-asks despite the fresh miss
    assert sorted(r["org_id"] for r in rows) == ["org:acme", "org:globex"]


def test_logo_hosted_when_provider_returns_validated_image(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(org_web, "load_kg", lambda _b: _KG)
    img = org_web.FetchedImage(data=b"\x89PNG\r\n\x1a\n", ext="png", license="CC0", artist=None)
    result = _run(OrgWebEnricher(provider=_LogoProvider({"org:acme"}, img)), tmp_path)
    row = result.data["orgs"][0]
    assert row["logo_hosted"] is True and row["logo_ext"] == "png" and row["logo_license"] == "CC0"
    assert org_web.org_logo_path(tmp_path, "org:acme") is not None


def test_unhostable_logo_caches_a_skip_and_never_refetches(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(org_web, "load_kg", lambda _b: _KG)
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


# --- Wikidata candidate ranking (2026-09-14) ---------------------------------------------------
# Both behaviours below were broken in the same way and found by the first real run of this
# enricher, on a 5-organization corpus that derived 0/5:
#
#   * the P31 allowlist held only generic shapes (organization / company / university...), which
#     real entities rarely declare. The NIH says "United States federal agency", Newsweek says
#     "news magazine", Nature says "scientific journal" — all three were refused as not-an-org.
#   * the search took hits[0] and then veto-ed it on P31. "Stanford" ranks the census-designated
#     place above the university, so the correct answer at rank 2 was discarded rather than used.
#
# The fix makes P31 the SELECTOR over ranked candidates instead of a veto on the top hit.


def _wd_payload(candidates: list[tuple[str, list[str]]]) -> dict:
    """A Wikidata raw payload: [(qid, [P31 qids])] in search-rank order."""
    return {
        "qid": candidates[0][0],
        "candidate_qids": [q for q, _ in candidates],
        "entity": {
            "entities": {
                q: {
                    "claims": {
                        "P31": [{"mainsnak": {"datavalue": {"value": {"id": p}}}} for p in p31]
                    },
                    "descriptions": {"en": {"value": f"description of {q}"}},
                }
                for q, p31 in candidates
            }
        },
    }


def test_ranked_candidates_skip_a_non_org_top_hit() -> None:
    """The real 'Stanford' case: town (Q173813) ranks above the university (Q41506)."""
    provider = org_web.WikidataProvider()
    raw = _wd_payload([("Q173813", ["Q498162"]), ("Q41506", ["Q902104"])])
    info = provider.derive("org:stanford", "Stanford", raw)
    assert info is not None, "rank-2 university must be selected when rank-1 is a place"
    # Provenance must cite the entity actually described, not the discarded top hit.
    assert info.source_url == "https://www.wikidata.org/wiki/Q41506"


def test_agency_and_periodical_p31s_are_accepted() -> None:
    """NIH / Newsweek / Nature shapes — each was rejected before the allowlist was widened."""
    provider = org_web.WikidataProvider()
    for qid, p31 in (
        ("Q390551", "Q20857065"),  # NIH — United States federal agency
        ("Q188413", "Q1684600"),  # Newsweek — news magazine
        ("Q180445", "Q5633421"),  # Nature — scientific journal
    ):
        info = provider.derive("org:x", "x", _wd_payload([(qid, [p31])]))
        assert info is not None, f"{qid} with P31={p31} must be accepted as an organization"


def test_non_organisations_are_still_refused() -> None:
    """The guard must keep doing its job: Shingrix is a vaccine brand the KG typed as an org."""
    provider = org_web.WikidataProvider()
    raw = _wd_payload([("Q42610038", ["Q431289", "Q105967696"])])  # brand + vaccine type
    assert provider.derive("org:shingrix", "Shingrix", raw) is None
