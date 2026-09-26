"""The WEB-tier coverage ceiling: dead entities must not hold the fetch budget forever.

WHAT BROKE ON PROD (measured 2026-09-26, `sha-e2dedbd`, 2003 episodes). An entity whose payload
fetches successfully but derives to no row left NO marker: not a row, and not a miss either,
because ``raw`` was not ``None``. The pre-budget filter excluded only rows and fresh misses, so
all 250 such persons and 229 such orgs were re-selected on EVERY run, took a budget slot,
re-derived from cache and produced nothing. Because the candidate list is ID-sorted and sliced
``[:max_persons]``, the same early-alphabet dead entries held the same slots every time:

    person_web_raw by first letter:  a=334 b=153 c=214 d=232 e=140 f=61 g=132 h=95 i=43 j=243
                                     -> NOTHING from k to z
    org_web_raw:                     digits=82  a=533  b=71
                                     -> NOTHING from c to z

Not one person sorting after "j" had ever been fetched. Three consecutive full passes with the
budget available added 7, then 2 rows.

This is the third disguise of a bug the enricher's own comment says it exists to remove — it was
guarded for the coverage slice, then for misses, and not for derive-failures.

These tests run the REAL enricher over several passes against a fixtured provider, and assert on
what lands on disk. ``test_pre_fix_behaviour_is_a_permanent_ceiling`` is the control: it disables
only the new filter and shows the tail stays unreachable no matter how many passes run.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from podcast_scraper.enrichment.enrichers import org_web, person_web
from podcast_scraper.enrichment.enrichers.person_web import PersonWebEnricher
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext

pytestmark = pytest.mark.integration

#: Two dead entities sorting FIRST, then four derivable ones. With a budget of 2 the dead pair
#: fills the whole budget on pass 1 — which is the prod shape in miniature.
_DEAD = ("person:aaa-dead", "person:bbb-dead")
_LIVE = ("person:ccc-live", "person:ddd-live", "person:eee-live", "person:zzz-live")

_GI = {
    "nodes": [
        {"id": pid, "type": "Person", "properties": {"name": pid.split(":", 1)[1]}}
        for pid in (*_DEAD, *_LIVE)
    ]
}


def _bundle() -> EpisodeArtifactBundle:
    p = Path("/unused/a.gi.json")
    return EpisodeArtifactBundle(
        metadata_path=p, gi_path=p, kg_path=None, bridge_path=None, episode_id="a", stem="a"
    )


def _ctx(enricher_id: str) -> RunContext:
    return RunContext(
        run_id="r",
        parent_run_id=None,
        enricher_id=enricher_id,
        enricher_version="0.1.0",
        tier="web",
        attempt=1,
        job_id="r",
        cancel_event=asyncio.Event(),
    )


class _PersonProvider:
    """Fetches everything; derives only the ``-live`` ids.

    The dead ones return a payload with a ``schema`` (so they are NOT treated as provider-#1
    fossils) that carries no usable prose — exactly the prod "human with no article" shape.
    """

    name = "fake-resolved"

    def __init__(self) -> None:
        self.fetched: list[str] = []

    def fetch_raw(self, person_id, display_name):
        self.fetched.append(person_id)
        if person_id in _DEAD:
            return {"schema": "wikidata+wikipedia/1", "candidates": [], "articled_ids": []}
        return {"schema": "wikidata+wikipedia/1", "wikidata_id": "Q1", "extract": "prose"}

    def derive(self, person_id, display_name, raw):
        if person_id in _DEAD:
            return None
        return person_web.PersonWebInfo(
            person_id=person_id,
            name=display_name,
            bio="prose",
            description=None,
            image_url=None,
            source=self.name,
            source_url=None,
            license="CC-BY-SA 4.0",
        )

    def classify_empty(self, raw):
        return person_web._REASON_NO_ARTICLE


def _derived_ids(tmp_path: Path) -> set[str]:
    art = tmp_path / "enrichments" / "person_web.json"
    if not art.is_file():
        return set()
    doc = json.loads(art.read_text())
    return {r["person_id"] for r in doc["data"]["persons"]}


def _pass(enricher, tmp_path: Path, budget: int) -> None:
    """One full pass, INCLUDING the artifact write the executor normally does.

    Persisting between passes is what makes carry-forward real: ``_existing_person_rows`` reads
    ``enrichments/person_web.json``, so without this every pass would start from an empty
    ``known`` and the multi-pass behaviour under test would not exist.
    """
    result = asyncio.run(
        enricher.enrich(
            bundle=None,
            corpus_root=tmp_path,
            all_bundles=[_bundle()],
            config={"max_persons": budget},
            ctx=_ctx("person_web"),
        )
    )
    art = tmp_path / "enrichments" / "person_web.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({"data": result.data}), encoding="utf-8")


class TestCoverageCeiling:
    def test_the_tail_is_reached_once_dead_entities_explain_themselves(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """THE FIX. Budget 2, two dead entities first, four live ones after.

        Pass 1 spends the budget on the dead pair and records WHY each is empty. Passes 2-3 then
        reach the live tail, including ``zzz`` — the entity that on prod would never be fetched.
        """
        monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
        provider = _PersonProvider()
        enricher = PersonWebEnricher(provider=provider)

        _pass(enricher, tmp_path, budget=2)
        after_first = _derived_ids(tmp_path)

        for _ in range(3):
            _pass(enricher, tmp_path, budget=2)

        assert after_first == set(), "pass 1 is consumed by the dead pair — that part is expected"
        assert _derived_ids(tmp_path) == set(_LIVE), (
            "every live person must be reachable across passes; "
            f"got {sorted(_derived_ids(tmp_path))}"
        )
        assert "person:zzz-live" in _derived_ids(tmp_path), "the sort-order tail must be reached"

    def test_pre_fix_behaviour_is_a_permanent_ceiling(self, monkeypatch, tmp_path: Path) -> None:
        """THE CONTROL — this is what prod did. Disable ONLY the new filter; change nothing else.

        The dead pair keeps its slots on every pass, so no live person is ever derived. Ten passes
        make no more progress than one, which is why running more reprocess passes could not have
        fixed prod.
        """
        monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
        monkeypatch.setattr(person_web, "_empty_reason_is_fresh", lambda *_a, **_k: False)
        provider = _PersonProvider()
        enricher = PersonWebEnricher(provider=provider)

        for _ in range(10):
            _pass(enricher, tmp_path, budget=2)

        assert _derived_ids(tmp_path) == set(), (
            "pre-fix, the dead pair holds the budget forever and the tail is unreachable; "
            f"got {sorted(_derived_ids(tmp_path))}"
        )

    def test_a_reason_is_recorded_without_destroying_the_payload(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """The reason must ANNOTATE the envelope, not replace it.

        The provider contract is that every candidate stays persisted so a future re-derive can
        revisit the choice without re-fetching. ``_write_miss`` replaces the envelope; the reason
        writer must not.
        """
        monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
        enricher = PersonWebEnricher(provider=_PersonProvider())

        _pass(enricher, tmp_path, budget=2)

        doc = json.loads(person_web._raw_path(tmp_path, "person:aaa-dead").read_text())
        assert doc["empty_reason"] == person_web._REASON_NO_ARTICLE
        assert isinstance(doc["empty_reason_at"], int)
        assert doc["payload"]["schema"] == "wikidata+wikipedia/1", "payload must survive"

    def test_an_explained_entity_is_not_refetched(self, monkeypatch, tmp_path: Path) -> None:
        """Budget relief is the point, but not at the cost of re-asking upstream every run."""
        monkeypatch.setattr(person_web, "load_gi", lambda _b: _GI)
        provider = _PersonProvider()
        enricher = PersonWebEnricher(provider=provider)

        _pass(enricher, tmp_path, budget=2)
        for _ in range(3):
            _pass(enricher, tmp_path, budget=2)

        assert (
            provider.fetched.count("person:aaa-dead") == 1
        ), f"a dead entity must be fetched once, not once per pass; got {provider.fetched}"


class TestLegacyFossilVersusOurOwnFallback:
    """A schema-less payload is only a fossil if WE did not just write it.

    ``WikidataWikipediaProvider.fetch_raw`` falls back to a direct-title lookup when no candidate
    has an article, and that fallback payload is a bare REST summary — shape-identical to a
    provider-#1 fossil. Treating it as a fossil would drop and re-fetch it on EVERY run, since
    the resolver would just take the same fallback again: a budget hog traded for a fetch hog.
    """

    def test_a_fossil_is_dropped_so_the_resolver_can_retry_it(self) -> None:
        provider = person_web.WikidataResolvedProvider()
        fossil = {"type": "disambiguation", "title": "Aaron Brown"}

        assert provider.classify_empty(fossil) == person_web._LEGACY_PAYLOAD_REASON

    def test_our_own_fallback_is_explained_not_dropped(self) -> None:
        provider = person_web.WikidataResolvedProvider()
        fallback = provider._stamp_fallback({"type": "disambiguation", "title": "Aaron Brown"})

        assert fallback is not None
        assert (
            provider.classify_empty(fallback) == person_web._REASON_DISAMBIGUATION
        ), "our own fallback must get a reason + TTL, never an invalidate-refetch loop"

    def test_stamping_keeps_the_payload_derivable_by_the_wikipedia_path(self) -> None:
        """The stamp must not break derive: it is still != _RESOLVED_SCHEMA, so it delegates."""
        provider = person_web.WikidataResolvedProvider()
        stamped = provider._stamp_fallback({"type": "standard", "extract": "Real prose."})

        assert stamped is not None
        info = provider.derive("person:x", "X", stamped)
        assert info is not None and info.bio == "Real prose."


class TestClassifyEmptyAgreesWithDerive:
    """A classifier that disagrees with ``derive`` records a false reason and, worse, suppresses
    an entity that would actually derive. Pin the agreement for every branch."""

    @pytest.mark.parametrize(
        "raw",
        [
            {"type": "disambiguation", "title": "X"},
            {"type": "standard"},
            {"type": "standard", "extract": "   "},
            {"schema": "wikidata+wikipedia/1", "articled_ids": []},
            {"schema": "wikidata+wikipedia/1", "articled_ids": ["Q1", "Q2"]},
            {"schema": "wikidata+wikipedia/1", "wikidata_id": "Q1", "wikipedia": None},
            {
                "schema": "wikidata+wikipedia/1",
                "wikidata_id": "Q1",
                "wikipedia": {"type": "disambiguation"},
            },
        ],
    )
    def test_person_classifier_only_speaks_when_derive_is_empty(self, raw) -> None:
        provider = person_web.WikidataResolvedProvider()

        assert provider.derive("person:x", "X", raw) is None, "fixture must be a derive-to-None"
        assert provider.classify_empty(raw), "every empty payload must get a non-empty reason"

    def test_org_classifier_names_not_an_org_for_a_law(self) -> None:
        """The prod case: `14th Amendment` resolves to a constitutional amendment, not an org."""
        provider = org_web.WikidataProvider()
        raw = {
            "qid": "Q188116",
            "candidate_qids": ["Q188116"],
            "entity": {
                "entities": {
                    "Q188116": {
                        "claims": {
                            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q1643989"}}}}]
                        },
                        "descriptions": {"en": {"value": "1868 amendment"}},
                    }
                }
            },
        }

        assert provider.derive("org:14th-amendment", "14th Amendment", raw) is None
        assert provider.classify_empty(raw) == org_web._REASON_NOT_AN_ORG

    def test_org_classifier_names_no_entity_for_an_empty_payload(self) -> None:
        provider = org_web.WikidataProvider()
        raw = {"qid": "Q1", "entity": {"entities": {}}}

        assert provider.derive("org:x", "X", raw) is None
        assert provider.classify_empty(raw) == org_web._REASON_NO_ENTITY
