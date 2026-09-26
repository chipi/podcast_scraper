"""The reason envelope that lifts the WEB-tier coverage ceiling (#2158).

`person_web` and `org_web` record WHY a fetched payload derived to no row, and the pre-budget filter
skips entities that already carry a fresh reason. Two properties make that safe rather than merely
convenient, and both live in these helpers:

1. **The payload must survive.** The provider contract is that every candidate stays persisted so a
   future re-derive can revisit the choice without re-fetching. `_write_miss` REPLACES the envelope;
   the reason writer must only annotate it.
2. **Every failure must fall back to "no reason recorded".** A reason is what suppresses an entity
   from the budget, so a corrupt or unreadable envelope has to read as absent — otherwise a bad file
   silently hides an entity from enrichment forever, which is the ceiling bug again in a new form.

The two modules deliberately duplicate these helpers (mirroring the existing `_write_miss` /
`_miss_is_fresh` split), so both are exercised here with the same battery.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from podcast_scraper.enrichment.enrichers import org_web, person_web

pytestmark = pytest.mark.unit


def _write_envelope(mod, root: Path, entity_id: str, payload: dict, **extra) -> Path:
    # ``mod`` is an untyped module parameter, so ``_raw_path`` is Any — annotate the local rather
    # than returning Any from a function declared to return Path.
    p: Path = mod._raw_path(root, entity_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {"payload": payload, "fetched_at": int(time.time())}
    doc.update(extra)
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


@pytest.mark.parametrize("mod", [person_web, org_web], ids=["person_web", "org_web"])
class TestReasonEnvelopeFailsToAbsent:
    """Anything unreadable must read as "no reason", never as "suppress this entity"."""

    def test_no_file_is_not_fresh(self, mod, tmp_path: Path) -> None:
        assert mod._empty_reason_is_fresh(tmp_path, "id:missing", int(time.time())) is False

    def test_unparsable_json_is_not_fresh(self, mod, tmp_path: Path) -> None:
        p = mod._raw_path(tmp_path, "id:broken")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json at all", encoding="utf-8")

        assert mod._empty_reason_is_fresh(tmp_path, "id:broken", int(time.time())) is False

    def test_an_envelope_with_no_reason_is_not_fresh(self, mod, tmp_path: Path) -> None:
        _write_envelope(mod, tmp_path, "id:plain", {"schema": "x"})

        assert mod._empty_reason_is_fresh(tmp_path, "id:plain", int(time.time())) is False

    def test_a_reason_without_a_usable_timestamp_is_not_fresh(self, mod, tmp_path: Path) -> None:
        """No timestamp means no TTL, and an un-expirable suppression is the bug we are fixing."""
        _write_envelope(
            mod,
            tmp_path,
            "id:nots",
            {"schema": "x"},
            empty_reason="whatever",
            empty_reason_at="soon",
        )

        assert mod._empty_reason_is_fresh(tmp_path, "id:nots", int(time.time())) is False

    def test_a_fresh_reason_is_fresh_and_an_expired_one_is_not(self, mod, tmp_path: Path) -> None:
        now = int(time.time())
        _write_envelope(
            mod, tmp_path, "id:ttl", {"schema": "x"}, empty_reason="r", empty_reason_at=now
        )

        assert mod._empty_reason_is_fresh(tmp_path, "id:ttl", now) is True
        assert mod._empty_reason_is_fresh(tmp_path, "id:ttl", now + mod._MISS_TTL_S + 1) is False


@pytest.mark.parametrize("mod", [person_web, org_web], ids=["person_web", "org_web"])
class TestReasonWriterPreservesThePayload:
    def test_it_annotates_without_dropping_the_payload(self, mod, tmp_path: Path) -> None:
        _write_envelope(mod, tmp_path, "id:keep", {"schema": "x", "candidates": [1, 2, 3]})
        now = int(time.time())

        mod._write_empty_reason(tmp_path, "id:keep", "no_article", now)

        doc = json.loads(mod._raw_path(tmp_path, "id:keep").read_text())
        assert doc["empty_reason"] == "no_article"
        assert doc["empty_reason_at"] == now
        assert doc["payload"]["candidates"] == [1, 2, 3], "the candidates must survive a re-derive"

    def test_writing_to_a_missing_file_is_a_no_op(self, mod, tmp_path: Path) -> None:
        mod._write_empty_reason(tmp_path, "id:absent", "no_article", int(time.time()))

        assert not mod._raw_path(tmp_path, "id:absent").exists(), "must not invent an envelope"

    def test_an_unparsable_envelope_is_left_alone(self, mod, tmp_path: Path) -> None:
        p = mod._raw_path(tmp_path, "id:broken")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json", encoding="utf-8")

        mod._write_empty_reason(tmp_path, "id:broken", "no_article", int(time.time()))

        assert p.read_text() == "{not json", "a corrupt file must not be half-rewritten"

    def test_a_non_dict_envelope_is_left_alone(self, mod, tmp_path: Path) -> None:
        p = mod._raw_path(tmp_path, "id:list")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[1, 2, 3]", encoding="utf-8")

        mod._write_empty_reason(tmp_path, "id:list", "no_article", int(time.time()))

        assert p.read_text() == "[1, 2, 3]"

    def test_a_failing_write_is_swallowed(self, mod, tmp_path: Path, monkeypatch) -> None:
        """Best-effort: a failed reason write costs a re-derive, it must not fail the run."""
        _write_envelope(mod, tmp_path, "id:ro", {"schema": "x"})

        def boom(*_a, **_k):
            raise OSError("read-only filesystem")

        monkeypatch.setattr(Path, "write_text", boom)

        mod._write_empty_reason(tmp_path, "id:ro", "no_article", int(time.time()))  # must not raise


class TestPersonWebInvalidation:
    def test_a_fossil_payload_is_dropped_so_the_resolver_retries(self, tmp_path: Path) -> None:
        p = _write_envelope(person_web, tmp_path, "person:x", {"type": "disambiguation"})
        assert p.is_file()

        person_web._invalidate_raw_cache(tmp_path, "person:x")

        assert not p.exists()

    def test_invalidating_something_absent_is_harmless(self, tmp_path: Path) -> None:
        person_web._invalidate_raw_cache(tmp_path, "person:never-existed")  # must not raise

    def test_a_failing_unlink_is_swallowed(self, tmp_path: Path, monkeypatch) -> None:
        _write_envelope(person_web, tmp_path, "person:y", {"type": "disambiguation"})

        def boom(*_a, **_k):
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "unlink", boom)

        person_web._invalidate_raw_cache(tmp_path, "person:y")  # must not raise


class TestClassifiersNameTheRemainingBranches:
    def test_wikipedia_no_extract(self) -> None:
        """An article that exists but carries no prose is `no_extract`, not a disambiguation."""
        p = person_web.WikipediaProvider()

        assert p.classify_empty({"type": "standard"}) == person_web._REASON_NO_EXTRACT

    def test_org_resolved_but_nothing_surfaceable(self) -> None:
        """Org-like with neither a description nor a logo — resolved, but nothing worth showing."""
        provider = org_web.WikidataProvider()
        org_qid = sorted(provider._ORG_INSTANCE_QIDS)[0]
        raw = {
            "qid": "Q1",
            "candidate_qids": ["Q1"],
            "entity": {
                "entities": {
                    "Q1": {
                        "claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": org_qid}}}}]}
                    }
                }
            },
        }

        assert provider.derive("org:x", "X", raw) is None
        assert provider.classify_empty(raw) == org_web._REASON_NOTHING_SURFACEABLE

    def test_org_candidate_that_is_not_a_dict_is_skipped(self) -> None:
        """A malformed candidate entry must not crash the selector, just fail to match."""
        provider = org_web.WikidataProvider()
        raw = {"qid": "Q1", "candidate_qids": ["Q1"], "entity": {"entities": {"Q1": "not-a-dict"}}}

        assert provider.classify_empty(raw) == org_web._REASON_NOT_AN_ORG


class TestOrgWebRecordsItsReason:
    """The org enricher's loop must write the reason, mirroring person_web's."""

    def test_a_derive_failure_writes_a_reason_into_the_envelope(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext

        gi = {
            "nodes": [
                {"id": "org:acme", "type": "Organization", "properties": {"name": "Acme"}},
            ]
        }
        # KG, not GI: ``Organization`` is a KG-only node type (org_web's own docstring records
        # that reading GI here made the enricher structurally incapable of finding anything).
        monkeypatch.setattr(org_web, "load_kg", lambda _b: gi)

        class _Provider:
            name = "fake"

            def fetch_raw(self, org_id, display_name):
                return {"qid": "Q1", "entity": {"entities": {}}}

            def derive(self, org_id, display_name, raw):
                return None

            def classify_empty(self, raw):
                return org_web._REASON_NO_ENTITY

        p = Path("/unused/a.gi.json")
        bundle = EpisodeArtifactBundle(
            metadata_path=p, gi_path=p, kg_path=p, bridge_path=None, episode_id="a", stem="a"
        )
        ctx = RunContext(
            run_id="r",
            parent_run_id=None,
            enricher_id="org_web",
            enricher_version="0.1.0",
            tier="web",
            attempt=1,
            job_id="r",
            cancel_event=asyncio.Event(),
        )

        asyncio.run(
            org_web.OrgWebEnricher(provider=_Provider()).enrich(
                bundle=None,
                corpus_root=tmp_path,
                all_bundles=[bundle],
                config={"max_orgs": 5},
                ctx=ctx,
            )
        )

        env = org_web._raw_path(tmp_path, "org:acme")
        assert env.is_file(), "the payload should have been cached"
        doc = json.loads(env.read_text())
        assert doc.get("empty_reason") == org_web._REASON_NO_ENTITY
        assert isinstance(doc.get("empty_reason_at"), int)


class TestRemainingDefensiveBranches:
    def test_org_derive_rejects_a_non_dict_entities_block(self) -> None:
        """``entities`` arriving as a LIST is malformed upstream, not an org with no match.

        Reachable because ``(raw.get("entity") or {}).get("entities") or {}`` passes a truthy list
        straight through.
        """
        provider = org_web.WikidataProvider()

        assert provider.derive("org:x", "X", {"entity": {"entities": ["a", "b"]}}) is None

    def test_a_legacy_fossil_payload_is_invalidated_by_the_enricher_loop(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The loop must DROP a fossil rather than record a reason for it.

        A payload the current provider cannot derive at all should be re-fetched through the
        resolver, so the envelope is removed instead of annotated. Recording a reason would pin the
        fossil in place for the whole TTL.
        """
        from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext

        gi = {
            "nodes": [
                {"id": "person:fossil", "type": "Person", "properties": {"name": "Aaron Brown"}}
            ]
        }
        monkeypatch.setattr(person_web, "load_gi", lambda _b: gi)

        # Pre-seed the cache with a provider-#1 shaped payload (no ``schema`` key).
        env = _write_envelope(person_web, tmp_path, "person:fossil", {"type": "disambiguation"})
        assert env.is_file()

        class _Provider:
            name = "fake-resolved"

            def fetch_raw(self, person_id, display_name):  # pragma: no cover - cache hit expected
                raise AssertionError("must read the cached payload, not re-fetch")

            def derive(self, person_id, display_name, raw):
                return None

            def classify_empty(self, raw):
                return person_web._LEGACY_PAYLOAD_REASON

        p = Path("/unused/a.gi.json")
        bundle = EpisodeArtifactBundle(
            metadata_path=p, gi_path=p, kg_path=None, bridge_path=None, episode_id="a", stem="a"
        )
        ctx = RunContext(
            run_id="r",
            parent_run_id=None,
            enricher_id="person_web",
            enricher_version="0.1.0",
            tier="web",
            attempt=1,
            job_id="r",
            cancel_event=asyncio.Event(),
        )

        asyncio.run(
            person_web.PersonWebEnricher(provider=_Provider()).enrich(
                bundle=None,
                corpus_root=tmp_path,
                all_bundles=[bundle],
                config={"max_persons": 5},
                ctx=ctx,
            )
        )

        assert not env.exists(), "the fossil must be dropped so the resolver re-fetches it"
