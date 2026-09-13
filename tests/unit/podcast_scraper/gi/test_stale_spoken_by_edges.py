"""Re-running enrichment must be able to REPLACE wrong SPOKEN_BY edges, not just add (#2062).

DEMONSTRATED ON A FRESH DGX INGEST (2026-09-13). Two episodes of "The Journal." were ingested with
the fixed attribution code, then ``enrich-edges`` was re-run over them:

    enrich-edges: episodes=2 HAS_EPISODE=0 MENTIONS=0 SPOKEN_BY=0

Nothing changed. The 57 edges pointing at the wrong ``person:twiggy`` were still there afterwards,
because :func:`add_spoken_by_edges` is idempotent in the ADDITIVE sense only: it skips an edge that
already exists and never removes one that should not. Every fix to attribution is therefore inert on
already-processed episodes — the corpus keeps the answer it was given the first time.

That is the difference between a fix and a remediation. ``replace=True`` makes the pass
authoritative: it drops the SPOKEN_BY edges it owns, drops the Person nodes that existed only to
receive them, and recomputes from the transcript. It stays OPT-IN because silently rewriting
historical artifacts is its own failure mode — the caller has to ask.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.gi.speakers import add_spoken_by_edges

pytestmark = pytest.mark.unit

HOST = "Kevin Roose"
TRANSCRIPT = f"{HOST}: The deal still stands as written.\n"
QUOTE = "The deal still stands as written."


def _artifact_with_a_wrong_edge() -> Dict[str, Any]:
    """A gi.json as prod has it: a quote already attributed to the WRONG person."""
    return {
        "episode_id": "ep-1",
        "nodes": [
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {"text": QUOTE, "char_start": TRANSCRIPT.index(QUOTE)},
            },
            {"id": "person:someone-else", "type": "Person", "properties": {"name": "Someone Else"}},
        ],
        "edges": [{"type": "SPOKEN_BY", "from": "quote:1", "to": "person:someone-else"}],
    }


def _spoken(a: Dict[str, Any]) -> set:
    return {(e["from"], e["to"]) for e in a["edges"] if e.get("type") == "SPOKEN_BY"}


def _person_ids(a: Dict[str, Any]) -> set:
    return {n["id"] for n in a["nodes"] if n.get("type") == "Person"}


class TestTheDefaultStaysAdditive:
    """Existing callers must not start deleting data because this capability exists."""

    def test_a_wrong_edge_survives_by_default(self) -> None:
        a = _artifact_with_a_wrong_edge()
        add_spoken_by_edges(a, TRANSCRIPT, hosts=[HOST], guests=[])
        assert ("quote:1", "person:someone-else") in _spoken(a)


class TestReplaceRebuildsAttribution:
    def test_the_wrong_edge_is_gone(self) -> None:
        a = _artifact_with_a_wrong_edge()
        add_spoken_by_edges(a, TRANSCRIPT, hosts=[HOST], guests=[], replace=True)
        assert ("quote:1", "person:someone-else") not in _spoken(a)

    def test_the_right_edge_is_there(self) -> None:
        a = _artifact_with_a_wrong_edge()
        add_spoken_by_edges(a, TRANSCRIPT, hosts=[HOST], guests=[], replace=True)
        assert ("quote:1", "person:kevin-roose") in _spoken(a)

    def test_the_orphaned_person_is_removed(self) -> None:
        # A Person node that existed only to receive the wrong edge would otherwise linger and
        # still render on the episode's people list.
        a = _artifact_with_a_wrong_edge()
        add_spoken_by_edges(a, TRANSCRIPT, hosts=[HOST], guests=[], replace=True)
        assert "person:someone-else" not in _person_ids(a)

    def test_a_person_still_referenced_by_another_edge_is_kept(self) -> None:
        # Only SPOKEN_BY is this function's to remove. A person the KG linked for another reason
        # must survive, or the pass would delete data it does not own.
        a = _artifact_with_a_wrong_edge()
        a["edges"].append(
            {"type": "MENTIONS", "from": "insight:1", "to": "person:someone-else"},
        )
        add_spoken_by_edges(a, TRANSCRIPT, hosts=[HOST], guests=[], replace=True)
        assert "person:someone-else" in _person_ids(a)
        assert ("quote:1", "person:someone-else") not in _spoken(a)

    def test_non_spoken_by_edges_are_untouched(self) -> None:
        a = _artifact_with_a_wrong_edge()
        a["edges"].append({"type": "HAS_EPISODE", "from": "podcast:x", "to": "episode:y"})
        add_spoken_by_edges(a, TRANSCRIPT, hosts=[HOST], guests=[], replace=True)
        assert {"type": "HAS_EPISODE", "from": "podcast:x", "to": "episode:y"} in a["edges"]

    def test_replace_is_idempotent(self) -> None:
        a = _artifact_with_a_wrong_edge()
        add_spoken_by_edges(a, TRANSCRIPT, hosts=[HOST], guests=[], replace=True)
        first = (_spoken(a), _person_ids(a), len(a["edges"]))
        add_spoken_by_edges(a, TRANSCRIPT, hosts=[HOST], guests=[], replace=True)
        assert (_spoken(a), _person_ids(a), len(a["edges"])) == first

    def test_an_episode_that_can_no_longer_be_attributed_ends_with_nothing(self) -> None:
        # The honest outcome when the fix says "we do not know": the wrong name is removed and
        # nothing replaces it. Leaving the old edge would be preferring a lie to a blank.
        a = _artifact_with_a_wrong_edge()
        anon = "SPEAKER_01: The deal still stands as written.\n"
        a["nodes"][0]["properties"]["char_start"] = anon.index(QUOTE)
        add_spoken_by_edges(a, anon, hosts=[], guests=[], replace=True)
        assert _spoken(a) == set()
        assert _person_ids(a) == set()
