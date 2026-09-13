"""A bare single-token speaker must not end up in the artifact twice (#2062).

FOUND BY A FRESH DGX INGEST on 2026-09-13: 2 episodes of "The Journal." produced a gi.json with 116
SPOKEN_BY edges for 59 quotes, because the SAME human was in the graph twice —

    person:unresolved-twiggy-bb004670-...   57 edges
    person:twiggy                           57 edges

WHERE THE RULE LIVES, and where it does NOT. ``identity.bare_name_scope`` is the rule: a
single-token name identifies one person within an episode and nobody globally, so it is scoped to
the episode. It is applied as ONE PASS over the finished payloads — by the pipeline
(``workflow/metadata_generation``) and by the m0007 migration — and deliberately NOT inside any id
mint, because it needs the episode's whole roster and because there are three mint families
(``entity_node_id``, ``person_node_id``, ``identity.slugify.person_id``) that would drift apart if
one of them scoped on its own.

MY FIRST FIX WAS WRONG. I put the scoping inside the GI speaker path's ``_person_node_id``. That
made that one family disagree with the other two, which ``test_entity_identity_invariants`` caught
on ``O'Brien``, ``will.i.am``, ``3Blue1Brown`` and ``Speakman`` — and it pre-empted the pass's
HEALING, which can bind a bare name to a real person's id instead of scoping it.

THE ACTUAL CAUSE is ordering: ``enrich-edges`` runs AFTER the pipeline's scoping pass, so the ids
it minted were never scoped and sat beside the already-scoped ones. That CLI now runs the same
shared pass. These tests therefore assert the ARTIFACT's outcome — the thing a reader sees — not
the return value of one mint function.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.gi.speakers import _person_node_id, add_spoken_by_edges
from podcast_scraper.identity.bare_name_scope import (
    is_bare_person_id,
    is_scoped_person_id,
    plan_bare_name_ids,
    rewrite_ids,
)

pytestmark = pytest.mark.unit

EP = "bb004670-ae00-11f1-8d2d-8fac93d999e1"


class TestTheMintStaysConsistentWithItsSiblings:
    """The invariant I broke: every mint family must agree on an id."""

    def test_the_gi_speaker_path_does_not_scope_on_its_own(self) -> None:
        from podcast_scraper.graph_id_utils import entity_node_id

        for name in ("Twiggy", "O'Brien", "will.i.am", "3Blue1Brown", "Lane Florsheim"):
            assert _person_node_id(name, EP) == entity_node_id("person", name, episode_id=EP)

    def test_a_bare_diarization_label_is_still_scoped_by_the_mint(self) -> None:
        # `SPEAKER_03` is a DIFFERENT rule and does belong in the mint — it needs no roster.
        assert _person_node_id("SPEAKER_03", EP) != "person:speaker-03"


class TestTheSharedPassScopesTheBareName:
    def _artifact(self) -> Dict[str, Any]:
        transcript = "Twiggy: I was sixteen when the photographs started.\n"
        quote = "I was sixteen when the photographs started."
        art: Dict[str, Any] = {
            "episode_id": EP,
            "nodes": [
                {
                    "id": "quote:1",
                    "type": "Quote",
                    "properties": {"text": quote, "char_start": transcript.index(quote)},
                }
            ],
            "edges": [],
        }
        add_spoken_by_edges(art, transcript, hosts=[], guests=["Twiggy"])
        return art

    def _scope(self, art: Dict[str, Any]) -> Dict[str, Any]:
        from podcast_scraper.identity.bare_name_scope import person_ids_in, person_node_ids_in

        id_map = plan_bare_name_ids(
            person_ids_in(art), EP, heal=True, candidate_ids=person_node_ids_in(art)
        )
        out, _changes = rewrite_ids(art, id_map)
        return out

    def test_the_mononym_is_scoped_after_the_pass(self) -> None:
        scoped = self._scope(self._artifact())
        ids = {n["id"] for n in scoped["nodes"] if n.get("type") == "Person"}
        assert any(is_scoped_person_id(i) for i in ids), ids

    def test_no_global_twin_survives(self) -> None:
        # THE DEFECT: one human, two identities, a full set of edges on each.
        scoped = self._scope(self._artifact())
        ids = {n["id"] for n in scoped["nodes"] if n.get("type") == "Person"}
        assert not any(is_bare_person_id(i) for i in ids), ids

    def test_one_quote_keeps_exactly_one_speaker(self) -> None:
        scoped = self._scope(self._artifact())
        sb = [e for e in scoped["edges"] if e.get("type") == "SPOKEN_BY"]
        assert len(sb) == 1, sb

    def test_a_full_name_is_left_global(self) -> None:
        transcript = "Lane Florsheim: The column runs on Mondays.\n"
        quote = "The column runs on Mondays."
        art: Dict[str, Any] = {
            "episode_id": EP,
            "nodes": [
                {
                    "id": "quote:1",
                    "type": "Quote",
                    "properties": {"text": quote, "char_start": transcript.index(quote)},
                }
            ],
            "edges": [],
        }
        add_spoken_by_edges(art, transcript, hosts=["Lane Florsheim"], guests=[])
        scoped = self._scope(art)
        ids = {n["id"] for n in scoped["nodes"] if n.get("type") == "Person"}
        assert "person:lane-florsheim" in ids
