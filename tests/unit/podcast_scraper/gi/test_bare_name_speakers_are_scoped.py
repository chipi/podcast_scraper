"""A bare single-token speaker must not be minted globally by SPOKEN_BY (#2062 follow-on).

FOUND BY A FRESH DGX INGEST, not by reasoning: 2 episodes of "The Journal." transcribed, diarized
and extracted on the DGX on 2026-09-13. One episode's guest is credited as "Twiggy" — a single
token. The episode's gi.json came out with 116 SPOKEN_BY edges for 59 quotes, because the SAME
human was in the graph twice:

    person:unresolved-twiggy-bb004670-ae00-11f1-8d2d-8fac93d999e1   57 edges
    person:twiggy                                                   57 edges

``identity.bare_name_scope`` is the rule for this: a single-token name is not safe to globalise —
every "Twiggy", "Jensen" or "Carly" in the corpus would merge into one person — so it is scoped to
the episode where the name means one person. The GI pipeline applies that rule and migration m0007
backfilled it corpus-wide. ``add_spoken_by_edges`` never did, so the enrichment pass re-introduced
the global id the migration had just removed, and the two layers disagreed about who a person is.

``entity_node_id`` only scopes a bare DIARIZATION LABEL (``SPEAKER_03``); a bare NAME is a different
rule in a different module, which is exactly why this slipped through.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from podcast_scraper.gi.speakers import _person_node_id, add_spoken_by_edges
from podcast_scraper.identity.bare_name_scope import is_bare_person_id, is_scoped_person_id

pytestmark = pytest.mark.unit

EP = "bb004670-ae00-11f1-8d2d-8fac93d999e1"


class TestASingleTokenNameIsScopedToItsEpisode:
    def test_a_bare_name_does_not_become_a_global_person(self) -> None:
        pid = _person_node_id("Twiggy", EP)
        assert not is_bare_person_id(
            pid
        ), f"{pid!r} claims every Twiggy in the corpus is this one person"

    def test_a_bare_name_is_scoped(self) -> None:
        assert is_scoped_person_id(_person_node_id("Twiggy", EP))

    def test_the_same_name_in_two_episodes_does_not_merge(self) -> None:
        a = _person_node_id("Twiggy", EP)
        b = _person_node_id("Twiggy", "some-other-episode")
        assert a != b

    def test_a_full_name_is_still_global(self) -> None:
        # Scoping a full name would fragment a real person across every episode they appear in.
        assert _person_node_id("Lane Florsheim", EP) == "person:lane-florsheim"

    def test_no_episode_id_leaves_the_name_global(self) -> None:
        # Nothing to scope TO. Callers that have the episode id must pass it.
        assert _person_node_id("Twiggy", None) == "person:twiggy"


class TestTheArtifactDoesNotCarryThePersonTwice:
    def test_one_quote_yields_one_spoken_by_edge(self) -> None:
        transcript = "Twiggy: I was sixteen when the photographs started.\n"
        artifact: Dict[str, Any] = {
            "episode_id": EP,
            "nodes": [
                {
                    "id": "quote:1",
                    "type": "Quote",
                    "properties": {
                        "text": "I was sixteen when the photographs started.",
                        "char_start": transcript.index("I was sixteen"),
                    },
                }
            ],
            "edges": [],
        }
        add_spoken_by_edges(artifact, transcript, hosts=[], guests=["Twiggy"])
        sb = [e for e in artifact["edges"] if e.get("type") == "SPOKEN_BY"]
        assert len(sb) == 1, f"one quote produced {len(sb)} SPOKEN_BY edges: {sb}"
        assert not is_bare_person_id(sb[0]["to"])
