"""SPOKEN_BY must never name the host on a quote the guest spoke (#2062).

This is the integration face of the unit tests in
``tests/unit/podcast_scraper/gi/test_mixed_marker_attribution.py``: it drives the real enrichment
entry point, ``add_spoken_by_edges``, over a gi.json artifact and asserts on the edges that end up
in the artifact — the same edges ``CorpusGraph._derive_speaker_links`` turns into Person->Insight,
and therefore the same edges behind the speaker name the insights panel renders.

THE OPERATOR-REPORTED SYMPTOM. "I click on insights, and I see a list of insights each one is
supported by a quote. I feel there's always same name listed on all insights." That is this defect:
the host's marker is recognised, the guest's is not, and attribution is sticky — so the host's name
extends over the guest's turns and every insight in the episode is published under one name.

WHY THE GUEST'S MARKER IS UNRECOGNISED. ``providers/ml/diarization/formatting.py`` writes
``f"{label}: "`` from ``roster.label_for()``: the real name when the roster named that voice, the
raw ``SPEAKER_xx`` when it did not. Guests are named far less reliably than hosts (the host is
known from feed metadata), so the mixed shape below is the ordinary prod shape, not a corner case.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from podcast_scraper.gi.speakers import add_spoken_by_edges

pytestmark = pytest.mark.integration

HOST = "Aaron Levie"

#: Roster named the host; left the guest's voice as a numbered label.
TRANSCRIPT = (
    f"{HOST}: Welcome back, today we get into storage economics.\n"
    "SPEAKER_01: The margin structure is what everyone gets wrong about this.\n"
    f"{HOST}: Why is that?\n"
    "SPEAKER_01: Because gross margin compounds with scale and nobody models it.\n"
)

GUEST_QUOTE = "The margin structure is what everyone gets wrong about this."
HOST_QUOTE = "Welcome back, today we get into storage economics."


def _artifact() -> Dict:
    """A gi.json with one host quote and one guest quote, offsets into TRANSCRIPT."""
    return {
        "episode_id": "ep-margin-1",
        "nodes": [
            {
                "id": "quote:host-1",
                "type": "Quote",
                "properties": {"text": HOST_QUOTE, "char_start": TRANSCRIPT.index(HOST_QUOTE)},
            },
            {
                "id": "quote:guest-1",
                "type": "Quote",
                "properties": {"text": GUEST_QUOTE, "char_start": TRANSCRIPT.index(GUEST_QUOTE)},
            },
        ],
        "edges": [],
    }


def _spoken_by(artifact: Dict) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for e in artifact["edges"]:
        if e.get("type") == "SPOKEN_BY":
            out.setdefault(e["from"], []).append(e["to"])
    return out


class TestTheGuestsQuoteDoesNotCarryTheHostsName:
    def test_the_guest_quote_is_not_spoken_by_the_host(self) -> None:
        art = _artifact()
        add_spoken_by_edges(art, TRANSCRIPT, hosts=[HOST], guests=[])
        speakers = _spoken_by(art)
        assert "person:aaron-levie" not in speakers.get("quote:guest-1", []), (
            "the guest's quote was published under the host's name — this is the edge that "
            "renders as the speaker on the insights panel"
        )

    def test_the_guest_quote_has_no_speaker_at_all(self) -> None:
        # Under-attribution is the contract this module states in its own docstring:
        # "a quote with no confident speaker stays None (under-attributed beats wrong)".
        art = _artifact()
        add_spoken_by_edges(art, TRANSCRIPT, hosts=[HOST], guests=[])
        assert _spoken_by(art).get("quote:guest-1") is None

    def test_the_two_quotes_do_not_collapse_onto_one_person(self) -> None:
        # The operator's exact symptom: every insight showing the same name.
        art = _artifact()
        add_spoken_by_edges(art, TRANSCRIPT, hosts=[HOST], guests=[])
        speakers = _spoken_by(art)
        assert speakers.get("quote:host-1") != speakers.get("quote:guest-1")

    def test_the_host_quote_keeps_its_speaker(self) -> None:
        # A "fix" that attributes nothing would pass the tests above and help nobody.
        art = _artifact()
        assert add_spoken_by_edges(art, TRANSCRIPT, hosts=[HOST], guests=[]) >= 1
        assert _spoken_by(art).get("quote:host-1") == ["person:aaron-levie"]

    def test_both_voices_named_still_attributes_both(self) -> None:
        # The control: when the roster names BOTH voices and both are detected, nothing regresses.
        guest = "Theo Jaffee"
        transcript = TRANSCRIPT.replace("SPEAKER_01", guest)
        art = _artifact()
        for n in art["nodes"]:
            n["properties"]["char_start"] = transcript.index(n["properties"]["text"])
        add_spoken_by_edges(art, transcript, hosts=[HOST], guests=[guest])
        speakers = _spoken_by(art)
        assert speakers.get("quote:host-1") == ["person:aaron-levie"]
        assert speakers.get("quote:guest-1") == ["person:theo-jaffee"]
