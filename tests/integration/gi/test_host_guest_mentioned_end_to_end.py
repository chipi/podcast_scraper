"""Host, guest and merely-mentioned person, end to end: extracted -> attributed -> persisted (#2062).

WHY THIS TEST EXISTS. The corpus shipped with 89.5% of Person nodes tagged ``mentioned``, 9.9%
``host`` and 0.6% ``guest`` (330-episode feed-stratified production sample, 2026-09-13). Every human
on an episode therefore rendered as a contributor, and the guest — the person most listeners open
the episode FOR — was never shown as the guest. Two independent defects produced that:

  A. the graph was handed the PRE-DIARIZATION host/guest hint instead of the diarization roster, so
     93.2% of roster-named guests never reached ``kg.json``; and
  B. attribution dropped unrecognised line-start markers instead of treating them as turn
     boundaries, so the host's name ran over the guest's words.

Each has its own focused tests. This one is deliberately different: it walks the WHOLE chain on one
episode and asserts the three roles stay distinct at every hop — because both defects were invisible
in unit tests of the individual stages and only showed up in what the reader finally saw.

THE THREE ROLES, and why a test that only checks two is worthless here:

  * **host**   — speaks, runs the show.
  * **guest**  — speaks, is not the host. The role the corpus lost.
  * **mentioned** — NAMED IN THE TRANSCRIPT BUT NEVER SPEAKS. The trap: the easy way to "fix" the
    guest count is to promote every extracted person to a speaker, which would publish a person who
    was merely discussed as though they had been in the room.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from podcast_scraper.gi.speakers import add_spoken_by_edges
from podcast_scraper.kg.io import write_artifact
from podcast_scraper.kg.pipeline import build_artifact
from podcast_scraper.server.app_kg_view import entities_from_kg

pytestmark = pytest.mark.integration

HOST = "Kevin Roose"
GUEST = "Brian Chesky"
MENTIONED = "Elon Musk"  # talked ABOUT; never holds the microphone

TRANSCRIPT = (
    f"{HOST}: Welcome back to the show, today we talk about design and scale.\n"
    f"{GUEST}: Thanks for having me. The design story really starts in 2008.\n"
    f"{HOST}: And what did {MENTIONED} say when you told him that?\n"
    f"{GUEST}: He said gross margin compounds with scale and nobody models it.\n"
)

HOST_QUOTE = "Welcome back to the show, today we talk about design and scale."
GUEST_QUOTE = "He said gross margin compounds with scale and nobody models it."


class _NoLLM:
    """`kg_extraction_source='metadata_only'` — entities arrive prefilled, nothing is generated."""

    kg_extraction_source = "metadata_only"


def _gi_artifact() -> Dict[str, Any]:
    return {
        "episode_id": "ep-design-1",
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


@pytest.fixture()
def kg_on_disk(tmp_path: Path) -> Dict[str, Any]:
    """Build the KG the way the pipeline does, persist it, and read it back off disk."""
    artifact = build_artifact(
        "ep-design-1",
        TRANSCRIPT,
        podcast_id="feed-1",
        episode_title="Design and scale",
        # The roster's answer — what #2062 wired through in place of the pre-diarization hint.
        detected_hosts=[HOST],
        detected_guests=[GUEST],
        cfg=_NoLLM(),
        prefilled_partial={
            "topics": ["design"],
            # Extracted from the transcript, where everyone starts life as "mentioned".
            "entities": [
                {"name": HOST, "kind": "person"},
                {"name": GUEST, "kind": "person"},
                {"name": MENTIONED, "kind": "person"},
            ],
        },
    )
    path = tmp_path / "ep.kg.json"
    write_artifact(path, artifact)
    loaded: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def _people(kg: Dict[str, Any]) -> Dict[str, str]:
    """``{person name: role}`` for every Person node in the artifact."""
    out: Dict[str, str] = {}
    for n in kg.get("nodes") or []:
        if n.get("type") == "Person":
            props = n.get("properties") or {}
            out[str(props.get("name") or n.get("id"))] = str(props.get("role") or "")
    return out


class TestTheThreeRolesSurvivePersistence:
    def test_the_host_is_persisted_as_host(self, kg_on_disk: Dict[str, Any]) -> None:
        assert _people(kg_on_disk).get(HOST) == "host"

    def test_the_guest_is_persisted_as_guest(self, kg_on_disk: Dict[str, Any]) -> None:
        # The headline defect: on prod this was "mentioned" 93.2% of the time.
        assert _people(kg_on_disk).get(GUEST) == "guest"

    def test_the_mentioned_person_is_not_promoted_to_a_speaker(
        self, kg_on_disk: Dict[str, Any]
    ) -> None:
        # The trap. Elon Musk is discussed, never present. Publishing him as host or guest would
        # be a worse bug than the one being fixed.
        assert _people(kg_on_disk).get(MENTIONED) == "mentioned"

    def test_all_three_are_present_and_distinct(self, kg_on_disk: Dict[str, Any]) -> None:
        people = _people(kg_on_disk)
        assert {HOST, GUEST, MENTIONED} <= set(people)
        assert len({people[HOST], people[GUEST], people[MENTIONED]}) == 3


class TestWhatTheClientActuallyReceives:
    """A role that is correct on disk but lost in the view is still wrong on the panel."""

    def test_the_view_reports_the_three_roles(self, kg_on_disk: Dict[str, Any]) -> None:
        persons, _orgs, _topics = entities_from_kg(kg_on_disk)
        by_name = {p.name: p.role for p in persons}
        assert by_name.get(HOST) == "host"
        assert by_name.get(GUEST) == "guest"
        assert by_name.get(MENTIONED) == "mentioned"

    def test_the_guest_is_not_served_as_a_contributor(self, kg_on_disk: Dict[str, Any]) -> None:
        persons, _orgs, _topics = entities_from_kg(kg_on_disk)
        guest = next((p for p in persons if p.name == GUEST), None)
        assert guest is not None, "the guest never reached the client at all"
        assert guest.role == "guest", (
            f"the client was served the guest as role={guest.role!r} — this is the operator-"
            "reported symptom: the guest always shown as a contributor"
        )


class TestAttributionAgreesWithTheRoles:
    """The graph says who they ARE; SPOKEN_BY says who SPOKE. They must not contradict."""

    def test_each_speaker_gets_their_own_quote(self) -> None:
        art = _gi_artifact()
        add_spoken_by_edges(art, TRANSCRIPT, hosts=[HOST], guests=[GUEST])
        spoken = {e["from"]: e["to"] for e in art["edges"] if e.get("type") == "SPOKEN_BY"}
        assert spoken.get("quote:host-1") == "person:kevin-roose"
        assert spoken.get("quote:guest-1") == "person:brian-chesky"

    def test_the_guest_quote_is_not_stamped_with_the_host(self) -> None:
        art = _gi_artifact()
        add_spoken_by_edges(art, TRANSCRIPT, hosts=[HOST], guests=[GUEST])
        spoken = {e["from"]: e["to"] for e in art["edges"] if e.get("type") == "SPOKEN_BY"}
        assert spoken.get("quote:guest-1") != "person:kevin-roose"

    def test_the_mentioned_person_never_speaks(self) -> None:
        art = _gi_artifact()
        add_spoken_by_edges(art, TRANSCRIPT, hosts=[HOST], guests=[GUEST])
        targets = {e["to"] for e in art["edges"] if e.get("type") == "SPOKEN_BY"}
        assert "person:elon-musk" not in targets, (
            "a person who is only TALKED ABOUT was attributed a quote — the name appears inside "
            "the host's turn, and a marker-blind reader will hand them the turn"
        )

    def test_an_undetected_guest_yields_nobody_not_the_host(self) -> None:
        # The mixed-marker shape: 63.1% of prod episodes. The roster named the host, left the
        # guest's voice as SPEAKER_NN, so the transcript carries a marker with no name behind it.
        mixed = TRANSCRIPT.replace(f"{GUEST}:", "SPEAKER_01:")
        art = _gi_artifact()
        for n in art["nodes"]:
            n["properties"]["char_start"] = mixed.index(n["properties"]["text"])
        add_spoken_by_edges(art, mixed, hosts=[HOST], guests=[])
        spoken = {e["from"]: e["to"] for e in art["edges"] if e.get("type") == "SPOKEN_BY"}
        assert spoken.get("quote:host-1") == "person:kevin-roose"
        assert "quote:guest-1" not in spoken
