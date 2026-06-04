"""Unit tests for GIL speaker attribution → Person / SPOKEN_BY (#874)."""

from __future__ import annotations

import pytest

from podcast_scraper.gi.speakers import (
    add_spoken_by_edges,
    attribute_quote_speakers,
    build_speaker_turns,
    map_clusters_to_people,
    speaker_for_char,
)

pytestmark = pytest.mark.unit

# Opening speaker (host) intros; the dominant other speaker is the guest.
_TRANSCRIPT = (
    "Speaker 1: Welcome to the show, today we talk markets. "  # host, opens
    "Speaker 2: Thanks for having me, inflation is the key story. "  # guest
    "Speaker 1: So what happens next? "  # host
    "Speaker 2: Rates stay higher for longer, and supply chains stay tight."  # guest (dominant)
)


def test_build_speaker_turns_parses_markers():
    turns = build_speaker_turns(_TRANSCRIPT)
    assert [label for _, label in turns] == ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2"]
    assert turns == sorted(turns)  # offsets ascending


def test_speaker_for_char_picks_containing_turn():
    turns = build_speaker_turns(_TRANSCRIPT)
    # a char inside the first guest turn resolves to Speaker 2
    guest_turn_off = turns[1][0]
    assert speaker_for_char(guest_turn_off + 5, turns) == "Speaker 2"
    assert speaker_for_char(0, turns) == "Speaker 1"


def test_role_heuristic_maps_guest_and_person_host():
    turns = build_speaker_turns(_TRANSCRIPT)
    cmap = map_clusters_to_people(turns, hosts=["Jane Host"], guests=["John Guest"])
    assert cmap["Speaker 2"] == "John Guest"  # dominant non-opening → guest
    assert cmap["Speaker 1"] == "Jane Host"  # opening, person-like → host


def test_publisher_host_label_is_not_attributed():
    # "Bloomberg" is a publisher, not a person → host stays None (guest still maps).
    turns = build_speaker_turns(_TRANSCRIPT)
    cmap = map_clusters_to_people(turns, hosts=["Bloomberg"], guests=["John Guest"])
    assert cmap["Speaker 1"] is None
    assert cmap["Speaker 2"] == "John Guest"


def test_attribute_quote_speakers_returns_canonical_person_ids():
    turns = build_speaker_turns(_TRANSCRIPT)
    guest_char = turns[1][0] + 3
    attribution = attribute_quote_speakers(
        _TRANSCRIPT, {"quote:1": guest_char}, hosts=["Bloomberg"], guests=["John Guest"]
    )
    assert attribution == {"quote:1": "person:john-guest"}


def test_attribute_skips_when_no_diarization():
    assert attribute_quote_speakers("plain text no labels", {"q": 0}, hosts=[], guests=["G"]) == {}


def test_add_spoken_by_edges_emits_person_and_edge_idempotently():
    turns = build_speaker_turns(_TRANSCRIPT)
    guest_char = turns[1][0] + 3
    artifact = {
        "nodes": [
            {"id": "quote:1", "type": "Quote", "properties": {"char_start": guest_char}},
            {"id": "insight:1", "type": "Insight", "properties": {}},
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}],
    }
    added = add_spoken_by_edges(artifact, _TRANSCRIPT, hosts=["Bloomberg"], guests=["John Guest"])
    assert added == 1
    assert {"type": "SPOKEN_BY", "from": "quote:1", "to": "person:john-guest"} in artifact["edges"]
    assert any(n["id"] == "person:john-guest" and n["type"] == "Person" for n in artifact["nodes"])
    # idempotent: a second pass adds nothing
    assert (
        add_spoken_by_edges(artifact, _TRANSCRIPT, hosts=["Bloomberg"], guests=["John Guest"]) == 0
    )
