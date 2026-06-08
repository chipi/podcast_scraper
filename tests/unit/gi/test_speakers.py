"""Unit tests for GIL speaker attribution → Person / SPOKEN_BY (#874)."""

from __future__ import annotations

import pytest

from podcast_scraper.gi.speakers import (
    add_spoken_by_edges,
    attribute_quote_speakers,
    build_named_turns,
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


# === #875: named diarized markers (panels / multi-guest) ===

# A 3-speaker panel — the named screenplay the new diarization writes. The 2-speaker
# role heuristic cannot tell Liam and Priya apart; named markers attribute each directly.
_PANEL_TRANSCRIPT = (
    "Maya: Welcome to the roundtable on AI policy.\n"
    "Liam: Thanks Maya. Regulation is moving fast in the EU.\n"
    "Priya: I'd push back, enforcement lags the rules badly.\n"
    "Liam: Fair, but the AI Act sets a real baseline.\n"
    "Maya: Let's dig into enforcement then.\n"
)
_PANEL_HOSTS = ["Maya"]
_PANEL_GUESTS = ["Liam", "Priya"]


def test_build_named_turns_matches_only_detected_people():
    known = {"maya": "Maya", "liam": "Liam", "priya": "Priya"}
    turns = build_named_turns(_PANEL_TRANSCRIPT, known)
    assert [name for _, name in turns] == ["Maya", "Liam", "Priya", "Liam", "Maya"]
    # prose colons (none here) and unknown labels are ignored
    assert build_named_turns("Note: a stray line.\nQ: another.\n", known) == []


def test_named_markers_attribute_each_panelist_directly():
    maya_c = _PANEL_TRANSCRIPT.index("Welcome to the roundtable")
    liam_c = _PANEL_TRANSCRIPT.index("Regulation is moving")
    priya_c = _PANEL_TRANSCRIPT.index("enforcement lags")
    attribution = attribute_quote_speakers(
        _PANEL_TRANSCRIPT,
        {"q:maya": maya_c, "q:liam": liam_c, "q:priya": priya_c},
        hosts=_PANEL_HOSTS,
        guests=_PANEL_GUESTS,
    )
    assert attribution == {
        "q:maya": "person:maya",
        "q:liam": "person:liam",
        "q:priya": "person:priya",
    }


def test_named_path_handles_single_token_first_names():
    # "Maya" (1 token) attributes via the named path — the role-heuristic host check
    # (>=2 tokens) would reject it.
    c = _PANEL_TRANSCRIPT.index("Let's dig into enforcement")
    out = attribute_quote_speakers(
        _PANEL_TRANSCRIPT, {"q": c}, hosts=["Maya"], guests=["Liam", "Priya"]
    )
    assert out == {"q": "person:maya"}


def test_named_publisher_label_not_attributed():
    transcript = "Bloomberg: Markets are volatile today.\nJohn Guest: Indeed, rates matter.\n"
    bbg_c = transcript.index("Markets are volatile")
    guest_c = transcript.index("Indeed, rates")
    out = attribute_quote_speakers(
        transcript,
        {"q:bbg": bbg_c, "q:guest": guest_c},
        hosts=["Bloomberg"],
        guests=["John Guest"],
    )
    # Publisher "Bloomberg" is excluded; the person guest still attributes.
    assert out == {"q:guest": "person:john-guest"}


def test_add_spoken_by_edges_panel_emits_all_panelists():
    artifact = {
        "nodes": [
            {
                "id": "quote:m",
                "type": "Quote",
                "properties": {"char_start": _PANEL_TRANSCRIPT.index("Welcome to the roundtable")},
            },
            {
                "id": "quote:l",
                "type": "Quote",
                "properties": {"char_start": _PANEL_TRANSCRIPT.index("Regulation is moving")},
            },
            {
                "id": "quote:p",
                "type": "Quote",
                "properties": {"char_start": _PANEL_TRANSCRIPT.index("enforcement lags")},
            },
        ],
        "edges": [],
    }
    added = add_spoken_by_edges(
        artifact, _PANEL_TRANSCRIPT, hosts=_PANEL_HOSTS, guests=_PANEL_GUESTS
    )
    assert added == 3
    persons = {n["id"] for n in artifact["nodes"] if n["type"] == "Person"}
    assert persons == {"person:maya", "person:liam", "person:priya"}
