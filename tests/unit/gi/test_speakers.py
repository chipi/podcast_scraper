"""Unit tests for GIL speaker attribution → Person / SPOKEN_BY (#874)."""

from __future__ import annotations

import pytest

from podcast_scraper.gi.speakers import (
    add_spoken_by_edges,
    attribute_quote_speakers,
    build_named_turns,
    speaker_for_char,
)

pytestmark = pytest.mark.unit

# A diarized transcript whose markers name NOBODY — only anonymous voices.
_TRANSCRIPT = (
    "Speaker 1: Welcome to the show, today we talk markets. "
    "Speaker 2: Thanks for having me, inflation is the key story. "
    "Speaker 1: So what happens next? "
    "Speaker 2: Rates stay higher for longer, and supply chains stay tight."
)


def test_speaker_for_char_picks_containing_turn():
    turns = [(0, "Speaker 1"), (55, "Speaker 2"), (116, "Speaker 1")]
    assert speaker_for_char(60, turns) == "Speaker 2"
    assert speaker_for_char(0, turns) == "Speaker 1"
    assert speaker_for_char(120, turns) == "Speaker 1"


class TestAnAnonymousTranscriptCreditsNobody:
    """#2075: no heuristic may put a name on a voice (#876).

    `attribute_quote_speakers` used to fall back to a role heuristic over `Speaker N` markers: the
    first voice to speak got `hosts[0]`, the most talkative other voice `guests[0]`. It credited
    81 quotes on 10 Odd Lots episodes in production and `Tracy Alloway` on the #2075 validation run,
    with no voice ever matched to her. With nothing naming a voice, nobody is credited.
    """

    def test_the_guest_is_not_guessed_from_talk_time(self):
        guest_char = _TRANSCRIPT.index("Thanks for having me")
        attribution = attribute_quote_speakers(
            _TRANSCRIPT, {"quote:1": guest_char}, hosts=["Jane Host"], guests=["John Guest"]
        )
        assert attribution == {}

    def test_the_host_is_not_guessed_from_who_spoke_first(self):
        attribution = attribute_quote_speakers(
            _TRANSCRIPT, {"quote:1": 3}, hosts=["Jane Host"], guests=["John Guest"]
        )
        assert attribution == {}

    def test_no_spoken_by_edge_is_written(self):
        artifact = {
            "nodes": [
                {"id": "quote:1", "type": "Quote", "properties": {"char_start": 60}},
                {"id": "insight:1", "type": "Insight", "properties": {}},
            ],
            "edges": [{"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}],
        }
        assert (
            add_spoken_by_edges(artifact, _TRANSCRIPT, hosts=["Jane Host"], guests=["John Guest"])
            == 0
        )
        assert not any(e["type"] == "SPOKEN_BY" for e in artifact["edges"])
        assert not any(n["type"] == "Person" for n in artifact["nodes"])


def test_attribute_skips_when_no_diarization():
    assert attribute_quote_speakers("plain text no labels", {"q": 0}, hosts=[], guests=["G"]) == {}


def test_add_spoken_by_edges_is_idempotent_on_a_named_transcript():
    transcript = "John Guest: Thanks for having me, inflation is the key story."
    artifact = {
        "nodes": [
            {"id": "quote:1", "type": "Quote", "properties": {"char_start": 12}},
            {"id": "insight:1", "type": "Insight", "properties": {}},
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "insight:1", "to": "quote:1"}],
    }
    assert add_spoken_by_edges(artifact, transcript, hosts=[], guests=["John Guest"]) == 1
    assert {"type": "SPOKEN_BY", "from": "quote:1", "to": "person:john-guest"} in artifact["edges"]
    assert add_spoken_by_edges(artifact, transcript, hosts=[], guests=["John Guest"]) == 0


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
    assert [name for _, name in turns if name] == ["Maya", "Liam", "Priya", "Liam", "Maya"]
    # Prose colons and unknown labels name NOBODY. Since #2062 they are still recorded as turn
    # BOUNDARIES (a None name), because an unrecognised line-start marker must END the previous
    # speaker's span rather than let it swallow the line — that is how the host's name ended up on
    # the guest's words. What matters here is that neither ever becomes a speaker.
    stray = build_named_turns("Note: a stray line.\nQ: another.\n", known)
    assert [name for _, name in stray if name] == []
    assert [name for _, name in stray] == [None, None]


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


def test_add_spoken_by_skips_misaligned_char_start(caplog):
    """#876/#925: a Quote whose char_start is in a different (pre-diarization)
    coordinate space than the transcript must NOT be attributed (would target the
    wrong speaker); it's skipped with a warning. Aligned quotes are unaffected."""
    import logging

    from podcast_scraper.identity.slugify import person_id

    transcript = (
        "Maya: " + "Welcome to the show. " * 10 + "\n"
        "Priya Sharma: Reliability is the real challenge here."
    )
    quote_text = "Reliability is the real challenge here."
    aligned = transcript.index(quote_text)

    art_ok = {
        "nodes": [
            {
                "id": "quote:1",
                "type": "Quote",
                "properties": {"char_start": aligned, "text": quote_text},
            }
        ],
        "edges": [],
    }
    assert add_spoken_by_edges(art_ok, transcript, hosts=["Maya"], guests=["Priya Sharma"]) == 1
    assert any(
        e.get("type") == "SPOKEN_BY" and e.get("to") == person_id("Priya Sharma")
        for e in art_ok["edges"]
    )

    art_bad = {
        "nodes": [
            {"id": "quote:1", "type": "Quote", "properties": {"char_start": 3, "text": quote_text}}
        ],
        "edges": [],
    }
    with caplog.at_level(logging.WARNING):
        added = add_spoken_by_edges(art_bad, transcript, hosts=["Maya"], guests=["Priya Sharma"])
    assert added == 0
    assert not [e for e in art_bad["edges"] if e.get("type") == "SPOKEN_BY"]
    assert "not aligned" in caplog.text
