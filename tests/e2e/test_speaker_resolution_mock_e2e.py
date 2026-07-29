"""E2E: the mock LLM server answers the ADR-110/135 speaker-resolution call in the real
`{"voices": {...}}` shape, and the pipeline's own parser round-trips it — happy paths (a voice
named from its self-introduction) and unhappy paths (a voice that declines, and an invented name the
closed-list guard rejects).

Locks the mock ↔ resolver contract so the LLM naming layer stays exercised end-to-end (it was
silently inert before — the mock only answered the older detect_speakers shape).
"""

from __future__ import annotations

import json

from podcast_scraper.speaker_detectors.resolution import (
    build_resolution_prompt,
    resolve_voices_and_roles,
)
from tests.e2e.fixtures.e2e_http_server import _resolution_response_json

# Samples must clear MIN_SAMPLE_CHARS (80) or build_resolution_prompt drops the voice.
_HOST = (
    "Hello and welcome to the show. I'm Ana Rodriguez, your host today, and we have a wonderful "
    "conversation lined up about the industry and where it is all heading next this year."
)
_GUEST = (
    "Thanks for having me. I'm Richard Gelfond and I have run IMAX for decades, and I will walk "
    "you through how the large-format cinema business actually came together over all these years."
)
_TAPE = (
    "just a random person on the street sharing a quick unrelated thought here today about the "
    "weather and the traffic and nothing at all to do with the show or its guests really at all."
)
_STATED = ["Ana Rodriguez", "Richard Gelfond"]


def _prompt(voice_texts):
    return build_resolution_prompt(
        stated_names=_STATED,
        voice_texts=voice_texts,
        known_hosts=["Ana Rodriguez"],
        ordered_turns=list(voice_texts.items()),
        episode_title="Test Episode",
    )


def test_mock_only_answers_resolution_prompts():
    assert _resolution_response_json("Summarize this episode in one paragraph.") is None
    assert _resolution_response_json(_prompt({"SPEAKER_00": _HOST})) is not None


def test_mock_tells_the_story_happy_and_unhappy():
    voice_texts = {"SPEAKER_00": _HOST, "SPEAKER_01": _GUEST, "SPEAKER_02": _TAPE}
    answer = json.loads(_resolution_response_json(_prompt(voice_texts)))["voices"]
    # HAPPY: self-introductions bind to the stated names, with roles from the speech acts.
    assert answer["SPEAKER_00"] == {"name": "Ana Rodriguez", "role": "host"}
    assert answer["SPEAKER_01"] == {"name": "Richard Gelfond", "role": "guest"}
    # UNHAPPY: a vox-pop with no name evidence declines rather than guessing.
    assert answer["SPEAKER_02"] == {"name": None, "role": None}


def test_pipeline_parser_round_trips_the_mock_answer():
    voice_texts = {"SPEAKER_00": _HOST, "SPEAKER_01": _GUEST, "SPEAKER_02": _TAPE}
    resolved = resolve_voices_and_roles(
        _STATED,
        voice_texts,
        complete=lambda p: _resolution_response_json(p) or "{}",
        known_hosts=["Ana Rodriguez"],
        ordered_turns=list(voice_texts.items()),
    )
    assert resolved["SPEAKER_00"].name == "Ana Rodriguez"
    assert resolved["SPEAKER_00"].role == "host"
    assert resolved["SPEAKER_01"].name == "Richard Gelfond"
    assert resolved["SPEAKER_01"].role == "guest"
    # the declined vox-pop is not named (it either drops out or carries no name).
    assert resolved.get("SPEAKER_02") is None or resolved["SPEAKER_02"].name is None


def test_mock_never_invents_a_name_off_the_stated_list():
    # A voice self-introduces as someone the metadata never stated -> the mock declines (the closed
    # list is the contract; inventing a name is the #876 failure the resolver guards against).
    off_list = (
        "Hi there. I'm Zebediah Quux and nobody in the show notes ever mentions me at all, but "
        "here I am talking for long enough to clear the sample floor and be shown to the model."
    )
    answer = json.loads(_resolution_response_json(_prompt({"SPEAKER_00": off_list})))["voices"]
    assert answer["SPEAKER_00"]["name"] is None
