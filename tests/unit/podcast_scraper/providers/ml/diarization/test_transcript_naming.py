"""Transcript naming: the model names every voice from the whole transcript (see the module)."""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization import transcript_naming as tn
from podcast_scraper.providers.ml.diarization.naming_trace import NamingTrace
from podcast_scraper.providers.ml.diarization.roster import SpeakerRole, SpeakerRoster

pytestmark = [pytest.mark.unit]

WORDS = " ".join(["word"] * 45)


def _aligned():
    return [
        (
            {"start": 0.0, "end": 5.0, "text": "Welcome to the show, I'm Russ Roberts."},
            "SPEAKER_00",
        ),
        ({"start": 5.0, "end": 9.0, "text": WORDS}, "SPEAKER_00"),
        ({"start": 9.0, "end": 60.0, "text": "Thanks for having me, Russ. " + WORDS}, "SPEAKER_01"),
        ({"start": 61.0, "end": 64.0, "text": "This episode is brought to you by"}, "SPEAKER_02"),
        ({"start": 64.0, "end": 65.0, "text": "uh"}, None),
    ]


def _roster():
    return SpeakerRoster(
        by_voice={
            # The ladder swapped the two people.
            "SPEAKER_00": SpeakerRole(
                name="Stephen Witt", role="guest", named=True, source="llm_resolution"
            ),
            "SPEAKER_01": SpeakerRole(
                name="Russ Roberts", role="host", named=True, source="llm_resolution"
            ),
            "SPEAKER_02": SpeakerRole(
                name="Daniel Atkinson", role="guest", named=True, source="self_intro"
            ),
        },
        num_speakers=3,
    )


def test_the_transcript_block_merges_turns_and_keeps_the_measured_layout() -> None:
    text = tn.transcript_block(
        _aligned(), "EconTalk", "Hosted by Russ Roberts.", "Ep 1", "With Stephen Witt."
    )
    lines = text.splitlines()
    assert lines[:6] == [
        "FEED: EconTalk",
        "FEED DESCRIPTION: Hosted by Russ Roberts.",
        "EPISODE: Ep 1",
        "EPISODE DESCRIPTION: With Stephen Witt.",
        "",
        "TRANSCRIPT (diarized voices; no names attached):",
    ]
    assert lines[6].startswith("[00:00] SPEAKER_00: Welcome to the show, I'm Russ Roberts. word")
    assert lines[7].startswith("[00:09] SPEAKER_01: Thanks for having me, Russ.")
    assert lines[8] == "[01:01] SPEAKER_02: This episode is brought to you by"
    assert lines[9] == "[01:04] UNATTRIBUTED: uh"
    assert tn.voices_to_judge(text) == ["SPEAKER_00", "SPEAKER_01"]


def test_the_prompt_is_the_measured_one() -> None:
    # podcast-scraper-eval-data autoresearch/naming_rethink/qwen_verify.py: change both together.
    assert tn.PROMPT.startswith("You identify the speakers of one podcast episode.")
    assert "A wrong name is worse than a missing one: when unsure, null." in tn.PROMPT
    assert tn.build_prompt("X", ["SPEAKER_00", "SPEAKER_01"]).endswith(
        "VOICES TO JUDGE: SPEAKER_00, SPEAKER_01\n"
    )


def test_the_answer_reads_null_strings_as_no_name() -> None:
    raw = (
        '{"voices": {"SPEAKER_00": {"name": "null", "role": "ad"},'
        ' "SPEAKER_01": {"name": "Kim Lee", "role": "Guest"}}}'
    )
    assert tn.parse_answer(raw) == {"SPEAKER_00": (None, "ad"), "SPEAKER_01": ("Kim Lee", "guest")}
    assert tn.parse_answer("not json") is None


def test_a_swap_is_corrected_and_a_voice_it_does_not_call_a_participant_is_unnamed() -> None:
    def complete(prompt: str) -> str:
        assert "VOICES TO JUDGE: SPEAKER_00, SPEAKER_01" in prompt
        return (
            '{"voices": {"SPEAKER_00": {"name": "Russ Roberts", "role": "host"},'
            ' "SPEAKER_01": {"name": "Stephen Witt", "role": "guest"}}}'
        )

    trace = NamingTrace()
    out = tn.name_from_transcript(
        _roster(),
        _aligned(),
        complete,
        feed_title="EconTalk",
        feed_description=None,
        episode_title="Ep 1",
        episode_description=None,
        trace=trace,
    )
    assert (out.by_voice["SPEAKER_00"].name, out.by_voice["SPEAKER_00"].role) == (
        "Russ Roberts",
        "host",
    )
    assert (out.by_voice["SPEAKER_01"].name, out.by_voice["SPEAKER_01"].role) == (
        "Stephen Witt",
        "guest",
    )
    assert out.by_voice["SPEAKER_02"].named is False
    assert out.by_voice["SPEAKER_02"].source == tn.SOURCE
    assert trace.inputs[tn.SOURCE]["voices_asked"] == ["SPEAKER_00", "SPEAKER_01"]
    assert trace.voices["SPEAKER_02"][0]["decision"] == "removed"


def test_a_failed_or_unreadable_call_leaves_the_ladders_roster() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("gateway down")

    before = _roster()
    kwargs = dict(
        feed_title=None, feed_description=None, episode_title=None, episode_description=None
    )
    assert tn.name_from_transcript(before, _aligned(), boom, **kwargs) is before
    assert tn.name_from_transcript(before, _aligned(), lambda p: "sorry", **kwargs) is before


def test_off_by_default() -> None:
    from podcast_scraper import config

    assert config.Config.model_fields["transcript_naming"].default is False
