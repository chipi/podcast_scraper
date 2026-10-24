"""The talk floor: a name comes off a voice that holds too little of the episode's talk.

Measured on the labelled naming sets (podcast-scraper-eval-data, NAMING-RETHINK-2026-10-22.md):
a person's name on a voice under 5% of the talk is almost always an ad, a promo or a clip voice.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.diarization.naming_trace import NamingTrace
from podcast_scraper.providers.ml.diarization.pipeline import _apply_talk_floor
from podcast_scraper.providers.ml.diarization.roster import SpeakerRole, SpeakerRoster

pytestmark = [pytest.mark.unit]


def _roster():
    return SpeakerRoster(
        by_voice={
            "SPEAKER_00": SpeakerRole(
                name="Russ Roberts", role="host", named=True, source="self_intro"
            ),
            "SPEAKER_01": SpeakerRole(
                name="Stephen Witt", role="guest", named=True, source="llm_resolution"
            ),
            "SPEAKER_02": SpeakerRole(
                name="Kim Cattrall", role="guest", named=True, source="llm_resolution"
            ),
            "SPEAKER_03": SpeakerRole(name="SPEAKER_03", role="unknown", named=False, source="raw"),
        },
        num_speakers=4,
    )


def _aligned():
    # Host 1,000 s, guest 1,800 s, a promo voice 40 s (1.4%), an unnamed cameo 10 s.
    return [
        ({"start": 0.0, "end": 1000.0}, "SPEAKER_00"),
        ({"start": 1000.0, "end": 2800.0}, "SPEAKER_01"),
        ({"start": 2800.0, "end": 2840.0}, "SPEAKER_02"),
        ({"start": 2840.0, "end": 2850.0}, "SPEAKER_03"),
        ({"start": 2850.0, "end": 2851.0}, None),
    ]


def test_a_name_on_a_voice_under_the_floor_comes_off() -> None:
    out = _apply_talk_floor(_roster(), _aligned(), 0.05)
    promo = out.by_voice["SPEAKER_02"]
    assert (promo.named, promo.name, promo.role, promo.source) == (
        False,
        "SPEAKER_02",
        "unknown",
        "talk_floor",
    )


def test_names_above_the_floor_and_unnamed_voices_are_untouched() -> None:
    before = _roster()
    out = _apply_talk_floor(before, _aligned(), 0.05)
    for voice in ("SPEAKER_00", "SPEAKER_01", "SPEAKER_03"):
        assert out.by_voice[voice] == before.by_voice[voice]


def test_off_by_default_and_at_zero() -> None:
    before = _roster()
    assert config.Config.model_fields["speaker_name_min_talk_share"].default == 0.0
    assert _apply_talk_floor(before, _aligned(), 0.0) is before


def test_the_decision_is_traced() -> None:
    trace = NamingTrace()
    _apply_talk_floor(_roster(), _aligned(), 0.05, trace)
    assert trace.voices == {
        "SPEAKER_02": [
            {
                "rung": "talk_floor",
                "decision": "removed",
                "name": "Kim Cattrall",
                "talk_share": 0.014,
            }
        ]
    }


def test_no_talk_at_all_changes_nothing() -> None:
    before = _roster()
    assert _apply_talk_floor(before, [], 0.05) is before


def _quiet_host_roster(role: str = "host"):
    # A moderator of a panel: seated host, 40 s of a 2,850 s episode (1.4%).
    return SpeakerRoster(
        by_voice={
            "SPEAKER_00": SpeakerRole(
                name="Russ Roberts", role="guest", named=True, source="self_intro"
            ),
            "SPEAKER_01": SpeakerRole(
                name="Stephen Witt", role="guest", named=True, source="llm_resolution"
            ),
            "SPEAKER_02": SpeakerRole(
                name="Ada Moderator", role=role, named=True, source="llm_resolution"
            ),
            "SPEAKER_03": SpeakerRole(name="SPEAKER_03", role="unknown", named=False, source="raw"),
        },
        num_speakers=4,
    )


def test_a_seated_host_the_llm_also_calls_host_keeps_the_name_under_the_floor() -> None:
    trace = NamingTrace()
    out = _apply_talk_floor(
        _quiet_host_roster(), _aligned(), 0.05, trace, llm_roles={"SPEAKER_02": "host"}
    )
    assert out.by_voice["SPEAKER_02"].name == "Ada Moderator"
    assert out.by_voice["SPEAKER_02"].named
    steps = trace.to_dict()["voices"]["SPEAKER_02"]
    assert {
        "rung": "talk_floor",
        "decision": "kept_host",
        "name": "Ada Moderator",
        "talk_share": 0.014,
    } in steps


@pytest.mark.parametrize(
    "role, llm_roles",
    [
        ("host", {"SPEAKER_02": "guest"}),  # seated, but the LLM resolution says guest
        ("host", {}),  # seated, the LLM resolution gave no role (a seated promo voice)
        ("guest", {"SPEAKER_02": "host"}),  # the LLM says host, the ladder did not seat it
        ("host", None),  # no LLM roles at all
    ],
)
def test_one_host_signal_alone_does_not_keep_the_name(role, llm_roles) -> None:
    out = _apply_talk_floor(_quiet_host_roster(role), _aligned(), 0.05, llm_roles=llm_roles)
    assert not out.by_voice["SPEAKER_02"].named
    assert out.by_voice["SPEAKER_02"].source == "talk_floor"
