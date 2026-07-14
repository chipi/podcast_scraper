"""`unidentified` means NOBODY NAMED THEM. It must never mean "we could not find the name".

The taxonomy exists to keep two things apart:

    unknown        a real person we FAILED to name  -> our defect, and it must be COUNTED
    unidentified   a real person NOBODY names       -> the vox-pop of a narrated show; not our fault

But `unidentified` was assigned to any voice that was not in the ``nameable`` set — and ``nameable``
was built from the names that SURVIVED corroboration. So when corroboration correctly rejected an
uncorroborated guest ("the episode text names them but never introduces them as speaking", #876),
the name vanished, nothing was left going spare, and the roster concluded that nobody could have
named the voice:

    [Physical AI] description: "Qasar Younis and Peter Ludwig have spent the last decade..."
                  SPEAKER_01, 35% of the episode -> "Unidentified speaker"

The show notes name him. We could not place him. Two correct safety rules compounding into a false
innocence — and the defect number that came out the other side (3.52% "truly unknown") was a lie.

METADATA IS THE AUTHORITY. If the metadata stated a name and we did not place it, that is OURS.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from podcast_scraper.providers.ml.diarization.base import (
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.ml.diarization.roster import (
    build_speaker_diagnostics,
    resolve_speaker_roster,
    VOICE_UNIDENTIFIED,
    VOICE_UNKNOWN,
)

pytestmark = pytest.mark.unit

HOST = "Alexi Horowitz-Gazi"


def _episode() -> tuple[DiarizationResult, Dict[str, str], List[tuple[str, str]]]:
    """A host who names himself, and a principal guest who speaks for a third of the episode."""
    turns = [
        ("SPEAKER_00", 0.0, 200.0, f"Welcome back to the show. I'm {HOST}, and today we dig in."),
        (
            "SPEAKER_01",
            200.0,
            500.0,
            "Traditionally robotics has been a vertically oriented field.",
        ),
        ("SPEAKER_00", 500.0, 700.0, "Say more about the data problem there."),
        ("SPEAKER_01", 700.0, 1000.0, "Any robot, any task, one brain. That is the whole thesis."),
    ]
    diar = DiarizationResult(
        segments=[DiarizationSegment(s, e, v) for v, s, e, _ in turns],
        num_speakers=2,
    )
    voice_texts: Dict[str, str] = {}
    for v, _, _, text in turns:
        voice_texts[v] = (voice_texts.get(v, "") + " " + text).strip()
    ordered = [(v, text) for v, _, _, text in turns]
    return diar, voice_texts, ordered


def _roster(metadata_named: List[str]):
    diar, voice_texts, ordered = _episode()
    return diar, resolve_speaker_roster(
        diar,
        " ".join(t for _, t in ordered),
        detected_guests=[],  # corroboration threw the guest away — this is the whole point
        known_hosts=[HOST],
        voice_texts=voice_texts,
        ordered_turns=ordered,
        metadata_named=metadata_named,
    )


class TestTheGuestWeCouldNotPlace:
    def test_a_stated_name_we_failed_to_place_is_our_DEFECT(self) -> None:
        """THE BUG. The description names him; we could not bind him; that is `unknown`."""
        _, roster = _roster(["Qasar Younis"])
        guest = roster.by_voice["SPEAKER_01"]

        assert not guest.named
        assert guest.voice_type == VOICE_UNKNOWN, (
            "a guest the show notes NAME was filed as 'nobody could have named them' — the "
            "corroboration gate rejected the name, and its absence then became our alibi"
        )

    def test_when_the_metadata_names_NOBODY_the_voice_is_genuinely_unidentified(self) -> None:
        """The vox-pop of a narrated show. No name existed anywhere, so nothing to fail at."""
        _, roster = _roster([])
        assert roster.by_voice["SPEAKER_01"].voice_type == VOICE_UNIDENTIFIED

    def test_a_stated_name_we_DID_place_does_not_incriminate_us(self) -> None:
        """The host is stated and bound. A name that reached a voice is not going spare."""
        _, roster = _roster([HOST])
        assert roster.by_voice["SPEAKER_01"].voice_type == VOICE_UNIDENTIFIED


class TestThePerEpisodeAlarm:
    """A headcount cannot tell three cameos from one lost principal. A share of TALK can."""

    def test_a_lost_principal_raises_the_alarm_and_names_who(self) -> None:
        diar, roster = _roster(["Qasar Younis"])
        summary = build_speaker_diagnostics(
            diar, roster, known_hosts=[HOST], metadata_named=["Qasar Younis"]
        )["summary"]

        assert summary["unattributed_talk_share"] == pytest.approx(0.6, abs=0.01)
        assert summary["unattributed_alarm"] is True
        assert summary["unbound_names"] == [
            "Qasar Younis"
        ], "the alarm must say WHO we lost — an alarm you cannot act on gets muted"

    def test_a_fully_attributed_episode_is_quiet(self) -> None:
        diar, roster = _roster([HOST])
        # Bind the guest too, so nothing substantive is left unattributed.
        summary = build_speaker_diagnostics(
            diar, roster, known_hosts=[HOST], metadata_named=[HOST]
        )["summary"]
        assert summary["unbound_names"] == []


def test_the_diarization_pipeline_actually_FORWARDS_the_stated_names() -> None:
    """A parameter nobody passes is a parameter that does not exist.

    The roster can only tell a defect from an innocent voice if the pipeline hands it the names the
    metadata stated — and the whole point of this change is that they were being thrown away.
    """
    import inspect

    from podcast_scraper.providers.ml.diarization import pipeline as diar_pipeline

    src = inspect.getsource(diar_pipeline.apply_diarization_to_result)
    assert (
        "metadata_named" in inspect.signature(diar_pipeline.apply_diarization_to_result).parameters
    )
    assert src.count("metadata_named=list(metadata_named or ())") == 2, (
        "apply_diarization_to_result must forward the stated names to BOTH the roster and the "
        "diagnostics — otherwise the defect accounting silently reverts to laundering our failures"
    )
