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


class TestNamingTheVoicesWeWereLosing:
    """Latent Space named NOBODY — 100% of the talk, on nearly every episode.

    The feed states no host (its author tag is the org, "Latent.Space"), so the only evidence is the
    conversation, and both of the show's habits defeated us.
    """

    def _run(self, turns, metadata_named, known_hosts=()):
        diar = DiarizationResult(
            segments=[DiarizationSegment(s, e, v) for v, s, e, _ in turns],
            num_speakers=len({v for v, _, _, _ in turns}),
        )
        voice_texts: Dict[str, str] = {}
        for v, _, _, text in turns:
            voice_texts[v] = (voice_texts.get(v, "") + " " + text).strip()
        return resolve_speaker_roster(
            diar,
            " ".join(t for _, _, _, t in turns),
            detected_guests=[],  # corroboration rejected them — that is the situation under test
            known_hosts=list(known_hosts),
            voice_texts=voice_texts,
            ordered_turns=[(v, t) for v, _, _, t in turns],
            metadata_named=metadata_named,
        )

    def test_a_FIRST_NAME_self_intro_binds_when_the_metadata_vouches_for_it(self) -> None:
        """ "Hey everyone, welcome to the Latent Space Podcast. This is Alessio."

        One token, so the guest path threw it away ("I'm American"). The metadata names him, and
        the conversation says which voice — together that is a warrant, and it costs us every host
        who uses their first name.
        """
        roster = self._run(
            [
                (
                    "SPEAKER_04",
                    0.0,
                    300.0,
                    "Hey everyone, welcome to the podcast. This is Alessio.",
                ),
                (
                    "SPEAKER_01",
                    300.0,
                    900.0,
                    "Thanks for having me. So the thesis is quite simple.",
                ),
            ],
            metadata_named=["Alessio Fanelli", "Marc Andreessen"],
        )
        host = roster.by_voice["SPEAKER_04"]
        assert host.named
        assert host.name == "Alessio Fanelli", "a bare first name the metadata states must bind"

    def test_a_bare_first_name_the_metadata_does_NOT_state_still_binds_nobody(self) -> None:
        """The guard that made the first+last rule right in the first place."""
        roster = self._run(
            [
                ("SPEAKER_04", 0.0, 300.0, "Welcome to the show. I'm American, and I love this."),
                ("SPEAKER_01", 300.0, 900.0, "Thanks for having me. The thesis is quite simple."),
            ],
            metadata_named=["Marc Andreessen"],
        )
        assert roster.by_voice["SPEAKER_04"].name != "American"

    def test_an_AMBIGUOUS_first_name_binds_nobody(self) -> None:
        """Two stated people share the first name. The conversation cannot say which — so neither
        of them gets it."""
        roster = self._run(
            [
                ("SPEAKER_04", 0.0, 300.0, "Welcome to the show. This is Chris."),
                ("SPEAKER_01", 300.0, 900.0, "Thanks for having me. The thesis is quite simple."),
            ],
            metadata_named=["Chris Wright", "Chris Lattner"],
        )
        assert not roster.by_voice["SPEAKER_04"].named

    def test_the_metadata_NEVER_names_a_voice_that_did_not_say_it(self) -> None:
        """THE RULE THAT WAS TESTED, BUILT, MEASURED, AND REMOVED.

        The tempting rule: let one confirmed guest vouch for the people named beside him. "Qasar
        Younis and Peter Ludwig have spent the last decade..." — Peter self-introduces, so surely
        Qasar (35% of the episode) is a speaker too?

        It was implemented and replayed over all 160 episodes. It admitted 8 names: 3 real guests
        and 5 people who were never in the room, including **HB Reese** — the founder of Reese's,
        discussed by a Planet Money episode, dead since 1956 — painted onto a voice. That is the
        #876 failure exactly.

        A description that names several people is a guest list or a topic list, and one member
        speaking does not tell you which. So the metadata alone vouches for NOBODY: it may only
        confirm a name the VOICE ITSELF uttered.
        """
        roster = self._run(
            [
                ("SPEAKER_00", 0.0, 300.0, "Welcome back. I'm Alexi Horowitz-Gazi."),
                (
                    "SPEAKER_03",
                    300.0,
                    900.0,
                    "Yeah, hi, I'm Brad Reese, and my family started the company.",
                ),
                (
                    "SPEAKER_09",
                    900.0,
                    1500.0,
                    "It was one of the greatest things that ever happened.",
                ),
            ],
            # Brad Reese speaks and is stated. HB Reese is stated too — dead since 1956.
            metadata_named=["Alexi Horowitz-Gazi", "Brad Reese", "HB Reese"],
            known_hosts=["Alexi Horowitz-Gazi"],
        )
        assert roster.by_voice["SPEAKER_03"].name == "Brad Reese", "his own mouth names him"

        tape = roster.by_voice["SPEAKER_09"]
        assert not tape.named, (
            "a confirmed speaker was allowed to vouch for the OTHER people the description names, "
            "and a man dead since 1956 was given a voice"
        )
        assert tape.name != "HB Reese"


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
