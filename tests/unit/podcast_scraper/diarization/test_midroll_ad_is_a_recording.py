"""#1188 — a voice that says the same thing every week is a RECORDING, not a person.

The edge-ad rule catches the pre-roll: short, and confined to the top or tail. A MID-ROLL house ad
defeats it by sitting in the middle — and the narrator reads its own name aloud, which is the
roster's most-trusted signal. So `Jonathan Knight` (NYT Games) entered 7 of 10 Hard Fork episodes as
a named person, with GUESTS_ON edges.

No keyword can see it: modern house ads carry no sponsor language at all. The signal exists only
ACROSS episodes — the ad is read from the same script every week, and the show is not. Measured on
the real corpus:

    Jonathan Knight    1.3-1.7% of talk, 77-100% of his words repeat across the feed  <- a recording
    Jack Clark         0.6% of talk,     80% repeat                                   <- a recording
    Robert Armstrong   3.4-3.6% of talk, 70-77% repeat (he reads the CREDITS)         <- a real HOST

BOTH tests must hold. Repetition alone strips the co-host who reads the credits; a small share alone
is just a brief speaker. Together they describe an advertisement and nothing else.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import (
    DiarizationResult,
    DiarizationSegment,
)
from podcast_scraper.providers.ml.diarization.roster import (
    resolve_speaker_roster,
    VOICE_COMMERCIAL,
)
from podcast_scraper.speaker_detectors.boilerplate import (
    recorded_voices,
    recurring_shingles,
    repeated_fraction,
)

pytestmark = pytest.mark.unit

# The NYT Games house ad, read from the same script every week. No sponsor language whatsoever.
AD = (
    "I'm Jonathan Knight and I'm the head of games here at the New York Times. If you play our "
    "games you probably know there is something a bit different about them, something that makes "
    "you come back day after day after day to play them."
)
# The show's credits — ALSO repeated every week, and read by a real co-host.
CREDITS = (
    "This episode was produced by Jake Harper and edited by Brian Erstadt. Our executive producer "
    "is Jacob Goldstein and Cheryl Brumley is the FT's global head of audio."
)
HOST = (
    "Welcome back to the show. Today we are talking about the bond market and what it means for "
    "the Federal Reserve, and whether the long end is finally telling us something real."
)


def _episode(extra: str = "") -> str:
    return f"{HOST} {AD} {CREDITS} {extra}"


def _shingles() -> set:
    """Five episodes of the same show: the ad and the credits recur, the content does not."""
    return recurring_shingles(
        [
            _episode("The jobs number came in hot this morning."),
            _episode("Nvidia reported earnings after the bell."),
            _episode("The ECB held rates for a fourth meeting."),
            _episode("Oil slipped below eighty dollars a barrel."),
            _episode("Housing starts fell for the third month."),
        ]
    )


class TestTheScriptTheShowRepeats:
    def test_the_ad_and_the_credits_are_both_found(self) -> None:
        sh = _shingles()
        assert repeated_fraction(AD, sh) > 0.8
        assert repeated_fraction(CREDITS, sh) > 0.8

    def test_the_EPISODE_CONTENT_is_not(self) -> None:
        """The thing that makes this work: a show does not repeat its own reporting."""
        sh = _shingles()
        assert repeated_fraction("The jobs number came in hot this morning.", sh) == 0.0


class TestBothTestsMustHold:
    def test_the_midroll_ad_narrator_is_a_recording(self) -> None:
        sh = _shingles()
        got = recorded_voices(
            {"SPEAKER_AD": AD, "SPEAKER_HOST": HOST * 40},
            {"SPEAKER_AD": 25.0, "SPEAKER_HOST": 2400.0},  # 1% of the talk
            sh,
        )
        assert got == {"SPEAKER_AD"}

    def test_a_CO_HOST_reading_the_credits_is_NOT(self) -> None:
        """THE FALSE POSITIVE THE CORPUS FOUND.

        Robert Armstrong reads the credits — 70-77% repeated text — and he is a host of the show.
        Repetition alone would have stripped his name. The share is what tells them apart, and it is
        the same 3% bar the edge-ad rule already uses.
        """
        sh = _shingles()
        got = recorded_voices(
            {"SPEAKER_COHOST": CREDITS, "SPEAKER_HOST": HOST * 40},
            {"SPEAKER_COHOST": 120.0, "SPEAKER_HOST": 2400.0},  # 4.8% — above the bar
            sh,
        )
        assert got == set(), "a co-host who reads the credits was typed as an advertisement"


def test_the_roster_types_the_midroll_ad_as_COMMERCIAL() -> None:
    """End to end: the mid-roll ad narrator must not become a named person (#1188)."""
    turns: List[Tuple[str, float, float, str]] = [
        ("SPEAKER_00", 0.0, 600.0, HOST * 10),
        ("SPEAKER_09", 600.0, 625.0, AD),  # mid-roll, 25s, reads its own name
        ("SPEAKER_00", 625.0, 1800.0, HOST * 10),
    ]
    diar = DiarizationResult(
        segments=[DiarizationSegment(s, e, v) for v, s, e, _ in turns], num_speakers=2
    )
    voice_texts: Dict[str, str] = {}
    for v, _, _, text in turns:
        voice_texts[v] = (voice_texts.get(v, "") + " " + text).strip()

    roster = resolve_speaker_roster(
        diar,
        " ".join(t for _, _, _, t in turns),
        known_hosts=["Katie Martin"],
        voice_texts=voice_texts,
        ordered_turns=[(v, t) for v, _, _, t in turns],
        recurring_text=_shingles(),
    )
    ad = roster.by_voice["SPEAKER_09"]
    assert ad.voice_type == VOICE_COMMERCIAL, (
        "the mid-roll house ad read its own name aloud and became a named person — the roster's "
        "most-trusted signal is the one an advertiser is built to exploit"
    )
    assert ad.name != "Jonathan Knight"
    assert not ad.named


def test_the_diarization_pipeline_actually_BUILDS_the_index() -> None:
    """A rule with no cross-episode evidence is a rule that cannot fire."""
    import inspect

    from podcast_scraper.providers.ml.diarization import pipeline as diar_pipeline

    src = inspect.getsource(diar_pipeline.apply_diarization_to_result)
    assert (
        "recurring_text=_feed_recurring_text(cfg)" in src
    ), "the roster is never given the feed's repeated script, so the mid-roll ad rule is inert"
