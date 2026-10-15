"""A person named anywhere in the episode title is a spelling reference for the stated-name snap.

The title readers take a guest from a title's opening ("X on Y", "X: ...") or its tail ("..., with
X", "... | X"). A name in the MIDDLE ("Why X Is Skeptical Of ...") was read by neither, so when the
metadata extraction stated nobody the guest's own heard spelling was published: the canonicalizer
had no correct spelling to snap it to (prod replay 2026-10-15, a Matt/Matthew nickname with a
one-letter surname mangling). The title name is a REFERENCE only: it may correct the spelling of a
name a voice already carries, and never names a voice by itself. Synthetic names throughout.
"""

from __future__ import annotations

from typing import List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _title_person_spellings,
    resolve_speaker_roster,
)

pytestmark = [pytest.mark.unit]

TITLE = "Why Matthew Ybarrola Is Skeptical Of Anti-Monopoly Policies"


def _diar(segs: List[Tuple[str, float, float]], n: int) -> DiarizationResult:
    return DiarizationResult(
        segments=[DiarizationSegment(s, e, v) for v, s, e in segs], num_speakers=n
    )


def _roster(guest_text: str, **kw):
    turns = [
        ("HOST", "Welcome to the show. I'm Dana Whitfield, and today we talk about antitrust."),
        ("GUEST", guest_text),
        ("HOST", "Thanks. Let's start with the stadium proposal and why it bothers you so much."),
    ]
    diar = _diar([("HOST", 0, 60), ("GUEST", 60, 400), ("HOST", 400, 700)], 2)
    return resolve_speaker_roster(
        diar,
        " ".join(t for _, t in turns),
        known_hosts=["Dana Whitfield"],
        voice_texts={
            "HOST": turns[0][1] + " " + turns[2][1],
            "GUEST": guest_text,
        },
        ordered_turns=turns,
        **kw,
    )


GUEST_SAYS = "Thanks for having me. I'm Matt Ibarrola, and I write a newsletter about economics."


def test_the_heard_spelling_snaps_to_the_title_spelling() -> None:
    r = _roster(GUEST_SAYS, episode_title=TITLE)
    assert r.by_voice["GUEST"].name == "Matthew Ybarrola"
    assert r.by_voice["HOST"].name == "Dana Whitfield"


def test_without_the_title_the_heard_spelling_stands() -> None:
    assert _roster(GUEST_SAYS).by_voice["GUEST"].name == "Matt Ibarrola"


def test_a_different_person_is_not_renamed_to_the_title_person() -> None:
    r = _roster(
        "Thanks for having me. I'm Carla Benton, and I write a newsletter about economics.",
        episode_title=TITLE,
    )
    assert r.by_voice["GUEST"].name == "Carla Benton"


def test_the_title_name_never_names_a_voice_by_itself() -> None:
    r = _roster(
        "Thanks for having me. The stadium question is mostly about complementary goods.",
        episode_title=TITLE,
    )
    assert not r.by_voice["GUEST"].named


def test_a_role_in_brackets_does_not_respell_a_correct_name() -> None:
    r = _roster(
        "Thanks for having me. I'm Grant Okafor, and I run a company that makes slides.",
        episode_title="Inside the rise of Gamma | Grant Okafor (CEO)",
    )
    assert r.by_voice["GUEST"].name == "Grant Okafor"


def test_the_metadatas_spelling_wins_over_the_titles() -> None:
    # Both are close to the heard "Matt Ybarra"; the metadata's is checked first.
    r = _roster(
        "Thanks for having me. I'm Matt Ybarra, and I write a newsletter about economics.",
        episode_title="Why Matthew Ybarrola Is Skeptical",
        metadata_named=["Matthew Ybarra"],
    )
    assert r.by_voice["GUEST"].name == "Matthew Ybarra"


def test_a_stated_host_keeps_its_own_spelling_over_a_near_title_name() -> None:
    # The corroborated reference (the feed's host) comes first; a title naming someone close to
    # the host does not respell the host.
    r = _roster(GUEST_SAYS, episode_title="Dana Whitfeld Is Wrong About Antitrust, With Matt")
    assert r.by_voice["HOST"].name == "Dana Whitfield"


@pytest.mark.parametrize(
    "title, expected",
    [
        (TITLE, ["Matthew Ybarrola"]),
        ("How Priya Raman's Lab Builds Robots", ["Priya Raman"]),
        ("Interest Rates, Inflation, and Lena Okoro on the Fed", ["Lena Okoro"]),
        ("Why We Are Skeptical", []),
        # A bracket closes the run: "Grant Lee CEO" respelled a guest's correct "Grant Lee" and
        # the plausibility gate then dropped it (drill replay 2026-10-15).
        ("Inside the rise of Gamma | Grant Okafor (CEO)", ["Grant Okafor"]),
        ("Hybrid Search — Lena Okoro of TurboPuffer [Sponsored]", ["Lena Okoro"]),
        ("[Lena Okoro] Explains Rates", ["Lena Okoro"]),
    ],
)
def test_title_person_spellings(title, expected) -> None:
    # Permissive by design ("Interest Rates" passes the shape check too): a reference that is no
    # person can only respell a published name close to it, which no real name is.
    got = _title_person_spellings(title)
    assert all(name in got for name in expected)
    assert not any(n.split()[-1].isupper() for n in got)
    if not expected:
        assert got == []


@pytest.mark.parametrize(
    "title, expected",
    [
        ("Inside the rise of Gamma | Grant Okafor (Gamma)", ["Grant Okafor"]),
        ("Rates With Lena Okoro CEO", ["Lena Okoro"]),
        # One word is no full name, four is a phrase: neither is a reference.
        ("Why Gamma Is Winning", []),
        ("Why Federal Reserve Rate Policy", []),
        ("Why We Are Skeptical", []),
        ("", []),
        (None, []),
    ],
)
def test_title_person_spellings_exactly(title, expected) -> None:
    assert _title_person_spellings(title) == expected
