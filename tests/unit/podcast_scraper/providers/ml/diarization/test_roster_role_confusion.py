"""Host/guest role confusion in guest-heavy interviews (#1169).

community-1's finer clustering splits a dominant guest and an under-speaking host into separate
clusters, exposing three ways the roster mislabels roles that deepgram's coarser clustering hid:

1. a guest whose thank-you the speech-act list did not match was flagged NEITHER role and could be
   crowned a host (fixed by widening ``_GUEST_SPEECH_ACTS``);
2. a host greeting a guest by a title + surname ("Professor Pape, thanks for coming") named the
   right voice with a degraded label that beat the metadata-stated full name ("Robert Pape");
3. when the host introduced the guest and the CO-HOST spoke next, the guest's name landed on the
   co-host AND blocked the co-host's own name — a guest name on a host voice, with the host dropped.

All fixtures are synthetic (never-commit-real-episodes); the shapes mirror the two real episodes
that surfaced the bugs (The Daily's Robert Pape, Hard Fork's Adam Rodman).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _canonicalize_to_stated_person,
    resolve_speaker_roster,
)
from podcast_scraper.speaker_detectors.hosts import roles_from_conversation

pytestmark = pytest.mark.unit


def _scripted(turns: List[Tuple[str, str]], seg_len: float = 25.0):
    """Build (diarization, voice_texts, ordered_turns) from a script of (speaker, text) turns.

    Segments start at 30s (past the edge-ad window) and each runs ``seg_len`` seconds so every voice
    clears the cameo floor (``CAMEO_MAX_TALK_S = 20``) and is a real speaker, not a clip.
    """
    segs: List[DiarizationSegment] = []
    vtext: Dict[str, str] = {}
    ordered: List[Tuple[str, str]] = []
    t = 30.0
    for spk, text in turns:
        segs.append(DiarizationSegment(start=t, end=t + seg_len, speaker=spk))
        t += seg_len
        vtext[spk] = vtext.get(spk, "") + " " + text
        ordered.append((spk, text))
    return DiarizationResult(segments=segs, num_speakers=len(vtext)), vtext, ordered


# --- Mechanism 1: the guest speech-act widen -------------------------------------------------


def test_roles_from_conversation_flags_a_thank_you_very_much_guest() -> None:
    # "thank you VERY much for having me" matched neither old fixed pattern, so the dominant guest
    # went unflagged and was eligible to be crowned a host.
    assert roles_from_conversation({"v": "Thank you very much for having me."}) == {"v": "guest"}
    assert roles_from_conversation({"v": "Thanks so much for having me on."}) == {"v": "guest"}


def test_a_host_act_still_wins_when_both_phrases_appear() -> None:
    # Precedence guard: a voice that performs a host act is a host even if it also quotes a thanks
    # (a host reading a listener note). Host is checked before guest in roles_from_conversation.
    text = "Welcome back to the show. A listener wrote in to say thank you very much for having me."
    assert roles_from_conversation({"v": text}) == {"v": "host"}


# --- Mechanism 2: title-form upgraded to the stated full name --------------------------------


@pytest.mark.parametrize(
    ("name", "stated", "expected"),
    [
        ("Professor Pape", ["Robert Pape"], "Robert Pape"),  # unique same-person → upgrade
        ("Dr. Pape", ["Robert Pape"], "Robert Pape"),
        ("Professor Pape", ["Robert Pape", "Karen Pape"], "Professor Pape"),  # two Papes → keep
        ("Professor Pape", [], "Professor Pape"),  # nothing stated → keep
        ("Professor Pape", ["Robert Chen"], "Professor Pape"),  # different surname → keep
        ("Robert Pape", ["Robert Pape"], "Robert Pape"),  # already full → no-op
    ],
)
def test_canonicalize_to_stated_person(name: str, stated: List[str], expected: str) -> None:
    assert _canonicalize_to_stated_person(name, stated) == expected


def test_a_title_form_greeting_is_upgraded_to_the_metadata_name() -> None:
    diar, vtext, ordered = _scripted(
        [
            ("HOST", "Hello and welcome to the show, I'm Michael Barbaro."),
            ("HOST", "Professor Pape, thank you so much for coming on the show today."),
            (
                "GUEST",
                "Thank you very much for having me. Political violence in America is rising.",
            ),
            ("HOST", "Tell me more about how we got here and what the data shows."),
            ("GUEST", "The most important fact is that tens of millions now condone it."),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["HOST"],
        detected_guests=["Robert Pape"],
        known_hosts=["Michael Barbaro"],
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=["Robert Pape"],
        diarization_provider="tailnet_dgx",
    )
    assert roster.by_voice["GUEST"].name == "Robert Pape"  # not the title-form "Professor Pape"
    assert roster.by_voice["GUEST"].role == "guest"


# --- Mechanism 3: the guest name must not land on the co-host, and the host is kept ----------


def test_the_introduced_guest_name_lands_on_the_guest_not_the_co_host() -> None:
    """THE WORST BUG. Host introduces the guest; the CO-HOST banters back before the guest answers.
    The intro reader used to name that next voice — the co-host — with the guest's name, which both
    mislabels the host and blocks the host's own name from the pool. The host is dropped entirely.
    """
    diar, vtext, ordered = _scripted(
        [
            ("H1", "Welcome back to the show. I'm Casey Newton, in New York."),
            ("H1", "Our guest today is a doctor. Adam Rodman, welcome to the show."),
            ("H2", "Welcome to the show, so glad we could finally make this happen."),
            ("GUEST", "Thank you very much for having me. AI in medicine is moving fast."),
            ("H1", "Where has it moved the most in the last year, would you say?"),
            ("GUEST", "Diagnostics, mostly — the models are getting genuinely good at it now."),
            ("H2", "That tracks with what we have been hearing on the show all season."),
            ("GUEST", "And the tooling around it has matured a great deal as well lately."),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["H1"],
        detected_guests=["Dr. Adam Rodman"],
        known_hosts=["Casey Newton", "Kevin Roose"],
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=["Dr. Adam Rodman"],
        diarization_provider="tailnet_dgx",
    )
    # the guest's name is on the guest voice, not a host
    assert "Rodman" in roster.by_voice["GUEST"].name
    assert roster.by_voice["GUEST"].role == "guest"
    # neither host carries the guest's name
    for v in ("H1", "H2"):
        assert "Rodman" not in roster.by_voice[v].name, f"{v} was painted with the guest's name"
    # and the second known host is recovered from the pool rather than dropped
    host_names = {r.name for r in roster.by_voice.values() if r.role == "host"}
    assert host_names == {"Casey Newton", "Kevin Roose"}


def test_an_introduction_of_a_stated_host_may_still_name_a_host_voice() -> None:
    """The exception the guard must preserve: "welcome back my co-host Kevin Roose" legitimately
    names a host voice — the skip only applies when the introduced name is NOT a stated host."""
    diar, vtext, ordered = _scripted(
        [
            ("H1", "Welcome to the show. I'm Casey Newton."),
            ("H1", "And joining me, as always, my co-host Kevin Roose."),
            ("H2", "Hello everyone, there is a lot to get through on the show today."),
            ("H1", "Lots to get through indeed, it has been a wild week in tech news."),
            ("H2", "It really has, let us get right into the first story of the day."),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["H1"],
        detected_guests=[],
        known_hosts=["Casey Newton", "Kevin Roose"],
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=[],
        diarization_provider="tailnet_dgx",
    )
    assert roster.by_voice["H2"].name == "Kevin Roose"
    assert roster.by_voice["H2"].role == "host"


def test_a_co_host_self_naming_survives_a_merged_ad_testimonial() -> None:
    """Regression from the mech-1 widen. community-1 merged an ad testimonial ("...thank you so much
    for having me") into a co-host's cluster; the widened guest-act then flipped him to a guest,
    dropped his host-candidacy, and his ASR-mangled self-intro never canonicalized. A voice naming
    itself as a STATED host is a host despite the stray guest phrase — matching deepgram's result.
    """
    diar, vtext, ordered = _scripted(
        [
            ("H1", "Welcome to the show. I'm Casey Newton."),
            ("H1", "Big week in tech — let us get into the first story before the break."),
            # the co-host names himself (ASR-mangled) AND his cluster absorbed an ad testimonial
            (
                "H2",
                "I'm Kevin Russo. It protects my child. Thank you so much for having me.",
            ),
            ("H1", "Right, so back to the story, what did the company actually announce today?"),
            (
                "H2",
                "They shipped a new model, and the benchmarks are genuinely striking this time.",
            ),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["H1"],
        detected_guests=[],
        known_hosts=["Casey Newton", "Kevin Roose"],
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=[],
        diarization_provider="tailnet_dgx",
    )
    assert (
        roster.by_voice["H2"].name == "Kevin Roose"
    )  # canonicalized, not the mangled "Kevin Russo"
    assert roster.by_voice["H2"].role == "host"
