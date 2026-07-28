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


# --- Mechanism 4: a guest must not fill an ABSENT co-host's seat -----------------------------
#
# The feed states N hosts; fewer than N are actually present this episode; the roster filled the
# empty host slot with the GUEST via its self-introduction. Surfaced on the v2.3.1 pilot: No Priors
# ("with Andy Fang") seated the DoorDash founder as a host over the absent Sarah Guo, and Unhedged
# seated Joshua Franklin over the absent Rob Armstrong. A voice that SAYS a name the feed did not
# state as a host is positive evidence it is NOT a stated host, so it may never fill a counted seat.


def test_a_self_introduced_guest_does_not_fill_an_absent_co_hosts_seat() -> None:
    """No Priors / Andy Fang. Two stated hosts, one present; the guest self-introduces a name that
    is NOT in the host pool and — with no thank-you cue — was seated into the vacant second seat."""
    diar, vtext, ordered = _scripted(
        [
            ("ELAD", "Welcome to No Priors, I'm Elad Gil, here as always this week."),
            ("ANDY", "I'm Andy Fang, co-founder of DoorDash, and I lead our engineering org."),
            ("ELAD", "Let us start with autonomous delivery. Where does that stand today?"),
            ("ANDY", "We have built it for years; robots now handle a real share of deliveries."),
            ("ELAD", "And what changed to make that work at the scale you run it at now?"),
            ("ANDY", "Cheaper sensors, better models, and a great deal of operational iteration."),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["ELAD"],
        detected_guests=[],  # Andy is caught by the pool contradiction alone, not a guest list
        known_hosts=["Elad Gil", "Sarah Guo"],  # Sarah is absent this episode
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=[],
        diarization_provider="tailnet_dgx",
    )
    assert roster.by_voice["ANDY"].name == "Andy Fang"
    assert roster.by_voice["ANDY"].role == "guest"  # NOT host
    assert roster.by_voice["ELAD"].name == "Elad Gil"
    assert roster.by_voice["ELAD"].role == "host"
    # only the present host is a host; the absent Sarah Guo is not painted onto the guest
    host_names = {r.name for r in roster.by_voice.values() if r.role == "host"}
    assert host_names == {"Elad Gil"}


def test_a_named_guest_does_not_fill_an_absent_co_hosts_seat() -> None:
    """Unhedged / Joshua Franklin. A co-hosted show with one host present; the guest is also on the
    episode's detected-guest list, yet was seated as the second host over the absent Rob Armstrong.
    """
    diar, vtext, ordered = _scripted(
        [
            ("KATIE", "Hello and welcome to Unhedged, I'm Katie Martin of the FT."),
            ("JOSH", "I'm Joshua Franklin, I cover the big banks here at the Financial Times."),
            ("KATIE", "So how exactly did JPMorgan end up winning the decade the way it did?"),
            ("JOSH", "Scale and discipline — they spent on technology while rivals pulled back."),
            ("KATIE", "And is Jamie Dimon the whole story, or is it deeper than one chief exec?"),
            ("JOSH", "Deeper. The bench he built is what actually compounds the advantage."),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["KATIE"],
        detected_guests=["Joshua Franklin"],  # even when the guest list DOES name him
        known_hosts=["Katie Martin", "Rob Armstrong"],  # Rob is absent this episode
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=["Joshua Franklin"],
        diarization_provider="tailnet_dgx",
    )
    assert roster.by_voice["JOSH"].name == "Joshua Franklin"
    assert roster.by_voice["JOSH"].role == "guest"  # NOT host
    assert roster.by_voice["KATIE"].role == "host"
    host_names = {r.name for r in roster.by_voice.values() if r.role == "host"}
    assert host_names == {"Katie Martin"}


def test_both_stated_hosts_present_are_still_both_seated() -> None:
    """The guard must not over-block: when both stated hosts ARE present (each self-naming into the
    pool) they are both seated, and only the true guest is a guest."""
    diar, vtext, ordered = _scripted(
        [
            ("ELAD", "Welcome to No Priors, I'm Elad Gil."),
            ("SARAH", "And I'm Sarah Guo, great to be co-hosting today's episode."),
            ("ANDY", "Thanks for having me. I'm Andy Fang, co-founder of DoorDash."),
            ("ELAD", "Let us get into autonomous delivery and where it stands right now."),
            ("SARAH", "Yes, walk us through how the robots handle real deliveries today."),
            ("ANDY", "Cheaper sensors and better models did most of the heavy lifting there."),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["ELAD"],
        detected_guests=["Andy Fang"],
        known_hosts=["Elad Gil", "Sarah Guo"],
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=["Andy Fang"],
        diarization_provider="tailnet_dgx",
    )
    host_names = {r.name for r in roster.by_voice.values() if r.role == "host"}
    assert host_names == {"Elad Gil", "Sarah Guo"}  # guard did not drop the present co-host
    assert roster.by_voice["ANDY"].role == "guest"
    assert roster.by_voice["ANDY"].name == "Andy Fang"


# --- ADR-135: the LLM host/guest verdict as BOUNDED advice (veto positional / anchor no-host) ----


def test_llm_guest_verdict_blocks_a_positional_host_seat_fill() -> None:
    """Two stated hosts, one present; a voice that neither self-introduces nor gives a guest cue
    fills the vacant seat positionally (step 4). The LLM's "guest" verdict blocks that (ADR-135)."""
    diar, vtext, ordered = _scripted(
        [
            ("ELAD", "Welcome to No Priors, I'm Elad Gil, here as always this week."),
            ("X", "The delivery robots now handle a real share of orders across several cities."),
            ("ELAD", "And what changed to make that work at the scale you run it at now?"),
            ("X", "Cheaper sensors and better models did most of the heavy lifting there lately."),
        ]
    )
    common = dict(
        detected_guests=[],
        known_hosts=["Elad Gil", "Sarah Guo"],  # Sarah absent → one seat is vacant
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=[],
        diarization_provider="tailnet_dgx",
    )
    # control: without the verdict, X fills the vacant second host seat positionally
    control = resolve_speaker_roster(diar, vtext["ELAD"], **common)
    assert (
        control.by_voice["X"].role == "host"
    ), "the positional seat-fill must be real to be vetoed"
    # with the LLM "guest" verdict, the seat-fill is blocked
    roster = resolve_speaker_roster(diar, vtext["ELAD"], llm_voice_roles={"X": "guest"}, **common)
    assert roster.by_voice["X"].role != "host"
    assert {r.name for r in roster.by_voice.values() if r.role == "host"} == {"Elad Gil"}


def test_llm_host_verdict_anchors_a_no_stated_host_show() -> None:
    """Planet Money-style: the feed states no hosts, so a second narrator the cues cannot anchor is
    labeled a guest. The LLM's "host" verdict (from title/description/intro) seats it (ADR-135)."""
    diar, vtext, ordered = _scripted(
        [
            ("S0", "Hello and welcome to Planet Money. Today, a town with a very strange problem."),
            (
                "S1",
                "That is right. It started when the money simply would not stop arriving there.",
            ),
            ("S0", "We went to find out what a town does with more cash than it can ever spend."),
            (
                "S1",
                "And what we found says something about how all of us really think about money.",
            ),
        ]
    )
    common = dict(
        detected_guests=[],
        known_hosts=[],  # no stated hosts — the empty-pool case
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=["Robert Smith", "Brittany Luce"],
        diarization_provider="tailnet_dgx",
    )
    # control: the second narrator is not a host without the anchor
    control = resolve_speaker_roster(diar, vtext["S0"], **common)
    assert (
        control.by_voice["S1"].role != "host"
    ), "S1 must start as a non-host for the anchor to bite"
    # the LLM anchors S1 as a host (and names it from the closed metadata list)
    roster = resolve_speaker_roster(
        diar,
        vtext["S0"],
        llm_voice_names={"S0": "Robert Smith", "S1": "Brittany Luce"},
        llm_voice_roles={"S0": "host", "S1": "host"},
        **common,
    )
    assert roster.by_voice["S1"].role == "host"
    assert roster.by_voice["S1"].name == "Brittany Luce"


def test_llm_host_anchor_ignores_an_unnamed_voice() -> None:
    """The over-assignment guard (found on the Planet Money pilot). A no-stated-host show is a
    narrated documentary as often as a rotating-host desk show, and the LLM will call field tape
    "host" too. An anonymous voice — no name from any source — is NOT anchored; only a NAMED host
    is, else the vox-pop of a documentary gets crowned (SPEAKER_04/11/19 were)."""
    diar, vtext, ordered = _scripted(
        [
            ("S0", "Hello and welcome to Planet Money. Today, a town with a very strange problem."),
            (
                "VOX",
                "I have lived here thirty years and I have never once seen anything like this.",
            ),
            (
                "S0",
                "We went to find out what a town does with more cash than it can ever spend now.",
            ),
            (
                "VOX",
                "The money just kept coming and coming, and nobody quite knew what to do at all.",
            ),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["S0"],
        detected_guests=[],
        known_hosts=[],
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=[],
        # the LLM over-eagerly calls the anonymous field voice a host; only S0 (named below) is real
        llm_voice_names={"S0": "Robert Smith"},
        llm_voice_roles={"S0": "host", "VOX": "host"},
        diarization_provider="tailnet_dgx",
    )
    assert roster.by_voice["VOX"].role != "host", "an unnamed voice must not be anchored as a host"


def test_llm_guest_verdict_never_unseats_a_self_intro_known_host() -> None:
    """The hard guardrail: a voice that self-introduces as a STATED host is seated at step 1, before
    the LLM verdict is consulted. A wrong "guest" from the model cannot unseat it (ADR-135)."""
    diar, vtext, ordered = _scripted(
        [
            ("H1", "Welcome to the show. I'm Casey Newton, in New York this week."),
            ("H1", "Today we dig into the week in AI, and there is a great deal to get through."),
            ("GUEST", "Thanks for having me. The models are moving genuinely fast this year."),
            ("H1", "Where has it moved the most, would you say, in the last few months or so?"),
        ]
    )
    roster = resolve_speaker_roster(
        diar,
        vtext["H1"],
        detected_guests=[],
        known_hosts=["Casey Newton"],
        voice_texts=vtext,
        ordered_turns=ordered,
        metadata_named=[],
        llm_voice_roles={"H1": "guest"},  # the model is WRONG about the host
        diarization_provider="tailnet_dgx",
    )
    assert roster.by_voice["H1"].role == "host"
    assert roster.by_voice["H1"].name == "Casey Newton"
