"""A voice that PRESENTS the episode on its own words is a host, whoever the feed's pool names.

Each test names one behaviour of the presenter-evidence rules in the roster: the show's branded
intro, the introduction of the person the episode states, the co-presenter formula, the guest host
the episode names, and the invariants that keep a guest (or an ad, or a promo) from being seated.

All fixtures are synthetic (never-commit-real-episodes): invented shows, hosts and guests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _copresenter_pair_voices,
    _guest_hosts_named,
    _introduces_stated_person,
    _one_name_per_person,
    _presenter_voices_by_evidence,
    resolve_speaker_roster,
    SpeakerRole,
    VoiceCleaning,
)

pytestmark = pytest.mark.unit

Turn = Tuple[str, str, float]

FILLER = "The tide tables changed again after the storm."


def _diar(segs: List[Tuple[str, float, float]], num_speakers: int) -> DiarizationResult:
    return DiarizationResult(
        segments=[DiarizationSegment(start=s, end=e, speaker=spk) for spk, s, e in segs],
        num_speakers=num_speakers,
    )


def _roster(
    turns: Sequence[Turn],
    *,
    ad: Sequence[str] = (),
    **kwargs,
):
    """Resolve a roster from ``(voice, text, seconds)`` turns laid end to end."""
    segs: List[DiarizationSegment] = []
    chunks: Dict[str, List[str]] = {}
    ordered: List[Tuple[str, str]] = []
    t = 0.0
    for spk, text, dur in turns:
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=spk))
        t += dur
        chunks.setdefault(spk, []).append(" " + text)
        ordered.append((spk, " " + text))
    voice_texts = {v: " ".join(c) for v, c in chunks.items()}
    cleaning: Optional[VoiceCleaning] = None
    if ad:
        cleaning = VoiceCleaning(
            ad=frozenset(ad),
            cameo=frozenset(),
            commercial=frozenset(),
            real=frozenset(v for v in voice_texts if v not in ad),
        )
    return resolve_speaker_roster(
        DiarizationResult(segments=segs, num_speakers=len(voice_texts)),
        " ".join(x for _, x in ordered),
        voice_texts=voice_texts,
        ordered_turns=ordered,
        cleaning=cleaning,
        **kwargs,
    )


def _host_and_name(roster, voice: str) -> Tuple[str, str]:
    r = roster.by_voice[voice]
    return r.name, r.role


# --- _introduces_stated_person -------------------------------------------------------------------


def test_a_full_name_before_the_on_the_show_tail_is_an_introduction() -> None:
    """ "<Full Name> is on the show today" introduces the stated person."""
    assert _introduces_stated_person("Ada Quill is on the show today.", ["Ada Quill"])


def test_a_given_name_before_welcome_back_is_an_introduction() -> None:
    """ "<Given name>, welcome back" greets the stated person; the tail is the evidence."""
    assert _introduces_stated_person("Garrett, welcome back.", ["Garrett Bern"])


def test_a_given_name_before_thank_you_for_being_on_the_show_is_an_introduction() -> None:
    """ "Hi, <Given name>. Thank you for being on the show." greets the stated person."""
    text = "Hi, Anna. Thank you for being on the show."
    assert _introduces_stated_person(text, ["Anna North"])


def test_my_guest_today_is_followed_by_the_full_name_is_an_introduction() -> None:
    """ "My guest today is <Full Name>" introduces the stated person."""
    assert _introduces_stated_person("My guest today is Ada Quill.", ["Ada Quill"])


def test_im_here_with_my_colleague_is_not_a_presenter_introduction() -> None:
    """A guest says "I'm here with my colleague" of a fellow guest, so it proves nothing."""
    assert not _introduces_stated_person("I'm here with my colleague Jane Doe", ["Jane Doe"])


def test_thanks_for_having_me_is_not_a_presenter_introduction() -> None:
    """A guest thanking the host by name does not introduce anyone."""
    assert not _introduces_stated_person("Thanks for having me, Garrett.", ["Garrett Bern"])


def test_a_given_name_alone_after_a_presenter_cue_is_not_enough() -> None:
    """After "my guest today is" only the FULL name counts; a bare given name does not."""
    assert not _introduces_stated_person("My guest today is Garrett.", ["Garrett Bern"])


def test_an_unstated_person_is_not_introduced() -> None:
    """Introducing somebody the episode does not state is not evidence."""
    assert not _introduces_stated_person("Ada Quill is on the show today.", ["Garrett Bern"])


# --- _copresenter_pair_voices --------------------------------------------------------------------

PAIR_TURNS = [
    ("V1", " I'm Julie Beckford, a staff writer at The Lantern."),
    ("V2", " And I'm Natalie Brennan, a producer at The Lantern."),
]
PAIR_NAMES = {"V1": "Julie Beckford", "V2": "Natalie Brennan"}


def test_two_voices_introducing_themselves_in_consecutive_turns_are_copresenters() -> None:
    """ "I'm X" then "And I'm Y" from two named voices makes each the other's partner."""
    pairs = _copresenter_pair_voices(PAIR_TURNS, PAIR_NAMES, set())
    assert pairs == {"V1": {"V2"}, "V2": {"V1"}}


def test_the_same_two_lines_said_by_one_voice_are_not_a_pair() -> None:
    """A single voice saying both introductions is one cluster, not two presenters."""
    one_voice = [("V1", PAIR_TURNS[0][1]), ("V1", PAIR_TURNS[1][1])]
    assert _copresenter_pair_voices(one_voice, PAIR_NAMES, set()) == {}


def test_an_ad_voice_is_never_half_of_a_pair() -> None:
    """A promo's "And I'm Y" from an ad voice does not pair with the real voice."""
    assert _copresenter_pair_voices(PAIR_TURNS, PAIR_NAMES, {"V2"}) == {}


def test_the_second_turn_must_open_with_and() -> None:
    """Without the conjunction the second self-introduction is not the two-host formula."""
    turns = [PAIR_TURNS[0], ("V2", " I'm Natalie Brennan, a producer at The Lantern.")]
    assert _copresenter_pair_voices(turns, PAIR_NAMES, set()) == {}


def test_a_voice_without_its_own_name_is_not_a_pair_member() -> None:
    """Both halves must already carry a name; an unnamed voice is not seated by the formula."""
    assert _copresenter_pair_voices(PAIR_TURNS, {"V1": "Julie Beckford"}, set()) == {}


# --- _presenter_voices_by_evidence ---------------------------------------------------------------

SHOW = "Lantern Hour"


def _evidence(
    segs: Sequence[Tuple[str, float, float]],
    texts: Dict[str, str],
    *,
    stated: Sequence[str] = (),
    intros: Optional[Dict[str, str]] = None,
    feed_title: Optional[str] = SHOW,
):
    return _presenter_voices_by_evidence(
        _diar(list(segs), len(texts)), texts, set(), feed_title, list(stated), intros or {}
    )


def test_a_voice_presenting_the_show_by_name_is_branded() -> None:
    """A minority voice saying "you're listening to <show>" presents the episode."""
    segs = [("P", 0, 40), ("G", 40, 100)]
    branded, introducers = _evidence(segs, {"P": "You're listening to Lantern Hour.", "G": FILLER})
    assert branded == {"P"} and introducers == set()


def test_a_voice_introducing_the_stated_guest_is_an_introducer() -> None:
    """A minority voice saying "<Guest> is on the show today" presents the episode."""
    segs = [("P", 0, 40), ("G", 40, 100)]
    texts = {"P": "Ada Quill is on the show today.", "G": FILLER}
    branded, introducers = _evidence(segs, texts, stated=["Ada Quill"])
    assert branded == set() and introducers == {"P"}


def test_a_dominant_voice_carrying_the_show_sting_is_not_branded() -> None:
    """The mid-roll "you're listening to <show>" bled into the guest's cluster must not brand it."""
    segs = [("P", 0, 30), ("G", 30, 100)]
    guest = f"{FILLER} You're listening to Lantern Hour. {FILLER}"
    branded, introducers = _evidence(segs, {"P": FILLER, "G": guest})
    assert branded == set() and introducers == set()


def test_a_dominant_voice_greeting_the_stated_guest_is_not_an_introducer() -> None:
    """A voice with half the talk or more is left to the positional steps, introduction or not."""
    segs = [("P", 0, 50), ("G", 50, 100)]
    texts = {"P": "Ada Quill is on the show today.", "G": FILLER}
    branded, introducers = _evidence(segs, texts, stated=["Ada Quill"])
    assert branded == set() and introducers == set()


def test_a_cluster_with_two_peoples_self_introductions_is_not_branded() -> None:
    """ "I'm Kevin... I'm Casey... this is <show>" is a merged cold open, not a presenter."""
    segs = [("M", 0, 40), ("G", 40, 100)]
    merged = "I'm Kevin Fairweather. I'm Casey Lindqvist. And this is Lantern Hour."
    branded, introducers = _evidence(segs, {"M": merged, "G": FILLER})
    assert branded == set() and introducers == set()


def test_a_voice_is_not_the_introducer_of_the_person_it_is_itself_named_as() -> None:
    """A host greeting bled into the guest's cluster, which carries the guest's own name."""
    segs = [("P", 0, 40), ("G", 40, 100)]
    texts = {"P": "Ada Quill is on the show today.", "G": FILLER}
    _, introducers = _evidence(segs, texts, stated=["Ada Quill"], intros={"P": "Ada Quill"})
    assert introducers == set()


def test_another_shows_intro_does_not_brand_a_voice() -> None:
    """Presenting a DIFFERENT show by name is not presenting this one."""
    segs = [("P", 0, 40), ("G", 40, 100)]
    branded, _ = _evidence(segs, {"P": "You're listening to Quiet Harbour.", "G": FILLER})
    assert branded == set()


# --- _guest_hosts_named --------------------------------------------------------------------------


def test_a_guest_host_named_in_the_description_is_read() -> None:
    """ "join the guest host Max Read and an array of reporters" names the episode's host."""
    text = "This week, join the guest host Max Read and an array of reporters."
    assert _guest_hosts_named(text) == ["Max Read"]


def test_a_person_sitting_in_is_a_guest_host() -> None:
    """ "<Name> is sitting in" names the person presenting instead of the feed's host."""
    assert _guest_hosts_named("Noah Smith is sitting in for the week.") == ["Noah Smith"]


def test_an_organisation_is_not_a_guest_host() -> None:
    """A network or organisation after the cue is not a person and is refused."""
    assert _guest_hosts_named("Join the guest host Public Radio Network today.") == []


def test_no_episode_text_names_no_guest_host() -> None:
    """Nothing to read, nobody named."""
    assert _guest_hosts_named(None) == []
    assert _guest_hosts_named("") == []


# --- end to end ----------------------------------------------------------------------------------


def test_host_introducing_the_guest_keeps_the_seat_despite_a_bled_guest_phrase() -> None:
    """How I Write: the host opens "<Guest> is on the show today"; the guest's "great to be here"
    bled onto the end of the host's cluster. The host is still the host."""
    turns: List[Turn] = [
        ("GUEST", f"A clip from later: {FILLER}", 20),
        ("HOST", "Welcome back, I'm Dov. Ada Quill is on the show today. She writes novels.", 40),
        ("GUEST", f"I started in a barn. {FILLER}", 340),
        ("HOST", "Thank you. It's great to be here.", 10),
    ]
    r = _roster(turns, known_hosts=["Dov Pell"], detected_guests=["Ada Quill"])
    assert _host_and_name(r, "HOST") == ("Dov Pell", "host")
    assert _host_and_name(r, "GUEST") == ("Ada Quill", "guest")


def test_a_voice_branded_with_the_show_outside_the_pool_takes_the_host_seat() -> None:
    """The feed names Shalma Wegsman, but the voice saying "You're listening to <show>. My name is
    Dan Hooper" presents; the opener is not painted with the absent pool name."""
    turns: List[Turn] = [
        ("OPEN", "Coming up, the sky is stranger than you think.", 20),
        (
            "PRES",
            "You're listening to Why This Universe. My name is Dan Hooper. Today, dark matter.",
            140,
        ),
        ("GUEST", f"Thanks for having me. {FILLER}", 200),
        ("PRES", "That is all for today. Thanks for listening.", 20),
    ]
    r = _roster(turns, known_hosts=["Shalma Wegsman"], feed_title="Why This Universe?")
    assert _host_and_name(r, "PRES") == ("Dan Hooper", "host")
    assert r.by_voice["OPEN"].name != "Shalma Wegsman"
    assert r.by_voice["GUEST"].role != "host"


def test_copresenters_on_an_empty_pool_both_take_host_seats() -> None:
    """ "I'm Julie..." then "And I'm Natalie..." seats both as hosts though the feed names none; the
    guest who says "thanks for having me" is not seated."""
    turns: List[Turn] = [
        ("V1", "I'm Julie Beckford, a staff writer at The Lantern.", 30),
        ("V2", "And I'm Natalie Brennan, a producer at The Lantern.", 30),
        ("GUEST", f"Thanks for having me. {FILLER}", 240),
    ]
    r = _roster(turns)
    assert _host_and_name(r, "V1") == ("Julie Beckford", "host")
    assert _host_and_name(r, "V2") == ("Natalie Brennan", "host")
    assert r.by_voice["GUEST"].role != "host"


def test_a_guest_host_the_episode_names_is_seated_over_the_feed_hosts() -> None:
    """The description states the guest host Max Read; the voice saying "I'm Max Reed" takes his
    stated spelling and the host seat, and the other voice does not get a pool host's name."""
    turns: List[Turn] = [
        ("MAX", "Welcome to the show. I'm Max Reed. This week we look at the harbour.", 120),
        ("OTHER", f"Thanks for having me. {FILLER}", 240),
    ]
    r = _roster(
        turns,
        known_hosts=["Casey Newton", "Kevin Roose"],
        episode_text="This week, join the guest host Max Read and an array of reporters.",
    )
    assert _host_and_name(r, "MAX") == ("Max Read", "host")
    assert r.by_voice["OTHER"].name not in {"Casey Newton", "Kevin Roose"}


def test_a_pool_entry_that_is_the_show_name_is_not_a_person_and_the_presenter_is_seated() -> None:
    """The pool says "Africa Tech Summit" (the show itself). Nobody is named that, the voice that
    presents the show by name and says "My name is Nixon Kanali" is the one host, and the pool's
    count of one still caps the seats."""
    turns: List[Turn] = [
        ("HOUSE", "Brought to you by the founders network.", 15),
        (
            "PRES",
            "Welcome to the Africa Tech Summit podcast. My name is Nixon Kanali. Today, payments.",
            160,
        ),
        ("GUEST", f"Thanks for having me. {FILLER}", 225),
    ]
    r = _roster(
        turns,
        known_hosts=["Africa Tech Summit"],
        feed_title="Africa Tech Summit Podcast",
    )
    assert _host_and_name(r, "PRES") == ("Nixon Kanali", "host")
    assert all(role.name != "Africa Tech Summit" for role in r.by_voice.values())
    assert [v for v, role in r.by_voice.items() if role.role == "host"] == ["PRES"]


def test_a_guest_who_self_introduces_stays_a_guest_and_the_host_keeps_the_pool_name() -> None:
    """A real guest says "thanks for having me" and "I'm Olaf Storbeck" while the host says "My
    guest today is Olaf Storbeck": the guest is not promoted, the host keeps the pool name."""
    turns: List[Turn] = [
        ("HOST", "Hello and welcome. My guest today is Olaf Storbeck. Olaf, welcome.", 60),
        ("GUEST", f"Thanks for having me. I'm Olaf Storbeck. {FILLER}", 300),
    ]
    r = _roster(turns, known_hosts=["Katie Martin"], detected_guests=["Olaf Storbeck"])
    assert _host_and_name(r, "HOST") == ("Katie Martin", "host")
    assert _host_and_name(r, "GUEST") == ("Olaf Storbeck", "guest")


def test_an_ad_voice_introduction_does_not_name_the_real_guest() -> None:
    """A promo for another show inside the episode ("I'm joined by Andy Baraghani") is read by an ad
    voice; the real guest who speaks next is not given that name."""
    turns: List[Turn] = [
        ("AD", "On this week's episode, I'm joined by Andy Baraghani.", 15),
        ("HOST", "Welcome to the show. I'm Dov Pell. Today we talk about harbours.", 60),
        ("GUEST", f"Thanks for having me. {FILLER}", 300),
    ]
    r = _roster(turns, known_hosts=["Dov Pell"], ad=["AD"])
    assert r.by_voice["GUEST"].name != "Andy Baraghani"
    assert not any(role.name == "Andy Baraghani" for role in r.by_voice.values())


def test_a_two_host_promo_inside_an_episode_is_not_seated_as_hosts() -> None:
    """Two slivers saying "I'm Kara Swisher." / "And I'm Scott Galloway." are another show's promo
    pairing with each other; neither takes a host seat."""
    turns: List[Turn] = [
        ("HOST", "Welcome to the show. I'm Dov Pell. Today we talk about harbours.", 100),
        ("GUEST", f"Thanks for having me. {FILLER}", 200),
        ("PROMO1", "I'm Kara Swisher.", 3),
        ("PROMO2", "And I'm Scott Galloway.", 3),
        ("GUEST", f"As I was saying. {FILLER}", 200),
    ]
    r = _roster(turns)
    assert r.by_voice["PROMO1"].role != "host"
    assert r.by_voice["PROMO2"].role != "host"


def test_an_evidence_host_takes_the_episodes_stated_spelling_of_the_name() -> None:
    """The branded voice says "I'm Imani Moiz"; the episode states "Imani Moise", and a second
    cluster says "I'm Imani Moise": both voices end as "Imani Moise", host."""
    turns: List[Turn] = [
        ("H1", "Welcome to Lantern Hour. I'm Imani Moiz. Today we talk about harbours.", 60),
        ("GUEST", f"Thanks for having me. {FILLER}", 300),
        ("H2", "I'm Imani Moise. That is all for today.", 40),
    ]
    r = _roster(turns, feed_title="Lantern Hour", metadata_named=["Imani Moise"])
    assert _host_and_name(r, "H1") == ("Imani Moise", "host")
    assert _host_and_name(r, "H2") == ("Imani Moise", "host")


# --- _one_name_per_person with evidence_hosts ----------------------------------------------------


def _split_person() -> Dict[str, SpeakerRole]:
    return {
        "V1": SpeakerRole(name="Imani Moise", role="host", source="self_intro", named=True),
        "V2": SpeakerRole(name="Imani Moise", role="guest", source="self_intro", named=True),
    }


def test_two_voices_of_one_person_are_both_host_when_one_seat_is_evidence() -> None:
    """The person made the host claim on its own evidence, so every voice of them is a host."""
    out = _one_name_per_person(
        _split_person(), {"V1": 100.0, "V2": 50.0}, [], [], evidence_hosts={"V1"}
    )
    assert out["V1"].role == "host" and out["V2"].role == "host"


def test_two_voices_of_one_person_are_guests_when_the_host_seat_is_positional() -> None:
    """Without evidence and without being a known host, the feed made no host claim: guest."""
    out = _one_name_per_person(_split_person(), {"V1": 100.0, "V2": 50.0}, [], [])
    assert out["V1"].role == "guest" and out["V2"].role == "guest"


def test_an_unnamed_opening_host_keeps_the_pool_name_despite_a_bled_guest_phrase() -> None:
    """The host opens with "<Guest> is on the show today" and never says their own name; the
    guest's "great to be here" bled onto the end of the host's cluster. The pool name still goes
    on the host, and the guest is not given it."""
    turns: List[Turn] = [
        ("HOST", "Welcome back. Ada Quill is on the show today. She writes novels.", 40),
        ("GUEST", f"I started in a barn. {FILLER}", 350),
        ("HOST", "Thank you. It's great to be here.", 10),
    ]
    r = _roster(turns, known_hosts=["Dov Pell"], detected_guests=["Ada Quill"])
    assert _host_and_name(r, "HOST") == ("Dov Pell", "host")
    assert _host_and_name(r, "GUEST") == ("Ada Quill", "guest")
