"""Roster-side rules of problem 3 (host pools), synthetic. One behaviour per test."""

from __future__ import annotations

from typing import Dict, List

import pytest

from podcast_scraper.providers.ml.diarization.roster import (
    _better_presenter_elsewhere,
    _is_show_mononym,
    _merged_host_cluster_owner,
    _name_a_talkative_host_once_the_guests_are_placed,
    _owns_the_conversation,
    _snap_near_identical_host,
    _SOLO_EPISODE,
    SpeakerRole,
)

HOSTS = ["Michael Stevens", "Hannah Fry"]


def test_a_merged_two_host_cluster_that_addresses_one_host_is_the_other() -> None:
    text = (
        "Hello, and welcome to The Rest Is Science. I'm Michael Stevens. And I'm Hannah Fry. "
        "And today I have brought you a gift, Hannah. There you go."
    )
    assert _merged_host_cluster_owner(text, HOSTS) == "Michael Stevens"


def test_a_merged_cluster_that_addresses_both_hosts_names_nobody() -> None:
    text = "I'm Michael Stevens. And I'm Hannah Fry. Hannah, look. Michael, no. Hannah, yes."
    assert _merged_host_cluster_owner(text, HOSTS) is None


def test_the_welcome_with_me_idiom_names_its_speaker() -> None:
    text = (
        "Welcome to The Rest Is Politics Leading with me, Alistair Campbell. And me, Rory Stewart."
    )
    assert _merged_host_cluster_owner(text, ["Alastair Campbell", "Rory Stewart"]) == (
        "Alastair Campbell"
    )


def test_a_cluster_with_one_intro_is_not_a_merge() -> None:
    assert _merged_host_cluster_owner("I'm Michael Stevens. Hannah, look at this.", HOSTS) is None


def test_the_interviewee_owns_a_cold_open_interview() -> None:
    assert _owns_the_conversation("G", {"G": 0.75, "H": 0.25}, seats=1)


def test_the_chattier_co_host_does_not_own_a_two_host_show() -> None:
    assert not _owns_the_conversation("R", {"R": 0.58, "K": 0.42}, seats=2)


def test_a_narrator_among_many_interviewees_does_not_own_it() -> None:
    # The Flip (2026-10-11): the narrator-host at 57% among three interviewees.
    shares = {"N": 0.57, "A": 0.22, "B": 0.15, "C": 0.06}
    assert not _owns_the_conversation("N", shares, seats=1)


def test_the_cold_open_guest_still_owns_it_among_many_voices() -> None:
    # The MLST case the rule exists for keeps it: the interviewee at 72%+, many voices or not.
    shares = {"G": 0.72, "H": 0.15, "C": 0.13}
    assert _owns_the_conversation("G", shares, seats=1)


def test_the_guest_of_a_two_host_show_still_owns_it() -> None:
    # The Rest Is Politics, Yanis Varoufakis at 67% with two hosts: the third voice IS the guest.
    shares = {"Y": 0.67, "A": 0.18, "R": 0.15}
    assert _owns_the_conversation("Y", shares, seats=2)


def _talkative_host(guest_placed: bool) -> SpeakerRole:
    by_voice = {
        "H": SpeakerRole(name="H", role="host", named=False, source="raw"),
        "G": (
            SpeakerRole(name="Liam", role="guest", named=True, source="metadata")
            if guest_placed
            else SpeakerRole(name="G", role="guest", named=False, source="raw")
        ),
    }

    def without_ownership(seats: List[str], used: set) -> Dict[str, SpeakerRole]:
        return {"H": SpeakerRole(name="Maya", role="host", named=True, source="known_hosts")}

    _name_a_talkative_host_once_the_guests_are_placed(
        by_voice, ["H"], ["Liam"], set(), without_ownership
    )
    return by_voice["H"]


def test_a_talkative_host_is_named_once_the_stated_guest_is_on_another_voice() -> None:
    assert _talkative_host(guest_placed=True).name == "Maya"


def test_a_talkative_seat_stays_unnamed_while_the_stated_guest_is_unplaced() -> None:
    # The seat may BE the guest (MLST, Lenny's Podcast, Latent Space cold opens).
    assert not _talkative_host(guest_placed=False).named


def test_a_solo_episode_is_recognised() -> None:
    assert _SOLO_EPISODE.search("In Dan's first solo episode, he takes us through theories.")
    assert not _SOLO_EPISODE.search("A solo artist joins us.")


def test_the_shows_first_word_is_not_a_name() -> None:
    assert _is_show_mononym("Trivium", "The Trivium China Podcast", [])


def test_a_host_who_goes_by_the_shows_first_word_keeps_it() -> None:
    assert not _is_show_mononym("Dwarkesh", "Dwarkesh Podcast", ["Dwarkesh Patel"])
    assert not _is_show_mononym("Lenny", "Lenny's Podcast: Product", ["Lenny Rachitsky"])


TEXTS = {
    "S0": "You're seeing someone else doing well and then you think, okay, I gotta do that.",
    "S1": "That's Babs Ogundeyi, the CEO of Kuda. So I want to start with you.",
}


def test_a_position_only_seat_is_outranked_by_an_introducer() -> None:
    assert _better_presenter_elsewhere(
        "S0",
        "Justin Norman",
        conv_host_voices=set(),
        host_evidence_voices=set(),
        introducer_voices={"S1"},
        voice_texts=TEXTS,
    )


def test_a_position_only_seat_is_outranked_by_the_shows_own_intro() -> None:
    texts = {
        "S0": "Designers using Claude get better results.",
        "S3": "Anish Acharya speaks with John Maeda.",
    }
    assert _better_presenter_elsewhere(
        "S0",
        "Anish Acharya",
        conv_host_voices=set(),
        host_evidence_voices=set(),
        introducer_voices=set(),
        voice_texts=texts,
    )


def test_a_seat_with_a_host_act_keeps_its_forced_name() -> None:
    assert not _better_presenter_elsewhere(
        "S0",
        "Justin Norman",
        conv_host_voices={"S0"},
        host_evidence_voices=set(),
        introducer_voices={"S1"},
        voice_texts=TEXTS,
    )


def test_a_respelt_guest_is_the_stated_guest() -> None:
    assert (
        _snap_near_identical_host("Christopher Moore", ["Cristopher Moore"]) == "Cristopher Moore"
    )


def test_a_co_host_naming_the_other_host_does_not_outrank_that_hosts_seat() -> None:
    texts = {
        "A": "Hello, I'm Katie Martin, joined as usual by mister Rob Armstrong.",
        "B": "Hi Katie.",
    }
    assert not _better_presenter_elsewhere(
        "B",
        "Robert Armstrong",
        conv_host_voices=set(),
        host_evidence_voices=set(),
        introducer_voices=set(),
        voice_texts=texts,
        host_voices={"A", "B"},
    )


def test_as_guest_host_names_the_guest_host() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _guest_hosts_named

    assert _guest_hosts_named(
        "This week on Sinica, I'm delighted to have Iza Ding as guest host."
    ) == ["Iza Ding"]


def test_being_invited_is_the_visitors_reply() -> None:
    # F4c (prod 2026-10-11): read only by the no-pool LLM-host step, never as a general guest act.
    from podcast_scraper.providers.ml.diarization.roster import _INVITED_REPLY

    for line in (
        "Thank you for inviting me. It's a pleasure.",
        "Well, thank you for the invitation and welcome to Moldova.",
        "Thanks so much for the invite!",
    ):
        assert _INVITED_REPLY.search(line), line
    assert not _INVITED_REPLY.search("We sent the invitation to him last week.")


def test_a_declared_guest_that_is_a_host_in_the_asr_spelling_is_not_a_guest() -> None:
    # F5 (Latent Space, 2026-10-11): "I'm joined by RJ Hanaki" is the co-host RJ Honicky.
    from podcast_scraper.providers.ml.diarization.roster import _declared_guests_not_hosts

    hosts = ["RJ Honicky"]
    lower = {"rj honicky"}
    assert _declared_guests_not_hosts(["RJ Hanaki", "Carina Hong"], hosts, lower) == ["Carina Hong"]
    # A guest who merely shares the host's first name stays a guest.
    assert _declared_guests_not_hosts(["RJ Smith"], hosts, lower) == ["RJ Smith"]


def test_the_model_never_names_a_voice_after_the_person_it_introduces() -> None:
    # F4a (prod 2026-10-11): the host's voice carried the guest's name on 10 served episodes.
    from podcast_scraper.providers.ml.diarization.naming_trace import NamingTrace
    from podcast_scraper.providers.ml.diarization.roster import _without_names_the_voice_introduces

    texts = {
        "H": "Hi, listeners. Welcome back to No Priors. Today, I'm here with Eric Alabest.",
        "G": "Thanks for having me.",
    }
    names = {"H": "Erik Allebest", "G": "Sarah Guo"}
    assert _without_names_the_voice_introduces(names, texts, NamingTrace()) == {"G": "Sarah Guo"}


def test_a_small_sponsor_read_never_takes_the_introduced_name() -> None:
    # F12 (Lenny's Podcast, prod 2026-10-11): the ad after "my guest today is X" took X's name.
    from podcast_scraper.providers.ml.diarization.roster import _is_a_small_sponsor_read

    guest = "We grew fast because we listened to users. " * 60
    ad = "Today's episode is brought to you by DX, the developer intelligence platform."
    assert _is_a_small_sponsor_read("AD", {"AD": ad, "G": guest})
    # A guest who mentions a sponsor once is not a sponsor read: the voice is not small.
    assert not _is_a_small_sponsor_read("G", {"G": guest + ad, "H": "Welcome."})


def test_the_titled_guest_is_declared_only_when_the_metadata_states_them() -> None:
    # F2 (prod 2026-10-11): ~40 episodes name their guest only in the title.
    from podcast_scraper.providers.ml.diarization.roster import _title_subject_guest

    text = "Neil Kinnock: Trump, Thatcher, and Why Labour Won. The former leader joins us."
    hosts = ["Alastair Campbell", "Rory Stewart"]
    assert _title_subject_guest(text, ["Neil Kinnock"], hosts, True) == "Neil Kinnock"
    assert _title_subject_guest(text, [], hosts, True) is None  # the metadata does not state him
    assert _title_subject_guest(text, ["Neil Kinnock"], hosts, False) is None  # no host seated
    assert (
        _title_subject_guest("Rory Stewart: on leadership", ["Rory Stewart"], hosts, True) is None
    )


def test_a_place_the_show_or_a_title_phrase_is_not_a_person_heard() -> None:
    # F12 (prod 2026-10-11): "Planet Money", "New York City", "Consumer AI Ads" named voices.
    from podcast_scraper.providers.ml.diarization.roster import _not_a_person_heard

    assert _not_a_person_heard("Planet Money", "Planet Money")
    assert _not_a_person_heard("New York City", None)
    assert _not_a_person_heard("Consumer AI Ads", None)
    assert not _not_a_person_heard("Sapna Maheshwari", None)
    assert not _not_a_person_heard("Anna Money", None)


@pytest.mark.parametrize(
    ("name", "text", "hosts", "expected"),
    [
        (
            "Alexander Carpi",
            "news roundup. My name is Alexander Carpi, and I'm joined by Adam Reichardt",
            ["Alexandra Karppi", "Adam Reichardt"],
            "Alexandra Karppi",
        ),
        (
            "Brennan Leung",
            "Welcome to Analyze Podcast, the premier podcast. I'm Brennan Leung and today",
            ["Bernard Leong"],
            "Bernard Leong",
        ),
        (
            "Jill Weisenthal",
            "Let's leave it there. And I'm Jill Weisenthal. You can follow me at the stalwart.",
            ["Joe Weisenthal", "Tracy Alloway"],
            "Joe Weisenthal",
        ),
        # Outside the host's formula the strict rules stand: different people stay apart.
        (
            "James Holland",
            "Thanks for having me. I'm James Holland, I write about the war.",
            ["Tom Holland", "Dominic Sandbrook"],
            "James Holland",
        ),
        (
            "Kevin Ross",
            "Hi, I'm Kevin Ross from Acme.",
            ["Kevin Roose", "Casey Newton"],
            "Kevin Ross",
        ),
        (
            "Anna Weisenthal",
            "Thanks for having me, Joe. I'm Anna Weisenthal.",
            ["Joe Weisenthal"],
            "Anna Weisenthal",
        ),
    ],
)
def test_a_self_intro_inside_the_host_formula_is_that_host(name, text, hosts, expected) -> None:
    # F5 (prod 2026-10-11): hosts published as guests of their own shows under the ASR's spelling.
    from podcast_scraper.providers.ml.diarization.roster import _snap_hosts_by_their_formula

    out = {"A": name}
    _snap_hosts_by_their_formula(out, {"A": text}, hosts)
    assert out["A"] == expected


def test_the_formula_never_takes_a_host_another_voice_already_carries() -> None:
    from podcast_scraper.providers.ml.diarization.roster import _snap_hosts_by_their_formula

    out = {"A": "Jill Weisenthal", "B": "Joe Weisenthal"}
    texts = {"A": "And I'm Jill Weisenthal. You can follow me at the stalwart.", "B": "I'm Joe."}
    _snap_hosts_by_their_formula(out, texts, ["Joe Weisenthal", "Tracy Alloway"])
    assert out["A"] == "Jill Weisenthal"


def test_a_promo_sliver_or_a_short_unlike_surname_is_not_snapped() -> None:
    # Prod replay 2026-10-11: "And I'm James Holland from Goalhangers" was a 0.5% cross-promo ad;
    # "Michael Wong" is not the host Zhao Yang (two edits on four-letter surnames).
    from podcast_scraper.providers.ml.diarization.roster import _snap_hosts_by_their_formula

    out = {"A": "James Holland"}
    texts = {"A": "And I'm James Holland from Goalhangers. We have ways of making you talk."}
    _snap_hosts_by_their_formula(out, texts, ["Tom Holland", "Dominic Sandbrook"], {"A": 0.005})
    assert out["A"] == "James Holland"
    out = {"A": "Michael Wong"}
    texts = {"A": "Welcome to Biz Talk. I'm Michael Wong and today"}
    _snap_hosts_by_their_formula(out, texts, ["Zhao Yang"], {"A": 0.4})
    assert out["A"] == "Michael Wong"
