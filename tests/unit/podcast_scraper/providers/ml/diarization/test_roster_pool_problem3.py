"""Roster-side rules of problem 3 (host pools), synthetic. One behaviour per test."""

from __future__ import annotations

from typing import Dict, List

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
