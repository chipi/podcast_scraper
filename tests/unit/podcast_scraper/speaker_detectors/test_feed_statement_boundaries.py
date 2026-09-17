"""Where a feed's host statement ENDS — and when an author tag is the publisher (#2075).

Each shape here was measured on production feeds, where it put a non-person on voices as a host:
`Norman Conquest` (49 voices), `Americas Online` (32), `Carnegie India` (6), `Anglo Canadian` (3).
The strings are synthetic; the grammar is the measured one. Across all 55 production feeds the
change alters exactly the 8 wrong ones and none of the other 47.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    detect_hosts_from_feed,
    distinct_self_introductions,
    extract_self_introduced_host,
    has_org_markers,
    hosts_from_feed_statement,
    looks_like_a_person_name,
)

pytestmark = pytest.mark.unit


class TestTheVerbBelongsToItsOwnSubject:
    def test_a_comma_ends_the_names_the_verb_can_reach(self) -> None:
        text = (
            "From the fall of the Western Empire to the Battle of Hastings in 1066, Ada and Ben "
            "bring history to life."
        )
        assert hosts_from_feed_statement("History Hour", text) == set()

    def test_an_organisation_before_a_comma_is_not_a_host(self) -> None:
        text = "At Meridian Policy, our lineup of experts will host discussions on trade."
        assert hosts_from_feed_statement("Policy Talk", text) == set()

    def test_commas_between_the_names_are_still_read(self) -> None:
        text = "Ada Brook, Ben Carver and other nerds at the Daily Ledger explain markets."
        assert hosts_from_feed_statement("Markets", text) == {"Ada Brook", "Ben Carver"}

    def test_the_tail_of_a_longer_proper_noun_is_not_a_name(self) -> None:
        text = "Each month the Council of the Andean Online team brings you the region."
        assert hosts_from_feed_statement("Andes", text) == set()


class TestTheStatementForms:
    def test_a_sentence_ending_on_with_names(self) -> None:
        text = "Take a deep dive into the past with Ada Brook & Ben Carver. New episodes weekly."
        assert hosts_from_feed_statement("History", text) == {"Ada Brook", "Ben Carver"}

    def test_with_names_mid_sentence_is_not_a_statement(self) -> None:
        text = "Conversations with Ada Brook about the economy and more."
        assert hosts_from_feed_statement("Econ", text) == set()

    def test_the_plural_noun_hosts(self) -> None:
        text = "Join hosts Ada Brook in Lagos and Ben Carver in Nairobi for interviews."
        assert "Ada Brook" in hosts_from_feed_statement("Africa Now", text)

    def test_hosted_by_a_descriptor_then_the_name(self) -> None:
        text = "Hosted by economist Ada Brook and business journalist Ben Carver, we explain."
        assert hosts_from_feed_statement("Econ", text) == {"Ada Brook", "Ben Carver"}

    def test_hosted_by_an_appositive(self) -> None:
        text = "The show is hosted by Anglo Canadian transplant to Peru, Ben Carver and friends."
        assert hosts_from_feed_statement("Peru Calling", text) == {"Ben Carver"}

    def test_a_nationality_is_not_the_host(self) -> None:
        assert hosts_from_feed_statement("X", "Hosted by Anglo Canadian writers.") == set()


class TestPresentingVerbs:
    def test_engages_is_what_a_host_does(self) -> None:
        """Conversations with Tyler: "<host> engages today's deepest thinkers" — the only feed of 55
        whose result changes, from nobody to its host."""
        text = "Ada Brook engages today's deepest thinkers in wide-ranging explorations."
        assert hosts_from_feed_statement("Conversations", text) == {"Ada Brook"}


class TestSelfIntroNamesAreWhitespaceClean:
    def test_word_level_segments_do_not_double_the_space(self) -> None:
        text = "Hello and welcome. I'm  Ada   Brook, and this is the show."
        assert extract_self_introduced_host(text) == "Ada Brook"
        assert distinct_self_introductions(text) == ["Ada Brook"]


class TestAFunctionWordIsNeverPartOfAName:
    """An ASR stretch that capitalises every word turns prose into capitalised runs; a closed-class
    function word inside the run is what gives it away (Latin America in Focus, measured)."""

    def test_a_title_cased_run_is_not_a_self_introduction(self) -> None:
        text = "We Hear That You Know I'm Super Willing To Be Aligned To The US To Do Trade Deals"
        assert extract_self_introduced_host(text) is None
        assert distinct_self_introductions(text) == []

    @pytest.mark.parametrize(
        "junk", ["Super Willing To Be", "One Factor That", "México She", "Karin Zesis This"]
    )
    def test_the_measured_junk_is_not_a_person(self, junk: str) -> None:
        assert not looks_like_a_person_name(junk)

    @pytest.mark.parametrize(
        "name", ["Cobus Van Staden", "Ben Carver", "Anh Do", "Mohammed Bin Salman"]
    )
    def test_names_with_particles_and_short_surnames_survive(self, name: str) -> None:
        assert looks_like_a_person_name(name)

    def test_a_real_intro_followed_by_another_intro_is_read(self) -> None:
        text = "I'm Ada Brook. And I'm Ben Carver. Today we talk about bonds."
        assert distinct_self_introductions(text) == ["Ada Brook", "Ben Carver"]


class TestAnAuthorTagThatIsThePublisher:
    @pytest.mark.parametrize("author", ["The Andean Report", "The Global South Project"])
    def test_no_person_is_the_x(self, author: str) -> None:
        assert detect_hosts_from_feed("Show", "A weekly show.", [author]) == set()

    def test_the_feed_uses_it_as_a_place(self) -> None:
        desc = "At Meridian Policy, our experts discuss the economy."
        assert detect_hosts_from_feed("Policy Talk", desc, ["Meridian Policy"]) == set()

    def test_a_podcast_by_a_person_still_credits_the_person(self) -> None:
        desc = "A podcast by Ada Brook about books."
        assert detect_hosts_from_feed("Books", desc, ["Ada Brook"]) == {"Ada Brook"}

    def test_a_broadcast_brand_suffix(self) -> None:
        assert has_org_markers("Andes Plus")
        assert detect_hosts_from_feed("Biz", "Business news.", ["Andes Plus"]) == set()
