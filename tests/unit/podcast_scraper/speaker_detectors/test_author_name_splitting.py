"""Unit tests for multi-author RSS tag splitting (#1652).

A single ``<itunes:author>`` often names the whole cast. Before this, the tag was kept whole,
so ``"Brandon Anderson, RJ Honicky, and Latent.Space"`` reached the roster as ONE host name
that could never match a voice — making the known-hosts fallback inert for every multi-author
feed. That is the fallback that would otherwise have limited #1646's damage on those shows.

The risk being managed here is asymmetric: failing to split loses a fallback, but splitting
badly INVENTS a person and paints their name onto a stranger's voice. Every test below is
written from that asymmetry — when in doubt, produce nothing rather than something wrong.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    detect_hosts_from_feed,
    is_network_or_org_author,
    looks_like_publisher,
    split_author_names,
)

pytestmark = [pytest.mark.unit]


class TestSplitAuthorNames:
    def test_the_real_latent_space_tag(self) -> None:
        """The exact string observed in the damaged corpus."""
        assert split_author_names("Brandon Anderson, RJ Honicky, and Latent.Space") == [
            "Brandon Anderson",
            "RJ Honicky",
            "Latent.Space",
        ]

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Kevin Roose and Casey Newton", ["Kevin Roose", "Casey Newton"]),
            ("Katie Martin & Robert Armstrong", ["Katie Martin", "Robert Armstrong"]),
            ("Sarah Guo; Elad Gil", ["Sarah Guo", "Elad Gil"]),
            ("Ryan Knutson, Jessica Mendoza", ["Ryan Knutson", "Jessica Mendoza"]),
        ],
    )
    def test_common_separators(self, raw: str, expected: list) -> None:
        assert split_author_names(raw) == expected

    def test_a_single_name_is_returned_unchanged(self) -> None:
        assert split_author_names("Lenny Rachitsky") == ["Lenny Rachitsky"]

    def test_empty_input_yields_nothing(self) -> None:
        assert split_author_names("") == []
        assert split_author_names("   ") == []

    @pytest.mark.parametrize("suffix", ["Jr.", "Jr", "Sr.", "III", "PhD", "MD"])
    def test_name_suffixes_are_not_torn_off_into_a_separate_person(self, suffix: str) -> None:
        """A bad split here would mint a person called "Jr."."""
        assert split_author_names(f"Martin Luther King, {suffix}") == [
            f"Martin Luther King, {suffix}"
        ]

    def test_a_name_containing_and_inside_a_word_is_not_split(self) -> None:
        """Word-bounded separator: "Alexander" must survive intact."""
        assert split_author_names("Alexander Anderson") == ["Alexander Anderson"]

    def test_suffix_after_a_multi_author_split_attaches_to_the_right_person(self) -> None:
        assert split_author_names("Ann Lee, Bob Fox, Jr., and Carla Diaz") == [
            "Ann Lee",
            "Bob Fox, Jr.",
            "Carla Diaz",
        ]


class TestSplittingCannotInventHosts:
    """The safety property: an over-eager split must degrade to "no host", never a fake one."""

    @pytest.mark.parametrize("fragment", ["Brandon", "NPR", "Latent.Space", "Vox", "Partners"])
    def test_fragments_that_are_not_person_names_are_rejected_downstream(
        self, fragment: str
    ) -> None:
        assert is_network_or_org_author(fragment) is True

    def test_an_org_style_tag_produces_no_hosts_at_all(self) -> None:
        hosts = detect_hosts_from_feed(
            feed_title="Some Show", feed_description=None, feed_authors=["Jones, Smith & Partners"]
        )
        assert hosts == set()


class TestDetectHostsFromFeedUsesTheSplit:
    def test_multi_author_tag_now_yields_each_person(self) -> None:
        hosts = detect_hosts_from_feed(
            feed_title="Latent Space",
            feed_description=None,
            feed_authors=["Brandon Anderson, RJ Honicky, and Latent.Space"],
        )
        # Both people are recovered; the show name is rejected as an organisation.
        assert hosts == {"Brandon Anderson", "RJ Honicky"}

    def test_single_author_behaviour_is_unchanged(self) -> None:
        hosts = detect_hosts_from_feed(
            feed_title="Lenny's Podcast", feed_description=None, feed_authors=["Lenny Rachitsky"]
        )
        assert hosts == {"Lenny Rachitsky"}

    def test_email_suffix_is_still_stripped_before_splitting(self) -> None:
        hosts = detect_hosts_from_feed(
            feed_title="A Show",
            feed_description=None,
            feed_authors=["Kevin Roose and Casey Newton <podcast@example.com>"],
        )
        assert hosts == {"Kevin Roose", "Casey Newton"}


class TestFirmBrandsAreNotPeople:
    """#1652: ``person:andreessen-horowitz`` was the corpus's top-ranked Person.

    54 episodes, 723 insights, and on one a16z episode it was the ONLY "person" while the
    actual speakers stayed unresolved. Two real-looking tokens with no generic org marker, so
    neither the mononym rule nor ``has_org_markers`` caught it.
    """

    @pytest.mark.parametrize("brand", ["Andreessen Horowitz", "andreessen horowitz", "a16z"])
    def test_firm_brands_are_recognised_as_publishers(self, brand: str) -> None:
        assert looks_like_publisher(brand) is True

    @pytest.mark.parametrize("person", ["Marc Andreessen", "Ben Horowitz", "Chris Dixon"])
    def test_real_people_from_the_same_firm_are_untouched(self, person: str) -> None:
        """Whole-name match only — the first-token check must not fire on "andreessen"."""
        assert looks_like_publisher(person) is False

    def test_a_firm_branded_author_tag_yields_no_host(self) -> None:
        hosts = detect_hosts_from_feed(
            feed_title="The a16z Show", feed_description=None, feed_authors=["Andreessen Horowitz"]
        )
        assert hosts == set()
