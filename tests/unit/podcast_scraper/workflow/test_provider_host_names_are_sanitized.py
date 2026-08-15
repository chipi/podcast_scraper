"""A provider's host answer goes through the same filter as the deterministic one (#1652).

Found in the #1657 acceptance run, not by a test. *The a16z Show* has no "Hosted by ..." blurb
and org-only author tags (``Andreessen Horowitz``, ``a16z``), so the deterministic parser
correctly returned nothing and the code fell through to ``speaker_detector.detect_hosts``. The
LLM answered with one string naming three people:

    "Erik Torenberg, Ben Horowitz, Travis Kalanick"

Nothing split it, so it anchored the roster and reached the graph as
``person:erik-torenberg-ben-horowitz-travis-kalanick`` — a person who does not exist.

That is worse than finding no host. A composite can never match a diarized voice (the roster
compares per name), so it disables the known-hosts anchor it was supposed to provide, while
adding a fake entity that cross-episode queries will join on.
"""

from __future__ import annotations

from typing import Any, Optional, Set

import pytest

from podcast_scraper.workflow.stages import processing

pytestmark = [pytest.mark.unit]


class _Feed:
    def __init__(self, title: str = "The a16z Show", description: str = "", authors=None) -> None:
        self.title = title
        self.description = description
        self.authors = authors or []


class _Detector:
    """Stands in for the LLM provider's ``detect_hosts``."""

    def __init__(self, returns: Set[str]) -> None:
        self._returns = returns

    def detect_hosts(self, **_kw: Any) -> Set[str]:
        return set(self._returns)


def _hosts(returns: Set[str], feed: Optional[_Feed] = None) -> Set[str]:
    return processing._detect_hosts_from_feed(feed or _Feed(), _Detector(returns))


class TestCompositeNamesAreSplit:
    def test_the_a16z_case(self) -> None:
        """The exact string observed in the acceptance run."""
        assert _hosts({"Erik Torenberg, Ben Horowitz, Travis Kalanick"}) == {
            "Erik Torenberg",
            "Ben Horowitz",
            "Travis Kalanick",
        }

    def test_no_composite_person_survives(self) -> None:
        out = _hosts({"Erik Torenberg, Ben Horowitz, Travis Kalanick"})
        assert not any("," in n for n in out), f"a composite reached the roster: {out}"

    def test_and_separated_names_split_too(self) -> None:
        assert _hosts({"Kevin Roose and Casey Newton"}) == {"Kevin Roose", "Casey Newton"}

    def test_a_single_name_is_untouched(self) -> None:
        assert _hosts({"Patrick O'Shaughnessy"}) == {"Patrick O'Shaughnessy"}

    def test_several_clean_names_are_preserved(self) -> None:
        assert _hosts({"Kevin Roose", "Casey Newton"}) == {"Kevin Roose", "Casey Newton"}


class TestOrganisationsAreRejected:
    """The provider short-circuits on RSS author tags and hands back the publisher verbatim —
    which is how ``person:andreessen-horowitz`` became the corpus's top-ranked Person (#1652)."""

    def test_a_publisher_is_dropped(self) -> None:
        assert _hosts({"Andreessen Horowitz"}) == set()

    def test_a_network_marker_is_dropped(self) -> None:
        assert _hosts({"Some Media Network"}) == set()

    def test_a_mixed_answer_keeps_only_the_person(self) -> None:
        assert _hosts({"a16z, Erik Torenberg"}) == {"Erik Torenberg"}


class TestSafeDegradation:
    """#876: an over-eager filter must fail to "no host", never to an invented one."""

    def test_nothing_usable_yields_no_hosts_not_a_guess(self) -> None:
        assert _hosts({"a16z"}) == set()

    def test_an_empty_answer_stays_empty(self) -> None:
        assert _hosts(set()) == set()

    def test_blank_strings_do_not_become_hosts(self) -> None:
        assert _hosts({"", "   "}) == set()


class TestTheDeterministicPathStillWins:
    def test_a_stated_host_short_circuits_the_provider(self) -> None:
        """When the feed says who hosts it, the provider is never consulted — so its composite
        cannot override a correct deterministic answer."""
        feed = _Feed(
            title="Hard Fork",
            description="Hosted by Kevin Roose and Casey Newton, journalists at the Times.",
        )

        class _Exploding:
            def detect_hosts(self, **_kw: Any) -> Set[str]:
                raise AssertionError("provider consulted despite a stated host")

        out = processing._detect_hosts_from_feed(feed, _Exploding())
        assert "Kevin Roose" in out and "Casey Newton" in out


class TestPlatformsAreNotHosts:
    """A publisher/platform is never the host, even inside a host phrase (#1652 extension).

    Real case from the #1657 acceptance run. *The a16z Show*'s episode blurb runs two sentences
    together with no full stop::

        ...Listen to the a16z Show on Spotify Listen to the a16z Show on Apple Podcasts
        Follow our host: https://twitter.com/eriktorenberg

    So ``"Spotify Listen"`` is a capitalised run spanning the sentence boundary, and the NOUN
    "host" 45 characters later satisfied the ``names ... presenting-verb`` pattern. The
    publisher check already existed and was simply not applied on this path.
    """

    def test_the_a16z_spotify_case(self) -> None:
        from podcast_scraper.speaker_detectors.hosts import hosts_from_feed_statement

        desc = (
            "Find a16z on LinkedIn Listen to the a16z Show on Spotify Listen to the a16z "
            "Show on Apple Podcasts Follow our host: https://twitter.com/eriktorenberg"
        )
        assert hosts_from_feed_statement("The a16z Show", desc) == set()

    def test_a_real_stated_host_still_survives(self) -> None:
        """The guard must not cost us the hosts it was protecting."""
        from podcast_scraper.speaker_detectors.hosts import hosts_from_feed_statement

        out = hosts_from_feed_statement(
            "Hard Fork", "Hosted by Kevin Roose and Casey Newton, journalists at the Times."
        )
        assert out == {"Kevin Roose", "Casey Newton"}

    def test_a_platform_mixed_with_a_person_keeps_the_person(self) -> None:
        from podcast_scraper.speaker_detectors.hosts import hosts_from_feed_statement

        out = hosts_from_feed_statement(
            "Some Show", "Hosted by Spotify Studios and Katie Martin, who explain markets."
        )
        assert "Katie Martin" in out
        assert not any("spotify" in n.lower() for n in out)
