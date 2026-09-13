"""A show's own name must not be seated as a host person (#2064).

MEASURED ON PRODUCTION (330-episode feed-stratified sample, 2026-09-13). 37 speaker entries name
their own show; 18 of those are legitimate — a real person whose name is IN the title ("Invest Like
the Best **with Patrick O'Shaughnessy**", "Macro Musings **with David Beckworth**"). The other 19
are the show itself, seated as a host person:

    Machine Learning Street   (from "Machine Learning Street Talk")   6 episodes
    Trivium China                                                     6
    Africa Tech Summit                                                5
    Conversations with Tyler                                          2

Reproduced on a fresh DGX ingest: the roster came back
``SPEAKER_01 'Africa Tech Summit' role=host`` beside a real guest, while the actual person who
self-introduced was ignored.

WHY IT MATTERS MORE NOW. Until #2062 the graph was built from the pre-diarization hint and most
people landed as ``mentioned``, so an org-as-host was one wrong label among many. The roster is now
authoritative and its roles reach ``kg.json`` directly, so this string becomes a Person node with
``role="host"`` — followable, rankable among speakers, counted in person metrics. It also defeats
the #2062 coherence guard, because the show name IS in ``content.speakers``: a wrong entry, not a
missing one. And it reaches migration m0009, which demoted a real guest on an episode whose roster
was "complete" while containing the non-person "Developer Survey".

THIS FILE COVERS THE TITLE PATH. ``_HOST_PHRASES`` includes a "names, then a presenting verb"
pattern for DESCRIPTIONS — "Joe Weisenthal and Tracy Alloway explore..." — and ``_PRESENTS``
includes ``talk``. Run over a TITLE, "Machine Learning Street Talk" parses as the names "Machine
Learning Street" followed by the verb "Talk". A title is not a sentence, and the host lives there in
a different shape ("... with Patrick O'Shaughnessy").
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    hosts_from_feed_statement,
    names_the_show,
    normalize_host_names,
)

pytestmark = pytest.mark.unit


class TestATitleIsNotASentence:
    def test_a_presenting_verb_in_the_title_does_not_name_a_host(self) -> None:
        # THE CASE: "Talk" is the last word of the show's name, not something people do.
        assert hosts_from_feed_statement("Machine Learning Street Talk", None) == set()

    @pytest.mark.parametrize(
        "title",
        [
            "Machine Learning Street Talk",
            "Trivium China",
            "Africa Tech Summit Podcast",
            "Conversations with Tyler",
            "The China-Global South Project",
        ],
    )
    def test_no_show_name_becomes_a_host(self, title: str) -> None:
        assert hosts_from_feed_statement(title, None) == set()


class TestTheHostInTheTitleStillWorks:
    """The 18 legitimate cases. A fix that loses these trades one bug for a worse one."""

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Invest Like the Best with Patrick O'Shaughnessy", "Patrick O'Shaughnessy"),
            ("Complex Systems with Patrick McKenzie", "Patrick McKenzie"),
            ("Macro Musings with David Beckworth", "David Beckworth"),
        ],
    )
    def test_the_with_pattern_still_names_the_host(self, title: str, expected: str) -> None:
        assert expected in hosts_from_feed_statement(title, None)


class TestTheDescriptionPathIsUnchanged:
    """The presenting-verb pattern exists for descriptions and must keep working there."""

    def test_names_then_a_presenting_verb_in_the_description(self) -> None:
        got = hosts_from_feed_statement(
            "Odd Lots",
            "Joe Weisenthal and Tracy Alloway explore the most interesting topics in finance.",
        )
        assert got == {"Joe Weisenthal", "Tracy Alloway"}

    def test_hosted_by_in_the_description(self) -> None:
        got = hosts_from_feed_statement(
            "The Journal.", "Hosted by Ryan Knutson and Jessica Mendoza."
        )
        assert got == {"Ryan Knutson", "Jessica Mendoza"}

    def test_a_description_echoing_the_show_name_is_still_rejected(self) -> None:
        # The guard that already existed for descriptions: "At Planet Money, we explore..."
        assert hosts_from_feed_statement("Planet Money", "At Planet Money, we explore...") == set()


class TestTheAuthorTagPath:
    """The other route in: ``<itunes:author>`` equal to the show's own name.

    ``is_network_or_org_author`` rejects org markers, known networks and mononyms — none of which
    "Africa Tech Summit", "Trivium China" or "Conversations with Tyler" trip. So the author tag was
    accepted as a host person, and `_validate_hosts_with_first_episode` then CONFIRMED it, because a
    show's name is always spoken in its own opening.

    The check is structural, not a wordlist: compare the candidate against the feed's own title.
    That is the one piece of evidence that actually distinguishes "this is the show" from "this is a
    person", and it is already on the artifact.
    """

    @pytest.mark.parametrize(
        "author,title",
        [
            ("Africa Tech Summit", "Africa Tech Summit Podcast"),
            ("Trivium China", "Trivium China"),
            ("Conversations with Tyler", "Conversations with Tyler"),
            ("Machine Learning Street", "Machine Learning Street Talk"),
            ("The China-Global South Project", "The China-Global South Project"),
        ],
    )
    def test_an_author_that_names_the_show_is_not_a_host(self, author: str, title: str) -> None:
        assert names_the_show(author, title) is True

    @pytest.mark.parametrize(
        "person,title",
        [
            # The 18 legitimate cases: the host's name is IN the title, after "with".
            ("Patrick O'Shaughnessy", "Invest Like the Best with Patrick O'Shaughnessy"),
            ("Patrick McKenzie", "Complex Systems with Patrick McKenzie"),
            ("David Beckworth", "Macro Musings with David Beckworth"),
            # A host whose name has nothing to do with the title.
            ("Ryan Knutson", "The Journal."),
            ("Eric Olander", "The China-Global South Project"),
        ],
    )
    def test_a_real_host_is_not_mistaken_for_the_show(self, person: str, title: str) -> None:
        assert names_the_show(person, title) is False

    def test_no_title_means_no_opinion(self) -> None:
        # Absence of the title is not evidence that the candidate IS the show.
        assert names_the_show("Africa Tech Summit", None) is False
        assert names_the_show("", "Africa Tech Summit") is False


class TestNormalizeHostNamesAppliesIt:
    """The centralising function — "a fifth seeding path cannot forget to call it"."""

    def test_the_show_name_is_dropped_when_the_title_is_known(self) -> None:
        got = normalize_host_names(["Africa Tech Summit"], feed_title="Africa Tech Summit Podcast")
        assert got == set()

    def test_a_real_host_survives(self) -> None:
        got = normalize_host_names(
            ["Patrick O'Shaughnessy"], feed_title="Invest Like the Best with Patrick O'Shaughnessy"
        )
        assert got == {"Patrick O'Shaughnessy"}

    def test_without_a_title_the_behaviour_is_unchanged(self) -> None:
        assert normalize_host_names(["Ryan Knutson"]) == {"Ryan Knutson"}
