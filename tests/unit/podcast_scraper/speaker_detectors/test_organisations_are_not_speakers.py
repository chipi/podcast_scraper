"""An organisation is not a speaker, and 150 of them were published as one (#2075).

A name that reaches `known_hosts` becomes a name a voice may be CALLED, so a publisher in that
list ends up on somebody's quotes. Measured on the production snapshot — 150 published speaker
entries across 144 episodes are a publisher or the show itself:

    128  source=known_hosts     "Andreessen Horowitz" 60, "Conversations with Tyler" 23,
                                "Machine Learning Street" 18, "Trivium China" 10
     50  source=llm_resolution  the same strings, via the closed candidate list
      5  source=self_intro      one-off ASR garbage ("Boston College", "Rindman University")

Both predicates already existed and neither was applied: `_clean_person_names` checks only
`has_org_markers` (12 of the 150) and `is_publishable_speaker_name` accepted all 150.
"""

from __future__ import annotations

from podcast_scraper.speaker_detectors.hosts import (
    drop_non_person_names,
    is_publishable_speaker_name,
)


class TestItDropsTheOrganisations:
    def test_a_publisher_is_not_a_person(self) -> None:
        assert drop_non_person_names(["Andreessen Horowitz"], "The a16z Show") == []

    def test_an_institution_is_not_a_person(self) -> None:
        out = drop_non_person_names(
            ["Mercatus Center at George Mason University"], "Conversations with Tyler"
        )
        assert out == []

    def test_the_show_itself_is_not_a_person(self) -> None:
        assert drop_non_person_names(["Conversations with Tyler"], "Conversations with Tyler") == []
        assert drop_non_person_names(["Trivium China"], "The Trivium China Podcast") == []
        assert (
            drop_non_person_names(
                ["Machine Learning Street"], "Machine Learning Street Talk (MLST)"
            )
            == []
        )


class TestItKeepsThePeople:
    def test_a_two_token_person_survives(self) -> None:
        assert drop_non_person_names(["Russ Roberts"], "EconTalk") == ["Russ Roberts"]

    def test_a_mononym_person_survives(self) -> None:
        # THE REASON `is_network_or_org_author` IS NOT USED HERE: it rejects every mononym, and a
        # one-token name in this list is a real person. #876 and `_clean_person_names` depend on it.
        assert drop_non_person_names(["Oprah", "Sting", "swyx"], "Latent Space") == [
            "Oprah",
            "Sting",
            "swyx",
        ]

    def test_a_host_whose_name_leads_the_show_title_survives(self) -> None:
        assert drop_non_person_names(["Peter Attia, MD"], "The Peter Attia Drive") == [
            "Peter Attia, MD"
        ]

    def test_no_feed_title_means_no_opinion_about_the_show(self) -> None:
        # Absence of evidence is not evidence that the candidate is the show.
        assert drop_non_person_names(["Machine Learning Street"], None) == [
            "Machine Learning Street"
        ]


class TestThePublishGate:
    def test_it_now_refuses_a_publisher(self) -> None:
        # The last thing between a name and a voice. It accepted all 150, including the five that
        # arrive as a transcript self-introduction, which no candidate-list filter can see.
        assert is_publishable_speaker_name("Andreessen Horowitz") is False
        assert is_publishable_speaker_name("Boston College") is False

    def test_it_still_accepts_real_people(self) -> None:
        assert is_publishable_speaker_name("Russ Roberts") is True
        assert is_publishable_speaker_name("swyx") is True
