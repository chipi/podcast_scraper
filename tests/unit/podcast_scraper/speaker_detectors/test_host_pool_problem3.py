"""Host pools, problem 3 of the naming scoreboard (2026-10-03): one behaviour per test, synthetic.

The feed's statement and author tags are both read; a junk name is dropped on its own; the stated
grammar reads multi-name titles, "join hosts", a role or place between names, initials, particles;
the episode's own description names its host and silences a byline; a respelt pool host is not a
stranger; a forced pool name never lands on the interviewee of a cold-open interview.
"""

from __future__ import annotations

import pytest

from podcast_scraper.speaker_detectors.hosts import (
    _HOST_SPEECH_ACTS,
    compose_episode_hosts,
    detect_hosts_from_feed,
    hosts_from_episode_description,
    hosts_from_feed_statement,
    names_the_show,
)


class TestAJunkNameIsDroppedOnItsOwn:
    def test_the_count_word_is_refused_and_the_author_tag_is_read(self) -> None:
        desc = "Two Carnegie Mellon faculty explore how AI is reshaping design."
        assert detect_hosts_from_feed("AI and Design", desc, ["Dan Saffer and Nik Martelaro"]) == {
            "Dan Saffer",
            "Nik Martelaro",
        }

    def test_the_tail_of_a_proper_noun_no_longer_blocks_the_author_tag(self) -> None:
        desc = "Twice a month the Council of the Andean Online team brings you the region."
        assert detect_hosts_from_feed("Andes Focus", desc, ["Ada Brook"]) == {"Ada Brook"}

    def test_the_real_name_beside_a_nationality_is_kept(self) -> None:
        desc = (
            "Hosted by Anglo Canadian transplant to Colombia, Richard McColl and the Briefing is "
            "reported by journalist Emily Hart."
        )
        assert hosts_from_feed_statement("Colombia Calling", desc) == {"Emily Hart"}
        assert detect_hosts_from_feed("Colombia Calling", desc, ["Richard McColl"]) == {
            "Emily Hart",
            "Richard McColl",
        }


class TestStatementAndAuthorTagAreUnioned:
    def test_statement_plus_personal_tag(self) -> None:
        desc = "Join theoretical physicist Dan Hooper and co-host Shalma Wegsman as they answer."
        assert detect_hosts_from_feed(
            "Why This Universe?", desc, ["Dan Hooper, Shalma Wegsman"]
        ) == {
            "Dan Hooper",
            "Shalma Wegsman",
        }

    def test_one_person_spelt_twice_is_one_entry(self) -> None:
        desc = "Hosted by Alastair Campbell."
        assert detect_hosts_from_feed("Leading", desc, ["Alistair Campbell"]) == {
            "Alastair Campbell"
        }

    def test_a_brand_author_tag_is_not_a_person(self) -> None:
        desc = "Join host Luke Timmerman for conversations with biotech newsmakers."
        assert detect_hosts_from_feed("The Long Run", desc, ["Timmerman Report"]) == {
            "Luke Timmerman"
        }


class TestTheStatedGrammar:
    def test_title_with_two_names(self) -> None:
        assert hosts_from_feed_statement(
            "The Curiosity Shop with Brené Brown and Adam Grant", ""
        ) == {
            "Brené Brown",
            "Adam Grant",
        }

    def test_title_with_a_credential(self) -> None:
        assert hosts_from_feed_statement("AI 4 UX with John Whalen, PhD", "") == {"John Whalen"}

    def test_join_hosts_with_places_and_a_particle(self) -> None:
        desc = (
            "Join hosts Eric Olander in Vietnam and Cobus van "
            "Staden in South Africa for interviews."
        )
        assert hosts_from_feed_statement("CGSP", desc) == {"Eric Olander", "Cobus van Staden"}

    def test_a_role_between_two_names(self) -> None:
        desc = (
            "Join mathematician Professor Hannah Fry and science "
            "creator Michael Stevens as they dig."
        )
        assert hosts_from_feed_statement("Science", desc) == {"Hannah Fry", "Michael Stevens"}

    def test_an_oxford_comma(self) -> None:
        desc = "Hosted by Adam Reichardt, Alexandra Karppi, and Nina Panikova, this podcast brings."
        assert hosts_from_feed_statement("Talk", desc) == {
            "Adam Reichardt",
            "Alexandra Karppi",
            "Nina Panikova",
        }

    def test_a_middle_initial_and_the_verb_uncover(self) -> None:
        desc = "Freakonomics co-author Stephen J. Dubner uncovers the hidden side of everything."
        assert hosts_from_feed_statement("Freakonomics Radio", desc) == {"Stephen J. Dubner"}

    def test_bare_host_before_the_name(self) -> None:
        desc = "Host Jonquilyn Hill will take you on a journey to find the answers."
        assert hosts_from_feed_statement("Explain It to Me", desc) == {"Jonquilyn Hill"}

    def test_run_by_and_a_job_title_in_front_of_the_name(self) -> None:
        assert hosts_from_feed_statement(
            "MLST", "MLST is run by Tim Scarfe, Ph.D and friends."
        ) == {"Tim Scarfe"}
        desc = "A podcast hosted by Senior User Experience Specialist Therese Fessenden."
        assert hosts_from_feed_statement("UX", desc) == {"Therese Fessenden"}

    def test_a_descriptor_between_hosted_by_and_the_name(self) -> None:
        desc = (
            "Produced and hosted by Johannesburg-based "
            "entrepreneur and American expat Justin Norman."
        )
        assert hosts_from_feed_statement("The Flip", desc) == {"Justin Norman"}

    def test_first_names_presenting_resolve_to_full_names_in_the_same_text(self) -> None:
        desc = (
            "Take a deep dive into history with Tom Holland & Dominic Sandbrook. From Rome to the "
            "Norman Conquest of England, Tom and Dominic bring the past to life."
        )
        assert hosts_from_feed_statement("The Rest Is History", desc) == {
            "Tom Holland",
            "Dominic Sandbrook",
        }

    def test_first_names_with_no_full_form_add_nobody(self) -> None:
        desc = "On this podcast Laolu, Furo, and Nosa talk about technology."
        assert hosts_from_feed_statement("Open Africa", desc) == set()

    def test_a_place_after_the_verb_is_not_a_host(self) -> None:
        desc = (
            "Join hosts Eric Olander in Vietnam and Cobus van "
            "Staden in South Africa for interviews."
        )
        assert "South Africa" not in hosts_from_feed_statement("CGSP", desc)


class TestAPossessiveTitleBelongsToThePerson:
    def test_names_the_show_is_false_for_the_owner(self) -> None:
        assert not names_the_show("Azeem Azhar", "Azeem Azhar's Exponential View")

    def test_the_author_tag_is_the_host(self) -> None:
        assert detect_hosts_from_feed("Azeem Azhar's Exponential View", "", ["Azeem Azhar"]) == {
            "Azeem Azhar"
        }

    def test_the_show_itself_is_still_refused(self) -> None:
        assert names_the_show("Trivium China", "The Trivium China Podcast")


class TestTheEpisodeNamesItsHost:
    def test_sits_down_with(self) -> None:
        desc = "Erik Torenberg sits down with Replit founder Amjad Masad to ask about college."
        assert hosts_from_episode_description("College", desc, "The a16z Show") == {
            "Erik Torenberg"
        }

    def test_a_byline_adds_nobody_once_the_description_names_the_host(self) -> None:
        pool = compose_episode_hosts(
            [],
            ["Amjad Masad", "Erik Torenberg", "Gagan Biyani"],
            episode_title="College",
            episode_description="Erik Torenberg sits down with Amjad Masad and Gagan Biyani.",
            feed_title="The a16z Show",
        )
        assert pool == ["Erik Torenberg"]

    def test_a_byline_is_kept_when_the_description_names_nobody(self) -> None:
        pool = compose_episode_hosts(
            [], ["Ada Brook"], episode_title="Ep 1", episode_description="A chat.", feed_title="X"
        )
        assert pool == ["Ada Brook"]

    def test_feed_hosts_outrank_and_merge_with_the_description(self) -> None:
        pool = compose_episode_hosts(
            ["Katie Martin", "Robert Armstrong"],
            [],
            episode_title="Ep",
            episode_description="Rob Armstrong speaks with John Paul Rathbone.",
            feed_title="Unhedged",
        )
        assert pool == ["Katie Martin", "Robert Armstrong"]

    def test_the_pool_never_holds_the_show_or_a_non_person(self) -> None:
        pool = compose_episode_hosts(
            ["Norman Conquest"],
            [],
            episode_title="Ep",
            feed_title="Machine Learning Street Talk (MLST)",
            more=["Machine Learning Street", "Trivium China"],
        )
        assert pool == []


class TestADescribedHostIsNotAParticipant:
    def test_a_stated_guest_before_the_cue_is_not_the_host(self) -> None:
        desc = "Mark Zuckerberg speaks with Sarah Guo and Elad Gil about Biohub."
        assert (
            hosts_from_episode_description(
                "Biohub", desc, "No Priors", feed_hosts=["Elad Gil", "Sarah Guo"]
            )
            == set()
        )

    def test_a_participant_named_by_the_cue_is_not_the_host(self) -> None:
        desc = "Fei-Fei Li sits down with Martin Casado to discuss spatial intelligence."
        assert (
            hosts_from_episode_description(
                "Spatial", desc, "The a16z Show", participants=["Fei-Fei Li"]
            )
            == set()
        )

    def test_the_host_before_the_cue_is_still_the_host(self) -> None:
        desc = "Erik Torenberg sits down with Replit founder Amjad Masad."
        assert hosts_from_episode_description(
            "College", desc, "The a16z Show", participants=["Amjad Masad"]
        ) == {"Erik Torenberg"}

    def test_a_role_word_before_the_name_outranks_the_guest_list(self) -> None:
        desc = "Neil deGrasse Tyson and comic co-host Jordan Klepper sit down with Lara Anderson."
        assert hosts_from_episode_description(
            "Quantum",
            desc,
            "StarTalk",
            feed_hosts=["Neil deGrasse Tyson"],
            participants=["Jordan Klepper", "Lara Anderson"],
        ) == {"Jordan Klepper"}

    def test_compose_passes_the_episode_people_through(self) -> None:
        pool = compose_episode_hosts(
            [],
            [],
            episode_title="Ep",
            episode_description="Fei-Fei Li sits down with Martin Casado.",
            feed_title="The a16z Show",
            episode_people=["Fei-Fei Li"],
        )
        assert pool == []


class TestHostActs:
    @pytest.mark.parametrize(
        "line",
        [
            "Thank you both very much for coming.",
            "Thanks all for joining us",
            "Thanks for coming on",
        ],
    )
    def test_thanking_the_guests_for_coming_is_a_host_act(self, line: str) -> None:
        assert any(p.search(line) for p in _HOST_SPEECH_ACTS)

    @pytest.mark.parametrize(
        "line", ["thank you for having me", "Boris, thank you so much for being here."]
    )
    def test_the_singular_close_that_bleeds_into_the_guest_is_not(self, line: str) -> None:
        assert not any(p.search(line) for p in _HOST_SPEECH_ACTS)
