"""When the feed names no host, the CONVERSATION does. The role is performed, not measured.

Metadata is the authority for who hosts a show — but three of our ten feeds state nobody (Planet
Money, Latent Space, the NVIDIA podcast). For those, the answer is still not a statistic. It is in
the transcript, because the host *performs* the role: he welcomes you to the show and introduces his
guest, and the guest thanks him for having them.

Every string below is real transcript text. Measured across those shows, this is decisive exactly
where talk time is worthless:

    Latent Space   the HOST talks 8.6% and the GUEST talks 84.5%
    NVIDIA         the cluster the pipeline LABELLED "Nicolas Cerisier" is the one that says
                   "I'm Noah Kravitz. My guest is Nicolas Serissier" — the labels were SWAPPED,
                   and only the conversation says so
"""

from __future__ import annotations

from podcast_scraper.speaker_detectors.hosts import (
    guests_introduced_by_the_host,
    roles_from_conversation,
)


class TestTheHostPerformsTheRole:
    def test_welcome_to_the_show(self) -> None:
        """Latent Space's host holds 8.6% of the episode. He is still the host."""
        roles = roles_from_conversation(
            {
                "HOST": "Welcome to the AI for Science podcast, part of Latent Space.",
                "GUEST": "So the thing about black holes is that they are not actually black.",
            }
        )
        assert roles["HOST"] == "host"
        assert "GUEST" not in roles, "a voice that performs no role must stay unclassified"

    def test_hello_and_welcome_plus_a_name(self) -> None:
        """Planet Money: "hello and welcome to Planet Money. I'm Alexi Horowitz-Gazi"."""
        roles = roles_from_conversation(
            {"A": "Hello and welcome to Planet Money. I'm Alexi Horowitz-Gazi."}
        )
        assert roles["A"] == "host"

    def test_the_host_names_the_guest(self) -> None:
        """ "I'm Noah Kravitz. My guest is Nicolas Serissier." — both roles, both names, one breath.

        This is the utterance that reveals the NVIDIA labels were swapped: the cluster the pipeline
        called "Nicolas Cerisier" is the one saying it.
        """
        texts = {
            "V1": "Welcome to the NVIDIA AI Podcast. I'm Noah Kravitz. My guest is Nicolas "
            "Serissier. Nicolas is vice president of engineering.",
            "V2": "Thanks Noah. So a virtual twin is a way to simulate an entire factory.",
        }
        assert roles_from_conversation(texts)["V1"] == "host"
        assert guests_introduced_by_the_host(texts) == {"Nicolas Serissier"}

    def test_my_guest_today_is(self) -> None:
        texts = {"H": "My guest today is Brian Chesky, the co-founder and CEO of Airbnb."}
        assert roles_from_conversation(texts)["H"] == "host"
        assert guests_introduced_by_the_host(texts) == {"Brian Chesky"}


class TestTheGuestPerformsTheirRoleToo:
    def test_thanks_for_having_me(self) -> None:
        roles = roles_from_conversation({"G": "Thank you. I'm really happy to be here. Thank you."})
        assert roles["G"] == "guest"

    def test_a_guest_who_out_talks_the_host_is_still_a_guest(self) -> None:
        """The whole point. Talk share says one thing; the conversation says the truth."""
        roles = roles_from_conversation(
            {
                "TALKS_MOST": "Thanks so much for having me. " + ("I think that... " * 400),
                "TALKS_LEAST": "Welcome back to the show.",
            }
        )
        assert roles["TALKS_MOST"] == "guest"
        assert roles["TALKS_LEAST"] == "host"


class TestItStaysSilentWhenNobodyPerforms:
    def test_no_cues_means_no_claim(self) -> None:
        """Saying nothing role-shaped must yield nothing. Guessing is what named an advert."""
        assert roles_from_conversation({"A": "So the tariffs went up.", "B": "They did."}) == {}

    def test_no_guest_named_means_no_guest(self) -> None:
        assert guests_introduced_by_the_host({"A": "So the tariffs went up."}) == set()

    def test_empty_input(self) -> None:
        assert roles_from_conversation({}) == {}
        assert roles_from_conversation(None) == {}
        assert guests_introduced_by_the_host(None) == set()
