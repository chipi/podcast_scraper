"""The feed says who hosts the show. Read the statement — do not run NER over the paragraph.

Every one of these strings is the REAL description (or title) of a feed we carry. They are here
because the roster spent two days inferring hosts from talk time, and the answer was written down
the whole time.

The two failure directions this pins:

* **NER cannot tell a host from a person.** Latent Space's description lists its PAST GUESTS — Bret
  Taylor, Chris Lattner, George Hotz — and NER offers every one of them as a host of the show. It is
  not wrong about them being people. It is being asked the wrong question.
* **A show that names nobody must return NOBODY.** Guessing is what let an advertiser's name be
  published as the host of a technology podcast. An empty result leaves the voice unnamed, which is
  the safe direction (#876).
"""

from __future__ import annotations

from podcast_scraper.speaker_detectors.hosts import hosts_from_feed_statement

# Real feed metadata, verbatim.
HARD_FORK = (
    "“Hard Fork” is a show about the future that’s already here. Each week, journalists Kevin "
    "Roose and Casey Newton explore and make sense of the latest in the rapidly changing world of "
    "tech. "
    "Subscribe today at nytimes.com/podcasts or on Apple Podcasts and Spotify."
)
THE_JOURNAL = (
    "The most important stories about money, business and power. Hosted by Ryan Knutson and "
    "Jessica Mendoza. The Journal is a co-production of Spotify and The Wall Street Journal."
)
THE_DAILY = (
    "This is what the news should sound like. The biggest stories of our time, told by the best "
    "journalists in the world. Hosted by Michael Barbaro, Rachel Abrams and Natalie Kitroeff. "
    "Twenty minutes a day, six days a week, ready by 6 a.m. Subscribe today at nytimes.com."
)
NO_PRIORS = (
    "At this moment of inflection in technology, co-hosts Elad Gil and Sarah Guo talk to the "
    "world's leading AI engineers, researchers and founders about the biggest questions."
)
ODD_LOTS = (
    "Bloomberg's Joe Weisenthal and Tracy Alloway explore the most interesting topics in finance, "
    "markets and economics. Join the conversation every Monday, Thursday, and Friday"
)
UNHEDGED = (
    "Katie Martin, Robert Armstrong and other markets nerds at the Financial Times explain the big "
    "ideas behind what’s happening in finance right now. Every Tuesday and Thursday."
)
PLANET_MONEY = (
    "Wanna see a trick? Give us any topic and we can tie it back to the economy. At Planet Money, "
    "we explore the forces that shape our lives and bring you along for the ride."
)
LATENT_SPACE = (
    "The podcast by and for AI Engineers! We cover Foundation Models changing every domain in Code "
    "Generation, Multimodality, AI Agents, GPU Infra. Past guests include Bret Taylor, Chris "
    "Lattner, George Hotz and Jeremy Howard."
)


class TestTheFeedStatesItsHosts:
    def test_hosted_by(self) -> None:
        assert hosts_from_feed_statement("The Journal.", THE_JOURNAL) == {
            "Ryan Knutson",
            "Jessica Mendoza",
        }

    def test_hosted_by_three(self) -> None:
        assert hosts_from_feed_statement("The Daily", THE_DAILY) == {
            "Michael Barbaro",
            "Rachel Abrams",
            "Natalie Kitroeff",
        }

    def test_co_hosts(self) -> None:
        assert hosts_from_feed_statement("No Priors", NO_PRIORS) == {"Elad Gil", "Sarah Guo"}

    def test_journalists_x_and_y(self) -> None:
        assert hosts_from_feed_statement("Hard Fork", HARD_FORK) == {
            "Kevin Roose",
            "Casey Newton",
        }

    def test_names_then_a_presenting_verb(self) -> None:
        """ "Joe Weisenthal and Tracy Alloway explore..." — no "hosted by" anywhere."""
        assert hosts_from_feed_statement("Odd Lots", ODD_LOTS) == {
            "Joe Weisenthal",
            "Tracy Alloway",
        }

    def test_the_publishers_possessive_is_not_part_of_the_name(self) -> None:
        """The feed says "Bloomberg's Joe Weisenthal". The host is not called Bloomberg's Joe."""
        assert "Bloomberg's Joe Weisenthal" not in hosts_from_feed_statement("Odd Lots", ODD_LOTS)

    def test_names_separated_from_the_verb_by_filler(self) -> None:
        """ "Katie Martin, Robert Armstrong and other markets nerds at the FT explain..."."""
        assert hosts_from_feed_statement("Unhedged", UNHEDGED) == {
            "Katie Martin",
            "Robert Armstrong",
        }

    def test_the_host_can_be_in_the_title(self) -> None:
        assert hosts_from_feed_statement("Invest Like the Best with Patrick O'Shaughnessy", "") == {
            "Patrick O'Shaughnessy"
        }


class TestItMustNotGuess:
    def test_past_guests_are_not_hosts(self) -> None:
        """THE NER TRAP. These are real people, correctly identified — and none of them host it.

        A better NER model does not fix this. `en_core_web_trf` returns exactly the same four names,
        because they ARE people. Being a person is not being the host, and only the statement says
        which is which.
        """
        hosts = hosts_from_feed_statement("Latent Space: The AI Engineer Podcast", LATENT_SPACE)
        for guest in ("Bret Taylor", "Chris Lattner", "George Hotz", "Jeremy Howard"):
            assert guest not in hosts, f"{guest} is a past GUEST listed in the description"
        assert hosts == set()

    def test_a_show_that_names_nobody_returns_nobody(self) -> None:
        """Planet Money's description names no host. The honest answer is none.

        It also opens "Wanna see a trick?", which the small NER model reports as a PERSON. Guessing
        here is what put an advertiser's name on a podcast — an empty result leaves the voice
        unnamed, which is the safe direction (#876).
        """
        hosts = hosts_from_feed_statement("Planet Money", PLANET_MONEY)
        assert hosts == set()

    def test_the_shows_own_name_is_not_a_person(self) -> None:
        """ "At Planet Money, we explore..." is a capitalised run followed by a presenting verb."""
        assert "Planet Money" not in hosts_from_feed_statement("Planet Money", PLANET_MONEY)
