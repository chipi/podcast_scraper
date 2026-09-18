"""Being greeted by name is proof you are NOT that person (#2078, #876).

MEASURED ON A REAL FAILURE. ChinaTalk: the voice that opens *"Hey, Jordan. Good morning."* was
published as `Jordan Schneider` while Jordan's own voice went unnamed — and by TWO independent
paths, the LLM resolution and the deterministic forced-host rule, which is why the veto is
applied at both sites.

Neither existing guard could see it. `_GUEST_SPEECH_ACTS` looks for "thanks for having me", and
being addressed is not a speech act. `_talks_about` matches the full name or the SURNAME, and a
greeting uses the first name.

WHY ONLY THE START OF THE TEXT. Measured over 6,121 named voices on the stored corpus, the
start-of-voice shape fires on 2 records — both this episode. A sentence-anywhere variant
(", Eric.") fires on 858 of 4,896 self-introduced voices and 78 of 821 forced hosts. Those are
diarization bleed: the guest's closing line merged into the host's cluster. On a two-voice show
the complement pass would then swap two names that were already correct.
"""

import pytest

from podcast_scraper.speaker_detectors.resolution import (
    _addressed_at_open,
    refuted_by_third_person,
)


class TestTheGreetingIsRecognised:
    @pytest.mark.parametrize(
        "text",
        [
            "Hey, Jordan. Good morning.",
            "hey jordan, thanks for setting this up",
            "Hi, Jordan! Let's get into it.",
            "Good morning, Jordan. Where should we start?",
            "  Hello, Jordan, and welcome back.",
        ],
    )
    def test_a_voice_greeted_by_first_name_at_the_open(self, text) -> None:
        assert _addressed_at_open(text, "Jordan Schneider")

    @pytest.mark.parametrize(
        "text",
        [
            # The person themselves, opening their own show. Must NOT be refused.
            "Hey everyone, this is Jordan Schneider and welcome to ChinaTalk.",
            # A greeting to somebody else entirely.
            "Hey, Sarah. Good morning.",
            # Mid-text address: the bleed shape, deliberately NOT matched.
            "So that is the whole picture on export controls. Thanks so much, Jordan.",
            "I think the tariffs matter more than people say, Jordan.",
            "",
        ],
    )
    def test_what_must_not_match(self, text) -> None:
        assert not _addressed_at_open(text, "Jordan Schneider")


class TestItReachesTheRefutation:
    def test_the_greeting_refutes_the_name(self) -> None:
        """The whole point: `refuted_by_third_person` is the shared veto, so widening it here
        reaches the LLM verdict loop, the complement pass and the forced guest path at once."""
        assert refuted_by_third_person("Hey, Jordan. Good morning.", "Jordan Schneider")

    def test_a_self_introduction_still_wins(self) -> None:
        """A voice that says who it is outranks every inference about it, and that precedence must
        survive this change — otherwise a host greeting a co-host by name loses their own name."""
        text = "Hey, Jordan. Good morning. I'm Jordan Schneider, and this is ChinaTalk."
        assert not refuted_by_third_person(text, "Jordan Schneider")

    def test_an_unrelated_voice_is_untouched(self) -> None:
        assert not refuted_by_third_person("Let's talk about export controls.", "Jordan Schneider")
