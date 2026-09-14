"""A guest's words must never be published under the host's name (#2062).

THE SHAPE OF A REAL PROD TRANSCRIPT. Diarization separates voices; the roster then tries to NAME
each one. `providers/ml/diarization/formatting.py:59` writes `f"{label}: "` where label comes from
`roster.label_for()` — the real name when the roster named that voice, the raw `SPEAKER_xx` when
it did not. So a transcript is a per-voice MIX:

    Aaron Levie: Welcome to the show.
    SPEAKER_01:  I think the margin structure is the interesting part.
    Aaron Levie: Say more.
    SPEAKER_01:  Gross margin compounds with scale.

THE DEFECT. Both attribution builders drop a marker that does not look like a person
(`_looks_like_person`, >=2 tokens) or is not in the detected-people whitelist. `speaker_for_char`
then returns the last SURVIVING marker at or before the quote — so with the guest's markers gone,
nothing closes the host's span and every guest quote inherits the host's name.

WHY IT MATTERS MORE THAN "NO GUEST". The guest is not merely unattributed: an insight the guest
stated is published under the host's name, and those quotes feed the `SPOKEN_BY` edges that
`top_people` ranks on. A wrong attribution is worse than a missing one.

THE RULE THESE TESTS PIN. Every line-start `Label:` is a turn BOUNDARY, recognised or not. An
unrecognised label ends the previous speaker's span and attributes to nobody. Under-attribution is
this module's stated contract (`gi/speakers.py`); mis-attribution is not.
"""

from __future__ import annotations

import pytest

from podcast_scraper.gi.speakers import (
    _detected_person_lookup,
    build_named_turns,
    build_unverified_named_turns,
    speaker_for_char,
)

pytestmark = pytest.mark.unit

HOST = "Aaron Levie"
GUEST = "Theo Jaffee"

#: The mixed shape: roster named the host, left the guest as a numbered voice.
MIXED = (
    f"{HOST}: Welcome to the show, today we talk about storage.\n"
    "SPEAKER_01: I think the margin structure is the interesting part.\n"
    f"{HOST}: Say more about that.\n"
    "SPEAKER_01: Gross margin compounds with scale in a way people miss.\n"
)

#: Both voices named — the case that already works, kept as the control.
BOTH_NAMED = (
    f"{HOST}: Welcome to the show, today we talk about storage.\n"
    f"{GUEST}: I think the margin structure is the interesting part.\n"
    f"{HOST}: Say more about that.\n"
    f"{GUEST}: Gross margin compounds with scale in a way people miss.\n"
)


def _at(transcript: str, needle: str) -> int:
    return transcript.index(needle)


GUEST_LINE_1 = "I think the margin structure"
GUEST_LINE_2 = "Gross margin compounds"
HOST_LINE_1 = "Welcome to the show"
HOST_LINE_2 = "Say more about that"


class TestTheUnverifiedBuilderDoesNotLeakTheHostOntoTheGuest:
    """`build_unverified_named_turns` — the artifact-build path (`gi/pipeline.py:2159`)."""

    def test_a_guest_quote_is_not_attributed_to_the_host(self) -> None:
        turns = build_unverified_named_turns(MIXED)
        who = speaker_for_char(_at(MIXED, GUEST_LINE_1), turns)
        assert who != HOST, (
            "the guest's words were published under the host's name — the operator-reported "
            "symptom: every insight on the panel showing the same speaker"
        )

    def test_the_second_guest_quote_is_not_the_host_either(self) -> None:
        turns = build_unverified_named_turns(MIXED)
        assert speaker_for_char(_at(MIXED, GUEST_LINE_2), turns) != HOST

    def test_an_unattributable_quote_attributes_to_nobody(self) -> None:
        # Under-attribution is the contract. None renders no speaker; a wrong name renders a lie.
        turns = build_unverified_named_turns(MIXED)
        assert speaker_for_char(_at(MIXED, GUEST_LINE_1), turns) is None

    def test_the_host_still_gets_their_own_quotes(self) -> None:
        # A fix that stops attributing anything would be no fix at all.
        turns = build_unverified_named_turns(MIXED)
        assert speaker_for_char(_at(MIXED, HOST_LINE_1), turns) == HOST
        assert speaker_for_char(_at(MIXED, HOST_LINE_2), turns) == HOST


class TestTheWhitelistedBuilderHasTheSameDefect:
    """`build_named_turns` — the enrich-edges path, which is the one the prod VIEW reads.

    Here the guest is dropped for a second reason: guests are rarely detected, so the guest's name
    is absent from `_detected_person_lookup` even when the transcript names it.
    """

    def test_a_guest_quote_is_not_attributed_to_the_host(self) -> None:
        known = _detected_person_lookup([HOST], [])  # guest never detected — the common case
        turns = build_named_turns(BOTH_NAMED, known)
        who = speaker_for_char(_at(BOTH_NAMED, GUEST_LINE_1), turns)
        assert who != HOST, "an undetected guest's quote inherited the host's name"

    def test_an_undetected_speaker_ends_the_previous_span(self) -> None:
        known = _detected_person_lookup([HOST], [])
        turns = build_named_turns(BOTH_NAMED, known)
        assert speaker_for_char(_at(BOTH_NAMED, GUEST_LINE_1), turns) is None

    def test_the_host_is_still_attributed(self) -> None:
        known = _detected_person_lookup([HOST], [])
        turns = build_named_turns(BOTH_NAMED, known)
        assert speaker_for_char(_at(BOTH_NAMED, HOST_LINE_1), turns) == HOST

    def test_both_detected_still_works(self) -> None:
        # The control: when detection succeeds, attribution was already correct.
        known = _detected_person_lookup([HOST], [GUEST])
        turns = build_named_turns(BOTH_NAMED, known)
        assert speaker_for_char(_at(BOTH_NAMED, GUEST_LINE_1), turns) == GUEST
        assert speaker_for_char(_at(BOTH_NAMED, HOST_LINE_1), turns) == HOST


class TestABoundaryIsABoundaryRegardlessOfLabel:
    """The rule itself, stated once."""

    def test_an_unrecognised_marker_closes_the_span(self) -> None:
        t = "Aaron Levie: one.\nSPEAKER_09: two.\n"
        turns = build_unverified_named_turns(t)
        assert speaker_for_char(t.index("two."), turns) is None

    def test_a_recognised_marker_after_an_unrecognised_one_reopens(self) -> None:
        t = "Aaron Levie: one.\nSPEAKER_09: two.\nAaron Levie: three.\n"
        turns = build_unverified_named_turns(t)
        assert speaker_for_char(t.index("three."), turns) == "Aaron Levie"

    def test_text_before_any_marker_belongs_to_nobody(self) -> None:
        t = "Some preamble with no speaker.\nAaron Levie: one.\n"
        turns = build_unverified_named_turns(t)
        assert speaker_for_char(t.index("Some preamble"), turns) is None
