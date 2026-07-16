"""ADR-110 — the model may IDENTIFY a voice. It may never AUTHOR a name.

`detect_speakers(title, description, known_hosts)` is asked who speaks before the audio is even
downloaded; its interface cannot take a transcript. So it answers from show notes, and show notes
name the people an episode is ABOUT as readily as the people in the room. That is how Elon Musk —
named only as the man SUING OpenAI — was returned as a speaker (#876).

This resolver asks the question where the answer lives: after diarization, against each voice's own
turns, with the mentions of every candidate name RETRIEVED from the transcript. Retrieval is what
separates a speaker from a subject: "Elon Musk is suing OpenAI" and "Jia Li is with us today" both
mention a person, and only the sentence says which one is in the room.

The candidate list is CLOSED. Anything outside it is discarded, whatever the model says.
"""

from __future__ import annotations

import json
from typing import Dict, List

import pytest

from podcast_scraper.speaker_detectors.resolution import (
    build_resolution_prompt,
    resolve_voices_from_conversation,
    retrieve_mentions,
)

pytestmark = pytest.mark.unit

HOST_TEXT = (
    "Welcome to the NVIDIA AI podcast. I'm Noah Kravitz. Jia Li is with us today to talk about "
    "holograms and what they mean for the future of telepresence. Great to have you."
)
GUEST_TEXT = (
    "Thanks for having me. So the core idea is that a hologram is really a light field, and "
    "reconstructing one in real time is a rendering problem before it is a display problem."
)

TURNS = [("SPEAKER_00", HOST_TEXT), ("SPEAKER_01", GUEST_TEXT)]
VOICES = {"SPEAKER_00": HOST_TEXT, "SPEAKER_01": GUEST_TEXT}


def _canned(mapping: Dict[str, object]):
    """An LLM that answers exactly this, so the test measures OUR handling, not a model."""

    def complete(_prompt: str) -> str:
        return json.dumps({"voices": mapping})

    return complete


class TestRetrievalIsWhatSeparatesASpeakerFromASubject:
    def test_it_finds_the_sentence_the_name_lives_in(self) -> None:
        hits = retrieve_mentions("Jia Li", TURNS)
        assert hits, "the host says her name out loud and retrieval found nothing"
        assert "is with us today" in hits[0]
        assert "SPEAKER_00" in hits[0]

    def test_it_says_that_the_speaker_of_a_mention_is_NOT_the_person(self) -> None:
        """The framing is load-bearing. "said by SPEAKER_01" reads as association, and that is how
        53.5% of an FT Unhedged episode was handed to Jay Powell — by the co-host discussing him."""
        assert "probably NOT them" in retrieve_mentions("Jia Li", TURNS)[0]

    def test_it_says_who_speaks_NEXT(self) -> None:
        """The person a host introduces is the person who speaks next — the model needs that."""
        assert "the NEXT voice to speak is SPEAKER_01" in retrieve_mentions("Jia Li", TURNS)[0]

    def test_a_name_nobody_says_is_reported_as_such(self) -> None:
        prompt = build_resolution_prompt(["Jensen Huang"], VOICES, [], TURNS)
        assert "NEVER SPOKEN ALOUD" in prompt

    def test_the_surname_alone_is_enough_to_retrieve(self) -> None:
        turns = [("SPEAKER_00", "And welcome back, Li. Good to see you again.")]
        assert retrieve_mentions("Jia Li", turns)


class TestTheModelMayOnlyChooseFromTheStatedNames:
    def test_a_voice_it_identifies_is_named(self) -> None:
        got = resolve_voices_from_conversation(
            ["Noah Kravitz", "Jia Li"],
            VOICES,
            _canned({"SPEAKER_00": "Noah Kravitz", "SPEAKER_01": "Jia Li"}),
            ordered_turns=TURNS,
        )
        assert got == {"SPEAKER_00": "Noah Kravitz", "SPEAKER_01": "Jia Li"}

    def test_a_name_NOBODY_STATED_is_discarded_however_confident_the_model(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """THE #876 FAILURE, refused at the door.

        The model is an identifier, not an author. If it returns a name the metadata never stated,
        the answer is thrown away — that is the entire safety property of this design.
        """
        with caplog.at_level("WARNING"):
            got = resolve_voices_from_conversation(
                ["Noah Kravitz", "Jia Li"],
                VOICES,
                _canned({"SPEAKER_00": "Noah Kravitz", "SPEAKER_01": "Elon Musk"}),
                ordered_turns=TURNS,
            )
        assert got == {"SPEAKER_00": "Noah Kravitz"}
        assert "Elon Musk" not in str(got)
        assert "never stated" in caplog.text

    def test_null_is_a_first_class_answer(self) -> None:
        """A voice nobody names stays unnamed. A wrong name is worse than no name."""
        got = resolve_voices_from_conversation(
            ["Noah Kravitz", "Jia Li"],
            VOICES,
            _canned({"SPEAKER_00": "Noah Kravitz", "SPEAKER_01": None}),
            ordered_turns=TURNS,
        )
        assert got == {"SPEAKER_00": "Noah Kravitz"}

    def test_one_person_cannot_be_two_voices(self) -> None:
        got = resolve_voices_from_conversation(
            ["Noah Kravitz"],
            VOICES,
            _canned({"SPEAKER_00": "Noah Kravitz", "SPEAKER_01": "Noah Kravitz"}),
            ordered_turns=TURNS,
        )
        assert list(got.values()) == ["Noah Kravitz"]

    def test_a_voice_that_TALKS_ABOUT_someone_is_not_that_someone(self) -> None:
        """FOUND ON THE REAL CORPUS. The retrieval that makes this work also misleads the model.

        FT Unhedged, "The Fed holds steady" — Katie Martin and Rob Armstrong DISCUSSING the Fed
        chair. The model was handed passages labelled `said by SPEAKER_01: "...Jay Powell, chair of
        the Federal Reserve, made a joke..."`, read the name sitting beside the voice as
        association, and gave **53.5% of the episode to Jay Powell**. SPEAKER_01 is Rob Armstrong.

        A prompt is not an enforcement mechanism (#876), so this is CHECKED. If you say somebody's
        name and never introduce yourself with it, you are talking about them, not being them.
        """
        armstrong = (
            "You did, you missed it. Jay Powell, chair of the Federal Reserve, made a joke "
            "yesterday, and Powell deeply bent his knees while the question was being asked."
        )
        got = resolve_voices_from_conversation(
            ["Katie Martin", "Rob Armstrong", "Jay Powell"],
            {"SPEAKER_01": armstrong},
            _canned({"SPEAKER_01": "Jay Powell"}),
            ordered_turns=[("SPEAKER_01", armstrong)],
        )
        assert got == {}, "a co-host discussing the Fed chair was published AS the Fed chair"

    def test_but_a_voice_that_introduces_itself_still_binds(self) -> None:
        """The rule must not eat the self-introduction it depends on."""
        own = "Hello and welcome back, I'm Rob Armstrong, and today we talk about Jay Powell."
        got = resolve_voices_from_conversation(
            ["Rob Armstrong", "Jay Powell"],
            {"SPEAKER_01": own},
            _canned({"SPEAKER_01": "Rob Armstrong"}),
            ordered_turns=[("SPEAKER_01", own)],
        )
        assert got == {"SPEAKER_01": "Rob Armstrong"}

    def test_a_dead_founder_the_notes_mention_is_not_given_a_voice(self) -> None:
        """HB Reese, founder of Reese's, died 1956 — and an earlier rule handed him a voice.

        Here he is a legitimate candidate (the notes DO name him), so the guard is not the closed
        list — it is that the model, shown that nobody speaks as him, answers null. And when it
        answers null we leave the voice unnamed rather than reaching for the next-best name.
        """
        turns = [("SPEAKER_00", "HB Reese founded the company in 1923."), ("SPEAKER_09", "Yeah.")]
        got = resolve_voices_from_conversation(
            ["HB Reese"],
            {"SPEAKER_09": "It was the greatest thing that ever happened to me, honestly, truly."},
            _canned({"SPEAKER_09": None}),
            ordered_turns=turns,
        )
        assert got == {}


class TestItFailsClosed:
    def test_an_unparsable_answer_names_nobody(self) -> None:
        assert (
            resolve_voices_from_conversation(
                ["Jia Li"], VOICES, lambda _p: "I'm afraid I can't do that", ordered_turns=TURNS
            )
            == {}
        )

    def test_a_provider_that_raises_names_nobody(self) -> None:
        def boom(_p: str) -> str:
            raise RuntimeError("503 high demand")

        assert resolve_voices_from_conversation(["Jia Li"], VOICES, boom, ordered_turns=TURNS) == {}

    def test_no_stated_names_means_no_call_at_all(self) -> None:
        calls: List[str] = []

        def spy(p: str) -> str:
            calls.append(p)
            return "{}"

        assert resolve_voices_from_conversation([], VOICES, spy, ordered_turns=TURNS) == {}
        assert not calls, "the metadata named nobody — there is nothing to match against"

    def test_a_reasoning_models_think_block_does_not_break_the_parse(self) -> None:
        def thinky(_p: str) -> str:
            return (
                "<think>SPEAKER_00 welcomes listeners, so that is the host.</think>"
                '{"voices": {"SPEAKER_00": "Noah Kravitz"}}'
            )

        got = resolve_voices_from_conversation(
            ["Noah Kravitz"], VOICES, thinky, ordered_turns=TURNS
        )
        assert got == {"SPEAKER_00": "Noah Kravitz"}


class TestTheRosterActuallyHonoursIt:
    """A resolver nobody calls is the defect this whole arc keeps producing.

    And it must rank BELOW the voice's own mouth: a voice that says "I'm Peter Ludwig" needs no
    model's opinion.
    """

    def test_the_roster_names_the_voice_the_model_matched(self) -> None:
        from podcast_scraper.providers.ml.diarization.base import (
            DiarizationResult,
            DiarizationSegment,
        )
        from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

        diar = DiarizationResult(
            segments=[
                DiarizationSegment(0.0, 300.0, "SPEAKER_00"),
                DiarizationSegment(300.0, 900.0, "SPEAKER_01"),
            ],
            num_speakers=2,
        )
        roster = resolve_speaker_roster(
            diar,
            HOST_TEXT + " " + GUEST_TEXT,
            detected_guests=[],  # corroboration deleted her — the situation ADR-110 exists for
            known_hosts=["Noah Kravitz"],
            voice_texts=VOICES,
            ordered_turns=TURNS,
            metadata_named=["Noah Kravitz", "Jia Li"],
            llm_voice_names={"SPEAKER_01": "Jia Li"},
        )
        assert roster.by_voice["SPEAKER_01"].named
        assert roster.by_voice["SPEAKER_01"].name == "Jia Li"

    def test_the_voices_OWN_self_introduction_outranks_the_model(self) -> None:
        from podcast_scraper.providers.ml.diarization.base import (
            DiarizationResult,
            DiarizationSegment,
        )
        from podcast_scraper.providers.ml.diarization.roster import resolve_speaker_roster

        own_words = "Oh sure, yeah, I'm Peter Ludwig, co-founder and CTO of Applied Intuition."
        diar = DiarizationResult(
            segments=[
                DiarizationSegment(0.0, 300.0, "SPEAKER_00"),
                DiarizationSegment(300.0, 900.0, "SPEAKER_01"),
            ],
            num_speakers=2,
        )
        roster = resolve_speaker_roster(
            diar,
            HOST_TEXT + " " + own_words,
            detected_guests=[],
            known_hosts=["Noah Kravitz"],
            voice_texts={"SPEAKER_00": HOST_TEXT, "SPEAKER_01": own_words},
            ordered_turns=[("SPEAKER_00", HOST_TEXT), ("SPEAKER_01", own_words)],
            metadata_named=["Noah Kravitz", "Peter Ludwig", "Qasar Younis"],
            llm_voice_names={"SPEAKER_01": "Qasar Younis"},  # the model got it wrong
        )
        assert roster.by_voice["SPEAKER_01"].name == "Peter Ludwig", (
            "the model overruled a voice that said its own name — the self-introduction is the "
            "strongest evidence there is and must win"
        )

    def test_the_diarization_pipeline_actually_CALLS_the_resolver(self) -> None:
        """Otherwise this is another gate wired to nothing."""
        import inspect

        from podcast_scraper.providers.ml.diarization import pipeline as diar_pipeline

        src = inspect.getsource(diar_pipeline.apply_diarization_to_result)
        assert "_resolve_voices_via_llm(" in src
        assert "llm_voice_names=llm_voice_names" in src, (
            "the resolver runs and its answer is thrown away — the voices stay unnamed and the "
            "LLM call is billed for nothing"
        )
