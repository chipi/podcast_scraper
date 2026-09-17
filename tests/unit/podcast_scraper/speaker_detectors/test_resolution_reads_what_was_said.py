"""The resolver shows the model what was actually said, and reads what the model actually answered.

Each case was measured on the DGX (vLLM) on one Odd Lots episode (#2075), with synthetic text here:

* the show notes say `Jeffrey Schmid`, the transcript says "Jeff Schmidt" — retrieval reported the
  name "NEVER SPOKEN ALOUD", which the prompt tells the model means "not in the room";
* "And I'm Joe Weisenthal" was presented as Joe's voice being "probably NOT" Joe;
* turns arrive per ASR segment, so the hand-off after an introduction never named the next voice;
* the model answered `SPEAKER_2` for `SPEAKER_02`, and once without the `voices` wrapper — both
  discarded every answer;
* the model copied the spoken spelling ("Tracy Allaway") and was discarded as inventing a name.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.speaker_detectors.resolution import (
    _parse,
    _voice_id_in,
    resolve_voices_and_roles,
    retrieve_mentions,
)

pytestmark = pytest.mark.unit


class TestRetrieval:
    def test_a_spoken_nickname_and_one_letter_surname_variant_is_a_mention(self) -> None:
        turns = [
            ("S2", "We're going to be speaking with Regional Fed PresidentJeff Schmidt. Welcome.")
        ]
        assert retrieve_mentions("Jeffrey Schmid", turns)

    def test_a_bare_different_surname_is_not_a_mention(self) -> None:
        turns = [("S2", "Eric Schmidt said something about search engines last week.")]
        assert retrieve_mentions("Jeffrey Schmid", turns) == []

    def test_a_self_introduction_is_presented_as_the_voice_being_them(self) -> None:
        hits = retrieve_mentions("Ben Weisenthal", [("S3", "And I'm Ben Weisenthal.")])
        assert hits and "INTRODUCES ITSELF" in hits[0] and "probably NOT" not in hits[0]

    def test_the_next_voice_skips_the_introducers_own_continuation(self) -> None:
        turns = [
            ("S2", "We're speaking with Cal Schmid."),
            ("S2", "Thank you so much for coming back on the show."),
            ("S1", "Glad to be here."),
        ]
        assert "the NEXT voice to speak is S1" in retrieve_mentions("Cal Schmid", turns)[0]

    def test_one_passage_is_listed_once(self) -> None:
        hits = retrieve_mentions("Ada Brook", [("S2", "I'm Ada Brook. Follow me at Ada Brook.")])
        assert len(hits) == len(set(hits))


class TestReadingTheAnswer:
    VOICES = {"SPEAKER_00": "a", "SPEAKER_01": "b", "SPEAKER_02": "c"}

    def test_a_voice_id_without_its_leading_zero_is_that_voice(self) -> None:
        assert _voice_id_in("SPEAKER_2", self.VOICES) == "SPEAKER_02"
        assert _voice_id_in("SPEAKER_9", self.VOICES) is None

    def test_an_answer_without_the_voices_wrapper_is_read(self) -> None:
        raw = json.dumps({"SPEAKER_1": {"name": "Ada Brook", "role": "host"}})
        assert _parse(raw)["SPEAKER_1"].name == "Ada Brook"

    def test_a_non_voice_top_level_object_is_not_read_as_voices(self) -> None:
        assert _parse(json.dumps({"answer": {"name": "Ada Brook"}})) == {}

    def test_the_spoken_spelling_of_a_stated_name_is_that_name(self) -> None:
        voices = {"SPEAKER_00": "Welcome to the show, I'm Ada Allaway.", "SPEAKER_01": "Thanks."}
        reply = json.dumps({"voices": {"SPEAKER_0": {"name": "Ada Allaway", "role": "host"}}})
        got = resolve_voices_and_roles(
            ["Ada Alloway"], voices, lambda _p: reply, known_hosts=["Ada Alloway"]
        )
        assert got["SPEAKER_00"].name == "Ada Alloway"

    def test_a_name_far_from_every_stated_name_is_still_discarded(self) -> None:
        voices = {"SPEAKER_00": "Welcome.", "SPEAKER_01": "Thanks."}
        reply = json.dumps({"voices": {"SPEAKER_00": {"name": "Elon Musk", "role": "guest"}}})
        got = resolve_voices_and_roles(["Ada Alloway"], voices, lambda _p: reply)
        assert all(v.name is None for v in got.values())


class TestThisIsIsNotASelfIntroduction:
    """Advisor review: "this is <full name>" is how a HOST introduces a GUEST (the roster already
    records the hazard). Framed as "INTRODUCES ITSELF as them", "this is Matthew Cobb's seventh
    book" told the model the host was the guest."""

    def test_this_is_a_name_is_a_mention_not_a_self_introduction(self) -> None:
        turns = [("S1", "Getting warmed up here, this is Ada Brook's seventh book."), ("S2", "Hi.")]
        hits = retrieve_mentions("Ada Brook", turns)
        assert hits and "INTRODUCES ITSELF" not in hits[0]
        assert "probably NOT" in hits[0]

    def test_im_a_name_is_still_a_self_introduction(self) -> None:
        hits = retrieve_mentions("Ada Brook", [("S1", "And I'm Ada Brook.")])
        assert hits and "INTRODUCES ITSELF" in hits[0]
