"""A host who never introduces themselves and names nobody asks the questions (#2366).

Every host rung reads words: a self-introduction, a branded intro, a greeting by name, the presenter
formula. A host who says none of them was never seated, so the show's one stated host went unnamed
and most of the episode was unattributed. Step 7 of `_select_host_voices` seats, as a last resort,
the voice that asks questions at least twice as often as every other substantial voice, and only
when the pool holds exactly one name for the forced naming to give it.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import (
    _question_lead_voice,
    _select_host_voices,
)

pytestmark = pytest.mark.unit

ASKER = " ".join(["What drew you to the work? How did it start? And then what happened."] * 12)
ANSWERER = " ".join(
    ["It started in the archive. I read every letter. Then I wrote it up. It took years."] * 12
)


def _diar(talk):
    segs, t = [], 0.0
    for voice, secs in talk.items():
        segs.append(DiarizationSegment(start=t, end=t + secs, speaker=voice))
        t += secs
    return DiarizationResult(segments=segs, num_speakers=len(talk))


def _call(texts=None, talk=None, **over):
    texts = texts or {"SPEAKER_00": ASKER, "SPEAKER_01": ANSWERER}
    talk = talk or {"SPEAKER_00": 600.0, "SPEAKER_01": 1800.0}
    kw = dict(
        diarization=_diar(talk),
        voice_intro={},
        host_pool=[("Ada Brightwell", "feed")],
        known_hosts=["Ada Brightwell"],
        conv_hosts=[],
        conv_guests=set(),
        voices_by_intro=[],
        llm_named=set(),
        llm_voice_roles=None,
        content_start=0.0,
        intro_window_s=0.0,
        ad_intervals=None,
        ad_voices=set(),
        voice_texts=texts,
    )
    kw.update(over)
    return _select_host_voices(**kw)


def test_the_voice_that_asks_the_questions_is_seated() -> None:
    assert _call() == ["SPEAKER_00"]


def test_two_stated_hosts_leave_it_to_the_other_rungs() -> None:
    # The forced naming needs exactly one name; with two, a seat could not be named.
    pool = [("Ada Brightwell", "feed"), ("Ben Okafor", "feed")]
    assert _call(host_pool=pool, known_hosts=[n for n, _ in pool]) == []


def test_a_show_with_no_stated_host_is_left_alone() -> None:
    assert _call(host_pool=[], known_hosts=[]) == []


def test_an_unpublishable_pool_name_is_not_a_host_to_seat() -> None:
    assert _call(host_pool=[("Speaker 1", "feed")], known_hosts=["Speaker 1"]) == []


@pytest.mark.parametrize(
    "guard",
    [
        {"conv_guests": {"SPEAKER_00"}},  # heard as a guest
        {"ad_voices": {"SPEAKER_00"}},  # out of the talk share, so never a candidate
        {"llm_voice_roles": {"SPEAKER_00": "guest"}},
        {"voice_intro": {"SPEAKER_00": "Cara Lindqvist"}},  # says a name outside the pool
    ],
)
def test_the_step_six_guards_hold(guard) -> None:
    assert "SPEAKER_00" not in _call(**guard)


def test_a_seated_host_is_never_joined_by_a_second() -> None:
    got = _call(conv_hosts=["SPEAKER_01"], voice_intro={"SPEAKER_01": "Ada Brightwell"})
    assert got == ["SPEAKER_01"]


def test_no_margin_no_seat() -> None:
    both_ask = {"SPEAKER_00": ASKER, "SPEAKER_01": ASKER}
    assert _question_lead_voice(both_ask, {"SPEAKER_00": 0.5, "SPEAKER_01": 0.5}) is None


def test_the_rate_is_per_word_not_per_voice() -> None:
    # A long talker with many questions in absolute terms but few per word does not win.
    long_answer = ANSWERER * 20 + " Why?" * 40  # 40 questions to the asker's 24
    texts = {"SPEAKER_00": ASKER, "SPEAKER_01": long_answer}
    assert _question_lead_voice(texts, {"SPEAKER_00": 0.2, "SPEAKER_01": 0.8}) == "SPEAKER_00"


def test_a_small_voice_is_not_a_candidate() -> None:
    texts = {"SPEAKER_00": ASKER, "SPEAKER_01": ANSWERER, "SPEAKER_02": "Why? Really? How?"}
    share = {"SPEAKER_00": 0.3, "SPEAKER_01": 0.65, "SPEAKER_02": 0.05}
    assert _question_lead_voice(texts, share) == "SPEAKER_00"


def test_one_substantial_voice_has_nobody_to_out_ask() -> None:
    assert _question_lead_voice({"SPEAKER_00": ASKER * 3}, {"SPEAKER_00": 0.95}) is None


def test_an_unpunctuated_transcript_says_nothing_about_questions() -> None:
    bare = {"SPEAKER_00": "what drew you to it", "SPEAKER_01": "it started in the archive"}
    assert _question_lead_voice(bare, {"SPEAKER_00": 0.4, "SPEAKER_01": 0.6}) is None


def test_question_marks_alone_are_not_a_punctuated_transcript() -> None:
    # A window punctuated only where Whisper heard a question says little about who asks more.
    texts = {
        "SPEAKER_00": "what drew you to it? and then? why",
        "SPEAKER_01": "it started in the archive",
    }
    assert _question_lead_voice(texts, {"SPEAKER_00": 0.4, "SPEAKER_01": 0.6}) is None


def test_nobody_asking_anything_is_no_signal() -> None:
    flat = {"SPEAKER_00": ANSWERER, "SPEAKER_01": ANSWERER}
    assert _question_lead_voice(flat, {"SPEAKER_00": 0.4, "SPEAKER_01": 0.6}) is None
