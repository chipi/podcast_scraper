"""A voice in the ad-geography language is an ad, in a show that is not in that language.

Publishers geo-target dynamic ad insertion by the IP that fetches the audio, and production fetches
from a German host. Measured 2026-10-02: 146 German-language voices in English shows, every one an
ad on hand review; 77 were classified as real speakers and 19 carried a host's or guest's name.

Each shape is its own case, including the ones that must NOT become ads.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

from typing import Dict

import pytest

from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import classify_voices

pytestmark = pytest.mark.unit

ENGLISH = (
    "Welcome back to the show. Today we talk about the history of the river trade and why it "
    "moved north. It is a story that the merchants of the time told in their letters, and we "
    "have read all of them for this episode, so stay with us for the whole of it."
)
GERMAN_AD = (
    "Die richtige Idee ist da, aber es fehlt an Zeit. Mit unserem Baukasten setzen Sie Ihre "
    "Idee schnell und einfach um, ob App oder Webseite, und wir sind auch noch jetzt für Sie da."
)
# An episode is overwhelmingly the show; an ad is a few percent of it.
HOST = ENGLISH * 30
SPANISH_CLIP = (
    "No vamos a dejar que un grupo detenga el país. Hoy presento los resultados de la reforma "
    "que el pueblo pidió y que el gobierno cumplió con todo el compromiso de este año."
)


def _classify(texts: Dict[str, str]):
    segs = []
    t = 0.0
    for v, text in texts.items():
        dur = 60.0 if v == "SPEAKER_00" else 40.0
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=v))
        t += dur
    dz = DiarizationResult(segments=segs, num_speakers=len(texts))
    return classify_voices(dz, None, voice_texts=texts, ordered_turns=list(texts.items()))


def test_a_german_ad_in_an_english_show_is_an_ad() -> None:
    cleaning = _classify({"SPEAKER_00": HOST, "SPEAKER_01": GERMAN_AD})
    assert "SPEAKER_01" in cleaning.ad
    assert "SPEAKER_00" not in cleaning.ad


def test_a_spanish_clip_in_an_english_show_is_not_an_ad() -> None:
    # On a Latin America show the Spanish and Portuguese tape IS the content.
    cleaning = _classify({"SPEAKER_00": HOST, "SPEAKER_01": SPANISH_CLIP})
    assert "SPEAKER_01" not in cleaning.ad


def test_german_voices_in_a_german_show_are_the_content() -> None:
    cleaning = _classify({"SPEAKER_00": GERMAN_AD * 3, "SPEAKER_01": GERMAN_AD})
    assert not cleaning.ad


def test_a_few_german_words_are_not_enough() -> None:
    cleaning = _classify({"SPEAKER_00": HOST, "SPEAKER_01": "Okay, das ist gut. Danke."})
    assert "SPEAKER_01" not in cleaning.ad


def test_an_english_speaker_who_uses_one_german_phrase_is_not_an_ad() -> None:
    cleaning = _classify(
        {"SPEAKER_00": HOST, "SPEAKER_01": ENGLISH + " As they say, das ist nicht gut."}
    )
    assert "SPEAKER_01" not in cleaning.ad
