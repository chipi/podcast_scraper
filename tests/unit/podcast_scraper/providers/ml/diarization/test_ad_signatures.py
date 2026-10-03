"""Cross-show ad signatures: each rule its own case, including the cases that must NOT be ads.

All fixtures are synthetic (never-commit-real-episodes).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from podcast_scraper.providers.ml.diarization import ad_signatures as A
from podcast_scraper.providers.ml.diarization.base import DiarizationResult, DiarizationSegment
from podcast_scraper.providers.ml.diarization.roster import classify_voices

pytestmark = pytest.mark.unit

HOST = (
    "Welcome back to the show. Today we look at the river trade and why it moved north, and "
    "what the merchants who lived through it wrote down about the change in their letters. "
) * 25  # >600 words: a host is never judged or indexed, as on the real corpus
AD = (
    "Build your website in minutes with the builder that does the work for you. Start your free "
    "trial today and get a domain on us for the first year of your plan."
)
GERMAN_ONE_OFF = (
    "Wir sind seit zehn Jahren für Sie da und auch noch jetzt mit dem besten Angebot. Ich sage "
    "es Ihnen, das ist nicht teuer und die Lieferung ist auch schnell bei Ihnen."
)
GERMAN_RECURRING = (
    "Die richtige Idee ist da, aber es fehlt an Zeit. Mit dem Baukasten setzen Sie sich und "
    "Ihre Idee schnell um, auch noch jetzt für Sie und ich bin mit dabei."
)
SPANISH_CLIP = (
    "No vamos a dejar que un grupo detenga el país. Hoy presento los resultados de la reforma "
    "que el pueblo pidió y que el gobierno cumplió con todo el compromiso de este año también."
)

Episode = Tuple[str, str, Dict[str, str]]


def _ep(feed: str, n: int, **voices: str) -> Episode:
    return (feed, f"{feed}-{n}", {"SPEAKER_00": HOST, **voices})


def _sig(episodes: List[Episode]) -> A.AdSignatures:
    doc = A.build(episodes)
    return A.AdSignatures(frozenset(doc["recurring"]), frozenset(doc["ad_languages"]))


# --- recurrence ------------------------------------------------------------------------------


def test_text_in_three_episodes_of_two_shows_is_an_ad() -> None:
    sig = _sig([_ep("a", 1, SPEAKER_01=AD), _ep("a", 2, SPEAKER_01=AD), _ep("b", 1, SPEAKER_01=AD)])
    assert sig.is_ad_voice(AD, "en")


def test_a_cross_posted_twin_is_not_recurrence() -> None:
    # The same episode published in two feeds is 2 episodes, not an ad seen across the corpus.
    guest = "So the ports moved north when the river silted up, and the trade went with them all."
    sig = _sig([_ep("a", 1, SPEAKER_01=guest), _ep("b", 1, SPEAKER_01=guest)])
    assert not sig.is_ad_voice(guest, "en")


def test_text_one_show_repeats_is_not_a_cross_show_ad() -> None:
    # A feed's own intro repeats every week; #1188 owns that, not the cross-show rule.
    sig = _sig([_ep("a", n, SPEAKER_01=AD) for n in range(5)])
    assert not sig.is_ad_voice(AD, "en")


def test_a_host_length_voice_is_never_judged() -> None:
    sig = _sig([_ep(f, n, SPEAKER_01=AD) for f in "ab" for n in range(3)])
    assert not sig.is_ad_voice(HOST + AD, "en")


# --- learned ad languages ----------------------------------------------------------------------


def _with_languages() -> A.AdSignatures:
    eps = [_ep(f, n, SPEAKER_01=GERMAN_RECURRING) for f in "abcd" for n in range(3)]
    eps += [_ep("latam", n, SPEAKER_01=SPANISH_CLIP.replace("año", f"año {n}")) for n in range(12)]
    return _sig(eps)


def test_a_language_whose_foreign_voices_recur_is_learned_as_an_ad_language() -> None:
    assert _with_languages().ad_languages == frozenset({"de"})


def test_a_one_off_voice_in_a_learned_ad_language_is_an_ad() -> None:
    assert _with_languages().is_ad_voice(GERMAN_ONE_OFF, "en")


def test_a_foreign_language_that_does_not_recur_stays_content() -> None:
    assert not _with_languages().is_ad_voice(SPANISH_CLIP, "en")


def test_the_ad_language_is_not_an_ad_in_a_show_in_that_language() -> None:
    assert not _with_languages().is_ad_voice(GERMAN_ONE_OFF, "de")


def test_english_is_not_read_as_portuguese() -> None:
    # Shared short words ("a", "de", "no") once taught the corpus a false ad language.
    assert A.detect_language(A.words(HOST)) == "en"


# --- the corpus file -----------------------------------------------------------------------------


def _write_episode(root: Path, feed: str, n: int, voices: Dict[str, str]) -> None:
    run = root / "feeds" / feed / f"run_{n}"
    (run / "metadata").mkdir(parents=True)
    (run / "transcripts").mkdir()
    rel = f"transcripts/{n}.txt"
    (run / "metadata" / f"{n}.metadata.json").write_text(
        json.dumps(
            {"episode": {"episode_id": f"{feed}-{n}"}, "content": {"transcript_file_path": rel}}
        )
    )
    segs, t = [], 0.0
    for v, text in voices.items():
        segs.append({"speaker": v, "start": t, "end": t + 30.0, "text": text})
        t += 30.0
    (run / "transcripts" / f"{n}.segments.json").write_text(json.dumps(segs))


def test_the_corpus_file_round_trips_and_is_found_from_a_feed_run_dir(tmp_path: Path) -> None:
    for f in ("a", "b"):
        for n in range(2):
            _write_episode(tmp_path, f, n, {"SPEAKER_00": HOST, "SPEAKER_01": AD})
    assert A.write_for_corpus(tmp_path, force=True) is not None
    sig = A.load_near(str(tmp_path / "feeds" / "a" / "run_0"))
    assert sig is not None and sig.is_ad_voice(AD, "en")


def test_a_fresh_file_is_not_rebuilt(tmp_path: Path) -> None:
    dest = tmp_path / "search" / A.FILENAME
    dest.parent.mkdir()
    dest.write_text(
        json.dumps({"schema_version": A.SCHEMA_VERSION, "recurring": [], "ad_languages": []})
    )
    before = dest.stat().st_mtime
    A.write_for_corpus(tmp_path)
    assert dest.stat().st_mtime == before


def test_a_stale_file_is_rebuilt(tmp_path: Path) -> None:
    dest = tmp_path / "search" / A.FILENAME
    dest.parent.mkdir()
    dest.write_text("{}")
    old = time.time() - A.MAX_AGE_S - 60
    os.utime(dest, (old, old))
    A.write_for_corpus(tmp_path)
    assert json.loads(dest.read_text())["schema_version"] == A.SCHEMA_VERSION


def test_no_file_means_no_opinion(tmp_path: Path) -> None:
    assert A.load_near(str(tmp_path / "feeds" / "a")) is None


# --- the classifier ------------------------------------------------------------------------------


def _dz(texts: Dict[str, str]) -> DiarizationResult:
    segs, t = [], 0.0
    for v in texts:
        dur = 600.0 if v == "SPEAKER_00" else 40.0
        segs.append(DiarizationSegment(start=t, end=t + dur, speaker=v))
        t += dur
    return DiarizationResult(segments=segs, num_speakers=len(texts))


def test_the_classifier_puts_a_signature_voice_in_ad() -> None:
    texts = {"SPEAKER_00": HOST, "SPEAKER_01": AD}
    sig = _sig([_ep("a", 1, SPEAKER_01=AD), _ep("a", 2, SPEAKER_01=AD), _ep("b", 1, SPEAKER_01=AD)])
    cleaning = classify_voices(_dz(texts), None, voice_texts=texts, ad_signatures=sig)
    assert "SPEAKER_01" in cleaning.ad
    assert "SPEAKER_00" not in cleaning.ad


def test_the_classifier_without_signatures_is_unchanged() -> None:
    texts = {"SPEAKER_00": HOST, "SPEAKER_01": AD}
    assert "SPEAKER_01" not in classify_voices(_dz(texts), None, voice_texts=texts).ad
