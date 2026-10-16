"""Which language an episode is in, when its transcript does not read as its show's (#2369).

The transcript check (``languages_guard.transcript_language_contradiction``) already says when an
episode is NOT in its show's language; it cannot say reliably WHICH language it is (function words
read Portuguese as Italian and Italian as Portuguese). The ASR can, but the service echoes the
language it is asked for, so it must be asked with none, on short clips.

A clip is chosen from a stretch the transcript shows is not the show's language, so an ad, a quote
or a cameo in the show's language is never the one listened to. Three clips spread over the
episode must agree; anything less is no answer, and the episode is refused as before.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

#: Whisper decides the language from the first 30 s of what it is given.
CLIP_SECONDS = 30.0
#: Clips listened to per episode; all must agree.
CLIPS = 3
#: A stretch with fewer words says too little about its language to be chosen.
MIN_STRETCH_WORDS = 30


def _share(ws: Sequence[str], function_words: frozenset) -> float:
    return sum(w in function_words for w in ws) / len(ws) if ws else 0.0


def _start(seg: Mapping[str, object]) -> float:
    return float(seg.get("start") or 0.0)  # type: ignore[arg-type]


def identification_clips(
    segments: Sequence[Mapping[str, object]], declared: Optional[str]
) -> list[Tuple[float, float]]:
    """``CLIPS`` ``(start, end)`` windows of about ``CLIP_SECONDS`` from stretches that do not read
    as ``declared``, spread early / middle / late; ``[]`` when there are fewer such stretches."""
    from ..languages import primary_language
    from ..languages_guard import DECLARED_LANGUAGE_MIN_SHARE
    from ..providers.ml.diarization.ad_signatures import FUNCTION_WORDS, words

    own = FUNCTION_WORDS.get(primary_language(declared) or "")
    if not own:
        return []
    stretches: list[Tuple[float, float, list[str]]] = []
    start: Optional[float] = None
    ws: list[str] = []
    for seg in sorted(segments, key=_start):
        a, b = _start(seg), float(seg.get("end") or 0.0)  # type: ignore[arg-type]
        start = a if start is None else start
        ws += words(str(seg.get("text") or ""))
        if b - start >= CLIP_SECONDS:
            stretches.append((start, min(b, start + CLIP_SECONDS), ws))
            start, ws = None, []
    other = [
        (a, b)
        for a, b, w in stretches
        if len(w) >= MIN_STRETCH_WORDS and _share(w, own) < DECLARED_LANGUAGE_MIN_SHARE
    ]
    if len(other) < CLIPS:
        return []
    return [other[(2 * k + 1) * len(other) // (2 * CLIPS)] for k in range(CLIPS)]


def agreed_language(answers: Sequence[Optional[str]]) -> Optional[str]:
    """The language every clip answered, or ``None`` when any clip disagreed or gave no answer."""
    langs = {str(a).strip().lower() for a in answers if a and str(a).strip()}
    if len(answers) < CLIPS or len(langs) != 1 or any(not a for a in answers):
        return None
    return langs.pop()
