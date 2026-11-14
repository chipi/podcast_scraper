"""Whisper-style transcription guardrail.

Catches the WER=1.0 garbage-content + empty-response failure modes observed
in #996 under GPU contention. Works against any whisper-style API
(self-hosted via faster-whisper-server, openai-whisper, the OpenAI cloud
Whisper API, etc.) — the check is on the returned transcript text + the
expected word count derived from audio duration.
"""

from __future__ import annotations

import re
from typing import Optional

from ._telemetry import raise_violation

REASON_TRANSCRIPTION_EMPTY = "empty_response"
REASON_TRANSCRIPTION_LENGTH_FLOOR = "length_floor_violated"

# Whisper length floor: 50% of expected word count, where the speaking rate
# estimate is 2.5 words/sec (median podcast pace). Catches the WER=1.0 mode
# without false-firing on natural silence stretches.
_WORDS_PER_SEC_ESTIMATE = 2.5
_LENGTH_FLOOR_FRACTION = 0.5

# Chinese and Japanese are written without spaces between words, so split() reads a full transcript
# as a few hundred "words": ChinaTalk's 中文版 episode (c18f57c6, 3816 s) failed the floor every
# night with 21,749 characters of Chinese, before the language guard could refuse it. Each of these
# characters counts as a word here.
_UNSPACED_SCRIPT = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]")


def _word_count(text: str) -> int:
    unspaced = len(_UNSPACED_SCRIPT.findall(text))
    if not unspaced:
        return len(text.split())
    return len(_UNSPACED_SCRIPT.sub(" ", text).split()) + unspaced


def check_whisper_response(
    text: str, *, audio_duration_sec: Optional[float], service: str = "whisper"
) -> None:
    """Raise :class:`..exceptions.GuardrailViolation` if the transcript fails
    its structural sanity check.

    Args:
        text: the returned transcript text.
        audio_duration_sec: audio duration in seconds. When None (probe
            failed), the length floor is skipped — only the empty-response
            check fires.
        service: service identifier. Defaults to ``"whisper"`` (one shared
            label for any whisper-style API today; if we ever wire multiple
            whisper backends side-by-side we'd specialize this).
    """
    if text is None or text == "":
        raise_violation(service, REASON_TRANSCRIPTION_EMPTY, "")
    if audio_duration_sec is None or audio_duration_sec <= 0:
        return
    word_count = _word_count(text)
    floor = int(audio_duration_sec * _WORDS_PER_SEC_ESTIMATE * _LENGTH_FLOOR_FRACTION)
    if word_count < floor:
        summary = f"word_count={word_count} floor={floor} " f"duration_sec={audio_duration_sec:.1f}"
        raise_violation(service, REASON_TRANSCRIPTION_LENGTH_FLOOR, summary)


__all__ = [
    "REASON_TRANSCRIPTION_EMPTY",
    "REASON_TRANSCRIPTION_LENGTH_FLOOR",
    "check_whisper_response",
]
