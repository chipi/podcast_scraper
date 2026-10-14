"""A punctuated re-decode that leaves out hesitations has not LOST speech (#2187, 2026-10-14).

Re-running the 48 refused prod windows: on Reiner Pope's episode 32 "um" and 6 "uh" alone pushed
the retained share to 0.895 and refused a clean repair. Fillers no longer count; real words do.
"""

from __future__ import annotations

import pytest

from podcast_scraper.transcription.punctuation_repair import _verdict

pytestmark = pytest.mark.unit

SPAN = (0.0, 60.0)
OLD_WORDS = (
    "this thing was called z i can recover y by z minus f of x because i have already "
    "recovered x so that means that this construction is invertible this was used in "
    "ciphers like a ton and it still is used today in many places"
).split()


def _seg(text: str, start: float = 0.0, end: float = 60.0):
    return {"start": start, "end": end, "text": text}


def _old_with(fillers: int) -> str:
    words = list(OLD_WORDS)
    for i in range(fillers):
        words.insert((i * 3) % len(words), "um" if i % 2 else "uh")
    return " ".join(words)


NEW = (
    "This thing was called z. I can recover y by z minus f of x, because I have already "
    "recovered x. So that means that this construction is invertible. This was used in "
    "ciphers, like, a ton. And it still is used today in many places."
)


def test_dropped_fillers_are_not_lost_words() -> None:
    old = _old_with(fillers=12)  # 12 of 57 words: retained would be ~0.79 counting them
    assert _verdict([_seg(old)], [_seg(NEW)], NEW, SPAN, "Hello.") is None


def test_dropped_real_words_are_still_lost_words() -> None:
    new = " ".join(OLD_WORDS[:30]) + "."
    assert _verdict([_seg(" ".join(OLD_WORDS))], [_seg(new)], new, SPAN, "Hello.") == "lost_words"
