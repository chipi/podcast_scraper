"""A sponsor mention must not delete the whole transcript (#1179).

`_paragraph_end` already refused to run to end-of-text when a transcript has no blank lines — "so
one sponsor mention does not wipe the entire transcript", as its own comment says. The same
protection was never applied to `_paragraph_start`, which returned 0: the start of the episode.

Screenplay transcripts separate speaker turns with a SINGLE newline, so a whole episode is one
"paragraph". A sponsor read late in the episode therefore produced a block of [0, match+800], and
the cleaner deleted everything before it. Measured on the DGX pilot: ten transcripts of 33k-96k
chars, every one cleaned to **0 chars** — after which summary, GI and KG each ran happily on an
empty string and the run reported green.

The cleaned transcript is the input to every downstream LLM stage, so destroying it destroys all
of them at once.
"""

from __future__ import annotations

import pytest

from podcast_scraper.cleaning import PatternBasedCleaner

pytestmark = pytest.mark.unit

_SPONSOR = (
    "Host: This episode is brought to you by Acme. Go to acme dot com slash deal "
    "to support the show."
)


def _screenplay(turns: int = 200, sponsor_at: int = 150) -> str:
    """A diarized transcript: speaker turns on single newlines, one sponsor read late on."""
    lines = []
    for i in range(turns):
        if i == sponsor_at:
            lines.append(_SPONSOR)
        else:
            speaker = "Host" if i % 2 else "Guest"
            lines.append(
                f"{speaker}: We were discussing the economy and interest rates, point {i}."
            )
    return "\n".join(lines)


def test_a_late_sponsor_read_does_not_delete_the_episode() -> None:
    """The bug: everything before the sponsor read was deleted. Real episodes went to 0 chars."""
    raw = _screenplay()
    cleaned = PatternBasedCleaner().clean(raw)

    assert cleaned, "the cleaner returned an empty transcript"
    # Pre-fix: ~0% on real episodes (21% on this fixture). Cleaning removes ads, not the episode —
    # a cleaner that keeps under half the transcript is destroying it, not cleaning it.
    assert len(cleaned) > 0.5 * len(raw), (
        f"cleaning destroyed the transcript: {len(raw)} -> {len(cleaned)} chars. "
        "Every downstream LLM stage would then run on the remnant."
    )


def test_content_before_the_sponsor_read_survives() -> None:
    """Exactly what `_paragraph_start` returning 0 destroyed."""
    cleaned = PatternBasedCleaner().clean(_screenplay())
    assert "point 10" in cleaned
    assert "point 100" in cleaned
    assert "point 149" in cleaned  # right up to the sponsor read


def test_the_sponsor_read_itself_is_still_removed() -> None:
    """The fix must not neuter the cleaner — the ad still goes."""
    cleaned = PatternBasedCleaner().clean(_screenplay())
    assert "acme dot com slash deal" not in cleaned.lower()
