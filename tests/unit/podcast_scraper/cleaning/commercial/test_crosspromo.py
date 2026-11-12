"""Opening host-read cross-promo excision (#1188).

The Athletic cross-promo survives every commercial/density pass because it carries
none of the usual markers (no promo code, no URL, no "brought to you by") — it is
two named people introducing their beat for a sister property. The distinctive
*diarization* shape is the signal: a leading block spoken by voices that never
recur later in the episode. These tests drive the shipped cleaning path
(``PatternBasedCleaner().clean``) with the real data contract: labelled transcript
text plus a sibling-style ``.segments.json`` list carrying pyannote speaker ids.
"""

from __future__ import annotations

import pytest

from podcast_scraper.cleaning.pattern_based import PatternBasedCleaner

pytestmark = pytest.mark.unit


def _athletic_crosspromo_transcript() -> tuple[str, list[dict]]:
    """A Hard-Fork-shaped episode: Athletic cross-promo read by two non-recurring
    voices, then the real two-host conversation. Returns (text, segments)."""
    opening = (
        "Paul Tenorio: I'm Paul Tenorio. I cover soccer for The Athletic.\n"
        "Amy Lawrence: And I'm Amy Lawrence. I cover football for The Athletic.\n"
        "Paul Tenorio: Whatever you call it, the biggest competition in the sport "
        "is happening right now.\n"
        "Amy Lawrence: We've got more than 70 obsessive reporters on the ground "
        "bringing you every angle.\n"
        "Paul Tenorio: Throughout the tournament you have free access to all the "
        "coverage in our app.\n"
    )
    body_turns = [
        "Today we're talking about AI agents and what actually changed this week.",
        "The new models can hold a plan across dozens of tool calls now.",
        "That reliability jump is the whole story for enterprise adoption.",
        "Regulators are starting to ask about autonomy and liability too.",
    ]
    body = ""
    for i in range(12):
        t = body_turns[i % len(body_turns)]
        body += f"Kevin Roose: {t}\n"
        body += f"Casey Newton: Right, and here's the counterpoint number {i}.\n"
    text = opening + body

    # Diarization segments (pyannote ids). Opening = SPEAKER_90/91, ~0-26s, who
    # never speak again. Body = SPEAKER_00 (Kevin, host) / SPEAKER_01 (Casey),
    # 26s onward across the rest of the episode.
    segments: list[dict] = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_90",
            "text": "I'm Paul Tenorio. I cover soccer for The Athletic.",
        },
        {
            "start": 5.0,
            "end": 10.0,
            "speaker": "SPEAKER_91",
            "text": "And I'm Amy Lawrence. I cover football for The Athletic.",
        },
        {
            "start": 10.0,
            "end": 16.0,
            "speaker": "SPEAKER_90",
            "text": (
                "Whatever you call it, the biggest competition in the sport "
                "is happening right now."
            ),
        },
        {
            "start": 16.0,
            "end": 21.0,
            "speaker": "SPEAKER_91",
            "text": (
                "We've got more than 70 obsessive reporters on the ground "
                "bringing you every angle."
            ),
        },
        {
            "start": 21.0,
            "end": 26.0,
            "speaker": "SPEAKER_90",
            "text": (
                "Throughout the tournament you have free access to all the " "coverage in our app."
            ),
        },
    ]
    clock = 26.0
    for i in range(12):
        segments.append(
            {
                "start": clock,
                "end": clock + 20.0,
                "speaker": "SPEAKER_00",
                "text": body_turns[i % len(body_turns)],
            }
        )
        clock += 20.0
        segments.append(
            {
                "start": clock,
                "end": clock + 12.0,
                "speaker": "SPEAKER_01",
                "text": f"Right, and here's the counterpoint number {i}.",
            }
        )
        clock += 12.0
    return text, segments


def test_recurring_host_intro_is_not_excised() -> None:
    """Guard against over-firing: hosts who introduce themselves at the top but
    recur throughout must NOT be treated as a cross-promo."""
    text = (
        "Kevin Roose: I'm Kevin Roose.\n"
        "Casey Newton: And I'm Casey Newton, and this is Hard Fork.\n"
    )
    segments = [
        {"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "I'm Kevin Roose."},
        {
            "start": 4.0,
            "end": 9.0,
            "speaker": "SPEAKER_01",
            "text": "And I'm Casey Newton, and this is Hard Fork.",
        },
    ]
    clock = 9.0
    for i in range(8):
        text += f"Kevin Roose: Point number {i} about the news.\n"
        segments.append(
            {
                "start": clock,
                "end": clock + 15.0,
                "speaker": "SPEAKER_00",
                "text": f"Point number {i} about the news.",
            }
        )
        clock += 15.0

    cleaned = PatternBasedCleaner().clean(
        text,
        diarization_segments=segments,
        host_speaker_id="SPEAKER_00",
    )

    assert "Kevin Roose" in cleaned
    assert "Casey Newton" in cleaned
    assert "I'm Kevin Roose" in cleaned


def _ad_with_only_feedspecific_promo() -> tuple[str, list[dict]]:
    """A cross-promo whose ONLY promotional word is feed-specific (not in the
    English-general defaults) — used to prove the config extension seam."""
    opening = (
        "Dana Lee: I'm Dana Lee. I cover markets for the Terminal Brief.\n"
        "Sam Ortiz: And I'm Sam Ortiz, on the markets desk.\n"
        "Dana Lee: Grab our terminal digest for the whole trading week.\n"
    )
    segments = [
        {
            "start": 0.0,
            "end": 5.0,
            "speaker": "SPEAKER_90",
            "text": "I'm Dana Lee. I cover markets for the Terminal Brief.",
        },
        {
            "start": 5.0,
            "end": 10.0,
            "speaker": "SPEAKER_91",
            "text": "And I'm Sam Ortiz, on the markets desk.",
        },
        {
            "start": 10.0,
            "end": 16.0,
            "speaker": "SPEAKER_90",
            "text": "Grab our terminal digest for the whole trading week.",
        },
    ]
    body = ""
    clock = 16.0
    for i in range(10):
        body += f"Kevin Roose: Real content point {i} about the news today.\n"
        segments.append(
            {
                "start": clock,
                "end": clock + 15.0,
                "speaker": "SPEAKER_00",
                "text": f"Real content point {i} about the news today.",
            }
        )
        clock += 15.0
    return opening + body, segments
