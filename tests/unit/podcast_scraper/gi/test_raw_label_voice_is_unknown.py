"""A raw-labelled voice with no voice type is a voice we failed to name (RFC-131, S3, write time).

Measured on the served corpus (2026-10-30): 15 of the 21 episodes whose insights showed a raw
label as speaker were written after #2126, the latest on 2026-10-27. A segment carrying
``speaker_label: "SPEAKER_07"`` and no ``voice_type`` fell through to a Person node named after the
label, and its insights stayed surfaceable. The read-time fix hides what is served; this stops new
episodes producing it.
"""

from __future__ import annotations

from podcast_scraper.gi.pipeline import _voice_type_for_char_range

_TRANSCRIPT = "Hello there. I agree with that."
_SEGMENTS = [
    {"start": 0.0, "end": 1.0, "text": "Hello there. ", "speaker_label": "Jorna Velt"},
    {"start": 1.0, "end": 2.0, "text": "I agree with that.", "speaker_label": "SPEAKER_07"},
]


def test_a_raw_label_without_a_voice_type_is_unknown() -> None:
    start = _TRANSCRIPT.index("I agree")
    assert _voice_type_for_char_range(_TRANSCRIPT, start, start + 5, _SEGMENTS) == "unknown"


def test_the_detector_default_without_a_voice_type_is_unknown() -> None:
    segs = [dict(_SEGMENTS[0]), {**_SEGMENTS[1], "speaker_label": "unknown_guest_1"}]
    start = _TRANSCRIPT.index("I agree")
    assert _voice_type_for_char_range(_TRANSCRIPT, start, start + 5, segs) == "unknown"


def test_a_named_voice_is_still_a_person() -> None:
    assert _voice_type_for_char_range(_TRANSCRIPT, 0, 5, _SEGMENTS) is None


def test_a_typed_voice_keeps_its_type() -> None:
    segs = [dict(_SEGMENTS[0]), {**_SEGMENTS[1], "voice_type": "cameo"}]
    start = _TRANSCRIPT.index("I agree")
    assert _voice_type_for_char_range(_TRANSCRIPT, start, start + 5, segs) == "cameo"
