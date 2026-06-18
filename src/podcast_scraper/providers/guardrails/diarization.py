"""Pyannote-style diarization guardrail (any provider).

Empty diarization output for non-trivial audio is structurally invalid
(every non-empty audio has at least one speech segment). Preventive
— no observed cases yet but the failure mode is structurally identical
to the chat-empty and whisper-empty cases.
"""

from __future__ import annotations

from typing import Any, Optional

from ._telemetry import raise_violation

REASON_DIARIZATION_EMPTY_SEGMENTS = "empty_segments"

# Below this duration the empty-segments check is skipped because pyannote
# legitimately can return no segments for very short / near-silent clips.
_MIN_DURATION_FOR_CHECK_SEC = 5.0


def check_pyannote_response(
    segments: list[Any],
    *,
    audio_duration_sec: Optional[float],
    service: str = "pyannote",
) -> None:
    """Raise :class:`..exceptions.GuardrailViolation` if the diarization output
    is empty for audio longer than ~5 s.

    Args:
        segments: the diarization segment list.
        audio_duration_sec: audio duration in seconds. When None or < 5 s,
            the check is skipped (the empty case can be legitimate for
            very short audio).
        service: service identifier. Defaults to ``"pyannote"``.
    """
    if not segments and audio_duration_sec and audio_duration_sec > _MIN_DURATION_FOR_CHECK_SEC:
        summary = f"duration_sec={audio_duration_sec:.1f} segments=[]"
        raise_violation(service, REASON_DIARIZATION_EMPTY_SEGMENTS, summary)


__all__ = ["REASON_DIARIZATION_EMPTY_SEGMENTS", "check_pyannote_response"]
