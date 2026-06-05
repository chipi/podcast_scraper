"""Map anonymous diarization speaker IDs to detected names."""

from __future__ import annotations

from typing import Dict, List

from .base import DiarizationResult

INTRO_WINDOW_SECONDS = 90.0


def _speaking_time_by_speaker(
    diarization: DiarizationResult,
    *,
    window_end: float | None = None,
) -> Dict[str, float]:
    totals: Dict[str, float] = {}
    for segment in diarization.segments:
        if window_end is not None and segment.start >= window_end:
            continue
        end = segment.end if window_end is None else min(segment.end, window_end)
        if end <= segment.start:
            continue
        duration = end - segment.start
        totals[segment.speaker] = totals.get(segment.speaker, 0.0) + duration
    return totals


def map_speakers_to_names(
    diarization: DiarizationResult,
    detected_names: List[str],
    *,
    intro_window_s: float = INTRO_WINDOW_SECONDS,
) -> Dict[str, str]:
    """Map diarized speaker IDs to detected names by intro + total speaking time."""
    if not diarization.segments:
        return {}

    intro_times = _speaking_time_by_speaker(diarization, window_end=intro_window_s)
    total_times = _speaking_time_by_speaker(diarization)
    speaker_ids = sorted(total_times, key=lambda sid: total_times[sid], reverse=True)

    ordered_ids: List[str] = []
    if intro_times:
        host_id = max(intro_times, key=lambda key: intro_times[key])
        ordered_ids.append(host_id)
        speaker_ids = [sid for sid in speaker_ids if sid != host_id]
    ordered_ids.extend(speaker_ids)

    mapping: Dict[str, str] = {}
    for index, speaker_id in enumerate(ordered_ids):
        if index < len(detected_names):
            mapping[speaker_id] = detected_names[index]
        else:
            mapping[speaker_id] = speaker_id
    return mapping
