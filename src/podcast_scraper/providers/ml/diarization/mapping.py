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
    """Map diarized speaker IDs to detected **guest** names by total speaking time.

    ``detected_names`` are guest names only — host names are stripped upstream in
    ``_detect_speakers_for_episode`` (the host is identified separately). The single caller
    (``apply_diarization_to_result``) always passes guest-only names.

    The most-talking speaker within the first ``intro_window_s`` is treated as the host and
    **keeps its raw ``SPEAKER_xx`` label**, so a guest's name is never painted onto the
    host's turns (#876 — previously the first guest name landed on the host slot, mis-
    attributing the host's intro/questions to the guest and leaving the real guest unnamed).
    Non-host speakers are named with ``detected_names`` in descending total-speaking-time
    order; any speaker past the supplied names keeps its raw label.
    """
    if not diarization.segments:
        return {}

    intro_times = _speaking_time_by_speaker(diarization, window_end=intro_window_s)
    total_times = _speaking_time_by_speaker(diarization)
    by_time = sorted(total_times, key=lambda sid: total_times[sid], reverse=True)

    host_id = max(intro_times, key=lambda key: intro_times[key]) if intro_times else None

    mapping: Dict[str, str] = {}
    guest_index = 0
    for speaker_id in by_time:
        if speaker_id == host_id:
            mapping[speaker_id] = speaker_id  # host kept raw; named separately if known
        elif guest_index < len(detected_names):
            mapping[speaker_id] = detected_names[guest_index]
            guest_index += 1
        else:
            mapping[speaker_id] = speaker_id
    return mapping
