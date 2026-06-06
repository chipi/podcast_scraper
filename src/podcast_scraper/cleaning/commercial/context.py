"""Helpers to pass diarization context into commercial cleaning."""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

from ...utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


def load_transcript_segments(full_transcript_path: str) -> Optional[List[dict]]:
    """Load sibling ``.segments.json`` when present."""
    segments_path = os.path.splitext(full_transcript_path)[0] + ".segments.json"
    if not os.path.isfile(segments_path):
        return None
    try:
        with open(segments_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug(
            "Could not load transcript segments from %s: %s",
            segments_path,
            format_exception_for_log(exc),
        )
        return None
    if not isinstance(raw, list):
        return None
    segments = [item for item in raw if isinstance(item, dict)]
    return segments or None


def segments_have_speaker_ids(segments: Optional[List[dict]]) -> bool:
    """True when segments include pyannote-style ``speaker`` ids."""
    if not segments:
        return False
    return any(isinstance(seg.get("speaker"), str) and seg.get("speaker") for seg in segments)


def _speaking_time_by_speaker(
    segments: List[dict], window_end: Optional[float]
) -> dict[str, float]:
    """Total speaking duration per speaker, optionally limited to segments starting
    before ``window_end``."""
    totals: dict[str, float] = {}
    for segment in segments:
        speaker = segment.get("speaker")
        if not isinstance(speaker, str) or not speaker:
            continue
        start = float(segment.get("start") or 0.0)
        if window_end is not None and start > window_end:
            continue
        duration = max(0.0, float(segment.get("end") or start) - start)
        totals[speaker] = totals.get(speaker, 0.0) + duration
    return totals


def infer_host_speaker_id(segments: List[dict]) -> Optional[str]:
    """Heuristic host speaker id from diarized segments, weighted by intro speaking time.

    The host usually dominates the intro, so sum each speaker's *duration* (not turn
    count, which over-weights a guest who interjects in many short segments) over the
    first 15%% of the episode; fall back to overall speaking time. Returns None when
    there are no labelled speakers. The result is a best-effort guess (a cold open
    spoken by a guest can still misfire) and is logged for debugging.
    """
    end_time = max(float(seg.get("end") or seg.get("start") or 0.0) for seg in segments) or 1.0
    intro_end = end_time * 0.15

    intro_totals = _speaking_time_by_speaker(segments, intro_end)
    chosen = intro_totals or _speaking_time_by_speaker(segments, None)
    if not chosen:
        return None
    host = max(chosen, key=lambda speaker: chosen[speaker])
    logger.debug(
        "Inferred commercial host speaker '%s' (intro-weighted=%s) from %d segments",
        host,
        bool(intro_totals),
        len(segments),
    )
    return host


def diarization_cleaning_context(
    full_transcript_path: str,
) -> Tuple[Optional[List[dict]], Optional[str]]:
    """Load segments + host speaker id for commercial Phase 2 cleaning."""
    segments = load_transcript_segments(full_transcript_path)
    if not segments_have_speaker_ids(segments):
        return None, None
    assert segments is not None
    host_id = infer_host_speaker_id(segments)
    return segments, host_id
