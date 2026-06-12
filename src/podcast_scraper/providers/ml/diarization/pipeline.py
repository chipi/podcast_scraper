"""Apply diarization to Whisper transcription results."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .... import config
from .alignment import align_segments_to_speakers
from .cache import (
    diarization_cache_dir_for_output,
    diarization_cache_path,
    load_cached_diarization,
    save_diarization_cache,
)
from .factory import create_diarization_provider
from .roster import resolve_speaker_roster

logger = logging.getLogger(__name__)


def _resolve_diarization_cache_dir(cfg: config.Config, cache_dir: Optional[str]) -> Optional[str]:
    if cache_dir:
        return cache_dir
    return diarization_cache_dir_for_output(cfg.output_dir)


def apply_diarization_to_result(
    result: dict,
    audio_path: str,
    cfg: config.Config,
    detected_speaker_names: Optional[List[str]],
    *,
    cache_dir: Optional[str] = None,
) -> dict:
    """Enrich transcription segments with diarized speaker labels."""
    segments = result.get("segments")
    if not isinstance(segments, list) or not segments:
        return result

    resolved_cache_dir = _resolve_diarization_cache_dir(cfg, cache_dir)
    diarization = None
    if resolved_cache_dir:
        cache_path = diarization_cache_path(audio_path, cfg, resolved_cache_dir)
        diarization = load_cached_diarization(cache_path)
        if diarization is not None:
            logger.info("Diarization cache hit: %s", os.path.basename(cache_path))

    if diarization is None:
        provider = create_diarization_provider(cfg)
        diarization = provider.diarize(
            audio_path,
            num_speakers=cfg.diarization_num_speakers,
            min_speakers=cfg.diarization_min_speakers,
            max_speakers=cfg.diarization_max_speakers,
        )
        if resolved_cache_dir:
            save_diarization_cache(
                diarization_cache_path(audio_path, cfg, resolved_cache_dir),
                diarization,
            )

    if not diarization.segments:
        # No speaker turns (silent/music-only audio, or a pyannote no-op). Returning
        # the result unchanged leaves segments without speaker_label, so the caller's
        # has_diarized_labels gate degrades to gap-based formatting instead of
        # attributing the whole episode to a phantom SPEAKER_00.
        logger.warning(
            "Diarization produced no speaker turns for %s; "
            "skipping speaker labels (gap-based formatting will be used).",
            os.path.basename(audio_path),
        )
        return result

    # Resolve every diarized voice once via the unified roster (#876): host = intro-dominant,
    # named by transcript self-intro ("I'm Patrick O'Shaughnessy") → config known_hosts;
    # guests by talk-time; leftovers kept raw; a guest's name never lands on a host. For
    # network-published feeds the host name isn't in the metadata (the author tag is the
    # network), so the transcript self-intro the roster reads is the only reliable source.
    transcript_text = result.get("text") or " ".join(
        str(seg.get("text", "")) for seg in segments if isinstance(seg, dict)
    )
    roster = resolve_speaker_roster(
        diarization,
        transcript_text,
        detected_guests=detected_speaker_names or [],
        known_hosts=list(getattr(cfg, "known_hosts", None) or []),
    )
    aligned = align_segments_to_speakers(segments, diarization)

    enriched_segments: List[Dict[str, Any]] = []
    for segment, speaker_id in aligned:
        enriched = dict(segment)
        enriched["speaker"] = speaker_id
        enriched["speaker_label"] = roster.label_for(speaker_id)
        enriched_segments.append(enriched)

    enriched_result = dict(result)
    enriched_result["segments"] = enriched_segments
    enriched_result["diarization_num_speakers"] = roster.num_speakers
    return enriched_result
