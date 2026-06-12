"""Apply diarization to Whisper transcription results."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .... import config
from ....speaker_detectors.hosts import extract_self_introduced_host
from .alignment import align_segments_to_speakers
from .cache import (
    diarization_cache_dir_for_output,
    diarization_cache_path,
    load_cached_diarization,
    save_diarization_cache,
)
from .factory import create_diarization_provider
from .mapping import map_speakers_to_names

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

    # Marry the host's transcript self-introduction ("I'm Patrick O'Shaughnessy") to the
    # diarized host speaker — for network-published feeds the host name isn't in the metadata
    # (the author tag is the network), so the intro is the only reliable source (#876).
    transcript_text = result.get("text") or " ".join(
        str(seg.get("text", "")) for seg in segments if isinstance(seg, dict)
    )
    host_name = extract_self_introduced_host(transcript_text)
    speaker_map = map_speakers_to_names(
        diarization, detected_speaker_names or [], host_name=host_name
    )
    aligned = align_segments_to_speakers(segments, diarization)

    enriched_segments: List[Dict[str, Any]] = []
    for segment, speaker_id in aligned:
        enriched = dict(segment)
        enriched["speaker"] = speaker_id
        enriched["speaker_label"] = speaker_map.get(speaker_id, speaker_id)
        enriched_segments.append(enriched)

    enriched_result = dict(result)
    enriched_result["segments"] = enriched_segments
    enriched_result["diarization_num_speakers"] = diarization.num_speakers
    return enriched_result
