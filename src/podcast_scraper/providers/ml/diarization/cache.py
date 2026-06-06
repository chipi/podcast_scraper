"""Disk cache for pyannote diarization results (RFC-058 Phase 4)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

from .... import config
from ....utils.log_redaction import format_exception_for_log
from .base import DiarizationResult, DiarizationSegment

logger = logging.getLogger(__name__)

DIARIZATION_CACHE_SUBDIR = ".cache/diarization"

_HASH_CHUNK_BYTES = 1024 * 1024


def diarization_audio_hash(audio_path: str) -> str:
    """Content hash of the **whole** audio file for diarization cache keys.

    The transcript cache hashes only the first 1MB for speed, which is unsafe
    here: diarization analyses the entire file, so two distinct episodes that
    share an identical intro (a common podcast bumper, or identical leading
    silence after ``loudnorm``) would collide on a 1MB hash and serve each
    other's full-file diarization. We stream the entire file plus its byte
    length into the digest; the cost is negligible next to the pyannote pass.
    """
    hasher = hashlib.sha256()
    try:
        hasher.update(str(os.path.getsize(audio_path)).encode("utf-8"))
        with open(audio_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                hasher.update(chunk)
    except OSError as exc:
        logger.warning(
            "Failed to hash audio file for diarization cache: %s",
            format_exception_for_log(exc),
        )
        # Path fallback keeps behaviour deterministic without crashing the run.
        hasher.update(audio_path.encode("utf-8"))
    return hasher.hexdigest()[:16]


def diarization_config_fingerprint(cfg: config.Config) -> str:
    """Stable hash of diarization settings that affect model output."""
    parts = (
        cfg.diarization_model,
        str(cfg.diarization_num_speakers),
        str(cfg.diarization_min_speakers),
        str(cfg.diarization_max_speakers),
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


def diarization_cache_dir_for_output(output_dir: Optional[str]) -> Optional[str]:
    """Return cache directory under corpus output, or None when unavailable."""
    if not output_dir or not str(output_dir).strip():
        return None
    return os.path.join(str(output_dir).strip(), DIARIZATION_CACHE_SUBDIR)


def diarization_cache_path(
    audio_path: str,
    cfg: config.Config,
    cache_dir: str,
) -> str:
    """Path to cached JSON for this audio + diarization config."""
    audio_hash = diarization_audio_hash(audio_path)
    config_fp = diarization_config_fingerprint(cfg)
    return os.path.join(cache_dir, f"{audio_hash}_{config_fp}.json")


def load_cached_diarization(cache_path: str) -> Optional[DiarizationResult]:
    """Load diarization result from cache file."""
    if not os.path.isfile(cache_path):
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list):
            return None
        segments = [
            DiarizationSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                speaker=str(item["speaker"]),
            )
            for item in raw_segments
            if isinstance(item, dict) and "speaker" in item
        ]
        return DiarizationResult(
            segments=segments,
            num_speakers=int(payload.get("num_speakers") or len({s.speaker for s in segments})),
            model_name=str(payload.get("model_name") or ""),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "Failed to read diarization cache %s: %s",
            cache_path,
            format_exception_for_log(exc),
        )
        return None


def save_diarization_cache(cache_path: str, result: DiarizationResult) -> None:
    """Persist diarization result to cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "segments": [
            {"start": seg.start, "end": seg.end, "speaker": seg.speaker} for seg in result.segments
        ],
        "num_speakers": result.num_speakers,
        "model_name": result.model_name,
    }
    try:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.debug("Wrote diarization cache: %s", cache_path)
    except OSError as exc:
        logger.warning(
            "Failed to write diarization cache %s: %s",
            cache_path,
            format_exception_for_log(exc),
        )
