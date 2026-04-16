"""Factory function for creating audio preprocessors."""

import logging
from typing import List, Optional

from .base import AudioPreprocessor
from .ffmpeg_processor import FFmpegAudioPreprocessor

logger = logging.getLogger(__name__)

# MP3 bitrate policy (GitHub #561)
_PREPROCESSING_MP3_BITRATE_MIN = 24
_PREPROCESSING_MP3_BITRATE_MAX = 128
# Default first-pass bitrate when ``preprocessing_mp3_bitrate_kbps`` is unset (auto).
_DEFAULT_MP3_KBPS_LOCAL_TRANSCRIBE = 64
_DEFAULT_MP3_KBPS_API_TRANSCRIBE = 48
# Strictly decreasing ladder used for cache probes and phase-2 re-encode steps.
_MP3_BITRATE_RUNG_DESC: List[int] = [64, 56, 48, 40, 32, 24]


def resolve_preprocessing_mp3_bitrate_kbps(cfg) -> int:
    """Resolve effective MP3 bitrate for the first full preprocess pass.

    ``None`` on ``cfg.preprocessing_mp3_bitrate_kbps`` selects **auto**:
    ``openai`` / ``gemini`` transcription → tighter default for 25 MB API caps;
    other providers (e.g. ``whisper``) → 64 kbps.

    Args:
        cfg: ``config.Config`` instance.

    Returns:
        Integer kbps in ``[24, 128]``.
    """
    raw = getattr(cfg, "preprocessing_mp3_bitrate_kbps", None)
    if raw is not None:
        return int(raw)
    tp = str(getattr(cfg, "transcription_provider", "") or "").lower()
    if tp in ("openai", "gemini"):
        return _DEFAULT_MP3_KBPS_API_TRANSCRIBE
    return _DEFAULT_MP3_KBPS_LOCAL_TRANSCRIBE


def mp3_bitrates_to_probe_for_cache(first_pass_kbps: int) -> List[int]:
    """Bitrates to probe for preprocessing cache hits (descending quality order).

    Includes ``first_pass_kbps`` plus any standard rung values at or below it so a file
    produced after phase-2 re-encode (lower bitrate) remains discoverable.

    Args:
        first_pass_kbps: First-pass bitrate from ``resolve_preprocessing_mp3_bitrate_kbps``.

    Returns:
        Sorted list high → low, values clamped to the allowed rung range.
    """
    kb = int(first_pass_kbps)
    kb = max(_PREPROCESSING_MP3_BITRATE_MIN, min(_PREPROCESSING_MP3_BITRATE_MAX, kb))
    candidates = {kb}
    for b in _MP3_BITRATE_RUNG_DESC:
        if b <= kb:
            candidates.add(b)
    return sorted(candidates, reverse=True)


def next_lower_mp3_bitrate_kbps(current_kbps: int) -> Optional[int]:
    """Next rung strictly below ``current_kbps``, or ``None`` if already at floor."""
    cur = int(current_kbps)
    lower = [b for b in reversed(_MP3_BITRATE_RUNG_DESC) if b < cur]
    if not lower:
        return None
    return max(lower)


def build_ffmpeg_preprocessor_with_bitrate(cfg, mp3_bitrate_kbps: int) -> FFmpegAudioPreprocessor:
    """Build FFmpeg preprocessor with an explicit MP3 bitrate (cache probes / keying)."""
    kb = int(mp3_bitrate_kbps)
    kb = max(_PREPROCESSING_MP3_BITRATE_MIN, min(_PREPROCESSING_MP3_BITRATE_MAX, kb))
    return FFmpegAudioPreprocessor(
        sample_rate=cfg.preprocessing_sample_rate,
        silence_threshold=cfg.preprocessing_silence_threshold,
        silence_duration=cfg.preprocessing_silence_duration,
        target_loudness=cfg.preprocessing_target_loudness,
        mp3_bitrate_kbps=kb,
    )


def create_audio_preprocessor(
    cfg,  # config.Config
) -> Optional[AudioPreprocessor]:
    """Create audio preprocessor based on configuration.

    Args:
        cfg: Configuration object with preprocessing settings

    Returns:
        AudioPreprocessor instance if enabled and available, None otherwise
    """
    if not cfg.preprocessing_enabled:
        return None

    mp3_kbps = resolve_preprocessing_mp3_bitrate_kbps(cfg)
    preprocessor = build_ffmpeg_preprocessor_with_bitrate(cfg, mp3_kbps)

    # Check if ffmpeg is available
    from .ffmpeg_processor import _check_ffmpeg_available

    if not _check_ffmpeg_available():
        logger.warning(
            "Audio preprocessing enabled but ffmpeg not found. "
            "Preprocessing will be disabled. Install ffmpeg to use audio preprocessing."
        )
        return None

    return preprocessor
