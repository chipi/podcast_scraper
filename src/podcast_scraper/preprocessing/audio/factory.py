"""Factory function for creating audio preprocessors."""

import logging
from typing import List, Optional

from .base import AudioPreprocessor
from .ffmpeg_processor import FFmpegAudioPreprocessor

logger = logging.getLogger(__name__)


class FFmpegUnavailableError(RuntimeError):
    """``preprocessing_enabled`` is on and ffmpeg is not installed. Not recoverable in-process.

    Deliberately fatal rather than a degrade-and-continue. Operating without a component we
    declared we need is not a scenario worth anticipating — the fix is always "install ffmpeg
    and re-run", and every episode on the host is affected identically, so continuing only
    produces a corpus that has to be redone.

    The soft path this replaces was worse than it looked. ffmpeg does the loudness
    normalisation, silence trim, mono/sample-rate conversion and MP3 bitrate reduction that get
    an episode under the cloud STT provider's 25 MB upload cap. Without it, every episode was
    uploaded raw: more expensive, worse transcripts, and some rejected outright for size — with
    a single factory-level WARNING as the only trace.

    This mistake has been made twice already, both times by reasoning that some variant did not
    need ffmpeg:

      5330582a (2026-04-24)  shipped the pipeline image with a conditional install and the
                             comment "ffmpeg is only needed for local Whisper transcription
                             (ML mode) / If using LLM-only mode, ffmpeg is not required". The
                             cloud profiles set ``preprocessing_enabled: true`` and preprocess
                             client-side, so that was false.
      954bac8f (2026-04-26)  fixed it two days later, after stack-test cloud-thin runs failed
                             with "ffmpeg is not installed or not in PATH".
      7d208f3e (2026-06-07)  the same omission again, in the builder stage, for pyannote.

    Note what caught it both times: a HARD failure. ``cli._validate_ffmpeg`` (#379) already
    exits at startup for any pipeline run. Raising here makes the library path agree with the
    front door instead of quietly disagreeing with it.
    """


# MP3 bitrate policy (GitHub #561)
_PREPROCESSING_MP3_BITRATE_MIN = 24
_PREPROCESSING_MP3_BITRATE_MAX = 128
# Default first-pass bitrate when ``preprocessing_mp3_bitrate_kbps`` is unset (auto).
_DEFAULT_MP3_KBPS_LOCAL_TRANSCRIBE = 64
_DEFAULT_MP3_KBPS_API_TRANSCRIBE = 48
# Strictly decreasing ladder used for cache probes and phase-2 re-encode steps.
_MP3_BITRATE_RUNG_DESC: List[int] = [64, 56, 48, 40, 32, 24]


def preprocessing_fingerprint(cfg) -> str:
    """Identity of the audio the transcriber will actually see (#1173).

    The transcript cache is keyed on the *original* media's hash, but transcription runs on the
    *preprocessed* file — so two runs with identical audio and identical models can still produce
    different transcripts if the preprocessing changed. Folding this fingerprint into the cache key
    is what stops a preprocessing fix from silently re-serving transcripts built under the old
    settings (in #1173, transcripts whose timestamps were drifted by silence removal).

    Only the knobs that change the produced audio belong here.
    """
    if not getattr(cfg, "preprocessing_enabled", False):
        return "pp=off"
    return "|".join(
        (
            "pp=on",
            f"sr={getattr(cfg, 'preprocessing_sample_rate', '')}",
            f"silrm={bool(getattr(cfg, 'preprocessing_silence_removal', False))}",
            f"silth={getattr(cfg, 'preprocessing_silence_threshold', '')}",
            f"sildur={getattr(cfg, 'preprocessing_silence_duration', '')}",
            f"loud={getattr(cfg, 'preprocessing_target_loudness', '')}",
            f"mp3={resolve_preprocessing_mp3_bitrate_kbps(cfg)}",
        )
    )


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
        silence_removal=getattr(cfg, "preprocessing_silence_removal", False),
    )


def create_audio_preprocessor(
    cfg,  # config.Config
) -> Optional[AudioPreprocessor]:
    """Create audio preprocessor based on configuration.

    Args:
        cfg: Configuration object with preprocessing settings

    Returns:
        AudioPreprocessor instance when preprocessing is enabled. ``None`` ONLY when
        ``preprocessing_enabled`` is false — that is the operator choosing not to preprocess,
        which is a legitimate configuration.

    Raises:
        FFmpegUnavailableError: Preprocessing is enabled but ffmpeg is missing. See that
            exception for why this is fatal rather than a degrade.
    """
    if not cfg.preprocessing_enabled:
        return None

    mp3_kbps = resolve_preprocessing_mp3_bitrate_kbps(cfg)
    preprocessor = build_ffmpeg_preprocessor_with_bitrate(cfg, mp3_kbps)

    # Check if ffmpeg is available
    from .ffmpeg_processor import _check_ffmpeg_available

    if not _check_ffmpeg_available():
        # Was a WARNING + "preprocessing will be disabled", which is the one thing it must not
        # do: the operator asked for preprocessing, the box cannot do it, and continuing sends
        # every episode to the provider raw and oversized under a log line nobody reads.
        raise FFmpegUnavailableError(
            "Audio preprocessing is enabled (preprocessing_enabled: true) but ffmpeg is not "
            "installed or not in PATH. Every episode would be transcribed from unnormalised, "
            "full-size audio — more expensive, worse quality, and some rejected by the "
            "provider's upload cap. Install ffmpeg (https://ffmpeg.org/download.html) and "
            "re-run, or set preprocessing_enabled: false to accept raw audio deliberately."
        )

    return preprocessor
