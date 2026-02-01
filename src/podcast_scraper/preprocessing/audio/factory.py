"""Factory function for creating audio preprocessors."""

import logging
from typing import Optional

from .base import AudioPreprocessor
from .ffmpeg_processor import FFmpegAudioPreprocessor

logger = logging.getLogger(__name__)


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

    # Currently only FFmpeg processor is supported
    preprocessor = FFmpegAudioPreprocessor(
        sample_rate=cfg.preprocessing_sample_rate,
        silence_threshold=cfg.preprocessing_silence_threshold,
        silence_duration=cfg.preprocessing_silence_duration,
        target_loudness=cfg.preprocessing_target_loudness,
    )

    # Check if ffmpeg is available
    from .ffmpeg_processor import _check_ffmpeg_available

    if not _check_ffmpeg_available():
        logger.warning(
            "Audio preprocessing enabled but ffmpeg not found. "
            "Preprocessing will be disabled. Install ffmpeg to use audio preprocessing."
        )
        return None

    return preprocessor
