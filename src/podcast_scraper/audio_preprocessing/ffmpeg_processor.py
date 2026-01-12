"""FFmpeg-based audio preprocessor implementation."""

import hashlib
import logging
import shutil
import subprocess
import time
from typing import Tuple

logger = logging.getLogger(__name__)


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system.

    Returns:
        True if ffmpeg is available, False otherwise
    """
    return shutil.which("ffmpeg") is not None


class FFmpegAudioPreprocessor:
    """Audio preprocessor using ffmpeg for format conversion and silence removal."""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: str = "-50dB",
        silence_duration: float = 2.0,
        target_loudness: int = -16,
    ):
        """Initialize preprocessor with configuration.

        Args:
            sample_rate: Target sample rate in Hz (default: 16000)
                Must be one of: 8000, 12000, 16000, 24000, 48000 (Opus supported rates)
            silence_threshold: Silence detection threshold (default: -50dB)
            silence_duration: Minimum silence duration to remove in seconds (default: 2.0)
            target_loudness: Target loudness in LUFS for normalization (default: -16)
        """
        # Opus codec only supports specific sample rates: 8000, 12000, 16000, 24000, 48000
        # If an unsupported rate is provided, use the closest supported rate
        OPUS_SUPPORTED_RATES = [8000, 12000, 16000, 24000, 48000]
        if sample_rate not in OPUS_SUPPORTED_RATES:
            # Find closest supported rate
            closest_rate = min(OPUS_SUPPORTED_RATES, key=lambda x: abs(x - sample_rate))
            logger.warning(
                "Sample rate %d Hz is not supported by Opus codec. "
                "Using closest supported rate: %d Hz",
                sample_rate,
                closest_rate,
            )
            sample_rate = closest_rate

        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.target_loudness = target_loudness

        # Check ffmpeg availability
        if not _check_ffmpeg_available():
            logger.warning(
                "FFmpeg not found. Audio preprocessing will fail. "
                "Install ffmpeg to use audio preprocessing."
            )

    def preprocess(self, input_path: str, output_path: str) -> Tuple[bool, float]:
        """Preprocess audio using ffmpeg pipeline.

        Pipeline stages:
        1. Convert to mono
        2. Resample to configured sample rate (default: 16 kHz)
        3. Remove silence (VAD)
        4. Normalize loudness to configured target (default: -16 LUFS)
        5. Compress using speech-optimized codec (Opus @ 24 kbps)

        Args:
            input_path: Path to raw audio file
            output_path: Path to save preprocessed audio

        Returns:
            Tuple of (success: bool, elapsed_time: float)
        """
        if not _check_ffmpeg_available():
            logger.error("FFmpeg not available. Cannot preprocess audio.")
            return False, 0.0

        start_time = time.time()

        # Build ffmpeg command
        # -ac 1: convert to mono
        # -ar <sample_rate>: resample to configured sample rate
        # -af silenceremove: remove silence (conservative thresholds)
        # -af loudnorm: normalize loudness to configured target
        # -c:a libopus: speech-optimized codec
        # -b:a 24k: 24 kbps bitrate
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ac",
            "1",  # Mono
            "-ar",
            str(self.sample_rate),  # Sample rate
            "-af",
            (
                f"silenceremove="
                f"start_periods=1:"
                f"start_threshold={self.silence_threshold}:"
                f"start_duration={self.silence_duration}:"
                f"stop_periods=-1:"
                f"stop_threshold={self.silence_threshold}:"
                f"stop_duration={self.silence_duration},"
                f"loudnorm=I={self.target_loudness}:LRA=11:TP=-1.5"
            ),
            "-c:a",
            "libopus",  # Speech-optimized codec
            "-b:a",
            "24k",  # 24 kbps
            "-y",  # Overwrite output
            output_path,
        ]

        try:
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=True,
            )
            elapsed = time.time() - start_time
            logger.debug("Audio preprocessing completed in %.1fs", elapsed)
            return True, elapsed

        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg preprocessing failed: %s", exc.stderr)
            return False, time.time() - start_time
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg preprocessing timed out after 300s")
            return False, time.time() - start_time
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install ffmpeg to use audio preprocessing.")
            return False, 0.0

    def get_cache_key(self, input_path: str) -> str:
        """Generate cache key from file content hash + preprocessing config.

        Args:
            input_path: Path to audio file

        Returns:
            Cache key (SHA256 hash of content + config)
        """
        hasher = hashlib.sha256()

        # Hash file content (first 1MB for performance)
        try:
            with open(input_path, "rb") as f:
                hasher.update(f.read(1024 * 1024))
        except OSError as exc:
            logger.warning("Failed to hash audio file: %s", exc)
            # Use file path as fallback
            hasher.update(input_path.encode("utf-8"))

        # Hash preprocessing config to invalidate cache when settings change
        config_str = (
            f"{self.sample_rate}|"
            f"{self.silence_threshold}|"
            f"{self.silence_duration}|"
            f"{self.target_loudness}"
        )
        hasher.update(config_str.encode("utf-8"))

        return hasher.hexdigest()[:16]  # 16 hex chars (64 bits)
