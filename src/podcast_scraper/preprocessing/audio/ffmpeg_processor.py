"""FFmpeg-based audio preprocessor implementation.

GitHub #561 tightens MP3 output for API transcription size limits; if a file is still
over the cap after stepped re-encodes, upload chunking is tracked separately (GitHub #286).
"""

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from podcast_scraper.utils.log_redaction import format_exception_for_log

logger = logging.getLogger(__name__)


def _run_text_subprocess(
    cmd: List[str],
    *,
    timeout: float,
    check: bool,
) -> "subprocess.CompletedProcess[str]":
    """Run subprocess with text mode and UTF-8 replace (GitHub #558).

    Avoids UnicodeDecodeError when ffmpeg/ffprobe write non-UTF-8 bytes to pipes.
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system.

    Returns:
        True if ffmpeg is available, False otherwise
    """
    return shutil.which("ffmpeg") is not None


def _check_ffprobe_available() -> bool:
    """Check if ffprobe is available on the system.

    Returns:
        True if ffprobe is available, False otherwise
    """
    return shutil.which("ffprobe") is not None


def extract_audio_metadata(audio_path: str) -> Optional[Dict[str, Any]]:
    """Extract audio metadata using ffprobe (Issue #387).

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with audio metadata (bitrate, sample_rate, codec, channels)
        or None if extraction fails
    """
    if not _check_ffprobe_available():
        logger.debug("FFprobe not available, skipping audio metadata extraction")
        return None

    if not audio_path or not os.path.exists(audio_path):
        logger.debug("Audio file does not exist, skipping metadata extraction: %s", audio_path)
        return None

    try:
        # Use ffprobe to extract audio stream metadata
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=bit_rate,sample_rate,codec_name,channels",
            "-of",
            "json",
            audio_path,
        ]

        result = _run_text_subprocess(cmd, timeout=10.0, check=True)

        data = json.loads(result.stdout)
        streams = data.get("streams", [])

        # Find audio stream (first stream with audio codec)
        for stream in streams:
            codec = stream.get("codec_name")
            if codec and codec.startswith(("pcm", "mp3", "aac", "opus", "flac", "wav")):
                metadata: Dict[str, Any] = {}
                if "bit_rate" in stream and stream["bit_rate"]:
                    try:
                        metadata["bitrate"] = int(stream["bit_rate"])
                    except (ValueError, TypeError):
                        pass
                if "sample_rate" in stream and stream["sample_rate"]:
                    try:
                        metadata["sample_rate"] = int(float(stream["sample_rate"]))
                    except (ValueError, TypeError):
                        pass
                if "codec_name" in stream and stream["codec_name"]:
                    metadata["codec"] = stream["codec_name"]
                if "channels" in stream and stream["channels"]:
                    try:
                        metadata["channels"] = int(stream["channels"])
                    except (ValueError, TypeError):
                        pass

                return metadata if metadata else None

        return None

    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        FileNotFoundError,
        UnicodeDecodeError,
    ) as exc:
        logger.debug("Failed to extract audio metadata: %s", exc)
        return None


class FFmpegAudioPreprocessor:
    """Audio preprocessor using ffmpeg for format conversion and silence removal."""

    def __init__(
        self,
        sample_rate: int = 16000,
        silence_threshold: str = "-50dB",
        silence_duration: float = 2.0,
        target_loudness: int = -16,
        mp3_bitrate_kbps: int = 64,
    ):
        """Initialize preprocessor with configuration.

        Args:
            sample_rate: Target sample rate in Hz (default: 16000).
                Recommended: 8000, 12000, 16000, 24000, or 48000.
            silence_threshold: Silence detection threshold (default: -50dB)
            silence_duration: Minimum silence duration to remove in seconds (default: 2.0)
            target_loudness: Target loudness in LUFS for normalization (default: -16)
            mp3_bitrate_kbps: libmp3lame constant bitrate for speech output (default: 64).
                GitHub #561: lower values for API transcription size limits; see CONFIGURATION.md.
        """
        STANDARD_RATES = [8000, 12000, 16000, 24000, 48000]
        if sample_rate not in STANDARD_RATES:
            closest_rate = min(STANDARD_RATES, key=lambda x: abs(x - sample_rate))
            logger.warning(
                "Sample rate %d Hz is non-standard for speech. "
                "Using closest standard rate: %d Hz",
                sample_rate,
                closest_rate,
            )
            sample_rate = closest_rate

        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.target_loudness = target_loudness
        self.mp3_bitrate_kbps = int(mp3_bitrate_kbps)

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
        5. Encode to MP3 via libmp3lame at ``mp3_bitrate_kbps`` (default 64; GitHub #561).

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
        # -vn: disable video (audio-only output)
        # -ac 1: convert to mono
        # -ar <sample_rate>: resample to configured sample rate
        # -af silenceremove: remove silence (conservative thresholds)
        # -af loudnorm: normalize loudness to configured target
        # -c:a libmp3lame: MP3 codec (widely supported by OpenAI API)
        # -b:a <n>k: constant bitrate for speech (GitHub #561 tunes this for API caps)
        br = int(self.mp3_bitrate_kbps)
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vn",  # No video (audio-only)
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
            "libmp3lame",  # MP3 codec (widely supported, reliable)
            "-b:a",
            f"{br}k",
            "-y",  # Overwrite output
            output_path,
        ]

        try:
            _run_text_subprocess(cmd, timeout=300.0, check=True)
            elapsed = time.time() - start_time
            logger.debug("Audio preprocessing completed in %.1fs", elapsed)
            return True, elapsed

        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg preprocessing failed: %s", exc.stderr)
            return False, time.time() - start_time
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg preprocessing timed out after 300s")
            return False, time.time() - start_time
        except UnicodeDecodeError as exc:
            logger.error(
                "FFmpeg preprocessing decode error: %s",
                format_exception_for_log(exc),
            )
            return False, time.time() - start_time
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install ffmpeg to use audio preprocessing.")
            return False, 0.0

    def reencode_mp3_to_bitrate(
        self, input_path: str, output_path: str, bitrate_kbps: int
    ) -> Tuple[bool, float]:
        """Re-encode an existing MP3 to a lower constant bitrate (GitHub #561 phase 2).

        Skips silenceremove/loudnorm; used when the full preprocess output is still over
        the API file-size budget. If still over after minimum bitrate, upload chunking may
        apply (GitHub #286 — not implemented here).

        Args:
            input_path: Path to source audio (typically already mono MP3 from preprocess).
            output_path: Destination path for re-encoded MP3.
            bitrate_kbps: Target libmp3lame bitrate in kbps.

        Returns:
            Tuple of (success, elapsed seconds).
        """
        if not _check_ffmpeg_available():
            logger.error("FFmpeg not available. Cannot re-encode audio.")
            return False, 0.0
        br = int(bitrate_kbps)
        start_time = time.time()
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            f"{br}k",
            "-y",
            output_path,
        ]
        try:
            _run_text_subprocess(cmd, timeout=300.0, check=True)
            return True, time.time() - start_time
        except subprocess.CalledProcessError as exc:
            logger.error("FFmpeg re-encode failed: %s", exc.stderr)
            return False, time.time() - start_time
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg re-encode timed out after 300s")
            return False, time.time() - start_time
        except UnicodeDecodeError as exc:
            logger.error(
                "FFmpeg re-encode decode error: %s",
                format_exception_for_log(exc),
            )
            return False, time.time() - start_time
        except FileNotFoundError:
            logger.error("FFmpeg not found during re-encode.")
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
            logger.warning(
                "Failed to hash audio file: %s",
                format_exception_for_log(exc),
            )
            # Use file path as fallback
            hasher.update(input_path.encode("utf-8"))

        # Hash preprocessing config to invalidate cache when settings change
        config_str = (
            f"{self.sample_rate}|"
            f"{self.silence_threshold}|"
            f"{self.silence_duration}|"
            f"{self.target_loudness}|"
            f"mp3={self.mp3_bitrate_kbps}"
        )
        hasher.update(config_str.encode("utf-8"))

        return hasher.hexdigest()[:16]  # 16 hex chars (64 bits)
