"""Base protocol for audio preprocessing providers."""

from typing import Protocol, Tuple


class AudioPreprocessor(Protocol):
    """Protocol for audio preprocessing providers."""

    def preprocess(
        self,
        input_path: str,
        output_path: str,
    ) -> Tuple[bool, float]:
        """Preprocess audio file for transcription.

        Args:
            input_path: Path to raw audio file
            output_path: Path to save preprocessed audio

        Returns:
            Tuple of (success: bool, elapsed_time: float)
        """
        ...

    def get_cache_key(self, input_path: str) -> str:
        """Generate content-based cache key for input audio.

        Args:
            input_path: Path to audio file

        Returns:
            Cache key (hash of audio content + preprocessing settings)
        """
        ...
