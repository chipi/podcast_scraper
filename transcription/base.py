"""TranscriptionProvider protocol definition.

This module defines the protocol that all transcription providers must implement.
"""

from __future__ import annotations

from typing import Protocol


class TranscriptionProvider(Protocol):
    """Protocol for transcription providers.

    All transcription providers must implement this protocol to ensure
    consistent interface across different implementations (Whisper, OpenAI, etc.).
    """

    def initialize(self) -> None:
        """Initialize provider (load models, setup API clients, etc.).

        This method should be called before transcribe() is used.
        It may be called multiple times safely (idempotent).
        """
        ...

    def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr")

        Returns:
            Transcribed text as string

        Raises:
            RuntimeError: If provider is not initialized
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails
        """
        ...

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

        This method returns the complete transcription result including segments,
        timestamps, and other metadata. Useful for screenplay formatting and
        advanced processing.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr")

        Returns:
            Tuple of (result_dict, elapsed_time):
            - result_dict: Dictionary with keys:
              - "text": str - Full transcribed text
              - "segments": List[dict] - List of segment dictionaries with:
                - "start": float - Start time in seconds
                - "end": float - End time in seconds
                - "text": str - Segment text
              - Other provider-specific metadata
            - elapsed_time: float - Transcription time in seconds

        Raises:
            RuntimeError: If provider is not initialized
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails

        Note:
            Default implementation calls transcribe() and constructs a minimal
            result dict. Providers should override for full segment support.
        """
        # Default implementation: call transcribe() and construct minimal result
        import time

        start_time = time.time()
        text = self.transcribe(audio_path, language)
        elapsed = time.time() - start_time
        return {"text": text, "segments": []}, elapsed

    def cleanup(self) -> None:
        """Cleanup provider resources (unload models, close connections, etc.).

        This method should be called when the provider is no longer needed.
        It may be called multiple times safely (idempotent).
        """
        ...
