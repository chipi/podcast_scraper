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
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails
        """
        ...
