"""Whisper transcription provider implementation.

This module provides a TranscriptionProvider implementation using OpenAI Whisper
for local transcription of audio files.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Import whisper_integration functions (keeping existing implementation)
from .. import config, whisper_integration


class WhisperTranscriptionProvider:
    """Whisper-based transcription provider.

    This provider uses OpenAI Whisper for local transcription of audio files.
    It implements the TranscriptionProvider protocol.

    Note:
        This provider always processes transcriptions sequentially (one at a time)
        due to memory and CPU constraints. The transcription_parallelism config
        field is ignored for Whisper provider (always uses parallelism = 1).
        For parallel transcription, use OpenAI provider (future implementation).
    """

    def __init__(self, cfg: config.Config):
        """Initialize Whisper transcription provider.

        Args:
            cfg: Configuration object with whisper_model and language settings
        """
        self.cfg = cfg
        self._model: Optional[Any] = None
        self._initialized = False

    def initialize(self) -> None:
        """Load Whisper model.

        This method loads the Whisper model using the configuration.
        It should be called before transcribe() is used.
        """
        if self._initialized:
            return

        logger.debug(
            "Initializing Whisper transcription provider (model: %s)", self.cfg.whisper_model
        )
        self._model = whisper_integration.load_whisper_model(self.cfg)
        if self._model is None:
            raise RuntimeError(
                "Failed to load Whisper model. "
                "Make sure 'openai-whisper' is installed: pip install openai-whisper"
            )
        self._initialized = True
        logger.debug("Whisper transcription provider initialized successfully")

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If None, uses cfg.language or defaults to "en"

        Returns:
            Transcribed text as string

        Raises:
            RuntimeError: If provider is not initialized
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails
        """
        if not self._initialized or self._model is None:
            raise RuntimeError(
                "WhisperTranscriptionProvider not initialized. Call initialize() first."
            )

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or "en")

        logger.debug("Transcribing audio file: %s (language: %s)", audio_path, effective_language)

        # Call transcribe_with_whisper
        # Note: transcribe_with_whisper uses cfg.language, so if a different language
        # is requested, we need to handle it. For now, we'll use cfg.language.
        # If language parameter differs from cfg.language, we log a warning.
        if language is not None and language != self.cfg.language:
            logger.warning(
                "Language parameter (%s) differs from config language (%s). "
                "Using config language for transcription.",
                language,
                self.cfg.language,
            )

        result_dict, elapsed = whisper_integration.transcribe_with_whisper(
            self._model, audio_path, self.cfg
        )

        # Extract text from result
        text = result_dict.get("text", "").strip()
        if not text:
            raise ValueError("Transcription returned empty text")

        logger.debug(
            "Transcription completed in %.2fs (text length: %d chars)",
            elapsed,
            len(text),
        )

        return str(text)  # Ensure we return str, not Any

    def transcribe_with_segments(
        self, audio_path: str, language: str | None = None
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

        Returns the complete Whisper transcription result including segments
        and timestamps for screenplay formatting.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If None, uses cfg.language or defaults to "en"

        Returns:
            Tuple of (result_dict, elapsed_time) where result_dict contains:
            - "text": Full transcribed text
            - "segments": List of segment dicts with start, end, text
            - Other Whisper metadata
        """
        if not self._initialized or self._model is None:
            raise RuntimeError(
                "WhisperTranscriptionProvider not initialized. Call initialize() first."
            )

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or "en")

        logger.debug(
            "Transcribing audio file with segments: %s (language: %s)",
            audio_path,
            effective_language,
        )

        # Call transcribe_with_whisper which returns (result_dict, elapsed)
        result_dict, elapsed = whisper_integration.transcribe_with_whisper(
            self._model, audio_path, self.cfg
        )

        logger.debug(
            "Transcription with segments completed in %.2fs (%d segments)",
            elapsed,
            len(result_dict.get("segments", [])),
        )

        return result_dict, elapsed

    def cleanup(self) -> None:
        """Cleanup resources.

        This method releases any resources held by the provider.
        For Whisper, there's no explicit cleanup needed, but we mark as uninitialized.
        """
        if not self._initialized:
            return

        logger.debug("Cleaning up Whisper transcription provider")
        # Whisper models don't need explicit cleanup (Python GC handles it)
        # But we can mark as uninitialized
        self._model = None
        self._initialized = False

    @property
    def model(self) -> Optional[Any]:
        """Get the loaded Whisper model (for backward compatibility).

        Returns:
            Loaded Whisper model or None if not initialized
        """
        return self._model

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized.

        Returns:
            True if provider is initialized, False otherwise
        """
        return self._initialized
