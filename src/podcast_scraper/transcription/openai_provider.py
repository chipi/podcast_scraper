"""OpenAI Whisper API transcription provider implementation.

This module provides a TranscriptionProvider implementation using OpenAI's
Whisper API for cloud-based transcription of audio files.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from .. import config

logger = logging.getLogger(__name__)


class OpenAITranscriptionProvider:
    """OpenAI Whisper API-based transcription provider.

    This provider uses OpenAI's Whisper API for cloud-based transcription.
    It implements the TranscriptionProvider protocol.

    Note:
        This provider supports parallel transcription via transcription_parallelism
        config field, as API calls can be made concurrently without shared state.
    """

    def __init__(self, cfg: config.Config):
        """Initialize OpenAI transcription provider.

        Args:
            cfg: Configuration object with openai_api_key and transcription settings

        Raises:
            ValueError: If OpenAI API key is not provided
        """
        if not cfg.openai_api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI transcription provider. "
                "Set OPENAI_API_KEY environment variable or openai_api_key in config."
            )

        self.cfg = cfg
        # Support custom base_url for E2E testing with mock servers
        client_kwargs: dict[str, Any] = {"api_key": cfg.openai_api_key}
        if cfg.openai_api_base:
            client_kwargs["base_url"] = cfg.openai_api_base
        self.client = OpenAI(**client_kwargs)
        # Default to environment-based model (whisper-1 is only option)
        self.model = getattr(
            cfg, "openai_transcription_model", config.PROD_DEFAULT_OPENAI_TRANSCRIPTION_MODEL
        )
        self._initialized = False

    def initialize(self) -> None:
        """Initialize provider (no local model loading needed for API).

        This method is called to prepare the provider for use.
        For OpenAI API, initialization is a no-op but we track it for consistency.
        """
        if self._initialized:
            return

        logger.debug("Initializing OpenAI transcription provider (model: %s)", self.model)
        # Verify API key works by making a test call (optional, can be skipped)
        # For now, we'll just mark as initialized
        self._initialized = True
        logger.debug("OpenAI transcription provider initialized successfully")

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If None, uses cfg.language or auto-detects

        Returns:
            Transcribed text as string

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        if not self._initialized:
            raise RuntimeError(
                "OpenAITranscriptionProvider not initialized. Call initialize() first."
            )

        import os

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or None)

        logger.debug(
            "Transcribing audio file via OpenAI API: %s (language: %s)",
            audio_path,
            effective_language or "auto",
        )

        try:
            with open(audio_path, "rb") as audio_file:
                # OpenAI API requires keyword-only arguments
                # When response_format="text", returns str directly
                if effective_language is not None:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=effective_language,
                        response_format="text",  # Simple text response
                    )
                else:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format="text",  # Simple text response
                    )

            # transcript is a string when response_format="text"
            text = str(transcript) if not isinstance(transcript, str) else transcript

            logger.debug(
                "OpenAI transcription completed: %d characters",
                len(text) if text else 0,
            )

            return text

        except Exception as exc:
            logger.error("OpenAI Whisper API error: %s", exc)
            raise ValueError(f"OpenAI transcription failed: {exc}") from exc

    def transcribe_with_segments(
        self, audio_path: str, language: str | None = None
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

        Returns the complete OpenAI transcription result including segments
        and timestamps for screenplay formatting.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If None, uses cfg.language or auto-detects

        Returns:
            Tuple of (result_dict, elapsed_time) where result_dict contains:
            - "text": Full transcribed text
            - "segments": List of segment dicts with start, end, text
            - Other OpenAI API metadata
        """
        if not self._initialized:
            raise RuntimeError(
                "OpenAITranscriptionProvider not initialized. Call initialize() first."
            )

        import os
        import time

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or None)

        logger.debug(
            "Transcribing audio file with segments via OpenAI API: %s (language: %s)",
            audio_path,
            effective_language or "auto",
        )

        start_time = time.time()
        try:
            with open(audio_path, "rb") as audio_file:
                # Use verbose_json format to get segments
                if effective_language is not None:
                    response = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        language=effective_language,
                        response_format="verbose_json",  # Get full response with segments
                    )
                else:
                    response = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format="verbose_json",  # Get full response with segments
                    )

            elapsed = time.time() - start_time

            # OpenAI API returns a Transcription object with text and segments
            # when verbose_json is used. Convert to dict format matching Whisper output.
            # Response structure: response.text (str), response.segments (List[Segment])
            # Each Segment has: start (float), end (float), text (str)
            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    # Handle both dict-like and object-like segment structures
                    if isinstance(seg, dict):
                        segments.append(
                            {
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                                "text": seg.get("text", ""),
                            }
                        )
                    else:
                        # Object-like structure (OpenAI SDK TranscriptionSegment)
                        segments.append(
                            {
                                "start": float(getattr(seg, "start", 0.0)),
                                "end": float(getattr(seg, "end", 0.0)),
                                "text": getattr(seg, "text", ""),
                            }
                        )

            result_dict: dict[str, object] = {
                "text": response.text if hasattr(response, "text") else str(response),
                "segments": segments,
            }

            segments_list = result_dict.get("segments", [])
            segment_count = len(segments_list) if isinstance(segments_list, list) else 0
            logger.debug(
                "OpenAI transcription with segments completed in %.2fs (%d segments)",
                elapsed,
                segment_count,
            )

            return result_dict, elapsed

        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error("OpenAI Whisper API error: %s", exc)
            raise ValueError(f"OpenAI transcription failed: {exc}") from exc

    def cleanup(self) -> None:
        """Cleanup provider resources (no-op for API provider)."""
        # No resources to clean up for API provider
        pass
