"""Audio preprocessing module for optimizing audio files before transcription.

This module provides audio preprocessing capabilities to reduce file size
and optimize audio for transcription providers, particularly for API-based
providers with file size limits (e.g., OpenAI 25 MB limit).
"""

from .factory import create_audio_preprocessor

__all__ = ["create_audio_preprocessor"]
