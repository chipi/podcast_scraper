"""Unified ML provider for transcription, speaker detection, and summarization.

This module provides a single MLProvider class that implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using:
- Whisper for transcription
- spaCy for speaker detection
- Transformers for summarization

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared ML libraries.
"""

from .ml_provider import MLProvider

__all__ = ["MLProvider"]
