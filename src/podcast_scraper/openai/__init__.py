"""Unified OpenAI provider for transcription, speaker detection, and summarization.

This module provides a single OpenAIProvider class that implements all three protocols
(TranscriptionProvider, SpeakerDetector, SummarizationProvider) using OpenAI's API:
- Whisper API for transcription
- GPT API for speaker detection
- GPT API for summarization

This unified approach matches the pattern of ML providers, where a single
provider type handles multiple capabilities using shared API client.
"""

from .openai_provider import OpenAIProvider

__all__ = ["OpenAIProvider"]
