"""Unified OpenAI provider for transcription, speaker detection, and summarization.

This module provides a single OpenAIProvider class that implements all three protocols:
- TranscriptionProvider (using Whisper API)
- SpeakerDetector (using GPT API)
- SummarizationProvider (using GPT API)

This unified approach matches the pattern of ML providers, where a single
provider type handles multiple capabilities using shared API client.

Note: This module was moved from root-level openai/ to providers/openai/ for better
organization. All provider implementations are now under providers/.
"""

from .openai_provider import OpenAIProvider

__all__ = ["OpenAIProvider"]
