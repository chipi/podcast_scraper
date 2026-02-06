"""Unified Gemini provider for transcription, speaker detection, and summarization.

This module provides a single GeminiProvider class that implements all three protocols:
- TranscriptionProvider (using native multimodal audio understanding)
- SpeakerDetector (using Gemini chat models)
- SummarizationProvider (using Gemini chat models)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Note: This module follows the unified provider pattern established by OpenAIProvider.
"""

from .gemini_provider import GeminiProvider

__all__ = ["GeminiProvider"]
