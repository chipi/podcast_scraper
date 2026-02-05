"""Unified Mistral provider for transcription, speaker detection, and summarization.

This module provides a single MistralProvider class that implements all three protocols:
- TranscriptionProvider (using Mistral Voxtral API)
- SpeakerDetector (using Mistral chat API)
- SummarizationProvider (using Mistral chat API)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Note: This module follows the unified provider pattern established by OpenAIProvider.
"""

from .mistral_provider import MistralProvider

__all__ = ["MistralProvider"]
