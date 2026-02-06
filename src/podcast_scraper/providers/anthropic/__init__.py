"""Anthropic provider for transcription, speaker detection, and summarization.

This package provides AnthropicProvider, a unified provider that implements:
- TranscriptionProvider (Note: Anthropic doesn't support native audio)
- SpeakerDetector (using Claude chat models)
- SummarizationProvider (using Claude chat models)
"""

from .anthropic_provider import AnthropicProvider

__all__ = ["AnthropicProvider"]
