"""Unified DeepSeek provider for speaker detection and summarization.

This module provides a single DeepSeekProvider class that implements two protocols:
- SpeakerDetector (using DeepSeek chat API via OpenAI SDK)
- SummarizationProvider (using DeepSeek chat API via OpenAI SDK)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Note: This module follows the unified provider pattern established by OpenAIProvider.
"""

from .deepseek_provider import DeepSeekProvider

__all__ = ["DeepSeekProvider"]
