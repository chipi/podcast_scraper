"""Unified Grok provider for speaker detection and summarization.

This module provides a single GrokProvider class that implements two protocols:
- SpeakerDetector (using Grok chat API via OpenAI SDK)
- SummarizationProvider (using Grok chat API via OpenAI SDK)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key insight: Grok uses an OpenAI-compatible API, so we reuse the OpenAI SDK
with a custom base_url. No new dependency required.

Key advantage: Real-time information access via X/Twitter integration.

Note: Grok does NOT support transcription (no audio API).

Note: This module follows the unified provider pattern established by OpenAIProvider.
"""

from .grok_provider import GrokProvider

__all__ = ["GrokProvider"]
