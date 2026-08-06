"""Qwen native provider for speaker detection and summarization (ADR-144).

A single :class:`QwenProvider` class implements both the SpeakerDetector and SummarizationProvider
protocols over the shared OpenAI-compatible transport, serving the Qwen3 family from any
OpenAI-compatible endpoint (a cloud host or the DGX vLLM slot). Sibling of the vLLM/DeepSeek/LiteLLM
providers; carries its own ``qwen`` telemetry namespace so cost is attributed to Qwen.
"""

from .qwen_provider import QwenProvider, QwenServedModelMismatch

__all__ = ["QwenProvider", "QwenServedModelMismatch"]
