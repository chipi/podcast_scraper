"""Groq native provider for speaker detection and summarization (ADR-147).

A single :class:`GroqProvider` class implements both the SpeakerDetector and SummarizationProvider
protocols over the shared OpenAI-compatible transport, serving Groq's hosted model catalog (Llama,
gpt-oss, Qwen3, DeepSeek-R1-distill, and more) over its low-latency OpenAI-compatible API. Sibling
of the vLLM/DeepSeek/Qwen/LiteLLM providers; carries its own ``groq`` telemetry namespace so cost
is attributed to Groq.
"""

from .groq_provider import GroqProvider

__all__ = ["GroqProvider"]
