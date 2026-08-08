"""vLLM provider — a first-class OpenAI-compatible serving stack for the DGX-local open-model
family (Qwen/DeepSeek/Llama). A sibling of the OpenAI provider, not a subclass (ADR-147)."""

from .vllm_provider import VLLMProvider

__all__ = ["VLLMProvider"]
