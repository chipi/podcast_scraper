"""Provider capability contract system.

This module provides a capability-based interface for providers, allowing
the pipeline to check provider capabilities instead of using provider name checks.

This enables:
- Graceful degradation when capabilities are missing
- Provider-agnostic code paths
- Better extensibility for new providers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderCapabilities:
    """Provider capability contract.

    This dataclass defines what capabilities a provider supports, allowing
    the pipeline to make decisions based on capabilities rather than provider names.

    Attributes:
        supports_transcription: Whether provider supports audio transcription
        supports_speaker_detection: Whether provider supports speaker detection
        supports_summarization: Whether provider supports text summarization
        supports_audio_input: Whether provider accepts audio files directly (vs text only)
        supports_json_mode: Whether provider supports structured JSON output mode
        max_context_tokens: Maximum context window size in tokens
        supports_tool_calls: Whether provider supports function/tool calling
        supports_system_prompt: Whether provider supports system prompts
        supports_streaming: Whether provider supports streaming responses
        provider_name: Human-readable provider name for logging/debugging
    """

    supports_transcription: bool
    supports_speaker_detection: bool
    supports_summarization: bool
    supports_audio_input: bool
    supports_json_mode: bool
    max_context_tokens: int
    supports_tool_calls: bool = True  # Most modern LLMs support this
    supports_system_prompt: bool = True  # Most modern LLMs support this
    supports_streaming: bool = False  # Not currently used, but available
    provider_name: str = "unknown"

    def __str__(self) -> str:
        """Human-readable string representation."""
        capabilities = []
        if self.supports_transcription:
            capabilities.append("transcription")
        if self.supports_speaker_detection:
            capabilities.append("speaker_detection")
        if self.supports_summarization:
            capabilities.append("summarization")
        if self.supports_audio_input:
            capabilities.append("audio_input")
        if self.supports_json_mode:
            capabilities.append("json_mode")
        return (
            f"{self.provider_name}({', '.join(capabilities)}, max_tokens={self.max_context_tokens})"
        )


def get_provider_capabilities(provider: Any) -> ProviderCapabilities:
    """Get capabilities for a provider instance.

    This function uses introspection to detect provider capabilities.
    Providers can optionally implement a `get_capabilities()` method for
    explicit capability declaration, otherwise capabilities are inferred.

    Args:
        provider: Provider instance (MLProvider, OpenAIProvider, etc.)

    Returns:
        ProviderCapabilities object describing provider capabilities

    Note:
        If provider has a `get_capabilities()` method, it's called directly.
        Otherwise, capabilities are inferred from provider attributes and type.
    """
    # Check if provider has explicit capability method
    if hasattr(provider, "get_capabilities") and callable(provider.get_capabilities):
        try:
            result = provider.get_capabilities()
            if isinstance(result, ProviderCapabilities):
                return result
            # If it's not the right type, fall through to inference
        except Exception as e:
            logger.warning(
                f"Provider {type(provider).__name__}.get_capabilities() raised error: {e}. "
                "Falling back to introspection."
            )

    # Fallback: infer capabilities from provider type and attributes
    return _infer_capabilities(provider)


def is_local_provider(provider: Any) -> bool:
    """Check if provider is local (ML-based) vs API-based.

    Args:
        provider: Provider instance

    Returns:
        True if provider is local (ML-based), False if API-based
    """
    provider_type = type(provider).__name__
    # MLProvider is the only local provider
    return provider_type == "MLProvider"


def _infer_capabilities(provider: Any) -> ProviderCapabilities:
    """Infer provider capabilities from provider instance.

    Args:
        provider: Provider instance

    Returns:
        ProviderCapabilities inferred from provider
    """
    provider_type = type(provider).__name__
    provider_name = provider_type.replace("Provider", "").lower()

    # Check if provider implements protocols (capability detection)
    has_transcribe = hasattr(provider, "transcribe") or hasattr(
        provider, "transcribe_with_segments"
    )
    has_detect_speakers = hasattr(provider, "detect_speakers")
    has_summarize = hasattr(provider, "summarize")

    # Get max_context_tokens if available
    max_tokens = getattr(provider, "max_context_tokens", 0)

    # Determine JSON mode support
    # Providers that use OpenAI SDK typically support JSON mode
    supports_json = False
    if provider_type in ("OpenAIProvider", "GrokProvider", "DeepSeekProvider", "OllamaProvider"):
        supports_json = True
    elif provider_type == "GeminiProvider":
        # Gemini supports JSON mode via response_schema
        supports_json = True
    elif provider_type in ("AnthropicProvider", "MistralProvider"):
        # Anthropic and Mistral support JSON mode
        supports_json = True
    elif provider_type == "MLProvider":
        # ML providers don't support JSON mode (text-only output)
        supports_json = False

    # Determine audio input support
    # Only transcription providers that accept audio files directly
    supports_audio = has_transcribe

    # Determine tool calls support
    # Most modern LLM providers support tool calls
    supports_tools = provider_type not in ("MLProvider",)

    # Determine system prompt support
    # Most modern LLM providers support system prompts
    supports_system = provider_type not in ("MLProvider",)

    return ProviderCapabilities(
        supports_transcription=has_transcribe,
        supports_speaker_detection=has_detect_speakers,
        supports_summarization=has_summarize,
        supports_audio_input=supports_audio,
        supports_json_mode=supports_json,
        max_context_tokens=max_tokens,
        supports_tool_calls=supports_tools,
        supports_system_prompt=supports_system,
        supports_streaming=False,  # Not currently used
        provider_name=provider_name,
    )
