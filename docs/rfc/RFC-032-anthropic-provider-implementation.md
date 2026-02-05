# RFC-032: Anthropic Provider Implementation (Revised)

- **Status**: Implemented
- **Revision**: 3
- **Date**: 2026-02-04
- **Authors**:
- **Stakeholders**: Maintainers, users wanting Anthropic API integration, developers implementing providers
- **Related PRDs**:
  - `docs/prd/PRD-009-anthropic-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference - unified provider pattern)
  - `docs/rfc/RFC-021-modularization-refactoring-plan.md` (architecture foundation)
  - `docs/rfc/RFC-017-prompt-management.md` (prompt system)

## Abstract

Design and implement Anthropic Claude API as a unified provider for speaker detection and summarization capabilities. This RFC builds on the existing modularization architecture (RFC-021) and follows the **unified provider pattern** established by OpenAI (RFC-013), where a single provider class implements multiple protocols. Anthropic does NOT support audio transcription.

**Architecture Alignment:** Anthropic provider follows the exact same unified provider pattern as `OpenAIProvider`, implementing two protocols (`SpeakerDetector`, `SummarizationProvider`) in a single class and integrating via the existing factory pattern with support for both Config-based and experiment-based modes.

## Problem Statement

Users want the option to use Anthropic Claude API as an alternative to OpenAI for:

1. **Speaker Detection**: Entity extraction using Claude models
2. **Summarization**: High-quality summaries using Claude models

**Note:** Transcription is NOT supported by Anthropic (no audio API).

Requirements:

- No changes to end-user experience or workflow when using defaults
- Secure API key management (environment variables, never in source code)
- Per-capability provider selection (can mix local, OpenAI, and Anthropic)
- Build on existing modularization and provider architecture
- Handle capability gaps gracefully (transcription not supported)
- Use Anthropic-specific prompts (prompts are provider-specific)
- Support both Config-based and experiment-based factory modes

## Constraints & Assumptions

**Constraints:**

- **Prerequisite**: Modularization refactoring (RFC-021) ✅ Completed
- **Prerequisite**: OpenAI provider implementation (RFC-013) ✅ Completed
- **Backward Compatibility**: Default providers (local) must remain unchanged
- **API Key Security**: API keys must never be in source code or committed files
- **Capability Gap**: Anthropic does not support audio transcription
- **Rate Limits**: Must respect Anthropic API rate limits and implement retry logic
- **Must follow unified provider pattern** (like OpenAI)

**Assumptions:**

- Anthropic API is stable and well-documented
- Anthropic Python SDK follows similar patterns to OpenAI SDK
- Prompts need to be optimized for Claude (different from GPT prompts)
- Users understand the capability matrix (what each provider supports)

## Design & Implementation

### 0. Anthropic API Overview

Anthropic's API differs from OpenAI in several ways:

| Feature | OpenAI | Anthropic |
| ------- | ------ | --------- |
| **Chat Endpoint** | `/v1/chat/completions` | `/v1/messages` |
| **Audio API** | ✅ Whisper API | ❌ Not available |
| **Context Window** | 128k tokens | 200k tokens |
| **Temperature Range** | 0.0 - 2.0 | 0.0 - 1.0 |
| **System Message** | In messages array | Separate `system` parameter |
| **Provider Pattern** | Unified (`OpenAIProvider`) | Unified (`AnthropicProvider`) |

### 1. Architecture Overview

**Unified Provider Pattern** (following OpenAI):

```text
src/podcast_scraper/
├── providers/
│   └── anthropic/                    # NEW: Unified Anthropic provider
│       ├── __init__.py
│       └── anthropic_provider.py     # Single class implementing 2 protocols
├── prompts/
│   └── anthropic/                    # NEW: Anthropic-specific prompts
│       ├── ner/
│       │   ├── system_ner_v1.j2
│       │   └── guest_host_v1.j2
│       └── summarization/
│           ├── system_v1.j2
│           └── long_v1.j2
├── speaker_detectors/
│   └── factory.py                    # Updated: Add "anthropic" option
├── summarization/
│   └── factory.py                    # Updated: Add "anthropic" option
└── config.py                         # Updated: Add Anthropic fields
```

**Key Architectural Decision:** Use unified provider pattern (single `AnthropicProvider` class) matching `OpenAIProvider`, not separate files per capability.

### 2. Configuration

Add to `config.py` following OpenAI pattern exactly:

```python
from typing import Literal, Optional

# Provider Selection (updated)
speaker_detector_provider: Literal["spacy", "openai", "anthropic"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "anthropic"] = Field(
    default="transformers",
    description="Summarization provider"
)

# Anthropic API Configuration (following OpenAI pattern)
anthropic_api_key: Optional[str] = Field(
    default=None,
    alias="anthropic_api_key",
    description="Anthropic API key (prefer ANTHROPIC_API_KEY env var or .env file)"
)

anthropic_api_base: Optional[str] = Field(
    default=None,
    alias="anthropic_api_base",
    description="Anthropic API base URL (for E2E testing with mock servers)"
)

# Anthropic Model Selection (environment-based defaults, like OpenAI)
anthropic_speaker_model: str = Field(
    default_factory=_get_default_anthropic_speaker_model,
    alias="anthropic_speaker_model",
    description="Anthropic model for speaker detection (default: environment-based)"
)

anthropic_summary_model: str = Field(
    default_factory=_get_default_anthropic_summary_model,
    alias="anthropic_summary_model",
    description="Anthropic model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
anthropic_temperature: float = Field(
    default=0.3,
    alias="anthropic_temperature",
    description="Temperature for Anthropic generation (0.0-1.0, lower = more deterministic)"
)

anthropic_max_tokens: Optional[int] = Field(
    default=None,
    alias="anthropic_max_tokens",
    description="Max tokens for Anthropic generation (None = model default)"
)

# Anthropic Prompt Configuration (following OpenAI pattern)
anthropic_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="anthropic_speaker_system_prompt",
    description="Anthropic system prompt for speaker detection (default: anthropic/ner/system_ner_v1)"
)

anthropic_speaker_user_prompt: str = Field(
    default="anthropic/ner/guest_host_v1",
    alias="anthropic_speaker_user_prompt",
    description="Anthropic user prompt for speaker detection"
)

anthropic_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="anthropic_summary_system_prompt",
    description="Anthropic system prompt for summarization (default: anthropic/summarization/system_v1)"
)

anthropic_summary_user_prompt: str = Field(
    default="anthropic/summarization/long_v1",
    alias="anthropic_summary_user_prompt",
    description="Anthropic user prompt for summarization"
)
```

**Environment-based defaults** (like OpenAI):

```python
# In config_constants.py
TEST_DEFAULT_ANTHROPIC_SPEAKER_MODEL = "claude-3-5-haiku-latest"
PROD_DEFAULT_ANTHROPIC_SPEAKER_MODEL = "claude-3-5-sonnet-latest"

TEST_DEFAULT_ANTHROPIC_SUMMARY_MODEL = "claude-3-5-haiku-latest"
PROD_DEFAULT_ANTHROPIC_SUMMARY_MODEL = "claude-3-5-sonnet-latest"

# In config.py
def _get_default_anthropic_speaker_model() -> str:
    """Get default Anthropic speaker detection model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_ANTHROPIC_SPEAKER_MODEL
    return PROD_DEFAULT_ANTHROPIC_SPEAKER_MODEL

def _get_default_anthropic_summary_model() -> str:
    """Get default Anthropic summarization model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_ANTHROPIC_SUMMARY_MODEL
    return PROD_DEFAULT_ANTHROPIC_SUMMARY_MODEL
```

### 3. API Key Management

Follow OpenAI pattern exactly:

```python
# In config.py

@field_validator("anthropic_api_key", mode="before")
@classmethod
def _load_anthropic_api_key_from_env(cls, value: Any) -> Optional[str]:
    """Load Anthropic API key from environment variable if not provided."""
    if value is not None:
        return value
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    return None

@field_validator("anthropic_api_base", mode="before")
@classmethod
def _load_anthropic_api_base_from_env(cls, value: Any) -> Optional[str]:
    """Load Anthropic API base URL from environment variable if not provided."""
    if value is not None:
        return value
    env_base = os.getenv("ANTHROPIC_API_BASE")
    if env_base:
        return env_base
    return None

@model_validator(mode="after")
def _validate_anthropic_provider_requirements(self) -> "Config":
    """Validate that Anthropic API key is provided when Anthropic providers are selected."""
    anthropic_providers_used = []
    if self.speaker_detector_provider == "anthropic":
        anthropic_providers_used.append("speaker_detection")
    if self.summary_provider == "anthropic":
        anthropic_providers_used.append("summarization")

    if anthropic_providers_used and not self.anthropic_api_key:
        providers_str = ", ".join(anthropic_providers_used)
        raise ValueError(
            f"Anthropic API key required for Anthropic providers: {providers_str}. "
            "Set ANTHROPIC_API_KEY environment variable or anthropic_api_key in config."
        )

    return self
```

### 4. Provider Capability Validation

Add validation to prevent using Anthropic for transcription:

```python
# config.py - Capability Validation

@model_validator(mode="after")
def _validate_provider_capabilities(self) -> "Config":
    """Validate provider supports requested capability."""
    if self.transcription_provider == "anthropic":
        raise ValueError(
            "Anthropic provider does not support transcription. "
            "Use 'whisper' (local) or 'openai' instead. "
            "See provider capability matrix in documentation."
        )
    return self
```

### 5. Unified Provider Implementation

**File**: `src/podcast_scraper/providers/anthropic/anthropic_provider.py`

Follow `OpenAIProvider` pattern exactly:

```python
"""Unified Anthropic provider for speaker detection and summarization.

This module provides a single AnthropicProvider class that implements two protocols:
- SpeakerDetector (using Claude API)
- SummarizationProvider (using Claude API)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Note: Anthropic does NOT support transcription (no audio API).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Set, Tuple

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore

from ... import config, models
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

# Anthropic API pricing constants (for cost estimation)
# Source: https://www.anthropic.com/pricing
# Last updated: 2026-02
ANTHROPIC_HAIKU_INPUT_COST_PER_1M_TOKENS = 0.25
ANTHROPIC_HAIKU_OUTPUT_COST_PER_1M_TOKENS = 1.25
ANTHROPIC_SONNET_INPUT_COST_PER_1M_TOKENS = 3.00
ANTHROPIC_SONNET_OUTPUT_COST_PER_1M_TOKENS = 15.00


class AnthropicProvider:
    """Unified Anthropic provider implementing SpeakerDetector and SummarizationProvider.

    This provider initializes and manages:
    - Anthropic Claude API for speaker detection
    - Anthropic Claude API for summarization

    All capabilities share the same Anthropic client, similar to how OpenAI providers
    share the same OpenAI client. The client is initialized once and reused.

    Note: Transcription is NOT supported (Anthropic has no audio API).
    """

    def __init__(self, cfg: config.Config):
        """Initialize unified Anthropic provider.

        Args:
            cfg: Configuration object with settings for both capabilities

        Raises:
            ValueError: If Anthropic API key is not provided
            ImportError: If anthropic package is not installed
        """
        if Anthropic is None:
            raise ImportError(
                "anthropic package required for Anthropic provider. "
                "Install with: pip install 'podcast-scraper[anthropic]'"
            )

        if not cfg.anthropic_api_key:
            raise ValueError(
                "Anthropic API key required for Anthropic provider. "
                "Set ANTHROPIC_API_KEY environment variable or anthropic_api_key in config."
            )

        self.cfg = cfg

        # Suppress verbose Anthropic SDK debug logs (if needed)
        # Similar to OpenAI provider pattern

        # Support custom base_url for E2E testing with mock servers
        client_kwargs: dict[str, Any] = {"api_key": cfg.anthropic_api_key}
        if cfg.anthropic_api_base:
            client_kwargs["base_url"] = cfg.anthropic_api_base
        self.client = Anthropic(**client_kwargs)

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "anthropic_speaker_model", "claude-3-5-haiku-latest")
        self.speaker_temperature = getattr(cfg, "anthropic_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "anthropic_summary_model", "claude-3-5-haiku-latest")
        self.summary_temperature = getattr(cfg, "anthropic_temperature", 0.3)
        # Claude 3.5 Sonnet supports 200k context window
        self.max_context_tokens = 200000  # Conservative estimate

        # Initialization state
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        self._requires_separate_instances = False

    @staticmethod
    def get_pricing(model: str, capability: str) -> Dict[str, float]:
        """Get pricing information for a specific model and capability.

        Args:
            model: Model name
            capability: Capability type ("speaker_detection", "summarization")

        Returns:
            Dictionary with pricing information
        """
        # Implementation similar to OpenAIProvider.get_pricing()
        # ...

    def initialize(self) -> None:
        """Initialize all Anthropic capabilities.

        For Anthropic API, initialization is a no-op but we track it for consistency.
        This method is idempotent and can be called multiple times safely.
        """
        # Initialize speaker detection if enabled
        if self.cfg.auto_speakers and not self._speaker_detection_initialized:
            self._initialize_speaker_detection()

        # Initialize summarization if enabled
        if self.cfg.generate_summaries and not self._summarization_initialized:
            self._initialize_summarization()

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing Anthropic speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Anthropic summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True

    # ============================================================================
    # SpeakerDetector Protocol Implementation
    # ============================================================================

    def detect_hosts(
        self,
        feed_title: str | None,
        feed_description: str | None,
        feed_authors: list[str] | None = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using Anthropic API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "AnthropicProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Anthropic API to detect hosts from feed metadata
        if not feed_title:
            return set()

        try:
            # Use detect_speakers with empty known_hosts to detect hosts
            speakers, detected_hosts, _ = self.detect_speakers(
                episode_title=feed_title,
                episode_description=feed_description,
                known_hosts=set(),
            )
            return detected_hosts
        except Exception as exc:
            logger.warning("Failed to detect hosts from feed metadata: %s", exc)
            return set()

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
        pipeline_metrics: metrics.Metrics | None = None,
    ) -> Tuple[list[str], Set[str], bool]:
        """Detect speaker names from episode metadata using Anthropic API.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)
            pipeline_metrics: Optional metrics tracker

        Returns:
            Tuple of:
            - List of detected speaker names (hosts + guests)
            - Set of detected host names (subset of known_hosts)
            - Success flag (True if detection succeeded)

        Raises:
            ValueError: If detection fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        # If auto_speakers is disabled, return defaults without requiring initialization
        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, detection failed")
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False

        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "AnthropicProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Anthropic API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                self.cfg.anthropic_speaker_system_prompt or "anthropic/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call Anthropic API (different from OpenAI - uses /v1/messages with separate system)
            message = self.client.messages.create(
                model=self.speaker_model,
                max_tokens=300,
                temperature=self.speaker_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            response_text = message.content[0].text if message.content else ""
            if not response_text:
                logger.warning("Anthropic API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Anthropic speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available
            if pipeline_metrics is not None and hasattr(message, "usage"):
                input_tokens = message.usage.input_tokens if message.usage else 0
                output_tokens = message.usage.output_tokens if message.usage else 0
                pipeline_metrics.record_llm_speaker_detection_call(input_tokens, output_tokens)

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Anthropic API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Anthropic API error in speaker detection: %s", exc)
            raise ValueError(f"Anthropic speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For Anthropic provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.anthropic_speaker_user_prompt
        user_prompt = render_prompt(
            user_prompt_name,
            episode_title=episode_title,
            episode_description=episode_description or "",
            known_hosts=", ".join(known_hosts) if known_hosts else "",
        )
        return user_prompt

    def _parse_speakers_from_response(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Parse speaker names from Anthropic API response."""
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                speakers = data.get("speakers", [])
                hosts = set(data.get("hosts", []))
                guests = data.get("guests", [])
                all_speakers = list(hosts) + guests if not speakers else speakers
                return all_speakers, hosts, True
        except json.JSONDecodeError:
            pass

        # Fallback: parse from plain text
        speakers = []
        for line in response_text.strip().split("\n"):
            for name in line.split(","):
                name = name.strip().strip("-").strip("*").strip()
                if name and len(name) > 1:
                    speakers.append(name)
        detected_hosts = set(s for s in speakers if s in known_hosts)
        return speakers, detected_hosts, len(speakers) > 0

    # ============================================================================
    # SummarizationProvider Protocol Implementation
    # ============================================================================

    def summarize(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: metrics.Metrics | None = None,
    ) -> Dict[str, Any]:
        """Summarize text using Anthropic Claude API.

        Can handle full transcripts directly due to large context window (200k tokens).
        No chunking needed for most podcast transcripts.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters dict with max_length, min_length, etc.
            pipeline_metrics: Optional metrics tracker

        Returns:
            Dictionary with summary results:
            {
                "summary": str,
                "summary_short": Optional[str],
                "metadata": {...}
            }

        Raises:
            ValueError: If summarization fails
            RuntimeError: If provider is not initialized
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "AnthropicProvider summarization not initialized. Call initialize() first."
            )

        # Extract parameters with defaults from config
        max_length = (
            (params.get("max_length") if params else None)
            or self.cfg.summary_reduce_params.get("max_new_tokens")
            or 800
        )
        min_length = (
            (params.get("min_length") if params else None)
            or self.cfg.summary_reduce_params.get("min_new_tokens")
            or 100
        )
        custom_prompt = params.get("prompt") if params else None

        logger.debug(
            "Summarizing text via Anthropic API (model: %s, max_tokens: %d)",
            self.summary_model,
            max_length,
        )

        try:
            # Build prompts using prompt_store (RFC-017)
            (
                system_prompt,
                user_prompt,
                system_prompt_name,
                user_prompt_name,
                paragraphs_min,
                paragraphs_max,
            ) = self._build_summarization_prompts(
                text, episode_title, episode_description, max_length, min_length, custom_prompt
            )

            # Call Anthropic API (different from OpenAI - uses /v1/messages with separate system)
            message = self.client.messages.create(
                model=self.summary_model,
                max_tokens=max_length,
                temperature=self.summary_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            summary = message.content[0].text if message.content else ""
            if not summary:
                logger.warning("Anthropic API returned empty summary")
                summary = ""

            logger.debug("Anthropic summarization completed: %d characters", len(summary))

            # Track LLM call metrics if available
            if pipeline_metrics is not None and hasattr(message, "usage"):
                input_tokens = message.usage.input_tokens if message.usage else 0
                output_tokens = message.usage.output_tokens if message.usage else 0
                pipeline_metrics.record_llm_summarization_call(input_tokens, output_tokens)

            # Get prompt metadata for tracking (RFC-017)
            from ...prompts.store import get_prompt_metadata

            prompt_metadata = {}
            if system_prompt_name:
                prompt_metadata["system"] = get_prompt_metadata(system_prompt_name)
            user_params = {
                "transcript": text[:100] + "..." if len(text) > 100 else text,
                "title": episode_title or "",
                "paragraphs_min": paragraphs_min,
                "paragraphs_max": paragraphs_max,
            }
            user_params.update(self.cfg.summary_prompt_params)
            prompt_metadata["user"] = get_prompt_metadata(user_prompt_name, params=user_params)

            return {
                "summary": summary,
                "summary_short": None,  # Anthropic provider doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "anthropic",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("Anthropic API error in summarization: %s", exc)
            raise ValueError(f"Anthropic summarization failed: {exc}") from exc

    def _build_summarization_prompts(
        self,
        text: str,
        episode_title: Optional[str],
        episode_description: Optional[str],
        max_length: int,
        min_length: int,
        custom_prompt: Optional[str],
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store (RFC-017)."""
        from ...prompts.store import render_prompt

        system_prompt_name = (
            self.cfg.anthropic_summary_system_prompt or "anthropic/summarization/system_v1"
        )
        user_prompt_name = self.cfg.anthropic_summary_user_prompt

        system_prompt = render_prompt(system_prompt_name)

        paragraphs_min = max(1, min_length // 100)
        paragraphs_max = max(paragraphs_min, max_length // 100)

        if custom_prompt:
            user_prompt = custom_prompt.replace("{{ transcript }}", text)
            if episode_title:
                user_prompt = user_prompt.replace("{{ title }}", episode_title)
            user_prompt_name = "custom"
        else:
            template_params = {
                "transcript": text,
                "title": episode_title or "",
                "paragraphs_min": paragraphs_min,
                "paragraphs_max": paragraphs_max,
            }
            template_params.update(self.cfg.summary_prompt_params)
            user_prompt = render_prompt(user_prompt_name, **template_params)

        return (
            system_prompt,
            user_prompt,
            system_prompt_name,
            user_prompt_name,
            paragraphs_min,
            paragraphs_max,
        )

    # ============================================================================
    # Cleanup Methods
    # ============================================================================

    def cleanup(self) -> None:
        """Cleanup all provider resources (no-op for API provider)."""
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

    def clear_cache(self) -> None:
        """Clear cache (no-op for API provider)."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component)."""
        return self._speaker_detection_initialized or self._summarization_initialized
```

### 6. Factory Updates

Update both factories to support both Config-based and experiment-based modes (like OpenAI):

**File**: `src/podcast_scraper/speaker_detectors/factory.py`

```python
def create_speaker_detector(
    cfg_or_provider_type: Union[config.Config, str],
    params: Optional[Union[SpeakerDetectionParams, Dict[str, Any]]] = None,
) -> SpeakerDetector:
    # ... existing code ...

    elif provider_type == "anthropic":
        from ..providers.anthropic.anthropic_provider import AnthropicProvider

        if experiment_mode:
            from ..config import Config
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",
                speaker_detector_provider="anthropic",
                anthropic_speaker_model=params.model_name if params.model_name else "claude-3-5-haiku-latest",
                anthropic_temperature=params.temperature if params.temperature is not None else 0.3,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            return AnthropicProvider(cfg)
        else:
            return AnthropicProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'spacy', 'openai', 'anthropic'"
        )
```

Similar update for `summarization/factory.py`.

### 7. Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
anthropic = [
    "anthropic>=0.30.0,<1.0.0",
]
```

### 8. Prompt Templates

Create Anthropic-specific prompts in `src/podcast_scraper/prompts/anthropic/`:

- `ner/system_ner_v1.j2` - System prompt for speaker detection
- `ner/guest_host_v1.j2` - User prompt for speaker detection
- `summarization/system_v1.j2` - System prompt for summarization
- `summarization/long_v1.j2` - User prompt for summarization

Follow OpenAI prompt patterns but optimize for Claude models.

## Testing Strategy

Same pattern as OpenAI provider:

1. **Unit tests**: Mock Anthropic API responses
2. **Integration tests**: Use E2E mock server with Anthropic endpoints
3. **E2E tests**: Full workflow with Anthropic provider

## Success Criteria

1. ✅ Anthropic supports speaker detection and summarization via unified provider
2. ✅ Clear error when attempting transcription with Anthropic
3. ✅ Free tier works for development (Haiku model)
4. ✅ E2E tests pass
5. ✅ Experiment mode supported from start
6. ✅ Environment-based model defaults (test vs prod)
7. ✅ Follows OpenAI provider pattern exactly

## Migration Notes

- **Breaking Changes**: None (new provider, backward compatible)
- **Configuration**: Add `ANTHROPIC_API_KEY` to `.env` file
- **Dependencies**: Install with `pip install 'podcast-scraper[anthropic]'`

## References

- **Related PRD**: `docs/prd/PRD-009-anthropic-provider-integration.md`
- **Reference Implementation**: `src/podcast_scraper/providers/openai/openai_provider.py`
- **Anthropic API Documentation**: https://docs.anthropic.com/en/api
- **Anthropic Python SDK**: https://github.com/anthropics/anthropic-sdk-python
