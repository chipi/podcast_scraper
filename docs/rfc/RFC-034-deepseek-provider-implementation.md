# RFC-034: DeepSeek Provider Implementation (Revised)

- **Status**: ✅ Completed (v2.5.0)
- **Revision**: 2
- **Date**: 2026-02-04
- **Authors**:
- **Stakeholders**: Maintainers, users wanting DeepSeek API integration, cost-conscious users
- **Related PRDs**:
  - `docs/prd/PRD-011-deepseek-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference - unified provider pattern)
  - `docs/rfc/RFC-032-anthropic-provider-implementation.md` (similar pattern - no transcription)
  - `docs/rfc/RFC-021-modularization-refactoring-plan.md` (architecture foundation)

## Abstract

Design and implement DeepSeek AI as a unified provider for speaker detection and summarization capabilities. DeepSeek offers an OpenAI-compatible API at significantly lower cost (90-95% cheaper), making it ideal for cost-conscious users. Like Anthropic, DeepSeek does NOT support audio transcription. This RFC follows the **unified provider pattern** established by OpenAI (RFC-013), where a single provider class implements multiple protocols. The key implementation detail is using the OpenAI SDK with a custom `base_url` pointing to DeepSeek's API.

**Architecture Alignment:** DeepSeek provider follows the exact same unified provider pattern as `OpenAIProvider`, implementing two protocols (`SpeakerDetector`, `SummarizationProvider`) in a single class and integrating via the existing factory pattern with support for both Config-based and experiment-based modes.

## Problem Statement

Users want the option to use DeepSeek AI as an extremely cost-effective alternative for:

1. **Speaker Detection**: Entity extraction using DeepSeek chat models
2. **Summarization**: High-quality summaries using DeepSeek chat models

**Note:** Transcription is NOT supported by DeepSeek (no audio API).

Key advantages of DeepSeek:

- **95% cheaper** than OpenAI for text processing
- **OpenAI-compatible API** - no new SDK required (uses OpenAI SDK)
- **Strong reasoning** with DeepSeek-R1 model

Requirements:

- No changes to end-user experience when using defaults
- Secure API key management
- Per-capability provider selection
- Use OpenAI SDK with custom base_url (no new dependency)
- Handle capability gaps gracefully (transcription not supported)
- Support both Config-based and experiment-based factory modes

## Constraints & Assumptions

**Constraints:**

- **Prerequisite**: OpenAI provider implementation (RFC-013) ✅ Completed
- **Backward Compatibility**: Default providers must remain unchanged
- **API Key Security**: API keys never in source code
- **Capability Gap**: DeepSeek does not support audio transcription
- **SDK Reuse**: Use existing OpenAI SDK, no new dependency
- **Must follow unified provider pattern** (like OpenAI)

**Assumptions:**

- DeepSeek API maintains OpenAI compatibility
- DeepSeek API is stable and available
- Prompts may need optimization for DeepSeek models

## Design & Implementation

### 0. DeepSeek API Overview

DeepSeek provides an OpenAI-compatible API:

| Feature | OpenAI | DeepSeek |
| ------- | ------ | -------- |
| **Base URL** | `https://api.openai.com/v1` | `https://api.deepseek.com` |
| **Chat Endpoint** | `/v1/chat/completions` | `/v1/chat/completions` |
| **Audio API** | ✅ Whisper | ❌ Not available |
| **Context Window** | 128k tokens | 64k tokens |
| **SDK** | `openai` | `openai` (with custom base_url) |
| **Provider Pattern** | Unified (`OpenAIProvider`) | Unified (`DeepSeekProvider`) |

### 1. Architecture Overview

**Unified Provider Pattern** (following OpenAI):

```text
src/podcast_scraper/
├── providers/
│   └── deepseek/                     # NEW: Unified DeepSeek provider
│       ├── __init__.py
│       └── deepseek_provider.py      # Single class implementing 2 protocols
├── prompts/
│   └── deepseek/                     # NEW: DeepSeek-specific prompts
│       ├── ner/
│       │   ├── system_ner_v1.j2
│       │   └── guest_host_v1.j2
│       └── summarization/
│           ├── system_v1.j2
│           └── long_v1.j2
├── speaker_detectors/
│   └── factory.py                    # Updated: Add "deepseek" option
├── summarization/
│   └── factory.py                    # Updated: Add "deepseek" option
└── config.py                         # Updated: Add DeepSeek fields
```

**Key Architectural Decision:** Use unified provider pattern (single `DeepSeekProvider` class) matching `OpenAIProvider`, not separate files per capability.

### 2. Configuration

Add to `config.py` following OpenAI pattern exactly:

```python
from typing import Literal, Optional

# Provider Selection (updated)
speaker_detector_provider: Literal["spacy", "openai", "anthropic", "deepseek"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "anthropic", "deepseek"] = Field(
    default="transformers",
    description="Summarization provider"
)

# DeepSeek API Configuration (following OpenAI pattern)
deepseek_api_key: Optional[str] = Field(
    default=None,
    alias="deepseek_api_key",
    description="DeepSeek API key (prefer DEEPSEEK_API_KEY env var or .env file)"
)

deepseek_api_base: Optional[str] = Field(
    default=None,
    alias="deepseek_api_base",
    description="DeepSeek API base URL (default: https://api.deepseek.com, for E2E testing)"
)

# DeepSeek Model Selection (environment-based defaults, like OpenAI)
deepseek_speaker_model: str = Field(
    default_factory=_get_default_deepseek_speaker_model,
    alias="deepseek_speaker_model",
    description="DeepSeek model for speaker detection (default: environment-based)"
)

deepseek_summary_model: str = Field(
    default_factory=_get_default_deepseek_summary_model,
    alias="deepseek_summary_model",
    description="DeepSeek model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
deepseek_temperature: float = Field(
    default=0.3,
    alias="deepseek_temperature",
    description="Temperature for DeepSeek generation (0.0-2.0, lower = more deterministic)"
)

deepseek_max_tokens: Optional[int] = Field(
    default=None,
    alias="deepseek_max_tokens",
    description="Max tokens for DeepSeek generation (None = model default)"
)

# DeepSeek Prompt Configuration (following OpenAI pattern)
deepseek_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="deepseek_speaker_system_prompt",
    description="DeepSeek system prompt for speaker detection (default: deepseek/ner/system_ner_v1)"
)

deepseek_speaker_user_prompt: str = Field(
    default="deepseek/ner/guest_host_v1",
    alias="deepseek_speaker_user_prompt",
    description="DeepSeek user prompt for speaker detection"
)

deepseek_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="deepseek_summary_system_prompt",
    description="DeepSeek system prompt for summarization (default: deepseek/summarization/system_v1)"
)

deepseek_summary_user_prompt: str = Field(
    default="deepseek/summarization/long_v1",
    alias="deepseek_summary_user_prompt",
    description="DeepSeek user prompt for summarization"
)
```

**Environment-based defaults** (like OpenAI):

```python
# In config_constants.py
TEST_DEFAULT_DEEPSEEK_SPEAKER_MODEL = "deepseek-chat"
PROD_DEFAULT_DEEPSEEK_SPEAKER_MODEL = "deepseek-chat"  # Same model, still very cheap

TEST_DEFAULT_DEEPSEEK_SUMMARY_MODEL = "deepseek-chat"
PROD_DEFAULT_DEEPSEEK_SUMMARY_MODEL = "deepseek-chat"  # Same model, still very cheap

# In config.py
def _get_default_deepseek_speaker_model() -> str:
    """Get default DeepSeek speaker detection model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_DEEPSEEK_SPEAKER_MODEL
    return PROD_DEFAULT_DEEPSEEK_SPEAKER_MODEL

def _get_default_deepseek_summary_model() -> str:
    """Get default DeepSeek summarization model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_DEEPSEEK_SUMMARY_MODEL
    return PROD_DEFAULT_DEEPSEEK_SUMMARY_MODEL
```

### 3. API Key Management

Follow OpenAI pattern exactly:

```python
# In config.py

@field_validator("deepseek_api_key", mode="before")
@classmethod
def _load_deepseek_api_key_from_env(cls, value: Any) -> Optional[str]:
    """Load DeepSeek API key from environment variable if not provided."""
    if value is not None:
        return value
    env_key = os.getenv("DEEPSEEK_API_KEY")
    if env_key:
        return env_key
    return None

@field_validator("deepseek_api_base", mode="before")
@classmethod
def _load_deepseek_api_base_from_env(cls, value: Any) -> Optional[str]:
    """Load DeepSeek API base URL from environment variable if not provided."""
    if value is not None:
        return value
    env_base = os.getenv("DEEPSEEK_API_BASE")
    if env_base:
        return env_base
    # Default to DeepSeek API base URL
    return "https://api.deepseek.com"

@model_validator(mode="after")
def _validate_deepseek_provider_requirements(self) -> "Config":
    """Validate that DeepSeek API key is provided when DeepSeek providers are selected."""
    deepseek_providers_used = []
    if self.speaker_detector_provider == "deepseek":
        deepseek_providers_used.append("speaker_detection")
    if self.summary_provider == "deepseek":
        deepseek_providers_used.append("summarization")

    if deepseek_providers_used and not self.deepseek_api_key:
        providers_str = ", ".join(deepseek_providers_used)
        raise ValueError(
            f"DeepSeek API key required for DeepSeek providers: {providers_str}. "
            "Set DEEPSEEK_API_KEY environment variable or deepseek_api_key in config."
        )

    return self
```

### 4. Provider Capability Validation

Add validation to prevent using DeepSeek for transcription:

```python
# config.py - Capability Validation

@model_validator(mode="after")
def _validate_provider_capabilities(self) -> "Config":
    """Validate provider supports requested capability."""
    if self.transcription_provider == "deepseek":
        raise ValueError(
            "DeepSeek provider does not support transcription. "
            "Use 'whisper' (local) or 'openai' instead. "
            "See provider capability matrix in documentation."
        )
    return self
```

### 5. Unified Provider Implementation

**File**: `src/podcast_scraper/providers/deepseek/deepseek_provider.py`

Follow `OpenAIProvider` pattern exactly, but use OpenAI SDK with custom base_url:

```python
"""Unified DeepSeek provider for speaker detection and summarization.

This module provides a single DeepSeekProvider class that implements two protocols:
- SpeakerDetector (using DeepSeek chat API via OpenAI SDK)
- SummarizationProvider (using DeepSeek chat API via OpenAI SDK)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key insight: DeepSeek uses an OpenAI-compatible API, so we reuse the OpenAI SDK
with a custom base_url. No new dependency required.

Note: DeepSeek does NOT support transcription (no audio API).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Set, Tuple

from openai import OpenAI

from ... import config, models
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

# DeepSeek API pricing constants (for cost estimation)
# Source: https://platform.deepseek.com/pricing
# Last updated: 2026-02
DEEPSEEK_CHAT_INPUT_COST_PER_1M_TOKENS = 0.28
DEEPSEEK_CHAT_OUTPUT_COST_PER_1M_TOKENS = 0.42
DEEPSEEK_CHAT_CACHE_HIT_INPUT_COST_PER_1M_TOKENS = 0.028  # 90% discount on cache hits


class DeepSeekProvider:
    """Unified DeepSeek provider implementing SpeakerDetector and SummarizationProvider.

    This provider initializes and manages:
    - DeepSeek chat API for speaker detection (via OpenAI SDK)
    - DeepSeek chat API for summarization (via OpenAI SDK)

    All capabilities share the same OpenAI client (configured with DeepSeek base_url),
    similar to how OpenAI providers share the same OpenAI client.

    Note: Transcription is NOT supported (DeepSeek has no audio API).
    """

    def __init__(self, cfg: config.Config):
        """Initialize unified DeepSeek provider.

        Args:
            cfg: Configuration object with settings for both capabilities

        Raises:
            ValueError: If DeepSeek API key is not provided
            ImportError: If openai package is not installed
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for DeepSeek provider. "
                "Install with: pip install 'podcast-scraper[openai]'"
            )

        if not cfg.deepseek_api_key:
            raise ValueError(
                "DeepSeek API key required for DeepSeek provider. "
                "Set DEEPSEEK_API_KEY environment variable or deepseek_api_key in config."
            )

        self.cfg = cfg

        # Suppress verbose OpenAI SDK debug logs (same as OpenAI provider)
        # Set OpenAI SDK loggers to WARNING level when root logger is DEBUG
        root_logger = logging.getLogger()
        root_level = root_logger.level if root_logger.level else logging.INFO
        if root_level <= logging.DEBUG:
            openai_loggers = [
                "openai",
                "openai._base_client",
                "openai.api_resources",
                "httpx",
                "httpcore",
            ]
            for logger_name in openai_loggers:
                openai_logger = logging.getLogger(logger_name)
                openai_logger.setLevel(logging.WARNING)

        # Support custom base_url for E2E testing with mock servers
        # Default to DeepSeek API base URL
        base_url = cfg.deepseek_api_base or "https://api.deepseek.com"
        client_kwargs: dict[str, Any] = {
            "api_key": cfg.deepseek_api_key,
            "base_url": base_url,
        }
        self.client = OpenAI(**client_kwargs)

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "deepseek_speaker_model", "deepseek-chat")
        self.speaker_temperature = getattr(cfg, "deepseek_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "deepseek_summary_model", "deepseek-chat")
        self.summary_temperature = getattr(cfg, "deepseek_temperature", 0.3)
        # DeepSeek supports 64k context window
        self.max_context_tokens = 64000  # Conservative estimate

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
        """Initialize all DeepSeek capabilities.

        For DeepSeek API, initialization is a no-op but we track it for consistency.
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
        logger.debug("Initializing DeepSeek speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing DeepSeek summarization (model: %s)", self.summary_model)
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
        """Detect host names from feed-level metadata using DeepSeek API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "DeepSeekProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use DeepSeek API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using DeepSeek API.

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
                "DeepSeekProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via DeepSeek API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                self.cfg.deepseek_speaker_system_prompt or "deepseek/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call DeepSeek API (OpenAI-compatible format)
            response = self.client.chat.completions.create(
                model=self.speaker_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.speaker_temperature,
                max_tokens=300,
                response_format={"type": "json_object"},  # Request JSON response
            )

            response_text = response.choices[0].message.content
            if not response_text:
                logger.warning("DeepSeek API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "DeepSeek speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available
            if pipeline_metrics is not None and hasattr(response, "usage"):
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                pipeline_metrics.record_llm_speaker_detection_call(input_tokens, output_tokens)

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse DeepSeek API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("DeepSeek API error in speaker detection: %s", exc)
            raise ValueError(f"DeepSeek speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For DeepSeek provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.deepseek_speaker_user_prompt
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
        """Parse speaker names from DeepSeek API response."""
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
        """Summarize text using DeepSeek API.

        Can handle full transcripts directly due to large context window (64k tokens).
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
                "DeepSeekProvider summarization not initialized. Call initialize() first."
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
            "Summarizing text via DeepSeek API (model: %s, max_tokens: %d)",
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

            # Call DeepSeek API (OpenAI-compatible format)
            response = self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.summary_temperature,
                max_tokens=max_length,
            )

            summary = response.choices[0].message.content
            if not summary:
                logger.warning("DeepSeek API returned empty summary")
                summary = ""

            logger.debug("DeepSeek summarization completed: %d characters", len(summary))

            # Track LLM call metrics if available
            if pipeline_metrics is not None and hasattr(response, "usage"):
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
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
                "summary_short": None,  # DeepSeek provider doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "deepseek",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("DeepSeek API error in summarization: %s", exc)
            raise ValueError(f"DeepSeek summarization failed: {exc}") from exc

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
            self.cfg.deepseek_summary_system_prompt or "deepseek/summarization/system_v1"
        )
        user_prompt_name = self.cfg.deepseek_summary_user_prompt

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

    elif provider_type == "deepseek":
        from ..providers.deepseek.deepseek_provider import DeepSeekProvider

        if experiment_mode:
            from ..config import Config
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",
                speaker_detector_provider="deepseek",
                deepseek_speaker_model=params.model_name if params.model_name else "deepseek-chat",
                deepseek_temperature=params.temperature if params.temperature is not None else 0.3,
                deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            )
            return DeepSeekProvider(cfg)
        else:
            return DeepSeekProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'spacy', 'openai', 'anthropic', 'deepseek'"
        )
```

Similar update for `summarization/factory.py`.

### 7. Dependencies

**No new dependencies required** - DeepSeek uses OpenAI SDK:

```toml
# pyproject.toml - No changes needed
# DeepSeek uses existing openai package with custom base_url
```

**Note:** DeepSeek provider requires `openai` package (already a dependency for OpenAI provider).

### 8. Prompt Templates

Create DeepSeek-specific prompts in `src/podcast_scraper/prompts/deepseek/`:

- `ner/system_ner_v1.j2` - System prompt for speaker detection
- `ner/guest_host_v1.j2` - User prompt for speaker detection
- `summarization/system_v1.j2` - System prompt for summarization
- `summarization/long_v1.j2` - User prompt for summarization

Follow OpenAI prompt patterns but optimize for DeepSeek models.

## Testing Strategy

Same pattern as OpenAI provider:

1. **Unit tests**: Mock DeepSeek API responses (using OpenAI SDK mocks)
2. **Integration tests**: Use E2E mock server with DeepSeek endpoints (reuse OpenAI format)
3. **E2E tests**: Full workflow with DeepSeek provider

## Success Criteria

1. ✅ DeepSeek supports speaker detection and summarization via unified provider
2. ✅ Clear error when attempting transcription with DeepSeek
3. ✅ No new SDK dependency (uses OpenAI SDK)
4. ✅ E2E tests pass
5. ✅ Experiment mode supported from start
6. ✅ Environment-based model defaults (test vs prod)
7. ✅ Follows OpenAI provider pattern exactly

## Migration Notes

- **Breaking Changes**: None (new provider, backward compatible)
- **Configuration**: Add `DEEPSEEK_API_KEY` to `.env` file
- **Dependencies**: No new dependencies (uses existing `openai` package)

## References

- **Related PRD**: `docs/prd/PRD-011-deepseek-provider-integration.md`
- **Reference Implementation**: `src/podcast_scraper/providers/openai/openai_provider.py`
- **DeepSeek API Documentation**: <https://platform.deepseek.com/docs>
- **DeepSeek Pricing**: <https://platform.deepseek.com/pricing>
