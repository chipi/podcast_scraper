# RFC-036: Groq Provider Implementation (Revised)

- **Status**: Draft (Revised)
- **Revision**: 2
- **Date**: 2026-02-04
- **Authors**:
- **Stakeholders**: Maintainers, users wanting ultra-fast inference
- **Related PRDs**:
  - `docs/prd/PRD-013-groq-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference - unified provider pattern)
  - `docs/rfc/RFC-034-deepseek-provider-implementation.md` (same pattern - OpenAI SDK)

## Abstract

Design and implement Groq as a unified provider for speaker detection and summarization capabilities. Groq offers **ultra-fast inference** (10x faster than other providers) by running models on custom LPU hardware. This RFC follows the **unified provider pattern** established by OpenAI (RFC-013), where a single provider class implements multiple protocols. The key implementation detail is using the OpenAI SDK with a custom `base_url` pointing to Groq's API.

**Key Advantage:** Groq processes at 500+ tokens/second vs ~50-100 for other providers.

**Architecture Alignment:** Groq provider follows the exact same unified provider pattern as `OpenAIProvider`, implementing two protocols (`SpeakerDetector`, `SummarizationProvider`) in a single class and integrating via the existing factory pattern with support for both Config-based and experiment-based modes.

## Problem Statement

Users want Groq for:

1. **Speaker Detection**: Entity extraction using Llama/Mixtral models
2. **Summarization**: High-quality summaries at 10x speed

**Note:** Transcription is NOT supported (Groq hosts LLMs only, no audio models).

Key advantages:

- **10x faster** inference than any other provider
- **Open source models** (Llama 3.3, Mixtral, Gemma)
- **OpenAI-compatible API** - no new SDK required
- **Generous free tier** - 14,400 tokens/minute

Requirements:

- No changes to end-user experience when using defaults
- Secure API key management
- Per-capability provider selection
- Use OpenAI SDK with custom base_url (no new dependency)
- Handle capability gaps gracefully (transcription not supported)
- Support both Config-based and experiment-based factory modes

## Constraints & Assumptions

**Constraints:**

- No audio transcription support
- Rate limits on free tier (30 RPM, 14,400 TPM)
- Model selection limited to Groq-hosted models
- **Must follow unified provider pattern** (like OpenAI)

**Assumptions:**

- Groq API maintains OpenAI compatibility
- Llama 3.3 70B quality is comparable to GPT-4o-mini

## Design & Implementation

### 0. Groq API Overview

Groq provides an OpenAI-compatible API:

| Feature | OpenAI | Groq |
| ------- | ------ | ---- |
| **Base URL** | `https://api.openai.com/v1` | `https://api.groq.com/openai/v1` |
| **SDK** | `openai` | `openai` (with custom base_url) |
| **Audio API** | ✅ Whisper | ❌ Not available |
| **Speed** | ~100 tokens/sec | **500+ tokens/sec** |
| **Models** | Proprietary | Open source (Llama, Mixtral, Gemma) |
| **Provider Pattern** | Unified (`OpenAIProvider`) | Unified (`GroqProvider`) |

### 1. Architecture Overview

**Unified Provider Pattern** (following OpenAI):

```text
src/podcast_scraper/
├── providers/
│   └── groq/                          # NEW: Unified Groq provider
│       ├── __init__.py
│       └── groq_provider.py           # Single class implementing 2 protocols
├── prompts/
│   └── groq/                          # NEW: Groq/Llama-specific prompts
│       ├── ner/
│       │   ├── system_ner_v1.j2
│       │   └── guest_host_v1.j2
│       └── summarization/
│           ├── system_v1.j2
│           └── long_v1.j2
├── speaker_detectors/
│   └── factory.py                     # Updated: Add "groq" option
├── summarization/
│   └── factory.py                     # Updated: Add "groq" option
└── config.py                         # Updated: Add Groq fields
```

**Key Architectural Decision:** Use unified provider pattern (single `GroqProvider` class) matching `OpenAIProvider`, not separate files per capability.

### 2. Configuration

Add to `config.py` following OpenAI pattern exactly:

```python
from typing import Literal, Optional

# Provider Selection (updated)
speaker_detector_provider: Literal["spacy", "openai", "anthropic", "deepseek", "groq"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "anthropic", "deepseek", "groq"] = Field(
    default="transformers",
    description="Summarization provider"
)

# Groq API Configuration (following OpenAI pattern)
groq_api_key: Optional[str] = Field(
    default=None,
    alias="groq_api_key",
    description="Groq API key (prefer GROQ_API_KEY env var or .env file)"
)

groq_api_base: Optional[str] = Field(
    default=None,
    alias="groq_api_base",
    description="Groq API base URL (default: https://api.groq.com/openai/v1, for E2E testing)"
)

# Groq Model Selection (environment-based defaults, like OpenAI)
groq_speaker_model: str = Field(
    default_factory=_get_default_groq_speaker_model,
    alias="groq_speaker_model",
    description="Groq model for speaker detection (default: environment-based)"
)

groq_summary_model: str = Field(
    default_factory=_get_default_groq_summary_model,
    alias="groq_summary_model",
    description="Groq model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
groq_temperature: float = Field(
    default=0.3,
    alias="groq_temperature",
    description="Temperature for Groq generation (0.0-2.0, lower = more deterministic)"
)

groq_max_tokens: Optional[int] = Field(
    default=None,
    alias="groq_max_tokens",
    description="Max tokens for Groq generation (None = model default)"
)

# Groq Prompt Configuration (following OpenAI pattern)
groq_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="groq_speaker_system_prompt",
    description="Groq system prompt for speaker detection (default: groq/ner/system_ner_v1)"
)

groq_speaker_user_prompt: str = Field(
    default="groq/ner/guest_host_v1",
    alias="groq_speaker_user_prompt",
    description="Groq user prompt for speaker detection"
)

groq_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="groq_summary_system_prompt",
    description="Groq system prompt for summarization (default: groq/summarization/system_v1)"
)

groq_summary_user_prompt: str = Field(
    default="groq/summarization/long_v1",
    alias="groq_summary_user_prompt",
    description="Groq user prompt for summarization"
)
```

**Environment-based defaults** (like OpenAI):

```python
# In config_constants.py
TEST_DEFAULT_GROQ_SPEAKER_MODEL = "llama-3.1-8b-instant"  # Free tier, ultra-fast
PROD_DEFAULT_GROQ_SPEAKER_MODEL = "llama-3.3-70b-versatile"  # Best quality, still fast

TEST_DEFAULT_GROQ_SUMMARY_MODEL = "llama-3.1-8b-instant"  # Free tier, ultra-fast
PROD_DEFAULT_GROQ_SUMMARY_MODEL = "llama-3.3-70b-versatile"  # Best quality, still fast

# In config.py
def _get_default_groq_speaker_model() -> str:
    """Get default Groq speaker detection model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_GROQ_SPEAKER_MODEL
    return PROD_DEFAULT_GROQ_SPEAKER_MODEL

def _get_default_groq_summary_model() -> str:
    """Get default Groq summarization model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_GROQ_SUMMARY_MODEL
    return PROD_DEFAULT_GROQ_SUMMARY_MODEL
```

### 3. API Key Management

Follow OpenAI pattern exactly:

```python
# In config.py

@field_validator("groq_api_key", mode="before")
@classmethod
def _load_groq_api_key_from_env(cls, value: Any) -> Optional[str]:
    """Load Groq API key from environment variable if not provided."""
    if value is not None:
        return value
    env_key = os.getenv("GROQ_API_KEY")
    if env_key:
        return env_key
    return None

@field_validator("groq_api_base", mode="before")
@classmethod
def _load_groq_api_base_from_env(cls, value: Any) -> Optional[str]:
    """Load Groq API base URL from environment variable if not provided."""
    if value is not None:
        return value
    env_base = os.getenv("GROQ_API_BASE")
    if env_base:
        return env_base
    # Default to Groq API base URL
    return "https://api.groq.com/openai/v1"

@model_validator(mode="after")
def _validate_groq_provider_requirements(self) -> "Config":
    """Validate that Groq API key is provided when Groq providers are selected."""
    groq_providers_used = []
    if self.speaker_detector_provider == "groq":
        groq_providers_used.append("speaker_detection")
    if self.summary_provider == "groq":
        groq_providers_used.append("summarization")

    if groq_providers_used and not self.groq_api_key:
        providers_str = ", ".join(groq_providers_used)
        raise ValueError(
            f"Groq API key required for Groq providers: {providers_str}. "
            "Set GROQ_API_KEY environment variable or groq_api_key in config."
        )

    return self
```

### 4. Provider Capability Validation

Add validation to prevent using Groq for transcription:

```python
# config.py - Capability Validation

@model_validator(mode="after")
def _validate_provider_capabilities(self) -> "Config":
    """Validate provider supports requested capability."""
    if self.transcription_provider == "groq":
        raise ValueError(
            "Groq provider does not support transcription. "
            "Use 'whisper' (local) or 'openai' instead. "
            "See provider capability matrix in documentation."
        )
    return self
```

### 5. Unified Provider Implementation

**File**: `src/podcast_scraper/providers/groq/groq_provider.py`

Follow `OpenAIProvider` pattern exactly, but use OpenAI SDK with custom base_url (same as DeepSeek):

```python
"""Unified Groq provider for speaker detection and summarization.

This module provides a single GroqProvider class that implements two protocols:
- SpeakerDetector (using Groq chat API via OpenAI SDK)
- SummarizationProvider (using Groq chat API via OpenAI SDK)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key insight: Groq uses an OpenAI-compatible API, so we reuse the OpenAI SDK
with a custom base_url. No new dependency required.

Key advantage: Ultra-fast inference (500+ tokens/second) - 10x faster than other providers.

Note: Groq does NOT support transcription (no audio API).
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

# Groq API pricing constants (for cost estimation)
# Source: https://console.groq.com/docs/pricing
# Last updated: 2026-02
GROQ_LLAMA_8B_INPUT_COST_PER_1M_TOKENS = 0.05
GROQ_LLAMA_8B_OUTPUT_COST_PER_1M_TOKENS = 0.08
GROQ_LLAMA_70B_INPUT_COST_PER_1M_TOKENS = 0.59
GROQ_LLAMA_70B_OUTPUT_COST_PER_1M_TOKENS = 0.79


class GroqProvider:
    """Unified Groq provider implementing SpeakerDetector and SummarizationProvider.

    This provider initializes and manages:
    - Groq chat API for speaker detection (via OpenAI SDK)
    - Groq chat API for summarization (via OpenAI SDK)

    All capabilities share the same OpenAI client (configured with Groq base_url),
    similar to how OpenAI providers share the same OpenAI client.

    Key advantage: Ultra-fast inference (500+ tokens/second) - 10x faster than other providers.

    Note: Transcription is NOT supported (Groq has no audio API).
    """

    def __init__(self, cfg: config.Config):
        """Initialize unified Groq provider.

        Args:
            cfg: Configuration object with settings for both capabilities

        Raises:
            ValueError: If Groq API key is not provided
            ImportError: If openai package is not installed
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for Groq provider. "
                "Install with: pip install 'podcast-scraper[openai]'"
            )

        if not cfg.groq_api_key:
            raise ValueError(
                "Groq API key required for Groq provider. "
                "Set GROQ_API_KEY environment variable or groq_api_key in config."
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
        # Default to Groq API base URL
        base_url = cfg.groq_api_base or "https://api.groq.com/openai/v1"
        client_kwargs: dict[str, Any] = {
            "api_key": cfg.groq_api_key,
            "base_url": base_url,
        }
        self.client = OpenAI(**client_kwargs)

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "groq_speaker_model", "llama-3.3-70b-versatile")
        self.speaker_temperature = getattr(cfg, "groq_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "groq_summary_model", "llama-3.3-70b-versatile")
        self.summary_temperature = getattr(cfg, "groq_temperature", 0.3)
        # Llama 3.3 70B supports 128k context window
        self.max_context_tokens = 128000  # Conservative estimate

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
        """Initialize all Groq capabilities.

        For Groq API, initialization is a no-op but we track it for consistency.
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
        logger.debug("Initializing Groq speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Groq summarization (model: %s)", self.summary_model)
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
        """Detect host names from feed-level metadata using Groq API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "GroqProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Groq API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using Groq API.

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
                "GroqProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Groq API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                self.cfg.groq_speaker_system_prompt or "groq/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call Groq API (OpenAI-compatible format)
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
                logger.warning("Groq API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Groq speaker detection completed: %d speakers, %d hosts, success=%s",
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
            logger.error("Failed to parse Groq API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Groq API error in speaker detection: %s", exc)
            raise ValueError(f"Groq speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For Groq provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.groq_speaker_user_prompt
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
        """Parse speaker names from Groq API response."""
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
        """Summarize text using Groq API.

        Can handle full transcripts directly due to large context window (128k tokens).
        No chunking needed for most podcast transcripts.

        Key advantage: Ultra-fast inference (500+ tokens/second) - 10x faster than other providers.

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
                "GroqProvider summarization not initialized. Call initialize() first."
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
            "Summarizing text via Groq API (model: %s, max_tokens: %d)",
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

            # Call Groq API (OpenAI-compatible format)
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
                logger.warning("Groq API returned empty summary")
                summary = ""

            logger.debug("Groq summarization completed: %d characters", len(summary))

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
                "summary_short": None,  # Groq provider doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "groq",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("Groq API error in summarization: %s", exc)
            raise ValueError(f"Groq summarization failed: {exc}") from exc

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
            self.cfg.groq_summary_system_prompt or "groq/summarization/system_v1"
        )
        user_prompt_name = self.cfg.groq_summary_user_prompt

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

    elif provider_type == "groq":
        from ..providers.groq.groq_provider import GroqProvider

        if experiment_mode:
            from ..config import Config
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",
                speaker_detector_provider="groq",
                groq_speaker_model=params.model_name if params.model_name else "llama-3.3-70b-versatile",
                groq_temperature=params.temperature if params.temperature is not None else 0.3,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            )
            return GroqProvider(cfg)
        else:
            return GroqProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'spacy', 'openai', 'anthropic', 'deepseek', 'groq'"
        )
```

Similar update for `summarization/factory.py`.

### 7. Dependencies

**No new dependencies required** - Groq uses OpenAI SDK:

```toml
# pyproject.toml - No changes needed
# Groq uses existing openai package with custom base_url
```

**Note:** Groq provider requires `openai` package (already a dependency for OpenAI provider).

### 8. Prompt Templates

Create Groq-specific prompts in `src/podcast_scraper/prompts/groq/`:

- `ner/system_ner_v1.j2` - System prompt for speaker detection
- `ner/guest_host_v1.j2` - User prompt for speaker detection
- `summarization/system_v1.j2` - System prompt for summarization
- `summarization/long_v1.j2` - User prompt for summarization

Follow OpenAI prompt patterns but optimize for Llama/Mixtral models.

## Testing Strategy

Same pattern as OpenAI provider:

1. **Unit tests**: Mock Groq API responses (using OpenAI SDK mocks)
2. **Integration tests**: Use E2E mock server with Groq endpoints (reuse OpenAI format)
3. **E2E tests**: Full workflow with Groq provider

## Success Criteria

1. ✅ Groq supports speaker detection and summarization via unified provider
2. ✅ Clear error when attempting transcription with Groq
3. ✅ 10x faster than other providers (500+ tokens/second)
4. ✅ Free tier works for development
5. ✅ No new SDK dependency (uses OpenAI SDK)
6. ✅ E2E tests pass
7. ✅ Experiment mode supported from start
8. ✅ Environment-based model defaults (test vs prod)
9. ✅ Follows OpenAI provider pattern exactly

## Migration Notes

- **Breaking Changes**: None (new provider, backward compatible)
- **Configuration**: Add `GROQ_API_KEY` to `.env` file
- **Dependencies**: No new dependencies (uses existing `openai` package)

## References

- **Related PRD**: `docs/prd/PRD-013-groq-provider-integration.md`
- **Reference Implementation**: `src/podcast_scraper/providers/openai/openai_provider.py`
- **DeepSeek RFC**: `docs/rfc/RFC-034-deepseek-provider-implementation.md` (similar pattern)
- **Groq Documentation**: https://console.groq.com/docs
- **Groq Models**: https://console.groq.com/docs/models
