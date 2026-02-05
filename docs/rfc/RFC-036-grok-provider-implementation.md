# RFC-036: Grok Provider Implementation (xAI)

- **Status**: Planned
- **Revision**: 3
- **Date**: 2026-02-05
- **Implementation**: Issue #1095
- **Authors**:
- **Stakeholders**: Maintainers, users wanting xAI's Grok integration
- **Related PRDs**:
  - `docs/prd/PRD-013-grok-provider-integration.md` (Updated)
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference - unified provider pattern)
  - `docs/rfc/RFC-034-deepseek-provider-implementation.md` (same pattern - OpenAI SDK)

## Abstract

Design and implement Grok (by xAI) as a unified provider for speaker detection and summarization capabilities. Grok is xAI's AI model with access to real-time information via X/Twitter integration. This RFC follows the **unified provider pattern** established by OpenAI (RFC-013), where a single provider class implements multiple protocols. The key implementation detail depends on whether Grok's API is OpenAI-compatible (use OpenAI SDK with custom `base_url`) or requires xAI's SDK.

**API Details (Researched):** Based on xAI's public API and OpenAI-compatible patterns:
- Base URL: `https://api.x.ai/v1` (OpenAI-compatible endpoint)
- SDK: Uses OpenAI SDK with custom `base_url` (no new dependency)
- Model names: `grok-beta` (beta), `grok-2` (production) - verify with your API access
- Pricing: Verify at https://console.x.ai or https://docs.x.ai

**Architecture Alignment:** Grok provider follows the exact same unified provider pattern as `OpenAIProvider`, implementing two protocols (`SpeakerDetector`, `SummarizationProvider`) in a single class and integrating via the existing factory pattern with support for both Config-based and experiment-based modes.

## Problem Statement

Users want Grok (xAI) for:

1. **Speaker Detection**: Entity extraction using Grok models
2. **Summarization**: High-quality summaries with real-time information access

**Note:** Transcription is NOT supported (Grok/xAI focuses on text-based LLMs, no audio models).

Key advantages:

- **Real-time information** access via X/Twitter integration
- **xAI's Grok model** - proprietary AI model
- **OpenAI-compatible API** - reuses OpenAI SDK with custom base_url
- **Free tier** - verify availability with your xAI account

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
- Rate limits (details need verification from xAI docs)
- Model selection limited to xAI's Grok models
- **Must follow unified provider pattern** (like OpenAI)

**Assumptions:**

- Grok API is OpenAI-compatible (confirmed - uses OpenAI SDK with custom base_url)
- Alternative: xAI may provide their own SDK (needs research)
- Grok model quality is suitable for speaker detection and summarization

## Design & Implementation

### 0. Grok API Overview

**API Details (Researched):** Based on xAI's public API and OpenAI-compatible patterns:

| Feature | OpenAI | Grok (xAI) |
| ------- | ------ | --------- |
| **Base URL** | `https://api.openai.com/v1` | `https://api.x.ai/v1` (OpenAI-compatible) |
| **SDK** | `openai` | `openai` (with custom base_url) |
| **Audio API** | ✅ Whisper | ❌ Not available |
| **Speed** | ~100 tokens/sec | Verify with API testing |
| **Models** | Proprietary | Grok models (`grok-beta`, `grok-2` - verify with your API) |
| **Provider Pattern** | Unified (`OpenAIProvider`) | Unified (`GrokProvider`) |
| **Real-time Info** | ❌ | ✅ (via X/Twitter integration) |

### 1. Architecture Overview

**Unified Provider Pattern** (following OpenAI):

```text
src/podcast_scraper/
├── providers/
│   └── grok/                          # NEW: Unified Grok provider
│       ├── __init__.py
│       └── grok_provider.py           # Single class implementing 2 protocols
├── prompts/
│   └── grok/                          # NEW: Grok-specific prompts
│       ├── ner/
│       │   ├── system_ner_v1.j2
│       │   └── guest_host_v1.j2
│       └── summarization/
│           ├── system_v1.j2
│           └── long_v1.j2
├── speaker_detectors/
│   └── factory.py                     # Updated: Add "grok" option
├── summarization/
│   └── factory.py                     # Updated: Add "grok" option
└── config.py                         # Updated: Add Grok fields
```

**Key Architectural Decision:** Use unified provider pattern (single `GrokProvider` class) matching `OpenAIProvider`, not separate files per capability.

### 2. Configuration

Add to `config.py` following OpenAI pattern exactly:

```python
from typing import Literal, Optional

# Provider Selection (updated)
speaker_detector_provider: Literal["spacy", "openai", "anthropic", "deepseek", "grok"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "anthropic", "deepseek", "grok"] = Field(
    default="transformers",
    description="Summarization provider"
)

# Grok API Configuration (following OpenAI pattern)
grok_api_key: Optional[str] = Field(
    default=None,
    alias="grok_api_key",
    description="Grok API key (prefer GROK_API_KEY env var or .env file)"
)

grok_api_base: Optional[str] = Field(
    default=None,
    alias="grok_api_base",
    description="Grok API base URL (default: https://api.x.ai/v1, for E2E testing)"
)

# Grok Model Selection (environment-based defaults, like OpenAI)
grok_speaker_model: str = Field(
    default_factory=_get_default_grok_speaker_model,
    alias="grok_speaker_model",
    description="Grok model for speaker detection (default: environment-based)"
)

grok_summary_model: str = Field(
    default_factory=_get_default_grok_summary_model,
    alias="grok_summary_model",
    description="Grok model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
grok_temperature: float = Field(
    default=0.3,
    alias="grok_temperature",
    description="Temperature for Grok generation (0.0-2.0, lower = more deterministic)"
)

grok_max_tokens: Optional[int] = Field(
    default=None,
    alias="grok_max_tokens",
    description="Max tokens for Grok generation (None = model default)"
)

# Grok Prompt Configuration (following OpenAI pattern)
grok_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="grok_speaker_system_prompt",
    description="Grok system prompt for speaker detection (default: grok/ner/system_ner_v1)"
)

grok_speaker_user_prompt: str = Field(
    default="grok/ner/guest_host_v1",
    alias="grok_speaker_user_prompt",
    description="Grok user prompt for speaker detection"
)

grok_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="grok_summary_system_prompt",
    description="Grok system prompt for summarization (default: grok/summarization/system_v1)"
)

grok_summary_user_prompt: str = Field(
    default="grok/summarization/long_v1",
    alias="grok_summary_user_prompt",
    description="Grok user prompt for summarization"
)
```

**Environment-based defaults** (verify model names with your xAI API access):

```python
# In config_constants.py
TEST_DEFAULT_GROK_SPEAKER_MODEL = "grok-beta"  # Beta model for development
PROD_DEFAULT_GROK_SPEAKER_MODEL = "grok-2"  # Production model

TEST_DEFAULT_GROK_SUMMARY_MODEL = "grok-beta"  # Beta model for development
PROD_DEFAULT_GROK_SUMMARY_MODEL = "grok-2"  # Production model

# In config.py
def _get_default_grok_speaker_model() -> str:
    """Get default Grok speaker detection model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_GROK_SPEAKER_MODEL
    return PROD_DEFAULT_GROK_SPEAKER_MODEL

def _get_default_grok_summary_model() -> str:
    """Get default Grok summarization model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_GROK_SUMMARY_MODEL
    return PROD_DEFAULT_GROK_SUMMARY_MODEL
```

### 3. API Key Management

Follow OpenAI pattern exactly:

```python
# In config.py

@field_validator("grok_api_key", mode="before")
@classmethod
def _load_grok_api_key_from_env(cls, value: Any) -> Optional[str]:
    """Load Grok API key from environment variable if not provided."""
    if value is not None:
        return value
    env_key = os.getenv("GROK_API_KEY")
    if env_key:
        return env_key
    return None

@field_validator("grok_api_base", mode="before")
@classmethod
def _load_grok_api_base_from_env(cls, value: Any) -> Optional[str]:
    """Load Grok API base URL from environment variable if not provided."""
    if value is not None:
        return value
    env_base = os.getenv("GROK_API_BASE")
    if env_base:
        return env_base
    # Default to Grok API base URL (OpenAI-compatible endpoint)
    return "https://api.x.ai/v1"

@model_validator(mode="after")
def _validate_grok_provider_requirements(self) -> "Config":
    """Validate that Grok API key is provided when Grok providers are selected."""
    grok_providers_used = []
    if self.speaker_detector_provider == "grok":
        grok_providers_used.append("speaker_detection")
    if self.summary_provider == "grok":
        grok_providers_used.append("summarization")

    if grok_providers_used and not self.grok_api_key:
        providers_str = ", ".join(grok_providers_used)
        raise ValueError(
            f"Grok API key required for Grok providers: {providers_str}. "
            "Set GROK_API_KEY environment variable or grok_api_key in config."
        )

    return self
```

### 4. Provider Capability Validation

Add validation to prevent using Grok for transcription:

```python
# config.py - Capability Validation

@model_validator(mode="after")
def _validate_provider_capabilities(self) -> "Config":
    """Validate provider supports requested capability."""
    if self.transcription_provider == "grok":
        raise ValueError(
            "Grok provider does not support transcription. "
            "Use 'whisper' (local) or 'openai' instead. "
            "See provider capability matrix in documentation."
        )
    return self
```

### 5. Unified Provider Implementation

**File**: `src/podcast_scraper/providers/grok/grok_provider.py`

Follow `OpenAIProvider` pattern exactly. Implementation depends on whether Grok API is OpenAI-compatible:

**Option A:** If Grok API is OpenAI-compatible, use OpenAI SDK with custom base_url (same as DeepSeek pattern).

**Option B:** If Grok requires xAI SDK, use xAI SDK directly.

```python
"""Unified Grok provider for speaker detection and summarization.

This module provides a single GrokProvider class that implements two protocols:
- SpeakerDetector (using Grok chat API)
- SummarizationProvider (using Grok chat API)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key insight: Grok API is OpenAI-compatible:
- Reuse OpenAI SDK with custom base_url (no new dependency)
- Same pattern as DeepSeek provider

Key advantage: Real-time information access via X/Twitter integration.

Note: Grok does NOT support transcription (no audio API).
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

# Grok API pricing constants (for cost estimation)
# Source: Verify at https://console.x.ai or https://docs.x.ai
# Last updated: 2026-02-05
# Note: Pricing should be verified from xAI's official documentation
GROK_BETA_INPUT_COST_PER_1M_TOKENS = 0.0  # Verify with xAI pricing
GROK_BETA_OUTPUT_COST_PER_1M_TOKENS = 0.0  # Verify with xAI pricing
GROK_2_INPUT_COST_PER_1M_TOKENS = 0.0  # Verify with xAI pricing
GROK_2_OUTPUT_COST_PER_1M_TOKENS = 0.0  # Verify with xAI pricing


class GrokProvider:
    """Unified Grok provider implementing SpeakerDetector and SummarizationProvider.

    This provider initializes and manages:
    - Grok chat API for speaker detection
    - Grok chat API for summarization

    All capabilities share the same API client (configured with Grok base_url),
    similar to how OpenAI providers share the same OpenAI client.

    Key advantage: Real-time information access via X/Twitter integration.

    Note: Transcription is NOT supported (Grok has no audio API).
    """

    def __init__(self, cfg: config.Config):
        """Initialize unified Grok provider.

        Args:
            cfg: Configuration object with settings for both capabilities

        Raises:
            ValueError: If Grok API key is not provided
            ImportError: If required SDK package is not installed
        """
        # Grok uses OpenAI-compatible API (reuse OpenAI SDK)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for Grok provider. "
                "Install with: pip install 'podcast-scraper[openai]'"
            )

        if not cfg.grok_api_key:
            raise ValueError(
                "Grok API key required for Grok provider. "
                "Set GROK_API_KEY environment variable or grok_api_key in config."
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
        # Default to Grok API base URL (OpenAI-compatible endpoint)
        base_url = cfg.grok_api_base or "https://api.x.ai/v1"
        client_kwargs: dict[str, Any] = {
            "api_key": cfg.grok_api_key,
            "base_url": base_url,
        }
        # Grok uses OpenAI-compatible API (same pattern as DeepSeek)
        self.client = OpenAI(**client_kwargs)

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "grok_speaker_model", "grok-2")  # Default: production model
        self.speaker_temperature = getattr(cfg, "grok_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "grok_summary_model", "grok-2")  # Default: production model
        self.summary_temperature = getattr(cfg, "grok_temperature", 0.3)
        # Context window size (verify with xAI documentation - common is 128k)
        self.max_context_tokens = 128000  # Conservative estimate, verify with API docs

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
        """Initialize all Grok capabilities.

        For Grok API, initialization is a no-op but we track it for consistency.
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
        logger.debug("Initializing Grok speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Grok summarization (model: %s)", self.summary_model)
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
        """Detect host names from feed-level metadata using Grok API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "GrokProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Grok API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using Grok API.

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
                "GrokProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Grok API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                self.cfg.grok_speaker_system_prompt or "grok/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call Grok API (OpenAI-compatible format)
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
                logger.warning("Grok API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Grok speaker detection completed: %d speakers, %d hosts, success=%s",
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
            logger.error("Failed to parse Grok API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Grok API error in speaker detection: %s", exc)
            raise ValueError(f"Grok speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For Grok provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.grok_speaker_user_prompt
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
        """Parse speaker names from Grok API response."""
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
        """Summarize text using Grok API.

        Can handle full transcripts directly if context window is large enough.
        Chunking may be needed depending on Grok's context window size (likely 128k, verify with API docs).

        Key advantage: Real-time information access via X/Twitter integration.

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
                "GrokProvider summarization not initialized. Call initialize() first."
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
            "Summarizing text via Grok API (model: %s, max_tokens: %d)",
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

            # Call Grok API (OpenAI-compatible format)
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
                logger.warning("Grok API returned empty summary")
                summary = ""

            logger.debug("Grok summarization completed: %d characters", len(summary))

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
                "summary_short": None,  # Grok provider doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "grok",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("Grok API error in summarization: %s", exc)
            raise ValueError(f"Grok summarization failed: {exc}") from exc

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
            self.cfg.grok_summary_system_prompt or "grok/summarization/system_v1"
        )
        user_prompt_name = self.cfg.grok_summary_user_prompt

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

    elif provider_type == "grok":
        from ..providers.grok.grok_provider import GrokProvider

        if experiment_mode:
            from ..config import Config
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",
                speaker_detector_provider="grok",
                grok_speaker_model=params.model_name if params.model_name else "grok-2",  # Default: production model
                grok_temperature=params.temperature if params.temperature is not None else 0.3,
                grok_api_key=os.getenv("GROK_API_KEY"),
            )
            return GrokProvider(cfg)
        else:
            return GrokProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'spacy', 'openai', 'anthropic', 'deepseek', 'grok'"
        )
```

Similar update for `summarization/factory.py`.

### 7. Dependencies

**Dependencies (OpenAI-compatible API confirmed):**

**Grok uses OpenAI-compatible API:**
```toml
# pyproject.toml - No changes needed
# Grok uses existing openai package with custom base_url (same as DeepSeek)
```

**Note:** No new dependencies required. Grok follows the same pattern as DeepSeek provider.

### 8. Prompt Templates

Create Grok-specific prompts in `src/podcast_scraper/prompts/grok/`:

- `ner/system_ner_v1.j2` - System prompt for speaker detection
- `ner/guest_host_v1.j2` - User prompt for speaker detection
- `summarization/system_v1.j2` - System prompt for summarization
- `summarization/long_v1.j2` - User prompt for summarization

Follow OpenAI prompt patterns but optimize for Grok models.

## Testing Strategy

Same pattern as OpenAI provider:

1. **Unit tests**: Mock Grok API responses (using OpenAI SDK mocks or xAI SDK mocks)
2. **Integration tests**: Use E2E mock server with Grok endpoints (reuse OpenAI format if compatible)
3. **E2E tests**: Full workflow with Grok provider

## Success Criteria

1. ✅ Grok supports speaker detection and summarization via unified provider
2. ✅ Clear error when attempting transcription with Grok
3. ✅ API integration works (OpenAI-compatible or xAI SDK)
4. ✅ Free tier works for development (if available)
5. ✅ Minimal new dependencies
6. ✅ E2E tests pass
7. ✅ Experiment mode supported from start
8. ✅ Environment-based model defaults (test vs prod)
9. ✅ Follows OpenAI provider pattern exactly

## Migration Notes

- **Breaking Changes**: None (new provider, backward compatible)
- **Configuration**: Add `GROK_API_KEY` to `.env` file
- **Dependencies**: Depends on API compatibility (may need xAI SDK or may reuse OpenAI SDK)

## Research Required

Before implementation, verify the following with your xAI API access:

1. ✅ **API Availability**: Grok API is publicly available (you have access)
2. ✅ **Base URL**: `https://api.x.ai/v1` (OpenAI-compatible endpoint)
3. ✅ **API Compatibility**: OpenAI-compatible (uses OpenAI SDK with custom base_url)
4. ⚠️ **Model Names**: Verify actual model names with your API (likely `grok-beta`, `grok-2`)
5. ⚠️ **Pricing**: Verify pricing at https://console.x.ai or https://docs.x.ai
6. ⚠️ **Free Tier**: Check your xAI account dashboard for free tier availability and limits
7. ⚠️ **Rate Limits**: Check your xAI account dashboard for rate limits
8. ⚠️ **Context Window**: Verify context window size (likely 128k, but confirm)
9. ✅ **SDK**: Uses OpenAI SDK (no xAI-specific SDK needed)

## References

- **Related PRD**: `docs/prd/PRD-013-grok-provider-integration.md` (Updated)
- **Reference Implementation**: `src/podcast_scraper/providers/openai/openai_provider.py`
- **DeepSeek RFC**: `docs/rfc/RFC-034-deepseek-provider-implementation.md` (similar pattern if OpenAI-compatible)
- **xAI Documentation**: https://docs.x.ai or https://console.x.ai
- **Grok Models**: Verify with your xAI API access (likely `grok-beta`, `grok-2`)
