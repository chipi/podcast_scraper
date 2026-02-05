# RFC-037: Ollama Provider Implementation (Revised)

- **Status**: Draft (Revised)
- **Revision**: 2
- **Date**: 2026-02-04
- **Authors**:
- **Stakeholders**: Maintainers, privacy-conscious users, offline users
- **Related PRDs**:
  - `docs/prd/PRD-014-ollama-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference - unified provider pattern)
  - `docs/rfc/RFC-034-deepseek-provider-implementation.md` (OpenAI SDK pattern)
  - `docs/rfc/RFC-036-grok-provider-implementation.md` (similar pattern)

## Abstract

Design and implement Ollama as a unified provider for speaker detection and summarization capabilities. Ollama is unique in being a **fully local/offline solution** that runs open-source LLMs on your own hardware with ZERO API costs. This RFC follows the **unified provider pattern** established by OpenAI (RFC-013), where a single provider class implements multiple protocols. The key implementation detail is using the OpenAI SDK with a custom `base_url` pointing to Ollama's local API.

**Key Advantages:**

- No internet required
- No API costs
- No rate limits
- Complete data privacy

**Architecture Alignment:** Ollama provider follows the exact same unified provider pattern as `OpenAIProvider`, implementing two protocols (`SpeakerDetector`, `SummarizationProvider`) in a single class and integrating via the existing factory pattern with support for both Config-based and experiment-based modes.

## Problem Statement

Users want Ollama for:

1. **Speaker Detection**: Entity extraction using local Llama/Mistral models
2. **Summarization**: High-quality summaries using local models

**Note:** Transcription is NOT supported (Ollama hosts LLMs only).

Key advantages:

- **Fully offline** - no internet required
- **Zero cost** - no per-token pricing
- **Complete privacy** - data never leaves local machine
- **OpenAI-compatible API** - uses same SDK

Requirements:

- No changes to end-user experience when using defaults
- Secure connection handling (no API key needed, but validate Ollama is running)
- Per-capability provider selection
- Use OpenAI SDK with custom base_url (no new dependency)
- Handle capability gaps gracefully (transcription not supported)
- Support both Config-based and experiment-based factory modes
- Validate Ollama server is running and models are available

## Constraints & Assumptions

**Constraints:**

- Ollama must be installed and running locally
- Models must be pulled before use (`ollama pull llama3.3`)
- Performance depends on local hardware
- No audio transcription support
- **Must follow unified provider pattern** (like OpenAI)
- No API key required (local service)

**Assumptions:**

- User has sufficient hardware for model of choice
- Ollama server is running on default port (11434)
- User has already pulled desired models

## Design & Implementation

### 0. Ollama API Overview

Ollama provides an OpenAI-compatible API:

| Feature | OpenAI | Ollama |
| ------- | ------ | ------ |
| **Base URL** | `https://api.openai.com/v1` | `http://localhost:11434/v1` |
| **API Key** | Required | Not required (local) |
| **SDK** | `openai` | `openai` (with custom base_url) |
| **Audio API** | ✅ Whisper | ❌ Not available |
| **Pricing** | Per token | **Free** |
| **Rate Limits** | Yes | **No** |
| **Provider Pattern** | Unified (`OpenAIProvider`) | Unified (`OllamaProvider`) |

### 1. Architecture Overview

**Unified Provider Pattern** (following OpenAI):

```text
src/podcast_scraper/
├── providers/
│   └── ollama/                        # NEW: Unified Ollama provider
│       ├── __init__.py
│       └── ollama_provider.py         # Single class implementing 2 protocols
├── prompts/
│   └── ollama/                        # NEW: Ollama/Llama-specific prompts
│       ├── ner/
│       │   ├── system_ner_v1.j2
│       │   └── guest_host_v1.j2
│       └── summarization/
│           ├── system_v1.j2
│           └── long_v1.j2
├── speaker_detectors/
│   └── factory.py                     # Updated: Add "ollama" option
├── summarization/
│   └── factory.py                     # Updated: Add "ollama" option
└── config.py                         # Updated: Add Ollama fields
```

**Key Architectural Decision:** Use unified provider pattern (single `OllamaProvider` class) matching `OpenAIProvider`, not separate files per capability.

### 2. Configuration

Add to `config.py` following OpenAI pattern exactly:

```python
from typing import Literal, Optional

# Provider Selection (updated)
speaker_detector_provider: Literal["spacy", "openai", "anthropic", "deepseek", "grok", "ollama"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "anthropic", "deepseek", "grok", "ollama"] = Field(
    default="transformers",
    description="Summarization provider"
)

# Ollama API Configuration (NO API KEY NEEDED - local service)
ollama_api_base: Optional[str] = Field(
    default=None,
    alias="ollama_api_base",
    description="Ollama API base URL (default: http://localhost:11434/v1, for E2E testing)"
)

# Ollama Model Selection (environment-based defaults, like OpenAI)
ollama_speaker_model: str = Field(
    default_factory=_get_default_ollama_speaker_model,
    alias="ollama_speaker_model",
    description="Ollama model for speaker detection (default: environment-based)"
)

ollama_summary_model: str = Field(
    default_factory=_get_default_ollama_summary_model,
    alias="ollama_summary_model",
    description="Ollama model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
ollama_temperature: float = Field(
    default=0.3,
    alias="ollama_temperature",
    description="Temperature for Ollama generation (0.0-2.0, lower = more deterministic)"
)

ollama_max_tokens: Optional[int] = Field(
    default=None,
    alias="ollama_max_tokens",
    description="Max tokens for Ollama generation (None = model default)"
)

# Ollama Connection Settings
ollama_timeout: int = Field(
    default=120,
    alias="ollama_timeout",
    description="Timeout in seconds for Ollama API calls (local inference can be slow)"
)

# Ollama Prompt Configuration (following OpenAI pattern)
ollama_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="ollama_speaker_system_prompt",
    description="Ollama system prompt for speaker detection (default: ollama/ner/system_ner_v1)"
)

ollama_speaker_user_prompt: str = Field(
    default="ollama/ner/guest_host_v1",
    alias="ollama_speaker_user_prompt",
    description="Ollama user prompt for speaker detection"
)

ollama_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="ollama_summary_system_prompt",
    description="Ollama system prompt for summarization (default: ollama/summarization/system_v1)"
)

ollama_summary_user_prompt: str = Field(
    default="ollama/summarization/long_v1",
    alias="ollama_summary_user_prompt",
    description="Ollama user prompt for summarization"
)
```

**Environment-based defaults** (like OpenAI):

```python
# In config_constants.py
TEST_DEFAULT_OLLAMA_SPEAKER_MODEL = "llama3.2:latest"  # Smaller, faster for testing
PROD_DEFAULT_OLLAMA_SPEAKER_MODEL = "llama3.3:latest"  # Best quality

TEST_DEFAULT_OLLAMA_SUMMARY_MODEL = "llama3.2:latest"  # Smaller, faster for testing
PROD_DEFAULT_OLLAMA_SUMMARY_MODEL = "llama3.3:latest"  # Best quality, 128k context

# In config.py
def _get_default_ollama_speaker_model() -> str:
    """Get default Ollama speaker detection model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_OLLAMA_SPEAKER_MODEL
    return PROD_DEFAULT_OLLAMA_SPEAKER_MODEL

def _get_default_ollama_summary_model() -> str:
    """Get default Ollama summarization model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_OLLAMA_SUMMARY_MODEL
    return PROD_DEFAULT_OLLAMA_SUMMARY_MODEL
```

### 3. API Base URL Management

Follow OpenAI pattern, but no API key needed:

```python
# In config.py

@field_validator("ollama_api_base", mode="before")
@classmethod
def _load_ollama_api_base_from_env(cls, value: Any) -> Optional[str]:
    """Load Ollama API base URL from environment variable if not provided."""
    if value is not None:
        return value
    env_base = os.getenv("OLLAMA_API_BASE")
    if env_base:
        return env_base
    # Default to local Ollama server
    return "http://localhost:11434/v1"
```

**Note:** No API key validation needed - Ollama is a local service.

### 4. Provider Capability Validation

Add validation to prevent using Ollama for transcription:

```python
# config.py - Capability Validation

@model_validator(mode="after")
def _validate_provider_capabilities(self) -> "Config":
    """Validate provider supports requested capability."""
    if self.transcription_provider == "ollama":
        raise ValueError(
            "Ollama provider does not support transcription. "
            "Use 'whisper' (local) or 'openai' instead. "
            "See provider capability matrix in documentation."
        )
    return self
```

### 5. Unified Provider Implementation

**File**: `src/podcast_scraper/providers/ollama/ollama_provider.py`

Follow `OpenAIProvider` pattern exactly, but use OpenAI SDK with custom base_url and add Ollama-specific validation:

```python
"""Unified Ollama provider for speaker detection and summarization.

This module provides a single OllamaProvider class that implements two protocols:
- SpeakerDetector (using Ollama chat API via OpenAI SDK)
- SummarizationProvider (using Ollama chat API via OpenAI SDK)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key insight: Ollama uses an OpenAI-compatible API, so we reuse the OpenAI SDK
with a custom base_url. No new dependency required.

Key advantages:
- Fully offline - no internet required
- Zero cost - no per-token pricing
- Complete privacy - data never leaves local machine

Note: Ollama does NOT support transcription (no audio API).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Set, Tuple

try:
import httpx
except ImportError:
    httpx = None  # type: ignore

from openai import OpenAI

from ... import config, models
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

# Error messages
OLLAMA_NOT_RUNNING_ERROR = """
Ollama server is not running. Please start it with:

    ollama serve

Or install Ollama from: https://ollama.ai
"""

MODEL_NOT_FOUND_ERROR_TEMPLATE = """
Model '{model}' is not available in Ollama. Install it with:

    ollama pull {model}

Available models can be listed with:

    ollama list
"""


class OllamaProvider:
    """Unified Ollama provider implementing SpeakerDetector and SummarizationProvider.

    This provider initializes and manages:
    - Ollama chat API for speaker detection (via OpenAI SDK)
    - Ollama chat API for summarization (via OpenAI SDK)

    All capabilities share the same OpenAI client (configured with Ollama base_url),
    similar to how OpenAI providers share the same OpenAI client.

    Key advantages:
    - Fully offline - no internet required
    - Zero cost - no per-token pricing
    - Complete privacy - data never leaves local machine

    Note: Transcription is NOT supported (Ollama has no audio API).
    """

    def __init__(self, cfg: config.Config):
        """Initialize unified Ollama provider.

        Args:
            cfg: Configuration object with settings for both capabilities

        Raises:
            ValueError: If Ollama server is not running or model is not available
            ImportError: If openai or httpx packages are not installed
            ConnectionError: If Ollama server is not accessible
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required for Ollama provider. "
                "Install with: pip install 'podcast-scraper[openai]'"
            )

        if httpx is None:
            raise ImportError(
                "httpx package required for Ollama provider (for health checks). "
                "Install with: pip install 'podcast-scraper[ollama]'"
            )

        self.cfg = cfg

        # Validate Ollama server is running
        base_url = cfg.ollama_api_base or "http://localhost:11434/v1"
        self._validate_ollama_running(base_url)

        # Suppress verbose OpenAI SDK debug logs (same as OpenAI provider)
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

        # Create OpenAI client with Ollama base_url
        # Ollama doesn't require API key, but OpenAI SDK requires one (use dummy)
        client_kwargs: dict[str, Any] = {
            "api_key": "ollama",  # Ollama ignores API key, but SDK requires one
            "base_url": base_url,
            "timeout": getattr(cfg, "ollama_timeout", 120),
        }
        self.client = OpenAI(**client_kwargs)

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "ollama_speaker_model", "llama3.3:latest")
        self.speaker_temperature = getattr(cfg, "ollama_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "ollama_summary_model", "llama3.3:latest")
        self.summary_temperature = getattr(cfg, "ollama_temperature", 0.3)
        # Modern Ollama models support 128k context window
        self.max_context_tokens = 128000  # Conservative estimate

        # Initialization state
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        self._requires_separate_instances = False

    def _validate_ollama_running(self, base_url: str) -> None:
        """Validate that Ollama server is running and accessible.

        Args:
            base_url: Ollama API base URL

        Raises:
            ConnectionError: If Ollama server is not running
        """
        try:
            # Remove /v1 suffix for health check endpoint
            health_url = base_url.rstrip("/v1") + "/api/version"
            response = httpx.get(health_url, timeout=5.0)
            response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ConnectionError(OLLAMA_NOT_RUNNING_ERROR) from exc

    def _validate_model_available(self, model: str) -> None:
        """Validate that model is available in Ollama.

        Args:
            model: Model name to check

        Raises:
            ValueError: If model is not available
        """
        try:
            base_url = self.cfg.ollama_api_base or "http://localhost:11434/v1"
            health_url = base_url.rstrip("/v1") + "/api/tags"
            response = httpx.get(health_url, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        available_models = [m.get("name", "") for m in data.get("models", [])]

            if model not in available_models:
                error_msg = MODEL_NOT_FOUND_ERROR_TEMPLATE.format(model=model)
                raise ValueError(error_msg)
        except httpx.RequestError as exc:
            logger.warning("Could not validate model availability: %s", exc)
            # Don't fail - model might still work, just warn

    def initialize(self) -> None:
        """Initialize all Ollama capabilities.

        For Ollama API, initialization validates models are available.
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
        logger.debug("Initializing Ollama speaker detection (model: %s)", self.speaker_model)
        # Validate model is available
        self._validate_model_available(self.speaker_model)
        self._speaker_detection_initialized = True

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Ollama summarization (model: %s)", self.summary_model)
        # Validate model is available
        self._validate_model_available(self.summary_model)
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
        """Detect host names from feed-level metadata using Ollama API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OllamaProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Ollama API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using Ollama API.

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
            ValueError: If detection fails
            RuntimeError: If provider is not initialized
        """
        # If auto_speakers is disabled, return defaults without requiring initialization
        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, detection failed")
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False

        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OllamaProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Ollama API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                self.cfg.ollama_speaker_system_prompt or "ollama/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call Ollama API (OpenAI-compatible format)
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
                logger.warning("Ollama API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Ollama speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available (Ollama doesn't report usage, but track anyway)
            if pipeline_metrics is not None and hasattr(response, "usage"):
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                pipeline_metrics.record_llm_speaker_detection_call(input_tokens, output_tokens)

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Ollama API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Ollama API error in speaker detection: %s", exc)
            raise ValueError(f"Ollama speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For Ollama provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.ollama_speaker_user_prompt
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
        """Parse speaker names from Ollama API response."""
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
        """Summarize text using Ollama API.

        Can handle full transcripts directly due to large context window (128k tokens).
        No chunking needed for most podcast transcripts.

        Key advantage: Fully offline, zero cost, complete privacy.

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
                "OllamaProvider summarization not initialized. Call initialize() first."
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
            "Summarizing text via Ollama API (model: %s, max_tokens: %d)",
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

            # Call Ollama API (OpenAI-compatible format)
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
                logger.warning("Ollama API returned empty summary")
                summary = ""

            logger.debug("Ollama summarization completed: %d characters", len(summary))

            # Track LLM call metrics if available (Ollama doesn't report usage, but track anyway)
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
                "summary_short": None,  # Ollama provider doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "ollama",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("Ollama API error in summarization: %s", exc)
            raise ValueError(f"Ollama summarization failed: {exc}") from exc

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
            self.cfg.ollama_summary_system_prompt or "ollama/summarization/system_v1"
        )
        user_prompt_name = self.cfg.ollama_summary_user_prompt

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

    elif provider_type == "ollama":
        from ..providers.ollama.ollama_provider import OllamaProvider

        if experiment_mode:
            from ..config import Config
            assert isinstance(params, SpeakerDetectionParams)
            cfg = Config(
                rss="",
                speaker_detector_provider="ollama",
                ollama_speaker_model=params.model_name if params.model_name else "llama3.3:latest",
                ollama_temperature=params.temperature if params.temperature is not None else 0.3,
                ollama_api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
            )
            return OllamaProvider(cfg)
        else:
            return OllamaProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported speaker detector type: {provider_type}. "
            "Supported types: 'spacy', 'openai', 'anthropic', 'deepseek', 'grok', 'ollama'"
        )
```

Similar update for `summarization/factory.py`.

### 7. Dependencies

Add `httpx` for health checks (likely already installed):

```toml
[project.optional-dependencies]
ollama = [
    "httpx>=0.24.0,<1.0.0",  # For Ollama health checks
]
```

**Note:** Ollama provider requires `openai` package (already a dependency for OpenAI provider) and `httpx` for health checks.

### 8. Prompt Templates

Create Ollama-specific prompts in `src/podcast_scraper/prompts/ollama/`:

- `ner/system_ner_v1.j2` - System prompt for speaker detection
- `ner/guest_host_v1.j2` - User prompt for speaker detection
- `summarization/system_v1.j2` - System prompt for summarization
- `summarization/long_v1.j2` - User prompt for summarization

Follow OpenAI prompt patterns but optimize for Llama/Mistral models.

## Testing Strategy

Same pattern as OpenAI provider:

1. **Unit tests**: Mock Ollama API responses (using OpenAI SDK mocks)
2. **Integration tests**: Use E2E mock server with Ollama endpoints (reuse OpenAI format)
3. **E2E tests**: Full workflow with Ollama provider (skip if Ollama not running)

## Success Criteria

1. ✅ Ollama supports speaker detection and summarization via unified provider
2. ✅ Clear error when attempting transcription with Ollama
3. ✅ Clear error when Ollama server is not running
4. ✅ Clear error when model is not installed
5. ✅ Works completely offline
6. ✅ Zero API costs
7. ✅ E2E tests pass
8. ✅ Experiment mode supported from start
9. ✅ Environment-based model defaults (test vs prod)
10. ✅ Follows OpenAI provider pattern exactly

## Migration Notes

- **Breaking Changes**: None (new provider, backward compatible)
- **Configuration**: Set `OLLAMA_API_BASE` environment variable if using remote Ollama server
- **Dependencies**: Install with `pip install 'podcast-scraper[ollama]'` (adds httpx)
- **Prerequisites**: Ollama must be installed and running locally

## References

- **Related PRD**: `docs/prd/PRD-014-ollama-provider-integration.md`
- **Reference Implementation**: `src/podcast_scraper/providers/openai/openai_provider.py`
- **Ollama Documentation**: https://ollama.ai
- **Ollama API Reference**: https://github.com/ollama/ollama/blob/main/docs/api.md
- **OpenAI Compatibility**: https://ollama.ai/blog/openai-compatibility
