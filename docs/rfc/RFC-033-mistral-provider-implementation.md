# RFC-033: Mistral Provider Implementation (Revised)

- **Status**: ✅ Completed (v2.5.0)
- **Revision**: 2
- **Date**: 2026-02-04
- **Authors**:
- **Stakeholders**: Maintainers, users wanting Mistral API integration, developers implementing providers
- **Related PRDs**:
  - `docs/prd/PRD-010-mistral-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference - unified provider pattern)
  - `docs/rfc/RFC-032-anthropic-provider-implementation.md` (similar pattern - no transcription)
  - `docs/rfc/RFC-021-modularization-refactoring-plan.md` (architecture foundation)
  - `docs/rfc/RFC-017-prompt-management.md` (prompt system)

## Abstract

Design and implement Mistral AI as a unified provider for transcription, speaker detection, and summarization capabilities. Mistral is unique among cloud providers in supporting ALL three capabilities, making it a complete OpenAI alternative. This RFC builds on the existing modularization architecture (RFC-021) and follows the **unified provider pattern** established by OpenAI (RFC-013), where a single provider class implements multiple protocols.

**Architecture Alignment:** Mistral provider follows the exact same unified provider pattern as `OpenAIProvider`, implementing three protocols (`TranscriptionProvider`, `SpeakerDetector`, `SummarizationProvider`) in a single class and integrating via the existing factory pattern with support for both Config-based and experiment-based modes.

## Problem Statement

Users want the option to use Mistral AI as a complete alternative to OpenAI for:

1. **Transcription**: Audio-to-text using Voxtral models
2. **Speaker Detection**: Entity extraction using Mistral chat models
3. **Summarization**: High-quality summaries using Mistral chat models

Unlike Anthropic, Mistral supports ALL three capabilities, making it a true OpenAI alternative.

Requirements:

- No changes to end-user experience or workflow when using defaults
- Secure API key management (environment variables, never in source code)
- Per-capability provider selection (can mix local, OpenAI, Anthropic, and Mistral)
- Build on existing modularization and provider architecture
- Use Mistral-specific prompts (prompts are provider-specific)
- Handle Voxtral API differences from OpenAI Whisper API
- Support both Config-based and experiment-based factory modes

## Constraints & Assumptions

**Constraints:**

- **Prerequisite**: Modularization refactoring (RFC-021) ✅ Completed
- **Prerequisite**: OpenAI provider implementation (RFC-013) ✅ Completed
- **Backward Compatibility**: Default providers (local) must remain unchanged
- **API Key Security**: API keys must never be in source code or committed files
- **Rate Limits**: Must respect Mistral API rate limits and implement retry logic
- **Must follow unified provider pattern** (like OpenAI)

**Assumptions:**

- Mistral API is stable and well-documented
- Mistral Python SDK follows similar patterns to OpenAI/Anthropic SDKs
- Voxtral transcription API follows similar patterns to OpenAI Whisper API
- Prompts need to be optimized for Mistral (may differ from GPT/Claude)

## Design & Implementation

### 0. Mistral API Overview

Mistral's API is similar to OpenAI but with some differences:

| Feature | OpenAI | Mistral |
| ------- | ------ | ------- |
| **Chat Endpoint** | `/v1/chat/completions` | `/v1/chat/completions` |
| **Transcription Endpoint** | `/v1/audio/transcriptions` | `/v1/audio/transcriptions` |
| **Audio Models** | whisper-1 | voxtral-mini-latest |
| **Context Window** | 128k tokens | 256k tokens (large) |
| **Temperature Range** | 0.0 - 2.0 | 0.0 - 1.0 |
| **Python SDK** | `openai` | `mistralai` |
| **Provider Pattern** | Unified (`OpenAIProvider`) | Unified (`MistralProvider`) |

### 1. Architecture Overview

**Unified Provider Pattern** (following OpenAI):

```text
src/podcast_scraper/
├── providers/
│   └── mistral/                      # NEW: Unified Mistral provider
│       ├── __init__.py
│       └── mistral_provider.py       # Single class implementing 3 protocols
├── prompts/
│   └── mistral/                      # NEW: Mistral-specific prompts
│       ├── ner/
│       │   ├── system_ner_v1.j2
│       │   └── guest_host_v1.j2
│       └── summarization/
│           ├── system_v1.j2
│           └── long_v1.j2
├── transcription/
│   └── factory.py                    # Updated: Add "mistral" option
├── speaker_detectors/
│   └── factory.py                    # Updated: Add "mistral" option
├── summarization/
│   └── factory.py                    # Updated: Add "mistral" option
└── config.py                         # Updated: Add Mistral fields
```

**Key Architectural Decision:** Use unified provider pattern (single `MistralProvider` class) matching `OpenAIProvider`, not separate files per capability.

### 2. Configuration

Add to `config.py` following OpenAI pattern exactly:

```python
from typing import Literal, Optional

# Provider Selection (updated to include mistral)
transcription_provider: Literal["whisper", "openai", "mistral"] = Field(
    default="whisper",
    description="Transcription provider"
)

speaker_detector_provider: Literal["spacy", "openai", "anthropic", "mistral"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "anthropic", "mistral"] = Field(
    default="transformers",
    description="Summarization provider"
)

# Mistral API Configuration (following OpenAI pattern)
mistral_api_key: Optional[str] = Field(
    default=None,
    alias="mistral_api_key",
    description="Mistral API key (prefer MISTRAL_API_KEY env var or .env file)"
)

mistral_api_base: Optional[str] = Field(
    default=None,
    alias="mistral_api_base",
    description="Mistral API base URL (for E2E testing with mock servers)"
)

# Mistral Model Selection (environment-based defaults, like OpenAI)
mistral_transcription_model: str = Field(
    default_factory=_get_default_mistral_transcription_model,
    alias="mistral_transcription_model",
    description="Mistral Voxtral model for transcription (default: environment-based)"
)

mistral_speaker_model: str = Field(
    default_factory=_get_default_mistral_speaker_model,
    alias="mistral_speaker_model",
    description="Mistral model for speaker detection (default: environment-based)"
)

mistral_summary_model: str = Field(
    default_factory=_get_default_mistral_summary_model,
    alias="mistral_summary_model",
    description="Mistral model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
mistral_temperature: float = Field(
    default=0.3,
    alias="mistral_temperature",
    description="Temperature for Mistral generation (0.0-1.0, lower = more deterministic)"
)

mistral_max_tokens: Optional[int] = Field(
    default=None,
    alias="mistral_max_tokens",
    description="Max tokens for Mistral generation (None = model default)"
)

# Mistral Prompt Configuration (following OpenAI pattern)
mistral_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="mistral_speaker_system_prompt",
    description="Mistral system prompt for speaker detection (default: mistral/ner/system_ner_v1)"
)

mistral_speaker_user_prompt: str = Field(
    default="mistral/ner/guest_host_v1",
    alias="mistral_speaker_user_prompt",
    description="Mistral user prompt for speaker detection"
)

mistral_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="mistral_summary_system_prompt",
    description="Mistral system prompt for summarization (default: mistral/summarization/system_v1)"
)

mistral_summary_user_prompt: str = Field(
    default="mistral/summarization/long_v1",
    alias="mistral_summary_user_prompt",
    description="Mistral user prompt for summarization"
)
```

**Environment-based defaults** (like OpenAI):

```python
# In config_constants.py
TEST_DEFAULT_MISTRAL_TRANSCRIPTION_MODEL = "voxtral-mini-latest"
PROD_DEFAULT_MISTRAL_TRANSCRIPTION_MODEL = "voxtral-mini-latest"  # Only option

TEST_DEFAULT_MISTRAL_SPEAKER_MODEL = "mistral-small-latest"  # Cheapest text
PROD_DEFAULT_MISTRAL_SPEAKER_MODEL = "mistral-large-latest"  # Best quality

TEST_DEFAULT_MISTRAL_SUMMARY_MODEL = "mistral-small-latest"  # Cheapest text
PROD_DEFAULT_MISTRAL_SUMMARY_MODEL = "mistral-large-latest"  # Best quality, 256k context

# In config.py
def _get_default_mistral_transcription_model() -> str:
    """Get default Mistral transcription model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_MISTRAL_TRANSCRIPTION_MODEL
    return PROD_DEFAULT_MISTRAL_TRANSCRIPTION_MODEL

def _get_default_mistral_speaker_model() -> str:
    """Get default Mistral speaker detection model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_MISTRAL_SPEAKER_MODEL
    return PROD_DEFAULT_MISTRAL_SPEAKER_MODEL

def _get_default_mistral_summary_model() -> str:
    """Get default Mistral summarization model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_MISTRAL_SUMMARY_MODEL
    return PROD_DEFAULT_MISTRAL_SUMMARY_MODEL
```

### 3. API Key Management

Follow OpenAI pattern exactly:

```python
# In config.py

@field_validator("mistral_api_key", mode="before")
@classmethod
def _load_mistral_api_key_from_env(cls, value: Any) -> Optional[str]:
    """Load Mistral API key from environment variable if not provided."""
    if value is not None:
        return value
    env_key = os.getenv("MISTRAL_API_KEY")
    if env_key:
        return env_key
    return None

@field_validator("mistral_api_base", mode="before")
@classmethod
def _load_mistral_api_base_from_env(cls, value: Any) -> Optional[str]:
    """Load Mistral API base URL from environment variable if not provided."""
    if value is not None:
        return value
    env_base = os.getenv("MISTRAL_API_BASE")
    if env_base:
        return env_base
    return None

@model_validator(mode="after")
def _validate_mistral_provider_requirements(self) -> "Config":
    """Validate that Mistral API key is provided when Mistral providers are selected."""
    mistral_providers_used = []
    if self.transcription_provider == "mistral":
        mistral_providers_used.append("transcription")
    if self.speaker_detector_provider == "mistral":
        mistral_providers_used.append("speaker_detection")
    if self.summary_provider == "mistral":
        mistral_providers_used.append("summarization")

    if mistral_providers_used and not self.mistral_api_key:
        providers_str = ", ".join(mistral_providers_used)
        raise ValueError(
            f"Mistral API key required for Mistral providers: {providers_str}. "
            "Set MISTRAL_API_KEY environment variable or mistral_api_key in config."
        )

    return self
```

### 4. Unified Provider Implementation

**File**: `src/podcast_scraper/providers/mistral/mistral_provider.py`

Follow `OpenAIProvider` pattern exactly, implementing all three protocols:

```python
"""Unified Mistral provider for transcription, speaker detection, and summarization.

This module provides a single MistralProvider class that implements three protocols:
- TranscriptionProvider (using Mistral Voxtral API)
- SpeakerDetector (using Mistral chat API)
- SummarizationProvider (using Mistral chat API)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key advantage: Mistral is the only cloud provider (besides OpenAI) that supports
ALL three capabilities, making it a complete OpenAI alternative.

Note: Uses mistralai Python SDK (not OpenAI SDK).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
from mistralai import Mistral
except ImportError:
    Mistral = None  # type: ignore

from ... import config, models
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]


class MistralProvider:
    """Unified Mistral provider implementing TranscriptionProvider, SpeakerDetector, and SummarizationProvider.

    This provider initializes and manages:
    - Mistral Voxtral API for transcription
    - Mistral chat API for speaker detection
    - Mistral chat API for summarization

    All capabilities share the same Mistral client, similar to how OpenAI providers
    share the same OpenAI client.

    Key advantage: Mistral is a complete OpenAI alternative (all three capabilities).
    """

    def __init__(self, cfg: config.Config):
        """Initialize unified Mistral provider.

        Args:
            cfg: Configuration object with settings for all capabilities

        Raises:
            ValueError: If Mistral API key is not provided
            ImportError: If mistralai package is not installed
        """
        if Mistral is None:
            raise ImportError(
                "mistralai package required for Mistral provider. "
                "Install with: pip install 'podcast-scraper[mistral]'"
            )

        if not cfg.mistral_api_key:
            raise ValueError(
                "Mistral API key required for Mistral provider. "
                "Set MISTRAL_API_KEY environment variable or mistral_api_key in config."
            )

        self.cfg = cfg

        # Support custom base_url for E2E testing with mock servers
        client_kwargs: dict[str, Any] = {"api_key": cfg.mistral_api_key}
        if cfg.mistral_api_base:
            client_kwargs["base_url"] = cfg.mistral_api_base
        self.client = Mistral(**client_kwargs)

        # Transcription settings
        self.transcription_model = getattr(
            cfg, "mistral_transcription_model", "voxtral-mini-latest"
        )

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "mistral_speaker_model", "mistral-small-latest")
        self.speaker_temperature = getattr(cfg, "mistral_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "mistral_summary_model", "mistral-small-latest")
        self.summary_temperature = getattr(cfg, "mistral_temperature", 0.3)
        # Mistral Large supports 256k context window
        self.max_context_tokens = 256000  # Conservative estimate

        # Initialization state
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        self._requires_separate_instances = False

    def initialize(self) -> None:
        """Initialize all Mistral capabilities.

        For Mistral API, initialization is a no-op but we track it for consistency.
        This method is idempotent and can be called multiple times safely.
        """
        # Initialize transcription if enabled
        if self.cfg.transcription_provider == "mistral" and not self._transcription_initialized:
            self._initialize_transcription()

        # Initialize speaker detection if enabled
        if self.cfg.auto_speakers and not self._speaker_detection_initialized:
            self._initialize_speaker_detection()

        # Initialize summarization if enabled
        if self.cfg.generate_summaries and not self._summarization_initialized:
            self._initialize_summarization()

    def _initialize_transcription(self) -> None:
        """Initialize transcription capability."""
        logger.debug(
            "Initializing Mistral transcription (model: %s)", self.transcription_model
        )
        self._transcription_initialized = True

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing Mistral speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Mistral summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True

    # ============================================================================
    # TranscriptionProvider Protocol Implementation
    # ============================================================================

    def transcribe(
        self, audio_path: Path, language: str | None = None
    ) -> str:
        """Transcribe audio file using Mistral Voxtral API.

        Args:
            audio_path: Path to audio file
            language: Optional language code (hint for transcription)

        Returns:
            Transcribed text

        Raises:
            ValueError: If transcription fails
            RuntimeError: If provider is not initialized
        """
        if not self._transcription_initialized:
            raise RuntimeError(
                "MistralProvider transcription not initialized. Call initialize() first."
            )

        logger.debug("Transcribing audio via Mistral Voxtral API: %s", audio_path)

        try:
            # Mistral Voxtral API uses similar format to OpenAI Whisper
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.transcription_model,
                    file=audio_file,
                    language=language,
                )

            text = transcription.text if hasattr(transcription, "text") else ""
            if not text:
                logger.warning("Mistral Voxtral API returned empty transcription")
                return ""

            logger.debug("Mistral transcription completed: %d characters", len(text))
            return text

        except Exception as exc:
            logger.error("Mistral API error in transcription: %s", exc)
            raise ValueError(f"Mistral transcription failed: {exc}") from exc

    def transcribe_with_segments(
        self, audio_path: Path, language: str | None = None
    ) -> tuple[str, list[dict[str, object]]]:
        """Transcribe audio file with timestamp segments using Mistral Voxtral API.

        Args:
            audio_path: Path to audio file
            language: Optional language code (hint for transcription)

        Returns:
            Tuple of (transcribed text, list of segment dictionaries with start/end/text)

        Raises:
            ValueError: If transcription fails
            RuntimeError: If provider is not initialized
        """
        if not self._transcription_initialized:
            raise RuntimeError(
                "MistralProvider transcription not initialized. Call initialize() first."
            )

        logger.debug("Transcribing audio with segments via Mistral Voxtral API: %s", audio_path)

        try:
            with open(audio_path, "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model=self.transcription_model,
                    file=audio_file,
                    language=language,
                    response_format="verbose_json",  # Request segments
                    timestamp_granularities=["segment"],
                )

            text = transcription.text if hasattr(transcription, "text") else ""
            segments = []
            if hasattr(transcription, "segments"):
                segments = [
                    {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", ""),
                    }
                    for seg in transcription.segments
                ]

            logger.debug(
                "Mistral transcription with segments completed: %d characters, %d segments",
                len(text),
                len(segments),
            )
            return text, segments

        except Exception as exc:
            logger.error("Mistral API error in transcription with segments: %s", exc)
            raise ValueError(f"Mistral transcription failed: {exc}") from exc

    # ============================================================================
    # SpeakerDetector Protocol Implementation
    # ============================================================================

    def detect_hosts(
        self,
        feed_title: str | None,
        feed_description: str | None,
        feed_authors: list[str] | None = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using Mistral API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "MistralProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Mistral API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using Mistral API.

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
                "MistralProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Mistral API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                self.cfg.mistral_speaker_system_prompt or "mistral/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call Mistral API (similar to OpenAI format)
            response = self.client.chat.complete(
                model=self.speaker_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.speaker_temperature,
                max_tokens=300,
            )

            response_text = response.choices[0].message.content
            if not response_text:
                logger.warning("Mistral API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Mistral speaker detection completed: %d speakers, %d hosts, success=%s",
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
            logger.error("Failed to parse Mistral API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Mistral API error in speaker detection: %s", exc)
            raise ValueError(f"Mistral speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For Mistral provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.mistral_speaker_user_prompt
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
        """Parse speaker names from Mistral API response."""
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
        """Summarize text using Mistral chat API.

        Can handle full transcripts directly due to large context window (256k tokens).
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
                "MistralProvider summarization not initialized. Call initialize() first."
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
            "Summarizing text via Mistral API (model: %s, max_tokens: %d)",
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

            # Call Mistral API (similar to OpenAI format)
            response = self.client.chat.complete(
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
                logger.warning("Mistral API returned empty summary")
                summary = ""

            logger.debug("Mistral summarization completed: %d characters", len(summary))

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
                "summary_short": None,  # Mistral provider doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "mistral",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("Mistral API error in summarization: %s", exc)
            raise ValueError(f"Mistral summarization failed: {exc}") from exc

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
            self.cfg.mistral_summary_system_prompt or "mistral/summarization/system_v1"
        )
        user_prompt_name = self.cfg.mistral_summary_user_prompt

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
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

    def clear_cache(self) -> None:
        """Clear cache (no-op for API provider)."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component)."""
        return (
            self._transcription_initialized
            or self._speaker_detection_initialized
            or self._summarization_initialized
        )
```

### 5. Factory Updates

Update all three factories to support both Config-based and experiment-based modes (like OpenAI):

**File**: `src/podcast_scraper/transcription/factory.py`

```python
def create_transcription_provider(
    cfg_or_provider_type: Union[config.Config, str],
    params: Optional[Union[TranscriptionParams, Dict[str, Any]]] = None,
) -> TranscriptionProvider:
    # ... existing code ...

    elif provider_type == "mistral":
        from ..providers.mistral.mistral_provider import MistralProvider

        if experiment_mode:
            from ..config import Config
            assert isinstance(params, TranscriptionParams)
            cfg = Config(
                rss="",
                transcription_provider="mistral",
                mistral_transcription_model=params.model_name if params.model_name else "voxtral-mini-latest",
                mistral_api_key=os.getenv("MISTRAL_API_KEY"),
            )
            return MistralProvider(cfg)
        else:
            return MistralProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider type: {provider_type}. "
            "Supported types: 'whisper', 'openai', 'mistral'"
        )
```

Similar updates for `speaker_detectors/factory.py` and `summarization/factory.py`.

### 6. Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
mistral = [
    "mistralai>=1.0.0,<2.0.0",
]
```

### 7. Prompt Templates

Create Mistral-specific prompts in `src/podcast_scraper/prompts/mistral/`:

- `ner/system_ner_v1.j2` - System prompt for speaker detection
- `ner/guest_host_v1.j2` - User prompt for speaker detection
- `summarization/system_v1.j2` - System prompt for summarization
- `summarization/long_v1.j2` - User prompt for summarization

Follow OpenAI prompt patterns but optimize for Mistral models.

## Testing Strategy

Same pattern as OpenAI provider:

1. **Unit tests**: Mock Mistral API responses
2. **Integration tests**: Use E2E mock server with Mistral endpoints
3. **E2E tests**: Full workflow with Mistral provider

## Success Criteria

1. ✅ Mistral supports transcription, speaker detection, and summarization via unified provider
2. ✅ Mistral is a complete OpenAI alternative (all three capabilities)
3. ✅ Free tier works for development (Small model)
4. ✅ E2E tests pass
5. ✅ Experiment mode supported from start
6. ✅ Environment-based model defaults (test vs prod)
7. ✅ Follows OpenAI provider pattern exactly

## Migration Notes

- **Breaking Changes**: None (new provider, backward compatible)
- **Configuration**: Add `MISTRAL_API_KEY` to `.env` file
- **Dependencies**: Install with `pip install 'podcast-scraper[mistral]'`

## References

- **Related PRD**: `docs/prd/PRD-010-mistral-provider-integration.md`
- **Reference Implementation**: `src/podcast_scraper/providers/openai/openai_provider.py`
- **Mistral API Documentation**: <https://docs.mistral.ai/>
- **Mistral Python SDK**: <https://github.com/mistralai/mistral-python>
- **Voxtral Documentation**: <https://docs.mistral.ai/capabilities/audio_transcription>
