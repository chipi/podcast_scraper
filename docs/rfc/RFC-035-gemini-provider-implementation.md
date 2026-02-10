# RFC-035: Google Gemini Provider Implementation (Revised)

- **Status**: ✅ Completed (v2.5.0)
- **Revision**: 2
- **Date**: 2026-02-04
- **Authors**:
- **Stakeholders**: Maintainers, users wanting Google Gemini integration
- **Related PRDs**:
  - `docs/prd/PRD-012-gemini-provider-integration.md`
- **Related RFCs**:
  - `docs/rfc/RFC-013-openai-provider-implementation.md` (reference - unified provider pattern)
  - `docs/rfc/RFC-033-mistral-provider-implementation.md` (similar - full capabilities)

## Abstract

Design and implement Google Gemini as a unified provider for transcription, speaker detection, and summarization capabilities. Gemini is unique in offering native multimodal audio understanding (no separate ASR step) and an industry-leading 2 million token context window. This RFC follows the **unified provider pattern** established by OpenAI (RFC-013), where a single provider class implements all three protocols.

**Architecture Alignment:** Gemini provider follows the exact same unified provider pattern as `OpenAIProvider`, implementing all three protocols (`TranscriptionProvider`, `SpeakerDetector`, `SummarizationProvider`) in a single class and integrating via the existing factory pattern with support for both Config-based and experiment-based modes.

## Problem Statement

Users want Google Gemini as a provider option for:

1. **Transcription**: Native audio understanding via multimodal input
2. **Speaker Detection**: Entity extraction using Gemini chat models
3. **Summarization**: High-quality summaries with massive context window

Key advantages of Gemini:

- **Native audio understanding** - no separate transcription step
- **2M token context** - process entire seasons without chunking
- **Generous free tier** - excellent for development
- **Google ecosystem** - familiar for Google Cloud users

## Constraints & Assumptions

**Constraints:**

- Must use Google Generative AI Python SDK (`google-genai`, migrated from `google-generativeai` in Issue #415)
- Audio files must be uploaded or passed as inline data (support both if available)
- Rate limits apply (especially free tier)
- Must follow unified provider pattern (like OpenAI)

**Assumptions:**

- Gemini API is stable
- Audio transcription quality is comparable to Whisper
- Prompts may need Gemini-specific optimization
- Package name is `google-genai` (migrated from `google-generativeai` in Issue #415)

## Design & Implementation

### 0. Gemini API Overview

| Feature | OpenAI | Gemini |
| ------- | ------ | ------ |
| **SDK** | `openai` | `google-genai` (migrated from `google-generativeai` in Issue #415) |
| **Audio** | Whisper (separate API) | Native multimodal |
| **Context Window** | 128k tokens | 2M tokens |
| **Audio Input** | File upload | File or inline data (both) |
| **Provider Pattern** | Unified (`OpenAIProvider`) | Unified (`GeminiProvider`) |

### 1. Architecture Overview

**Unified Provider Pattern** (following OpenAI):

```text
src/podcast_scraper/
├── providers/
│   └── gemini/                        # NEW: Unified Gemini provider
│       ├── __init__.py
│       └── gemini_provider.py         # Single class implementing all 3 protocols
├── prompts/
│   └── gemini/                        # NEW: Gemini-specific prompts
│       ├── ner/
│       │   ├── system_ner_v1.j2
│       │   └── guest_host_v1.j2
│       └── summarization/
│           ├── system_v1.j2
│           └── long_v1.j2
├── transcription/
│   └── factory.py                     # Updated: Add "gemini" option
├── speaker_detectors/
│   └── factory.py                     # Updated: Add "gemini" option
├── summarization/
│   └── factory.py                     # Updated: Add "gemini" option
└── config.py                          # Updated: Add Gemini fields
```

**Key Architectural Decision:** Use unified provider pattern (single `GeminiProvider` class) matching `OpenAIProvider`, not separate files per capability.

### 2. Configuration

Add to `config.py` following OpenAI pattern exactly:

```python
from typing import Literal, Optional

# Provider Selection (updated)
transcription_provider: Literal["whisper", "openai", "gemini"] = Field(
    default="whisper",
    description="Transcription provider"
)

speaker_detector_provider: Literal["spacy", "openai", "gemini"] = Field(
    default="spacy",
    description="Speaker detection provider"
)

summary_provider: Literal["transformers", "openai", "gemini"] = Field(
    default="transformers",
    description="Summarization provider"
)

# Gemini API Configuration (following OpenAI pattern)
gemini_api_key: Optional[str] = Field(
    default=None,
    alias="gemini_api_key",
    description="Google AI API key (prefer GEMINI_API_KEY env var or .env file)"
)

gemini_api_base: Optional[str] = Field(
    default=None,
    alias="gemini_api_base",
    description="Gemini API base URL (for E2E testing with mock servers)"
)

# Gemini Model Selection (environment-based defaults, like OpenAI)
gemini_transcription_model: str = Field(
    default_factory=_get_default_gemini_transcription_model,
    alias="gemini_transcription_model",
    description="Gemini model for transcription (default: environment-based)"
)

gemini_speaker_model: str = Field(
    default_factory=_get_default_gemini_speaker_model,
    alias="gemini_speaker_model",
    description="Gemini model for speaker detection (default: environment-based)"
)

gemini_summary_model: str = Field(
    default_factory=_get_default_gemini_summary_model,
    alias="gemini_summary_model",
    description="Gemini model for summarization (default: environment-based)"
)

# Shared settings (like OpenAI)
gemini_temperature: float = Field(
    default=0.3,
    alias="gemini_temperature",
    description="Temperature for Gemini generation (0.0-2.0, lower = more deterministic)"
)

gemini_max_tokens: Optional[int] = Field(
    default=None,
    alias="gemini_max_tokens",
    description="Max tokens for Gemini generation (None = model default)"
)

# Gemini Prompt Configuration (following OpenAI pattern)
gemini_summary_system_prompt: Optional[str] = Field(
    default=None,
    alias="gemini_summary_system_prompt",
    description="Gemini system prompt for summarization (default: gemini/summarization/system_v1)"
)

gemini_summary_user_prompt: str = Field(
    default="gemini/summarization/long_v1",
    alias="gemini_summary_user_prompt",
    description="Gemini user prompt for summarization"
)

gemini_speaker_system_prompt: Optional[str] = Field(
    default=None,
    alias="gemini_speaker_system_prompt",
    description="Gemini system prompt for speaker detection (default: gemini/ner/system_ner_v1)"
)

gemini_speaker_user_prompt: str = Field(
    default="gemini/ner/guest_host_v1",
    alias="gemini_speaker_user_prompt",
    description="Gemini user prompt for speaker detection"
)
```

**Environment-based defaults** (like OpenAI):

```python
# In config_constants.py
TEST_DEFAULT_GEMINI_TRANSCRIPTION_MODEL = "gemini-2.0-flash"
PROD_DEFAULT_GEMINI_TRANSCRIPTION_MODEL = "gemini-1.5-pro"

TEST_DEFAULT_GEMINI_SPEAKER_MODEL = "gemini-2.0-flash"
PROD_DEFAULT_GEMINI_SPEAKER_MODEL = "gemini-1.5-pro"

TEST_DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-2.0-flash"
PROD_DEFAULT_GEMINI_SUMMARY_MODEL = "gemini-1.5-pro"

# In config.py
def _get_default_gemini_transcription_model() -> str:
    """Get default Gemini transcription model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_GEMINI_TRANSCRIPTION_MODEL
    return PROD_DEFAULT_GEMINI_TRANSCRIPTION_MODEL

def _get_default_gemini_speaker_model() -> str:
    """Get default Gemini speaker detection model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_GEMINI_SPEAKER_MODEL
    return PROD_DEFAULT_GEMINI_SPEAKER_MODEL

def _get_default_gemini_summary_model() -> str:
    """Get default Gemini summarization model based on environment."""
    if _is_test_environment():
        return TEST_DEFAULT_GEMINI_SUMMARY_MODEL
    return PROD_DEFAULT_GEMINI_SUMMARY_MODEL
```

### 3. API Key Management

Follow OpenAI pattern exactly:

```python
# In config.py

@field_validator("gemini_api_key", mode="before")
@classmethod
def _load_gemini_api_key_from_env(cls, value: Any) -> Optional[str]:
    """Load Gemini API key from environment variable if not provided."""
    if value is not None:
        return value
    # Check environment variable
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key
    return None

@field_validator("gemini_api_base", mode="before")
@classmethod
def _load_gemini_api_base_from_env(cls, value: Any) -> Optional[str]:
    """Load Gemini API base URL from environment variable if not provided."""
    if value is not None:
        return value
    env_base = os.getenv("GEMINI_API_BASE")
    if env_base:
        return env_base
    return None

@model_validator(mode="after")
def _validate_gemini_provider_requirements(self) -> "Config":
    """Validate that Gemini API key is provided when Gemini providers are selected."""
    gemini_providers_used = []
    if self.transcription_provider == "gemini":
        gemini_providers_used.append("transcription")
    if self.speaker_detector_provider == "gemini":
        gemini_providers_used.append("speaker_detection")
    if self.summary_provider == "gemini":
        gemini_providers_used.append("summarization")

    if gemini_providers_used and not self.gemini_api_key:
        providers_str = ", ".join(gemini_providers_used)
        raise ValueError(
            f"Gemini API key required for Gemini providers: {providers_str}. "
            "Set GEMINI_API_KEY environment variable or gemini_api_key in config."
        )

    return self
```

### 4. Unified Provider Implementation

**File**: `src/podcast_scraper/providers/gemini/gemini_provider.py`

Follow `OpenAIProvider` pattern exactly:

```python
"""Unified Gemini provider for transcription, speaker detection, and summarization.

This module provides a single GeminiProvider class that implements all three protocols:
- TranscriptionProvider (using native multimodal audio understanding)
- SpeakerDetector (using Gemini chat models)
- SummarizationProvider (using Gemini chat models)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Set, Tuple

# Import Gemini SDK (migrated from google.generativeai to google.genai in Issue #415)
try:
    import google.genai as genai
except ImportError:
    genai = None  # type: ignore

from ... import config, models
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

# Gemini API pricing constants (for cost estimation)
# Source: https://ai.google.dev/pricing
# Last updated: 2026-02
# Note: Prices subject to change. Always verify current rates
GEMINI_AUDIO_COST_PER_SECOND = 0.00025  # ~$0.90 per hour
GEMINI_2_FLASH_INPUT_COST_PER_1M_TOKENS = 0.10
GEMINI_2_FLASH_OUTPUT_COST_PER_1M_TOKENS = 0.40
GEMINI_1_5_PRO_INPUT_COST_PER_1M_TOKENS = 1.25
GEMINI_1_5_PRO_OUTPUT_COST_PER_1M_TOKENS = 5.00


class GeminiProvider:
    """Unified Gemini provider implementing TranscriptionProvider, SpeakerDetector, and SummarizationProvider.

    This provider initializes and manages:
    - Gemini native multimodal audio understanding for transcription
    - Gemini chat models for speaker detection
    - Gemini chat models for summarization

    All three capabilities share the same Gemini client, similar to how OpenAI providers
    share the same OpenAI client. The client is initialized once and reused.
    """

    def __init__(self, cfg: config.Config):
        """Initialize unified Gemini provider.

        Args:
            cfg: Configuration object with settings for all three capabilities

        Raises:
            ValueError: If Gemini API key is not provided
            ImportError: If google-genai package is not installed
        """
        if genai is None:
            raise ImportError(
                "google-genai package required for Gemini provider. "
                "Install with: pip install 'podcast-scraper[gemini]'"
            )

        if not cfg.gemini_api_key:
            raise ValueError(
                "Gemini API key required for Gemini provider. "
                "Set GEMINI_API_KEY environment variable or gemini_api_key in config."
            )

        self.cfg = cfg

        # Suppress verbose Gemini SDK debug logs (if needed)
        # Similar to OpenAI provider pattern

        # Configure Gemini client
        genai.configure(api_key=cfg.gemini_api_key)

        # Support custom base_url for E2E testing with mock servers
        # (Check Gemini SDK documentation for how to set custom base URL)

        # Transcription settings
        self.transcription_model = getattr(cfg, "gemini_transcription_model", "gemini-2.0-flash")

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "gemini_speaker_model", "gemini-2.0-flash")
        self.speaker_temperature = getattr(cfg, "gemini_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "gemini_summary_model", "gemini-2.0-flash")
        self.summary_temperature = getattr(cfg, "gemini_temperature", 0.3)
        # Gemini 1.5 Pro supports 2M context window
        self.max_context_tokens = 2000000  # Conservative estimate

        # Initialization state
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        self._requires_separate_instances = False

    @staticmethod
    def get_pricing(model: str, capability: str) -> Dict[str, float]:
        """Get pricing information for a specific model and capability.

        Args:
            model: Model name
            capability: Capability type ("transcription", "speaker_detection", "summarization")

        Returns:
            Dictionary with pricing information
        """
        # Implementation similar to OpenAIProvider.get_pricing()
        # ...

    def initialize(self) -> None:
        """Initialize all Gemini capabilities.

        For Gemini API, initialization is a no-op but we track it for consistency.
        This method is idempotent and can be called multiple times safely.
        """
        # Initialize transcription if enabled
        if self.cfg.transcribe_missing and not self._transcription_initialized:
            self._initialize_transcription()

        # Initialize speaker detection if enabled
        if self.cfg.auto_speakers and not self._speaker_detection_initialized:
            self._initialize_speaker_detection()

        # Initialize summarization if enabled
        if self.cfg.generate_summaries and not self._summarization_initialized:
            self._initialize_summarization()

    def _initialize_transcription(self) -> None:
        """Initialize transcription capability."""
        logger.debug("Initializing Gemini transcription (model: %s)", self.transcription_model)
        self._transcription_initialized = True

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing Gemini speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Gemini summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True

    # ============================================================================
    # TranscriptionProvider Protocol Implementation
    # ============================================================================

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text using Gemini's native multimodal audio understanding.

        Args:
            audio_path: Path to audio file (str, not Path)
            language: Optional language code (e.g., "en", "fr").
                     If provided (not None), uses that language.
                     If not provided (default None), uses cfg.language if available.
                     If explicitly passed as None, auto-detects.

        Returns:
            Transcribed text as string

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        if not self._transcription_initialized:
            raise RuntimeError(
                "GeminiProvider transcription not initialized. Call initialize() first."
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided language or fall back to config (like OpenAI)
        if language is not None:
            effective_language = language
        elif hasattr(self.cfg, "language") and self.cfg.language is not None:
            effective_language = self.cfg.language
        else:
            effective_language = None

        logger.debug(
            "Transcribing audio file via Gemini API: %s (language: %s)",
            audio_path,
            effective_language or "auto",
        )

        try:
            # Load audio file
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()

            # Determine MIME type from file extension
            suffix = os.path.splitext(audio_path)[1].lower()
            mime_types = {
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
                ".m4a": "audio/mp4",
                ".ogg": "audio/ogg",
                ".flac": "audio/flac",
            }
            mime_type = mime_types.get(suffix, "audio/mpeg")

            # Use Gemini's native multimodal API
            # Support both file upload and inline data (check SDK docs)
            model = genai.GenerativeModel(self.transcription_model)

            # Create content with audio
            # (Exact API call depends on SDK - verify during implementation)
            response = model.generate_content(
                [
                    {"mime_type": mime_type, "data": audio_data},
                    "Transcribe this audio file to text.",
                ]
            )

            text = response.text if hasattr(response, "text") else str(response)
            if not text:
                logger.warning("Gemini returned empty transcription")
                text = ""

            logger.debug("Gemini transcription completed: %d characters", len(text))
            return text

        except Exception as exc:
            logger.error("Gemini API error in transcription: %s", exc)
            from podcast_scraper.exceptions import ProviderRuntimeError

            raise ProviderRuntimeError(
                message=f"Gemini transcription failed: {exc}",
                provider="GeminiProvider/Transcription",
            ) from exc

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: metrics.Metrics | None = None,
        episode_duration_seconds: int | None = None,
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

        Gemini doesn't provide native segments, so we return text with empty segments.

        Args:
            audio_path: Path to audio file
            language: Optional language code
            pipeline_metrics: Optional metrics tracker
            episode_duration_seconds: Optional episode duration

        Returns:
            Tuple of (result_dict, elapsed_time) where result_dict contains:
            - "text": Full transcribed text
            - "segments": Empty list (Gemini doesn't provide segments)
        """
        start_time = time.time()
        text = self.transcribe(audio_path, language)
        elapsed = time.time() - start_time

        return {"text": text, "segments": []}, elapsed

    # ============================================================================
    # SpeakerDetector Protocol Implementation
    # ============================================================================

    def detect_hosts(
        self,
        feed_title: str | None,
        feed_description: str | None,
        feed_authors: list[str] | None = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using Gemini API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "GeminiProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Gemini API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using Gemini API.

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
                "GeminiProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Gemini API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                self.cfg.gemini_speaker_system_prompt or "gemini/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call Gemini API
            model = genai.GenerativeModel(
                model_name=self.speaker_model,
                system_instruction=system_prompt,
            )

            response = model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": self.speaker_temperature,
                    "max_output_tokens": 300,
                },
            )

            response_text = response.text if hasattr(response, "text") else str(response)
            if not response_text:
                logger.warning("Gemini API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Gemini speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available
            # (Check Gemini SDK for usage information)

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Gemini API error in speaker detection: %s", exc)
            raise ValueError(f"Gemini speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For Gemini provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.gemini_speaker_user_prompt
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
        """Parse speaker names from Gemini API response."""
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
        """Summarize text using Gemini API.

        Can handle full transcripts directly due to massive context window (2M tokens).
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
                "GeminiProvider summarization not initialized. Call initialize() first."
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
            "Summarizing text via Gemini API (model: %s, max_tokens: %d)",
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

            # Call Gemini API
            model = genai.GenerativeModel(
                model_name=self.summary_model,
                system_instruction=system_prompt,
            )

            response = model.generate_content(
                user_prompt,
                generation_config={
                    "temperature": self.summary_temperature,
                    "max_output_tokens": max_length,
                },
            )

            summary = response.text if hasattr(response, "text") else str(response)
            if not summary:
                logger.warning("Gemini API returned empty summary")
                summary = ""

            logger.debug("Gemini summarization completed: %d characters", len(summary))

            # Track LLM call metrics if available
            # (Check Gemini SDK for usage information)

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
                "summary_short": None,  # Gemini provider doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "gemini",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("Gemini API error in summarization: %s", exc)
            raise ValueError(f"Gemini summarization failed: {exc}") from exc

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
            self.cfg.gemini_summary_system_prompt or "gemini/summarization/system_v1"
        )
        user_prompt_name = self.cfg.gemini_summary_user_prompt

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

    elif provider_type == "gemini":
        from ..providers.gemini.gemini_provider import GeminiProvider

        if experiment_mode:
            from ..config import Config
            assert isinstance(params, TranscriptionParams)
            cfg = Config(
                rss="",
                transcription_provider="gemini",
                gemini_transcription_model=params.model_name if params.model_name else "gemini-2.0-flash",
                gemini_api_key=os.getenv("GEMINI_API_KEY"),
            )
            return GeminiProvider(cfg)
        else:
            return GeminiProvider(cfg)
    else:
        raise ValueError(
            f"Unsupported transcription provider: {provider_type}. "
            "Supported providers: 'whisper', 'openai', 'gemini'"
        )
```

Similar updates for `speaker_detectors/factory.py` and `summarization/factory.py`.

### 6. Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
gemini = [
    "google-genai>=0.1.0,<1.0.0",  # Migrated from google-generativeai in Issue #415
]
```

**Note:** Package name is `google-genai` (migrated from `google-generativeai` in Issue #415).

### 7. Prompt Templates

Create Gemini-specific prompts in `src/podcast_scraper/prompts/gemini/`:

- `ner/system_ner_v1.j2` - System prompt for speaker detection
- `ner/guest_host_v1.j2` - User prompt for speaker detection
- `summarization/system_v1.j2` - System prompt for summarization
- `summarization/long_v1.j2` - User prompt for summarization

Follow OpenAI prompt patterns but optimize for Gemini models.

## Testing Strategy

Same pattern as OpenAI provider:

1. **Unit tests**: Mock Gemini API responses
2. **Integration tests**: Use E2E mock server with Gemini endpoints
3. **E2E tests**: Full workflow with Gemini provider

## Success Criteria

1. ✅ Gemini supports all three capabilities via unified provider
2. ✅ Native audio transcription works (file upload and inline data)
3. ✅ 2M context window is available for summarization
4. ✅ Free tier works for development
5. ✅ E2E tests pass
6. ✅ Experiment mode supported from start
7. ✅ Environment-based model defaults (test vs prod)
8. ✅ Follows OpenAI provider pattern exactly

## Migration Notes

- **Breaking Changes**: None (new provider, backward compatible)
- **Configuration**: Add `GEMINI_API_KEY` to `.env` file
- **Dependencies**: Install with `pip install 'podcast-scraper[gemini]'`

## References

- **Related PRD**: `docs/prd/PRD-012-gemini-provider-integration.md`
- **Reference Implementation**: `src/podcast_scraper/providers/openai/openai_provider.py`
- **Google AI Documentation**: <https://ai.google.dev/docs>
- **Gemini API Reference**: <https://ai.google.dev/api/rest>
