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
import warnings
from typing import Any, cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

# Suppress Pydantic ArbitraryTypeWarning from google.genai (uses built-in "any" not typing.Any)
try:
    from pydantic.warnings import ArbitraryTypeWarning

    warnings.filterwarnings(
        "ignore",
        message=r".*built-in function any.*is not a Python type",
        category=ArbitraryTypeWarning,
    )
except ImportError:
    pass

# Import Gemini SDK (migrated from google.generativeai to google.genai in Issue #415)
try:
    import google.genai as genai
    from google.genai import types as genai_types
except ImportError:
    genai = None  # type: ignore
    genai_types = None  # type: ignore

from ... import config

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ...cleaning import PatternBasedCleaner
from ...cleaning.base import TranscriptCleaningProcessor
from ...utils.cleaning_max_tokens import (
    clamp_cleaning_max_tokens,
    estimate_cleaning_output_tokens,
    GEMINI_CLEANING_MAX_OUTPUT_TOKENS,
)
from ...utils.log_redaction import format_exception_for_log, redact_for_log
from ...workflow import metrics
from ..capabilities import ProviderCapabilities

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES

# Gemini API pricing constants (for cost estimation)
# Source: https://ai.google.dev/pricing
# Last updated: 2026-02
# Note: Prices subject to change. Always verify current rates
GEMINI_AUDIO_COST_PER_SECOND = 0.00025  # ~$0.90 per hour
GEMINI_2_FLASH_INPUT_COST_PER_1M_TOKENS = 0.10
GEMINI_2_FLASH_OUTPUT_COST_PER_1M_TOKENS = 0.40
GEMINI_1_5_PRO_INPUT_COST_PER_1M_TOKENS = 1.25
GEMINI_1_5_PRO_OUTPUT_COST_PER_1M_TOKENS = 5.00
GEMINI_1_5_FLASH_INPUT_COST_PER_1M_TOKENS = 0.075
GEMINI_1_5_FLASH_OUTPUT_COST_PER_1M_TOKENS = 0.30


def _should_disable_thinking_for_model(model: str) -> bool:
    """True for Gemini 2.5 Flash (non-lite): default thinking consumes max_output_tokens."""
    m = (model or "").lower()
    if "flash-lite" in m:
        return False
    return "2.5-flash" in m


def _merge_generate_content_config(
    model: str, base: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge generation config; set thinking_budget=0 for 2.5-flash unless thinking_config set."""
    merged: Dict[str, Any] = dict(base or ())
    # Explicit None is treated as absent so callers do not block injection by mistake.
    if merged.get("thinking_config") is None and "thinking_config" in merged:
        del merged["thinking_config"]
    if "thinking_config" in merged:
        return merged
    if _should_disable_thinking_for_model(model):
        merged["thinking_config"] = {"thinking_budget": 0}
    return merged


class GeminiProvider:
    """Unified Gemini provider: TranscriptionProvider, SpeakerDetector, SummarizationProvider.

    Uses Gemini native audio for transcription and chat models for speaker detection
    and summarization. All capabilities share the same Gemini client.
    """

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

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
                'Install with: pip install -e ".[llm]" (or pip install "podcast-scraper[llm]")'
            )

        if not cfg.gemini_api_key:
            raise ValueError(
                "Gemini API key required for Gemini provider. "
                "Set GEMINI_API_KEY environment variable or gemini_api_key in config."
            )

        # Validate API key format
        from ...utils.provider_metadata import validate_api_key_format

        is_valid, _ = validate_api_key_format(
            cfg.gemini_api_key,
            "Gemini",
            expected_prefixes=None,  # Gemini keys don't have standard prefix
        )
        if not is_valid:
            # Do not log validation detail: CodeQL taints any message from this API-key path.
            logger.warning(
                "Gemini API key validation failed (missing or too short); "
                "credentials are never logged."
            )

        self.cfg = cfg
        self.api_key = cfg.gemini_api_key

        # Set up transcript cleaning processor based on strategy (Issue #418)
        from ...cleaning import HybridCleaner, LLMBasedCleaner

        cleaning_strategy = getattr(cfg, "transcript_cleaning_strategy", "hybrid")
        if cleaning_strategy == "pattern":
            self.cleaning_processor = PatternBasedCleaner()  # type: ignore[assignment]
        elif cleaning_strategy == "llm":
            self.cleaning_processor = LLMBasedCleaner()  # type: ignore[assignment]
        else:  # hybrid (default)
            self.cleaning_processor = HybridCleaner()  # type: ignore[assignment]

        # Cleaning model settings (cheaper model for cost efficiency)
        # Note: gemini-1.5-flash not available in new API; default flash-tier model in config
        self.cleaning_model = getattr(cfg, "gemini_cleaning_model", "gemini-2.5-flash-lite")
        self.cleaning_temperature = getattr(cfg, "gemini_cleaning_temperature", 0.2)

        # Suppress verbose Gemini SDK debug logs (if needed)
        # Similar to OpenAI provider pattern
        root_logger = logging.getLogger()
        root_level = root_logger.level if root_logger.level else logging.INFO
        if root_level <= logging.DEBUG:
            gemini_loggers = [
                "google.genai",
                "google.api_core",
            ]
            for logger_name in gemini_loggers:
                gemini_logger = logging.getLogger(logger_name)
                gemini_logger.setLevel(logging.WARNING)

        # Create Gemini client using new API (google-genai 1.x; Issue #415 / #572)
        # New API uses Client instead of configure() + GenerativeModel
        client_kwargs: Dict[str, Any] = {"api_key": cfg.gemini_api_key}
        gemini_api_base = getattr(cfg, "gemini_api_base", None)
        if gemini_api_base:
            # E2E testing: route requests to mock server via http_options.base_url
            if genai_types is not None:
                client_kwargs["http_options"] = genai_types.HttpOptions(
                    base_url=gemini_api_base.rstrip("/")
                )
            else:
                logger.warning(
                    "gemini_api_base is set but google.genai.types not available; "
                    "requests will use default API."
                )
        self.client = genai.Client(**client_kwargs)

        # Log non-sensitive provider metadata (for debugging)
        from ...utils.provider_metadata import log_provider_metadata

        # Gemini may have project/region in config (if supported)
        project = getattr(cfg, "gemini_project", None)
        region = getattr(cfg, "gemini_region", None)
        log_provider_metadata(
            provider_name="Gemini",
            project=project,
            region=region,
        )

        # Note: HTTP timeout configuration
        # Gemini SDK uses global configure() and may not support per-client timeout configuration.
        # Timeout behavior is controlled by the SDK's default settings.
        # If timeout configuration is needed, it may require SDK version update or workaround.

        # Transcription settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.transcription_model = getattr(
            cfg, "gemini_transcription_model", "gemini-2.5-flash-lite"
        )

        # Speaker detection settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.speaker_model = getattr(cfg, "gemini_speaker_model", "gemini-2.5-flash-lite")
        self.speaker_temperature = getattr(cfg, "gemini_temperature", 0.3)

        # Summarization settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.summary_model = getattr(cfg, "gemini_summary_model", "gemini-2.5-flash-lite")
        self.summary_temperature = getattr(cfg, "gemini_temperature", 0.3)
        # Gemini 1.5 Pro supports 2M context window
        self.max_context_tokens = 2000000  # Conservative estimate

        # Initialization state
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        # API providers handle rate limiting and retries internally via SDK
        # Gemini SDK automatically handles retries with exponential backoff
        self._requires_separate_instances = False

    @staticmethod
    def get_pricing(model: str, capability: str) -> Dict[str, float]:
        """Get pricing information for a specific model and capability.

        Args:
            model: Model name (e.g., "gemini-1.5-pro", "gemini-2.5-flash-lite")
            capability: Capability type ("transcription", "speaker_detection", "summarization")

        Returns:
            Dictionary with pricing information
        """
        pricing: Dict[str, float] = {}

        if capability == "transcription":
            # Audio pricing is per second
            pricing["cost_per_second"] = GEMINI_AUDIO_COST_PER_SECOND
            pricing["cost_per_hour"] = GEMINI_AUDIO_COST_PER_SECOND * 3600
        else:
            # Text-based pricing (speaker detection, summarization)
            if "2.0-flash" in model.lower() or "flash-lite" in model.lower():
                pricing["input_cost_per_1m_tokens"] = GEMINI_2_FLASH_INPUT_COST_PER_1M_TOKENS
                pricing["output_cost_per_1m_tokens"] = GEMINI_2_FLASH_OUTPUT_COST_PER_1M_TOKENS
            elif "1.5-pro" in model.lower():
                pricing["input_cost_per_1m_tokens"] = GEMINI_1_5_PRO_INPUT_COST_PER_1M_TOKENS
                pricing["output_cost_per_1m_tokens"] = GEMINI_1_5_PRO_OUTPUT_COST_PER_1M_TOKENS
            elif "1.5-flash" in model.lower():
                pricing["input_cost_per_1m_tokens"] = GEMINI_1_5_FLASH_INPUT_COST_PER_1M_TOKENS
                pricing["output_cost_per_1m_tokens"] = GEMINI_1_5_FLASH_OUTPUT_COST_PER_1M_TOKENS
            else:
                # Default to 2.0-flash pricing
                pricing["input_cost_per_1m_tokens"] = GEMINI_2_FLASH_INPUT_COST_PER_1M_TOKENS
                pricing["output_cost_per_1m_tokens"] = GEMINI_2_FLASH_OUTPUT_COST_PER_1M_TOKENS

        return pricing

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
        logger.debug("Gemini transcription initialized successfully")

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing Gemini speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True
        logger.debug("Gemini speaker detection initialized successfully")

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Gemini summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True
        logger.debug("Gemini summarization initialized successfully")

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

            # Use Gemini's native multimodal API with new Client API
            # Build prompt with language hint if provided
            prompt_text = "Transcribe this audio file to text."
            if effective_language:
                prompt_text += f" The language is {effective_language}."

            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                retry_with_metrics,
            )

            contents = [
                {
                    "mime_type": mime_type,
                    "data": audio_data,
                },
                prompt_text,
            ]
            response = retry_with_metrics(
                lambda: self.client.models.generate_content(
                    model=self.transcription_model,
                    contents=cast(Any, contents),
                    config=cast(
                        Any,
                        _merge_generate_content_config(self.transcription_model, {}),
                    ),
                ),
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_gemini_retryable(),
            )

            # Extract text from response
            text = response.text if hasattr(response, "text") else str(response)
            if not text:
                logger.warning("Gemini returned empty transcription")
                text = ""

            logger.debug("Gemini transcription completed: %d characters", len(text))
            return text

        except Exception as exc:
            # Enhanced error logging for diagnostics
            error_type = type(exc).__name__
            error_msg = str(exc)
            error_msg_lower = error_msg.lower()

            # Log full error details for 429/rate limit errors
            if (
                "429" in error_msg
                or "quota" in error_msg_lower
                or "rate limit" in error_msg_lower
                or "resource exhausted" in error_msg_lower
            ):
                logger.error(
                    "Gemini API 429/rate limit error in transcription:\n"
                    "  Error type: %s\n"
                    "  Error message: %s\n"
                    "  Full exception: %s",
                    error_type,
                    redact_for_log(error_msg),
                    format_exception_for_log(exc),
                    exc_info=True,
                )
                # Check if exception has additional attributes (some SDKs provide rate limit info)
                if hasattr(exc, "status_code"):
                    logger.error("  HTTP status code: %s", exc.status_code)
                if hasattr(exc, "response"):
                    # Do not log exc.response body: may echo API key or other secrets.
                    resp = getattr(exc, "response", None)
                    if resp is not None:
                        r_status = getattr(resp, "status_code", None)
                        logger.error(
                            "  Response attachment: type=%s, status_code=%s",
                            type(resp).__name__,
                            r_status,
                        )
                if hasattr(exc, "retry_after"):
                    logger.error("  Retry after: %s seconds", exc.retry_after)
            else:
                logger.error("Gemini API error in transcription: %s", format_exception_for_log(exc))

            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Gemini-specific error types
            if (
                "api key" in error_msg_lower
                or "authentication" in error_msg_lower
                or "permission" in error_msg_lower
            ):
                raise ProviderAuthError(
                    message=f"Gemini authentication failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Transcription",
                    suggestion="Check your GEMINI_API_KEY environment variable or config setting",
                ) from exc
            elif (
                "429" in error_msg
                or "quota" in error_msg_lower
                or "rate limit" in error_msg_lower
                or "resource exhausted" in error_msg_lower
            ):
                raise ProviderRuntimeError(
                    message=f"Gemini rate limit exceeded (429): {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Transcription",
                    suggestion=(
                        "Gemini API rate limit exceeded. This usually means:\n"
                        "1. Too many requests per minute - reduce parallelism or add delays\n"
                        "2. Daily quota exceeded - check your API quota at "
                        "https://aistudio.google.com/app/apikey\n"
                        "3. Account limits - free tier has lower limits than paid\n"
                        "Wait a few minutes and retry, or check your API quota/limits"
                    ),
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini invalid model: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Transcription",
                    suggestion="Check gemini_transcription_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Gemini transcription failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Transcription",
                ) from exc

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: metrics.Metrics | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,  # ProviderCallMetrics from utils.provider_metrics
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

        Gemini doesn't provide native segments, so we return text with empty segments.

        Args:
            audio_path: Path to audio file
            language: Optional language code
            pipeline_metrics: Optional metrics tracker
            episode_duration_seconds: Optional episode duration
            call_metrics: Optional per-episode metrics tracker

        Returns:
            Tuple of (result_dict, elapsed_time) where result_dict contains:
            - "text": Full transcribed text
            - "segments": Empty list (Gemini doesn't provide segments)
        """
        # Track retries and rate limits
        from ...utils.provider_metrics import ProviderCallMetrics

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("gemini")

        from ...utils.provider_metrics import (
            _safe_gemini_retryable,
            retry_with_metrics,
        )

        start_time = time.time()
        try:
            text = retry_with_metrics(
                lambda: self.transcribe(audio_path, language),
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_gemini_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise

        elapsed = time.time() - start_time
        call_metrics.finalize()

        # Track LLM call metrics if available (aggregate)
        if pipeline_metrics is not None and episode_duration_seconds is not None:
            audio_minutes = episode_duration_seconds / 60.0
            # Note: Gemini doesn't provide token usage for audio, so we track by duration
            # This is an approximation - actual pricing is per second of audio
            pipeline_metrics.record_llm_transcription_call(audio_minutes)

        # Calculate cost for transcription (per minute pricing)
        if episode_duration_seconds is not None:
            audio_minutes = episode_duration_seconds / 60.0
            from ...workflow.helpers import calculate_provider_cost

            cost = calculate_provider_cost(
                cfg=self.cfg,
                provider_type="gemini",
                capability="transcription",
                model=self.transcription_model,
                audio_minutes=audio_minutes,
            )
            call_metrics.set_cost(cost)

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
            speakers, detected_hosts, _, _ = self.detect_speakers(
                episode_title=feed_title,
                episode_description=feed_description,
                known_hosts=set(),
            )
            return detected_hosts
        except Exception as exc:
            logger.warning(
                "Failed to detect hosts from feed metadata: %s", format_exception_for_log(exc)
            )
            return set()

    def detect_speakers(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
        pipeline_metrics: metrics.Metrics | None = None,
    ) -> Tuple[list[str], Set[str], bool, bool]:
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
            - used_defaults: True if default names were returned (e.g. on failure)

        Raises:
            ValueError: If detection fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        # If auto_speakers is disabled, return defaults without requiring initialization
        if not self.cfg.auto_speakers:
            logger.debug("Auto-speakers disabled, detection failed")
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

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

            system_prompt_name = self.cfg.gemini_speaker_system_prompt or "gemini/ner/system_ner_v1"
            system_prompt = render_prompt(system_prompt_name)

            # Call Gemini API using new Client API
            # Request JSON response
            # Use dict format for generation config (more compatible with SDK versions)
            # Note: system_instruction is part of config in new API
            generation_config = _merge_generate_content_config(
                self.speaker_model,
                {
                    "temperature": self.speaker_temperature,
                    "max_output_tokens": 300,
                    "response_mime_type": "application/json",
                    "system_instruction": system_prompt,
                },
            )

            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                retry_with_metrics,
            )

            response = retry_with_metrics(
                lambda: self.client.models.generate_content(
                    model=self.speaker_model,
                    contents=user_prompt,
                    config=cast(Any, generation_config),
                ),
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_gemini_retryable(),
            )

            response_text = response.text if hasattr(response, "text") else str(response)
            if not response_text:
                logger.warning("Gemini API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

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
            # Note: Check Gemini SDK for usage information structure
            if pipeline_metrics is not None and hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                # Safely extract token counts, handling Mock objects in tests
                input_tokens = getattr(usage, "prompt_token_count", 0)
                output_tokens = getattr(usage, "candidates_token_count", 0)
                # Convert to int (handles Mock objects in tests)
                try:
                    input_tokens = int(input_tokens) if input_tokens is not None else 0
                except (TypeError, ValueError):
                    input_tokens = 0
                try:
                    output_tokens = int(output_tokens) if output_tokens is not None else 0
                except (TypeError, ValueError):
                    output_tokens = 0
                pipeline_metrics.record_llm_speaker_detection_call(input_tokens, output_tokens)

            return speakers, detected_hosts, success, False

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse Gemini API JSON response: %s", format_exception_for_log(exc)
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True
        except Exception as exc:
            logger.error("Gemini API error in speaker detection: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Gemini-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Gemini authentication failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Check your GEMINI_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini invalid model: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Check gemini_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Gemini speaker detection failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/SpeakerDetection",
                ) from exc

    def analyze_patterns(
        self,
        episodes: list[Episode],  # type: ignore[valid-type]
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
        template_params = {
            "episode_title": episode_title,
            "episode_description": episode_description or "",
            "known_hosts": ", ".join(sorted(known_hosts)) if known_hosts else "",
        }
        template_params.update(self.cfg.ner_prompt_params)
        user_prompt = render_prompt(user_prompt_name, **template_params)
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
            if response_text.strip().startswith("{"):
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False
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
        call_metrics: Any | None = None,  # ProviderCallMetrics from utils.provider_metrics
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

            # Track retries and rate limits
            from ...utils.provider_metrics import ProviderCallMetrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("gemini")

            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                generation_config = _merge_generate_content_config(
                    self.summary_model,
                    {
                        "temperature": self.summary_temperature,
                        "max_output_tokens": max_length,
                        "system_instruction": system_prompt,
                    },
                )

                return self.client.models.generate_content(
                    model=self.summary_model,
                    contents=user_prompt,
                    config=cast(Any, generation_config),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_gemini_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            summary = response.text if hasattr(response, "text") else str(response)
            if not summary:
                logger.warning("Gemini API returned empty summary")
                summary = ""

            logger.debug("Gemini summarization completed: %d characters", len(summary))

            # Extract token counts and populate call_metrics
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = response.usage_metadata
                # Safely extract token counts, handling Mock objects in tests
                input_tokens_raw = getattr(usage, "prompt_token_count", 0)
                output_tokens_raw = getattr(usage, "candidates_token_count", 0)
                # Convert to int (handles Mock objects in tests)
                try:
                    input_tokens = int(input_tokens_raw) if input_tokens_raw is not None else None
                except (TypeError, ValueError):
                    input_tokens = None
                try:
                    output_tokens = (
                        int(output_tokens_raw) if output_tokens_raw is not None else None
                    )
                except (TypeError, ValueError):
                    output_tokens = None

                if input_tokens is not None and output_tokens is not None:
                    call_metrics.set_tokens(input_tokens, output_tokens)

            # Track LLM call metrics if available (aggregate tracking)
            if (
                pipeline_metrics is not None
                and input_tokens is not None
                and output_tokens is not None
            ):
                pipeline_metrics.record_llm_summarization_call(input_tokens, output_tokens)

            # Calculate cost
            if input_tokens is not None:
                from ...workflow.helpers import calculate_provider_cost

                cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="gemini",
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )
                call_metrics.set_cost(cost)

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
                "summary_short": None,  # Gemini doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "gemini",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("Gemini API error in summarization: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Gemini-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Gemini authentication failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Summarization",
                    suggestion="Check your GEMINI_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini invalid model: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Summarization",
                    suggestion="Check gemini_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Gemini summarization failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Summarization",
                ) from exc

    def summarize_bundled(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: metrics.Metrics | None = None,
        call_metrics: Any | None = None,
    ) -> Dict[str, Any]:
        """One completion: semantic transcript clean + JSON title/bullets (Issue #477)."""
        if not self._summarization_initialized:
            raise RuntimeError(
                "GeminiProvider summarization not initialized. Call initialize() first."
            )

        from ...prompts.store import get_prompt_metadata, render_prompt
        from ...utils.provider_metrics import (
            _safe_gemini_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        max_out = int(getattr(self.cfg, "llm_bundled_max_output_tokens", 16384) or 16384)
        tmpl_kwargs = dict(self.cfg.summary_prompt_params or {})
        system_prompt = render_prompt(
            "gemini/summarization/bundled_clean_summary_system_v1",
            **tmpl_kwargs,
        )
        user_prompt = render_prompt(
            "gemini/summarization/bundled_clean_summary_user_v1",
            transcript=text,
            title=episode_title or "",
            **tmpl_kwargs,
        )

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("gemini")

        def _make_api_call() -> Any:
            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": self.summary_temperature,
                    "max_output_tokens": max_out,
                    "response_mime_type": "application/json",
                    "system_instruction": system_prompt,
                },
            )
            return self.client.models.generate_content(
                model=self.summary_model,
                contents=user_prompt,
                config=cast(Any, generation_config),
            )

        try:
            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_gemini_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise

        call_metrics.finalize()

        raw = (response.text if hasattr(response, "text") else str(response) or "").strip()
        if not raw:
            raise ValueError("Gemini bundled call returned empty content")

        try:
            data = json.loads(raw, strict=False)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Bundled response is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Bundled JSON must be an object")
        summary_prose = data.get("summary")
        bullets = data.get("bullets")
        if not isinstance(summary_prose, str) or not summary_prose.strip():
            raise ValueError("Bundled JSON missing non-empty summary string")
        if not isinstance(bullets, list) or not bullets:
            raise ValueError("Bundled JSON missing non-empty bullets list")

        input_tokens = None
        output_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens_raw = getattr(usage, "prompt_token_count", 0)
            output_tokens_raw = getattr(usage, "candidates_token_count", 0)
            try:
                input_tokens = int(input_tokens_raw) if input_tokens_raw is not None else None
            except (TypeError, ValueError):
                input_tokens = None
            try:
                output_tokens = int(output_tokens_raw) if output_tokens_raw is not None else None
            except (TypeError, ValueError):
                output_tokens = None
            if input_tokens is not None and output_tokens is not None:
                call_metrics.set_tokens(input_tokens, output_tokens)

        if pipeline_metrics is not None and input_tokens is not None and output_tokens is not None:
            pipeline_metrics.record_llm_bundled_clean_summary_call(input_tokens, output_tokens)

        if input_tokens is not None:
            from ...workflow.helpers import calculate_provider_cost

            cost = calculate_provider_cost(
                cfg=self.cfg,
                provider_type="gemini",
                capability="summarization",
                model=self.summary_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            call_metrics.set_cost(cost)

        prompt_metadata = {
            "system": get_prompt_metadata(
                "gemini/summarization/bundled_clean_summary_system_v1",
                params=tmpl_kwargs,
            ),
            "user": get_prompt_metadata(
                "gemini/summarization/bundled_clean_summary_user_v1",
                params={
                    **tmpl_kwargs,
                    "transcript": text[:100] + "..." if len(text) > 100 else text,
                },
            ),
        }

        return {
            "summary": raw,
            "summary_short": None,
            "metadata": {
                "model": self.summary_model,
                "provider": "gemini",
                "bundled": True,
                "max_output_tokens": max_out,
                "prompts": prompt_metadata,
            },
        }

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

    def generate_insights(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_insights: int = 5,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> List[str]:
        """Generate a list of short insight statements from transcript (GIL).

        Uses gemini/insight_extraction/v1 prompt; parses response as one insight per line.
        Returns empty list on failure so GIL can fall back to stub.
        """
        if not self._summarization_initialized:
            logger.warning("Gemini summarization not initialized for generate_insights")
            return []

        from ...prompts.store import render_prompt

        max_insights = min(max(1, max_insights), 10)
        text_slice = (text or "").strip()
        if len(text_slice) > 120000:
            text_slice = text_slice[:120000] + "\n\n[Transcript truncated.]"

        try:
            user_prompt = render_prompt(
                "gemini/insight_extraction/v1",
                transcript=text_slice,
                title=episode_title or "",
                max_insights=max_insights,
            )
            system_prompt = (
                "Output only the list of key takeaways, one per line. "
                "No numbering, bullets, or extra text."
            )
            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": 0.3,
                    "max_output_tokens": min(1024, max_insights * 150),
                    "system_instruction": system_prompt,
                },
            )
            response = self.client.models.generate_content(
                model=self.summary_model,
                contents=user_prompt,
                config=cast(Any, generation_config),
            )
            content = response.text if hasattr(response, "text") else str(response)
            content = (content or "").strip()
            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            cleaned = []
            for line in lines:
                s = line.strip()
                if not s:
                    continue
                if len(s) >= 2 and s[0].isdigit() and s[1] in ".)":
                    s = s[2:].strip()
                if s.startswith("- ") or s.startswith("* "):
                    s = s[2:].strip()
                if s:
                    cleaned.append(s)
            return cleaned[:max_insights]
        except Exception as e:
            logger.debug("Gemini generate_insights failed: %s", e, exc_info=True)
            return []

    def extract_kg_graph(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_topics: int = 5,
        max_entities: int = 15,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract topics and entities as JSON (KG layer). Returns None on failure."""
        if not self._summarization_initialized:
            logger.warning("Gemini summarization not initialized for extract_kg_graph")
            return None
        from ...kg.llm_extract import (
            build_kg_transcript_system_prompt,
            build_kg_user_prompt,
            parse_kg_graph_response,
            resolve_kg_model_id,
            truncate_transcript_for_kg,
        )

        max_topics = min(max(1, max_topics), 20)
        max_entities = min(max(1, max_entities), 50)
        text_slice = truncate_transcript_for_kg(text or "")
        if not text_slice.strip():
            return None
        model = resolve_kg_model_id(self, params)
        user_prompt = build_kg_user_prompt(
            text_slice, episode_title or "", max_topics, max_entities
        )
        system_msg = build_kg_transcript_system_prompt(max_topics, max_entities)
        try:
            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                retry_with_metrics,
            )

            generation_config = _merge_generate_content_config(
                model,
                {
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                    "system_instruction": system_msg,
                },
            )

            def _make_api_call():
                return self.client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=cast(Any, generation_config),
                )

            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_gemini_retryable(),
            )
            raw = response.text if hasattr(response, "text") else str(response)
            return parse_kg_graph_response(
                (raw or "").strip(),
                max_topics=max_topics,
                max_entities=max_entities,
            )
        except Exception as e:
            logger.debug("Gemini extract_kg_graph failed: %s", e, exc_info=True)
            return None

    def extract_kg_from_summary_bullets(
        self,
        bullet_labels: List[str],
        episode_title: Optional[str] = None,
        max_topics: int = 5,
        max_entities: int = 15,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Derive KG topics/entities from summary bullets (no transcript)."""
        if not self._summarization_initialized:
            logger.warning(
                "Gemini summarization not initialized for extract_kg_from_summary_bullets"
            )
            return None
        from ...kg.llm_extract import (
            build_kg_from_bullets_system_prompt,
            build_kg_from_bullets_user_prompt,
            normalize_bullet_labels_for_kg,
            parse_kg_graph_response,
            resolve_kg_model_id,
        )

        max_topics = min(max(1, max_topics), 20)
        max_entities = min(max(1, max_entities), 50)
        bullets = normalize_bullet_labels_for_kg(bullet_labels)
        if not bullets:
            return None
        model = resolve_kg_model_id(self, params)
        user_prompt = build_kg_from_bullets_user_prompt(
            bullets, episode_title or "", max_topics, max_entities
        )
        system_msg = build_kg_from_bullets_system_prompt(max_topics, max_entities)
        try:
            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                retry_with_metrics,
            )

            generation_config = _merge_generate_content_config(
                model,
                {
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                    "system_instruction": system_msg,
                },
            )

            def _make_api_call():
                return self.client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=cast(Any, generation_config),
                )

            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_gemini_retryable(),
            )
            raw = response.text if hasattr(response, "text") else str(response)
            return parse_kg_graph_response(
                (raw or "").strip(),
                max_topics=max_topics,
                max_entities=max_entities,
            )
        except Exception as e:
            logger.debug("Gemini extract_kg_from_summary_bullets failed: %s", e, exc_info=True)
            return None

    def extract_quotes(
        self,
        transcript: str,
        insight_text: str,
        **kwargs: Any,
    ) -> List[Any]:
        """Extract candidate quote span that supports the insight (GIL QA via LLM)."""
        if not self._summarization_initialized or not (transcript and insight_text):
            return []
        import json

        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span

        system = (
            "Extract all short verbatim quotes from the transcript that "
            "support the given insight. CRITICAL: each quote must be a "
            "DIFFERENT passage — never repeat the same text. Find evidence "
            "from separate parts of the transcript. "
            "Reply with ONLY a JSON object: "
            '{"quotes": ["quote from early in transcript", '
            '"quote from middle", "quote from end"]}'
        )
        user = (
            f"Transcript (excerpt):\n{transcript.strip()[:50000]}\n\n"
            f"Insight: {insight_text.strip()}\n\n"
            "Return JSON with quote_text only."
        )
        try:
            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                apply_gil_evidence_llm_call_metrics,
                gemini_generate_usage_tokens,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("gemini")
            pm = kwargs.get("pipeline_metrics")

            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": 0.0,
                    "max_output_tokens": 512,
                    "system_instruction": system,
                },
            )

            def _make_api_call():
                return self.client.models.generate_content(
                    model=self.summary_model,
                    contents=user,
                    config=cast(Any, generation_config),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_gemini_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = gemini_generate_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(call_metrics, pm, in_tok, out_tok)
            content = response.text if hasattr(response, "text") else str(response)
            content = (content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            obj = json.loads(content)
            quotes_raw = obj.get("quotes") or []
            if isinstance(quotes_raw, str):
                quotes_raw = [quotes_raw]
            # Backward compat: fall back to single quote_text
            if not quotes_raw:
                qt = (obj.get("quote_text") or "").strip()
                if qt:
                    quotes_raw = [qt]
            results_q: list = []
            for qt_str in quotes_raw:
                qt_clean = str(qt_str).strip()
                if not qt_clean:
                    continue
                resolved = resolve_llm_quote_span(transcript, qt_clean)
                if resolved is None:
                    continue
                r_start, r_end, r_verbatim = resolved
                results_q.append(
                    QuoteCandidate(
                        char_start=r_start,
                        char_end=r_end,
                        text=r_verbatim,
                        qa_score=1.0,
                    )
                )
            # Deduplicate: LLMs sometimes return the same quote multiple times
            seen_texts: set = set()
            deduped: list = []
            for q in results_q:
                if q.text not in seen_texts:
                    seen_texts.add(q.text)
                    deduped.append(q)
            results_q = deduped
            return results_q
        except Exception as e:
            logger.debug("Gemini extract_quotes failed: %s", e, exc_info=True)
            return []

    def score_entailment(
        self,
        premise: str,
        hypothesis: str,
        **kwargs: Any,
    ) -> float:
        """Score entailment of hypothesis given premise (GIL NLI via LLM). 0–1."""
        if not self._summarization_initialized or not (premise and hypothesis):
            return 0.0
        system = (
            "You rate how much the premise supports the hypothesis. "
            "Reply with ONLY a number between 0 and 1 (0=not at all, 1=fully supports)."
        )
        user = f"Premise: {premise.strip()}\n\nHypothesis: {hypothesis.strip()}"
        try:
            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                apply_gil_evidence_llm_call_metrics,
                gemini_generate_usage_tokens,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("gemini")
            pm = kwargs.get("pipeline_metrics")

            generation_config = _merge_generate_content_config(
                self.summary_model,
                {
                    "temperature": 0.0,
                    "max_output_tokens": 10,
                    "system_instruction": system,
                },
            )

            def _make_api_call():
                return self.client.models.generate_content(
                    model=self.summary_model,
                    contents=user,
                    config=cast(Any, generation_config),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_gemini_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = gemini_generate_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(call_metrics, pm, in_tok, out_tok)
            content = response.text if hasattr(response, "text") else str(response)
            content = (content or "0").strip()
            for part in content.replace(",", " ").split():
                try:
                    v = float(part)
                    return max(0.0, min(1.0, v))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            logger.debug("Gemini score_entailment failed: %s", e, exc_info=True)
            return 0.0

    def clean_transcript(self, text: str, pipeline_metrics: Optional[Any] = None) -> str:
        """Clean transcript using LLM for semantic filtering.

        Args:
            text: Transcript text to clean (should already be pattern-cleaned)

        Returns:
            Cleaned transcript text

        Raises:
            RuntimeError: If provider is not initialized or cleaning fails
        """
        if not self._summarization_initialized:
            raise RuntimeError("GeminiProvider not initialized. Call initialize() first.")

        from ...prompts.store import render_prompt

        # Build cleaning prompt using prompt_store (RFC-017)
        prompt_name = "gemini/cleaning/v1"
        user_prompt = render_prompt(prompt_name, transcript=text)

        # Use system instruction (Gemini pattern)
        system_prompt = (
            "You are a transcript cleaning assistant. "
            "Remove sponsors, ads, intros, outros, and meta-commentary. "
            "Preserve all substantive content and speaker information. "
            "Return only the cleaned text, no explanations."
        )

        logger.debug(
            "Cleaning transcript via Gemini API (model: %s, text length: %d chars)",
            self.cleaning_model,
            len(text),
        )

        try:
            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("gemini")

            def _make_api_call():
                generation_config = _merge_generate_content_config(
                    self.cleaning_model,
                    {
                        "temperature": self.cleaning_temperature,
                        "max_output_tokens": clamp_cleaning_max_tokens(
                            estimate_cleaning_output_tokens(len(text.split())),
                            GEMINI_CLEANING_MAX_OUTPUT_TOKENS,
                        ),
                        "system_instruction": system_prompt,
                    },
                )

                return self.client.models.generate_content(
                    model=self.cleaning_model,
                    contents=user_prompt,
                    config=cast(Any, generation_config),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_gemini_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            cleaned = response.text if hasattr(response, "text") else str(response)
            if not cleaned:
                logger.warning("Gemini API returned empty cleaned text, using original")
                return text

            logger.debug("Gemini cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except Exception as exc:
            logger.error("Gemini API error in cleaning: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle Gemini-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Gemini authentication failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Cleaning",
                    suggestion="Check your GEMINI_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Cleaning",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Gemini cleaning failed: {format_exception_for_log(exc)}",
                    provider="GeminiProvider/Cleaning",
                ) from exc

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities object describing Gemini provider capabilities
        """
        return ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_semantic_cleaning=True,  # Gemini supports LLM-based cleaning
            supports_audio_input=True,  # Gemini supports multimodal audio
            supports_json_mode=True,  # Gemini supports JSON mode via response_schema
            max_context_tokens=self.max_context_tokens,
            supports_tool_calls=True,  # Gemini supports function calling
            supports_system_prompt=True,  # Gemini supports system prompts
            supports_streaming=True,  # Gemini API supports streaming
            provider_name="gemini",
            supports_gi_segment_timing=False,
        )
