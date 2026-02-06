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

# Import Gemini SDK (verify package name during implementation)
try:
    import google.generativeai as genai
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
GEMINI_1_5_FLASH_INPUT_COST_PER_1M_TOKENS = 0.075
GEMINI_1_5_FLASH_OUTPUT_COST_PER_1M_TOKENS = 0.30


class GeminiProvider:
    """Unified Gemini provider implementing TranscriptionProvider, SpeakerDetector, and
    SummarizationProvider.

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
            ImportError: If google-generativeai package is not installed
        """
        if genai is None:
            raise ImportError(
                "google-generativeai package required for Gemini provider. "
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
        root_logger = logging.getLogger()
        root_level = root_logger.level if root_logger.level else logging.INFO
        if root_level <= logging.DEBUG:
            gemini_loggers = [
                "google.generativeai",
                "google.api_core",
            ]
            for logger_name in gemini_loggers:
                gemini_logger = logging.getLogger(logger_name)
                gemini_logger.setLevel(logging.WARNING)

        # Configure Gemini client
        genai.configure(api_key=cfg.gemini_api_key)

        # Support custom base_url for E2E testing with mock servers
        # Note: Gemini SDK may not support custom base_url directly
        # This would need to be handled via environment variable or SDK configuration
        # For now, we'll document this limitation
        if cfg.gemini_api_base:
            logger.warning(
                "gemini_api_base is set but Gemini SDK may not support custom base URLs. "
                "This is primarily for E2E testing - verify SDK support."
            )

        # Transcription settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.transcription_model = getattr(cfg, "gemini_transcription_model", "gemini-2.0-flash")

        # Speaker detection settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.speaker_model = getattr(cfg, "gemini_speaker_model", "gemini-2.0-flash")
        self.speaker_temperature = getattr(cfg, "gemini_temperature", 0.3)

        # Summarization settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.summary_model = getattr(cfg, "gemini_summary_model", "gemini-2.0-flash")
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
            model: Model name (e.g., "gemini-1.5-pro", "gemini-2.0-flash")
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
            if "2.0-flash" in model.lower():
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

            # Use Gemini's native multimodal API
            # Create model instance
            model = genai.GenerativeModel(self.transcription_model)

            # Build prompt with language hint if provided
            prompt_text = "Transcribe this audio file to text."
            if effective_language:
                prompt_text += f" The language is {effective_language}."

            # Create content with audio (multimodal input)
            # Gemini SDK supports file upload and inline data
            response = model.generate_content(
                [
                    {
                        "mime_type": mime_type,
                        "data": audio_data,
                    },
                    prompt_text,
                ]
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
                    error_msg,
                    exc,
                    exc_info=True,
                )
                # Check if exception has additional attributes (some SDKs provide rate limit info)
                if hasattr(exc, "status_code"):
                    logger.error("  HTTP status code: %s", exc.status_code)
                if hasattr(exc, "response"):
                    logger.error("  Response object: %s", exc.response)
                if hasattr(exc, "retry_after"):
                    logger.error("  Retry after: %s seconds", exc.retry_after)
            else:
                logger.error("Gemini API error in transcription: %s", exc)

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
                    message=f"Gemini authentication failed: {exc}",
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
                    message=f"Gemini rate limit exceeded (429): {exc}",
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
                    message=f"Gemini invalid model: {exc}",
                    provider="GeminiProvider/Transcription",
                    suggestion="Check gemini_transcription_model configuration",
                ) from exc
            else:
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

        # Wrap transcribe call with retry tracking
        from google.api_core import exceptions as google_exceptions

        from ...utils.provider_metrics import retry_with_metrics

        start_time = time.time()
        try:
            text = retry_with_metrics(
                lambda: self.transcribe(audio_path, language),
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=(
                    google_exceptions.ResourceExhausted,
                    google_exceptions.ServiceUnavailable,
                    ConnectionError,
                ),
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

            system_prompt_name = self.cfg.gemini_speaker_system_prompt or "gemini/ner/system_ner_v1"
            system_prompt = render_prompt(system_prompt_name)

            # Call Gemini API
            # Gemini uses system_instruction parameter instead of system message
            model = genai.GenerativeModel(
                model_name=self.speaker_model,
                system_instruction=system_prompt,
            )

            # Request JSON response
            # Use dict format for generation config (more compatible with SDK versions)
            generation_config = {
                "temperature": self.speaker_temperature,
                "max_output_tokens": 300,
                "response_mime_type": "application/json",
            }

            # Type ignore: Gemini SDK accepts dict for generation_config
            response = model.generate_content(
                user_prompt,
                generation_config=generation_config,  # type: ignore[arg-type]
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

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Gemini API error in speaker detection: %s", exc)
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Gemini-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Gemini authentication failed: {exc}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Check your GEMINI_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini rate limit exceeded: {exc}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini invalid model: {exc}",
                    provider="GeminiProvider/SpeakerDetection",
                    suggestion="Check gemini_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Gemini speaker detection failed: {exc}",
                    provider="GeminiProvider/SpeakerDetection",
                ) from exc

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

            # Wrap API call with retry tracking
            from google.api_core import exceptions as google_exceptions

            from ...utils.provider_metrics import retry_with_metrics

            def _make_api_call():
                # Call Gemini API
                # Gemini uses system_instruction parameter instead of system message
                model = genai.GenerativeModel(
                    model_name=self.summary_model,
                    system_instruction=system_prompt,
                )

                # Use dict format for generation config (more compatible with SDK versions)
                generation_config = {
                    "temperature": self.summary_temperature,
                    "max_output_tokens": max_length,
                }

                # Type ignore: Gemini SDK accepts dict for generation_config
                return model.generate_content(
                    user_prompt,
                    generation_config=generation_config,  # type: ignore[arg-type]
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=(
                        google_exceptions.ResourceExhausted,
                        google_exceptions.ServiceUnavailable,
                        ConnectionError,
                    ),
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
            logger.error("Gemini API error in summarization: %s", exc)
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Gemini-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Gemini authentication failed: {exc}",
                    provider="GeminiProvider/Summarization",
                    suggestion="Check your GEMINI_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini rate limit exceeded: {exc}",
                    provider="GeminiProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Gemini invalid model: {exc}",
                    provider="GeminiProvider/Summarization",
                    suggestion="Check gemini_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Gemini summarization failed: {exc}",
                    provider="GeminiProvider/Summarization",
                ) from exc

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
