"""Unified Anthropic provider for transcription, speaker detection, and summarization.

This module provides a single AnthropicProvider class that implements all three protocols:
- TranscriptionProvider (Note: Anthropic doesn't support native audio, so this raises
  NotImplementedError)
- SpeakerDetector (using Claude chat models)
- SummarizationProvider (using Claude chat models)

This unified approach matches the pattern of OpenAI and Gemini providers, where a single
provider type handles multiple capabilities using shared API client.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

# Import Anthropic SDK
try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None  # type: ignore
    Anthropic = None  # type: ignore

from ... import config

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ...cleaning import PatternBasedCleaner
from ...cleaning.base import TranscriptCleaningProcessor
from ...utils.cleaning_max_tokens import (
    ANTHROPIC_CLEANING_MAX_TOKENS,
    clamp_cleaning_max_tokens,
    estimate_cleaning_output_tokens,
)
from ...utils.log_redaction import format_exception_for_log
from ...utils.timeout_config import get_http_timeout
from ...workflow import metrics
from ..capabilities import ProviderCapabilities

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES

# Anthropic API pricing constants (for cost estimation)
# Source: https://www.anthropic.com/pricing
# Last updated: 2026-02
# Note: Prices subject to change. Always verify current rates
ANTHROPIC_CLAUDE_3_5_SONNET_INPUT_COST_PER_1M_TOKENS = 3.00
ANTHROPIC_CLAUDE_3_5_SONNET_OUTPUT_COST_PER_1M_TOKENS = 15.00
ANTHROPIC_CLAUDE_3_OPUS_INPUT_COST_PER_1M_TOKENS = 15.00
ANTHROPIC_CLAUDE_3_OPUS_OUTPUT_COST_PER_1M_TOKENS = 75.00
ANTHROPIC_CLAUDE_3_HAIKU_INPUT_COST_PER_1M_TOKENS = 0.25
ANTHROPIC_CLAUDE_3_HAIKU_OUTPUT_COST_PER_1M_TOKENS = 1.25
ANTHROPIC_CLAUDE_3_5_HAIKU_INPUT_COST_PER_1M_TOKENS = 0.80
ANTHROPIC_CLAUDE_3_5_HAIKU_OUTPUT_COST_PER_1M_TOKENS = 4.00
# Claude Haiku 4.5 (alias e.g. claude-haiku-4-5) — see Anthropic pricing page
ANTHROPIC_CLAUDE_HAIKU_4_5_INPUT_COST_PER_1M_TOKENS = 1.00
ANTHROPIC_CLAUDE_HAIKU_4_5_OUTPUT_COST_PER_1M_TOKENS = 5.00


class AnthropicProvider:
    """Unified Anthropic provider: SpeakerDetector and SummarizationProvider (no transcription).

    Manages Claude chat models for speaker detection and summarization. All capabilities
    share the same Anthropic client. Transcription raises NotImplementedError.
    """

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

    def __init__(self, cfg: config.Config):
        """Initialize unified Anthropic provider.

        Args:
            cfg: Configuration object with settings for all three capabilities

        Raises:
            ValueError: If Anthropic API key is not provided
            ImportError: If anthropic package is not installed
        """
        if Anthropic is None:
            raise ImportError(
                "anthropic package required for Anthropic provider. "
                "Install with: pip install -e '.[llm]' (anthropic is in the llm extra)"
            )

        if not cfg.anthropic_api_key:
            raise ValueError(
                "Anthropic API key required for Anthropic provider. "
                "Set ANTHROPIC_API_KEY environment variable or anthropic_api_key in config."
            )

        # Validate API key format
        from ...utils.provider_metadata import validate_api_key_format

        is_valid, _ = validate_api_key_format(
            cfg.anthropic_api_key,
            "Anthropic",
            expected_prefixes=["sk-ant-"],
        )
        if not is_valid:
            # Do not log validation detail: CodeQL taints any message from this API-key path.
            logger.warning(
                "Anthropic API key validation failed (missing, too short, or wrong prefix); "
                "credentials are never logged."
            )

        self.cfg = cfg

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
        self.cleaning_model = getattr(cfg, "anthropic_cleaning_model", "claude-haiku-4-5")
        self.cleaning_temperature = getattr(cfg, "anthropic_cleaning_temperature", 0.2)

        # Suppress verbose Anthropic SDK debug logs (if needed)
        # Similar to OpenAI and Gemini provider pattern
        root_logger = logging.getLogger()
        root_level = root_logger.level if root_logger.level else logging.INFO
        if root_level <= logging.DEBUG:
            anthropic_loggers = [
                "anthropic",
                "anthropic._client",
            ]
            for logger_name in anthropic_loggers:
                anthropic_logger = logging.getLogger(logger_name)
                anthropic_logger.setLevel(logging.WARNING)

        # Log non-sensitive provider metadata (for debugging)
        from ...utils.provider_metadata import extract_region_from_endpoint, log_provider_metadata

        # Extract region from base_url if possible
        base_url = getattr(cfg, "anthropic_api_base", None)
        region = extract_region_from_endpoint(base_url)
        log_provider_metadata(
            provider_name="Anthropic",
            base_url=base_url,
            region=region,
        )

        # Configure Anthropic client
        # Support custom base_url for E2E testing with mock servers
        client_kwargs: Dict[str, Any] = {"api_key": cfg.anthropic_api_key}
        if cfg.anthropic_api_base:
            client_kwargs["base_url"] = cfg.anthropic_api_base

        # Configure HTTP timeouts with separate connect/read timeouts
        # Note: Anthropic SDK may support timeout parameter (verify SDK version)
        # If not supported, this will be ignored but won't break
        timeout_config = get_http_timeout(cfg)
        if timeout_config is not None:
            client_kwargs["timeout"] = timeout_config

        self.client = Anthropic(**client_kwargs)  # type: ignore[arg-type]

        # Transcription settings
        # Note: Anthropic doesn't support native audio transcription
        # This is a placeholder for future support or error handling
        self.transcription_model = getattr(
            cfg, "anthropic_transcription_model", "claude-3-5-sonnet-20241022"
        )

        # Speaker detection settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.speaker_model = getattr(cfg, "anthropic_speaker_model", "claude-3-5-sonnet-20241022")
        self.speaker_temperature = getattr(cfg, "anthropic_temperature", 0.3)

        # Summarization settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.summary_model = getattr(cfg, "anthropic_summary_model", "claude-3-5-sonnet-20241022")
        self.summary_temperature = getattr(cfg, "anthropic_temperature", 0.3)
        # Claude 3.5 Sonnet supports 200K context window
        self.max_context_tokens = 200000

        # Initialization state
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        # API providers handle rate limiting and retries internally via SDK
        # Anthropic SDK automatically handles retries with exponential backoff
        self._requires_separate_instances = False

    @staticmethod
    def get_pricing(model: str, capability: str) -> Dict[str, float]:
        """Get pricing information for a specific model and capability.

        Args:
            model: Model name (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229")
            capability: Capability type ("transcription", "speaker_detection", "summarization")

        Returns:
            Dictionary with pricing information
        """
        pricing: Dict[str, float] = {}

        if capability == "transcription":
            # Anthropic doesn't support native audio transcription
            # Return placeholder pricing (should not be used)
            pricing["cost_per_second"] = 0.0
            pricing["cost_per_hour"] = 0.0
        else:
            # Text-based pricing (speaker detection, summarization)
            # Model names use "3-5" (dash) but we check for both "3.5" and "3-5"
            model_lower = model.lower()
            if "3.5-sonnet" in model_lower or "3-5-sonnet" in model_lower:
                pricing["input_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_5_SONNET_INPUT_COST_PER_1M_TOKENS
                )
                pricing["output_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_5_SONNET_OUTPUT_COST_PER_1M_TOKENS
                )
            elif "haiku-4-5" in model_lower:
                pricing["input_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_HAIKU_4_5_INPUT_COST_PER_1M_TOKENS
                )
                pricing["output_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_HAIKU_4_5_OUTPUT_COST_PER_1M_TOKENS
                )
            elif "3.5-haiku" in model_lower or "3-5-haiku" in model_lower:
                pricing["input_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_5_HAIKU_INPUT_COST_PER_1M_TOKENS
                )
                pricing["output_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_5_HAIKU_OUTPUT_COST_PER_1M_TOKENS
                )
            elif "3-opus" in model.lower():
                pricing["input_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_OPUS_INPUT_COST_PER_1M_TOKENS
                )
                pricing["output_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_OPUS_OUTPUT_COST_PER_1M_TOKENS
                )
            elif "3-haiku" in model.lower():
                pricing["input_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_HAIKU_INPUT_COST_PER_1M_TOKENS
                )
                pricing["output_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_HAIKU_OUTPUT_COST_PER_1M_TOKENS
                )
            else:
                # Default to 3.5-sonnet pricing
                pricing["input_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_5_SONNET_INPUT_COST_PER_1M_TOKENS
                )
                pricing["output_cost_per_1m_tokens"] = (
                    ANTHROPIC_CLAUDE_3_5_SONNET_OUTPUT_COST_PER_1M_TOKENS
                )

        return pricing

    def initialize(self) -> None:
        """Initialize all Anthropic capabilities.

        For Anthropic API, initialization is a no-op but we track it for consistency.
        This method is idempotent and can be called multiple times safely.
        """
        # Initialize transcription if enabled
        # Note: Anthropic doesn't support native audio, so this is a no-op
        if self.cfg.transcribe_missing and not self._transcription_initialized:
            self._initialize_transcription()

        # Initialize speaker detection if enabled
        if self.cfg.auto_speakers and not self._speaker_detection_initialized:
            self._initialize_speaker_detection()

        # Initialize summarization if enabled
        if self.cfg.generate_summaries and not self._summarization_initialized:
            self._initialize_summarization()

    def _initialize_transcription(self) -> None:
        """Initialize transcription capability.

        Note: Anthropic doesn't support native audio transcription.
        This method exists for API compatibility but will raise NotImplementedError
        when transcribe() is called.
        """
        logger.debug(
            "Initializing Anthropic transcription (model: %s) - "
            "Note: Anthropic doesn't support native audio transcription",
            self.transcription_model,
        )
        self._transcription_initialized = True
        logger.debug("Anthropic transcription initialized (no-op)")

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing Anthropic speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True
        logger.debug("Anthropic speaker detection initialized successfully")

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Anthropic summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True
        logger.debug("Anthropic summarization initialized successfully")

    # ============================================================================
    # TranscriptionProvider Protocol Implementation
    # ============================================================================

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text.

        Note: Anthropic doesn't support native audio transcription.
        This method raises NotImplementedError.

        Args:
            audio_path: Path to audio file (str, not Path)
            language: Optional language code (ignored, kept for API compatibility)

        Returns:
            Transcribed text as string (never reached due to NotImplementedError)

        Raises:
            NotImplementedError: Anthropic doesn't support native audio transcription
        """
        if not self._transcription_initialized:
            raise RuntimeError(
                "AnthropicProvider transcription not initialized. Call initialize() first."
            )

        raise NotImplementedError(
            "Anthropic doesn't support native audio transcription. "
            "Use Whisper or another transcription provider, then use Anthropic for "
            "speaker detection and summarization."
        )

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: metrics.Metrics | None = None,
        episode_duration_seconds: int | None = None,
        call_metrics: Any | None = None,  # ProviderCallMetrics from utils.provider_metrics
    ) -> tuple[dict[str, object], float]:
        """Transcribe audio file and return full result with segments.

        Note: Anthropic doesn't support native audio transcription.
        This method raises NotImplementedError.

        Args:
            audio_path: Path to audio file
            language: Optional language code (ignored)
            pipeline_metrics: Optional metrics tracker (ignored)
            episode_duration_seconds: Optional episode duration (ignored)

        Returns:
            Tuple of (result_dict, elapsed_time) (never reached due to NotImplementedError)

        Raises:
            NotImplementedError: Anthropic doesn't support native audio transcription
        """
        start_time = time.time()
        text = self.transcribe(audio_path, language)
        elapsed = time.time() - start_time

        # Finalize call_metrics (Anthropic doesn't support audio transcription,
        # but finalize for consistency)
        if call_metrics is not None:
            call_metrics.finalize()

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

        # Prefer RSS author tags if available (like OpenAI and Gemini)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Anthropic API to detect hosts from feed metadata
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

            # Call Anthropic API with retry
            from ...utils.provider_metrics import (
                _safe_anthropic_retryable,
                retry_with_metrics,
            )

            response = retry_with_metrics(
                lambda: self.client.messages.create(
                    model=self.speaker_model,
                    max_tokens=300,
                    temperature=self.speaker_temperature,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                ),
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_anthropic_retryable(),
            )

            # Extract text from response - handle different block types
            response_text = ""
            if response.content and len(response.content) > 0:
                first_block = response.content[0]
                if hasattr(first_block, "text"):
                    response_text = first_block.text
                elif isinstance(first_block, str):
                    response_text = first_block
            if not response_text:
                logger.warning("Anthropic API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

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
            if pipeline_metrics is not None and hasattr(response, "usage"):
                usage = response.usage
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                pipeline_metrics.record_llm_speaker_detection_call(input_tokens, output_tokens)

            return speakers, detected_hosts, success, False

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse Anthropic API JSON response: %s", format_exception_for_log(exc)
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True
        except Exception as exc:
            logger.error(
                "Anthropic API error in speaker detection: %s", format_exception_for_log(exc)
            )
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Anthropic-specific error types
            error_msg = str(exc).lower()
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
            ):
                raise ProviderAuthError(
                    message=f"Anthropic authentication failed: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/SpeakerDetection",
                    suggestion=(
                        "Check your ANTHROPIC_API_KEY environment variable or config setting"
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic invalid model: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/SpeakerDetection",
                    suggestion="Check anthropic_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Anthropic speaker detection failed: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/SpeakerDetection",
                ) from exc

    def analyze_patterns(
        self,
        episodes: list[Episode],  # type: ignore[valid-type]
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
            # Only return defaults for JSON-like content that failed to parse
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
        """Summarize text using Anthropic API.

        Can handle full transcripts directly due to large context window (200K tokens).
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

            # Track retries and rate limits
            from ...utils.provider_metrics import (
                _safe_anthropic_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("anthropic")

            # Wrap API call with retry tracking
            def _make_api_call():
                return self.client.messages.create(
                    model=self.summary_model,
                    max_tokens=max_length,
                    temperature=self.summary_temperature,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_anthropic_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            # Extract text from response - handle different block types
            summary = ""
            if response.content and len(response.content) > 0:
                first_block = response.content[0]
                if hasattr(first_block, "text"):
                    summary = first_block.text
                elif isinstance(first_block, str):
                    summary = first_block
            if not summary:
                logger.warning("Anthropic API returned empty summary")
                summary = ""

            logger.debug("Anthropic summarization completed: %d characters", len(summary))

            # Extract token counts and populate call_metrics
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage") and response.usage:
                input_tokens_val = getattr(response.usage, "input_tokens", None)
                output_tokens_val = getattr(response.usage, "output_tokens", None)
                # Convert to int if they're actual numbers, otherwise use 0
                # Handle Mock objects from tests by checking type
                input_tokens = (
                    int(input_tokens_val) if isinstance(input_tokens_val, (int, float)) else 0
                )
                output_tokens = (
                    int(output_tokens_val) if isinstance(output_tokens_val, (int, float)) else 0
                )
                if input_tokens > 0 or output_tokens > 0:
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
                    provider_type="anthropic",
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
                "summary_short": None,  # Anthropic doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "anthropic",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("Anthropic API error in summarization: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Anthropic-specific error types
            error_msg = str(exc).lower()
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
            ):
                raise ProviderAuthError(
                    message=f"Anthropic authentication failed: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/Summarization",
                    suggestion=(
                        "Check your ANTHROPIC_API_KEY environment variable or config setting"
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic invalid model: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/Summarization",
                    suggestion="Check anthropic_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Anthropic summarization failed: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/Summarization",
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
                "AnthropicProvider summarization not initialized. Call initialize() first."
            )

        from ...prompts.store import get_prompt_metadata, render_prompt
        from ...utils.provider_metrics import (
            _safe_anthropic_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        max_out = int(getattr(self.cfg, "llm_bundled_max_output_tokens", 16384) or 16384)
        tmpl_kwargs = dict(self.cfg.summary_prompt_params or {})
        system_prompt = render_prompt(
            "anthropic/summarization/bundled_clean_summary_system_v1",
            **tmpl_kwargs,
        )
        user_prompt = render_prompt(
            "anthropic/summarization/bundled_clean_summary_user_v1",
            transcript=text,
            title=episode_title or "",
            **tmpl_kwargs,
        )

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("anthropic")

        def _make_api_call() -> Any:
            return self.client.messages.create(
                model=self.summary_model,
                max_tokens=max_out,
                temperature=self.summary_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

        try:
            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_anthropic_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise

        call_metrics.finalize()

        summary = ""
        if response.content and len(response.content) > 0:
            first_block = response.content[0]
            if hasattr(first_block, "text"):
                summary = first_block.text
            elif isinstance(first_block, str):
                summary = first_block
        raw = (summary or "").strip()
        if not raw:
            raise ValueError("Anthropic bundled call returned empty content")

        try:
            data = json.loads(raw)
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
        if hasattr(response, "usage") and response.usage:
            in_toks = getattr(response.usage, "input_tokens", None)
            out_toks = getattr(response.usage, "output_tokens", None)
            input_tokens = int(in_toks) if isinstance(in_toks, (int, float)) else 0
            output_tokens = int(out_toks) if isinstance(out_toks, (int, float)) else 0
            if input_tokens > 0 or output_tokens > 0:
                call_metrics.set_tokens(input_tokens, output_tokens)

        if pipeline_metrics is not None and input_tokens is not None and output_tokens is not None:
            pipeline_metrics.record_llm_bundled_clean_summary_call(input_tokens, output_tokens)

        if input_tokens is not None:
            from ...workflow.helpers import calculate_provider_cost

            cost = calculate_provider_cost(
                cfg=self.cfg,
                provider_type="anthropic",
                capability="summarization",
                model=self.summary_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            call_metrics.set_cost(cost)

        prompt_metadata = {
            "system": get_prompt_metadata(
                "anthropic/summarization/bundled_clean_summary_system_v1",
                params=tmpl_kwargs,
            ),
            "user": get_prompt_metadata(
                "anthropic/summarization/bundled_clean_summary_user_v1",
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
                "provider": "anthropic",
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
            raise RuntimeError("AnthropicProvider not initialized. Call initialize() first.")

        from ...prompts.store import render_prompt

        # Build cleaning prompt using prompt_store (RFC-017)
        prompt_name = "anthropic/cleaning/v1"
        user_prompt = render_prompt(prompt_name, transcript=text)

        # Use system prompt (Anthropic pattern)
        system_prompt = (
            "You are a transcript cleaning assistant. "
            "Remove sponsors, ads, intros, outros, and meta-commentary. "
            "Preserve all substantive content and speaker information. "
            "Return only the cleaned text, no explanations."
        )

        logger.debug(
            "Cleaning transcript via Anthropic API (model: %s, text length: %d chars)",
            self.cleaning_model,
            len(text),
        )

        try:
            # Track retries and rate limits
            from ...utils.provider_metrics import (
                _safe_anthropic_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("anthropic")

            # Wrap API call with retry tracking
            def _make_api_call():
                return self.client.messages.create(
                    model=self.cleaning_model,
                    max_tokens=clamp_cleaning_max_tokens(
                        estimate_cleaning_output_tokens(len(text.split())),
                        ANTHROPIC_CLEANING_MAX_TOKENS,
                    ),
                    temperature=self.cleaning_temperature,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": user_prompt,
                        }
                    ],
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_anthropic_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            # Extract text from response
            cleaned = ""
            if response.content and len(response.content) > 0:
                first_block = response.content[0]
                if hasattr(first_block, "text"):
                    cleaned = first_block.text
                elif isinstance(first_block, str):
                    cleaned = first_block

            if not cleaned:
                logger.warning("Anthropic API returned empty cleaned text, using original")
                return text

            logger.debug("Anthropic cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except Exception as exc:
            logger.error("Anthropic API error in cleaning: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle Anthropic-specific error types
            error_msg = str(exc).lower()
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
            ):
                raise ProviderAuthError(
                    message=f"Anthropic authentication failed: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/Cleaning",
                    suggestion="Check your ANTHROPIC_API_KEY environment variable or config setting",  # noqa: E501
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/Cleaning",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Anthropic cleaning failed: {format_exception_for_log(exc)}",
                    provider="AnthropicProvider/Cleaning",
                ) from exc

    def generate_insights(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_insights: int = 5,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> List[str]:
        """Generate a list of short insight statements from transcript (GIL).

        Uses anthropic/insight_extraction/v1 prompt; parses response as one insight per line.
        Returns empty list on failure so GIL can fall back to stub.
        """
        if not self._summarization_initialized:
            logger.warning("Anthropic summarization not initialized for generate_insights")
            return []

        from ...prompts.store import render_prompt

        max_insights = min(max(1, max_insights), 10)
        text_slice = (text or "").strip()
        if len(text_slice) > 120000:
            text_slice = text_slice[:120000] + "\n\n[Transcript truncated.]"

        try:
            user_prompt = render_prompt(
                "anthropic/insight_extraction/v1",
                transcript=text_slice,
                title=episode_title or "",
                max_insights=max_insights,
            )
            system_prompt = (
                "Output only the list of key takeaways, one per line. "
                "No numbering, bullets, or extra text."
            )
            response = self.client.messages.create(
                model=self.summary_model,
                max_tokens=min(1024, max_insights * 150),
                temperature=0.3,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = ""
            if response.content and len(response.content) > 0:
                first = response.content[0]
                raw = getattr(first, "text", None)
                content = raw if isinstance(raw, str) else ""
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
            logger.debug("Anthropic generate_insights failed: %s", e, exc_info=True)
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
            logger.warning("Anthropic summarization not initialized for extract_kg_graph")
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
                _safe_anthropic_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.messages.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.1,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_prompt}],
                )

            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_anthropic_retryable(),
            )
            content = ""
            if response.content and len(response.content) > 0:
                first = response.content[0]
                raw = getattr(first, "text", None)
                content = raw if isinstance(raw, str) else ""
            return parse_kg_graph_response(
                (content or "").strip(),
                max_topics=max_topics,
                max_entities=max_entities,
            )
        except Exception as e:
            logger.debug("Anthropic extract_kg_graph failed: %s", e, exc_info=True)
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
                "Anthropic summarization not initialized for " "extract_kg_from_summary_bullets"
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
                _safe_anthropic_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.messages.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.1,
                    system=system_msg,
                    messages=[{"role": "user", "content": user_prompt}],
                )

            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_anthropic_retryable(),
            )
            content = ""
            if response.content and len(response.content) > 0:
                first = response.content[0]
                raw = getattr(first, "text", None)
                content = raw if isinstance(raw, str) else ""
            return parse_kg_graph_response(
                (content or "").strip(),
                max_topics=max_topics,
                max_entities=max_entities,
            )
        except Exception as e:
            logger.debug("Anthropic extract_kg_from_summary_bullets failed: %s", e, exc_info=True)
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
            "You extract a single short quote from the transcript that best supports "
            "the given insight. Reply with ONLY a JSON object: "
            '{"quote_text": "exact quote from transcript"}'
        )
        user = (
            f"Transcript (excerpt):\n{transcript.strip()[:50000]}\n\n"
            f"Insight: {insight_text.strip()}\n\n"
            "Return JSON with quote_text only."
        )
        try:
            from ...utils.provider_metrics import (
                _safe_anthropic_retryable,
                anthropic_message_usage_tokens,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("anthropic")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.messages.create(
                    model=self.summary_model,
                    max_tokens=512,
                    temperature=0.0,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_anthropic_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = anthropic_message_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(call_metrics, pm, in_tok, out_tok)
            content = ""
            if response.content and len(response.content) > 0:
                first = response.content[0]
                raw = getattr(first, "text", None)
                content = raw if isinstance(raw, str) else ""
            content = (content or "").strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            obj = json.loads(content)
            quote_text = (obj.get("quote_text") or "").strip()
            if not quote_text:
                return []
            resolved = resolve_llm_quote_span(transcript, quote_text)
            if resolved is None:
                return []
            start, end, verbatim = resolved
            return [
                QuoteCandidate(
                    char_start=start,
                    char_end=end,
                    text=verbatim,
                    qa_score=1.0,
                )
            ]
        except Exception as e:
            logger.debug("Anthropic extract_quotes failed: %s", e, exc_info=True)
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
                _safe_anthropic_retryable,
                anthropic_message_usage_tokens,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("anthropic")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.messages.create(
                    model=self.summary_model,
                    max_tokens=10,
                    temperature=0.0,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_anthropic_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = anthropic_message_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(call_metrics, pm, in_tok, out_tok)
            content = ""
            if response.content and len(response.content) > 0:
                first = response.content[0]
                raw = getattr(first, "text", None)
                content = raw if isinstance(raw, str) else "0"
            content = (content or "0").strip()
            for part in content.replace(",", " ").split():
                try:
                    v = float(part)
                    return max(0.0, min(1.0, v))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            logger.debug("Anthropic score_entailment failed: %s", e, exc_info=True)
            return 0.0

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities object describing Anthropic provider capabilities
        """
        return ProviderCapabilities(
            supports_transcription=False,  # Anthropic doesn't support audio transcription
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_semantic_cleaning=True,  # Anthropic supports LLM-based cleaning
            supports_audio_input=False,  # Anthropic doesn't accept audio files
            supports_json_mode=True,  # Anthropic supports JSON mode
            max_context_tokens=self.max_context_tokens,
            supports_tool_calls=True,  # Anthropic supports tool use
            supports_system_prompt=True,  # Anthropic supports system prompts
            supports_streaming=True,  # Anthropic API supports streaming
            provider_name="anthropic",
            supports_gi_segment_timing=False,
        )
