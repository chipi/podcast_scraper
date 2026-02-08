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
from typing import Any, Dict, Optional, Set, Tuple, TYPE_CHECKING

# Import Anthropic SDK
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore

from ... import config

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ...utils.timeout_config import get_http_timeout
from ...workflow import metrics
from ..capabilities import ProviderCapabilities

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

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


class AnthropicProvider:
    """Unified Anthropic provider implementing TranscriptionProvider, SpeakerDetector, and
    SummarizationProvider.

    This provider initializes and manages:
    - Note: Anthropic doesn't support native audio transcription, so transcription raises
      NotImplementedError
    - Claude chat models for speaker detection
    - Claude chat models for summarization

    All capabilities share the same Anthropic client, similar to how OpenAI and Gemini
    providers share the same API client. The client is initialized once and reused.
    """

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
                "Install with: pip install 'podcast-scraper[anthropic]'"
            )

        if not cfg.anthropic_api_key:
            raise ValueError(
                "Anthropic API key required for Anthropic provider. "
                "Set ANTHROPIC_API_KEY environment variable or anthropic_api_key in config."
            )

        # Validate API key format
        from ...utils.provider_metadata import validate_api_key_format

        is_valid, error_msg = validate_api_key_format(
            cfg.anthropic_api_key,
            "Anthropic",
            expected_prefixes=["sk-ant-"],
        )
        if not is_valid:
            # Note: error_msg does not contain the API key itself, only validation status
            logger.warning("Anthropic API key validation failed: %s", error_msg)

        self.cfg = cfg

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

            # Call Anthropic API
            # Anthropic uses messages API with system parameter
            response = self.client.messages.create(
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
            if pipeline_metrics is not None and hasattr(response, "usage"):
                usage = response.usage
                input_tokens = getattr(usage, "input_tokens", 0)
                output_tokens = getattr(usage, "output_tokens", 0)
                pipeline_metrics.record_llm_speaker_detection_call(input_tokens, output_tokens)

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Anthropic API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Anthropic API error in speaker detection: %s", exc)
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
                    message=f"Anthropic authentication failed: {exc}",
                    provider="AnthropicProvider/SpeakerDetection",
                    suggestion=(
                        "Check your ANTHROPIC_API_KEY environment variable or config setting"
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic rate limit exceeded: {exc}",
                    provider="AnthropicProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic invalid model: {exc}",
                    provider="AnthropicProvider/SpeakerDetection",
                    suggestion="Check anthropic_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Anthropic speaker detection failed: {exc}",
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
            from ...utils.provider_metrics import ProviderCallMetrics, retry_with_metrics

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
                    retryable_exceptions=(Exception,),  # Anthropic SDK handles specific errors
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
            logger.error("Anthropic API error in summarization: %s", exc)
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
                    message=f"Anthropic authentication failed: {exc}",
                    provider="AnthropicProvider/Summarization",
                    suggestion=(
                        "Check your ANTHROPIC_API_KEY environment variable or config setting"
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic rate limit exceeded: {exc}",
                    provider="AnthropicProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Anthropic invalid model: {exc}",
                    provider="AnthropicProvider/Summarization",
                    suggestion="Check anthropic_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Anthropic summarization failed: {exc}",
                    provider="AnthropicProvider/Summarization",
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

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities object describing Anthropic provider capabilities
        """
        return ProviderCapabilities(
            supports_transcription=False,  # Anthropic doesn't support audio transcription
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_audio_input=False,  # Anthropic doesn't accept audio files
            supports_json_mode=True,  # Anthropic supports JSON mode
            max_context_tokens=self.max_context_tokens,
            supports_tool_calls=True,  # Anthropic supports tool use
            supports_system_prompt=True,  # Anthropic supports system prompts
            supports_streaming=True,  # Anthropic API supports streaming
            provider_name="anthropic",
        )
