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
from typing import Any, cast, Dict, Optional, Set, Tuple, TYPE_CHECKING

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from ... import config

if TYPE_CHECKING:
    from ...models import Episode
    from ..capabilities import ProviderCapabilities
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ...cleaning import PatternBasedCleaner
from ...cleaning.base import TranscriptCleaningProcessor
from ...utils.timeout_config import get_http_timeout
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES

# DeepSeek API pricing constants (for cost estimation)
# Source: https://platform.deepseek.com/pricing
# Last updated: 2026-02
# Note: Prices subject to change. Always verify current rates
DEEPSEEK_CHAT_INPUT_COST_PER_1M_TOKENS = 0.28
DEEPSEEK_CHAT_OUTPUT_COST_PER_1M_TOKENS = 0.42
DEEPSEEK_CHAT_CACHE_HIT_INPUT_COST_PER_1M_TOKENS = 0.028  # 90% discount on cache hits


class DeepSeekProvider:
    """Unified DeepSeek provider: SpeakerDetector and SummarizationProvider (no transcription).

    Uses DeepSeek chat API via OpenAI SDK for speaker detection and summarization.
    Transcription is not supported.
    """

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

    def __init__(self, cfg: config.Config):
        """Initialize unified DeepSeek provider.

        Args:
            cfg: Configuration object with settings for both capabilities

        Raises:
            ValueError: If DeepSeek API key is not provided
            ImportError: If openai package is not installed
        """
        if OpenAI is None:
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
        self.cleaning_model = getattr(cfg, "deepseek_cleaning_model", "deepseek-chat")
        self.cleaning_temperature = getattr(cfg, "deepseek_cleaning_temperature", 0.2)

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

        # Configure HTTP timeouts with separate connect/read timeouts
        client_kwargs["timeout"] = get_http_timeout(cfg)

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
            model: Model name (e.g., "deepseek-chat", "deepseek-reasoner")
            capability: Capability type ("speaker_detection", "summarization")

        Returns:
            Dictionary with pricing information
        """
        pricing: Dict[str, float] = {}

        if capability in ("speaker_detection", "summarization"):
            # Text-based pricing
            pricing["input_cost_per_1m_tokens"] = DEEPSEEK_CHAT_INPUT_COST_PER_1M_TOKENS
            pricing["output_cost_per_1m_tokens"] = DEEPSEEK_CHAT_OUTPUT_COST_PER_1M_TOKENS
            pricing["cache_hit_input_cost_per_1m_tokens"] = (
                DEEPSEEK_CHAT_CACHE_HIT_INPUT_COST_PER_1M_TOKENS
            )

        return pricing

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
        logger.debug("DeepSeek speaker detection initialized successfully")

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing DeepSeek summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True
        logger.debug("DeepSeek summarization initialized successfully")

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
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle DeepSeek-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"DeepSeek authentication failed: {exc}",
                    provider="DeepSeekProvider/SpeakerDetection",
                    suggestion="Check your DEEPSEEK_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"DeepSeek rate limit exceeded: {exc}",
                    provider="DeepSeekProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"DeepSeek invalid model: {exc}",
                    provider="DeepSeekProvider/SpeakerDetection",
                    suggestion="Check deepseek_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"DeepSeek speaker detection failed: {exc}",
                    provider="DeepSeekProvider/SpeakerDetection",
                ) from exc

    def analyze_patterns(
        self,
        episodes: list[Episode],  # type: ignore[valid-type]
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

            # Track retries and rate limits
            from ...utils.provider_metrics import ProviderCallMetrics, retry_with_metrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("deepseek")

            # Wrap API call with retry tracking
            from openai import APIError, RateLimitError

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.summary_temperature,
                    max_tokens=max_length,
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=(RateLimitError, APIError, ConnectionError),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            summary = response.choices[0].message.content
            if not summary:
                logger.warning("DeepSeek API returned empty summary")
                summary = ""

            logger.debug("DeepSeek summarization completed: %d characters", len(summary))

            # Extract token counts and populate call_metrics
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage") and response.usage:
                prompt_tokens_val = getattr(response.usage, "prompt_tokens", None)
                completion_tokens_val = getattr(response.usage, "completion_tokens", None)
                # Convert to int if they're actual numbers, otherwise use 0
                # Handle Mock objects from tests by checking type
                input_tokens = (
                    int(prompt_tokens_val) if isinstance(prompt_tokens_val, (int, float)) else 0
                )
                output_tokens = (
                    int(completion_tokens_val)
                    if isinstance(completion_tokens_val, (int, float))
                    else 0
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
                    provider_type="deepseek",
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
                "summary_short": None,  # DeepSeek doesn't generate short summaries separately
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
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle DeepSeek-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"DeepSeek authentication failed: {exc}",
                    provider="DeepSeekProvider/Summarization",
                    suggestion="Check your DEEPSEEK_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"DeepSeek rate limit exceeded: {exc}",
                    provider="DeepSeekProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"DeepSeek invalid model: {exc}",
                    provider="DeepSeekProvider/Summarization",
                    suggestion="Check deepseek_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"DeepSeek summarization failed: {exc}",
                    provider="DeepSeekProvider/Summarization",
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

    def clean_transcript(self, text: str) -> str:
        """Clean transcript using LLM for semantic filtering.

        Args:
            text: Transcript text to clean (should already be pattern-cleaned)

        Returns:
            Cleaned transcript text

        Raises:
            RuntimeError: If provider is not initialized or cleaning fails
        """
        if not self._summarization_initialized:
            raise RuntimeError("DeepSeekProvider not initialized. Call initialize() first.")

        from ...prompts.store import render_prompt

        # Build cleaning prompt using prompt_store (RFC-017)
        prompt_name = "deepseek/cleaning/v1"
        user_prompt = render_prompt(prompt_name, transcript=text)

        # Use system prompt (OpenAI-compatible pattern)
        system_prompt = (
            "You are a transcript cleaning assistant. "
            "Remove sponsors, ads, intros, outros, and meta-commentary. "
            "Preserve all substantive content and speaker information. "
            "Return only the cleaned text, no explanations."
        )

        logger.debug(
            "Cleaning transcript via DeepSeek API (model: %s, text length: %d chars)",
            self.cleaning_model,
            len(text),
        )

        try:
            # Track retries and rate limits
            from ...utils.provider_metrics import ProviderCallMetrics, retry_with_metrics

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("deepseek")

            # Wrap API call with retry tracking
            from openai import APIError, RateLimitError

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.cleaning_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cleaning_temperature,
                    max_tokens=int(len(text.split()) * 0.85 * 1.3),  # Rough token estimate
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=(RateLimitError, APIError, ConnectionError),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            cleaned = response.choices[0].message.content
            if not cleaned:
                logger.warning("DeepSeek API returned empty cleaned text, using original")
                return text

            logger.debug("DeepSeek cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except Exception as exc:
            logger.error("DeepSeek API error in cleaning: %s", exc)
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle DeepSeek-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"DeepSeek authentication failed: {exc}",
                    provider="DeepSeekProvider/Cleaning",
                    suggestion="Check your DEEPSEEK_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"DeepSeek rate limit exceeded: {exc}",
                    provider="DeepSeekProvider/Cleaning",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"DeepSeek cleaning failed: {exc}",
                    provider="DeepSeekProvider/Cleaning",
                ) from exc

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component)."""
        return self._speaker_detection_initialized or self._summarization_initialized

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities object describing DeepSeek provider capabilities
        """
        from ..capabilities import ProviderCapabilities  # noqa: PLC0415

        return ProviderCapabilities(
            supports_transcription=False,  # DeepSeek doesn't support audio transcription
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_semantic_cleaning=True,  # DeepSeek supports LLM-based cleaning
            supports_audio_input=False,  # DeepSeek doesn't accept audio files
            supports_json_mode=True,  # DeepSeek supports JSON mode
            max_context_tokens=self.max_context_tokens,
            supports_tool_calls=True,  # DeepSeek supports function calling
            supports_system_prompt=True,  # DeepSeek supports system prompts
            supports_streaming=True,  # DeepSeek API supports streaming
            provider_name="deepseek",
        )
