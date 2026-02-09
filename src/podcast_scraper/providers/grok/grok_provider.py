"""Unified Grok provider for speaker detection and summarization.

This module provides a single GrokProvider class that implements two protocols:
- SpeakerDetector (using Grok chat API via OpenAI SDK)
- SummarizationProvider (using Grok chat API via OpenAI SDK)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key insight: Grok uses an OpenAI-compatible API, so we reuse the OpenAI SDK
with a custom base_url. No new dependency required.

Key advantage: Real-time information access via X/Twitter integration.

Note: Grok does NOT support transcription (no audio API).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Set, Tuple, TYPE_CHECKING

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from ... import config

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ...utils.timeout_config import get_http_timeout
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
    - Grok chat API for speaker detection (via OpenAI SDK)
    - Grok chat API for summarization (via OpenAI SDK)

    All capabilities share the same OpenAI client (configured with Grok base_url),
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
            ImportError: If openai package is not installed
        """
        if OpenAI is None:
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

        # Configure HTTP timeouts with separate connect/read timeouts
        client_kwargs["timeout"] = get_http_timeout(cfg)

        self.client = OpenAI(**client_kwargs)

        # Speaker detection settings
        # Model validation happens at API call time - invalid models will raise clear errors
        # Config model provides defaults via default_factory, so attribute always exists
        self.speaker_model = cfg.grok_speaker_model
        self.speaker_temperature = cfg.grok_temperature
        logger.debug(
            "Grok speaker model from config: %s (type: %s)",
            self.speaker_model,
            type(self.speaker_model).__name__,
        )

        # Summarization settings
        # Model validation happens at API call time - invalid models will raise clear errors
        # Config model provides defaults via default_factory, so attribute always exists
        self.summary_model = cfg.grok_summary_model
        self.summary_temperature = cfg.grok_temperature
        logger.debug(
            "Grok summary model from config: %s (type: %s)",
            self.summary_model,
            type(self.summary_model).__name__,
        )
        # Context window size (verify with xAI documentation - common is 128k)
        self.max_context_tokens = 128000  # Conservative estimate, verify with API docs

        # Initialization state
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        # API providers handle rate limiting and retries internally via SDK
        self._requires_separate_instances = False

    @staticmethod
    def get_pricing(model: str, capability: str) -> Dict[str, float]:
        """Get pricing information for a specific model and capability.

        Args:
            model: Model name (e.g., "grok-beta", "grok-2")
            capability: Capability type ("speaker_detection", "summarization")

        Returns:
            Dictionary with pricing information
        """
        pricing: Dict[str, float] = {}

        # Text-based pricing (speaker detection, summarization)
        model_lower = model.lower()

        # Check for specific model families
        if "grok-2" in model_lower or "grok2" in model_lower:
            pricing["input_cost_per_1m_tokens"] = GROK_2_INPUT_COST_PER_1M_TOKENS
            pricing["output_cost_per_1m_tokens"] = GROK_2_OUTPUT_COST_PER_1M_TOKENS
        elif "grok-beta" in model_lower or "betagrok" in model_lower:
            pricing["input_cost_per_1m_tokens"] = GROK_BETA_INPUT_COST_PER_1M_TOKENS
            pricing["output_cost_per_1m_tokens"] = GROK_BETA_OUTPUT_COST_PER_1M_TOKENS
        else:
            # Default to grok-2 pricing for unknown models (conservative estimate)
            pricing["input_cost_per_1m_tokens"] = GROK_2_INPUT_COST_PER_1M_TOKENS
            pricing["output_cost_per_1m_tokens"] = GROK_2_OUTPUT_COST_PER_1M_TOKENS

        return pricing

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
        logger.debug("Grok speaker detection initialized successfully")

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Grok summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True
        logger.debug("Grok summarization initialized successfully")

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

            system_prompt_name = self.cfg.grok_speaker_system_prompt or "grok/ner/system_ner_v1"
            system_prompt = render_prompt(system_prompt_name)

            # Call Grok API (OpenAI-compatible format)
            # This call may raise exceptions (RateLimitError, AuthenticationError, etc.)
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

            # Extract response text (this will raise if response structure is invalid)
            # Only access response if the API call succeeded
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
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Grok-specific error types (OpenAI-compatible errors)
            # Check exception type first (more reliable than string matching)
            exc_type_name = type(exc).__name__
            try:
                from openai import AuthenticationError, BadRequestError, RateLimitError

                # Try isinstance check first (works when openai is not mocked)
                try:
                    if isinstance(exc, AuthenticationError):
                        raise ProviderAuthError(
                            message=f"Grok authentication failed: {exc}",
                            provider="GrokProvider/SpeakerDetection",
                            suggestion=(
                                "Check your GROK_API_KEY environment variable " "or config setting"
                            ),
                        ) from exc
                    elif isinstance(exc, RateLimitError):
                        raise ProviderRuntimeError(
                            message=f"Grok rate limit exceeded: {exc}",
                            provider="GrokProvider/SpeakerDetection",
                            suggestion="Wait before retrying or check your API quota",
                        ) from exc
                    elif isinstance(exc, BadRequestError):
                        error_msg = str(exc).lower()
                        if "invalid" in error_msg and "model" in error_msg:
                            raise ProviderRuntimeError(
                                message=f"Grok invalid model: {exc}",
                                provider="GrokProvider/SpeakerDetection",
                                suggestion="Check grok_speaker_model configuration",
                            ) from exc
                except TypeError:
                    # isinstance() failed (likely because exception classes are mocked)
                    # Fall through to type name checking
                    pass

                # Fallback: check by exception type name (works even when mocked)
                if exc_type_name == "AuthenticationError":
                    raise ProviderAuthError(
                        message=f"Grok authentication failed: {exc}",
                        provider="GrokProvider/SpeakerDetection",
                        suggestion="Check your GROK_API_KEY environment variable or config setting",
                    ) from exc
                elif exc_type_name == "RateLimitError":
                    raise ProviderRuntimeError(
                        message=f"Grok rate limit exceeded: {exc}",
                        provider="GrokProvider/SpeakerDetection",
                        suggestion="Wait before retrying or check your API quota",
                    ) from exc
                elif exc_type_name == "BadRequestError":
                    error_msg = str(exc).lower()
                    if "invalid" in error_msg and "model" in error_msg:
                        raise ProviderRuntimeError(
                            message=f"Grok invalid model: {exc}",
                            provider="GrokProvider/SpeakerDetection",
                            suggestion="Check grok_speaker_model configuration",
                        ) from exc
            except ImportError:
                # Fallback if OpenAI exceptions not available
                pass

            # Fallback to string-based error detection
            # This handles cases where exceptions are mocked (type name won't match)
            error_msg = str(exc).lower()
            exc_repr = repr(exc).lower()

            # Check both error message and exception representation
            # For mocked exceptions, the repr contains the class name
            # (e.g., "mock.AuthenticationError()")
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "authenticationerror" in exc_repr
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"Grok authentication failed: {exc}",
                    provider="GrokProvider/SpeakerDetection",
                    suggestion="Check your GROK_API_KEY environment variable or config setting",
                ) from exc
            elif (
                "quota" in error_msg
                or "rate limit" in error_msg
                or "ratelimiterror" in exc_repr
                or exc_type_name == "RateLimitError"
            ):
                raise ProviderRuntimeError(
                    message=f"Grok rate limit exceeded: {exc}",
                    provider="GrokProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif (
                ("invalid" in error_msg and "model" in error_msg)
                or "badrequesterror" in exc_repr
                or exc_type_name == "BadRequestError"
            ):
                raise ProviderRuntimeError(
                    message=f"Grok invalid model: {exc}",
                    provider="GrokProvider/SpeakerDetection",
                    suggestion="Check grok_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Grok speaker detection failed: {exc}",
                    provider="GrokProvider/SpeakerDetection",
                ) from exc

    def analyze_patterns(
        self,
        episodes: list[Episode],  # type: ignore[valid-type]
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
        """Parse speaker names from Grok API response."""
        try:
            # Ensure response_text is a string, not a Mock or other object
            if not isinstance(response_text, str):
                response_text = str(response_text)
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
        """Summarize text using Grok API.

        Can handle full transcripts directly if context window is large enough.
        Chunking may be needed depending on Grok's context window size
        (likely 128k, verify with API docs).

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

            # Track retries and rate limits
            from ...utils.provider_metrics import ProviderCallMetrics, retry_with_metrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("grok")

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
                logger.warning("Grok API returned empty summary")
                summary = ""

            logger.debug("Grok summarization completed: %d characters", len(summary))

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
                    provider_type="grok",
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
                "summary_short": None,  # Grok doesn't generate short summaries separately
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
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Grok-specific error types (OpenAI-compatible errors)
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Grok authentication failed: {exc}",
                    provider="GrokProvider/Summarization",
                    suggestion="Check your GROK_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Grok rate limit exceeded: {exc}",
                    provider="GrokProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Grok invalid model: {exc}",
                    provider="GrokProvider/Summarization",
                    suggestion="Check grok_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Grok summarization failed: {exc}",
                    provider="GrokProvider/Summarization",
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

        system_prompt_name = self.cfg.grok_summary_system_prompt or "grok/summarization/system_v1"
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
