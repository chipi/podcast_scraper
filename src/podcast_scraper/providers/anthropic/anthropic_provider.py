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

import logging
import time
from typing import Any, cast, Dict, Optional, Set, TYPE_CHECKING

# Import Anthropic SDK
try:
    import anthropic
    from anthropic import Anthropic
except ImportError:
    anthropic = None  # type: ignore
    Anthropic = None  # type: ignore

from ... import config, config_constants
from ..common.summary_use_case import (
    build_use_case_prompts,
    reply_budget,
    SummaryUseCase,
    use_case_from_params,
    use_case_prompt_metadata,
    use_case_stage,
)

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
from .. import guardrails as _guardrails, insight_salvage as _insight_salvage
from ..capabilities import ProviderCapabilities
from ..common.transcript_cache import anthropic_style_system as _anthropic_style_system

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
from ._anthropic_provider_calls import (  # re-exported
    _anthropic_finish_reason,
    _record_anthropic_llm_call,
)
from .anthropic_provider_gi_naming import AnthropicProviderGiNamingMethods

# Pricing for Anthropic models lives in ``config/pricing_assumptions.yaml`` (#651).


# The Claude 5 generation (claude-sonnet-5, claude-opus-4-8) DEPRECATED the `temperature` parameter:
# the API rejects any request that carries it with a 400 "`temperature` is deprecated for this
# model", which crashes the whole run at episode 1. Older 4.x models (sonnet-4-5/4-6, haiku-4-5)
# still accept it. Rather than hardcode a model list that rots as Anthropic ships new models, the
# provider LEARNS: it seeds the known rejecters, omits temperature for them, and on an unexpected
# "temperature deprecated" 400 it records the model and retries without temperature. So a future
# model that drops temperature self-heals after one failed call instead of taking down the run.
_TEMPERATURE_DEPRECATED_SEED = frozenset({"claude-sonnet-5", "claude-opus-4-8"})


class AnthropicProvider(AnthropicProviderGiNamingMethods):
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
        # RFC-115: relocate the transcript to a cache_control'd leading system block (default on).
        self._cache_transcript_prefix = bool(getattr(cfg, "cache_transcript_prefix", True))

        # Set up transcript cleaning processor based on strategy (Issue #418)
        from ...cleaning import HybridCleaner, LLMBasedCleaner

        cleaning_strategy = getattr(cfg, "transcript_cleaning_strategy", "hybrid")
        if cleaning_strategy == "pattern":
            self.cleaning_processor = PatternBasedCleaner()  # type: ignore[assignment]
        elif cleaning_strategy == "llm":
            self.cleaning_processor = LLMBasedCleaner()  # type: ignore[assignment]
        else:  # hybrid (default)
            self.cleaning_processor = HybridCleaner()  # type: ignore[assignment]

        # Cleaning model settings (cheaper model for cost efficiency).
        # Temperature tuned in #594 (data/eval/runs/baseline_cleaning_autoresearch_v1):
        # 0.4 won on v2 smoke sweep AND on 2/3 real-prod episodes from manual-run-10.
        self.cleaning_model = getattr(cfg, "anthropic_cleaning_model", "claude-haiku-4-5")
        self.cleaning_temperature = getattr(cfg, "anthropic_cleaning_temperature", 0.4)

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

        # The generic HTTP timeout is sized for RSS fetches: a 20s read. That is far too short for
        # an LLM call summarizing a 74k-char transcript, which takes 30-60s+, so anthropic cells
        # died with "Request timed out or interrupted" while every other provider completed.
        # Read gets the summarization budget; connect/write/pool keep the short HTTP values.
        timeout_config = get_http_timeout(
            cfg, read_timeout=float(config_constants.DEFAULT_SUMMARIZATION_TIMEOUT_SECONDS)
        )
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
        # The window this DEPLOYMENT serves (#2050). Claude publishes 200k, but a
        # gateway or a self-hosted stand-in can serve less — and a budget derived from an
        # overstated window is a prompt the server rejects or silently truncates. So the
        # published figure is the FALLBACK, and a profile/StageOption can declare the real one.
        self.max_context_tokens = (
            int(getattr(cfg, "anthropic_max_context_tokens", 0) or 0) or 200000
        )

        # Models known to reject `temperature` (see _TEMPERATURE_DEPRECATED_SEED). Grows at runtime
        # when the API teaches us a new one. Shared across instances so the lesson is learned once.
        self._temp_deprecated: Set[str] = set(_TEMPERATURE_DEPRECATED_SEED)

        # Initialization state
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        # API providers handle rate limiting and retries internally via SDK
        # Anthropic SDK automatically handles retries with exponential backoff
        self._requires_separate_instances = False

    def _messages_create(self, **kwargs: Any) -> Any:
        """``client.messages.create`` that survives models which dropped ``temperature``.

        Every Anthropic call routes through here. If the target model is known to reject
        ``temperature`` (Claude 5 generation), the parameter is stripped before the call. If a model
        we did NOT know about rejects it — a 400 whose message names ``temperature`` — the model is
        recorded and the call retried once without the parameter, so the failure self-heals instead
        of crashing the run at episode 1 (which is exactly what claude-sonnet-5 did).
        """
        model = kwargs.get("model")
        if model in self._temp_deprecated:
            kwargs.pop("temperature", None)
        try:
            return self.client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 — inspect, re-raise unless it is the temp case
            msg = str(exc).lower()
            if (
                "temperature" in kwargs
                and "temperature" in msg
                and ("deprecat" in msg or "not support" in msg or "unsupported" in msg)
            ):
                if isinstance(model, str):
                    self._temp_deprecated.add(model)
                kwargs.pop("temperature", None)
                return self.client.messages.create(**kwargs)
            raise

    @staticmethod
    def get_pricing(model: str, capability: str) -> Dict[str, float]:
        """Read pricing from ``config/pricing_assumptions.yaml`` (#651)."""
        from podcast_scraper.pricing_assumptions import (
            get_loaded_table,
            lookup_external_pricing,
        )

        table, _ = get_loaded_table("config/pricing_assumptions.yaml")
        if not table:
            return {}
        ext = lookup_external_pricing(table, "anthropic", capability, model)
        return dict(ext) if ext else {}

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

        # ALWAYS initialized — this capability also serves extract_kg_graph / generate_insights /
        # clean_transcript / the evidence stack, so gating it on the generate_summaries STAGE flag
        # starved KG-only runs and lazily-built failover tiers (#1720).
        if not self._summarization_initialized:
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
        # Plain-text summary path — bypass the structured-JSON floor that
        # was silently clamping max_tokens up to 4096 even for plain text
        # (#1023 follow-up to #1016 Phase 2b). Bundled / structured-JSON
        # callsites (summarize_mega_bundled / summarize_bundled below)
        # keep the default ``structured=True`` and the floor still applies.
        from ..common.output_tokens import cloud_structured_max_output_tokens

        max_length = cloud_structured_max_output_tokens(self.cfg, max_length, structured=False)
        min_length = (
            (params.get("min_length") if params else None)
            or self.cfg.summary_reduce_params.get("min_new_tokens")
            or 100
        )
        custom_prompt = params.get("prompt") if params else None
        # A summary other than the episode's (a board's) runs this same flow with its
        # own prompts, stage and budget; absent, this is the episode summary exactly as before.
        use_case = use_case_from_params(params)
        max_length = reply_budget(
            use_case,
            max_length,
            lambda n: cloud_structured_max_output_tokens(self.cfg, n, structured=False),
        )

        logger.debug(
            "Summarizing text via Anthropic API (model: %s, max_tokens: %d)",
            self.summary_model,
            max_length,
        )

        try:
            # Build prompts using prompt_store
            (
                system_prompt,
                user_prompt,
                system_prompt_name,
                user_prompt_name,
                paragraphs_min,
                paragraphs_max,
            ) = self._build_summarization_prompts(
                text,
                episode_title,
                episode_description,
                max_length,
                min_length,
                custom_prompt,
                use_case=use_case,
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
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            # Wrap API call with retry tracking
            # RFC-115: transcript block leads the system with a cache_control breakpoint.
            _sys, _usr = _anthropic_style_system(
                text, system_prompt, user_prompt, enabled=self._cache_transcript_prefix
            )

            def _make_api_call():
                return self._messages_create(
                    model=self.summary_model,
                    max_tokens=max_length,
                    temperature=self.summary_temperature,
                    system=_sys,
                    messages=[{"role": "user", "content": _usr}],
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
            # Extract token counts up-front so the cost event can fire on
            # both the happy path and the GuardrailViolation path
            # (ADR-100 paid-but-rejected cost-attribution).
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage") and response.usage:
                input_tokens_val = getattr(response.usage, "input_tokens", None)
                output_tokens_val = getattr(response.usage, "output_tokens", None)
                input_tokens = (
                    int(input_tokens_val) if isinstance(input_tokens_val, (int, float)) else 0
                )
                output_tokens = (
                    int(output_tokens_val) if isinstance(output_tokens_val, (int, float)) else 0
                )
                if input_tokens > 0 or output_tokens > 0:
                    call_metrics.set_tokens(input_tokens, output_tokens)

            cost: Optional[float] = None
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

            def _record_cost(*, triggered_guardrail: bool = False) -> None:
                if input_tokens is None:
                    return
                from ...utils.provider_metrics import record_provider_call_cost

                record_provider_call_cost(
                    call_metrics,
                    cost,
                    cfg=self.cfg,
                    provider_type="anthropic",
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    triggered_guardrail=triggered_guardrail,
                    response=response,  # RFC-115: surfaces cache_read/creation tokens in llm_cost
                    stage=use_case_stage(use_case),
                )

            # Response-shape guardrail (ADR-100, #1003): empty / thinking-prose /
            # finish_reason=length. Cost event fires in BOTH branches so the
            # operator can pivot on paid-but-rejected spend; the raw
            # GuardrailViolation propagates so FallbackAware can route the
            # configured degradation_policy.fallback_provider_on_failure.
            try:
                _guardrails.check_chat_response(
                    summary,
                    service="anthropic",
                    finish_reason=_anthropic_finish_reason(response),
                )
            except _guardrails.GuardrailViolation:
                _record_cost(triggered_guardrail=True)
                raise

            logger.debug("Anthropic summarization completed: %d characters", len(summary))

            _record_cost()

            # Track LLM call metrics if available (aggregate tracking)
            if (
                pipeline_metrics is not None
                and input_tokens is not None
                and output_tokens is not None
            ):
                pipeline_metrics.record_llm_summarization_call(
                    input_tokens, output_tokens, cost_usd=cost
                )

            # Get prompt metadata for tracking
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

            # A use case records its own templates; the episode summary keeps the above.
            prompt_metadata = use_case_prompt_metadata("anthropic", use_case) or prompt_metadata
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

        except _guardrails.GuardrailViolation:
            # ADR-100: propagate the raw violation so FallbackAwareSummarizationProvider
            # can route to the degradation policy's fallback. Wrapping into
            # ProviderRuntimeError would hide the type from the fallback layer.
            raise
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

    def _build_summarization_prompts(
        self,
        text: str,
        episode_title: Optional[str],
        episode_description: Optional[str],
        max_length: int,
        min_length: int,
        custom_prompt: Optional[str],
        use_case: Optional[SummaryUseCase] = None,
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store."""
        from ...prompts.store import render_prompt

        if use_case is not None:  # a use case's own prompts
            system, user, system_name, user_name = build_use_case_prompts("anthropic", use_case)
            return system, user, system_name, user_name, 0, 0

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

        # Build cleaning prompt using prompt_store
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
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            # Wrap API call with retry tracking
            def _make_api_call():
                return self._messages_create(
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

            _record_anthropic_llm_call(
                response,
                pipeline_metrics,
                recorder_name="record_llm_cleaning_call",
                cfg=self.cfg,
                model=self.cleaning_model,
            )

            # Extract text from response
            cleaned = ""
            if response.content and len(response.content) > 0:
                first_block = response.content[0]
                if hasattr(first_block, "text"):
                    cleaned = first_block.text
                elif isinstance(first_block, str):
                    cleaned = first_block

            # Cost capture up-front so cost emits on both happy path and the
            # GuardrailViolation catch-and-degrade path (ADR-100).
            in_tok_cl = None
            out_tok_cl = None
            if hasattr(response, "usage") and response.usage:
                ir = getattr(response.usage, "input_tokens", None)
                or_ = getattr(response.usage, "output_tokens", None)
                in_tok_cl = int(ir) if isinstance(ir, (int, float)) else None
                out_tok_cl = int(or_) if isinstance(or_, (int, float)) else None
            cleaning_cost: Optional[float] = None
            if in_tok_cl is not None and out_tok_cl is not None:
                from ...workflow.helpers import calculate_provider_cost

                cleaning_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="anthropic",
                    capability="cleaning",
                    model=self.cleaning_model,
                    prompt_tokens=in_tok_cl,
                    completion_tokens=out_tok_cl,
                )

            def _emit_cleaning_cost(*, triggered_guardrail: bool = False) -> None:
                if in_tok_cl is None or out_tok_cl is None:
                    return
                try:
                    from ...workflow.cost_monitoring import emit_llm_cost_event

                    emit_llm_cost_event(
                        self.cfg,
                        provider="anthropic",
                        stage="cleaning",
                        model=self.cleaning_model,
                        estimated_cost_usd=float(cleaning_cost or 0.0),
                        prompt_tokens=in_tok_cl,
                        completion_tokens=out_tok_cl,
                        triggered_guardrail=triggered_guardrail,
                        served_model=getattr(response, "model", None),
                        response=response,
                    )
                except Exception:  # noqa: BLE001
                    pass

            if not cleaned:
                logger.warning("Anthropic API returned empty cleaned text, using original")
                _emit_cleaning_cost()
                return text
            # ADR-100: cleaning catch-and-degrade. Cost emitted in both branches.
            try:
                _guardrails.check_chat_response(
                    cleaned,
                    service="anthropic",
                    finish_reason=_anthropic_finish_reason(response),
                )
            except _guardrails.GuardrailViolation:
                _emit_cleaning_cost(triggered_guardrail=True)
                logger.warning(
                    "Anthropic cleaning output failed guardrail; "
                    "returning original transcript text"
                )
                return text

            _emit_cleaning_cost()
            logger.debug("Anthropic cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except _guardrails.GuardrailViolation:
            # Defensive: outer catch in case a future change adds a guardrail
            # call outside the inline-handled if/else. Re-affirms the
            # catch-and-degrade contract.
            logger.warning(
                "Anthropic cleaning output failed guardrail (outer); "
                "returning original transcript text"
            )
            return text
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

    def complete_text(
        self, prompt: str, max_tokens: int = 800, stage: str = "speaker_detection"
    ) -> str:
        """One prompt in, one JSON answer out — the generic call ADR-110's resolver needs.

        `detect_speakers` is asked who speaks BEFORE the audio is downloaded, so it can only answer
        from show notes and returns the people an episode is ABOUT (that is how Elon Musk became a
        speaker, #876). The resolver asks the same model the same question AFTER diarization, with
        each voice's own words in hand — and that needs a plain completion, not a task method whose
        prompt is baked in.

        A provider WITHOUT this silently resolves nobody: the pipeline degrades instead of failing,
        and the model gets blamed for a method we never wrote.
        """
        if not self._summarization_initialized:
            raise RuntimeError("Anthropic summarization not initialized for complete_text")

        response = self._messages_create(
            model=self.summary_model,
            max_tokens=int(max_tokens),
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            response,
            provider_type="anthropic",
            capability="speaker_detection",
            stage=stage,
            model=self.summary_model,
        )
        content = ""
        if response.content and len(response.content) > 0:
            first = response.content[0]
            content = (getattr(first, "text", "") or "").strip()
        _guardrails.check_chat_response(content, service="anthropic", expect_json=True)
        return _insight_salvage.strip_json_fence(content)

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
            raise RuntimeError(
                "AnthropicProvider summarization not initialized. Call initialize() first."
            )
        from ...kg.llm_extract import (
            build_kg_transcript_system_prompt,
            build_kg_user_prompt,
            KG_RESPONSE_TOKENS,
            parse_kg_graph_response,
            resolve_kg_model_id,
            truncate_transcript_for_kg,
        )

        max_topics = min(max(1, max_topics), 20)
        max_entities = min(max(1, max_entities), 50)
        # #2050: derive the KG clip from this deployment's window, not a shared 120,000 literal
        # that fit no model in particular. Char-derived rather than tokenizer-counted: the KG path
        # accounts for 0 of the 565 measured context rejections, so a /tokenize round trip per
        # call is not justified here.
        text_slice = truncate_transcript_for_kg(
            text or "",
            limit=config_constants.transcript_budget_chars(
                getattr(self, "max_context_tokens", None),
                response_tokens=KG_RESPONSE_TOKENS,
            ),
        )
        if not text_slice.strip():
            return None
        model = resolve_kg_model_id(self, params)
        user_prompt = build_kg_user_prompt(
            text_slice,
            episode_title or "",
            max_topics,
            max_entities,
            prompt_version=(params or {}).get("kg_prompt_version", "v4"),
            ner_entity_hints=(params or {}).get("ner_entity_hints"),
        )
        system_msg = build_kg_transcript_system_prompt(max_topics, max_entities)
        try:
            from ...utils.provider_metrics import (
                _safe_anthropic_retryable,
                retry_with_metrics,
            )

            _sys, _usr = _anthropic_style_system(  # RFC-115 transcript-first + cache_control
                text_slice, system_msg, user_prompt, enabled=self._cache_transcript_prefix
            )

            def _make_api_call():
                return self._messages_create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.1,
                    system=_sys,
                    messages=[{"role": "user", "content": _usr}],
                )

            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_anthropic_retryable(),
            )
            _record_anthropic_llm_call(
                response,
                pipeline_metrics,
                recorder_name="record_llm_kg_call",
                cfg=self.cfg,
                model=model,
            )
            from ...utils.provider_metrics import record_response_cost

            record_response_cost(
                self.cfg,
                response,
                provider_type="anthropic",
                capability="kg",
                stage="kg",
                model=model,
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
