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

import logging
from typing import Any, cast, Dict, Optional, Set, TYPE_CHECKING

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

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
    from ..capabilities import ProviderCapabilities
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]
from ...cleaning import PatternBasedCleaner
from ...cleaning.base import TranscriptCleaningProcessor
from ...utils.cleaning_max_tokens import (
    clamp_cleaning_max_tokens,
    estimate_cleaning_output_tokens,
    GROK_CLEANING_MAX_TOKENS,
)
from ...utils.log_redaction import format_exception_for_log
from ...utils.provider_metadata import warn_if_truncated
from ...utils.timeout_config import get_http_timeout
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
from .. import guardrails as _guardrails, insight_salvage as _insight_salvage
from ..common.transcript_cache import openai_style_messages as _openai_style_messages
from ._grok_provider_calls import (  # noqa: F401 - re-exported: moved there with S1
    _record_grok_llm_call,
)
from .grok_provider_gi_naming import GrokProviderGiNamingMethods

# Pricing for Grok models lives in ``config/pricing_assumptions.yaml`` (#651).


class GrokProvider(GrokProviderGiNamingMethods):
    """Unified Grok provider: SpeakerDetector and SummarizationProvider (no transcription).

    Uses Grok chat API via OpenAI SDK. Real-time info via X/Twitter. No audio API.
    """

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

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
                "Install the project (OpenAI SDK is a core dependency), e.g. pip install -e ."
            )

        if not cfg.grok_api_key:
            raise ValueError(
                "Grok API key required for Grok provider. "
                "Set GROK_API_KEY environment variable or grok_api_key in config."
            )

        from ...utils.provider_metadata import validate_api_key_format

        is_valid, _ = validate_api_key_format(
            cfg.grok_api_key,
            "Grok",
            expected_prefixes=None,
        )
        if not is_valid:
            # Do not log validation detail: CodeQL taints any message from this API-key path.
            logger.warning(
                "Grok API key validation failed (missing or too short); "
                "credentials are never logged."
            )

        self.cfg = cfg
        # RFC-115: relocate the transcript to a cacheable leading system block (default on).
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

        # Cleaning model settings (cheaper model for cost efficiency)
        self.cleaning_model = getattr(cfg, "grok_cleaning_model", "grok-4.3")
        self.cleaning_temperature = getattr(cfg, "grok_cleaning_temperature", 0.2)

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

        # Configure HTTP timeouts — Grok is a reasoning model that takes longer
        # per call than other providers (thinking before responding). Use
        # grok_timeout (default 1800s) as read timeout instead of generic timeout.
        grok_read_timeout = float(getattr(cfg, "grok_timeout", 1800))
        client_kwargs["timeout"] = get_http_timeout(cfg, read_timeout=grok_read_timeout)

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
        # The window this DEPLOYMENT serves (#2050). xAI publishes 128k, but a
        # gateway or a self-hosted stand-in can serve less — and a budget derived from an
        # overstated window is a prompt the server rejects or silently truncates. So the
        # published figure is the FALLBACK, and a profile/StageOption can declare the real one.
        self.max_context_tokens = int(getattr(cfg, "grok_max_context_tokens", 0) or 0) or 128000

        # Initialization state
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        # API providers handle rate limiting and retries internally via SDK
        self._requires_separate_instances = False

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
        ext = lookup_external_pricing(table, "grok", capability, model)
        return dict(ext) if ext else {}

    def initialize(self) -> None:
        """Initialize all Grok capabilities.

        For Grok API, initialization is a no-op but we track it for consistency.
        This method is idempotent and can be called multiple times safely.
        """
        # Initialize speaker detection if enabled
        if self.cfg.auto_speakers and not self._speaker_detection_initialized:
            self._initialize_speaker_detection()

        # ALWAYS initialized — this capability also serves extract_kg_graph / generate_insights /
        # clean_transcript / the evidence stack, so gating it on the generate_summaries STAGE flag
        # starved KG-only runs and lazily-built failover tiers (#1720).
        if not self._summarization_initialized:
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
        # Plain-text summary path — bypass the structured-JSON floor that
        # was silently clamping max_tokens up to 4096 even for plain text
        # (#1023 follow-up to #1016 Phase 2b). Bundled / structured-JSON
        # callsites (summarize_mega_bundled below) keep the default
        # ``structured=True`` and the floor still applies.
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
            "Summarizing text via Grok API (model: %s, max_tokens: %d)",
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
                _safe_openai_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("grok")
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=_openai_style_messages(  # RFC-115 transcript-first
                        text, system_prompt, user_prompt, enabled=self._cache_transcript_prefix
                    ),
                    temperature=self.summary_temperature,
                    max_tokens=max_length,
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            warn_if_truncated(
                response.choices[0].finish_reason,
                "grok",
                "summarize",
            )

            summary = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Token + cost capture up-front so cost emits in both branches (ADR-100).
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage") and response.usage:
                prompt_tokens_val = getattr(response.usage, "prompt_tokens", None)
                completion_tokens_val = getattr(response.usage, "completion_tokens", None)
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

            cost: Optional[float] = None
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

            def _record_cost(*, triggered_guardrail: bool = False) -> None:
                if input_tokens is None:
                    return
                from ...utils.provider_metrics import record_provider_call_cost

                record_provider_call_cost(
                    call_metrics,
                    cost,
                    cfg=self.cfg,
                    provider_type="grok",
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    triggered_guardrail=triggered_guardrail,
                    stage=use_case_stage(use_case),
                )

            # ADR-100 response-shape guardrail; Grok is OpenAI-compatible.
            try:
                _guardrails.check_chat_response(
                    summary, service="grok", finish_reason=finish_reason
                )
            except _guardrails.GuardrailViolation:
                _record_cost(triggered_guardrail=True)
                raise

            logger.debug(
                "Grok summarization completed: %d characters",
                len(summary),
            )

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
            prompt_metadata = use_case_prompt_metadata("grok", use_case) or prompt_metadata
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

        except _guardrails.GuardrailViolation:
            # ADR-100: propagate the raw violation so FallbackAware routes.
            raise
        except Exception as exc:
            logger.error("Grok API error in summarization: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Grok-specific error types (OpenAI-compatible errors)
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Grok authentication failed: {format_exception_for_log(exc)}",
                    provider="GrokProvider/Summarization",
                    suggestion="Check your GROK_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Grok rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="GrokProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Grok invalid model: {format_exception_for_log(exc)}",
                    provider="GrokProvider/Summarization",
                    suggestion="Check grok_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Grok summarization failed: {format_exception_for_log(exc)}",
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
        use_case: Optional[SummaryUseCase] = None,
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store."""
        from ...prompts.store import render_prompt

        if use_case is not None:  # a use case's own prompts
            system, user, system_name, user_name = build_use_case_prompts("grok", use_case)
            return system, user, system_name, user_name, 0, 0

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
            raise RuntimeError("Grok summarization not initialized for complete_text")

        response = self.client.chat.completions.create(
            model=self.summary_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=int(max_tokens),
            response_format={"type": "json_object"},
        )
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            response,
            provider_type="grok",
            capability="speaker_detection",
            stage=stage,
            model=self.summary_model,
        )
        content = (response.choices[0].message.content or "").strip()
        _guardrails.check_chat_response(content, service="grok", expect_json=True)
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
                "GrokProvider summarization not initialized. Call initialize() first."
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
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=model,
                    messages=_openai_style_messages(  # RFC-115 transcript-first
                        text_slice, system_msg, user_prompt, enabled=self._cache_transcript_prefix
                    ),
                    temperature=0.1,
                    max_tokens=2048,
                )

            response = retry_with_metrics(
                _make_api_call,
                max_retries=3,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
            )
            _record_grok_llm_call(
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
                provider_type="grok",
                capability="kg",
                stage="kg",
                model=model,
            )
            raw = (response.choices[0].message.content or "").strip()
            return parse_kg_graph_response(raw, max_topics=max_topics, max_entities=max_entities)
        except Exception as e:
            logger.debug("Grok extract_kg_graph failed: %s", e, exc_info=True)
            return None

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
            raise RuntimeError("GrokProvider not initialized. Call initialize() first.")

        from ...prompts.store import render_prompt

        # Build cleaning prompt using prompt_store
        prompt_name = "grok/cleaning/v1"
        user_prompt = render_prompt(prompt_name, transcript=text)

        # Use system prompt (OpenAI-compatible pattern)
        system_prompt = (
            "You are a transcript cleaning assistant. "
            "Remove sponsors, ads, intros, outros, and meta-commentary. "
            "Preserve all substantive content and speaker information. "
            "Return only the cleaned text, no explanations."
        )

        logger.debug(
            "Cleaning transcript via Grok API (model: %s, text length: %d chars)",
            self.cleaning_model,
            len(text),
        )

        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("grok")
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.cleaning_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cleaning_temperature,
                    max_tokens=clamp_cleaning_max_tokens(
                        estimate_cleaning_output_tokens(len(text.split())),
                        GROK_CLEANING_MAX_TOKENS,
                    ),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_openai_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            _record_grok_llm_call(
                response,
                pipeline_metrics,
                recorder_name="record_llm_cleaning_call",
                cfg=self.cfg,
                model=self.cleaning_model,
            )

            cleaned = response.choices[0].message.content

            # Cost capture up-front (ADR-100 cost-attribution).
            in_tok_cl = None
            out_tok_cl = None
            if hasattr(response, "usage") and response.usage:
                pt = getattr(response.usage, "prompt_tokens", None)
                ct = getattr(response.usage, "completion_tokens", None)
                in_tok_cl = int(pt) if isinstance(pt, (int, float)) else None
                out_tok_cl = int(ct) if isinstance(ct, (int, float)) else None
            cleaning_cost: Optional[float] = None
            if in_tok_cl is not None and out_tok_cl is not None:
                from ...workflow.helpers import calculate_provider_cost

                cleaning_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="grok",
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
                        provider="grok",
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
                logger.warning("Grok API returned empty cleaned text, using original")
                _emit_cleaning_cost()
                return text
            # ADR-100: cleaning catch-and-degrade; cost emitted in both branches.
            try:
                _guardrails.check_chat_response(
                    cleaned,
                    service="grok",
                    finish_reason=response.choices[0].finish_reason,
                )
            except _guardrails.GuardrailViolation:
                _emit_cleaning_cost(triggered_guardrail=True)
                logger.warning(
                    "Grok cleaning output failed guardrail; " "returning original transcript text"
                )
                return text

            _emit_cleaning_cost()
            logger.debug("Grok cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except _guardrails.GuardrailViolation:
            # Defensive outer catch — preserve contract.
            logger.warning(
                "Grok cleaning output failed guardrail (outer); "
                "returning original transcript text"
            )
            return text
        except Exception as exc:
            logger.error("Grok API error in cleaning: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle Grok-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Grok authentication failed: {format_exception_for_log(exc)}",
                    provider="GrokProvider/Cleaning",
                    suggestion="Check your GROK_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Grok rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="GrokProvider/Cleaning",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Grok cleaning failed: {format_exception_for_log(exc)}",
                    provider="GrokProvider/Cleaning",
                ) from exc

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component)."""
        return self._speaker_detection_initialized or self._summarization_initialized

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities object describing Grok provider capabilities
        """
        from ..capabilities import ProviderCapabilities  # noqa: PLC0415

        return ProviderCapabilities(
            supports_transcription=False,  # Grok doesn't support audio transcription
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_semantic_cleaning=True,  # Grok supports LLM-based cleaning
            supports_audio_input=False,  # Grok doesn't accept audio files
            supports_json_mode=True,  # Grok supports JSON mode
            max_context_tokens=self.max_context_tokens,
            supports_tool_calls=True,  # Grok supports function calling
            supports_system_prompt=True,  # Grok supports system prompts
            supports_streaming=True,  # Grok API supports streaming
            provider_name="grok",
            supports_gi_segment_timing=False,
        )
