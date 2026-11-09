"""Unified Gemini provider for transcription, speaker detection, and summarization.

This module provides a single GeminiProvider class that implements all three protocols:
- TranscriptionProvider (using native multimodal audio understanding)
- SpeakerDetector (using Gemini chat models)
- SummarizationProvider (using Gemini chat models)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any, cast, Dict, Optional, Set, Tuple, TYPE_CHECKING

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
    clamp_cleaning_max_tokens,
    estimate_cleaning_output_tokens,
    GEMINI_CLEANING_MAX_OUTPUT_TOKENS,
)
from ...utils.log_redaction import format_exception_for_log, redact_for_log
from ...workflow import metrics
from .. import guardrails as _guardrails, insight_salvage as _insight_salvage
from ..capabilities import ProviderCapabilities

logger = logging.getLogger(__name__)

from ._gemini_provider_calls import (  # noqa: F401 - re-exported: moved there with S1
    _gemini_finish_reason,
    _GEMINI_PRO_THINKING_BUDGET,
    _merge_generate_content_config,
    _record_gemini_llm_call,
    _should_bound_thinking_for_pro,
    _should_disable_thinking_for_model,
)
from .gemini_provider_gi_naming import GeminiProviderGiNamingMethods

# Pricing for Gemini models lives in ``config/pricing_assumptions.yaml`` (#651).


class GeminiProvider(GeminiProviderGiNamingMethods):
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
        # Note: gemini-1.5-flash not available in new API; default flash-tier model in config.
        # Temperature tuned in #594 (data/eval/runs/baseline_cleaning_autoresearch_v1):
        # 0.4 won on v2 smoke sweep AND on 1/3 real-prod eps (tied on 2/3, lost none).
        self.cleaning_model = getattr(cfg, "gemini_cleaning_model", "gemini-2.5-flash-lite")
        self.cleaning_temperature = getattr(cfg, "gemini_cleaning_temperature", 0.4)

        # Suppress verbose Gemini SDK debug logs when DEBUG is enabled
        # globally (same pattern as the OpenAI provider).
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
        if genai_types is not None:
            # google-genai ``HttpOptions.timeout`` is documented in milliseconds.
            # ADR-100: honour ``summarization_timeout`` (read_timeout in our
            # shared helper) so a tight client-timeout config plumbs through
            # to the SDK — matches the contract on the other 3 cloud providers
            # (OpenAI / Anthropic / DeepSeek).
            from ...utils.timeout_config import get_http_timeout

            timeout_config = get_http_timeout(cfg)
            timeout_ms: Optional[int] = None
            try:
                read_t = getattr(timeout_config, "read", None) if timeout_config else None
                if read_t is None and isinstance(timeout_config, (int, float)):
                    read_t = float(timeout_config)
                if read_t is not None:
                    timeout_ms = int(float(read_t) * 1000)
            except Exception:  # noqa: BLE001 - defensive; never break client construction
                timeout_ms = None

            http_options_kwargs: Dict[str, Any] = {}
            if gemini_api_base:
                http_options_kwargs["base_url"] = gemini_api_base.rstrip("/")
            if timeout_ms is not None and timeout_ms > 0:
                http_options_kwargs["timeout"] = timeout_ms
            if http_options_kwargs:
                client_kwargs["http_options"] = genai_types.HttpOptions(**http_options_kwargs)
        elif gemini_api_base:
            # gemini_api_base is set but the SDK's types aren't importable to apply it.
            # FAIL LOUD instead of silently building a client against the hosted Gemini
            # API (same trap as the Deepgram provider): a configured base — the CI mock
            # server or a self-hosted endpoint — must be honored or error, never quietly
            # fall through to production.
            raise RuntimeError(
                "gemini_api_base is set but google.genai.types is unavailable to apply "
                "it; refusing to build a client against the default hosted Gemini API."
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
        # The window this DEPLOYMENT serves (#2050). Gemini 1.5/2.x publish a 1M-2M window, but a
        # gateway or a self-hosted stand-in can serve less — and a budget derived from an
        # overstated window is a prompt the server rejects or silently truncates. So the
        # published figure is the FALLBACK, and a profile/StageOption can declare the real one.
        self.max_context_tokens = int(getattr(cfg, "gemini_max_context_tokens", 0) or 0) or 2000000

        # RFC-115: Gemini EXPLICIT context caching (cachedContent). Off by default (stateful +
        # storage-billed, unlike the auto-cache providers). ``_gemini_cache_handles`` maps
        # "{model}:{sha256(transcript)}" -> cache resource name so an episode's stages reuse one
        # handle; a short TTL auto-expires it and we evict/delete stale handles on a new transcript.
        self._gemini_context_cache_enabled = bool(
            getattr(cfg, "gemini_context_cache_enabled", False)
        )
        self._gemini_cache_ttl_s = int(getattr(cfg, "gemini_context_cache_ttl_seconds", 300))
        self._gemini_cache_handles: Dict[str, str] = {}

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
        """Read pricing from ``config/pricing_assumptions.yaml`` (#651)."""
        from podcast_scraper.pricing_assumptions import (
            get_loaded_table,
            lookup_external_pricing,
        )

        table, _ = get_loaded_table("config/pricing_assumptions.yaml")
        if not table:
            return {}
        ext = lookup_external_pricing(table, "gemini", capability, model)
        return dict(ext) if ext else {}

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

        # ALWAYS initialized — this capability also serves extract_kg_graph / generate_insights /
        # clean_transcript / the evidence stack, so gating it on the generate_summaries STAGE flag
        # starved KG-only runs and lazily-built failover tiers (#1720).
        if not self._summarization_initialized:
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

    def _gemini_retry_params(
        self,
        ctx: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Kwargs for ``retry_with_metrics`` on Gemini HTTP/SDK calls (configurable).

        Passes ``pipeline_metrics`` through so #988 per-stage retry-reason
        attribution fires from inside the retry loop. The ``ctx["stage"]``
        already carries a per-call-site tag (e.g. ``gemini_megabundle``,
        ``gemini_gil_extract_quotes``) which the canonical-stage normalizer
        maps onto pipeline buckets (summarization, gi, …).
        """
        out: Dict[str, Any] = {
            "max_retries": self.cfg.gemini_retry_max_retries,
            "initial_delay": self.cfg.gemini_retry_initial_delay_seconds,
            "max_delay": self.cfg.gemini_retry_max_delay_seconds,
        }
        if ctx:
            out["retry_context"] = ctx
        if pipeline_metrics is not None:
            out["pipeline_metrics"] = pipeline_metrics
        return out

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

            # google-genai SDK expects Part objects, not raw dicts.
            contents = [
                genai_types.Part.from_bytes(data=audio_data, mime_type=mime_type),
                "Transcribe verbatim.",
            ]

            # Transcription-specific config (#577 Exp 3 research findings):
            #   - system_instruction: explicit anti-summary directive. Without it,
            #     gemini-2.5-flash-lite tends to return summary-like output rather
            #     than verbatim word-for-word text.
            #   - max_output_tokens=65536: 2.5 series cap. Default 8192 silently
            #     truncates long transcripts (32-min audio = ~6-9K tokens) and the
            #     model compresses toward summary rather than hard-cutting.
            #   - temperature=0.0: verbatim tasks should be deterministic.
            #   - thinking_budget=2048: python-genai #2204 documents that audio tasks
            #     on 2.5 series produce low-quality output when thinking is disabled.
            #     flash-lite defaults to 0; enable explicitly for transcription paths.
            transcription_system_instruction = (
                "You are a verbatim audio transcription service. "
                "Transcribe every spoken word exactly as said, including filler words "
                "and repetitions. Do NOT summarize, paraphrase, shorten, or omit any "
                "content. Output plain text only, no timestamps, no speaker labels "
                "unless explicitly requested, no formatting or commentary."
            )
            if effective_language:
                transcription_system_instruction += f" Language: {effective_language}."

            transcription_config = _merge_generate_content_config(
                self.transcription_model,
                {
                    "system_instruction": transcription_system_instruction,
                    "max_output_tokens": 65536,
                    "temperature": 0.0,
                    "thinking_config": {"thinking_budget": 2048},
                },
            )

            response = retry_with_metrics(
                lambda: self.client.models.generate_content(
                    model=self.transcription_model,
                    contents=cast(Any, contents),
                    config=cast(Any, transcription_config),
                ),
                **self._gemini_retry_params({"stage": "gemini_transcribe"}),
                retryable_exceptions=_safe_gemini_retryable(),
            )

            # Silent-truncation and safety-block diagnostics: response.text returns
            # empty on SAFETY/RECITATION finish reasons and silently concatenates
            # truncated output on MAX_TOKENS. Surface these so we can tell the
            # difference between "empty transcript" and "client blocked it".
            # Broad except because this is a purely diagnostic branch — callers
            # must never fail transcription because introspection hiccupped on
            # an atypical response object (e.g. test doubles).
            finish_reason: Optional[str] = None
            try:
                candidates = getattr(response, "candidates", None)
                first = candidates[0] if candidates else None
                fr = getattr(first, "finish_reason", None) if first is not None else None
                finish_reason = str(fr) if fr is not None else None
            except Exception:
                finish_reason = None
            if finish_reason and any(
                x in finish_reason.upper() for x in ("MAX_TOKENS", "SAFETY", "RECITATION")
            ):
                logger.warning(
                    "Gemini transcription finished with non-STOP reason: %s (audio=%s). "
                    "Output may be truncated or content-filtered.",
                    finish_reason,
                    audio_path,
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
        call_metrics.set_breaker_config_from_cfg(self.cfg)

        from ...utils.provider_metrics import (
            _safe_gemini_retryable,
            retry_with_metrics,
        )

        start_time = time.time()
        try:
            text = retry_with_metrics(
                lambda: self.transcribe(audio_path, language),
                **self._gemini_retry_params(
                    {"stage": "gemini_transcribe_outer"},
                    pipeline_metrics=pipeline_metrics,
                ),
                retryable_exceptions=_safe_gemini_retryable(),
                metrics=call_metrics,
            )
        except Exception:
            call_metrics.finalize()
            raise

        elapsed = time.time() - start_time
        call_metrics.finalize()

        # Resolve audio_minutes with a file-size fallback so cost isn't silently 0
        # when the caller doesn't pass episode_duration_seconds (#650 Finding 18
        # symmetry — OpenAI fix applies here too). Bitrate-aware: scales by
        # cfg.preprocessing_mp3_bitrate_kbps (32 kbps on speech_optimal_v1 would
        # otherwise under-report 4× with a naive 128-kbps assumption).
        audio_minutes = 0.0
        if episode_duration_seconds is not None:
            audio_minutes = float(episode_duration_seconds) / 60.0
        else:
            try:
                file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                bitrate_kbps_cfg = getattr(self.cfg, "preprocessing_mp3_bitrate_kbps", None)
                bitrate_kbps = float(bitrate_kbps_cfg) if bitrate_kbps_cfg else 128.0
                audio_minutes = file_size_mb * (128.0 / bitrate_kbps)
            except OSError:
                pass

        # Calculate cost first so the value flows to both call_metrics
        # (per-episode) and pipeline_metrics (per-stage aggregate).
        if audio_minutes > 0:
            from ...utils.provider_metrics import (
                record_provider_call_cost,
                record_transcription_cost_to_pipeline,
            )
            from ...workflow.helpers import calculate_provider_cost

            record_provider_call_cost(
                call_metrics,
                calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="gemini",
                    capability="transcription",
                    model=self.transcription_model,
                    audio_minutes=audio_minutes,
                ),
                cfg=self.cfg,
                provider_type="gemini",
                capability="transcription",
                model=self.transcription_model,
                audio_minutes=audio_minutes,
            )
            record_transcription_cost_to_pipeline(
                pipeline_metrics, call_metrics, audio_minutes, call_metrics.estimated_cost
            )

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

        For Gemini provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.
        """
        return None

    # ============================================================================
    # SummarizationProvider Protocol Implementation
    # ============================================================================

    # --- RFC-115: Gemini explicit context-cache lifecycle ------------------------------------
    # Below Gemini's minimum cacheable size (~1k tokens) an explicit cache can't be created, so we
    # skip and use the legacy layout. Real episode transcripts are far above this.
    _GEMINI_CACHE_MIN_CHARS = 4096

    def _gemini_cache_for(self, transcript: str, model: str) -> Optional[str]:
        """Get-or-create a ``cachedContent`` handle for this (model, transcript). ``None`` = legacy.

        One handle per (model, transcript) with a short TTL, reused across the episode's stages;
        stale handles are deleted when a new transcript arrives (bounds rent). Any failure, a
        disabled flag, or a transcript below the min cacheable size falls back to ``None``.
        """
        if not self._gemini_context_cache_enabled:
            return None
        canonical = (transcript or "").strip()
        if len(canonical) < self._GEMINI_CACHE_MIN_CHARS:
            return None
        import hashlib

        key = f"{model}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
        existing = self._gemini_cache_handles.get(key)
        if existing:
            return existing
        if genai_types is None:
            return None
        try:
            cache = self.client.caches.create(
                model=model,
                config=genai_types.CreateCachedContentConfig(
                    contents=cast(Any, [canonical]), ttl=f"{self._gemini_cache_ttl_s}s"
                ),
            )
            name = getattr(cache, "name", None)
            if not name:
                return None
            self._evict_gemini_caches(keep=key)  # new transcript -> drop the previous episode's
            self._gemini_cache_handles[key] = name
            return str(name)
        except Exception as exc:  # noqa: BLE001 - never fail a stage on a cache hiccup
            logger.debug("Gemini context-cache create failed; using legacy layout: %s", exc)
            return None

    def _evict_gemini_caches(self, *, keep: Optional[str] = None) -> None:
        """Delete every cache handle except ``keep`` (bounds accumulation + storage rent)."""
        for k, name in list(self._gemini_cache_handles.items()):
            if k == keep:
                continue
            try:
                self.client.caches.delete(name=name)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Gemini context-cache delete failed (will TTL-expire): %s", exc)
            self._gemini_cache_handles.pop(k, None)

    def _gemini_stage_request(
        self, transcript: str, system_prompt: str, user_prompt: str, model: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Return ``(contents, config_overrides)`` for a Gemini stage.

        Cache on + transcript cacheable: the transcript goes into a ``cachedContent`` handle and the
        stage's system instructions + task (transcript removed) become the request ``contents``,
        with ``cached_content`` referencing the handle. Otherwise legacy: the transcript stays in
        ``contents`` and the stage system prompt is passed as ``system_instruction``.
        """
        handle = self._gemini_cache_for(transcript, model)
        if handle is None:
            return user_prompt, {"system_instruction": system_prompt}
        from ..common.transcript_cache import relocate_transcript

        block, _sys, user_wo = relocate_transcript(
            transcript, system_prompt, user_prompt, enabled=True
        )
        if block is None:  # transcript not present verbatim -> legacy (keeps the input whole)
            return user_prompt, {"system_instruction": system_prompt}
        contents = f"{system_prompt}\n\n{user_wo}" if system_prompt else user_wo
        return contents, {"cached_content": handle}

    def summarize(  # noqa: C901
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
            "Summarizing text via Gemini API (model: %s, max_tokens: %d)",
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
            from ...utils.provider_metrics import ProviderCallMetrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("gemini")
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            from ...utils.provider_metrics import (
                _safe_gemini_retryable,
                retry_with_metrics,
            )

            # RFC-115: transcript in a cachedContent handle (if enabled); else legacy.
            _contents, _cfg_over = self._gemini_stage_request(
                text, system_prompt, user_prompt, self.summary_model
            )

            def _make_api_call():
                generation_config = _merge_generate_content_config(
                    self.summary_model,
                    {
                        "temperature": self.summary_temperature,
                        "max_output_tokens": max_length,
                        **_cfg_over,
                    },
                )

                return self.client.models.generate_content(
                    model=self.summary_model,
                    contents=_contents,
                    config=cast(Any, generation_config),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    **self._gemini_retry_params(
                        {
                            "stage": "gemini_summarize",
                            "episode_title": (episode_title or "")[:200],
                        },
                        pipeline_metrics=pipeline_metrics,
                    ),
                    retryable_exceptions=_safe_gemini_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            summary = response.text if hasattr(response, "text") else str(response)
            # Extract token counts up-front so the cost event can fire on
            # both the happy path and the GuardrailViolation path
            # (ADR-100 paid-but-rejected cost-attribution).
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

            # Truncation diagnostic: warn on MAX_TOKENS/SAFETY/RECITATION finish
            # reasons or near-budget output. Surfaces the class of silent
            # truncation that caused Flightcast failure on 2026-04-20.
            try:
                from ..common.output_tokens import warn_if_output_truncated

                finish_reason: Optional[str] = None
                try:
                    candidates = getattr(response, "candidates", None)
                    first = candidates[0] if candidates else None
                    fr = getattr(first, "finish_reason", None) if first is not None else None
                    finish_reason = str(fr) if fr is not None else None
                except Exception:
                    finish_reason = None
                warn_if_output_truncated(
                    provider_name="gemini",
                    finish_reason=finish_reason,
                    output_tokens=output_tokens,
                    max_output_tokens=max_length,
                )
            except Exception:
                pass

            cost: Optional[float] = None
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

            def _record_cost(*, triggered_guardrail: bool = False) -> None:
                if input_tokens is None:
                    return
                from ...utils.provider_metrics import record_provider_call_cost

                record_provider_call_cost(
                    call_metrics,
                    cost,
                    cfg=self.cfg,
                    provider_type="gemini",
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    triggered_guardrail=triggered_guardrail,
                    response=response,  # RFC-115: surfaces cached_content tokens in llm_cost
                    stage=use_case_stage(use_case),
                )

            # Response-shape guardrail (ADR-100, #1003): empty / thinking-prose /
            # finish_reason=length. Gemini "thinking" models (2.5-flash w/o
            # explicit budget=0) can burn output budget on hidden reasoning
            # and return empty/short content — caught here. Cost event fires
            # in BOTH branches so the operator can pivot on paid-but-rejected
            # spend; raw GuardrailViolation propagates so FallbackAware
            # routes the configured fallback.
            try:
                _guardrails.check_chat_response(
                    summary, service="gemini", finish_reason=_gemini_finish_reason(response)
                )
            except _guardrails.GuardrailViolation:
                _record_cost(triggered_guardrail=True)
                raise

            logger.debug("Gemini summarization completed: %d characters", len(summary))

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
            prompt_metadata = use_case_prompt_metadata("gemini", use_case) or prompt_metadata
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

        except _guardrails.GuardrailViolation:
            # ADR-100: propagate the raw violation so FallbackAwareSummarizationProvider
            # can route to the degradation policy's fallback. Wrapping into
            # ProviderRuntimeError would hide the type from the fallback layer.
            raise
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
            system, user, system_name, user_name = build_use_case_prompts("gemini", use_case)
            return system, user, system_name, user_name, 0, 0

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
        # RFC-115: delete any live context-cache handles so we don't pay storage rent past the run
        # (they would otherwise linger until their TTL).
        self._evict_gemini_caches(keep=None)

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

    def complete_text(
        self, prompt: str, max_tokens: int = 800, stage: str = "speaker_detection"
    ) -> str:
        """One prompt in, one JSON answer out — the generic call ADR-110's resolver needs.

        `detect_speakers` is asked who speaks BEFORE the audio is downloaded, so it can only answer
        from show notes and returns the people an episode is ABOUT (that is how Elon Musk became a
        speaker, #876). The resolver asks the same model the same question AFTER diarization, with
        each voice's own words in hand — and that needs a plain completion, not a task method whose
        prompt is baked in.
        """
        if not self._summarization_initialized:
            raise RuntimeError("Gemini summarization not initialized for complete_text")

        generation_config = _merge_generate_content_config(
            self.summary_model,
            {
                "temperature": 0.0,
                "max_output_tokens": int(max_tokens),
                "response_mime_type": "application/json",
            },
        )
        response = self.client.models.generate_content(
            model=self.summary_model,
            contents=prompt,
            config=cast(Any, generation_config),
        )
        # Observability (ADR-137): a real per-episode gemini call (speaker resolution, transcript
        # naming). Priced and counted against the run budget like every other stage; it used to
        # emit tokens with a cost of 0 (2026-10-23 cost audit). Never raises.
        from ...utils.provider_metrics import record_response_cost

        record_response_cost(
            self.cfg,
            response,
            provider_type="gemini",
            capability="speaker_detection",
            stage=stage,
            model=self.summary_model,
        )
        content = (getattr(response, "text", "") or "").strip()
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
                "GeminiProvider summarization not initialized. Call initialize() first."
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
                _safe_gemini_retryable,
                retry_with_metrics,
            )

            _contents, _cfg_over = self._gemini_stage_request(  # RFC-115
                text_slice, system_msg, user_prompt, model
            )
            generation_config = _merge_generate_content_config(
                model,
                {
                    "temperature": 0.1,
                    "max_output_tokens": 2048,
                    **_cfg_over,
                },
            )

            def _make_api_call():
                return self.client.models.generate_content(
                    model=model,
                    contents=_contents,
                    config=cast(Any, generation_config),
                )

            response = retry_with_metrics(
                _make_api_call,
                **self._gemini_retry_params(
                    {
                        "stage": "gemini_kg_transcript",
                        "episode_title": (episode_title or "")[:200],
                    },
                    pipeline_metrics=pipeline_metrics,
                ),
                retryable_exceptions=_safe_gemini_retryable(),
            )
            _record_gemini_llm_call(
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
                provider_type="gemini",
                capability="kg",
                stage="kg",
                model=model,
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

        # Build cleaning prompt using prompt_store
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
            call_metrics.set_breaker_config_from_cfg(self.cfg)

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
                    **self._gemini_retry_params(
                        {"stage": "gemini_clean_transcript"},
                        pipeline_metrics=pipeline_metrics,
                    ),
                    retryable_exceptions=_safe_gemini_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            _record_gemini_llm_call(
                response,
                pipeline_metrics,
                recorder_name="record_llm_cleaning_call",
                cfg=self.cfg,
                model=self.cleaning_model,
            )

            cleaned = response.text if hasattr(response, "text") else str(response)

            # Cost capture up-front (ADR-100 cost-attribution in both branches).
            in_tok_cl = None
            out_tok_cl = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                u = response.usage_metadata
                ir = getattr(u, "prompt_token_count", 0)
                or_ = getattr(u, "candidates_token_count", 0)
                try:
                    in_tok_cl = int(ir) if ir is not None else None
                except (TypeError, ValueError):
                    in_tok_cl = None
                try:
                    out_tok_cl = int(or_) if or_ is not None else None
                except (TypeError, ValueError):
                    out_tok_cl = None
            cleaning_cost: Optional[float] = None
            if in_tok_cl is not None and out_tok_cl is not None:
                from ...workflow.helpers import calculate_provider_cost

                cleaning_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="gemini",
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
                        provider="gemini",
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
                logger.warning("Gemini API returned empty cleaned text, using original")
                _emit_cleaning_cost()
                return text
            # ADR-100: cleaning catch-and-degrade; cost emitted in both branches.
            try:
                _guardrails.check_chat_response(
                    cleaned, service="gemini", finish_reason=_gemini_finish_reason(response)
                )
            except _guardrails.GuardrailViolation:
                _emit_cleaning_cost(triggered_guardrail=True)
                logger.warning(
                    "Gemini cleaning output failed guardrail; " "returning original transcript text"
                )
                return text

            _emit_cleaning_cost()
            logger.debug("Gemini cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except _guardrails.GuardrailViolation:
            # Defensive outer catch — preserve the catch-and-degrade contract
            # if a future change adds a guardrail call outside the inline block.
            logger.warning(
                "Gemini cleaning output failed guardrail (outer); "
                "returning original transcript text"
            )
            return text
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
