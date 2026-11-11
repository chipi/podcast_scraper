"""Unified Mistral provider for transcription, speaker detection, and summarization.

This module provides a single MistralProvider class that implements all three protocols:
- TranscriptionProvider (using Mistral Voxtral API)
- SpeakerDetector (using Mistral chat API)
- SummarizationProvider (using Mistral chat API)

This unified approach matches the pattern of OpenAI providers, where a single
provider type handles multiple capabilities using shared API client.

Key advantage: Mistral is the only cloud provider (besides OpenAI) that supports
ALL three capabilities, making it a complete OpenAI alternative.

Note: Uses mistralai Python SDK (not OpenAI SDK).
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from typing import Any, cast, Dict, Optional, Set, TYPE_CHECKING


def _load_mistral_sdk() -> tuple[Any, Any]:
    """Return (Mistral client class, SDKError) for mistralai 2.x or 1.x."""
    try:
        sdk_mod = importlib.import_module("mistralai.client.sdk")
        err_mod = importlib.import_module("mistralai.client.errors")
        return sdk_mod.Mistral, err_mod.SDKError
    except (ImportError, AttributeError):
        pass
    try:
        pkg = importlib.import_module("mistralai")
        return pkg.Mistral, pkg.SDKError
    except (ImportError, AttributeError):
        return None, None


def _mistral_file_class() -> Any:
    """Resolve ``File`` model for mistralai 2.x (client.models) or 1.x (models.file)."""
    for mod_name in ("mistralai.client.models", "mistralai.models.file"):
        try:
            mod = importlib.import_module(mod_name)
            return mod.File
        except (ImportError, AttributeError):
            continue
    raise ImportError("mistralai File model not found")


Mistral, MistralSDKError = _load_mistral_sdk()

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
from ...cleaning.base import TranscriptCleaningProcessor
from ...utils.cleaning_max_tokens import (
    clamp_cleaning_max_tokens,
    estimate_cleaning_output_tokens,
    MISTRAL_CLEANING_MAX_TOKENS,
)
from ...utils.log_redaction import format_exception_for_log
from ...workflow import metrics
from ..extensible import ExtensibleProvider

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
from .. import guardrails as _guardrails, insight_salvage as _insight_salvage
from ..common.transcript_cache import openai_style_messages as _openai_style_messages
from ._mistral_provider_calls import (  # noqa: F401 - re-exported: moved there with S1
    _record_mistral_llm_call,
)
from .mistral_provider_gi_naming import MistralProviderGiNamingMethods  # noqa: F401

# Pricing for Mistral models lives in ``config/pricing_assumptions.yaml`` (#651).


class MistralProvider(ExtensibleProvider):
    """Unified Mistral provider: TranscriptionProvider, SpeakerDetector, SummarizationProvider.

    Uses Voxtral for transcription and chat API for speaker detection and summarization.
    Full OpenAI alternative; all capabilities share the same Mistral client.
    """

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

    @staticmethod
    def _probe_audio_duration_s(audio_path: str) -> float:
        """Best-effort ffprobe duration in seconds. Returns 0.0 on any failure."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",  # tags on stderr need not be UTF-8 (#558)
                timeout=5,
            )
            return float(result.stdout.strip()) if result.returncode == 0 else 0.0
        except (subprocess.SubprocessError, ValueError, OSError):
            return 0.0

    def __init__(self, cfg: config.Config):
        """Initialize unified Mistral provider.

        Args:
            cfg: Configuration object with settings for all three capabilities

        Raises:
            ValueError: If Mistral API key is not provided
            ImportError: If mistralai package is not installed
        """
        if Mistral is None:
            raise ImportError(
                "mistralai package required for Mistral provider. "
                "Install with: pip install -e '.[llm]' (mistralai is in the llm extra)"
            )

        if not cfg.mistral_api_key:
            raise ValueError(
                "Mistral API key required for Mistral provider. "
                "Set MISTRAL_API_KEY environment variable or mistral_api_key in config."
            )

        from ...utils.provider_metadata import validate_api_key_format

        is_valid, _ = validate_api_key_format(
            cfg.mistral_api_key,
            "Mistral",
            expected_prefixes=None,
        )
        if not is_valid:
            # Do not log validation detail: CodeQL taints any message from this API-key path.
            logger.warning(
                "Mistral API key validation failed (missing or too short); "
                "credentials are never logged."
            )

        self.cfg = cfg
        # RFC-115: relocate the transcript to a cacheable leading system block (default on).
        self._cache_transcript_prefix = bool(getattr(cfg, "cache_transcript_prefix", True))

        # Set up transcript cleaning processor based on strategy (Issue #418)
        from ... import stage_hooks

        cleaning_strategy = getattr(cfg, "transcript_cleaning_strategy", "hybrid")
        self.cleaning_processor = stage_hooks.transcript_cleaner(cleaning_strategy)

        # Cleaning model settings (cheaper model for cost efficiency)
        self.cleaning_model = getattr(cfg, "mistral_cleaning_model", "mistral-small")
        self.cleaning_temperature = getattr(cfg, "mistral_cleaning_temperature", 0.2)

        # Suppress verbose Mistral SDK debug logs (if needed)
        # Similar to OpenAI provider pattern
        root_logger = logging.getLogger()
        root_level = root_logger.level if root_logger.level else logging.INFO
        if root_level <= logging.DEBUG:
            mistral_loggers = [
                "mistralai",
                "httpx",
                "httpcore",
            ]
            for logger_name in mistral_loggers:
                mistral_logger = logging.getLogger(logger_name)
                mistral_logger.setLevel(logging.WARNING)

        # Support custom base_url for E2E testing with mock servers
        # Mistral SDK: server_url is the API root; SDK appends /v1/... (do not pass .../v1).
        # 'server' is for named servers only (e.g. "eu") and rejects arbitrary URLs.
        client_kwargs: dict[str, Any] = {"api_key": cfg.mistral_api_key}
        if cfg.mistral_api_base:
            client_kwargs["server_url"] = cfg.mistral_api_base

        # Per-request timeout so a hung Voxtral / large-context call can't block a
        # pipeline worker forever (review 2026-07-17 M12). The mistralai SDK takes
        # ``timeout_ms`` (int milliseconds) in __init__ — the old "SDK does not
        # support timeout" comment was wrong.
        client_kwargs["timeout_ms"] = int(float(getattr(cfg, "mistral_timeout", 600)) * 1000)

        self.client = Mistral(**client_kwargs)

        # Transcription settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.transcription_model = getattr(
            cfg, "mistral_transcription_model", "voxtral-mini-latest"
        )

        # Speaker detection settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.speaker_model = getattr(cfg, "mistral_speaker_model", "mistral-small-latest")
        self.speaker_temperature = getattr(cfg, "mistral_temperature", 0.3)

        # Summarization settings
        # Model validation happens at API call time - invalid models will raise clear errors
        self.summary_model = getattr(cfg, "mistral_summary_model", "mistral-small-latest")
        self.summary_temperature = getattr(cfg, "mistral_temperature", 0.3)
        # Mistral Large supports 256k context window
        # The window this DEPLOYMENT serves (#2050). Mistral Large publishes 256k, but a
        # gateway or a self-hosted stand-in can serve less — and a budget derived from an
        # overstated window is a prompt the server rejects or silently truncates. So the
        # published figure is the FALLBACK, and a profile/StageOption can declare the real one.
        self.max_context_tokens = int(getattr(cfg, "mistral_max_context_tokens", 0) or 0) or 256000

        # Initialization state
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        # API providers handle rate limiting and retries internally via SDK
        # Mistral SDK automatically handles retries with exponential backoff
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
        ext = lookup_external_pricing(table, "mistral", capability, model)
        return dict(ext) if ext else {}

    def initialize(self) -> None:
        """Initialize all Mistral capabilities.

        For Mistral API, initialization is a no-op but we track it for consistency.
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
        logger.debug("Initializing Mistral transcription (model: %s)", self.transcription_model)
        self._transcription_initialized = True
        logger.debug("Mistral transcription initialized successfully")

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing Mistral speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True
        logger.debug("Mistral speaker detection initialized successfully")

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing Mistral summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True
        logger.debug("Mistral summarization initialized successfully")

    # ============================================================================
    # TranscriptionProvider Protocol Implementation
    # ============================================================================

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text using Mistral Voxtral API.

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
                "MistralProvider transcription not initialized. Call initialize() first."
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
            "Transcribing audio file via Mistral Voxtral API: %s (language: %s)",
            audio_path,
            effective_language or "auto",
        )

        try:
            # Mistral Voxtral API uses 'complete' method and requires File object
            # Create File object from file path (Mistral SDK handles file opening)
            from pathlib import Path

            MistralFile = _mistral_file_class()

            # File object requires file_name and binary content
            file_name = Path(audio_path).name
            with open(audio_path, "rb") as audio_file:
                file_content = audio_file.read()
            mistral_file = MistralFile(file_name=file_name, content=file_content)

            # Mistral Voxtral API with retry
            from ...utils.provider_metrics import (
                _safe_mistral_retryable,
                retry_with_metrics,
            )

            # temperature=0.0 is Mistral's documented recommendation for transcription
            # (vs 0.2 for audio chat). Without it, voxtral-mini-latest has been observed
            # to enter runaway repetition loops — its LLM-based decoder has no built-in
            # anti-hallucination filters (unlike Whisper's compression_ratio_threshold,
            # logprob_threshold, no_speech_threshold). See #577 Exp 3 investigation.
            def _make_transcribe_call():
                if effective_language is not None:
                    return self.client.audio.transcriptions.complete(
                        model=self.transcription_model,
                        file=mistral_file,
                        language=effective_language,
                        temperature=0.0,
                    )
                else:
                    return self.client.audio.transcriptions.complete(
                        model=self.transcription_model,
                        file=mistral_file,
                        temperature=0.0,
                    )

            transcription = retry_with_metrics(
                _make_transcribe_call,
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_mistral_retryable(),
            )

            # Extract text from response
            text = transcription.text if hasattr(transcription, "text") else str(transcription)
            if not text:
                logger.warning("Mistral Voxtral API returned empty transcription")
                text = ""

            # Runaway-output sanity check. Natural conversational speech tops out near
            # ~120 chars/s. Anything above 200 chars/s is almost certainly a decoder
            # repetition loop (observed on voxtral-mini-latest for audio >30 min, the
            # model's documented transcription ceiling). Warn but do not raise — caller
            # decides fallback policy. Threshold tuned to catch the 3,400 chars/s loop
            # seen in #577 Exp 3 without tripping on legitimate fast-talking content.
            duration_s = self._probe_audio_duration_s(audio_path)
            if duration_s and duration_s > 0:
                chars_per_s = len(text) / duration_s
                if chars_per_s > 200:
                    logger.warning(
                        "Mistral transcription looks like a hallucination loop: %d chars "
                        "over %.1fs audio = %.0f chars/s (sane ceiling ~200). Model: %s. "
                        "Consider switching to voxtral-mini-transcribe-2 or pre-chunking "
                        "audio to <25 min windows.",
                        len(text),
                        duration_s,
                        chars_per_s,
                        self.transcription_model,
                    )

            logger.debug("Mistral transcription completed: %d characters", len(text))
            return text

        except Exception as exc:
            logger.error("Mistral API error in transcription: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Mistral-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Mistral authentication failed: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Transcription",
                    suggestion="Check your MISTRAL_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Transcription",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral invalid model: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Transcription",
                    suggestion="Check mistral_transcription_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Mistral transcription failed: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Transcription",
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

        Mistral Voxtral may provide segments similar to OpenAI Whisper.

        Args:
            audio_path: Path to audio file
            language: Optional language code
            pipeline_metrics: Optional metrics tracker
            episode_duration_seconds: Optional episode duration

        Returns:
            Tuple of (result_dict, elapsed_time) where result_dict contains:
            - "text": Full transcribed text
            - "segments": List of segment dicts with start/end/text (if available)
        """
        start_time = time.time()
        text = self.transcribe(audio_path, language)
        elapsed = time.time() - start_time

        # Resolve audio_minutes with a file-size fallback so cost isn't silently 0
        # when the caller doesn't pass episode_duration_seconds (#650 Finding 18
        # symmetry — OpenAI fix applies here too). Bitrate-aware: scales by
        # cfg.preprocessing_mp3_bitrate_kbps.
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

        # Compute cost first so the same value flows into both call_metrics
        # (per-episode) and pipeline_metrics (per-stage aggregate).
        tr_cost: Optional[float] = None
        if audio_minutes > 0:
            from ...workflow.helpers import calculate_provider_cost

            tr_cost = calculate_provider_cost(
                cfg=self.cfg,
                provider_type="mistral",
                capability="transcription",
                model=self.transcription_model,
                audio_minutes=audio_minutes,
            )
            if call_metrics is not None:
                from ...utils.provider_metrics import record_provider_call_cost

                record_provider_call_cost(
                    call_metrics,
                    tr_cost,
                    cfg=self.cfg,
                    provider_type="mistral",
                    capability="transcription",
                    model=self.transcription_model,
                    audio_minutes=audio_minutes,
                )

        # Track LLM call metrics if available (#1523: idempotent via the call_metrics latch).
        if pipeline_metrics is not None and audio_minutes > 0:
            from ...utils.provider_metrics import record_transcription_cost_to_pipeline

            record_transcription_cost_to_pipeline(
                pipeline_metrics, call_metrics, audio_minutes, tr_cost
            )

        # Finalize call_metrics (Mistral audio transcription may not have tokens,
        # but finalize for consistency)
        if call_metrics is not None:
            call_metrics.finalize()

        # Mistral Voxtral may not provide segments in the same format as OpenAI
        # Return text with empty segments for now (can be enhanced if Mistral supports it)
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

        For Mistral provider, pattern analysis is not implemented.
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
        """Summarize text using Mistral API.

        Can handle full transcripts directly due to large context window (256k tokens).
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
                "MistralProvider summarization not initialized. Call initialize() first."
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
            "Summarizing text via Mistral API (model: %s, max_tokens: %d)",
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
                _safe_mistral_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("mistral")
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            # Wrap API call with retry tracking
            def _make_api_call():
                return self.client.chat.complete(
                    model=self.summary_model,
                    messages=_openai_style_messages(  # type: ignore[arg-type]  # RFC-115
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
                    retryable_exceptions=_safe_mistral_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            # Mistral SDK response structure: response.choices[0].message.content
            summary = response.choices[0].message.content
            finish_reason = getattr(response.choices[0], "finish_reason", None)

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
                    provider_type="mistral",
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
                    provider_type="mistral",
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                    triggered_guardrail=triggered_guardrail,
                    stage=use_case_stage(use_case),
                )

            # Response-shape guardrail (ADR-100): cost emitted in BOTH branches;
            # raw GuardrailViolation propagates so FallbackAware routes the
            # configured fallback. Mistral uses an OpenAI-compatible response shape.
            try:
                _guardrails.check_chat_response(
                    summary, service="mistral", finish_reason=finish_reason
                )
            except _guardrails.GuardrailViolation:
                _record_cost(triggered_guardrail=True)
                raise

            logger.debug("Mistral summarization completed: %d characters", len(summary))

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
            prompt_metadata = use_case_prompt_metadata("mistral", use_case) or prompt_metadata
            return {
                "summary": summary,
                "summary_short": None,  # Mistral doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "mistral",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except _guardrails.GuardrailViolation:
            # ADR-100: propagate the raw violation so FallbackAware routes.
            raise
        except Exception as exc:
            logger.error("Mistral API error in summarization: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Mistral-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Mistral authentication failed: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Summarization",
                    suggestion="Check your MISTRAL_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral invalid model: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Summarization",
                    suggestion="Check mistral_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Mistral summarization failed: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Summarization",
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
            system, user, system_name, user_name = build_use_case_prompts("mistral", use_case)
            return system, user, system_name, user_name, 0, 0

        system_prompt_name = (
            self.cfg.mistral_summary_system_prompt or "mistral/summarization/system_v1"
        )
        user_prompt_name = self.cfg.mistral_summary_user_prompt

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
            raise RuntimeError("Mistral summarization not initialized for complete_text")

        response = self.client.chat.complete(
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
            provider_type="mistral",
            capability="speaker_detection",
            stage=stage,
            model=self.summary_model,
        )
        raw = response.choices[0].message.content
        content = ((raw if isinstance(raw, str) else "") or "").strip()
        _guardrails.check_chat_response(content, service="mistral", expect_json=True)
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
                "MistralProvider summarization not initialized. Call initialize() first."
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
                _safe_mistral_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.chat.complete(
                    model=model,
                    messages=_openai_style_messages(  # type: ignore[arg-type]  # RFC-115
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
                retryable_exceptions=_safe_mistral_retryable(),
            )
            _record_mistral_llm_call(
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
                provider_type="mistral",
                capability="kg",
                stage="kg",
                model=model,
            )
            raw = response.choices[0].message.content
            content = (raw if isinstance(raw, str) else "") or ""
            return parse_kg_graph_response(
                content.strip(),
                max_topics=max_topics,
                max_entities=max_entities,
            )
        except Exception as e:
            logger.debug("Mistral extract_kg_graph failed: %s", e, exc_info=True)
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
            raise RuntimeError("MistralProvider not initialized. Call initialize() first.")

        from ...prompts.store import render_prompt

        # Build cleaning prompt using prompt_store
        prompt_name = "mistral/cleaning/v1"
        user_prompt = render_prompt(prompt_name, transcript=text)

        # Use system prompt (Mistral pattern)
        system_prompt = (
            "You are a transcript cleaning assistant. "
            "Remove sponsors, ads, intros, outros, and meta-commentary. "
            "Preserve all substantive content and speaker information. "
            "Return only the cleaned text, no explanations."
        )

        logger.debug(
            "Cleaning transcript via Mistral API (model: %s, text length: %d chars)",
            self.cleaning_model,
            len(text),
        )

        try:
            # Track retries and rate limits
            from ...utils.provider_metrics import (
                _safe_mistral_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("mistral")
            call_metrics.set_breaker_config_from_cfg(self.cfg)

            # Wrap API call with retry tracking
            def _make_api_call():
                return self.client.chat.complete(
                    model=self.cleaning_model,
                    messages=[  # type: ignore[arg-type]
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.cleaning_temperature,
                    max_tokens=clamp_cleaning_max_tokens(
                        estimate_cleaning_output_tokens(len(text.split())),
                        MISTRAL_CLEANING_MAX_TOKENS,
                    ),
                )

            try:
                response = retry_with_metrics(
                    _make_api_call,
                    max_retries=3,
                    initial_delay=1.0,
                    max_delay=30.0,
                    retryable_exceptions=_safe_mistral_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            _record_mistral_llm_call(
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
                    provider_type="mistral",
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
                        provider="mistral",
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
                logger.warning("Mistral API returned empty cleaned text, using original")
                _emit_cleaning_cost()
                return text
            # ADR-100: cleaning catch-and-degrade; cost emitted in both branches.
            try:
                _guardrails.check_chat_response(
                    cleaned,
                    service="mistral",
                    finish_reason=getattr(response.choices[0], "finish_reason", None),
                )
            except _guardrails.GuardrailViolation:
                _emit_cleaning_cost(triggered_guardrail=True)
                logger.warning(
                    "Mistral cleaning output failed guardrail; "
                    "returning original transcript text"
                )
                return text

            _emit_cleaning_cost()
            logger.debug("Mistral cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except _guardrails.GuardrailViolation:
            # Defensive outer catch — preserve contract.
            logger.warning(
                "Mistral cleaning output failed guardrail (outer); "
                "returning original transcript text"
            )
            return text
        except Exception as exc:
            logger.error("Mistral API error in cleaning: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle Mistral-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Mistral authentication failed: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Cleaning",
                    suggestion="Check your MISTRAL_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Cleaning",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Mistral cleaning failed: {format_exception_for_log(exc)}",
                    provider="MistralProvider/Cleaning",
                ) from exc

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
            ProviderCapabilities object describing Mistral provider capabilities
        """
        from ..capabilities import ProviderCapabilities  # noqa: PLC0415

        return ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_semantic_cleaning=True,  # Mistral supports LLM-based cleaning
            supports_audio_input=True,  # Mistral Voxtral accepts audio files
            supports_json_mode=True,  # Mistral supports JSON mode
            max_context_tokens=self.max_context_tokens,
            supports_tool_calls=True,  # Mistral supports function calling
            supports_system_prompt=True,  # Mistral supports system prompts
            supports_streaming=True,  # Mistral API supports streaming
            provider_name="mistral",
            supports_gi_segment_timing=False,
        )
