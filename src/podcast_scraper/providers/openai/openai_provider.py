"""Unified OpenAI provider for transcription, speaker detection, and summarization.

This module provides a single OpenAIProvider class that implements all three protocols:
- TranscriptionProvider (using Whisper API)
- SpeakerDetector (using GPT API)
- SummarizationProvider (using GPT API)

This unified approach matches the pattern of ML providers, where a single
provider type handles multiple capabilities using shared API client.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from ... import config, models

if TYPE_CHECKING:
    from ...models import Episode
else:
    Episode = models.Episode  # type: ignore[assignment]
from ...cleaning import PatternBasedCleaner
from ...cleaning.base import TranscriptCleaningProcessor
from ...utils.cleaning_max_tokens import (
    clamp_cleaning_max_tokens,
    estimate_cleaning_output_tokens,
    OPENAI_CLEANING_MAX_TOKENS,
)
from ...utils.log_redaction import format_exception_for_log
from ...utils.provider_metadata import (
    extract_region_from_endpoint,
    log_provider_metadata,
    validate_api_key_format,
    warn_if_truncated,
)
from ...utils.timeout_config import get_openai_client_timeout
from ...workflow import metrics
from ..capabilities import ProviderCapabilities

logger = logging.getLogger(__name__)


def _openai_chat_usage_tokens(response: Any) -> Tuple[Optional[int], Optional[int]]:
    """Best-effort prompt/completion token counts from a chat completion response."""
    if not hasattr(response, "usage") or not response.usage:
        return None, None
    prompt_tokens = getattr(response.usage, "prompt_tokens", None)
    completion_tokens = getattr(response.usage, "completion_tokens", None)
    input_tokens = int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else None
    output_tokens = int(completion_tokens) if isinstance(completion_tokens, (int, float)) else None
    return input_tokens, output_tokens


# Protocol types imported for type hints (used in docstrings and type annotations)
# from ..speaker_detectors.base import SpeakerDetector  # noqa: F401
# from ..summarization.base import SummarizationProvider  # noqa: F401
# from ..transcription.base import TranscriptionProvider  # noqa: F401

# Use canonical default from speaker_detection (Issue #428: typed placeholder, not "Guest")
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES

# Pricing for OpenAI models lives in ``config/pricing_assumptions.yaml`` (#651).
# See ``get_pricing()`` below — it delegates to the YAML loader so there is a
# single source of truth. CI guard
# (``scripts/validate/check_profile_pricing_coverage.py``) fails PRs that
# reference a model without a matching YAML row.


def _record_openai_summarization_call(
    response: Any,
    pipeline_metrics: Optional[Any],
    *,
    cfg: Any,
    model: str,
) -> None:
    """Record one OpenAI summarization LLM call into pipeline_metrics.

    Used by bundle-mode methods (summarize_bundled, summarize_mega_bundled,
    summarize_extraction_bundled) — these make one summary-pricing LLM call
    and historically skipped ``record_llm_summarization_call`` entirely, so
    cost/tokens vanished at pipeline level even though call_metrics saw them.
    """
    if pipeline_metrics is None or not hasattr(pipeline_metrics, "record_llm_summarization_call"):
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    in_raw = getattr(usage, "prompt_tokens", None)
    out_raw = getattr(usage, "completion_tokens", None)
    if not isinstance(in_raw, (int, float)) or not isinstance(out_raw, (int, float)):
        return
    in_tok = int(in_raw)
    out_tok = int(out_raw)
    from ...workflow.helpers import calculate_provider_cost

    cost = calculate_provider_cost(
        cfg=cfg,
        provider_type="openai",
        capability="summarization",
        model=model,
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
    )
    pipeline_metrics.record_llm_summarization_call(in_tok, out_tok, cost_usd=cost)


class OpenAIProvider:
    """Unified OpenAI provider implementing TranscriptionProvider, SpeakerDetector, and SummarizationProvider.

    This provider initializes and manages:
    - OpenAI Whisper API for transcription
    - OpenAI GPT API for speaker detection
    - OpenAI GPT API for summarization

    All three capabilities share the same OpenAI client, similar to how ML providers
    share the same ML libraries. The client is initialized once and reused.
    """  # noqa: E501

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

    def __init__(self, cfg: config.Config):
        """Initialize unified OpenAI provider.

        Args:
            cfg: Configuration object with settings for all three capabilities

        Raises:
            ValueError: If OpenAI API key is not provided
            ImportError: If openai package is not installed (core dependency)
        """
        # Lazy import to allow unit tests without openai installed (Issue #405)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI provider. "
                "Install the project (OpenAI SDK is a core dependency), e.g. pip install -e ."
            ) from exc

        if not cfg.openai_api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI provider. "
                "Set OPENAI_API_KEY environment variable or openai_api_key in config."
            )

        # Validate API key format
        is_valid, _ = validate_api_key_format(
            cfg.openai_api_key,
            "OpenAI",
            expected_prefixes=["sk-", "sk-proj-"],
        )
        if not is_valid:
            # Do not log validation detail: CodeQL taints any message from this API-key path.
            logger.warning(
                "OpenAI API key validation failed (missing, too short, or wrong prefix); "
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

        # Cleaning model settings (config default: gpt-4o-mini; override for cheaper tiers)
        self.cleaning_model = getattr(
            cfg, "openai_cleaning_model", config.PROD_DEFAULT_OPENAI_CLEANING_MODEL
        )
        self.cleaning_temperature = getattr(cfg, "openai_cleaning_temperature", 0.2)

        # Suppress verbose OpenAI SDK debug logs (they're too long and clutter the output)
        # Set OpenAI SDK loggers to WARNING level when root logger is DEBUG
        # This keeps our debug logs visible while hiding OpenAI's verbose request/response logs
        root_logger = logging.getLogger()
        root_level = root_logger.level if root_logger.level else logging.INFO
        if root_level <= logging.DEBUG:
            openai_loggers = [
                "openai",
                "openai._base_client",
                "openai.api_resources",
                "httpx",
                "httpcore",
                "httpcore.connection",
                "httpcore.http11",
            ]
            for logger_name in openai_loggers:
                openai_logger = logging.getLogger(logger_name)
                openai_logger.setLevel(logging.WARNING)

        # Support custom base_url for E2E testing with mock servers
        client_kwargs: dict[str, Any] = {"api_key": cfg.openai_api_key}
        if cfg.openai_api_base:
            client_kwargs["base_url"] = cfg.openai_api_base

        # Read timeout must cover Whisper and long chat calls (cleaning, summarize);
        # cfg.timeout alone is often too low (RSS/download tuning).
        client_kwargs["timeout"] = get_openai_client_timeout(cfg)

        self.client = OpenAI(**client_kwargs)

        # Log non-sensitive provider metadata (for debugging)
        # Extract region from base_url if possible
        region = extract_region_from_endpoint(cfg.openai_api_base)
        log_provider_metadata(
            provider_name="OpenAI",
            organization=getattr(cfg, "openai_organization", None),
            base_url=cfg.openai_api_base,
            region=region,
        )

        # Transcription settings
        self.transcription_model = getattr(cfg, "openai_transcription_model", "whisper-1")

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "openai_speaker_model", "gpt-4o-mini")
        self.speaker_temperature = getattr(cfg, "openai_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "openai_summary_model", "gpt-4o-mini")
        _insight_override = getattr(cfg, "openai_insight_model", None)
        if isinstance(_insight_override, str) and _insight_override.strip():
            self.insight_model = _insight_override.strip()
        else:
            self.insight_model = self.summary_model
        self.summary_temperature = getattr(cfg, "openai_temperature", 0.3)
        _seed = getattr(cfg, "openai_summary_seed", None)
        self.summary_seed: Optional[int] = int(_seed) if _seed is not None else None
        # GPT-4o-mini supports 128k context window - can handle full transcripts
        self.max_context_tokens = 128000  # Conservative estimate

        # Initialization state
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

        # Mark provider as thread-safe (API clients can be shared across threads)
        # API providers handle rate limiting internally, so parallelism isn't needed
        self._requires_separate_instances = False

    @staticmethod
    def get_pricing(model: str, capability: str) -> Dict[str, float]:
        """Read pricing from ``config/pricing_assumptions.yaml`` (#651).

        YAML is the single source of truth; this thin wrapper is kept for API
        stability (test fixtures, tooling). Production cost calc goes through
        :func:`podcast_scraper.workflow.helpers._get_provider_pricing`.

        Returns the rate dict for ``(provider=openai, capability, model)`` or
        ``{}`` when no matching row exists.
        """
        from podcast_scraper.pricing_assumptions import (
            get_loaded_table,
            lookup_external_pricing,
        )

        table, _ = get_loaded_table("config/pricing_assumptions.yaml")
        if not table:
            return {}
        ext = lookup_external_pricing(table, "openai", capability, model)
        return dict(ext) if ext else {}

    def initialize(self) -> None:
        """Initialize all OpenAI capabilities.

        For OpenAI API, initialization is a no-op but we track it for consistency.
        This method is idempotent and can be called multiple times safely.

        Note:
            All capabilities share the same OpenAI client, so initialization
            is lightweight and doesn't require separate setup for each capability.
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
        logger.debug("Initializing OpenAI transcription (model: %s)", self.transcription_model)
        self._transcription_initialized = True
        logger.debug("OpenAI transcription initialized successfully")

    def _initialize_speaker_detection(self) -> None:
        """Initialize speaker detection capability."""
        logger.debug("Initializing OpenAI speaker detection (model: %s)", self.speaker_model)
        self._speaker_detection_initialized = True
        logger.debug("OpenAI speaker detection initialized successfully")

    def _initialize_summarization(self) -> None:
        """Initialize summarization capability."""
        logger.debug("Initializing OpenAI summarization (model: %s)", self.summary_model)
        self._summarization_initialized = True
        logger.debug("OpenAI summarization initialized successfully")

    # ============================================================================
    # TranscriptionProvider Protocol Implementation
    # ============================================================================

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """Transcribe audio file to text using OpenAI Whisper API.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If provided (not None), uses that language.
                     If not provided (default None), uses cfg.language if available.
                     If explicitly passed as None, auto-detects (ignores cfg.language).
                     Note: To explicitly request auto-detect, pass None. To use cfg.language,
                     omit the argument.

        Returns:
            Transcribed text as string

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If transcription fails or API key is invalid
            RuntimeError: If provider is not initialized
        """
        if not self._transcription_initialized:
            raise RuntimeError(
                "OpenAIProvider transcription not initialized. Call initialize() first."
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided language or fall back to config
        # We can't perfectly distinguish "not provided" from "explicitly None" with default args
        # So we use a simple rule: if language is provided (not None), use it.
        # If language is None, use cfg.language if available, otherwise auto-detect.
        # Note: To explicitly request auto-detect when cfg.language is set, the caller
        # would need to pass a special value, but for backward compatibility we use None.
        # The test case where language=None is explicitly passed and cfg.language is set
        # is ambiguous - we choose to use cfg.language in this case for consistency.
        if language is not None:
            effective_language = language
        elif hasattr(self.cfg, "language") and self.cfg.language is not None:
            # Use config language when parameter is None (default or explicitly passed)
            effective_language = self.cfg.language
        else:
            # No language specified - auto-detect
            effective_language = None

        logger.debug(
            "Transcribing audio file via OpenAI API: %s (language: %s)",
            audio_path,
            effective_language or "auto",
        )

        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_transcribe_call():
                with open(audio_path, "rb") as audio_file:
                    if effective_language is not None:
                        return self.client.audio.transcriptions.create(
                            model=self.transcription_model,
                            file=audio_file,
                            language=effective_language,
                            response_format="verbose_json",
                        )
                    else:
                        return self.client.audio.transcriptions.create(
                            model=self.transcription_model,
                            file=audio_file,
                            response_format="verbose_json",
                        )

            transcript = retry_with_metrics(
                _make_transcribe_call,
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
            )

            # verbose_json returns a Transcription object with `.text` (and `.duration`,
            # `.segments`, etc.). We return just the text here; cost tracking in
            # transcribe_with_segments uses the full response.
            if hasattr(transcript, "text"):
                text_attr = getattr(transcript, "text", None)
                text = text_attr if isinstance(text_attr, str) else ""
            elif isinstance(transcript, dict):
                text = str(transcript.get("text", ""))
            elif isinstance(transcript, str):
                text = transcript
            else:
                text = str(transcript)

            logger.debug(
                "OpenAI transcription completed: %d characters",
                len(text) if text else 0,
            )

            return text

        except Exception as exc:
            logger.error("OpenAI Whisper API error: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderRuntimeError

            raise ProviderRuntimeError(
                message=f"OpenAI transcription failed: {format_exception_for_log(exc)}",
                provider="OpenAIProvider/Transcription",
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

        Returns the complete OpenAI transcription result including segments
        and timestamps for screenplay formatting.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., "en", "fr").
                     If None, uses cfg.language or auto-detects

        Returns:
            Tuple of (result_dict, elapsed_time) where result_dict contains:
            - "text": Full transcribed text
            - "segments": List of segment dicts with start, end, text
            - Other OpenAI API metadata
        """
        if not self._transcription_initialized:
            raise RuntimeError(
                "OpenAIProvider transcription not initialized. Call initialize() first."
            )

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use provided language or fall back to config
        effective_language = language if language is not None else (self.cfg.language or None)

        logger.debug(
            "Transcribing audio file with segments via OpenAI API: %s (language: %s)",
            audio_path,
            effective_language or "auto",
        )

        start_time = time.time()
        try:
            # Track retries and rate limits
            from ...utils.provider_metrics import ProviderCallMetrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("openai")

            # Wrap API call with retry tracking
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                with open(audio_path, "rb") as audio_file:
                    # Use verbose_json format to get segments
                    if effective_language is not None:
                        return self.client.audio.transcriptions.create(
                            model=self.transcription_model,
                            file=audio_file,
                            language=effective_language,
                            response_format="verbose_json",  # Get full response with segments
                        )
                    else:
                        return self.client.audio.transcriptions.create(
                            model=self.transcription_model,
                            file=audio_file,
                            response_format="verbose_json",  # Get full response with segments
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

            elapsed = time.time() - start_time
            call_metrics.finalize()

            # Extract token counts (OpenAI Whisper API doesn't return tokens,
            # but track audio minutes)
            # Get audio duration from episode metadata, response, or estimate from file size
            audio_minutes = 0.0
            if episode_duration_seconds is not None:
                # Handle Mock objects from tests by checking type
                if isinstance(episode_duration_seconds, (int, float)):
                    audio_minutes = float(episode_duration_seconds) / 60.0
            elif hasattr(response, "duration") and response.duration:
                duration_val = getattr(response, "duration", None)
                if isinstance(duration_val, (int, float)):
                    audio_minutes = float(duration_val) / 60.0
            else:
                # Fallback only fires when neither episode_duration_seconds nor
                # response.duration is available (rare with verbose_json). Scale
                # by the configured bitrate: at 128 kbps ≈ 1 MB/min, but low-bitrate
                # preprocessing (speech_optimal_v1 uses 32 kbps → ~0.25 MB/min) would
                # under-report ~4× if we assumed 128. Default to 128 when unset
                # (matches pre-preprocessing podcast MP3s).
                try:
                    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                    bitrate_kbps_cfg = getattr(self.cfg, "preprocessing_mp3_bitrate_kbps", None)
                    bitrate_kbps = float(bitrate_kbps_cfg) if bitrate_kbps_cfg else 128.0
                    audio_minutes = file_size_mb * (128.0 / bitrate_kbps)
                except OSError:
                    pass

            # Calculate cost for transcription (per minute pricing)
            if audio_minutes > 0:
                # Compute cost first so the same value can flow into both
                # call_metrics (per-episode) and pipeline_metrics (per-stage).
                from ...workflow.helpers import calculate_provider_cost

                cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="openai",
                    capability="transcription",
                    model=self.transcription_model,
                    audio_minutes=audio_minutes,
                )
                call_metrics.set_cost(cost)
                if pipeline_metrics is not None:
                    pipeline_metrics.record_llm_transcription_call(audio_minutes, cost_usd=cost)

            # OpenAI API returns a Transcription object with text and segments
            # when verbose_json is used. Convert to dict format matching Whisper output.
            # Response structure: response.text (str), response.segments (List[Segment])
            # Each Segment has: start (float), end (float), text (str)
            segments = []
            if hasattr(response, "segments") and response.segments:
                for seg in response.segments:
                    # Handle both dict-like and object-like segment structures
                    if isinstance(seg, dict):
                        segments.append(
                            {
                                "start": float(seg.get("start", 0.0)),
                                "end": float(seg.get("end", 0.0)),
                                "text": seg.get("text", ""),
                            }
                        )
                    else:
                        # Object-like structure (OpenAI SDK TranscriptionSegment)
                        segments.append(
                            {
                                "start": float(getattr(seg, "start", 0.0)),
                                "end": float(getattr(seg, "end", 0.0)),
                                "text": getattr(seg, "text", ""),
                            }
                        )

            result_dict: dict[str, object] = {
                "text": response.text if hasattr(response, "text") else str(response),
                "segments": segments,
            }

            segments_list = result_dict.get("segments", [])
            segment_count = len(segments_list) if isinstance(segments_list, list) else 0
            logger.debug(
                "OpenAI transcription with segments completed in %.2fs (%d segments)",
                elapsed,
                segment_count,
            )

            return result_dict, elapsed

        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error("OpenAI Whisper API error: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle OpenAI-specific error types
            error_msg = str(exc).lower()
            exc_type_name = type(exc).__name__
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"OpenAI authentication failed: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Transcription",
                    suggestion=(
                        "Check your OPENAI_API_KEY environment variable or config setting. "
                        "Verify the key is valid and has not expired."
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"OpenAI rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Transcription",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"OpenAI transcription failed: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Transcription",
                ) from exc

    # ============================================================================
    # SpeakerDetector Protocol Implementation
    # ============================================================================

    def detect_hosts(
        self,
        feed_title: str | None,
        feed_description: str | None,
        feed_authors: list[str] | None = None,
    ) -> Set[str]:
        """Detect host names from feed-level metadata using OpenAI API.

        Args:
            feed_title: Feed title
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "OpenAIProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use OpenAI API to detect hosts from feed metadata
        # This is a simplified version - could be enhanced with dedicated prompt
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
        """Detect speaker names from episode metadata using OpenAI API.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names (for context)

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
                "OpenAIProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via OpenAI API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = self.cfg.openai_speaker_system_prompt or "openai/ner/system_ner_v1"
            system_prompt = render_prompt(system_prompt_name)

            # Call OpenAI API with retry
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            response = retry_with_metrics(
                lambda: self.client.chat.completions.create(
                    model=self.speaker_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.speaker_temperature,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                ),
                max_retries=2,
                initial_delay=1.0,
                max_delay=30.0,
                retryable_exceptions=_safe_openai_retryable(),
            )

            response_text = response.choices[0].message.content
            if not response_text:
                logger.warning("OpenAI API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "OpenAI speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available
            if pipeline_metrics is not None and hasattr(response, "usage"):
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                # Compute cost for the aggregate so metrics.json reflects speaker
                # detection spend (previously silently 0 in total_stage_cost_usd).
                sd_cost: Optional[float] = None
                if input_tokens > 0 or output_tokens > 0:
                    from ...workflow.helpers import calculate_provider_cost

                    sd_cost = calculate_provider_cost(
                        cfg=self.cfg,
                        provider_type="openai",
                        capability="speaker_detection",
                        model=self.speaker_model,
                        prompt_tokens=int(input_tokens),
                        completion_tokens=int(output_tokens),
                    )
                pipeline_metrics.record_llm_speaker_detection_call(
                    input_tokens, output_tokens, cost_usd=sd_cost
                )

            return speakers, detected_hosts, success, False

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse OpenAI API JSON response: %s", format_exception_for_log(exc)
            )
            logger.debug(
                "Response text: %s", response_text if "response_text" in locals() else "N/A"
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True
        except Exception as exc:
            logger.error("OpenAI API error in speaker detection: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle OpenAI-specific error types
            error_msg = str(exc).lower()
            exc_type_name = type(exc).__name__
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"OpenAI authentication failed: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/SpeakerDetection",
                    suggestion=(
                        "Check your OPENAI_API_KEY environment variable or config setting. "
                        "Verify the key is valid and has not expired."
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"OpenAI rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"OpenAI speaker detection failed: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/SpeakerDetection",
                ) from exc

    def analyze_patterns(
        self,
        episodes: list[Episode],  # type: ignore[valid-type]
        known_hosts: Set[str],
    ) -> dict[str, object] | None:
        """Analyze patterns across multiple episodes (optional).

        For OpenAI provider, pattern analysis is not implemented.
        Returns None to use local pattern analysis logic.

        Args:
            episodes: List of episodes to analyze
            known_hosts: Set of known host names

        Returns:
            None (uses local pattern analysis)
        """
        # OpenAI provider doesn't implement pattern analysis
        # Return None to use local logic
        return None

    def _build_speaker_detection_prompt(
        self,
        episode_title: str,
        episode_description: str | None,
        known_hosts: Set[str],
    ) -> str:
        """Build prompt for speaker detection using prompt_store.

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names

        Returns:
            Rendered prompt string
        """
        from ...prompts.store import render_prompt

        # Use prompt_store to load versioned prompt template
        prompt_name = self.cfg.openai_speaker_user_prompt

        # Merge config params with template params
        template_params = {
            "episode_title": episode_title,
            "episode_description": episode_description or "",
            "known_hosts": ", ".join(sorted(known_hosts)) if known_hosts else "",
        }
        template_params.update(self.cfg.ner_prompt_params)

        return render_prompt(prompt_name, **template_params)

    def _parse_speakers_from_response(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Parse speaker names from OpenAI API JSON response.

        Args:
            response_text: JSON response from OpenAI API
            known_hosts: Set of known host names (for filtering)

        Returns:
            Tuple of (speaker_names_list, detected_hosts_set, detection_succeeded)
        """
        try:
            data = json.loads(response_text)

            # Extract speakers, hosts, and guests from JSON
            all_speakers = data.get("speakers", [])
            detected_hosts_list = data.get("hosts", [])
            guests_list = data.get("guests", [])

            # Normalize names (strip whitespace, filter empty)
            all_speakers = [name.strip() for name in all_speakers if name.strip()]
            detected_hosts_list = [name.strip() for name in detected_hosts_list if name.strip()]
            guests_list = [name.strip() for name in guests_list if name.strip()]

            # Filter hosts to only include those in known_hosts
            detected_hosts = {host for host in detected_hosts_list if host in known_hosts}

            # Build speaker names list: hosts first, then guests
            speaker_names = list(detected_hosts) + guests_list

            # Ensure we have at least MIN_SPEAKERS_REQUIRED speakers
            min_speakers = getattr(self.cfg, "screenplay_num_speakers", 2)
            if len(speaker_names) < min_speakers:
                # Add default speakers if needed
                defaults_needed = min_speakers - len(speaker_names)
                speaker_names.extend(DEFAULT_SPEAKER_NAMES[:defaults_needed])

            # Detection succeeded if we have real names (not just defaults)
            detection_succeeded = bool(detected_hosts or guests_list or (len(all_speakers) > 0))

            return speaker_names[:min_speakers], detected_hosts, detection_succeeded

        except (json.JSONDecodeError, KeyError, AttributeError) as exc:
            logger.warning(
                "Failed to parse OpenAI response as JSON: %s", format_exception_for_log(exc)
            )
            logger.debug("Response text: %s", response_text[:200])
            # Fallback: try to extract names from text response
            return self._parse_speakers_from_text(response_text, known_hosts)

    def _parse_speakers_from_text(
        self, response_text: str, known_hosts: Set[str]
    ) -> Tuple[list[str], Set[str], bool]:
        """Fallback: Parse speaker names from text response (not JSON).

        Args:
            response_text: Text response from OpenAI API
            known_hosts: Set of known host names

        Returns:
            Tuple of (speaker_names_list, detected_hosts_set, detection_succeeded)
        """
        # Try to extract names using simple patterns
        # Look for common patterns like "Speakers: Name1, Name2" or "Hosts: Name1"
        names = set()

        # Pattern: "speakers": ["Name1", "Name2"]
        json_pattern = r'"speakers"\s*:\s*\[(.*?)\]'
        match = re.search(json_pattern, response_text, re.IGNORECASE)
        if match:
            names_str = match.group(1)
            # Extract quoted strings
            quoted_names = re.findall(r'"([^"]+)"', names_str)
            names.update(quoted_names)

        # Pattern: "Hosts: Name1, Name2"
        hosts_pattern = r"hosts?\s*:\s*([^\n]+)"
        match = re.search(hosts_pattern, response_text, re.IGNORECASE)
        if match:
            hosts_str = match.group(1)
            host_names = [n.strip() for n in re.split(r"[,;]", hosts_str)]
            names.update(host_names)

        # Pattern: "Guests: Name1, Name2"
        guests_pattern = r"guests?\s*:\s*([^\n]+)"
        match = re.search(guests_pattern, response_text, re.IGNORECASE)
        if match:
            guests_str = match.group(1)
            guest_names = [n.strip() for n in re.split(r"[,;]", guests_str)]
            names.update(guest_names)

        # Filter out generic names
        names = {n for n in names if n.lower() not in ("host", "guest", "speaker")}

        # Separate hosts and guests
        detected_hosts = {name for name in names if name in known_hosts}
        guests = [name for name in names if name not in known_hosts]

        # Build speaker names list
        speaker_names = list(detected_hosts) + guests
        min_speakers = getattr(self.cfg, "screenplay_num_speakers", 2)
        if len(speaker_names) < min_speakers:
            speaker_names.extend(DEFAULT_SPEAKER_NAMES[: min_speakers - len(speaker_names)])

        detection_succeeded = bool(detected_hosts or guests)

        return speaker_names[:min_speakers], detected_hosts, detection_succeeded

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
        """Summarize text using OpenAI GPT API.

        Can handle full transcripts directly due to large context window (128k+ tokens).
        No chunking needed for most podcast transcripts.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            params: Optional parameters dict with:
                - max_length: Maximum summary length in tokens
                    (default from summary_reduce_params.max_new_tokens)
                - min_length: Minimum summary length in tokens
                    (default from config)
                - prompt: Optional custom prompt (overrides default)

        Returns:
            Dictionary with summary results:
            {
                "summary": str,
                "summary_short": Optional[str],
                "metadata": {
                    "model": str,
                    "provider": "openai",
                    ...
                }
            }

        Raises:
            ValueError: If summarization fails
            RuntimeError: If provider is not initialized
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        # Extract parameters with defaults from config
        # OpenAI doesn't have map/reduce, so use reduce params for final summary
        max_length = (
            (params.get("max_length") if params else None)
            or self.cfg.summary_reduce_params.get("max_new_tokens")
            or 800
        )
        # Enforce cloud-LLM structured-JSON output floor (Flightcast 2026-04-20).
        from ..common.output_tokens import cloud_structured_max_output_tokens

        max_length = cloud_structured_max_output_tokens(self.cfg, max_length)
        min_length = (
            (params.get("min_length") if params else None)
            or self.cfg.summary_reduce_params.get("min_new_tokens")
            or 100
        )
        custom_prompt = params.get("prompt") if params else None

        logger.debug(
            "Summarizing text via OpenAI API (model: %s, max_tokens: %d)",
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
                text, episode_title, episode_description, max_length, min_length, custom_prompt
            )

            # Track retries and rate limits
            from ...utils.provider_metrics import ProviderCallMetrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("openai")

            # Wrap API call with retry tracking
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                retry_with_metrics,
            )

            # Newer OpenAI models (o1/o3/gpt-5 series) require max_completion_tokens.
            _uses_completion_tokens = self.summary_model.startswith(("o1", "o3", "gpt-5"))
            _token_kwarg = (
                {"max_completion_tokens": max_length}
                if _uses_completion_tokens
                else {"max_tokens": max_length}
            )

            def _make_api_call():
                kwargs: Dict[str, Any] = {
                    "model": self.summary_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": self.summary_temperature,
                    **_token_kwarg,
                }
                if self.summary_seed is not None:
                    kwargs["seed"] = self.summary_seed
                return self.client.chat.completions.create(**kwargs)

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
                "openai",
                "summarize",
            )

            summary = response.choices[0].message.content
            if not summary:
                logger.warning("OpenAI API returned empty summary")
                summary = ""

            logger.debug(
                "OpenAI summarization completed: %d characters",
                len(summary),
            )

            # Extract token counts and populate call_metrics
            input_tokens = None
            output_tokens = None
            if hasattr(response, "usage") and response.usage:
                prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                completion_tokens = getattr(response.usage, "completion_tokens", None)
                # Convert to int if they're actual numbers, otherwise use 0
                # Handle Mock objects from tests by checking type
                input_tokens = int(prompt_tokens) if isinstance(prompt_tokens, (int, float)) else 0
                output_tokens = (
                    int(completion_tokens) if isinstance(completion_tokens, (int, float)) else 0
                )
                if input_tokens > 0 or output_tokens > 0:
                    call_metrics.set_tokens(input_tokens, output_tokens)

            # Calculate cost first so the value flows into both call_metrics and
            # pipeline_metrics.record_llm_summarization_call(cost_usd=...).
            cost: Optional[float] = None
            if input_tokens is not None:
                from ...workflow.helpers import calculate_provider_cost

                cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="openai",
                    capability="summarization",
                    model=self.summary_model,
                    prompt_tokens=input_tokens,
                    completion_tokens=output_tokens,
                )
                call_metrics.set_cost(cost)

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
                "transcript": (
                    text[:100] + "..." if len(text) > 100 else text
                ),  # Truncate for metadata
                "title": episode_title or "",
                "paragraphs_min": paragraphs_min,
                "paragraphs_max": paragraphs_max,
            }
            user_params.update(self.cfg.summary_prompt_params)
            prompt_metadata["user"] = get_prompt_metadata(user_prompt_name, params=user_params)

            return {
                "summary": summary,
                # OpenAI provider doesn't generate short summaries separately
                "summary_short": None,
                "metadata": {
                    "model": self.summary_model,
                    "provider": "openai",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

        except Exception as exc:
            logger.error("OpenAI API error in summarization: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle OpenAI-specific error types
            error_msg = str(exc).lower()
            exc_type_name = type(exc).__name__
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"OpenAI authentication failed: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Summarization",
                    suggestion=(
                        "Check your OPENAI_API_KEY environment variable or config setting. "
                        "Verify the key is valid and has not expired."
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"OpenAI rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"OpenAI summarization failed: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Summarization",
                ) from exc

    def summarize_mega_bundled(
        self,
        text: str,
        *,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: "metrics.Metrics | None" = None,
        call_metrics: Any | None = None,
    ) -> Any:
        """Single-call mega-bundle: summary + bullets + insights + topics + entities (#643).

        #632 research flagged OpenAI as "not tier-1" because KG/entity quality
        regressed in mega-bundle mode on fixture transcripts. #646 real-episode
        validation revisits this claim. See
        ``docs/guides/AI_PROVIDER_COMPARISON_GUIDE.md`` for the tier table.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        from ...prompting.megabundle import build_megabundle_prompt
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )
        from ..common.megabundle_parser import parse_megabundle_response

        max_out = int(
            (params or {}).get("max_tokens")
            or getattr(self.cfg, "llm_bundled_max_output_tokens", 16384)
            or 16384
        )
        language = getattr(self.cfg, "language", "en") or None
        system_prompt, user_prompt = build_megabundle_prompt(text, language=language)

        _uses_completion_tokens = self.summary_model.startswith(("o1", "o3", "gpt-5"))
        _token_kwarg: Dict[str, Any] = (
            {"max_completion_tokens": max_out}
            if _uses_completion_tokens
            else {"max_tokens": max_out}
        )

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("openai")

        def _make_api_call() -> Any:
            kwargs: Dict[str, Any] = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                **_token_kwarg,
            }
            if not _uses_completion_tokens:
                kwargs["response_format"] = {"type": "json_object"}
            if self.summary_seed is not None:
                kwargs["seed"] = self.summary_seed
            return self.client.chat.completions.create(**kwargs)

        try:
            resp = retry_with_metrics(
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

        raw_text = (resp.choices[0].message.content or "").strip() or "{}"
        if hasattr(resp, "usage") and resp.usage is not None:
            try:
                call_metrics.set_tokens(
                    int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    int(getattr(resp.usage, "completion_tokens", 0) or 0),
                )
            except (TypeError, ValueError):
                pass

        _record_openai_summarization_call(
            resp, pipeline_metrics, cfg=self.cfg, model=self.summary_model
        )
        return parse_megabundle_response(raw_text)

    def summarize_extraction_bundled(
        self,
        text: str,
        *,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: "metrics.Metrics | None" = None,
        call_metrics: Any | None = None,
    ) -> Any:
        """Single-call extraction bundle: insights + topics + entities (#643).

        Companion to :meth:`summarize_bundled` for the 2-call pipeline:
        bundled summary/bullets + bundled extraction. Omits summary/title/bullets
        from the prompt so the model can spend its budget on KG quality.

        Returns:
            :class:`MegaBundleResult` with empty title/summary/bullets and
            populated insights/topics/entities.
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        from ...prompting.megabundle import build_extraction_bundle_prompt
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )
        from ..common.megabundle_parser import parse_extraction_bundle_response

        max_out = int(
            (params or {}).get("max_tokens")
            or getattr(self.cfg, "llm_bundled_max_output_tokens", 16384)
            or 16384
        )
        language = getattr(self.cfg, "language", "en") or None
        system_prompt, user_prompt = build_extraction_bundle_prompt(text, language=language)

        _uses_completion_tokens = self.summary_model.startswith(("o1", "o3", "gpt-5"))
        _token_kwarg: Dict[str, Any] = (
            {"max_completion_tokens": max_out}
            if _uses_completion_tokens
            else {"max_tokens": max_out}
        )

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("openai")

        def _make_api_call() -> Any:
            kwargs: Dict[str, Any] = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                **_token_kwarg,
            }
            if not _uses_completion_tokens:
                kwargs["response_format"] = {"type": "json_object"}
            if self.summary_seed is not None:
                kwargs["seed"] = self.summary_seed
            return self.client.chat.completions.create(**kwargs)

        try:
            resp = retry_with_metrics(
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

        raw_text = (resp.choices[0].message.content or "").strip() or "{}"
        if hasattr(resp, "usage") and resp.usage is not None:
            try:
                call_metrics.set_tokens(
                    int(getattr(resp.usage, "prompt_tokens", 0) or 0),
                    int(getattr(resp.usage, "completion_tokens", 0) or 0),
                )
            except (TypeError, ValueError):
                pass

        _record_openai_summarization_call(
            resp, pipeline_metrics, cfg=self.cfg, model=self.summary_model
        )
        return parse_extraction_bundle_response(raw_text)

    def summarize_bundled(
        self,
        text: str,
        episode_title: Optional[str] = None,
        episode_description: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: metrics.Metrics | None = None,
        call_metrics: Any | None = None,
    ) -> Dict[str, Any]:
        """One completion: semantic transcript clean + JSON title/bullets (Issue #477).

        Returns the same ``summary`` shape as :meth:`summarize` (JSON string
        with ``title``, ``summary``, and ``bullets``).
        """
        if not self._summarization_initialized:
            raise RuntimeError(
                "OpenAIProvider summarization not initialized. Call initialize() first."
            )

        from ...prompts.store import get_prompt_metadata, render_prompt
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        max_out = int(getattr(self.cfg, "llm_bundled_max_output_tokens", 16384) or 16384)
        _uses_completion_tokens = self.summary_model.startswith(("o1", "o3", "gpt-5"))
        _token_kwarg: Dict[str, Any] = (
            {"max_completion_tokens": max_out}
            if _uses_completion_tokens
            else {"max_tokens": max_out}
        )

        tmpl_kwargs = dict(self.cfg.summary_prompt_params or {})
        system_prompt = render_prompt(
            "openai/summarization/bundled_clean_summary_system_v1",
            **tmpl_kwargs,
        )
        user_prompt = render_prompt(
            "openai/summarization/bundled_clean_summary_user_v1",
            transcript=text,
            title=episode_title or "",
            **tmpl_kwargs,
        )

        if call_metrics is None:
            call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("openai")

        def _make_api_call() -> Any:
            kwargs: Dict[str, Any] = {
                "model": self.summary_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.summary_temperature,
                **_token_kwarg,
            }
            if not self.summary_model.startswith(("o1", "o3", "gpt-5")):
                kwargs["response_format"] = {"type": "json_object"}
            if self.summary_seed is not None:
                kwargs["seed"] = self.summary_seed
            return self.client.chat.completions.create(**kwargs)

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
            "openai",
            "summarize_bundled",
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            raise ValueError("OpenAI bundled call returned empty content")

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
            pt = getattr(response.usage, "prompt_tokens", None)
            ct = getattr(response.usage, "completion_tokens", None)
            input_tokens = int(pt) if isinstance(pt, (int, float)) else 0
            output_tokens = int(ct) if isinstance(ct, (int, float)) else 0
            if input_tokens > 0 or output_tokens > 0:
                call_metrics.set_tokens(input_tokens, output_tokens)

        cost: Optional[float] = None
        if input_tokens is not None:
            from ...workflow.helpers import calculate_provider_cost

            cost = calculate_provider_cost(
                cfg=self.cfg,
                provider_type="openai",
                capability="summarization",
                model=self.summary_model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            call_metrics.set_cost(cost)

        if pipeline_metrics is not None and input_tokens is not None and output_tokens is not None:
            pipeline_metrics.record_llm_bundled_clean_summary_call(
                input_tokens, output_tokens, cost_usd=cost
            )

        prompt_metadata = {
            "system": get_prompt_metadata(
                "openai/summarization/bundled_clean_summary_system_v1",
                params=tmpl_kwargs,
            ),
            "user": get_prompt_metadata(
                "openai/summarization/bundled_clean_summary_user_v1",
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
                "provider": "openai",
                "bundled": True,
                "max_output_tokens": max_out,
                "prompts": prompt_metadata,
            },
        }

    def generate_insights(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_insights: int = 5,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> List[str]:
        """Generate a list of short insight statements from transcript (GIL).

        Uses openai/insight_extraction/v1 prompt; parses response as one insight per line.
        Returns empty list on failure so GIL can fall back to stub.
        """
        if not self._summarization_initialized:
            logger.warning("OpenAI summarization not initialized for generate_insights")
            return []

        from ...prompts.store import render_prompt

        max_insights = min(max(1, max_insights), 10)
        # Truncate transcript for context (e.g. ~100k chars) to avoid token limits
        text_slice = (text or "").strip()
        if len(text_slice) > 120000:
            text_slice = text_slice[:120000] + "\n\n[Transcript truncated.]"

        try:
            user_prompt = render_prompt(
                "openai/insight_extraction/v1",
                transcript=text_slice,
                title=episode_title or "",
                max_insights=max_insights,
            )
            system_prompt = (
                "Output only the list of key takeaways, one per line. "
                "No numbering, bullets, or extra text."
            )
            response = self.client.chat.completions.create(
                model=self.insight_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=min(1024, max_insights * 150),
            )
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            if (
                pipeline_metrics is not None
                and in_tok is not None
                and out_tok is not None
                and hasattr(pipeline_metrics, "record_llm_gi_call")
            ):
                gi_cost: Optional[float] = None
                from ...workflow.helpers import calculate_provider_cost

                gi_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="openai",
                    capability="summarization",
                    model=self.insight_model,
                    prompt_tokens=int(in_tok),
                    completion_tokens=int(out_tok),
                )
                pipeline_metrics.record_llm_gi_call(in_tok, out_tok, cost_usd=gi_cost)
            content = (response.choices[0].message.content or "").strip()
            lines = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            # Drop common list prefixes: "1. ", "2) ", "- ", "* "
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
            logger.debug("OpenAI generate_insights failed: %s", e, exc_info=True)
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
            logger.warning("OpenAI summarization not initialized for extract_kg_graph")
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
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
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
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            pm = pipeline_metrics
            if (
                pm is not None
                and in_tok is not None
                and out_tok is not None
                and hasattr(pm, "record_llm_kg_call")
            ):
                from ...workflow.helpers import calculate_provider_cost

                kg_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="openai",
                    capability="summarization",
                    model=model,
                    prompt_tokens=int(in_tok),
                    completion_tokens=int(out_tok),
                )
                pm.record_llm_kg_call(in_tok, out_tok, cost_usd=kg_cost)
            raw = (response.choices[0].message.content or "").strip()
            return parse_kg_graph_response(raw, max_topics=max_topics, max_entities=max_entities)
        except Exception as e:
            logger.debug("OpenAI extract_kg_graph failed: %s", e, exc_info=True)
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
                "OpenAI summarization not initialized for extract_kg_from_summary_bullets"
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
                _safe_openai_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_prompt},
                    ],
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
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            pm = pipeline_metrics
            if (
                pm is not None
                and in_tok is not None
                and out_tok is not None
                and hasattr(pm, "record_llm_kg_call")
            ):
                from ...workflow.helpers import calculate_provider_cost

                kg_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="openai",
                    capability="summarization",
                    model=model,
                    prompt_tokens=int(in_tok),
                    completion_tokens=int(out_tok),
                )
                pm.record_llm_kg_call(in_tok, out_tok, cost_usd=kg_cost)
            raw = (response.choices[0].message.content or "").strip()
            return parse_kg_graph_response(raw, max_topics=max_topics, max_entities=max_entities)
        except Exception as e:
            logger.debug("OpenAI extract_kg_from_summary_bullets failed: %s", e, exc_info=True)
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
        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span
        from ...prompts.store import render_prompt

        system = (
            "Extract all short verbatim quotes from the transcript that support "
            "the given insight. Quotes must be from different parts of the "
            "transcript. Reply with ONLY a JSON object: "
            '{"quotes": ["exact quote 1", "exact quote 2"]}'
        )
        user = render_prompt(
            "openai/evidence/extract_quote/v1",
            transcript=transcript.strip()[:50000],
            insight=insight_text.strip(),
        )
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("openai")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=512,
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
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics, pm, in_tok, out_tok, stage="extract_quotes"
            )
            content = (response.choices[0].message.content or "").strip()
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
            logger.debug("OpenAI extract_quotes failed: %s", e, exc_info=True)
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
        from ...prompts.store import render_prompt

        system = (
            "You rate how much the premise supports the hypothesis. "
            "Reply with ONLY a number between 0 and 1 (0=not at all, 1=fully supports)."
        )
        user = render_prompt(
            "openai/evidence/entailment/v1",
            premise=premise.strip(),
            hypothesis=hypothesis.strip(),
        )
        try:
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("openai")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.chat.completions.create(
                    model=self.summary_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=10,
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
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(
                call_metrics, pm, in_tok, out_tok, stage="score_entailment"
            )
            content = (response.choices[0].message.content or "0").strip()
            # Take first number
            for part in content.replace(",", " ").split():
                try:
                    v = float(part)
                    return max(0.0, min(1.0, v))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            logger.debug("OpenAI score_entailment failed: %s", e, exc_info=True)
            return 0.0

    def extract_quotes_bundled(
        self,
        transcript: str,
        insight_texts: List[str],
        **kwargs: Any,
    ) -> Dict[int, List[Any]]:
        """Bundle ``extract_quotes`` across all insights into one OpenAI call (#698 Layer A).

        Mirrors :meth:`GeminiProvider.extract_quotes_bundled`. Same shared prompt, same
        parser, same QuoteCandidate output shape — only the SDK glue differs. See
        ``providers/common/bundled_prompts.py`` for the prompt contract.
        """
        if not self._summarization_initialized or not transcript:
            return {idx: [] for idx in range(len(insight_texts))}
        if not insight_texts:
            return {}

        from ...gi.grounding import QuoteCandidate, resolve_llm_quote_span
        from ...providers.common.bundle_extract_parser import (
            BundleExtractParseError,
            parse_bundled_extract_response,
        )
        from ...providers.common.bundled_prompts import (
            extract_quotes_bundled_max_tokens,
            EXTRACT_QUOTES_BUNDLED_SYSTEM,
            extract_quotes_bundled_user,
            transcript_clip,
        )
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            apply_gil_evidence_llm_call_metrics,
            merge_gil_evidence_call_metrics_on_failure,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = EXTRACT_QUOTES_BUNDLED_SYSTEM
        user = extract_quotes_bundled_user(transcript_clip(transcript), insight_texts)

        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("openai")
        pm = kwargs.get("pipeline_metrics")
        max_out = extract_quotes_bundled_max_tokens(len(insight_texts))

        def _make_api_call() -> Any:
            return self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_out,
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
            merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
            raise
        in_tok, out_tok = _openai_chat_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pm,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type="openai",
            model=self.summary_model,
            stage="extract_quotes",
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            parsed = parse_bundled_extract_response(content, expected_count=len(insight_texts))
        except BundleExtractParseError as exc:
            logger.debug("OpenAI extract_quotes_bundled parse failed: %s", exc)
            raise

        out: Dict[int, List[Any]] = {}
        for idx in range(len(insight_texts)):
            quote_strings = parsed.get(idx, [])
            seen: set = set()
            candidates: List[Any] = []
            for qt_str in quote_strings:
                qt_clean = str(qt_str).strip()
                if not qt_clean:
                    continue
                resolved = resolve_llm_quote_span(transcript, qt_clean)
                if resolved is None:
                    continue
                r_start, r_end, r_verbatim = resolved
                if r_verbatim in seen:
                    continue
                seen.add(r_verbatim)
                candidates.append(
                    QuoteCandidate(
                        char_start=r_start,
                        char_end=r_end,
                        text=r_verbatim,
                        qa_score=1.0,
                    )
                )
            out[idx] = candidates
        return out

    def score_entailment_bundled(
        self,
        pairs: List[Tuple[str, str]],
        chunk_size: int = 15,
        **kwargs: Any,
    ) -> Dict[int, float]:
        """Bundle ``score_entailment`` across many pairs (#698 Layer B).

        Mirrors :meth:`GeminiProvider.score_entailment_bundled`. Chunks at
        ``chunk_size`` (default 15) and issues one OpenAI call per chunk.
        """
        if not self._summarization_initialized or not pairs:
            return {}
        chunk_size = max(1, int(chunk_size))
        out: Dict[int, float] = {}
        pm = kwargs.get("pipeline_metrics")
        for chunk_start in range(0, len(pairs), chunk_size):
            chunk = pairs[chunk_start : chunk_start + chunk_size]
            chunk_scores = self._score_entailment_bundled_chunk(
                chunk_pairs=chunk, pipeline_metrics=pm
            )
            for local_idx, score in chunk_scores.items():
                out[chunk_start + local_idx] = score
            if pm is not None and hasattr(pm, "gi_evidence_score_entailment_bundled_pairs_total"):
                pm.gi_evidence_score_entailment_bundled_pairs_total += len(chunk)
        return out

    def _score_entailment_bundled_chunk(
        self,
        chunk_pairs: List[Tuple[str, str]],
        pipeline_metrics: Optional[Any],
    ) -> Dict[int, float]:
        """One bundled NLI OpenAI call for up to ``chunk_size`` pairs."""
        from ...providers.common.bundle_nli_parser import (
            BundleNliParseError,
            parse_bundled_nli_response,
        )
        from ...providers.common.bundled_prompts import (
            score_entailment_bundled_max_tokens,
            SCORE_ENTAILMENT_BUNDLED_SYSTEM,
            score_entailment_bundled_user,
        )
        from ...utils.provider_metrics import (
            _safe_openai_retryable,
            apply_gil_evidence_llm_call_metrics,
            merge_gil_evidence_call_metrics_on_failure,
            ProviderCallMetrics,
            retry_with_metrics,
        )

        system = SCORE_ENTAILMENT_BUNDLED_SYSTEM
        user = score_entailment_bundled_user(chunk_pairs)

        call_metrics = ProviderCallMetrics()
        call_metrics.set_provider_name("openai")
        max_out = score_entailment_bundled_max_tokens(len(chunk_pairs))

        def _make_api_call() -> Any:
            return self.client.chat.completions.create(
                model=self.summary_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=max_out,
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
            merge_gil_evidence_call_metrics_on_failure(call_metrics, pipeline_metrics)
            raise
        in_tok, out_tok = _openai_chat_usage_tokens(response)
        apply_gil_evidence_llm_call_metrics(
            call_metrics,
            pipeline_metrics,
            in_tok,
            out_tok,
            cfg=self.cfg,
            provider_type="openai",
            model=self.summary_model,
            stage="score_entailment",
        )

        content = (response.choices[0].message.content or "").strip()
        try:
            return parse_bundled_nli_response(content, expected_count=len(chunk_pairs))
        except BundleNliParseError as exc:
            logger.debug("OpenAI score_entailment_bundled parse failed: %s", exc)
            raise

    def _build_summarization_prompts(
        self,
        text: str,
        episode_title: Optional[str],
        episode_description: Optional[str],
        max_length: int,
        min_length: int,
        custom_prompt: Optional[str],
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store.

        Args:
            text: Transcript text to summarize
            episode_title: Optional episode title
            episode_description: Optional episode description
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            custom_prompt: Optional custom prompt (overrides default)

        Returns:
            Tuple of (system_prompt, user_prompt, system_prompt_name,
            user_prompt_name, paragraphs_min, paragraphs_max)
        """
        from ...prompts.store import render_prompt

        # Use prompt_store to load versioned prompt templates
        system_prompt_name = (
            self.cfg.openai_summary_system_prompt or "openai/summarization/system_v1"
        )
        user_prompt_name = self.cfg.openai_summary_user_prompt

        # Render system prompt
        system_prompt = render_prompt(system_prompt_name)

        # Estimate paragraphs: roughly 100 tokens per paragraph
        paragraphs_min = max(1, min_length // 100)
        paragraphs_max = max(paragraphs_min, max_length // 100)

        # Render user prompt
        if custom_prompt:
            # Use custom prompt if provided
            user_prompt = custom_prompt.replace("{{ transcript }}", text)
            if episode_title:
                user_prompt = user_prompt.replace("{{ title }}", episode_title)
            # For custom prompts, use a placeholder name
            user_prompt_name = "custom"
        else:
            # Use prompt_store template
            # Merge config params with template params
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
        """Cleanup all provider resources (no-op for API provider).

        This method releases any resources held by the provider.
        For OpenAI API, there are no local resources to clean up.
        It may be called multiple times safely (idempotent).
        """
        # No resources to clean up for API provider
        # But we can mark as uninitialized
        self._transcription_initialized = False
        self._speaker_detection_initialized = False
        self._summarization_initialized = False

    def clear_cache(self) -> None:
        """Clear cache (no-op for API provider).

        OpenAI provider doesn't use cached models, so this is a no-op.
        """
        # No cache to clear for API provider
        pass

    # ============================================================================

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component).

        Returns:
            True if any component is initialized, False otherwise
        """
        return (
            self._transcription_initialized
            or self._speaker_detection_initialized
            or self._summarization_initialized
        )

    def clean_transcript(
        self,
        text: str,
        pipeline_metrics: Optional[Any] = None,
    ) -> str:
        """Clean transcript using LLM for semantic filtering.

        Args:
            text: Transcript text to clean (should already be pattern-cleaned)
            pipeline_metrics: Optional pipeline metrics for LLM usage accounting

        Returns:
            Cleaned transcript text

        Raises:
            RuntimeError: If provider is not initialized or cleaning fails
        """
        if not self._summarization_initialized:
            raise RuntimeError("OpenAIProvider not initialized. Call initialize() first.")

        from ...prompts.store import render_prompt

        # Build cleaning prompt using prompt_store
        prompt_name = "openai/cleaning/v1"
        user_prompt = render_prompt(prompt_name, transcript=text)

        # Use system prompt (optional, can be empty for cleaning)
        system_prompt = (
            "You are a transcript cleaning assistant. "
            "Remove sponsors, ads, intros, outros, and meta-commentary. "
            "Preserve all substantive content and speaker information. "
            "Return only the cleaned text, no explanations."
        )

        logger.debug(
            "Cleaning transcript via OpenAI API (model: %s, text length: %d chars)",
            self.cleaning_model,
            len(text),
        )

        try:
            # Track retries and rate limits
            from ...utils.provider_metrics import (
                _safe_openai_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("openai")

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
                        OPENAI_CLEANING_MAX_TOKENS,
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

            cleaned = response.choices[0].message.content
            in_tok, out_tok = _openai_chat_usage_tokens(response)
            if (
                pipeline_metrics is not None
                and in_tok is not None
                and out_tok is not None
                and hasattr(pipeline_metrics, "record_llm_cleaning_call")
            ):
                from ...workflow.helpers import calculate_provider_cost

                cleaning_cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="openai",
                    capability="summarization",
                    model=self.cleaning_model,
                    prompt_tokens=int(in_tok),
                    completion_tokens=int(out_tok),
                )
                pipeline_metrics.record_llm_cleaning_call(in_tok, out_tok, cost_usd=cleaning_cost)
            if not cleaned:
                logger.warning("OpenAI API returned empty cleaned text, using original")
                return text

            logger.debug("OpenAI cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

        except Exception as exc:
            logger.error("OpenAI API error in cleaning: %s", format_exception_for_log(exc))
            from podcast_scraper.exceptions import ProviderAuthError, ProviderRuntimeError

            # Handle OpenAI-specific error types
            error_msg = str(exc).lower()
            exc_type_name = type(exc).__name__
            if (
                "api key" in error_msg
                or "authentication" in error_msg
                or "permission" in error_msg
                or "401" in error_msg
                or "unauthorized" in error_msg
                or exc_type_name == "AuthenticationError"
            ):
                raise ProviderAuthError(
                    message=f"OpenAI authentication failed: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Cleaning",
                    suggestion=(
                        "Check your OPENAI_API_KEY environment variable or config setting. "
                        "Verify the key is valid and has not expired."
                    ),
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg or "429" in error_msg:
                raise ProviderRuntimeError(
                    message=f"OpenAI rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Cleaning",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"OpenAI cleaning failed: {format_exception_for_log(exc)}",
                    provider="OpenAIProvider/Cleaning",
                ) from exc

    def get_capabilities(self) -> ProviderCapabilities:
        """Get provider capabilities.

        Returns:
            ProviderCapabilities object describing OpenAI provider capabilities
        """
        return ProviderCapabilities(
            supports_transcription=True,
            supports_speaker_detection=True,
            supports_summarization=True,
            supports_semantic_cleaning=True,  # OpenAI supports LLM-based cleaning
            supports_audio_input=True,  # Whisper API accepts audio files
            supports_json_mode=True,  # GPT models support JSON mode
            max_context_tokens=self.max_context_tokens,
            supports_tool_calls=True,  # GPT models support function calling
            supports_system_prompt=True,  # GPT models support system prompts
            supports_streaming=True,  # OpenAI API supports streaming
            provider_name="openai",
            supports_gi_segment_timing=True,
        )
