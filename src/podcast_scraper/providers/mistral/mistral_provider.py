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
import json
import logging
import os
import time
from typing import Any, cast, Dict, List, Optional, Set, Tuple, TYPE_CHECKING


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

from ... import config

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
    MISTRAL_CLEANING_MAX_TOKENS,
)
from ...utils.log_redaction import format_exception_for_log
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
from ..ml.speaker_detection import DEFAULT_SPEAKER_NAMES

# Mistral API pricing constants (for cost estimation)
# Source: https://docs.mistral.ai/pricing/
# Last updated: 2026-02
# Note: Prices subject to change. Always verify current rates
MISTRAL_VOXTRAL_COST_PER_MINUTE = 0.006  # Voxtral: $0.006 per minute of audio
MISTRAL_SMALL_INPUT_COST_PER_1M_TOKENS = 0.20  # mistral-small: $0.20 per 1M input tokens
MISTRAL_SMALL_OUTPUT_COST_PER_1M_TOKENS = 0.20  # mistral-small: $0.20 per 1M output tokens
MISTRAL_LARGE_INPUT_COST_PER_1M_TOKENS = 2.00  # mistral-large: $2.00 per 1M input tokens
MISTRAL_LARGE_OUTPUT_COST_PER_1M_TOKENS = 6.00  # mistral-large: $6.00 per 1M output tokens


class MistralProvider:
    """Unified Mistral provider: TranscriptionProvider, SpeakerDetector, SummarizationProvider.

    Uses Voxtral for transcription and chat API for speaker detection and summarization.
    Full OpenAI alternative; all capabilities share the same Mistral client.
    """

    cleaning_processor: TranscriptCleaningProcessor  # Type annotation for mypy

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

        # Configure HTTP timeouts with separate connect/read timeouts
        # Note: Mistral SDK does not support timeout parameter in __init__
        # Timeout configuration would need to be handled at the HTTP client level
        # if needed in the future. For now, we skip timeout configuration.
        # timeout_config = get_http_timeout(cfg)  # Not supported by Mistral SDK

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
        self.max_context_tokens = 256000  # Conservative estimate

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
        """Get pricing information for a specific model and capability.

        Args:
            model: Model name (e.g., "mistral-large-latest", "voxtral-mini-latest")
            capability: Capability type ("transcription", "speaker_detection", "summarization")

        Returns:
            Dictionary with pricing information
        """
        pricing: Dict[str, float] = {}

        if capability == "transcription":
            # Audio pricing is per minute
            pricing["cost_per_minute"] = MISTRAL_VOXTRAL_COST_PER_MINUTE
            pricing["cost_per_hour"] = MISTRAL_VOXTRAL_COST_PER_MINUTE * 60
        else:
            # Text-based pricing (speaker detection, summarization)
            if "small" in model.lower():
                pricing["input_cost_per_1m_tokens"] = MISTRAL_SMALL_INPUT_COST_PER_1M_TOKENS
                pricing["output_cost_per_1m_tokens"] = MISTRAL_SMALL_OUTPUT_COST_PER_1M_TOKENS
            elif "large" in model.lower():
                pricing["input_cost_per_1m_tokens"] = MISTRAL_LARGE_INPUT_COST_PER_1M_TOKENS
                pricing["output_cost_per_1m_tokens"] = MISTRAL_LARGE_OUTPUT_COST_PER_1M_TOKENS
            else:
                # Default to small pricing
                pricing["input_cost_per_1m_tokens"] = MISTRAL_SMALL_INPUT_COST_PER_1M_TOKENS
                pricing["output_cost_per_1m_tokens"] = MISTRAL_SMALL_OUTPUT_COST_PER_1M_TOKENS

        return pricing

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

        # Initialize summarization if enabled
        if self.cfg.generate_summaries and not self._summarization_initialized:
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

            def _make_transcribe_call():
                if effective_language is not None:
                    return self.client.audio.transcriptions.complete(
                        model=self.transcription_model,
                        file=mistral_file,
                        language=effective_language,
                    )
                else:
                    return self.client.audio.transcriptions.complete(
                        model=self.transcription_model,
                        file=mistral_file,
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

        # Track LLM call metrics if available
        if pipeline_metrics is not None and episode_duration_seconds is not None:
            audio_minutes = episode_duration_seconds / 60.0
            # Note: Mistral doesn't provide token usage for audio, so we track by duration
            # This is an approximation - actual pricing is per minute of audio
            pipeline_metrics.record_llm_transcription_call(audio_minutes)

        # Finalize call_metrics (Mistral audio transcription may not have tokens,
        # but finalize for consistency)
        if call_metrics is not None:
            if episode_duration_seconds is not None:
                from ...workflow.helpers import calculate_provider_cost

                audio_minutes = episode_duration_seconds / 60.0
                cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="mistral",
                    capability="transcription",
                    model=self.transcription_model,
                    audio_minutes=audio_minutes,
                )
                call_metrics.set_cost(cost)
            call_metrics.finalize()

        # Mistral Voxtral may not provide segments in the same format as OpenAI
        # Return text with empty segments for now (can be enhanced if Mistral supports it)
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
        """Detect host names from feed-level metadata using Mistral API.

        Args:
            feed_title: Feed title (can be None)
            feed_description: Optional feed description
            feed_authors: Optional list of author names from RSS feed (preferred source)

        Returns:
            Set of detected host names
        """
        if not self._speaker_detection_initialized:
            raise RuntimeError(
                "MistralProvider speaker detection not initialized. Call initialize() first."
            )

        # Prefer RSS author tags if available (like OpenAI)
        if feed_authors:
            return set(feed_authors)

        # Otherwise, use Mistral API to detect hosts from feed metadata
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
        """Detect speaker names from episode metadata using Mistral API.

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
                "MistralProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via Mistral API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store
            from ...prompts.store import render_prompt

            system_prompt_name = (
                self.cfg.mistral_speaker_system_prompt or "mistral/ner/system_ner_v1"
            )
            system_prompt = render_prompt(system_prompt_name)

            # Call Mistral API with retry
            from ...utils.provider_metrics import (
                _safe_mistral_retryable,
                retry_with_metrics,
            )

            response = retry_with_metrics(
                lambda: self.client.chat.complete(
                    model=self.speaker_model,
                    messages=[  # type: ignore[arg-type]
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
                retryable_exceptions=_safe_mistral_retryable(),
            )

            # Mistral SDK response structure: response.choices[0].message.content
            response_content = response.choices[0].message.content
            # Handle both string and list of chunks
            if isinstance(response_content, str):
                response_text = response_content
            elif isinstance(response_content, list) and len(response_content) > 0:
                # Extract text from first chunk if it's a list
                first_chunk = response_content[0]
                if hasattr(first_chunk, "text"):
                    response_text = first_chunk.text
                elif isinstance(first_chunk, str):
                    response_text = first_chunk
                else:
                    response_text = str(first_chunk)
            else:
                response_text = ""
            if not response_text:
                logger.warning("Mistral API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True

            # Parse JSON response
            speakers, detected_hosts, success = self._parse_speakers_from_response(
                response_text, known_hosts
            )

            logger.debug(
                "Mistral speaker detection completed: %d speakers, %d hosts, success=%s",
                len(speakers),
                len(detected_hosts),
                success,
            )

            # Track LLM call metrics if available
            if pipeline_metrics is not None and hasattr(response, "usage"):
                input_tokens = (
                    int(response.usage.prompt_tokens)
                    if response.usage and response.usage.prompt_tokens
                    else 0
                )
                output_tokens = (
                    int(response.usage.completion_tokens)
                    if response.usage and response.usage.completion_tokens
                    else 0
                )
                pipeline_metrics.record_llm_speaker_detection_call(input_tokens, output_tokens)

            return speakers, detected_hosts, success, False

        except json.JSONDecodeError as exc:
            logger.error(
                "Failed to parse Mistral API JSON response: %s", format_exception_for_log(exc)
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False, True
        except Exception as exc:
            logger.error(
                "Mistral API error in speaker detection: %s", format_exception_for_log(exc)
            )
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Mistral-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Mistral authentication failed: {format_exception_for_log(exc)}",
                    provider="MistralProvider/SpeakerDetection",
                    suggestion="Check your MISTRAL_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral rate limit exceeded: {format_exception_for_log(exc)}",
                    provider="MistralProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral invalid model: {format_exception_for_log(exc)}",
                    provider="MistralProvider/SpeakerDetection",
                    suggestion="Check mistral_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Mistral speaker detection failed: {format_exception_for_log(exc)}",
                    provider="MistralProvider/SpeakerDetection",
                ) from exc

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

    def _build_speaker_detection_prompt(
        self, episode_title: str, episode_description: str | None, known_hosts: Set[str]
    ) -> str:
        """Build user prompt for speaker detection using prompt_store."""
        from ...prompts.store import render_prompt

        user_prompt_name = self.cfg.mistral_speaker_user_prompt
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
        """Parse speaker names from Mistral API response."""
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
        min_length = (
            (params.get("min_length") if params else None)
            or self.cfg.summary_reduce_params.get("min_new_tokens")
            or 100
        )
        custom_prompt = params.get("prompt") if params else None

        logger.debug(
            "Summarizing text via Mistral API (model: %s, max_tokens: %d)",
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
                _safe_mistral_retryable,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("mistral")

            # Wrap API call with retry tracking
            def _make_api_call():
                return self.client.chat.complete(
                    model=self.summary_model,
                    messages=[  # type: ignore[arg-type]
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
                    retryable_exceptions=_safe_mistral_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                call_metrics.finalize()
                raise

            call_metrics.finalize()

            # Mistral SDK response structure: response.choices[0].message.content
            summary = response.choices[0].message.content
            if not summary:
                logger.warning("Mistral API returned empty summary")
                summary = ""

            logger.debug("Mistral summarization completed: %d characters", len(summary))

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
                    provider_type="mistral",
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
                "summary_short": None,  # Mistral doesn't generate short summaries separately
                "metadata": {
                    "model": self.summary_model,
                    "provider": "mistral",
                    "max_length": max_length,
                    "min_length": min_length,
                    "prompts": prompt_metadata,
                },
            }

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
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store (RFC-017)."""
        from ...prompts.store import render_prompt

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

    def generate_insights(
        self,
        text: str,
        episode_title: Optional[str] = None,
        max_insights: int = 5,
        params: Optional[Dict[str, Any]] = None,
        pipeline_metrics: Optional[Any] = None,
    ) -> List[str]:
        """Generate a list of short insight statements from transcript (GIL).

        Uses mistral/insight_extraction/v1 prompt; parses response as one insight per line.
        Returns empty list on failure so GIL can fall back to stub.
        """
        if not self._summarization_initialized:
            logger.warning("Mistral summarization not initialized for generate_insights")
            return []

        from ...prompts.store import render_prompt

        max_insights = min(max(1, max_insights), 10)
        text_slice = (text or "").strip()
        if len(text_slice) > 120000:
            text_slice = text_slice[:120000] + "\n\n[Transcript truncated.]"

        try:
            user_prompt = render_prompt(
                "mistral/insight_extraction/v1",
                transcript=text_slice,
                title=episode_title or "",
                max_insights=max_insights,
            )
            system_prompt = (
                "Output only the list of key takeaways, one per line. "
                "No numbering, bullets, or extra text."
            )
            response = self.client.chat.complete(
                model=self.summary_model,
                messages=[  # type: ignore[arg-type]
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=min(1024, max_insights * 150),
            )
            raw = response.choices[0].message.content
            content = (raw if isinstance(raw, str) else "") or ""
            content = content.strip()
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
            logger.debug("Mistral generate_insights failed: %s", e, exc_info=True)
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
            logger.warning("Mistral summarization not initialized for extract_kg_graph")
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
                _safe_mistral_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.chat.complete(
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
                retryable_exceptions=_safe_mistral_retryable(),
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
                "Mistral summarization not initialized for extract_kg_from_summary_bullets"
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
                _safe_mistral_retryable,
                retry_with_metrics,
            )

            def _make_api_call():
                return self.client.chat.complete(
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
                retryable_exceptions=_safe_mistral_retryable(),
            )
            raw = response.choices[0].message.content
            content = (raw if isinstance(raw, str) else "") or ""
            return parse_kg_graph_response(
                content.strip(),
                max_topics=max_topics,
                max_entities=max_entities,
            )
        except Exception as e:
            logger.debug("Mistral extract_kg_from_summary_bullets failed: %s", e, exc_info=True)
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
                _safe_mistral_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                openai_compatible_chat_usage_tokens,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("mistral")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.chat.complete(
                    model=self.summary_model,
                    messages=[  # type: ignore[arg-type]
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
                    retryable_exceptions=_safe_mistral_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(call_metrics, pm, in_tok, out_tok)
            raw = response.choices[0].message.content
            content = (raw if isinstance(raw, str) else "") or ""
            content = content.strip()
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
            logger.debug("Mistral extract_quotes failed: %s", e, exc_info=True)
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
                _safe_mistral_retryable,
                apply_gil_evidence_llm_call_metrics,
                merge_gil_evidence_call_metrics_on_failure,
                openai_compatible_chat_usage_tokens,
                ProviderCallMetrics,
                retry_with_metrics,
            )

            call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("mistral")
            pm = kwargs.get("pipeline_metrics")

            def _make_api_call():
                return self.client.chat.complete(
                    model=self.summary_model,
                    messages=[  # type: ignore[arg-type]
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
                    retryable_exceptions=_safe_mistral_retryable(),
                    metrics=call_metrics,
                )
            except Exception:
                merge_gil_evidence_call_metrics_on_failure(call_metrics, pm)
                raise
            in_tok, out_tok = openai_compatible_chat_usage_tokens(response)
            apply_gil_evidence_llm_call_metrics(call_metrics, pm, in_tok, out_tok)
            raw = response.choices[0].message.content
            content = (raw if isinstance(raw, str) else "0") or "0"
            content = content.strip()
            for part in content.replace(",", " ").split():
                try:
                    v = float(part)
                    return max(0.0, min(1.0, v))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            logger.debug("Mistral score_entailment failed: %s", e, exc_info=True)
            return 0.0

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

        # Build cleaning prompt using prompt_store (RFC-017)
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

            cleaned = response.choices[0].message.content
            if not cleaned:
                logger.warning("Mistral API returned empty cleaned text, using original")
                return text

            logger.debug("Mistral cleaning completed: %d -> %d chars", len(text), len(cleaned))
            return cast(str, cleaned)

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
        )
