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

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Set, Tuple

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None  # type: ignore

from ... import config, models
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

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
    """Unified Mistral provider implementing TranscriptionProvider, SpeakerDetector, and
    SummarizationProvider.

    This provider initializes and manages:
    - Mistral Voxtral API for transcription
    - Mistral chat API for speaker detection
    - Mistral chat API for summarization

    All three capabilities share the same Mistral client, similar to how OpenAI providers
    share the same OpenAI client. The client is initialized once and reused.

    Key advantage: Mistral is a complete OpenAI alternative (all three capabilities).
    """

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
                "Install with: pip install 'podcast-scraper[mistral]'"
            )

        if not cfg.mistral_api_key:
            raise ValueError(
                "Mistral API key required for Mistral provider. "
                "Set MISTRAL_API_KEY environment variable or mistral_api_key in config."
            )

        self.cfg = cfg

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
        # Mistral SDK uses 'server' or 'server_url' parameter (not 'endpoint' or 'base_url')
        client_kwargs: dict[str, Any] = {"api_key": cfg.mistral_api_key}
        if cfg.mistral_api_base:
            # Mistral SDK uses 'server' parameter for custom base URL
            client_kwargs["server"] = cfg.mistral_api_base
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

            from mistralai.models.file import File

            # File object requires file_name and binary content
            file_name = Path(audio_path).name
            with open(audio_path, "rb") as audio_file:
                file_content = audio_file.read()
            mistral_file = File(file_name=file_name, content=file_content)

            # Mistral Voxtral API uses 'complete' method (not 'create' like OpenAI)
            if effective_language is not None:
                transcription = self.client.audio.transcriptions.complete(
                    model=self.transcription_model,
                    file=mistral_file,
                    language=effective_language,
                )
            else:
                transcription = self.client.audio.transcriptions.complete(
                    model=self.transcription_model,
                    file=mistral_file,
                )

            # Extract text from response
            text = transcription.text if hasattr(transcription, "text") else str(transcription)
            if not text:
                logger.warning("Mistral Voxtral API returned empty transcription")
                text = ""

            logger.debug("Mistral transcription completed: %d characters", len(text))
            return text

        except Exception as exc:
            logger.error("Mistral API error in transcription: %s", exc)
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Mistral-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Mistral authentication failed: {exc}",
                    provider="MistralProvider/Transcription",
                    suggestion="Check your MISTRAL_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral rate limit exceeded: {exc}",
                    provider="MistralProvider/Transcription",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral invalid model: {exc}",
                    provider="MistralProvider/Transcription",
                    suggestion="Check mistral_transcription_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Mistral transcription failed: {exc}",
                    provider="MistralProvider/Transcription",
                ) from exc

    def transcribe_with_segments(
        self,
        audio_path: str,
        language: str | None = None,
        pipeline_metrics: metrics.Metrics | None = None,
        episode_duration_seconds: int | None = None,
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

            # Call Mistral API - uses 'complete' method (not 'completions.create' like OpenAI)
            response = self.client.chat.complete(
                model=self.speaker_model,
                messages=[  # type: ignore[arg-type]
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.speaker_temperature,
                max_tokens=300,
                response_format={"type": "json_object"},  # Request JSON response
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
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

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

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Mistral API JSON response: %s", exc)
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("Mistral API error in speaker detection: %s", exc)
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Mistral-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Mistral authentication failed: {exc}",
                    provider="MistralProvider/SpeakerDetection",
                    suggestion="Check your MISTRAL_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral rate limit exceeded: {exc}",
                    provider="MistralProvider/SpeakerDetection",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral invalid model: {exc}",
                    provider="MistralProvider/SpeakerDetection",
                    suggestion="Check mistral_speaker_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Mistral speaker detection failed: {exc}",
                    provider="MistralProvider/SpeakerDetection",
                ) from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
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

            # Call Mistral API - uses 'complete' method (not 'completions.create' like OpenAI)
            response = self.client.chat.complete(
                model=self.summary_model,
                messages=[  # type: ignore[arg-type]
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.summary_temperature,
                max_tokens=max_length,
            )

            # Mistral SDK response structure: response.choices[0].message.content
            summary = response.choices[0].message.content
            if not summary:
                logger.warning("Mistral API returned empty summary")
                summary = ""

            logger.debug("Mistral summarization completed: %d characters", len(summary))

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
                pipeline_metrics.record_llm_summarization_call(input_tokens, output_tokens)

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
            logger.error("Mistral API error in summarization: %s", exc)
            from podcast_scraper.exceptions import (
                ProviderAuthError,
                ProviderRuntimeError,
            )

            # Handle Mistral-specific error types
            error_msg = str(exc).lower()
            if "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                raise ProviderAuthError(
                    message=f"Mistral authentication failed: {exc}",
                    provider="MistralProvider/Summarization",
                    suggestion="Check your MISTRAL_API_KEY environment variable or config setting",
                ) from exc
            elif "quota" in error_msg or "rate limit" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral rate limit exceeded: {exc}",
                    provider="MistralProvider/Summarization",
                    suggestion="Wait before retrying or check your API quota",
                ) from exc
            elif "invalid" in error_msg and "model" in error_msg:
                raise ProviderRuntimeError(
                    message=f"Mistral invalid model: {exc}",
                    provider="MistralProvider/Summarization",
                    suggestion="Check mistral_summary_model configuration",
                ) from exc
            else:
                raise ProviderRuntimeError(
                    message=f"Mistral summarization failed: {exc}",
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

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized (any component)."""
        return (
            self._transcription_initialized
            or self._speaker_detection_initialized
            or self._summarization_initialized
        )
