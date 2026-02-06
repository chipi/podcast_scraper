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
from typing import Any, Dict, Optional, Set, Tuple

from ... import config, models
from ...workflow import metrics

logger = logging.getLogger(__name__)

# Protocol types imported for type hints (used in docstrings and type annotations)
# from ..speaker_detectors.base import SpeakerDetector  # noqa: F401
# from ..summarization.base import SummarizationProvider  # noqa: F401
# from ..transcription.base import TranscriptionProvider  # noqa: F401

# Default speaker names when detection fails
DEFAULT_SPEAKER_NAMES = ["Host", "Guest"]

# OpenAI API pricing constants (for cost estimation)
# Source: https://openai.com/pricing
# Last updated: 2025-01
# Note: Prices subject to change. Always verify current rates at https://openai.com/pricing
#
# Pricing is model-based, not environment-based. The same model costs the same
# whether used in test or production. Test/prod distinction is only in which
# models are selected (test uses cheaper models, prod uses higher quality models).
#
# Only models actually defined in config_constants.py are tracked here:
# - whisper-1: Used for transcription (test and prod)
# - gpt-4o-mini: Used for speaker detection (test and prod) and summarization (test)
# - gpt-4o: Used for summarization (prod)
OPENAI_WHISPER_COST_PER_MINUTE = 0.006  # Whisper API: $0.006 per minute of audio
OPENAI_GPT4O_MINI_INPUT_COST_PER_1M_TOKENS = 0.15  # GPT-4o-mini: $0.15 per 1M input tokens
OPENAI_GPT4O_MINI_OUTPUT_COST_PER_1M_TOKENS = 0.60  # GPT-4o-mini: $0.60 per 1M output tokens
OPENAI_GPT4O_INPUT_COST_PER_1M_TOKENS = 2.50  # GPT-4o: $2.50 per 1M input tokens
OPENAI_GPT4O_OUTPUT_COST_PER_1M_TOKENS = 10.00  # GPT-4o: $10.00 per 1M output tokens


class OpenAIProvider:
    """Unified OpenAI provider implementing TranscriptionProvider, SpeakerDetector, and SummarizationProvider.

    This provider initializes and manages:
    - OpenAI Whisper API for transcription
    - OpenAI GPT API for speaker detection
    - OpenAI GPT API for summarization

    All three capabilities share the same OpenAI client, similar to how ML providers
    share the same ML libraries. The client is initialized once and reused.
    """  # noqa: E501

    def __init__(self, cfg: config.Config):
        """Initialize unified OpenAI provider.

        Args:
            cfg: Configuration object with settings for all three capabilities

        Raises:
            ValueError: If OpenAI API key is not provided
            ImportError: If openai package is not installed (install with [llm] extra)
        """
        # Lazy import to allow unit tests without openai installed (Issue #405)
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI provider. "
                "Install it with: pip install podcast-scraper[llm]"
            ) from exc

        if not cfg.openai_api_key:
            raise ValueError(
                "OpenAI API key required for OpenAI provider. "
                "Set OPENAI_API_KEY environment variable or openai_api_key in config."
            )

        self.cfg = cfg

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
        self.client = OpenAI(**client_kwargs)

        # Transcription settings
        self.transcription_model = getattr(cfg, "openai_transcription_model", "whisper-1")

        # Speaker detection settings
        self.speaker_model = getattr(cfg, "openai_speaker_model", "gpt-4o-mini")
        self.speaker_temperature = getattr(cfg, "openai_temperature", 0.3)

        # Summarization settings
        self.summary_model = getattr(cfg, "openai_summary_model", "gpt-4o-mini")
        self.summary_temperature = getattr(cfg, "openai_temperature", 0.3)
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
        """Get pricing information for a specific model and capability.

        Only models defined in config_constants.py are supported:
        - whisper-1: Transcription (test and prod)
        - gpt-4o-mini: Speaker detection (test and prod), Summarization (test)
        - gpt-4o: Summarization (prod)

        Pricing is model-based, not environment-based. The same model costs the same
        whether used in test or production. Test/prod distinction is only in which
        models are selected.

        Args:
            model: Model name (must match one of the models in config_constants.py)
            capability: Capability type ("transcription", "speaker_detection", "summarization")

        Returns:
            Dictionary with pricing information:
            - For transcription: {"cost_per_minute": float}
            - For chat models: {"input_cost_per_1m_tokens": float,
              "output_cost_per_1m_tokens": float}
            - Empty dict if pricing not available for the model/capability
        """
        if capability == "transcription":
            if model == "whisper-1":
                return {"cost_per_minute": OPENAI_WHISPER_COST_PER_MINUTE}
        elif capability in ("speaker_detection", "summarization"):
            model_lower = model.lower()
            # Only support models actually defined in config_constants.py
            if "gpt-4o-mini" in model_lower:
                return {
                    "input_cost_per_1m_tokens": OPENAI_GPT4O_MINI_INPUT_COST_PER_1M_TOKENS,
                    "output_cost_per_1m_tokens": OPENAI_GPT4O_MINI_OUTPUT_COST_PER_1M_TOKENS,
                }
            elif "gpt-4o" in model_lower and "mini" not in model_lower:
                return {
                    "input_cost_per_1m_tokens": OPENAI_GPT4O_INPUT_COST_PER_1M_TOKENS,
                    "output_cost_per_1m_tokens": OPENAI_GPT4O_OUTPUT_COST_PER_1M_TOKENS,
                }
        return {}

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
            with open(audio_path, "rb") as audio_file:
                # OpenAI API requires keyword-only arguments
                # When response_format="text", returns str directly
                if effective_language is not None:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.transcription_model,
                        file=audio_file,
                        language=effective_language,
                        response_format="text",  # Simple text response
                    )
                else:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.transcription_model,
                        file=audio_file,
                        response_format="text",  # Simple text response
                    )

            # transcript is a string when response_format="text"
            text = str(transcript) if not isinstance(transcript, str) else transcript

            logger.debug(
                "OpenAI transcription completed: %d characters",
                len(text) if text else 0,
            )

            return text

        except Exception as exc:
            logger.error("OpenAI Whisper API error: %s", exc)
            from podcast_scraper.exceptions import ProviderRuntimeError

            raise ProviderRuntimeError(
                message=f"OpenAI transcription failed: {exc}",
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
            from openai import APIError, RateLimitError

            from ...utils.provider_metrics import retry_with_metrics

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
                    retryable_exceptions=(RateLimitError, APIError, ConnectionError),
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
                # Estimate from file size (rough: 1MB ≈ 1 minute for typical MP3)
                try:
                    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
                    audio_minutes = file_size_mb  # Rough estimate
                except OSError:
                    pass

            # Calculate cost for transcription (per minute pricing)
            if audio_minutes > 0:
                if pipeline_metrics is not None:
                    pipeline_metrics.record_llm_transcription_call(audio_minutes)
                # Calculate cost
                from ...workflow.helpers import calculate_provider_cost

                cost = calculate_provider_cost(
                    cfg=self.cfg,
                    provider_type="openai",
                    capability="transcription",
                    model=self.transcription_model,
                    audio_minutes=audio_minutes,
                )
                call_metrics.set_cost(cost)

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
            logger.error("OpenAI Whisper API error: %s", exc)
            raise ValueError(f"OpenAI transcription failed: {exc}") from exc

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
                "OpenAIProvider speaker detection not initialized. Call initialize() first."
            )

        logger.debug("Detecting speakers via OpenAI API for episode: %s", episode_title[:50])

        try:
            # Build prompt using prompt_store (RFC-017)
            user_prompt = self._build_speaker_detection_prompt(
                episode_title, episode_description, known_hosts
            )

            # Get system prompt from prompt_store (RFC-017)
            from ...prompts.store import render_prompt

            system_prompt_name = self.cfg.openai_speaker_system_prompt or "openai/ner/system_ner_v1"
            system_prompt = render_prompt(system_prompt_name)

            # Call OpenAI API
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
                logger.warning("OpenAI API returned empty response")
                return DEFAULT_SPEAKER_NAMES.copy(), set(), False

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
                pipeline_metrics.record_llm_speaker_detection_call(input_tokens, output_tokens)

            return speakers, detected_hosts, success

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse OpenAI API JSON response: %s", exc)
            logger.debug(
                "Response text: %s", response_text if "response_text" in locals() else "N/A"
            )
            return DEFAULT_SPEAKER_NAMES.copy(), set(), False
        except Exception as exc:
            logger.error("OpenAI API error in speaker detection: %s", exc)
            raise ValueError(f"OpenAI speaker detection failed: {exc}") from exc

    def analyze_patterns(
        self,
        episodes: list[models.Episode],
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
        """Build prompt for speaker detection using prompt_store (RFC-017).

        Args:
            episode_title: Episode title
            episode_description: Optional episode description
            known_hosts: Set of known host names

        Returns:
            Rendered prompt string
        """
        from ...prompts.store import render_prompt

        # Use prompt_store to load versioned prompt template (RFC-017)
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
            logger.warning("Failed to parse OpenAI response as JSON: %s", exc)
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
            from ...utils.provider_metrics import ProviderCallMetrics

            if call_metrics is None:
                call_metrics = ProviderCallMetrics()
            call_metrics.set_provider_name("openai")

            # Wrap API call with retry tracking
            from openai import APIError, RateLimitError

            from ...utils.provider_metrics import retry_with_metrics

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
                logger.warning("OpenAI API returned empty summary")
                summary = ""

            logger.debug("OpenAI summarization completed: %d characters", len(summary))

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
                    provider_type="openai",
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
            logger.error("OpenAI API error in summarization: %s", exc)
            raise ValueError(f"OpenAI summarization failed: {exc}") from exc

    def _build_summarization_prompts(
        self,
        text: str,
        episode_title: Optional[str],
        episode_description: Optional[str],
        max_length: int,
        min_length: int,
        custom_prompt: Optional[str],
    ) -> tuple[str, str, Optional[str], str, int, int]:
        """Build system and user prompts for summarization using prompt_store (RFC-017).

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

        # Use prompt_store to load versioned prompt templates (RFC-017)
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
