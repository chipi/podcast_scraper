"""Application configuration. Low MI: see docs/ci/CODE_QUALITY_TRENDS.md."""

from __future__ import annotations

import json
import logging
import os
import threading
import warnings
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, TYPE_CHECKING
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from . import config_constants

logger = logging.getLogger(__name__)

# GitHub #562: one INFO per process when coercing screenplay off for API transcription.
_screenplay_tx_api_coerce_lock = threading.Lock()
_screenplay_tx_api_coerce_state: dict[str, bool] = {"logged": False}


def reset_screenplay_transcription_api_coerce_log_for_tests() -> None:
    """Reset #562 process-wide log gate (unit tests only)."""
    with _screenplay_tx_api_coerce_lock:
        _screenplay_tx_api_coerce_state["logged"] = False


def reset_screenplay_issue_562_gates() -> None:
    """Reset all GitHub #562 gates (unit tests and between ``run_pipeline`` invocations)."""
    reset_screenplay_transcription_api_coerce_log_for_tests()
    from .workflow import episode_processor as _ep562

    _ep562.reset_screenplay_unsupported_provider_warning_for_tests()
    _ep562.reset_screenplay_format_failure_warning_for_tests()


def _raw_screenplay_requested(value: Any) -> bool:
    """Whether raw input should be treated as screenplay enabled (GitHub #562 follow-up).

    Accepts common YAML/JSON shapes (bool, 1/0, yes/no strings). Unknown non-empty strings
    are treated as **not** requested so we do not guess.
    """
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, (int, float)) and int(value) == 1:
        return True
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("false", "no", "off", "0", "", "none"):
            return False
        if v in ("true", "yes", "on", "1"):
            return True
        return False
    return False


def _screenplay_strict_env_enabled() -> bool:
    """When set, invalid screenplay + API transcription is an error instead of coercion."""
    v = os.environ.get("PODCAST_SCRAPER_SCREENPLAY_STRICT", "")
    return str(v).strip().lower() in ("1", "true", "yes", "on")


if TYPE_CHECKING:
    from podcast_scraper.evaluation.experiment_config import GenerationParams, TokenizeConfig
else:
    # Lazy import to avoid circular dependency
    GenerationParams = None
    TokenizeConfig = None


# Load .env file if it exists (RFC-013: OpenAI API key management)
# Check for .env in project root
# Use get_project_root() for robust path resolution
# SKIP .env loading in test environments - tests should use Config objects and
# environment variables directly, never rely on .env files
def _is_test_environment() -> bool:
    """Check if we're running in a test environment."""
    import sys

    # Check for pytest (most common test runner)
    if "pytest" in sys.modules or "PYTEST_CURRENT_TEST" in os.environ:
        return True
    # Check for unittest
    if "unittest" in sys.modules:
        return True
    # Check for explicit test environment variable
    if os.environ.get("TESTING", "").lower() in ("1", "true", "yes"):
        return True
    return False


# Only load .env file if NOT in test environment
# Tests should use Config objects with explicit values or environment variables
if not _is_test_environment():
    try:
        from .cache import get_project_root

        env_path = get_project_root() / ".env"
        if env_path.exists():
            try:
                load_dotenv(env_path, override=False)
            except (PermissionError, OSError):
                # Handle permission errors gracefully
                # Fallback to current working directory
                try:
                    load_dotenv(override=False)
                except (PermissionError, OSError):
                    # If both fail, continue without .env file
                    pass
        else:
            # Also check current working directory (for flexibility)
            try:
                load_dotenv(override=False)
            except (PermissionError, OSError):
                # If loading fails, continue without .env file
                pass
    except ImportError:
        # Fallback if cache module not available (shouldn't happen in normal usage)
        # Also check current working directory (for flexibility)
        try:
            load_dotenv(override=False)
        except (PermissionError, OSError):
            # If loading fails, continue without .env file
            pass

# Import constants from config_constants.py to avoid duplication
# These are re-exported here for backward compatibility
DEFAULT_LOG_LEVEL = config_constants.DEFAULT_LOG_LEVEL
DEFAULT_NUM_SPEAKERS = config_constants.DEFAULT_NUM_SPEAKERS
DEFAULT_SCREENPLAY_GAP_SECONDS = config_constants.DEFAULT_SCREENPLAY_GAP_SECONDS
DEFAULT_TIMEOUT_SECONDS = config_constants.DEFAULT_TIMEOUT_SECONDS
DEFAULT_USER_AGENT = config_constants.DEFAULT_USER_AGENT
DEFAULT_WORKERS = config_constants.DEFAULT_WORKERS
DEFAULT_LANGUAGE = config_constants.DEFAULT_LANGUAGE
DEFAULT_PREPROCESSING_CACHE_DIR = config_constants.DEFAULT_PREPROCESSING_CACHE_DIR

# Speaker detection defaults
# Note: DEFAULT_NER_MODEL is now set via _get_default_ner_model() function
# to support dev/prod distinction (TEST vs PROD defaults)
TEST_DEFAULT_NER_MODEL = config_constants.TEST_DEFAULT_NER_MODEL
PROD_DEFAULT_NER_MODEL = config_constants.PROD_DEFAULT_NER_MODEL
DEFAULT_MAX_DETECTED_NAMES = config_constants.DEFAULT_MAX_DETECTED_NAMES
MIN_NUM_SPEAKERS = config_constants.MIN_NUM_SPEAKERS
MIN_TIMEOUT_SECONDS = config_constants.MIN_TIMEOUT_SECONDS

# Test defaults (smaller, faster models for CI/local dev)
TEST_DEFAULT_WHISPER_MODEL = config_constants.TEST_DEFAULT_WHISPER_MODEL
TEST_DEFAULT_SUMMARY_MODEL = config_constants.TEST_DEFAULT_SUMMARY_MODEL
TEST_DEFAULT_SUMMARY_REDUCE_MODEL = config_constants.TEST_DEFAULT_SUMMARY_REDUCE_MODEL

# Production defaults (quality models for production use)
PROD_DEFAULT_WHISPER_MODEL = config_constants.PROD_DEFAULT_WHISPER_MODEL
PROD_DEFAULT_SUMMARY_MODEL = config_constants.PROD_DEFAULT_SUMMARY_MODEL
PROD_DEFAULT_SUMMARY_REDUCE_MODEL = config_constants.PROD_DEFAULT_SUMMARY_REDUCE_MODEL

# OpenAI model defaults
TEST_DEFAULT_OPENAI_TRANSCRIPTION_MODEL = config_constants.TEST_DEFAULT_OPENAI_TRANSCRIPTION_MODEL
TEST_DEFAULT_OPENAI_SPEAKER_MODEL = config_constants.TEST_DEFAULT_OPENAI_SPEAKER_MODEL
TEST_DEFAULT_OPENAI_SUMMARY_MODEL = config_constants.TEST_DEFAULT_OPENAI_SUMMARY_MODEL
TEST_DEFAULT_OPENAI_CLEANING_MODEL = config_constants.TEST_DEFAULT_OPENAI_CLEANING_MODEL
PROD_DEFAULT_OPENAI_TRANSCRIPTION_MODEL = config_constants.PROD_DEFAULT_OPENAI_TRANSCRIPTION_MODEL
PROD_DEFAULT_OPENAI_SPEAKER_MODEL = config_constants.PROD_DEFAULT_OPENAI_SPEAKER_MODEL
PROD_DEFAULT_OPENAI_SUMMARY_MODEL = config_constants.PROD_DEFAULT_OPENAI_SUMMARY_MODEL
PROD_DEFAULT_OPENAI_CLEANING_MODEL = config_constants.PROD_DEFAULT_OPENAI_CLEANING_MODEL

# Validation constants
VALID_WHISPER_MODELS = config_constants.VALID_WHISPER_MODELS
VALID_LOG_LEVELS = config_constants.VALID_LOG_LEVELS
MAX_RUN_ID_LENGTH = config_constants.MAX_RUN_ID_LENGTH
MAX_METADATA_SUBDIRECTORY_LENGTH = config_constants.MAX_METADATA_SUBDIRECTORY_LENGTH

# Summarization defaults
DEFAULT_SUMMARY_BATCH_SIZE = config_constants.DEFAULT_SUMMARY_BATCH_SIZE
DEFAULT_SUMMARY_MAX_WORKERS_CPU = config_constants.DEFAULT_SUMMARY_MAX_WORKERS_CPU
DEFAULT_SUMMARY_MAX_WORKERS_CPU_TEST = config_constants.DEFAULT_SUMMARY_MAX_WORKERS_CPU_TEST
DEFAULT_SUMMARY_MAX_WORKERS_GPU = config_constants.DEFAULT_SUMMARY_MAX_WORKERS_GPU
DEFAULT_SUMMARY_MAX_WORKERS_GPU_TEST = config_constants.DEFAULT_SUMMARY_MAX_WORKERS_GPU_TEST
DEFAULT_SUMMARY_CHUNK_SIZE = config_constants.DEFAULT_SUMMARY_CHUNK_SIZE
DEFAULT_SUMMARY_WORD_CHUNK_SIZE = config_constants.DEFAULT_SUMMARY_WORD_CHUNK_SIZE
DEFAULT_SUMMARY_WORD_OVERLAP = config_constants.DEFAULT_SUMMARY_WORD_OVERLAP


def _get_default_summary_mode_id() -> Optional[str]:
    """Get default summarization mode ID (RFC-044) based on environment.

    Returns:
        None in test environments (to avoid altering unit test defaults),
        dev or production mode ID otherwise.
    """
    if _is_test_environment():
        return None
    profile = (os.environ.get("PODCAST_SCRAPER_PROFILE") or "").strip().lower()
    if profile in ("dev", "development", "local"):
        return getattr(config_constants, "DEV_DEFAULT_SUMMARY_MODE_ID", None) or getattr(
            config_constants, "PROD_DEFAULT_SUMMARY_MODE_ID", None
        )
    if profile and profile not in ("prod", "production"):
        warnings.warn(
            f"Unknown PODCAST_SCRAPER_PROFILE={profile!r}; defaulting to production defaults",
            RuntimeWarning,
        )
    return getattr(config_constants, "PROD_DEFAULT_SUMMARY_MODE_ID", None)


def _get_default_openai_transcription_model() -> str:
    """Get default OpenAI transcription model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return TEST_DEFAULT_OPENAI_TRANSCRIPTION_MODEL
    return PROD_DEFAULT_OPENAI_TRANSCRIPTION_MODEL


def _get_default_openai_speaker_model() -> str:
    """Get default OpenAI speaker detection model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return TEST_DEFAULT_OPENAI_SPEAKER_MODEL
    return PROD_DEFAULT_OPENAI_SPEAKER_MODEL


def _get_default_openai_summary_model() -> str:
    """Get default OpenAI summarization model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return TEST_DEFAULT_OPENAI_SUMMARY_MODEL
    return PROD_DEFAULT_OPENAI_SUMMARY_MODEL


def _get_default_gemini_transcription_model() -> str:
    """Get default Gemini transcription model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_GEMINI_TRANSCRIPTION_MODEL
    return config_constants.PROD_DEFAULT_GEMINI_TRANSCRIPTION_MODEL


def _get_default_gemini_speaker_model() -> str:
    """Get default Gemini speaker detection model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_GEMINI_SPEAKER_MODEL
    return config_constants.PROD_DEFAULT_GEMINI_SPEAKER_MODEL


def _get_default_gemini_summary_model() -> str:
    """Get default Gemini summarization model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_GEMINI_SUMMARY_MODEL
    return config_constants.PROD_DEFAULT_GEMINI_SUMMARY_MODEL


def _get_default_anthropic_transcription_model() -> str:
    """Get default Anthropic transcription model based on environment.

    Note: Anthropic doesn't support native audio transcription.
    This is a placeholder for API compatibility.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_ANTHROPIC_TRANSCRIPTION_MODEL
    return config_constants.PROD_DEFAULT_ANTHROPIC_TRANSCRIPTION_MODEL


def _get_default_anthropic_speaker_model() -> str:
    """Get default Anthropic speaker detection model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_ANTHROPIC_SPEAKER_MODEL
    return config_constants.PROD_DEFAULT_ANTHROPIC_SPEAKER_MODEL


def _get_default_anthropic_summary_model() -> str:
    """Get default Anthropic summarization model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_ANTHROPIC_SUMMARY_MODEL
    return config_constants.PROD_DEFAULT_ANTHROPIC_SUMMARY_MODEL


def _get_default_deepseek_speaker_model() -> str:
    """Get default DeepSeek speaker detection model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_DEEPSEEK_SPEAKER_MODEL
    return config_constants.PROD_DEFAULT_DEEPSEEK_SPEAKER_MODEL


def _get_default_deepseek_summary_model() -> str:
    """Get default DeepSeek summarization model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_DEEPSEEK_SUMMARY_MODEL
    return config_constants.PROD_DEFAULT_DEEPSEEK_SUMMARY_MODEL


def _get_default_grok_speaker_model() -> str:
    """Get default Grok speaker detection model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_GROK_SPEAKER_MODEL
    return config_constants.PROD_DEFAULT_GROK_SPEAKER_MODEL


def _get_default_grok_summary_model() -> str:
    """Get default Grok summarization model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_GROK_SUMMARY_MODEL
    return config_constants.PROD_DEFAULT_GROK_SUMMARY_MODEL


def _get_default_ollama_speaker_model() -> str:
    """Get default Ollama speaker detection model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_OLLAMA_SPEAKER_MODEL
    return config_constants.PROD_DEFAULT_OLLAMA_SPEAKER_MODEL


def _get_default_ollama_summary_model() -> str:
    """Get default Ollama summarization model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_OLLAMA_SUMMARY_MODEL
    return config_constants.PROD_DEFAULT_OLLAMA_SUMMARY_MODEL


def _get_default_mistral_transcription_model() -> str:
    """Get default Mistral transcription model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_MISTRAL_TRANSCRIPTION_MODEL
    return config_constants.PROD_DEFAULT_MISTRAL_TRANSCRIPTION_MODEL


def _get_default_mistral_speaker_model() -> str:
    """Get default Mistral speaker detection model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_MISTRAL_SPEAKER_MODEL
    return config_constants.PROD_DEFAULT_MISTRAL_SPEAKER_MODEL


def _get_default_mistral_summary_model() -> str:
    """Get default Mistral summarization model based on environment.

    Returns:
        Test default if in test environment, production default otherwise.
    """
    if _is_test_environment():
        return config_constants.TEST_DEFAULT_MISTRAL_SUMMARY_MODEL
    return config_constants.PROD_DEFAULT_MISTRAL_SUMMARY_MODEL


def _get_default_openai_cleaning_model() -> str:
    """Get default OpenAI transcript-cleaning model based on environment.

    Returns:
        Test default in test/CI environments; production default otherwise.
        Override with ``openai_cleaning_model`` (e.g. ``gpt-3.5-turbo``) to reduce cost.
    """
    if _is_test_environment():
        return TEST_DEFAULT_OPENAI_CLEANING_MODEL
    return PROD_DEFAULT_OPENAI_CLEANING_MODEL


def _get_default_anthropic_cleaning_model() -> str:
    """Get default Anthropic cleaning model (cheaper than summary model).

    Returns:
        ``claude-haiku-4-5`` (Anthropic alias for Claude Haiku 4.5)
    """
    return "claude-haiku-4-5"


def _get_default_gemini_cleaning_model() -> str:
    """Get default Gemini cleaning model (cheaper than summary model).

    Returns:
        ``gemini-2.5-flash-lite`` (``gemini-1.5-flash`` is not available on current Generative API)
    """
    return "gemini-2.5-flash-lite"


def _get_default_mistral_cleaning_model() -> str:
    """Get default Mistral cleaning model (cheaper than summary model).

    Returns:
        'mistral-small' (cheaper model for cleaning)
    """
    return "mistral-small"


def _get_default_deepseek_cleaning_model() -> str:
    """Get default DeepSeek cleaning model (cheaper than summary model).

    Returns:
        'deepseek-chat' (same as summary, but can be overridden)
    """
    return "deepseek-chat"


def _get_default_ollama_cleaning_model() -> str:
    """Get default Ollama cleaning model (smaller than summary model).

    Returns:
        'llama3.1:8b' (smaller model for cleaning)
    """
    return "llama3.1:8b"


def _get_default_grok_cleaning_model() -> str:
    """Get default Grok cleaning model (cheaper than summary model).

    Returns:
        'grok-beta' (same as summary, but can be overridden)
    """
    return "grok-beta"


def _get_default_ner_model() -> str:
    """Get default spaCy NER model based on environment.

    Returns:
        Test default (en_core_web_sm) if in test environment,
        production default (en_core_web_trf) otherwise.
    """
    if _is_test_environment():
        return TEST_DEFAULT_NER_MODEL
    return PROD_DEFAULT_NER_MODEL


# Set DEFAULT_NER_MODEL to use the environment-aware function
DEFAULT_NER_MODEL = _get_default_ner_model()


# Default generation parameters (aligned with baseline_ml_prod_authority_v1)
# Note: These are NOT in config_constants.py as they're specific to Config defaults
# These are used as defaults when summary_map_params/summary_reduce_params are not provided
# Map stage defaults (for chunk-level summarization) - Pegasus-CNN settings
DEFAULT_MAP_MAX_NEW_TOKENS = 200  # Production baseline: 200 tokens for map stage
DEFAULT_MAP_MIN_NEW_TOKENS = 80  # Production baseline: 80 tokens for map stage
DEFAULT_MAP_NUM_BEAMS = 6  # Production baseline: 6 beams (Pegasus-CNN optimized)
DEFAULT_MAP_NO_REPEAT_NGRAM_SIZE = 3  # Production baseline: 3 (prevents repetition)
DEFAULT_MAP_LENGTH_PENALTY = 1.0  # Production baseline: 1.0 (no length bias)
DEFAULT_MAP_EARLY_STOPPING = True  # Production baseline: true
DEFAULT_MAP_REPETITION_PENALTY = 1.1  # Production baseline: 1.1 (Pegasus-CNN optimized)

# Reduce stage defaults (for episode-level summarization) - LED-base settings
# Note: These defaults are dynamically capped based on input size to prevent expansion
# When reduce input is short (< 500 tokens), min_new_tokens is overridden to 0
DEFAULT_REDUCE_MAX_NEW_TOKENS = 650  # Production baseline: 650 tokens for reduce stage
DEFAULT_REDUCE_MIN_NEW_TOKENS = (
    220  # Production baseline: 220 tokens (dynamically capped to prevent expansion)
)
DEFAULT_REDUCE_NUM_BEAMS = 4  # Production baseline: 4 beams
DEFAULT_REDUCE_NO_REPEAT_NGRAM_SIZE = 3  # Production baseline: 3
DEFAULT_REDUCE_LENGTH_PENALTY = 1.0  # Production baseline: 1.0
DEFAULT_REDUCE_EARLY_STOPPING = False  # Production baseline: false (ensures min_new_tokens)
DEFAULT_REDUCE_REPETITION_PENALTY = 1.12  # Production baseline: 1.12 (LED-base optimized)


# Default tokenization limits (moved from model defaults)
def _get_default_summary_tokenize() -> Dict[str, Any]:
    """Get default tokenization settings for local transformers summarization.

    In production, defaults are sourced from the promoted production mode
    (RFC-044) to ensure baseline == app behavior.

    In tests, use safe defaults to keep unit tests stable and independent of
    production mode registry contents.
    """
    if not _is_test_environment():
        profile = (os.environ.get("PODCAST_SCRAPER_PROFILE") or "").strip().lower()
        mode_id = (
            getattr(config_constants, "DEV_DEFAULT_SUMMARY_MODE_ID", None)
            if profile in ("dev", "development", "local")
            else getattr(config_constants, "PROD_DEFAULT_SUMMARY_MODE_ID", None)
        )
        if mode_id:
            try:
                from podcast_scraper.providers.ml.model_registry import ModelRegistry

                return dict(ModelRegistry.get_mode_configuration(mode_id).tokenize)
            except Exception:
                # Fall back to safe defaults if registry is unavailable or missing mode.
                pass
    return {
        "map_max_input_tokens": 1024,
        "reduce_max_input_tokens": 4096,
        "truncation": True,
    }


# Default distill parameters (for final compression pass)
DEFAULT_DISTILL_MAX_TOKENS = 200
DEFAULT_DISTILL_MIN_TOKENS = 120
DEFAULT_DISTILL_NUM_BEAMS = 4
DEFAULT_DISTILL_NO_REPEAT_NGRAM_SIZE = 6
DEFAULT_DISTILL_LENGTH_PENALTY = 0.75
DEFAULT_DISTILL_REPETITION_PENALTY = 1.3  # Same as map/reduce for consistency

# Default token overlap for chunking
DEFAULT_TOKEN_OVERLAP = 200


def _default_summary_prompt_params() -> Dict[str, Any]:
    """Defaults for shared summarization Jinja params (JSON bullet templates, etc.)."""
    return {
        "bullet_min": config_constants.DEFAULT_SUMMARY_BULLET_MIN,
        "max_words_per_bullet": 45,
    }


# Summary providers for which default GIL evidence can align to the same backend (validator).
GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS: frozenset[str] = frozenset(
    {
        "openai",
        "gemini",
        "anthropic",
        "mistral",
        "deepseek",
        "grok",
        "ollama",
        "hybrid_ml",
    }
)

# Top-level keys still allowed in CLI YAML merge before ``Config.model_validate``;
# stripped or mapped in ``Config._handle_deprecated_fields``.
DEPRECATED_CONFIG_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"multi_feed_soft_fail_exit_zero"})


class Config(BaseModel):
    """Configuration model for podcast scraping pipeline.

    This Pydantic model defines all configuration options for podcast_scraper, with
    automatic validation and type checking. Configuration can be created programmatically
    or loaded from JSON/YAML files using `load_config_file()`.

    The configuration is organized into several categories:

    - **RSS Feed**: Source feed URL and episode limits
    - **Output**: Directory structure, file naming, and run management
    - **HTTP**: Request behavior, timeouts, and user agents
    - **Transcription**: Whisper model selection and language settings
    - **Screenplay**: Speaker detection and formatting options
    - **Metadata**: Episode metadata generation settings
    - **Summarization**: AI-powered episode summary generation
    - **Processing**: Parallelization and execution control
    - **Logging**: Log levels and output destinations

    All fields support validation and provide sensible defaults. The model is immutable
    (frozen) after creation to prevent accidental modification.

    Attributes:
        rss_url: RSS feed URL to scrape. Required unless loading from config file.
        output_dir: Output directory path. Auto-generated from RSS URL if not provided.
        max_episodes: Maximum number of episodes to process. None processes all episodes.
        episode_order: ``newest`` (feed document order) or ``oldest`` (reversed) before filters.
        episode_offset: Items to skip after order and optional date filter (GitHub #521).
        episode_since: Optional inclusive lower bound on episode pubDate (calendar date).
        episode_until: Optional inclusive upper bound on episode pubDate (calendar date).
        user_agent: HTTP User-Agent header for requests.
        timeout: Request timeout in seconds (minimum: 1).
        delay_ms: Delay between requests in milliseconds.
        http_retry_total: Max urllib3 retries for media/transcript downloads (default 8).
        http_backoff_factor: Backoff factor for HTTP retries (default 1.0).
        rss_retry_total: Max urllib3 retries for RSS feed fetches (default 5).
        rss_backoff_factor: Backoff factor for RSS retries (default 1.0).
        episode_retry_max: App-level retries per episode after urllib3 exhaustion (default 1).
        episode_retry_delay_sec: Initial delay between episode retries (default 10.0).
        host_request_interval_ms: Min ms between requests to same host (0=off; Issue #522).
        host_max_concurrent: Max concurrent downloads per host (0=unlimited; Issue #522).
        circuit_breaker_enabled: Enable HTTP circuit breaker (Issue #522).
        circuit_breaker_failure_threshold: Failures in window before open.
        circuit_breaker_window_seconds: Rolling window for breaker counts.
        circuit_breaker_cooldown_seconds: Cooldown while circuit is open.
        circuit_breaker_scope: ``feed`` (RSS URL) or ``host`` (netloc).
        rss_conditional_get: RSS If-None-Match / If-Modified-Since (Issue #522).
        rss_cache_dir: Directory for RSS conditional cache (optional).
        prefer_types: Preferred transcript types or extensions (e.g., ["text/vtt", ".srt"]).
        transcribe_missing: Enable Whisper transcription for episodes without transcripts
            (default: True). Set to False to only download existing transcripts.
        whisper_model: Whisper model name (e.g., "base", "small", "medium").
        whisper_device: Device for Whisper execution ("cpu", "cuda", "mps", or None for auto).
        screenplay: Format transcripts as screenplay with speaker labels (Whisper-only;
            see Field description / GitHub #562 for API transcription).
        screenplay_gap_s: Minimum gap in seconds between speaker segments.
        screenplay_num_speakers: Number of speakers for Whisper diarization.
        screenplay_speaker_names: Manual speaker names list (overrides auto-detection).
        run_id: Optional run identifier. Use "auto" for timestamp-based ID.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path for file output.
        workers: Number of parallel download workers.
        skip_existing: Skip episodes with existing output files.
        backfill_transcript_segments: Re-transcribe when ``.segments.json`` missing (#542);
            ``speaker_id`` when segment rows carry speaker labels (#541).
        clean_output: Remove output directory before processing.
        reuse_media: Reuse existing media files instead of re-downloading.
        dry_run: Preview planned work without saving files.
        preload_models: Preload ML models at startup if configured to use them (default: True).
        language: Language code for transcription (e.g., "en", "fr", "de").
        ner_model: spaCy NER model name for speaker detection.
        auto_speakers: Enable automatic speaker name detection using NER.
        cache_detected_hosts: Cache detected host names across episodes.
        generate_metadata: Generate per-episode metadata documents.
        metadata_format: Metadata file format ("json" or "yaml").
        metadata_subdirectory: Optional subdirectory for metadata files.
        generate_summaries: Generate episode summaries using AI models.
        summary_provider: Summary generation provider ("transformers" or "openai").
        summary_model: Model identifier for MAP-phase summarization.
        summary_reduce_model: Optional separate model for REDUCE-phase summarization.
        summary_device: Device for model execution ("cpu", "cuda", "mps", or None).
        summary_batch_size: Batch size for episode-level parallel processing (episodes in parallel).
        summary_chunk_parallelism: Number of chunks to process in parallel within a single episode
            (CPU-bound, local providers only).
        summary_chunk_size: Chunk size in tokens for long transcripts.
        summary_word_chunk_size: Chunk size in words for word-based chunking.
        summary_word_overlap: Overlap in words for word-based chunking.
        summary_cache_dir: Custom cache directory for transformer models.
        summary_prompt: Optional custom prompt for summarization.
        save_cleaned_transcript: Save cleaned transcript to separate file.
        speaker_detector_provider: Speaker detection provider type
            ("spacy" or "openai").
        transcription_provider: Transcription provider type ("whisper" or "openai").
        openai_api_key: OpenAI API key (loaded from environment variable or .env file).

    Example:
        Create configuration programmatically:

        >>> from podcast_scraper import Config
        >>> cfg = Config(
        ...     rss_url="https://example.com/feed.xml",
        ...     output_dir="./transcripts",
        ...     max_episodes=50,
        ...     transcribe_missing=True,
        ...     whisper_model="base"
        ... )

    Example:
        Load configuration from file:

        >>> from podcast_scraper import Config, load_config_file
        >>> config_dict = load_config_file("config.yaml")
        >>> cfg = Config(**config_dict)

    Example:
        Configuration with metadata and summaries:

        >>> cfg = Config(
        ...     rss_url="https://example.com/feed.xml",
        ...     generate_metadata=True,
        ...     metadata_format="yaml",
        ...     generate_summaries=True,
        ...     summary_model="facebook/bart-base",
        ...     summary_device="mps"
        ... )

    See Also:
        - `load_config_file()`: Load configuration from JSON/YAML files
        - `run_pipeline()`: Execute pipeline with configuration
        - API Reference: https://chipi.github.io/podcast_scraper/api/configuration/
    """

    rss_url: Optional[str] = Field(default=None, alias="rss")
    rss_urls: Optional[List[str]] = Field(
        default=None,
        validation_alias=AliasChoices("rss_urls", "feeds"),
        description=(
            "Multiple RSS feed URLs (GitHub #440). When two or more are set, output_dir must "
            "be the corpus parent; each feed is written under output_dir/feeds/<stable_name>/."
        ),
    )
    multi_feed_strict: bool = Field(
        default=False,
        description=(
            "Multi-feed only (two or more rss_urls; GitHub #559). When False (default), a run "
            "counts as successful if every failed feed is a **soft** failure; aggregated text "
            "is on ``ServiceResult.soft_failures``. When True, **strict** CI semantics: any feed "
            "failure yields ``success=False`` / non-zero exit even when failures are "
            "soft-classified. Hard failures are always failures. In YAML/JSON dicts passed to "
            "``Config.model_validate``, deprecated key ``multi_feed_soft_fail_exit_zero`` is "
            "accepted and mapped to ``multi_feed_strict = not`` that legacy boolean. "
            "Programmatic ``Config(...)`` must use ``multi_feed_strict`` only (the legacy name "
            "is not a model field; ``extra=forbid``)."
        ),
    )
    incident_log_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional absolute path to ``corpus_incidents.jsonl`` (GitHub #557). When unset, "
            "``run_pipeline`` defaults to ``<effective_output_dir>/corpus_incidents.jsonl``. "
            "Multi-feed ``service.run`` sets this to the corpus parent so all feeds append to "
            "one log."
        ),
    )
    output_dir: Optional[str] = Field(
        default=None,
        alias="output_dir",
        description="Output directory path. Auto-generated from RSS URL if not provided. "
        "Can be set via OUTPUT_DIR environment variable.",
    )
    max_episodes: Optional[int] = Field(default=None, alias="max_episodes")
    episode_order: Literal["newest", "oldest"] = Field(
        default="newest",
        alias="episode_order",
        description=(
            "RSS item order before date filter and offset/limit: newest keeps document order; "
            "oldest reverses (GitHub #521)."
        ),
    )
    episode_offset: int = Field(
        default=0,
        alias="episode_offset",
        description=(
            "Skip this many items after order and optional date filter, before max_episodes "
            "(GitHub #521)."
        ),
    )
    episode_since: Optional[date] = Field(
        default=None,
        alias="episode_since",
        description=(
            "Keep items with pubDate on or after this calendar date (UTC date for "
            "timezone-aware pubDate). Items without a parseable date are kept; see logs "
            "(GitHub #521)."
        ),
    )
    episode_until: Optional[date] = Field(
        default=None,
        alias="episode_until",
        description=(
            "Keep items with pubDate on or before this calendar date (UTC date for "
            "timezone-aware pubDate). Items without a parseable date are kept (GitHub #521)."
        ),
    )
    user_agent: str = Field(default=DEFAULT_USER_AGENT, alias="user_agent")
    timeout: int = Field(default=DEFAULT_TIMEOUT_SECONDS, alias="timeout")
    transcription_timeout: Optional[int] = Field(
        default=config_constants.DEFAULT_TRANSCRIPTION_TIMEOUT_SECONDS,
        alias="transcription_timeout",
        description=(
            "Timeout in seconds for transcription operations (default: 1800 = 30 minutes). "
            "Set to None to disable timeout. Prevents hangs on very long audio files."
        ),
    )
    summarization_timeout: Optional[int] = Field(
        default=config_constants.DEFAULT_SUMMARIZATION_TIMEOUT_SECONDS,
        alias="summarization_timeout",
        description=(
            "Timeout in seconds for summarization operations (default: 600 = 10 minutes). "
            "Set to None to disable timeout. Prevents hangs on very long transcripts."
        ),
    )
    delay_ms: int = Field(default=0, alias="delay_ms")
    http_retry_total: int = Field(
        default=8,
        alias="http_retry_total",
        ge=0,
        le=20,
        description=(
            "Max urllib3 retries for media/transcript downloads "
            "(default 8). RSS feeds use rss_retry_total."
        ),
    )
    http_backoff_factor: float = Field(
        default=1.0,
        alias="http_backoff_factor",
        ge=0.0,
        le=10.0,
        description=(
            "Exponential backoff factor for HTTP retries "
            "(default 1.0). Delay = factor * 2^(attempt-1)."
        ),
    )
    rss_retry_total: int = Field(
        default=5,
        alias="rss_retry_total",
        ge=0,
        le=20,
        description=(
            "Max urllib3 retries for RSS feed fetches " "(default 5). Caps total wait at ~31 s."
        ),
    )
    rss_backoff_factor: float = Field(
        default=1.0,
        alias="rss_backoff_factor",
        ge=0.0,
        le=10.0,
        description=(
            "Exponential backoff factor for RSS feed retries "
            "(default 1.0). Delay = factor * 2^(attempt-1)."
        ),
    )
    episode_retry_max: int = Field(
        default=1,
        alias="episode_retry_max",
        ge=0,
        le=10,
        description=(
            "Application-level retries per episode after all "
            "urllib3 retries are exhausted (default 1). Only "
            "retries on transient network errors. Set to 0 to "
            "disable."
        ),
    )
    episode_retry_delay_sec: float = Field(
        default=10.0,
        alias="episode_retry_delay_sec",
        ge=0.0,
        le=120.0,
        description=(
            "Initial delay in seconds between episode-level "
            "retries (default 10.0). Uses exponential backoff."
        ),
    )
    host_request_interval_ms: int = Field(
        default=0,
        alias="host_request_interval_ms",
        ge=0,
        le=600_000,
        description=(
            "Minimum milliseconds between HTTP requests to the same host (0=off). "
            "Issue #522 fair usage."
        ),
    )
    host_max_concurrent: int = Field(
        default=0,
        alias="host_max_concurrent",
        ge=0,
        le=64,
        description="Maximum concurrent downloads per host (0=unlimited). Issue #522.",
    )
    circuit_breaker_enabled: bool = Field(
        default=False,
        alias="circuit_breaker_enabled",
        description="Enable per-feed or per-host circuit breaker for HTTP (Issue #522).",
    )
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        alias="circuit_breaker_failure_threshold",
        ge=1,
        le=100,
        description="Failures in the rolling window before the circuit opens.",
    )
    circuit_breaker_window_seconds: int = Field(
        default=60,
        alias="circuit_breaker_window_seconds",
        ge=1,
        le=86400,
        description="Rolling window (seconds) for circuit breaker failure counts.",
    )
    circuit_breaker_cooldown_seconds: int = Field(
        default=120,
        alias="circuit_breaker_cooldown_seconds",
        ge=1,
        le=86400,
        description="Cooldown (seconds) while the circuit stays open before a probe.",
    )
    circuit_breaker_scope: Literal["feed", "host"] = Field(
        default="feed",
        alias="circuit_breaker_scope",
        description="Scope key: rss_url (feed) or request host (host).",
    )
    rss_conditional_get: bool = Field(
        default=False,
        alias="rss_conditional_get",
        description="Use If-None-Match / If-Modified-Since on RSS GET (Issue #522).",
    )
    rss_cache_dir: Optional[str] = Field(
        default=None,
        alias="rss_cache_dir",
        description=(
            "Directory for RSS conditional validators/body cache. "
            "Default: PODCAST_SCRAPER_RSS_CACHE_DIR or ~/.cache/podcast_scraper/rss."
        ),
    )
    prefer_types: List[str] = Field(default_factory=list, alias="prefer_type")
    transcribe_missing: bool = Field(default=True, alias="transcribe_missing")
    whisper_model: str = Field(default="base.en", alias="whisper_model")
    whisper_device: Optional[str] = Field(default=None, alias="whisper_device")
    screenplay: bool = Field(
        default=False,
        alias="screenplay",
        description=(
            "Format transcripts as screenplay with speaker labels. Only "
            "`transcription_provider='whisper'` applies screenplay; OpenAI / Gemini / "
            "Mistral audio paths emit plain text, so `screenplay: true` is coerced to "
            "`false` at validation with a single INFO (GitHub #562)."
        ),
    )
    screenplay_gap_s: float = Field(default=DEFAULT_SCREENPLAY_GAP_SECONDS, alias="screenplay_gap")
    screenplay_num_speakers: int = Field(default=DEFAULT_NUM_SPEAKERS, alias="num_speakers")
    screenplay_speaker_names: List[str] = Field(default_factory=list, alias="speaker_names")
    run_id: Optional[str] = Field(default=None, alias="run_id")
    seed: Optional[int] = Field(
        default=None,
        alias="seed",
        description=(
            "Random seed for reproducibility (Issue #429). If set, torch, numpy, and "
            "transformers seeds are set at pipeline start. MPS (Apple Silicon) may still "
            "be non-deterministic; see docs for details."
        ),
    )
    log_level: str = Field(default=DEFAULT_LOG_LEVEL, alias="log_level")
    log_file: Optional[str] = Field(
        default=None,
        alias="log_file",
        description="Path to log file (logs will be written to both console and file). "
        "Can be set via LOG_FILE environment variable.",
    )
    json_logs: bool = Field(
        default=False,
        alias="json_logs",
        description="Output structured JSON logs for monitoring/alerting (Issue #379)",
    )
    workers: int = Field(default=DEFAULT_WORKERS, alias="workers")
    fail_fast: bool = Field(
        default=False,
        alias="fail_fast",
        description="Stop on first episode failure (Issue #379)",
    )
    max_failures: Optional[int] = Field(
        default=None,
        alias="max_failures",
        description="Stop after N episode failures (Issue #379). None = no limit.",
    )
    skip_existing: bool = Field(default=False, alias="skip_existing")
    backfill_transcript_segments: bool = Field(
        default=False,
        alias="backfill_transcript_segments",
        description=(
            "When True with generate_gi: if an existing Whisper ``transcripts/*.txt`` "
            "has no sibling ``.segments.json``, do not treat transcription as complete "
            "under ``skip_existing`` — re-run transcription so GI quote "
            "``timestamp_*_ms`` can be populated (GitHub #542); ``speaker_id`` is set only "
            "when segment rows include speaker labels (GitHub #541). Append mode also "
            "treats episodes as incomplete until the sidecar exists. Default False "
            "preserves legacy skip behavior."
        ),
    )
    append: bool = Field(
        default=False,
        alias="append",
        description=(
            "Resume into a stable run directory and skip episodes that already have valid "
            "artifacts (GitHub #444). Uses episode_id + on-disk metadata validation; "
            "incompatible with clean_output."
        ),
    )
    clean_output: bool = Field(default=False, alias="clean_output")
    reuse_media: bool = Field(
        default=False,
        alias="reuse_media",
        description="Reuse existing media files instead of re-downloading (for faster testing)",
    )
    dry_run: bool = Field(default=False, alias="dry_run")
    monitor: bool = Field(
        default=False,
        alias="monitor",
        description=(
            "Spawn a subprocess with a live RSS/CPU/stage dashboard (RFC-065, GitHub #512). "
            "Writes .pipeline_status.json under the output directory."
        ),
    )
    memray: bool = Field(
        default=False,
        alias="memray",
        description=(
            "Re-exec the CLI or service under memray for heap profiling (RFC-065 Phase 3; "
            "optional extra .[monitor]). Sets PODCAST_SCRAPER_MEMRAY_ACTIVE=1 in the child."
        ),
    )
    memray_output: Optional[str] = Field(
        default=None,
        alias="memray_output",
        description=(
            "Destination .bin for memray run (default: <output_dir>/debug/memray_<timestamp>.bin "
            "or cwd/debug when output_dir is unset)."
        ),
    )
    preload_models: bool = Field(
        default=True,
        alias="preload_models",
        description="Preload ML models at startup if configured to use them (default: True)",
    )
    language: str = Field(default=DEFAULT_LANGUAGE, alias="language")
    ner_model: Optional[str] = Field(
        default_factory=_get_default_ner_model,
        alias="ner_model",
        description=(
            "spaCy NER model name for speaker detection. "
            "Defaults to en_core_web_sm (dev) or en_core_web_trf (prod) based on environment."
        ),
    )
    auto_speakers: bool = Field(default=True, alias="auto_speakers")
    cache_detected_hosts: bool = Field(default=True, alias="cache_detected_hosts")
    known_hosts: List[str] = Field(
        default_factory=list,
        alias="known_hosts",
        description=(
            "Known host names for the podcast (show-level override). "
            "Useful when RSS metadata doesn't provide clean host names. "
            "When the RSS author is an organization (e.g. NPR, BBC), set this to "
            "the actual host names if auto-detection finds none. "
            "These will be used as hosts if auto-detection fails or finds no hosts."
        ),
    )
    # Provider selection fields (Stage 0: Foundation)
    speaker_detector_provider: Literal[
        "spacy", "openai", "gemini", "mistral", "grok", "deepseek", "anthropic", "ollama"
    ] = Field(
        default="spacy",
        alias="speaker_detector_provider",
        description="Speaker detection provider type (default: 'spacy' for spaCy NER).",
    )
    transcription_provider: Literal["whisper", "openai", "gemini", "mistral"] = Field(
        default="whisper",
        alias="transcription_provider",
        description="Transcription provider type (default: 'whisper' for local Whisper)",
    )
    transcription_parallelism: int = Field(
        default=1,
        alias="transcription_parallelism",
        description=(
            "Episode-level parallelism: Number of episodes to transcribe in parallel "
            "(default: 1 for sequential). "
            "For Whisper provider: Values > 1 are EXPERIMENTAL and not production-ready. "
            "May cause memory/GPU contention. Use with caution. "
            "For API providers (OpenAI, etc.): Uses parallelism for parallel API calls."
        ),
    )
    transcription_queue_size: int = Field(
        default=50,
        alias="transcription_queue_size",
        description=(
            "Maximum size of the transcription job queue. "
            "When the queue is full, downloads will block until space is available (backpressure). "
            "Prevents unbounded memory growth when downloads outpace transcription. "
            "Default: 50 episodes."
        ),
    )
    transcription_device: Optional[str] = Field(
        default=None,
        alias="transcription_device",
        description=(
            "Device for transcription stage (CPU, CUDA, MPS, or None for auto-detection). "
            "Overrides provider-specific device (e.g., whisper_device) if set. "
            "Allows CPU/GPU mix to regain overlap (Issue #387). "
            "Valid values: 'cpu', 'cuda', 'mps', or None/empty for auto-detect."
        ),
    )
    processing_parallelism: int = Field(
        default=2,
        alias="processing_parallelism",
        description=(
            "Episode-level parallelism: Number of episodes to process "
            "(metadata/summarization) in parallel (default: 2)"
        ),
    )
    # OpenAI API configuration (RFC-013)
    openai_api_key: Optional[str] = Field(
        default=None,
        alias="openai_api_key",
        description="OpenAI API key (prefer OPENAI_API_KEY env var or .env file)",
    )
    openai_api_base: Optional[str] = Field(
        default=None,
        alias="openai_api_base",
        description="OpenAI API base URL (e.g., 'https://api.openai.com/v1' or custom endpoint). "
        "Can be set via OPENAI_API_BASE environment variable. "
        "Used for E2E testing with mock servers.",
    )
    openai_transcription_model: str = Field(
        default_factory=_get_default_openai_transcription_model,
        alias="openai_transcription_model",
        description="OpenAI Whisper API model version (default: environment-based)",
    )
    openai_speaker_model: str = Field(
        default_factory=_get_default_openai_speaker_model,
        alias="openai_speaker_model",
        description="OpenAI model for speaker detection (default: environment-based)",
    )
    openai_summary_model: str = Field(
        default_factory=_get_default_openai_summary_model,
        alias="openai_summary_model",
        description="OpenAI model for summarization (default: environment-based)",
    )
    openai_insight_model: Optional[str] = Field(
        default=None,
        alias="openai_insight_model",
        description=(
            "Optional OpenAI chat model used only for GIL generate_insights when "
            "gi_insight_source is provider; when unset, openai_summary_model is used."
        ),
    )
    openai_temperature: float = Field(
        default=0.3,
        alias="openai_temperature",
        description="Temperature for OpenAI generation (0.0-2.0, lower = more deterministic)",
    )
    openai_summary_seed: Optional[int] = Field(
        default=None,
        alias="openai_summary_seed",
        description=(
            "Optional deterministic-sampling seed for OpenAI summarization calls. "
            "Combined with temperature=0, yields approximately reproducible outputs "
            "(same seed + prompt + model → near-identical output). Primarily used "
            "by the autoresearch ratchet to stabilise smoke-scale scoring."
        ),
    )
    openai_cleaning_model: str = Field(
        default_factory=_get_default_openai_cleaning_model,
        alias="openai_cleaning_model",
        description=(
            "OpenAI model for hybrid/LLM transcript cleaning before summarization "
            "(default: gpt-4o-mini; test/prod via config constants). "
            "Set to e.g. gpt-3.5-turbo for lower cost."
        ),
    )
    openai_cleaning_temperature: float = Field(
        default=0.2,
        alias="openai_cleaning_temperature",
        description="Temperature for OpenAI cleaning (0.0-2.0, default: 0.2, lower than summarization)",  # noqa: E501
    )
    openai_max_tokens: Optional[int] = Field(
        default=None,
        alias="openai_max_tokens",
        description="Max tokens for OpenAI generation (None = model default)",
    )
    # Prompt configuration (RFC-017)
    openai_summary_system_prompt: str = Field(
        default="openai/summarization/system_bullets_v1",
        alias="openai_summary_system_prompt",
        description=(
            "System prompt for summarization (default: JSON bullet schema). "
            "Set to 'openai/summarization/system_v1' for legacy paragraph-style. "
            "Uses prompt_store (RFC-017); shared templates: prompts/shared/summarization/."
        ),
    )
    openai_summary_user_prompt: str = Field(
        default="openai/summarization/bullets_json_v1",
        alias="openai_summary_user_prompt",
        description=(
            "User prompt for summarization (default: JSON bullets). "
            "Use 'openai/summarization/long_v1' for paragraphs. Uses prompt_store (RFC-017)."
        ),
    )
    summary_prompt_params: Dict[str, Any] = Field(
        default_factory=_default_summary_prompt_params,
        alias="summary_prompt_params",
        description=(
            "Template parameters for summary prompts (Jinja). Defaults include bullet_min "
            "and max_words_per_bullet for shared bullet templates; optional bullet_max caps count."
        ),
    )
    openai_speaker_system_prompt: Optional[str] = Field(
        default=None,
        alias="openai_speaker_system_prompt",
        description="System prompt name for speaker detection/NER. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    openai_speaker_user_prompt: str = Field(
        default="openai/ner/guest_host_v1",
        alias="openai_speaker_user_prompt",
        description="User prompt name for speaker detection/NER. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    ner_prompt_params: Dict[str, Any] = Field(
        default_factory=dict,
        alias="ner_prompt_params",
        description="Template parameters for NER prompts (passed to Jinja2 templates).",
    )
    # Gemini API configuration (Issue #194)
    gemini_api_key: Optional[str] = Field(
        default=None,
        alias="gemini_api_key",
        description="Google AI API key (prefer GEMINI_API_KEY env var or .env file)",
    )
    gemini_api_base: Optional[str] = Field(
        default=None,
        alias="gemini_api_base",
        description="Gemini API base URL (for E2E testing with mock servers). "
        "Can be set via GEMINI_API_BASE environment variable.",
    )
    gemini_transcription_model: str = Field(
        default_factory=_get_default_gemini_transcription_model,
        alias="gemini_transcription_model",
        description="Gemini model for transcription (default: environment-based)",
    )
    gemini_speaker_model: str = Field(
        default_factory=_get_default_gemini_speaker_model,
        alias="gemini_speaker_model",
        description="Gemini model for speaker detection (default: environment-based)",
    )
    gemini_summary_model: str = Field(
        default_factory=_get_default_gemini_summary_model,
        alias="gemini_summary_model",
        description="Gemini model for summarization (default: environment-based)",
    )
    gemini_temperature: float = Field(
        default=0.3,
        alias="gemini_temperature",
        description="Temperature for Gemini generation (0.0-2.0, lower = more deterministic)",
    )
    gemini_cleaning_model: str = Field(
        default_factory=_get_default_gemini_cleaning_model,
        alias="gemini_cleaning_model",
        description=(
            "Gemini model for transcript cleaning "
            "(default: gemini-2.5-flash-lite, cheaper than summary model)"
        ),
    )
    gemini_cleaning_temperature: float = Field(
        default=0.2,
        alias="gemini_cleaning_temperature",
        description="Temperature for Gemini cleaning (0.0-2.0, default: 0.2, lower than summarization)",  # noqa: E501
    )
    gemini_max_tokens: Optional[int] = Field(
        default=None,
        alias="gemini_max_tokens",
        description="Max tokens for Gemini generation (None = model default)",
    )
    # Gemini Prompt Configuration (following OpenAI pattern)
    gemini_summary_system_prompt: str = Field(
        default="gemini/summarization/system_bullets_v1",
        alias="gemini_summary_system_prompt",
        description=(
            "Gemini system prompt for summarization (default: JSON bullets). "
            "Uses prompt_store (RFC-017); resolves shared/summarization/ when provider file absent."
        ),
    )
    gemini_summary_user_prompt: str = Field(
        default="gemini/summarization/bullets_json_v1",
        alias="gemini_summary_user_prompt",
        description="Gemini user prompt for summarization. Uses prompt_store (RFC-017).",
    )
    gemini_speaker_system_prompt: Optional[str] = Field(
        default=None,
        alias="gemini_speaker_system_prompt",
        description=(
            "Gemini system prompt for speaker detection (default: gemini/ner/system_ner_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    gemini_speaker_user_prompt: str = Field(
        default="gemini/ner/guest_host_v1",
        alias="gemini_speaker_user_prompt",
        description="Gemini user prompt for speaker detection. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    # Anthropic API configuration (Issue #106)
    anthropic_api_key: Optional[str] = Field(
        default=None,
        alias="anthropic_api_key",
        description="Anthropic API key (prefer ANTHROPIC_API_KEY env var or .env file)",
    )
    anthropic_api_base: Optional[str] = Field(
        default=None,
        alias="anthropic_api_base",
        description="Anthropic API base URL (for E2E testing with mock servers). "
        "Can be set via ANTHROPIC_API_BASE environment variable.",
    )
    anthropic_transcription_model: str = Field(
        default_factory=_get_default_anthropic_transcription_model,
        alias="anthropic_transcription_model",
        description="Anthropic model for transcription (default: environment-based). "
        "Note: Anthropic doesn't support native audio transcription.",
    )
    anthropic_speaker_model: str = Field(
        default_factory=_get_default_anthropic_speaker_model,
        alias="anthropic_speaker_model",
        description="Anthropic model for speaker detection (default: environment-based)",
    )
    anthropic_summary_model: str = Field(
        default_factory=_get_default_anthropic_summary_model,
        alias="anthropic_summary_model",
        description="Anthropic model for summarization (default: environment-based)",
    )
    anthropic_temperature: float = Field(
        default=0.3,
        alias="anthropic_temperature",
        description="Temperature for Anthropic generation (0.0-1.0, lower = more deterministic)",
    )
    anthropic_cleaning_model: str = Field(
        default_factory=_get_default_anthropic_cleaning_model,
        alias="anthropic_cleaning_model",
        description="Anthropic model for transcript cleaning (default: claude-haiku-4-5, cheaper than summary model)",  # noqa: E501
    )
    anthropic_cleaning_temperature: float = Field(
        default=0.2,
        alias="anthropic_cleaning_temperature",
        description="Temperature for Anthropic cleaning (0.0-1.0, default: 0.2, lower than summarization)",  # noqa: E501
    )
    anthropic_max_tokens: Optional[int] = Field(
        default=None,
        alias="anthropic_max_tokens",
        description="Max tokens for Anthropic generation (None = model default)",
    )
    # Anthropic Prompt Configuration (following OpenAI/Gemini pattern)
    anthropic_summary_system_prompt: str = Field(
        default="anthropic/summarization/system_bullets_v1",
        alias="anthropic_summary_system_prompt",
        description=(
            "Anthropic system prompt for summarization (default: JSON bullets). "
            "Uses prompt_store (RFC-017)."
        ),
    )
    anthropic_summary_user_prompt: str = Field(
        default="anthropic/summarization/bullets_json_v1",
        alias="anthropic_summary_user_prompt",
        description="Anthropic user prompt for summarization. Uses prompt_store (RFC-017).",
    )
    anthropic_speaker_system_prompt: Optional[str] = Field(
        default=None,
        alias="anthropic_speaker_system_prompt",
        description=(
            "Anthropic system prompt for speaker detection (default: anthropic/ner/system_ner_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    anthropic_speaker_user_prompt: str = Field(
        default="anthropic/ner/guest_host_v1",
        alias="anthropic_speaker_user_prompt",
        description="Anthropic user prompt for speaker detection. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    # Ollama API configuration (Issue #196)
    ollama_api_base: Optional[str] = Field(
        default=None,
        alias="ollama_api_base",
        description="Ollama API base URL (default: http://localhost:11434/v1, for E2E testing). "
        "Can be set via OLLAMA_API_BASE environment variable.",
    )
    ollama_speaker_model: str = Field(
        default_factory=_get_default_ollama_speaker_model,
        alias="ollama_speaker_model",
        description="Ollama model for speaker detection (default: environment-based)",
    )
    ollama_summary_model: str = Field(
        default_factory=_get_default_ollama_summary_model,
        alias="ollama_summary_model",
        description="Ollama model for summarization (default: environment-based)",
    )
    ollama_temperature: float = Field(
        default=0.3,
        alias="ollama_temperature",
        description="Temperature for Ollama generation (0.0-2.0, lower = more deterministic)",
    )
    ollama_num_ctx: int = Field(
        default=32768,
        alias="ollama_num_ctx",
        ge=512,
        description=(
            "Ollama context window (num_ctx) passed to the /v1/chat/completions call. "
            "Ollama 0.19.0 tiers defaults by VRAM (48GB → 32k), but silent truncation "
            "still happens when prompt + output exceed the set limit. 32768 is the "
            "research-recommended safe default on 48GB hardware. Note: gemma2 is "
            "structurally capped at 8192 regardless of this setting (model spec limit); "
            "Ollama will silently clamp. Reduce this value to save memory if all prompts "
            "are small, but monitor for held-out-size silent truncation."
        ),
    )
    ollama_reduce_temperature: Optional[float] = Field(
        default=None,
        alias="ollama_reduce_temperature",
        description=(
            "Override temperature for Ollama reduce stage in hybrid_ml pipeline. "
            "When None, falls back to ollama_temperature."
        ),
    )
    ollama_reduce_top_p: Optional[float] = Field(
        default=None,
        alias="ollama_reduce_top_p",
        description="Top-p (nucleus sampling) for Ollama reduce stage. When None, uses 0.9.",
    )
    ollama_reduce_frequency_penalty: Optional[float] = Field(
        default=None,
        alias="ollama_reduce_frequency_penalty",
        description="Frequency penalty for Ollama reduce stage. When None, uses 0.0.",
    )
    ollama_cleaning_model: str = Field(
        default_factory=_get_default_ollama_cleaning_model,
        alias="ollama_cleaning_model",
        description="Ollama model for transcript cleaning (default: llama3.1:8b, smaller than summary model)",  # noqa: E501
    )
    ollama_cleaning_temperature: float = Field(
        default=0.2,
        alias="ollama_cleaning_temperature",
        description="Temperature for Ollama cleaning (0.0-2.0, default: 0.2, lower than summarization)",  # noqa: E501
    )
    ollama_max_tokens: Optional[int] = Field(
        default=None,
        alias="ollama_max_tokens",
        description="Max tokens for Ollama generation (None = model default)",
    )
    ollama_timeout: int = Field(
        default=120,
        alias="ollama_timeout",
        description="Timeout in seconds for Ollama API calls (local inference can be slow)",
    )
    # Ollama Prompt Configuration (following OpenAI pattern)
    ollama_speaker_system_prompt: Optional[str] = Field(
        default=None,
        alias="ollama_speaker_system_prompt",
        description=(
            "Ollama system prompt for speaker detection (default: ollama/ner/system_ner_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    ollama_speaker_user_prompt: str = Field(
        default="ollama/ner/guest_host_v1",
        alias="ollama_speaker_user_prompt",
        description="Ollama user prompt for speaker detection. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    ollama_summary_system_prompt: str = Field(
        default="ollama/summarization/system_bullets_v1",
        alias="ollama_summary_system_prompt",
        description=(
            "Ollama system prompt for summarization (default: JSON bullets). "
            "Uses prompt_store (RFC-017)."
        ),
    )
    ollama_summary_user_prompt: str = Field(
        default="ollama/summarization/bullets_json_v1",
        alias="ollama_summary_user_prompt",
        description="Ollama user prompt for summarization. Uses prompt_store (RFC-017).",
    )
    # DeepSeek API configuration (Issue #107)
    deepseek_api_key: Optional[str] = Field(
        default=None,
        alias="deepseek_api_key",
        description="DeepSeek API key (prefer DEEPSEEK_API_KEY env var or .env file)",
    )
    deepseek_api_base: Optional[str] = Field(
        default=None,
        alias="deepseek_api_base",
        description="DeepSeek API base URL (default: https://api.deepseek.com, for E2E testing). "
        "Can be set via DEEPSEEK_API_BASE environment variable.",
    )
    deepseek_speaker_model: str = Field(
        default_factory=_get_default_deepseek_speaker_model,
        alias="deepseek_speaker_model",
        description="DeepSeek model for speaker detection (default: environment-based)",
    )
    deepseek_summary_model: str = Field(
        default_factory=_get_default_deepseek_summary_model,
        alias="deepseek_summary_model",
        description="DeepSeek model for summarization (default: environment-based)",
    )
    deepseek_temperature: float = Field(
        default=0.3,
        alias="deepseek_temperature",
        description="Temperature for DeepSeek generation (0.0-2.0, lower = more deterministic)",
    )
    deepseek_cleaning_model: str = Field(
        default_factory=_get_default_deepseek_cleaning_model,
        alias="deepseek_cleaning_model",
        description="DeepSeek model for transcript cleaning (default: deepseek-chat)",
    )
    deepseek_cleaning_temperature: float = Field(
        default=0.2,
        alias="deepseek_cleaning_temperature",
        description="Temperature for DeepSeek cleaning (0.0-2.0, default: 0.2, lower than summarization)",  # noqa: E501
    )
    deepseek_max_tokens: Optional[int] = Field(
        default=None,
        alias="deepseek_max_tokens",
        description="Max tokens for DeepSeek generation (None = model default)",
    )
    deepseek_timeout: int = Field(
        default=600,
        alias="deepseek_timeout",
        ge=30,
        description=(
            "HTTP read timeout in seconds for DeepSeek API calls (default: 600 = 10min). "
            "DeepSeek mega-bundle / large JSON responses can exceed the default 120s "
            "generic HTTP timeout, so DeepSeek follows Grok's pattern of a per-provider "
            "override (#646 real-episode validation)."
        ),
    )
    # DeepSeek Prompt Configuration (following OpenAI pattern)
    deepseek_summary_system_prompt: str = Field(
        default="deepseek/summarization/system_bullets_v1",
        alias="deepseek_summary_system_prompt",
        description=(
            "DeepSeek system prompt for summarization (default: JSON bullets). "
            "Uses prompt_store (RFC-017)."
        ),
    )
    deepseek_summary_user_prompt: str = Field(
        default="deepseek/summarization/bullets_json_v1",
        alias="deepseek_summary_user_prompt",
        description="DeepSeek user prompt for summarization. Uses prompt_store (RFC-017).",
    )
    deepseek_speaker_system_prompt: Optional[str] = Field(
        default=None,
        alias="deepseek_speaker_system_prompt",
        description=(
            "DeepSeek system prompt for speaker detection (default: deepseek/ner/system_ner_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    deepseek_speaker_user_prompt: str = Field(
        default="deepseek/ner/guest_host_v1",
        alias="deepseek_speaker_user_prompt",
        description="DeepSeek user prompt for speaker detection. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    # Grok API configuration (Issue #1095)
    grok_api_key: Optional[str] = Field(
        default=None,
        alias="grok_api_key",
        description="Grok API key (prefer GROK_API_KEY env var or .env file)",
    )
    grok_api_base: Optional[str] = Field(
        default=None,
        alias="grok_api_base",
        description=(
            "Grok API base URL (default: https://api.x.ai/v1, for E2E testing with mock servers). "
            "Can be set via GROK_API_BASE environment variable."
        ),
    )
    grok_speaker_model: str = Field(
        default_factory=_get_default_grok_speaker_model,
        alias="grok_speaker_model",
        description="Grok model for speaker detection (default: environment-based)",
    )
    grok_summary_model: str = Field(
        default_factory=_get_default_grok_summary_model,
        alias="grok_summary_model",
        description="Grok model for summarization (default: environment-based)",
    )
    grok_temperature: float = Field(
        default=0.3,
        alias="grok_temperature",
        description="Temperature for Grok generation (0.0-2.0, lower = more deterministic)",
    )
    grok_cleaning_model: str = Field(
        default_factory=_get_default_grok_cleaning_model,
        alias="grok_cleaning_model",
        description="Grok model for transcript cleaning (default: grok-beta)",
    )
    grok_cleaning_temperature: float = Field(
        default=0.2,
        alias="grok_cleaning_temperature",
        description="Temperature for Grok cleaning (0.0-2.0, default: 0.2, lower than summarization)",  # noqa: E501
    )
    grok_timeout: int = Field(
        default=1800,
        alias="grok_timeout",
        ge=30,
        description=(
            "HTTP read timeout in seconds for Grok API calls (default: 1800 = 30min). "
            "Grok is a reasoning model — GI evidence grounding makes 60+ calls per "
            "episode, each requiring thinking time. Default is 3x summarization_timeout."
        ),
    )
    grok_max_tokens: Optional[int] = Field(
        default=None,
        alias="grok_max_tokens",
        description="Max tokens for Grok generation (None = model default)",
    )
    grok_summary_system_prompt: str = Field(
        default="grok/summarization/system_bullets_v1",
        alias="grok_summary_system_prompt",
        description=(
            "Grok system prompt for summarization (default: JSON bullets). "
            "Uses prompt_store (RFC-017)."
        ),
    )
    grok_summary_user_prompt: str = Field(
        default="grok/summarization/bullets_json_v1",
        alias="grok_summary_user_prompt",
        description="Grok user prompt for summarization. Uses prompt_store (RFC-017).",
    )
    grok_speaker_system_prompt: Optional[str] = Field(
        default=None,
        alias="grok_speaker_system_prompt",
        description=(
            "Grok system prompt for speaker detection (default: grok/ner/system_ner_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    grok_speaker_user_prompt: str = Field(
        default="grok/ner/guest_host_v1",
        alias="grok_speaker_user_prompt",
        description="Grok user prompt for speaker detection. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    # Mistral API configuration (Issue #106)
    mistral_api_key: Optional[str] = Field(
        default=None,
        alias="mistral_api_key",
        description="Mistral API key (prefer MISTRAL_API_KEY env var or .env file)",
    )
    mistral_api_base: Optional[str] = Field(
        default=None,
        alias="mistral_api_base",
        description="Mistral API base URL (for E2E testing with mock servers). "
        "Can be set via MISTRAL_API_BASE environment variable.",
    )
    mistral_transcription_model: str = Field(
        default_factory=_get_default_mistral_transcription_model,
        alias="mistral_transcription_model",
        description="Mistral Voxtral model for transcription (default: environment-based)",
    )
    mistral_speaker_model: str = Field(
        default_factory=_get_default_mistral_speaker_model,
        alias="mistral_speaker_model",
        description="Mistral model for speaker detection (default: environment-based)",
    )
    mistral_summary_model: str = Field(
        default_factory=_get_default_mistral_summary_model,
        alias="mistral_summary_model",
        description="Mistral model for summarization (default: environment-based)",
    )
    mistral_temperature: float = Field(
        default=0.3,
        alias="mistral_temperature",
        description="Temperature for Mistral generation (0.0-1.0, lower = more deterministic)",
    )
    mistral_cleaning_model: str = Field(
        default_factory=_get_default_mistral_cleaning_model,
        alias="mistral_cleaning_model",
        description="Mistral model for transcript cleaning (default: mistral-small, cheaper than summary model)",  # noqa: E501
    )
    mistral_cleaning_temperature: float = Field(
        default=0.2,
        alias="mistral_cleaning_temperature",
        description="Temperature for Mistral cleaning (0.0-1.0, default: 0.2, lower than summarization)",  # noqa: E501
    )
    mistral_max_tokens: Optional[int] = Field(
        default=None,
        alias="mistral_max_tokens",
        description="Max tokens for Mistral generation (None = model default)",
    )
    # Mistral Prompt Configuration (following OpenAI pattern)
    mistral_speaker_system_prompt: Optional[str] = Field(
        default=None,
        alias="mistral_speaker_system_prompt",
        description=(
            "Mistral system prompt for speaker detection (default: mistral/ner/system_ner_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    mistral_speaker_user_prompt: str = Field(
        default="mistral/ner/guest_host_v1",
        alias="mistral_speaker_user_prompt",
        description="Mistral user prompt for speaker detection. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    mistral_summary_system_prompt: str = Field(
        default="mistral/summarization/system_bullets_v1",
        alias="mistral_summary_system_prompt",
        description=(
            "Mistral system prompt for summarization (default: JSON bullets). "
            "Uses prompt_store (RFC-017)."
        ),
    )
    mistral_summary_user_prompt: str = Field(
        default="mistral/summarization/bullets_json_v1",
        alias="mistral_summary_user_prompt",
        description="Mistral user prompt for summarization. Uses prompt_store (RFC-017).",
    )
    generate_metadata: bool = Field(
        default=True,
        alias="generate_metadata",
        description="Write per-episode metadata documents when running the pipeline (PRD-004).",
    )
    metadata_format: Literal["json", "yaml"] = Field(default="json", alias="metadata_format")
    metadata_subdirectory: Optional[str] = Field(default=None, alias="metadata_subdirectory")
    download_podcast_artwork: bool = Field(
        default=True,
        alias="download_podcast_artwork",
        description=(
            "When True with generate_metadata, download feed/episode image_url targets into "
            "<output>/.podcast_scraper/corpus-art/ and set image_local_relpath fields."
        ),
    )
    generate_summaries: bool = Field(default=False, alias="generate_summaries")
    # GIL evidence stack (Issue #435): loaded lazily when GIL or dependent feature enabled
    embedding_model: str = Field(
        default=config_constants.DEFAULT_EMBEDDING_MODEL,
        alias="embedding_model",
        description="Model for sentence embeddings (GIL grounding). Alias or full HF ID.",
    )
    embedding_device: Optional[str] = Field(
        default=None,
        alias="embedding_device",
        description="Device for embedding model (cpu, cuda, mps, or None for auto).",
    )
    extractive_qa_model: str = Field(
        default=config_constants.DEFAULT_EXTRACTIVE_QA_MODEL,
        alias="extractive_qa_model",
        description="Model for extractive QA (GIL quote extraction). Alias or full HF ID.",
    )
    extractive_qa_device: Optional[str] = Field(
        default=None,
        alias="extractive_qa_device",
        description="Device for extractive QA model (cpu, cuda, mps, or None for auto).",
    )
    nli_model: str = Field(
        default=config_constants.DEFAULT_NLI_MODEL,
        alias="nli_model",
        description="Model for NLI entailment (GIL grounding). Alias or full HF ID.",
    )
    nli_device: Optional[str] = Field(
        default=None,
        alias="nli_device",
        description="Device for NLI model (cpu, cuda, mps, or None for auto).",
    )
    # GIL extraction (Issue #356): per-episode gi.json when enabled
    generate_gi: bool = Field(
        default=False,
        alias="generate_gi",
        description="Enable Grounded Insight Layer extraction; writes gi.json per episode.",
    )
    gi_qa_model: str = Field(
        default=config_constants.DEFAULT_EXTRACTIVE_QA_MODEL,
        alias="gi_qa_model",
        description="Model for quote extraction (reuses evidence stack QA).",
    )
    gi_embedding_model: str = Field(
        default=config_constants.DEFAULT_EMBEDDING_MODEL,
        alias="gi_embedding_model",
        description="Model for embeddings when GIL uses similarity (reuses evidence stack).",
    )
    gi_nli_model: str = Field(
        default=config_constants.DEFAULT_NLI_MODEL,
        alias="gi_nli_model",
        description="Model for entailment when grounding (reuses evidence stack).",
    )
    gi_require_grounding: bool = Field(
        default=True,
        alias="gi_require_grounding",
        description="If True, only emit SUPPORTED_BY edges when QA+NLI pass thresholds.",
    )
    gi_fail_on_missing_grounding: bool = Field(
        default=False,
        alias="gi_fail_on_missing_grounding",
        description=(
            "If True, raise GILGroundingUnsatisfiedError when gi_require_grounding is True "
            "but evidence produces zero grounded quotes for an episode (strict CI / QA)."
        ),
    )
    gi_evidence_extract_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        alias="gi_evidence_extract_retries",
        description=(
            "Provider GIL extract_quotes: extra attempts when the first returns no span; "
            "retries append a verbatim-copy hint to the insight text."
        ),
    )
    gi_qa_score_min: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        alias="gi_qa_score_min",
        description=(
            "Minimum extractive QA score to keep a quote candidate before NLI (GIL grounding)."
        ),
    )
    gi_nli_entailment_min: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        alias="gi_nli_entailment_min",
        description="Minimum NLI entailment probability to attach a grounded quote (GIL).",
    )
    gi_qa_window_chars: int = Field(
        default=1800,
        ge=0,
        le=500_000,
        alias="gi_qa_window_chars",
        description=(
            "GIL local QA: when > 0 and the transcript is longer, scan overlapping windows "
            "of this many characters and keep the best-scoring span. Use 0 to disable "
            "windowing (single QA call on the full transcript)."
        ),
    )
    gi_qa_window_overlap_chars: int = Field(
        default=300,
        ge=0,
        le=100_000,
        alias="gi_qa_window_overlap_chars",
        description=(
            "Overlap between consecutive QA windows when gi_qa_window_chars > 0. "
            "Must be less than gi_qa_window_chars."
        ),
    )
    quote_extraction_provider: Literal[
        "transformers",
        "hybrid_ml",
        "openai",
        "gemini",
        "grok",
        "mistral",
        "deepseek",
        "anthropic",
        "ollama",
    ] = Field(
        default="transformers",
        alias="quote_extraction_provider",
        description=(
            "Provider for GIL quote extraction (QA). Same backends as summary_provider; "
            "default 'transformers' uses local extractive QA."
        ),
    )
    entailment_provider: Literal[
        "transformers",
        "hybrid_ml",
        "openai",
        "gemini",
        "grok",
        "mistral",
        "deepseek",
        "anthropic",
        "ollama",
    ] = Field(
        default="transformers",
        alias="entailment_provider",
        description=(
            "Provider for GIL entailment (NLI). Same backends as summary_provider; "
            "default 'transformers' uses local NLI model."
        ),
    )
    gil_evidence_match_summary_provider: bool = Field(
        default=True,
        alias="gil_evidence_match_summary_provider",
        description=(
            "When True (default) and generate_gi is True: if summary_provider is an API LLM "
            "(openai, gemini, anthropic, mistral, deepseek, grok, ollama) or hybrid_ml, and both "
            "quote_extraction_provider and entailment_provider are still the default "
            "'transformers', they are set to summary_provider so GIL grounding uses the same "
            "backend as summaries (applied in a model_validator before init, so Config(**d) "
            "and Config.model_validate(d) both work). Set False to keep local extractive QA + "
            "NLI with an API summary (advanced)."
        ),
    )
    gi_insight_source: Literal["provider", "summary_bullets", "stub"] = Field(
        default="stub",
        alias="gi_insight_source",
        description=(
            "Source of insight texts for GIL: 'provider' = call generate_insights() on "
            "the summarization provider (LLM only; ML providers return empty), "
            "'summary_bullets' = first N summary bullets (needs generate_summaries + bullets), "
            "or 'stub' (default placeholder). See GROUNDED_INSIGHTS_GUIDE.md."
        ),
    )
    gi_max_insights: int = Field(
        default=config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX,
        ge=1,
        le=50,
        alias="gi_max_insights",
        description=(
            "Max insights when gi_insight_source is provider or summary_bullets. "
            "Default matches DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX (avoid truncating long lists)."
        ),
    )
    vector_search: bool = Field(
        default=False,
        alias="vector_search",
        description=(
            "When True, embed and index corpus vectors after pipeline finalize "
            "(PRD-021 / RFC-061). Index defaults to <output_dir>/search; use podcast search / "
            "podcast index CLIs and semantic gi explore when a topic filter is set. "
            "Implies embedding model preload alongside GIL evidence models when set without "
            "generate_gi."
        ),
    )
    skip_auto_vector_index: bool = Field(
        default=False,
        alias="skip_auto_vector_index",
        description=(
            "When True, pipeline finalize skips automatic ``index_corpus`` even if "
            "vector_search is on (GitHub #505). Multi-feed runs set this on per-feed "
            "children and build one parent index after all feeds complete."
        ),
    )
    vector_index_path: Optional[str] = Field(
        default=None,
        alias="vector_index_path",
        description=(
            "Directory for the FAISS corpus index. Relative paths resolve under output_dir. "
            "Default when unset: <output_dir>/search."
        ),
    )
    vector_chunk_size_tokens: int = Field(
        default=300,
        ge=20,
        le=2000,
        alias="vector_chunk_size_tokens",
        description="Target transcript chunk size (whitespace token count) for embeddings.",
    )
    vector_chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        le=500,
        alias="vector_chunk_overlap_tokens",
        description="Token overlap between consecutive transcript chunks.",
    )
    vector_embedding_model: str = Field(
        default=config_constants.DEFAULT_EMBEDDING_MODEL,
        alias="vector_embedding_model",
        description="Sentence-transformers model id for semantic corpus embeddings (GitHub #484).",
    )
    vector_backend: Literal["faiss"] = Field(
        default="faiss",
        alias="vector_backend",
        description=(
            "Vector index backend. Currently faiss only. qdrant is reserved for a future "
            "platform-mode release (RFC-070) — will be re-added to this Literal once wired."
        ),
    )
    vector_index_types: Optional[
        List[Literal["insight", "quote", "summary", "transcript", "kg_topic", "kg_entity"]]
    ] = Field(
        default=None,
        alias="vector_index_types",
        description=(
            "Doc types to embed (default: all). "
            "Values: insight, quote, summary, transcript, kg_topic, kg_entity."
        ),
    )
    vector_faiss_index_mode: Literal["auto", "flat", "ivf_flat", "ivfpq"] = Field(
        default="auto",
        alias="vector_faiss_index_mode",
        description=(
            "FAISS structure: auto uses #484 thresholds (Flat / IVFFlat / IVFPQ by size); "
            "or force flat, ivf_flat, ivfpq after indexing."
        ),
    )
    # Knowledge Graph Layer (PRD-019 / RFC-055): per-episode kg.json when enabled
    generate_kg: bool = Field(
        default=False,
        alias="generate_kg",
        description=(
            "Enable Knowledge Graph extraction; writes kg.json per episode " "(separate from GIL)."
        ),
    )
    kg_extraction_source: Literal["stub", "summary_bullets", "provider"] = Field(
        default="summary_bullets",
        alias="kg_extraction_source",
        description=(
            "KG topics/entities source: 'provider' = LLM JSON extraction via "
            "extract_kg_graph on the transcript (see kg_extraction_provider; "
            "default uses summary_provider; ML providers no-op); "
            "'summary_bullets' = when an API summarization provider is available, "
            "derive short topics/entities from bullets via extract_kg_from_summary_bullets; "
            "otherwise topic labels mirror bullet text; "
            "'stub' = episode + pipeline hosts/guests only (no summary topics)."
        ),
    )
    kg_extraction_provider: Optional[
        Literal[
            "transformers",
            "hybrid_ml",
            "openai",
            "gemini",
            "grok",
            "mistral",
            "deepseek",
            "anthropic",
            "ollama",
        ]
    ] = Field(
        default=None,
        alias="kg_extraction_provider",
        description=(
            "When kg_extraction_source is 'provider', which backend runs extract_kg_graph; "
            "when 'summary_bullets', which backend runs extract_kg_from_summary_bullets "
            "(bullet-derived topics). None means use summary_provider (same instance). "
            "Same names as summary_provider."
        ),
    )
    kg_max_topics: int = Field(
        default=config_constants.DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX,
        ge=1,
        le=20,
        alias="kg_max_topics",
        description=(
            "Max topic nodes for summary_bullets or provider KG extraction. "
            "Default matches DEFAULT_SUMMARY_BULLETS_DOWNSTREAM_MAX (schema max 20)."
        ),
    )
    kg_max_entities: int = Field(
        default=15,
        ge=1,
        le=50,
        alias="kg_max_entities",
        description="Max entity nodes from provider KG extraction.",
    )
    kg_extraction_model: Optional[str] = Field(
        default=None,
        alias="kg_extraction_model",
        description=(
            "Optional model override for KG LLM extraction; default uses the "
            "summarization model for the active provider."
        ),
    )
    kg_merge_pipeline_entities: bool = Field(
        default=True,
        alias="kg_merge_pipeline_entities",
        description=(
            "When True, merge detected hosts/guests into kg.json after provider extraction "
            "(deduped by entity_kind + name, same as LLM entities). When False, only LLM "
            "entities (plus Episode)."
        ),
    )
    metrics_output: Optional[str] = Field(
        default=None,
        alias="metrics_output",
        description="Path to save pipeline metrics JSON file. "
        "If not specified, defaults to {effective_output_dir}/metrics.json "
        "(same level as transcripts/ and metadata/ subdirectories). "
        "Set to empty string to disable metrics export.",
    )
    jsonl_metrics_enabled: bool = Field(
        default=False,
        alias="jsonl_metrics_enabled",
        description=(
            "Enable JSONL streaming metrics output (default: False, opt-in). "
            "JSONL metrics stream during pipeline execution, complementing "
            "the single JSON file output at the end. Useful for real-time monitoring."
        ),
    )
    jsonl_metrics_path: Optional[str] = Field(
        default=None,
        alias="jsonl_metrics_path",
        description=(
            "Path to JSONL metrics output file. "
            "If not specified and jsonl_metrics_enabled=True, "
            "defaults to {effective_output_dir}/run.jsonl."
        ),
    )
    pricing_assumptions_file: str = Field(
        default="",
        alias="pricing_assumptions_file",
        description=(
            "Optional YAML file with USD rates for LLM cost estimates (transcription + tokens). "
            "When empty, only built-in provider constants are used. "
            "Relative paths are resolved from the current working directory and then from "
            "ancestor directories (repo root). See config/pricing_assumptions.yaml."
        ),
    )
    summary_provider: Literal[
        "transformers",
        "hybrid_ml",
        "summllama",
        "openai",
        "gemini",
        "grok",
        "mistral",
        "deepseek",
        "anthropic",
        "ollama",
    ] = Field(
        default="transformers",
        alias="summary_provider",
        description=(
            "Summary generation provider " "(default: 'transformers' for HuggingFace Transformers)."
        ),
    )
    hybrid_map_model: str = Field(
        default="longt5-base",
        alias="hybrid_map_model",
        description=(
            "Hybrid MAP model (classic summarizer). "
            "Recommended: longt5-base (8k context) for medium-long transcripts."
        ),
    )
    hybrid_reduce_model: str = Field(
        default="google/flan-t5-base",
        alias="hybrid_reduce_model",
        description=(
            "Hybrid REDUCE model (instruction-tuned). "
            "Tier 1 default: google/flan-t5-base via transformers backend."
        ),
    )
    hybrid_reduce_backend: Literal["transformers", "ollama", "llama_cpp"] = Field(
        default="transformers",
        alias="hybrid_reduce_backend",
        description=(
            "Hybrid REDUCE backend. "
            "transformers = FLAN-T5 via local transformers; "
            "ollama = send reduce step to local Ollama server; "
            "llama_cpp = GGUF via llama.cpp (optional)."
        ),
    )
    hybrid_map_device: Optional[str] = Field(
        default=None,
        alias="hybrid_map_device",
        description="Device for hybrid MAP model (cpu/cuda/mps/auto). Defaults to summary_device.",
    )
    hybrid_reduce_device: Optional[str] = Field(
        default=None,
        alias="hybrid_reduce_device",
        description=(
            "Device for hybrid REDUCE model (cpu/cuda/mps/auto). "
            "Defaults to summarization_device/summary_device."
        ),
    )
    hybrid_quantization: Optional[str] = Field(
        default=None,
        alias="hybrid_quantization",
        description=(
            "Optional quantization hint for hybrid REDUCE backend "
            "(e.g., '4bit', '8bit', 'q4'). Backend-specific; ignored if unsupported."
        ),
    )
    hybrid_llama_n_ctx: Optional[int] = Field(
        default=None,
        alias="hybrid_llama_n_ctx",
        description=(
            "Context length for llama_cpp REDUCE backend (e.g., 4096). "
            "When unset, provider uses 4096. Only used when hybrid_reduce_backend is llama_cpp."
        ),
    )
    hybrid_reduce_instruction_style: Optional[Literal["structured", "paragraph"]] = Field(
        default=None,
        alias="hybrid_reduce_instruction_style",
        description=(
            "REDUCE instruction style: 'structured' = Takeaways/Outline/Actions (default); "
            "'paragraph' = silver-style 4-6 paragraphs, no headings. Used for tuning toward silver."
        ),
    )
    hybrid_internal_preprocessing_after_pattern: str = Field(
        default="cleaning_hybrid_after_pattern",
        alias="hybrid_internal_preprocessing_after_pattern",
        description=(
            "Registered preprocessing profile applied inside HybridMLProvider.summarize() when "
            "summary_provider is hybrid_ml and transcript_cleaning_strategy is 'pattern', after "
            "the workflow has already run PatternBasedCleaner (Issue #419). Avoids redundant "
            "sponsor/outro passes versus full cleaning_v4 while keeping v4-only steps "
            "(header strip, junk filter, anonymization, artifact_scrub_v1). "
            "Must be a profile ID from preprocessing.profiles."
        ),
    )
    summary_2nd_pass_distill: bool = Field(
        default=False,
        alias="summary_2nd_pass_distill",
        description=(
            "Enable optional 2nd-pass distillation with faithfulness prompt (Issue #387). "
            "When enabled, applies an additional distillation pass with a prompt that guides "
            "the model to be faithful to the source and reduce hallucinations. "
            "Useful for hallucination-prone summaries. Only effective with OpenAI provider "
            "(BART/LED models don't use prompts effectively)."
        ),
    )
    summary_mode_id: Optional[str] = Field(
        default_factory=_get_default_summary_mode_id,
        alias="summary_mode_id",
        description=(
            "Summarization mode ID (RFC-044). When set, providers may use a promoted "
            "ModeConfiguration from the Model Registry as the source of defaults "
            "(models, preprocessing_profile, and runtime params)."
        ),
    )
    summary_mode_precedence: Literal["mode", "config"] = Field(
        default="mode",
        alias="summary_mode_precedence",
        description=(
            "When summary_mode_id is set, controls precedence between the promoted "
            "ModeConfiguration and explicit config fields. "
            "'mode' = mode overrides config; 'config' = config overrides mode. "
            "Params dict passed at runtime always overrides both."
        ),
    )
    summary_model: Optional[str] = Field(default=None, alias="summary_model")
    # Optional separate model for reduce phase (e.g., BART for map, LED for reduce).
    # If not set, the same model is used for both map and reduce.
    summary_reduce_model: Optional[str] = Field(
        default=None,
        alias="summary_reduce_model",
        description="Optional separate model (or key) to use for reduce phase; "
        "falls back to summary_model when not set.",
    )
    summary_device: Optional[str] = Field(default=None, alias="summary_device")
    summarization_device: Optional[str] = Field(
        default=None,
        alias="summarization_device",
        description=(
            "Device for summarization stage (CPU, CUDA, MPS, or None for auto-detection). "
            "Overrides provider-specific device (e.g., summary_device) if set. "
            "Allows CPU/GPU mix to regain overlap (Issue #387). "
            "Valid values: 'cpu', 'cuda', 'mps', or None/empty for auto-detect."
        ),
    )
    mps_exclusive: bool = Field(
        default=True,
        alias="mps_exclusive",
        description=(
            "Serialize GPU work on MPS to prevent memory contention between "
            "Whisper transcription and summarization (default: True). "
            "When enabled and both Whisper and summarization use MPS, "
            "transcription completes before summarization starts. "
            "I/O operations (downloads, parsing) remain parallel."
        ),
    )
    summary_batch_size: int = Field(
        default=DEFAULT_SUMMARY_BATCH_SIZE,
        alias="summary_batch_size",
        description=(
            "Episode-level parallelism: Number of episodes to summarize in parallel "
            "(memory-bound for local, rate-limited for API providers)"
        ),
    )
    summary_chunk_parallelism: int = Field(
        default=1,
        alias="summary_chunk_parallelism",
        description=(
            "Chunk-level parallelism: Number of chunks to process in parallel "
            "within a single episode (CPU-bound, local providers only. "
            "API providers handle internally via rate limiting)"
        ),
    )
    summary_max_workers_cpu: Optional[int] = Field(
        default=None,  # Will be set by validator based on environment
        alias="summary_max_workers_cpu",
        description=(
            "Maximum parallel workers for episode summarization on CPU. "
            "Defaults to 1 in test environments, 4 in production. "
            "Lower values reduce memory usage."
        ),
    )
    summary_max_workers_gpu: Optional[int] = Field(
        default=None,  # Will be set by validator based on environment
        alias="summary_max_workers_gpu",
        description=(
            "Maximum parallel workers for episode summarization on GPU. "
            "Defaults to 1 in test environments, 2 in production. "
            "Lower values reduce memory usage."
        ),
    )
    summary_chunk_size: Optional[int] = Field(default=None, alias="summary_chunk_size")
    summary_word_chunk_size: Optional[int] = Field(
        default=DEFAULT_SUMMARY_WORD_CHUNK_SIZE,
        alias="summary_word_chunk_size",
        description=(
            f"Chunk size in words for word-based chunking "
            f"({config_constants.RECOMMENDED_WORD_CHUNK_SIZE_MIN}-"
            f"{config_constants.RECOMMENDED_WORD_CHUNK_SIZE_MAX} recommended)"
        ),
    )
    summary_word_overlap: Optional[int] = Field(
        default=DEFAULT_SUMMARY_WORD_OVERLAP,
        alias="summary_word_overlap",
        description=(
            f"Overlap in words for word-based chunking "
            f"({config_constants.RECOMMENDED_WORD_OVERLAP_MIN}-"
            f"{config_constants.RECOMMENDED_WORD_OVERLAP_MAX} recommended)"
        ),
    )
    summary_cache_dir: Optional[str] = Field(
        default=None,
        alias="summary_cache_dir",
        description="Custom cache directory for transformer models. "
        "Can be set via SUMMARY_CACHE_DIR or CACHE_DIR environment variable.",
    )
    summary_prompt: Optional[str] = Field(default=None, alias="summary_prompt")
    save_cleaned_transcript: bool = Field(
        default=True, alias="save_cleaned_transcript"
    )  # Save cleaned transcript to separate file for testing (default: True)
    # Transcript cleaning configuration (Issue #418)
    transcript_cleaning_strategy: Literal["pattern", "llm", "hybrid"] = Field(
        default="hybrid",
        alias="transcript_cleaning_strategy",
        description=(
            "Transcript cleaning strategy (default: 'hybrid'). "
            "Options: 'pattern' (pattern-based only), 'llm' (LLM-based only), "
            "'hybrid' (pattern-based + conditional LLM when needed)."
        ),
    )
    llm_pipeline_mode: Literal["staged", "bundled", "mega_bundled", "extraction_bundled"] = Field(
        default="staged",
        alias="llm_pipeline_mode",
        description=(
            "LLM transcript pipeline (Issue #477, extended by #643). "
            "'staged' = separate clean, summarize, GI, KG calls (default). "
            "'bundled' = one structured completion for clean+summary+bullets "
            "(Issue #477). "
            "'mega_bundled' = one call for summary+bullets+insights+topics+entities. "
            "Validated on Anthropic Claude Haiku 4.5 (#632, #643). Falls back "
            "to 'staged' on parser failure. "
            "'extraction_bundled' = two calls: summary standalone + "
            "insights/topics/entities bundled. Viable on OpenAI, Gemini where "
            "mega_bundled compresses the summary (#643)."
        ),
    )
    llm_bundled_max_output_tokens: int = Field(
        default=16384,
        ge=256,
        alias="llm_bundled_max_output_tokens",
        description=(
            "Max completion/output tokens for bundled clean+summary+bullets calls "
            "(large default: output includes full cleaned transcript JSON)."
        ),
    )
    cloud_llm_structured_min_output_tokens: int = Field(
        default=4096,
        ge=512,
        alias="cloud_llm_structured_min_output_tokens",
        description=(
            "Minimum max_output_tokens enforced on cloud-LLM structured summary "
            "(non-bundled) calls. Prevents mid-JSON truncation on long transcripts "
            "when summary_reduce_params.max_new_tokens is sized for LED-base local "
            "ML (~650). Discovered via Flightcast episode failure 2026-04-20 "
            "(Gemini truncated summary JSON at ~650 tokens). Providers that read "
            "params.max_length or summary_reduce_params.max_new_tokens should "
            "clamp the resulting value to at least this floor before calling the "
            "API. Applies to: openai, anthropic, gemini, deepseek, mistral, grok."
        ),
    )
    single_feed_uses_corpus_layout: bool = Field(
        default=False,
        alias="single_feed_uses_corpus_layout",
        description=(
            "When True, single-feed runs write under <output_dir>/feeds/<slug>/ "
            "instead of <output_dir>/run_<id>/, matching the corpus layout used "
            "by multi-feed runs (GitHub #644). Unifies output shape for viewer, "
            "eval tooling, and corpus-level artifacts. Default False for "
            "backwards compatibility; recommended True for new corpora and when "
            "an existing output_dir has been migrated. Migration helper: "
            "scripts/tools/migrate_single_feed_to_corpus.py."
        ),
    )
    # ML generation parameters (all defaults come from Config, no hardcoded values)
    # These provide fine-grained control over generation parameters
    summary_map_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_new_tokens": DEFAULT_MAP_MAX_NEW_TOKENS,
            "min_new_tokens": DEFAULT_MAP_MIN_NEW_TOKENS,
            "num_beams": DEFAULT_MAP_NUM_BEAMS,
            "no_repeat_ngram_size": DEFAULT_MAP_NO_REPEAT_NGRAM_SIZE,
            "length_penalty": DEFAULT_MAP_LENGTH_PENALTY,
            "early_stopping": DEFAULT_MAP_EARLY_STOPPING,
            "repetition_penalty": DEFAULT_MAP_REPETITION_PENALTY,
        },
        alias="summary_map_params",
        description=(
            "Generation parameters for map stage (hf_local backend only). "
            "Dict with: max_new_tokens, min_new_tokens, num_beams, no_repeat_ngram_size, "
            "length_penalty, early_stopping, repetition_penalty. "
            "Defaults aligned with baseline_ml_prod_authority_v1 (Pegasus-CNN)."
        ),
    )
    summary_reduce_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "max_new_tokens": DEFAULT_REDUCE_MAX_NEW_TOKENS,
            "min_new_tokens": DEFAULT_REDUCE_MIN_NEW_TOKENS,
            "num_beams": DEFAULT_REDUCE_NUM_BEAMS,
            "no_repeat_ngram_size": DEFAULT_REDUCE_NO_REPEAT_NGRAM_SIZE,
            "length_penalty": DEFAULT_REDUCE_LENGTH_PENALTY,
            "early_stopping": DEFAULT_REDUCE_EARLY_STOPPING,
            "repetition_penalty": DEFAULT_REDUCE_REPETITION_PENALTY,
        },
        alias="summary_reduce_params",
        description=(
            "Generation parameters for reduce stage (hf_local backend only). "
            "Dict with: max_new_tokens, min_new_tokens, num_beams, no_repeat_ngram_size, "
            "length_penalty, early_stopping, repetition_penalty. "
            "Defaults aligned with baseline_ml_prod_authority_v1 (LED-base)."
        ),
    )
    summary_tokenize: Dict[str, Any] = Field(
        default_factory=_get_default_summary_tokenize,
        alias="summary_tokenize",
        description=(
            "Tokenization configuration for input text (hf_local backend only). "
            "Dict with: map_max_input_tokens, reduce_max_input_tokens, truncation. "
            "Defaults are set in Config, no hardcoded values."
        ),
    )

    # Audio Preprocessing (RFC-040)
    preprocessing_enabled: bool = Field(
        default=True,
        alias="preprocessing_enabled",
        description="Enable audio preprocessing before transcription (default: True). "
        "Preprocessing optimizes audio for API providers with file size limits.",
    )
    preprocessing_cache_dir: Optional[str] = Field(
        default=None,
        alias="preprocessing_cache_dir",
        description=f"Custom cache directory for preprocessed audio "
        f"(default: {DEFAULT_PREPROCESSING_CACHE_DIR}).",
    )
    transcript_cache_enabled: bool = Field(
        default=True,
        alias="transcript_cache_enabled",
        description=(
            "Enable transcript caching by audio hash (default: True). "
            "Cached transcripts skip transcription entirely, enabling fast "
            "multi-provider experimentation without re-transcribing the same audio."
        ),
    )
    transcript_cache_dir: Optional[str] = Field(
        default=None,
        alias="transcript_cache_dir",
        description=(
            "Custom cache directory for transcripts "
            "(default: .cache/transcripts). "
            "Transcripts are cached by audio hash to enable fast re-runs."
        ),
    )

    # Graceful Degradation Policy
    degradation_policy: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="degradation_policy",
        description=(
            "Graceful degradation policy for handling component failures. "
            "If None, uses default policy (save partial results, continue on failures). "
            "Dict with keys: save_transcript_on_summarization_failure, "
            "save_summary_on_entity_extraction_failure, fallback_provider_on_failure, "
            "continue_on_stage_failure. See DegradationPolicy for details."
        ),
    )
    preprocessing_sample_rate: int = Field(
        default=16000,
        alias="preprocessing_sample_rate",
        description="Target sample rate for preprocessing in Hz (default: 16000).",
    )
    preprocessing_silence_threshold: str = Field(
        default="-50dB",
        alias="preprocessing_silence_threshold",
        description="Silence detection threshold (default: -50dB).",
    )
    preprocessing_silence_duration: float = Field(
        default=2.0,
        alias="preprocessing_silence_duration",
        description="Minimum silence duration to remove in seconds (default: 2.0).",
    )
    preprocessing_target_loudness: int = Field(
        default=-16,
        alias="preprocessing_target_loudness",
        description="Target loudness in LUFS for normalization (default: -16).",
    )
    preprocessing_mp3_bitrate_kbps: Optional[int] = Field(
        default=None,
        alias="preprocessing_mp3_bitrate_kbps",
        description=(
            "libmp3lame bitrate (kbps) for FFmpeg preprocessing output; null = auto "
            "(64 kbps for local transcription e.g. whisper, 48 kbps for openai/gemini — "
            "GitHub #561). Allowed range 24–128 when set. After the first pass, "
            "openai/gemini may re-encode to lower standard rungs until under the API cap; "
            "if still too large, upload chunking is tracked separately (GitHub #286)."
        ),
    )

    audio_preprocessing_profile: Optional[str] = Field(
        default=None,
        alias="audio_preprocessing_profile",
        description=(
            "Named audio preset from config/profiles/audio/<name>.yaml (GitHub #634). "
            "Merged under the deployment profile so one change to "
            "audio/speech_optimal_v1.yaml updates every profile that references it. "
            "Individual preprocessing_* fields in the deployment profile or on the "
            "CLI still override the preset. Orthogonal to ml_preprocessing_profile, "
            "which controls text cleaning for the ML summarizer."
        ),
    )

    ml_preprocessing_profile: Optional[str] = Field(
        default=None,
        alias="ml_preprocessing_profile",
        description=(
            "ML-only text cleaning profile ID (e.g. 'cleaning_v4', 'cleaning_v3') "
            "applied before BART/LED/SummLlama summarization. When set, overrides "
            "the mode_cfg.preprocessing_profile default in ml_provider.py / "
            "hybrid_ml_provider.py. Ignored by cloud LLM providers and Ollama, "
            "which send raw transcripts (GitHub #634 Scope 2, Option A: ML-only "
            "scope made explicit in name)."
        ),
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _resolve_profile(cls, data: Any) -> Any:
        """Resolve ``profile`` field: load named preset, merge as defaults (#593).

        If ``profile: cloud_balanced`` is set, loads
        ``config/profiles/cloud_balanced.yaml`` and merges its values as
        defaults — explicit fields in ``data`` override profile defaults.
        The ``profile`` key is consumed and not passed to Pydantic fields.
        """
        if not isinstance(data, dict):
            return data
        profile_name = data.pop("profile", None)
        if not profile_name:
            # Still resolve audio preprocessing preset if user set it directly,
            # without a deployment profile.
            return cls._merge_audio_preprocessing_preset(data)

        profile_name = str(profile_name).strip()
        # Resolve profile YAML path
        from pathlib import Path

        # Try config/profiles/ relative to project root, then package resources
        candidates = [
            Path("config/profiles") / f"{profile_name}.yaml",
            Path(__file__).parent.parent.parent / "config" / "profiles" / f"{profile_name}.yaml",
        ]
        profile_path = None
        for c in candidates:
            if c.is_file():
                profile_path = c
                break

        if profile_path is None:
            import logging

            logging.getLogger(__name__).warning(
                "Profile '%s' not found in config/profiles/; ignoring", profile_name
            )
            # Still resolve audio preset if set directly on data.
            return cls._merge_audio_preprocessing_preset(data)

        # Load profile YAML
        import yaml

        profile_dict = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}

        # Merge: profile provides defaults, explicit data overrides
        merged = dict(profile_dict)
        merged.update(data)  # explicit fields win

        # After deployment profile resolution, also resolve
        # audio_preprocessing_profile if set (#634 Scope 1). Inlined here rather
        # than as a separate @model_validator because Pydantic v2 runs mode=before
        # validators in reverse definition order, making cross-validator ordering
        # fragile when one depends on the output of another.
        merged = cls._merge_audio_preprocessing_preset(merged)
        return merged

    @classmethod
    def _merge_audio_preprocessing_preset(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Load audio/<name>.yaml preset and merge its preprocessing_* fields
        as defaults beneath ``data``. Explicit data (deployment profile or CLI)
        wins on overlap.

        No-op if ``audio_preprocessing_profile`` is unset. Warns if the named
        preset YAML is not found.
        """
        preset_name = data.get("audio_preprocessing_profile")
        if not preset_name:
            return data

        from pathlib import Path

        preset_name = str(preset_name).strip()
        candidates = [
            Path("config/profiles/audio") / f"{preset_name}.yaml",
            (
                Path(__file__).parent.parent.parent
                / "config"
                / "profiles"
                / "audio"
                / f"{preset_name}.yaml"
            ),
        ]
        preset_path = None
        for c in candidates:
            if c.is_file():
                preset_path = c
                break

        if preset_path is None:
            import logging

            logging.getLogger(__name__).warning(
                "Audio preprocessing profile '%s' not found in config/profiles/audio/; " "ignoring",
                preset_name,
            )
            return data

        import yaml

        preset_dict = yaml.safe_load(preset_path.read_text(encoding="utf-8")) or {}
        merged = dict(preset_dict)
        merged.update(data)
        return merged

    @model_validator(mode="before")
    @classmethod
    def _normalize_multi_rss_input(cls, data: Any) -> Any:
        """Promote ``rss`` list or merge ``rss`` + ``feeds``/``rss_urls`` (GitHub #440)."""
        if not isinstance(data, dict):
            return data
        d = dict(data)
        rss_raw = d.get("rss")
        rss_urls_key = d.get("rss_urls")
        feeds_key = d.get("feeds")
        feeds_val = rss_urls_key if rss_urls_key is not None else feeds_key

        if isinstance(rss_raw, list):
            urls: List[str] = []
            for u in rss_raw:
                if u is not None and str(u).strip():
                    t = str(u).strip()
                    if t not in urls:
                        urls.append(t)
            if isinstance(feeds_val, list):
                for u in feeds_val:
                    if u is not None and str(u).strip():
                        t = str(u).strip()
                        if t not in urls:
                            urls.append(t)
            d.pop("rss", None)
            d.pop("feeds", None)
            d.pop("rss_urls", None)
            d["rss_urls"] = urls
            return d

        if isinstance(rss_raw, str) and rss_raw.strip() and isinstance(feeds_val, list):
            urls = [rss_raw.strip()]
            for u in feeds_val:
                if u is not None and str(u).strip():
                    t = str(u).strip()
                    if t not in urls:
                        urls.append(t)
            if len(urls) >= 2:
                d.pop("rss", None)
                d.pop("feeds", None)
                d.pop("rss_urls", None)
                d["rss_urls"] = urls
            else:
                d.pop("feeds", None)
                if rss_urls_key is not None:
                    d.pop("rss_urls", None)
            return d

        only_list: Optional[List[Any]] = None
        if isinstance(feeds_key, list):
            only_list = feeds_key
        elif isinstance(rss_urls_key, list):
            only_list = rss_urls_key
        if (
            only_list is not None
            and len(only_list) == 1
            and not (isinstance(rss_raw, str) and rss_raw.strip())
        ):
            one = str(only_list[0]).strip()
            if one:
                d["rss"] = one
                d.pop("feeds", None)
                d.pop("rss_urls", None)
        return d

    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_fields(cls, data: Any) -> Any:
        """Handle deprecated field names for backward compatibility."""
        if isinstance(data, dict):
            if "multi_feed_soft_fail_exit_zero" in data and "multi_feed_strict" in data:
                raise ValueError(
                    "Use only multi_feed_strict; remove deprecated multi_feed_soft_fail_exit_zero"
                )
            if "multi_feed_soft_fail_exit_zero" in data:
                import warnings

                warnings.warn(
                    "multi_feed_soft_fail_exit_zero is deprecated; use multi_feed_strict "
                    "(strict means fail the run on any feed failure, including soft-only)",
                    DeprecationWarning,
                    stacklevel=2,
                )
                legacy = data.pop("multi_feed_soft_fail_exit_zero")
                data["multi_feed_strict"] = not bool(legacy)
            # Map deprecated speaker_detector_type to speaker_detector_provider
            if "speaker_detector_type" in data and "speaker_detector_provider" not in data:
                import warnings

                warnings.warn(
                    "speaker_detector_type is deprecated, use speaker_detector_provider instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Map "ner" -> "spacy" for backward compatibility
                old_value = data["speaker_detector_type"]
                if old_value == "ner":
                    data["speaker_detector_provider"] = "spacy"
                elif old_value == "openai":
                    data["speaker_detector_provider"] = "openai"
                else:
                    # Unknown value, pass through and let validation catch it
                    data["speaker_detector_provider"] = old_value
                # Remove deprecated field
                del data["speaker_detector_type"]
        return data

    @model_validator(mode="before")
    @classmethod
    def _coerce_screenplay_for_api_transcription_before(cls, data: Any) -> Any:
        """Screenplay applies only to local Whisper; coerce YAML/CLI for API paths (GitHub #562).

        Implemented as ``mode='before'`` so ``Config(**kwargs)`` applies: frozen Config ignores
        non-``self`` returns from ``mode='after'`` validators under ``__init__`` (see
        ``_align_gil_evidence_with_summary_provider`` docstring).
        """
        if not isinstance(data, dict):
            return data
        merged = dict(data)
        tp = merged.get("transcription_provider", "whisper")
        if not _raw_screenplay_requested(merged.get("screenplay")):
            return merged
        if tp == "whisper":
            return merged
        if _screenplay_strict_env_enabled():
            raise ValueError(
                "screenplay is only supported with transcription_provider='whisper' "
                f"(got transcription_provider={tp!r}). Remove screenplay or switch to "
                "whisper. To allow automatic coercion to screenplay=false, unset "
                "PODCAST_SCRAPER_SCREENPLAY_STRICT (GitHub #562)."
            )
        merged["screenplay"] = False
        with _screenplay_tx_api_coerce_lock:
            should_log = not _screenplay_tx_api_coerce_state["logged"]
            if should_log:
                _screenplay_tx_api_coerce_state["logged"] = True
        if should_log:
            logger.info(
                "screenplay applies only to transcription_provider='whisper'; "
                "with transcription_provider=%r screenplay has no effect — "
                "coercing screenplay=false (GitHub #562).",
                tp,
            )
        return merged

    @model_validator(mode="before")
    @classmethod
    def _align_gil_evidence_with_summary_provider(cls, data: Any) -> Any:
        """Set quote/entail providers from summary_provider when still default transformers.

        Implemented as ``mode='before'`` so ``Config(**kwargs)`` picks up changes: Pydantic v2
        does not substitute instances returned from ``mode='after'`` / ``mode='wrap'`` when
        validating via ``__init__``.
        """
        if not isinstance(data, dict):
            return data
        merged = dict(data)
        if merged.get("gil_evidence_match_summary_provider", True) is not True:
            return merged
        if not merged.get("generate_gi", False):
            return merged
        summary = merged.get("summary_provider", "transformers")
        if summary not in GIL_EVIDENCE_ALIGN_SUMMARY_PROVIDERS:
            return merged
        quote = merged.get("quote_extraction_provider", "transformers")
        entail = merged.get("entailment_provider", "transformers")
        if quote == "transformers" and entail == "transformers":
            merged["quote_extraction_provider"] = summary
            merged["entailment_provider"] = summary
        return merged

    @field_validator("vector_index_types", mode="before")
    @classmethod
    def _normalize_vector_index_types(cls, value: Any) -> Any:
        if value is None or value == []:
            return None
        return value

    @field_validator("rss_url", mode="before")
    @classmethod
    def _strip_rss(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("rss_url", mode="after")
    @classmethod
    def _validate_rss_url_scheme(cls, value: Optional[str]) -> Optional[str]:
        """Require http(s) and hostname when rss_url is set (matches CLI validation)."""
        if not value:
            return value
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"RSS URL must use http or https (got {parsed.scheme!r}): {value}")
        if not parsed.netloc:
            raise ValueError(f"RSS URL must include a valid hostname: {value}")
        return value

    @field_validator("rss_urls", mode="before")
    @classmethod
    def _coerce_rss_urls_list(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError("rss_urls must be a list of URL strings")
        out = [str(u).strip() for u in value if u is not None and str(u).strip()]
        return out or None

    @field_validator("rss_urls", mode="after")
    @classmethod
    def _validate_rss_urls_entries(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if not value:
            return value
        for u in value:
            parsed = urlparse(u)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"RSS URL must use http or https (got {parsed.scheme!r}): {u}")
            if not parsed.netloc:
                raise ValueError(f"RSS URL must include a valid hostname: {u}")
        return value

    @field_validator("output_dir", mode="before")
    @classmethod
    def _load_output_dir_from_env(cls, value: Any) -> Optional[str]:
        """Load output directory from environment variable if not provided."""
        # Check environment variable first (loaded from .env by dotenv)
        # This allows env var to be used when value is None (default)
        env_output_dir = os.getenv("OUTPUT_DIR")
        if env_output_dir:
            env_value = str(env_output_dir).strip()
            if env_value:
                # If explicitly provided in config, config takes precedence
                if value is not None and str(value).strip():
                    return str(value).strip() or None
                return env_value

        # If explicitly provided in config, use it
        if value is not None:
            value_str = str(value).strip()
            return value_str or None

        return None

    @field_validator("whisper_model", "language", mode="before")
    @classmethod
    def _coerce_string(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("user_agent", mode="before")
    @classmethod
    def _coerce_user_agent(cls, value: Any) -> str:
        if value is None:
            return DEFAULT_USER_AGENT
        value_str = str(value).strip()
        if not value_str:
            return DEFAULT_USER_AGENT
        return value_str

    @field_validator("user_agent", mode="after")
    @classmethod
    def _validate_user_agent(cls, value: str) -> str:
        """Validate user agent is not empty."""
        if not value:
            return DEFAULT_USER_AGENT
        return value

    @field_validator("whisper_model", mode="after")
    @classmethod
    def _validate_whisper_model(cls, value: str) -> str:
        if value and value not in VALID_WHISPER_MODELS:
            raise ValueError(f"whisper_model must be one of {VALID_WHISPER_MODELS}, got: {value}")
        return value

    @staticmethod
    def _load_string_env_var(
        data: Dict[str, Any],
        field_name: str,
        env_var_name: str,
        validator: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Load a string environment variable into config data if field is missing.

        Args:
            data: Configuration data dictionary
            field_name: Name of the field in config
            env_var_name: Name of the environment variable
            validator: Optional validator function (env_value) -> bool
        """
        if field_name not in data or data.get(field_name) is None:
            env_value = os.getenv(env_var_name)
            if env_value:
                value = str(env_value).strip()
                if value and (validator is None or validator(value)):
                    data[field_name] = value

    @staticmethod
    def _load_int_env_var(
        data: Dict[str, Any],
        field_name: str,
        env_var_name: str,
        min_value: int = 1,
        max_value: Optional[int] = None,
    ) -> None:
        """Load an integer environment variable into config data if field is missing.

        Args:
            data: Configuration data dictionary
            field_name: Name of the field in config
            env_var_name: Name of the environment variable
            min_value: Minimum valid value (default: 1)
            max_value: Optional inclusive maximum (invalid env values are ignored)
        """
        if field_name not in data or data.get(field_name) is None:
            env_value = os.getenv(env_var_name)
            if env_value:
                try:
                    int_value = int(env_value)
                    if int_value < min_value:
                        return
                    if max_value is not None and int_value > max_value:
                        return
                    data[field_name] = int_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

    @staticmethod
    def _load_float_env_var(
        data: Dict[str, Any],
        field_name: str,
        env_var_name: str,
        min_value: float = 0.0,
        max_value: Optional[float] = None,
    ) -> None:
        """Load a float environment variable into config data if field is missing.

        Args:
            data: Configuration data dictionary
            field_name: Name of the field in config
            env_var_name: Name of the environment variable
            min_value: Minimum valid value (default: 0.0)
            max_value: Optional maximum valid value
        """
        if field_name not in data or data.get(field_name) is None:
            env_value = os.getenv(env_var_name)
            if env_value:
                try:
                    float_value = float(env_value)
                    if float_value >= min_value and (max_value is None or float_value <= max_value):
                        data[field_name] = float_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

    @staticmethod
    def _load_bool_env_var(data: Dict[str, Any], field_name: str, env_var_name: str) -> None:
        """Load a boolean environment variable into config data if field is missing.

        Args:
            data: Configuration data dictionary
            field_name: Name of the field in config
            env_var_name: Name of the environment variable
        """
        if field_name not in data:
            env_value = os.getenv(env_var_name)
            if env_value:
                value = str(env_value).strip().lower()
                if value in ("1", "true", "yes", "on"):
                    data[field_name] = True
                elif value in ("0", "false", "no", "off"):
                    data[field_name] = False

    @staticmethod
    def _load_device_env_var(data: Dict[str, Any], field_name: str, env_var_name: str) -> None:
        """Load a device environment variable into config data if field is missing.

        Args:
            data: Configuration data dictionary
            field_name: Name of the field in config
            env_var_name: Name of the environment variable
        """
        if field_name not in data or data.get(field_name) is None:
            env_value = os.getenv(env_var_name)
            if env_value:
                value = str(env_value).strip().lower()
                if value in ("cpu", "cuda", "mps"):
                    data[field_name] = value

    @classmethod
    def _load_summary_cache_dir_from_env(cls, data: Dict[str, Any]) -> None:
        """Load summary cache directory from environment variables.

        Checks SUMMARY_CACHE_DIR, CACHE_DIR, and local project cache.

        Args:
            data: Configuration data dictionary
        """
        if "summary_cache_dir" not in data or data.get("summary_cache_dir") is None:
            # Check SUMMARY_CACHE_DIR first (explicit override)
            summary_cache = os.getenv("SUMMARY_CACHE_DIR")
            if summary_cache:
                value = str(summary_cache).strip()
                if value:
                    data["summary_cache_dir"] = value
                    return

            # If CACHE_DIR is set, derive summary cache from it
            cache_dir = os.getenv("CACHE_DIR")
            if cache_dir:
                # Derive Transformers cache path from CACHE_DIR
                derived_cache = str(Path(cache_dir) / "huggingface" / "hub")
                data["summary_cache_dir"] = derived_cache
                return

            # Check for local cache in project root
            try:
                from .cache import get_project_root

                project_root = get_project_root()
                local_cache = project_root / ".cache" / "huggingface" / "hub"
                if local_cache.exists():
                    data["summary_cache_dir"] = str(local_cache)
            except Exception:
                # If cache_utils not available, continue without local cache
                pass

    @model_validator(mode="before")
    @classmethod
    def _preprocess_config_data(cls, data: Any) -> Any:
        """Preprocess configuration data before validation.

        Handles:
        - Mapping deprecated field names to new names
        - Loading configuration from environment variables
        """
        if not isinstance(data, dict):
            return data

        # Load configuration from environment variables
        # LOG_LEVEL: Environment variable takes precedence (special case)
        env_log_level = os.getenv("LOG_LEVEL")
        if env_log_level:
            env_value = str(env_log_level).strip().upper()
            if env_value and env_value in VALID_LOG_LEVELS:
                data["log_level"] = env_value

        # Load string environment variables
        cls._load_string_env_var(data, "output_dir", "OUTPUT_DIR")
        cls._load_string_env_var(data, "log_file", "LOG_FILE")
        cls._load_string_env_var(data, "openai_api_key", "OPENAI_API_KEY")
        cls._load_string_env_var(data, "openai_api_base", "OPENAI_API_BASE")
        cls._load_string_env_var(data, "gemini_api_key", "GEMINI_API_KEY")
        cls._load_string_env_var(data, "gemini_api_base", "GEMINI_API_BASE")
        cls._load_string_env_var(data, "anthropic_api_key", "ANTHROPIC_API_KEY")
        cls._load_string_env_var(data, "anthropic_api_base", "ANTHROPIC_API_BASE")
        cls._load_string_env_var(data, "mistral_api_key", "MISTRAL_API_KEY")
        cls._load_string_env_var(data, "mistral_api_base", "MISTRAL_API_BASE")
        cls._load_string_env_var(data, "deepseek_api_key", "DEEPSEEK_API_KEY")
        cls._load_string_env_var(data, "deepseek_api_base", "DEEPSEEK_API_BASE")
        cls._load_string_env_var(data, "grok_api_key", "GROK_API_KEY")
        cls._load_string_env_var(data, "grok_api_base", "GROK_API_BASE")
        cls._load_string_env_var(data, "ollama_api_base", "OLLAMA_API_BASE")
        cls._load_string_env_var(data, "pricing_assumptions_file", "PRICING_ASSUMPTIONS_FILE")

        # Load integer environment variables
        cls._load_int_env_var(data, "workers", "WORKERS")
        cls._load_int_env_var(data, "transcription_parallelism", "TRANSCRIPTION_PARALLELISM")
        cls._load_int_env_var(data, "processing_parallelism", "PROCESSING_PARALLELISM")
        cls._load_int_env_var(data, "summary_batch_size", "SUMMARY_BATCH_SIZE")
        cls._load_int_env_var(data, "summary_chunk_parallelism", "SUMMARY_CHUNK_PARALLELISM")
        cls._load_int_env_var(data, "timeout", "TIMEOUT")
        cls._load_int_env_var(data, "seed", "SEED")

        # Download resilience (optional; prefixed env vars — config file / kwargs win if set)
        cls._load_int_env_var(
            data, "http_retry_total", "PODCAST_SCRAPER_HTTP_RETRY_TOTAL", min_value=0, max_value=20
        )
        cls._load_float_env_var(
            data, "http_backoff_factor", "PODCAST_SCRAPER_HTTP_BACKOFF_FACTOR", 0.0, 10.0
        )
        cls._load_int_env_var(
            data, "rss_retry_total", "PODCAST_SCRAPER_RSS_RETRY_TOTAL", min_value=0, max_value=20
        )
        cls._load_float_env_var(
            data, "rss_backoff_factor", "PODCAST_SCRAPER_RSS_BACKOFF_FACTOR", 0.0, 10.0
        )
        cls._load_int_env_var(
            data,
            "episode_retry_max",
            "PODCAST_SCRAPER_EPISODE_RETRY_MAX",
            min_value=0,
            max_value=10,
        )
        cls._load_float_env_var(
            data,
            "episode_retry_delay_sec",
            "PODCAST_SCRAPER_EPISODE_RETRY_DELAY_SEC",
            0.0,
            120.0,
        )
        cls._load_int_env_var(
            data,
            "host_request_interval_ms",
            "PODCAST_SCRAPER_HOST_REQUEST_INTERVAL_MS",
            min_value=0,
            max_value=600_000,
        )
        cls._load_int_env_var(
            data,
            "host_max_concurrent",
            "PODCAST_SCRAPER_HOST_MAX_CONCURRENT",
            min_value=0,
            max_value=64,
        )
        cls._load_bool_env_var(
            data, "circuit_breaker_enabled", "PODCAST_SCRAPER_CIRCUIT_BREAKER_ENABLED"
        )
        cls._load_int_env_var(
            data,
            "circuit_breaker_failure_threshold",
            "PODCAST_SCRAPER_CIRCUIT_BREAKER_FAILURE_THRESHOLD",
            min_value=1,
            max_value=100,
        )
        cls._load_int_env_var(
            data,
            "circuit_breaker_window_seconds",
            "PODCAST_SCRAPER_CIRCUIT_BREAKER_WINDOW_SECONDS",
            min_value=1,
            max_value=86_400,
        )
        cls._load_int_env_var(
            data,
            "circuit_breaker_cooldown_seconds",
            "PODCAST_SCRAPER_CIRCUIT_BREAKER_COOLDOWN_SECONDS",
            min_value=1,
            max_value=86_400,
        )
        _raw_cb_scope = os.getenv("PODCAST_SCRAPER_CIRCUIT_BREAKER_SCOPE")
        if _raw_cb_scope and (
            "circuit_breaker_scope" not in data or data.get("circuit_breaker_scope") is None
        ):
            _s = str(_raw_cb_scope).strip().lower()
            if _s in ("feed", "host"):
                data["circuit_breaker_scope"] = _s
        cls._load_bool_env_var(data, "rss_conditional_get", "PODCAST_SCRAPER_RSS_CONDITIONAL_GET")
        cls._load_string_env_var(data, "rss_cache_dir", "PODCAST_SCRAPER_RSS_CACHE_DIR")

        # Load float environment variables
        cls._load_float_env_var(data, "openai_temperature", "OPENAI_TEMPERATURE", 0.0, 2.0)

        # Load boolean environment variables
        cls._load_bool_env_var(data, "mps_exclusive", "MPS_EXCLUSIVE")

        # Load device environment variables
        cls._load_device_env_var(data, "summary_device", "SUMMARY_DEVICE")
        cls._load_device_env_var(data, "whisper_device", "WHISPER_DEVICE")

        # Load summary cache directory (special handling)
        cls._load_summary_cache_dir_from_env(data)

        return data

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> str:
        """Normalize log level value."""
        if value is None:
            return DEFAULT_LOG_LEVEL
        return str(value).strip().upper() or DEFAULT_LOG_LEVEL

    @field_validator("log_level", mode="after")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Validate log level is one of the valid levels."""
        if value not in VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {VALID_LOG_LEVELS}, got: {value}")
        return value

    @field_validator("log_file", mode="before")
    @classmethod
    def _load_log_file_from_env(cls, value: Any) -> Optional[str]:
        """Load log file path from environment variable if not provided."""
        # Check environment variable first (loaded from .env by dotenv)
        # This allows env var to be used when value is None (default)
        env_log_file = os.getenv("LOG_FILE")
        if env_log_file:
            env_value = str(env_log_file).strip()
            if env_value:
                # If explicitly provided in config, config takes precedence
                if value is not None and str(value).strip():
                    return str(value).strip() or None
                return env_value

        # If explicitly provided in config, use it
        if value is not None:
            value_str = str(value).strip()
            return value_str or None

        return None

    @field_validator("run_id", mode="before")
    @classmethod
    def _strip_run_id(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    @field_validator("run_id", mode="after")
    @classmethod
    def _validate_run_id(cls, value: Optional[str]) -> Optional[str]:
        """Validate run_id length and characters."""
        if value is None:
            return None
        if len(value) > MAX_RUN_ID_LENGTH:
            raise ValueError(
                f"run_id must be at most {MAX_RUN_ID_LENGTH} characters, got {len(value)}"
            )
        # Check for invalid characters (path separators, control characters)
        if "/" in value or "\\" in value:
            raise ValueError("run_id cannot contain path separators (/, \\)")
        if any(ord(c) < 32 for c in value):  # Control characters
            raise ValueError("run_id cannot contain control characters")
        return value

    @field_validator("max_episodes", mode="before")
    @classmethod
    def _coerce_max_episodes(cls, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_episodes must be an integer") from exc
        return parsed if parsed > 0 else None

    @field_validator("episode_order", mode="before")
    @classmethod
    def _normalize_episode_order(cls, value: Any) -> str:
        if value is None or value == "":
            return "newest"
        s = str(value).strip().lower()
        if s not in ("newest", "oldest"):
            raise ValueError("episode_order must be 'newest' or 'oldest'")
        return s

    @field_validator("episode_offset", mode="before")
    @classmethod
    def _coerce_episode_offset(cls, value: Any) -> int:
        if value is None or value == "":
            return 0
        try:
            offset = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("episode_offset must be an integer") from exc
        if offset < 0:
            raise ValueError("episode_offset must be non-negative")
        return offset

    @field_validator("timeout", mode="before")
    @classmethod
    def _ensure_timeout(cls, value: Any) -> int:
        if value is None or value == "":
            return DEFAULT_TIMEOUT_SECONDS
        try:
            timeout = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("timeout must be an integer") from exc
        return max(MIN_TIMEOUT_SECONDS, timeout)

    @field_validator("delay_ms", mode="before")
    @classmethod
    def _ensure_delay_ms(cls, value: Any) -> int:
        if value is None or value == "":
            return 0
        try:
            delay = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("delay_ms must be an integer") from exc
        if delay < 0:
            raise ValueError("delay_ms must be non-negative")
        return delay

    @field_validator("prefer_types", mode="before")
    @classmethod
    def _coerce_prefer_types(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("prefer_types must be a string or list of strings")

    @field_validator("screenplay_speaker_names", mode="before")
    @classmethod
    def _coerce_speaker_names(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        raise TypeError("speaker_names must be a string or list of strings")

    @field_validator("screenplay_gap_s", mode="before")
    @classmethod
    def _ensure_screenplay_gap(cls, value: Any) -> float:
        if value is None or value == "":
            return DEFAULT_SCREENPLAY_GAP_SECONDS
        try:
            gap = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("screenplay_gap must be a number") from exc
        if gap <= 0:
            raise ValueError("screenplay_gap must be positive")
        return gap

    @field_validator("screenplay_num_speakers", mode="before")
    @classmethod
    def _ensure_num_speakers(cls, value: Any) -> int:
        if value is None or value == "":
            return DEFAULT_NUM_SPEAKERS
        try:
            speakers = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("num_speakers must be an integer") from exc
        if speakers < MIN_NUM_SPEAKERS:
            raise ValueError(f"num_speakers must be at least {MIN_NUM_SPEAKERS}")
        return speakers

    @field_validator("workers", mode="before")
    @classmethod
    def _ensure_workers(cls, value: Any) -> int:
        if value is None or value == "":
            return DEFAULT_WORKERS
        try:
            workers = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("workers must be an integer") from exc
        if workers < 1:
            raise ValueError("workers must be at least 1")
        return workers

    @field_validator("language", mode="after")
    @classmethod
    def _normalize_language(cls, value: str) -> str:
        """Normalize language code to lowercase."""
        if not value:
            return DEFAULT_LANGUAGE
        return value.lower().strip() or DEFAULT_LANGUAGE

    @field_validator("ner_model", mode="before")
    @classmethod
    def _coerce_ner_model(cls, value: Any) -> Optional[str]:
        """Coerce NER model to string or None."""
        if value is None or value == "":
            return None
        return str(value).strip() or None

    @field_validator("metadata_format", mode="before")
    @classmethod
    def _validate_metadata_format(cls, value: Any) -> Literal["json", "yaml"]:
        """Validate metadata format."""
        if value is None or value == "":
            return "json"
        value_str = str(value).strip().lower()
        if value_str not in ("json", "yaml"):
            raise ValueError("metadata_format must be 'json' or 'yaml'")
        return value_str  # type: ignore[return-value]

    @field_validator("metadata_subdirectory", mode="before")
    @classmethod
    def _strip_metadata_subdirectory(cls, value: Any) -> Optional[str]:
        """Strip and validate metadata subdirectory."""
        if value is None or value == "":
            return None
        value_str = str(value).strip()
        return value_str or None

    @field_validator("metadata_subdirectory", mode="after")
    @classmethod
    def _validate_metadata_subdirectory(cls, value: Optional[str]) -> Optional[str]:
        """Validate metadata subdirectory is a valid directory name."""
        if value is None:
            return None
        if not value:
            raise ValueError("metadata_subdirectory cannot be empty if provided")
        if len(value) > MAX_METADATA_SUBDIRECTORY_LENGTH:
            raise ValueError(
                f"metadata_subdirectory must be at most "
                f"{MAX_METADATA_SUBDIRECTORY_LENGTH} characters, got {len(value)}"
            )
        # Check for invalid path separators (absolute paths or parent directory references)
        if value.startswith("/") or value.startswith("\\"):
            raise ValueError("metadata_subdirectory cannot be an absolute path")
        if ".." in value:
            raise ValueError(
                "metadata_subdirectory cannot contain '..' (parent directory references)"
            )
        # Check for invalid characters
        invalid_chars = set(value) & set(
            "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f"
            "\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f"
        )
        if invalid_chars:
            raise ValueError("metadata_subdirectory cannot contain control characters")
        return value

    @staticmethod
    def _validate_path_no_traversal(v: Optional[str], field_name: str) -> Optional[str]:
        """Reject paths containing '..' traversal components."""
        if v is None:
            return v
        from pathlib import PurePosixPath, PureWindowsPath

        for cls in (PurePosixPath, PureWindowsPath):
            if ".." in cls(v).parts:
                raise ValueError(f"{field_name} must not contain " f"'..' path traversal")
        return v

    @field_validator("output_dir", mode="after")
    @classmethod
    def _validate_output_dir_traversal(cls, v: Optional[str]) -> Optional[str]:
        return cls._validate_path_no_traversal(v, "output_dir")

    @field_validator("log_file", mode="after")
    @classmethod
    def _validate_log_file_traversal(cls, v: Optional[str]) -> Optional[str]:
        return cls._validate_path_no_traversal(v, "log_file")

    @field_validator("gemini_cleaning_model", mode="after")
    @classmethod
    def _migrate_legacy_gemini_cleaning_model(cls, v: str) -> str:
        """Remap deprecated model IDs that 404 on current Generative API."""
        if v == "gemini-1.5-flash":
            return "gemini-2.5-flash-lite"
        return v

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def _load_openai_api_key_from_env(cls, value: Any) -> Optional[str]:
        """Load OpenAI API key from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key.strip() or None
        return None

    @field_validator("openai_api_base", mode="before")
    @classmethod
    def _load_openai_api_base_from_env(cls, value: Any) -> Optional[str]:
        """Load OpenAI API base URL from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_base = os.getenv("OPENAI_API_BASE")
        if env_base:
            return env_base.strip() or None
        return None

    @field_validator("mistral_api_key", mode="before")
    @classmethod
    def _load_mistral_api_key_from_env(cls, value: Any) -> Optional[str]:
        """Load Mistral API key from environment variable if not provided."""
        # If value is provided and not empty, use it
        if value is not None and str(value).strip():
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_key = os.getenv("MISTRAL_API_KEY")
        if env_key:
            return env_key.strip() or None
        return None

    @field_validator("mistral_api_base", mode="before")
    @classmethod
    def _load_mistral_api_base_from_env(cls, value: Any) -> Optional[str]:
        """Load Mistral API base URL from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_base = os.getenv("MISTRAL_API_BASE")
        if env_base:
            return env_base.strip() or None
        return None

    @field_validator("speaker_detector_provider", mode="before")
    @classmethod
    def _validate_speaker_detector_provider(
        cls, value: Any
    ) -> Literal[
        "spacy", "ner", "openai", "gemini", "mistral", "deepseek", "anthropic", "grok", "ollama"
    ]:
        """Validate speaker detector provider type."""
        if value is None or value == "":
            return "spacy"
        value_str = str(value).strip().lower()

        if value_str not in (
            "spacy",
            "openai",
            "gemini",
            "mistral",
            "deepseek",
            "anthropic",
            "grok",
            "ollama",
        ):
            raise ValueError(
                "speaker_detector_provider must be 'spacy', 'openai', 'gemini', "
                "'mistral', 'deepseek', 'anthropic', 'grok', or 'ollama'"
            )
        return value_str  # type: ignore[return-value]

    @field_validator("transcription_provider", mode="before")
    @classmethod
    def _validate_transcription_provider(
        cls, value: Any
    ) -> Literal["whisper", "openai", "gemini", "mistral", "anthropic"]:
        """Validate transcription provider."""
        if value is None or value == "":
            return "whisper"
        value_str = str(value).strip().lower()
        if value_str not in ("whisper", "openai", "gemini", "mistral", "anthropic"):
            raise ValueError(
                "transcription_provider must be 'whisper', 'openai', 'gemini', "
                "'mistral', or 'anthropic'"
            )
        return value_str  # type: ignore[return-value]

    @field_validator("summary_provider", mode="before")
    @classmethod
    def _validate_summary_provider(cls, value: Any) -> Literal[
        "transformers",
        "hybrid_ml",
        "summllama",
        "openai",
        "gemini",
        "grok",
        "deepseek",
        "anthropic",
        "ollama",
    ]:
        """Validate summary provider."""
        if value is None or value == "":
            return "transformers"
        value_str = str(value).strip().lower()

        if value_str not in (
            "transformers",
            "hybrid_ml",
            "summllama",
            "openai",
            "gemini",
            "grok",
            "mistral",
            "deepseek",
            "anthropic",
            "ollama",
        ):
            raise ValueError(
                "summary_provider must be 'transformers', 'hybrid_ml', 'summllama', "
                "'openai', 'gemini', 'grok', 'mistral', 'deepseek', 'anthropic', or 'ollama'"
            )
        return value_str  # type: ignore[return-value]

    @field_validator("quote_extraction_provider", "entailment_provider", mode="before")
    @classmethod
    def _validate_evidence_providers(cls, value: Any) -> Literal[
        "transformers",
        "hybrid_ml",
        "openai",
        "gemini",
        "grok",
        "deepseek",
        "anthropic",
        "ollama",
    ]:
        """Validate quote_extraction_provider and entailment_provider (same as summary)."""
        if value is None or value == "":
            return "transformers"
        value_str = str(value).strip().lower()
        if value_str not in (
            "transformers",
            "hybrid_ml",
            "openai",
            "gemini",
            "grok",
            "mistral",
            "deepseek",
            "anthropic",
            "ollama",
        ):
            raise ValueError(
                "quote_extraction_provider/entailment_provider must be one of: "
                "'transformers', 'hybrid_ml', 'openai', 'gemini', 'grok', "
                "'mistral', 'deepseek', 'anthropic', 'ollama'"
            )
        return value_str  # type: ignore[return-value]

    @field_validator("kg_extraction_provider", mode="before")
    @classmethod
    def _validate_kg_extraction_provider(cls, value: Any) -> Optional[str]:
        """Optional KG LLM backend; empty means use summary_provider."""
        if value is None or value == "":
            return None
        value_str = str(value).strip().lower()
        if value_str not in (
            "transformers",
            "hybrid_ml",
            "openai",
            "gemini",
            "grok",
            "mistral",
            "deepseek",
            "anthropic",
            "ollama",
        ):
            raise ValueError(
                "kg_extraction_provider must be one of: "
                "'transformers', 'hybrid_ml', 'openai', 'gemini', 'grok', "
                "'mistral', 'deepseek', 'anthropic', 'ollama'"
            )
        return value_str

    @field_validator("hybrid_map_device", "hybrid_reduce_device", mode="before")
    @classmethod
    def _coerce_hybrid_devices(cls, value: Any) -> Optional[str]:
        """Coerce hybrid device fields to string or None (supports 'auto')."""
        if value is None or value == "":
            return None
        value_str = str(value).strip().lower()
        if value_str == "auto":
            return None
        if value_str not in ("cuda", "mps", "cpu"):
            raise ValueError("hybrid_*_device must be 'cuda', 'mps', 'cpu', or 'auto'")
        return value_str

    @field_validator("openai_temperature", mode="before")
    @classmethod
    def _validate_openai_temperature(cls, value: Any) -> float:
        """Validate OpenAI temperature is in valid range (0.0-2.0)."""
        if value is None or value == "":
            return 0.3
        try:
            temp = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("openai_temperature must be a number") from exc
        if temp < 0.0 or temp > 2.0:
            raise ValueError("openai_temperature must be between 0.0 and 2.0")
        return temp

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _load_gemini_api_key_from_env(cls, value: Any) -> Optional[str]:
        """Load Gemini API key from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            return env_key.strip() or None
        return None

    @field_validator("gemini_api_base", mode="before")
    @classmethod
    def _load_gemini_api_base_from_env(cls, value: Any) -> Optional[str]:
        """Load Gemini API base URL from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_base = os.getenv("GEMINI_API_BASE")
        if env_base:
            return env_base.strip() or None
        return None

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _load_anthropic_api_key_from_env(cls, value: Any) -> Optional[str]:
        """Load Anthropic API key from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            return env_key.strip() or None
        return None

    @field_validator("anthropic_api_base", mode="before")
    @classmethod
    def _load_anthropic_api_base_from_env(cls, value: Any) -> Optional[str]:
        """Load Anthropic API base URL from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_base = os.getenv("ANTHROPIC_API_BASE")
        if env_base:
            return env_base.strip() or None
        return None

    @field_validator("gemini_temperature", mode="before")
    @classmethod
    def _validate_gemini_temperature(cls, value: Any) -> float:
        """Validate Gemini temperature is in valid range (0.0-2.0)."""
        if value is None or value == "":
            return 0.3
        try:
            temp = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("gemini_temperature must be a number") from exc
        if temp < 0.0 or temp > 2.0:
            raise ValueError("gemini_temperature must be between 0.0 and 2.0")
        return temp

    @field_validator("deepseek_api_key", mode="before")
    @classmethod
    def _load_deepseek_api_key_from_env(cls, value: Any) -> Optional[str]:
        """Load DeepSeek API key from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_key = os.getenv("DEEPSEEK_API_KEY")
        if env_key:
            return env_key.strip() or None
        return None

    @field_validator("deepseek_api_base", mode="before")
    @classmethod
    def _load_deepseek_api_base_from_env(cls, value: Any) -> Optional[str]:
        """Load DeepSeek API base URL from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_base = os.getenv("DEEPSEEK_API_BASE")
        if env_base:
            return env_base.strip() or None
        # Default to DeepSeek API base URL
        return "https://api.deepseek.com"

    @field_validator("ollama_api_base", mode="before")
    @classmethod
    def _load_ollama_api_base_from_env(cls, value: Any) -> Optional[str]:
        """Load Ollama API base URL from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_base = os.getenv("OLLAMA_API_BASE")
        if env_base:
            return env_base.strip() or None
        # Default to local Ollama server
        return "http://localhost:11434/v1"

    @field_validator("ollama_temperature", mode="before")
    @classmethod
    def _validate_ollama_temperature(cls, value: Any) -> float:
        """Validate Ollama temperature is in valid range (0.0-2.0)."""
        if value is None or value == "":
            return 0.3
        try:
            temp = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("ollama_temperature must be a number") from exc
        if temp < 0.0 or temp > 2.0:
            raise ValueError("ollama_temperature must be between 0.0 and 2.0")
        return temp

    @field_validator("deepseek_temperature", mode="before")
    @classmethod
    def _validate_deepseek_temperature(cls, value: Any) -> float:
        """Validate DeepSeek temperature is in valid range (0.0-2.0)."""
        if value is None or value == "":
            return 0.3
        try:
            temp = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("deepseek_temperature must be a number") from exc
        if temp < 0.0 or temp > 2.0:
            raise ValueError("deepseek_temperature must be between 0.0 and 2.0")
        return temp

    @field_validator("grok_api_key", mode="before")
    @classmethod
    def _load_grok_api_key_from_env(cls, value: Any) -> Optional[str]:
        """Load Grok API key from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_key = os.getenv("GROK_API_KEY")
        if env_key:
            return env_key.strip() or None
        return None

    @field_validator("grok_api_base", mode="before")
    @classmethod
    def _load_grok_api_base_from_env(cls, value: Any) -> Optional[str]:
        """Load Grok API base URL from environment variable if not provided."""
        if value is not None:
            return str(value).strip() or None
        # Check environment variable (loaded from .env by dotenv)
        env_base = os.getenv("GROK_API_BASE")
        if env_base:
            return env_base.strip() or None
        # Default to Grok API base URL (OpenAI-compatible endpoint)
        return "https://api.x.ai/v1"

    @field_validator("grok_temperature", mode="before")
    @classmethod
    def _validate_grok_temperature(cls, value: Any) -> float:
        """Validate Grok temperature is in valid range (0.0-2.0)."""
        if value is None or value == "":
            return 0.3
        try:
            temp = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("grok_temperature must be a number") from exc
        if temp < 0.0 or temp > 2.0:
            raise ValueError("grok_temperature must be between 0.0 and 2.0")
        return temp

    @field_validator("anthropic_temperature", mode="after")
    @classmethod
    def _validate_anthropic_temperature(
        cls,
        v: float,
    ) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("anthropic_temperature must be between 0.0 and 1.0")
        return v

    @field_validator("mistral_temperature", mode="after")
    @classmethod
    def _validate_mistral_temperature(
        cls,
        v: float,
    ) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError("mistral_temperature must be between 0.0 and 2.0")
        return v

    @field_validator(
        "openai_cleaning_temperature",
        "gemini_cleaning_temperature",
        "ollama_cleaning_temperature",
        "deepseek_cleaning_temperature",
        "grok_cleaning_temperature",
        "mistral_cleaning_temperature",
        mode="after",
    )
    @classmethod
    def _validate_cleaning_temperature(
        cls,
        v: float,
    ) -> float:
        if v < 0.0 or v > 2.0:
            raise ValueError("cleaning temperature must be between 0.0 and 2.0")
        return v

    @field_validator("anthropic_cleaning_temperature", mode="after")
    @classmethod
    def _validate_anthropic_cleaning_temp(
        cls,
        v: float,
    ) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError("anthropic_cleaning_temperature must be " "between 0.0 and 1.0")
        return v

    @field_validator(
        "openai_max_tokens",
        "gemini_max_tokens",
        "anthropic_max_tokens",
        "ollama_max_tokens",
        "deepseek_max_tokens",
        "grok_max_tokens",
        "mistral_max_tokens",
        mode="after",
    )
    @classmethod
    def _validate_max_tokens(
        cls,
        v: Optional[int],
    ) -> Optional[int]:
        if v is not None and v < 1:
            raise ValueError("max_tokens must be >= 1")
        return v

    @model_validator(mode="after")
    def _multi_feed_requires_output_dir(self) -> "Config":
        """Corpus parent is mandatory when two or more feeds are configured (GitHub #440)."""
        urls = self.rss_urls or []
        if len(urls) < 2:
            return self
        if not self.output_dir or not str(self.output_dir).strip():
            raise ValueError(
                "When rss_urls has two or more feeds (GitHub #440), output_dir is required "
                "as the corpus parent directory (set output_dir or OUTPUT_DIR)."
            )
        return self

    @model_validator(mode="after")
    def _validate_openai_provider_requirements(self) -> "Config":
        """Validate that OpenAI API key is provided when OpenAI providers are selected."""
        openai_providers_used = []
        if self.transcription_provider == "openai":
            openai_providers_used.append("transcription")
        if self.speaker_detector_provider == "openai":
            openai_providers_used.append("speaker_detection")
        if self.summary_provider == "openai":
            openai_providers_used.append("summarization")
        if self.quote_extraction_provider == "openai":
            openai_providers_used.append("quote_extraction")
        if self.entailment_provider == "openai":
            openai_providers_used.append("entailment")

        if openai_providers_used and not self.openai_api_key:
            providers_str = ", ".join(openai_providers_used)
            raise ValueError(
                f"OpenAI API key required for OpenAI providers: {providers_str}. "
                "Set OPENAI_API_KEY environment variable or openai_api_key in config."
            )

        return self

    @model_validator(mode="after")
    def _validate_gemini_provider_requirements(self) -> "Config":
        """Validate that Gemini API key is provided when Gemini providers are selected."""
        gemini_providers_used = []
        if self.transcription_provider == "gemini":
            gemini_providers_used.append("transcription")
        if self.speaker_detector_provider == "gemini":
            gemini_providers_used.append("speaker_detection")
        if self.summary_provider == "gemini":
            gemini_providers_used.append("summarization")
        if self.quote_extraction_provider == "gemini":
            gemini_providers_used.append("quote_extraction")
        if self.entailment_provider == "gemini":
            gemini_providers_used.append("entailment")

        if gemini_providers_used and not self.gemini_api_key:
            providers_str = ", ".join(gemini_providers_used)
            raise ValueError(
                f"Gemini API key required for Gemini providers: {providers_str}. "
                "Set GEMINI_API_KEY environment variable or gemini_api_key in config."
            )

        return self

    @model_validator(mode="after")
    def _validate_anthropic_provider_requirements(self) -> "Config":
        """Validate that Anthropic API key is provided when Anthropic providers are selected."""
        anthropic_providers_used = []
        if self.transcription_provider == "anthropic":
            anthropic_providers_used.append("transcription")
        if self.speaker_detector_provider == "anthropic":
            anthropic_providers_used.append("speaker_detection")
        if self.summary_provider == "anthropic":
            anthropic_providers_used.append("summarization")
        if self.quote_extraction_provider == "anthropic":
            anthropic_providers_used.append("quote_extraction")
        if self.entailment_provider == "anthropic":
            anthropic_providers_used.append("entailment")

        if anthropic_providers_used and not self.anthropic_api_key:
            providers_str = ", ".join(anthropic_providers_used)
            raise ValueError(
                f"Anthropic API key required for Anthropic providers: {providers_str}. "
                "Set ANTHROPIC_API_KEY environment variable or anthropic_api_key in config."
            )

        return self

    @model_validator(mode="after")
    def _validate_deepseek_provider_requirements(self) -> "Config":
        """Validate that DeepSeek API key is provided when DeepSeek providers are selected."""
        deepseek_providers_used = []
        if self.speaker_detector_provider == "deepseek":
            deepseek_providers_used.append("speaker_detection")
        if self.summary_provider == "deepseek":
            deepseek_providers_used.append("summarization")
        if self.quote_extraction_provider == "deepseek":
            deepseek_providers_used.append("quote_extraction")
        if self.entailment_provider == "deepseek":
            deepseek_providers_used.append("entailment")

        if deepseek_providers_used and not self.deepseek_api_key:
            providers_str = ", ".join(deepseek_providers_used)
            raise ValueError(
                f"DeepSeek API key required for DeepSeek providers: {providers_str}. "
                "Set DEEPSEEK_API_KEY environment variable or deepseek_api_key in config."
            )

        return self

    @model_validator(mode="after")
    def _validate_provider_capabilities(self) -> "Config":
        """Validate provider supports requested capability."""
        # DeepSeek does not support transcription
        if self.transcription_provider == "deepseek":
            raise ValueError(
                "DeepSeek provider does not support transcription. "
                "Use 'whisper' (local), 'openai', 'gemini', or 'mistral' instead. "
                "See provider capability matrix in documentation."
            )
        # Anthropic does not support transcription
        if self.transcription_provider == "anthropic":
            raise ValueError(
                "Anthropic provider does not support native audio transcription. "
                "Use 'whisper' (local), 'openai', 'gemini', or 'mistral' instead. "
                "Anthropic can be used for speaker detection and summarization after transcription."
            )
        # Grok does not support transcription
        if self.transcription_provider == "grok":
            raise ValueError(
                "Grok provider does not support transcription. "
                "Use 'whisper' (local) or 'openai' instead. "
                "See provider capability matrix in documentation."
            )
        # Ollama does not support transcription
        if self.transcription_provider == "ollama":
            raise ValueError(
                "Ollama provider does not support transcription. "
                "Use 'whisper' (local) or 'openai' instead. "
                "See provider capability matrix in documentation."
            )
        return self

    @model_validator(mode="after")
    def _validate_grok_provider_requirements(self) -> "Config":
        """Validate that Grok API key is provided when Grok providers are selected."""
        grok_providers_used = []
        if self.speaker_detector_provider == "grok":
            grok_providers_used.append("speaker_detection")
        if self.summary_provider == "grok":
            grok_providers_used.append("summarization")
        if self.quote_extraction_provider == "grok":
            grok_providers_used.append("quote_extraction")
        if self.entailment_provider == "grok":
            grok_providers_used.append("entailment")

        if grok_providers_used and not self.grok_api_key:
            providers_str = ", ".join(grok_providers_used)
            raise ValueError(
                f"Grok API key required for Grok providers: {providers_str}. "
                "Set GROK_API_KEY environment variable or grok_api_key in config."
            )

        return self

    @model_validator(mode="after")
    def _validate_mistral_provider_requirements(self) -> "Config":
        """Validate that Mistral API key is provided when Mistral providers are selected."""
        mistral_providers_used = []
        if self.transcription_provider == "mistral":
            mistral_providers_used.append("transcription")
        if self.speaker_detector_provider == "mistral":
            mistral_providers_used.append("speaker_detection")
        if self.summary_provider == "mistral":
            mistral_providers_used.append("summarization")
        if self.quote_extraction_provider == "mistral":
            mistral_providers_used.append("quote_extraction")
        if self.entailment_provider == "mistral":
            mistral_providers_used.append("entailment")

        if mistral_providers_used and not self.mistral_api_key:
            providers_str = ", ".join(mistral_providers_used)
            raise ValueError(
                f"Mistral API key required for Mistral providers: {providers_str}. "
                "Set MISTRAL_API_KEY environment variable or mistral_api_key in config."
            )

        return self

    @field_validator("summary_model", mode="before")
    @classmethod
    def _coerce_summary_model(cls, value: Any) -> Optional[str]:
        """Coerce summary model to string or None."""
        if value is None or value == "":
            return None
        return str(value).strip() or None

    @field_validator("summary_mode_id", mode="before")
    @classmethod
    def _coerce_summary_mode_id(cls, value: Any) -> Optional[str]:
        """Coerce summary_mode_id to string or None."""
        if value is None or value == "":
            return None
        return str(value).strip() or None

    @field_validator("summary_mode_precedence", mode="before")
    @classmethod
    def _coerce_summary_mode_precedence(cls, value: Any) -> str:
        """Coerce summary_mode_precedence to 'mode' or 'config'."""
        if value is None or value == "":
            return "mode"
        lowered = str(value).strip().lower()
        if lowered in ("mode", "config"):
            return lowered
        raise ValueError("summary_mode_precedence must be 'mode' or 'config'")

    @field_validator("summary_device", mode="before")
    @classmethod
    def _coerce_summary_device(cls, value: Any) -> Optional[str]:
        """Coerce summary device to string or None."""
        if value is None or value == "":
            return None
        value_str = str(value).strip().lower()
        if value_str == "auto":
            return None
        if value_str not in ("cuda", "mps", "cpu"):
            raise ValueError("summary_device must be 'cuda', 'mps', 'cpu', or 'auto'")
        return value_str

    @field_validator("whisper_device", mode="before")
    @classmethod
    def _coerce_whisper_device(cls, value: Any) -> Optional[str]:
        """Coerce whisper device to string or None."""
        if value is None or value == "":
            return None
        value_str = str(value).strip().lower()
        if value_str == "auto":
            return None
        if value_str not in ("cuda", "mps", "cpu"):
            raise ValueError("whisper_device must be 'cuda', 'mps', 'cpu', or 'auto'")
        return value_str

    @field_validator("transcription_parallelism", mode="before")
    @classmethod
    def _ensure_transcription_parallelism(cls, value: Any) -> int:
        """Ensure transcription parallelism is a positive integer."""
        if value is None or value == "":
            return 1
        try:
            parallelism = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("transcription_parallelism must be an integer") from exc
        if parallelism < 1:
            raise ValueError("transcription_parallelism must be at least 1")
        return parallelism

    @field_validator("transcription_queue_size", mode="before")
    @classmethod
    def _ensure_transcription_queue_size(cls, value: Any) -> int:
        """Ensure transcription queue size is a positive integer."""
        if value is None or value == "":
            return 50
        try:
            queue_size = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("transcription_queue_size must be an integer") from exc
        if queue_size < 1:
            raise ValueError("transcription_queue_size must be at least 1")
        return queue_size

    @field_validator("processing_parallelism", mode="before")
    @classmethod
    def _ensure_processing_parallelism(cls, value: Any) -> int:
        """Ensure processing parallelism is a positive integer."""
        if value is None or value == "":
            return 2
        try:
            parallelism = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("processing_parallelism must be an integer") from exc
        if parallelism < 1:
            raise ValueError("processing_parallelism must be at least 1")
        return parallelism

    @field_validator("summary_batch_size", mode="before")
    @classmethod
    def _ensure_batch_size(cls, value: Any) -> int:
        """Ensure batch size is a positive integer."""
        if value is None or value == "":
            return DEFAULT_SUMMARY_BATCH_SIZE
        try:
            batch_size = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("summary_batch_size must be an integer") from exc
        if batch_size < 1:
            raise ValueError("summary_batch_size must be at least 1")
        return batch_size

    @field_validator("summary_max_workers_cpu", mode="before")
    @classmethod
    def _validate_summary_max_workers_cpu(cls, v: Any) -> Optional[int]:
        """Set default max workers for CPU based on environment."""
        if v is None or v == "":
            # Return None - will be handled in workflow code based on environment
            return None
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError) as exc:
                raise ValueError("summary_max_workers_cpu must be an integer") from exc
        if v < 1:
            raise ValueError("summary_max_workers_cpu must be at least 1")
        return int(v)

    @field_validator("summary_max_workers_gpu", mode="before")
    @classmethod
    def _validate_summary_max_workers_gpu(cls, v: Any) -> Optional[int]:
        """Set default max workers for GPU based on environment."""
        if v is None or v == "":
            # Return None - will be handled in workflow code based on environment
            return None
        if not isinstance(v, int):
            try:
                v = int(v)
            except (ValueError, TypeError) as exc:
                raise ValueError("summary_max_workers_gpu must be an integer") from exc
        if v < 1:
            raise ValueError("summary_max_workers_gpu must be at least 1")
        return int(v)

    @field_validator("summary_chunk_parallelism", mode="before")
    @classmethod
    def _ensure_chunk_parallelism(cls, value: Any) -> int:
        """Ensure chunk parallelism is a positive integer."""
        if value is None or value == "":
            return 1
        try:
            chunk_parallelism = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("summary_chunk_parallelism must be an integer") from exc
        if chunk_parallelism < 1:
            raise ValueError("summary_chunk_parallelism must be at least 1")
        return chunk_parallelism

    @field_validator("summary_chunk_size", mode="before")
    @classmethod
    def _coerce_chunk_size(cls, value: Any) -> Optional[int]:
        """Coerce chunk size to positive integer or None."""
        if value is None or value == "":
            return None
        try:
            chunk_size = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("summary_chunk_size must be an integer") from exc
        if chunk_size < 1:
            raise ValueError("summary_chunk_size must be at least 1")
        return chunk_size

    @field_validator("summary_cache_dir", mode="before")
    @classmethod
    def _coerce_cache_dir(cls, value: Any) -> Optional[str]:
        """Coerce cache directory to string or None."""
        if value is None or value == "":
            return None
        value_str = str(value).strip()
        return value_str or None

    @field_validator("summary_cache_dir", mode="after")
    @classmethod
    def _validate_summary_cache_dir_traversal(cls, v: Optional[str]) -> Optional[str]:
        return cls._validate_path_no_traversal(v, "summary_cache_dir")

    @field_validator("transcript_cache_dir", mode="after")
    @classmethod
    def _validate_transcript_cache_dir_traversal(cls, v: Optional[str]) -> Optional[str]:
        return cls._validate_path_no_traversal(v, "transcript_cache_dir")

    @field_validator("preprocessing_cache_dir", mode="after")
    @classmethod
    def _validate_preprocessing_cache_dir_traversal(cls, v: Optional[str]) -> Optional[str]:
        return cls._validate_path_no_traversal(v, "preprocessing_cache_dir")

    @field_validator("preprocessing_mp3_bitrate_kbps", mode="before")
    @classmethod
    def _coerce_preprocessing_mp3_bitrate_kbps(cls, v: Any) -> Optional[int]:
        """Allow null / unset; coerce ints from YAML strings."""
        if v is None or v == "":
            return None
        if isinstance(v, str) and not str(v).strip():
            return None
        return int(v)

    @field_validator("preprocessing_mp3_bitrate_kbps", mode="after")
    @classmethod
    def _validate_preprocessing_mp3_bitrate_kbps(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if v < 24 or v > 128:
            raise ValueError(
                "preprocessing_mp3_bitrate_kbps must be between 24 and 128 inclusive, "
                "or null for automatic bitrate (GitHub #561)."
            )
        return v

    @model_validator(mode="after")
    def _validate_cross_field_settings(self) -> "Config":
        """Cross-field validation for configuration settings.

        Also sets default values for summary_max_workers based on environment.

        Validates logical relationships and dependencies between configuration fields:

        Summary Settings:
        - summary_word_overlap must be less than summary_word_chunk_size
        - Warns if word-based chunking parameters are outside recommended ranges
        - Ensures generate_summaries requires generate_metadata

        Output Control:
        - Prevents contradictory flag combinations (clean_output with skip_existing/reuse_media)

        Transcription:
        - Ensures transcribe_missing has a valid whisper_model

        Returns:
            Self for method chaining

        Raises:
            ValueError: If any validation check fails
        """
        # Episode selection (GitHub #521)
        if (
            self.episode_since is not None
            and self.episode_until is not None
            and self.episode_since > self.episode_until
        ):
            raise ValueError(
                f"episode_since ({self.episode_since}) must be on or before "
                f"episode_until ({self.episode_until})"
            )

        # === Summary Settings Validation ===

        # 1. Validate ML params structure (always present now, with defaults from Config)
        # Validate map_params structure
        required_map_keys = {"max_new_tokens", "min_new_tokens"}
        if not all(key in self.summary_map_params for key in required_map_keys):
            raise ValueError(
                f"summary_map_params must include: {required_map_keys}. "
                f"Got: {list(self.summary_map_params.keys())}"
            )

        # Validate reduce_params structure
        required_reduce_keys = {"max_new_tokens", "min_new_tokens"}
        if not all(key in self.summary_reduce_params for key in required_reduce_keys):
            raise ValueError(
                f"summary_reduce_params must include: {required_reduce_keys}. "
                f"Got: {list(self.summary_reduce_params.keys())}"
            )

        # Validate tokenize structure
        required_tokenize_keys = {"map_max_input_tokens", "reduce_max_input_tokens"}
        if not all(key in self.summary_tokenize for key in required_tokenize_keys):
            raise ValueError(
                f"summary_tokenize must include: {required_tokenize_keys}. "
                f"Got: {list(self.summary_tokenize.keys())}"
            )

        # 2. summary_word_chunk_size should be in recommended range
        #    Warn but don't error to allow experimentation
        if self.summary_word_chunk_size is not None:
            min_val = config_constants.RECOMMENDED_WORD_CHUNK_SIZE_MIN
            max_val = config_constants.RECOMMENDED_WORD_CHUNK_SIZE_MAX
            if self.summary_word_chunk_size < min_val or self.summary_word_chunk_size > max_val:
                warnings.warn(
                    f"summary_word_chunk_size ({self.summary_word_chunk_size}) is outside "
                    f"recommended range ({min_val}-{max_val}). This may affect summary quality.",
                    UserWarning,
                    stacklevel=2,
                )

        # 4. summary_word_overlap should be in recommended range
        #    Warn but don't error to allow experimentation
        if self.summary_word_overlap is not None:
            min_val = config_constants.RECOMMENDED_WORD_OVERLAP_MIN
            max_val = config_constants.RECOMMENDED_WORD_OVERLAP_MAX
            if self.summary_word_overlap < min_val or self.summary_word_overlap > max_val:
                warnings.warn(
                    f"summary_word_overlap ({self.summary_word_overlap}) is outside "
                    f"recommended range ({min_val}-{max_val}). This may affect summary quality.",
                    UserWarning,
                    stacklevel=2,
                )

        # 5. summary_word_overlap must be less than summary_word_chunk_size
        if self.summary_word_chunk_size is not None and self.summary_word_overlap is not None:
            if self.summary_word_overlap >= self.summary_word_chunk_size:
                raise ValueError(
                    f"summary_word_overlap ({self.summary_word_overlap}) must be less than "
                    f"summary_word_chunk_size ({self.summary_word_chunk_size})"
                )

        # 6. generate_summaries requires generate_metadata
        #    Summaries are stored in metadata files, so metadata generation must be enabled
        if self.generate_summaries and not self.generate_metadata:
            raise ValueError(
                "generate_summaries=True requires generate_metadata=True "
                "(summaries are stored in metadata files)"
            )

        # 6b. generate_gi requires generate_metadata
        #     GIL artifact is per-episode alongside metadata
        if self.generate_gi and not self.generate_metadata:
            raise ValueError(
                "generate_gi=True requires generate_metadata=True "
                "(GIL artifact is written alongside episode metadata)"
            )

        # 6c. generate_kg requires generate_metadata (artifact co-located with metadata)
        if self.generate_kg and not self.generate_metadata:
            raise ValueError(
                "generate_kg=True requires generate_metadata=True "
                "(KG artifact is written alongside episode metadata)"
            )

        # 6d. GIL windowed QA: overlap must be smaller than window when windowing is on
        if self.gi_qa_window_chars > 0:
            if self.gi_qa_window_overlap_chars >= self.gi_qa_window_chars:
                raise ValueError(
                    "gi_qa_window_overlap_chars must be strictly less than gi_qa_window_chars "
                    "when gi_qa_window_chars > 0"
                )

        # === Output Control Validation ===

        # 7. clean_output and skip_existing are mutually exclusive
        #    clean_output removes all files, making skip_existing meaningless
        if self.clean_output and self.skip_existing:
            raise ValueError(
                "clean_output and skip_existing are mutually exclusive "
                "(clean_output removes all existing files, making skip_existing meaningless)"
            )

        # 7b. clean_output and append are mutually exclusive (GitHub #444)
        if self.clean_output and self.append:
            raise ValueError(
                "clean_output and append are mutually exclusive "
                "(append requires a durable workspace; clean_output deletes it)"
            )

        # 8. clean_output and reuse_media are mutually exclusive
        #    clean_output removes media files that reuse_media would reuse
        if self.clean_output and self.reuse_media:
            raise ValueError(
                "clean_output and reuse_media are mutually exclusive "
                "(clean_output removes media files that would be reused)"
            )

        # === Transcription Validation ===

        # 9. transcribe_missing requires a valid whisper_model
        #    Can't transcribe without specifying which model to use
        if self.transcribe_missing and not self.whisper_model:
            raise ValueError(
                "transcribe_missing=True requires a valid whisper_model "
                "(e.g., 'base', 'small', 'medium')"
            )

        if self.vector_chunk_overlap_tokens >= self.vector_chunk_size_tokens:
            raise ValueError(
                "vector_chunk_overlap_tokens must be strictly less than vector_chunk_size_tokens"
            )

        return self


def load_config_file(
    path: str,
) -> Dict[str, Any]:  # noqa: C901 - file parsing handles multiple formats
    """Load configuration from a JSON or YAML file.

    This function reads a configuration file and returns a dictionary of configuration values.
    The file format is auto-detected from the file extension (`.json`, `.yaml`, or `.yml`).

    The returned dictionary can be unpacked into the `Config` constructor to create a
    configuration object.

    Args:
        path: Path to configuration file (JSON or YAML). Supports tilde expansion for
              home directory (e.g., "~/config.yaml").

    Returns:
        Dict[str, Any]: Dictionary containing configuration values from the file.
            Keys correspond to `Config` field names (using aliases where applicable).

    Raises:
        ValueError: If any of the following occur:

            - Config path is empty
            - Config file does not exist
            - File format is invalid (not JSON or YAML)
            - JSON parsing fails
            - YAML parsing fails

        OSError: If file cannot be read due to permissions or I/O errors

    Example:
        >>> from podcast_scraper import Config, load_config_file, run_pipeline
        >>>
        >>> # Load from YAML file
        >>> config_dict = load_config_file("config.yaml")
        >>> cfg = Config(**config_dict)
        >>> count, summary = run_pipeline(cfg)

    Example with JSON:
        >>> config_dict = load_config_file("config.json")
        >>> cfg = Config(**config_dict)

    Example with direct usage:
        >>> from podcast_scraper import load_config_file, service
        >>>
        >>> # Service API provides load_config_file convenience
        >>> result = service.run_from_config_file("config.yaml")

    Supported Formats:
        **JSON** (`.json`):

            {
              "rss": "https://example.com/feed.xml",
              "output_dir": "./transcripts",
              "max_episodes": 50
            }

        **YAML** (`.yaml`, `.yml`):

            rss: https://example.com/feed.xml
            output_dir: ./transcripts
            max_episodes: 50

    Note:
        - Field aliases are supported (e.g., both "rss" and "rss_url" work)
        - See `Config` documentation for all available configuration options
        - Configuration files should not contain sensitive data (API keys, passwords)

    See Also:
        - `Config`: Configuration model and field documentation
        - `service.run_from_config_file()`: Direct service API from config file
        - Configuration examples: `config/examples/config.example.json`,
          `config/examples/config.example.yaml`
    """
    if not path:
        raise ValueError("Config path cannot be empty")

    cfg_path = Path(path).expanduser()
    try:
        resolved = cfg_path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Invalid config path: {path} ({exc})") from exc

    if not resolved.exists():
        raise ValueError(f"Config file not found: {resolved}")

    suffix = resolved.suffix.lower()
    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read config file {resolved}: {exc}") from exc

    if suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON config file {resolved}: {exc}") from exc
    elif suffix in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"Invalid YAML config file {resolved}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported config file type: {resolved.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping/object at the top level")

    return data
