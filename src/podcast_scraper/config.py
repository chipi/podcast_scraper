from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import config_constants

if TYPE_CHECKING:
    from podcast_scraper.evaluation.config import GenerationParams, TokenizeConfig
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
PROD_DEFAULT_OPENAI_TRANSCRIPTION_MODEL = config_constants.PROD_DEFAULT_OPENAI_TRANSCRIPTION_MODEL
PROD_DEFAULT_OPENAI_SPEAKER_MODEL = config_constants.PROD_DEFAULT_OPENAI_SPEAKER_MODEL
PROD_DEFAULT_OPENAI_SUMMARY_MODEL = config_constants.PROD_DEFAULT_OPENAI_SUMMARY_MODEL

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
DEFAULT_MAP_MAX_INPUT_TOKENS = 1024  # BART model limit
DEFAULT_REDUCE_MAX_INPUT_TOKENS = 4096  # LED model limit
DEFAULT_TRUNCATION = True

# Default distill parameters (for final compression pass)
DEFAULT_DISTILL_MAX_TOKENS = 200
DEFAULT_DISTILL_MIN_TOKENS = 120
DEFAULT_DISTILL_NUM_BEAMS = 4
DEFAULT_DISTILL_NO_REPEAT_NGRAM_SIZE = 6
DEFAULT_DISTILL_LENGTH_PENALTY = 0.75
DEFAULT_DISTILL_REPETITION_PENALTY = 1.3  # Same as map/reduce for consistency

# Default token overlap for chunking
DEFAULT_TOKEN_OVERLAP = 200


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
        user_agent: HTTP User-Agent header for requests.
        timeout: Request timeout in seconds (minimum: 1).
        delay_ms: Delay between requests in milliseconds.
        prefer_types: Preferred transcript types or extensions (e.g., ["text/vtt", ".srt"]).
        transcribe_missing: Enable Whisper transcription for episodes without transcripts
            (default: True). Set to False to only download existing transcripts.
        whisper_model: Whisper model name (e.g., "base", "small", "medium").
        whisper_device: Device for Whisper execution ("cpu", "cuda", "mps", or None for auto).
        screenplay: Format transcripts as screenplay with speaker labels.
        screenplay_gap_s: Minimum gap in seconds between speaker segments.
        screenplay_num_speakers: Number of speakers for Whisper diarization.
        screenplay_speaker_names: Manual speaker names list (overrides auto-detection).
        run_id: Optional run identifier. Use "auto" for timestamp-based ID.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional log file path for file output.
        workers: Number of parallel download workers.
        skip_existing: Skip episodes with existing output files.
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
    output_dir: Optional[str] = Field(
        default=None,
        alias="output_dir",
        description="Output directory path. Auto-generated from RSS URL if not provided. "
        "Can be set via OUTPUT_DIR environment variable.",
    )
    max_episodes: Optional[int] = Field(default=None, alias="max_episodes")
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
    prefer_types: List[str] = Field(default_factory=list, alias="prefer_type")
    transcribe_missing: bool = Field(default=True, alias="transcribe_missing")
    whisper_model: str = Field(default="base.en", alias="whisper_model")
    whisper_device: Optional[str] = Field(default=None, alias="whisper_device")
    screenplay: bool = Field(default=False, alias="screenplay")
    screenplay_gap_s: float = Field(default=DEFAULT_SCREENPLAY_GAP_SECONDS, alias="screenplay_gap")
    screenplay_num_speakers: int = Field(default=DEFAULT_NUM_SPEAKERS, alias="num_speakers")
    screenplay_speaker_names: List[str] = Field(default_factory=list, alias="speaker_names")
    run_id: Optional[str] = Field(default=None, alias="run_id")
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
    clean_output: bool = Field(default=False, alias="clean_output")
    reuse_media: bool = Field(
        default=False,
        alias="reuse_media",
        description="Reuse existing media files instead of re-downloading (for faster testing)",
    )
    dry_run: bool = Field(default=False, alias="dry_run")
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
    openai_temperature: float = Field(
        default=0.3,
        alias="openai_temperature",
        description="Temperature for OpenAI generation (0.0-2.0, lower = more deterministic)",
    )
    openai_max_tokens: Optional[int] = Field(
        default=None,
        alias="openai_max_tokens",
        description="Max tokens for OpenAI generation (None = model default)",
    )
    # Prompt configuration (RFC-017)
    openai_summary_system_prompt: Optional[str] = Field(
        default=None,
        alias="openai_summary_system_prompt",
        description="System prompt name for summarization (e.g. 'openai/summarization/system_v1'). "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    openai_summary_user_prompt: str = Field(
        default="openai/summarization/long_v1",
        alias="openai_summary_user_prompt",
        description="User prompt name for summarization. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    summary_prompt_params: Dict[str, Any] = Field(
        default_factory=dict,
        alias="summary_prompt_params",
        description="Template parameters for summary prompts (passed to Jinja2 templates).",
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
    gemini_max_tokens: Optional[int] = Field(
        default=None,
        alias="gemini_max_tokens",
        description="Max tokens for Gemini generation (None = model default)",
    )
    # Gemini Prompt Configuration (following OpenAI pattern)
    gemini_summary_system_prompt: Optional[str] = Field(
        default=None,
        alias="gemini_summary_system_prompt",
        description=(
            "Gemini system prompt for summarization (default: gemini/summarization/system_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    gemini_summary_user_prompt: str = Field(
        default="gemini/summarization/long_v1",
        alias="gemini_summary_user_prompt",
        description="Gemini user prompt for summarization. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
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
    anthropic_max_tokens: Optional[int] = Field(
        default=None,
        alias="anthropic_max_tokens",
        description="Max tokens for Anthropic generation (None = model default)",
    )
    # Anthropic Prompt Configuration (following OpenAI/Gemini pattern)
    anthropic_summary_system_prompt: Optional[str] = Field(
        default=None,
        alias="anthropic_summary_system_prompt",
        description=(
            "Anthropic system prompt for summarization "
            "(default: anthropic/summarization/system_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    anthropic_summary_user_prompt: str = Field(
        default="anthropic/summarization/long_v1",
        alias="anthropic_summary_user_prompt",
        description="Anthropic user prompt for summarization. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
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
    ollama_summary_system_prompt: Optional[str] = Field(
        default=None,
        alias="ollama_summary_system_prompt",
        description=(
            "Ollama system prompt for summarization (default: ollama/summarization/system_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    ollama_summary_user_prompt: str = Field(
        default="ollama/summarization/long_v1",
        alias="ollama_summary_user_prompt",
        description="Ollama user prompt for summarization. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
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
    deepseek_max_tokens: Optional[int] = Field(
        default=None,
        alias="deepseek_max_tokens",
        description="Max tokens for DeepSeek generation (None = model default)",
    )
    # DeepSeek Prompt Configuration (following OpenAI pattern)
    deepseek_summary_system_prompt: Optional[str] = Field(
        default=None,
        alias="deepseek_summary_system_prompt",
        description=(
            "DeepSeek system prompt for summarization (default: deepseek/summarization/system_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    deepseek_summary_user_prompt: str = Field(
        default="deepseek/summarization/long_v1",
        alias="deepseek_summary_user_prompt",
        description="DeepSeek user prompt for summarization. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
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
    grok_max_tokens: Optional[int] = Field(
        default=None,
        alias="grok_max_tokens",
        description="Max tokens for Grok generation (None = model default)",
    )
    grok_summary_system_prompt: Optional[str] = Field(
        default=None,
        alias="grok_summary_system_prompt",
        description=(
            "Grok system prompt for summarization (default: grok/summarization/system_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    grok_summary_user_prompt: str = Field(
        default="grok/summarization/long_v1",
        alias="grok_summary_user_prompt",
        description="Grok user prompt for summarization. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
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
    mistral_summary_system_prompt: Optional[str] = Field(
        default=None,
        alias="mistral_summary_system_prompt",
        description=(
            "Mistral system prompt for summarization (default: mistral/summarization/system_v1). "
            "Uses prompt_store (RFC-017) for versioned prompts."
        ),
    )
    mistral_summary_user_prompt: str = Field(
        default="mistral/summarization/long_v1",
        alias="mistral_summary_user_prompt",
        description="Mistral user prompt for summarization. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    generate_metadata: bool = Field(default=False, alias="generate_metadata")
    metadata_format: Literal["json", "yaml"] = Field(default="json", alias="metadata_format")
    metadata_subdirectory: Optional[str] = Field(default=None, alias="metadata_subdirectory")
    generate_summaries: bool = Field(default=False, alias="generate_summaries")
    metrics_output: Optional[str] = Field(
        default=None,
        alias="metrics_output",
        description="Path to save pipeline metrics JSON file. "
        "If not specified, defaults to {effective_output_dir}/metrics.json "
        "(same level as transcripts/ and metadata/ subdirectories). "
        "Set to empty string to disable metrics export.",
    )
    summary_provider: Literal[
        "transformers",
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
        default_factory=lambda: {
            "map_max_input_tokens": DEFAULT_MAP_MAX_INPUT_TOKENS,
            "reduce_max_input_tokens": DEFAULT_REDUCE_MAX_INPUT_TOKENS,
            "truncation": DEFAULT_TRUNCATION,
        },
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

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _handle_deprecated_fields(cls, data: Any) -> Any:
        """Handle deprecated field names for backward compatibility."""
        if isinstance(data, dict):
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

    @field_validator("rss_url", mode="before")
    @classmethod
    def _strip_rss(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

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

        # OUTPUT_DIR: Only set from env if not in config
        if "output_dir" not in data or data.get("output_dir") is None:
            env_output_dir = os.getenv("OUTPUT_DIR")
            if env_output_dir:
                env_value = str(env_output_dir).strip()
                if env_value:
                    data["output_dir"] = env_value

        # LOG_FILE: Only set from env if not in config
        if "log_file" not in data or data.get("log_file") is None:
            env_log_file = os.getenv("LOG_FILE")
            if env_log_file:
                env_value = str(env_log_file).strip()
                if env_value:
                    data["log_file"] = env_value

        # SUMMARY_CACHE_DIR / CACHE_DIR: Only set from env if not in config
        # Also check for local cache in project root
        if "summary_cache_dir" not in data or data.get("summary_cache_dir") is None:
            # Check SUMMARY_CACHE_DIR first (explicit override)
            summary_cache = os.getenv("SUMMARY_CACHE_DIR")
            if summary_cache:
                env_value = str(summary_cache).strip()
                if env_value:
                    data["summary_cache_dir"] = env_value
            else:
                # If CACHE_DIR is set, derive summary cache from it
                cache_dir = os.getenv("CACHE_DIR")
                if cache_dir:
                    # Derive Transformers cache path from CACHE_DIR
                    from pathlib import Path

                    derived_cache = str(Path(cache_dir) / "huggingface" / "hub")
                    data["summary_cache_dir"] = derived_cache
                else:
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

        # WORKERS: Only set from env if not in config
        if "workers" not in data or data.get("workers") is None:
            env_workers = os.getenv("WORKERS")
            if env_workers:
                try:
                    workers_value = int(env_workers)
                    if workers_value > 0:
                        data["workers"] = workers_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

        # TRANSCRIPTION_PARALLELISM: Only set from env if not in config
        if "transcription_parallelism" not in data or data.get("transcription_parallelism") is None:
            env_parallelism = os.getenv("TRANSCRIPTION_PARALLELISM")
            if env_parallelism:
                try:
                    parallelism_value = int(env_parallelism)
                    if parallelism_value > 0:
                        data["transcription_parallelism"] = parallelism_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

        # PROCESSING_PARALLELISM: Only set from env if not in config
        if "processing_parallelism" not in data or data.get("processing_parallelism") is None:
            env_parallelism = os.getenv("PROCESSING_PARALLELISM")
            if env_parallelism:
                try:
                    parallelism_value = int(env_parallelism)
                    if parallelism_value > 0:
                        data["processing_parallelism"] = parallelism_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

        # SUMMARY_BATCH_SIZE: Only set from env if not in config
        if "summary_batch_size" not in data or data.get("summary_batch_size") is None:
            env_batch_size = os.getenv("SUMMARY_BATCH_SIZE")
            if env_batch_size:
                try:
                    batch_size_value = int(env_batch_size)
                    if batch_size_value > 0:
                        data["summary_batch_size"] = batch_size_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

        # SUMMARY_CHUNK_PARALLELISM: Only set from env if not in config
        if "summary_chunk_parallelism" not in data or data.get("summary_chunk_parallelism") is None:
            env_parallelism = os.getenv("SUMMARY_CHUNK_PARALLELISM")
            if env_parallelism:
                try:
                    parallelism_value = int(env_parallelism)
                    if parallelism_value > 0:
                        data["summary_chunk_parallelism"] = parallelism_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

        # TIMEOUT: Only set from env if not in config
        if "timeout" not in data or data.get("timeout") is None:
            env_timeout = os.getenv("TIMEOUT")
            if env_timeout:
                try:
                    timeout_value = int(env_timeout)
                    if timeout_value > 0:
                        data["timeout"] = timeout_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

        # SUMMARY_DEVICE: Only set from env if not in config
        if "summary_device" not in data or data.get("summary_device") is None:
            env_device = os.getenv("SUMMARY_DEVICE")
            if env_device:
                env_value = str(env_device).strip().lower()
                if env_value in ("cpu", "cuda", "mps"):
                    data["summary_device"] = env_value

        # WHISPER_DEVICE: Only set from env if not in config
        if "whisper_device" not in data or data.get("whisper_device") is None:
            env_device = os.getenv("WHISPER_DEVICE")
            if env_device:
                env_value = str(env_device).strip().lower()
                if env_value in ("cpu", "cuda", "mps"):
                    data["whisper_device"] = env_value

        # MPS_EXCLUSIVE: Only set from env if not in config
        if "mps_exclusive" not in data:
            env_mps_exclusive = os.getenv("MPS_EXCLUSIVE")
            if env_mps_exclusive:
                env_value = str(env_mps_exclusive).strip().lower()
                # Support various boolean representations
                if env_value in ("1", "true", "yes", "on"):
                    data["mps_exclusive"] = True
                elif env_value in ("0", "false", "no", "off"):
                    data["mps_exclusive"] = False

        # OPENAI_API_KEY: Only set from env if not in config
        if "openai_api_key" not in data or data.get("openai_api_key") is None:
            env_key = os.getenv("OPENAI_API_KEY")
            if env_key:
                env_value = str(env_key).strip()
                if env_value:
                    data["openai_api_key"] = env_value

        # OPENAI_API_BASE: Only set from env if not in config
        if "openai_api_base" not in data or data.get("openai_api_base") is None:
            env_base = os.getenv("OPENAI_API_BASE")
            if env_base:
                env_value = str(env_base).strip()
                if env_value:
                    data["openai_api_base"] = env_value

        # GEMINI_API_KEY: Only set from env if not in config
        if "gemini_api_key" not in data or data.get("gemini_api_key") is None:
            env_key = os.getenv("GEMINI_API_KEY")
            if env_key:
                env_value = str(env_key).strip()
                if env_value:
                    data["gemini_api_key"] = env_value

        # GEMINI_API_BASE: Only set from env if not in config
        if "gemini_api_base" not in data or data.get("gemini_api_base") is None:
            env_base = os.getenv("GEMINI_API_BASE")
            if env_base:
                env_value = str(env_base).strip()
                if env_value:
                    data["gemini_api_base"] = env_value

        # ANTHROPIC_API_KEY: Only set from env if not in config
        if "anthropic_api_key" not in data or data.get("anthropic_api_key") is None:
            env_key = os.getenv("ANTHROPIC_API_KEY")
            if env_key:
                env_value = str(env_key).strip()
                if env_value:
                    data["anthropic_api_key"] = env_value

        # ANTHROPIC_API_BASE: Only set from env if not in config
        if "anthropic_api_base" not in data or data.get("anthropic_api_base") is None:
            env_base = os.getenv("ANTHROPIC_API_BASE")
            if env_base:
                env_value = str(env_base).strip()
                if env_value:
                    data["anthropic_api_base"] = env_value

        # MISTRAL_API_KEY: Only set from env if not in config
        if "mistral_api_key" not in data or data.get("mistral_api_key") is None:
            env_key = os.getenv("MISTRAL_API_KEY")
            if env_key:
                env_value = str(env_key).strip()
                if env_value:
                    data["mistral_api_key"] = env_value

        # MISTRAL_API_BASE: Only set from env if not in config
        if "mistral_api_base" not in data or data.get("mistral_api_base") is None:
            env_base = os.getenv("MISTRAL_API_BASE")
            if env_base:
                env_value = str(env_base).strip()
                if env_value:
                    data["mistral_api_base"] = env_value

        # DEEPSEEK_API_KEY: Only set from env if not in config
        if "deepseek_api_key" not in data or data.get("deepseek_api_key") is None:
            env_key = os.getenv("DEEPSEEK_API_KEY")
            if env_key:
                env_value = str(env_key).strip()
                if env_value:
                    data["deepseek_api_key"] = env_value

        # DEEPSEEK_API_BASE: Only set from env if not in config
        if "deepseek_api_base" not in data or data.get("deepseek_api_base") is None:
            env_base = os.getenv("DEEPSEEK_API_BASE")
            if env_base:
                env_value = str(env_base).strip()
                if env_value:
                    data["deepseek_api_base"] = env_value

        # GROK_API_KEY: Only set from env if not in config
        if "grok_api_key" not in data or data.get("grok_api_key") is None:
            env_key = os.getenv("GROK_API_KEY")
            if env_key:
                env_value = str(env_key).strip()
                if env_value:
                    data["grok_api_key"] = env_value

        # GROK_API_BASE: Only set from env if not in config
        if "grok_api_base" not in data or data.get("grok_api_base") is None:
            env_base = os.getenv("GROK_API_BASE")
            if env_base:
                env_value = str(env_base).strip()
                if env_value:
                    data["grok_api_base"] = env_value

        # OPENAI_TEMPERATURE: Only set from env if not in config
        if "openai_temperature" not in data or data.get("openai_temperature") is None:
            env_temp = os.getenv("OPENAI_TEMPERATURE")
            if env_temp:
                try:
                    temp_value = float(env_temp)
                    if 0.0 <= temp_value <= 2.0:
                        data["openai_temperature"] = temp_value
                except (ValueError, TypeError):
                    pass  # Invalid value, skip

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
    def _validate_summary_provider(
        cls, value: Any
    ) -> Literal["transformers", "openai", "gemini", "grok", "deepseek", "anthropic", "ollama"]:
        """Validate summary provider."""
        if value is None or value == "":
            return "transformers"
        value_str = str(value).strip().lower()

        if value_str not in (
            "transformers",
            "openai",
            "gemini",
            "grok",
            "mistral",
            "deepseek",
            "anthropic",
            "ollama",
        ):
            raise ValueError(
                "summary_provider must be 'transformers', 'openai', 'gemini', "
                "'grok', 'mistral', 'deepseek', 'anthropic', or 'ollama'"
            )
        return value_str  # type: ignore[return-value]

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

        # === Output Control Validation ===

        # 7. clean_output and skip_existing are mutually exclusive
        #    clean_output removes all files, making skip_existing meaningless
        if self.clean_output and self.skip_existing:
            raise ValueError(
                "clean_output and skip_existing are mutually exclusive "
                "(clean_output removes all existing files, making skip_existing meaningless)"
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
        - Configuration examples: `examples/config.example.json`, `examples/config.example.yaml`
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
