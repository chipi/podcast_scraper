from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Load .env file if it exists (RFC-013: OpenAI API key management)
# Check for .env in project root (where config.py is located)
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=False)
else:
    # Also check current working directory (for flexibility)
    load_dotenv(override=False)

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_NUM_SPEAKERS = 2
DEFAULT_SCREENPLAY_GAP_SECONDS = 1.25
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0 Safari/537.36"
)
DEFAULT_WORKERS = max(1, min(8, os.cpu_count() or 4))
DEFAULT_LANGUAGE = "en"
DEFAULT_NER_MODEL = "en_core_web_sm"
DEFAULT_MAX_DETECTED_NAMES = 4
MIN_NUM_SPEAKERS = 1
MIN_TIMEOUT_SECONDS = 1
VALID_WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "large.en",
)

VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
MAX_RUN_ID_LENGTH = 100
MAX_METADATA_SUBDIRECTORY_LENGTH = 255
DEFAULT_SUMMARY_MAX_LENGTH = 160  # Per SUMMARY_REVIEW.md: chunk summaries should be ~160 tokens
DEFAULT_SUMMARY_MIN_LENGTH = (
    60  # Per SUMMARY_REVIEW.md: chunk summaries should be at least 60 tokens
)
DEFAULT_SUMMARY_BATCH_SIZE = 1
DEFAULT_SUMMARY_CHUNK_SIZE = (
    2048  # Default token chunk size (BART models support up to 1024, but larger chunks work safely)
)
DEFAULT_SUMMARY_WORD_CHUNK_SIZE = (
    900  # Per SUMMARY_REVIEW.md: 800-1200 words recommended for encoder-decoder models
)
DEFAULT_SUMMARY_WORD_OVERLAP = 150  # Per SUMMARY_REVIEW.md: 100-200 words recommended


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
        transcribe_missing: Enable Whisper transcription for episodes without transcripts.
        whisper_model: Whisper model name (e.g., "base", "small", "medium").
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
        language: Language code for transcription (e.g., "en", "fr", "de").
        ner_model: spaCy NER model name for speaker detection.
        auto_speakers: Enable automatic speaker name detection using NER.
        cache_detected_hosts: Cache detected host names across episodes.
        generate_metadata: Generate per-episode metadata documents.
        metadata_format: Metadata file format ("json" or "yaml").
        metadata_subdirectory: Optional subdirectory for metadata files.
        generate_summaries: Generate episode summaries using AI models.
        summary_provider: Summary generation provider ("local", "openai", "anthropic").
        summary_model: Model identifier for MAP-phase summarization.
        summary_reduce_model: Optional separate model for REDUCE-phase summarization.
        summary_max_length: Maximum summary length in tokens.
        summary_min_length: Minimum summary length in tokens.
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
        speaker_detector_provider: Speaker detection provider type ("ner" or "openai").
        Deprecated: speaker_detector_type (use speaker_detector_provider instead).
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
    delay_ms: int = Field(default=0, alias="delay_ms")
    prefer_types: List[str] = Field(default_factory=list, alias="prefer_type")
    transcribe_missing: bool = Field(default=False, alias="transcribe_missing")
    whisper_model: str = Field(default="base", alias="whisper_model")
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
    workers: int = Field(default=DEFAULT_WORKERS, alias="workers")
    skip_existing: bool = Field(default=False, alias="skip_existing")
    clean_output: bool = Field(default=False, alias="clean_output")
    reuse_media: bool = Field(
        default=False,
        alias="reuse_media",
        description="Reuse existing media files instead of re-downloading (for faster testing)",
    )
    dry_run: bool = Field(default=False, alias="dry_run")
    language: str = Field(default=DEFAULT_LANGUAGE, alias="language")
    ner_model: Optional[str] = Field(default=None, alias="ner_model")
    auto_speakers: bool = Field(default=True, alias="auto_speakers")
    cache_detected_hosts: bool = Field(default=True, alias="cache_detected_hosts")
    # Provider selection fields (Stage 0: Foundation)
    speaker_detector_provider: Literal["ner", "openai"] = Field(
        default="ner",
        alias="speaker_detector_provider",
        description="Speaker detection provider type (default: 'ner' for spaCy NER). "
        "Deprecated alias 'speaker_detector_type' is supported for backward compatibility.",
    )
    # Deprecated field for backward compatibility - handled in _preprocess_config_data
    speaker_detector_type: Optional[Literal["ner", "openai"]] = Field(
        default=None,
        exclude=True,
        description="Deprecated: use speaker_detector_provider instead.",
    )
    transcription_provider: Literal["whisper", "openai"] = Field(
        default="whisper",
        alias="transcription_provider",
        description="Transcription provider type (default: 'whisper' for local Whisper)",
    )
    transcription_parallelism: int = Field(
        default=1,
        alias="transcription_parallelism",
        description=(
            "Episode-level parallelism: Number of episodes to transcribe in parallel "
            "(default: 1 for sequential. Whisper ignores >1, OpenAI uses for parallel API calls)"
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
    openai_transcription_model: str = Field(
        default="whisper-1",
        alias="openai_transcription_model",
        description="OpenAI Whisper API model version (default: 'whisper-1')",
    )
    openai_speaker_model: str = Field(
        default="gpt-4o-mini",
        alias="openai_speaker_model",
        description="OpenAI model for speaker detection (default: 'gpt-4o-mini')",
    )
    openai_summary_model: str = Field(
        default="gpt-4o-mini",
        alias="openai_summary_model",
        description="OpenAI model for summarization (default: 'gpt-4o-mini')",
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
        description="System prompt name for summarization (e.g. 'summarization/system_v1'). "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    openai_summary_user_prompt: str = Field(
        default="summarization/long_v1",
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
        default="ner/guest_host_v1",
        alias="openai_speaker_user_prompt",
        description="User prompt name for speaker detection/NER. "
        "Uses prompt_store (RFC-017) for versioned prompts.",
    )
    ner_prompt_params: Dict[str, Any] = Field(
        default_factory=dict,
        alias="ner_prompt_params",
        description="Template parameters for NER prompts (passed to Jinja2 templates).",
    )
    generate_metadata: bool = Field(default=False, alias="generate_metadata")
    metadata_format: Literal["json", "yaml"] = Field(default="json", alias="metadata_format")
    metadata_subdirectory: Optional[str] = Field(default=None, alias="metadata_subdirectory")
    generate_summaries: bool = Field(default=False, alias="generate_summaries")
    summary_provider: Literal["local", "openai", "anthropic"] = Field(
        default="local", alias="summary_provider"
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
    summary_max_length: int = Field(default=DEFAULT_SUMMARY_MAX_LENGTH, alias="summary_max_length")
    summary_min_length: int = Field(default=DEFAULT_SUMMARY_MIN_LENGTH, alias="summary_min_length")
    summary_device: Optional[str] = Field(default=None, alias="summary_device")
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
    summary_chunk_size: Optional[int] = Field(default=None, alias="summary_chunk_size")
    summary_word_chunk_size: Optional[int] = Field(
        default=DEFAULT_SUMMARY_WORD_CHUNK_SIZE,
        alias="summary_word_chunk_size",
        description="Chunk size in words for word-based chunking (800-1200 recommended)",
    )
    summary_word_overlap: Optional[int] = Field(
        default=DEFAULT_SUMMARY_WORD_OVERLAP,
        alias="summary_word_overlap",
        description="Overlap in words for word-based chunking (100-200 recommended)",
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

    model_config = ConfigDict(extra="forbid", populate_by_name=True, frozen=True)

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

        # Map deprecated speaker_detector_type to speaker_detector_provider
        if "speaker_detector_type" in data and "speaker_detector_provider" not in data:
            import warnings

            warnings.warn(
                "speaker_detector_type is deprecated, use speaker_detector_provider instead",
                DeprecationWarning,
                stacklevel=2,
            )
            data["speaker_detector_provider"] = data["speaker_detector_type"]
            # Remove deprecated key to avoid Pydantic extra field validation error
            del data["speaker_detector_type"]

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
        if "summary_cache_dir" not in data or data.get("summary_cache_dir") is None:
            env_cache_dir = os.getenv("SUMMARY_CACHE_DIR") or os.getenv("CACHE_DIR")
            if env_cache_dir:
                env_value = str(env_cache_dir).strip()
                if env_value:
                    data["summary_cache_dir"] = env_value

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

    @model_validator(mode="before")
    @classmethod
    def _map_deprecated_speaker_detector_field(cls, data: Any) -> Any:
        """Map deprecated speaker_detector_type to speaker_detector_provider."""
        if isinstance(data, dict):
            # If speaker_detector_type is provided but speaker_detector_provider is not,
            # copy the value to the new field name
            if "speaker_detector_type" in data and "speaker_detector_provider" not in data:
                import warnings

                warnings.warn(
                    "speaker_detector_type is deprecated, use speaker_detector_provider instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                data["speaker_detector_provider"] = data["speaker_detector_type"]
        return data

    @field_validator("speaker_detector_provider", mode="before")
    @classmethod
    def _validate_speaker_detector_provider(cls, value: Any) -> Literal["ner", "openai"]:
        """Validate speaker detector provider type."""
        if value is None or value == "":
            return "ner"
        value_str = str(value).strip().lower()
        if value_str not in ("ner", "openai"):
            raise ValueError("speaker_detector_provider must be 'ner' or 'openai'")
        return value_str  # type: ignore[return-value]

    @field_validator("transcription_provider", mode="before")
    @classmethod
    def _validate_transcription_provider(cls, value: Any) -> Literal["whisper", "openai"]:
        """Validate transcription provider."""
        if value is None or value == "":
            return "whisper"
        value_str = str(value).strip().lower()
        if value_str not in ("whisper", "openai"):
            raise ValueError("transcription_provider must be 'whisper' or 'openai'")
        return value_str  # type: ignore[return-value]

    @field_validator("summary_provider", mode="before")
    @classmethod
    def _validate_summary_provider(cls, value: Any) -> Literal["local", "openai", "anthropic"]:
        """Validate summary provider."""
        if value is None or value == "":
            return "local"
        value_str = str(value).strip().lower()
        if value_str not in ("local", "openai", "anthropic"):
            raise ValueError("summary_provider must be 'local', 'openai', or 'anthropic'")
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

    @field_validator("summary_model", mode="before")
    @classmethod
    def _coerce_summary_model(cls, value: Any) -> Optional[str]:
        """Coerce summary model to string or None."""
        if value is None or value == "":
            return None
        return str(value).strip() or None

    @field_validator("summary_max_length", mode="before")
    @classmethod
    def _ensure_summary_max_length(cls, value: Any) -> int:
        """Ensure summary max length is a positive integer."""
        if value is None or value == "":
            return DEFAULT_SUMMARY_MAX_LENGTH
        try:
            length = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("summary_max_length must be an integer") from exc
        if length < 1:
            raise ValueError("summary_max_length must be at least 1")
        return length

    @field_validator("summary_min_length", mode="before")
    @classmethod
    def _ensure_summary_min_length(cls, value: Any) -> int:
        """Ensure summary min length is a positive integer."""
        if value is None or value == "":
            return DEFAULT_SUMMARY_MIN_LENGTH
        try:
            length = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("summary_min_length must be an integer") from exc
        if length < 1:
            raise ValueError("summary_min_length must be at least 1")
        return length

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

        Validates logical relationships and dependencies between configuration fields:

        Summary Settings:
        - summary_max_length must be greater than summary_min_length
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

        # 1. summary_max_length must be greater than summary_min_length
        if self.summary_max_length <= self.summary_min_length:
            raise ValueError(
                f"summary_max_length ({self.summary_max_length}) must be greater than "
                f"summary_min_length ({self.summary_min_length})"
            )

        # 2. summary_word_chunk_size should be in recommended range (800-1200)
        #    Warn but don't error to allow experimentation
        if self.summary_word_chunk_size is not None:
            if self.summary_word_chunk_size < 800 or self.summary_word_chunk_size > 1200:
                warnings.warn(
                    f"summary_word_chunk_size ({self.summary_word_chunk_size}) is outside "
                    f"recommended range (800-1200). This may affect summary quality.",
                    UserWarning,
                    stacklevel=2,
                )

        # 3. summary_word_overlap should be in recommended range (100-200)
        #    Warn but don't error to allow experimentation
        if self.summary_word_overlap is not None:
            if self.summary_word_overlap < 100 or self.summary_word_overlap > 200:
                warnings.warn(
                    f"summary_word_overlap ({self.summary_word_overlap}) is outside "
                    f"recommended range (100-200). This may affect summary quality.",
                    UserWarning,
                    stacklevel=2,
                )

        # 4. summary_word_overlap must be less than summary_word_chunk_size
        if self.summary_word_chunk_size is not None and self.summary_word_overlap is not None:
            if self.summary_word_overlap >= self.summary_word_chunk_size:
                raise ValueError(
                    f"summary_word_overlap ({self.summary_word_overlap}) must be less than "
                    f"summary_word_chunk_size ({self.summary_word_chunk_size})"
                )

        # 5. generate_summaries requires generate_metadata
        #    Summaries are stored in metadata files, so metadata generation must be enabled
        if self.generate_summaries and not self.generate_metadata:
            raise ValueError(
                "generate_summaries=True requires generate_metadata=True "
                "(summaries are stored in metadata files)"
            )

        # === Output Control Validation ===

        # 6. clean_output and skip_existing are mutually exclusive
        #    clean_output removes all files, making skip_existing meaningless
        if self.clean_output and self.skip_existing:
            raise ValueError(
                "clean_output and skip_existing are mutually exclusive "
                "(clean_output removes all existing files, making skip_existing meaningless)"
            )

        # 7. clean_output and reuse_media are mutually exclusive
        #    clean_output removes media files that reuse_media would reuse
        if self.clean_output and self.reuse_media:
            raise ValueError(
                "clean_output and reuse_media are mutually exclusive "
                "(clean_output removes media files that would be reused)"
            )

        # === Transcription Validation ===

        # 8. transcribe_missing requires a valid whisper_model
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
