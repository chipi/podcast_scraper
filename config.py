from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class Config(BaseModel):
    rss_url: Optional[str] = Field(default=None, alias="rss")
    output_dir: Optional[str] = Field(default=None, alias="output_dir")
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
    workers: int = Field(default=DEFAULT_WORKERS, alias="workers")
    skip_existing: bool = Field(default=False, alias="skip_existing")
    clean_output: bool = Field(default=False, alias="clean_output")
    dry_run: bool = Field(default=False, alias="dry_run")
    language: str = Field(default=DEFAULT_LANGUAGE, alias="language")
    ner_model: Optional[str] = Field(default=None, alias="ner_model")
    auto_speakers: bool = Field(default=True, alias="auto_speakers")
    cache_detected_hosts: bool = Field(default=True, alias="cache_detected_hosts")
    generate_metadata: bool = Field(default=False, alias="generate_metadata")
    metadata_format: Literal["json", "yaml"] = Field(default="json", alias="metadata_format")
    metadata_subdirectory: Optional[str] = Field(default=None, alias="metadata_subdirectory")

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
    def _strip_output_dir(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None

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

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> str:
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


def load_config_file(
    path: str,
) -> Dict[str, Any]:  # noqa: C901 - file parsing handles multiple formats
    """Load JSON or YAML configuration file and return a dictionary."""
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
