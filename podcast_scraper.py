#!/usr/bin/env python3

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import time
import warnings
from platformdirs import PlatformDirs

import requests
from requests.utils import requote_uri

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse, urlunparse

from defusedxml.ElementTree import ParseError as DefusedXMLParseError, fromstring as safe_fromstring
from pydantic import BaseModel, ValidationError, field_validator

from tqdm import tqdm

import xml.etree.ElementTree as ET

__version__ = "1.0.0"

DEFAULT_LOG_LEVEL = "INFO"

# Set up logging to stderr
logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL, logging.INFO),
    format="%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def resolve_log_level(level: str) -> int:
    if level is None:
        raise ValueError("Log level cannot be None")
    numeric_level = getattr(logging, str(level).upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    return numeric_level


def apply_log_level(level: str) -> None:
    numeric_level = resolve_log_level(level)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    for handler in root_logger.handlers:
        handler.setLevel(numeric_level)
    logger.setLevel(numeric_level)

REQUEST_SESSION = requests.Session()

XML_Element = ET.Element

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/119.0 Safari/537.36"
)

# Constants
DOWNLOAD_CHUNK_SIZE = 1024 * 256  # 256KB chunks for file downloads
URL_HASH_LENGTH = 8  # Length of hash suffix in output directory names
DEFAULT_TIMEOUT_SECONDS = 20  # Default HTTP request timeout
DEFAULT_SCREENPLAY_GAP_SECONDS = 1.25  # Default gap between speakers for screenplay formatting
DEFAULT_NUM_SPEAKERS = 2  # Default number of speakers for screenplay formatting
MIN_TIMEOUT_SECONDS = 1  # Minimum allowed timeout value
MIN_NUM_SPEAKERS = 1  # Minimum number of speakers
TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"  # Format for auto-generated run IDs
EPISODE_NUMBER_FORMAT_WIDTH = 4  # Width for episode number formatting (e.g., 0001)
BYTES_PER_MB = 1024 * 1024  # Bytes in a megabyte
MS_TO_SECONDS = 1000.0  # Milliseconds to seconds conversion factor
TEMP_DIR_NAME = ".tmp_media"  # Name of temporary directory for media files


@dataclass
class Config:
    rss_url: str
    output_dir: str
    max_episodes: Optional[int]
    user_agent: str
    timeout: int
    delay_ms: int
    prefer_types: List[str]
    transcribe_missing: bool
    whisper_model: str
    screenplay: bool
    screenplay_gap_s: float
    screenplay_num_speakers: int
    screenplay_speaker_names: List[str]
    run_id: Optional[str]
    log_level: str = DEFAULT_LOG_LEVEL


class ConfigFileModel(BaseModel):
    rss: Optional[str] = None
    output_dir: Optional[str] = None
    max_episodes: Optional[int] = None
    user_agent: Optional[str] = None
    timeout: Optional[int] = None
    delay_ms: Optional[int] = None
    prefer_type: Optional[List[str]] = None
    transcribe_missing: Optional[bool] = None
    whisper_model: Optional[str] = None
    screenplay: Optional[bool] = None
    screenplay_gap: Optional[float] = None
    num_speakers: Optional[int] = None
    speaker_names: Optional[List[str]] = None
    run_id: Optional[str] = None
    log_level: Optional[str] = None

    @field_validator("prefer_type", mode="before")
    @classmethod
    def _coerce_prefer_type(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        raise TypeError("prefer_type must be a string or list of strings")

    @field_validator("speaker_names", mode="before")
    @classmethod
    def _coerce_speaker_names(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [name.strip() for name in value.split(",") if name.strip()]
        if isinstance(value, list):
            return [str(item) for item in value]
        raise TypeError("speaker_names must be a string or list of strings")

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: Any) -> Optional[str]:
        if value is None:
            return None
        return str(value).upper()

# Precompiled regex patterns for performance
RE_NEWLINES_TABS = re.compile(r"[\r\n\t]")  # Match newlines and tabs
RE_MULTIPLE_WHITESPACE = re.compile(r"\s+")  # Match multiple whitespace characters
RE_NON_FILENAME_CHARS = re.compile(r"[^\w\- .]")  # Match characters not allowed in filenames

def normalize_url(url: str) -> str:
    """Normalize URLs while preserving already-encoded segments."""
    return requote_uri(url)


def sanitize_filename(name: str) -> str:
    name = RE_NEWLINES_TABS.sub(" ", name)
    name = RE_MULTIPLE_WHITESPACE.sub(" ", name).strip()
    name = RE_NON_FILENAME_CHARS.sub("_", name)
    return name or "untitled"


def validate_and_normalize_output_dir(path: str) -> str:
    """Validate and normalize output directory path to prevent path traversal attacks."""
    if not path or not path.strip():
        raise ValueError("Output directory path cannot be empty")
    
    # Use pathlib for better path handling
    path_obj = Path(path).expanduser()
    
    # Resolve the path (normalizes .. and . components, resolves symlinks)
    try:
        resolved = path_obj.resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid output directory path: {path} ({e})")
    
    app_dirs = PlatformDirs("podcast_scraper", ensure_exists=False)
    safe_roots = {
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path(app_dirs.user_data_dir).resolve(),
        Path(app_dirs.user_cache_dir).resolve(),
    }

    if any(resolved == root or resolved.is_relative_to(root) for root in safe_roots):
        return str(resolved)

    logger.warning(
        "Output directory %s is outside recommended locations (home or app data).",
        resolved,
    )
    return str(resolved)


def derive_output_dir(rss_url: str, override: Optional[str]) -> str:
    if override:
        return validate_and_normalize_output_dir(override)
    parsed = urlparse(rss_url)
    base = parsed.netloc or "feed"
    # include a short hash of the full URL to disambiguate same host feeds
    h = hashlib.sha1(rss_url.encode("utf-8")).hexdigest()[:URL_HASH_LENGTH]
    return f"output_rss_{sanitize_filename(base)}_{h}"


def load_config_file(path: str) -> Dict[str, Any]:
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
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                "YAML config requires PyYAML; install with 'pip install pyyaml'."
            ) from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"Invalid YAML config file {resolved}: {exc}") from exc
    else:
        raise ValueError(f"Unsupported config file type: {resolved.suffix}")

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping/object at the top level")

    return data


def parse_rss_items(xml_bytes: bytes) -> Tuple[str, List[ET.Element]]:
    # Return (feed_title, item_elements)
    # Use defusedxml for secure parsing (protects against XML bombs, external entity attacks, etc.)
    root = safe_fromstring(xml_bytes)
    channel = root.find("channel")
    if channel is None:
        # some feeds use namespaces; attempt generic search
        channel = next((e for e in root.iter() if e.tag.endswith("channel")), None)
    title = ""
    if channel is not None:
        t = channel.find("title") or next((e for e in channel.iter() if e.tag.endswith("title")), None)
        if t is not None and t.text:
            title = t.text.strip()
        items = list(channel.findall("item"))
        if not items:
            items = [e for e in channel if isinstance(e.tag, str) and e.tag.endswith("item")]
    else:
        title = ""
        items = [e for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("item")]
    return title, items


def find_transcript_urls(item: ET.Element, base_url: str) -> List[Tuple[str, Optional[str]]]:
    # Returns list of (url, type) with relative URLs resolved against base_url
    candidates: List[Tuple[str, Optional[str]]] = []

    # podcast:transcript (Podcasting 2.0)
    for el in item.iter():
        tag = el.tag
        if isinstance(tag, str) and tag.lower().endswith("transcript"):
            url_attr = el.attrib.get("url") or el.attrib.get("href")
            if url_attr:
                t = el.attrib.get("type")
                # Resolve relative URLs against base_url
                resolved_url = urljoin(base_url, url_attr.strip())
                candidates.append((resolved_url, (t.strip() if t else None)))

    # Some feeds use <transcript> without namespace
    for el in item.findall("transcript"):
        if el.text and el.text.strip():
            # Resolve relative URLs against base_url
            resolved_url = urljoin(base_url, el.text.strip())
            candidates.append((resolved_url, None))

    # Deduplicate preserving order
    seen = set()
    unique: List[Tuple[str, Optional[str]]] = []
    for u, t in candidates:
        key = (u, t or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append((u, t))
    return unique


def find_enclosure_media(item: ET.Element, base_url: str) -> Optional[Tuple[str, Optional[str]]]:
    # Look for <enclosure url="..." type="audio/mpeg"/> and resolve relative URLs
    for el in item.iter():
        if isinstance(el.tag, str) and el.tag.lower().endswith("enclosure"):
            url_attr = el.attrib.get("url")
            if url_attr:
                # Resolve relative URLs against base_url
                resolved_url = urljoin(base_url, url_attr.strip())
                return resolved_url, (el.attrib.get("type") or None)
    return None


def choose_transcript_url(candidates: List[Tuple[str, Optional[str]]], prefer_types: List[str]) -> Optional[Tuple[str, Optional[str]]]:
    if not candidates:
        return None
    if not prefer_types:
        return candidates[0]
    lowered = [(u, t.lower() if t else None) for (u, t) in candidates]
    for pref in prefer_types:
        p = pref.lower().strip()
        for idx, (u, t_lower) in enumerate(lowered):
            orig_url, orig_type = candidates[idx]
            if (t_lower and p in t_lower) or orig_url.lower().endswith(p):
                return orig_url, orig_type
    return candidates[0]

def _format_screenplay_from_segments(segments: List[dict], num_speakers: int, speaker_names: List[str], gap_s: float) -> str:
    if not segments:
        return ""
    # Sort by start time to be safe
    segs = sorted(segments, key=lambda s: float(s.get("start") or 0.0))
    current_speaker_idx = 0
    lines: List[Tuple[int, str]] = []  # (speaker_idx, text)
    prev_end: Optional[float] = None

    for s in segs:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        start = float(s.get("start") or 0.0)
        end = float(s.get("end") or start)
        if prev_end is not None and start - prev_end > gap_s:
            current_speaker_idx = (current_speaker_idx + 1) % max(MIN_NUM_SPEAKERS, num_speakers)
        prev_end = end
        if lines and lines[-1][0] == current_speaker_idx:
            # merge with previous same speaker line
            lines[-1] = (lines[-1][0], lines[-1][1] + (" " if lines[-1][1] else "") + text)
        else:
            lines.append((current_speaker_idx, text))

    def speaker_label(idx: int) -> str:
        if 0 <= idx < len(speaker_names):
            return speaker_names[idx]
        return f"SPEAKER {idx + 1}"

    out_lines = [f"{speaker_label(idx)}: {txt}" for (idx, txt) in lines]
    return "\n".join(out_lines) + "\n"


def write_file(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def validate_args(args: argparse.Namespace) -> None:
    """Validate input parameters and raise ValueError with descriptive message if invalid."""
    errors: List[str] = []
    
    # Validate RSS URL
    if not args.rss or not args.rss.strip():
        errors.append("RSS URL is required")
    else:
        parsed = urlparse(args.rss)
        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            errors.append(f"RSS URL must be http or https: {args.rss}")
        if not parsed.netloc:
            errors.append(f"RSS URL must have a valid hostname: {args.rss}")
    
    # Validate max-episodes
    if args.max_episodes is not None and args.max_episodes <= 0:
        errors.append(f"--max-episodes must be positive, got: {args.max_episodes}")
    
    # Validate timeout
    if args.timeout <= 0:
        errors.append(f"--timeout must be positive, got: {args.timeout}")
    
    # Validate delay-ms
    if args.delay_ms < 0:
        errors.append(f"--delay-ms must be non-negative, got: {args.delay_ms}")
    
    # Validate whisper-model
    valid_models = (
        "tiny", "base", "small", "medium", "large", "large-v2", "large-v3",
        "tiny.en", "base.en", "small.en", "medium.en", "large.en"
    )
    if args.transcribe_missing and args.whisper_model not in valid_models:
        errors.append(f"--whisper-model must be one of {valid_models}, got: {args.whisper_model}")
    
    # Validate screenplay speakers
    if args.screenplay:
        if args.num_speakers < MIN_NUM_SPEAKERS:
            errors.append(f"--num-speakers must be at least {MIN_NUM_SPEAKERS}, got: {args.num_speakers}")
    
    # Validate speaker names
    if args.speaker_names:
        names = [n.strip() for n in args.speaker_names.split(",") if n.strip()]
        if len(names) < MIN_NUM_SPEAKERS:
            errors.append("At least two speaker names required when specifying --speaker-names")

    # Validate log level
    try:
        resolve_log_level(args.log_level)
    except ValueError as exc:
        errors.append(str(exc))

    # Validate output directory if provided
    if args.output_dir:
        try:
            validate_and_normalize_output_dir(args.output_dir)
        except ValueError as e:
            errors.append(str(e))
    
    if errors:
        raise ValueError("Invalid input parameters:\n  " + "\n  ".join(errors))


CONFIG_FIELD_TYPES: Dict[str, type] = {
    "rss": str,
    "output_dir": str,
    "max_episodes": int,
    "user_agent": str,
    "timeout": int,
    "delay_ms": int,
    "transcribe_missing": bool,
    "whisper_model": str,
    "screenplay": bool,
    "screenplay_gap": float,
    "num_speakers": int,
    "run_id": str,
}

_BOOL_TRUE = {"true", "yes", "1", "on"}
_BOOL_FALSE = {"false", "no", "0", "off"}


def _coerce_config_field(key: str, value: Any) -> Any:
    expected = CONFIG_FIELD_TYPES[key]
    if value is None:
        return None
    if expected is str:
        if isinstance(value, str):
            return value
        return str(value)
    if expected is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError as exc:
                raise ValueError(f"Config field '{key}' must be an integer") from exc
        raise ValueError(f"Config field '{key}' must be an integer")
    if expected is float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError as exc:
                raise ValueError(f"Config field '{key}' must be a number") from exc
        raise ValueError(f"Config field '{key}' must be a number")
    if expected is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _BOOL_TRUE:
                return True
            if lowered in _BOOL_FALSE:
                return False
        raise ValueError(f"Config field '{key}' must be a boolean")
    raise ValueError(f"Unsupported config field '{key}'")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download podcast episode transcripts from an RSS feed.")
    parser.add_argument("--config", default=None, help="Path to configuration file (JSON or YAML)")
    parser.add_argument("rss", nargs="?", default=None, help="Podcast RSS feed URL")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: output_rss_<host>_<hash>)")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of episodes to process")
    parser.add_argument("--prefer-type", action="append", default=[], help="Preferred transcript types or extensions (repeatable), e.g. text/plain, .vtt, .srt")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="User-Agent header")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="Request timeout in seconds")
    parser.add_argument("--delay-ms", type=int, default=0, help="Delay between requests (ms)")
    parser.add_argument("--transcribe-missing", action="store_true", help="Use Whisper to transcribe when no transcript is provided")
    parser.add_argument("--whisper-model", default="base", help="Whisper model to use (e.g., tiny, base, small, medium, large, or tiny.en, base.en, small.en, medium.en, large.en for English-only)")
    parser.add_argument("--screenplay", action="store_true", help="Format Whisper transcript as screenplay with speaker turns")
    parser.add_argument("--screenplay-gap", type=float, default=DEFAULT_SCREENPLAY_GAP_SECONDS, help="Gap (seconds) to trigger speaker change when formatting screenplay")
    parser.add_argument("--num-speakers", type=int, default=DEFAULT_NUM_SPEAKERS, help="Number of speakers to alternate between when formatting screenplay")
    parser.add_argument("--speaker-names", default="", help="Comma-separated speaker names to use instead of SPEAKER 1..N")
    parser.add_argument("--run-id", default=None, help="Optional run identifier to create a unique subfolder under the output directory; use 'auto' for timestamp")
    parser.add_argument("--version", action="store_true", help="Show program version and exit")
    parser.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, type=str.upper, help="Logging level (e.g., DEBUG, INFO, WARNING, ERROR)")
    # Parse once to check for config file, then re-parse with config defaults if needed
    initial_args, _ = parser.parse_known_args(argv)

    if initial_args.version:
        print(f"podcast_scraper {__version__}")
        raise SystemExit(0)

    if initial_args.config:
        config_data = load_config_file(initial_args.config)
        valid_dests = {action.dest for action in parser._actions if action.dest}
        unknown_keys = [key for key in config_data.keys() if key not in valid_dests]
        if unknown_keys:
            raise ValueError(
                "Unknown config option(s): " + ", ".join(sorted(unknown_keys))
            )

        try:
            config_model = ConfigFileModel.model_validate(config_data)
        except ValidationError as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc

        defaults_updates: Dict[str, Any] = config_model.model_dump(exclude_none=True)

        prefer_list = defaults_updates.pop("prefer_type", None)
        if prefer_list is not None:
            defaults_updates["prefer_type"] = prefer_list

        speaker_list = defaults_updates.pop("speaker_names", None)
        if speaker_list is not None:
            defaults_updates["speaker_names"] = ",".join(speaker_list)

        parser.set_defaults(**defaults_updates)
        args = parser.parse_args(argv)
        if not args.rss:
            raise ValueError("RSS URL is required (provide in config as 'rss' or via CLI)")
        if args.version:
            print(f"podcast_scraper {__version__}")
            raise SystemExit(0)
    else:
        args = parser.parse_args(argv)
        if args.version:
            print(f"podcast_scraper {__version__}")
            raise SystemExit(0)

    validate_args(args)
    return args


def load_whisper_model(cfg: Config):
    """Load Whisper model if transcription is enabled. Returns model or None."""
    if not cfg.transcribe_missing:
        return None
    try:
        import whisper  # type: ignore
        logger.info(f"Loading Whisper model ({cfg.whisper_model})...")
        model = whisper.load_model(cfg.whisper_model)
        logger.info("Whisper model loaded successfully.")
        # If running on CPU, let the user know FP16 won't be used so they see the message before transcription starts.
        device = getattr(model, "device", None)
        device_type = getattr(device, "type", None)
        is_cpu_device = device_type == "cpu"
        setattr(model, "_is_cpu_device", is_cpu_device)
        if is_cpu_device:
            logger.info("Whisper is running on CPU; FP16 is unavailable so FP32 will be used.")
        return model
    except ImportError:
        logger.warning("openai-whisper not installed. Install with: pip install openai-whisper && brew install ffmpeg")
        return None
    except (RuntimeError, OSError) as e:
        logger.warning(f"Failed to load Whisper model: {e}")
        return None


def extract_episode_title(item: ET.Element, idx: int) -> Tuple[str, str]:
    """Extract episode title from RSS item. Returns (title, sanitized_title)."""
    title_el = item.find("title") or next((e for e in item.iter() if e.tag.endswith("title")), None)
    ep_title = (title_el.text.strip() if title_el is not None and title_el.text else f"episode_{idx}")
    ep_title_safe = sanitize_filename(ep_title)
    return ep_title, ep_title_safe


def derive_media_extension(media_type: Optional[str], media_url: str) -> str:
    """Derive file extension from media type or URL. Returns extension with leading dot."""
    ext = ".bin"
    if media_type and "/" in media_type:
        ext_guess = media_type.split("/", 1)[1].lower()
        if ext_guess in ("mpeg", "mp3"):
            ext = ".mp3"
        elif ext_guess in ("m4a", "mp4", "aac"):
            ext = ".m4a"
        elif ext_guess in ("ogg", "oga"):
            ext = ".ogg"
        elif ext_guess in ("wav",):
            ext = ".wav"
        elif ext_guess in ("webm",):
            ext = ".webm"
    else:
        low = media_url.lower()
        for cand in (".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".wav", ".webm"):
            if low.endswith(cand):
                ext = cand
                break
    return ext


def derive_transcript_extension(transcript_type: Optional[str], content_type: Optional[str], transcript_url: str) -> str:
    """Derive file extension from transcript type, content type, or URL. Returns extension with leading dot."""
    ext = ".txt"
    if transcript_type:
        if "vtt" in transcript_type.lower():
            ext = ".vtt"
        elif "srt" in transcript_type.lower():
            ext = ".srt"
        elif "json" in transcript_type.lower():
            ext = ".json"
        elif "html" in transcript_type.lower():
            ext = ".html"
    else:
        # infer from content-type header or URL
        if content_type and "vtt" in content_type.lower():
            ext = ".vtt"
        elif content_type and "srt" in content_type.lower():
            ext = ".srt"
        elif content_type and "json" in content_type.lower():
            ext = ".json"
        elif content_type and "html" in content_type.lower():
            ext = ".html"
        else:
            low = transcript_url.lower()
            for cand in (".vtt", ".srt", ".json", ".html", ".txt"):
                if low.endswith(cand):
                    ext = cand
                    break
    return ext


def transcribe_with_whisper(whisper_model, temp_media: str, cfg: Config) -> Tuple[dict, float]:
    """Transcribe audio/video file using Whisper. Returns (transcription_result_dict, elapsed_seconds)."""
    logger.info(f"    transcribing with Whisper ({cfg.whisper_model})...")
    tc_start = time.time()
    # Show simple progress indicator for transcription (indeterminate since Whisper doesn't expose progress)
    # Use a simple format without bar animation to avoid duplicate displays
    with tqdm(
        total=None,
        desc="Transcribing",
        unit="",
        ncols=80,
        bar_format="{desc}: {elapsed}",
        leave=False,
        dynamic_ncols=False,
        miniters=1,
        mininterval=0.5,
    ) as pbar:
        suppress_fp16_warning = getattr(whisper_model, "_is_cpu_device", False)
        if suppress_fp16_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="FP16 is not supported on CPU",
                    category=UserWarning,
                )
                result = whisper_model.transcribe(temp_media, task="transcribe", language="en", verbose=False)
        else:
            result = whisper_model.transcribe(temp_media, task="transcribe", language="en", verbose=False)
        # Update once to show completion
        pbar.update(1)
    tc_elapsed = time.time() - tc_start
    return result, tc_elapsed


def _open_http_request(url: str, user_agent: str, timeout: int, stream: bool = False):
    """Shared helper to execute HTTP GET requests with consistent error handling."""
    normalized_url = normalize_url(url)
    headers = {"User-Agent": user_agent}
    try:
        resp = REQUEST_SESSION.get(normalized_url, headers=headers, timeout=timeout, stream=stream)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def http_get(url: str, user_agent: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    resp = _open_http_request(url, user_agent, timeout, stream=True)
    if resp is None:
        return None, None
    try:
        ctype = resp.headers.get("Content-Type", "")
        content_length = resp.headers.get("Content-Length")
        try:
            total_size = int(content_length) if content_length else None
        except (TypeError, ValueError):
            total_size = None

        body_parts: List[bytes] = []
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Downloading",
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                body_parts.append(chunk)
                pbar.update(len(chunk))

        return b"".join(body_parts), ctype
    except (requests.RequestException, OSError) as exc:
        logger.warning("Failed to read response from %s: %s", url, exc)
        return None, None
    finally:
        resp.close()


def http_download_to_file(url: str, user_agent: str, timeout: int, out_path: str) -> Tuple[bool, int]:
    resp = _open_http_request(url, user_agent, timeout, stream=True)
    if resp is None:
        return False, 0
    try:
        content_length = resp.headers.get("Content-Length")
        try:
            total_size = int(content_length) if content_length else None
        except (TypeError, ValueError):
            total_size = None

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        filename = os.path.basename(out_path) or os.path.basename(urlparse(url).path)

        total_bytes = 0
        with open(out_path, "wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading {filename}" if filename else "Downloading",
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                chunk_size = len(chunk)
                total_bytes += chunk_size
                pbar.update(chunk_size)
        return True, total_bytes
    except (requests.RequestException, OSError) as exc:
        logger.warning("Failed to download %s to %s: %s", url, out_path, exc)
        return False, 0
    finally:
        resp.close()


def download_and_transcribe_media(
    item: ET.Element,
    idx: int,
    ep_title: str,
    ep_title_safe: str,
    cfg: Config,
    whisper_model,
    temp_dir: str,
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_base_url: str,
) -> bool:
    """Download media and transcribe with Whisper. Returns True if successful."""
    media = find_enclosure_media(item, feed_base_url)
    if not media:
        logger.info(f"[{idx}] no transcript or enclosure for: {ep_title}")
        return False
    
    media_url, media_type = media
    logger.info(f"[{idx}] no transcript; downloading media for Whisper: {ep_title}")

    # Derive extension from media type or URL
    ext = derive_media_extension(media_type, media_url)

    ep_num_str = f"{idx:0{EPISODE_NUMBER_FORMAT_WIDTH}d}"
    temp_media = os.path.join(temp_dir, f"{ep_num_str}_{ep_title_safe}{ext}")
    dl_start = time.time()
    ok, total_bytes = http_download_to_file(media_url, cfg.user_agent, cfg.timeout, temp_media)
    dl_elapsed = time.time() - dl_start
    if not ok:
        logger.warning("    failed to download media for transcription")
        return False
    
    try:
        mb = total_bytes / BYTES_PER_MB
        logger.info(f"    downloaded {mb:.2f} MB in {dl_elapsed:.1f}s")
    except (ValueError, ZeroDivisionError, TypeError) as e:
        logger.warning(f"    failed to format download size: {e}")

    # Use pre-loaded Whisper model
    if whisper_model is None:
        logger.warning("    Skipping transcription: Whisper model not available")
        # Clean up temp file
        try:
            os.remove(temp_media)
        except OSError as e:
            logger.warning(f"    failed to remove temp media file {temp_media}: {e}")
        return False

    try:
        result, tc_elapsed = transcribe_with_whisper(whisper_model, temp_media, cfg)
        text = (result.get("text") or "").strip()
        # Optionally format as screenplay using segments
        if cfg.screenplay and isinstance(result, dict) and isinstance(result.get("segments"), list):
            try:
                formatted = _format_screenplay_from_segments(
                    result["segments"], cfg.screenplay_num_speakers, cfg.screenplay_speaker_names, cfg.screenplay_gap_s
                )
                if formatted.strip():
                    text = formatted
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"    failed to format as screenplay, using plain transcript: {e}")
        if not text:
            raise RuntimeError("empty transcription")
        # Include run identifier in filename for easy comparison across runs
        run_tag = f"_{run_suffix}" if run_suffix else ""
        out_name = f"{idx:0{EPISODE_NUMBER_FORMAT_WIDTH}d} - {ep_title_safe}{run_tag}.txt"
        out_path = os.path.join(effective_output_dir, out_name)
        write_file(out_path, text.encode("utf-8"))
        logger.info(f"    saved transcript: {out_path} (transcribed in {tc_elapsed:.1f}s)")
        return True
    except (RuntimeError, OSError) as e:
        logger.error(f"    Whisper transcription failed: {e}")
        return False
    finally:
        # best-effort cleanup of temp media
        try:
            os.remove(temp_media)
        except OSError as e:
            logger.warning(f"    failed to remove temp media file {temp_media}: {e}")


def process_transcript_download(
    transcript_url: str,
    transcript_type: Optional[str],
    idx: int,
    ep_title: str,
    ep_title_safe: str,
    cfg: Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> bool:
    """Download and save transcript. Returns True if successful."""
    logger.info(f"[{idx}] downloading transcript: {ep_title} -> {transcript_url}")

    data, ctype = http_get(transcript_url, cfg.user_agent, cfg.timeout)
    if data is None:
        logger.warning(f"    failed to download transcript")
        return False

    # Decide extension
    ext = derive_transcript_extension(transcript_type, ctype, transcript_url)

    # Include run identifier in filename for easy comparison across runs
    run_tag = f"_{run_suffix}" if run_suffix else ""
    out_name = f"{idx:0{EPISODE_NUMBER_FORMAT_WIDTH}d} - {ep_title_safe}{run_tag}{ext}"
    out_path = os.path.join(effective_output_dir, out_name)
    try:
        write_file(out_path, data)
        logger.info(f"    saved: {out_path}")
        return True
    except (IOError, OSError) as e:
        logger.error(f"    failed to write file: {e}")
        return False


def process_episode(
    item: ET.Element,
    idx: int,
    cfg: Config,
    whisper_model,
    temp_dir: Optional[str],
    effective_output_dir: str,
    run_suffix: Optional[str],
    feed_base_url: str,
) -> bool:
    """Process a single episode. Returns True if transcript was saved."""
    ep_title, ep_title_safe = extract_episode_title(item, idx)

    candidates = find_transcript_urls(item, feed_base_url)
    chosen = choose_transcript_url(candidates, cfg.prefer_types)

    if not chosen:
        # fallback to enclosure media if requested
        if cfg.transcribe_missing and temp_dir:
            success = download_and_transcribe_media(
                item,
                idx,
                ep_title,
                ep_title_safe,
                cfg,
                whisper_model,
                temp_dir,
                effective_output_dir,
                run_suffix,
                feed_base_url,
            )
            if cfg.delay_ms:
                time.sleep(cfg.delay_ms / MS_TO_SECONDS)
            return success
        else:
            logger.info(f"[{idx}] no transcript for: {ep_title}")
            return False

    t_url, t_type = chosen
    success = process_transcript_download(
        t_url, t_type, idx, ep_title, ep_title_safe, cfg, effective_output_dir, run_suffix
    )
    if cfg.delay_ms:
        time.sleep(cfg.delay_ms / MS_TO_SECONDS)
    return success


def fetch_and_parse_rss(cfg: Config) -> Tuple[str, List[ET.Element], str]:
    """Fetch and parse RSS feed. Returns (feed_title, items, feed_base_url)."""
    resp = _open_http_request(cfg.rss_url, cfg.user_agent, cfg.timeout, stream=False)
    if resp is None:
        raise ValueError("Failed to fetch RSS feed.")
    try:
        rss_bytes = resp.content
        feed_base_url = resp.url or cfg.rss_url
    finally:
        resp.close()

    try:
        feed_title, items = parse_rss_items(rss_bytes)
    except (DefusedXMLParseError, ValueError) as e:
        raise ValueError(f"Failed to parse RSS XML: {e}") from e

    return feed_title, items, feed_base_url


def setup_output_directory(cfg: Config) -> Tuple[str, Optional[str]]:
    """Setup output directory with run suffix. Returns (effective_output_dir, run_suffix)."""
    run_suffix: Optional[str] = None
    if cfg.run_id:
        run_suffix = time.strftime(TIMESTAMP_FORMAT) if cfg.run_id.lower() == "auto" else sanitize_filename(cfg.run_id)
        # Append Whisper model name if using transcription
        if cfg.transcribe_missing:
            model_part = sanitize_filename(cfg.whisper_model)
            run_suffix = f"{run_suffix}_whisper_{model_part}" if run_suffix else f"whisper_{model_part}"
    elif cfg.transcribe_missing:
        # Auto-create run-id with model if Whisper is used but no run-id specified
        model_part = sanitize_filename(cfg.whisper_model)
        run_suffix = f"whisper_{model_part}"
    effective_output_dir = os.path.join(cfg.output_dir, f"run_{run_suffix}") if run_suffix else cfg.output_dir
    return effective_output_dir, run_suffix


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        args = parse_args(argv)
    except ValueError as e:
        logger.error(f"Error: {e}")
        return 1

    apply_log_level(args.log_level)

    cfg = Config(
        rss_url=args.rss,
        output_dir=derive_output_dir(args.rss, args.output_dir),
        max_episodes=(args.max_episodes if args.max_episodes and args.max_episodes > 0 else None),
        user_agent=args.user_agent,
        timeout=max(MIN_TIMEOUT_SECONDS, args.timeout),
        delay_ms=max(0, args.delay_ms),
        prefer_types=args.prefer_type,
        transcribe_missing=bool(args.transcribe_missing),
        whisper_model=str(args.whisper_model or "base"),
        screenplay=bool(args.screenplay),
        screenplay_gap_s=float(args.screenplay_gap),
        screenplay_num_speakers=max(MIN_NUM_SPEAKERS, int(args.num_speakers)),
        screenplay_speaker_names=[s.strip() for s in (args.speaker_names or "").split(",") if s.strip()],
        run_id=(str(args.run_id) if args.run_id else None),
        log_level=args.log_level,
    )

    # Setup output directory
    effective_output_dir, run_suffix = setup_output_directory(cfg)

    try:
        logger.info("Starting podcast transcript scrape")
        logger.info(f"  rss: {cfg.rss_url}")
        logger.info(f"  output_dir: {effective_output_dir}")
        logger.info(f"  max_episodes: {cfg.max_episodes or 'all'}")
        logger.info(f"  log_level: {cfg.log_level}")
    except (IOError, OSError) as e:
        logger.warning(f"Failed to write startup message: {e}")

    # Fetch and parse RSS feed
    try:
        feed_title, items, feed_base_url = fetch_and_parse_rss(cfg)
    except ValueError as e:
        logger.error(f"{e}")
        return 1

    total_items = len(items)
    if cfg.max_episodes is not None:
        items = items[: cfg.max_episodes]

    logger.info(f"Episodes to process: {len(items)} of {total_items}")

    # Load Whisper model once if transcription is enabled
    whisper_model = load_whisper_model(cfg)

    # Create temp directory once if transcription is enabled
    temp_dir = None
    if cfg.transcribe_missing:
        temp_dir = os.path.join(effective_output_dir, TEMP_DIR_NAME)
        os.makedirs(temp_dir, exist_ok=True)

    saved = 0
    for idx, item in enumerate(items, start=1):
        if process_episode(item, idx, cfg, whisper_model, temp_dir, effective_output_dir, run_suffix, feed_base_url):
            saved += 1

    # Clean up temp directory if it was created
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temp directory: {temp_dir}")
        except OSError as e:
            logger.warning(f"Failed to remove temp directory {temp_dir}: {e}")

    logger.info(f"Done. transcripts_saved={saved} in {effective_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
