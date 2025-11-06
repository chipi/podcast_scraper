#!/usr/bin/env python3

import argparse
import hashlib
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, build_opener, HTTPSHandler, HTTPHandler

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

# Precompiled regex patterns for performance
RE_NEWLINES_TABS = re.compile(r"[\r\n\t]")  # Match newlines and tabs
RE_MULTIPLE_WHITESPACE = re.compile(r"\s+")  # Match multiple whitespace characters
RE_NON_FILENAME_CHARS = re.compile(r"[^\w\- .]")  # Match characters not allowed in filenames

@dataclass
class Config:
    rss_url: str
    output_dir: Optional[str]
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


def _open_http_request(url: str, user_agent: str, timeout: int):
    """Shared helper to open an HTTP request. Returns response object or None on error."""
    headers = {"User-Agent": user_agent, "Accept": "*/*"}
    req = Request(url, headers=headers)
    opener = build_opener(HTTPHandler(), HTTPSHandler())
    try:
        return opener.open(req, timeout=timeout)
    except (HTTPError, URLError) as e:
        sys.stderr.write(f"Warning: failed to fetch {url}: {e}\n")
        sys.stderr.flush()
        return None


def http_get(url: str, user_agent: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    resp = _open_http_request(url, user_agent, timeout)
    if resp is None:
        return None, None
    try:
        with resp:
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read()
            return body, ctype
    except (IOError, OSError) as e:
        sys.stderr.write(f"Warning: failed to read response from {url}: {e}\n")
        sys.stderr.flush()
        return None, None


def http_download_to_file(url: str, user_agent: str, timeout: int, out_path: str) -> Tuple[bool, int]:
    resp = _open_http_request(url, user_agent, timeout)
    if resp is None:
        return False, 0
    try:
        with resp:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "wb") as f:
                total = 0
                while True:
                    chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
        return True, total
    except (HTTPError, URLError, OSError) as e:
        sys.stderr.write(f"Warning: failed to download {url} to {out_path}: {e}\n")
        sys.stderr.flush()
        return False, 0


def sanitize_filename(name: str) -> str:
    name = RE_NEWLINES_TABS.sub(" ", name)
    name = RE_MULTIPLE_WHITESPACE.sub(" ", name).strip()
    name = RE_NON_FILENAME_CHARS.sub("_", name)
    return name or "untitled"


def derive_output_dir(rss_url: str, override: Optional[str]) -> str:
    if override:
        return override
    parsed = urlparse(rss_url)
    base = parsed.netloc or "feed"
    # include a short hash of the full URL to disambiguate same host feeds
    h = hashlib.sha1(rss_url.encode("utf-8")).hexdigest()[:URL_HASH_LENGTH]
    return f"output_rss_{sanitize_filename(base)}_{h}"


def parse_rss_items(xml_bytes: bytes) -> Tuple[str, List[ET.Element]]:
    # Return (feed_title, item_elements)
    root = ET.fromstring(xml_bytes)
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


def find_transcript_urls(item: ET.Element) -> List[Tuple[str, Optional[str]]]:
    # Returns list of (url, type)
    candidates: List[Tuple[str, Optional[str]]] = []

    # podcast:transcript (Podcasting 2.0)
    for el in item.iter():
        tag = el.tag
        if isinstance(tag, str) and tag.lower().endswith("transcript"):
            url_attr = el.attrib.get("url") or el.attrib.get("href")
            if url_attr:
                t = el.attrib.get("type")
                candidates.append((url_attr.strip(), (t.strip() if t else None)))

    # Some feeds use <transcript> without namespace
    for el in item.findall("transcript"):
        if el.text and el.text.strip():
            candidates.append((el.text.strip(), None))

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


def find_enclosure_media(item: ET.Element) -> Optional[Tuple[str, Optional[str]]]:
    # Look for <enclosure url="..." type="audio/mpeg"/>
    for el in item.iter():
        if isinstance(el.tag, str) and el.tag.lower().endswith("enclosure"):
            url_attr = el.attrib.get("url")
            if url_attr:
                return url_attr.strip(), (el.attrib.get("type") or None)
    return None


def choose_transcript_url(candidates: List[Tuple[str, Optional[str]]], prefer_types: List[str]) -> Optional[Tuple[str, Optional[str]]]:
    if not candidates:
        return None
    if not prefer_types:
        return candidates[0]
    # Try to match preferred types in order (substring match on type or URL extension)
    lowered = [(u, (t.lower() if t else None)) for (u, t) in candidates]
    for pref in prefer_types:
        p = pref.lower().strip()
        for (u, t) in lowered:
            if (t and p in t) or u.lower().endswith(p):
                return (u, t)
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
    
    # Validate screenplay-gap
    if args.screenplay_gap <= 0:
        errors.append(f"--screenplay-gap must be positive, got: {args.screenplay_gap}")
    
    # Validate num-speakers
    if args.num_speakers <= 0:
        errors.append(f"--num-speakers must be positive, got: {args.num_speakers}")
    
    # Validate output-dir (if provided)
    if args.output_dir:
        parent_dir = os.path.dirname(os.path.abspath(args.output_dir)) or os.getcwd()
        if not os.path.exists(parent_dir):
            errors.append(f"--output-dir parent directory does not exist: {parent_dir}")
        elif not os.path.isdir(parent_dir):
            errors.append(f"--output-dir parent path is not a directory: {parent_dir}")
    
    if errors:
        raise ValueError("Invalid input parameters:\n  " + "\n  ".join(errors))


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download podcast episode transcripts from an RSS feed.")
    parser.add_argument("rss", help="Podcast RSS feed URL")
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
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def main(argv: Optional[Iterable[str]] = None) -> int:
    try:
        args = parse_args(argv)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.stderr.flush()
        return 1

    cfg = Config(
        rss_url=args.rss,
        output_dir=derive_output_dir(args.rss, args.output_dir),
        max_episodes=(args.max_episodes if args.max_episodes and args.max_episodes > 0 else None),
        user_agent=args.user_agent,
        timeout=max(MIN_TIMEOUT_SECONDS, args.timeout),
        delay_ms=max(0, args.delay_ms),
        prefer_types=list(args.prefer_type or []),
        transcribe_missing=bool(args.transcribe_missing),
        whisper_model=str(args.whisper_model or "base"),
        screenplay=bool(args.screenplay),
        screenplay_gap_s=float(args.screenplay_gap),
        screenplay_num_speakers=max(MIN_NUM_SPEAKERS, int(args.num_speakers)),
        screenplay_speaker_names=[s.strip() for s in (args.speaker_names or "").split(",") if s.strip()],
        run_id=(str(args.run_id) if args.run_id else None),
    )

    # Resolve effective output directory (with optional run subfolder)
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

    try:
        sys.stderr.write(
            "Starting podcast transcript scrape\n"
            f"  rss: {cfg.rss_url}\n"
            f"  output_dir: {effective_output_dir}\n"
            f"  max_episodes: {cfg.max_episodes or 'all'}\n"
        )
        sys.stderr.flush()
    except (IOError, OSError) as e:
        sys.stderr.write(f"Warning: failed to write startup message: {e}\n")
        sys.stderr.flush()

    rss_bytes, rss_ctype = http_get(cfg.rss_url, cfg.user_agent, cfg.timeout)
    if not rss_bytes:
        sys.stderr.write("Failed to fetch RSS feed.\n")
        return 1

    try:
        feed_title, items = parse_rss_items(rss_bytes)
    except (ET.ParseError, ValueError) as e:
        sys.stderr.write(f"Failed to parse RSS XML: {e}\n")
        return 1

    total_items = len(items)
    if cfg.max_episodes is not None:
        items = items[: cfg.max_episodes]

    sys.stderr.write(f"Episodes to process: {len(items)} of {total_items}\n")
    sys.stderr.flush()

    # Load Whisper model once if transcription is enabled
    whisper_model = None
    if cfg.transcribe_missing:
        try:
            import whisper  # type: ignore
            sys.stderr.write(f"Loading Whisper model ({cfg.whisper_model})...\n")
            sys.stderr.flush()
            whisper_model = whisper.load_model(cfg.whisper_model)
            sys.stderr.write("Whisper model loaded successfully.\n")
            sys.stderr.flush()
        except ImportError:
            sys.stderr.write("Warning: openai-whisper not installed. Install with: pip install openai-whisper && brew install ffmpeg\n")
            sys.stderr.flush()
        except (RuntimeError, OSError) as e:
            sys.stderr.write(f"Warning: failed to load Whisper model: {e}\n")
            sys.stderr.flush()

    # Create temp directory once if transcription is enabled
    temp_dir = None
    if cfg.transcribe_missing:
        temp_dir = os.path.join(effective_output_dir, TEMP_DIR_NAME)
        os.makedirs(temp_dir, exist_ok=True)

    saved = 0
    for idx, item in enumerate(items, start=1):
        title_el = item.find("title") or next((e for e in item.iter() if e.tag.endswith("title")), None)
        ep_title = (title_el.text.strip() if title_el is not None and title_el.text else f"episode_{idx}")
        ep_title_safe = sanitize_filename(ep_title)

        candidates = find_transcript_urls(item)
        chosen = choose_transcript_url(candidates, cfg.prefer_types)

        if not chosen:
            # fallback to enclosure media if requested
            if cfg.transcribe_missing:
                media = find_enclosure_media(item)
                if not media:
                    sys.stderr.write(f"[{idx}] no transcript or enclosure for: {ep_title}\n")
                    sys.stderr.flush()
                    continue
                media_url, media_type = media
                sys.stderr.write(f"[{idx}] no transcript; downloading media for Whisper: {ep_title}\n")
                sys.stderr.flush()

                # derive extension from media type or URL
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

                ep_num_str = f"{idx:0{EPISODE_NUMBER_FORMAT_WIDTH}d}"  # Using EPISODE_NUMBER_FORMAT_WIDTH (4)
                temp_media = os.path.join(temp_dir, f"{ep_num_str}_{ep_title_safe}{ext}")
                dl_start = time.time()
                ok, total_bytes = http_download_to_file(media_url, cfg.user_agent, cfg.timeout, temp_media)
                dl_elapsed = time.time() - dl_start
                if not ok:
                    sys.stderr.write("    failed to download media for transcription\n")
                    sys.stderr.flush()
                    continue
                try:
                    mb = total_bytes / BYTES_PER_MB
                    sys.stderr.write(f"    downloaded {mb:.2f} MB in {dl_elapsed:.1f}s\n")
                    sys.stderr.flush()
                except (ValueError, ZeroDivisionError, TypeError) as e:
                    sys.stderr.write(f"    Warning: failed to format download size: {e}\n")
                    sys.stderr.flush()

                # Use pre-loaded Whisper model
                if whisper_model is None:
                    sys.stderr.write("    Skipping transcription: Whisper model not available\n")
                    sys.stderr.flush()
                    # Clean up temp file
                    try:
                        os.remove(temp_media)
                    except OSError as e:
                        sys.stderr.write(f"    Warning: failed to remove temp media file {temp_media}: {e}\n")
                        sys.stderr.flush()
                    if cfg.delay_ms:
                        time.sleep(cfg.delay_ms / MS_TO_SECONDS)
                    continue

                try:
                    sys.stderr.write(f"    transcribing with Whisper ({cfg.whisper_model})...\n")
                    sys.stderr.flush()
                    # Transcribe in English using pre-loaded model
                    tc_start = time.time()
                    result = whisper_model.transcribe(temp_media, task="transcribe", language="en")
                    tc_elapsed = time.time() - tc_start
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
                            sys.stderr.write(f"    Warning: failed to format as screenplay, using plain transcript: {e}\n")
                            sys.stderr.flush()
                    if not text:
                        raise RuntimeError("empty transcription")
                    # Include run identifier in filename for easy comparison across runs
                    run_tag = f"_{run_suffix}" if run_suffix else ""
                    out_name = f"{idx:0{EPISODE_NUMBER_FORMAT_WIDTH}d} - {ep_title_safe}{run_tag}.txt"
                    out_path = os.path.join(effective_output_dir, out_name)
                    write_file(out_path, text.encode("utf-8"))
                    saved += 1
                    sys.stderr.write(f"    saved transcript: {out_path} (transcribed in {tc_elapsed:.1f}s)\n")
                    sys.stderr.flush()
                except (RuntimeError, OSError) as e:
                    sys.stderr.write(f"    Whisper transcription failed: {e}\n")
                    sys.stderr.flush()
                finally:
                    # best-effort cleanup of temp media
                    try:
                        os.remove(temp_media)
                    except OSError as e:
                        sys.stderr.write(f"    Warning: failed to remove temp media file {temp_media}: {e}\n")
                        sys.stderr.flush()
                if cfg.delay_ms:
                    time.sleep(cfg.delay_ms / 1000.0)
                continue
            else:
                sys.stderr.write(f"[{idx}] no transcript for: {ep_title}\n")
                sys.stderr.flush()
                continue

        t_url, t_type = chosen
        sys.stderr.write(f"[{idx}] downloading transcript: {ep_title} -> {t_url}\n")
        sys.stderr.flush()

        data, ctype = http_get(t_url, cfg.user_agent, cfg.timeout)
        if not data:
            sys.stderr.write(f"    failed to download transcript\n")
            sys.stderr.flush()
            continue

        # decide extension
        ext = ".txt"
        if t_type:
            if "vtt" in t_type.lower():
                ext = ".vtt"
            elif "srt" in t_type.lower():
                ext = ".srt"
            elif "json" in t_type.lower():
                ext = ".json"
            elif "html" in t_type.lower():
                ext = ".html"
        else:
            # infer from content-type header or URL
            if ctype and "vtt" in ctype.lower():
                ext = ".vtt"
            elif ctype and "srt" in ctype.lower():
                ext = ".srt"
            elif ctype and "json" in ctype.lower():
                ext = ".json"
            elif ctype and "html" in ctype.lower():
                ext = ".html"
            else:
                low = t_url.lower()
                for cand in (".vtt", ".srt", ".json", ".html", ".txt"):
                    if low.endswith(cand):
                        ext = cand
                        break

        # Include run identifier in filename for easy comparison across runs
        run_tag = f"_{run_suffix}" if run_suffix else ""
        out_name = f"{idx:0{EPISODE_NUMBER_FORMAT_WIDTH}d} - {ep_title_safe}{run_tag}{ext}"
        out_path = os.path.join(effective_output_dir, out_name)
        try:
            write_file(out_path, data)
            saved += 1
            sys.stderr.write(f"    saved: {out_path}\n")
            sys.stderr.flush()
        except (IOError, OSError) as e:
            sys.stderr.write(f"    failed to write file: {e}\n")
            sys.stderr.flush()

        if cfg.delay_ms:
            time.sleep(cfg.delay_ms / 1000.0)

    # Clean up temp directory if it was created
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            sys.stderr.write(f"Cleaned up temp directory: {temp_dir}\n")
            sys.stderr.flush()
        except OSError as e:
            sys.stderr.write(f"Warning: failed to remove temp directory {temp_dir}: {e}\n")
            sys.stderr.flush()

    sys.stderr.write(f"Done. transcripts_saved={saved} in {effective_output_dir}\n")
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
