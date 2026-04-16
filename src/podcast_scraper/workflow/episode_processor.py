"""Episode-level processing: downloads, transcripts, and Whisper transcription."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

from .. import config, config_constants, models

if TYPE_CHECKING:
    from ..models import Episode, TranscriptionJob
else:
    Episode = models.Episode  # type: ignore[assignment]
    TranscriptionJob = models.TranscriptionJob  # type: ignore[assignment]
from ..exceptions import ProviderError, ProviderRuntimeError
from ..rss import choose_transcript_url, downloader
from ..rss.downloader import OPENAI_MAX_FILE_SIZE_BYTES
from ..transcript_formats import parse_srt, parse_webvtt
from ..utils import filesystem
from ..utils.audio_payload_limits import is_provider_audio_payload_limit_error
from ..utils.corpus_incidents import append_corpus_incident
from ..utils.log_redaction import format_exception_for_log, redact_for_log

logger = logging.getLogger(__name__)

# GitHub #561: stop stepping MP3 bitrate once under this size (headroom below API cap).
_PREPROCESSING_API_REENCODE_TARGET_BYTES = OPENAI_MAX_FILE_SIZE_BYTES - (1024 * 1024)

# GitHub #562: warn at most once if screenplay is requested but the provider has no formatter.
_screenplay_unsupported_warn_lock = threading.Lock()
_screenplay_unsupported_warn_state: dict[str, bool] = {"emitted": False}

# GitHub #562: format_screenplay_from_segments raised — dedupe per process / per run reset.
_screenplay_format_fail_warn_lock = threading.Lock()
_screenplay_format_fail_warn_state: dict[str, bool] = {"emitted": False}


def reset_screenplay_unsupported_provider_warning_for_tests() -> None:
    """Reset #562 episode-level warning gate (unit tests only)."""
    with _screenplay_unsupported_warn_lock:
        _screenplay_unsupported_warn_state["emitted"] = False


def reset_screenplay_format_failure_warning_for_tests() -> None:
    """Reset #562 screenplay format exception warning gate (unit tests only)."""
    with _screenplay_format_fail_warn_lock:
        _screenplay_format_fail_warn_state["emitted"] = False


MS_TO_SECONDS = 1000.0
DEFAULT_MEDIA_EXTENSION = config_constants.DEFAULT_MEDIA_EXTENSION
MEDIA_TYPE_EXTENSION_MAP = {
    "mpeg": ".mp3",
    "mp3": ".mp3",
    "m4a": ".m4a",
    "mp4": ".m4a",
    "aac": ".m4a",
    "ogg": ".ogg",
    "oga": ".ogg",
    "wav": ".wav",
    "webm": ".webm",
}
MEDIA_URL_EXTENSION_FALLBACKS = (".mp3", ".m4a", ".mp4", ".aac", ".ogg", ".wav", ".webm")
TRANSCRIPT_EXTENSION_TOKENS = ("vtt", "srt", "json", "html")
TITLE_HASH_PREFIX_LENGTH = 6


def _job_has_episode_for_metrics(job: Any) -> bool:
    """True when job.episode is a real Episode (not a test Mock with auto-children)."""
    ep = getattr(job, "episode", None)
    return ep is not None and isinstance(ep, Episode)


def _audio_sec_for_transcription_job(
    job: TranscriptionJob,  # type: ignore[valid-type]
) -> Optional[float]:
    """Best-effort episode duration in seconds for per-episode metrics (RSS or job attr)."""
    episode_duration_seconds = getattr(job, "episode_duration_seconds", None)
    if episode_duration_seconds is not None and isinstance(episode_duration_seconds, (int, float)):
        return float(episode_duration_seconds)
    ep = getattr(job, "episode", None)
    if ep is None or not getattr(ep, "item", None):
        return None
    from ..rss.parser import extract_episode_metadata

    _, _, _, duration, _, _ = extract_episode_metadata(ep.item, "")
    if duration is not None and isinstance(duration, (int, float)):
        return float(duration)
    return None


def derive_media_extension(media_type: Optional[str], media_url: str) -> str:
    """Derive file extension for media file based on MIME type or URL.

    Args:
        media_type: MIME type of the media
        media_url: URL of the media file

    Returns:
        File extension with leading dot (e.g., '.mp3')
    """
    ext = DEFAULT_MEDIA_EXTENSION
    if media_type and "/" in media_type:
        ext_guess = media_type.split("/", 1)[1].lower()
        mapped_ext = MEDIA_TYPE_EXTENSION_MAP.get(ext_guess)
        if mapped_ext:
            return mapped_ext
    low = media_url.lower()
    for cand in MEDIA_URL_EXTENSION_FALLBACKS:
        if low.endswith(cand):
            return cand
    return ext


def derive_transcript_extension(
    transcript_type: Optional[str], content_type: Optional[str], transcript_url: str
) -> str:  # noqa: C901 - intentionally verbose heuristics
    """Derive file extension for transcript based on type, content-type, or URL.

    Args:
        transcript_type: Declared transcript type
        content_type: HTTP Content-Type header
        transcript_url: URL of the transcript

    Returns:
        File extension with leading dot (e.g., '.vtt')
    """

    def _match_extension(candidate: Optional[str]) -> Optional[str]:
        if not candidate:
            return None
        low = candidate.lower()

        def _match_path(path: str) -> Optional[str]:
            for token in TRANSCRIPT_EXTENSION_TOKENS:
                if path.endswith(f".{token}"):
                    return f".{token}"
            return None

        if "://" in low or low.startswith("/"):
            parsed = urlparse(low)
            path = parsed.path or ""
            ext = _match_path(path)
            if ext:
                return ext
            filename = os.path.basename(path)
            if filename:
                return _match_path(filename)
            return None

        if "/" in low:
            path = low.split("?", 1)[0].split("#", 1)[0]
            ext = _match_path(path)
            if ext:
                return ext

        if "." in low:
            ext = _match_path(low.split("?", 1)[0].split("#", 1)[0])
            if ext:
                return ext

        if "/" in low:
            subtype = low.split("/", 1)[1]
            subtype = subtype.split(";", 1)[0].strip()
            for token in TRANSCRIPT_EXTENSION_TOKENS:
                if subtype == token or subtype.endswith(f"+{token}"):
                    return f".{token}"
            return None

        return None

    for candidate in (transcript_type, content_type, transcript_url):
        ext = _match_extension(candidate)
        if ext:
            return ext
    return config_constants.DEFAULT_TRANSCRIPT_EXTENSION


def _download_or_reuse_media(
    episode: Episode,  # type: ignore[valid-type]
    cfg: config.Config,
    temp_media: str,
    pipeline_metrics: Any,
) -> tuple[bool, int, float]:
    """Download media or reuse existing file. Returns (success, total_bytes, dl_elapsed)."""
    if episode.media_url is None:
        logger.warning("    media_url is missing; cannot download")
        return False, 0, 0.0
    if pipeline_metrics is not None:
        pipeline_metrics.record_download_media_attempt()
    if cfg.reuse_media and os.path.exists(temp_media):
        try:
            file_size = os.path.getsize(temp_media)
            if file_size > 0:
                logger.debug("    reusing existing media file: %s", temp_media)
                return True, file_size, 0.0
            logger.warning("    media file is empty, re-downloading: %s", temp_media)
        except OSError as exc:
            logger.warning(
                "    error checking media file, re-downloading: %s",
                format_exception_for_log(exc),
            )
        if pipeline_metrics is not None:
            pipeline_metrics.record_download_media_attempt()
    dl_start = time.time()
    ok, total_bytes = downloader.http_download_to_file(
        episode.media_url, cfg.user_agent, cfg.timeout, temp_media
    )
    dl_elapsed = time.time() - dl_start
    if not ok:
        logger.warning("    failed to download media")
        return False, 0, 0.0
    if downloader.should_log_download_summary():
        try:
            mb = total_bytes / downloader.BYTES_PER_MB
            logger.info("    downloaded %.2f MB in %.1fs", mb, dl_elapsed)
        except (ValueError, ZeroDivisionError, TypeError):
            pass
    return True, total_bytes, dl_elapsed


def transcript_txt_missing_segments(full_txt_path: str) -> bool:
    """Return True if *full_txt_path* is an existing ``.txt`` with no sibling ``.segments.json``.

    Whisper-style outputs use a sidecar for GI quote audio timestamps. When only the ``.txt``
    exists (for example after an older ``--skip-existing`` run), GI timing stays at zero until
    segments exist (GitHub #542).
    """
    if not full_txt_path.endswith(".txt"):
        return False
    if not os.path.isfile(full_txt_path):
        return False
    seg_path = os.path.splitext(full_txt_path)[0] + ".segments.json"
    return not os.path.isfile(seg_path)


def _should_retranscribe_for_gi_segments(cfg: config.Config, whisper_txt_path: str) -> bool:
    """Whether to bypass skip/reuse so transcription can populate ``.segments.json`` for GI."""
    if not cfg.backfill_transcript_segments:
        return False
    if not cfg.generate_gi:
        return False
    return transcript_txt_missing_segments(whisper_txt_path)


def download_media_for_transcription(
    episode: Episode,  # type: ignore[valid-type]
    cfg: config.Config,
    temp_dir: str,
    effective_output_dir: str,
    run_suffix: Optional[str],
    detected_speaker_names: Optional[List[str]] = None,
    pipeline_metrics=None,
) -> Optional[TranscriptionJob]:  # type: ignore[valid-type]
    """Download media file for Whisper transcription.

    Args:
        episode: Episode object with metadata
        cfg: Configuration object
        temp_dir: Temporary directory for downloads
        effective_output_dir: Output directory path
        run_suffix: Optional suffix for output filename

    Returns:
        TranscriptionJob object or None if skipped/failed
    """
    final_out_path = filesystem.build_whisper_output_path(
        episode.idx, episode.title_safe, run_suffix, effective_output_dir
    )
    if cfg.skip_existing and os.path.exists(final_out_path):
        if _should_retranscribe_for_gi_segments(cfg, final_out_path):
            logger.info(
                "[%s] Transcript exists without .segments.json; will re-transcribe to populate "
                "sidecar for GI quote timestamps and segment-backed speaker_id when segments "
                "carry speaker labels (backfill_transcript_segments + generate_gi): %s",
                episode.idx,
                final_out_path,
            )
            # Fall through: do not return — schedule download/transcribe to populate sidecar (#542).
        # If generate_summaries is enabled, still return a job so transcript path can be used
        # for summarization (even though we won't re-transcribe)
        elif cfg.generate_summaries:
            logger.debug(
                "[%s] Transcript exists, but will use for summarization: %s",
                episode.idx,
                final_out_path,
            )
            # Return a job with empty temp_media since we won't download/transcribe
            # CRITICAL: Create a copy of detected_speaker_names to prevent shared mutable state
            # This prevents speaker names from one episode leaking to another
            speaker_names_copy = list(detected_speaker_names) if detected_speaker_names else None
            return TranscriptionJob(  # type: ignore[no-any-return]
                idx=episode.idx,
                ep_title=episode.title,
                ep_title_safe=episode.title_safe,
                temp_media="",  # Empty since we're reusing existing transcript
                detected_speaker_names=speaker_names_copy,
                episode=episode,
            )
        else:
            prefix = "[dry-run] " if cfg.dry_run else ""
            logger.info(
                "[%s] %stranscript already exists; skipping (--skip-existing): %s",
                episode.idx,
                prefix,
                final_out_path,
            )
            return None

    if not episode.media_url:
        logger.debug("[%s] Episode missing media_url; cannot schedule transcription", episode.idx)
        logger.info(f"[{episode.idx}] no transcript or enclosure for: {episode.title}")
        return None

    display_title = filesystem.truncate_whisper_title(episode.title, for_log=True)
    if cfg.dry_run:
        logger.info(
            "[%s] (dry-run) would download media for Whisper: %s -> %s",
            episode.idx,
            display_title,
            episode.media_url,
        )
        logger.info(f"    [dry-run] Whisper output would be: {final_out_path}")
        return TranscriptionJob(  # type: ignore[no-any-return,valid-type]
            idx=episode.idx,
            ep_title=episode.title,
            ep_title_safe=episode.title_safe,
            temp_media="",
            episode=episode,
        )
    else:
        logger.info(f"[{episode.idx}] no transcript; downloading media: {display_title}")

    ext = derive_media_extension(episode.media_type, episode.media_url)
    ep_num_str = f"{episode.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d}"
    short_title = filesystem.truncate_whisper_title(episode.title_safe, for_log=False)
    title_hash_input = f"{episode.media_url}|{episode.idx}|{cfg.rss_url}"
    # Hash is only for stable filenames (not security sensitive)
    title_hash = hashlib.sha1(title_hash_input.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :TITLE_HASH_PREFIX_LENGTH
    ]
    temp_media = os.path.join(temp_dir, f"{ep_num_str}_{short_title}_{title_hash}{ext}")

    ok, _total_bytes, dl_elapsed = _download_or_reuse_media(
        episode, cfg, temp_media, pipeline_metrics
    )
    if not ok:
        return None

    # CRITICAL: Create a copy of detected_speaker_names to prevent shared mutable state
    # This prevents speaker names from one episode leaking to another
    speaker_names_copy = list(detected_speaker_names) if detected_speaker_names else None
    return TranscriptionJob(  # type: ignore[no-any-return,valid-type]
        idx=episode.idx,
        ep_title=episode.title,
        ep_title_safe=episode.title_safe,
        temp_media=temp_media,
        detected_speaker_names=speaker_names_copy,
        episode=episode,
        media_download_elapsed=dl_elapsed,
    )


def _format_transcript_if_needed(
    result: dict,
    cfg: config.Config,
    detected_speaker_names: Optional[List[str]],
    transcription_provider=None,
) -> str:
    """Format transcript as screenplay if configured.

    Args:
        result: Transcription result dictionary
        cfg: Configuration object
        detected_speaker_names: List of detected speaker names
        transcription_provider: Optional TranscriptionProvider instance for formatting

    Returns:
        Formatted transcript text (screenplay or plain)
    """
    text = (result.get("text") or "").strip()
    if cfg.screenplay and isinstance(result, dict) and isinstance(result.get("segments"), list):
        # Use detected speaker names (manual names are already used as fallback in workflow)
        speaker_names = detected_speaker_names or []
        try:
            # Use provider's format_screenplay_from_segments if available (Whisper provider)
            if transcription_provider and hasattr(
                transcription_provider, "format_screenplay_from_segments"
            ):
                formatted = transcription_provider.format_screenplay_from_segments(
                    result["segments"],
                    cfg.screenplay_num_speakers,
                    speaker_names,
                    cfg.screenplay_gap_s,
                )
            else:
                # Fallback: log at most once per process (GitHub #562; config also coerces).
                with _screenplay_unsupported_warn_lock:
                    if not _screenplay_unsupported_warn_state["emitted"]:
                        _screenplay_unsupported_warn_state["emitted"] = True
                        logger.warning(
                            "Screenplay formatting requested but provider doesn't support it. "
                            "Using plain transcript (GitHub #562; see CONFIGURATION.md)."
                        )
                formatted = None
            if formatted and formatted.strip():
                text = formatted
        except (ValueError, KeyError, TypeError) as exc:
            with _screenplay_format_fail_warn_lock:
                if not _screenplay_format_fail_warn_state["emitted"]:
                    _screenplay_format_fail_warn_state["emitted"] = True
                    logger.warning(
                        "    failed to format as screenplay, using plain transcript: %s "
                        "(GitHub #562: repeats suppressed until pipeline gate reset)",
                        format_exception_for_log(exc),
                    )
                else:
                    logger.debug(
                        "screenplay format failure suppressed (repeat; GitHub #562): %s",
                        format_exception_for_log(exc),
                    )
    return text


def _save_transcript_file(
    text: str,
    job: TranscriptionJob,  # type: ignore[valid-type]
    run_suffix: Optional[str],
    effective_output_dir: str,
    pipeline_metrics=None,
) -> str:
    """Save transcript text to file.

    Args:
        text: Transcript text to save
        job: TranscriptionJob object
        run_suffix: Optional run suffix
        effective_output_dir: Output directory path

    Returns:
        Relative path to saved transcript file

    Raises:
        RuntimeError: If text is empty
        OSError: If file writing fails
    """
    if not text:
        raise RuntimeError("empty transcription")
    out_path = filesystem.build_whisper_output_path(
        job.idx, job.ep_title_safe, run_suffix, effective_output_dir
    )
    # write_file() now logs detailed I/O metrics: file path, bytes, elapsed time
    filesystem.write_file(out_path, text.encode("utf-8"), pipeline_metrics=pipeline_metrics)
    rel_path = os.path.relpath(out_path, effective_output_dir)
    return rel_path


def _save_transcript_segments_file(
    segments: List[Dict[str, Any]],
    rel_transcript_path: str,
    effective_output_dir: str,
) -> None:
    """Save transcription segments to a .segments.json file for GIL timestamp mapping.

    When transcription returns segments (start/end in seconds, text), persist them
    so the GIL pipeline can attach precise timestamp_start_ms/timestamp_end_ms to
    quotes (FR2.2). File is written next to the transcript (same base name, .segments.json).

    Args:
        segments: List of {"start": float, "end": float, "text": str}.
        rel_transcript_path: Relative path to the transcript file (e.g. transcripts/01 - ep.txt).
        effective_output_dir: Output directory path.
    """
    if not segments or not rel_transcript_path:
        return
    full_path = os.path.join(effective_output_dir, rel_transcript_path)
    base, _ = os.path.splitext(full_path)
    segments_path = base + ".segments.json"
    try:
        with open(segments_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, indent=0, allow_nan=False)
        logger.debug("Saved transcription segments for GIL timestamps: %s", segments_path)
    except OSError as e:
        logger.debug("Could not save segments file %s: %s", segments_path, e)


def _cleanup_temp_media(temp_media: str, cfg: Optional[config.Config] = None) -> None:
    """Clean up temporary media file.

    Args:
        temp_media: Path to temporary media file
        cfg: Configuration object (optional, for reuse_media check)
    """
    # Skip cleanup if reuse_media is enabled
    if cfg and cfg.reuse_media:
        logger.debug(f"    keeping media file for reuse: {temp_media}")
        return

    try:
        os.remove(temp_media)
    except OSError as exc:
        logger.debug(f"    failed to remove temp media file {temp_media}: {exc}")


def _check_and_reuse_existing_transcript(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    run_suffix: Optional[str],
    effective_output_dir: str,
    pipeline_metrics=None,
) -> Optional[tuple[bool, Optional[str], int]]:
    """Check if existing transcript can be reused and return if found.

    Args:
        job: TranscriptionJob with episode info
        cfg: Configuration object
        run_suffix: Optional suffix for output filename
        effective_output_dir: Output directory path
        pipeline_metrics: Optional metrics object

    Returns:
        Tuple of (success, rel_path, bytes_downloaded) if transcript exists, None otherwise
    """
    final_out_path = filesystem.build_whisper_output_path(
        job.idx, job.ep_title_safe, run_suffix, effective_output_dir
    )

    # If temp_media is empty and transcript exists, we're reusing existing transcript
    # (happens when skip_existing=True and generate_summaries=True)
    if not job.temp_media and cfg.skip_existing and os.path.exists(final_out_path):
        if _should_retranscribe_for_gi_segments(cfg, final_out_path):
            return None
        rel_path = os.path.relpath(final_out_path, effective_output_dir)
        logger.debug(
            "[%s] Reusing existing Whisper transcript for summarization: %s",
            job.idx,
            rel_path,
        )
        # Update episode status: transcribed (reused existing) (Issue #391)
        if pipeline_metrics is not None and _job_has_episode_for_metrics(job):
            from podcast_scraper.workflow.helpers import get_episode_id_from_episode

            assert job.episode is not None
            episode_id, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
            pipeline_metrics.update_episode_status(episode_id=episode_id, stage="transcribed")
        return True, rel_path, 0
    return None


def _check_transcript_cache(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    temp_media: str,
    run_suffix: Optional[str],
    effective_output_dir: str,
    pipeline_metrics=None,
) -> Optional[tuple[bool, Optional[str], int]]:
    """Check transcript cache and return cached transcript if found.

    Args:
        job: TranscriptionJob with episode info
        cfg: Configuration object
        temp_media: Path to temporary media file
        run_suffix: Optional suffix for output filename
        effective_output_dir: Output directory path
        pipeline_metrics: Optional metrics object

    Returns:
        Tuple of (success, rel_path, bytes_downloaded) if cache hit, None otherwise
    """
    if not (cfg.transcript_cache_enabled and temp_media and os.path.exists(temp_media)):
        return None

    from podcast_scraper.cache import transcript_cache

    cache_dir = cfg.transcript_cache_dir or transcript_cache.TRANSCRIPT_CACHE_DIR
    audio_hash = transcript_cache.get_audio_hash(temp_media)
    cached_entry = transcript_cache.get_cached_transcript_entry(audio_hash, cache_dir)
    if cached_entry:
        cached_transcript, cached_segments = cached_entry
        # Save cached transcript to output file
        rel_path = _save_transcript_file(
            cached_transcript,
            job,
            run_suffix,
            effective_output_dir,
            pipeline_metrics=pipeline_metrics,
        )
        if isinstance(cached_segments, list) and len(cached_segments) > 0:
            _save_transcript_segments_file(cached_segments, rel_path, effective_output_dir)
        logger.info(
            "[%s] Transcript cache hit: global cache entry audio_hash=%s "
            "(no API transcribe; same file bytes can repeat across feeds in multi-feed) -> %s",
            job.idx,
            audio_hash,
            rel_path,
        )
        # Update episode status: transcribed (from cache)
        if pipeline_metrics is not None and _job_has_episode_for_metrics(job):
            from podcast_scraper.workflow.helpers import get_episode_id_from_episode

            assert job.episode is not None
            episode_id, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
            pipeline_metrics.update_episode_status(episode_id=episode_id, stage="transcribed")
            # Per-episode metrics: no provider transcription time (cache hit); duration from RSS/job
            audio_sec = _audio_sec_for_transcription_job(job)
            pipeline_metrics.update_episode_metrics(
                episode_id=episode_id,
                audio_sec=audio_sec,
                transcribe_sec=0.0,
            )
        # Audio was downloaded only to hash into transcript cache; treat as 0 for metrics/UI
        if pipeline_metrics is not None:
            pipeline_metrics.record_download_media_time(0.0, job.idx)
        _cleanup_temp_media(temp_media, cfg)
        bytes_downloaded = 0
        if os.path.exists(temp_media):
            try:
                bytes_downloaded = os.path.getsize(temp_media)
            except OSError:
                pass
        return True, rel_path, bytes_downloaded
    return None


def _preprocessing_probe_preprocessed_cache(
    cfg: config.Config,
    temp_media: str,
    cache_dir: str,
    cache_probe_bitrates: List[int],
    transcription_provider: str,
) -> Tuple[Optional[str], str, float]:
    """Return cached preprocessed path, cache key, and cache check duration (GitHub #561)."""
    from podcast_scraper.preprocessing.audio import cache as preprocessing_cache
    from podcast_scraper.preprocessing.audio.factory import build_ffmpeg_preprocessor_with_bitrate

    cache_check_start = time.time()
    cached_path: Optional[str] = None
    cache_key = ""
    for kb in cache_probe_bitrates:
        probe_pre = build_ffmpeg_preprocessor_with_bitrate(cfg, kb)
        ck = probe_pre.get_cache_key(temp_media)
        hit = preprocessing_cache.get_cached_audio_path(ck, cache_dir)
        if not hit:
            continue
        if transcription_provider in ("openai", "gemini"):
            try:
                hit_sz = os.path.getsize(hit)
            except OSError:
                hit_sz = 0
            # Shared upload-style cap (constant name says OpenAI; same check for Gemini).
            if hit_sz > OPENAI_MAX_FILE_SIZE_BYTES:
                continue
        cached_path = hit
        cache_key = ck
        break
    return cached_path, cache_key, time.time() - cache_check_start


def _preprocessing_reencode_mp3_until_target(
    job_idx: int,
    audio_preprocessor: Any,
    temp_media: str,
    preprocessed_path: str,
    transcription_provider: str,
    preprocess_elapsed: float,
) -> Tuple[str, int, float]:
    """GitHub #561 phase 2: step MP3 bitrate down until under target size or floor."""
    from podcast_scraper.preprocessing.audio.factory import next_lower_mp3_bitrate_kbps

    working_path = preprocessed_path
    final_kbps = int(audio_preprocessor.mp3_bitrate_kbps)
    total_preprocess_elapsed = float(preprocess_elapsed)
    if transcription_provider not in ("openai", "gemini"):
        return working_path, final_kbps, total_preprocess_elapsed

    while True:
        try:
            sz_now = os.path.getsize(working_path)
        except OSError:
            break
        if sz_now <= _PREPROCESSING_API_REENCODE_TARGET_BYTES:
            break
        nxt = next_lower_mp3_bitrate_kbps(final_kbps)
        if nxt is None:
            break
        out_next = f"{temp_media}.re_encode.{nxt}.mp3"
        ok_re, step_elapsed = audio_preprocessor.reencode_mp3_to_bitrate(
            working_path, out_next, nxt
        )
        total_preprocess_elapsed += float(step_elapsed)
        if not ok_re or not os.path.exists(out_next):
            break
        if working_path != preprocessed_path and os.path.abspath(working_path) != os.path.abspath(
            temp_media
        ):
            try:
                os.remove(working_path)
            except OSError:
                pass
        working_path = out_next
        final_kbps = int(nxt)
        logger.info(
            "[%s] Preprocess re-encode (GitHub #561): %d kbps MP3, %.2fs",
            job_idx,
            final_kbps,
            step_elapsed,
        )
    if working_path != preprocessed_path and os.path.exists(preprocessed_path):
        try:
            os.remove(preprocessed_path)
        except OSError:
            pass
    return working_path, final_kbps, total_preprocess_elapsed


def _preprocess_audio_if_needed(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    temp_media: str,
    pipeline_metrics=None,
) -> str:
    """Preprocess audio if enabled and return path to audio file for transcription.

    Args:
        job: TranscriptionJob with episode info
        cfg: Configuration object
        temp_media: Path to temporary media file
        pipeline_metrics: Optional metrics object

    Returns:
        Path to audio file to use for transcription (preprocessed or original)
    """
    media_for_transcription = temp_media
    if not (cfg.preprocessing_enabled and temp_media and os.path.exists(temp_media)):
        return media_for_transcription

    from podcast_scraper.preprocessing.audio import cache as preprocessing_cache
    from podcast_scraper.preprocessing.audio.factory import (
        build_ffmpeg_preprocessor_with_bitrate,
        create_audio_preprocessor,
        mp3_bitrates_to_probe_for_cache,
        resolve_preprocessing_mp3_bitrate_kbps,
    )

    # Log before preprocessing
    try:
        original_size = os.path.getsize(temp_media)
        original_size_mb = original_size / (1024 * 1024)
        logger.debug(
            "[%s] Audio preprocessing: starting with original file size: %.2f MB",
            job.idx,
            original_size_mb,
        )
    except OSError:
        original_size = 0
        logger.debug("[%s] Audio preprocessing: starting (size unknown)", job.idx)

    audio_preprocessor = create_audio_preprocessor(cfg)
    if not audio_preprocessor:
        return media_for_transcription

    # Record preprocessing attempt (regardless of cache hit/miss)
    if pipeline_metrics is not None:
        pipeline_metrics.record_preprocessing_attempt()

    cache_dir = cfg.preprocessing_cache_dir or preprocessing_cache.PREPROCESSING_CACHE_DIR
    transcription_provider = str(cfg.transcription_provider or "").lower()
    first_pass_kbps = resolve_preprocessing_mp3_bitrate_kbps(cfg)
    if transcription_provider in ("openai", "gemini"):
        cache_probe_bitrates = mp3_bitrates_to_probe_for_cache(first_pass_kbps)
    else:
        cache_probe_bitrates = [first_pass_kbps]

    # Track wall time for preprocessing (Issue #387)
    preprocessing_wall_start = time.time()

    # Extract audio metadata from original file (Issue #387)
    from podcast_scraper.preprocessing.audio.ffmpeg_processor import extract_audio_metadata

    audio_metadata = extract_audio_metadata(temp_media)
    if audio_metadata and pipeline_metrics is not None:
        pipeline_metrics.record_preprocessing_audio_metadata(
            bitrate=audio_metadata.get("bitrate"),
            sample_rate=audio_metadata.get("sample_rate"),
            codec=audio_metadata.get("codec"),
            channels=audio_metadata.get("channels"),
        )

    # Check cache first (GitHub #561: probe lower bitrates for API transcription)
    cached_path, cache_key, cache_check_elapsed = _preprocessing_probe_preprocessed_cache(
        cfg,
        temp_media,
        cache_dir,
        cache_probe_bitrates,
        transcription_provider,
    )

    if cached_path:
        logger.debug(
            "[%s] Audio preprocessing: cache hit, using cached preprocessed audio: %s",
            job.idx,
            cache_key,
        )
        media_for_transcription = cached_path
        preprocessing_wall_elapsed = time.time() - preprocessing_wall_start

        # Record cache hit metrics (Issue #387)
        if pipeline_metrics is not None:
            pipeline_metrics.record_preprocessing_cache_hit()
            pipeline_metrics.record_preprocessing_time(cache_check_elapsed)
            pipeline_metrics.record_preprocessing_wall_time(preprocessing_wall_elapsed)
            pipeline_metrics.record_preprocessing_cache_hit_time(preprocessing_wall_elapsed)
            pipeline_metrics.record_preprocessing_cache_hit_flag(True)
            try:
                cached_size = os.path.getsize(cached_path)
                cached_size_mb = cached_size / (1024 * 1024)
                reduction = (1 - cached_size / original_size) * 100 if original_size > 0 else 0.0
                logger.debug(
                    "[%s] Audio preprocessing: cached file size: %.2f MB "
                    "(%.1f%% reduction from original)",
                    job.idx,
                    cached_size_mb,
                    reduction,
                )
                # Record metrics for cached file
                pipeline_metrics.record_preprocessing_size_reduction(original_size, cached_size)
            except OSError:
                pass
    else:
        # Record cache miss
        if pipeline_metrics is not None:
            pipeline_metrics.record_preprocessing_cache_miss()
        logger.debug("[%s] Audio preprocessing: cache miss, preprocessing audio file", job.idx)

        # Preprocess audio
        preprocessed_path = f"{temp_media}.preprocessed.mp3"
        success, preprocess_elapsed = audio_preprocessor.preprocess(temp_media, preprocessed_path)

        preprocessing_wall_elapsed = time.time() - preprocessing_wall_start

        if success and os.path.exists(preprocessed_path):
            working_path, final_kbps, total_preprocess_elapsed = (
                _preprocessing_reencode_mp3_until_target(
                    job.idx,
                    audio_preprocessor,
                    temp_media,
                    preprocessed_path,
                    transcription_provider,
                    preprocess_elapsed,
                )
            )

            cache_save_pre = build_ffmpeg_preprocessor_with_bitrate(cfg, final_kbps)
            cache_key = cache_save_pre.get_cache_key(temp_media)
            cached_path = preprocessing_cache.save_to_cache(working_path, cache_key, cache_dir)
            media_for_transcription = cached_path
            if os.path.abspath(working_path) != os.path.abspath(cached_path):
                try:
                    os.remove(working_path)
                except OSError:
                    pass

            # Log after preprocessing with metrics
            try:
                preprocessed_size = os.path.getsize(cached_path)
                preprocessed_size_mb = preprocessed_size / (1024 * 1024)
                reduction = (
                    (1 - preprocessed_size / original_size) * 100 if original_size > 0 else 0.0
                )
                logger.debug(
                    "[%s] Audio preprocessing: completed in %.2fs, "
                    "preprocessed file size: %.2f MB (%.1f%% reduction from %.2f MB)",
                    job.idx,
                    total_preprocess_elapsed,
                    preprocessed_size_mb,
                    reduction,
                    original_size_mb,
                )
                logger.info(
                    "[%s] Preprocessed audio: %.1f%% smaller " "(%.1fMB -> %.1fMB) in %.1fs",
                    job.idx,
                    reduction,
                    original_size_mb,
                    preprocessed_size_mb,
                    total_preprocess_elapsed,
                )

                # Record metrics (Issue #387)
                if pipeline_metrics is not None:
                    pipeline_metrics.record_preprocessing_time(total_preprocess_elapsed)
                    pipeline_metrics.record_preprocessing_wall_time(preprocessing_wall_elapsed)
                    pipeline_metrics.record_preprocessing_cache_miss_time(
                        preprocessing_wall_elapsed
                    )
                    pipeline_metrics.record_preprocessing_cache_hit_flag(False)
                    pipeline_metrics.record_preprocessing_size_reduction(
                        original_size, preprocessed_size
                    )
            except OSError:
                logger.debug(
                    "[%s] Audio preprocessing: completed in %.2fs (size unknown)",
                    job.idx,
                    total_preprocess_elapsed,
                )
                # Still record time even if size is unknown (Issue #387)
                if pipeline_metrics is not None:
                    pipeline_metrics.record_preprocessing_time(total_preprocess_elapsed)
                    pipeline_metrics.record_preprocessing_wall_time(preprocessing_wall_elapsed)
                    pipeline_metrics.record_preprocessing_cache_miss_time(
                        preprocessing_wall_elapsed
                    )
                    pipeline_metrics.record_preprocessing_cache_hit_flag(False)
        else:
            # Preprocessing failed, use original audio
            logger.warning("[%s] Audio preprocessing failed, using original audio", job.idx)
            _append_preprocessing_incident(
                cfg,
                job,
                message=(
                    "Audio preprocessing failed (ffmpeg); using original audio for transcription "
                    "(GitHub #558)"
                ),
            )
            media_for_transcription = temp_media
            # Still record wall time even on failure (Issue #387)
            if pipeline_metrics is not None:
                pipeline_metrics.record_preprocessing_wall_time(preprocessing_wall_elapsed)
                pipeline_metrics.record_preprocessing_cache_hit_flag(False)
            # Clean up failed preprocessed file if it exists
            if os.path.exists(preprocessed_path):
                try:
                    os.remove(preprocessed_path)
                except OSError:
                    pass

    return media_for_transcription


def _get_provider_model_name(transcription_provider: Any, cfg: config.Config) -> Optional[str]:
    """Extract model name from transcription provider for cache metadata.

    Args:
        transcription_provider: Transcription provider instance
        cfg: Configuration object

    Returns:
        Model name string or None
    """
    if not transcription_provider:
        return None

    # Try to get model name from provider
    # Note: MLProvider.model returns Whisper object (not JSON serializable),
    # so we need to get model name from config or provider attributes
    if hasattr(transcription_provider, "model"):
        model = getattr(transcription_provider, "model", None)
        # If model is not a string (e.g., Whisper model object), get name from config
        if model is not None and not isinstance(model, str):
            # Get model name from config based on provider type
            if cfg.transcription_provider == "whisper":
                return cfg.whisper_model
            elif cfg.transcription_provider == "openai":
                return getattr(cfg, "openai_transcription_model", "whisper-1")
            elif cfg.transcription_provider == "gemini":
                return getattr(cfg, "gemini_transcription_model", "gemini-2.5-flash-lite")
            elif cfg.transcription_provider == "mistral":
                return getattr(cfg, "mistral_transcription_model", None)
            elif cfg.transcription_provider == "anthropic":
                return getattr(cfg, "anthropic_transcription_model", None)
            elif cfg.transcription_provider == "deepseek":
                return getattr(cfg, "deepseek_transcription_model", None)
            elif cfg.transcription_provider == "grok":
                return getattr(cfg, "grok_transcription_model", None)
            elif cfg.transcription_provider == "ollama":
                return getattr(cfg, "ollama_transcription_model", None)
            else:
                # Fallback: try to get transcription_model attribute from provider
                return getattr(transcription_provider, "transcription_model", None)
    # If provider has transcription_model attribute (like OpenAIProvider), prefer that
    elif hasattr(transcription_provider, "transcription_model"):
        return getattr(transcription_provider, "transcription_model", None)

    return None


def _save_transcript_to_cache_if_needed(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    temp_media: str,
    text: str,
    transcription_provider: Any,
    segments: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save transcript to cache if caching is enabled.

    Args:
        job: TranscriptionJob with episode info
        cfg: Configuration object
        temp_media: Path to temporary media file
        text: Transcribed text
        transcription_provider: Transcription provider instance
        segments: Optional provider segments for GI ``.segments.json`` parity on cache hit
    """
    if not (cfg.transcript_cache_enabled and temp_media and os.path.exists(temp_media)):
        return

    from podcast_scraper.cache import transcript_cache

    cache_dir = cfg.transcript_cache_dir or transcript_cache.TRANSCRIPT_CACHE_DIR
    audio_hash = transcript_cache.get_audio_hash(temp_media)
    # Get provider name and model for metadata
    provider_name = None
    if transcription_provider:
        provider_name = (
            getattr(transcription_provider, "name", None)
            or type(transcription_provider).__name__.replace("Provider", "").lower()
        )
    model = _get_provider_model_name(transcription_provider, cfg)
    try:
        transcript_cache.save_transcript_to_cache(
            audio_hash,
            text,
            provider_name=provider_name,
            model=model,
            cache_dir=cache_dir,
            segments=segments,
        )
        logger.debug("[%s] Saved transcript to cache (hash=%s)", job.idx, audio_hash)
    except Exception as exc:
        # Cache save failure is non-fatal - log and continue
        logger.warning(
            "[%s] Failed to save transcript to cache: %s",
            job.idx,
            format_exception_for_log(exc),
        )


def _record_transcription_metrics(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    tc_elapsed: float,
    call_metrics: Any,
    pipeline_metrics=None,
) -> None:
    """Record transcription metrics after successful transcription.

    Args:
        job: TranscriptionJob with episode info
        cfg: Configuration object
        tc_elapsed: Transcription elapsed time in seconds
        call_metrics: Provider call metrics
        pipeline_metrics: Optional metrics object
    """
    if pipeline_metrics is None:
        return

    pipeline_metrics.record_transcribe_time(tc_elapsed, job.idx)
    # Update episode status: transcribed (Issue #391)
    if _job_has_episode_for_metrics(job):
        from podcast_scraper.workflow.helpers import get_episode_id_from_episode
        from podcast_scraper.workflow.orchestration import _log_episode_metrics

        assert job.episode is not None
        episode_id, episode_number = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
        pipeline_metrics.update_episode_status(episode_id=episode_id, stage="transcribed")

        # Log standardized per-episode metrics after transcription
        audio_sec = _audio_sec_for_transcription_job(job)
        _log_episode_metrics(
            episode_id=episode_id,
            episode_number=episode_number,
            pipeline_metrics=pipeline_metrics,
            cfg=cfg,
            audio_sec=audio_sec,
            transcribe_sec=tc_elapsed,
            retries=call_metrics.retries,
            rate_limit_sleep_sec=call_metrics.rate_limit_sleep_sec,
            prompt_tokens=call_metrics.prompt_tokens,
            completion_tokens=call_metrics.completion_tokens,
            estimated_cost=call_metrics.estimated_cost,
        )


def _episode_id_and_idx_for_incident(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
) -> tuple[Optional[str], int]:
    if job.episode is None:
        return None, int(job.idx)
    from podcast_scraper.workflow.helpers import get_episode_id_from_episode

    episode_id, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
    return episode_id, int(job.idx)


def _append_transcription_incident(
    cfg: config.Config,
    job: TranscriptionJob,  # type: ignore[valid-type]
    *,
    category: str,
    message: str,
    exception_type: str,
    stage: str = "transcription",
) -> None:
    path = getattr(cfg, "incident_log_path", None)
    if not path:
        return
    episode_id, episode_idx = _episode_id_and_idx_for_incident(job, cfg)
    append_corpus_incident(
        path,
        scope="episode",
        category=category,  # type: ignore[arg-type]
        message=message,
        exception_type=exception_type,
        stage=stage,
        feed_url=cfg.rss_url,
        episode_id=episode_id,
        episode_idx=episode_idx,
    )


def _append_preprocessing_incident(
    cfg: config.Config,
    job: TranscriptionJob,  # type: ignore[valid-type]
    *,
    message: str,
    exception_type: str = "PreprocessFailed",
) -> None:
    """Append episode-scoped row when preprocessing fails and we fall back (GitHub #558)."""
    path = getattr(cfg, "incident_log_path", None)
    if not path:
        return
    episode_id, episode_idx = _episode_id_and_idx_for_incident(job, cfg)
    append_corpus_incident(
        path,
        scope="episode",
        category="policy",
        message=message,
        exception_type=exception_type,
        stage="preprocessing",
        feed_url=cfg.rss_url,
        episode_id=episode_id,
        episode_idx=episode_idx,
    )


def _mark_episode_skipped_policy(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    pipeline_metrics: Any,
    reason: str,
) -> None:
    if pipeline_metrics is None or job.episode is None:
        return
    from podcast_scraper.workflow.helpers import get_episode_id_from_episode, update_metric_safely

    episode_id, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
    pipeline_metrics.update_episode_status(
        episode_id=episode_id,
        status="skipped",
        stage="transcription",
        error_type="PolicySkip",
        error_message=redact_for_log(reason, max_len=500),
    )
    update_metric_safely(pipeline_metrics, "episodes_skipped_total", 1)


def transcribe_media_to_text(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    whisper_model,
    run_suffix: Optional[str],
    effective_output_dir: str,
    transcription_provider=None,  # Stage 2: Optional TranscriptionProvider
    pipeline_metrics=None,
) -> tuple[bool, Optional[str], int]:
    """Transcribe media file using Whisper and save result.

    Args:
        job: TranscriptionJob with media file path
        cfg: Configuration object
        whisper_model: Loaded Whisper model (for backward compatibility)
        run_suffix: Optional suffix for output filename
        effective_output_dir: Output directory path
        transcription_provider: Optional TranscriptionProvider instance (Stage 2)
        pipeline_metrics: Optional metrics object

    Returns:
        Tuple of (success: bool, transcript_file_path: Optional[str], bytes_downloaded: int)
        transcript_file_path is relative to effective_output_dir
        bytes_downloaded is the size of the media file downloaded (if any)
    """
    if cfg.dry_run:
        final_path = filesystem.build_whisper_output_path(
            job.idx, job.ep_title_safe, run_suffix, effective_output_dir
        )
        logger.info(f"[{job.idx}] (dry-run) would transcribe media -> {final_path}")
        rel_path = os.path.relpath(final_path, effective_output_dir)
        return True, rel_path, 0

    temp_media = job.temp_media

    # Check if existing transcript can be reused
    reuse_result = _check_and_reuse_existing_transcript(
        job, cfg, run_suffix, effective_output_dir, pipeline_metrics
    )
    if reuse_result:
        return reuse_result

    # Transcript cache before requiring a provider (cache hit skips API and keeps download at 0).
    cache_result = _check_transcript_cache(
        job, cfg, temp_media, run_suffix, effective_output_dir, pipeline_metrics
    )
    if cache_result:
        return cache_result

    # Record media download wall time only after a cache miss (avoids attributing full HTTP
    # time when the transcript is served from cache).
    if pipeline_metrics is not None and getattr(job, "media_download_elapsed", None) is not None:
        pipeline_metrics.record_download_media_time(job.media_download_elapsed, job.idx)

    # Log detected speaker names (hosts + guests) before transcription
    # IMPORTANT: Log episode idx to catch speaker name leaks between episodes
    if job.detected_speaker_names:
        speaker_names_display = ", ".join(job.detected_speaker_names)
        logger.debug(
            "[%s] Speaker names for transcription: %s",
            job.idx,
            speaker_names_display,
        )

    # Stage 2: Require transcription provider
    if transcription_provider is None:
        logger.warning(
            "    Skipping transcription: Transcription provider not available",
        )
        _cleanup_temp_media(temp_media, cfg)
        return False, None, 0

    # Get bytes downloaded (media file size)
    bytes_downloaded = 0
    if temp_media and os.path.exists(temp_media):
        try:
            bytes_downloaded = os.path.getsize(temp_media)
        except OSError:
            # File size check is optional (for metrics only)
            # Use default value of 0 if stat fails
            pass

    # Audio preprocessing (RFC-040): Preprocess audio BEFORE passing to any provider
    # This happens at the pipeline level, not within providers
    # All providers receive optimized audio (Whisper, OpenAI, future providers)
    media_for_transcription = _preprocess_audio_if_needed(job, cfg, temp_media, pipeline_metrics)

    if (
        cfg.transcription_provider in ("openai", "gemini")
        and media_for_transcription
        and os.path.exists(media_for_transcription)
    ):
        try:
            preprocessed_bytes = os.path.getsize(media_for_transcription)
        except OSError:
            preprocessed_bytes = None
        # OPENAI_MAX_FILE_SIZE_BYTES names the historical constant; both OpenAI and Gemini
        # transcription paths use this same post-preprocess ceiling (GitHub #557 / #561).
        if preprocessed_bytes is not None and preprocessed_bytes > OPENAI_MAX_FILE_SIZE_BYTES:
            msg = (
                f"Preprocessed audio ({preprocessed_bytes} B) exceeds API limit "
                f"({OPENAI_MAX_FILE_SIZE_BYTES} B); skipping transcription (GitHub #557)"
            )
            logger.warning("[%s] %s", job.idx, msg)
            _append_transcription_incident(
                cfg, job, category="policy", message=msg, exception_type="PolicySkip"
            )
            _mark_episode_skipped_policy(job, cfg, pipeline_metrics, msg)
            _cleanup_temp_media(temp_media, cfg)
            return False, None, bytes_downloaded

    try:
        # Stage 2: Use provider's transcribe_with_segments method for full result with segments
        # This supports both plain text and screenplay formatting
        # Pass pipeline_metrics and episode duration for LLM call tracking (if OpenAI provider)
        # Note: Provider receives preprocessed audio (if preprocessing was successful)
        # Provider is agnostic to whether audio was preprocessed
        episode_duration_seconds = getattr(job, "episode_duration_seconds", None)
        # Apply timeout enforcement for transcription (Issue #379)
        # Create call metrics for tracking per-episode provider metrics
        from ..utils.provider_metrics import ProviderCallMetrics
        from ..utils.timeout import timeout_context, TimeoutError

        call_metrics = ProviderCallMetrics()

        try:
            with timeout_context(cfg.transcription_timeout, f"transcription for episode {job.idx}"):
                # All providers must support call_metrics (no backward compatibility)
                result, tc_elapsed = transcription_provider.transcribe_with_segments(
                    media_for_transcription,
                    language=cfg.language,
                    pipeline_metrics=pipeline_metrics,
                    episode_duration_seconds=episode_duration_seconds,
                    call_metrics=call_metrics,
                )
        except TimeoutError as e:
            logger.error(
                f"[{job.idx}] Transcription timeout after {cfg.transcription_timeout}s: {e}"
            )
            raise
        text = _format_transcript_if_needed(
            result, cfg, job.detected_speaker_names, transcription_provider
        )
        rel_path = _save_transcript_file(
            text, job, run_suffix, effective_output_dir, pipeline_metrics=pipeline_metrics
        )
        logger.info(f"    saved transcript: {rel_path} (transcribed in {tc_elapsed:.1f}s)")
        segments = result.get("segments") if isinstance(result, dict) else None
        if isinstance(segments, list) and len(segments) > 0:
            _save_transcript_segments_file(segments, rel_path, effective_output_dir)

        # Save transcript to cache for future use (enables fast multi-provider experimentation)
        _save_transcript_to_cache_if_needed(
            job,
            cfg,
            temp_media,
            text,
            transcription_provider,
            segments=segments if isinstance(segments, list) else None,
        )

        # Record transcription time if metrics available
        _record_transcription_metrics(job, cfg, tc_elapsed, call_metrics, pipeline_metrics)

        return True, rel_path, bytes_downloaded
    except (ValueError, ProviderRuntimeError) as exc:
        if is_provider_audio_payload_limit_error(exc):
            logger.warning(
                "[%s] Skipping episode due to provider payload / file size limit: %s",
                job.idx,
                redact_for_log(str(exc)),
            )
            _append_transcription_incident(
                cfg,
                job,
                category="policy",
                message=str(exc),
                exception_type=type(exc).__name__,
            )
            _mark_episode_skipped_policy(job, cfg, pipeline_metrics, str(exc))
            return False, None, bytes_downloaded
        logger.error(
            "    Transcription validation failed: %s",
            format_exception_for_log(exc),
        )
        _append_transcription_incident(
            cfg,
            job,
            category="hard",
            message=str(exc),
            exception_type=type(exc).__name__,
        )
        return False, None, bytes_downloaded
    except (RuntimeError, OSError, ProviderError) as exc:
        _append_transcription_incident(
            cfg,
            job,
            category="hard",
            message=str(exc),
            exception_type=type(exc).__name__,
        )
        logger.error(
            "    Whisper transcription failed: %s",
            format_exception_for_log(exc),
        )
        return False, None, bytes_downloaded
    finally:
        _cleanup_temp_media(temp_media, cfg)


def _determine_output_path(
    episode: Episode,  # type: ignore[valid-type]
    transcript_url: str,
    transcript_type: Optional[str],
    effective_output_dir: str,
    run_suffix: Optional[str],
    planned_ext: str,
) -> str:
    """Determine output path for transcript file.

    Transcripts are stored in the transcripts/ subdirectory within the output directory.

    Args:
        episode: Episode object
        transcript_url: Transcript URL
        transcript_type: Transcript type
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        planned_ext: Planned file extension

    Returns:
        Full path to output file
    """
    run_tag = f"_{run_suffix}" if run_suffix else ""
    base_name = (
        f"{episode.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} - {episode.title_safe}{run_tag}"
    )
    out_name = f"{base_name}{planned_ext}"
    transcripts_dir = os.path.join(effective_output_dir, filesystem.TRANSCRIPTS_SUBDIR)
    return os.path.join(transcripts_dir, out_name)


def _check_existing_transcript(
    episode: Episode,  # type: ignore[valid-type]
    effective_output_dir: str,
    run_suffix: Optional[str],
    cfg: config.Config,
) -> bool:
    """Check if transcript already exists and should be skipped.

    Checks in the transcripts/ subdirectory within the output directory.

    Args:
        episode: Episode object
        effective_output_dir: Output directory path
        run_suffix: Optional run suffix
        cfg: Configuration object

    Returns:
        True if transcript exists and should be skipped, False otherwise
    """
    if not cfg.skip_existing:
        return False

    run_tag = f"_{run_suffix}" if run_suffix else ""
    base_name = (
        f"{episode.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} - {episode.title_safe}{run_tag}"
    )
    transcripts_dir = os.path.join(effective_output_dir, filesystem.TRANSCRIPTS_SUBDIR)
    existing_matches = list(Path(transcripts_dir).glob(f"{base_name}*"))
    for candidate in existing_matches:
        if candidate.is_file():
            prefix = "[dry-run] " if cfg.dry_run else ""
            logger.info(
                "    %stranscript already exists, skipping (--skip-existing): %s",
                prefix,
                candidate,
            )
            return True
    return False


def _fetch_transcript_content(
    transcript_url: str, cfg: config.Config
) -> Optional[tuple[bytes, Optional[str]]]:
    """Fetch transcript content from URL.

    Args:
        transcript_url: URL of the transcript
        cfg: Configuration object

    Returns:
        Tuple of (data, content_type) or None if download fails
    """
    logger.debug(
        "[%s] Downloading transcript from %s",
        transcript_url,
        transcript_url,
    )
    data, ctype = downloader.http_get(transcript_url, cfg.user_agent, cfg.timeout)
    if data is None:
        logger.warning("    failed to download transcript")
        return None
    return (data, ctype)


def _write_transcript_file(
    data: bytes,
    out_path: str,
    cfg: config.Config,
    episode: Episode,  # type: ignore[valid-type]
    effective_output_dir: str,
) -> Optional[str]:
    """Write transcript data to file.

    Args:
        data: Transcript data bytes
        out_path: Output file path
        cfg: Configuration object
        episode: Episode object
        effective_output_dir: Output directory path

    Returns:
        Relative path to saved file, or None if writing fails
    """
    if cfg.skip_existing and os.path.exists(out_path):
        logger.info(f"    transcript already exists, skipping (--skip-existing): {out_path}")
        return None

    try:
        # Note: pipeline_metrics not available in this function,
        # but write_file will still log I/O time
        filesystem.write_file(out_path, data, pipeline_metrics=None)
        logger.info(f"    saved: {out_path}")
        # Return relative path from output_dir
        rel_path = os.path.relpath(out_path, effective_output_dir)
        return rel_path
    except (IOError, OSError) as exc:
        logger.error("    failed to write file: %s", format_exception_for_log(exc))
        return None


def process_transcript_download(
    episode: Episode,  # type: ignore[valid-type]
    transcript_url: str,
    transcript_type: Optional[str],
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> tuple[bool, Optional[str], Optional[str], int]:
    """Download and save a transcript file.

    Args:
        episode: Episode object with metadata
        transcript_url: URL of the transcript
        transcript_type: Declared transcript type
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional suffix for output filename

    Returns:
        Tuple of (success: bool, transcript_file_path: Optional[str],
        transcript_source: Optional[str], bytes_downloaded: int)
        transcript_source is "direct_download" or None
    """
    # Check if transcript already exists
    # If skip_existing is True but generate_summaries is enabled, still return transcript path
    # so summaries can be generated even when transcript exists
    if _check_existing_transcript(episode, effective_output_dir, run_suffix, cfg):
        if cfg.generate_summaries:
            # Find existing transcript file to return its path for summarization
            # Transcripts are now in the transcripts/ subdirectory
            run_tag = f"_{run_suffix}" if run_suffix else ""
            base_name = (
                f"{episode.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} "
                f"- {episode.title_safe}{run_tag}"
            )
            transcripts_dir = os.path.join(effective_output_dir, filesystem.TRANSCRIPTS_SUBDIR)
            existing_matches = list(Path(transcripts_dir).glob(f"{base_name}*"))
            for candidate in existing_matches:
                if candidate.is_file():
                    rel_path = os.path.relpath(str(candidate), effective_output_dir)
                    logger.debug(
                        "[%s] Transcript exists, but will use for summarization: %s",
                        episode.idx,
                        rel_path,
                    )
                    return False, rel_path, "direct_download", 0
        return False, None, None, 0

    planned_ext = derive_transcript_extension(transcript_type, None, transcript_url)
    out_path = _determine_output_path(
        episode, transcript_url, transcript_type, effective_output_dir, run_suffix, planned_ext
    )

    if cfg.dry_run:
        dry_path = out_path
        if planned_ext in (".vtt", ".srt"):
            dry_path = os.path.splitext(out_path)[0] + ".txt"
        logger.info(
            "[%s] (dry-run) transcript available: %s -> %s",
            episode.idx,
            episode.title,
            transcript_url,
        )
        logger.info(f"    [dry-run] would save as: {dry_path}")
        return True, dry_path, "direct_download", 0

    logger.info(f"[{episode.idx}] downloading transcript: {episode.title} -> {transcript_url}")

    # Fetch transcript content
    fetch_result = _fetch_transcript_content(transcript_url, cfg)
    if fetch_result is None:
        return False, None, None, 0
    data, ctype = fetch_result
    bytes_downloaded = len(data) if data else 0

    # Determine final extension (may differ from planned)
    ext = derive_transcript_extension(transcript_type, ctype, transcript_url)
    if ext != planned_ext:
        out_path = _determine_output_path(
            episode, transcript_url, transcript_type, effective_output_dir, run_suffix, ext
        )

    if ext in (".vtt", ".srt"):
        try:
            body = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            body = data.decode("utf-8", errors="replace")
        if ext == ".vtt":
            plain, segments = parse_webvtt(body)
        else:
            plain, segments = parse_srt(body)
        if plain.strip() and segments:
            txt_path = os.path.splitext(out_path)[0] + ".txt"
            rel_path_result = _write_transcript_file(
                plain.encode("utf-8"), txt_path, cfg, episode, effective_output_dir
            )
            if rel_path_result is None:
                return False, None, None, bytes_downloaded
            _save_transcript_segments_file(segments, rel_path_result, effective_output_dir)
            logger.info(
                "[%s] normalized %s to .txt with %d segment(s) for GI timing",
                episode.idx,
                ext,
                len(segments),
            )
            return True, rel_path_result, "direct_download", bytes_downloaded
        logger.warning(
            "[%s] %s parse yielded no usable cues; saving raw caption bytes",
            episode.idx,
            ext,
        )

    rel_path_result = _write_transcript_file(data, out_path, cfg, episode, effective_output_dir)
    if rel_path_result is None:
        return False, None, None, bytes_downloaded

    return True, rel_path_result, "direct_download", bytes_downloaded


def process_episode_download(
    episode: Episode,  # type: ignore[valid-type]
    cfg: config.Config,
    temp_dir: Optional[str],
    effective_output_dir: str,
    run_suffix: Optional[str],
    transcription_jobs: queue.Queue[TranscriptionJob],  # type: ignore[valid-type]
    transcription_jobs_lock: Optional[threading.Lock],
    detected_speaker_names: Optional[List[str]] = None,
    pipeline_metrics=None,
) -> tuple[bool, Optional[str], Optional[str], int]:
    """Process a single episode: download transcript or prepare for Whisper transcription.

    Args:
        episode: Episode object with metadata and URLs
        cfg: Configuration object
        temp_dir: Temporary directory for downloads
        effective_output_dir: Output directory path
        run_suffix: Optional suffix for output filename
        transcription_jobs: Queue to put TranscriptionJob objects into (bounded queue)
        transcription_jobs_lock: Lock for thread-safe access (may be redundant with Queue)

    Returns:
        Tuple of (success: bool, transcript_file_path: Optional[str],
        transcript_source: Optional[str], bytes_downloaded: int)
        transcript_source is "direct_download" or "whisper_transcription" or None
    """
    chosen = choose_transcript_url(episode.transcript_urls, cfg.prefer_types)

    if chosen:
        t_url, t_type = chosen
        logger.debug(
            "[%s] Selected transcript candidate %s (type=%s) from %s options",
            episode.idx,
            t_url,
            t_type,
            len(episode.transcript_urls),
        )
        success, transcript_path, transcript_source, bytes_downloaded = process_transcript_download(
            episode, t_url, t_type, cfg, effective_output_dir, run_suffix
        )
        if success and cfg.delay_ms:
            time.sleep(cfg.delay_ms / MS_TO_SECONDS)
        return success, transcript_path, transcript_source, bytes_downloaded

    if cfg.transcribe_missing and temp_dir:
        logger.debug("[%s] No transcript; enqueueing Whisper transcription", episode.idx)
        job = download_media_for_transcription(
            episode,
            cfg,
            temp_dir,
            effective_output_dir,
            run_suffix,
            detected_speaker_names=detected_speaker_names,
            pipeline_metrics=pipeline_metrics,
        )
        if job:
            # Use queue.put() with blocking=True to provide backpressure when queue is full
            # This prevents unbounded memory growth when downloads outpace transcription
            # The lock is kept for compatibility but Queue is already thread-safe
            if transcription_jobs_lock:
                with transcription_jobs_lock:
                    transcription_jobs.put(job, block=True, timeout=None)
            else:
                transcription_jobs.put(job, block=True, timeout=None)
            logger.debug(
                "[%s] Added transcription job (queue size=%s/%s)",
                episode.idx,
                transcription_jobs.qsize(),
                transcription_jobs.maxsize,
            )
            if cfg.delay_ms:
                time.sleep(cfg.delay_ms / MS_TO_SECONDS)
        else:
            # Issue #429: record failed episode so run index has status/error_type/stage
            if pipeline_metrics is not None:
                from .helpers import get_episode_id_from_episode

                episode_id, _ = get_episode_id_from_episode(episode, cfg.rss_url or "")
                pipeline_metrics.update_episode_status(
                    episode_id=episode_id,
                    status="failed",
                    stage="transcription",
                    error_type="DownloadError",
                    error_message="failed to download media",
                )
        return False, None, None, 0

    logger.info(f"[{episode.idx}] no transcript for: {episode.title}")
    if cfg.delay_ms:
        time.sleep(cfg.delay_ms / MS_TO_SECONDS)
    return False, None, None, 0
