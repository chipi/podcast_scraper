"""Episode-level processing: downloads, transcripts, and Whisper transcription."""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from . import config, downloader, filesystem, models, whisper
from .rss_parser import choose_transcript_url

logger = logging.getLogger(__name__)

MS_TO_SECONDS = 1000.0
DEFAULT_MEDIA_EXTENSION = ".bin"
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
    return ".txt"


def download_media_for_transcription(
    episode: models.Episode,
    cfg: config.Config,
    temp_dir: str,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> Optional[models.TranscriptionJob]:
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
        return models.TranscriptionJob(
            idx=episode.idx, ep_title=episode.title, ep_title_safe=episode.title_safe, temp_media=""
        )
    else:
        logger.info(f"[{episode.idx}] no transcript; downloading media: {display_title}")

    ext = derive_media_extension(episode.media_type, episode.media_url)
    ep_num_str = f"{episode.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d}"
    short_title = filesystem.truncate_whisper_title(episode.title_safe, for_log=False)
    title_hash_input = f"{episode.media_url}|{episode.idx}|{cfg.rss_url}"
    title_hash = hashlib.sha1(  # nosec B324 - used only for stable filenames
        title_hash_input.encode("utf-8")
    ).hexdigest()[:TITLE_HASH_PREFIX_LENGTH]
    temp_media = os.path.join(temp_dir, f"{ep_num_str}_{short_title}_{title_hash}{ext}")

    dl_start = time.time()
    ok, total_bytes = downloader.http_download_to_file(
        episode.media_url, cfg.user_agent, cfg.timeout, temp_media
    )
    dl_elapsed = time.time() - dl_start
    if not ok:
        logger.warning("    failed to download media")
        return None

    if downloader.should_log_download_summary():
        try:
            mb = total_bytes / downloader.BYTES_PER_MB
            logger.info(f"    downloaded {mb:.2f} MB in {dl_elapsed:.1f}s")
        except (ValueError, ZeroDivisionError, TypeError) as exc:
            logger.warning(f"    failed to format download size: {exc}")
    return models.TranscriptionJob(
        idx=episode.idx,
        ep_title=episode.title,
        ep_title_safe=episode.title_safe,
        temp_media=temp_media,
    )


def transcribe_media_to_text(  # noqa: C901 - orchestrates Whisper transcription
    job: models.TranscriptionJob,
    cfg: config.Config,
    whisper_model,
    run_suffix: Optional[str],
    effective_output_dir: str,
) -> bool:
    """Transcribe media file using Whisper and save result.

    Args:
        job: TranscriptionJob with media file path
        cfg: Configuration object
        whisper_model: Loaded Whisper model
        run_suffix: Optional suffix for output filename
        effective_output_dir: Output directory path

    Returns:
        True if transcription succeeded, False otherwise
    """
    if cfg.dry_run:
        final_path = filesystem.build_whisper_output_path(
            job.idx, job.ep_title_safe, run_suffix, effective_output_dir
        )
        logger.info(f"[{job.idx}] (dry-run) would transcribe media -> {final_path}")
        return True

    temp_media = job.temp_media
    if whisper_model is None:
        logger.warning("    Skipping transcription: Whisper model not available")
        try:
            os.remove(temp_media)
        except OSError as exc:
            logger.warning(f"    failed to remove temp media file {temp_media}: {exc}")
        return False

    try:
        result, tc_elapsed = whisper.transcribe_with_whisper(whisper_model, temp_media, cfg)
        text = (result.get("text") or "").strip()
        if cfg.screenplay and isinstance(result, dict) and isinstance(result.get("segments"), list):
            try:
                formatted = whisper.format_screenplay_from_segments(
                    result["segments"],
                    cfg.screenplay_num_speakers,
                    cfg.screenplay_speaker_names,
                    cfg.screenplay_gap_s,
                )
                if formatted.strip():
                    text = formatted
            except (ValueError, KeyError, TypeError) as exc:
                logger.warning(f"    failed to format as screenplay, using plain transcript: {exc}")
        if not text:
            raise RuntimeError("empty transcription")
        out_path = filesystem.build_whisper_output_path(
            job.idx, job.ep_title_safe, run_suffix, effective_output_dir
        )
        filesystem.write_file(out_path, text.encode("utf-8"))
        logger.info(f"    saved transcript: {out_path} (transcribed in {tc_elapsed:.1f}s)")
        return True
    except (RuntimeError, OSError) as exc:
        logger.error(f"    Whisper transcription failed: {exc}")
        return False
    finally:
        try:
            os.remove(temp_media)
        except OSError as exc:
            logger.warning(f"    failed to remove temp media file {temp_media}: {exc}")


def process_transcript_download(
    episode: models.Episode,
    transcript_url: str,
    transcript_type: Optional[str],
    cfg: config.Config,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> bool:
    """Download and save a transcript file.

    Args:
        episode: Episode object with metadata
        transcript_url: URL of the transcript
        transcript_type: Declared transcript type
        cfg: Configuration object
        effective_output_dir: Output directory path
        run_suffix: Optional suffix for output filename

    Returns:
        True if transcript was downloaded and saved, False otherwise
    """
    run_tag = f"_{run_suffix}" if run_suffix else ""
    base_name = (
        f"{episode.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} - {episode.title_safe}{run_tag}"
    )
    if cfg.skip_existing:
        existing_matches = list(Path(effective_output_dir).glob(f"{base_name}*"))
        for candidate in existing_matches:
            if candidate.is_file():
                prefix = "[dry-run] " if cfg.dry_run else ""
                logger.info(
                    "    %stranscript already exists, skipping (--skip-existing): %s",
                    prefix,
                    candidate,
                )
                return False

    planned_ext = derive_transcript_extension(transcript_type, None, transcript_url)
    out_name = f"{base_name}{planned_ext}"
    out_path = os.path.join(effective_output_dir, out_name)

    if cfg.dry_run:
        logger.info(
            "[%s] (dry-run) transcript available: %s -> %s",
            episode.idx,
            episode.title,
            transcript_url,
        )
        logger.info(f"    [dry-run] would save as: {out_path}")
        return True

    logger.debug(
        "[%s] Downloading transcript from %s (planned extension=%s)",
        episode.idx,
        transcript_url,
        planned_ext,
    )
    logger.info(f"[{episode.idx}] downloading transcript: {episode.title} -> {transcript_url}")

    data, ctype = downloader.http_get(transcript_url, cfg.user_agent, cfg.timeout)
    if data is None:
        logger.warning("    failed to download transcript")
        return False

    ext = derive_transcript_extension(transcript_type, ctype, transcript_url)
    if ext != planned_ext:
        out_name = f"{base_name}{ext}"
        out_path = os.path.join(effective_output_dir, out_name)

    if cfg.skip_existing and os.path.exists(out_path):
        logger.info(f"    transcript already exists, skipping (--skip-existing): {out_path}")
        return False
    try:
        filesystem.write_file(out_path, data)
        logger.info(f"    saved: {out_path}")
        return True
    except (IOError, OSError) as exc:
        logger.error(f"    failed to write file: {exc}")
        return False


def process_episode_download(
    episode: models.Episode,
    cfg: config.Config,
    temp_dir: Optional[str],
    effective_output_dir: str,
    run_suffix: Optional[str],
    transcription_jobs: List[models.TranscriptionJob],
    transcription_jobs_lock: Optional[threading.Lock],
) -> bool:
    """Process a single episode: download transcript or prepare for Whisper transcription.

    Args:
        episode: Episode object with metadata and URLs
        cfg: Configuration object
        temp_dir: Temporary directory for downloads
        effective_output_dir: Output directory path
        run_suffix: Optional suffix for output filename
        transcription_jobs: List to append TranscriptionJob objects to
        transcription_jobs_lock: Lock for thread-safe access to transcription_jobs

    Returns:
        True if transcript was downloaded, False otherwise
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
        success = process_transcript_download(
            episode, t_url, t_type, cfg, effective_output_dir, run_suffix
        )
        if success and cfg.delay_ms:
            time.sleep(cfg.delay_ms / MS_TO_SECONDS)
        return success

    if cfg.transcribe_missing and temp_dir:
        logger.debug("[%s] No transcript; enqueueing Whisper transcription", episode.idx)
        job = download_media_for_transcription(
            episode,
            cfg,
            temp_dir,
            effective_output_dir,
            run_suffix,
        )
        if job:
            if transcription_jobs_lock:
                with transcription_jobs_lock:
                    transcription_jobs.append(job)
            else:
                transcription_jobs.append(job)
            logger.debug(
                "[%s] Added transcription job (queue size=%s)", episode.idx, len(transcription_jobs)
            )
            if cfg.delay_ms:
                time.sleep(cfg.delay_ms / MS_TO_SECONDS)
        return False

    logger.info(f"[{episode.idx}] no transcript for: {episode.title}")
    if cfg.delay_ms:
        time.sleep(cfg.delay_ms / MS_TO_SECONDS)
    return False
