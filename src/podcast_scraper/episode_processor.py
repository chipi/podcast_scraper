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

from . import config, downloader, filesystem, models
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
    detected_speaker_names: Optional[List[str]] = None,
    pipeline_metrics=None,
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
        # If generate_summaries is enabled, still return a job so transcript path can be used
        # for summarization (even though we won't re-transcribe)
        if cfg.generate_summaries:
            logger.debug(
                "[%s] Transcript exists, but will use for summarization: %s",
                episode.idx,
                final_out_path,
            )
            # Return a job with empty temp_media since we won't download/transcribe
            return models.TranscriptionJob(
                idx=episode.idx,
                ep_title=episode.title,
                ep_title_safe=episode.title_safe,
                temp_media="",  # Empty since we're reusing existing transcript
                detected_speaker_names=detected_speaker_names,
            )
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
    # Hash is only for stable filenames (not security sensitive)
    title_hash = hashlib.sha1(title_hash_input.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :TITLE_HASH_PREFIX_LENGTH
    ]
    temp_media = os.path.join(temp_dir, f"{ep_num_str}_{short_title}_{title_hash}{ext}")

    # Check if media file already exists and reuse it if configured
    total_bytes = 0
    dl_elapsed = 0.0
    if cfg.reuse_media and os.path.exists(temp_media):
        logger.debug(f"    reusing existing media file: {temp_media}")
        # Verify file size is reasonable (not empty or corrupted)
        try:
            file_size = os.path.getsize(temp_media)
            if file_size > 0:
                total_bytes = file_size
                logger.debug(f"    media file size: {file_size} bytes")
            else:
                logger.warning(f"    media file is empty, re-downloading: {temp_media}")
                # File exists but is empty, re-download
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
                        logger.debug(f"    downloaded {mb:.2f} MB in {dl_elapsed:.1f}s")
                    except (ValueError, ZeroDivisionError, TypeError) as exc:
                        logger.debug(f"    failed to format download size: {exc}")
        except OSError as exc:
            logger.warning(f"    error checking media file, re-downloading: {exc}")
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
    else:
        # Download media file
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

    # Record download time if metrics available and download actually happened
    if pipeline_metrics is not None and dl_elapsed > 0:
        pipeline_metrics.record_download_media_time(dl_elapsed)

    return models.TranscriptionJob(
        idx=episode.idx,
        ep_title=episode.title,
        ep_title_safe=episode.title_safe,
        temp_media=temp_media,
        detected_speaker_names=detected_speaker_names,
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
                # Fallback: log warning and use plain text
                logger.warning(
                    "Screenplay formatting requested but provider doesn't support it. "
                    "Using plain transcript."
                )
                formatted = None
            if formatted and formatted.strip():
                text = formatted
        except (ValueError, KeyError, TypeError) as exc:
            logger.warning(f"    failed to format as screenplay, using plain transcript: {exc}")
    return text


def _save_transcript_file(
    text: str, job: models.TranscriptionJob, run_suffix: Optional[str], effective_output_dir: str
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
    filesystem.write_file(out_path, text.encode("utf-8"))
    rel_path = os.path.relpath(out_path, effective_output_dir)
    return rel_path


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


def transcribe_media_to_text(
    job: models.TranscriptionJob,
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
    final_out_path = filesystem.build_whisper_output_path(
        job.idx, job.ep_title_safe, run_suffix, effective_output_dir
    )

    # If temp_media is empty and transcript exists, we're reusing existing transcript
    # (happens when skip_existing=True and generate_summaries=True)
    if not temp_media and cfg.skip_existing and os.path.exists(final_out_path):
        rel_path = os.path.relpath(final_out_path, effective_output_dir)
        logger.debug(
            "[%s] Reusing existing Whisper transcript for summarization: %s",
            job.idx,
            rel_path,
        )
        return True, rel_path, 0

    # Log detected speaker names (hosts + guests) before transcription
    if job.detected_speaker_names:
        speaker_names_display = ", ".join(job.detected_speaker_names)
        logger.debug("    Speaker names for transcription: %s", speaker_names_display)

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

    try:
        # Stage 2: Use provider's transcribe_with_segments method for full result with segments
        # This supports both plain text and screenplay formatting
        result, tc_elapsed = transcription_provider.transcribe_with_segments(
            temp_media, language=cfg.language
        )
        text = _format_transcript_if_needed(
            result, cfg, job.detected_speaker_names, transcription_provider
        )
        rel_path = _save_transcript_file(text, job, run_suffix, effective_output_dir)
        logger.info(f"    saved transcript: {rel_path} (transcribed in {tc_elapsed:.1f}s)")

        # Record transcription time if metrics available
        if pipeline_metrics is not None:
            pipeline_metrics.record_transcribe_time(tc_elapsed)

        return True, rel_path, bytes_downloaded
    except ValueError as exc:
        # Handle file size validation errors gracefully
        error_msg = str(exc)
        if "exceeds" in error_msg and "limit" in error_msg:
            logger.warning(f"[{job.idx}] Skipping episode due to file size limit: {error_msg}")
            # Return False to indicate episode was skipped (not failed)
            return False, None, bytes_downloaded
        else:
            # Re-raise if it's a different ValueError
            logger.error(f"    Transcription validation failed: {exc}")
            return False, None, bytes_downloaded
    except (RuntimeError, OSError) as exc:
        logger.error(f"    Whisper transcription failed: {exc}")
        return False, None, bytes_downloaded
    finally:
        _cleanup_temp_media(temp_media, cfg)


def _determine_output_path(
    episode: models.Episode,
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
    episode: models.Episode,
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
    episode: models.Episode,
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
        filesystem.write_file(out_path, data)
        logger.info(f"    saved: {out_path}")
        # Return relative path from output_dir
        rel_path = os.path.relpath(out_path, effective_output_dir)
        return rel_path
    except (IOError, OSError) as exc:
        logger.error(f"    failed to write file: {exc}")
        return None


def process_transcript_download(
    episode: models.Episode,
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
        logger.info(
            "[%s] (dry-run) transcript available: %s -> %s",
            episode.idx,
            episode.title,
            transcript_url,
        )
        logger.info(f"    [dry-run] would save as: {out_path}")
        return True, out_path, "direct_download", 0

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

    # Write transcript file
    rel_path_result = _write_transcript_file(data, out_path, cfg, episode, effective_output_dir)
    if rel_path_result is None:
        return False, None, None, bytes_downloaded

    return True, rel_path_result, "direct_download", bytes_downloaded


def process_episode_download(
    episode: models.Episode,
    cfg: config.Config,
    temp_dir: Optional[str],
    effective_output_dir: str,
    run_suffix: Optional[str],
    transcription_jobs: List[models.TranscriptionJob],
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
        transcription_jobs: List to append TranscriptionJob objects to
        transcription_jobs_lock: Lock for thread-safe access to transcription_jobs

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
        return False, None, None, 0

    logger.info(f"[{episode.idx}] no transcript for: {episode.title}")
    if cfg.delay_ms:
        time.sleep(cfg.delay_ms / MS_TO_SECONDS)
    return False, None, None, 0
