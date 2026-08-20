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
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

from .. import config, config_constants, models

if TYPE_CHECKING:
    from ..models import Episode, TranscriptionJob
else:
    Episode = models.Episode  # type: ignore[assignment]
    TranscriptionJob = models.TranscriptionJob  # type: ignore[assignment]
from ..exceptions import ProviderError, ProviderRuntimeError
from ..preprocessing.audio.factory import preprocessing_fingerprint
from ..rss import choose_transcript_url, downloader
from ..rss.downloader import OPENAI_MAX_FILE_SIZE_BYTES
from ..transcript_formats import parse_srt, parse_webvtt
from ..utils import filesystem
from ..utils.audio_payload_limits import is_provider_audio_payload_limit_error
from ..utils.corpus_incidents import append_corpus_incident
from ..utils.log_redaction import format_exception_for_log, redact_for_log
from . import run_index

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
    if not _job_has_episode_for_metrics(job):
        return None
    ep = job.episode
    assert ep is not None
    item = getattr(ep, "item", None)
    if item is None:
        return None
    from ..rss.parser import extract_episode_metadata

    _, _, _, duration, _, _ = extract_episode_metadata(item, "")
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
    effective_output_dir: Optional[str] = None,
) -> tuple[bool, int, float]:
    """Download media or reuse existing file. Returns (success, total_bytes, dl_elapsed).

    #947: before re-fetching from the feed, try the durable raw-audio cache (keyed by
    episode GUID). On a cache hit the audio is copied into ``temp_media`` and no network
    request is made — this is what lets a reprocess (re-diarization) run without the live
    feed. After a fresh download the audio is stored into the cache for next time.
    """
    if episode.media_url is None:
        logger.warning("    media_url is missing; cannot download")
        return False, 0, 0.0
    if pipeline_metrics is not None:
        pipeline_metrics.record_download_media_attempt()

    # #947 audio archive lookup (GUID-keyed) — highest priority source.
    # #1199: the archive is local (default) or remote object storage (rclone).
    from ..rss import extract_item_guid
    from ..utils import audio_cache

    guid = extract_item_guid(episode.item) if getattr(episode, "item", None) is not None else None
    archive = audio_cache.resolve_backend(cfg, effective_output_dir)
    if archive is not None and guid:
        if audio_cache.fetch_into(archive, guid, temp_media):
            try:
                size = os.path.getsize(temp_media)
            except OSError:
                size = 0
            if size > 0:
                logger.info(
                    "    [#947] audio archive HIT (guid=%s) via %s (no feed fetch)",
                    guid,
                    archive.describe(),
                )
                return True, size, 0.0

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
    # #947/#1199: archive the freshly-downloaded raw audio for future reprocessing (best-effort).
    if archive is not None and guid:
        # H1: was the archive ALREADY holding an object for this GUID before we stored? If so,
        # store_via dedupes (upload() returns success without writing), and the cold object may be
        # a DIFFERENT (dynamic-ad re-encoded) copy than the bytes we just downloaded + transcribed.
        # Provenance must then NOT claim byte-identical.
        pre_existing = False
        try:
            from ..archive.backfill import already_archived

            pre_existing = already_archived(archive, guid) is not None
        except Exception:  # noqa: BLE001 - a probe failure just means we can't prove dedupe
            pre_existing = False

        stored = audio_cache.store_via(archive, guid, temp_media)
        if stored:
            logger.info(
                "    [#947] audio archive STORE (guid=%s) -> %s via %s",
                guid,
                stored,
                archive.describe(),
            )
            # #1789 (+M3): stamp provenance at the download choke point so every archived episode
            # is traceable. Write to the CORPUS ROOT (same place backfill + finalize use), not the
            # per-run dir, so a corpus has ONE provenance file rather than a scatter.
            try:
                from ..archive.backfill import record_pipeline_provenance

                corpus_root = str(effective_output_dir)
                if getattr(cfg, "single_feed_uses_corpus_layout", False):
                    from .corpus_operations import corpus_parent_for_manifest_stamp_from_cfg

                    _root = corpus_parent_for_manifest_stamp_from_cfg(cfg)
                    if _root:
                        corpus_root = str(_root)

                record_pipeline_provenance(
                    corpus_root,
                    guid=str(guid),
                    rel_key=stored,
                    source_url=str(episode.media_url or ""),
                    byte_identical=not pre_existing,
                )
            except Exception:  # noqa: BLE001 - provenance is a breadcrumb, never block ingestion
                logger.debug("audio provenance: record failed (non-fatal)", exc_info=True)
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
    metadata_named: Optional[List[str]] = None,
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
    # Skip-existing must key on the STABLE guid, not the run-local idx (which shifts when the feed
    # grows → silent reprocess + duplicates). Resolve the on-disk idx by guid for the existence
    # check; a genuinely new episode falls back to its run-local idx (its real output path).
    skip_idx = (
        run_index.resolve_ondisk_idx_for_episode(episode, effective_output_dir)
        if cfg.skip_existing
        else episode.idx
    )
    # Whether speaker detection ran for this episode, read from the stage ledger rather than
    # threaded through the positional download-args tuple (#1647). ``detected_speaker_names``
    # cannot answer it — empty means both "ran, found nobody" and "never ran" — and the roster
    # needs the difference to tell an accepted unnamed voice from an unmeasured one.
    detection_ran = (
        pipeline_metrics.stage_did_run("speaker_detection", episode.idx)
        if pipeline_metrics is not None and hasattr(pipeline_metrics, "stage_did_run")
        else None
    )
    final_out_path = filesystem.build_whisper_output_path(
        skip_idx, episode.title_safe, run_suffix, effective_output_dir
    )
    # pipeline_stage=relabel_only reuses the on-disk transcript + diarization and re-runs
    # only the speaker-name resolution — no audio is needed. Return a no-download job so
    # transcribe_media_to_text reaches the relabel branch (which loads from disk).
    if cfg.pipeline_stage == "relabel_only":
        speaker_names_copy = list(detected_speaker_names) if detected_speaker_names else None
        return TranscriptionJob(  # type: ignore[no-any-return]
            idx=episode.idx,
            ep_title=episode.title,
            ep_title_safe=episode.title_safe,
            temp_media="",
            detected_speaker_names=speaker_names_copy,
            metadata_named=list(metadata_named) if metadata_named else None,
            speaker_detection_ran=detection_ran,
            episode=episode,
        )
    # D7: under --single-feed-uses-corpus-layout each run writes a FRESH run dir, so an
    # already-processed episode's transcript is in a PRIOR run dir — NOT final_out_path (this run's
    # OUTPUT path). Resolve presence corpus-wide by stable guid; else skip-existing scoped to the
    # empty run dir silently re-transcribes it (the Step-1 NO-GO, 2026-08-11).
    _corpus_layout = bool(getattr(cfg, "single_feed_uses_corpus_layout", False))
    if cfg.skip_existing and _corpus_layout and cfg.output_dir:
        _existing_transcript = run_index.existing_transcript_path_in_corpus(
            episode, str(cfg.output_dir)
        )
    elif cfg.skip_existing and os.path.exists(final_out_path):
        _existing_transcript = final_out_path
    else:
        _existing_transcript = None

    if cfg.skip_existing and _existing_transcript is not None:
        if _force_reprocess_for_source(episode, effective_output_dir, run_suffix, cfg):
            # #925: a scoped reprocess (--reprocess-source) forces matching episodes
            # (e.g. whisper_transcription) back through download+transcribe so diarization
            # re-runs and the GI/KG/CIL cascade with it. This is the Whisper path (episodes
            # with no transcript URL); the override in _check_existing_transcript covers the
            # direct-download path. Falling through schedules a real download, so the
            # job.temp_media reuse skip below is bypassed too.
            logger.info(
                "[%s] [#925] forcing re-transcription + diarization (reprocess-source=%s): %s",
                episode.idx,
                cfg.reprocess_source,
                _existing_transcript,
            )
            # Fall through: schedule download/transcribe.
        elif _should_retranscribe_for_gi_segments(cfg, _existing_transcript):
            logger.info(
                "[%s] Transcript exists without .segments.json; will re-transcribe to populate "
                "sidecar for GI quote timestamps and segment-backed speaker_id when segments "
                "carry speaker labels (backfill_transcript_segments + generate_gi): %s",
                episode.idx,
                _existing_transcript,
            )
            # Fall through: do not return — schedule download/transcribe to populate sidecar (#542).
        # If generate_summaries is enabled, still return a job so transcript path can be used for
        # summarization (even though we won't re-transcribe). NOT in corpus-layout: there the
        # episode is already fully processed in a prior run — reusing re-summarizes; skip instead.
        elif cfg.generate_summaries and not _corpus_layout:
            logger.debug(
                "[%s] Transcript exists, but will use for summarization: %s",
                episode.idx,
                _existing_transcript,
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
                speaker_detection_ran=detection_ran,
                episode=episode,
            )
        else:
            prefix = "[dry-run] " if cfg.dry_run else ""
            logger.info(
                "[%s] %stranscript already exists; skipping (--skip-existing): %s",
                episode.idx,
                prefix,
                _existing_transcript,
            )
            _mark_episode_skipped_existing(
                episode,
                cfg,
                pipeline_metrics,
                f"transcript already exists: {_existing_transcript}",
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
        episode, cfg, temp_media, pipeline_metrics, effective_output_dir
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
        metadata_named=list(metadata_named) if metadata_named else None,
        speaker_detection_ran=detection_ran,
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
            segments = result["segments"]
            has_diarized_labels = any(
                isinstance(seg, dict) and seg.get("speaker_label") for seg in segments
            )
            # ``speaker_label`` on segments means a roster pass ran (local diarizer OR the native
            # roster pass) — render via the shared diarized formatter, not the provider's positional
            # one, so roster-resolved names win. (No ``diarize=false`` path emits speaker_label
            # except a roster pass, so this stays inert for plain provider screenplays.)
            if has_diarized_labels:
                from ..providers.ml.diarization.formatting import (
                    format_diarized_screenplay_from_segments,
                )

                formatted = format_diarized_screenplay_from_segments(segments)
            elif transcription_provider and hasattr(
                transcription_provider, "format_screenplay_from_segments"
            ):
                formatted = transcription_provider.format_screenplay_from_segments(
                    result["segments"],
                    cfg.screenplay_num_speakers,
                    speaker_names,
                    cfg.screenplay_gap_s,
                )
            else:
                formatted = None
            if formatted is None and not cfg.diarize:
                # Fallback: log at most once per process (GitHub #562; config also coerces).
                with _screenplay_unsupported_warn_lock:
                    if not _screenplay_unsupported_warn_state["emitted"]:
                        _screenplay_unsupported_warn_state["emitted"] = True
                        logger.warning(
                            "Screenplay formatting requested but provider doesn't support it. "
                            "Using plain transcript (GitHub #562; see CONFIGURATION.md)."
                        )
            elif formatted and formatted.strip():
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

    # #1212: if no screenplay was produced, ``text`` is still the raw ``result["text"]`` — the
    # space-joined Whisper segments on one unbroken line, which grounds to ~zero quotes (#1182).
    # Give it segment structure: one line per segment, with each segment's char range recorded so
    # the char->timestamp mapping stays exact (the inserted newlines would otherwise blow the
    # cumulative-length alignment guard). Only when we actually have >=2 real segments to delimit.
    raw_plain = (result.get("text") or "").strip()
    if text == raw_plain and isinstance(result, dict) and isinstance(result.get("segments"), list):
        real_segs = [
            s for s in result["segments"] if isinstance(s, dict) and (s.get("text") or "").strip()
        ]
        if len(real_segs) >= 2:
            from ..transcript_formats.plain import format_plain_transcript_with_offsets

            delimited, offset_segments = format_plain_transcript_with_offsets(real_segs)
            if delimited.strip():
                text = delimited
                result["segments"] = offset_segments
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
        # WARNING, not debug: a missing .segments.json silently zeroes GI quote
        # timestamps downstream (#542) with no operator signal (review M16).
        logger.warning("Could not save segments file %s: %s", segments_path, e)


def _save_speaker_diagnostics_file(
    diagnostics: Optional[Dict[str, Any]],
    rel_transcript_path: str,
    effective_output_dir: str,
) -> None:
    """Persist per-episode speaker-resolution diagnostics next to the transcript.

    ``<base>.speakers.diagnostics.json`` records what the roster tried, what it resolved, and
    why each unresolved voice failed — so an operator can explain unrecognized speakers without
    re-running the pipeline. No-op when diarization produced no diagnostics.
    """
    if not diagnostics or not rel_transcript_path:
        return
    full_path = os.path.join(effective_output_dir, rel_transcript_path)
    base, _ = os.path.splitext(full_path)
    diag_path = base + ".speakers.diagnostics.json"
    try:
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2, allow_nan=False)
        logger.debug("Saved speaker diagnostics: %s", diag_path)
    except OSError as e:
        logger.warning("Could not save speaker diagnostics %s: %s", diag_path, e)


def _attach_speech_audio_ratio(
    result: Any, media_for_transcription: str, episode_duration_seconds: Optional[float]
) -> None:
    """Attach the speech_audio_ratio METRIC (not a gate) to ``result`` in place.

    ``Σ(transcript segments) / total audio`` — the fraction of runtime that is actual audio content
    vs music/silence/dead-air. Computed ALWAYS (independent of any coverage gate) and surfaced
    in the ASR provenance + manifest so every episode records how much of its runtime is signal.
    raw ADR-123 coverage value repurposed as an observability metric after that gate was retired in
    favour of the speech-normalized ADR-131 one.
    """
    if not isinstance(result, dict):
        return
    from ..providers.resilience.fallback import _segment_coverage

    dur = int(episode_duration_seconds) if episode_duration_seconds is not None else None
    sar = _segment_coverage(result, media_for_transcription, dur)
    if sar is not None:
        result["speech_audio_ratio"] = round(sar, 3)


def _save_asr_provenance_file(
    result: Optional[Dict[str, Any]],
    cfg: config.Config,
    rel_transcript_path: str,
    effective_output_dir: str,
) -> None:
    """ADR-131: record the ACTUAL per-episode ASR model + speech coverage next to the transcript.

    ``<base>.asr.json``. The pipeline otherwise records only the CONFIGURED transcription model, so
    a speech-coverage failover (turbo -> large-v3) left no per-episode trace of which model actually
    produced the transcript. This closes that provenance gap: the few episodes turbo dropped speech
    on are recorded as the failover model, the rest as the primary. No-op when the gate did not run
    (off, or no diarization speech denominator) so there is nothing meaningful to record.
    """
    if not isinstance(result, dict) or not rel_transcript_path:
        return
    cov = result.get("asr_speech_coverage")
    failover = result.get("speech_coverage_failover")
    sar = result.get("speech_audio_ratio")
    if cov is None and not failover and sar is None:
        return
    provenance: Dict[str, Any] = {
        "model": (
            result.get("model_used")
            or getattr(cfg, "dgx_whisper_model", None)
            or getattr(cfg, "transcription_provider", None)
        ),
        "speech_coverage": cov,
        # Σ(segments)/total-audio content signal — always present, gate or not (see caller).
        "speech_audio_ratio": sar,
        "failed_over": bool(failover),
    }
    if failover:
        provenance["speech_coverage_failover"] = failover
    full_path = os.path.join(effective_output_dir, rel_transcript_path)
    base, _ = os.path.splitext(full_path)
    asr_path = base + ".asr.json"
    try:
        with open(asr_path, "w", encoding="utf-8") as f:
            json.dump(provenance, f, indent=2, allow_nan=False)
        logger.debug("Saved ASR provenance: %s", asr_path)
    except OSError as e:
        logger.warning("Could not save ASR provenance %s: %s", asr_path, e)


def _episode_naming_cost(pipeline_metrics: Any, job: Any) -> Optional[float]:
    """This episode's speaker-detection cost, or None when it was never measured.

    None and 0.0 are different claims and the manifest keeps them apart: 0.0 means detection ran
    and made no priced LLM call; None means detection never ran for this episode (or the caller
    passed a metrics object that predates the per-episode store), so the cost is unknown.
    """
    by_episode = getattr(pipeline_metrics, "speaker_detection_cost_usd_by_episode", None)
    if not isinstance(by_episode, dict):
        return None
    value = by_episode.get(getattr(job, "idx", None))
    return None if value is None else float(value)


def _write_processing_manifest(
    result: Optional[Dict[str, Any]],
    cfg: config.Config,
    job: TranscriptionJob,  # type: ignore[valid-type]
    rel_transcript_path: str,
    effective_output_dir: str,
    asr_elapsed: Optional[float] = None,
    asr_call_metrics: Any = None,
    pipeline_metrics: Any = None,
) -> None:
    """RFC-109 / ADR-132: write the per-episode processing manifest's ASR/diarization/naming blocks.

    Each block is built from that stage's OWN result fields (never from ``cfg``) at the one site
    where all three results are in hand. Complements ``metadata.json`` (ADR-133). Best-effort — a
    manifest write never fails the episode. The downstream stages (summary / GI / KG) append their
    own blocks to the same manifest from ``metadata_generation._write_downstream_manifest_blocks``.
    ``.asr.json`` stays (ADR-133 write-both migration) until readers move to the manifest.
    """
    if not isinstance(result, dict) or not rel_transcript_path:
        return
    from ..utils import correlation
    from . import processing_manifest as pm

    episode_id, _ = _episode_id_and_idx_for_incident(job, cfg)
    feed_id = getattr(cfg, "rss_url", None)
    # cfg.run_id defaults to None and is frozen; the RESOLVED run id lives in the correlation global
    # (like llm_cost events). Sourcing from cfg gave every manifest/event run_id=null (advisor #1).
    run_id = correlation.get_run_id() or getattr(cfg, "run_id", None)

    # --- ASR: actual model + speech coverage + failover (ADR-131 provenance) ---
    cov = result.get("asr_speech_coverage")
    failover = result.get("speech_coverage_failover")
    sar = result.get("speech_audio_ratio")
    # The metric alone is enough to record the ASR stage — so the block is present on every episode
    # (not only the rare failover / speech-gate run), closing the "no asr stage in manifest" gap.
    if cov is not None or failover or sar is not None:
        asr_flags: List[str] = []
        if failover:
            asr_flags.append("asr_failover")
        thresh = getattr(cfg, "transcription_speech_coverage_min", None)
        if cov is not None and thresh and cov < thresh:
            asr_flags.append("asr_speech_coverage_low")
        # Total ASR cost = primary call + any failover re-transcription (both 0 for local models;
        # a cloud ASR that failed over billed twice — RFC-109).
        _primary_cost = getattr(asr_call_metrics, "estimated_cost", None)
        _failover_cost = result.get("asr_failover_cost_usd")
        asr_cost = None
        if _primary_cost is not None or _failover_cost is not None:
            asr_cost = float(_primary_cost or 0.0) + float(_failover_cost or 0.0)
        asr = pm.stage_block(
            ran=True,
            method=getattr(cfg, "transcription_provider", None),
            model=(result.get("model_used") or getattr(cfg, "dgx_whisper_model", None)),
            method_version=pm.METHOD_VERSIONS["asr"],
            duration_s=asr_elapsed,
            # 0.0 when a LOCAL engine ran (DGX/whisper — measured, and genuinely free); real USD
            # for cloud ASR (OpenAI/Deepgram) from apply_estimated_cost_if_missing; None only
            # when nobody measured it. See ``measured_or_unmeasured``.
            cost_usd=pm.measured_or_unmeasured(
                asr_cost,
                getattr(cfg, "transcription_provider", None),
                pm.LOCAL_TRANSCRIPTION_PROVIDERS,
            ),
            metrics={"speech_coverage": cov, "speech_audio_ratio": sar},
            failover=failover or None,
        )
        pm.update_stage(
            effective_output_dir,
            rel_transcript_path,
            "asr",
            asr,
            quality_flags=asr_flags,
            episode_id=episode_id,
            feed_id=feed_id,
            run_id=run_id,
        )

    # --- Diarization: speaker count + speech seconds, from the diarizer's own result ---
    num_spk = result.get("diarization_num_speakers")
    speech_s = result.get("diarization_speech_seconds")
    if num_spk is not None or speech_s is not None:
        diar = pm.stage_block(
            ran=True,
            method=getattr(cfg, "diarization_provider", None),
            # ADR-132 provenance: the ACTUAL served model (from the DiarizationResult), falling back
            # to the configured field for the provider — so the diar block records WHICH model, not
            # just the method. Previously absent (the diar block carried method but no model).
            model=(
                result.get("diarization_model_name")
                or getattr(cfg, "dgx_diarize_model", None)
                or getattr(cfg, "diarization_model", None)
            ),
            method_version=pm.METHOD_VERSIONS["diarization"],
            # 0.0 for local diarizers (pyannote/DGX — measured, and genuinely free); real USD for
            # cloud (Deepgram/Gemini) via DiarizationResult.cost_usd; None only when unmeasured.
            cost_usd=pm.measured_or_unmeasured(
                result.get("diarization_cost_usd"),
                getattr(cfg, "diarization_provider", None),
                pm.LOCAL_DIARIZATION_PROVIDERS,
            ),
            metrics={"num_speakers": num_spk, "speech_seconds": speech_s},
        )
        pm.update_stage(
            effective_output_dir,
            rel_transcript_path,
            "diarization",
            diar,
            episode_id=episode_id,
            feed_id=feed_id,
            run_id=run_id,
        )

    # --- Naming: detected-vs-named + attribution health, from the roster's own diagnostics ---
    diag = result.get("speaker_diagnostics")
    if isinstance(diag, dict) and isinstance(diag.get("summary"), dict):
        summary = diag["summary"]
        voices_raw = diag.get("voices")
        voices = voices_raw if isinstance(voices_raw, list) else []
        host_named = any(
            isinstance(v, dict) and v.get("role") == "host" and v.get("named") for v in voices
        )
        name_flags: List[str] = []
        if summary.get("unattributed_alarm"):
            name_flags.append("unnamed_dominant_voice")
        if summary.get("unbound_names"):
            name_flags.append("guest_in_title_not_placed")
        if not host_named and not summary.get("show_centric"):
            name_flags.append("empty_host_anchor")
        # Naming is NOT free by definition: cloud_balanced sets speaker_detector_provider:
        # litellm, so voice resolution is a real LLM call. This reads the PER-EPISODE figure the
        # detection stage recorded (0.0 when only the deterministic path ran — measured, and
        # honestly free); no entry means detection never ran for this episode and the cost is
        # genuinely unmeasured, which stays null rather than becoming a fabricated zero.
        #
        # It used to read ``speaker_detection_cost_usd`` straight off ``pipeline_metrics``. That
        # attribute exists only on an EpisodeCostProbe, and no probe ever wrapped the naming
        # stage — the probes are built later, for summary/GI/KG — so the getattr returned None on
        # every episode and the key was dropped from the block entirely. The run-level
        # ``llm_speaker_detection_cost_usd`` was accruing the whole time; it is shared across
        # parallel episodes, so it could never have been used here directly.
        naming_cost = _episode_naming_cost(pipeline_metrics, job)
        naming = pm.stage_block(
            ran=True,
            method_version=pm.METHOD_VERSIONS["naming"],
            cost_usd=naming_cost,
            metrics={
                "num_speakers": summary.get("num_speakers"),
                "named": summary.get("named"),
                "unresolved": summary.get("unresolved"),
                "truly_unknown": summary.get("truly_unknown"),
                "unattributed_talk_share": summary.get("unattributed_talk_share"),
                "by_voice_type": summary.get("by_voice_type"),
                # ADR-135/#1220: the labeling OUTPUT — real speakers exposed to GI/KG after
                # cameo/commercial cleanup, split named vs Voice (unresolved). Lets the sidecar
                # answer the clean named-vs-Voice rate without opening the graph.
                "exposed": summary.get("exposed"),
                "unbound_names": summary.get("unbound_names"),
                "host_named": host_named,
            },
        )
        pm.update_stage(
            effective_output_dir,
            rel_transcript_path,
            "naming",
            naming,
            quality_flags=name_flags,
            episode_id=episode_id,
            feed_id=feed_id,
            run_id=run_id,
        )


def _maybe_produce_adfree(
    cfg: config.Config,
    text: str,
    segments: Optional[List[Dict[str, Any]]],
    rel_transcript_path: str,
    effective_output_dir: str,
) -> None:
    """Derive + save the ad-free processing-base sidecars (#974), if enabled.

    No-op when ``save_adfree_transcript`` is off or there are no segments. Keeps the
    raw ``.txt`` untouched; writes ``<base>.adfree.{txt,segments.json,admap.json}``.
    """
    if not cfg.save_adfree_transcript or not rel_transcript_path:
        return
    if not isinstance(segments, list) or not segments:
        return
    from .adfree_transcript import produce_adfree_transcript

    adfree_rel = produce_adfree_transcript(
        text,
        segments,
        rel_transcript_path,
        effective_output_dir,
        extra_cue_patterns=cfg.crosspromo_cue_patterns,
    )
    if adfree_rel:
        logger.info("    saved ad-free transcript base: %s", adfree_rel)


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


def _resolve_audio_cache_entry(
    cfg: config.Config, effective_output_dir: str, episode: Any
) -> Optional[str]:
    """Retained #947 GUID audio-cache path for this episode (G6 hardlink source), or None."""
    try:
        from ..rss import extract_item_guid
        from ..utils import audio_cache

        item = getattr(episode, "item", None)
        guid = extract_item_guid(item) if item is not None else None
        if not guid:
            return None
        # #1199: the hardlink/symlink source is a LOCAL cache path; a remote
        # archive has no local file to link, so corpus media falls back to copy.
        if getattr(cfg, "audio_storage_backend", "local") == "remote":
            return None
        cache_root = audio_cache.resolve_cache_root(cfg, effective_output_dir)
        return audio_cache.lookup_by_guid(cache_root, guid)
    except Exception:  # best-effort optimisation; never block persistence
        return None


def _maybe_persist_episode_media(
    cfg: config.Config,
    temp_media: str,
    effective_output_dir: str,
    transcript_relpath: Optional[str],
    episode: Any = None,
) -> None:
    """Persist temp media into corpus ``media/`` when enabled (copy, or G6 hard/sym-link)."""
    if not getattr(cfg, "persist_episode_media", True):
        return
    if not transcript_relpath or not temp_media or not os.path.exists(temp_media):
        return
    from ..utils.corpus_media import persist_episode_media

    link_mode = getattr(cfg, "corpus_media_link_mode", "copy") or "copy"
    link_source = (
        _resolve_audio_cache_entry(cfg, effective_output_dir, episode)
        if link_mode in ("hardlink", "symlink") and episode is not None
        else None
    )
    persist_episode_media(
        temp_media,
        effective_output_dir,
        transcript_relpath,
        link_source=link_source,
        link_mode=link_mode,
    )


def _check_transcript_cache(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    temp_media: str,
    run_suffix: Optional[str],
    effective_output_dir: str,
    pipeline_metrics=None,
    transcription_provider=None,
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
    provider_name = None
    if transcription_provider:
        provider_name = (
            getattr(transcription_provider, "name", None)
            or type(transcription_provider).__name__.replace("Provider", "").lower()
        )
    model = _get_provider_model_name(transcription_provider, cfg)
    cached_entry = transcript_cache.get_cached_transcript_entry(
        audio_hash,
        cache_dir,
        provider_name=provider_name,
        model=model,
        preprocessing=preprocessing_fingerprint(cfg),
    )
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
            _maybe_produce_adfree(
                cfg, cached_transcript, cached_segments, rel_path, effective_output_dir
            )
        _maybe_persist_episode_media(
            cfg, temp_media, effective_output_dir, rel_path, episode=job.episode
        )
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
    if transcription_provider not in ("openai", "gemini", "mistral", "deepgram"):
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


def _record_preprocessing_outcome(
    pipeline_metrics: Any,
    job_idx: int,
    outcome: str,
    *,
    reason: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    duration_seconds: Optional[float] = None,
) -> None:
    """Write one ``audio_preprocessing`` row to the stage ledger (#1647).

    Guarded and best-effort: ``pipeline_metrics`` is duck-typed across callers, and an
    observability write must never be the thing that kills an episode. Lives here rather than
    inline at the three call sites so ``_preprocess_audio_if_needed`` keeps its complexity
    budget — the branches belong to reporting, not to the preprocessing decision.
    """
    if pipeline_metrics is None or not hasattr(pipeline_metrics, "record_stage_outcome"):
        return
    try:
        pipeline_metrics.record_stage_outcome(
            "audio_preprocessing",
            job_idx,
            outcome,
            reason=reason,
            detail=detail,
            duration_seconds=duration_seconds,
        )
    except Exception:  # pragma: no cover - reporting must not break the pipeline
        logger.debug("[%s] could not record audio_preprocessing outcome", job_idx)


def _record_preprocessing_degraded(
    pipeline_metrics: Any, job_idx: int, temp_media: str, elapsed: Optional[float]
) -> None:
    """Ledger row for "transcribing from UNPREPROCESSED audio" (#1647 / #558).

    ``degraded``, not ``failed``: transcription still proceeds, just from worse input — no
    mono/16 kHz/loudness normalisation, and a file that may genuinely exceed the 25 MB upload
    cap. The size of what actually went to the provider is the number an operator needs when
    asking why an episode cost more or scored worse.
    """
    detail: Dict[str, Any] = {"fallback": "original_audio"}
    try:
        detail["media_bytes"] = os.path.getsize(temp_media)
    except OSError:
        pass
    _record_preprocessing_outcome(
        pipeline_metrics,
        job_idx,
        "degraded",
        reason="preprocessing_failed_using_original_audio",
        detail=detail,
        duration_seconds=elapsed,
    )


def _preprocessing_cannot_run(
    cfg: config.Config,
    job: TranscriptionJob,  # type: ignore[valid-type]
    temp_media: str,
    pipeline_metrics: Any,
) -> bool:
    """Report why preprocessing will not run, and answer whether that is the case.

    Both branches used to be a single silent ``return`` inside ``_preprocess_audio_if_needed``,
    so an episode that never preprocessed and one that preprocessed cleanly produced the same
    manifest — no ``audio_preprocessing`` block at all, which reads as "nothing to report"
    rather than "skipped, and here is why". That is the ambiguity the stage ledger (#1647)
    exists to remove.

    Extracted rather than left inline for the same reason ``_record_preprocessing_outcome`` is:
    the branches belong to reporting, not to the preprocessing decision, and the caller is at
    its complexity budget.
    """
    if not cfg.preprocessing_enabled:
        _record_preprocessing_outcome(
            pipeline_metrics, job.idx, "skipped", reason="preprocessing_disabled"
        )
        return True
    if not (temp_media and os.path.exists(temp_media)):
        # Distinct from disabled: preprocessing was ASKED for and could not run. Still `skipped`
        # rather than `failed` — the missing download is the upstream stage's failure to report,
        # and double-reporting it here would inflate the preprocessing failure rate.
        _record_preprocessing_outcome(
            pipeline_metrics,
            job.idx,
            "skipped",
            reason="media_file_missing",
            detail={"path": str(temp_media) if temp_media else ""},
        )
        return True
    return False


def _build_preprocessor_or_report(
    cfg: config.Config,
    job: TranscriptionJob,  # type: ignore[valid-type]
    pipeline_metrics: Any,
) -> Any:
    """Construct the audio preprocessor, writing a ledger row if it cannot be built.

    Missing ffmpeg propagates as ``FFmpegUnavailableError`` — #26, decided: yell, do not
    degrade. The operator asked for preprocessing and the host cannot do it, which is a
    deployment fault affecting EVERY episode identically, not a per-episode quality wobble to
    absorb. Continuing would produce a whole corpus that has to be redone, under one WARNING
    line nobody reads.

    The ledger row is written BEFORE re-raising, so the episode's own record names the cause
    instead of the run simply dying with nothing in the artifact.

    Extracted from ``_preprocess_audio_if_needed`` for its complexity budget, like
    ``_preprocessing_cannot_run`` and ``_record_preprocessing_outcome`` before it.
    """
    from podcast_scraper.preprocessing.audio.factory import (
        create_audio_preprocessor,
        FFmpegUnavailableError,
    )

    try:
        preprocessor = create_audio_preprocessor(cfg)
    except FFmpegUnavailableError:
        _record_preprocessing_outcome(
            pipeline_metrics,
            job.idx,
            "failed",
            reason="ffmpeg_unavailable",
            detail={"fatal": True},
        )
        raise

    # `cfg.preprocessing_enabled` is known true by the time this is called, and the factory now
    # raises on missing ffmpeg, so None is unreachable. Kept as a belt-and-braces guard rather
    # than an assert: if a future factory change reintroduces a None path, transcription should
    # carry on with the original audio rather than crash on a NoneType.
    if preprocessor is None:  # pragma: no cover - unreachable via the factory's contract
        _record_preprocessing_outcome(
            pipeline_metrics,
            job.idx,
            "degraded",
            reason="preprocessor_unavailable",
            detail={"fallback": "original_audio"},
        )
    return preprocessor


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
    # Every exit writes exactly one ledger row (#1647). These two returns used to be silent, so
    # an episode that never preprocessed and an episode whose preprocessing succeeded were
    # indistinguishable downstream — the manifest simply had no audio_preprocessing block, which
    # reads as "nothing to report" rather than "we skipped it, here is why".
    if _preprocessing_cannot_run(cfg, job, temp_media, pipeline_metrics):
        return media_for_transcription

    from podcast_scraper.preprocessing.audio import cache as preprocessing_cache
    from podcast_scraper.preprocessing.audio.factory import (
        build_ffmpeg_preprocessor_with_bitrate,
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

    audio_preprocessor = _build_preprocessor_or_report(cfg, job, pipeline_metrics)
    if audio_preprocessor is None:  # pragma: no cover - unreachable via the factory's contract
        return media_for_transcription

    # Record preprocessing attempt (regardless of cache hit/miss)
    if pipeline_metrics is not None:
        pipeline_metrics.record_preprocessing_attempt()

    cache_dir = cfg.preprocessing_cache_dir or preprocessing_cache.PREPROCESSING_CACHE_DIR
    transcription_provider = str(cfg.transcription_provider or "").lower()
    first_pass_kbps = resolve_preprocessing_mp3_bitrate_kbps(cfg)
    if transcription_provider in ("openai", "gemini", "mistral", "deepgram"):
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
                # A cache HIT still means this episode was transcribed from preprocessed audio,
                # so the ledger must say ``ran`` here too — otherwise every cached episode reads
                # as "preprocessing never happened", which is the same silence the fresh path
                # had. ``cache_hit`` in the detail keeps the two distinguishable without
                # inventing a fifth outcome.
                _record_preprocessing_outcome(
                    pipeline_metrics,
                    job.idx,
                    "ran",
                    detail={
                        "original_bytes": original_size,
                        "preprocessed_bytes": cached_size,
                        "cache_hit": True,
                    },
                )
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
                    # Record the SUCCESS too, not only the degradation below. A ledger that
                    # speaks up only when something goes wrong cannot distinguish "ran fine"
                    # from "never ran" — the exact ambiguity that let #1646 hide.
                    _record_preprocessing_outcome(
                        pipeline_metrics,
                        job.idx,
                        "ran",
                        detail={
                            "original_bytes": original_size,
                            "preprocessed_bytes": preprocessed_size,
                        },
                        duration_seconds=preprocessing_wall_elapsed,
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
                    # Preprocessing SUCCEEDED here — only the size stat failed. Omitting the row
                    # would report a successful stage as if it never ran, which is the same
                    # ambiguity #1646 hid behind. The byte counts are simply absent.
                    _record_preprocessing_outcome(
                        pipeline_metrics,
                        job.idx,
                        "ran",
                        detail={"sizes_unavailable": True},
                        duration_seconds=preprocessing_wall_elapsed,
                    )
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
                # #1647: the episode is now being transcribed from UNPREPROCESSED audio — no
                # mono/16 kHz/loudness normalisation, and a file that may genuinely exceed the
                # 25 MB upload cap. Before this the only trace was a WARNING and a corpus
                # incident, so the ledger — the artifact built to answer "what actually happened
                # to this episode" — showed nothing, and a degraded episode was indistinguishable
                # from a clean one. `degraded`, not `failed`: transcription still proceeds.
                _record_preprocessing_degraded(
                    pipeline_metrics, job.idx, temp_media, preprocessing_wall_elapsed
                )
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
        # A plain string model (e.g. Deepgram "nova-3") is already the name — use it
        # directly. Without this it fell through to None and every model of that
        # provider collapsed to one cache key (H3).
        if isinstance(model, str) and model.strip():
            return model.strip()
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


def _preprocessing_fingerprint_would_lie(
    cfg: config.Config,
    temp_media: str,
    media_for_transcription: Optional[str],
) -> bool:
    """True when the cache key would claim preprocessed audio the transcriber never saw (#35).

    ``preprocessing_fingerprint(cfg)`` is computed from CONFIG and its docstring calls itself
    "identity of the audio the transcriber will actually see". Those agree only when preprocessing
    was enabled AND actually produced a file. When it was enabled and fell back to the original —
    the #18/#558 failure, where a flat 300 s budget killed preprocessing on long episodes — the
    key says ``pp=on|…`` over a transcript built from RAW audio.

    Three cases, and only one is a lie:

    * preprocessing disabled   -> key is ``pp=off``, audio was raw          -> HONEST, cache it
    * preprocessing produced a file (path differs) -> key is ``pp=on|…``    -> HONEST, cache it
    * preprocessing enabled, path unchanged (fell back)                     -> LIE, do not cache

    Compared on ``realpath`` so a symlinked or non-normalised temp dir cannot make one file look
    like two and turn the lie back on.
    """
    if not getattr(cfg, "preprocessing_enabled", False):
        return False
    if not media_for_transcription:
        # Nothing to compare against. Treat as a fallback rather than assume success: a wrongly
        # skipped cache write costs one re-transcription, a wrongly kept one silently defeats the
        # repair this whole epic exists for.
        return True
    try:
        return os.path.realpath(media_for_transcription) == os.path.realpath(temp_media)
    except OSError:
        return True


def _save_transcript_to_cache_if_needed(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    temp_media: str,
    text: str,
    transcription_provider: Any,
    segments: Optional[List[Dict[str, Any]]] = None,
    *,
    media_for_transcription: Optional[str],
) -> None:
    """Save transcript to cache if caching is enabled AND the cache key would be honest (#35).

    Args:
        job: TranscriptionJob with episode info
        cfg: Configuration object
        temp_media: Path to temporary media file (the ORIGINAL download)
        text: Transcribed text
        transcription_provider: Transcription provider instance
        segments: Optional provider segments for GI ``.segments.json`` parity on cache hit
        media_for_transcription: The file the provider ACTUALLY received. Keyword-only and
            required — no default — because a default is exactly how this bug got here: the
            helper was handed ``temp_media`` and had no way to know whether that was what the
            provider saw. A new call site must state it.
    """
    if not (cfg.transcript_cache_enabled and temp_media and os.path.exists(temp_media)):
        return
    if _preprocessing_fingerprint_would_lie(cfg, temp_media, media_for_transcription):
        # The transcript is fine and this run uses it. It just must not be REPLAYED under a key
        # claiming preprocessed audio, or the #18 repair run scores a cache hit on the very
        # transcript it was launched to replace.
        logger.warning(
            "[%s] Not caching transcript: preprocessing was enabled but fell back to raw audio, "
            "so the cache key (%s) would misdescribe what was transcribed. "
            "The transcript is still used for this run.",
            job.idx,
            preprocessing_fingerprint(cfg),
        )
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
            preprocessing=preprocessing_fingerprint(cfg),
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
    from podcast_scraper.utils.provider_metrics import (
        apply_estimated_cost_if_missing,
        record_transcription_cost_to_pipeline,
        transcription_model_for_cfg,
    )

    provider = getattr(cfg, "transcription_provider", None) or "whisper"
    audio_sec = _audio_sec_for_transcription_job(job) if job else None
    audio_min = (audio_sec / 60.0) if audio_sec is not None else None
    apply_estimated_cost_if_missing(
        call_metrics,
        cfg=cfg,
        provider_type=str(provider),
        capability="transcription",
        model=transcription_model_for_cfg(cfg),
        audio_minutes=audio_min,
    )
    # #1523: backstop — record this call's transcription cost onto the run-level metrics so it
    # reaches the manifest cost_rollup. No-op (via the call_metrics latch) when the provider already
    # self-recorded; covers every path where it didn't (duration unknown, deepgram audio<=0 bail).
    record_transcription_cost_to_pipeline(
        pipeline_metrics, call_metrics, audio_min, getattr(call_metrics, "estimated_cost", None)
    )
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


def _mark_episode_skipped_existing(
    episode: Episode,  # type: ignore[valid-type]
    cfg: config.Config,
    pipeline_metrics: Any,
    reason: str,
    *,
    stage: str = "transcription",
) -> None:
    """Record a skip-existing skip as ``skipped`` rather than leaving it untallied (F1/C1).

    Before this, the skip-existing branches returned ``None`` without touching
    ``pipeline_metrics``. A clean all-skip run therefore reported ``{failed: 1}`` and
    failed the Step-0/Step-1 EXIT criteria despite doing exactly the right thing. Only
    the policy-skip and exception paths ever set ``status="skipped"``.

    Never raises: a metrics problem must not turn a successful skip into a failure.
    """
    if pipeline_metrics is None or episode is None:
        return
    try:
        from podcast_scraper.workflow.helpers import (
            get_episode_id_from_episode,
            update_metric_safely,
        )

        episode_id, _ = get_episode_id_from_episode(episode, cfg.rss_url or "")
        pipeline_metrics.update_episode_status(
            episode_id=episode_id,
            status="skipped",
            stage=stage,
            error_type="SkipExisting",
            error_message=redact_for_log(reason, max_len=500),
        )
        update_metric_safely(pipeline_metrics, "episodes_skipped_total", 1)
    except Exception:  # noqa: BLE001 — telemetry must never break a successful skip
        logger.debug("failed to record skip-existing status", exc_info=True)


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


def _resolve_episode_duration_seconds(job) -> Optional[int]:
    """Resolve the episode duration in seconds for cost-metric attribution.

    ``ProcessingJob`` / ``TranscriptionJob`` NamedTuples don't carry
    ``episode_duration_seconds``, so direct attribute access always
    returns ``None``. Fall back to parsing the RSS ``<itunes:duration>``
    element off ``job.episode.item`` (#665 hotfix). Without this, the
    cost recorder's guard (``if audio_minutes > 0``) skips emission
    whenever the provider path is fully mocked and no audio bytes land
    on disk — the default for the e2e provider-mock tests that exercise
    the #650/#651 cost-chain assertions.
    """
    duration = getattr(job, "episode_duration_seconds", None)
    if duration is not None:
        return int(duration) if isinstance(duration, (int, float)) else None
    try:
        from ..rss.parser import _extract_duration_seconds

        ep_item = getattr(getattr(job, "episode", None), "item", None)
        if ep_item is not None:
            parsed = _extract_duration_seconds(ep_item)
            if isinstance(parsed, (int, float)) and parsed > 0:
                return int(parsed)
    except Exception:  # pragma: no cover — defensive: provider will try file-size next
        pass
    return None


# ``moss`` is a local DGX service, not an API, but it shares the same need: its 128k
# context truncates episodes past ~30 min, so it chunks by duration through the same
# AudioChunker path (#1174/#1177). The duration governor lives in
# ``transcription_max_chunk_duration_seconds``; the byte cap is set high for it.
_API_CHUNKING_PROVIDERS = frozenset({"openai", "groq", "gemini", "mistral", "deepgram", "moss"})


def _transcription_provider_supports_chunking(cfg: config.Config) -> bool:
    return cfg.transcription_provider in _API_CHUNKING_PROVIDERS


def _transcribe_with_segments_maybe_chunked(
    media_for_transcription: str,
    *,
    cfg: config.Config,
    job: TranscriptionJob,  # type: ignore[valid-type]
    transcription_provider: Any,
    pipeline_metrics: Any,
    episode_duration_seconds: Optional[float],
    call_metrics: Any,
) -> Tuple[Dict[str, Any], float]:
    """Transcribe media, splitting into chunks when post-preprocess size exceeds API cap."""
    from ..preprocessing.audio.chunker import AudioChunker, transcribe_file_in_chunks
    from ..utils.audio_payload_limits import (
        transcription_max_bytes,
        transcription_max_chunk_duration_seconds,
    )
    from ..utils.timeout import timeout_context, TimeoutError

    # #1046 — when the operator has wired up the sniff-pass gate, run the
    # sniff (cheap) model first and only call the deep (large) model on
    # episodes whose NER density meets the gate threshold. Disabled by
    # default; activation gated on cfg.dgx_whisper_sniff_model being set
    # AND the active provider being tailnet_dgx_whisper. See
    # src/podcast_scraper/workflow/sniff_gate.py + issue #1046.
    from . import sniff_gate as _sniff_gate

    def _transcribe_one(path: str) -> Tuple[Dict[str, Any], float]:
        # The completed result is stashed BEFORE leaving the ``with`` block, and returned from
        # outside it. ``timeout_context`` raises from ``__exit__`` — after the block has already
        # finished — so a ``return`` written inside the block has its value evaluated and then
        # discarded by that exception. The transcript existed and was thrown away: the episode
        # died of a deadline it had already met. Stashing first means an overrun costs a loud
        # log line instead of the work.
        done: Dict[str, Tuple[Dict[str, Any], float]] = {}
        try:
            with timeout_context(cfg.transcription_timeout, f"transcription for episode {job.idx}"):
                if _sniff_gate.is_enabled(cfg):
                    done["r"] = _sniff_gate.transcribe_with_sniff_gate(
                        media_path=path,
                        cfg=cfg,
                        provider=transcription_provider,
                        pipeline_metrics=pipeline_metrics,
                        episode_duration_seconds=episode_duration_seconds,
                        call_metrics=call_metrics,
                    )
                else:
                    result, elapsed = transcription_provider.transcribe_with_segments(
                        path,
                        language=cfg.language,
                        pipeline_metrics=pipeline_metrics,
                        episode_duration_seconds=episode_duration_seconds,
                        call_metrics=call_metrics,
                    )
                    done["r"] = (result, elapsed)
        except TimeoutError:
            if "r" not in done:
                # Nothing was produced — the deadline is not why, but there is no result to
                # keep, so the caller's failure handling is correct here.
                raise
            logger.error(
                "[%s] Transcription OVERRAN its %ss deadline but COMPLETED; keeping the "
                "transcript rather than discarding finished work.",
                job.idx,
                cfg.transcription_timeout,
            )
        return done["r"]

    chunker = AudioChunker(
        # Per-provider byte cap minus 1 MiB headroom (multipart/form overhead),
        # rather than the OpenAI-specific constant for every provider (F1).
        max_bytes=transcription_max_bytes(cfg) - (1024 * 1024),
        max_duration_seconds=transcription_max_chunk_duration_seconds(cfg),
    )
    if _transcription_provider_supports_chunking(cfg) and chunker.needs_chunking(
        media_for_transcription
    ):
        logger.info(
            "[%s] Preprocessed audio exceeds API limit; transcribing in chunks",
            job.idx,
        )
        # D3: Deepgram numbers speakers per request, so chunking fragments speaker
        # ids across chunk seams. The 2 GB cap means this only fires for extreme
        # files; warn so a garbled multi-speaker screenplay isn't a silent mystery.
        if str(getattr(cfg, "transcription_provider", "") or "").lower() == "deepgram":
            logger.warning(
                "[%s] Deepgram audio over the 2 GB single-request cap is being chunked; "
                "diarization speaker ids are chunk-local and not reconciled across chunks.",
                job.idx,
            )
        return transcribe_file_in_chunks(
            media_for_transcription,
            chunker=chunker,
            transcribe_fn=_transcribe_one,
        )

    try:
        return _transcribe_one(media_for_transcription)
    except TimeoutError:
        raise


def _feed_hosts_from_sibling_metadata(txt_path: Path) -> List[str]:
    """Read the feed-stated host names from a transcript's sibling metadata JSON.

    ``<run>/transcripts/<name>.txt`` -> ``<run>/metadata/<name>.metadata.json``. The feed blurb
    usually names the hosts; ``detect_hosts_from_feed`` extracts them. Any failure (missing file,
    malformed JSON, no feed block) returns ``[]`` — the roster then falls back to self-intro only.
    """
    import json as _json

    from ..speaker_detectors.hosts import detect_hosts_from_feed

    stem = txt_path.name[: -len(".txt")] if txt_path.name.endswith(".txt") else txt_path.stem
    md_path = txt_path.parent.parent / filesystem.METADATA_SUBDIR / f"{stem}.metadata.json"
    try:
        feed = _json.loads(md_path.read_text(encoding="utf-8")).get("feed", {})
    except (OSError, ValueError):
        return []
    if not isinstance(feed, dict):
        return []
    return sorted(
        detect_hosts_from_feed(
            feed.get("title"), feed.get("description"), feed.get("authors") or []
        )
    )


def _relabel_existing_transcript(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    run_suffix: Optional[str],
    effective_output_dir: str,
    transcription_provider,
    pipeline_metrics,
) -> tuple[bool, Optional[str], int]:
    """pipeline_stage=relabel_only: re-resolve speaker names on the existing on-disk
    transcript + frozen ``SPEAKER_NN`` diarization, re-render the screenplay, and overwrite
    the transcript / segments / ad-free base in place. No audio, no re-ASR, no re-diarize.
    """
    import json as _json

    from ..providers.ml.diarization.base import DiarizationResult, DiarizationSegment
    from ..providers.ml.diarization.pipeline import apply_diarization_to_result

    # effective_output_dir is the *new* run dir this invocation created; the existing corpus
    # transcript lives in a sibling run_<old-tag>/transcripts/ with a truncated title + run-tag
    # suffix. Search the whole feed root by the unique episode-index prefix ("0001 - "),
    # requiring a .segments.json sibling, and overwrite that file in place.
    run_dir = Path(effective_output_dir)
    search_root = run_dir.parent if run_dir.name.startswith("run_") else run_dir
    idx_prefix = f"{job.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} - "
    matches = [
        p
        for p in search_root.glob(f"**/{filesystem.TRANSCRIPTS_SUBDIR}/{idx_prefix}*.txt")
        if ".adfree." not in p.name
        and p.with_name(p.name[: -len(".txt")] + ".segments.json").exists()
    ]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        logger.warning(
            "[%s] relabel_only: no on-disk transcript to relabel under %s (idx prefix %r)",
            job.idx,
            search_root,
            idx_prefix,
        )
        return False, None, 0
    txt_path = matches[0]
    if len(matches) > 1:
        # The feed root can hold several run_* dirs for the same episode (pilots, prior reprocesses,
        # enrich_only passes). We pick the newest by mtime — surface that choice and the skipped
        # alternates so a relabel that targeted the wrong run is diagnosable (B1).
        logger.warning(
            "[%s] relabel_only: %d on-disk transcripts match idx %r; using newest-mtime %s "
            "(skipped %d older)",
            job.idx,
            len(matches),
            idx_prefix,
            txt_path,
            len(matches) - 1,
        )
    seg_path = txt_path.with_name(txt_path.name[: -len(".txt")] + ".segments.json")
    if not seg_path.exists():
        logger.warning(
            "[%s] relabel_only: transcript has no .segments.json; cannot relabel", job.idx
        )
        return False, None, 0

    text = txt_path.read_text(encoding="utf-8")
    segs = _json.loads(seg_path.read_text(encoding="utf-8"))
    # Frozen-clustering source. A finished corpus stores the diarization identity in
    # ``speaker_label`` (``speaker`` is None), and for already-resolved voices that label is v2's
    # RESOLVED NAME ("Amy Lawrence"), not a raw ``SPEAKER_NN``. Relabel must re-resolve from the
    # transcript, NOT inherit v2's names — so remap every distinct label (name or SPEAKER_NN) to a
    # fresh anonymous ``SPEAKER_NN`` in first-appearance order. This preserves the clustering (same
    # label -> same voice) while stripping v2's naming, so the roster decides afresh.
    _cluster_ids: Dict[str, str] = {}

    def _anon(label: str) -> str:
        if label not in _cluster_ids:
            _cluster_ids[label] = f"SPEAKER_{len(_cluster_ids):02d}"
        return _cluster_ids[label]

    def _identity(s: dict) -> Optional[str]:
        # ``is not None`` so a native diarizer's int-0 speaker id is not dropped by a falsy ``or``.
        v = s.get("speaker")
        if v is None:
            v = s.get("speaker_label")
        return str(v) if v is not None else None

    dsegs = [
        DiarizationSegment(
            start=float(s["start"]),
            end=float(s["end"]),
            speaker=_anon(_identity(s)),  # type: ignore[arg-type]
        )
        for s in segs
        if isinstance(s, dict) and _identity(s) is not None
    ]
    if not dsegs:
        logger.warning(
            "[%s] relabel_only: segments carry no speaker identity; nothing to relabel", job.idx
        )
        return False, None, 0
    diar = DiarizationResult(segments=dsegs, num_speakers=len(_cluster_ids))
    result: dict = {
        "text": text,
        "segments": [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
            for s in segs
            if isinstance(s, dict)
        ],
    }
    # The corpus's own stored metadata carries the feed blurb that names the hosts
    # ("journalists Kevin Roose and Casey Newton explore..."). Feed it to the roster so an
    # ASR-garbled spoken surname ("Kevin Russo") canonicalizes to the feed's spelling. Read from
    # the sibling <run>/metadata/<name>.metadata.json; absence is non-fatal (relabel still runs).
    #
    # Q3 (advisor review): keep the FROZEN sibling metadata as the host anchor — a relabel of a
    # stored corpus must be reproducible, not track live feed drift. But when the sibling is
    # missing/unreadable it silently returns [], the WORST anchor state, while the live detection
    # (job.feed_hosts, statement + NER fallback + corroboration) was already computed — so fall back
    # to it only then. Log any sibling-vs-live divergence so the freeze-vs-live choice can later be
    # decided from data rather than assumption.
    sibling_hosts = _feed_hosts_from_sibling_metadata(txt_path)
    live_hosts = list(getattr(job, "feed_hosts", None) or [])
    feed_hosts = sibling_hosts or live_hosts
    if sibling_hosts and live_hosts and sorted(sibling_hosts) != sorted(live_hosts):
        logger.info(
            "[%s] relabel_only: feed_hosts divergence — sibling-metadata %s vs live %s (using "
            "sibling for a reproducible relabel)",
            job.idx,
            sibling_hosts,
            live_hosts,
        )
    result = apply_diarization_to_result(
        result,
        "",
        cfg,
        job.detected_speaker_names,
        metadata_named=job.metadata_named,
        precomputed_diarization=diar,
        feed_hosts=feed_hosts,
        # ADR-137 — title + description feed the LLM's host/guest role determination and gate
        # role-only resolution. FULL passes both; relabel_only omitting them resolved on a strictly
        # weaker prompt ("(not provided)"), the structural half of the relabel!=full confound.
        episode_title=job.ep_title,
        episode_description=getattr(job.episode, "description", None),
        detection_ran=getattr(job, "speaker_detection_ran", None),
    )
    new_text = _format_transcript_if_needed(
        result, cfg, job.detected_speaker_names, transcription_provider
    )
    txt_path.write_text(new_text, encoding="utf-8")
    rel_path = os.path.relpath(str(txt_path), effective_output_dir)
    new_segs = result.get("segments") if isinstance(result, dict) else None
    if isinstance(new_segs, list) and new_segs:
        _save_transcript_segments_file(new_segs, rel_path, effective_output_dir)
        _save_speaker_diagnostics_file(
            result.get("speaker_diagnostics") if isinstance(result, dict) else None,
            rel_path,
            effective_output_dir,
        )
        _maybe_produce_adfree(cfg, new_text, new_segs, rel_path, effective_output_dir)
    # advisor #2: relabel rewrites naming on disk — the manifest MUST record the new naming
    # method_version, or "reprocess episodes below naming-3" never converges. result has no ASR/
    # diarization fields (frozen), so only the naming block is written + a pipeline_stage emitted.
    _write_processing_manifest(result, cfg, job, rel_path, effective_output_dir)
    logger.info("[%s] relabel_only: re-resolved speaker names in place -> %s", job.idx, rel_path)
    return True, rel_path, 0


def _segments_carry_native_speakers(result: Any) -> bool:
    """True when a transcription provider self-diarized (each segment tagged with a ``speaker`` id)
    but the local roster pass did NOT run — the native-screenplay case (deepgram / moss with
    ``diarize`` off). Gated on the DATA, not a provider name, so it is provider-agnostic and inert
    for non-diarizing providers (whisper/openai segments never carry ``speaker``)."""
    if not isinstance(result, dict):
        return False
    segs = result.get("segments")
    if not isinstance(segs, list):
        return False
    return any(isinstance(s, dict) and s.get("speaker") is not None for s in segs)


def _apply_native_speaker_roster(result: dict, cfg: config.Config, job: Any) -> dict:
    """Route a natively-diarized transcript through the SINGLE role authority (the roster).

    A provider that diarizes server-side (deepgram/moss) tags each segment with a ``speaker`` id
    but assigns names positionally and never computes host/guest roles — the same guess-from-
    ordering class of bug the roster exists to replace. Feed the provider's OWN clustering to the
    roster (``precomputed_diarization`` — no re-diarization, no extra API call) so names AND roles
    come from evidence and land as ``speaker_label`` + ``speaker_role`` on the durable segments,
    exactly like the local-diarizer path.

    No-op unless the segments carry a native ``speaker`` id (provider-agnostic, gated on data). A
    roster failure is swallowed so the successfully-transcribed text is never lost. ``cost_usd=0.0``
    so the reused diarization does not fabricate a cloud-diarizer charge in the manifest."""
    if not _segments_carry_native_speakers(result):
        return result

    from ..exceptions import ProviderDependencyError
    from ..providers.ml.diarization.base import DiarizationResult, DiarizationSegment
    from ..providers.ml.diarization.pipeline import apply_diarization_to_result

    segs = result.get("segments") or []
    cluster_ids: Dict[str, str] = {}

    def _anon(key: str) -> str:
        if key not in cluster_ids:
            cluster_ids[key] = f"SPEAKER_{len(cluster_ids):02d}"
        return cluster_ids[key]

    dsegs = [
        DiarizationSegment(
            start=float(s.get("start") or 0.0),
            end=float(s.get("end") or 0.0),
            speaker=_anon(str(s.get("speaker"))),
        )
        for s in segs
        if isinstance(s, dict) and s.get("speaker") is not None
    ]
    if not dsegs:
        return result
    diar = DiarizationResult(segments=dsegs, num_speakers=len(cluster_ids), cost_usd=0.0)
    # Preserve ASR-provenance top-level fields; only the segments are re-clustered by the roster.
    clean = dict(result)
    clean["segments"] = [
        {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
        for s in segs
        if isinstance(s, dict)
    ]
    try:
        return apply_diarization_to_result(
            clean,
            "",
            cfg,
            job.detected_speaker_names,
            metadata_named=job.metadata_named,
            precomputed_diarization=diar,
            feed_hosts=job.feed_hosts,
            detection_ran=getattr(job, "speaker_detection_ran", None),
        )
    except (ProviderDependencyError, ValueError, OSError, RuntimeError) as exc:
        logger.warning(
            "[%s] Native-speaker roster pass failed; keeping the provider screenplay: %s",
            job.idx,
            format_exception_for_log(exc),
        )
        _capture_stage_exception(exc, stage="diarization")
        return result


def _rediarize_existing_transcript(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    run_suffix: Optional[str],
    effective_output_dir: str,
    transcription_provider,
    pipeline_metrics,
) -> tuple[bool, Optional[str], int]:
    """pipeline_stage=rediarize_only (v2.2): re-diarize the downloaded audio with the profile's
    diarizer (DGX pyannote) and align the FRESH voices to the existing on-disk ASR transcript —
    reusing its text + timestamps, so NO re-ASR — then re-resolve names, re-render the screenplay,
    and overwrite in place. GI/KG cascade. The decoupled sibling of ``relabel_only``: same
    read/render/save machinery, but the diarization is regenerated from audio rather than frozen.
    """
    import json as _json

    from ..providers.ml.diarization.pipeline import apply_diarization_to_result

    audio_path = job.temp_media
    if not audio_path or not os.path.exists(audio_path):
        logger.warning("[%s] rediarize_only: no downloaded audio; cannot re-diarize", job.idx)
        return False, None, 0

    # Locate the existing transcript (same discovery as relabel): unique idx prefix, feed root.
    run_dir = Path(effective_output_dir)
    search_root = run_dir.parent if run_dir.name.startswith("run_") else run_dir
    idx_prefix = f"{job.idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} - "
    matches = [
        p
        for p in search_root.glob(f"**/{filesystem.TRANSCRIPTS_SUBDIR}/{idx_prefix}*.txt")
        if ".adfree." not in p.name
        and p.with_name(p.name[: -len(".txt")] + ".segments.json").exists()
    ]
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        logger.warning(
            "[%s] rediarize_only: no on-disk transcript to align under %s (idx prefix %r)",
            job.idx,
            search_root,
            idx_prefix,
        )
        return False, None, 0
    txt_path = matches[0]
    if len(matches) > 1:
        # Several run_* dirs can match the same episode; we pick the newest by mtime. Surface it and
        # the skipped alternates so a rediarize that targeted the wrong run is diagnosable (B1).
        logger.warning(
            "[%s] rediarize_only: %d on-disk transcripts match idx %r; using newest-mtime %s "
            "(skipped %d older)",
            job.idx,
            len(matches),
            idx_prefix,
            txt_path,
            len(matches) - 1,
        )
    seg_path = txt_path.with_name(txt_path.name[: -len(".txt")] + ".segments.json")

    text = txt_path.read_text(encoding="utf-8")
    segs = _json.loads(seg_path.read_text(encoding="utf-8"))
    # Reuse the existing ASR text + per-segment timestamps; the fresh diarization aligns to THESE.
    result: dict = {
        "text": text,
        "segments": [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "")}
            for s in segs
            if isinstance(s, dict)
        ],
    }
    if not result["segments"]:
        logger.warning("[%s] rediarize_only: transcript has no segments to align", job.idx)
        return False, None, 0

    feed_hosts = _feed_hosts_from_sibling_metadata(txt_path)
    # No precomputed diarization + bypass_cache_read -> apply_diarization diarizes the AUDIO fresh
    # with the profile's diarizer and aligns the new voices to result["segments"] (the ASR text).
    result = apply_diarization_to_result(
        result,
        audio_path,
        cfg,
        job.detected_speaker_names,
        metadata_named=job.metadata_named,
        feed_hosts=feed_hosts,
        bypass_cache_read=True,
    )
    new_text = _format_transcript_if_needed(
        result, cfg, job.detected_speaker_names, transcription_provider
    )
    txt_path.write_text(new_text, encoding="utf-8")
    rel_path = os.path.relpath(str(txt_path), effective_output_dir)
    new_segs = result.get("segments") if isinstance(result, dict) else None
    if isinstance(new_segs, list) and new_segs:
        _save_transcript_segments_file(new_segs, rel_path, effective_output_dir)
        _save_speaker_diagnostics_file(
            result.get("speaker_diagnostics") if isinstance(result, dict) else None,
            rel_path,
            effective_output_dir,
        )
        _maybe_produce_adfree(cfg, new_text, new_segs, rel_path, effective_output_dir)
    # advisor #2: rediarize regenerates diarization + naming on disk — record both into run metrics
    # and the manifest (fresh diarization + naming blocks + pipeline_stage), else the rerun is
    # invisible in diarization_* metrics and the manifest keeps the old versions.
    _record_episode_diarization(pipeline_metrics, result)
    _write_processing_manifest(
        result, cfg, job, rel_path, effective_output_dir, pipeline_metrics=pipeline_metrics
    )
    logger.info("[%s] rediarize_only: re-diarized + re-resolved in place -> %s", job.idx, rel_path)
    return True, rel_path, 0


def _maybe_dispatch_reprocess_stage(
    job: TranscriptionJob,  # type: ignore[valid-type]
    cfg: config.Config,
    run_suffix: Optional[str],
    effective_output_dir: str,
    transcription_provider,
    pipeline_metrics,
) -> Optional[tuple[bool, Optional[str], int]]:
    """Intercept the transcribe stage for the reprocess modes that must NOT re-ASR.

    ``relabel_only`` re-resolves names on frozen diarization; ``rediarize_only`` re-diarizes the
    audio and aligns to the existing transcript. Returns the stage result, or ``None`` to continue
    with normal transcription.
    """
    if cfg.pipeline_stage == "relabel_only":
        return _relabel_existing_transcript(
            job, cfg, run_suffix, effective_output_dir, transcription_provider, pipeline_metrics
        )
    if cfg.pipeline_stage == "rediarize_only":
        return _rediarize_existing_transcript(
            job, cfg, run_suffix, effective_output_dir, transcription_provider, pipeline_metrics
        )
    return None


def _maybe_speech_coverage_failover(
    result: Dict[str, Any],
    media_for_transcription: str,
    cfg: config.Config,
    job: TranscriptionJob,  # type: ignore[valid-type]
    effective_output_dir: str,
    pipeline_metrics: Any,
    episode_duration_seconds: Optional[float],
) -> Dict[str, Any]:
    """ADR-131: re-transcribe on the failover model when the diarized transcript covers too little
    of the diarizer's SPEECH.

    ``speech_coverage = Σ(transcript segments) / Σ(diarization speech)`` (both merged). Unlike the
    raw ADR-123 gate (÷ total audio), music/ads/silence — which produce no diarization turn — are
    excluded, so a music-heavy episode turbo transcribed fully does not falsely failover, while the
    long-episode cliff (turbo dropping real speech a speaker was talking through) still does.

    Provider-agnostic (any ``DiarizationResult``). A **no-op** when there is no speech denominator
    (diarization off, or no speaker turns): the raw ``transcription_coverage_min`` gate governs that
    case instead, so nothing regresses. The failover result carries a ``speech_coverage_failover``
    breadcrumb + ``model_used`` for per-episode provenance.
    """
    min_cov = float(getattr(cfg, "transcription_speech_coverage_min", 0.0) or 0.0)
    fo_model = getattr(cfg, "transcription_coverage_failover_model", None)
    if min_cov <= 0 or not fo_model:
        return result
    speech = float(result.get("diarization_speech_seconds") or 0.0)
    if speech <= 0:
        return result  # no speech denominator — defer to the raw-coverage gate

    from ..providers.ml.diarization.pipeline import (
        apply_diarization_to_result,
        merged_speech_seconds,
    )

    covered = merged_speech_seconds(result.get("segments") or [])
    speech_cov = min(1.0, covered / speech)
    result["asr_speech_coverage"] = round(speech_cov, 3)
    primary_model = getattr(cfg, "dgx_whisper_model", None)
    if speech_cov >= min_cov:
        # Observable pass (ADR-131): log every evaluation so a run shows the gate ran + its
        # coverage, not only the rare failover ("the gate works, it just didn't need to fire").
        logger.info(
            "[%s] speech coverage %.1f%% >= %.1f%% — keeping primary %r (ADR-131 gate passed)",
            job.idx,
            speech_cov * 100,
            min_cov * 100,
            primary_model,
        )
        return result

    logger.info(
        "[%s] speech coverage %.1f%% < %.1f%% — primary %r dropped real speech; "
        "re-transcribing on failover model %s (ADR-131)",
        job.idx,
        speech_cov * 100,
        min_cov * 100,
        primary_model,
        fo_model,
    )
    from ..transcription.factory import create_transcription_provider
    from ..utils.provider_metrics import (
        apply_estimated_cost_if_missing,
        ProviderCallMetrics,
        transcription_model_for_cfg,
    )

    # Route the failover to the configured provider — parity with the ADR-123 raw gate's factory
    # logic (#1273): 'moss' re-transcribes on the DGX MOSS service (:8004) with moss_model, not the
    # speaches whisper service, so the MOSS model id is never sent to a server that cannot serve it.
    # None (default) keeps the historical whisper-on-whisper swap via dgx_whisper_model. Both gates
    # are disabled on the failover pass so it cannot recurse into another re-transcription.
    cov_provider = getattr(cfg, "transcription_coverage_failover_provider", None)
    model_field = "moss_model" if cov_provider == "moss" else "dgx_whisper_model"
    fo_updates: Dict[str, Any] = {
        model_field: fo_model,
        "transcription_coverage_min": 0.0,
        "transcription_speech_coverage_min": 0.0,
    }
    if cov_provider:
        fo_updates["transcription_provider"] = cov_provider
    fo_cfg = cfg.model_copy(update=fo_updates)
    fo_provider = create_transcription_provider(fo_cfg)
    fo_call_metrics = ProviderCallMetrics()
    fo_result, _ = _transcribe_with_segments_maybe_chunked(
        media_for_transcription,
        cfg=fo_cfg,
        job=job,
        transcription_provider=fo_provider,
        pipeline_metrics=pipeline_metrics,
        episode_duration_seconds=episode_duration_seconds,
        call_metrics=fo_call_metrics,
    )
    # RFC-109: the failover re-transcription is a SECOND ASR call. Local models cost 0; a cloud
    # failover model bills again, so its cost must be added to the ASR block (not just the primary).
    apply_estimated_cost_if_missing(
        fo_call_metrics,
        cfg=fo_cfg,
        provider_type=str(getattr(fo_cfg, "transcription_provider", None) or "whisper"),
        capability="transcription",
        model=transcription_model_for_cfg(fo_cfg),
        audio_minutes=(episode_duration_seconds / 60.0) if episode_duration_seconds else None,
    )
    fo_result["asr_failover_cost_usd"] = fo_call_metrics.estimated_cost
    # Diarization is audio-based (cache hit on the same audio) — only the transcript→speaker
    # alignment re-runs, so the failover pays for re-transcription, not re-diarization.
    fo_result = apply_diarization_to_result(
        fo_result,
        media_for_transcription,
        fo_cfg,
        job.detected_speaker_names,
        metadata_named=job.metadata_named,
        cache_dir=os.path.join(effective_output_dir, ".cache", "diarization"),
        feed_hosts=job.feed_hosts,
    )
    fo_speech = float(fo_result.get("diarization_speech_seconds") or speech)
    fo_cov = (
        min(1.0, merged_speech_seconds(fo_result.get("segments") or []) / fo_speech)
        if fo_speech > 0
        else None
    )
    fo_result["model_used"] = fo_result.get("model_used") or f"{fo_model}:speech_coverage_failover"
    if fo_cov is not None:
        fo_result["asr_speech_coverage"] = round(fo_cov, 3)
    fo_result["speech_coverage_failover"] = {
        "primary_model": primary_model,
        "primary_speech_coverage": round(speech_cov, 3),
        "failover_speech_coverage": round(fo_cov, 3) if fo_cov is not None else None,
        "speech_coverage_min": min_cov,
    }
    return fo_result


def _capture_stage_exception(exc: BaseException, *, stage: str) -> None:
    """Best-effort Sentry capture for a swallowed stage failure (o11y P1). Never raises."""
    try:
        from ..utils.sentry_init import capture_stage_exception

        capture_stage_exception(exc, stage=stage)
    except Exception:  # pragma: no cover - telemetry must never break the pipeline
        pass


def _bind_episode_correlation(
    job: TranscriptionJob, cfg: config.Config  # type: ignore[valid-type]
) -> None:
    """#1053 / o11y: bind the correlation episode id for this consumer thread.

    The transcription consumer processes one job then the next, each re-binding at entry, so a bare
    set (no reset) is safe — no id leaks into another episode's emissions. Never blocks on failure.
    """
    try:
        from ..utils import correlation
        from .helpers import get_episode_id_from_episode

        if job.episode is None:
            # No episode → clear, don't inherit the PREVIOUS job's id on this reused thread (#9).
            correlation.set_episode_id(None)
            return
        _corr_ep_id, _ = get_episode_id_from_episode(job.episode, cfg.rss_url or "")
        correlation.set_episode_id(_corr_ep_id)
    except Exception:  # never block transcription on correlation
        pass


def _record_episode_diarization(pipeline_metrics: Any, result: Any) -> None:
    """o11y P1: roll one episode's diarization stats into run-level metrics (metrics.json /
    run.jsonl), where diarization was previously invisible. Detail stays in the manifest. No-op
    when diarization did not run or metrics are absent."""
    if pipeline_metrics is None or not isinstance(result, dict):
        return
    num_spk = result.get("diarization_num_speakers")
    speech_s = result.get("diarization_speech_seconds")
    if num_spk is None and speech_s is None:
        return
    try:
        pipeline_metrics.record_diarization(
            num_speakers=num_spk,
            speech_seconds=speech_s,
            cost_usd=result.get("diarization_cost_usd"),
        )
    except Exception:  # never block transcription on a metrics update
        logger.debug("record_diarization failed", exc_info=True)


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
    # #1053 / o11y: bind the episode id for this consumer thread so ASR / diarization / naming logs,
    # incidents, cost events, and Langfuse spans carry it (see helper).
    _bind_episode_correlation(job, cfg)

    if cfg.dry_run:
        final_path = filesystem.build_whisper_output_path(
            job.idx, job.ep_title_safe, run_suffix, effective_output_dir
        )
        logger.info(f"[{job.idx}] (dry-run) would transcribe media -> {final_path}")
        rel_path = os.path.relpath(final_path, effective_output_dir)
        return True, rel_path, 0

    temp_media = job.temp_media

    # #947 phase 1: download_only. The media has already been downloaded + cached by
    # download_media_for_transcription (via _download_or_reuse_media). Stop here — do NOT
    # transcribe/diarize. This is the "get the audio down first, reprocess later" stage.
    if cfg.pipeline_stage == "download_only":
        bytes_dl = 0
        if temp_media and os.path.exists(temp_media):
            try:
                bytes_dl = os.path.getsize(temp_media)
            except OSError:
                bytes_dl = 0
        logger.info(
            "[%s] [#947] download-only: audio downloaded + cached, skipping transcription (%s)",
            job.idx,
            job.ep_title_safe,
        )
        _cleanup_temp_media(temp_media, cfg)
        return True, None, bytes_dl

    # relabel_only: re-resolve speaker NAMES on the existing on-disk transcript + frozen
    # SPEAKER_NN diarization, then re-render + re-save in place. No audio, no re-ASR, no
    # re-diarize — just the profile's resolver over the existing voices.
    _reprocess = _maybe_dispatch_reprocess_stage(
        job, cfg, run_suffix, effective_output_dir, transcription_provider, pipeline_metrics
    )
    if _reprocess is not None:
        return _reprocess

    # Check if existing transcript can be reused
    reuse_result = _check_and_reuse_existing_transcript(
        job, cfg, run_suffix, effective_output_dir, pipeline_metrics
    )
    if reuse_result:
        return reuse_result

    # Transcript cache before requiring a provider (cache hit skips API and keeps download at 0).
    cache_result = _check_transcript_cache(
        job,
        cfg,
        temp_media,
        run_suffix,
        effective_output_dir,
        pipeline_metrics,
        transcription_provider=transcription_provider,
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

    # Audio preprocessing: Preprocess audio BEFORE passing to any provider
    # This happens at the pipeline level, not within providers
    # All providers receive optimized audio (Whisper, OpenAI, future providers)
    media_for_transcription = _preprocess_audio_if_needed(job, cfg, temp_media, pipeline_metrics)

    try:
        # Stage 2: Use provider's transcribe_with_segments (chunked when over API cap)
        episode_duration_seconds = _resolve_episode_duration_seconds(job)
        from ..utils.provider_metrics import ProviderCallMetrics
        from ..utils.timeout import TimeoutError

        call_metrics = ProviderCallMetrics()

        try:
            result, tc_elapsed = _transcribe_with_segments_maybe_chunked(
                media_for_transcription,
                cfg=cfg,
                job=job,
                transcription_provider=transcription_provider,
                pipeline_metrics=pipeline_metrics,
                episode_duration_seconds=episode_duration_seconds,
                call_metrics=call_metrics,
            )
        except TimeoutError as e:
            logger.error(
                f"[{job.idx}] Transcription timeout after {cfg.transcription_timeout}s: {e}"
            )
            raise
        if cfg.diarize:
            from ..exceptions import ProviderDependencyError
            from ..providers.ml.diarization.pipeline import apply_diarization_to_result

            try:
                result = apply_diarization_to_result(
                    result,
                    media_for_transcription,
                    cfg,
                    job.detected_speaker_names,
                    metadata_named=job.metadata_named,
                    cache_dir=os.path.join(effective_output_dir, ".cache", "diarization"),
                    feed_hosts=job.feed_hosts,
                    # ADR-137 — title + description feed the LLM's host/guest role determination.
                    episode_title=job.ep_title,
                    episode_description=getattr(job.episode, "description", None),
                )
            except (ProviderDependencyError, ValueError, OSError, RuntimeError) as exc:
                # Broadened catch (Whisper-e2e diagnosis, #1180 follow-up).
                # A diarization failure MUST NOT lose the successfully-computed
                # transcript. HuggingFace Hub errors from an uncached pyannote
                # model raise OSError; torch/pyannote instantiation issues raise
                # RuntimeError; both were previously escaping to the outer
                # transcription except clause and getting mislabeled as
                # "Whisper transcription failed" while dropping the transcript.
                logger.warning(
                    "[%s] Diarization failed; falling back to gap-based screenplay: %s",
                    job.idx,
                    format_exception_for_log(exc),
                )
                _capture_stage_exception(exc, stage="diarization")
            # ADR-131: speech-normalized quality gate — re-transcribe on the failover model if the
            # diarized transcript covers too little of the diarizer's SPEECH (music/ads excluded).
            # No-op unless configured + diarization produced a speech denominator.
            result = _maybe_speech_coverage_failover(
                result,
                media_for_transcription,
                cfg,
                job,
                effective_output_dir,
                pipeline_metrics,
                episode_duration_seconds,
            )
        else:
            # Native-screenplay provider (deepgram/moss) may have self-diarized without a roster
            # pass — route it through the single role authority (no-op otherwise). Guard + failure
            # handling live in the helper so this stays one branch.
            result = _apply_native_speaker_roster(result, cfg, job)
        _attach_speech_audio_ratio(result, media_for_transcription, episode_duration_seconds)
        text = _format_transcript_if_needed(
            result, cfg, job.detected_speaker_names, transcription_provider
        )
        rel_path = _save_transcript_file(
            text, job, run_suffix, effective_output_dir, pipeline_metrics=pipeline_metrics
        )
        logger.info(f"    saved transcript: {rel_path} (transcribed in {tc_elapsed:.1f}s)")
        # ADR-131: per-episode ASR provenance (actual model + speech coverage), incl. any failover.
        _save_asr_provenance_file(result, cfg, rel_path, effective_output_dir)
        segments = result.get("segments") if isinstance(result, dict) else None
        if isinstance(segments, list) and len(segments) > 0:
            _save_transcript_segments_file(segments, rel_path, effective_output_dir)
            _save_speaker_diagnostics_file(
                result.get("speaker_diagnostics") if isinstance(result, dict) else None,
                rel_path,
                effective_output_dir,
            )
            # #974: derive the ad-free processing-base sibling. Raw .txt left untouched.
            _maybe_produce_adfree(cfg, text, segments, rel_path, effective_output_dir)

        _maybe_persist_episode_media(
            cfg, temp_media, effective_output_dir, rel_path, episode=job.episode
        )

        # Save transcript to cache for future use (enables fast multi-provider experimentation)
        _save_transcript_to_cache_if_needed(
            job,
            cfg,
            temp_media,
            text,
            transcription_provider,
            segments=segments if isinstance(segments, list) else None,
            # What the provider ACTUALLY received, which is not always what preprocessing was
            # asked to produce (#35).
            media_for_transcription=media_for_transcription,
        )

        # Record transcription time if metrics available
        _record_transcription_metrics(job, cfg, tc_elapsed, call_metrics, pipeline_metrics)

        # o11y P1: roll this episode's diarization stats into run-level metrics (see helper).
        _record_episode_diarization(pipeline_metrics, result)

        # RFC-109 / ADR-132: per-episode processing manifest (ASR/diarization/naming stage blocks).
        # After metrics recording so the ASR ``estimated_cost`` (cloud providers) is populated.
        _write_processing_manifest(
            result,
            cfg,
            job,
            rel_path,
            effective_output_dir,
            asr_elapsed=tc_elapsed,
            asr_call_metrics=call_metrics,
            pipeline_metrics=pipeline_metrics,
        )

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
        _capture_stage_exception(exc, stage="transcription")
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


def _episode_existing_transcript_source(
    episode: Episode,  # type: ignore[valid-type]
    effective_output_dir: str,
    run_suffix: Optional[str],
    cfg: config.Config,
) -> Optional[str]:
    """Return the episode's existing ``content.transcript_source`` from its on-disk
    metadata, or None if absent/unreadable (#925). Handles both JSON and YAML
    metadata (``_determine_metadata_path`` returns ``.metadata.yaml`` when
    ``metadata_format == 'yaml'``).

    RESOLVES CORPUS-WIDE under corpus layout. ``_determine_metadata_path`` builds a path inside
    THIS run's directory, and under ``--single-feed-uses-corpus-layout`` every run gets a fresh
    ``run_<ts>/`` — so the prior metadata lives in a different run dir, the open raises, this
    returns None, and ``_force_reprocess_for_source`` concludes the episode does not match.

    The effect was that ``--reprocess-source`` NEVER FIRED on a corpus: the "#925 forcing
    re-transcription" branch is unreachable and every episode falls through to the ordinary
    "transcript already exists; skipping" path. ``make redo-diarization`` is built entirely on
    that flag, so it reported success and re-diarized nothing. Verified 2026-08-16 on a corpus
    copy whose metadata declared ``transcript_source: whisper_transcription`` while
    ``--reprocess-source whisper_transcription`` was passed: the forcing log line never appeared.

    This is the same defect the TRANSCRIPT lookup already fixed (see the D7 note above
    ``existing_transcript_path_in_corpus``); the metadata lookup never got the same treatment.
    """
    from .metadata_generation import _determine_metadata_path  # local: avoid import cycle

    metadata_path: Optional[str] = None
    if getattr(cfg, "single_feed_uses_corpus_layout", False) and cfg.output_dir:
        from . import run_index

        meta_rel = run_index.episode_metadata_rel_in_corpus(episode, str(cfg.output_dir))
        if meta_rel:
            metadata_path = os.path.join(str(cfg.output_dir), meta_rel)

    try:
        if metadata_path is None:
            metadata_path = _determine_metadata_path(episode, effective_output_dir, run_suffix, cfg)
        with open(metadata_path, "r", encoding="utf-8") as fh:
            if metadata_path.endswith((".yaml", ".yml")):
                import yaml

                data = yaml.safe_load(fh)
            else:
                data = json.load(fh)
    except (OSError, ValueError, KeyError, AttributeError):
        return None
    content = data.get("content") if isinstance(data, dict) else None
    src = content.get("transcript_source") if isinstance(content, dict) else None
    return src if isinstance(src, str) else None


def _force_reprocess_for_source(
    episode: Episode,  # type: ignore[valid-type]
    effective_output_dir: str,
    run_suffix: Optional[str],
    cfg: config.Config,
) -> bool:
    """True when this episode must be forced back through download+transcribe, overriding
    ``--skip-existing`` for it alone (re-runs diarization under the profile and cascades
    GI/KG/CIL).

    TWO selection modes, checked in order:

    1. ``--reprocess-episode-ids`` (#32) — an EXPLICIT list. Needed because the damage that
       motivates a re-transcription is usually not expressible as a transcript_source. Measured
       2026-08-17: every episode in a corpus carrying #18 damage had
       ``transcript_source: whisper_transcription`` — and so did every HEALTHY one. Selecting by
       source there would re-transcribe 6 healthy episodes to reach 9 damaged ones. A detector
       that can only produce a list needs a selector that can consume one.
    2. ``--reprocess-source`` (#925) — matches the recorded ``transcript_source``. Right tool
       when the whole class needs redoing (e.g. re-diarizing every whisper-sourced episode).
    """
    wanted_ids = getattr(cfg, "reprocess_episode_ids", None) or ()
    if wanted_ids:
        for candidate in _episode_identity_candidates(episode, effective_output_dir, cfg):
            if candidate in wanted_ids:
                return True

    target = getattr(cfg, "reprocess_source", None)
    if not target:
        return False
    existing = _episode_existing_transcript_source(episode, effective_output_dir, run_suffix, cfg)
    return bool(existing == target)


def _episode_identity_candidates(
    episode: Episode,  # type: ignore[valid-type]
    effective_output_dir: str,
    cfg: config.Config,
) -> Set[str]:
    """Every id this episode could legitimately be named by in a work-list.

    Detectors emit whatever the artifact carries — ``episode_id`` from the metadata, or the RSS
    ``guid``. Matching on only one of them makes an operator's list silently miss episodes, which
    for a repair work-list is the worst possible failure: it looks like the episode was already
    fine.
    """
    out: Set[str] = set()
    for attr in ("episode_id", "guid"):
        value = getattr(episode, attr, None)
        if isinstance(value, str) and value.strip():
            out.add(value.strip())

    from . import run_index

    guid = run_index._episode_guid(episode)
    if guid:
        out.add(guid)
        if cfg.output_dir:
            entry = run_index.corpus_metadata_index(str(cfg.output_dir))["by_guid"].get(guid)
            if entry is not None and entry.episode_id:
                out.add(str(entry.episode_id))
    return out


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
    # #925: a scoped reprocess (--reprocess-source) forces matching episodes
    # through transcription again so diarization re-runs (and the downstream
    # GI/KG/CIL cascade with it), instead of being skipped by --skip-existing.
    if _force_reprocess_for_source(episode, effective_output_dir, run_suffix, cfg):
        logger.info(
            "    [#925] forcing re-transcription (reprocess-source=%s): %s",
            cfg.reprocess_source,
            episode.title_safe,
        )
        return False

    # D7: under --single-feed-uses-corpus-layout each run writes a FRESH run dir, so the episode's
    # transcript lives in a PRIOR run dir, not effective_output_dir. Resolve presence corpus-wide by
    # stable guid (all feeds/runs) — else skip-existing scoped to the empty run dir silently
    # re-transcribes an already-present episode (the Step-1 NO-GO, 2026-08-11).
    if getattr(cfg, "single_feed_uses_corpus_layout", False) and cfg.output_dir:
        present = run_index.episode_metadata_rel_in_corpus(episode, str(cfg.output_dir))
        if present:
            prefix = "[dry-run] " if cfg.dry_run else ""
            logger.info(
                "    %salready present in corpus, skipping (--skip-existing): %s", prefix, present
            )
            return True
        return False

    run_tag = f"_{run_suffix}" if run_suffix else ""
    # Key on the STABLE guid, not the run-local idx (which shifts when the feed grows → silent
    # reprocess + duplicates). New episodes fall back to their run-local idx.
    skip_idx = run_index.resolve_ondisk_idx_for_episode(episode, effective_output_dir)
    base_name = (
        f"{skip_idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} - {episode.title_safe}{run_tag}"
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
            # Find existing transcript file to return its path for summarization.
            # Resolve the on-disk idx by STABLE guid — episode.idx shifts when the feed grows, so
            # rebuilding the glob from it would miss the transcript and silently skip summarization
            # for an already-present episode (the same P0.1 drift, one branch missed; Fable-5 #2).
            run_tag = f"_{run_suffix}" if run_suffix else ""
            skip_idx = run_index.resolve_ondisk_idx_for_episode(episode, effective_output_dir)
            base_name = (
                f"{skip_idx:0{filesystem.EPISODE_NUMBER_FORMAT_WIDTH}d} "
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
                    # success=True so the result handlers ENQUEUE the summarization
                    # / metadata ProcessingJob — they gate that on `if success`, and
                    # returning False here silently skipped summarization for every
                    # direct-download episode with a pre-existing transcript (review
                    # 2026-07-17 H4). Cost: transcripts_downloaded is +1 high for a
                    # reused transcript (cosmetic metric only).
                    return True, rel_path, "direct_download", 0
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
            _maybe_produce_adfree(cfg, plain, segments, rel_path_result, effective_output_dir)
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
    metadata_named: Optional[List[str]] = None,
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
            metadata_named=metadata_named,
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
