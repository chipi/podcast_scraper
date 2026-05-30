"""Split oversized audio files for API transcription providers."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from podcast_scraper.preprocessing.audio.ffmpeg_processor import _run_text_subprocess
from podcast_scraper.rss.downloader import OPENAI_MAX_FILE_SIZE_BYTES

logger = logging.getLogger(__name__)

DEFAULT_OVERLAP_SECONDS = 5.0
DEFAULT_MAX_BYTES = OPENAI_MAX_FILE_SIZE_BYTES - (1024 * 1024)


@dataclass(frozen=True)
class AudioChunk:
    """One segment of a split audio file."""

    path: str
    start_seconds: float
    index: int


def _probe_duration_seconds(audio_path: str) -> Optional[float]:
    """Return media duration in seconds via ffprobe."""
    if not shutil.which("ffprobe"):
        logger.warning("ffprobe not available; cannot chunk audio")
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        audio_path,
    ]
    try:
        result = _run_text_subprocess(cmd, timeout=30.0, check=True)
        data = json.loads(result.stdout)
        duration = float(data.get("format", {}).get("duration", 0))
        return duration if duration > 0 else None
    except (subprocess.CalledProcessError, ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Failed to probe audio duration for chunking: %s", exc)
        return None


def _dedupe_overlap_text(prev_text: str, next_text: str, *, overlap_seconds: float) -> str:
    """Remove duplicated words at chunk boundary using suffix/prefix overlap."""
    if not prev_text or not next_text:
        return next_text

    prev_words = prev_text.split()
    next_words = next_text.split()
    max_overlap_words = max(3, int(overlap_seconds * 2.5))
    best = 0
    for size in range(min(max_overlap_words, len(prev_words), len(next_words)), 0, -1):
        if prev_words[-size:] == next_words[:size]:
            best = size
            break
    if best:
        return " ".join(next_words[best:])
    return next_text


class AudioChunker:
    """Split long audio files into provider-sized chunks with overlap."""

    def __init__(
        self,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    ) -> None:
        self.max_bytes = max_bytes
        self.overlap_seconds = overlap_seconds

    def needs_chunking(self, audio_path: str) -> bool:
        """Return True when file exceeds configured byte limit."""
        try:
            return os.path.getsize(audio_path) > self.max_bytes
        except OSError:
            return False

    def split(self, audio_path: str, work_dir: Optional[str] = None) -> List[AudioChunk]:
        """Split audio into chunks using ffmpeg stream copy."""
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not available for audio chunking")

        file_size = os.path.getsize(audio_path)
        duration = _probe_duration_seconds(audio_path)
        if duration is None or duration <= 0:
            raise RuntimeError(f"Could not determine duration for chunking: {audio_path}")

        chunk_duration = duration * (self.max_bytes / file_size) * 0.95
        chunk_duration = max(chunk_duration, self.overlap_seconds + 1.0)
        step = max(chunk_duration - self.overlap_seconds, 1.0)
        num_chunks = max(1, math.ceil((duration - self.overlap_seconds) / step))

        out_dir = work_dir or tempfile.mkdtemp(prefix="audio_chunks_")
        os.makedirs(out_dir, exist_ok=True)
        chunks: List[AudioChunk] = []

        for index in range(num_chunks):
            start = index * step
            if start >= duration:
                break
            end = min(duration, start + chunk_duration)
            out_path = os.path.join(out_dir, f"chunk_{index:03d}.mp3")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-to",
                str(end),
                "-i",
                audio_path,
                "-c",
                "copy",
                out_path,
            ]
            _run_text_subprocess(cmd, timeout=120.0, check=True)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                chunks.append(AudioChunk(path=out_path, start_seconds=start, index=index))

        if not chunks:
            raise RuntimeError(f"ffmpeg produced no audio chunks for {audio_path}")
        return chunks

    def merge_transcript_results(
        self,
        chunk_results: Sequence[Tuple[Dict[str, Any], float]],
        chunks: Sequence[AudioChunk],
    ) -> Tuple[Dict[str, Any], float]:
        """Merge per-chunk transcription dicts into one result with time offsets."""
        if not chunk_results:
            return {"text": "", "segments": []}, 0.0

        merged_segments: List[Dict[str, Any]] = []
        merged_text_parts: List[str] = []
        total_elapsed = 0.0

        for (result, elapsed), chunk in zip(chunk_results, chunks):
            total_elapsed += elapsed
            offset = chunk.start_seconds
            text = str(result.get("text") or "").strip()
            if merged_text_parts and text:
                text = _dedupe_overlap_text(
                    merged_text_parts[-1], text, overlap_seconds=self.overlap_seconds
                )
            if text:
                merged_text_parts.append(text)

            segments = result.get("segments")
            if not isinstance(segments, list):
                continue
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                start = float(seg.get("start", 0)) + offset
                end = float(seg.get("end", start)) + offset
                seg_text = str(seg.get("text") or "").strip()
                if not seg_text:
                    continue
                merged: Dict[str, Any] = {
                    "start": start,
                    "end": end,
                    "text": seg_text,
                }
                if "speaker" in seg:
                    merged["speaker"] = seg["speaker"]
                merged_segments.append(merged)

        merged_segments.sort(key=lambda s: (float(s["start"]), float(s["end"])))
        full_text = " ".join(merged_text_parts).strip()
        if not full_text and merged_segments:
            full_text = " ".join(str(s.get("text", "")) for s in merged_segments).strip()

        return {"text": full_text, "segments": merged_segments}, total_elapsed


def transcribe_file_in_chunks(
    audio_path: str,
    *,
    chunker: AudioChunker,
    transcribe_fn: Callable[[str], Tuple[Dict[str, Any], float]],
    work_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], float]:
    """Split audio, transcribe each chunk, merge results."""
    chunks = chunker.split(audio_path, work_dir=work_dir)
    chunk_dir = os.path.dirname(chunks[0].path)
    try:
        results: List[Tuple[Dict[str, Any], float]] = []
        for chunk in chunks:
            logger.info(
                "Transcribing audio chunk %d (start=%.1fs): %s",
                chunk.index + 1,
                chunk.start_seconds,
                chunk.path,
            )
            results.append(transcribe_fn(chunk.path))
        return chunker.merge_transcript_results(results, chunks)
    finally:
        if work_dir is None and os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir, ignore_errors=True)
