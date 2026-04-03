"""Transcript chunking for semantic indexing (RFC-061 §3 / #484 Step 2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class TranscriptChunk:
    """One overlapping window of transcript text with source offsets."""

    text: str
    chunk_index: int
    char_start: int
    char_end: int
    timestamp_start_ms: Optional[int] = None
    timestamp_end_ms: Optional[int] = None


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PAR_SPLIT = re.compile(r"\n\s*\n+")


def _token_count(text: str) -> int:
    return len(text.split())


def _sentences_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Split on sentence-ending punctuation; return (sentence, char_start, char_end)."""
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    out: List[Tuple[str, int, int]] = []
    search_from = 0
    for part in parts:
        stripped = part.strip()
        if not stripped:
            search_from = min(search_from + len(part), len(text))
            continue
        idx = text.find(stripped, search_from)
        if idx < 0:
            idx = text.find(stripped)
        if idx < 0:
            continue
        end = idx + len(stripped)
        out.append((stripped, idx, end))
        search_from = end
    if not out and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        if start < 0:
            start = 0
        end = start + len(stripped)
        out.append((stripped, start, end))
    return out


def _paragraph_spans(text: str) -> List[Tuple[str, int, int]]:
    """Fallback when punctuation splitting yields a single long span."""
    out: List[Tuple[str, int, int]] = []
    pos = 0
    for m in _PAR_SPLIT.finditer(text):
        block = text[pos : m.start()]
        stripped = block.strip()
        if stripped:
            rel = block.find(stripped)
            start = pos + (rel if rel >= 0 else 0)
            out.append((stripped, start, start + len(stripped)))
        pos = m.end()
    tail = text[pos:]
    stripped = tail.strip()
    if stripped:
        rel = tail.find(stripped)
        start = pos + (rel if rel >= 0 else 0)
        out.append((stripped, start, start + len(stripped)))
    return out


def _merge_time_for_span(
    char_start: int,
    char_end: int,
    segments: Sequence[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int]]:
    """Map chunk char span to ms using segment dicts (optional char + time fields)."""
    if not segments:
        return None, None
    starts: List[int] = []
    ends: List[int] = []
    for seg in segments:
        cs = seg.get("char_start")
        ce = seg.get("char_end")
        sm = seg.get("start_ms")
        em = seg.get("end_ms")
        if sm is None:
            sm = seg.get("timestamp_start_ms")
        if em is None:
            em = seg.get("timestamp_end_ms")
        try:
            cs_i = int(cs) if cs is not None else None
            ce_i = int(ce) if ce is not None else None
            sm_i = int(sm) if sm is not None else None
            em_i = int(em) if em is not None else None
        except (TypeError, ValueError):
            continue
        if cs_i is None or ce_i is None or sm_i is None or em_i is None:
            continue
        if ce_i <= char_start or cs_i >= char_end:
            continue
        starts.append(sm_i)
        ends.append(em_i)
    if not starts:
        return None, None
    return min(starts), max(ends)


def chunk_transcript(
    text: str,
    target_tokens: int = 300,
    overlap_tokens: int = 50,
    timestamps: Optional[List[Dict[str, Any]]] = None,
) -> List[TranscriptChunk]:
    """Split transcript into overlapping sentence-based chunks (RFC-061).

    Sentences are split on ``.?!`` followed by whitespace. If that yields at most
    one span and the transcript is long, paragraph boundaries (blank lines) are
    used as a secondary split.

    Args:
        text: Full transcript.
        target_tokens: Soft target chunk size (whitespace token count).
        overlap_tokens: Approximate token overlap carried into the next chunk.
        timestamps: Optional segment dicts with ``char_start``, ``char_end``,
            and ``start_ms`` / ``end_ms`` (or ``timestamp_*``) for interpolation.

    Returns:
        Ordered chunks with inclusive-exclusive ``char_start`` / ``char_end`` in
        the original ``text`` coordinate space.
    """
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= target_tokens:
        raise ValueError("overlap_tokens must be less than target_tokens")

    if not text or not text.strip():
        return []

    sentences = _sentences_with_spans(text)
    if len(sentences) <= 1 and _token_count(text) > target_tokens:
        para = _paragraph_spans(text)
        if len(para) > 1:
            sentences = para
        elif sentences and _token_count(sentences[0][0]) > target_tokens * 2:
            # Very long single sentence: split on single newlines as last resort.
            lines = []
            pos = 0
            for line in text.split("\n"):
                stripped = line.strip()
                if stripped:
                    idx = text.find(stripped, pos)
                    if idx < 0:
                        idx = text.find(stripped)
                    if idx >= 0:
                        lines.append((stripped, idx, idx + len(stripped)))
                pos += len(line) + 1
            if len(lines) > 1:
                sentences = lines

    if not sentences:
        return []

    chunks: List[TranscriptChunk] = []
    i = 0
    chunk_index = 0
    n = len(sentences)

    while i < n:
        tok = 0
        j = i
        while j < n and (tok < target_tokens or j == i):
            tok += _token_count(sentences[j][0])
            j += 1
            if tok >= target_tokens:
                break

        sent_slice = sentences[i:j]
        joined = " ".join(s[0] for s in sent_slice)
        c_start = sent_slice[0][1]
        c_end = sent_slice[-1][2]
        ts0, ts1 = _merge_time_for_span(c_start, c_end, timestamps or [])
        chunks.append(
            TranscriptChunk(
                text=joined,
                chunk_index=chunk_index,
                char_start=c_start,
                char_end=c_end,
                timestamp_start_ms=ts0,
                timestamp_end_ms=ts1,
            )
        )
        chunk_index += 1

        if j >= n:
            break

        otok = 0
        k = j - 1
        while k >= i and otok < overlap_tokens:
            otok += _token_count(sentences[k][0])
            k -= 1
        i_next = k + 1
        if i_next <= i:
            i_next = i + 1
        if i_next >= j:
            i_next = j
        i = i_next

    return chunks
