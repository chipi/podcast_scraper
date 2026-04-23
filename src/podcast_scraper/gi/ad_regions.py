"""Pre-extraction transcript ad-region detection + excision (#663 option 2).

The ``gi.filters`` post-extraction filter catches 0/1200 insights on real
corpora because the LLM paraphrases sponsor reads into generic-sounding
claims (``"Ramp saves companies 5%"``) that carry no ad markers — by the
time a filter sees insight text, the signatures are gone. The only layer
that can reliably prevent ad content from reaching GI/KG/summary artifacts
is **before** the LLM reads the transcript.

This module implements a **position-scoped** detector: it scans only the
first and last ``SCAN_CHARS`` of the transcript (where pre-rolls and
post-rolls live) and requires ``THRESHOLD`` distinct ad-pattern hits to
declare a region. Mid-transcript ads are **not** targeted — generic
sliding-window detection produced ~37% false positives on content regions
in the ``my-manual-run4`` sweep (see
``scripts/validate/sweep_transcript_ad_regions.py``), so we stay out of
the middle entirely.

Public API:

* :func:`detect_preroll_ad_end` — returns char position where a detected
  pre-roll ends, or ``None``.
* :func:`detect_postroll_ad_start` — returns char position where a
  detected post-roll starts, or ``None``.
* :func:`excise_ad_regions` — returns the cleaned transcript, optionally
  re-aligned segments, and metadata describing what was cut (used by
  callers that want to record observability telemetry).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .filters import _AD_PATTERNS

logger = logging.getLogger(__name__)

SCAN_CHARS = 5000
PREROLL_THRESHOLD = 3
POSTROLL_THRESHOLD = 3
# Below this length, the "transcript" is almost certainly a test fixture or
# a heavily-preprocessed snippet — running pre-roll/post-roll detection on
# it produces nonsense (scan window = whole input, so both ends collapse).
MIN_TRANSCRIPT_CHARS = 2000
# Ad-pattern hits must fall inside a window of this size to count as a
# *single* pre-roll / post-roll block. Planet Money-style shows scatter
# discrete short ads across the first 4,000 chars separated by legitimate
# content; cutting the full span would delete the interleaved content. A
# tight cluster cap keeps excision focused on genuinely contiguous ad
# blocks like the Invest-Like-the-Best pre-roll stack.
MAX_AD_CLUSTER_SPAN = 2000
# After the last ad-pattern hit, extend the cut forward (pre-roll) or
# backward (post-roll) to the next sentence terminator so we don't leave
# ragged mid-sentence fragments on the content side.
SENTENCE_TERMINATORS = (". ", "! ", "? ")
SENTENCE_BOUNDARY_LOOKAHEAD = 300


@dataclass
class AdRegionMetadata:
    """Describes what ``excise_ad_regions`` did on one transcript — cut
    positions, pattern hit counts, and the resulting excised char ranges
    — so callers can log / surface telemetry without re-running the
    detector."""

    preroll_cut_end: Optional[int] = None
    postroll_cut_start: Optional[int] = None
    chars_removed: int = 0
    preroll_pattern_hits: int = 0
    postroll_pattern_hits: int = 0
    source_length: int = 0
    excised_ranges: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise as JSON-friendly plain dict (for manifest / logs)."""
        return {
            "preroll_cut_end": self.preroll_cut_end,
            "postroll_cut_start": self.postroll_cut_start,
            "chars_removed": self.chars_removed,
            "preroll_pattern_hits": self.preroll_pattern_hits,
            "postroll_pattern_hits": self.postroll_pattern_hits,
            "source_length": self.source_length,
            "excised_ranges": [list(r) for r in self.excised_ranges],
        }


def _distinct_hits(text: str) -> List[Tuple[int, int]]:
    """Return sorted ``(start, end)`` positions of each distinct pattern
    that matches in ``text`` (at most one match per pattern)."""
    hits: List[Tuple[int, int]] = []
    for pat in _AD_PATTERNS:
        m = pat.search(text)
        if m:
            hits.append((m.start(), m.end()))
    hits.sort()
    return hits


def _snap_forward_to_sentence_end(text: str, pos: int) -> int:
    """Move ``pos`` forward to the end of the current sentence, bounded by
    ``SENTENCE_BOUNDARY_LOOKAHEAD`` chars. Falls back to ``pos`` if no
    terminator is found — better to keep a trailing fragment of ad than
    risk cutting into content mid-sentence.
    """
    window = text[pos : pos + SENTENCE_BOUNDARY_LOOKAHEAD]
    best = -1
    for term in SENTENCE_TERMINATORS:
        idx = window.find(term)
        if idx >= 0 and (best < 0 or idx < best):
            best = idx + len(term)
    if best < 0:
        return pos
    return pos + best


def _snap_backward_to_sentence_start(text: str, pos: int) -> int:
    """Move ``pos`` backward to the start of the current sentence (char
    after the previous sentence terminator). Bounded by
    ``SENTENCE_BOUNDARY_LOOKAHEAD`` chars backward.
    """
    lo = max(0, pos - SENTENCE_BOUNDARY_LOOKAHEAD)
    window = text[lo:pos]
    best = -1
    for term in SENTENCE_TERMINATORS:
        idx = window.rfind(term)
        if idx >= 0 and idx > best:
            best = idx + len(term)
    if best < 0:
        return pos
    return lo + best


def _hits_are_clustered(hits: List[Tuple[int, int]], threshold: int) -> bool:
    """Return True when ≥ ``threshold`` pattern hits exist AND the first
    and last hit are within ``MAX_AD_CLUSTER_SPAN`` chars of each other
    — i.e., the hits plausibly describe a single contiguous ad block.

    Rejects Planet Money-style episodes where 3 short ads are spread
    across ~4,000 chars interleaved with legitimate content: the
    first-to-last span is too wide to be a coherent pre-roll, so we
    decline to cut and keep the content intact.
    """
    if len(hits) < threshold:
        return False
    first_start = min(start for start, _ in hits)
    last_end = max(end for _, end in hits)
    return (last_end - first_start) <= MAX_AD_CLUSTER_SPAN


def detect_preroll_ad_end(
    text: str,
    *,
    scan_chars: int = SCAN_CHARS,
    threshold: int = PREROLL_THRESHOLD,
) -> Optional[int]:
    """Return the char position where a detected pre-roll region ends.

    Scans only the first ``scan_chars`` of the transcript. If ≥ ``threshold``
    distinct ad patterns match AND the first-to-last hit span fits inside
    ``MAX_AD_CLUSTER_SPAN`` chars, the returned end position is the end of
    the last matching phrase, snapped forward to the next sentence
    terminator. Scattered hits (e.g., Planet Money-style short ads
    separated by real content) are intentionally ignored. ``None`` if no
    coherent pre-roll is detected.
    """
    if not text:
        return None
    prefix = text[:scan_chars]
    hits = _distinct_hits(prefix)
    if not _hits_are_clustered(hits, threshold):
        return None
    last_end = max(end for _, end in hits)
    return _snap_forward_to_sentence_end(text, last_end)


def _expand_postroll_backward(
    text: str,
    cut_pos: int,
    floor: int,
    *,
    chunk_size: int = 500,
    max_iterations: int = 4,
) -> int:
    """Iteratively extend the post-roll cut backward one chunk at a time
    while the preceding ``chunk_size`` window still contains at least one
    ``_AD_PATTERNS`` hit. Stops when a chunk is clean or when we reach
    ``floor``. Ad blocks typically run for multiple sentences (product
    pitch → call-to-action → URL) where only the CTA sentence carries an
    explicit pattern — this expansion catches the full block without
    over-reaching into content.
    """
    pos = cut_pos
    for _ in range(max_iterations):
        chunk_start = max(floor, pos - chunk_size)
        if chunk_start >= pos:
            break
        chunk = text[chunk_start:pos]
        any_hit = any(pat.search(chunk) for pat in _AD_PATTERNS)
        if not any_hit:
            break
        pos = chunk_start
    return _snap_backward_to_sentence_start(text, pos)


def detect_postroll_ad_start(
    text: str,
    *,
    scan_chars: int = SCAN_CHARS,
    threshold: int = POSTROLL_THRESHOLD,
) -> Optional[int]:
    """Return the char position where a detected post-roll region starts.

    Scans only the last ``scan_chars`` of the transcript. If ≥ ``threshold``
    distinct ad patterns match, the returned start position is the start of
    the first matching phrase in that window, snapped backward to the
    previous sentence boundary, then iteratively extended backward while
    preceding chunks still carry ad-pattern hits (multi-sentence ad
    blocks). ``None`` if no post-roll is detected.
    """
    if not text:
        return None
    suffix_start = max(0, len(text) - scan_chars)
    suffix = text[suffix_start:]
    hits = _distinct_hits(suffix)
    if not _hits_are_clustered(hits, threshold):
        return None
    first_start_in_suffix = min(start for start, _ in hits)
    absolute_start = suffix_start + first_start_in_suffix
    initial_cut = _snap_backward_to_sentence_start(text, absolute_start)
    return _expand_postroll_backward(text, initial_cut, floor=suffix_start)


def _realign_segments(
    segments: List[Dict[str, Any]],
    excised_ranges: List[Tuple[int, int]],
    source_length: int,
) -> List[Dict[str, Any]]:
    """Drop segments whose text falls inside excised ranges; shift char
    offsets on surviving segments so they align with the cleaned text.

    Segments are expected to carry a ``text`` key. Char offset in the
    transcript is rebuilt by summing segment lengths, matching how the
    existing ``gi.pipeline._char_range_to_ms`` does it.
    """
    if not segments or not excised_ranges:
        return list(segments)

    cleaned: List[Dict[str, Any]] = []
    cursor = 0
    for seg in segments:
        seg_len = len(str(seg.get("text") or ""))
        seg_start = cursor
        seg_end = cursor + seg_len
        cursor = seg_end
        # Skip the segment if the majority of its char range overlaps an
        # excised range (forgiving tolerance for small alignment drift).
        inside = False
        for lo, hi in excised_ranges:
            overlap = max(0, min(seg_end, hi) - max(seg_start, lo))
            if overlap > 0 and overlap >= 0.5 * max(seg_len, 1):
                inside = True
                break
        if not inside:
            cleaned.append(seg)
    return cleaned


def excise_ad_regions(
    text: str,
    *,
    segments: Optional[List[Dict[str, Any]]] = None,
    scan_chars: int = SCAN_CHARS,
    preroll_threshold: int = PREROLL_THRESHOLD,
    postroll_threshold: int = POSTROLL_THRESHOLD,
    dry_run: bool = False,
) -> Tuple[str, Optional[List[Dict[str, Any]]], AdRegionMetadata]:
    """Detect and (optionally) excise pre-roll and post-roll ad regions.

    Args:
        text: Raw transcript text.
        segments: Optional word/utterance segments carrying ``text`` keys.
            When provided, survivors are returned with segments inside
            excised ranges dropped.
        scan_chars: Size of the head/tail window to scan (default 5,000).
        preroll_threshold: Distinct ad-pattern hits required to confirm a
            pre-roll region.
        postroll_threshold: Same, for the tail.
        dry_run: When ``True``, return the source ``text`` + ``segments``
            unchanged but still populate the metadata describing what
            *would* have been cut. Intended for observability / audit
            before enabling live excision in production.

    Returns:
        ``(cleaned_text, cleaned_segments_or_None, AdRegionMetadata)``.
    """
    meta = AdRegionMetadata(source_length=len(text or ""))
    if not text or len(text) < MIN_TRANSCRIPT_CHARS:
        return text, segments, meta

    # Count distinct hits in head/tail for observability, regardless of
    # whether we cut (so dry-run mode still reports signal).
    preroll_hits = _distinct_hits(text[:scan_chars])
    meta.preroll_pattern_hits = len(preroll_hits)
    tail_start = max(0, len(text) - scan_chars)
    postroll_hits = _distinct_hits(text[tail_start:])
    meta.postroll_pattern_hits = len(postroll_hits)

    preroll_end = detect_preroll_ad_end(text, scan_chars=scan_chars, threshold=preroll_threshold)
    postroll_start = detect_postroll_ad_start(
        text, scan_chars=scan_chars, threshold=postroll_threshold
    )

    meta.preroll_cut_end = preroll_end
    meta.postroll_cut_start = postroll_start

    ranges: List[Tuple[int, int]] = []
    if preroll_end is not None:
        ranges.append((0, preroll_end))
    if postroll_start is not None and (not ranges or postroll_start > ranges[-1][1]):
        ranges.append((postroll_start, len(text)))
    meta.excised_ranges = list(ranges)
    meta.chars_removed = sum(hi - lo for lo, hi in ranges)

    if dry_run or not ranges:
        return text, segments, meta

    # Build the cleaned text by keeping the complement of the excised ranges.
    kept_parts: List[str] = []
    prev = 0
    for lo, hi in ranges:
        if prev < lo:
            kept_parts.append(text[prev:lo])
        prev = hi
    if prev < len(text):
        kept_parts.append(text[prev:])
    cleaned_text = "".join(kept_parts)

    cleaned_segments: Optional[List[Dict[str, Any]]] = None
    if segments is not None:
        cleaned_segments = _realign_segments(segments, ranges, len(text))

    return cleaned_text, cleaned_segments, meta
