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
# Per-cluster excision (the fix for "several ad blocks in one episode → cut nothing"): ad-pattern
# hits within CLUSTER_GAP chars of each other belong to ONE ad block; a larger gap starts a new
# block. Each block is excised on its own, so real content BETWEEN blocks is kept — the whole reason
# the old first-to-last-span check declined to cut scattered hits.
CLUSTER_GAP = 750
# A block whose snapped end reaches within this many chars of the very end is a POST-roll — extend
# its cut to the end. Measured against len(text), NOT the scan window, so it stays correct for
# transcripts shorter than SCAN_CHARS (where the tail window starts at 0).
POSTROLL_TRAILOUT = 350
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
    # The ACTUAL text removed at each range, in order — so an operator chasing a bad cut can see
    # exactly WHAT was excised (not just how many chars), straight from the ad-map sidecar.
    excised_texts: List[str] = field(default_factory=list)

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
            "excised_texts": list(self.excised_texts),
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


def _merge_ranges(ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort and union overlapping/adjacent ``[lo, hi)`` ranges."""
    out: List[Tuple[int, int]] = []
    for lo, hi in sorted(ranges):
        if lo >= hi:
            continue
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _cluster_hits(
    hits: List[Tuple[int, int]], max_gap: int = CLUSTER_GAP
) -> List[List[Tuple[int, int]]]:
    """Group sorted ``(start, end)`` hits into clusters. A gap larger than ``max_gap`` between one
    hit's end and the next hit's start starts a new cluster — i.e., a run of nearby ad-pattern hits
    is one ad block; a stretch of real content between hits splits the blocks apart."""
    clusters: List[List[Tuple[int, int]]] = []
    for hit in sorted(hits):
        if clusters and hit[0] - clusters[-1][-1][1] <= max_gap:
            clusters[-1].append(hit)
        else:
            clusters.append([hit])
    return clusters


def detect_ad_cut_ranges(
    text: str,
    *,
    scan_chars: int = SCAN_CHARS,
    threshold: int = PREROLL_THRESHOLD,
) -> List[Tuple[int, int]]:
    """Return every ad block's ``[lo, hi)`` cut range across the head + tail scan windows.

    Replaces the old single-contiguous-block model (which cut ONE pre-roll span and declined
    entirely when hits were scattered, leaving mid-episode sponsor reads in the transcript). Each
    ad block is cut on its own, sentence-aligned, so content between blocks survives:

    - Episode-level gate: fewer than ``threshold`` DISTINCT ad patterns anywhere → return ``[]``
      (weak signal; do not risk cutting a lone false-positive phrase).
    - Cluster the hits, then cut ``[snap_back(first) .. snap_forward(last)]`` per cluster.
    - A block whose start lands within ``PREROLL_LEADIN`` of the top is a true pre-roll → extend to
      0. A block reaching the tail window's end is a post-roll → extend to the end and expand
      backward over its multi-sentence body.
    """
    if not text:
        return []
    head_hits = _distinct_hits(text[:scan_chars])
    tail_start = max(0, len(text) - scan_chars)
    tail_hits = [(tail_start + s, tail_start + e) for s, e in _distinct_hits(text[tail_start:])]
    # De-dupe by position (a hit found in both windows on short transcripts) and gate on distinct
    # PATTERN count — Σ head+tail distinct patterns — so a strong ad signal is required to cut.
    all_hits = sorted(set(head_hits) | set(tail_hits))
    if len(all_hits) < threshold:
        return []
    ranges: List[Tuple[int, int]] = []
    prev_end = 0
    for i, cluster in enumerate(_cluster_hits(all_hits)):
        first = cluster[0][0]
        last = cluster[-1][1]
        start = _snap_backward_to_sentence_start(text, first)
        end = _snap_forward_to_sentence_end(text, last)
        # Post-roll ONLY when the cut reaches the very end of the text (measured vs len, not
        # window — for short transcripts the tail window starts at 0, which would misclassify every
        # block as a post-roll). Extend to end + expand backward over the ad body, bounded by
        # previous kept region so it never eats real content between blocks.
        if end >= len(text) - POSTROLL_TRAILOUT:
            start = _expand_postroll_backward(text, start, floor=prev_end)
            end = len(text)
        elif i == 0 and first < scan_chars:
            # The first ad block in the head window is the PRE-roll: cut from 0 so the ad PITCH that
            # precedes its first URL/CTA pattern goes too (the pattern marks the CTA, not the ad's
            # start; snapping back to a sentence boundary can't reach the start in a screenplay
            # where sentences end in ".\n"). Matches the pre-per-cluster [0..end] model. Cost:
            # intro before a near-start sponsor is also cut — a #1385 refinement (needs NLP).
            start = 0
        ranges.append((start, end))
        prev_end = end
    return _merge_ranges(ranges)


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

    # Per-cluster excision: cut EVERY ad block (pre-, mid-, post-roll) independently, keeping the
    # content between them — instead of the old single-span model that declined to cut at all when
    # an episode had more than one ad block (the "5 hits, excised 0" bug).
    ranges = detect_ad_cut_ranges(text, scan_chars=scan_chars, threshold=preroll_threshold)
    meta.excised_ranges = list(ranges)
    meta.chars_removed = sum(hi - lo for lo, hi in ranges)
    meta.excised_texts = [text[lo:hi] for lo, hi in ranges]
    # Backward-compatible boundary breadcrumbs: first range if it opens at 0 is the pre-roll; last
    # range if it reaches the end is the post-roll (mid-roll blocks live only in excised_ranges).
    meta.preroll_cut_end = ranges[0][1] if ranges and ranges[0][0] == 0 else None
    meta.postroll_cut_start = ranges[-1][0] if ranges and ranges[-1][1] == len(text) else None

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


def _shift_for(pos: int, ranges: List[Tuple[int, int]]) -> int:
    """Total chars excised strictly before ``pos`` (how far ``pos`` moves left)."""
    return sum(hi - lo for lo, hi in ranges if hi <= pos)


def _overlaps_any(cs: int, ce: int, ranges: List[Tuple[int, int]]) -> bool:
    return any(max(cs, lo) < min(ce, hi) for lo, hi in ranges)


def _complement_text(text: str, ranges: List[Tuple[int, int]]) -> str:
    """Return ``text`` with every ``[lo, hi)`` range removed (ranges are sorted,
    non-overlapping — pre-roll then post-roll)."""
    kept: List[str] = []
    prev = 0
    for lo, hi in ranges:
        if prev < lo:
            kept.append(text[prev:lo])
        prev = hi
    if prev < len(text):
        kept.append(text[prev:])
    return "".join(kept)


def merge_preroll_range(
    meta: AdRegionMetadata, preroll_end: int, *, text: Optional[str] = None
) -> None:
    """Fold an extra pre-roll range ``[0, preroll_end)`` into ``meta`` in place.

    The single seam both ad-free branches use to add a diarization-detected opening
    cross-promo (#1188) to the ad-map: merge it with any density-detected pre-roll so
    ``excised_ranges`` / ``chars_removed`` / ``preroll_cut_end`` stay one coordinate
    space for consumers and raw reconciliation."""
    if preroll_end <= 0:
        return
    # MERGE the extra pre-roll into the existing ranges (which now may include mid-roll blocks), not
    # replace them — replacing would silently drop every non-pre-roll ad block from the ad-map.
    merged = _merge_ranges(list(meta.excised_ranges) + [(0, preroll_end)])
    meta.excised_ranges = merged
    meta.preroll_cut_end = merged[0][1] if merged and merged[0][0] == 0 else preroll_end
    if merged and merged[-1][1] == meta.source_length:
        meta.postroll_cut_start = merged[-1][0]
    meta.chars_removed = sum(hi - lo for lo, hi in merged)
    # Refresh the audit trail of WHAT was cut when the source text is available (ranges just moved).
    if text is not None:
        meta.excised_texts = [text[lo:hi] for lo, hi in merged]


def excise_ad_regions_with_offsets(
    text: str,
    offset_segments: List[Dict[str, Any]],
    *,
    scan_chars: int = SCAN_CHARS,
    preroll_threshold: int = PREROLL_THRESHOLD,
    postroll_threshold: int = POSTROLL_THRESHOLD,
    extra_preroll_end: int = 0,
) -> Tuple[str, List[Dict[str, Any]], AdRegionMetadata]:
    """Excise ad regions from a screenplay while keeping segment char ranges exact.

    Unlike :func:`excise_ad_regions`, the input ``offset_segments`` already carry a
    ``char_start`` / ``char_end`` range into ``text`` (as emitted by
    :func:`...diarization.formatting.format_diarized_screenplay_with_offsets`). This
    returns the ad-free text plus the surviving segments **re-offset into the ad-free
    text**, so a downstream consumer can map ``cleaned_text[seg["char_start"]:
    seg["char_end"]] == seg["text"]`` with no length-guard heuristic (#974, Fault B).

    A segment that overlaps an excised range at all is dropped (it's inside the ad);
    survivors shift left by the number of excised chars before them. Returns
    ``(cleaned_text, surviving_offset_segments, AdRegionMetadata)``. The metadata is
    the ad-map (excised ranges in the *source* / raw-screenplay space) used to
    reconcile the ad-free text back to the raw transcript for the future player.
    """
    cleaned_text, _, meta = excise_ad_regions(
        text,
        segments=None,
        scan_chars=scan_chars,
        preroll_threshold=preroll_threshold,
        postroll_threshold=postroll_threshold,
    )
    if extra_preroll_end > 0:
        # Caller-supplied opening cut (e.g. a diarization-detected cross-promo, #1188)
        # the density pass missed. Merge it in and re-cut the text from the new ranges.
        merge_preroll_range(meta, extra_preroll_end, text=text)
        cleaned_text = _complement_text(text, meta.excised_ranges)
    ranges = meta.excised_ranges
    if not ranges:
        # Nothing cut → ranges/text unchanged; return segments verbatim.
        return cleaned_text, [dict(s) for s in offset_segments], meta

    survivors: List[Dict[str, Any]] = []
    for seg in offset_segments:
        cs = int(seg.get("char_start", 0))
        ce = int(seg.get("char_end", 0))
        if _overlaps_any(cs, ce, ranges):
            continue
        shift = _shift_for(cs, ranges)
        moved = dict(seg)
        moved["char_start"] = cs - shift
        moved["char_end"] = ce - shift
        survivors.append(moved)
    return cleaned_text, survivors, meta
