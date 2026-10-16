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
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# ragged mid-sentence fragments on the content side. A line break ends a screenplay turn: without
# it, an ad ending its turn (".\n") snapped into the next speaker's first sentence (D30).
SENTENCE_TERMINATORS = (". ", "! ", "? ", "\n")
SENTENCE_BOUNDARY_LOOKAHEAD = 300
# The first ad block is a true PRE-roll, cut from 0, only when its first ad pattern starts within
# this many chars of the top (#2360). Past it, what precedes the block is the episode's own talk —
# a host's opening monologue, a cold open — and cutting from 0 took it all (46 prod episodes,
# 114 min; Colombia Calling lost 2-4 minutes of every opening). Measured on the 289 prod episodes
# with a pre-roll cut: every pure pitch or show promo put its first pattern within 896 chars, every
# episode that lost real talk had it at 1,051 or later.
PREROLL_LEADIN = 1000
# A block that starts past the lead-in with a call to action ("check out ramp.com") was pitched
# first; the cut reaches back over at most this many chars for the pitch. The longest pure pitch
# before a first pattern on prod was 546 chars.
PREROLL_PITCH = 600
# A block in the middle of the episode is cut only when it holds at least this many KINDS of ad cue
# (a URL and a code, an opener and a URL ...): a single "dot com" in conversation is not a read.
MIDROLL_MIN_KINDS = 2
# Ad patterns that OPEN a read ("brought to you by", "sponsored by", "our sponsor" ...): when the
# block's first pattern is one of these, the ad starts there and nothing before it is pitch.
_AD_OPENER = re.compile(
    r"\b(?:brought to you by|sponsored by|this episode is sponsored|"
    r"this (?:show|podcast) is (?:brought|sponsored)|our sponsors?|today'?s sponsor|"
    r"thanks (?:to )?our sponsors?|support for (?:this|the) (?:show|podcast|episode) comes from)\b",
    re.IGNORECASE,
)
# The opener forms that announce a sponsor read and nothing else. A bare "sponsored by" is also
# ordinary talk ("we were sponsored by a church"), so it never brackets a read on its own.
_READ_OPENER = re.compile(
    r"\b(?:brought to you by|this episode is sponsored|this (?:show|podcast) is "
    r"(?:brought|sponsored)|today'?s sponsor|support for (?:this|the) (?:show|podcast|episode) "
    r"comes from)\b",
    re.IGNORECASE,
)
# A call to action: where to go or what code to use.
_CALL_TO_ACTION = re.compile(
    r"\b(?:go to|visit|check out|learn more at|head (?:over )?to|start at|sign up)\b"
    r"|\b(?:promo|use) code\b|\b(?:dot com|\.(?:com|ai|io|co)) slash\b|\bslash\s+\w+\b"
    r"|\bfree (?:trial|month)\b|(?:percent|%) off\b",
    re.IGNORECASE,
)
# An opener and its call to action are one read when the call to action follows within this many
# chars in the same turn. They clustered apart whenever the pitch between them ran past
# CLUSTER_GAP: the opener alone is a lone cue (skipped mid-episode), the call to action starts its
# own cut, and the pitch between them stayed in the ad-free text (WorkOS, Mercury, Vanta, Sentry on
# the 2026-10-15 prod snapshot).
READ_SPAN = 2500


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


def _all_hits(text: str) -> List[Tuple[int, int, int]]:
    """Every ad-pattern match in ``text`` as ``(start, end, pattern_index)``, sorted.

    :func:`_distinct_hits` keeps one match per pattern, which is right for counting how many KINDS
    of cue an episode has but blind to a second read that repeats a pattern already seen (#2168)."""
    hits = [
        (m.start(), m.end(), i) for i, pat in enumerate(_AD_PATTERNS) for m in pat.finditer(text)
    ]
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
        # The top of the text starts a sentence: without this, an opener on the first line left
        # its own "SPEAKER_00: This episode is" behind.
        return 0 if lo == 0 else pos
    return lo + best


def _preroll_start(text: str, first: int, sentence_start: int, *, lone: bool = False) -> int:
    """Where the first ad block in the head window starts, given its first pattern at ``first``.

    An opener ("brought to you by") starts the read at its own sentence, at the top too: cut from 0,
    a cold open before it went with it (Rest Is Science, prod 2026-10-15). A call to action near the
    top (``PREROLL_LEADIN``) is the pre-roll: cut from 0, so the pitch before it goes too (the
    pattern marks the CTA, not the ad's start). Further in, the episode has already begun: a call
    to action takes at most ``PREROLL_PITCH`` chars of pitch before it (#2360) — but only in a
    block with more than one kind of cue. A ``lone`` call to action is one sentence, and the reach
    back took what came before it: a guest's introduction before "check out lennysproductpass.com",
    a founder's own account of his product before "go to dreamer.com".
    """
    if _AD_OPENER.match(text, first):
        return sentence_start
    if first <= PREROLL_LEADIN:
        return 0
    if lone:
        return sentence_start
    return _read_start(text, first, sentence_start, floor=0)


def _read_start(text: str, first: int, sentence_start: int, floor: int) -> int:
    """Where an ad read whose first pattern is at ``first`` starts, never before ``floor``.

    An opener ("sponsored by") starts the read at its own sentence; a call to action ("check out
    ramp.com") was pitched first, so the read reaches back at most ``PREROLL_PITCH`` chars to a
    sentence start (#2360, #2168)."""
    if _AD_OPENER.match(text, first):
        return max(floor, sentence_start)
    # A pitch names what it sells. The reach goes back only to the first sentence that names the
    # call to action's own domain: without that, it took what came before a show's OWN plug — a
    # guest's introduction before "check out lennysproductpass.com", the host's talk before
    # "patreon.com slash colombia calling" (8 of the 17 reaches on the 2026-10-15 prod snapshot).
    brand = _cta_brand(text, first)
    if brand is None:
        return max(floor, sentence_start)
    reach = max(floor, first - PREROLL_PITCH)
    named = brand.search(text, reach, sentence_start)
    if named is None:
        return max(floor, sentence_start)
    return max(floor, min(sentence_start, _snap_backward_to_sentence_start(text, named.start())))


_CTA_DOMAIN = re.compile(r"\b([a-z][\w-]{2,})\s*(?:\.|\bdot\s+)(?:com|ai|io|co)\b", re.IGNORECASE)


def _cta_brand(text: str, first: int) -> Optional[re.Pattern[str]]:
    """The domain a call to action at ``first`` names, as a pattern for the brand in speech: its
    first six letters, any non-letters between them ("alpha-sense.com" -> "AlphaSense"; ASR
    respellings past six letters, "antisysus.com" -> "Antisys", still match). ``None`` when the
    call to action names no domain."""
    m = _CTA_DOMAIN.search(text, first, first + SENTENCE_BOUNDARY_LOOKAHEAD)
    letters = re.sub(r"[^a-z]", "", (m.group(1) if m else "").lower())[:6]
    if len(letters) < 4:
        return None
    return re.compile(r"[\W_]*".join(re.escape(ch) for ch in letters), re.IGNORECASE)


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
    hits: Sequence[Tuple[int, ...]], max_gap: int = CLUSTER_GAP
) -> List[List[Tuple[int, ...]]]:
    """Group sorted ``(start, end)`` hits into clusters. A gap larger than ``max_gap`` between one
    hit's end and the next hit's start starts a new cluster — i.e., a run of nearby ad-pattern hits
    is one ad block; a stretch of real content between hits splits the blocks apart."""
    clusters: List[List[Tuple[int, ...]]] = []
    for hit in sorted(hits):
        if clusters and hit[0] - max(h[1] for h in clusters[-1]) <= max_gap:
            clusters[-1].append(hit)
        else:
            clusters.append([hit])
    return clusters


def _bracket_reads(text: str, clusters: List[List[Tuple[int, ...]]]) -> List[List[Tuple[int, ...]]]:
    """Join a block that an opener starts to the next block when that one starts with a call to
    action within ``READ_SPAN`` chars and no turn change (a screenplay line break) between."""
    out: List[List[Tuple[int, ...]]] = []
    for cluster in clusters:
        if out:
            prev = out[-1]
            gap_lo, gap_hi = max(h[1] for h in prev), cluster[0][0]
            if (
                _READ_OPENER.match(text, prev[0][0])
                and _CALL_TO_ACTION.search(text, cluster[0][0], cluster[0][1])
                and gap_hi - gap_lo <= READ_SPAN
                and "\n" not in text[gap_lo:gap_hi]
            ):
                out[-1] = prev + list(cluster)
                continue
        out.append(list(cluster))
    return out


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
    - A block whose first pattern lands within ``PREROLL_LEADIN`` of the top is a true pre-roll →
      extend to 0; further in, the cut starts at the read's opener or a pitch's length before its
      call to action (:func:`_preroll_start`, #2360).
    - A block reaching the tail window's end is a post-roll → extend to the end and expand backward
      over its multi-sentence body.
    """
    if not text:
        return []
    # The WHOLE text, every match (#2168): the head and tail windows alone never saw a sponsor read
    # in the middle of an episode, and one match per pattern hid a second read of the same kind.
    # The episode gate adds up the distinct cue KINDS of each block, as the old head + tail count
    # did (a show whose pre-roll and post-roll use the same two cues still has 4, not 2).
    hits = _all_hits(text)
    clusters = _cluster_hits(hits)
    if sum(len({h[2] for h in c}) for c in clusters) < threshold:
        return []
    clusters = _bracket_reads(text, clusters)
    ranges: List[Tuple[int, int]] = []
    prev_end = 0
    for i, cluster in enumerate(clusters):
        first = cluster[0][0]
        last = max(h[1] for h in cluster)
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
            start = _preroll_start(
                text, first, start, lone=len({kind for _, _, kind in cluster}) < MIDROLL_MIN_KINDS
            )
        elif len({kind for _, _, kind in cluster}) < MIDROLL_MIN_KINDS:
            # A lone cue in the middle of an episode ("as I said on cnn dot com") is talk, not a
            # read: a mid-roll needs two kinds of cue together (#2168).
            continue
        else:
            # Mid-episode, the cut starts at the read's first cue sentence and reaches no further
            # back: read on 12 prod mid-rolls, the pre-roll's pitch reach took the guest's talk
            # before a call to action in 4 of them. A little pitch left in beats talk cut out.
            start = max(prev_end, start)
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


def ad_time_intervals(segments: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Ad regions of an episode as ``(start_s, end_s)`` intervals over its timed segments.

    The detector works in *character* space over the transcript; diarization and the player work in
    *time*. This runs :func:`excise_ad_regions` over the segments' text, joined with one space, and
    maps every excised range back to the first and last segment it overlaps. ``[]`` when nothing is
    detected. Never raises: a caller that loses its ads keeps working, just ad-blind.
    """
    spans: List[Tuple[int, int, Dict[str, Any]]] = []
    parts: List[str] = []
    cursor = 0
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = str(seg.get("text", "") or "")
        parts.append(text)
        spans.append((cursor, cursor + len(text), seg))
        cursor += len(text) + 1  # the space " ".join puts between segments
    try:
        _, _, meta = excise_ad_regions(" ".join(parts))
    except Exception as exc:  # noqa: BLE001 — ad detection must never break its caller
        logger.warning("Ad-region detection failed; continuing without ad intervals: %s", exc)
        return []
    intervals: List[Tuple[float, float]] = []
    for char_start, char_end in meta.excised_ranges:
        covered = [
            seg
            for start, end, seg in spans
            if not (end <= char_start or start >= char_end)  # overlaps the ad span
        ]
        if covered:
            intervals.append(
                (float(covered[0].get("start") or 0.0), float(covered[-1].get("end") or 0.0))
            )
    return intervals


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
