"""Confidence-scored commercial segment detection and removal."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

from .diarization_signals import diarization_sponsor_signals
from .patterns import (
    BLOCK_END_PATTERNS,
    BLOCK_START_PATTERNS,
    BRAND_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    SPONSOR_PATTERNS,
    SponsorPattern,
)
from .positions import position_score

_BACKSCAN_CHARS = 500
_FORWARD_SCAN_CHARS = 2000
_MAX_LOW_CONFIDENCE_BLOCK_RATIO = 0.4
_HIGH_CONFIDENCE_FOR_LARGE_BLOCK = 0.85
# When the transcript has no paragraph breaks (e.g. single-line Whisper output)
# cap the sponsor-block removal at this many chars so one sponsor mention does
# not wipe the entire transcript.
_SPONSOR_BLOCK_MAX_CHARS = 800

# Inline CTAs (a bare ".com" or "sign up today") are common in ordinary speech.
# Below this pattern confidence we refuse to remove on the CTA alone — it must be
# corroborated by another sponsor signal nearby, or we risk deleting real content.
_INLINE_STANDALONE_CONFIDENCE = 0.7
_INLINE_CORROBORATION_WINDOW = 600
_PROMO_CODE_RE = re.compile(r"(?:use (?:code|promo|coupon)|promo code|discount code)", re.I)


@dataclass(frozen=True)
class CommercialCandidate:
    """One detected sponsor region."""

    start: int
    end: int
    confidence: float
    matched_pattern: str


class CommercialDetector:
    """Multi-signal commercial detection (patterns, position, optional diarization)."""

    def __init__(
        self,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        diarization_segments: Optional[List[dict]] = None,
        host_speaker_id: Optional[str] = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.diarization_segments = diarization_segments
        self.host_speaker_id = host_speaker_id

    def detect(self, text: str) -> List[CommercialCandidate]:
        """Return sponsor candidates above threshold after boundary expansion."""
        if not text:
            return []

        candidates: List[CommercialCandidate] = []
        seen_spans: set[Tuple[int, int]] = set()
        text_len = len(text)

        for sponsor_pattern in SPONSOR_PATTERNS:
            for match in sponsor_pattern.pattern.finditer(text):
                # A low-confidence inline CTA must be corroborated by another
                # sponsor signal, else "check out github.com" in normal speech
                # would clear the threshold via the position boost and delete a
                # paragraph of real content.
                if (
                    sponsor_pattern.boundary_hint == "inline"
                    and sponsor_pattern.confidence < _INLINE_STANDALONE_CONFIDENCE
                    and not _inline_cta_corroborated(text, match.start(), match.end())
                ):
                    continue
                base_conf = sponsor_pattern.confidence
                base_conf += position_score(match.start(), text_len)
                if _text_mentions_brand(text, match.start(), match.end()):
                    base_conf += 0.1
                if self.diarization_segments and self.host_speaker_id:
                    signals = diarization_sponsor_signals(
                        match.start(),
                        match.end(),
                        text,
                        self.diarization_segments,
                        self.host_speaker_id,
                    )
                    if signals.disqualify:
                        continue
                    base_conf += signals.confidence_delta
                if base_conf < self.confidence_threshold:
                    continue
                start, end = _detect_sponsor_boundaries(
                    text, match.start(), match.end(), sponsor_pattern
                )
                if _span_too_large_for_confidence(start, end, text_len, base_conf):
                    continue
                span = (start, end)
                if span in seen_spans or start >= end:
                    continue
                seen_spans.add(span)
                candidates.append(
                    CommercialCandidate(
                        start=start,
                        end=end,
                        confidence=min(base_conf, 1.0),
                        matched_pattern=sponsor_pattern.pattern.pattern,
                    )
                )

        candidates.sort(key=lambda c: c.start)
        return _merge_overlapping_candidates(candidates)

    def remove(self, text: str) -> str:
        """Remove detected commercial regions from text.

        Logs an audit trail of what was excised (span, confidence, char count) so a
        too-aggressive removal is recoverable from logs rather than silent.
        """
        candidates = self.detect(text)
        cleaned = text
        for candidate in reversed(candidates):
            cleaned = cleaned[: candidate.start] + cleaned[candidate.end :]
        if candidates:
            removed_chars = sum(c.end - c.start for c in candidates)
            logger.info(
                "Commercial cleaning removed %d block(s), %d chars (%.1f%% of %d); " "spans=%s",
                len(candidates),
                removed_chars,
                100.0 * removed_chars / max(len(text), 1),
                len(text),
                [(c.start, c.end, round(c.confidence, 2)) for c in candidates],
            )
        return cleaned


def _span_too_large_for_confidence(start: int, end: int, text_len: int, confidence: float) -> bool:
    """Reject low-confidence detections that would excise most of the transcript."""
    if text_len <= 0 or end <= start:
        return True
    if confidence >= _HIGH_CONFIDENCE_FOR_LARGE_BLOCK:
        return False
    span_ratio = (end - start) / text_len
    return span_ratio > _MAX_LOW_CONFIDENCE_BLOCK_RATIO


def _text_mentions_brand(text: str, start: int, end: int) -> bool:
    window = text[max(0, start - 200) : min(len(text), end + 400)].lower()
    return any(brand in window for brand in BRAND_NAMES)


def _inline_cta_corroborated(text: str, match_start: int, match_end: int) -> bool:
    """True if a low-confidence inline CTA has independent sponsor evidence nearby.

    Corroboration = a known brand name, an explicit promo/coupon code, or a
    sponsor-intro phrase shortly before the CTA. Without any of these, a bare
    URL/sign-up phrase is treated as ordinary conversation and left in place.
    """
    if _text_mentions_brand(text, match_start, match_end):
        return True
    window_start = max(0, match_start - _INLINE_CORROBORATION_WINDOW)
    window_end = min(len(text), match_end + _INLINE_CORROBORATION_WINDOW)
    if _PROMO_CODE_RE.search(text[window_start:window_end]):
        return True
    preceding = text[window_start:match_start]
    return any(pattern.pattern.search(preceding) for pattern in BLOCK_START_PATTERNS)


def _paragraph_start(text: str, index: int) -> int:
    prev_break = text.rfind("\n\n", 0, index)
    if prev_break != -1:
        return prev_break + 2

    # No paragraph break before the match. Returning 0 here — the start of the transcript — is what
    # ``_paragraph_end`` already refuses to do at the other end, and for the same reason: a
    # screenplay transcript separates speaker turns with a SINGLE newline, so a whole episode is
    # one "paragraph". A sponsor mention 70 000 chars in then yields a block of [0, 70 800] and the
    # cleaner deletes the episode. Measured: a 77 868-char transcript cleaned to **0 chars**, after
    # which every LLM stage — summary, GI, KG — ran happily on nothing.
    #
    # The cap was already written for exactly this case; it was only ever applied to the end.
    # Symmetry: walk back to a sentence boundary within the same window.
    window_start = max(0, index - _SPONSOR_BLOCK_MAX_CHARS)
    sentence_starts = list(re.finditer(r"[.!?](?:\s|$)|\n", text[window_start:index]))
    if sentence_starts:
        return window_start + sentence_starts[-1].end()
    return window_start


def _paragraph_end(text: str, index: int) -> int:
    next_break = text.find("\n\n", index)
    if next_break != -1:
        return next_break
    # No paragraph break — common when transcript comes from Whisper as one
    # continuous line. Removing to end-of-text would wipe everything when a
    # single sponsor mention appears. Fall back to sentence boundary: find
    # the next sentence terminator within a reasonable window.
    sentence_end = re.search(r"[.!?](?:\s|$)", text[index : index + _SPONSOR_BLOCK_MAX_CHARS])
    if sentence_end:
        return index + sentence_end.end()
    return min(index + _SPONSOR_BLOCK_MAX_CHARS, len(text))


def _scan_for_pattern(
    text: str,
    patterns: List[SponsorPattern],
    start: int,
    end: int,
    *,
    reverse: bool,
) -> Optional[int]:
    segment = text[start:end]
    if reverse:
        segment = segment[::-1]
    for sponsor_pattern in patterns:
        match = sponsor_pattern.pattern.search(segment)
        if match:
            if reverse:
                return end - match.end()
            return start + match.start()
    return None


def _detect_sponsor_boundaries(
    text: str,
    match_start: int,
    match_end: int,
    matched: SponsorPattern,
) -> Tuple[int, int]:
    """Find block start/end using boundary patterns with paragraph fallback."""
    back_start = max(0, match_start - _BACKSCAN_CHARS)
    block_start = _scan_for_pattern(
        text, BLOCK_START_PATTERNS, back_start, match_start, reverse=True
    )
    if block_start is None:
        block_start = _paragraph_start(text, match_start)
    else:
        block_start = max(back_start, block_start)

    forward_end = min(len(text), match_end + _FORWARD_SCAN_CHARS)
    block_end = _scan_for_pattern(text, BLOCK_END_PATTERNS, match_end, forward_end, reverse=False)
    if block_end is None:
        block_end = _paragraph_end(text, match_end)
    else:
        block_end = min(len(text), block_end)

    if matched.boundary_hint == "inline" and block_end - block_start > 2000:
        block_end = min(block_start + 2000, len(text))

    return block_start, max(block_end, match_end)


def _merge_overlapping_candidates(
    candidates: List[CommercialCandidate],
) -> List[CommercialCandidate]:
    if not candidates:
        return []
    merged: List[CommercialCandidate] = [candidates[0]]
    for candidate in candidates[1:]:
        prev = merged[-1]
        if candidate.start <= prev.end:
            merged[-1] = CommercialCandidate(
                start=prev.start,
                end=max(prev.end, candidate.end),
                confidence=max(prev.confidence, candidate.confidence),
                matched_pattern=prev.matched_pattern,
            )
        else:
            merged.append(candidate)
    return merged
