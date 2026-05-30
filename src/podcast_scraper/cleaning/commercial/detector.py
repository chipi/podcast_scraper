"""Confidence-scored commercial segment detection and removal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


@dataclass(frozen=True)
class CommercialCandidate:
    """One detected sponsor region."""

    start: int
    end: int
    confidence: float
    matched_pattern: str


class CommercialDetector:
    """Multi-signal commercial detection (Phase 1: patterns + position)."""

    def __init__(
        self,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold

    def detect(self, text: str) -> List[CommercialCandidate]:
        """Return sponsor candidates above threshold after boundary expansion."""
        if not text:
            return []

        candidates: List[CommercialCandidate] = []
        seen_spans: set[Tuple[int, int]] = set()
        text_len = len(text)

        for sponsor_pattern in SPONSOR_PATTERNS:
            for match in sponsor_pattern.pattern.finditer(text):
                base_conf = sponsor_pattern.confidence
                base_conf += position_score(match.start(), text_len)
                if _text_mentions_brand(text, match.start(), match.end()):
                    base_conf += 0.1
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
        """Remove detected commercial regions from text."""
        cleaned = text
        for candidate in reversed(self.detect(text)):
            cleaned = cleaned[: candidate.start] + cleaned[candidate.end :]
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


def _paragraph_start(text: str, index: int) -> int:
    prev_break = text.rfind("\n\n", 0, index)
    return 0 if prev_break == -1 else prev_break + 2


def _paragraph_end(text: str, index: int) -> int:
    next_break = text.find("\n\n", index)
    if next_break == -1:
        return len(text)
    return next_break


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
