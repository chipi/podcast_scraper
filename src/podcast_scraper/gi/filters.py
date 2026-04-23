"""#652 Part B — deterministic post-extraction validators for GI insights.

Two filters that run on the final insight list regardless of source
(``provider``, ``summary_bullets``, transcript chunks, prefilled):

1. Ad filter — drops insights that sit inside a transcript window that
   matches ≥ 2 sponsor-ad regex patterns. Threshold tuned conservatively
   (≥ 2 not ≥ 1) so a CEO legitimately describing their own product
   doesn't trip it.

2. Dialogue filter — drops insights that (a) start with conversational
   filler, (b) exceed a first-person-pronoun density threshold, or
   (c) are dominated by a verbatim quote (quote > 60 % of insight text).

Filters are pure functions — no side effects. Callers (``gi/pipeline.py``)
wire them into :func:`~podcast_scraper.workflow.metrics.Metrics` counters
for observability.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Ad filter (Finding 14-lite)
# ---------------------------------------------------------------------------

# Patterns are tuned for SPOKEN transcript text (Whisper output) where URLs
# become phonetic — "bloomberg.com/odd_lots" → "Bloomberg dot com slash Odd
# Lots". Pre-overnight-#652-stabilization the patterns assumed machine-URL
# form (``\bvisit\s+[\w.]+/\w+\b``) and caught 0/1200 insights on a 100-ep
# real corpus. Verified spoken-form patterns hit ad reads in 11/100 eps
# at the ≥ 2-distinct-patterns threshold.
_AD_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # Sponsor disclosure phrases — high-precision standalone signals.
        r"\bbrought to you by\b",
        r"\bsponsored by\b",
        r"\bthis episode is sponsored\b",
        r"\bthis (?:show|podcast) is (?:brought|sponsored)\b",
        r"\bour sponsors?\b",
        r"\btoday'?s sponsor\b",
        r"\bthanks (?:to )?our sponsors?\b",
        r"\bsupport the (?:show|podcast)\b",
        # Spoken-URL signatures (Whisper output).
        r"\b\w+\s+dot\s+com\b",  # "ramp dot com"
        r"\bdot\s+com\s+slash\b",  # "dot com slash promo"
        r"\bslash\s+(?:promo|code|deal|free|trial|save|offer|pod)\b",
        r"\bgo to\s+\w+\s+dot\s+com\b",
        r"\bvisit\s+\w+\s+dot\s+com\b",
        r"\bhead (?:over )?to\s+\w+\s+dot\s+com\b",
        # Promo / CTA phrases — common in ad reads.
        r"\b(?:promo|use)\s+code\s+\w+",
        r"\bsave\s+(?:up to\s+)?\d+\s*(?:percent|%)\b",
        r"\bget\s+\d+\s*(?:percent|%)\s*off\b",
        r"\b\d+\s*(?:percent|%)\s*off\b",  # bare "20% off" (discount offer)
        r"\bfree (?:trial|month|shipping|delivery)\b",
        r"\bfor a limited time\b",
        r"\bsign up\s+(?:today|now)\b",
    )
)

_AD_HITS_THRESHOLD = 2


def _ad_pattern_hits(text: str) -> int:
    if not text:
        return 0
    return sum(1 for p in _AD_PATTERNS if p.search(text))


def insight_looks_like_ad(insight_text: str, source_text: Optional[str] = None) -> bool:
    """Return True when ``source_text`` (or the insight itself) matches ≥ 2
    sponsor-ad regex patterns.

    ``source_text`` should be the transcript window the insight was distilled
    from (quote context) when available; we fall back to scanning the insight
    text directly. The ≥ 2-pattern threshold keeps false positives low — a
    single "go to example.com/X" on its own can appear in genuine content, but
    two or more ad-phrase hits within the same passage is reliably a sponsor
    read.
    """
    hits = _ad_pattern_hits(insight_text)
    if source_text and source_text != insight_text:
        hits += _ad_pattern_hits(source_text)
    return hits >= _AD_HITS_THRESHOLD


# ---------------------------------------------------------------------------
# Dialogue filter (Finding 12)
# ---------------------------------------------------------------------------

_FILLER_PREFIXES: Tuple[str, ...] = (
    # Post-#652 audit: dropped "so", "and", "but" — they're natural sentence
    # connectors used by genuine substantive insights ("So stable coins like
    # USDC ..." was a real technical claim, not filler). The remaining prefixes
    # are pure conversational filler (yeah/uh/well/etc.) with low collision
    # risk on real insight starts.
    "yeah",
    "yep",
    "nope",
    "okay",
    "ok",
    "well",
    "i mean",
    "you know",
    "um",
    "uh",
    "right",
    "exactly",
)

_FIRST_PERSON_PRONOUNS: frozenset[str] = frozenset(
    {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
)

# Bumped from 0.15 → 0.25 after #652 audit. The lower threshold dropped
# substantive first-person CEO/expert claims like "We made our first
# investment before the AI trade started" (3/12 = 0.25 pronoun density —
# right at the old threshold but legitimate analysis content). The higher
# threshold still catches dialogue-heavy insights without false-positiving
# on opinion/analysis bullets.
_PRONOUN_DENSITY_THRESHOLD = 0.25
_QUOTE_COVERAGE_THRESHOLD = 0.60


def _normalize_first_words(text: str, n: int = 3) -> List[str]:
    """Return lowercase first ``n`` tokens (strip punctuation)."""
    tokens = re.findall(r"[A-Za-z']+", text or "")
    return [t.lower() for t in tokens[:n]]


def _starts_with_filler(text: str) -> bool:
    if not text:
        return False
    first_words = _normalize_first_words(text, 3)
    if not first_words:
        return False
    for pref in _FILLER_PREFIXES:
        pref_tokens = pref.split()
        if len(pref_tokens) <= len(first_words):
            if first_words[: len(pref_tokens)] == pref_tokens:
                return True
    return False


def _first_person_density(text: str) -> float:
    tokens = re.findall(r"[A-Za-z']+", text or "")
    if not tokens:
        return 0.0
    pronouns = sum(1 for t in tokens if t.lower() in _FIRST_PERSON_PRONOUNS)
    return pronouns / len(tokens)


def _quote_coverage(insight_text: str, quote_text: Optional[str]) -> float:
    """Fraction of ``insight_text`` character-length that ``quote_text`` covers
    (case-insensitive substring length). Returns 0 when quote is absent."""
    if not insight_text or not quote_text:
        return 0.0
    insight_len = len(insight_text.strip())
    if insight_len == 0:
        return 0.0
    q = quote_text.strip()
    if not q:
        return 0.0
    if q.lower() in insight_text.lower():
        return len(q) / insight_len
    return 0.0


def insight_looks_like_dialogue(insight_text: str, quote_text: Optional[str] = None) -> bool:
    """Return True when an insight is likely dialogue/filler rather than a
    distilled third-person claim.

    Any one of the three rules is sufficient:

    * Starts with a conversational filler token (yeah/okay/well/so/…).
    * First-person pronoun density > 0.15.
    * A verbatim quote covers > 60 % of the insight text.
    """
    if not insight_text:
        return False
    if _starts_with_filler(insight_text):
        return True
    if _first_person_density(insight_text) > _PRONOUN_DENSITY_THRESHOLD:
        return True
    if _quote_coverage(insight_text, quote_text) > _QUOTE_COVERAGE_THRESHOLD:
        return True
    return False


# ---------------------------------------------------------------------------
# Public entry points used by gi/pipeline.py
# ---------------------------------------------------------------------------


def apply_insight_filters(
    insights: Sequence[dict],
    *,
    transcript_window_by_index: Optional[dict[int, str]] = None,
) -> Tuple[List[dict], int, int]:
    """Apply the two filters to a list of insight dicts.

    Args:
        insights: Each dict has at least ``text``; may also have ``quote`` /
            ``quote_text`` for the dialogue filter and a resolvable transcript
            source window via ``transcript_window_by_index``.
        transcript_window_by_index: Optional mapping of positional index → the
            transcript window the insight came from (used by the ad filter).

    Returns:
        ``(kept_insights, ads_dropped_count, dialogue_dropped_count)``.
    """
    kept: List[dict] = []
    ads_dropped = 0
    dialogue_dropped = 0
    for i, ins in enumerate(insights):
        text = str(ins.get("text") or "").strip()
        if not text:
            kept.append(ins)  # let downstream validation drop empties
            continue
        window = None
        if transcript_window_by_index is not None:
            window = transcript_window_by_index.get(i)
        if insight_looks_like_ad(text, window):
            ads_dropped += 1
            continue
        quote = ins.get("quote") or ins.get("quote_text")
        if insight_looks_like_dialogue(text, quote):
            dialogue_dropped += 1
            continue
        kept.append(ins)
    return kept, ads_dropped, dialogue_dropped


__all__ = [
    "apply_insight_filters",
    "insight_looks_like_ad",
    "insight_looks_like_dialogue",
]
