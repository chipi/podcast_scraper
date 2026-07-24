"""Opening host-read cross-promo detection (#1188).

Some shows open with a *cross-promo* for a sister property, read as editorial by
voices that appear nowhere else in the episode (Hard Fork's "The Athletic"
segment: two named reporters introduce their beat, then never speak again). It
carries none of the commercial markers the sponsor detector keys on — no promo
code, no URL, no "brought to you by" — so :mod:`.detector` and the density-based
:func:`...gi.ad_regions.excise_ad_regions` both leave it in every episode.

Detection is *diarization-corroborated, content-bounded*, and deliberately NOT
tuned to any one feed:

* **Structural core (feed-agnostic):** a short leading run of segments where at
  least one voice never recurs later in the episode. Real hosts recur; ad readers
  do not. This is the strongest general signal and needs no per-feed tuning.
* **Linguistic corroboration (English-general, extensible):** the run must read
  like an ad — a self-introduction plus promotional/CTA language. The cue set is
  general English ad-speak (``DEFAULT_PROMO_CUE_PATTERNS``), NOT a specific
  brand/feed. It is *data*, extended per feed at onboarding via
  ``extra_cue_patterns`` — the reality is this is an evolving surface, so adding a
  pattern must not mean editing code.
* **Boundary = the cue run, not speaker-recurrence.** Using content as the cut
  point keeps a following non-promo block (a "we're on vacation this week"
  note) intact, and survives diarization merging an ad reader's voice into a
  recurring cluster (measured on real data).

Diarization is a mandatory stage in this pipeline, so this detector assumes
segments are present and does nothing without them.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Pattern, Sequence, Tuple

logger = logging.getLogger(__name__)

# Corroboration: a speaker is "recurring" (host-like) if still speaking past this
# fraction of the episode. The run is an ad only if >=1 of its voices is NOT.
_RECURRING_AFTER_FRACTION = 0.5

# A cross-promo is short. Refuse to treat a leading block spanning more than this
# fraction of the episode as an ad — that is content.
_MAX_CROSSPROMO_FRACTION = 0.25

# How many consecutive non-cue segments the scan tolerates before ending the run
# (an ad has bridging lines). The run is always trimmed back to its last cue
# segment, so this only bounds the look-ahead.
_CUE_GAP = 2

# A person naming themselves / their role. General, feed-agnostic.
_SELF_INTRO_PATTERNS: Tuple[str, ...] = (
    r"\bI'?m\s+[A-Z]",
    r"\bI\s+(?:cover|report|write|host|produce|co-?host)\b",
    r"\bthis\s+is\s+[A-Z][a-z]+",
    r"\bwelcome\s+to\b",
)

# Promotional / call-to-action register — English-general ad-speak, NOT a specific
# brand or feed. Deliberately PRECISE: only phrasings rare in ordinary conversation
# (a host saying "let's visit the mothership" or discussing "media coverage" must not
# trip it). Ads whose bridge language is topic-specific (a sports cross-promo's
# "coverage"/"reporters") are closed per feed at onboarding via ``extra_cue_patterns``
# — the intended evolving surface, not a reason to broaden the default and over-cut.
DEFAULT_PROMO_CUE_PATTERNS: Tuple[str, ...] = (
    r"\b(?:download|subscribe|sign\s?up|tune\s+in|follow\s+us|learn\s+more)\b",
    r"\b(?:promo|discount|coupon)\s+code\b|\buse\s+code\b",
    r"\bsponsored\s+by\b|\bbrought\s+to\s+you\s+by\b|\bour\s+sponsor\b",
    r"\bfree\s+(?:access|trial)\b|\bour\s+app\b|\bour\s+newsletter\b"
    r"|\bour\s+(?:other\s+)?(?:show|podcast)\b",
    r"\b\w[\w-]*\.(?:com|org|net|ai|io|fm|co|app)\b",
)

_SELF_INTRO_RE = re.compile("|".join(_SELF_INTRO_PATTERNS), re.I)
_DEFAULT_PROMO_RE = re.compile("|".join(DEFAULT_PROMO_CUE_PATTERNS), re.I)


def _promo_re(extra_cue_patterns: Optional[Sequence[str]]) -> Pattern[str]:
    if not extra_cue_patterns:
        return _DEFAULT_PROMO_RE
    return re.compile("|".join((*DEFAULT_PROMO_CUE_PATTERNS, *extra_cue_patterns)), re.I)


def _spk(seg: Dict[str, Any]) -> str:
    """Speaker id, tolerating both the raw ``speaker`` and the screenplay
    ``speaker_label`` field name (offset segments use the latter)."""
    return str(seg.get("speaker") or seg.get("speaker_label") or "")


def _is_cue_segment(text: str, promo_re: Pattern[str]) -> bool:
    return bool(_SELF_INTRO_RE.search(text) or promo_re.search(text))


def _looks_like_crosspromo(block_text: str, promo_re: Pattern[str]) -> bool:
    """The run reads like a promo only with BOTH a self-introduction and a
    promotional cue — not just any non-recurring cold-open voice."""
    return bool(_SELF_INTRO_RE.search(block_text) and promo_re.search(block_text))


def _timed(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [s for s in segments if _spk(s) and (s.get("text") or "").strip()]
    out.sort(key=lambda s: float(s.get("start") or 0.0))
    return out


def detect_opening_crosspromo(
    segments: Optional[List[Dict[str, Any]]],
    *,
    extra_cue_patterns: Optional[Sequence[str]] = None,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """Detect a leading host-read cross-promo from diarized segments.

    Returns ``(run, first_post)`` — the list of cross-promo segments and the first
    segment after them — or ``None`` when there is no cross-promo. Works on both
    raw diarization segments (``speaker``/``start``/``end``/``text``) and screenplay
    offset segments (``speaker_label`` + ``char_start``/``char_end``).
    """
    if not segments:
        return None
    segs = _timed(segments)
    if len(segs) < 2:
        return None

    total_end = max(float(s.get("end") or s.get("start") or 0.0) for s in segs)
    if total_end <= 0.0:
        return None

    promo_re = _promo_re(extra_cue_patterns)

    # Recurring (host-like) voices — needed to bridge the run structurally.
    last_end: Dict[str, float] = {}
    for s in segs:
        spk = _spk(s)
        last_end[spk] = max(last_end.get(spk, 0.0), float(s.get("end") or s.get("start") or 0.0))
    recurring = {
        spk for spk, end in last_end.items() if end >= total_end * _RECURRING_AFTER_FRACTION
    }

    # Walk the leading run. A segment CONTINUES the run when it is a cue, OR spoken by
    # a non-recurring voice (the ad readers' own voices bridge their narrative — the
    # general, feed-agnostic signal), OR within _CUE_GAP of the last cue (tolerate a
    # cluster-merged reader's few no-cue lines). The run always ENDS at the last cue,
    # so a following non-recurring housekeeping note (no cue) is trimmed off.
    last_cue_idx = -1
    for i, seg in enumerate(segs):
        is_cue = _is_cue_segment(seg.get("text") or "", promo_re)
        if is_cue:
            last_cue_idx = i
            continue
        bridged = (_spk(seg) not in recurring) or (
            last_cue_idx >= 0 and (i - last_cue_idx) <= _CUE_GAP
        )
        if not bridged:
            break
    if last_cue_idx < 0 or last_cue_idx + 1 >= len(segs):
        return None

    run = segs[: last_cue_idx + 1]
    if float(run[-1].get("end") or 0.0) > total_end * _MAX_CROSSPROMO_FRACTION:
        return None  # too long to be an opening ad — leave content alone

    # Diarization corroboration: >=1 voice in the run must never recur later.
    if not ({_spk(s) for s in run} - recurring):
        return None  # all voices recur -> this is the hosts, not an ad

    if not _looks_like_crosspromo(" ".join((s.get("text") or "") for s in run), promo_re):
        return None

    return run, segs[last_cue_idx + 1]


def crosspromo_char_end(
    offset_segments: List[Dict[str, Any]],
    *,
    extra_cue_patterns: Optional[Sequence[str]] = None,
) -> int:
    """Char offset in the screenplay where an opening cross-promo ends (0 if none).

    For the ad-free pipeline: ``offset_segments`` carry ``char_start``/``char_end``,
    so the cut is the first post-run segment's ``char_start`` — exact, no mapping.
    """
    det = detect_opening_crosspromo(offset_segments, extra_cue_patterns=extra_cue_patterns)
    if not det:
        return 0
    _, first_post = det
    return int(first_post.get("char_start") or 0)


def _cut_before_segment(text: str, segment_text: str, approx_frac: float) -> int:
    """Char offset in ``text`` where the opening block ends: the start of the first
    post-run turn. Prefer an exact match of that turn's text (robust to
    non-uniform speaking rate); fall back to a proportional estimate snapped back
    to a line start so we never cut mid-turn."""
    needle = segment_text.strip()
    if needle:
        pos = text.find(needle)
        if pos != -1:
            line_start = text.rfind("\n", 0, pos)
            return line_start + 1 if line_start != -1 else 0
    approx = int(max(0.0, min(1.0, approx_frac)) * len(text))
    line_start = text.rfind("\n", 0, approx)
    return line_start + 1 if line_start != -1 else 0


def opening_crosspromo_cut(
    text: str,
    *,
    diarization_segments: Optional[List[Dict[str, Any]]] = None,
    extra_cue_patterns: Optional[Sequence[str]] = None,
) -> int:
    """Char offset in ``text`` where an opening cross-promo ends (0 if none)."""
    if not text or not diarization_segments:
        return 0
    det = detect_opening_crosspromo(diarization_segments, extra_cue_patterns=extra_cue_patterns)
    if not det:
        return 0
    run, first_post = det
    total_end = max(float(s.get("end") or s.get("start") or 0.0) for s in run) or 1.0
    return _cut_before_segment(
        text,
        str(first_post.get("text") or ""),
        approx_frac=float(first_post.get("start") or 0.0) / total_end,
    )


def excise_opening_crosspromo(
    text: str,
    *,
    diarization_segments: Optional[List[Dict[str, Any]]] = None,
    host_speaker_id: Optional[str] = None,
    extra_cue_patterns: Optional[Sequence[str]] = None,
) -> str:
    """Remove a leading host-read cross-promo from summarization text (#1188).

    ``host_speaker_id`` is accepted for call-site symmetry; detection is driven by
    speaker-recurrence across all segments, not a single configured host.
    """
    cut = opening_crosspromo_cut(
        text, diarization_segments=diarization_segments, extra_cue_patterns=extra_cue_patterns
    )
    if cut <= 0:
        return text
    excised = text[cut:].lstrip()
    logger.debug("excise_opening_crosspromo: removed %d chars of opening cross-promo", cut)
    return excised or text
