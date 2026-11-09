"""Character spans of transcript segments: which segment a char offset falls in (moved verbatim
from ``gi.pipeline``, E04 S4; the search indexer times transcript chunks with it).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Max |len(transcript) - sum(segment text)| before skipping segment-based timestamps (FR2.2).
# Avoids mapping quote char offsets to wrong audio after reformatting or edited transcripts.
SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA = 50


def _segment_char_spans(
    transcript: str, segments: List[Dict[str, Any]]
) -> Optional[List[Tuple[int, int, Dict[str, Any]]]]:
    """Return ``[(seg_start, seg_end, seg)]`` char spans for the segments.

    #974: when segments carry explicit ``char_start`` / ``char_end`` (the ad-free
    processing base, where each segment knows its exact position in the screenplay —
    markers and all), use those directly — a quote ``char_start`` maps to its segment
    with no heuristic. Otherwise fall back to cumulative ``len(text)`` positions and
    apply the legacy alignment guard (returning ``None`` when the transcript length and
    the summed segment text diverge by more than ``SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA``,
    so callers skip an unreliable mapping on a reformatted transcript).
    """
    dict_segs = [s for s in segments if isinstance(s, dict)]
    if dict_segs and all("char_start" in s and "char_end" in s for s in dict_segs):
        return [(int(s["char_start"]), int(s["char_end"]), s) for s in dict_segs]

    spans: List[Tuple[int, int, Dict[str, Any]]] = []
    pos = 0
    for seg in dict_segs:
        text = seg.get("text") or ""
        spans.append((pos, pos + len(text), seg))
        pos += len(text)
    if not spans:
        return None
    if abs(len(transcript) - pos) > SEGMENT_TRANSCRIPT_ALIGNMENT_MAX_DELTA:
        return None
    return spans
