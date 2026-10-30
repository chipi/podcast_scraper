"""The corpus scoreboard: what a listener can see, measured the same way every time.

The yardstick of `docs/guides/CORPUS_CHANGE_METHOD.md` §3. Every row has one definition here, so a
before/after table on two corpus copies compares like with like. Over the served episodes
(`select_served_artifacts`), per language and in total:

- ``S1``: insights hidden because a voice went unnamed (`surfaceable` False, type ``unknown``);
- ``S1n``: insights under a real name that are still hidden (type ``unknown``/``unidentified``);
- ``S1c``: insights whose visibility contradicts their voice type (must be 0);
- ``S2``: all hidden insights, by voice type;
- ``S3``: hidden ``unidentified`` insights on a voice holding at least 30% of the talk. The voice
  is found through the insight's first grounded quote and the segment it sits in (the raw voice
  id), never through the insight's ``speaker``, which is a name after a rename;
- ``S10a`` / ``S10b``: episodes with insights of which none is visible, without and with the
  episode panel's unattributed fallback (`server.gi_view.insights_from_gi`);
- ``S12``: quotes whose offsets do not slice to their own text in the file they name;
- ``A1``: sponsor-marker hits in the served ad-free text (an indicator: `AD_MARKERS`).

Every counter in `KEYS` is written out, zeros included, with the population gaps beside the rows
(no GI, quotes without offsets or with an unresolved file, episodes without an ad-free text, S3
insights whose voice cannot be found).

**Not measured here**, though §3 lists them:
- S4/S5, named-quote coverage: `quality.naming_coverage`. It cannot gate, because it improved under
  a migration that hid more insights (0031 then, now 0032, 2026-10-17);
- S6–S8, the gold numbers: `scripts/measure/naming_gate.py`, with S7/S8 still to build;
- S9, unattributed talk share;
- S11, non-person names on shown insights;
- A2–A4, which compare two corpus states (talk cut with an ad, ad-built output removed, promo
  restored).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PRINCIPAL_SHARE = 0.30
_RAW_VOICE = re.compile(r"^SPEAKER(_\d+)?$")

#: Sponsor-read phrases (regex, matched case-insensitively between word boundaries). A hit in
#: served ad-free text is ad copy the cut left. Bare "use code" and "sponsored by" are out: read in
#: full on 2026-10-16, 44 of 60 "use code" hits were talk ("because code is", "use Codex") and about
#: 21 of 88 "sponsored by" hits were legislation, programmes and departments.
AD_MARKERS: Tuple[str, ...] = (
    r"brought to you by",
    r"(?:is|are|also) sponsored by",
    r"(?:season's|our) presenting sponsor",
    r"promo code",
    r"discount code",
    r"members fdic",
    r"terms and conditions apply",
    r"after a short word from our sponsors?",
    r"a word from our sponsors?",
    r"this message comes from",
    r"at checkout",
)
_AD_MARKER = re.compile(r"\b(?:" + "|".join(AD_MARKERS) + r")\b", re.IGNORECASE)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _rows(payload: Any) -> List[Dict[str, Any]]:
    rows = payload if isinstance(payload, list) else (payload or {}).get("segments")
    return [r for r in rows or [] if isinstance(r, dict)]


class _Texts:
    """The episode's transcript files, read once, keyed by file name."""

    def __init__(self, transcripts: Path) -> None:
        self.dir = transcripts
        self._text: Dict[str, Optional[str]] = {}
        self._spans: Dict[str, Optional[List[Tuple[int, int, Dict[str, Any]]]]] = {}

    def text(self, name: str) -> Optional[str]:
        if name not in self._text:
            p = self.dir / name
            self._text[name] = p.read_text(encoding="utf-8") if p.is_file() else None
        return self._text[name]

    def spans(self, name: str) -> Optional[List[Tuple[int, int, Dict[str, Any]]]]:
        """Segment char spans for the transcript ``name`` (``x.txt`` → ``x.segments.json``)."""
        if name not in self._spans:
            from ..gi.pipeline import _segment_char_spans

            text = self.text(name)
            segs = _rows(_load(self.dir / (name[: -len(".txt")] + ".segments.json")))
            self._spans[name] = _segment_char_spans(text, segs) if text and segs else None
        return self._spans[name]


def _talk_share(diagnostics: Any) -> Dict[str, float]:
    voices = [v for v in (diagnostics or {}).get("voices") or [] if isinstance(v, dict)]
    total = sum(float(v.get("talk_time_s") or 0) for v in voices)
    if not total:
        return {}
    return {str(v.get("voice")): float(v.get("talk_time_s") or 0) / total for v in voices}


def _voice_of_quote(quote: Dict[str, Any], texts: _Texts) -> Optional[str]:
    props = quote.get("properties") or {}
    ref, start = props.get("transcript_ref"), props.get("char_start")
    if not isinstance(ref, str) or not isinstance(start, int):
        return None
    spans = texts.spans(Path(ref).name)
    for a, b, seg in spans or ():
        if a <= start < b:
            v = seg.get("speaker")
            return str(v) if v else None
    return None


def _count_ad_markers(c: Counter, texts: _Texts, tr: str) -> None:
    adfree = texts.text(tr[: -len(".txt")] + ".adfree.txt") if tr.endswith(".txt") else None
    if adfree is None:
        c["A1_episodes_no_adfree"] += 1
        return
    hits = len(_AD_MARKER.findall(adfree))
    c["A1_marker_hits"] += hits
    c["A1_episodes_with_hits"] += 1 if hits else 0


def _count_quote(c: Counter, props: Dict[str, Any], texts: _Texts) -> None:
    c["quotes"] += 1
    ref, cs, ce = props.get("transcript_ref"), props.get("char_start"), props.get("char_end")
    if not (isinstance(ref, str) and isinstance(cs, int) and isinstance(ce, int)):
        c["quotes_no_offsets"] += 1
        return
    body = texts.text(Path(ref).name)
    if body is None:
        c["quotes_ref_unresolved"] += 1
    elif body[cs:ce] != props.get("text"):
        c["S12_quotes_off_text"] += 1


def _count_insight(c: Counter, props: Dict[str, Any]) -> bool:
    """S1, S1n, S1c, S2 for one insight; True when it is a hidden ``unidentified`` one (S3)."""
    vt = props.get("speaker_voice_type") or None
    hidden = props.get("surfaceable") is False
    speaker = str(props.get("speaker") or "")
    c["insights"] += 1
    if (hidden and vt in (None, "person")) or (vt == "unknown" and not hidden):
        c["S1c_inconsistent"] += 1
    if not hidden:
        return False
    c["S2_hidden"] += 1
    c[f"S2_hidden_{vt or 'none'}"] += 1
    if vt == "unknown":
        c["S1_hidden_unknown"] += 1
    if vt in ("unknown", "unidentified") and speaker and not _RAW_VOICE.match(speaker):
        c["S1n_named_but_hidden"] += 1
    return vt == "unidentified"


def _count_principal(c: Counter, voice: Optional[str], share: Dict[str, float]) -> None:
    if not share:
        c["S3_outside_no_diagnostics"] += 1
    elif voice is None:
        c["S3_outside_no_voice"] += 1
    elif voice not in share:
        c["S3_voice_not_in_diagnostics"] += 1
    elif share[voice] >= PRINCIPAL_SHARE:
        c["S3_principal_unidentified"] += 1


def score_episode(meta_path: Path) -> Optional[Dict[str, Any]]:
    """One served episode's counters, or None when it has no readable metadata."""
    from ..server.gi_view import insights_from_gi

    meta = _load(meta_path)
    if not isinstance(meta, dict):
        return None
    run = meta_path.parent.parent
    stem = meta_path.name[: -len(".metadata.json")]
    c: Counter = Counter()
    lang = str((meta.get("episode") or {}).get("language") or "unknown")
    tr = Path(str((meta.get("content") or {}).get("transcript_file_path") or "")).name
    texts = _Texts(run / "transcripts")
    _count_ad_markers(c, texts, tr)
    gi = _load(run / "metadata" / f"{stem}.gi.json")
    if not isinstance(gi, dict):
        c["episodes_no_gi"] += 1
        return {"language": lang, "counts": c}
    c["episodes"] += 1
    nodes = [n for n in gi.get("nodes") or [] if isinstance(n, dict)]
    by_id = {str(n.get("id")): n for n in nodes}
    first_quote: Dict[str, str] = {}
    for e in gi.get("edges") or []:
        if isinstance(e, dict) and e.get("type") == "SUPPORTED_BY":
            first_quote.setdefault(str(e.get("from")), str(e.get("to")))
    share: Optional[Dict[str, float]] = None
    insights = 0
    for n in nodes:
        props = n.get("properties") or {}
        if n.get("type") == "Quote":
            _count_quote(c, props, texts)
            continue
        if n.get("type") != "Insight":
            continue
        insights += 1
        if not _count_insight(c, props):
            continue
        if share is None:
            diag_name = tr[: -len(".txt")] + ".speakers.diagnostics.json" if tr else None
            share = _talk_share(_load(run / "transcripts" / diag_name) if diag_name else None)
        q = by_id.get(first_quote.get(str(n.get("id")), ""))
        _count_principal(c, _voice_of_quote(q, texts) if q else None, share)
    if insights:
        if not insights_from_gi(gi):
            c["S10a_zero_visible"] += 1
        if not insights_from_gi(gi, unattributed_fallback=True):
            c["S10b_zero_visible_panel"] += 1
    return {"language": lang, "counts": c}


def score_corpus(root: Path) -> Dict[str, Any]:
    """The scoreboard over every served episode under ``root``: totals and per language."""
    from ..upgrade.corpus_selection import select_served_artifacts

    total: Counter = Counter()
    per_lang: Dict[str, Counter] = defaultdict(Counter)
    for meta in select_served_artifacts(root, ".metadata.json")[0]:
        row = score_episode(meta)
        if row is None:
            total["metadata_unreadable"] += 1
            continue
        total.update(row["counts"])
        per_lang[row["language"]].update(row["counts"])
    return {
        "total": _with_zeros(total),
        "per_language": {k: _with_zeros(v) for k, v in sorted(per_lang.items())},
    }


#: Every counter the scoreboard emits. A zero is written out, so "0" never reads as "not measured".
KEYS: Tuple[str, ...] = (
    "metadata_unreadable",
    "episodes",
    "episodes_no_gi",
    "insights",
    "S1_hidden_unknown",
    "S1n_named_but_hidden",
    "S1c_inconsistent",
    "S2_hidden",
    "S2_hidden_unknown",
    "S2_hidden_unidentified",
    "S2_hidden_commercial",
    "S2_hidden_none",
    "S3_principal_unidentified",
    "S3_outside_no_diagnostics",
    "S3_outside_no_voice",
    "S3_voice_not_in_diagnostics",
    "S10a_zero_visible",
    "S10b_zero_visible_panel",
    "quotes",
    "quotes_no_offsets",
    "quotes_ref_unresolved",
    "S12_quotes_off_text",
    "A1_marker_hits",
    "A1_episodes_with_hits",
    "A1_episodes_no_adfree",
)


def _with_zeros(counts: Counter) -> Dict[str, int]:
    out = {k: int(counts.get(k, 0)) for k in KEYS}
    out.update({k: int(v) for k, v in counts.items() if k not in out})
    return dict(sorted(out.items()))
