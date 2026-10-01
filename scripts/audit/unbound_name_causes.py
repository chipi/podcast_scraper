"""Why did a name the metadata STATED end up on no voice? One bucket per served episode (#2200).

The M3 measurement (2026-10-01) assumed guest corroboration kept stated names away from voices.
It does not: every stated name reaches the ADR-110 resolver as a candidate. What decides the
outcome is whether the resolver's retrieval finds the name SPOKEN, and what the model and the
guards do with it. This classifies each served episode that hides ``unknown``-voice insights by
its best-placed unbound name:

    A  retrieval finds the name — the model abstained or a guard discarded its answer
    B  retrieval misses; the opening has the first name AND a surname within 2 edits (ASR)
    C  retrieval misses; the opening has the first name only
    D  the name is not in the opening (first 180 s) at all
    single-token   the unbound name is one word
    no unbound name recorded / no segments or diagnostics

Read-only. Judged on the SERVED copy (``dedupe_metadata_paths_newest_run_per_episode``), the
same selection the API makes.

Usage::

    python scripts/audit/unbound_name_causes.py --corpus-dir /app/output [--json]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

OPENING_S = 180.0

BUCKET_A = "A retrieval finds the name (model/guards declined)"
BUCKET_B = "B retrieval misses; first name + surname within 2 edits in the opening"
BUCKET_C = "C retrieval misses; first name only in the opening"
BUCKET_D = "D not in the opening"
SINGLE = "single-token name"
NO_UNBOUND = "no unbound name recorded"
NO_FILES = "no segments/diagnostics"


def _unknown_insights(gi_path: Path) -> int:
    try:
        gi = json.loads(gi_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return sum(
        1
        for n in gi.get("nodes") or []
        if n.get("type") == "Insight"
        and (n.get("properties") or {}).get("speaker_voice_type") == "unknown"
    )


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein, case-insensitive. Local, so the audit runs against any deployed image."""
    a, b = a.lower(), b.lower()
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def classify_name(name: str, turns: Sequence[Tuple[str, str]], opening: str) -> str:
    """The bucket for one unbound name, judged by the DEPLOYED resolver's retrieval."""
    from podcast_scraper.speaker_detectors.resolution import retrieve_mentions

    toks = name.split()
    if len(toks) < 2:
        return SINGLE
    if retrieve_mentions(name, turns):
        return BUCKET_A
    first, last = toks[0], toks[-1]
    words = re.findall(r"[A-Za-z'’\-]+", opening)
    near = len(last) >= 4 and any(_edit_distance(w, last) <= 2 for w in words if len(w) >= 4)
    first_in = re.search(rf"\b{re.escape(first)}\b", opening) is not None
    if near and first_in:
        return BUCKET_B
    if first_in:
        return BUCKET_C
    return BUCKET_D


def classify_episode(meta_path: Path) -> Tuple[str, int, Optional[str]]:
    """``(bucket, hidden unknown-voice insights, example name)`` for one served episode."""
    stem = meta_path.name[: -len(".metadata.json")]
    hidden = _unknown_insights(meta_path.with_name(f"{stem}.gi.json"))
    tdir = meta_path.parent.parent / "transcripts"
    seg = tdir / f"{stem}.segments.json"
    diag = tdir / f"{stem}.speakers.diagnostics.json"
    if not seg.is_file() or not diag.is_file():
        return NO_FILES, hidden, None
    raw = json.loads(seg.read_text(encoding="utf-8"))
    segs: List[Dict[str, Any]] = raw if isinstance(raw, list) else raw.get("segments", [])
    turns = [(str(s.get("speaker") or ""), str(s.get("text") or "")) for s in segs]
    opening = " ".join(
        str(s.get("text") or "") for s in segs if float(s.get("start") or 0.0) < OPENING_S
    )
    summary = json.loads(diag.read_text(encoding="utf-8")).get("summary") or {}
    best: Optional[Tuple[str, str]] = None
    for name in summary.get("unbound_names") or []:
        bucket = classify_name(str(name), turns, opening)
        if best is None or bucket < best[0]:
            best = (bucket, str(name))
    if best is None:
        return NO_UNBOUND, hidden, None
    return best[0], hidden, best[1]


def run(corpus_dir: Path) -> Dict[str, Dict[str, Any]]:
    from podcast_scraper.search.corpus_scope import dedupe_metadata_paths_newest_run_per_episode

    paths = list(corpus_dir.glob("feeds/*/run_*/metadata/*.metadata.json"))
    out: Dict[str, Dict[str, Any]] = collections.defaultdict(
        lambda: {"episodes": 0, "hidden_insights": 0, "example": None}
    )
    for p in dedupe_metadata_paths_newest_run_per_episode(corpus_dir, paths):
        meta = Path(p)
        stem = meta.name[: -len(".metadata.json")]
        if _unknown_insights(meta.with_name(f"{stem}.gi.json")) == 0:
            continue
        bucket, hidden, example = classify_episode(meta)
        row = out[bucket]
        row["episodes"] += 1
        row["hidden_insights"] += hidden
        row["example"] = row["example"] or example
    return dict(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = ap.parse_args(argv)
    result = run(args.corpus_dir)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    for bucket in sorted(result):
        row = result[bucket]
        print(
            f"{row['episodes']:5d} eps {row['hidden_insights']:7d} hidden  {bucket}"
            + (f"   e.g. {row['example']!r}" if row["example"] else "")
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
