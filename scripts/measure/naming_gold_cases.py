#!/usr/bin/env python3
"""Build BLIND case files for a naming gold set (docs/guides/NAMING_GATE_RUNBOOK.md). Read-only.

One case per selected served episode: feed title / description / authors, episode title /
description, per diarized voice its talk share, first and last second, the opening ~1,800 and
closing 400 characters of its own turns, and the opening line of the first 45 turns. It
deliberately carries NO published speaker names (``speaker_label``), so a label cannot copy the
pipeline's answer. Descriptions and turn lines are capped as in the first sets.

Selection is seeded and reproducible: served episodes (newest run per episode) in one language,
with at least two diarized voices, excluding every episode an existing set already holds
(``--exclude`` the ``cases`` folders of dev/val), so a new set is disjoint from them.

    python scripts/measure/naming_gold_cases.py --corpus /app/output --out /tmp/val_v2/cases \\
        --n 150 --seed 20261012 --prefix w \\
        --exclude naming_gold/dev_v1/cases naming_gold/val_v1/cases

The cases are derived from real episodes: they live under .test_outputs/naming_gold/ and are
NEVER committed. First used 2026-10-12 for val_v2, after val_v1 was read while tuning (#2304).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

OPENING_CHARS = 1800
CLOSING_CHARS = 400
FIRST_TURNS = 45
TURN_CHARS = 220
FEED_DESCRIPTION_CHARS = 1500
EPISODE_DESCRIPTION_CHARS = 2500


def _load(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cap(text: Any, chars: int) -> Any:
    return text[:chars] if isinstance(text, str) else text


def _segments(meta: Path, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
    if not rel.endswith(".txt"):
        return []
    raw = _load(meta.parent.parent / rel.replace(".txt", ".segments.json"))
    rows = raw if isinstance(raw, list) else (raw or {}).get("segments") or []
    return [r for r in rows if isinstance(r, dict) and r.get("speaker")]


def _turns(segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The first segment at each change of voice (its text only), the first 45 of them."""
    out: List[Dict[str, Any]] = []
    for s in segs:
        if out and out[-1]["voice"] == s["speaker"]:
            continue
        out.append(
            {
                "voice": s["speaker"],
                "start": round(float(s.get("start") or 0)),
                "text": str(s.get("text") or "")[:TURN_CHARS],
            }
        )
        if len(out) == FIRST_TURNS:
            break
    return out


def build_case(
    meta: Path, corpus: Path, case_id: str, selected_as: str
) -> Optional[Dict[str, Any]]:
    """The blind case for one episode, or None when it has fewer than two voices."""
    doc = _load(meta)
    if not isinstance(doc, dict):
        return None
    segs = _segments(meta, doc)
    talk: Dict[str, float] = {}
    first: Dict[str, float] = {}
    last: Dict[str, float] = {}
    texts: Dict[str, List[str]] = {}
    for s in segs:
        v, a, b = s["speaker"], float(s.get("start") or 0), float(s.get("end") or 0)
        talk[v] = talk.get(v, 0.0) + max(0.0, b - a)
        first[v] = min(first.get(v, a), a)
        last[v] = b
        texts.setdefault(v, []).append(str(s.get("text") or ""))
    if len(talk) < 2:
        return None
    total = sum(talk.values()) or 1.0
    feed = doc.get("feed") or {}
    episode = doc.get("episode") or {}
    voices = []
    for v in sorted(talk, key=lambda x: -talk[x]):
        joined = " ".join(texts[v])
        voices.append(
            {
                "voice": v,
                "share": round(talk[v] / total, 3),
                "first_s": round(first[v]),
                "last_s": round(last[v]),
                "opening": joined[:OPENING_CHARS],
                "closing": joined[-CLOSING_CHARS:],
            }
        )
    return {
        "case_id": case_id,
        "selected_as": selected_as,
        "meta_relpath": str(meta.relative_to(corpus)),
        "feed": {
            "title": feed.get("title"),
            "description": _cap(feed.get("description"), FEED_DESCRIPTION_CHARS),
            "authors": feed.get("authors") or [],
        },
        "episode": {
            "title": episode.get("title"),
            "description": _cap(episode.get("description"), EPISODE_DESCRIPTION_CHARS),
        },
        "duration_s": round(max(last.values())),
        "voices": voices,
        "first_turns": _turns(segs),
    }


def episode_key(meta: Path, corpus: Path) -> str:
    """Feed folder plus episode id: a reprocessed episode keeps it while its run path changes."""
    rel = meta.relative_to(corpus)
    doc = _load(meta)
    eid = ((doc or {}).get("episode") or {}).get("episode_id") if isinstance(doc, dict) else None
    return f"{rel.parts[1]}/{eid}" if eid and len(rel.parts) > 1 else str(rel)


def excluded_episodes(folders: Iterable[Path], corpus: Path) -> Set[str]:
    """The episodes the existing sets' cases already hold."""
    out: Set[str] = set()
    for folder in folders:
        for p in Path(folder).glob("*.json"):
            doc = _load(p)
            if isinstance(doc, dict) and doc.get("meta_relpath"):
                out.add(episode_key(corpus / str(doc["meta_relpath"]), corpus))
    return out


def _language(doc: Dict[str, Any]) -> str:
    feed = doc.get("feed") or {}
    return str(feed.get("language") or (doc.get("content") or {}).get("language") or "")[:2]


def select(
    metas: List[Path], corpus: Path, n: int, seed: int, exclude: Set[str], lang: Optional[str]
) -> List[Path]:
    pool = []
    for meta in sorted(metas):
        doc = _load(meta)
        if episode_key(meta, corpus) in exclude:
            continue
        if not isinstance(doc, dict) or (lang and _language(doc).lower() not in (lang, "")):
            continue
        pool.append(meta)
    random.Random(seed).shuffle(pool)
    return pool[: n * 2]  # twice n: some candidates have fewer than two voices


def main(argv: Optional[List[str]] = None) -> int:
    from podcast_scraper.upgrade.corpus_selection import select_served_artifacts

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prefix", default="w", help="case id prefix (dev=c, val_v1=v)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--exclude", nargs="*", type=Path, default=[])
    args = ap.parse_args(argv)
    corpus = args.corpus
    metas = select_served_artifacts(corpus, ".metadata.json")[0]
    exclude = excluded_episodes(args.exclude, corpus)
    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for meta in select(metas, corpus, args.n, args.seed, exclude, args.lang or None):
        if written >= args.n:
            break
        case = build_case(meta, corpus, f"{args.prefix}{written + 1:03d}", "validation")
        if case is None:
            continue
        (args.out / f"{case['case_id']}.json").write_text(
            json.dumps(case, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        written += 1
    print(f"wrote {written} case(s) to {args.out}; excluded {len(exclude)} existing episode(s)")
    return 0 if written == args.n else 1


if __name__ == "__main__":
    sys.exit(main())
