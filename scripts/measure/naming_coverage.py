#!/usr/bin/env python3
"""Naming COVERAGE of a corpus: how much of what we publish is attributed to a named person.

Every other naming check compares old code against new code, so an episode that both name nobody
"agrees" and stays invisible. This one measures the absolute level (#2304: 296 English episodes with
insights published no named voice on 2026-10-10, and nothing reported it).

Per served episode with a GI artifact:

- ``quotes`` / ``named_quotes``: GI quotes, and those whose ``speaker_id`` is a person;
- ``named_voices``: diarized voices published under a name in the segments;
- ``group``: why an episode with no named voice is unnamed:
  ``no_segments``, ``no_diagnostics`` (an older record that cannot be replayed), or
  ``stored_unnamed`` (replayable).

With ``--replay``, each ``stored_unnamed`` episode is replayed through the roster (the installed
one, or ``--roster PATH``), with the host pool rebuilt from the feed (``roster_replay --repool``).
The group is then refined to:

- ``replay_names_main``: today's code names the voice with the most segments;
- ``replay_names_some``: it names some voice, but not that one;
- ``replay_hosts_known``: it names nobody although the host pool is non-empty;
- ``replay_no_host``: it names nobody and no host is known.

The replay cannot run the LLM naming step (it uses stored verdicts only), bylines or per-feed
config. Its result is therefore a floor on what production would do, not a prediction.

    python scripts/measure/naming_coverage.py --corpus /app/output [--replay] [--lang en] \\
        [--json out.json] [--max-unnamed N]

Exit 1 when ``--max-unnamed`` is given and more episodes than that publish no named voice.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

_DIARIZATION_LABEL = re.compile(r"^SPEAKER(_\d+)?$")


def _segments(path: Path) -> Optional[List[Dict[str, Any]]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rows = raw if isinstance(raw, list) else raw.get("segments", [])
    return [r for r in rows if isinstance(r, dict)]


def _language(doc: Dict[str, Any]) -> str:
    feed = doc.get("feed") or {}
    content = doc.get("content") or {}
    return str(feed.get("language") or content.get("language") or "")[:2].lower()


def scan_episode(meta: Path, lang: Optional[str]) -> Optional[Dict[str, Any]]:
    """One episode's coverage row, or None when it is out of scope."""
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if lang and _language(doc) not in (lang, ""):
        return None
    stem = meta.name[: -len(".metadata.json")]
    gi = next(iter(meta.parent.glob(stem + "*.gi.json")), None)
    if gi is None:
        return None
    try:
        nodes = json.loads(gi.read_text(encoding="utf-8")).get("nodes") or []
    except (OSError, ValueError):
        return None
    quotes = [n for n in nodes if isinstance(n, dict) and n.get("type") == "Quote"]
    named_quotes = sum(
        1
        for q in quotes
        if str((q.get("properties") or {}).get("speaker_id") or "").startswith("person:")
    )
    rel = str((doc.get("content") or {}).get("transcript_file_path") or "")
    run = meta.parent.parent
    row: Dict[str, Any] = {
        "meta": str(meta),
        "feed": str((doc.get("feed") or {}).get("title") or "")[:40],
        "title": str((doc.get("episode") or {}).get("title") or "")[:60],
        "quotes": len(quotes),
        "named_quotes": named_quotes,
    }
    segs = _segments(run / rel.replace(".txt", ".segments.json")) if rel else None
    if segs is None:
        row.update(named_voices=0, group="no_segments")
        return row
    labels = collections.Counter(
        str(s.get("speaker_label") or s.get("speaker") or "") for s in segs
    )
    named = {lab for lab in labels if lab and not _DIARIZATION_LABEL.match(lab)}
    row["named_voices"] = len(named)
    row["main_voice"] = labels.most_common(1)[0][0] if labels else None
    if named:
        row["group"] = "named"
    elif (run / rel.replace(".txt", ".speakers.diagnostics.json")).is_file():
        row["group"] = "stored_unnamed"
    else:
        row["group"] = "no_diagnostics"
    return row


def refine_by_replay(rows: List[Dict[str, Any]], corpus: Path, roster_path: Optional[Path]) -> None:
    """Replace ``stored_unnamed`` with what today's roster does on replay (see the module doc)."""
    import roster_replay as rr

    variant = rr.load_variant(rr._variant_arg([f"roster={roster_path}"] if roster_path else []))
    roster = variant["roster"]
    if getattr(roster, "_replay_hosts", None) is None:
        import podcast_scraper.speaker_detectors.hosts as hosts

        roster._replay_hosts = hosts
    rr.REPOOL = True
    rr.index_siblings(corpus)
    for row in rows:
        if row["group"] != "stored_unnamed":
            continue
        try:
            ep = rr.load_episode(Path(row["meta"]))
            pool = rr.roster_inputs(ep, roster)["known_hosts"]
            result = rr.replay(roster, ep)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not stop the scan
            row["group"] = "replay_error"
            row["error"] = type(exc).__name__
            continue
        named = {v: r.name for v, r in result.by_voice.items() if getattr(r, "named", False)}
        row["replay_named"] = named
        row["host_pool"] = list(pool)
        if row.get("main_voice") in named:
            row["group"] = "replay_names_main"
        elif named:
            row["group"] = "replay_names_some"
        elif pool:
            row["group"] = "replay_hosts_known"
        else:
            row["group"] = "replay_no_host"


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = collections.Counter(r["group"] for r in rows)
    unnamed = [r for r in rows if r["group"] != "named"]
    return {
        "episodes": len(rows),
        "episodes_no_named_voice": len(unnamed),
        "quotes": sum(r["quotes"] for r in rows),
        "named_quotes": sum(r["named_quotes"] for r in rows),
        "groups": dict(groups),
        "unnamed_by_feed": dict(collections.Counter(r["feed"] for r in unnamed).most_common(15)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    from podcast_scraper.upgrade.corpus_selection import select_served_artifacts

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--lang", default="en", help="two-letter feed language to keep ('' = all)")
    ap.add_argument("--replay", action="store_true", help="refine unnamed groups by replay")
    ap.add_argument("--roster", type=Path, help="roster module to replay (default: installed)")
    ap.add_argument("--json", type=Path, help="write every row here")
    ap.add_argument("--max-unnamed", type=int, help="exit 1 above this many unnamed episodes")
    args = ap.parse_args(argv)

    rows = [
        row
        for meta in select_served_artifacts(args.corpus, ".metadata.json")[0]
        if (row := scan_episode(meta, args.lang or None)) is not None
    ]
    if args.replay:
        refine_by_replay(rows, args.corpus, args.roster)
    summary = summarise(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False))
    if args.max_unnamed is not None and summary["episodes_no_named_voice"] > args.max_unnamed:
        print(f"FAIL: {summary['episodes_no_named_voice']} > {args.max_unnamed} unnamed episodes")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
