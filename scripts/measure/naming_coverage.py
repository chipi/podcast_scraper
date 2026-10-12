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
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from podcast_scraper.quality.naming_coverage import (  # noqa: E402 - the scan lives in src/
    scan_episode,
    summarise,
)


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
