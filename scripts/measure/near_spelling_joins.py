#!/usr/bin/env python3
"""List the served ledger's person joins that rest on a near spelling alone and no read has
reviewed (``identity/near_joins``). Read-only. Run it after each ledger rebuild, read every pair,
and add the verdicts to ``upgrade/lists/s5d_reviewed_fuzzy_joins.json``.

    python scripts/measure/near_spelling_joins.py --corpus <root> [--json out.jsonl]

Prints the count of open pairs and one line per pair (ids, names, episodes, voice and publisher
flags), most episodes first.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.identity.guards import collect_episode_evidence, load_reviewed  # noqa: E402
from podcast_scraper.identity.ledger import read_ledger  # noqa: E402
from podcast_scraper.identity.near_joins import pairs, unreviewed_near_joins  # noqa: E402
from podcast_scraper.kg.entity_clusters import collect_entity_candidates  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    ledger, state = read_ledger(args.corpus)
    if ledger is None:
        print(f"no valid ledger ({state}): nothing is joined")
        return 0
    rows = unreviewed_near_joins(
        ledger,
        collect_entity_candidates(args.corpus),
        collect_episode_evidence(args.corpus),
        load_reviewed(),
    )
    open_pairs = pairs(rows)
    print(f"unreviewed near-spelling joins: {len(open_pairs)} pairs, {len(rows)} members")
    for r in sorted(rows, key=lambda r: (-r["episodes"], r["id"])):
        flags = ("V" if r["voice"] else "") + ("P" if r["publisher"] else "") or "-"
        near = "; ".join(f"{b['id']} {b['name']!r} {b['episodes']}ep" for b in r["near"])
        print(f"  {r['id']} {r['name']!r} {r['episodes']}ep {flags}  ~  {near}")
    if args.json:
        args.json.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
