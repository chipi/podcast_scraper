"""Which SERVED episodes carry a KG no extractor produced — and fail if any do.

WHY THIS EXISTS (#2199). The 2026-09-29 ADR-156 repair measured its population by taking the newest
``kg.json`` per episode **by file mtime**. A migration had rewritten every ``kg.json`` at the same
minute (2026-09-22 15:46), so mtime tied, and for two episodes the count landed on a good copy in
an OLDER run while the app served a fabricated ``topic_labels`` graph from the newer one. Those two
were never repaired, and the repair reported them as fine.

The app decides what it serves with ``dedupe_metadata_paths_newest_run_per_episode``. So does this:
every episode is judged on the copy the API actually serves, never on "the newest file".

A served KG is BAD when its ``extraction.model_version`` is one of:

    topic_labels                 topics fabricated from summary bullets (ADR-156)
    provider:extraction_failed   extraction ran and produced nothing
    no_extractor                 no extractor was configured
    (missing)                    the served run has no kg.json at all

Read-only; ``--worklist`` writes only the path you name (one episode_id per line, the format
``--reprocess-episode-ids`` takes). Exit 1 when any served KG is bad, so it can gate a repair.

Usage::

    python scripts/audit/served_kg_provenance.py --corpus-dir /app/output
    python scripts/audit/served_kg_provenance.py --corpus-dir /app/output --worklist /tmp/ids.txt
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

BAD_PROVENANCE = frozenset({"topic_labels", "provider:extraction_failed", "no_extractor"})
MISSING = "(missing kg.json)"


def _load(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def served_metadata_paths(root: Path) -> List[Path]:
    """The metadata record the app serves for each episode — the central membership rule."""
    from podcast_scraper.search.corpus_scope import (
        dedupe_metadata_paths_newest_run_per_episode,
    )

    metas = [
        p
        for p in sorted(root.glob("feeds/*/run_*/metadata/*.metadata.json"))
        if ".trash" not in p.parts
    ]
    return sorted(dedupe_metadata_paths_newest_run_per_episode(root, metas))


def scan(root: Path) -> Dict[str, Any]:
    """``{"counts": {prov: n}, "bad": [{episode_id, feed, run, provenance}], "served": n}``."""
    counts: collections.Counter[str] = collections.Counter()
    bad: List[Dict[str, str]] = []
    served = served_metadata_paths(root)
    for md_path in served:
        stem = str(md_path)[: -len(".metadata.json")]
        kg_path = Path(stem + ".kg.json")
        if kg_path.is_file():
            prov = str((_load(kg_path).get("extraction") or {}).get("model_version") or "(none)")
        else:
            prov = MISSING
        counts[prov] += 1
        if prov in BAD_PROVENANCE or prov == MISSING:
            md = _load(md_path)
            parts = md_path.relative_to(root).parts
            bad.append(
                {
                    "episode_id": str((md.get("episode") or {}).get("episode_id") or ""),
                    "feed": parts[1] if len(parts) > 1 else "?",
                    "run": parts[2] if len(parts) > 2 else "?",
                    "provenance": prov,
                }
            )
    return {"served": len(served), "counts": dict(counts), "bad": bad}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus-dir", required=True, type=Path)
    ap.add_argument("--worklist", type=Path, default=None, help="Write bad episode_ids here.")
    ap.add_argument("--json", type=Path, default=None, help="Write the full result here.")
    args = ap.parse_args(argv)

    result = scan(args.corpus_dir)
    print(f"served episodes: {result['served']}")
    for prov, n in sorted(result["counts"].items(), key=lambda kv: -kv[1]):
        flag = "  <- BAD" if prov in BAD_PROVENANCE or prov == MISSING else ""
        print(f"  {n:6d}  {prov}{flag}")
    for row in result["bad"]:
        print(f"BAD {row['provenance']:28s} {row['episode_id']}  {row['feed']}/{row['run']}")
    if args.worklist is not None:
        ids = sorted({r["episode_id"] for r in result["bad"] if r["episode_id"]})
        args.worklist.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    return 1 if result["bad"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
