"""Re-key or flag every listener's stored references, once (#2345). Dry-run unless ``--apply``.

Why: theme (``tc:``) and storyline (``thc:``) ids changed whenever a re-clustering picked a new
label, and person/topic spellings were merged by migrations, while listeners' follows, favorites,
boards, notes, Learn paths and "Less like this" kept the old ids. The board-summarizer sidecar now
runs this pass every cycle; this command runs it over the state written before, with the renames
that happened before ids were kept read from saved artifacts (``--prior-storylines``,
``--prior-themes``: the snapshots' ``topic_theme_clusters.json`` and ``topic_clusters.json``). It
can only repair renames it has an artifact for; older broken references are flagged.

A reference with a successor is re-keyed; one that is gone is flagged ("No longer available" in
the player) and never deleted. Every change is receipted in
``<appdata>/rekey_user_references.jsonl``; ``--undo`` puts back each value the apply wrote, where
it is still what the apply wrote.

    python -m podcast_scraper.server.rekey_references --appdata /app/appdata --corpus /app/output \\
        --prior-storylines snap/topic_theme_clusters.json --prior-themes snap/topic_clusters.json
    ... --apply
    python -m podcast_scraper.server.rekey_references --appdata /app/appdata --undo
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

RECEIPTS_FILE = "rekey_user_references.jsonl"


def _clusters(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw.get("data", raw) if isinstance(raw, dict) else {}
    return list(data.get("clusters") or [])


def _prior_history(corpus: Path, storylines: Path | None, themes: Path | None) -> dict[str, Any]:
    from podcast_scraper.search.stable_cluster_ids import map_prior_ids

    successors: dict[str, str] = {}
    gone: set[str] = set()
    for prior, current in (
        (storylines, corpus / "enrichments" / "topic_theme_clusters.json"),
        (themes, corpus / "search" / "topic_clusters.json"),
    ):
        if prior is None:
            continue
        history = map_prior_ids(_clusters(prior), _clusters(current))
        successors.update(history["successors"])
        gone.update(history["gone"])
    return {"successors": successors, "gone": sorted(gone)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--appdata", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, help="corpus root (not needed for --undo)")
    ap.add_argument("--prior-storylines", type=Path, help="an earlier topic_theme_clusters.json")
    ap.add_argument("--prior-themes", type=Path, help="an earlier search/topic_clusters.json")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="write (default: dry-run)")
    mode.add_argument("--undo", action="store_true", help="put back what the apply changed")
    args = ap.parse_args(argv)

    from podcast_scraper.server import reference_integrity

    receipts_path = args.appdata / RECEIPTS_FILE
    if args.undo:
        if not receipts_path.is_file():
            print(f"no receipts at {receipts_path}; nothing to undo")
            return 1
        receipts = [
            json.loads(line) for line in receipts_path.read_text("utf-8").splitlines() if line
        ]
        restored = reference_integrity.undo(args.appdata, receipts)
        print(f"undo: restored {restored} of {len(receipts)} changes")
        return 0 if restored == len(receipts) else 1

    if args.corpus is None:
        ap.error("--corpus is required unless --undo")
    extra = _prior_history(args.corpus, args.prior_storylines, args.prior_themes)
    results = reference_integrity.repair_all(
        args.appdata, args.corpus, apply=args.apply, extra_history=extra
    )
    receipts = [r for res in results for r in res.receipts]
    by_store = Counter(r["store"].split(":")[0] for r in receipts)
    flagged = Counter(f["kind"] for res in results for f in res.flagged)
    verb = "re-keyed" if args.apply else "would re-key"
    print(f"users: {len(results)}; prior renames known: {len(extra['successors'])}")
    print(f"{verb} {len(receipts)}: {dict(sorted(by_store.items()))}")
    print(f"flagged as gone: {sum(flagged.values())}: {dict(sorted(flagged.items()))}")
    if args.apply and receipts:
        with receipts_path.open("a", encoding="utf-8") as fh:
            for rec in receipts:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"receipts appended to {receipts_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
