#!/usr/bin/env python3
"""How many hidden insights sit in the episodes one migration wrote, now and before it ran.

    python scripts/measure/hidden_by_migration.py --corpus <clone> \\
        --receipts naming_arc_voices_repaired.jsonl --backup-tag 0029

Reads the migration's receipts (the files it wrote), keeps the episodes still served, and scores
them with the corpus scoreboard twice: as they are, and with the migration's backups
(``.podcast_scraper/upgrade-backups/<tag>/``) laid over a temporary copy, which is the state the
migration found. On the 2026-10-16 snapshot this showed 0029 left 1,779 of the corpus's 1,786
named-but-hidden insights (S1n): 0 before it, 1,779 after (#2380). Read-only on the corpus.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.quality.corpus_scoreboard import score_episode  # noqa: E402
from podcast_scraper.upgrade.corpus_selection import select_served_artifacts  # noqa: E402

KEYS = ("insights", "S1_hidden_unknown", "S1n_named_but_hidden", "S2_hidden")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--receipts", required=True, help="receipts file name at the corpus root")
    ap.add_argument("--backup-tag", required=True)
    args = ap.parse_args()
    root: Path = args.corpus
    metas = set()
    for line in open(root / args.receipts, encoding="utf-8"):
        rel = json.loads(line).get("relpath", "")
        if rel.endswith(".metadata.json"):
            metas.add(rel)
    served = {str(m.relative_to(root)) for m in select_served_artifacts(root, ".metadata.json")[0]}
    still = sorted(metas & served)
    now: Counter = Counter()
    for rel in still:
        now.update(
            {
                k: v
                for k, v in (score_episode(root / rel) or {}).get("counts", {}).items()
                if k in KEYS
            }
        )
    before: Counter = Counter()
    backups = root / ".podcast_scraper" / "upgrade-backups" / args.backup_tag
    with tempfile.TemporaryDirectory() as tmp:
        for rel in still:
            run = (root / rel).parent.parent
            dst = Path(tmp) / run.relative_to(root)
            shutil.copytree(run, dst, dirs_exist_ok=True)
            if (backups / run.relative_to(root)).is_dir():
                shutil.copytree(backups / run.relative_to(root), dst, dirs_exist_ok=True)
        for rel in still:
            counts = (score_episode(Path(tmp) / rel) or {}).get("counts", {})
            before.update({k: v for k, v in counts.items() if k in KEYS})
    print(f"metadata files the migration wrote: {len(metas)}; still served: {len(still)}")
    print(f"{'':24}{'before it':>12}{'now':>12}")
    for k in KEYS:
        print(f"{k:24}{before[k]:>12}{now[k]:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
