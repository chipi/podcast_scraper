#!/usr/bin/env python3
"""The loss ranking over a snapshot clone and the naming gold (INGESTION-ARC 2.2, 2.3). Read-only.

    python scripts/measure/loss_ranking.py --corpus <snapshot clone> \\
        --gold .test_outputs/naming_gold --snapshot <snapshot id> --out <dir outside the repo>

Writes ``gold_wrong.jsonl``, ``corpus_hidden.jsonl``, ``seat_check.jsonl`` and ``summary.json`` to
``--out``, which must lie outside the git checkout: the rows carry gold-derived names, which are
never committed. Prints the summary. Definitions: ``podcast_scraper.quality.loss_ranking``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.quality import gold_insights, loss_ranking  # noqa: E402


def _inside_repo(path: Path) -> bool:
    top = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    return bool(top) and Path(path).resolve().is_relative_to(Path(top).resolve())


def main(argv: list[str] | None = None) -> int:
    """Run the ranking; non-zero on a refused ``--out`` or a failed conservation check."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    if _inside_repo(args.out):
        print(f"--out must lie outside the git checkout: {args.out}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    report = gold_insights.run(args.corpus, args.gold, list(loss_ranking.GOLD_SETS), args.snapshot)
    gold = loss_ranking.gold_wrong_names(report, args.gold, args.corpus)
    hidden = loss_ranking.corpus_hidden(args.corpus)
    served = {
        str(m.relative_to(args.corpus))
        for m in __import__(
            "podcast_scraper.upgrade.corpus_selection", fromlist=["x"]
        ).select_served_artifacts(args.corpus, ".metadata.json")[0]
    }
    ldirs = {s: str(report["sets"][s]["labels"]) for s in report["sets"]}
    seat = loss_ranking.seat_check(args.corpus, args.gold, ldirs, served)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    summary = {
        "snapshot": args.snapshot,
        "corpus_root": str(args.corpus),
        "code_sha": sha,
        "gold_wrong": gold["summary"],
        "gold_outside_s7_wrong": gold["outside_s7_wrong"],
        "corpus_hidden": hidden["summary"],
        "seat_check": seat["summary"],
    }
    for r in gold["rows"]:
        r["definitions_sha"] = sha
    loss_ranking.write(args.out, "gold_wrong", gold["rows"])
    loss_ranking.write(args.out, "corpus_hidden", hidden["rows"])
    loss_ranking.write(args.out, "seat_check", seat["rows"])
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
