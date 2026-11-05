#!/usr/bin/env python3
"""Read-only: the outcome hook's own plan diff for one migration against a corpus, before an apply.

    python scripts/measure/plan_diff_ro.py --migration 0043_non_participant_names_removed \\
        --corpus /app/output [--out /tmp/0043.plan-diff.json]

Prints one JSON line: the gate's and the corpus's episode counts, the kinds of difference, and
whether the installed code's digest equals the outcome's. ``--out`` (a path outside the corpus)
gets the entries in the hook's own format. Writes nothing in the corpus.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.upgrade import outcome_hook as oh  # noqa: E402
from podcast_scraper.upgrade.registry import get_migrations  # noqa: E402


def plan_report(mid: str, corpus: Path) -> Dict[str, Any]:
    """The plan diff's summary and entries for ``mid`` on ``corpus``."""
    mig = next((m for m in get_migrations() if m.id == mid), None)
    if mig is None:
        raise SystemExit(f"not registered: {mid}")
    outcome, sha = oh.load_outcome(mid)
    prod = oh.episode_plan(mig, corpus)
    diff: List[Dict[str, Any]] = oh.plan_diff(outcome["plan"], prod)
    return {
        "migration": mid,
        "outcome_sha256": sha,
        "clone_episodes": len(outcome["plan"]),
        "prod_episodes": len(prod),
        "diff_kinds": dict(Counter(e["kind"] for e in diff)),
        "digest_matches": oh.migration_digest(mig) == outcome["code_digest"],
        "entries": diff,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print the summary line; write the entries to ``--out`` when given."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--migration", required=True)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    rep = plan_report(args.migration, args.corpus)
    entries = rep.pop("entries")
    print(json.dumps(rep))
    if args.out is not None:
        out = {
            "migration": rep["migration"],
            "outcome_sha256": rep["outcome_sha256"],
            "entries": entries,
        }
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
