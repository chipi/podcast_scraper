#!/usr/bin/env python3
"""The corpus scoreboard (`docs/guides/CORPUS_CHANGE_METHOD.md` §3) over one corpus copy.

Read-only. Run it on a snapshot and on a clone with the change applied, and compare the two:

    python scripts/measure/corpus_scoreboard.py --corpus <root> --label <snapshot id> \\
        [--json out.json]

The JSON carries the label, the totals and the per-language counts. Definitions live in
`podcast_scraper.quality.corpus_scoreboard`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.quality.corpus_scoreboard import score_corpus  # noqa: E402

ROWS = (
    ("episodes", "episodes with GI"),
    ("episodes_no_gi", "  population gap: no GI"),
    ("metadata_unreadable", "  population gap: metadata unreadable"),
    ("insights", "insights"),
    ("S1_hidden_unknown", "S1  hidden by our naming defect"),
    ("S1n_named_but_hidden", "S1n named but hidden"),
    ("S1c_inconsistent", "S1c visibility contradicts type (must be 0)"),
    ("S2_hidden", "S2  all hidden"),
    ("S2_hidden_unknown", "      unknown"),
    ("S2_hidden_unidentified", "      unidentified"),
    ("S2_hidden_commercial", "      commercial"),
    ("S3_principal_unidentified", "S3  principal voices typed unidentified"),
    ("S3_outside_no_diagnostics", "  outside S3: no diagnostics"),
    ("S3_outside_no_voice", "  outside S3: voice not found"),
    ("S3_voice_not_in_diagnostics", "  outside S3: voice not in diagnostics"),
    ("S10a_zero_visible", "S10a episodes with zero visible insights"),
    ("S10b_zero_visible_panel", "S10b episodes whose panel is empty"),
    ("quotes", "quotes"),
    ("S12_quotes_off_text", "S12 quotes off their text"),
    ("quotes_ref_unresolved", "  outside S12: transcript file not found"),
    ("quotes_no_offsets", "  outside S12: no offsets"),
    ("A1_marker_hits", "A1  sponsor-marker hits in ad-free text"),
    ("A1_episodes_with_hits", "A1  episodes with a hit"),
    ("A1_episodes_no_adfree", "  outside A1: no ad-free text"),
)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument(
        "--label", required=True, help="snapshot id or state name, carried into the output"
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    board = score_corpus(args.corpus)
    board = {"label": args.label, "corpus": str(args.corpus), **board}
    total = board["total"]
    print(f"scoreboard [{args.label}] episodes={total.get('episodes', 0)}")
    for key, name in ROWS:
        print(f"  {name:48s} {total.get(key, 0):>8}")
    if args.json:
        args.json.write_text(json.dumps(board, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
