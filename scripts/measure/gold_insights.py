#!/usr/bin/env python3
"""S7 / S8 on the naming gold sets, insight-weighted (``podcast_scraper.quality.gold_insights``).

    python scripts/measure/gold_insights.py --corpus <snapshot clone> \\
        --gold .test_outputs/naming_gold \\
        --snapshot snapshot-prod-20261016+0030 --out report.json [--set dev_v1 --set val_v1 ...] \\
        [--labels val_v1=labels_v1] [--voice-map <rows>.voicemap.json]

``--snapshot`` names the corpus state exactly, including any migration applied outside the
runner (it is not in the corpus's ledger; the report's ``ledger_applied`` shows what the ledger
records). ``--labels SET=DIR`` pins a set's label directory instead of the newest ``labels_vN``.
``--voice-map`` measures an eval run's output (``--corpus`` = the run's work dir): the eval
scorer's voice-map file follows each case to its after run and re-keys the labels to its voices.

Read-only. Prints the per-set table; ``--out`` writes the full report (per-insight rows included),
deterministic: no clock, sorted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from podcast_scraper.quality.gold_insights import BUCKETS, load_voice_maps, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--snapshot", required=True, help="the snapshot id this corpus is")
    ap.add_argument("--set", action="append", dest="sets")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--labels", action="append", default=[], metavar="SET=DIR")
    ap.add_argument("--voice-map", type=Path, help="the eval scorer's <rows>.voicemap.json")
    args = ap.parse_args()
    sets = args.sets or sorted(p.name for p in args.gold.iterdir() if (p / "cases").is_dir())
    if any("=" not in x for x in args.labels):
        ap.error("--labels must be SET=DIR")
    labels = dict(x.split("=", 1) for x in args.labels)
    if set(labels) - set(sets):
        ap.error(f"--labels names a set not measured: {sorted(set(labels) - set(sets))}")
    maps = load_voice_maps(args.voice_map) if args.voice_map else None
    report = run(args.corpus, args.gold, sets, args.snapshot, labels or None, maps)
    print(f"snapshot {report['snapshot']}  ({report['matcher_limits']})")
    for gset, s in report["sets"].items():
        t = s["total"]
        print(
            f"\n{gset} (labels {s['labels']}): included {s['included']}, "
            f"skipped {len(s['skipped_not_served'])}, no_label {s['no_label']}, "
            f"gi_missing {s['gi_missing']}, "
            + (f"no voice map {len(s['skipped_no_voice_map'])}, " if maps else "")
            + f"insights {t['insights']}"
        )
        for b in BUCKETS:
            print(f"  {b:22} {t[b]:>6}   shown {t[b + '_shown']:>6}   hidden {t[b + '_hidden']:>6}")
        print(f"  S7_correct by tier {t['S7_correct_by_tier']}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
