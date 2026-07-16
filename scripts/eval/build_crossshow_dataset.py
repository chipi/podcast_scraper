#!/usr/bin/env python3
"""Build a CROSS-SHOW eval dataset — a few episodes from every show, not ten from one.

The previous 10-episode set was ten Hard Fork episodes. That is one format: two co-hosts, a
pre-roll ad, a guest in the second half. Every rule we tuned against it broke on the next show —
because the variance that matters is FORMAT variance, not episode variance:

    Invest Like the Best   1 host, 1 guest; the GUEST talks 82%
    Latent Space           1 host, holding 8.6% of the episode
    Planet Money           a narrated desk, 13 voices, rotating hosts
    The Journal            2 reporters, no guest
    NVIDIA                 1 host + 1 guest; the shipped labels were SWAPPED
    Hard Fork              2 co-hosts, ads, a guest

Ten episodes of one show teach us one format ten times. Two episodes of nine shows teach us nine.

Reads the relabelled transcripts (`scripts/backfill/relabel_corpus.py` — zero GPU) so the dataset
carries transcripts that actually say who spoke. Emits the standard `data/eval/datasets` manifest:
path + sha256 + title only, never transcript content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("relabelled_root", type=Path, help="output of relabel_corpus.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--per-show", type=int, default=2)
    ap.add_argument("--profile", default="cleaning_v5")
    args = ap.parse_args(argv)

    episodes = []
    for show_dir in sorted(args.relabelled_root.glob("*")):
        if not show_dir.is_dir():
            continue
        txts = sorted(show_dir.glob("*.txt"))[: args.per_show]
        for i, txt in enumerate(txts, 1):
            body = txt.read_text(encoding="utf-8")
            episodes.append(
                {
                    "episode_id": f"{show_dir.name.replace('rss_', '')[:22]}_e{i:02d}",
                    "show": show_dir.name,
                    "title": txt.stem.split("_2026")[0].strip(),
                    "transcript_path": str(txt),
                    "transcript_hash": hashlib.sha256(body.encode()).hexdigest(),
                    "preprocessing_profile": args.profile,
                    "chars": len(body),
                }
            )

    payload = {
        "dataset_id": args.dataset_id,
        "task": "grounded_insights",
        "note": (
            "Cross-show: a few episodes from every show. The variance that breaks the pipeline is "
            "FORMAT variance, not episode variance — ten episodes of one show teach one format ten "
            "times. Transcripts are relabelled (metadata + conversation), zero GPU."
        ),
        "episodes": episodes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    by_show: dict = {}
    for e in episodes:
        by_show.setdefault(e["show"], []).append(e)
    print(f"{len(episodes)} episodes across {len(by_show)} shows\n")
    for show, eps in sorted(by_show.items()):
        print(
            f"  {show.replace('rss_', '')[:34]:36} {len(eps)} ep, "
            f"{sum(e['chars'] for e in eps)//1000}k chars"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
