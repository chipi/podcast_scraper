#!/usr/bin/env python3
"""Does the interests picker offer a real choice, or a decorative one? (#1669)

WHAT THIS MEASURES
``top_clusters_by_member_count`` selects the picker's options by PREVALENCE — the clusters with
the most members win. Prevalence is the opposite of discriminating power: a cluster that every
episode carries adds the same affinity constant to every row, so following it cannot re-rank
anything. Taken to its limit, every option produces the identical feed and the choice is theatre.

This script answers three questions against a real corpus:

  1. How much of the corpus does each OFFERED option actually cover?
  2. How many DISTINCT feeds do the offered options produce between them? (1 = decorative)
  3. How many tokens sit in a discriminating band, and how many distinct feeds do THEY produce?

WHY IT TAKES A CORPUS ROOT
Measured on the 36-episode v3 fixture the answer is "2 options, both at 100%, one feed" — but that
fixture only has two clusters at all, so it cannot distinguish "the picker ranks badly" from "this
corpus is too small to cluster". Only a production-scale corpus separates those. Point it at one::

    python scripts/eval/score/picker_discriminating_power_v1.py --corpus-root /path/to/output

Read-only: it loads artifacts and ranks in-process, and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from podcast_scraper.search.theme_clusters import consumer_theme_cluster_map  # noqa: E402
from podcast_scraper.search.topic_clusters import (  # noqa: E402
    consumer_topic_cluster_map,
    top_clusters_by_member_count,
)
from podcast_scraper.server.app_discover_view import _episode_features, rank_discover  # noqa: E402
from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative  # noqa: E402

#: A token is a candidate option when it covers at least this many episodes (below it, following
#: it is indistinguishable from following one episode) and at most this share of them (above it,
#: it stops separating the corpus).
MIN_EPISODES = 2
MAX_SHARE = 0.6


def _coverage(root: Path, rows) -> Dict[str, int]:
    """token -> how many episodes carry it."""
    cluster_map = consumer_topic_cluster_map(root)
    theme_map = consumer_theme_cluster_map(root)
    counts: Dict[str, int] = {}
    for row in rows:
        clusters, topics, persons = _episode_features(root, row, cluster_map, theme_map)
        for token in (*clusters, *topics, *persons):
            counts[token] = counts.get(token, 0) + 1
    return counts


def _feed(root: Path, rows, tokens: Sequence[str], limit: int) -> tuple:
    return tuple(s.slug for s in rank_discover(root, list(tokens), rows, limit=limit))


def _table(title: str, entries: List[tuple], total: int) -> None:
    print(f"\n{title}")
    print(f"  {'token':52} {'episodes':>9} {'coverage':>9}")
    for token, n in entries:
        flag = "  <-- every episode" if total and n >= total else ""
        print(f"  {token[:52]:52} {n:>9} {n / total * 100:>8.1f}%{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-root", type=Path, required=True, help="corpus parent directory")
    ap.add_argument("--top-n", type=int, default=12, help="options the picker offers (default 12)")
    ap.add_argument("--limit", type=int, default=10, help="feed length to compare (default 10)")
    args = ap.parse_args()

    root: Path = args.corpus_root.expanduser().resolve()
    if not root.is_dir():
        print(f"corpus root is not a directory: {root}", file=sys.stderr)
        return 2

    rows = build_catalog_rows_cumulative(root)
    rows.sort(key=lambda r: (r.publish_date or ""), reverse=True)
    total = len(rows)
    if not total:
        print(f"corpus has no episodes: {root}", file=sys.stderr)
        return 2

    counts = _coverage(root, rows)
    offered = top_clusters_by_member_count(root, args.top_n)

    print(f"corpus: {root}")
    print(f"episodes: {total}   distinct tokens: {len(counts)}   picker offers: {len(offered)}")

    _table(
        "OFFERED — what the picker actually shows (member_count desc)",
        [(c["id"], counts.get(c["id"], 0)) for c in offered],
        total,
    )
    offered_feeds = {c["id"]: _feed(root, rows, [c["id"]], args.limit) for c in offered}
    distinct_offered = len(set(offered_feeds.values()))
    print(f"\n  -> {distinct_offered} distinct feed(s) from {len(offered)} option(s)")
    if len(offered) > 1 and distinct_offered == 1:
        print("  -> DECORATIVE: every option yields the same feed.")

    band = sorted(
        ((t, n) for t, n in counts.items() if n >= MIN_EPISODES and n <= total * MAX_SHARE),
        key=lambda kv: -kv[1],
    )
    print(
        f"\ntokens in the discriminating band "
        f"({MIN_EPISODES} <= n <= {MAX_SHARE:.0%} of corpus): {len(band)}"
    )
    _table(
        "ALTERNATIVE — the same count of options, drawn from that band", band[: args.top_n], total
    )
    alt_feeds = {t: _feed(root, rows, [t], args.limit) for t, _ in band[: args.top_n]}
    print(f"\n  -> {len(set(alt_feeds.values()))} distinct feed(s) from {len(alt_feeds)} option(s)")

    # A verdict, so the output cannot be read as "numbers were printed, therefore fine".
    universal = [c["id"] for c in offered if counts.get(c["id"], 0) >= total]
    print("\nVERDICT")
    print(f"  offered options covering EVERY episode: {len(universal)}/{len(offered)}")
    print(f"  distinct feeds — offered {distinct_offered}  vs  band {len(set(alt_feeds.values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
