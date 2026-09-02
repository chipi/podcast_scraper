#!/usr/bin/env python
"""Sweep ``topic_theme_clusters`` knobs and score themes by CROSS-FEED reach.

Why this exists
---------------
Theme clusters power the operator plane's top-down zoom-out (``graphTopDown.ts``: *"when the user
taps a SuperTheme node… inject the super-theme's child TopicClusters + their tagged Topics"*) and
the player's Storylines. The enricher shipped with ``merge_threshold=2.0`` and
``min_pair_episode_count=2`` and had never been swept against a real corpus — it was switched off
for nine days across the entire Batch A expansion (765 -> 1,066 episodes), so the surface rendered
pre-expansion themes until 2026-09-02.

The trap this harness exists to make visible
--------------------------------------------
``min_pair_episode_count`` looks like a recall knob — lower it, get more themes. **It is not.**
``_average_linkage`` carries a guard::

    if n > _MAX_LINKAGE_TOPICS:      # 400
        return [{i} for i in range(n)]

where ``n`` is the count of topics in the *co-occurring subgraph*, i.e. topics touching at least
one pair seen in ``>= min_pair`` episodes. Measured on the 1,066-episode corpus:

===========  =================  =====================  ==================================
 min_pair     qualifying pairs   topics involved (n)    outcome
===========  =================  =====================  ==================================
 1            45,009             9,344                  **over the cap — ZERO themes**
 2 (shipped)  258                192                    54 themes
 3            1                  2                      nothing to cluster
===========  =================  =====================  ==================================

So lowering ``min_pair`` to 1 does not loosen the filter, it **silently empties the surface** — the
enricher still reports ``ok``, having returned every topic as its own singleton. And raising it to
3 empties it the honest way. ``min_pair=2`` is the only viable value at this corpus size, which
means ``merge_threshold`` is the only real knob and this sweep is how it gets picked.

Scoring
-------
Theme count is the wrong metric for the same reason cluster count is wrong for semantic clusters:
a theme confined to one episode is a co-occurrence artifact, not a storyline. The operator surface
is a *browse* surface, so what matters is themes that reach across episodes and feeds, at a
legend-sized count. Reported per threshold:

* ``themes``        — clusters with >= 2 members
* ``topics_used``   — topics landing in a real theme (coverage of the subgraph)
* ``x_episode``     — themes whose topics span >= 2 distinct episodes
* ``x_feed``        — themes whose topics span >= 2 distinct feeds  <- the storyline claim
* ``intra_ep``      — themes confined to ONE episode (the failure mode)
* ``max``           — largest theme (a giant blob is unbrowsable)

Uses the enricher's own ``_average_linkage`` and lift arithmetic, so results are comparable to what
production emits rather than to a re-implementation.

Usage
-----
    python scripts/eval/score/theme_cluster_threshold_sweep.py \\
        --cooccurrence enrichments/topic_cooccurrence_corpus.json \\
        --topics topics.jsonl \\
        --thresholds 1.5,2.0,3.0,5.0,10.0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def feed_of(ep_relpath: str) -> str:
    parts = ep_relpath.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "feeds" else ep_relpath


def load_topic_episodes(topics_jsonl: Path) -> Dict[str, Set[str]]:
    """topic_id -> set of episode relpaths, from the per-episode KG pull."""
    out: Dict[str, Set[str]] = defaultdict(set)
    with topics_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = str(row.get("ep") or "")
            for t in row.get("topics") or []:
                tid = t.get("id")
                if isinstance(tid, str) and tid and ep:
                    out[tid].add(ep)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cooccurrence", type=Path, required=True)
    ap.add_argument("--topics", type=Path, required=True)
    ap.add_argument("--thresholds", default="1.5,2.0,3.0,5.0,10.0")
    ap.add_argument("--min-pair", type=int, default=2)
    args = ap.parse_args()

    from podcast_scraper.enrichment.enrichers.topic_theme_clusters import (
        _MAX_LINKAGE_TOPICS,
        _average_linkage,
    )

    raw = json.loads(args.cooccurrence.read_text(encoding="utf-8"))
    data = raw.get("data", raw)
    pairs = data.get("pairs") or []
    eps_by_topic = load_topic_episodes(args.topics)

    # Rebuild the lift edge set exactly as the enricher does: pairs seen in >= min_pair episodes.
    edges: Dict[Tuple[str, str], float] = {}
    involved: Set[str] = set()
    for p in pairs:
        cnt = p.get("episode_count") or 0
        if cnt < args.min_pair:
            continue
        lift = p.get("lift")
        if not isinstance(lift, (int, float)) or lift <= 0:
            continue
        a, b = p.get("topic_a_id"), p.get("topic_b_id")
        if not (isinstance(a, str) and isinstance(b, str)):
            continue
        edges[(a, b)] = float(lift)
        involved.add(a)
        involved.add(b)

    idx = sorted(involved)
    pos = {t: i for i, t in enumerate(idx)}
    w: Dict[Tuple[int, int], float] = {}
    for (a, b), lift in edges.items():
        i, j = pos[a], pos[b]
        w[(min(i, j), max(i, j))] = lift

    def weight(i: int, j: int) -> float:
        return 0.0 if i == j else w.get((min(i, j), max(i, j)), 0.0)

    n = len(idx)
    print(f"min_pair={args.min_pair}: {len(edges)} qualifying pairs, {n} topics in the subgraph")
    if n > _MAX_LINKAGE_TOPICS:
        print(
            f"  *** n > _MAX_LINKAGE_TOPICS ({_MAX_LINKAGE_TOPICS}): the enricher SILENTLY"
            f" returns\n"
            f"      all-singletons here, so every threshold below would yield ZERO themes. The\n"
            f"      enricher still reports status=ok. Raise min_pair or the cap.",
            file=sys.stderr,
        )
        return 1
    print(f"  ({_MAX_LINKAGE_TOPICS - n} headroom before the silent all-singleton cliff)\n")

    print(
        f"{'thresh':>7} {'themes':>7} {'topics_used':>12} {'x_episode':>10} "
        f"{'x_feed':>7} {'intra_ep':>9} {'median':>7} {'max':>5}"
    )
    for raw_t in args.thresholds.split(","):
        t = float(raw_t)
        clusters = _average_linkage(n, weight, t)
        real = [c for c in clusters if len(c) >= 2]
        used = sum(len(c) for c in real)
        xep = xfeed = intra = 0
        sizes: List[int] = []
        for c in real:
            sizes.append(len(c))
            e: Set[str] = set()
            for i in c:
                e |= eps_by_topic.get(idx[i], set())
            f = {feed_of(x) for x in e}
            if len(e) >= 2:
                xep += 1
            if len(f) >= 2:
                xfeed += 1
            if len(e) <= 1:
                intra += 1
        med = statistics.median(sizes) if sizes else 0
        print(
            f"{t:7.1f} {len(real):7d} {used:12d} {xep:10d} {xfeed:7d} "
            f"{intra:9d} {med:7.1f} {max(sizes, default=0):5d}"
        )

    print(
        "\nPick for a BROWSE surface: enough x_feed themes to be worth zooming out to, at a legend-"
        "sized count, without one giant blob. Theme count alone is not the goal.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
