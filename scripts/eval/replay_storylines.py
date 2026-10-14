#!/usr/bin/env python3
"""Replay the storyline clustering month by month over a corpus: today's rule against RFC-128's.

For every month of the corpus's history, the episodes published up to the end of that month are
clustered with the production enricher (``topic_theme_clusters``):

- **old rule**: each month on its own, with no previous artifact — what production did before the
  registry (every run started from scratch);
- **new rule**: each month reconciled against the previous month's artifact, with the enricher's
  clock set to that month's end — the storyline registry (``search/storyline_registry.py``).

The corpus is only read; each rule's artifact lives in its own temporary directory. Prints the
table RFC-128 reports (storylines, short ones, renames, follow survival, episodes covered) and,
with ``--probe``, a named storyline's id and names under each rule.

    python scripts/eval/replay_storylines.py --corpus <corpus copy> --probe "prompt engineering"
"""

from __future__ import annotations

import argparse
import calendar
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers import topic_theme_clusters as ttc
from podcast_scraper.enrichment.paths import discover_episode_bundles


def _month(bundle: Any) -> str:
    try:
        meta = json.loads(Path(bundle.metadata_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str((meta.get("episode") or {}).get("published_date") or "")[:7]


def _month_end(month: str) -> float:
    y, m = (int(x) for x in month.split("-"))
    day = calendar.monthrange(y, m)[1]
    return datetime(y, m, day, 23, 59, tzinfo=timezone.utc).timestamp()


def _run(bundles: list[Any], state_dir: Path, now: float, carry: bool) -> dict[str, Any]:
    art = state_dir / "enrichments" / "topic_theme_clusters.json"
    if not carry:  # from scratch: neither the artifact nor the registry's copy
        art.unlink(missing_ok=True)
        (art.parent / ttc.REGISTRY_COPY).unlink(missing_ok=True)
    ttc._now = lambda: now  # the run's clock (the enricher's own hook)
    ctx = type("Ctx", (), {"run_id": "replay", "enricher_id": "topic_theme_clusters"})()
    data: dict[str, Any] = ttc._compute(None, state_dir, bundles, {}, ctx)
    art.parent.mkdir(parents=True, exist_ok=True)
    art.write_text(json.dumps({"data": data}), encoding="utf-8")
    return data


def _snapshot(data: dict[str, Any]) -> dict[str, tuple[str, frozenset[str], frozenset[str]]]:
    """``{id: (name, episodes, topics)}``."""
    return {
        c["graph_compound_parent_id"]: (
            c["canonical_label"],
            frozenset(e for m in c["members"] for e in m.get("episode_ids") or []),
            frozenset(m["topic_id"] for m in c["members"]),
        )
        for c in data["clusters"]
    }


def replay(corpus: Path, min_episodes: int = 50) -> dict[str, Any]:
    """``{rule: [(month, {id: (name, episodes, topics)})]}`` for both rules, and under
    ``successors`` each rule's last ``id_history`` successors (a merged storyline's follows move
    to its successor)."""
    bundles = discover_episode_bundles(corpus)
    dated = [(_month(b), b) for b in bundles]
    months = sorted({m for m, _ in dated if m})
    out: dict[str, Any] = {"old": [], "new": [], "successors": {"old": {}, "new": {}}}
    with tempfile.TemporaryDirectory() as old_dir, tempfile.TemporaryDirectory() as new_dir:
        for month in months:
            sub = [b for m, b in dated if m and m <= month]
            if len(sub) < min_episodes:
                continue
            now = _month_end(month)
            out["old"].append((month, _snapshot(_run(sub, Path(old_dir), now, carry=False))))
            new = _run(sub, Path(new_dir), now, carry=True)
            out["new"].append((month, _snapshot(new)))
            out["successors"]["new"] = dict((new.get("id_history") or {}).get("successors") or {})
    return out


def summarise(
    snaps: list[tuple[str, dict]],
    follow_months: int = 6,
    successors: dict[str, str] | None = None,
) -> dict[str, Any]:
    """The RFC-128 measures for one rule's snapshots. A follow is kept when its storyline is still
    published, or its id has a published successor (a merge moves its followers there)."""
    final = snaps[-1][1]
    succ = successors or {}

    def kept(sid: str) -> bool:
        seen_ids = {sid}
        while sid not in final and sid in succ and succ[sid] not in seen_ids:
            sid = succ[sid]
            seen_ids.add(sid)
        return sid in final

    renamed: set[str] = set()
    for (_m, prev), (_n, cur) in zip(snaps, snaps[1:]):
        renamed |= {i for i in cur if i in prev and cur[i][0] != prev[i][0]}
    start = snaps[max(0, len(snaps) - 1 - follow_months)][1]
    seen = [t for _n, _e, topics in final.values() for t in topics]
    names = [n for n, _e, _t in final.values()]
    return {
        "month": snaps[-1][0],
        "storylines": len(final),
        "under_5_episodes": sum(1 for _n, e, _t in final.values() if len(e) < 5),
        "episodes_covered": len(set().union(*(e for _n, e, _t in final.values()))),
        # Today's clustering is a partition; the registry must stay one (RFC-128).
        "topics_in_two_storylines": len(seen) - len(set(seen)),
        "storylines_sharing_a_name": len(names) - len(set(names)),
        "widest": max((len(t) for _n, _e, t in final.values()), default=0),
        # Same id, new name. Under the old rule a new name is a new id, so its renames show as
        # follows lost instead.
        "names_changed_same_id": len(renamed),
        "follows_kept": f"{sum(1 for i in start if kept(i))}/{len(start)}",
    }


def main(argv: list[str] | None = None) -> int:
    """The command: replay, print both rules' measures, and each probe's history."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--probe", action="append", default=[], help="a storyline name to follow")
    args = ap.parse_args(argv)
    result = replay(args.corpus)
    for rule in ("old", "new"):
        print(rule, json.dumps(summarise(result[rule], successors=result["successors"][rule])))
    for probe in args.probe:
        for rule in ("old", "new"):
            seen = [
                (m, sid, name)
                for m, snap in result[rule]
                for sid, (name, _e, _t) in snap.items()
                if probe.lower() in name.lower()
            ]
            first = seen[0] if seen else None
            print(f"{rule} '{probe}': first {first}, last {seen[-1] if seen else None}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
