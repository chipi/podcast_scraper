#!/usr/bin/env python3
"""Count episodes in a corpus tree two ways, to catch backup under-capture (#877).

The prod corpus stores episodes under ``feeds/<feed>/run_<ts>_<hash>/metadata/`` and,
with ``append=false`` (the prod default), each scheduled run writes a NEW run dir — so a
feed's episodes scatter across multiple run dirs over time. The app's default discovery
keeps only the latest run per feed (load-bearing for index rebuild), which under-reports
the true episode total. #877 ("snapshot has 100 vs prod's 110") was this counting
artifact, not lost data — the tarball holds every run dir.

This prints both counts so an operator (or the backup-verify workflow) can confirm a
restored snapshot holds every episode:

  - distinct  : unique ``(feed_id, episode_id)`` across ALL run dirs (the true total)
  - latest_run: what the app's default latest-run-only discovery reports

With ``--expect N`` the script exits non-zero unless ``distinct == N`` — wire that into
the backup-verify workflow to fail loudly if a future snapshot really does drop episodes.

Deliberately STDLIB-ONLY (no ``podcast_scraper`` import): a backup checker must run against
a bare tarball without the app's deps installed, and an independent re-implementation of
the run-selection logic cross-checks the app's own discovery rather than trusting it.

Usage:
  count_distinct_episodes.py <corpus_dir>
  count_distinct_episodes.py <corpus_dir> --expect 110
  count_distinct_episodes.py <corpus_dir> --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_METADATA_SUFFIXES = (".metadata.json", ".metadata.yaml", ".metadata.yml")


def _iter_metadata_files(corpus_root: Path):
    """Yield every ``*.metadata.*`` path under the corpus (all runs, all feeds + flat)."""
    for dirpath, _dirnames, filenames in os.walk(corpus_root):
        if os.path.basename(dirpath) != "metadata":
            continue
        for name in filenames:
            if name.endswith(_METADATA_SUFFIXES):
                yield Path(dirpath) / name


def _feed_episode_key(path: Path) -> tuple[str, str]:
    """Best-effort ``(feed_id, episode_id)`` from a metadata doc; falls back to the stem."""
    feed_id = ""
    episode_id = ""
    if path.name.endswith(".metadata.json"):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(doc, dict):
                feed = doc.get("feed")
                if isinstance(feed, dict) and feed.get("feed_id"):
                    feed_id = str(feed["feed_id"]).strip()
                ep = doc.get("episode")
                if isinstance(ep, dict) and ep.get("episode_id"):
                    episode_id = str(ep["episode_id"]).strip()
        except (OSError, ValueError):
            pass
    if not episode_id:
        # Stem without the .metadata.* suffix — stable within a feed for dedup.
        episode_id = path.name
        for suf in _METADATA_SUFFIXES:
            if episode_id.endswith(suf):
                episode_id = episode_id[: -len(suf)]
                break
    return feed_id, episode_id


def _run_segment(path: Path, corpus_root: Path) -> tuple[str | None, str | None]:
    """``(feedDir, run_seg)`` for a ``feeds/.../run_*/metadata/`` path, else ``(None, None)``."""
    try:
        rel = path.relative_to(corpus_root).as_posix()
    except ValueError:
        return None, None
    parts = [p for p in rel.split("/") if p]
    if (
        len(parts) >= 5
        and parts[0] == "feeds"
        and parts[2].startswith("run_")
        and parts[3] == "metadata"
    ):
        return parts[1], parts[2]
    return None, None


def count_corpus_episodes(corpus_dir: str) -> dict[str, int]:
    """Return ``{"distinct": N_all_runs, "latest_run": N_latest_only}`` for *corpus_dir*."""
    root = Path(corpus_dir)
    files = list(_iter_metadata_files(root))

    # Latest run_* per feed dir (lexicographic max), matching the app's discovery.
    latest_seg: dict[str, str] = {}
    for f in files:
        feed_dir, seg = _run_segment(f, root)
        if feed_dir is None or seg is None:
            continue
        if feed_dir not in latest_seg or seg > latest_seg[feed_dir]:
            latest_seg[feed_dir] = seg

    distinct: set[tuple[str, str]] = set()
    latest: set[tuple[str, str]] = set()
    for f in files:
        key = _feed_episode_key(f)
        distinct.add(key)
        feed_dir, seg = _run_segment(f, root)
        if feed_dir is None:
            latest.add(key)  # flat metadata/ — always counted
        elif seg == latest_seg.get(feed_dir):
            latest.add(key)
    return {"distinct": len(distinct), "latest_run": len(latest)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_dir", help="Path to the corpus root (the dir containing feeds/).")
    parser.add_argument(
        "--expect",
        type=int,
        default=None,
        help="Fail (exit 2) unless the distinct episode count equals this value.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.corpus_dir):
        print(f"ERROR: not a directory: {args.corpus_dir}", file=sys.stderr)
        return 1

    counts = count_corpus_episodes(args.corpus_dir)
    dropped = counts["distinct"] - counts["latest_run"]

    if args.json:
        print(json.dumps({**counts, "latest_run_dropped": dropped}))
    else:
        print(f"distinct episodes (all runs):     {counts['distinct']}")
        print(f"latest-run-only (app default):    {counts['latest_run']}")
        if dropped > 0:
            print(
                f"NOTE: {dropped} episode(s) live only in older run dirs — present in the "
                "snapshot, but hidden from latest-run-only views (#877)."
            )

    if args.expect is not None and counts["distinct"] != args.expect:
        print(
            f"FAIL: expected {args.expect} distinct episodes, found {counts['distinct']} "
            "— snapshot under-captured.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
