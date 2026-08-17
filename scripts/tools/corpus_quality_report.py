#!/usr/bin/env python3
"""Answer "I ran N episodes — how did it go?" from a corpus on disk (#1647).

    python scripts/tools/corpus_quality_report.py --corpus output/
    python scripts/tools/corpus_quality_report.py --corpus output/ --run run_0b2f616f_20260813
    python scripts/tools/corpus_quality_report.py --corpus output/ --json report.json

Reads each episode's metadata sidecar for its stage ledger and its ``*.gi.json`` for
attribution counts, then prints the aggregate — including a NOT MEASURED section, because a
report that lists only what it checked lets silence read as health. That is the exact failure
this exists to prevent: for two months every available signal reported a healthy corpus while
72 % of episodes had speaker detection skipped (#1646).

Exit codes: 0 = report produced. 1 = nothing to report (no episodes found). 2 = bad usage.
The report itself never exits non-zero for finding damage — it is an instrument, not a gate;
wire thresholds in the caller so the instrument stays honest.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make ``src`` importable when run directly from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from podcast_scraper.quality.attribution import (  # noqa: E402
    build_report,
    EpisodeQuality,
    format_report,
)

METADATA_SUFFIX = ".metadata.json"
GI_SUFFIX = ".gi.json"


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _episode_from_metadata(metadata_path: Path) -> EpisodeQuality:
    """Build one quality record; every unreadable input becomes a NOTE, never a silent zero."""
    record = EpisodeQuality()
    meta = _read_json(metadata_path)
    if not isinstance(meta, dict):
        record.notes.append("metadata_unreadable")
        return record

    episode = meta.get("episode") or {}
    feed = meta.get("feed") or {}
    processing = meta.get("processing") or {}
    record.episode_id = episode.get("episode_id")
    record.feed = feed.get("title") or feed.get("feed_title")
    record.duration_seconds = episode.get("duration_seconds")

    ledger = processing.get("stage_ledger")
    if isinstance(ledger, dict):
        record.stage_ledger = {k: v for k, v in ledger.items() if isinstance(v, dict)}
    else:
        # Pre-#1647 episode: no ledger at all. Recorded as a note so the report can say how
        # much of the corpus it cannot speak for, rather than quietly averaging it in.
        record.notes.append("no_stage_ledger (episode predates #1647)")

    gi_path = metadata_path.with_name(metadata_path.name[: -len(METADATA_SUFFIX)] + GI_SUFFIX)
    gi = _read_json(gi_path)
    if not isinstance(gi, dict):
        record.notes.append("gi_unreadable")
        return record

    nodes = gi.get("nodes") or []
    insights = [n for n in nodes if isinstance(n, dict) and n.get("type") == "Insight"]
    record.insights_total = len(insights)
    # Mirrors is_surfaceable_insight(): absent means surfaceable, only explicit False excludes.
    record.insights_surfaceable = sum(
        1 for n in insights if (n.get("properties") or {}).get("surfaceable") is not False
    )
    speakers = {(n.get("properties") or {}).get("speaker") for n in insights if isinstance(n, dict)}
    speakers.discard(None)
    record.voices_total = len(speakers)
    # A raw diarization label (SPEAKER_00) is an unnamed voice; anything else was resolved.
    record.voices_named = sum(1 for s in speakers if not str(s).startswith("SPEAKER_"))
    return record


def collect(corpus_root: Path, run: Optional[str]) -> List[EpisodeQuality]:
    """Find every episode metadata sidecar under *corpus_root*, optionally scoped to one run."""
    pattern = f"**/{run}/**/*{METADATA_SUFFIX}" if run else f"**/*{METADATA_SUFFIX}"
    return [_episode_from_metadata(p) for p in sorted(corpus_root.glob(pattern))]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", required=True, type=Path, help="Corpus root (e.g. output/).")
    parser.add_argument("--run", default=None, help="Limit to one run_* directory.")
    parser.add_argument("--json", dest="json_out", type=Path, help="Also write the report as JSON.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.corpus.is_dir():
        print(f"not a directory: {args.corpus}", file=sys.stderr)
        return 2

    episodes = collect(args.corpus, args.run)
    if not episodes:
        print(f"no episodes found under {args.corpus}", file=sys.stderr)
        return 1

    report: Dict[str, Any] = build_report(episodes)
    print(format_report(report))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
