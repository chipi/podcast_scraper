"""Insight-density report — where insights cluster across a corpus (#1140).

Reads every ``insight_density`` envelope and shows the early/mid/late
distribution corpus-wide and per show, so you can see (and tune) the signal that
drives the player's skip-guide: is the substance front-loaded, even, or
back-loaded? Only episodes with timed insights count toward the distribution.

Direct-Python entry point (no Make wrapper), matching the other eval scripts.

Usage:
    python scripts/eval/insight_density_report.py --corpus path/to/corpus [--json]

Exit codes:
    0 — report rendered (even if the corpus has no density envelopes)
    2 — invocation error (corpus dir missing)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

SEGMENTS = ("early", "mid", "late")


def find_density_envelopes(corpus_root: Path) -> list[Path]:
    """Every per-episode ``*.insight_density.json`` under the corpus."""
    return sorted(corpus_root.glob("**/enrichments/*.insight_density.json"))


def _show_title(density_path: Path) -> str:
    """The show/feed title for a density envelope, from its sibling metadata."""
    stem = density_path.name[: -len(".insight_density.json")]
    meta = density_path.parent.parent / f"{stem}.metadata.json"
    if not meta.is_file():
        return "unknown"
    try:
        doc = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unknown"
    feed = doc.get("feed") if isinstance(doc, dict) else None
    title = feed.get("title") if isinstance(feed, dict) else None
    return title.strip() if isinstance(title, str) and title.strip() else "unknown"


def counts_of(envelope: Any) -> dict[str, int]:
    """The early/mid/late counts from a density envelope (``data.counts``)."""
    data = envelope.get("data", envelope) if isinstance(envelope, dict) else {}
    counts = data.get("counts") if isinstance(data, dict) else {}
    if not isinstance(counts, dict):
        return {s: 0 for s in SEGMENTS}
    return {s: int(counts.get(s, 0) or 0) for s in SEGMENTS}


def aggregate(items: list[tuple[str, dict[str, int]]]) -> dict[str, Any]:
    """Pure: ``(show, {early,mid,late})`` rows → corpus + per-show totals.

    Episodes whose thirds are all zero (no timed insights) are ignored so they
    don't dilute the distribution.
    """
    corpus = {s: 0 for s in SEGMENTS}
    shows: dict[str, dict[str, int]] = defaultdict(
        lambda: {**{s: 0 for s in SEGMENTS}, "episodes": 0}
    )
    episodes = 0
    for show, counts in items:
        if sum(counts.get(s, 0) for s in SEGMENTS) <= 0:
            continue
        episodes += 1
        for s in SEGMENTS:
            corpus[s] += counts.get(s, 0)
            shows[show][s] += counts.get(s, 0)
        shows[show]["episodes"] += 1
    return {
        "corpus": corpus,
        "total_insights": sum(corpus.values()),
        "episodes": episodes,
        "shows": dict(shows),
    }


def classify(counts: dict[str, int]) -> str:
    """front-loaded / even / back-loaded from the early-vs-late gap."""
    total = sum(counts.get(s, 0) for s in SEGMENTS) or 1
    early, late = counts.get("early", 0) / total, counts.get("late", 0) / total
    if early - late >= 0.15:
        return "front-loaded"
    if late - early >= 0.15:
        return "back-loaded"
    return "even"


def _bar(frac: float, width: int = 10) -> str:
    filled = max(0, min(width, round(frac * width)))
    return "█" * filled + "░" * (width - filled)


def render(agg: dict[str, Any]) -> str:
    """A scannable markdown/ASCII report."""
    corpus = agg["corpus"]
    total = agg["total_insights"] or 1
    lines = [
        f"# Insight density — {agg['episodes']} episodes · {agg['total_insights']} insights",
        "",
    ]
    lines.append("## Corpus distribution")
    for seg in SEGMENTS:
        frac = corpus[seg] / total
        lines.append(f"  {seg:<5} {_bar(frac)} {frac * 100:4.0f}%  ({corpus[seg]})")
    lines.append("")
    lines.append("## By show (early/mid/late %)")
    rows = sorted(agg["shows"].items(), key=lambda kv: -sum(kv[1][s] for s in SEGMENTS))
    for show, counts in rows:
        show_total = sum(counts[s] for s in SEGMENTS) or 1
        dist = "/".join(str(round(counts[s] / show_total * 100)) for s in SEGMENTS)
        lines.append(
            f"  {show[:34]:<34} {classify(counts):<12} {dist:>10}  ({counts['episodes']} ep)"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Insight-density report (#1140).")
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus output dir.")
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON instead of the report."
    )
    args = parser.parse_args(argv)
    if not args.corpus.is_dir():
        print(f"corpus not found: {args.corpus}", file=sys.stderr)
        return 2
    items: list[tuple[str, dict[str, int]]] = []
    for path in find_density_envelopes(args.corpus):
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        items.append((_show_title(path), counts_of(envelope)))
    agg = aggregate(items)
    if args.json:
        print(json.dumps(agg, indent=2, sort_keys=True))
    else:
        print(render(agg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
