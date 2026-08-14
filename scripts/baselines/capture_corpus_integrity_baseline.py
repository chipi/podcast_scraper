#!/usr/bin/env python3
"""Capture the corpus-integrity baseline that epic #1657 is measured against.

Why this exists
---------------
#1646 (speaker detection silently skipped for every episode over 25 MB) destroyed insight
attribution across most of the corpus, and nothing noticed because every existing quality
signal measures *artifact presence* — episodes indexed, GI present, KG present — none of
which changes when attribution fails.

The repair in #1655 has to prove it worked. "It looks better" is not an acceptance test, so
this script freezes the pre-fix numbers into a checked-in artifact. Re-run it after the
repair and diff: the same script, the same corpus, two files.

What it measures
----------------
Three families, deliberately including the ones that were *green* while the corpus was
broken — a baseline that only records the damaged dimension cannot show that a fix left the
rest alone.

1. ``stages``       — per episode, whether speaker detection actually ran. The tell is
                      ``processing.stage_timings.extract_names_time``: ``null`` means the
                      stage returned before it recorded a timing (``processing.py:840/845``),
                      which is indistinguishable from "never configured". That ambiguity is
                      itself the subject of #1647.
2. ``attribution``  — per episode, GI insight counts split by ``surfaceable``, plus voice
                      types. This is the dimension that was silently destroyed.
3. ``artifacts``    — coverage and enricher record counts. Expected to be unchanged by the
                      repair; recorded so that "unchanged" is evidenced rather than assumed.

Usage
-----
    PODCAST_OPERATOR_KEY=...  \
    PODCAST_BASE_URL=https://prod-podcast.<TAILNET>.ts.net \
        python scripts/baselines/capture_corpus_integrity_baseline.py \
            --out data/baselines/corpus-integrity-2026-08-14.json

The operator host comes from ``PODCAST_BASE_URL`` or ``--base-url`` and has **no default**:
operator identifiers are never committed to this repo (CONTRIBUTING.md § "No operator
identifiers in the repo"), and a wrong-but-plausible default would be worse than an explicit
error. ``--limit`` caps the episode count for a smoke run; omit it for the real baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests

# Placeholder only — never a real host. The operator FQDN is supplied at run time via
# PODCAST_BASE_URL / --base-url and is deliberately absent from the repo (deny-list gate in
# .github/workflows/secret-scan.yml).
BASE_URL_PLACEHOLDER = "https://prod-podcast.<TAILNET>.ts.net"
# _check_episode_size_skip compares the media Content-Length against this (rss/downloader.py).
# Recorded here so the baseline is self-describing about the threshold under test.
OPENAI_MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
SCHEMA_VERSION = "1"


class OperatorClient:
    """Minimal read-only operator-API client."""

    def __init__(self, base_url: str, key: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["X-Operator-Key"] = key

    def get_json(self, path: str, **params: Any) -> Optional[Any]:
        """GET a JSON endpoint; None on any transport or decode failure (never raises)."""
        try:
            resp = self.session.get(
                f"{self.base_url}{path}", params=params or None, timeout=self.timeout
            )
            if resp.status_code != 200:
                return None
            return resp.json()
        except (requests.RequestException, ValueError):
            return None

    def text_file(self, relpath: str) -> Optional[Any]:
        """Read a corpus-relative JSON artifact through /api/corpus/text-file."""
        return self.get_json("/api/corpus/text-file", relpath=relpath)


def iter_episodes(client: OperatorClient, limit: Optional[int]) -> List[Dict[str, Any]]:
    """Page the full episode catalog via next_cursor.

    ``offset`` is silently ignored by this endpoint and the page caps at 200 regardless of
    ``limit`` (#1654) — so cursor paging is the only correct way to reach every episode.
    """
    episodes: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    while True:
        payload = client.get_json("/api/corpus/episodes", limit=200, cursor=cursor or "")
        if not payload:
            break
        items = payload.get("items") or []
        episodes.extend(items)
        cursor = payload.get("next_cursor")
        if not cursor or not items or (limit and len(episodes) >= limit):
            break
    return episodes[:limit] if limit else episodes


def _gi_relpath(metadata_relpath: str) -> str:
    """GI sits beside the metadata sidecar with the same stem."""
    return metadata_relpath[: -len(".metadata.json")] + ".gi.json"


def probe_episode(client: OperatorClient, episode: Dict[str, Any]) -> Dict[str, Any]:
    """Collect the stage + attribution facts for one episode."""
    relpath = episode.get("metadata_relative_path") or ""
    row: Dict[str, Any] = {
        "feed": episode.get("feed_display_title"),
        "episode_id": episode.get("episode_id"),
        "metadata_relpath": relpath,
        "duration_seconds": None,
        "speaker_detection_ran": None,
        "extract_names_time": None,
        "insights_total": None,
        "insights_surfaceable": None,
        "voices_total": None,
        "voices_unidentified": None,
        "persons": None,
        "errors": [],
    }

    meta = client.text_file(relpath) if relpath else None
    if meta is None:
        row["errors"].append("metadata_unreadable")
    else:
        timings = (meta.get("processing") or {}).get("stage_timings") or {}
        ent = timings.get("extract_names_time")
        row["extract_names_time"] = ent
        # null is NOT "took zero time" — it is "returned before recording", i.e. skipped.
        row["speaker_detection_ran"] = ent is not None
        row["duration_seconds"] = (meta.get("episode") or {}).get("duration_seconds")

    gi = client.text_file(_gi_relpath(relpath)) if relpath else None
    if gi is None:
        row["errors"].append("gi_unreadable")
    else:
        nodes = gi.get("nodes") or []
        insights = [n for n in nodes if n.get("type") == "Insight"]
        row["insights_total"] = len(insights)
        # is_surfaceable_insight(): absent means surfaceable; only an explicit False excludes.
        row["insights_surfaceable"] = sum(
            1 for n in insights if (n.get("properties") or {}).get("surfaceable") is not False
        )
        row["voices_unidentified"] = sum(
            1
            for n in insights
            if (n.get("properties") or {}).get("speaker_voice_type") == "unidentified"
        )
        row["voices_total"] = len(
            {(n.get("properties") or {}).get("speaker") for n in insights if n.get("properties")}
        )
        row["persons"] = sum(1 for n in nodes if n.get("type") == "Person")
    return row


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 6) if denominator else None


def summarise(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-episode rows corpus-wide and per feed."""
    usable = [r for r in rows if not r["errors"]]
    skipped = [r for r in usable if r["speaker_detection_ran"] is False]
    ins_total = sum(r["insights_total"] or 0 for r in usable)
    ins_surf = sum(r["insights_surfaceable"] or 0 for r in usable)
    zeroed = [
        r
        for r in usable
        if (r["insights_total"] or 0) > 0 and (r["insights_surfaceable"] or 0) == 0
    ]

    per_feed: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"episodes": 0, "skipped": 0, "insights": 0, "surfaceable": 0, "zeroed": 0}
    )
    for r in usable:
        cell = per_feed[r["feed"] or "(unknown)"]
        cell["episodes"] += 1
        cell["skipped"] += 1 if r["speaker_detection_ran"] is False else 0
        cell["insights"] += r["insights_total"] or 0
        cell["surfaceable"] += r["insights_surfaceable"] or 0
        cell["zeroed"] += 1 if r in zeroed else 0
    for cell in per_feed.values():
        cell["attribution_ratio"] = _ratio(cell["surfaceable"], cell["insights"])

    return {
        "episodes_probed": len(rows),
        "episodes_usable": len(usable),
        "episodes_with_errors": len(rows) - len(usable),
        "speaker_detection_skipped": len(skipped),
        "speaker_detection_skipped_ratio": _ratio(len(skipped), len(usable)),
        "insights_total": ins_total,
        "insights_surfaceable": ins_surf,
        "attribution_ratio": _ratio(ins_surf, ins_total),
        "episodes_fully_zeroed": len(zeroed),
        "episodes_fully_zeroed_ratio": _ratio(len(zeroed), len(usable)),
        "audio_minutes_total": round(sum((r["duration_seconds"] or 0) for r in usable) / 60, 1),
        "audio_minutes_skipped": round(sum((r["duration_seconds"] or 0) for r in skipped) / 60, 1),
        "per_feed": dict(sorted(per_feed.items())),
    }


def capture(client: OperatorClient, limit: Optional[int], workers: int) -> Dict[str, Any]:
    episodes = iter_episodes(client, limit)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda ep: probe_episode(client, ep), episodes))

    return {
        "schema_version": SCHEMA_VERSION,
        "epic": "https://github.com/chipi/podcast_scraper/issues/1657",
        "size_gate_bytes": OPENAI_MAX_FILE_SIZE_BYTES,
        "artifacts": {
            "coverage": client.get_json("/api/corpus/coverage"),
            "enrichment_run_summary": client.get_json("/api/enrichment/run-summary"),
            "enrichments": client.get_json("/api/corpus/enrichments"),
            "cost_rollup": (client.get_json("/api/corpus/documents/run-summary") or {}).get(
                "cost_rollup"
            ),
        },
        "summary": summarise(rows),
        "episodes": rows,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-url",
        default=os.environ.get("PODCAST_BASE_URL", ""),
        help=f"Operator API base URL, e.g. {BASE_URL_PLACEHOLDER} (or set PODCAST_BASE_URL).",
    )
    parser.add_argument("--out", required=True, help="Path to write the baseline JSON.")
    parser.add_argument("--limit", type=int, default=None, help="Cap episodes (smoke runs).")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    key = os.environ.get("PODCAST_OPERATOR_KEY", "")
    if not key:
        print("PODCAST_OPERATOR_KEY is not set", file=sys.stderr)
        return 2
    if not args.base_url:
        # No fallback host on purpose: a plausible-looking default would either fail
        # confusingly for someone else's deployment, or silently point at the wrong corpus.
        print(
            "operator base URL is not set — pass --base-url or set PODCAST_BASE_URL "
            f"(e.g. {BASE_URL_PLACEHOLDER})",
            file=sys.stderr,
        )
        return 2

    client = OperatorClient(args.base_url, key)
    if client.get_json("/api/corpus/coverage") is None:
        print(f"operator API not reachable at {args.base_url}", file=sys.stderr)
        return 3

    baseline = capture(client, args.limit, args.workers)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(baseline, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary = baseline["summary"]
    print(f"wrote {args.out}")
    print(
        f"  episodes usable            : {summary['episodes_usable']} "
        f"(errors: {summary['episodes_with_errors']})"
    )
    print(
        f"  speaker detection skipped  : {summary['speaker_detection_skipped']} "
        f"({summary['speaker_detection_skipped_ratio']})"
    )
    print(
        f"  attribution ratio          : {summary['insights_surfaceable']}/"
        f"{summary['insights_total']} = {summary['attribution_ratio']}"
    )
    print(f"  episodes fully zeroed      : {summary['episodes_fully_zeroed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
