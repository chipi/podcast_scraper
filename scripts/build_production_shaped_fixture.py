#!/usr/bin/env python3
"""Build a production-shaped fixture for the viewer's Tier-2 test matrix.

RFC-086 / ADR-095. Deterministic + idempotent: given the same source corpus
and pick, produces byte-identical output. Re-run to refresh fixture when
behaviour drifts.

Usage:
    python scripts/build_production_shaped_fixture.py \\
        --corpus /abs/path/to/your/real-corpus \\
        --api http://localhost:8000 \\
        --output web/gi-kg-viewer/e2e/fixtures/production-shaped \\
        [--episodes-per-feed 5] [--max-feeds 5]

The ``--corpus`` argument is operator-supplied; committed code never
names a specific local corpus path (copyright + per-operator privacy).

Requires ``make serve`` running so the API endpoints respond. The script
trims artifact files to keep the fixture under ~5 MB.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def fetch_json(api: str, path: str, params: dict[str, Any]) -> Any:
    qs = urllib.parse.urlencode({k: str(v) for k, v in params.items() if v is not None})
    url = f"{api.rstrip('/')}{path}{'?' + qs if qs else ''}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        if resp.status != 200:
            raise SystemExit(f"GET {url} -> HTTP {resp.status}")
        return json.loads(resp.read())


def write_json(target: Path, payload: Any) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def slim_episode_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields the viewer reads from /api/corpus/episodes items."""
    keep = {
        "metadata_relative_path",
        "feed_id",
        "feed_display_title",
        "feed_rss_url",
        "feed_description",
        "topics",
        "summary_title",
        "summary_bullets_preview",
        "summary_bullet_graph_topic_ids",
        "summary_preview",
        "episode_id",
        "episode_title",
        "publish_date",
        "feed_image_url",
        "episode_image_url",
        "duration_seconds",
        "gi_relative_path",
        "kg_relative_path",
        "has_gi",
        "has_kg",
        "cil_digest_topics",
    }
    return {k: item[k] for k in keep if k in item}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, type=Path, help="absolute corpus path")
    p.add_argument("--api", default="http://localhost:8000", help="API base URL")
    p.add_argument("--output", required=True, type=Path, help="fixture output dir")
    p.add_argument("--episodes-per-feed", type=int, default=5)
    p.add_argument("--max-feeds", type=int, default=5)
    args = p.parse_args()

    if not args.corpus.is_dir():
        sys.exit(f"corpus path is not a directory: {args.corpus}")

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus").mkdir(exist_ok=True)
    (out / "artifacts").mkdir(exist_ok=True)
    (out / "search").mkdir(exist_ok=True)

    api = args.api
    corpus = str(args.corpus.resolve())

    # 1. Feeds — pick first N deterministically by feed_id alphabetical.
    feeds_resp = fetch_json(api, "/api/corpus/feeds", {"path": corpus})
    all_feeds = sorted(feeds_resp.get("feeds", []), key=lambda f: f["feed_id"])
    picked_feeds = all_feeds[: args.max_feeds]
    picked_feed_ids = {f["feed_id"] for f in picked_feeds}
    print(f"feeds: {len(picked_feeds)}/{len(all_feeds)} picked")

    # 2. Episodes — pick first N per feed by publish_date desc within feed.
    by_feed: dict[str, list[dict[str, Any]]] = {}
    cursor: str | None = None
    while True:
        params: dict[str, Any] = {"path": corpus, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        eps_resp = fetch_json(api, "/api/corpus/episodes", params)
        for item in eps_resp.get("items", []):
            fid = item.get("feed_id")
            if fid not in picked_feed_ids:
                continue
            by_feed.setdefault(fid, []).append(item)
        cursor = eps_resp.get("next_cursor")
        if not cursor:
            break
    picked_episodes: list[dict[str, Any]] = []
    for fid in picked_feed_ids:
        feed_eps = sorted(
            by_feed.get(fid, []),
            key=lambda e: e.get("publish_date") or "",
            reverse=True,
        )[: args.episodes_per_feed]
        picked_episodes.extend(feed_eps)
    picked_episodes.sort(key=lambda e: (e["feed_id"], e.get("publish_date") or ""))
    print(f"episodes: {len(picked_episodes)} picked across {len(by_feed)} feeds")

    # 3. Trim feeds response to the same count.
    trimmed_feeds = [
        {**f, "episode_count": sum(1 for e in picked_episodes if e["feed_id"] == f["feed_id"])}
        for f in picked_feeds
    ]
    write_json(out / "corpus" / "feeds.json", {"path": corpus, "feeds": trimmed_feeds})

    # 4. Episodes (slimmed).
    write_json(
        out / "corpus" / "episodes.json",
        {
            "path": corpus,
            "feed_id": None,
            "items": [slim_episode_item(e) for e in picked_episodes],
            "next_cursor": None,
        },
    )

    # 5. Episode details — fetch per metadata_relative_path.
    details: dict[str, Any] = {}
    for ep in picked_episodes:
        rel = ep["metadata_relative_path"]
        try:
            d = fetch_json(
                api,
                "/api/corpus/episodes/detail",
                # API uses ``metadata_relpath`` query arg name (server-internal),
                # despite responses + most other code paths using ``metadata_relative_path``.
                {"path": corpus, "metadata_relpath": rel},
            )
            details[rel] = d
        except Exception as exc:  # noqa: BLE001
            print(f"  skip detail for {rel}: {exc}")
    write_json(out / "corpus" / "episode-details.json", details)
    print(f"episode details: {len(details)} captured")

    # 6. Artifacts — copy GI + KG from disk.
    copied = 0
    for ep in picked_episodes:
        for kind_key in ("gi_relative_path", "kg_relative_path"):
            rel = ep.get(kind_key)
            if not rel:
                continue
            src = args.corpus / rel
            if not src.is_file():
                continue
            # Flatten to artifacts/<episode_id>.<kind>.json
            kind = "gi" if kind_key == "gi_relative_path" else "kg"
            dst = out / "artifacts" / f"{ep['episode_id']}.{kind}.json"
            dst.write_bytes(src.read_bytes())
            copied += 1
    print(f"artifacts copied: {copied}")

    # 7. Digest, topic clusters, stats, coverage, persons/top, runs/summary, index/stats.
    for path, name in [
        ("/api/corpus/digest", "digest.json"),
        ("/api/corpus/topic-clusters", "topic-clusters.json"),
        ("/api/corpus/stats", "stats.json"),
        ("/api/corpus/coverage", "coverage.json"),
        ("/api/corpus/persons/top", "persons-top.json"),
        ("/api/corpus/runs/summary", "runs-summary.json"),
        ("/api/index/stats", "index-stats.json"),
    ]:
        try:
            payload = fetch_json(api, path, {"path": corpus})
            write_json(out / "corpus" / name, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {path}: {exc}")

    # 8. Pre-recorded search results for a fixed query list.
    queries = ["AI", "policy", "economy", "Taiwan", "climate"]
    search_by_query: dict[str, Any] = {}
    for q in queries:
        try:
            r = fetch_json(api, "/api/search", {"q": q, "path": corpus, "top_k": 5})
            search_by_query[q] = r
        except Exception as exc:  # noqa: BLE001
            print(f"  skip search {q!r}: {exc}")
            search_by_query[q] = {"query": q, "results": []}
    write_json(out / "search" / "results-by-query.json", search_by_query)

    # 9. Manifest — index of everything for the mock helper.
    manifest = {
        "schema_version": "1",
        "source_corpus": corpus,
        "picked": {
            "feeds": [f["feed_id"] for f in picked_feeds],
            "episodes": [
                {
                    "episode_id": e["episode_id"],
                    "feed_id": e["feed_id"],
                    "metadata_relative_path": e["metadata_relative_path"],
                    "gi_relative_path": e.get("gi_relative_path"),
                    "kg_relative_path": e.get("kg_relative_path"),
                    "publish_date": e.get("publish_date"),
                }
                for e in picked_episodes
            ],
            "queries": queries,
        },
        "artifact_count": copied,
        "detail_count": len(details),
        "total_size_bytes": sum(
            (p.stat().st_size for p in out.rglob("*") if p.is_file()),
            0,
        ),
    }
    write_json(out / "manifest.json", manifest)
    print(f"manifest written; total fixture size = {manifest['total_size_bytes'] / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
