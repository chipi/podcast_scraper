#!/usr/bin/env python3
"""Where the API's memory goes: RSS and Python allocations at each stage of a warm start.

Builds the app the way ``serve`` does (installed extensions included) on a corpus, then:

1. ``import``: after importing the app factory;
2. ``startup``: after the lifespan (``server_started`` hooks, the cache warmer) has run and RSS
   has stopped growing;
3. ``routes``: after one request to each listed route.

At each stage it prints RSS (``VmRSS``, ``RssAnon``, ``RssFile``) and the bytes ``tracemalloc``
traces. The gap between ``RssAnon`` and the traced bytes is native memory (numpy, Arrow, LanceDB,
torch), which ``tracemalloc`` cannot see. At the end it prints the top allocation sites: by file,
and by the innermost frame inside an app package (``podcast_scraper``, ``closelistening_*``) with
the app frame that called it, so a cache shows up under the code that fills it. ``tracemalloc``
itself costs memory (``API_MEMORY_FRAMES``, default 10), which RSS includes: compare stages, not
absolute RSS, with prod.

Usage::

    PODCAST_DATABASE_URL=... python scripts/measure/api_memory.py --output-dir <corpus> \\
        [--route /api/app/episodes ...] [--settle-seconds 20] [--json out.json]

It needs a migrated database (``scripts/tools/dev_db.py run -- ...`` gives one) and writes nothing
except what the app itself writes under the corpus (its caches). Point it at a copy, never at a
corpus another process serves.
"""

from __future__ import annotations

import os
import tracemalloc

# Frames kept per allocation: more frames attribute better and cost more memory, which RSS
# then includes.
tracemalloc.start(int(os.environ.get("API_MEMORY_FRAMES", "10")))

import argparse  # noqa: E402
import json  # noqa: E402
import linecache  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

APP_PACKAGES = ("podcast_scraper", "closelistening_")

DEFAULT_ROUTES = (
    "/api/health",
    "/api/app/episodes",
    "/api/app/podcasts",
    "/api/app/trending",
    "/api/app/storylines",
    "/api/app/themes",
    "/api/app/discover",
    "/api/app/corpus/trending-topics",
    "/api/corpus/digest",
    "/api/corpus/stats",
    "/api/corpus/coverage",
    "/api/corpus/persons/top",
    "/api/app/search?q=climate",
    "/api/app/discover/rails",
    "/api/app/discover/learn",
    "/api/app/discover/reel-topics",
    "/api/app/your-week",
    "/api/app/recommended",
    "/api/app/key-voices",
    "/api/app/whats-new",
)

#: Per-episode routes, asked for the first ``--episodes`` slugs ``/api/app/episodes`` returns.
EPISODE_ROUTES = (
    "/api/app/episodes/{slug}",
    "/api/app/episodes/{slug}/related",
    "/api/app/episodes/{slug}/entities",
    "/api/app/episodes/{slug}/insights",
    "/api/app/episodes/{slug}/segments",
    "/api/app/episodes/{slug}/moments",
)


def rss() -> dict[str, int]:
    """``VmRSS``, ``RssAnon``, ``RssFile`` of this process, in MB."""
    out: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "RssAnon", "RssFile"):
            out[key] = int(value.split()[0]) // 1024
    return out


def stage(name: str, stages: list[dict[str, Any]]) -> None:
    traced, peak = tracemalloc.get_traced_memory()
    row = {"stage": name, **rss(), "traced_mb": traced // 2**20, "traced_peak_mb": peak // 2**20}
    stages.append(row)
    print(json.dumps(row), flush=True)


def settle(seconds: float, limit: float = 300.0) -> None:
    """Wait until RSS has not grown by more than 5 MB for ``seconds`` (warmers run on threads)."""
    start = time.monotonic()
    last, since = rss()["VmRSS"], time.monotonic()
    while time.monotonic() - start < limit:
        time.sleep(2)
        now = rss()["VmRSS"]
        if now - last > 5:
            last, since = now, time.monotonic()
        elif time.monotonic() - since >= seconds:
            return


def _app_label(filename: str) -> str | None:
    parts = Path(filename).parts
    idx = [i for i, p in enumerate(parts) if p.startswith(APP_PACKAGES)]
    return "/".join(parts[idx[-1] :]) if idx else None


def _app_frame(tb: tracemalloc.Traceback) -> tuple[str, str, int]:
    """``(label, file, line)`` of the innermost frame inside an app package, labelled with the app
    frame that called it (so a shared loader shows which cache filled it); else the innermost."""
    app = [(f, label) for f in reversed(tb) if (label := _app_label(f.filename))]  # newest first
    if not app:
        frame = tb[-1]
        return frame.filename, frame.filename, frame.lineno
    (frame, label), caller = app[0], app[1] if len(app) > 1 else None
    via = f" <- {caller[1]}:{caller[0].lineno}" if caller else ""
    return f"{label}:{frame.lineno}{via}", frame.filename, frame.lineno


def top(snapshot: tracemalloc.Snapshot, n: int) -> dict[str, list[dict[str, Any]]]:
    by_file = [
        {"site": s.traceback[0].filename, "mb": round(s.size / 2**20, 1), "count": s.count}
        for s in snapshot.statistics("filename")[:n]
    ]
    grouped: dict[tuple[str, str, int], list[int]] = {}
    for s in snapshot.statistics("traceback"):
        key = _app_frame(s.traceback)
        acc = grouped.setdefault(key, [0, 0])
        acc[0] += s.size
        acc[1] += s.count
    by_app = [
        {
            "site": label,
            "mb": round(v[0] / 2**20, 1),
            "count": v[1],
            "line": linecache.getline(path, line).strip(),
        }
        for (label, path, line), v in sorted(grouped.items(), key=lambda kv: -kv[1][0])[:n]
    ]
    return {"by_file": by_file, "by_app_frame": by_app}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route", action="append", default=None)
    parser.add_argument("--settle-seconds", type=float, default=20.0)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--sign-in-as",
        default=None,
        help="mock-provider identity hint (APP_OAUTH_PROVIDER=mock), e.g. pat-player",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if not os.environ.get("PODCAST_DATABASE_URL"):
        print("PODCAST_DATABASE_URL is not set (run under scripts/tools/dev_db.py run --)")
        return 2

    stages: list[dict[str, Any]] = []
    from fastapi.testclient import TestClient

    from podcast_scraper.server.app import serve_feature_kwargs_from_environ
    from podcast_scraper.server.cli_handlers import _load_create_app

    stage("import", stages)
    kw = serve_feature_kwargs_from_environ()
    app = _load_create_app()(
        args.output_dir.resolve(),
        static_dir=False,
        enable_feeds_api=bool(kw["enable_feeds_api"]),
        enable_operator_config_api=bool(kw["enable_operator_config_api"]),
        enable_jobs_api=bool(kw["enable_jobs_api"]),
        operator_config_file=kw["operator_config_file"],
    )
    statuses: dict[str, int] = {}
    with TestClient(app) as client:
        settle(args.settle_seconds)
        stage("startup", stages)
        if args.sign_in_as:
            client.get("/api/app/auth/login", params={"as": args.sign_in_as}, follow_redirects=True)
            me = client.get("/api/app/me")
            print(f"{me.status_code} signed in as {args.sign_in_as}", flush=True)
            if me.status_code != 200:
                return 3
        routes = list(args.route or DEFAULT_ROUTES)
        listing = client.get("/api/app/episodes")
        if listing.status_code == 200:
            body = listing.json()
            items = body.get("items", body) if isinstance(body, dict) else body
            slugs = [str(i["slug"]) for i in items if isinstance(i, dict) and i.get("slug")]
            routes += [r.format(slug=s) for s in slugs[: args.episodes] for r in EPISODE_ROUTES]
        for route in routes:
            t0 = time.monotonic()
            statuses[route] = client.get(route).status_code
            print(f"{statuses[route]} {route} {time.monotonic() - t0:.2f}s", flush=True)
        settle(args.settle_seconds)
        stage("routes", stages)
        tops = top(tracemalloc.take_snapshot(), args.top)
    report: dict[str, Any] = {"stages": stages, "statuses": statuses, **tops}
    for key in ("by_app_frame", "by_file"):
        print(f"--- top by {key}")
        for row in tops[key]:
            print(f"{row['mb']:8.1f} MB  {row['count']:8d}  {row['site']}  {row.get('line', '')}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
