#!/usr/bin/env python3
"""Graph API-side perf capturer (perf-traces Chunk 2).

The API-only companion to ``capture-graph-lcp`` (browser paint): measures the
endpoint latency of the graph *load* path — the fan-out the viewer pays before
Cytoscape ever renders. Per the graph-v3 tuning report, corpus-envelope fetch +
per-episode artifact parse dominate graph time-to-canvas, so those are what this
measures, no browser involved.

Invoked from ``capture-graph-api.sh``. Records p50/p95/p99/max/mean over:

1. ``api-artifacts-list``   — GET /api/artifacts (the corpus envelope listing).
2. ``api-artifact-fetch``   — GET /api/artifacts/<relpath> for a sample of the
   listed artifacts (the per-episode GI/KG fan-out the merged graph consumes).
3. ``api-topic-clusters``   — GET /api/corpus/topic-clusters (graph overlay).
4. ``api-concurrent-4``     — 4 parallel workers over the artifact sample;
   asserts no socket death (the #1205 SIGSEGV signature is a killed api worker).

Emits one JSON under ``--out``. Plain-Python + stdlib http, self-contained
(matches capture_search_api.py / the graph runbook philosophy).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ScenarioResult:
    name: str
    iterations: int
    request_count: int
    ok_count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    sigsegv_free: bool | None
    notes: list[str] = field(default_factory=list)


def _fetch_ms(api: str, path: str, params: dict[str, Any]) -> tuple[int, int]:
    """Return (status_code, elapsed_ms) for one API GET. status -1 = socket died."""
    qs = urllib.parse.urlencode({k: str(v) for k, v in params.items() if v is not None})
    url = f"{api.rstrip('/')}{path}{'?' + qs if qs else ''}"
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            resp.read()  # exhaust body — realistic for measure
            return resp.status, int((time.perf_counter() - t0) * 1000)
    except Exception:
        return -1, int((time.perf_counter() - t0) * 1000)


def _percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    idx = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return float(s[idx])


def _summarize(
    name: str, latencies: list[int], iterations: int, ok: int, sigsegv_free: bool | None
) -> ScenarioResult:
    return ScenarioResult(
        name=name,
        iterations=iterations,
        request_count=len(latencies),
        ok_count=ok,
        p50_ms=_percentile(latencies, 50),
        p95_ms=_percentile(latencies, 95),
        p99_ms=_percentile(latencies, 99),
        max_ms=float(max(latencies)) if latencies else 0.0,
        mean_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        sigsegv_free=sigsegv_free,
    )


def _artifact_relpaths(api: str, corpus: str, limit: int) -> list[str]:
    """Discover a sample of fetchable artifact relative_paths from /api/artifacts.

    Prefer graph-relevant artifacts (gi / kg / bridge) since those are what the
    merged graph consumes; fall back to any listed artifact.
    """
    qs = urllib.parse.urlencode({"path": corpus})
    url = f"{api.rstrip('/')}/api/artifacts?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.loads(resp.read())
    except Exception as exc:  # noqa: BLE001 — surfaced as an empty sample below
        print(f"  WARN: /api/artifacts fetch failed: {exc}", file=sys.stderr)
        return []
    items = data.get("artifacts") or []
    graphy = [
        it["relative_path"]
        for it in items
        if isinstance(it, dict)
        and isinstance(it.get("relative_path"), str)
        and any(tok in it["relative_path"] for tok in (".gi.json", ".kg.json", ".bridge.json"))
    ]
    if graphy:
        return graphy[:limit]
    return [
        it["relative_path"]
        for it in items
        if isinstance(it, dict) and isinstance(it.get("relative_path"), str)
    ][:limit]


def _run_list(api: str, corpus: str, iterations: int) -> ScenarioResult:
    latencies: list[int] = []
    ok = 0
    for _ in range(iterations):
        status, ms = _fetch_ms(api, "/api/artifacts", {"path": corpus})
        latencies.append(ms)
        if status == 200:
            ok += 1
    return _summarize("api-artifacts-list", latencies, iterations, ok, sigsegv_free=None)


def _run_artifact_fetch(
    api: str, corpus: str, relpaths: list[str], iterations: int, name: str
) -> ScenarioResult:
    latencies: list[int] = []
    ok = 0
    for _ in range(iterations):
        for rel in relpaths:
            path = "/api/artifacts/" + urllib.parse.quote(rel)
            status, ms = _fetch_ms(api, path, {"path": corpus})
            latencies.append(ms)
            if status == 200:
                ok += 1
    return _summarize(name, latencies, iterations, ok, sigsegv_free=None)


def _run_topic_clusters(api: str, corpus: str, iterations: int) -> ScenarioResult:
    latencies: list[int] = []
    ok = 0
    for _ in range(iterations):
        status, ms = _fetch_ms(api, "/api/corpus/topic-clusters", {"path": corpus})
        latencies.append(ms)
        if status == 200:
            ok += 1
    return _summarize("api-topic-clusters", latencies, iterations, ok, sigsegv_free=None)


def _run_concurrent_4(
    api: str, corpus: str, relpaths: list[str], iterations: int
) -> ScenarioResult:
    latencies: list[int] = []
    ok = 0
    sigsegv_free = True
    for _ in range(iterations):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(
                    _fetch_ms,
                    api,
                    "/api/artifacts/" + urllib.parse.quote(rel),
                    {"path": corpus},
                )
                for rel in relpaths * 4
            ]
            for fut in concurrent.futures.as_completed(futures):
                status, ms = fut.result()
                latencies.append(ms)
                if status == 200:
                    ok += 1
                elif status < 0:
                    sigsegv_free = False
    return _summarize("api-concurrent-4", latencies, iterations, ok, sigsegv_free=sigsegv_free)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=3)
    ap.add_argument("--sample", type=int, default=20, help="Artifacts to fetch for the fan-out.")
    args = ap.parse_args()

    relpaths = _artifact_relpaths(args.api, args.corpus, args.sample)
    if not relpaths:
        print(f"no artifacts discovered at {args.api}/api/artifacts", file=sys.stderr)

    scenarios: list[ScenarioResult] = [_run_list(args.api, args.corpus, args.iterations)]
    if relpaths:
        scenarios.append(
            _run_artifact_fetch(
                args.api, args.corpus, relpaths, args.iterations, "api-artifact-fetch"
            )
        )
    scenarios.append(_run_topic_clusters(args.api, args.corpus, args.iterations))
    if relpaths:
        # Concurrency guard on a small subset so the fan-out stays bounded.
        scenarios.append(_run_concurrent_4(args.api, args.corpus, relpaths[:5], args.iterations))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "label": args.label,
        "api": args.api,
        "corpus": args.corpus,
        "iterations": args.iterations,
        "artifact_sample": len(relpaths),
        "scenarios": [asdict(s) for s in scenarios],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\ncapture_graph_api: {len(scenarios)} scenarios → {args.out.name}")
    for s in scenarios:
        sig = "" if s.sigsegv_free is None else f" sigsegv_free={s.sigsegv_free}"
        print(f"  {s.name:24s} p50={s.p50_ms:.0f}ms p95={s.p95_ms:.0f}ms n={s.request_count}{sig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
