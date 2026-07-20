#!/usr/bin/env python3
"""Search v3 API-side perf capturer (RFC-107 §P; #1230 S0(d)).

Invoked from ``capture-search-api.sh``. Records p50/p95/p99 for /api/search
across:

1. Per-intent scenarios (5 queries each, all 5 RFC-092 intent classes).
2. top_k grid (10/25/50/100) on 3 queries each.
3. ``api-concurrent-4`` — 4 parallel workers × 5 queries; asserts no SIGSEGV
   (the runtime companion to the compile-time lint at
   scripts/check/lint_search_v3_forbidden_imports.py).

Emits one JSON per scenario under ``<out>``; a small aggregate at the top level.

Deliberately kept as plain-Python + stdlib http; no third-party HTTP client
(matches the graph runbook's philosophy of self-contained perf tooling).
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
    """Return (status_code, elapsed_ms) for one API GET."""
    qs = urllib.parse.urlencode({k: str(v) for k, v in params.items() if v is not None})
    url = f"{api.rstrip('/')}{path}{'?' + qs if qs else ''}"
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            resp.read()  # exhaust body — realistic for measure
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return resp.status, elapsed_ms
    except Exception:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return -1, elapsed_ms


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


def _queries_by_intent(queries_path: Path) -> dict[str, list[str]]:
    data = json.loads(queries_path.read_text())
    by: dict[str, list[str]] = {}
    for q in data.get("queries", []):
        intent = q.get("intent_expected") or "unknown"
        by.setdefault(intent, []).append(q.get("q", ""))
    return by


def _run_serial(
    api: str, corpus: str, queries: list[str], iterations: int, top_k: int, name: str
) -> ScenarioResult:
    latencies: list[int] = []
    ok = 0
    for _ in range(iterations):
        for q in queries:
            status, ms = _fetch_ms(
                api,
                "/api/search",
                {"q": q, "top_k": top_k, "path": corpus},
            )
            latencies.append(ms)
            if status == 200:
                ok += 1
    return _summarize(name, latencies, iterations, ok, sigsegv_free=None)


def _run_concurrent_4(api: str, corpus: str, queries: list[str], iterations: int) -> ScenarioResult:
    latencies: list[int] = []
    ok = 0
    sigsegv_free = True
    for _ in range(iterations):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(_fetch_ms, api, "/api/search", {"q": q, "top_k": 10, "path": corpus})
                for q in queries * 4  # 4 workers, ~queries*4 requests
            ]
            for fut in concurrent.futures.as_completed(futures):
                status, ms = fut.result()
                latencies.append(ms)
                if status == 200:
                    ok += 1
                elif status < 0:
                    # A -1 status from _fetch_ms means the socket died — could be
                    # SIGSEGV in the api worker (the #1205 signature is a killed
                    # api process, which produces exactly this HTTP-side symptom).
                    sigsegv_free = False
    return _summarize("api-concurrent-4", latencies, iterations, ok, sigsegv_free=sigsegv_free)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iterations", type=int, default=3)
    args = ap.parse_args()

    by_intent = _queries_by_intent(args.queries)
    if not by_intent:
        print(f"no queries loaded from {args.queries}", file=sys.stderr)
        return 2

    scenarios: list[ScenarioResult] = []

    # Per-intent scenarios (top_k = 10)
    for intent in (
        "entity_lookup",
        "raw_evidence",
        "temporal_tracking",
        "cross_show_synthesis",
        "semantic",
    ):
        qs = by_intent.get(intent, [])
        if not qs:
            print(f"  skip api-intent-{intent}: no queries for this intent")
            continue
        scenarios.append(
            _run_serial(
                args.api, args.corpus, qs, args.iterations, top_k=10, name=f"api-intent-{intent}"
            )
        )

    # top_k grid — reuse a stable sub-set of 3 queries (from raw_evidence class if available)
    grid_queries = (by_intent.get("raw_evidence") or by_intent.get("semantic") or [])[:3]
    if grid_queries:
        for top_k in (10, 25, 50, 100):
            scenarios.append(
                _run_serial(
                    args.api,
                    args.corpus,
                    grid_queries,
                    args.iterations,
                    top_k=top_k,
                    name=f"api-top_k-{top_k}",
                )
            )

    # Concurrent-4 SIGSEGV assertion — 5 queries × 4 workers per iteration
    concurrent_queries = (by_intent.get("raw_evidence") or by_intent.get("semantic") or [])[:5]
    if concurrent_queries:
        scenarios.append(
            _run_concurrent_4(args.api, args.corpus, concurrent_queries, args.iterations)
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "label": args.label,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "api": args.api,
        "corpus": args.corpus,
        "queries_path": str(args.queries),
        "iterations": args.iterations,
        "scenarios": [asdict(s) for s in scenarios],
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"\ncapture-search-api: {len(scenarios)} scenarios captured -> {args.out.name}")
    for s in scenarios:
        sigsegv_tag = (
            "" if s.sigsegv_free is None else (" ✓SIGSEGV-free" if s.sigsegv_free else " ✗SIGSEGV")
        )
        print(
            f"  {s.name:<38} p50={s.p50_ms:6.0f}  p95={s.p95_ms:6.0f}  "
            f"p99={s.p99_ms:6.0f}  ok={s.ok_count}/{s.request_count}{sigsegv_tag}"
        )

    # Exit 1 if the concurrent-4 scenario detected any non-200 (proxy for #1205).
    for s in scenarios:
        if s.name == "api-concurrent-4" and s.sigsegv_free is False:
            print("\nFATAL: api-concurrent-4 detected socket-level failures. Investigate #1205.")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
