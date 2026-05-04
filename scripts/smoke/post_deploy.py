#!/usr/bin/env python3
"""Post-deploy codespace smoke (#710).

Runs *inside* the pre-prod codespace via ``gh codespace ssh``. Validates
that the freshly rebuilt codespace has actually-working internals — not
just that the container started. Steps:

1. ``GET /api/health`` returns 200 with the expected capability flags.
2. ``GET /api/scheduled-jobs`` returns 200 with a valid shape (catches
   APScheduler boot regressions).
3. ``POST /api/jobs`` with the operator-pinned smoke RSS feed +
   ``cloud_balanced`` profile + ``max_episodes: 1``. Poll
   ``GET /api/jobs/{id}`` until status is ``succeeded`` or ``failed``,
   bounded by ``SMOKE_BUDGET_SECONDS``.
4. After the job succeeds, walk the corpus on disk and assert at least
   one episode has the four expected artifacts (``.gi.json`` /
   ``.kg.json`` / ``.bridge.json`` / ``.metadata.json``). Quick eye-check
   on the GI: ``model_version`` field non-empty, ``nodes`` contains at
   least one ``Insight`` (regression catch for #701-class stub
   regressions).

Exit codes:
    0  — all steps green.
    1  — one or more steps failed (reason printed to stderr; the GHA
         workflow leaves the codespace running so an operator can SSH
         in and continue debugging).
    2  — usage / setup error (missing env vars, etc).

Reads from environment (set by the GHA workflow):
    SMOKE_FEED_URL          — RSS URL to ingest. Required.
    SMOKE_FEED_NAME         — corpus directory name (default: ``smoke``).
    SMOKE_BUDGET_SECONDS    — wall-clock budget for the pipeline run
                              (default: 420 = 7 min).
    EXPECTED_SHORT_SHA      — commit SHA the smoke is validating (used
                              for the diagnostic banner only; image
                              freshness check is intentionally omitted
                              because docker compose images metadata
                              isn't a reliable freshness signal —
                              GHCR pulls the same :main tag regardless).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "http://localhost:8090"
CORPUS_ROOT = Path("/app/output")


def _http_json(method: str, url: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    """Tiny stdlib HTTP client — codespace may not have requests installed."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = resp.read().decode()
            try:
                return resp.status, json.loads(payload) if payload else None
            except json.JSONDecodeError:
                return resp.status, payload
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        return e.code, body_text


def step_health() -> bool:
    print("==> [1/4] GET /api/health", flush=True)
    code, body = _http_json("GET", f"{API_BASE}/api/health")
    if code != 200:
        print(f"   FAIL: status={code} body={body!r}", file=sys.stderr)
        return False
    if not isinstance(body, dict):
        print(f"   FAIL: body is not JSON object: {body!r}", file=sys.stderr)
        return False
    # Capability flags surface api factory state. ``jobs_api`` /
    # ``feeds_api`` should both be true on a healthy codespace; absence
    # is a regression we want to catch.
    for flag in ("jobs_api", "feeds_api"):
        if not body.get(flag):
            print(f"   FAIL: capability flag {flag!r} missing/false: {body!r}", file=sys.stderr)
            return False
    print(f"   OK: {body}", flush=True)
    return True


def step_scheduled_jobs() -> bool:
    print("==> [2/4] GET /api/scheduled-jobs", flush=True)
    code, body = _http_json("GET", f"{API_BASE}/api/scheduled-jobs")
    if code != 200:
        print(f"   FAIL: status={code} body={body!r}", file=sys.stderr)
        return False
    # Shape: {jobs: [...], timezone: "UTC", ...}. APScheduler boot
    # regressions show as 500 / missing timezone / lifespan crash.
    if not isinstance(body, dict) or "jobs" not in body or not body.get("timezone"):
        print(f"   FAIL: bad shape: {body!r}", file=sys.stderr)
        return False
    print(f"   OK: {len(body['jobs'])} scheduled jobs, tz={body['timezone']}", flush=True)
    return True


def step_real_episode(feed_url: str, feed_name: str, budget_s: int) -> tuple[bool, str | None]:
    print(f"==> [3/4] POST /api/jobs feed={feed_url!r}", flush=True)
    corpus_path = f"/app/output/{feed_name}"
    payload = {
        "feed_url": feed_url,
        "path": corpus_path,
        "profile": "cloud_balanced",
        "max_episodes": 1,
    }
    code, body = _http_json("POST", f"{API_BASE}/api/jobs", body=payload)
    if code not in (200, 201, 202):
        print(f"   FAIL: status={code} body={body!r}", file=sys.stderr)
        return False, None
    if not isinstance(body, dict) or "job_id" not in body:
        print(f"   FAIL: response missing job_id: {body!r}", file=sys.stderr)
        return False, None
    job_id = body["job_id"]
    print(f"   queued job_id={job_id}; polling (budget={budget_s}s)", flush=True)

    deadline = time.monotonic() + budget_s
    last_status = "?"
    quoted_path = urllib.parse.quote(corpus_path, safe="")
    while time.monotonic() < deadline:
        code, body = _http_json("GET", f"{API_BASE}/api/jobs/{job_id}?path={quoted_path}")
        if code == 200 and isinstance(body, dict):
            last_status = body.get("status", "?")
            if last_status == "succeeded":
                print(f"   OK: job succeeded in {body.get('finished_at')}", flush=True)
                return True, corpus_path
            if last_status == "failed":
                print(
                    f"   FAIL: job failed exit_code={body.get('exit_code')} "
                    f"reason={body.get('error_reason')}",
                    file=sys.stderr,
                )
                return False, corpus_path
        time.sleep(5)
    print(
        f"   FAIL: budget exhausted after {budget_s}s (last_status={last_status!r})",
        file=sys.stderr,
    )
    return False, corpus_path


def step_artifacts(corpus_path: str) -> bool:
    print(f"==> [4/4] verify artifacts in {corpus_path}", flush=True)
    metadata_dir = Path(corpus_path) / "metadata"
    if not metadata_dir.is_dir():
        print(f"   FAIL: {metadata_dir} not found", file=sys.stderr)
        return False
    # Find at least one episode with all four artifact suffixes.
    suffixes = (".gi.json", ".kg.json", ".bridge.json", ".metadata.json")
    by_episode: dict[str, set[str]] = {}
    for f in metadata_dir.iterdir():
        for sfx in suffixes:
            if f.name.endswith(sfx):
                stem = f.name[: -len(sfx)]
                by_episode.setdefault(stem, set()).add(sfx)
    complete = [ep for ep, found in by_episode.items() if set(found) == set(suffixes)]
    if not complete:
        print(
            f"   FAIL: no episode with all 4 artifacts. by_episode={by_episode}",
            file=sys.stderr,
        )
        return False
    ep = complete[0]
    print(f"   OK: episode {ep!r} has all 4 artifacts", flush=True)

    # Eye-check on GI: the #701-class regression looked like a stub
    # artifact (1 placeholder Insight, model_version=stub) sneaking past
    # the file-existence check. Refuse stubs explicitly.
    gi_path = metadata_dir / f"{ep}.gi.json"
    try:
        gi = json.loads(gi_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"   FAIL: cannot read {gi_path}: {e}", file=sys.stderr)
        return False
    model_version = gi.get("model_version", "")
    if not model_version or "stub" in model_version.lower():
        print(
            f"   FAIL: GI artifact looks stubbed (model_version={model_version!r})",
            file=sys.stderr,
        )
        return False
    insight_nodes = [n for n in gi.get("nodes", []) if n.get("type") == "Insight"]
    if not insight_nodes:
        print("   FAIL: GI artifact has no Insight nodes", file=sys.stderr)
        return False
    print(
        f"   OK: GI model_version={model_version!r}, " f"insights={len(insight_nodes)}",
        flush=True,
    )
    return True


def main() -> int:
    feed_url = os.environ.get("SMOKE_FEED_URL", "").strip()
    feed_name = os.environ.get("SMOKE_FEED_NAME", "smoke").strip()
    try:
        budget_s = int(os.environ.get("SMOKE_BUDGET_SECONDS", "420"))
    except ValueError:
        print("SMOKE_BUDGET_SECONDS must be int", file=sys.stderr)
        return 2
    if not feed_url:
        print("SMOKE_FEED_URL is required", file=sys.stderr)
        return 2
    expected = os.environ.get("EXPECTED_SHORT_SHA", "?")[:7]
    print(f"==> Smoke against codespace, expected sha={expected}", flush=True)

    if not step_health():
        return 1
    if not step_scheduled_jobs():
        return 1
    ok, corpus_path = step_real_episode(feed_url, feed_name, budget_s)
    if not ok or not corpus_path:
        return 1
    if not step_artifacts(corpus_path):
        return 1

    print("==> Smoke green.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
