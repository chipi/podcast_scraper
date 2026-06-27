"""Probes against a deploy's ``/api/enrichment/*`` + jobs surface (RFC-088).

Layered on top of :mod:`podcast_obs.sources.prod_api` — credential-free, only needs
``api_base``. Lets an agent see what the enrichment layer is doing (last run, per-enricher
health, rollup metrics, JSONL events), trigger a re-enable after a transient auto-disable,
and cancel an in-flight job.

``eval_history`` reads the local on-disk eval roots when present (the operator's machine,
or a deploy mount); it's the one probe that's not a remote HTTP call — the eval root is
operator-side data per :doc:`AGENTS.md` (no remote /api/enrichment/eval-history endpoint
exists yet, by design — eval artefacts are frozen-once-written under data/eval/runs/).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .._http import get_json, post_json
from ..config import TargetConfig
from ..result import err, ok

_NOT_CONFIGURED = "api_base not set (PODCAST_OBS_API_BASE or targets.<name>.api_base)"

# Must match ``podcast_scraper.server.jobs.COMMAND_ENRICHMENT``. We keep a
# local copy so this module stays import-decoupled from the application
# package; a unit test asserts the two constants stay in lockstep.
COMMAND_ENRICHMENT = "corpus_enrichment"

# Compact subset of PipelineJobRecord we surface for an enrichment-run listing
# (mirrors prod_api._RUN_FIELDS but filtered to command_type == corpus_enrichment).
_RUN_FIELDS = (
    "job_id",
    "status",
    "created_at",
    "started_at",
    "ended_at",
    "exit_code",
    "command_type",
    "error_reason",
)


def _base(target: TargetConfig) -> Optional[str]:
    return target.api_base.rstrip("/") if target.api_base else None


# --- read probes -------------------------------------------------------------------


def run_status(target: TargetConfig) -> dict:
    """GET ``/api/enrichment/status`` — last status snapshot for the corpus."""
    base = _base(target)
    if not base:
        return err("enrichment.status", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/enrichment/status"
    try:
        data = get_json(url, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001 — surface transport/HTTP error as a result
        return err("enrichment.status", f"GET {url} failed: {exc}")
    return ok("enrichment.status", data)


def recent_runs(target: TargetConfig, limit: int = 10) -> dict:
    """GET ``/api/jobs`` filtered to ``command_type == corpus_enrichment`` (newest first)."""
    base = _base(target)
    if not base:
        return err("enrichment.runs", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/jobs"
    try:
        data = get_json(url, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err("enrichment.runs", f"GET {url} failed: {exc}")
    jobs = data.get("jobs", []) if isinstance(data, dict) else []
    enrich_only = [j for j in jobs if (j or {}).get("command_type") == COMMAND_ENRICHMENT]
    newest_first = sorted(enrich_only, key=lambda job: job.get("created_at") or "", reverse=True)
    runs = [{key: job.get(key) for key in _RUN_FIELDS} for job in newest_first[: max(limit, 0)]]
    return ok(
        "enrichment.runs",
        {
            "path": data.get("path") if isinstance(data, dict) else None,
            "count": len(runs),
            "runs": runs,
        },
    )


def health(target: TargetConfig, enricher_id: Optional[str] = None) -> dict:
    """GET ``/api/enrichment/health`` — per-enricher health (or a single enricher's record)."""
    base = _base(target)
    if not base:
        return err("enrichment.health", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/enrichment/health"
    params = {"enricher_id": enricher_id} if enricher_id else None
    try:
        data = get_json(url, params=params, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err("enrichment.health", f"GET {url} failed: {exc}")
    return ok("enrichment.health", data)


def metrics(target: TargetConfig, window: str = "24h") -> dict:
    """GET ``/api/enrichment/metrics?window=`` — rollup metrics over a window."""
    base = _base(target)
    if not base:
        return err("enrichment.metrics", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/enrichment/metrics"
    try:
        data = get_json(url, params={"window": window}, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err("enrichment.metrics", f"GET {url} failed: {exc}")
    return ok("enrichment.metrics", data)


def run_summary(target: TargetConfig) -> dict:
    """GET ``/api/enrichment/run-summary`` — last completed run summary."""
    base = _base(target)
    if not base:
        return err("enrichment.run_summary", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/enrichment/run-summary"
    try:
        data = get_json(url, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err("enrichment.run_summary", f"GET {url} failed: {exc}")
    return ok("enrichment.run_summary", data)


def recent_events(
    target: TargetConfig,
    *,
    enricher_id: Optional[str] = None,
    event_type: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 50,
) -> dict:
    """GET ``/api/enrichment/events`` — JSONL tail (filterable).

    ``run_id`` is applied client-side after the fetch (the route doesn't
    yet accept it as a query param; this keeps the route minimal and the
    join logic in the observability layer where it belongs).
    """
    base = _base(target)
    if not base:
        return err("enrichment.events", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/enrichment/events"
    params: dict[str, Any] = {"limit": int(limit)}
    if enricher_id:
        params["enricher_id"] = enricher_id
    if event_type:
        params["event_type"] = event_type
    try:
        data = get_json(url, params=params, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err("enrichment.events", f"GET {url} failed: {exc}")
    if run_id and isinstance(data, dict):
        events = data.get("events")
        if isinstance(events, list):
            filtered = [e for e in events if isinstance(e, dict) and e.get("run_id") == run_id]
            data = {**data, "events": filtered, "count": len(filtered)}
    return ok("enrichment.events", data)


def eval_history(
    target: TargetConfig,
    *,
    eval_root: Optional[str] = None,
    limit: int = 10,
) -> dict:
    """List the last *limit* enrichment-tagged eval runs from ``data/eval/runs/`` on disk.

    No remote endpoint — eval artefacts are operator-side, frozen-once-written
    ([[feedback_never_mutate_historical_artifacts]]). Falls back to
    ``data/eval/runs`` relative to CWD when ``eval_root`` is not given.

    Picks up runs whose ``run_id`` starts with ``enrichment-`` or whose ``metadata.json``
    has ``kind == "enrichment"``. Each entry is the parsed metadata.json plus the run dir.
    """
    root = Path(eval_root or "data/eval/runs")
    if not root.is_dir():
        return err(
            "enrichment.eval_history",
            f"eval root {root} not found (pass eval_root= or run from repo root)",
            configured=False,
        )
    runs: list[dict[str, Any]] = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "metadata.json"
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                parsed = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    meta = parsed
            except Exception:  # noqa: BLE001 — skip malformed metadata, don't crash the listing
                meta = {}
        is_enrich = run_dir.name.startswith("enrichment-") or meta.get("kind") == "enrichment"
        if not is_enrich:
            continue
        runs.append({"run_id": run_dir.name, "path": str(run_dir), "metadata": meta})
        if len(runs) >= max(limit, 0):
            break
    return ok(
        "enrichment.eval_history",
        {"eval_root": str(root), "count": len(runs), "runs": runs},
    )


# --- write probes (re-enable + cancel) ---------------------------------------------


def re_enable(target: TargetConfig, enricher_id: str, reason: Optional[str] = None) -> dict:
    """POST ``/api/enrichment/health/{enricher_id}/re-enable`` — clears auto_disabled."""
    base = _base(target)
    if not base:
        return err("enrichment.re_enable", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/enrichment/health/{enricher_id}/re-enable"
    body = {"reason": reason} if reason else {}
    try:
        data = post_json(url, json=body, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err("enrichment.re_enable", f"POST {url} failed: {exc}")
    return ok("enrichment.re_enable", data)


def cancel(target: TargetConfig, job_id: str) -> dict:
    """POST ``/api/jobs/{job_id}/cancel`` — command_type-agnostic cancel (works on
    enrichment jobs since the registry doesn't distinguish)."""
    base = _base(target)
    if not base:
        return err("enrichment.cancel", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/jobs/{job_id}/cancel"
    try:
        data = post_json(url, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err("enrichment.cancel", f"POST {url} failed: {exc}")
    return ok("enrichment.cancel", data)
