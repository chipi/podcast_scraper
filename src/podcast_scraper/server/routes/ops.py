"""GET /api/ops/summary — prod observability control-plane glance (#803).

Returns the same cross-source summary the ``podcast_obs`` CLI/MCP produce, so the viewer's Ops
view shows a human exactly what an agent sees. Reads ``PODCAST_OBS_*`` from the server env;
defaults the observed target to *this* server so health/version/runs work with zero config,
and the external sources (deploys / cost / logs / errors / alerts) light up when their read-scoped
tokens are present. ``podcast_obs`` is light-dep and pulls no MCP SDK, so importing it is cheap.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["ops"])

# The api container serves /api on 8000 (GH-745); observe ourselves when no target is configured.
_LOCAL_API_BASE = "http://127.0.0.1:8000"
# Keep the endpoint responsive: a configured-but-unreachable source shouldn't hang the dashboard.
# summary() fans out to several backends sequentially, so keep the per-probe timeout tight.
_WEB_TIMEOUT = 4.0


# Deliberately a SYNC ``def`` (not ``async``): ``summary()`` makes blocking ``httpx`` calls, so
# Starlette runs this handler in a threadpool — it must not block the event loop.
@router.get("/ops/summary")
def ops_summary() -> dict:
    """Cross-source prod-state summary (live / unconfigured / failed + per-source envelopes)."""
    from podcast_obs.aggregate import summary as obs_summary
    from podcast_obs.config import ObservabilityConfig

    target = ObservabilityConfig.load().target()
    overrides: dict = {"timeout": _WEB_TIMEOUT}
    if not target.api_base:
        overrides["api_base"] = _LOCAL_API_BASE
    target = replace(target, **overrides)
    data: dict = obs_summary(target)["data"]
    return data


@router.get("/ops/cache-stats")
def ops_cache_stats() -> dict:
    """Per-namespace hit/miss/size + build-time-saved for the central in-process perf caches
    (``podcast_scraper.perf_cache``): the consumer read caches (``app_catalog_rows``,
    ``app_slug_index``, ``app_kg_entity_index``, ``app_corpus_artifact``, ``app_corpus_signals``)
    plus the operator ones (``index_stats``, ``digest_bands``, ``catalog_feeds``). Each carries
    ``hit_rate_pct``, ``avg_build_ms`` (miss cost) and ``est_saved_seconds`` (wall-clock saved on
    hits) — "are the caches earning their keep?" without a profiler. Surfaced to the obs MCP via
    ``prod_cache_stats``.
    """
    from podcast_scraper import perf_cache

    return {"namespaces": perf_cache.stats()}


def _corpus_root(request: Request) -> Path:
    root = getattr(request.app.state, "output_dir", None)
    if root is None:
        raise HTTPException(status_code=400, detail="No corpus configured on this server.")
    return Path(str(root))


# The three corpus reads below replace inspect-prod-corpus.yml's measurements (#1688): reading
# prod state must not cost an approval click, a browser, and ~40s of runner start. All are
# side-effect-free GETs — notably `preprocessing` returns the worklist AS JSON, where the
# workflow's write_worklist=true wrote a file INTO the corpus during a "read-only" audit and
# killed a backup mid-window ('tar: file changed as we read it', 2026-08-18). Sync `def`s:
# blocking filesystem walks belong in Starlette's threadpool, not on the event loop.


@router.get("/ops/corpus/integrity")
def ops_corpus_integrity(request: Request) -> dict:
    """GI integrity for every corpus-member episode — counts, defect lists, verdict."""
    from podcast_scraper.gi.integrity import assess_gi_integrity

    a = assess_gi_integrity(_corpus_root(request))
    defects = (
        len(a["legacy_placeholders"])
        + len(a["episodes_without_gi_block"])
        + len(a["missing_artifact"])
        + len(a["unreadable_artifact"])
        + len(a["empty_artifacts"])
        + len(a["unreadable_metadata"])
    )
    return {
        "metadata_scanned": a["metadata_scanned"],
        "membership_rule_applied": a["membership_rule_applied"],
        "healthy_gi": len(a["healthy"]),
        "legacy_placeholders": len(a["legacy_placeholders"]),
        "no_gi_block": len(a["episodes_without_gi_block"]),
        "missing_artifact": a["missing_artifact"],
        "unreadable_artifact": a["unreadable_artifact"],
        "unreadable_metadata": a["unreadable_metadata"],
        "empty_artifacts": a["empty_artifacts"],
        "undeclared_artifact": len(a["undeclared_artifact"]),
        "verdict": "PASS" if defects == 0 else "FAIL",
    }


@router.get("/ops/corpus/preprocessing")
def ops_corpus_preprocessing(request: Request) -> dict:
    """Preprocessing damage assessment; the repair worklist is returned AS JSON, never written."""
    from podcast_scraper.preprocessing.audit import assess_preprocessing, damaged_episode_ids

    root = _corpus_root(request)
    runs = assess_preprocessing(root)
    damaged = [r for r in runs if r.damaged]
    return {
        "runs_with_metrics": len(runs),
        "damaged_runs": [
            {
                "run_dir": r.run_dir,
                "episodes_in_run": r.episodes_in_run,
                "attempts": r.attempts,
                "completed": r.completed,
                "episode_ids": r.episode_ids,
                "hit_legacy_wall": r.hit_legacy_wall,
            }
            for r in damaged
        ],
        "runs_no_attempt": sum(1 for r in runs if not r.attempts),
        "unpreprocessed_episodes": damaged_episode_ids(root),
        "verdict": "PASS" if not damaged else "FAIL",
    }


@router.get("/ops/corpus/usage")
def ops_corpus_usage(request: Request) -> dict:
    """Corpus disk footprint by top-level directory (bytes actually on disk under the root)."""
    root = _corpus_root(request)
    by_directory: list[dict[str, Any]] = []
    total = 0
    for entry in sorted(root.iterdir()):
        if entry.is_file():
            size = entry.stat().st_size
        elif entry.is_dir():
            size = 0
            for dirpath, _dirnames, filenames in os.walk(entry):
                for fname in filenames:
                    try:
                        size += os.path.getsize(os.path.join(dirpath, fname))
                    except OSError:
                        continue  # a file deleted mid-walk must not fail the read
        else:
            continue
        total += size
        by_directory.append({"path": entry.name, "bytes": size})
    by_directory.sort(key=lambda d: -d["bytes"])
    return {"total_bytes": total, "by_directory": by_directory}
