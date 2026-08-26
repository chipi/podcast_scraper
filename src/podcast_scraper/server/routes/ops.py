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


#: The 11 credentials deploy-prod stages into tmpfs (ADR-115) — mirrors
#: .github/actions/stage-prod-secrets. On 2026-08-18 the ENTIRE staging dir was missing and it
#: took ~5 hours, two wrong diagnoses, a deleted live key and three deploys to establish that.
#: GET /api/ops/secrets/status answers it in one request (#1690).
_EXPECTED_SECRETS = (
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "mistral_api_key",
    "deepseek_api_key",
    "grok_api_key",
    "deepgram_api_key",
    "litellm_api_key",
    "podcast_sentry_dsn_api",
    "podcast_sentry_dsn_pipeline",
    "app_operator_api_key",
)

#: Where staged secrets appear: docker-secrets mount inside a container, host tmpfs outside.
_SECRETS_DIRS = ("/run/secrets", "/dev/shm/podcast-secrets")


@router.get("/ops/secrets/status")
def ops_secrets_status() -> dict:
    """Presence/size/hash-prefix of every staged secret — VALUES NEVER CROSS THIS BOUNDARY.

    Three states, deliberately distinct (the first diagnostic written during the outage
    collapsed them into one 'ABSENT' by swallowing the read error — the same bug class it was
    hunting): ``present=false`` (missing), ``present=true, bytes=0`` (empty), and
    ``present=true, readable=false`` (exists but cannot be read). The 12-char sha256 prefix is
    what makes 'present' actionable: compare it against the gateway's token hash to tell
    'present' from 'present and correct' without ever seeing the value.
    """
    import hashlib

    override = os.environ.get("PODCAST_SECRETS_STATUS_DIR")
    candidates = (override,) if override else _SECRETS_DIRS
    base = next((Path(d) for d in candidates if d and Path(d).is_dir()), None)
    rows = []
    for name in _EXPECTED_SECRETS:
        path = base / name if base is not None else None
        if path is None or not path.exists():
            rows.append(
                {
                    "name": name,
                    "present": False,
                    "readable": None,
                    "bytes": None,
                    "sha256_prefix": None,
                }
            )
            continue
        try:
            value = path.read_bytes()
        except OSError:
            rows.append(
                {
                    "name": name,
                    "present": True,
                    "readable": False,
                    "bytes": None,
                    "sha256_prefix": None,
                }
            )
            continue
        rows.append(
            {
                "name": name,
                "present": True,
                "readable": True,
                "bytes": len(value),
                "sha256_prefix": hashlib.sha256(value).hexdigest()[:12] if value else None,
            }
        )
    return {"dir": str(base) if base is not None else None, "secrets": rows}


#: Minimum seconds between live gateway auth probes — the endpoint makes a real upstream call.
_GATEWAY_PROBE_MIN_INTERVAL_S = 10.0
_gateway_probe_last: list[float] = []


@router.get("/ops/gateway/auth")
def ops_gateway_auth() -> dict:
    """Does the credential THIS CONTAINER actually holds authenticate against the gateway?

    The probe deliberately uses ``LITELLM_API_KEY`` from the process env — the value the
    secrets shim exported from ``/run/secrets/`` — because that is what a container actually
    receives. The 2026-08-18 D5 probe reported 401 for an evening while the stack held a good
    key, through two bugs this design cannot reproduce: it ran without the secrets overlay
    (key never mounted) and with ``--entrypoint python`` (shim skipped, env never exported).
    Probing from inside the serving process tests the mounted+exported path end to end.

    Returns hash prefix only, never the key. The gateway KEYS listing stays workflow-side
    (mint-prod-gateway-key action=list): it needs the LiteLLM master key, which lives outside
    this container on purpose (#1689).
    """
    import hashlib
    import time as _time

    import httpx

    now = _time.monotonic()
    if _gateway_probe_last and now - _gateway_probe_last[-1] < _GATEWAY_PROBE_MIN_INTERVAL_S:
        raise HTTPException(status_code=429, detail="gateway auth probe rate-limited; retry soon.")
    _gateway_probe_last[:] = [now]

    base = (os.environ.get("LITELLM_API_BASE") or "").rstrip("/")
    key = os.environ.get("LITELLM_API_KEY") or ""
    if not base:
        return {
            "base": None,
            "key_present": bool(key),
            "http_status": None,
            "ok": False,
            "detail": "LITELLM_API_BASE not set in this container",
        }
    if not key:
        return {
            "base": base,
            "key_present": False,
            "http_status": None,
            "ok": False,
            "detail": "LITELLM_API_KEY not set — the shim exported nothing (the 2026-08-18 shape)",
        }
    try:
        resp = httpx.get(f"{base}/models", headers={"Authorization": f"Bearer {key}"}, timeout=5.0)
        status = resp.status_code
    except httpx.HTTPError as exc:
        return {
            "base": base,
            "key_present": True,
            "http_status": None,
            "ok": False,
            "detail": f"gateway unreachable: {type(exc).__name__}",
        }
    return {
        "base": base,
        "key_present": True,
        "key_sha256_prefix": hashlib.sha256(key.encode()).hexdigest()[:12],
        "http_status": status,
        "ok": status == 200,
    }


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
