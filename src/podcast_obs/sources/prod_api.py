"""Probes against a deploy's own ``/api`` surface — credential-free (needs only ``api_base``).

Works against any deploy: a local ``make serve`` stack (``http://localhost:8080``), prod over
Tailscale (``https://prod-podcast.<tailnet>``), or a drill. These three are the "basics" you can
observe before any external integration (GitHub/Sentry/Grafana) is wired.
"""

from __future__ import annotations

from typing import Optional

from .._http import get_json
from ..config import TargetConfig
from ..result import err, ok

_NOT_CONFIGURED = "api_base not set (PODCAST_OBS_API_BASE or targets.<name>.api_base)"

# Compact subset of PipelineJobRecord we surface for a run listing.
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


def health(target: TargetConfig) -> dict:
    """GET ``{api_base}/api/health`` — full health payload (versions + feature flags)."""
    base = _base(target)
    if not base:
        return err("prod_api.health", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/health"
    try:
        data = get_json(url, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001 — surface any transport/HTTP error as a result
        return err("prod_api.health", f"GET {url} failed: {exc}")
    return ok("prod_api.health", data)


def deployed_version(target: TargetConfig) -> dict:
    """The running code version + the corpus stamp it's serving (derived from health)."""
    result = health(target)
    if not result["ok"]:
        return {**result, "source": "prod_api.version"}
    data = result["data"] if isinstance(result["data"], dict) else {}
    produced_by = data.get("corpus_produced_by") or {}
    return ok(
        "prod_api.version",
        {
            "status": data.get("status"),
            "code_version": data.get("code_version"),
            "corpus_code_version": data.get("corpus_code_version"),
            "corpus_git_sha": produced_by.get("git_sha"),
            "corpus_produced_at": produced_by.get("produced_at"),
            "corpus_version_warning": data.get("corpus_version_warning"),
        },
    )


def recent_pipeline_runs(target: TargetConfig, limit: int = 10) -> dict:
    """GET ``{api_base}/api/jobs`` — the last *limit* pipeline jobs, newest first."""
    base = _base(target)
    if not base:
        return err("prod_api.runs", _NOT_CONFIGURED, configured=False)
    url = f"{base}/api/jobs"
    try:
        data = get_json(url, timeout=target.timeout)
    except Exception as exc:  # noqa: BLE001
        return err("prod_api.runs", f"GET {url} failed: {exc}")
    jobs = data.get("jobs", []) if isinstance(data, dict) else []
    newest_first = sorted(jobs, key=lambda job: job.get("created_at") or "", reverse=True)
    runs = [{key: job.get(key) for key in _RUN_FIELDS} for job in newest_first[: max(limit, 0)]]
    return ok(
        "prod_api.runs",
        {
            "path": data.get("path") if isinstance(data, dict) else None,
            "count": len(runs),
            "runs": runs,
        },
    )
