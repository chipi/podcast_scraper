"""Recent unresolved error issues for the deploy's environment (Sentry-compatible API, bearer auth).

Complements :func:`podcast_obs.sources.victoria.recent_logs`: this source holds SDK-captured
exceptions; logs hold everything the containers logged. Use both for a full error picture.

Backend is Sentry-API-compatible: point ``sentry_url`` at self-hosted **GlitchTip**
(e.g. ``http://homelab:8090``) for the current stack, or leave it unset for Sentry SaaS.
"""

from __future__ import annotations

from typing import Optional

from .._http import get_json
from ..config import TargetConfig
from ..result import err, ok

_SOURCE = "sentry.errors"


def _api_base(target: TargetConfig) -> str:
    """``<sentry_url>/api/0`` — self-hosted GlitchTip when set, else Sentry SaaS (gap #3)."""
    base = (target.sentry_url or "https://sentry.io").rstrip("/")
    return f"{base}/api/0"


def recent_errors(
    target: TargetConfig,
    window: str = "24h",
    limit: int = 10,
    *,
    run_id: Optional[str] = None,
) -> dict:
    """Top issues per Sentry project for ``environment=<target.sentry_environment>``.

    With ``run_id`` set (#1053), filters to issues tagged ``run_id:<id>`` (and drops the
    ``is:unresolved`` filter so a run's *full* error picture surfaces for correlation).
    """
    if not target.sentry_token:
        return err(_SOURCE, "sentry token not set (PODCAST_OBS_SENTRY_TOKEN)", configured=False)
    if not target.sentry_org or not target.sentry_projects:
        return err(_SOURCE, "sentry org/projects not set", configured=False)
    headers = {"Authorization": f"Bearer {target.sentry_token}"}
    if run_id:
        # safe charset, but quote so an id with spaces can't split the query
        query = f'environment:{target.sentry_environment} run_id:"{run_id}"'
    else:
        query = f"is:unresolved environment:{target.sentry_environment}"
    api = _api_base(target)
    projects: list[dict] = []
    total = 0
    for project in target.sentry_projects:
        url = f"{api}/projects/{target.sentry_org}/{project}/issues/"
        params = {
            "query": query,
            "statsPeriod": window,
            "limit": max(limit, 1),
        }
        try:
            issues = get_json(url, headers=headers, params=params, timeout=target.timeout)
        except Exception as exc:  # noqa: BLE001
            projects.append({"project": project, "ok": False, "error": str(exc)})
            continue
        items = [
            {
                "title": issue.get("title"),
                "culprit": issue.get("culprit"),
                "level": issue.get("level"),
                "count": issue.get("count"),
                "lastSeen": issue.get("lastSeen"),
                "permalink": issue.get("permalink"),
            }
            for issue in (issues if isinstance(issues, list) else [])
        ]
        total += len(items)
        projects.append({"project": project, "ok": True, "issues": items})
    data = {
        "window": window,
        "environment": target.sentry_environment,
        "total_issues": total,
        "projects": projects,
    }
    # Don't report "live" when every configured project failed (e.g. wrong slugs / missing
    # scope) — that's a misconfiguration, not a healthy zero.
    if projects and not any(p["ok"] for p in projects):
        return err(_SOURCE, "all configured Sentry projects failed (check slugs / token scopes)")
    return ok(_SOURCE, data)
