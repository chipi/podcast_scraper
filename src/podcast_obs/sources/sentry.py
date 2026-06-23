"""Recent unresolved Sentry issues for the deploy's environment (Sentry API, bearer auth).

Complements :func:`podcast_obs.sources.loki.recent_logs`: Sentry holds SDK-captured exceptions;
Loki holds everything the containers logged. Use both for a full error picture.
"""

from __future__ import annotations

from typing import Optional

from .._http import get_json
from ..config import TargetConfig
from ..result import err, ok

_SOURCE = "sentry.errors"
_API = "https://sentry.io/api/0"


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
    projects: list[dict] = []
    total = 0
    for project in target.sentry_projects:
        url = f"{_API}/projects/{target.sentry_org}/{project}/issues/"
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
