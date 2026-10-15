"""Recent prod deploys from the GitHub Actions API, across every prod deploy workflow.

Needs a read-only token (fine-grained, Actions:read) — the control plane never triggers runs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from .._http import get_json
from ..config import TargetConfig
from ..result import err, ok

_SOURCE = "github.deploys"
_API = "https://api.github.com"
# Every workflow that changes prod. deploy-all-prod calls deploy-prod/-player/-operator as
# reusable workflows; those appear only as the caller's run, so nothing is counted twice.
_WORKFLOWS = (
    "deploy-all-prod.yml",
    "deploy-prod.yml",
    "deploy-player.yml",
    "deploy-operator.yml",
    "deploy-config.yml",
    "deploy-litellm.yml",
    "deploy-vps-observability-endpoints.yml",
)
_HEADERS_BASE = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def _duration_s(started: Optional[str], ended: Optional[str]) -> Optional[float]:
    if not started or not ended:
        return None
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended.replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((end - start).total_seconds(), 1)


def _deploy(workflow: str, run: dict) -> dict:
    return {
        "workflow": workflow,
        "run_number": run.get("run_number"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "sha": (run.get("head_sha") or "")[:7],
        "actor": (run.get("actor") or {}).get("login"),
        "event": run.get("event"),
        "created_at": run.get("created_at"),
        # Only meaningful once the run has concluded — updated_at moves on re-runs/annotations,
        # and is "started → now-ish" for in-progress runs.
        "duration_s": (
            _duration_s(run.get("run_started_at"), run.get("updated_at"))
            if run.get("conclusion")
            else None
        ),
        "url": run.get("html_url"),
    }


def recent_deploys(target: TargetConfig, limit: int = 10) -> dict:
    """The last *limit* prod deploy runs across ``_WORKFLOWS``, newest first, + failure rate.

    A workflow whose runs can't be read is listed in ``errors``; only all of them failing is
    an error result.
    """
    if not target.github_token:
        return err(
            _SOURCE,
            "github token not set (PODCAST_OBS_GITHUB_TOKEN; needs Actions:read)",
            configured=False,
        )
    if not target.github_repo:
        return err(_SOURCE, "github repo not set (PODCAST_OBS_GITHUB_REPO)", configured=False)
    headers = {**_HEADERS_BASE, "Authorization": f"Bearer {target.github_token}"}
    deploys: list[dict] = []
    errors: list[str] = []
    for workflow in _WORKFLOWS:
        url = f"{_API}/repos/{target.github_repo}/actions/workflows/{workflow}/runs"
        try:
            data = get_json(
                url, headers=headers, params={"per_page": max(limit, 1)}, timeout=target.timeout
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"GET {url} failed: {exc}")
            continue
        runs = data.get("workflow_runs", []) if isinstance(data, dict) else []
        deploys.extend(_deploy(workflow, run) for run in runs)
    if len(errors) == len(_WORKFLOWS):
        return err(_SOURCE, "; ".join(errors))
    deploys.sort(key=lambda d: d["created_at"] or "", reverse=True)
    deploys = deploys[: max(limit, 0)]
    concluded = [d for d in deploys if d["conclusion"]]
    failed = [d for d in concluded if d["conclusion"] != "success"]
    failure_rate = round(len(failed) / len(concluded), 3) if concluded else None
    data_out: dict = {"count": len(deploys), "failure_rate": failure_rate, "deploys": deploys}
    if errors:
        data_out["errors"] = errors
    return ok(_SOURCE, data_out)
