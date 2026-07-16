#!/usr/bin/env python3
"""Sync Sentry issue-alert rules to each tenant's Sentry project (ADR-117).

Sentry is per-account by DSN (§3a): each tenant's rules live in its own org/project. Reads
``config/obs/tenants.yaml`` + ``config/sentry/<tenant>/alerts.yaml`` and posts rules via the
Sentry API. Account creds from env: ``SENTRY_<ACCOUNT>_ORG`` / ``SENTRY_<ACCOUNT>_TOKEN``,
a per-tenant ``SENTRY_<ACCOUNT>_PROJECT``, and an optional ``SENTRY_<ACCOUNT>_URL`` base
(default ``https://sentry.io``).

Default is a DRY RUN; pass ``--apply`` to upload.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
TENANTS = REPO / "config" / "obs" / "tenants.yaml"


def _acct(account: str, suffix: str) -> str:
    return f"SENTRY_{account.upper().replace('-', '_')}_{suffix}"


def _post(base_url: str, token: str, org: str, project: str, rule: dict, *, apply: bool) -> None:
    path = f"/api/0/projects/{org}/{project}/rules/"
    name = rule.get("name", "<unnamed>")
    if not apply:
        print(f"  DRY-RUN POST {path} :: {name}")
        return
    req = urllib.request.Request(  # noqa: S310 - operator-configured Sentry URL
        base_url.rstrip("/") + path,
        data=json.dumps(rule).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            print(f"  POST {path} :: {name} -> {r.status}")
    except urllib.error.HTTPError as e:
        # 400 (rule already exists) is fine for an idempotent re-run.
        print(f"  POST {path} :: {name} -> {e.code} {e.read().decode(errors='replace')[:200]}")


def _sync_tenant(name: str, cfg: dict, *, apply: bool) -> None:
    if cfg.get("enabled") is False:
        print(f"[{name}] disabled — skipping")
        return
    account = (cfg.get("sentry") or {}).get("account", "common")
    base_url = os.environ.get(_acct(account, "URL"), "https://sentry.io")
    org = os.environ.get(_acct(account, "ORG"), "")
    token = os.environ.get(_acct(account, "TOKEN"), "")
    project = os.environ.get(_acct(account, "PROJECT"), name)
    alerts = REPO / "config" / "sentry" / name / "alerts.yaml"
    if not alerts.exists():
        print(f"[{name}] no config/sentry/{name}/alerts.yaml — skipping")
        return
    rules = (yaml.safe_load(alerts.read_text()) or {}).get("rules", [])
    print(
        f"[{name}] account={account} org={org or '(dry)'} project={project}: {len(rules)} rule(s)"
    )
    if apply and not (org and token):
        print(f"[{name}] missing {_acct(account, 'ORG')}/TOKEN — skipping", file=sys.stderr)
        return
    for rule in rules:
        _post(base_url, token, org, project, rule, apply=apply)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--tenant")
    args = ap.parse_args()
    tenants: dict[str, Any] = (yaml.safe_load(TENANTS.read_text()) or {}).get("tenants", {})
    for name, cfg in tenants.items():
        if args.tenant and name != args.tenant:
            continue
        _sync_tenant(name, cfg or {}, apply=args.apply)
    print("DRY RUN — nothing uploaded (pass --apply)" if not args.apply else "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
