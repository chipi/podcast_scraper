#!/usr/bin/env python3
"""Sync Grafana dashboards + alert rules to each tenant's account (ADR-117).

Config-as-code → upload. Reads ``config/obs/tenants.yaml`` and, per tenant, pushes
``config/grafana/dashboards/<tenant>/*.json`` and ``config/grafana/alerts/<tenant>/*.yaml``
to that tenant's Grafana account. The account's endpoint + token come from env (never
committed): ``GRAFANA_<ACCOUNT>_URL`` / ``GRAFANA_<ACCOUNT>_TOKEN`` (account upper-cased,
``-`` → ``_``); the alert-rule ``${LOKI_UID}`` placeholder from ``GRAFANA_<ACCOUNT>_LOKI_UID``.

Default is a DRY RUN (prints what it would POST); pass ``--apply`` to actually upload.
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


def _acct_env(account: str, suffix: str) -> str:
    return f"GRAFANA_{account.upper().replace('-', '_')}_{suffix}"


def _post(url: str, token: str, path: str, payload: dict, *, apply: bool) -> None:
    body = json.dumps(payload).encode()
    if not apply:
        print(f"  DRY-RUN POST {path} ({len(body)} bytes)")
        return
    req = urllib.request.Request(  # noqa: S310 - operator-configured Grafana URL
        url.rstrip("/") + path,
        data=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            print(f"  POST {path} -> {r.status}")
    except urllib.error.HTTPError as e:
        # 409/412 (already exists / version) is fine for an idempotent upsert.
        print(f"  POST {path} -> {e.code} {e.read().decode(errors='replace')[:200]}")


def _sync_tenant(name: str, cfg: dict, *, apply: bool) -> None:
    if cfg.get("enabled") is False:
        print(f"[{name}] disabled — skipping")
        return
    gcfg = cfg.get("grafana") or {}
    account = gcfg.get("account", "common")
    folder = gcfg.get("folder", name)
    url = os.environ.get(_acct_env(account, "URL"), "")
    token = os.environ.get(_acct_env(account, "TOKEN"), "")
    loki_uid = os.environ.get(_acct_env(account, "LOKI_UID"), "${LOKI_UID}")
    if apply and not (url and token):
        print(f"[{name}] missing {_acct_env(account, 'URL')}/TOKEN — skipping", file=sys.stderr)
        return

    print(f"[{name}] account={account} folder={folder!r} url={'set' if url else '(dry)'}")
    _post(url, token, "/api/folders", {"title": folder}, apply=apply)

    dash_dir = REPO / "config" / "grafana" / "dashboards" / name
    for f in sorted(dash_dir.glob("*.json")):
        try:
            dash = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  SKIP {f.name}: invalid JSON ({e})", file=sys.stderr)
            continue
        _post(url, token, "/api/dashboards/db", {"dashboard": dash, "overwrite": True}, apply=apply)

    alert_dir = REPO / "config" / "grafana" / "alerts" / name
    for f in sorted(alert_dir.glob("*.yaml")):
        raw = f.read_text().replace("${LOKI_UID}", loki_uid)
        doc = yaml.safe_load(raw) or {}
        for group in doc.get("groups", []):
            for rule in group.get("rules", []):
                _post(url, token, "/api/v1/provisioning/alert-rules", rule, apply=apply)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="actually POST (default: dry run)")
    ap.add_argument("--tenant", help="sync only this tenant")
    args = ap.parse_args()

    tenants: dict[str, Any] = (yaml.safe_load(TENANTS.read_text()) or {}).get("tenants", {})
    if not tenants:
        print("no tenants in config/obs/tenants.yaml", file=sys.stderr)
        return 1
    for name, cfg in tenants.items():
        if args.tenant and name != args.tenant:
            continue
        _sync_tenant(name, cfg or {}, apply=args.apply)
    print("DRY RUN — nothing uploaded (pass --apply)" if not args.apply else "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
