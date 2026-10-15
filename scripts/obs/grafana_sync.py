#!/usr/bin/env python3
"""Sync Grafana dashboards to each tenant's account (ADR-117).

Config-as-code → upload. Reads ``config/obs/tenants.yaml`` and, per tenant, pushes
``config/grafana/dashboards/<tenant>/*.json`` into that tenant's ``folder`` in its Grafana
account. The account's endpoint + token come from env (never committed):
``GRAFANA_<ACCOUNT>_URL`` / ``GRAFANA_<ACCOUNT>_TOKEN`` (account upper-cased, ``-`` → ``_``).

Alert rules are not synced from here: since 2026-10-15 agentic-ai-homelab's Grafana alerting
``rules.yaml`` owns them (with a drift check), so a podcast alert is added there.

Default is a DRY RUN. With the account's URL + token set it reads the live Grafana and prints,
per dashboard, NEW / CHANGED / MOVE (same content, other folder) / UNCHANGED, without writing
anything. Without them it prints what it would POST. Pass ``--apply`` to actually upload.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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


def _slug(text: str) -> str:
    """A stable Grafana folder UID from a title (alphanumeric + hyphens, <=40)."""
    out = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return out[:40] or "folder"


def _api(url: str, token: str, method: str, path: str, payload: dict, *, apply: bool) -> int:
    body = json.dumps(payload).encode()
    if not apply:
        print(f"  DRY-RUN {method} {path} ({len(body)} bytes)")
        return 0
    req = urllib.request.Request(  # noqa: S310 - operator-configured Grafana URL
        url.rstrip("/") + path,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            # Let a subsequent sync overwrite an API-provisioned rule instead of it being locked.
            "X-Disable-Provenance": "true",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            print(f"  {method} {path} -> {r.status}")
            return int(r.status)
    except urllib.error.HTTPError as e:
        # 409/412 (already exists / version) is fine for an idempotent upsert.
        print(f"  {method} {path} -> {e.code} {e.read().decode(errors='replace')[:200]}")
        return int(e.code)


def _post(url: str, token: str, path: str, payload: dict, *, apply: bool) -> int:
    return _api(url, token, "POST", path, payload, apply=apply)


def _get(url: str, token: str, path: str) -> Any:
    """GET *path* from the live Grafana; ``None`` when it answers 404."""
    req = urllib.request.Request(  # noqa: S310 - operator-configured Grafana URL
        url.rstrip("/") + path, headers={"Authorization": f"Bearer {token}"}, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            return json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


_VOLATILE = {"id", "version", "iteration"}


def _same(a: dict, b: dict) -> bool:
    def norm(d: dict) -> str:
        return json.dumps({k: v for k, v in d.items() if k not in _VOLATILE}, sort_keys=True)

    return norm(a) == norm(b)


def _dashboard_state(url: str, token: str, dash: dict, folder_uid: str) -> str:
    live = _get(url, token, f"/api/dashboards/uid/{dash.get('uid')}")
    if not live:
        return "NEW      "
    if not _same(dash, live.get("dashboard") or {}):
        return "CHANGED  "
    if (live.get("meta") or {}).get("folderUid", "") != folder_uid:
        return "MOVE     "
    return "UNCHANGED"


def _sync_tenant(name: str, cfg: dict, *, apply: bool) -> None:
    if cfg.get("enabled") is False:
        print(f"[{name}] disabled — skipping")
        return
    gcfg = cfg.get("grafana") or {}
    account = gcfg.get("account", "common")
    folder = gcfg.get("folder", name)
    url = os.environ.get(_acct_env(account, "URL"), "")
    token = os.environ.get(_acct_env(account, "TOKEN"), "")
    if apply and not (url and token):
        print(f"[{name}] missing {_acct_env(account, 'URL')}/TOKEN — skipping", file=sys.stderr)
        return
    live = bool(url and token)

    # Deterministic folder UID: create the folder with a slug-derived uid so we KNOW it (a plain
    # ``{title}`` POST returns a random uid we'd have to GET back, and 409s on re-sync).
    folder_uid = _slug(folder)
    where = "set" if url else "(dry)"
    print(f"[{name}] account={account} folder={folder!r} uid={folder_uid} url={where}")
    dash_files = sorted((REPO / "config" / "grafana" / "dashboards" / name).glob("*.json"))
    if dash_files and (apply or not live):  # no empty folder for a tenant with no dashboards
        _post(url, token, "/api/folders", {"uid": folder_uid, "title": folder}, apply=apply)

    for f in dash_files:
        try:
            dash = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            print(f"  SKIP {f.name}: invalid JSON ({e})", file=sys.stderr)
            continue
        if live and not apply:
            print(f"  {_dashboard_state(url, token, dash, folder_uid)} {f.name}")
            continue
        _post(
            url,
            token,
            "/api/dashboards/db",
            {"dashboard": {**dash, "id": None}, "folderUid": folder_uid, "overwrite": True},
            apply=apply,
        )


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
