#!/usr/bin/env python3
"""Drop stale Tailscale devices tagged ``tag:gha-deployer`` (GitHub Actions runners).

Ephemeral CI nodes sometimes remain listed after jobs end. This script lists
them and optionally DELETEs via the Tailscale API (same pattern as
``.github/workflows/drill-infra-destroy.yml``).

Env (required):

  TS_API_KEY   — Tailscale API key (``tskey-api-...``, not TS_AUTHKEY).
  TAILNET_NAME or TAILNET — tailnet id (e.g. ``tail6d0ed4.ts.net`` or org DNS name).

Examples:

  # List only (default)
  TS_API_KEY=... TAILNET_NAME=tail-xxxx.ts.net \\
    python3 scripts/ops/cleanup_tailscale_gha_deployers.py

  # Remove nodes idle for at least 3 hours (never touches nodes newer than cutoff)
  TS_API_KEY=... TAILNET_NAME=tail-xxxx.ts.net \\
    python3 scripts/ops/cleanup_tailscale_gha_deployers.py --apply --min-hours 3
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
from urllib.parse import quote

TAG = "tag:gha-deployer"
API_BASE = "https://api.tailscale.com/api/v2"


@dataclass(frozen=True)
class Candidate:
    device_id: str
    hostname: str
    name: str
    last_seen: str | None
    idle_hours: float | None
    skip_reason: str | None


def _tailnet() -> str:
    t = (os.environ.get("TAILNET_NAME") or os.environ.get("TAILNET") or "").strip()
    if not t:
        print("Set TAILNET_NAME or TAILNET (e.g. tail6d0ed4.ts.net).", file=sys.stderr)
        sys.exit(1)
    return t


def _api_key() -> str:
    k = (os.environ.get("TS_API_KEY") or "").strip()
    if not k:
        print("Set TS_API_KEY (Tailscale API key, not device TS_AUTHKEY).", file=sys.stderr)
        sys.exit(1)
    return k


def _http_json(method: str, url: str, api_key: str) -> Any:
    req = urllib.request.Request(
        url,
        method=method,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code} {method} {url}\n{detail}", file=sys.stderr)
        raise
    if not body.strip():
        return None
    return json.loads(body)


def _parse_last_seen(raw: str | None) -> datetime | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip().replace("Z", "+00:00")
    # API may return fractional seconds; fromiso8601 handles .ffffff
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _devices_payload(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("devices"), list):
        return data["devices"]
    if isinstance(data, list):
        return data
    return []


def _iter_candidates(
    devices: Iterable[dict[str, Any]],
    min_idle: timedelta,
    now: datetime,
) -> list[Candidate]:
    out: list[Candidate] = []
    for d in devices:
        did = str(d.get("id", "")).strip()
        if not did:
            continue
        hostname = str(d.get("hostname", "") or d.get("name", "") or "")
        name = str(d.get("name", "") or "")
        tags = d.get("tags") or []
        last_raw = d.get("lastSeen") or d.get("last_seen")
        last_raw_s = str(last_raw).strip() if last_raw is not None else None
        ls_dt = _parse_last_seen(last_raw_s)
        skip: str | None = None
        idle_h: float | None = None
        if d.get("online") is True:
            skip = "online=true (still active)"
        elif ls_dt is None:
            skip = "no lastSeen (use --allow-unknown-last-seen to delete anyway)"
        else:
            idle = now - ls_dt
            idle_h = idle.total_seconds() / 3600.0
            if idle < min_idle:
                skip = f"idle {idle_h:.2f}h < min {min_idle.total_seconds() / 3600.0:.2f}h"
        
        # DEBUG: Add tags to skip reason so we can see them
        full_skip = f"{skip or ''} tags={tags}".strip()
        
        out.append(
            Candidate(
                device_id=did,
                hostname=hostname,
                name=name,
                last_seen=last_raw_s,
                idle_hours=idle_h,
                skip_reason=full_skip,
            )
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--apply",
        action="store_true",
        help="DELETE matching devices (default is list-only).",
    )
    p.add_argument(
        "--min-hours",
        type=float,
        default=2.0,
        metavar="H",
        help="Only delete when lastSeen is at least this many hours ago (default: 2).",
    )
    p.add_argument(
        "--allow-unknown-last-seen",
        action="store_true",
        help="Allow deleting tag:gha-deployer devices with no parseable lastSeen (risky).",
    )
    args = p.parse_args()

    tailnet = _tailnet()
    api_key = _api_key()
    now = datetime.now(timezone.utc)
    min_idle = timedelta(hours=float(args.min_hours))

    tail_enc = quote(tailnet, safe="")
    url = f"{API_BASE}/tailnet/{tail_enc}/devices?fields=all"
    data = _http_json("GET", url, api_key)
    devices = _devices_payload(data)
    candidates = _iter_candidates(devices, min_idle=min_idle, now=now)

    if not candidates:
        print("No devices with tag:gha-deployer.")
        return

    print(f"tailnet={tailnet}  tag={TAG}  candidates={len(candidates)}")
    to_delete: list[Candidate] = []
    for c in candidates:
        reason = f"DEBUG_SKIP {c.skip_reason}"
        if reason and "no lastSeen" in reason and args.allow_unknown_last_seen:
            reason = None
        if reason:
            print(f"SKIP id={c.device_id} host={c.hostname!r} name={c.name!r} {reason}")
        else:
            idle = f"idle_h={c.idle_hours:.2f}" if c.idle_hours is not None else "idle_h=?"
            print(
                f"DROP id={c.device_id} host={c.hostname!r} name={c.name!r} "
                f"lastSeen={c.last_seen!r} {idle}"
            )
            to_delete.append(c)

    if not args.apply:
        print(f"\nDry run. {len(to_delete)} would be deleted. Re-run with --apply to delete.")
        return

    if not to_delete:
        print("Nothing to delete (all skipped).")
        return

    for c in to_delete:
        del_url = f"{API_BASE}/device/{quote(c.device_id, safe='')}"
        print(f"DELETE {del_url}")
        _http_json("DELETE", del_url, api_key)
    print(f"Deleted {len(to_delete)} device(s).")


if __name__ == "__main__":
    main()
