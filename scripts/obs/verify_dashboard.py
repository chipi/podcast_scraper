#!/usr/bin/env python3
"""Verify a published Grafana dashboard actually RENDERS DATA, panel by panel (ADR-117, #2268).

Why this exists
───────────────
``make obs-sync`` proves a dashboard was *uploaded*. It cannot prove any panel on it will show
anything, and those are very different claims. The analytics arc this was written for turned on
exactly that gap: three log streams read empty for weeks because an Alloy ``local.file_match`` glob
named the wrong directory *and* the wrong filename. Nothing errored. The dashboards were present,
correct, and blank — and a blank analytics panel is ambiguous in the worst possible direction,
because it looks identical to a product nobody used.

So this reads the dashboard back OUT of Grafana (checking what was stored, not the local file),
substitutes the template variable, and runs every panel's query through ``/api/ds/query`` — the same
path the browser takes. A panel that returns no frames here is a panel that will look empty to the
operator, whatever the underlying store contains.

It is deliberately usable as a GATE: exit 1 when any panel comes back empty.

Usage
─────
    GRAFANA_URL=http://127.0.0.1:3000 GRAFANA_TOKEN=glsa_… \\
      python scripts/obs/verify_dashboard.py --uid podcast-player-beta-usage --var instance=dev-local

    # Expect-empty is a legitimate assertion too: run it against prod to DEMONSTRATE a broken
    # shipper rather than describe one.
    … --var instance=prod-podcast --expect-empty

Exit codes: 0 = every panel returned data (or every panel was empty, with --expect-empty); 1 = not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def _api(
    base: str, token: str, path: str, payload: dict[str, Any] | None = None, method: str = "GET"
) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{base.rstrip('/')}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _frame_rows(result: dict[str, Any]) -> tuple[int, list[Any]]:
    """Total row count across the result's frames, plus a short sample of the last value column."""
    total = 0
    sample: list[Any] = []
    for frame in result.get("frames", []) or []:
        values = frame.get("data", {}).get("values", []) or []
        if not values:
            continue
        total += len(values[0])
        for column in values[1:]:
            sample = list(column[:6])
    return total, sample


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uid", required=True, help="dashboard uid")
    ap.add_argument(
        "--var",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="substitute a template variable, e.g. instance=prod-podcast (repeatable)",
    )
    ap.add_argument("--from", dest="time_from", default="now-14d")
    ap.add_argument("--to", dest="time_to", default="now")
    ap.add_argument(
        "--expect-empty",
        action="store_true",
        help="invert the gate: succeed only when EVERY panel is empty. Use to demonstrate a broken "
        "shipper as a measurement rather than a claim.",
    )
    args = ap.parse_args()

    base = os.environ.get("GRAFANA_URL", "")
    token = os.environ.get("GRAFANA_TOKEN", "")
    if not base or not token:
        print("GRAFANA_URL and GRAFANA_TOKEN must be set", file=sys.stderr)
        return 2

    substitutions = dict(v.split("=", 1) for v in args.var)

    try:
        dashboard = _api(base, token, f"/api/dashboards/uid/{args.uid}")["dashboard"]
    except urllib.error.HTTPError as exc:
        print(f"could not read dashboard {args.uid!r}: HTTP {exc.code}", file=sys.stderr)
        return 2

    panels = dashboard.get("panels", []) or []
    print(f"dashboard {dashboard.get('title')!r} — {len(panels)} panel(s), vars={substitutions}\n")

    non_empty = 0
    failures: list[str] = []
    for panel in panels:
        targets = panel.get("targets") or []
        if not targets:
            continue
        target = targets[0]
        expr = str(target.get("expr", ""))
        for name, value in substitutions.items():
            expr = expr.replace(f"${name}", value).replace(f"${{{name}}}", value)

        body = {
            "from": args.time_from,
            "to": args.time_to,
            "queries": [
                {
                    "refId": "A",
                    "datasource": target.get("datasource"),
                    "queryType": target.get("queryType"),
                    "expr": expr,
                    "maxDataPoints": 200,
                    "intervalMs": 3_600_000,
                }
            ],
        }
        title = str(panel.get("title", ""))[:56]
        try:
            result = _api(base, token, "/api/ds/query", body, method="POST")["results"]["A"]
        except Exception as exc:  # noqa: BLE001 — any failure is a failed panel, and we report it
            failures.append(f"panel {panel.get('id')} {title!r}: {exc}")
            print(f"[FAIL ] panel {panel.get('id')} {title!r}: {exc}")
            continue

        rows, sample = _frame_rows(result)
        error = result.get("error")
        if rows and not error:
            non_empty += 1
        tag = "OK   " if (rows and not error) else "EMPTY"
        print(f"[{tag}] panel {panel.get('id')} {title!r}")
        print(f"         rows={rows} sample={sample} err={error}")

    print()
    if args.expect_empty:
        good = non_empty == 0 and not failures
        print(
            "EVERY PANEL EMPTY, as expected — the stream genuinely carries nothing"
            if good
            else f"{non_empty} panel(s) returned data, but --expect-empty was given"
        )
        return 0 if good else 1

    good = non_empty == len(panels) and not failures
    print(
        "ALL PANELS RETURNED DATA"
        if good
        else f"{len(panels) - non_empty} of {len(panels)} panel(s) returned nothing"
    )
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
