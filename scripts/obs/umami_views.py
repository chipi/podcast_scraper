#!/usr/bin/env python3
"""Create the player's beta-usage Umami views as code (ADR-126, slice 5 of epic #2263).

Why as code
───────────
A report clicked together in the Umami UI exists in exactly one place and is described nowhere. The
beta's numbers are supposed to be reviewable — "which events feed Discovery share?" has to be
answerable from the repo, not by opening a dashboard and reading a funnel's step list. This is the
same argument ADR-117 makes for Grafana dashboards, applied to the other analytics surface.

Idempotent by NAME per website, so re-running updates rather than duplicating. Umami happily stores
two reports with the same name, and a silently-duplicated funnel is worse than a missing one: both
versions look authoritative and they drift.

The step vocabulary is the typed registry in `web/learning-player/src/services/analytics.ts`. Every
`value` below must be a member of `EVENT_NAMES` — `--check-registry` asserts exactly that, because a
funnel step naming an event the app never emits renders as a clean 0% conversion rather than as an
error, which is indistinguishable from a product nobody uses.

Auth
────
Umami reading and writing needs admin auth. Supply either a token or a username/password pair, which
is exchanged via `/api/auth/login`:

    PODCAST_OBS_UMAMI_URL=http://127.0.0.1:3001 \\
    PODCAST_OBS_UMAMI_USERNAME=… PODCAST_OBS_UMAMI_PASSWORD=… \\
      python scripts/obs/umami_views.py --website-id <uuid> --apply

Dry run by default, matching `make obs-sync`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "web" / "learning-player" / "src" / "services" / "analytics.ts"

# The beta window. Deliberately explicit rather than "last 30 days": the report is a record of a
# specific beta, and a rolling window silently changes what a reviewer is looking at.
START = "2026-09-01T00:00:00.000Z"
END = "2026-12-31T23:59:59.999Z"


def _funnel(name: str, description: str, steps: list[str]) -> dict[str, Any]:
    return {
        "type": "funnel",
        "name": name,
        "description": description,
        "parameters": {
            "steps": [{"type": "event", "value": s} for s in steps],
            # Days a visitor has to complete the funnel. 30 matches the operator's existing orrery
            # funnels, so the two products' numbers mean the same thing.
            "window": 30,
            "startDate": START,
            "endDate": END,
        },
    }


def _goal(name: str, description: str, event: str) -> dict[str, Any]:
    return {
        "type": "goal",
        "name": name,
        "description": description,
        "parameters": {"type": "event", "value": event, "startDate": START, "endDate": END},
    }


#: The six views. Two funnels answer "do people get through", four goals answer "did the thing that
#: matters happen at all" — which is the split the beta spec draws.
VIEWS: list[dict[str, Any]] = [
    _funnel(
        "Onboarding — landing to first play",
        "The spec's headline funnel. Each step is a separate event precisely so the drop-off is "
        "attributable: a listener who saw the landing but never set off to sign in is a different "
        "problem from one who signed in and never pressed play.",
        ["landing_view", "auth_started", "auth_completed", "interests_saved", "play_start"],
    ),
    _funnel(
        "Discovery — pivot to a real listen",
        "The question the beta exists to answer: do people MOVE across the corpus and then actually"
        "listen to what they found? `entity_open` is the pivot, `episode_open` is the commitment, "
        "`play_start` is the listen. A wide gap between the first two means the cards look "
        "interesting and read badly.",
        ["entity_open", "episode_open", "play_start"],
    ),
    _goal(
        "First play",
        "Pressing play at all. The single most important binary in the beta: everything upstream is"
        "only interesting if it ends here.",
        "play_start",
    ),
    _goal(
        "Followed a show",
        "A feed subscription — the strongest voluntary signal a listener gives, because it is a "
        "commitment to future episodes rather than a reaction to this one.",
        "follow",
    ),
    _goal(
        "Made a capture",
        "A highlight or a note. The product's thesis is that people want to REMEMBER what they "
        "heard; this is the event that either supports that or does not.",
        "capture_created",
    ),
    _goal(
        "Shared something",
        "The only event that reaches someone who is not a participant, so it is both an engagement "
        "signal and the beta's only organic growth channel.",
        "share",
    ),
]


def _registry_names() -> set[str]:
    src = REGISTRY.read_text()
    block = re.search(r"export const EVENT_NAMES = \[(.*?)\] as const", src, re.S)
    if not block:
        raise SystemExit(f"could not find EVENT_NAMES in {REGISTRY}")
    return set(re.findall(r"'([a-z0-9_]+)'", block.group(1)))


def _check_registry() -> int:
    known = _registry_names()
    bad: list[str] = []
    for view in VIEWS:
        params = view["parameters"]
        values = (
            [s["value"] for s in params["steps"]] if view["type"] == "funnel" else [params["value"]]
        )
        for v in values:
            if v not in known:
                bad.append(f"{view['name']!r} references {v!r}, which is not in EVENT_NAMES")
    for line in bad:
        print(f"  MISMATCH {line}", file=sys.stderr)
    if bad:
        print(
            "\nA view naming an event the app never emits renders as a clean 0% conversion, not as "
            "an error — indistinguishable from a product nobody used.",
            file=sys.stderr,
        )
        return 1
    print(f"all {len(VIEWS)} view(s) reference only registry events ({len(known)} known)")
    return 0


class Umami:
    def __init__(self, base: str, token: str) -> None:
        self.base = base.rstrip("/")
        self.token = token

    @classmethod
    def from_env(cls) -> "Umami":
        base = os.environ.get("PODCAST_OBS_UMAMI_URL", "http://127.0.0.1:3001")
        token = os.environ.get("PODCAST_OBS_UMAMI_TOKEN", "")
        if not token:
            user = os.environ.get("PODCAST_OBS_UMAMI_USERNAME", "")
            pw = os.environ.get("PODCAST_OBS_UMAMI_PASSWORD", "")
            if not user or not pw:
                raise SystemExit(
                    "need PODCAST_OBS_UMAMI_TOKEN, or PODCAST_OBS_UMAMI_USERNAME + _PASSWORD"
                )
            body = json.dumps({"username": user, "password": pw}).encode()
            req = urllib.request.Request(
                f"{base.rstrip('/')}/api/auth/login",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                token = json.loads(resp.read().decode())["token"]
        return cls(base, token)

    def _call(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else None

    def reports(self, website_id: str) -> list[dict[str, Any]]:
        q = urllib.parse.urlencode({"websiteId": website_id, "pageSize": 200})
        out = self._call("GET", f"/api/reports?{q}")
        return (out or {}).get("data", []) if isinstance(out, dict) else (out or [])

    def create(self, website_id: str, view: dict[str, Any]) -> Any:
        return self._call(
            "POST",
            "/api/reports",
            {
                "websiteId": website_id,
                "type": view["type"],
                "name": view["name"],
                "description": view["description"],
                "parameters": view["parameters"],
            },
        )

    def run_report(self, website_id: str, view: dict[str, Any]) -> Any:
        """Execute a view and return its rows.

        The report endpoints take `{type, websiteId, filters, parameters}` — `parameters` NESTED,
        not
        spread at the top level, and `filters` required even when empty. Getting either wrong
        answers
        400 with a schema complaint rather than an empty result, which is the one helpful thing
        about
        this API: a malformed view cannot masquerade as a view with no data.
        """
        return self._call(
            "POST",
            f"/api/reports/{view['type']}",
            {
                "type": view["type"],
                "websiteId": website_id,
                "filters": {},
                "parameters": view["parameters"],
            },
        )

    def update(self, report_id: str, website_id: str, view: dict[str, Any]) -> Any:
        return self._call(
            "POST",
            f"/api/reports/{report_id}",
            {
                "websiteId": website_id,
                "type": view["type"],
                "name": view["name"],
                "description": view["description"],
                "parameters": view["parameters"],
            },
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--website-id", help="the Umami website uuid to attach the views to")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="execute every view and report whether it returns data. Creating a view proves it was "
        "stored; it does not prove it will ever show a number, and the whole point of this arc is "
        "that an empty analytics surface looks exactly like a product nobody used.",
    )
    ap.add_argument(
        "--check-registry",
        action="store_true",
        help="only verify every referenced event exists in EVENT_NAMES, then exit",
    )
    args = ap.parse_args()

    if args.check_registry:
        return _check_registry()
    if _check_registry() != 0:
        return 1
    if not args.website_id:
        print("--website-id is required (or pass --check-registry)", file=sys.stderr)
        return 2

    umami = Umami.from_env()

    if args.verify:
        empty: list[str] = []
        for view in VIEWS:
            try:
                rows = umami.run_report(args.website_id, view)
            except urllib.error.HTTPError as exc:
                print(f"  [FAIL ] {view['name']}: HTTP {exc.code} {exc.read().decode()[:160]}")
                empty.append(view["name"])
                continue
            # TYPE-AWARE, and this cost a wrong answer to learn.
            #
            # A goal answers `{"num": <achievements>, "total": <denominator>}`. `total` is the
            # SESSION COUNT in the window, not the goal count, and it is IDENTICAL for every goal on
            # a site. Summing it reported four goals that had never fired once on prod as healthy,
            # with the plausible number 32 — measured by asking for a goal named
            # `zzz_definitely_not_an_event`, which answered the same `{"num":0,"total":32}`.
            #
            # That is precisely the failure this whole arc exists to prevent, produced by the tool
            # written to catch it. `num` is the only field that says whether the goal happened.
            #
            # A funnel answers one row per step; entering it at all is the first step's `visitors`.
            rows = rows if isinstance(rows, list) else [rows]
            hits = 0.0
            denom: float | None = None
            if view["type"] == "goal":
                first = rows[0] if rows and isinstance(rows[0], dict) else {}
                hits = float(first.get("num") or 0)
                denom = float(first.get("total") or 0)
            else:
                first = rows[0] if rows and isinstance(rows[0], dict) else {}
                hits = float(first.get("visitors") or 0)

            tag = "OK   " if hits else "EMPTY"
            if not hits:
                empty.append(view["name"])
            print(f"  [{tag}] [{view['type']:6s}] {view['name']}")
            detail = f"rows={len(rows)} hits={hits:g}"
            if denom is not None:
                detail += f" of {denom:g} sessions"
            print(f"           {detail}")
        print()
        if empty:
            print(f"{len(empty)} of {len(VIEWS)} view(s) returned nothing: {', '.join(empty)}")
            return 1
        print(f"all {len(VIEWS)} view(s) returned data")
        return 0

    existing = {r.get("name"): r for r in umami.reports(args.website_id)}
    print(f"website {args.website_id}: {len(existing)} existing report(s)\n")

    for view in VIEWS:
        found = existing.get(view["name"])
        action = "UPDATE" if found else "CREATE"
        print(f"  {'' if args.apply else 'DRY-RUN '}{action} [{view['type']:6s}] {view['name']}")
        if not args.apply:
            continue
        try:
            if found:
                umami.update(str(found.get("id") or found.get("reportId")), args.website_id, view)
            else:
                umami.create(args.website_id, view)
        except urllib.error.HTTPError as exc:
            print(f"    FAILED HTTP {exc.code}: {exc.read().decode()[:200]}", file=sys.stderr)
            return 1

    print("\nDRY RUN — nothing written (pass --apply)" if not args.apply else "\ndone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
