#!/usr/bin/env python3
"""Seed one test account so a contact-sheet tour photographs a POPULATED app, in seconds.

The contact sheets used to get their data by running two whole device suites (AppJourney +
Personalisation) first: ~80 minutes of tapping to arrive at a few follows, listens and highlights
(2026-10-08). The tour only needs the data to exist, and every piece of it has an API, so this
writes it directly against the e2e api (the mock sign-in it exposes) and the tour runs straight
after.

What it writes, as ``--identity`` (default ``simtest``, the account the iOS and Android tours sign
in as):
- 3 followed shows, 3 interests (trending topics) — Following, Profile → Interests, Recommended
- 4 listens with progress — Continue listening, Jump back in, Your week, Recently played
- 2 queued episodes, 2 favourite episodes
- 4 highlights over 2 episodes, one with a colour — Saved, Your week, the colour popover
- 3 boards (episodes and topics) and 3 notes — Library → Boards, the board picker

Rerunnable: boards and notes are ensured by name every time; the listening data is written once
(skipped when the account already has highlights).

Usage::

    python scripts/dev/seed_contact_sheet.py [--api http://127.0.0.1:8011] [--identity simtest]
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import sys
import time
import urllib.parse
import urllib.request
from typing import Any


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/") + "/api/app"
        jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def call(self, method: str, path: str, body: Any = None) -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with self.opener.open(req, timeout=30) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def sign_in(self, identity: str) -> dict[str, Any]:
        # The e2e api's mock provider: login → callback → session cookie. The final redirect lands
        # on "/", which the api does not serve; the cookie is what matters.
        try:
            self.opener.open(
                f"{self.base}/auth/login?as={urllib.parse.quote(identity)}", timeout=30
            )
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        me = self.call("GET", "/me")
        if not me or not me.get("user_id"):
            raise SystemExit(f"FAIL: sign-in as {identity!r} did not produce a session")
        return me


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--api", default="http://127.0.0.1:8011")
    ap.add_argument("--identity", default="simtest")
    args = ap.parse_args()

    t0 = time.monotonic()
    api = Api(args.api)
    me = api.sign_in(args.identity)
    print(f"signed in as {args.identity} ({me['user_id']})")

    episodes = api.call("GET", "/episodes?page_size=12")["items"]
    if len(episodes) < 6:
        raise SystemExit(
            f"FAIL: the corpus has {len(episodes)} episodes; the tour needs at least 6"
        )
    slugs = [e["slug"] for e in episodes]

    # Boards and notes are ensured on every run — the Boards tab is a screen of its own, and an
    # account seeded before they were added still gets them (operator 2026-10-08: "a few boards and
    # a few notes").
    boards, notes = ensure_boards_and_notes(api, slugs)

    existing = api.call("GET", "/highlights")
    if existing.get("items") if isinstance(existing, dict) else existing:
        print(f"OK: {args.identity} already had listening data; boards {boards}, notes {notes}")
        return 0

    podcasts = api.call("GET", "/podcasts")
    shows = podcasts.get("items", podcasts) if isinstance(podcasts, dict) else podcasts
    for p in shows[:3]:
        api.call("POST", "/library", {"feed_id": p["feed_id"], "title": p.get("title")})

    trending = api.call("GET", "/trending?kind=topic&scope=corpus&limit=3").get("items", [])
    for t in trending[:3]:
        api.call("POST", f"/interests/{urllib.parse.quote(t['entity_id'], safe='')}")

    tz = -int(time.timezone / 60)
    for i, slug in enumerate(slugs[:4]):
        # Two saves, the way the player writes them, so the listen counts as progress.
        for pos in (10, 120 + 60 * i):
            api.call(
                "PUT",
                f"/playback/{urllib.parse.quote(slug)}",
                {"position_seconds": pos, "tz_offset_minutes": tz},
            )

    for slug in slugs[4:6]:
        api.call("POST", "/queue/items", {"slug": slug, "after": None})
    for slug in slugs[:2]:
        api.call("PUT", "/favorites", {"kind": "episode", "ref": slug, "slug": slug})

    quotes = [
        "The real exposure is correlation, not component failure.",
        "Build the system so one mistake cannot take the rest down with it.",
        "You learn the most from the incidents nobody predicted.",
        "Write it down while it is still fresh.",
    ]
    highlight_ids = []
    for i, quote in enumerate(quotes):
        h = api.call(
            "POST",
            "/highlights",
            {
                "episode_slug": slugs[i % 2],
                "kind": "moment",
                "start_ms": 30_000 * (i + 1),
                "quote_text": quote,
            },
        )
        highlight_ids.append(h["id"])
    api.call("PATCH", f"/highlights/{highlight_ids[0]}", {"color": "amber"})

    print(
        f"OK: seeded {args.identity} in {time.monotonic() - t0:.1f}s — 3 shows, "
        f"{len(trending[:3])} interests, 4 listens, 2 queued, 2 favourites, "
        f"{len(highlight_ids)} highlights, {notes} notes, {boards} boards"
    )
    return 0


BOARDS = {
    "Risk & resilience": [("episode", 0), ("episode", 1), ("topic", "topic:risk-management")],
    "Systems thinking reading list": [
        ("episode", 2),
        ("topic", "topic:systems-thinking"),
        ("episode", 3),
    ],
    "Listen again": [("episode", 4), ("episode", 5)],
}
NOTES = [
    ("episode", 0, "Come back to the part about correlated failures."),
    ("episode", 2, "The example about delayed feedback loops is worth sharing with the team."),
    ("topic", "topic:risk-management", "Compare how each show defines risk — few agree."),
]


def ensure_boards_and_notes(api: Api, slugs: list[str]) -> tuple[int, int]:
    """Three boards with a mix of items, and three notes; anything already there is kept as is."""
    got = api.call("GET", "/collections")
    have = {c["name"] for c in (got.get("items", []) if isinstance(got, dict) else got)}
    for name, items in BOARDS.items():
        if name in have:
            continue
        board = api.call("POST", "/collections", {"name": name})
        for kind, ref in items:
            ref = slugs[ref] if isinstance(ref, int) else ref
            api.call(
                "POST",
                f"/collections/{urllib.parse.quote(board['id'])}/items",
                {"kind": kind, "ref": ref},
            )
    got_notes = api.call("GET", "/notes")
    texts = {
        n["text"]
        for n in (got_notes.get("items", []) if isinstance(got_notes, dict) else got_notes)
    }
    for target, ref, text in NOTES:
        if text in texts:
            continue
        api.call(
            "POST",
            "/notes",
            {
                "target": target,
                "target_id": slugs[ref] if isinstance(ref, int) else ref,
                "text": text,
            },
        )
    got = api.call("GET", "/collections")
    got_notes = api.call("GET", "/notes")
    count = lambda r: len(r.get("items", []) if isinstance(r, dict) else r)  # noqa: E731
    return count(got), count(got_notes)


if __name__ == "__main__":
    sys.exit(main())
