"""Reference client for the consumer platform API (#1072, RFC-098 task F6).

Drives the whole ``/api/app`` spine for a signed-in user — the tracer bullet that proves
the contract end-to-end: identity -> episode detail -> segments -> origin audio -> insights
-> entities -> grounded search -> playback resume -> queue -> library.

Works against any object with ``.get/.put/.post`` (Starlette ``TestClient`` in tests, or an
``httpx.Client`` against a live server via ``python -m
podcast_scraper.server.app_reference_client``). It is a *contract proof*, not the product UI
(the consumer PWA is RFC-099).
"""

from __future__ import annotations

import json
from typing import Any

from podcast_scraper.server import app_sessions


def walk_app_contract(client: Any, slug: str) -> dict[str, Any]:
    """Exercise the read+write spine for ``slug``; return a summary. Raises on any break."""

    def _ok(resp: Any) -> Any:
        resp.raise_for_status()
        return resp.json()

    summary: dict[str, Any] = {}
    summary["user"] = _ok(client.get("/api/app/me"))["email"]

    detail = _ok(client.get(f"/api/app/episodes/{slug}"))
    summary["title"] = detail["title"]
    feed_id = detail["feed_id"]

    summary["segments"] = len(_ok(client.get(f"/api/app/episodes/{slug}/segments"))["segments"])
    summary["audio_url"] = _ok(client.get(f"/api/app/episodes/{slug}/audio-source"))["url"]
    summary["insights"] = len(_ok(client.get(f"/api/app/episodes/{slug}/insights"))["insights"])
    summary["persons"] = len(_ok(client.get(f"/api/app/episodes/{slug}/entities"))["persons"])

    _ok(client.put(f"/api/app/playback/{slug}", json={"position_seconds": 12.0}))
    summary["resume_seconds"] = _ok(client.get(f"/api/app/playback/{slug}"))["position_seconds"]

    _ok(client.put("/api/app/queue", json={"items": [slug]}))
    summary["queue"] = _ok(client.get("/api/app/queue"))["items"]

    _ok(client.post("/api/app/library", json={"feed_id": feed_id}))
    summary["library"] = len(_ok(client.get("/api/app/library"))["items"])
    return summary


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin live wrapper
    import argparse

    import httpx

    parser = argparse.ArgumentParser(prog="app-reference-client")
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--session", required=True, help="lp_session cookie value")
    parser.add_argument("--slug", required=True, help="episode slug to walk")
    args = parser.parse_args(argv)
    with httpx.Client(
        base_url=args.base_url, cookies={app_sessions.SESSION_COOKIE: args.session}, timeout=15.0
    ) as client:
        print(json.dumps(walk_app_contract(client, args.slug), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
