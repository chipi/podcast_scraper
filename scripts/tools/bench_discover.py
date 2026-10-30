"""Time the Discover / Explore endpoints in-process on a corpus (operator 2026-10-11: performance).

Signs in a listener with a small world (one followed storyline or topic, two captures), then calls
each endpoint twice — cold (first call in the process) and warm — and prints milliseconds. Read-only
on the corpus; the listener's state lives in a temp dir.

Usage:  .venv/bin/python scripts/tools/bench_discover.py <corpus_root> [--repeat N]
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()

    from fastapi.testclient import TestClient

    from podcast_scraper.search.storylines import storyline_map_by_topic
    from podcast_scraper.server import app_sessions, app_user_state
    from podcast_scraper.server.app import create_app
    from podcast_scraper.server.app_access import AccessPolicy
    from podcast_scraper.server.app_user_store import get_or_create_user
    from podcast_scraper.server.catalog_cache import cached_catalog
    from podcast_scraper.server.slugs import slug_for_row

    root = args.corpus.resolve()
    data = Path(tempfile.mkdtemp())
    app = create_app(root, static_dir=False)
    app.state.session_secret = "bench"
    app.state.app_data_dir = data
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(data, provider="stub", subject="b", email="b@x.com", name="B")
    client.cookies.set(
        app_sessions.SESSION_COOKIE,
        app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "bench"),
    )

    rows = [r for r in cached_catalog(root) if r.has_gi]
    from podcast_scraper.server.app_discover_view import interest_episode_index

    by_storyline: dict[str, set] = {}
    index = interest_episode_index(root)
    for tid, info in storyline_map_by_topic(root).items():
        by_storyline.setdefault(info["storyline_id"], set()).update(index.get(tid, set()))
    storylines = sorted(by_storyline, key=lambda s: -len(by_storyline[s]))
    subject = storylines[0] if storylines else None  # the broadest: the heaviest reel and path
    if subject:
        app_user_state.add_interest(data, user.user_id, subject)
    for i, row in enumerate(rows[:2]):
        app_user_state.add_highlight(
            data,
            user.user_id,
            {
                "id": f"h{i}",
                "episode_slug": slug_for_row(row),
                "kind": "moment",
                "quote_text": "k",
                "created_at": i + 1,
            },
        )
    print(f"corpus: {root}  episodes with GI: {len(rows)}  storylines: {len(storylines)}")

    calls = [
        ("rails (Mine, first pages)", "/api/app/discover/rails", {"scope": "mine"}),
        ("rails (Everyone)", "/api/app/discover/rails", {"scope": "corpus"}),
        ("reel-topics (Mine)", "/api/app/discover/reel-topics", {"scope": "mine"}),
        ("reel-topics (Everyone)", "/api/app/discover/reel-topics", {"scope": "corpus"}),
        ("learn (Everyone offers)", "/api/app/discover/learn", {"scope": "corpus"}),
        ("trending topic page", "/api/app/trending", {"kind": "topic", "limit": 5}),
    ]
    if subject:
        calls += [
            ("reel (storyline, subject-page door)", "/api/app/discover/reel", {"subject": subject}),
            ("learn subject (storyline door)", f"/api/app/discover/learn/subjects/{subject}", {}),
        ]
    for label, url, params in calls:
        times = []
        status = None
        for _ in range(1 + args.repeat):
            t0 = time.perf_counter()
            r = client.get(url, params=params)
            times.append((time.perf_counter() - t0) * 1000)
            status = r.status_code
        warm = statistics.median(times[1:]) if len(times) > 1 else times[0]
        body = r.json() if status == 200 else {}
        size = (
            sum(len(x.get("items") or []) for x in body.get("rails") or [])
            or len(body.get("items") or body.get("clips") or body.get("candidates") or [])
            or body.get("steps", 0)
        )
        print(f"{label:40} cold {times[0]:8.0f} ms  warm {warm:8.0f} ms  [{status}, {size} items]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
