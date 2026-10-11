"""A show's licence declarations reach the show page and its full-text page (#2325).

The newest episode's record is the show's; each distinct set of declarations seen across
episodes is kept with its read dates; a corpus written before licence capture reads "not
checked", never "no explicit licence found".
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server import app_sessions
from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_access import AccessPolicy
from podcast_scraper.server.app_user_store import get_or_create_user

pytestmark = [pytest.mark.integration]

NYT = (
    "© 2020-2021 THE NEW YORK TIMES COMPANY; The New York Times encourages the use of RSS "
    "feeds for personal use in a news reader or as part of a non-commercial blog, subject to "
    "your agreement to our Terms of Service."
)


def _record(raw: str | None, read: str) -> dict:
    decls: list[dict[str, Any]] = (
        []
        if raw is None
        else [
            {
                "level": "channel",
                "mechanism": "rss_copyright",
                "raw": raw,
                "url": None,
                "asset_scope": "unspecified",
                "known": None,
                "notes": [],
            }
        ]
    )
    return {
        "status": "declared" if decls else "no_explicit_licence_found",
        "declarations": decls,
        "conflicts": [],
        "documents_for_review": [],
        "source_feed_url": "https://feeds.example/x",
        "retrieved_at": read,
    }


def _episode(
    root: Path,
    stem: str,
    feed_id: str,
    published: str,
    licence: dict | None,
    own: dict | None = None,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    feed: dict[str, Any] = {
        "feed_id": feed_id,
        "title": feed_id.title(),
        "url": f"https://{feed_id}.example/f",
    }
    if licence is not None:
        feed["licence"] = licence
    episode: dict[str, Any] = {"episode_id": stem, "title": stem, "published_date": published}
    if own is not None:
        episode["licence"] = own
    doc = {
        "feed": feed,
        "episode": episode,
        "summary": {"title": "s", "bullets": ["a"]},
        "content": {},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


def _client(root: Path) -> TestClient:
    data_dir = root / "_appdata"
    app = create_app(root, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    client = TestClient(app)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def _corpus(root: Path) -> None:
    # "daily": the notice changed between two reads; the newer episode carries the newer one.
    _episode(
        root,
        "d1",
        "daily",
        "2024-01-01T00:00:00",
        _record("© 2020 NYT", "2024-01-02T00:00:00+00:00"),
    )
    _episode(root, "d2", "daily", "2024-03-01T00:00:00", _record(NYT, "2024-03-02T00:00:00+00:00"))
    _episode(root, "d3", "daily", "2024-04-01T00:00:00", _record(NYT, "2024-04-02T00:00:00+00:00"))
    # "quiet": read, nothing declared.  "old": written before licence capture.
    _episode(root, "q1", "quiet", "2024-02-01T00:00:00", _record(None, "2024-02-02T00:00:00+00:00"))
    _episode(root, "o1", "old", "2024-02-01T00:00:00", None)


def test_the_show_list_carries_the_one_line(tmp_path: Path) -> None:
    _corpus(tmp_path)
    items = {p["feed_id"]: p for p in _client(tmp_path).get("/api/app/podcasts").json()["items"]}
    assert items["daily"]["licence_summary"].startswith("© 2020-2021 THE NEW YORK TIMES COMPANY")
    assert items["daily"]["licence_summary"].endswith("…")
    assert items["quiet"]["licence_summary"] == "No explicit licence found"
    assert items["old"]["licence_summary"] is None


def test_the_licence_page_has_the_whole_text_and_its_history(tmp_path: Path) -> None:
    _corpus(tmp_path)
    body = _client(tmp_path).get("/api/app/podcasts/daily/licence").json()
    assert body["status"] == "declared"
    assert body["licence"]["declarations"][0]["raw"] == NYT
    assert [v["declarations"][0]["raw"] for v in body["history"]] == [NYT, "© 2020 NYT"]
    newest = body["history"][0]
    assert (newest["first_read"], newest["last_read"]) == (
        "2024-03-02T00:00:00+00:00",
        "2024-04-02T00:00:00+00:00",
    )


def test_not_checked_and_unknown_shows(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    old = client.get("/api/app/podcasts/old/licence").json()
    assert old["status"] == "not_checked" and old["licence"] is None and old["summary"] is None
    assert client.get("/api/app/podcasts/nope/licence").status_code == 404


def test_an_episode_carries_the_line_its_feed_had_when_it_was_ingested(tmp_path: Path) -> None:
    _corpus(tmp_path)
    client = _client(tmp_path)
    old = client.get("/api/app/podcasts/daily/episodes").json()["items"]
    by_title = {e["title"]: e["slug"] for e in old}
    first = client.get(f"/api/app/episodes/{by_title['d1']}").json()
    newest = client.get(f"/api/app/episodes/{by_title['d3']}").json()
    assert first["licence_summary"] == "© 2020 NYT"
    assert newest["licence_summary"].startswith("© 2020-2021 THE NEW YORK TIMES COMPANY")
    unchecked = client.get("/api/app/podcasts/old/episodes").json()["items"][0]["slug"]
    assert client.get(f"/api/app/episodes/{unchecked}").json()["licence_summary"] is None


CC_BY = {
    "status": "declared",
    "declarations": [
        {
            "level": "episode",
            "mechanism": "podcast_license",
            "raw": "cc-by-4.0",
            "url": None,
            "asset_scope": "audio_video",
            "known": {
                "id": "CC-BY-4.0",
                "name": "CC BY 4.0",
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "commercial_use": True,
                "adaptations": True,
                "attribution": True,
                "share_alike": False,
            },
            "notes": [],
        }
    ],
    "conflicts": [],
    "documents_for_review": [],
    "source_feed_url": "https://feeds.example/x",
    "retrieved_at": "2024-05-02T00:00:00+00:00",
}


def test_an_episode_licence_tag_is_the_episodes_line_and_both_are_on_its_page(
    tmp_path: Path,
) -> None:
    """Podcasting 2.0 lets an item carry its own <podcast:license>; the listener sees that one
    on the episode, and the page shows it beside the show's declaration."""
    _corpus(tmp_path)
    show = _record(NYT, "2024-05-02T00:00:00+00:00")
    _episode(tmp_path, "d4", "daily", "2024-05-01T00:00:00", show, own=CC_BY)
    nothing_own = dict(_record(None, "2024-05-02T00:00:00+00:00"))
    _episode(tmp_path, "d5", "daily", "2024-06-01T00:00:00", show, own=nothing_own)
    client = _client(tmp_path)
    slugs = {
        e["title"]: e["slug"]
        for e in client.get("/api/app/podcasts/daily/episodes").json()["items"]
    }

    own = client.get(f"/api/app/episodes/{slugs['d4']}").json()
    assert (own["licence_summary"], own["licence_scope"]) == ("CC BY 4.0", "episode")
    page = client.get(f"/api/app/episodes/{slugs['d4']}/licence").json()
    assert page["scope"] == "episode"
    assert page["episode"]["declarations"][0]["raw"] == "cc-by-4.0"
    assert page["show"]["declarations"][0]["raw"] == NYT

    inherits = client.get(f"/api/app/episodes/{slugs['d5']}").json()
    assert inherits["licence_scope"] == "show"
    assert inherits["licence_summary"].startswith("© 2020-2021 THE NEW YORK TIMES COMPANY")
    page = client.get(f"/api/app/episodes/{slugs['d5']}/licence").json()
    assert page["episode"]["status"] == "no_explicit_licence_found"
