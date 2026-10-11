"""More from what you captured (ADR-163; Discover arc Step 2.2) — the rail's eval.

Frozen listener state over the COMMITTED fixture corpus (``tests/fixtures/app-validation-corpus/v3``,
read only): two highlights of insights, on p07 and p02. The rail must recommend episodes ELSEWHERE,
each linked to one of those captures, round-robin across them, with a grounded quote from the
recommended episode — and its top results are pinned, so a ranking change shows up in review.
Deterministic, disk-only, no LLM.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server import app_sessions, app_user_state  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from podcast_scraper.server.app_access import AccessPolicy  # noqa: E402
from podcast_scraper.server.app_captured_rail import captured_rail  # noqa: E402
from podcast_scraper.server.app_corpus_access import load_json_artifact  # noqa: E402
from podcast_scraper.server.app_slugs import resolve_slug  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
SEEDS = (("p07-769d04686d", 2000), ("p02-8fdd7b7f79", 1000))  # (episode, created_at) newest first


def _first_about_insight(slug: str) -> str:
    row = resolve_slug(ROOT, slug)
    assert row is not None
    gi = load_json_artifact(ROOT, row.gi_relative_path)
    assert gi is not None
    return str(next(e["from"] for e in gi["edges"] if e.get("type") == "ABOUT"))


def _seed(data_dir: Path, user_id: str) -> None:
    for slug, at in SEEDS:
        app_user_state.add_highlight(
            data_dir,
            user_id,
            {
                "id": f"h-{slug}",
                "episode_slug": slug,
                "kind": "insight",
                "source_insight_id": _first_about_insight(slug),
                "quote_text": f"kept from {slug}",
                "created_at": at,
            },
        )


def test_recommends_elsewhere_round_robin_with_a_grounded_quote(tmp_path: Path) -> None:
    _seed(tmp_path, "u1")
    cards = captured_rail(ROOT, tmp_path, "u1", limit=6)
    assert cards, "two topic-linked highlights must find something elsewhere in the fixture"
    captured_shows = {"p07", "p02"}
    for card in cards:
        w = card.why
        row = resolve_slug(ROOT, card.slug)
        assert row is not None and row.feed_id not in captured_shows  # elsewhere
        assert card.slug not in {s for s, _ in SEEDS}
        assert w.kind == "captured" and w.capture is not None and w.capture.kind == "highlight"
        assert w.capture.text == f"kept from {w.capture.episode_slug}"  # the listener's own words
        assert w.subject is not None and w.subject.kind in ("topic", "person", "storyline")
        assert w.quote is not None and w.quote.episode_slug == card.slug and w.quote.text
    # Round-robin: the first two cards come from different captures, newest capture first.
    assert [c.why.capture.episode_slug for c in cards[:2]] == [s for s, _ in SEEDS]  # type: ignore[union-attr]


def test_top_results_are_pinned(tmp_path: Path) -> None:
    """The eval's frozen answer (2026-10-11, re-frozen on the grounded corpus, ADR-165). A change
    here is a ranking change: review it. Three, not five: the seeded highlights' insights now link
    only through what their lines actually discuss."""
    _seed(tmp_path, "u1")
    got = [
        (c.slug, c.why.capture.episode_slug, c.why.subject.kind)  # type: ignore[union-attr]
        for c in captured_rail(ROOT, tmp_path, "u1", limit=5)
    ]
    assert got == [
        ("p09-db2d39ef2e", "p07-769d04686d", "topic"),
        ("p09-a4bbb5dde3", "p02-8fdd7b7f79", "topic"),
        ("p01-74565a3a19", "p07-769d04686d", "storyline"),
    ]


def test_never_an_episode_already_heard_or_saved(tmp_path: Path) -> None:
    _seed(tmp_path, "u1")
    first = captured_rail(ROOT, tmp_path, "u1", limit=6)[0].slug
    app_user_state.add_favorite(tmp_path, "u1", {"kind": "episode", "ref": first, "added_at": 3000})
    assert first not in [c.slug for c in captured_rail(ROOT, tmp_path, "u1", limit=6)]


def test_nothing_captured_means_no_rail(tmp_path: Path) -> None:
    assert captured_rail(ROOT, tmp_path, "nobody", limit=6) == []


def _client(data_dir: Path, *, signed_in: bool) -> TestClient:
    app = create_app(ROOT, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    if signed_in:
        user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
        _seed(data_dir, user.user_id)
        token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
        client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_the_route_returns_cards_with_their_reason(tmp_path: Path) -> None:
    body = (
        _client(tmp_path, signed_in=True)
        .get("/api/app/discover/rails/captured", params={"scope": "mine", "limit": 12})
        .json()
    )
    assert body["rail"] == "captured" and body["items"]
    item = body["items"][0]
    assert item["why"]["kind"] == "captured"
    assert item["why"]["capture"]["kind"] == "highlight"
    assert item["episode"]["slug"] == item["why"]["quote"]["episode_slug"]
