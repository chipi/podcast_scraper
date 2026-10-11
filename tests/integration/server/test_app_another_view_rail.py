"""Hear another view (ADR-163; Discover arc Step 3) — the rail's eval.

Frozen listener state over the COMMITTED fixture corpus (``tests/fixtures/app-validation-corpus/v3``,
read only): highlights in three episodes about risk management and systems thinking, where the
listener heard Dr. Elena Fischer, Skanda Amarnath, Alex Morgan, Priya Nair and Ethan. The rail must
offer speakers they have NOT heard on those topics, never the umbrella topics every fixture episode
carries, never a speaker recorded as agreeing with one already heard. Its answer is pinned.
Deterministic, disk-only, no LLM.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server import app_sessions, app_user_state  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from podcast_scraper.server.app_access import AccessPolicy  # noqa: E402
from podcast_scraper.server.app_another_view_rail import (  # noqa: E402
    agreeing_pairs,
    another_view_rail,
    engaged_topics,
)
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"
HEARD = ("p07-a5eb82076a", "p07-2aceab172c", "p02-8fdd7b7f79")
HEARD_SPEAKERS = {"Dr. Elena Fischer", "Skanda Amarnath", "Alex Morgan", "Priya Nair", "Ethan"}


def _seed(data_dir: Path, user_id: str) -> None:
    for i, slug in enumerate(HEARD):
        app_user_state.add_highlight(
            data_dir,
            user_id,
            {
                "id": f"h-{slug}",
                "episode_slug": slug,
                "kind": "moment",
                "quote_text": "kept",
                "created_at": 1000 + i,
            },
        )


def test_umbrella_topics_are_not_subjects(tmp_path: Path) -> None:
    _seed(tmp_path, "u1")
    topics = [t for t, _ in engaged_topics(ROOT, tmp_path, "u1")]
    assert topics[:2] == ["topic:risk-management", "topic:systems-thinking"]
    # In all 55 fixture episodes: an umbrella, so "another view on" it says nothing.
    assert "topic:expert-interviews" not in topics and "topic:lifelong-learning" not in topics


def test_offers_speakers_not_yet_heard_on_the_topic(tmp_path: Path) -> None:
    _seed(tmp_path, "u1")
    cards = another_view_rail(ROOT, tmp_path, "u1", limit=10)
    assert cards
    for card in cards:
        w = card.why
        assert w.kind == "another_view" and w.subject is not None and w.subject.kind == "topic"
        assert w.quote is not None and w.quote.episode_slug == card.slug
        assert w.quote.speaker not in HEARD_SPEAKERS
        assert card.slug not in HEARD


def test_the_answer_is_pinned(tmp_path: Path) -> None:
    """The eval's frozen answer (2026-10-11). A change here is a ranking change: review it."""
    _seed(tmp_path, "u1")
    got = [
        (c.slug, c.why.subject.label, c.why.quote.speaker)  # type: ignore[union-attr]
        for c in another_view_rail(ROOT, tmp_path, "u1")
    ]
    assert got == [
        ("p05-25cb576153", "risk management", "Daniel Cho"),
        ("p03-5b8ca67393", "systems thinking", "Marco Silva"),
        ("p01-f2b497cfa3", "risk management", "Maya"),
    ]


def test_a_speaker_recorded_as_agreeing_with_one_heard_is_not_another_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture records no agreement involving the speakers here, so one is supplied: Daniel
    Cho agreeing with Dr. Elena Fischer (heard) on risk management takes him off the rail."""
    import podcast_scraper.server.app_another_view_rail as av

    _seed(tmp_path, "u1")
    assert another_view_rail(ROOT, tmp_path, "u1")[0].why.quote.speaker == "Daniel Cho"  # type: ignore[union-attr]
    pair = ("topic:risk-management", frozenset({"person:daniel-cho", "person:dr-elena-fischer"}))
    monkeypatch.setattr(av, "agreeing_pairs", lambda root: {pair})
    speakers = [c.why.quote.speaker for c in another_view_rail(ROOT, tmp_path, "u1")]  # type: ignore[union-attr]
    assert "Daniel Cho" not in speakers


def test_the_fixture_records_agreement_pairs(tmp_path: Path) -> None:
    raw = json.loads((ROOT / "enrichments" / "topic_consensus.json").read_text())
    assert len(agreeing_pairs(ROOT)) == len(raw["data"]["consensus"]) > 0


def test_nothing_heard_means_no_rail(tmp_path: Path) -> None:
    assert another_view_rail(ROOT, tmp_path, "nobody") == []


def test_the_route_returns_cards_with_their_reason(tmp_path: Path) -> None:
    app = create_app(ROOT, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = tmp_path
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(tmp_path, provider="stub", subject="s", email="u@x.com", name="U")
    _seed(tmp_path, user.user_id)
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    body = client.get(
        "/api/app/discover/rails/another_view", params={"scope": "mine", "limit": 12}
    ).json()
    assert body["rail"] == "another_view" and body["items"]
    item = body["items"][0]
    assert item["why"]["kind"] == "another_view" and item["why"]["subject"]["kind"] == "topic"
    assert (
        TestClient(app)
        .get("/api/app/discover/rails/another_view", params={"scope": "mine", "limit": 12})
        .status_code
        == 401
    )
