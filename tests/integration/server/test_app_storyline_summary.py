"""#2344: the storyline page reads its summary — by the storyline id or any of its topics."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server import app_sessions, storyline_summaries  # noqa: E402
from podcast_scraper.server.app import create_app  # noqa: E402
from podcast_scraper.server.app_access import AccessPolicy  # noqa: E402
from podcast_scraper.server.app_user_store import get_or_create_user  # noqa: E402

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def _client(data_dir: Path) -> TestClient:
    app = create_app(ROOT, static_dir=False)
    app.state.session_secret = "test-secret"
    app.state.app_data_dir = data_dir
    app.state.access_policy = AccessPolicy("open", frozenset(), frozenset())
    client = TestClient(app)
    user = get_or_create_user(data_dir, provider="stub", subject="s", email="u@x.com", name="U")
    token = app_sessions.sign({"user_id": user.user_id, "iat": int(time.time())}, "test-secret")
    client.cookies.set(app_sessions.SESSION_COOKIE, token)
    return client


def test_the_summary_is_found_by_the_storyline_or_any_of_its_topics(tmp_path: Path) -> None:
    client = _client(tmp_path)
    risk = next(s for s in storyline_summaries.load_storylines(ROOT) if s.id == "thc:managing-risk")
    url = "/api/app/storylines/{}/summary"
    assert client.get(url.format("thc:managing-risk")).status_code == 404  # none written yet

    calls: list[str] = []

    def summarize(material: dict) -> str:
        calls.append(material["label"])
        return f"A thread on {material['label']}."

    storyline_summaries.sweep(
        ROOT,
        tmp_path,
        summarize=summarize,
        price=lambda m: 0.0,
        model="m",
        prompt="p",
    )
    assert len(calls) == len(storyline_summaries.load_storylines(ROOT)), "every storyline"
    for ref in ("thc:managing-risk", risk.anchor_topic_id, risk.topics[-1][0]):
        body = client.get(url.format(ref)).json()
        assert body["text"].startswith("A thread on") and body["stale"] is False, ref
    assert client.get(url.format("topic:not-in-any-storyline")).status_code == 404
