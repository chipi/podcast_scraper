"""GET /api/app/discover/rails: each card carries its reason in the shared shape (ADR-163).

The selection itself is unit-tested (``test_app_trending_episodes``); this pins the seam — the
builder's ``Reason`` becomes the response's ``why``, beside the episode it recommends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from podcast_scraper.server.app_reasons import grounded_quote, Reason, ReasonSubject  # noqa: E402
from podcast_scraper.server.app_trending_episodes import TrendingEpisodeCard  # noqa: E402
from podcast_scraper.server.schemas import AppQuote  # noqa: E402
from tests.integration.server.test_app_moments_route import _client, _episode, _slug  # noqa: E402

pytestmark = [pytest.mark.integration]


def test_each_card_carries_its_reason_beside_the_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _episode(tmp_path, "0001-a", "ep1", insights=1, segments=False)
    slug = _slug(tmp_path, "ep1")
    quote = grounded_quote(
        [AppQuote(text="Rates will fall.", speaker=None, start_ms=61_000, end_ms=None)],
        slug,
        speaker="Ann",
        insight_id="insight:0",
    )
    card = TrendingEpisodeCard(
        slug=slug,
        why=Reason(
            kind="rising_topic",
            quote=quote,
            subject=ReasonSubject(kind="topic", id="topic:rates", label="Rates"),
        ),
    )
    monkeypatch.setattr(
        "podcast_scraper.server.app_discover_rails.trending_episodes", lambda *a, **k: [card]
    )
    body = _client(tmp_path).get("/api/app/discover/rails").json()
    [rail] = body["rails"]
    assert (rail["rail"], rail["total"], rail["fallback"]) == ("quotes", 1, False)
    [item] = rail["items"]
    assert item["episode"]["slug"] == slug
    why = item["why"]
    assert why["kind"] == "rising_topic" and "outside_habits" not in why
    assert why["subject"] == {"kind": "topic", "id": "topic:rates", "label": "Rates"}
    assert why["quote"] == {
        "text": "Rates will fall.",
        "episode_slug": slug,
        "speaker": "Ann",
        "start_ms": 61_000,
        "insight_id": "insight:0",
    }
    assert why["capture"] is None and why["events"] is None
