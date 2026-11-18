"""GET /api/feeds lists the content types and transports a feed entry may name (ADR-177), so the
Workbench's feed editor offers only registered ones (UXS-019, News 1.0's acceptance)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper import content_types as ct
from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]

NEWS = ct.ContentType(
    name="news_article",
    id_scheme="item",
    stage_profile=ct.StageProfile(name="text", stages=frozenset({"summary", "kg"})),
    default_transport="rss_atom",
    source_kind="news",
)
RSS_ATOM = ct.Transport(name="rss_atom", list_items=lambda source, cursor: ([], cursor))


def _feeds(tmp_path: Path) -> dict:
    app = create_app(tmp_path, static_dir=False, enable_feeds_api=True)
    app.state.session_secret = ""
    body: dict = TestClient(app).get("/api/feeds", params={"path": str(tmp_path)}).json()
    return body


def test_a_public_server_offers_podcasts_over_rss_enclosures(tmp_path: Path) -> None:
    with use_extensions([]):
        body = _feeds(tmp_path)
    assert body["content_types"] == [
        {"name": "podcast_episode", "default_transport": "rss_enclosure", "source_kind": "podcast"}
    ]
    assert body["transports"] == ["rss_enclosure"]


def test_a_type_an_extension_registers_is_offered_with_its_transport(tmp_path: Path) -> None:
    ext = Extension(name="news", content_types=lambda: [NEWS], transports=lambda: [RSS_ATOM])
    with use_extensions([ext]):
        body = _feeds(tmp_path)
    names = {t["name"]: t for t in body["content_types"]}
    assert names["news_article"] == {
        "name": "news_article",
        "default_transport": "rss_atom",
        "source_kind": "news",
    }
    assert body["transports"] == ["rss_atom", "rss_enclosure"]
