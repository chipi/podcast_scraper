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

REPORT = ct.ContentType(
    name="test_report",
    id_scheme="item",
    stage_profile=ct.StageProfile(name="text", stages=frozenset({"summary", "kg"})),
    default_transport="report_api",
    source_kind="report",
)
REPORT_API = ct.Transport(name="report_api", list_items=lambda source, cursor: ([], cursor))


def _feeds(tmp_path: Path) -> dict:
    app = create_app(tmp_path, static_dir=False, enable_feeds_api=True)
    app.state.session_secret = ""
    body: dict = TestClient(app).get("/api/feeds", params={"path": str(tmp_path)}).json()
    return body


def test_a_public_server_offers_podcasts_and_news(tmp_path: Path) -> None:
    """Public's built-in types: podcast over RSS enclosures, news over RSS (ADR-177 decision 5)."""
    with use_extensions([]):
        body = _feeds(tmp_path)
    assert body["content_types"] == [
        {"name": "podcast_episode", "default_transport": "rss_enclosure", "source_kind": "podcast"},
        {"name": "news_article", "default_transport": "rss", "source_kind": "news"},
    ]
    assert body["transports"] == ["rss", "rss_enclosure"]


def test_a_type_an_extension_registers_is_offered_with_its_transport(tmp_path: Path) -> None:
    ext = Extension(name="reports", content_types=lambda: [REPORT], transports=lambda: [REPORT_API])
    with use_extensions([ext]):
        body = _feeds(tmp_path)
    names = {t["name"]: t for t in body["content_types"]}
    assert names["test_report"] == {
        "name": "test_report",
        "default_transport": "report_api",
        "source_kind": "report",
    }
    assert body["transports"] == ["report_api", "rss", "rss_enclosure"]
