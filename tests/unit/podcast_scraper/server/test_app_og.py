"""OG image route + SPA OG-injection tests (#2036), over the app-validation fixture corpus."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_kg_index import build_kg_index
from podcast_scraper.server.app_slugs import slug_for_row

_CORPUS = Path("tests/fixtures/app-validation-corpus/v3")
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_INDEX_HTML = (
    "<!doctype html><html><head><title>Learning Player</title>"
    '<meta name="description" content="x"></head><body><div id="app"></div></body></html>'
)


@pytest.fixture(scope="module")
def ids() -> dict[str, str]:
    idx = build_kg_index(_CORPUS)
    row = cached_catalog(_CORPUS)[0]
    return {
        "topic": next(iter(idx.topic_to_eps)),
        "person": next(iter(idx.person_to_eps)),
        "show": row.feed_id,
        "episode": slug_for_row(row),
    }


@pytest.fixture(scope="module")
def client() -> TestClient:
    static = Path(tempfile.mkdtemp())
    (static / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    (static / "assets").mkdir()
    (static / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    app = create_app(output_dir=_CORPUS, static_dir=static)
    return TestClient(app)


@pytest.mark.parametrize("kind", ["topic", "person", "show", "episode"])
def test_og_route_renders_a_png_per_kind(
    client: TestClient, ids: dict[str, str], kind: str
) -> None:
    r = client.get(f"/og/{kind}/{ids[kind]}.png")
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == _PNG_MAGIC


def test_og_route_is_public_no_auth_required(client: TestClient, ids: dict[str, str]) -> None:
    # No Authorization header — unfurl bots carry no session. A 200 proves the route is ungated.
    assert client.get(f"/og/topic/{ids['topic']}.png").status_code == 200


def test_og_route_404_for_unknown_entity(client: TestClient) -> None:
    assert client.get("/og/topic/topic:does-not-exist.png").status_code == 404


def test_og_route_404_for_unknown_kind(client: TestClient) -> None:
    assert client.get("/og/nonsense/whatever.png").status_code == 404


def test_entity_document_injects_og_tags(client: TestClient, ids: dict[str, str]) -> None:
    r = client.get(f"/topic/{ids['topic']}")
    assert r.status_code == 200
    body = r.text
    assert 'property="og:image"' in body
    assert 'property="og:title"' in body
    assert 'name="twitter:card" content="summary_large_image"' in body
    # og:image points at this topic's card route.
    assert "/og/topic/" in body and ".png" in body


def test_non_entity_route_falls_back_to_index_without_injection(client: TestClient) -> None:
    r = client.get("/catalog")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    # SPA shell, but no per-entity OG (this is the history-mode fallback, not a shareable entity).
    assert 'property="og:image"' not in r.text


def test_missing_asset_still_404s(client: TestClient) -> None:
    # A path with a file extension must NOT fall back to index.html (else a broken chunk import
    # would get HTML and fail with a MIME error) — it stays a hard 404.
    assert client.get("/assets/missing.js").status_code == 404


def test_real_asset_is_served(client: TestClient) -> None:
    r = client.get("/assets/app.js")
    assert r.status_code == 200
    assert "javascript" in r.headers["content-type"]
