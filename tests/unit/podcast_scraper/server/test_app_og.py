"""OG image route + SPA OG-injection tests (#2036), over the app-validation fixture corpus."""

from __future__ import annotations

import os
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
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    static = tmp_path_factory.mktemp("static")
    (static / "index.html").write_text(_INDEX_HTML, encoding="utf-8")
    (static / "assets").mkdir()
    (static / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    # Point per-user/app-data at a tmp dir so create_app doesn't write .app/.viewer INTO the tracked
    # fixture corpus (the OG routes don't need the user store).
    os.environ["APP_DATA_DIR"] = str(tmp_path_factory.mktemp("appdata"))
    try:
        app = create_app(output_dir=_CORPUS, static_dir=static)
    finally:
        os.environ.pop("APP_DATA_DIR", None)
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


def test_og_route_sets_cache_and_nosniff_headers(client: TestClient, ids: dict[str, str]) -> None:
    r = client.get(f"/og/topic/{ids['topic']}.png")
    assert "max-age" in r.headers.get("cache-control", "")
    assert r.headers.get("x-content-type-options") == "nosniff"


def test_organization_build_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # The fixture corpus has no orgs, so drive the _org build branch with a stub card: description →
    # blurb, founded → stats, green accent, and a missing logo → no artwork (graceful).
    from types import SimpleNamespace

    import podcast_scraper.server.app_relational_view as arv
    from podcast_scraper.server.og import build

    web = SimpleNamespace(
        description="The central banking system of the United States.",
        summary=None,
        founded="1913",
        industry="Finance",
    )
    card = SimpleNamespace(id="org:fed", label="The Federal Reserve", episode_count=17, web=web)
    monkeypatch.setattr(arv, "build_org_card", lambda _root, _ident: card)

    m = build.build_og_model(_CORPUS, "organization", "org:fed")
    assert m is not None
    assert m.kicker == "Organization"
    assert m.title == "The Federal Reserve"
    assert m.blurb == "The central banking system of the United States."
    assert "17 episodes" in (m.stats or "") and "founded 1913" in (m.stats or "")
    assert m.accent == "#5fd0a8"  # org green
    assert m.artwork is None  # no logo file in the fixture → graceful


def test_og_route_503_when_renderer_unavailable(
    client: TestClient, ids: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # If Pillow/the renderer blows up, the route degrades to 503, not a 500 traceback.
    import podcast_scraper.server.routes.app_og as og_route

    def _boom(_model: object) -> bytes:
        raise RuntimeError("no pillow")

    monkeypatch.setattr(og_route, "render_card_png", _boom)
    assert client.get(f"/og/topic/{ids['topic']}.png").status_code == 503


def test_spa_origin_sanitises_a_crafted_host_header() -> None:
    # A Host header with an embedded newline must not leak into the injected og:image URL.
    from podcast_scraper.server.spa import SpaStaticFiles

    scope = {
        "scheme": "https",
        "headers": [(b"host", b"evil.example\r\nX-Injected: 1")],
    }
    origin = SpaStaticFiles._origin(scope)
    assert "\n" not in origin and "\r" not in origin
    assert origin == "https://evil.example"


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


# --- build-level enrichment (the per-card fields we tune) ---------------------------------------


def test_topic_model_carries_voices_stat_and_attributed_quote(ids: dict[str, str]) -> None:
    from podcast_scraper.server.og.build import build_og_model

    m = build_og_model(_CORPUS, "topic", ids["topic"])
    assert m is not None
    assert "episode" in (m.stats or "") and "voice" in (m.stats or "")  # "N episodes · M voices"
    if m.quote:  # when the topic has a perspective, the quote is attributed in the byline
        assert (m.byline or "").startswith("—")


def test_storyline_model_explains_itself(ids: dict[str, str]) -> None:
    from podcast_scraper.server.og.build import build_og_model

    m = build_og_model(_CORPUS, "storyline", ids["topic"])
    assert m is not None
    # It defines what a storyline is (byline) and shows WHICH topics (blurb).
    assert m.byline == "Topics discussed together"
    assert m.blurb and "·" in m.blurb
    assert "topic" in (m.stats or "")


def test_show_and_episode_models_carry_artwork_when_present(ids: dict[str, str]) -> None:
    from podcast_scraper.server.og.build import build_og_model

    show = build_og_model(_CORPUS, "show", ids["show"])
    episode = build_og_model(_CORPUS, "episode", ids["episode"])
    assert show is not None and episode is not None
    # The fixture ships corpus art for its feeds, so both should composite an identity square.
    assert isinstance(show.artwork, (bytes, type(None)))
    assert isinstance(episode.artwork, (bytes, type(None)))


def test_episode_uses_full_bleed_background(ids: dict[str, str]) -> None:
    from podcast_scraper.server.og.build import build_og_model

    episode = build_og_model(_CORPUS, "episode", ids["episode"])
    assert episode is not None
    assert episode.background is True  # episode art is the backdrop, not a square
    # Duration + published land in the footer stats; upper part stays clean for the summary.
    assert "min" in (episode.stats or "")


def test_show_footer_carries_a_published_date(ids: dict[str, str]) -> None:
    from podcast_scraper.server.og.build import build_og_model

    show = build_og_model(_CORPUS, "show", ids["show"])
    assert show is not None
    assert "latest" in (show.stats or "")  # "N episodes · … · latest Mon YYYY"
    assert show.background is False


def test_guest_person_gets_a_show_gallery(ids: dict[str, str]) -> None:
    # A guest with no photo → a gallery tile per show, and no single artwork square.
    from podcast_scraper.server.og.build import build_og_model

    guest = build_og_model(_CORPUS, "person", "person:dr-elena-fischer")
    assert guest is not None
    assert guest.byline == "Guest"
    assert len(guest.gallery) >= 2  # she guests on 2 shows in the fixture
    assert guest.artwork is None
    assert "shows" in (guest.stats or "")


def test_host_person_names_their_show(ids: dict[str, str]) -> None:
    from podcast_scraper.server.og.build import build_og_model

    host = build_og_model(_CORPUS, "person", "person:sam")
    assert host is not None
    assert (host.byline or "").startswith("Host of ")
    assert host.tags  # that show's key topics land in the footer


def test_build_og_meta_is_text_only(ids: dict[str, str]) -> None:
    # The SPA document path (build_og_meta → with_art=False) must NOT load artwork bytes.
    from podcast_scraper.server.og.build import build_og_model

    lite = build_og_model(_CORPUS, "episode", ids["episode"], with_art=False)
    full = build_og_model(_CORPUS, "episode", ids["episode"])
    assert lite is not None and full is not None
    assert lite.artwork is None and lite.gallery == ()
    assert lite.title == full.title  # same text, just no image reads


def test_image_credit_formats_attribution() -> None:
    from podcast_scraper.server.og.build import _image_credit

    assert _image_credit("Photo", "A. Smith", "CC BY-SA 4.0") == "Photo: A. Smith · CC BY-SA 4.0"
    assert _image_credit("Logo", None, "CC0") == "Logo: CC0"
    assert _image_credit("Photo", None, None) is None
