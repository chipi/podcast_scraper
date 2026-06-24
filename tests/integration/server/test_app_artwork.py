"""Integration tests for consumer artwork serving (#1078 follow-up).

GET /api/app/artwork serves the locally-stored corpus art (large=original, thumb=downscale)
with immutable caching; AppEpisode summary/detail expose `artwork_url` when local art exists.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient
from PIL import Image

from podcast_scraper.server.app import create_app
from podcast_scraper.utils.corpus_artwork import CORPUS_ART_REL_PREFIX

pytestmark = [pytest.mark.integration]

ART_REL = f"{CORPUS_ART_REL_PREFIX}/sha256/ab/cd/abcd1234.png"


def _write_corpus_with_art(root: Path) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / "transcripts" / "0001.txt").write_text("hi", encoding="utf-8")
    # A real 600x600 PNG in the corpus-art store.
    art_path = root / ART_REL
    art_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (600, 600), (200, 80, 40)).save(art_path, format="PNG")

    doc = {
        "feed": {"feed_id": "showa", "title": "Show A", "url": "https://a.example/f"},
        "episode": {
            "episode_id": "a1",
            "title": "A One",
            "published_date": "2024-03-01T00:00:00",
            "duration_seconds": 1800,
            "image_local_relpath": ART_REL,
        },
        "summary": {"title": "Sum", "bullets": ["one", "two"]},
        "content": {"transcript_file": "transcripts/0001.txt"},
    }
    (root / "metadata" / "0001.metadata.json").write_text(json.dumps(doc), encoding="utf-8")


def _client(root: Path) -> TestClient:
    return TestClient(create_app(root, static_dir=False))


def test_large_serves_original_with_immutable_cache(tmp_path: Path) -> None:
    _write_corpus_with_art(tmp_path)
    resp = _client(tmp_path).get("/api/app/artwork", params={"ref": ART_REL, "size": "large"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("image/")
    assert "immutable" in resp.headers.get("cache-control", "")
    img = Image.open(io.BytesIO(resp.content))
    assert img.size == (600, 600)  # original, not resized


def test_thumb_downscales(tmp_path: Path) -> None:
    _write_corpus_with_art(tmp_path)
    resp = _client(tmp_path).get("/api/app/artwork", params={"ref": ART_REL, "size": "thumb"})
    assert resp.status_code == 200, resp.text
    img = Image.open(io.BytesIO(resp.content))
    assert max(img.size) <= 320  # downscaled for list density


def test_rejects_path_outside_art_store(tmp_path: Path) -> None:
    _write_corpus_with_art(tmp_path)
    resp = _client(tmp_path).get(
        "/api/app/artwork", params={"ref": "metadata/0001.metadata.json", "size": "large"}
    )
    assert resp.status_code == 400


def test_missing_art_is_404(tmp_path: Path) -> None:
    _write_corpus_with_art(tmp_path)
    missing = f"{CORPUS_ART_REL_PREFIX}/sha256/zz/zz/nope.png"
    assert _client(tmp_path).get("/api/app/artwork", params={"ref": missing}).status_code == 404


def test_invalid_size_rejected(tmp_path: Path) -> None:
    _write_corpus_with_art(tmp_path)
    resp = _client(tmp_path).get("/api/app/artwork", params={"ref": ART_REL, "size": "huge"})
    assert resp.status_code == 422


def test_episode_endpoints_expose_artwork_url(tmp_path: Path) -> None:
    _write_corpus_with_art(tmp_path)
    client = _client(tmp_path)
    listing = client.get("/api/app/episodes").json()
    item = listing["items"][0]
    assert item["artwork_url"] is not None
    assert "size=thumb" in item["artwork_url"]

    detail = client.get(f"/api/app/episodes/{item['slug']}").json()
    assert detail["artwork_url"] is not None
    assert "size=large" in detail["artwork_url"]
    # And that URL actually serves an image.
    art = client.get(detail["artwork_url"])
    assert art.status_code == 200 and art.headers["content-type"].startswith("image/")
