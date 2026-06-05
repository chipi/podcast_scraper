"""GET /api/corpus/media — episode audio under media/."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def test_corpus_media_returns_audio(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True)
    target = media_dir / "ep.mp3"
    target.write_bytes(b"ID3test")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/media",
        params={"path": str(tmp_path), "relpath": "media/ep.mp3"},
    )
    assert r.status_code == 200
    assert r.content == b"ID3test"
    assert r.headers.get("accept-ranges") == "bytes"


def test_corpus_media_rejects_non_media_prefix(tmp_path: Path) -> None:
    p = tmp_path / "secret.mp3"
    p.write_bytes(b"ID3")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/media",
        params={"path": str(tmp_path), "relpath": "secret.mp3"},
    )
    assert r.status_code == 400
