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


def test_corpus_media_resolves_real_extension_by_stem(tmp_path: Path) -> None:
    """A .mp3 request resolves the actually-persisted .m4a for the same stem (I1)."""
    media_dir = tmp_path / "media"
    media_dir.mkdir(parents=True)
    (media_dir / "ep.m4a").write_bytes(b"ftypM4A")

    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/media",
        params={"path": str(tmp_path), "relpath": "media/ep.mp3"},
    )
    assert r.status_code == 200
    assert r.content == b"ftypM4A"


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


@pytest.mark.parametrize(
    "relpath",
    [
        "media/../secret.mp3",
        "media/../../etc/passwd.mp3",
        "/etc/passwd.mp3",
        "media/sub/../../secret.mp3",
    ],
)
def test_corpus_media_blocks_path_traversal(tmp_path: Path, relpath: str) -> None:
    """Traversal out of media/ is rejected before any file access (G3)."""
    (tmp_path / "media").mkdir(parents=True)
    (tmp_path / "secret.mp3").write_bytes(b"SECRET")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/media",
        params={"path": str(tmp_path), "relpath": relpath},
    )
    assert r.status_code in (400, 404)
    assert r.content != b"SECRET"


def test_corpus_media_rejects_disallowed_suffix(tmp_path: Path) -> None:
    (tmp_path / "media").mkdir(parents=True)
    (tmp_path / "media" / "notes.txt").write_bytes(b"text")
    app = create_app(tmp_path, static_dir=False)
    client = TestClient(app)
    r = client.get(
        "/api/corpus/media",
        params={"path": str(tmp_path), "relpath": "media/notes.txt"},
    )
    assert r.status_code == 400
