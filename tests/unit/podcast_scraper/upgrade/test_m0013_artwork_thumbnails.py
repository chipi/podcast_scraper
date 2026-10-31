"""Thumbnails exist on disk, so a read-only server serves them instead of the full image.

Measured on prod (2026-10-01): `size=thumb` and `size=large` both returned the same 200,194 bytes
— the API mounts the corpus read-only, so it could never write the thumbnail it looked for.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from podcast_scraper.server.artwork import ensure_thumbnail
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0013_artwork_thumbnails import ArtworkThumbnailsMigration
from podcast_scraper.upgrade.registry import get_migrations
from podcast_scraper.utils.corpus_artwork import (
    download_podcast_artwork,
    THUMB_MAX_PX,
    thumbnail_path,
)

pytestmark = [pytest.mark.unit]


def _jpeg(size: int = 1400, colour: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buf, format="JPEG")
    return buf.getvalue()


def _store(root: Path, body: bytes) -> Path:
    h = hashlib.sha256(body).hexdigest()
    path = root / ".podcast_scraper" / "corpus-art" / "sha256" / h[:2] / h[2:4] / f"{h}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_the_writer_makes_the_thumbnail_when_it_downloads(tmp_path: Path) -> None:
    body = _jpeg()
    with patch("podcast_scraper.utils.corpus_artwork.http_get", return_value=(body, "image/jpeg")):
        rel = download_podcast_artwork(
            "https://cdn.example/a.jpg", tmp_path, user_agent="t", timeout=5
        )
    assert rel
    thumb = thumbnail_path(tmp_path, str(tmp_path / rel))
    assert thumb.is_file()
    with Image.open(thumb) as im:
        assert max(im.size) == THUMB_MAX_PX


def test_a_read_only_server_serves_an_existing_thumbnail(tmp_path: Path) -> None:
    original = _store(tmp_path, _jpeg())
    ArtworkThumbnailsMigration().apply(MigrationContext(corpus_root=tmp_path))
    with patch("podcast_scraper.server.artwork.write_thumbnail", side_effect=AssertionError):
        path, media_type = ensure_thumbnail(tmp_path, str(original))
    assert path == str(thumbnail_path(tmp_path, str(original))) and media_type == "image/jpeg"
    assert Path(path).stat().st_size < original.stat().st_size


def test_backfill_writes_every_missing_thumbnail_once(tmp_path: Path) -> None:
    a, b = _store(tmp_path, _jpeg(colour="red")), _store(tmp_path, _jpeg(colour="blue"))
    m = ArtworkThumbnailsMigration()
    ctx = MigrationContext(corpus_root=tmp_path)
    assert not m.verify(ctx)[0]
    assert m.apply(ctx).details["written"] == 2
    assert thumbnail_path(tmp_path, str(a)).is_file() and thumbnail_path(tmp_path, str(b)).is_file()
    assert m.apply(ctx).details["missing"] == 0  # idempotent
    assert m.verify(ctx)[0]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    original = _store(tmp_path, _jpeg())
    result = ArtworkThumbnailsMigration().apply(
        MigrationContext(corpus_root=tmp_path, dry_run=True)
    )
    assert result.details["missing"] == 1
    assert not thumbnail_path(tmp_path, str(original)).exists()


def test_an_undecodable_image_is_recorded_not_fatal(tmp_path: Path) -> None:
    _store(tmp_path, b"\xff\xd8\xff not really a jpeg")
    m = ArtworkThumbnailsMigration()
    ctx = MigrationContext(corpus_root=tmp_path)
    result = m.apply(ctx)
    assert result.details["written"] == 0 and len(result.details["failed"]) == 1
    ok, msg = m.verify(ctx)
    assert ok, msg


def test_registered_after_0012() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0013_artwork_thumbnails") == ids.index("0012_org_speakers_removed") + 1
