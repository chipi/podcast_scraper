"""The player is served a ≤1024px copy, never the original (2026-10-08).

Measured on prod: 677 of 1,318 originals are 2001-3000px (~19 MB each once a phone decodes them),
and the episode detail handed them to the player hero AND to 116px cards. On a Pixel 8 emulator,
build 1.0.2 against prod, Home held 3000² and 2048² originals in "Jump back in".
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from podcast_scraper.server.artwork import artwork_url, ensure_medium
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0026_artwork_medium import ArtworkMediumMigration
from podcast_scraper.upgrade.registry import get_migrations
from podcast_scraper.utils.corpus_artwork import (
    download_podcast_artwork,
    MEDIUM_MAX_PX,
    medium_path,
    thumbnail_path,
)

pytestmark = [pytest.mark.unit]


def _jpeg(size: int = 3000, colour: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buf, format="JPEG")
    return buf.getvalue()


def _store(root: Path, body: bytes) -> Path:
    h = hashlib.sha256(body).hexdigest()
    path = root / ".podcast_scraper" / "corpus-art" / "sha256" / h[:2] / h[2:4] / f"{h}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def test_the_writer_makes_the_medium_and_the_thumb_when_it_downloads(tmp_path: Path) -> None:
    body = _jpeg()
    with patch("podcast_scraper.utils.corpus_artwork.http_get", return_value=(body, "image/jpeg")):
        rel = download_podcast_artwork(
            "https://cdn.example/a.jpg", tmp_path, user_agent="t", timeout=5
        )
    assert rel
    original = str(tmp_path / rel)
    with Image.open(medium_path(tmp_path, original)) as im:
        assert max(im.size) == MEDIUM_MAX_PX
    assert thumbnail_path(tmp_path, original).is_file()


def test_a_small_original_is_never_upscaled(tmp_path: Path) -> None:
    original = _store(tmp_path, _jpeg(size=600))
    ArtworkMediumMigration().apply(MigrationContext(corpus_root=tmp_path))
    with Image.open(medium_path(tmp_path, str(original))) as im:
        assert im.size == (600, 600)


def test_a_read_only_server_serves_an_existing_medium(tmp_path: Path) -> None:
    original = _store(tmp_path, _jpeg())
    ArtworkMediumMigration().apply(MigrationContext(corpus_root=tmp_path))
    with patch("podcast_scraper.server.artwork.write_medium", side_effect=AssertionError):
        path, media_type = ensure_medium(tmp_path, str(original))
    assert path == str(medium_path(tmp_path, str(original))) and media_type == "image/jpeg"


def test_a_missing_medium_on_a_read_only_server_falls_back_to_the_original(
    tmp_path: Path,
) -> None:
    original = _store(tmp_path, _jpeg())
    with patch("podcast_scraper.server.artwork.write_medium", return_value=False):
        path, media_type = ensure_medium(tmp_path, str(original))
    assert path == str(original) and media_type == "image/jpeg"


def test_backfill_writes_every_missing_medium_once(tmp_path: Path) -> None:
    a, b = _store(tmp_path, _jpeg(colour="red")), _store(tmp_path, _jpeg(colour="blue"))
    m = ArtworkMediumMigration()
    ctx = MigrationContext(corpus_root=tmp_path)
    assert not m.verify(ctx)[0]
    assert m.apply(ctx).details["written"] == 2
    assert medium_path(tmp_path, str(a)).is_file() and medium_path(tmp_path, str(b)).is_file()
    assert m.apply(ctx).details["missing"] == 0  # idempotent
    assert m.verify(ctx)[0]


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    original = _store(tmp_path, _jpeg())
    result = ArtworkMediumMigration().apply(MigrationContext(corpus_root=tmp_path, dry_run=True))
    assert result.details["missing"] == 1
    assert not medium_path(tmp_path, str(original)).exists()


def test_an_undecodable_image_is_recorded_not_fatal(tmp_path: Path) -> None:
    _store(tmp_path, b"\xff\xd8\xff not really a jpeg")
    m = ArtworkMediumMigration()
    ctx = MigrationContext(corpus_root=tmp_path)
    result = m.apply(ctx)
    assert result.details["written"] == 0 and len(result.details["failed"]) == 1
    ok, msg = m.verify(ctx)
    assert ok, msg


def test_the_default_url_size_is_the_thumb() -> None:
    assert (artwork_url(".podcast_scraper/corpus-art/sha256/aa/bb/x.jpg") or "").endswith(
        "&size=thumb"
    )


def test_registered_after_0025() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0026_artwork_medium") == ids.index("0025_one_person_one_entry") + 1
