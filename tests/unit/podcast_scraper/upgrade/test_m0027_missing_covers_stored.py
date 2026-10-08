"""Covers the corpus only links to are stored, so phones get a downscale (2026-10-08).

On prod "The China-Global South Podcast" cover — a 3000x3000 PNG of 11.9 MB — was never stored
(the writer's cap was 8 MB), so every phone downloaded it in full for a 116 px tile.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest
from PIL import Image

from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations import m0027_missing_covers_stored as m0027
from podcast_scraper.upgrade.migrations.m0027_missing_covers_stored import (
    MissingCoversStoredMigration,
)
from podcast_scraper.upgrade.registry import get_migrations
from podcast_scraper.utils.corpus_artwork import medium_path, thumbnail_path

pytestmark = [pytest.mark.unit]

COVER = "https://cdn.example/big-cover.png"


def _png(size: int = 3000) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), "green").save(buf, format="PNG")
    return buf.getvalue()


def _episode(root: Path, episode_id: str, *, cover: str | None = COVER) -> Path:
    meta_dir = root / "feeds" / "f1" / "run_20260101-000000" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    path = meta_dir / f"{episode_id}.metadata.json"
    feed: Dict[str, Any] = {"feed_id": "f1", "title": "Show", "url": "https://example.com/f1.xml"}
    if cover:
        feed["image_url"] = cover
    path.write_text(
        json.dumps(
            {
                "feed": feed,
                "episode": {"episode_id": episode_id, "title": episode_id},
                "content": {"transcript_file_path": f"transcripts/{episode_id}.txt"},
                "schema_version": "1.0",
            }
        ),
        encoding="utf-8",
    )
    return path


def _ctx(root: Path, **kw: Any) -> MigrationContext:
    return MigrationContext(corpus_root=root, **kw)


def test_one_fetch_stores_the_cover_its_downscales_and_every_episode_records_it(
    tmp_path: Path,
) -> None:
    a, b = _episode(tmp_path, "e1"), _episode(tmp_path, "e2")
    body = _png()
    with patch(
        "podcast_scraper.utils.corpus_artwork.http_get", return_value=(body, "image/png")
    ) as get:
        res = MissingCoversStoredMigration().apply(_ctx(tmp_path))
    assert get.call_count == 1  # once per URL, not per episode
    rel = res.details["stored"][COVER]
    original = tmp_path / rel
    assert original.is_file()
    assert thumbnail_path(tmp_path, str(original)).is_file()
    assert medium_path(tmp_path, str(original)).is_file()
    for path in (a, b):
        assert json.loads(path.read_text())["feed"]["image_local_relpath"] == rel
    assert MissingCoversStoredMigration().verify(_ctx(tmp_path))[0]


def test_a_cover_over_8_mb_is_now_stored(tmp_path: Path) -> None:
    _episode(tmp_path, "e1")
    big = b"\x89PNG\r\n\x1a\n" + b"\0" * (12 * 1024 * 1024)  # 12 MB, past the old 8 MB cap
    with patch("podcast_scraper.utils.corpus_artwork.http_get", return_value=(big, "image/png")):
        res = MissingCoversStoredMigration().apply(_ctx(tmp_path))
    assert COVER in res.details["stored"]


def test_dry_run_fetches_and_writes_nothing(tmp_path: Path) -> None:
    path = _episode(tmp_path, "e1")
    before = path.read_text()
    with patch("podcast_scraper.utils.corpus_artwork.http_get", side_effect=AssertionError):
        res = MissingCoversStoredMigration().apply(_ctx(tmp_path, dry_run=True))
    assert res.details["to_fetch"] == 1
    assert path.read_text() == before


def test_an_unfetchable_cover_is_recorded_not_fatal_and_not_refetched(tmp_path: Path) -> None:
    _episode(tmp_path, "e1")
    with patch("podcast_scraper.utils.corpus_artwork.http_get", return_value=(None, "")):
        res = MissingCoversStoredMigration().apply(_ctx(tmp_path))
    assert res.details["failed"] == [COVER]
    ok, msg = MissingCoversStoredMigration().verify(_ctx(tmp_path))
    assert ok, msg
    with patch("podcast_scraper.utils.corpus_artwork.http_get", side_effect=AssertionError):
        assert MissingCoversStoredMigration().apply(_ctx(tmp_path)).details["to_fetch"] == 0


def test_a_stored_cover_is_left_alone_and_undo_restores_the_metadata(tmp_path: Path) -> None:
    path = _episode(tmp_path, "e1")
    original_text = path.read_text()
    with patch(
        "podcast_scraper.utils.corpus_artwork.http_get", return_value=(_png(600), "image/png")
    ):
        MissingCoversStoredMigration().apply(_ctx(tmp_path))
    with patch("podcast_scraper.utils.corpus_artwork.http_get", side_effect=AssertionError):
        assert MissingCoversStoredMigration().apply(_ctx(tmp_path)).details["to_fetch"] == 0
    restored, _errors = m0027.undo(tmp_path)
    assert restored == 1
    assert path.read_text() == original_text


def test_registered_after_0026() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0027_missing_covers_stored") == ids.index("0026_artwork_medium") + 1
