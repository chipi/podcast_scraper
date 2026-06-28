"""Unit tests for :mod:`podcast_scraper.server.app_favorites_view`.

Exercise ``hydrate_favorites`` directly: episode favorites re-hydrate from the catalog,
insight favorites render from the stored snapshot, newest-first, with malformed/unknown
entries dropped.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server.app_favorites_view import hydrate_favorites

pytestmark = [pytest.mark.unit]


def _write_episode(root: Path, *, stem: str, episode_id: str) -> str:
    """Write a minimal episode and return its resolved slug."""
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f", "title": "Show", "url": "https://p.example/f.xml"},
        "episode": {"episode_id": episode_id, "title": "Hello", "published_date": "2024-01-01"},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hi", encoding="utf-8")
    rows = build_catalog_rows_cumulative(root)
    return slug_for_row(rows[0])


def test_hydrate_empty_is_empty_groups(tmp_path: Path) -> None:
    resp = hydrate_favorites(tmp_path, [])
    assert resp.episodes == [] and resp.insights == []


def test_hydrate_episode_favorite_rehydrates_from_catalog(tmp_path: Path) -> None:
    slug = _write_episode(tmp_path, stem="0001-hello", episode_id="ep1")
    resp = hydrate_favorites(tmp_path, [{"kind": "episode", "ref": slug, "label": "Hello"}])
    assert [e.slug for e in resp.episodes] == [slug]
    assert resp.insights == []


def test_hydrate_episode_favorite_uses_slug_fallback_key(tmp_path: Path) -> None:
    # An episode favorite stored under "slug" rather than "ref" still resolves.
    slug = _write_episode(tmp_path, stem="0001-hello", episode_id="ep1")
    resp = hydrate_favorites(tmp_path, [{"kind": "episode", "slug": slug}])
    assert [e.slug for e in resp.episodes] == [slug]


def test_hydrate_unknown_episode_slug_is_dropped(tmp_path: Path) -> None:
    _write_episode(tmp_path, stem="0001-hello", episode_id="ep1")
    resp = hydrate_favorites(tmp_path, [{"kind": "episode", "ref": "no-such-slug"}])
    assert resp.episodes == []


def test_hydrate_episode_favorite_without_ref_is_dropped(tmp_path: Path) -> None:
    resp = hydrate_favorites(tmp_path, [{"kind": "episode"}])
    assert resp.episodes == []


def test_hydrate_insight_favorite_from_snapshot(tmp_path: Path) -> None:
    resp = hydrate_favorites(
        tmp_path,
        [
            {
                "kind": "insight",
                "ref": "ep1#i1",
                "label": "A claim",
                "slug": "ep1-slug",
                "sublabel": "My Show",
                "start_ms": 5000,
            }
        ],
    )
    assert len(resp.insights) == 1
    ins = resp.insights[0]
    assert ins.ref == "ep1#i1"
    assert ins.text == "A claim"
    assert ins.episode_slug == "ep1-slug"
    assert ins.podcast_title == "My Show"
    assert ins.start_ms == 5000


def test_hydrate_insight_coerces_non_string_optional_fields_to_none(tmp_path: Path) -> None:
    # Non-string slug/sublabel and non-int start_ms fall back to None.
    resp = hydrate_favorites(
        tmp_path,
        [{"kind": "insight", "ref": "ep1#i1", "slug": 123, "sublabel": 9, "start_ms": "x"}],
    )
    ins = resp.insights[0]
    assert ins.episode_slug is None
    assert ins.podcast_title is None
    assert ins.start_ms is None
    assert ins.text == ""  # missing label → empty string


def test_hydrate_insight_without_ref_is_dropped(tmp_path: Path) -> None:
    resp = hydrate_favorites(tmp_path, [{"kind": "insight", "label": "no ref"}])
    assert resp.insights == []


def test_hydrate_unknown_kind_is_ignored(tmp_path: Path) -> None:
    resp = hydrate_favorites(tmp_path, [{"kind": "bookmark", "ref": "x"}])
    assert resp.episodes == [] and resp.insights == []


def test_hydrate_presents_newest_first(tmp_path: Path) -> None:
    # Stored newest-last; presentation reverses to newest-first.
    resp = hydrate_favorites(
        tmp_path,
        [
            {"kind": "insight", "ref": "old#i", "label": "old"},
            {"kind": "insight", "ref": "new#i", "label": "new"},
        ],
    )
    assert [i.ref for i in resp.insights] == ["new#i", "old#i"]
