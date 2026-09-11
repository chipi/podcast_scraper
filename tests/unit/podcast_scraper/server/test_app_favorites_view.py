"""Unit tests for :mod:`podcast_scraper.server.app_favorites_view`.

Exercise ``hydrate_favorites`` directly: episode favorites re-hydrate from the catalog, entity
favorites (show/topic/person/storyline) render from the save-time snapshot, newest-first, with
malformed/unknown entries dropped. Insights are NOT favorites (RFC-121 two-class model — they are
captures, served by the highlights path), so an ``insight`` entry is dropped here.
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
    assert resp.episodes == [] and resp.entities == []


def test_hydrate_episode_favorite_rehydrates_from_catalog(tmp_path: Path) -> None:
    slug = _write_episode(tmp_path, stem="0001-hello", episode_id="ep1")
    resp = hydrate_favorites(tmp_path, [{"kind": "episode", "ref": slug, "label": "Hello"}])
    assert [e.slug for e in resp.episodes] == [slug]
    assert resp.entities == []


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


def test_hydrate_entity_favorite_from_snapshot(tmp_path: Path) -> None:
    # RFC-121: shows/topics/people/storylines are saved as entity favorites, rendered from the
    # save-time snapshot (no per-row catalog hydration here).
    resp = hydrate_favorites(
        tmp_path,
        [{"kind": "person", "ref": "person:jane", "label": "Jane", "sublabel": "Host"}],
    )
    assert len(resp.entities) == 1
    ent = resp.entities[0]
    assert ent.kind == "person"
    assert ent.ref == "person:jane"
    assert ent.label == "Jane"
    assert ent.sublabel == "Host"


def test_hydrate_entity_coerces_non_string_sublabel_and_falls_back_label(tmp_path: Path) -> None:
    # A non-string sublabel falls back to None; a missing label falls back to the ref.
    resp = hydrate_favorites(tmp_path, [{"kind": "topic", "ref": "topic:ai", "sublabel": 9}])
    ent = resp.entities[0]
    assert ent.sublabel is None
    assert ent.label == "topic:ai"


def test_hydrate_entity_without_ref_is_dropped(tmp_path: Path) -> None:
    resp = hydrate_favorites(tmp_path, [{"kind": "show", "label": "no ref"}])
    assert resp.entities == []


def test_hydrate_insight_is_not_a_favorite(tmp_path: Path) -> None:
    # RFC-121 two-class model: an insight is a capture (highlight), never a favorite — dropped here.
    resp = hydrate_favorites(tmp_path, [{"kind": "insight", "ref": "ep1#i1", "label": "A claim"}])
    assert resp.episodes == [] and resp.entities == []


def test_hydrate_unknown_kind_is_ignored(tmp_path: Path) -> None:
    resp = hydrate_favorites(tmp_path, [{"kind": "bookmark", "ref": "x"}])
    assert resp.episodes == [] and resp.entities == []


def test_hydrate_presents_newest_first(tmp_path: Path) -> None:
    # Stored newest-last; presentation reverses to newest-first.
    resp = hydrate_favorites(
        tmp_path,
        [
            {"kind": "topic", "ref": "topic:old", "label": "old"},
            {"kind": "topic", "ref": "topic:new", "label": "new"},
        ],
    )
    assert [e.ref for e in resp.entities] == ["topic:new", "topic:old"]
