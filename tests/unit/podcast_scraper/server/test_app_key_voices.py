"""Unit tests for per-user key voices (wave-G)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.server import app_key_voices

pytestmark = pytest.mark.unit

_ROOT = Path("/unused")


def _person(pid: str, label: str) -> SimpleNamespace:
    # Mirror the real AppEntity shape entities_from_kg yields: the display field is ``name``, NOT
    # ``label`` (a SimpleNamespace(label=…) fake hid a prod crash where the rail read ``.label``).
    return SimpleNamespace(id=pid, name=label)


def _row(slug: str, *, has_kg: bool = True) -> SimpleNamespace:
    # Mirror the CatalogEpisodeRow contract the ranker relies on: has_kg, kg_relative_path, and a
    # newest-first sort_key (here keyed on the slug so ordering in tests is deterministic).
    return SimpleNamespace(has_kg=has_kg, kg_relative_path=slug, sort_key=lambda s=slug: (0, 0, s))


def _wire(
    monkeypatch, *, heard: set[str], per_slug: dict[str, list], photos: dict[str, str] | None = None
) -> None:
    monkeypatch.setattr(app_key_voices, "user_episode_set", lambda *a, **k: set(heard))
    monkeypatch.setattr(app_key_voices, "resolve_slug", lambda _root, slug: _row(slug))
    monkeypatch.setattr(app_key_voices, "load_json_artifact", lambda _root, rel: rel)
    monkeypatch.setattr(app_key_voices, "hosted_photo_urls", lambda _root: dict(photos or {}))
    monkeypatch.setattr(
        app_key_voices,
        "entities_from_kg",
        lambda rel: (per_slug.get(rel, []), [], []),
    )


def test_ranks_people_by_episode_presence(tmp_path: Path, monkeypatch) -> None:
    # jane in 3 eps, john in 2, amir in 1 → that order.
    _wire(
        monkeypatch,
        heard={"s1", "s2", "s3"},
        per_slug={
            "s1": [_person("person:jane", "Jane"), _person("person:john", "John")],
            "s2": [_person("person:jane", "Jane"), _person("person:john", "John")],
            "s3": [_person("person:jane", "Jane"), _person("person:amir", "Amir")],
        },
    )
    voices = app_key_voices.key_voices_for_user(_ROOT, tmp_path, "u_x", limit=8)
    assert [v["id"] for v in voices] == ["person:jane", "person:john", "person:amir"]
    assert voices[0] == {
        "id": "person:jane",
        "kind": "person",
        "label": "Jane",
        "episode_count": 3,
        "image_url": None,
    }


def test_hosted_photo_hydrates_image_url(tmp_path: Path, monkeypatch) -> None:
    # A person the web enricher hosts a photo for carries the served route; others stay None.
    _wire(
        monkeypatch,
        heard={"s1"},
        per_slug={"s1": [_person("person:jane", "Jane"), _person("person:john", "John")]},
        photos={"person:jane": "/api/app/persons/person%3Ajane/photo"},
    )
    voices = app_key_voices.key_voices_for_user(_ROOT, tmp_path, "u_x")
    by_id = {v["id"]: v["image_url"] for v in voices}
    assert by_id["person:jane"] == "/api/app/persons/person%3Ajane/photo"
    assert by_id["person:john"] is None


def test_limit_caps_the_rail(tmp_path: Path, monkeypatch) -> None:
    _wire(
        monkeypatch,
        heard={"s1"},
        per_slug={"s1": [_person(f"person:p{i}", f"P{i}") for i in range(10)]},
    )
    assert len(app_key_voices.key_voices_for_user(_ROOT, tmp_path, "u_x", limit=3)) == 3


def test_empty_when_no_graph(tmp_path: Path, monkeypatch) -> None:
    _wire(monkeypatch, heard=set(), per_slug={})
    assert app_key_voices.key_voices_for_user(_ROOT, tmp_path, "u_x") == []


def test_unresolvable_heard_slugs_are_skipped(tmp_path: Path, monkeypatch) -> None:
    # A heard slug that no longer resolves to a catalog row (re-scrape churn) is dropped, not fatal.
    monkeypatch.setattr(app_key_voices, "user_episode_set", lambda *a, **k: {"gone", "s1"})
    monkeypatch.setattr(
        app_key_voices, "resolve_slug", lambda _root, slug: None if slug == "gone" else _row(slug)
    )
    monkeypatch.setattr(app_key_voices, "load_json_artifact", lambda _root, rel: rel)
    monkeypatch.setattr(app_key_voices, "hosted_photo_urls", lambda _root: {})
    monkeypatch.setattr(
        app_key_voices, "entities_from_kg", lambda rel: ([_person("person:jane", "Jane")], [], [])
    )
    voices = app_key_voices.key_voices_for_user(_ROOT, tmp_path, "u_x")
    assert [v["id"] for v in voices] == ["person:jane"]


def test_skips_rows_without_kg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_key_voices, "user_episode_set", lambda *a, **k: {"s1"})
    monkeypatch.setattr(
        app_key_voices, "resolve_slug", lambda _root, slug: _row(slug, has_kg=False)
    )
    monkeypatch.setattr(app_key_voices, "load_json_artifact", lambda *a: {})
    monkeypatch.setattr(app_key_voices, "hosted_photo_urls", lambda _root: {})
    monkeypatch.setattr(app_key_voices, "entities_from_kg", lambda *a: (["nope"], [], []))
    assert app_key_voices.key_voices_for_user(_ROOT, tmp_path, "u_x") == []
