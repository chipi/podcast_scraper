"""Unit tests for per-user key voices (wave-G)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.server import app_key_voices

pytestmark = pytest.mark.unit

_ROOT = Path("/unused")


def _person(pid: str, label: str) -> SimpleNamespace:
    return SimpleNamespace(id=pid, label=label)


def _wire(monkeypatch, *, heard: set[str], per_slug: dict[str, list]) -> None:
    monkeypatch.setattr(app_key_voices, "user_episode_set", lambda *a, **k: set(heard))
    monkeypatch.setattr(
        app_key_voices,
        "resolve_slug",
        lambda _root, slug: SimpleNamespace(has_kg=True, kg_relative_path=slug),
    )
    monkeypatch.setattr(app_key_voices, "load_json_artifact", lambda _root, rel: rel)
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
    assert voices[0] == {"id": "person:jane", "kind": "person", "label": "Jane", "episode_count": 3}


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


def test_skips_rows_without_kg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(app_key_voices, "user_episode_set", lambda *a, **k: {"s1"})
    monkeypatch.setattr(
        app_key_voices,
        "resolve_slug",
        lambda _root, slug: SimpleNamespace(has_kg=False, kg_relative_path=slug),
    )
    monkeypatch.setattr(app_key_voices, "load_json_artifact", lambda *a: {})
    monkeypatch.setattr(app_key_voices, "entities_from_kg", lambda *a: (["nope"], [], []))
    assert app_key_voices.key_voices_for_user(_ROOT, tmp_path, "u_x") == []
