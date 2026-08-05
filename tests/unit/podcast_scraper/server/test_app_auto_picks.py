"""Unit tests for the auto-highlight seed (#1416, app_auto_picks)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from podcast_scraper.server import app_auto_picks, app_graph_refs

pytestmark = pytest.mark.unit

_ROOT = Path("/unused")
_REFS = [{"id": "person:jane-doe", "kind": "person", "label": "Jane Doe"}]


def _insight(*, grounded: bool = True, start_ms: int | None = 60_000, text: str = "a key point"):
    quote = SimpleNamespace(start_ms=start_ms)
    return SimpleNamespace(grounded=grounded, quotes=[quote] if grounded else [], text=text)


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_auto_picks,
        "resolve_slug",
        lambda root, slug: SimpleNamespace(has_gi=True, gi_relative_path="g"),
    )
    monkeypatch.setattr(app_auto_picks, "load_json_artifact", lambda root, rel: {})
    monkeypatch.setattr(
        app_auto_picks, "insights_from_gi", lambda artifact, limit=None: [_insight()]
    )
    monkeypatch.setattr(app_graph_refs, "refs_for_slug", lambda root, slug, *, limit=3: list(_REFS))


def _heard(monkeypatch: pytest.MonkeyPatch, slugs: set[str]) -> None:
    monkeypatch.setattr(app_auto_picks, "user_episode_set", lambda root, dd, uid: set(slugs))


def test_picks_from_heard_but_uncaptured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _heard(monkeypatch, {"ep-a", "ep-b"})
    items = app_auto_picks.auto_pick_items(_ROOT, tmp_path, "u_x", exclude_slugs={"ep-b"}, limit=5)
    assert [i["episode_slug"] for i in items] == ["ep-a"]
    assert items[0]["source"] == "auto"
    assert items[0]["graph_refs"] == _REFS
    assert items[0]["deep_link"] == "/player/ep-a?t=60"
    assert items[0]["quote"] == "a key point"


def test_respects_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _heard(monkeypatch, {"ep-a", "ep-b", "ep-c"})
    items = app_auto_picks.auto_pick_items(_ROOT, tmp_path, "u_x", exclude_slugs=set(), limit=2)
    assert len(items) == 2


def test_limit_zero_returns_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _heard(monkeypatch, {"ep-a"})
    assert (
        app_auto_picks.auto_pick_items(_ROOT, tmp_path, "u_x", exclude_slugs=set(), limit=0) == []
    )


def test_ungrounded_insight_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _heard(monkeypatch, {"ep-a"})
    monkeypatch.setattr(
        app_auto_picks, "insights_from_gi", lambda artifact, limit=None: [_insight(grounded=False)]
    )
    assert (
        app_auto_picks.auto_pick_items(_ROOT, tmp_path, "u_x", exclude_slugs=set(), limit=5) == []
    )


def test_no_graph_refs_dropped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _heard(monkeypatch, {"ep-a"})
    monkeypatch.setattr(app_graph_refs, "refs_for_slug", lambda root, slug, *, limit=3: [])
    assert (
        app_auto_picks.auto_pick_items(_ROOT, tmp_path, "u_x", exclude_slugs=set(), limit=5) == []
    )
