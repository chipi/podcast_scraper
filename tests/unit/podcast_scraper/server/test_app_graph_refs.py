"""Unit tests for the shared highlight→graph-refs resolver (#1419, app_graph_refs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_graph_refs

pytestmark = pytest.mark.unit

_ROOT = Path("/unused")


def test_refs_for_highlight_prefers_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    # A highlight with its own refs never re-resolves the episode KG.
    def _boom(*_a: object, **_k: object) -> list[dict[str, str]]:
        raise AssertionError("should not resolve when stored refs present")

    monkeypatch.setattr(app_graph_refs, "refs_for_slug", _boom)
    stored = [{"id": "person:x", "kind": "person", "label": "X"}]
    out = app_graph_refs.refs_for_highlight(_ROOT, {"episode_slug": "ep", "graph_refs": stored})
    assert out == stored


def test_refs_for_highlight_falls_back_to_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        app_graph_refs,
        "refs_for_slug",
        lambda root, slug, *, limit=3: [{"id": "topic:y", "kind": "topic", "label": "Y"}],
    )
    out = app_graph_refs.refs_for_highlight(_ROOT, {"episode_slug": "ep"})  # no stored refs
    assert out == [{"id": "topic:y", "kind": "topic", "label": "Y"}]


def test_refs_for_highlight_drops_malformed_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_graph_refs, "refs_for_slug", lambda *a, **k: [])
    stored = [{"id": "person:ok", "kind": "person", "label": "OK"}, {"no": "id"}, "junk"]
    out = app_graph_refs.refs_for_highlight(_ROOT, {"episode_slug": "ep", "graph_refs": stored})
    assert out == [{"id": "person:ok", "kind": "person", "label": "OK"}]


def test_refs_for_slug_empty_when_no_slug() -> None:
    assert app_graph_refs.refs_for_slug(_ROOT, "") == []
