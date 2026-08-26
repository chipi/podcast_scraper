"""Unit tests for the interest-follows digest section (#1836) — new_in_interests_items.

Deterministic materialisation of topic/person follows: recent UNHEARD episodes about a followed
topic or featuring a followed person, no ranking score. Dependencies are stubbed so the test pins
the section's own logic (topic+person union, heard filtering, de-dup, newest-first, graph-gating).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.server import app_digest_sections as mod, app_graph_refs, app_user_state

pytestmark = [pytest.mark.unit]


class _Row:
    def __init__(self, slug: str, title: str, sk: int) -> None:
        self.slug = slug
        self.episode_title = title
        self._sk = sk

    def sort_key(self) -> int:  # smaller = newer (matches new_in_follows ordering)
        return self._sk


class _Ep:
    def __init__(self, row: _Row) -> None:
        self.row = row


def _wire(monkeypatch, *, interests, topic_eps, person_eps, heard, refs=True) -> None:
    monkeypatch.setattr(app_user_state, "get_interests", lambda dd, uid: interests)
    monkeypatch.setattr(mod, "user_episode_set", lambda root, dd, uid: set(heard))
    monkeypatch.setattr(mod, "slug_for_row", lambda r: r.slug)

    class _Idx:
        def topic_episodes(self, tid):
            return [_Ep(r) for r in topic_eps.get(tid, [])]

        def person_episodes(self, pid):
            return [_Ep(r) for r in person_eps.get(pid, [])]

    monkeypatch.setattr(mod, "get_kg_index", lambda root: _Idx())
    monkeypatch.setattr(
        app_graph_refs,
        "refs_for_slug",
        lambda root, slug: [{"id": "topic:x", "kind": "topic", "label": "X"}] if refs else [],
    )


def test_unheard_followed_topic_and_person_episodes(monkeypatch) -> None:
    new = _Row("ep-new", "New AI ep", 0)
    old = _Row("ep-old", "Old AI ep", 1)  # heard → dropped
    jane = _Row("ep-jane", "Jane ep", 2)
    _wire(
        monkeypatch,
        interests=["topic:ai", "person:jane", "tc:ignored"],
        topic_eps={"topic:ai": [new, old]},
        person_eps={"person:jane": [jane]},
        heard={"ep-old"},
    )
    out = mod.new_in_interests_items(Path("/root"), Path("/data"), "u1", limit=10)
    slugs = [i["episode_slug"] for i in out]
    assert slugs == ["ep-new", "ep-jane"]  # heard dropped, newest-first
    assert all(i["graph_refs"] for i in out)


def test_dedupes_episode_matching_multiple_follows(monkeypatch) -> None:
    shared = _Row("ep-shared", "Shared", 0)
    _wire(
        monkeypatch,
        interests=["topic:ai", "person:jane"],
        topic_eps={"topic:ai": [shared]},
        person_eps={"person:jane": [shared]},
        heard=set(),
    )
    out = mod.new_in_interests_items(Path("/root"), Path("/data"), "u1", limit=10)
    assert [i["episode_slug"] for i in out] == ["ep-shared"]  # once, not twice


def test_empty_when_no_topic_or_person_follows(monkeypatch) -> None:
    _wire(
        monkeypatch,
        interests=["tc:cluster", "thc:story"],
        topic_eps={},
        person_eps={},
        heard=set(),
    )
    assert mod.new_in_interests_items(Path("/root"), Path("/data"), "u1", limit=10) == []


def test_graphless_episodes_are_dropped(monkeypatch) -> None:
    _wire(
        monkeypatch,
        interests=["topic:ai"],
        topic_eps={"topic:ai": [_Row("ep-nograph", "No graph", 0)]},
        person_eps={},
        heard=set(),
        refs=False,
    )
    assert mod.new_in_interests_items(Path("/root"), Path("/data"), "u1", limit=10) == []
