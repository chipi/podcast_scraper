"""Per-topic CIL reads touch only the episodes that mention the topic (prod 2026-10-04).

Opening a topic, theme or storyline read EVERY episode's bridge, GI and KG from disk, uncached,
before discarding the ones that did not mention it — perspectives averaged 1.25s on prod and the
topic card peaked at 4.6s. ``iter_cil_episode_bundles_where`` takes bridges from the corpus-keyed
cache and reads GI/KG only for bridges the caller keeps. These tests pin both halves: it yields
exactly what the plain walk yields for those episodes, and it never parses a rejected one.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from podcast_scraper.server import cil_queries

FIXTURE = Path("tests/fixtures/app-validation-corpus/v3")


@pytest.fixture()
def root() -> str:
    if not FIXTURE.is_dir():
        pytest.fail(f"fixture corpus missing: {FIXTURE}")
    return os.path.abspath(FIXTURE)


def _a_topic(root: str) -> str:
    for _p, bridge in cil_queries.iter_cil_bridge_bundles(root, root):
        for tid in cil_queries._bridge_all_ids(bridge):
            if tid.startswith("topic:"):
                return tid
    pytest.fail("fixture corpus has no topic in any bridge")


def test_yields_exactly_what_the_plain_walk_keeps(root: str) -> None:
    topic = _a_topic(root)

    def keep(b: dict) -> bool:
        return topic in cil_queries._bridge_all_ids(b)

    plain = [
        (p, b, g, k) for p, b, g, k in cil_queries.iter_cil_episode_bundles(root, root) if keep(b)
    ]
    lazy = list(cil_queries.iter_cil_episode_bundles_where(root, root, keep))
    assert plain, "the topic matched no episode, so the comparison proves nothing"
    assert json.dumps(lazy, sort_keys=True) == json.dumps(plain, sort_keys=True)


def test_never_reads_gi_or_kg_for_a_rejected_episode(
    root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    list(cil_queries.iter_cil_episode_bundles_where(root, root, lambda _b: False))  # warm cache
    reads: list[str] = []
    real = cil_queries._read_json

    def spy(path: str) -> dict | None:
        reads.append(path)
        return real(path)

    monkeypatch.setattr(cil_queries, "_read_json", spy)
    assert list(cil_queries.iter_cil_episode_bundles_where(root, root, lambda _b: False)) == []
    assert [p for p in reads if p.endswith((".gi.json", ".kg.json"))] == []


def test_topic_perspectives_matches_the_full_walk(
    root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller end to end: same answer whether bundles come lazily or from the full walk."""
    topic = _a_topic(root)
    lazy = cil_queries.topic_perspectives(root, root, topic)

    def full_walk(r: str, a: str, ids: set[str]):  # parse everything, filter after
        return (
            t
            for t in cil_queries.iter_cil_episode_bundles(r, a)
            if ids & cil_queries._bridge_all_ids(t[1])
        )

    monkeypatch.setattr(cil_queries, "iter_cil_episode_bundles_naming", full_walk)
    assert cil_queries.topic_perspectives(root, root, topic) == lazy


def test_naming_yields_what_the_filter_yields_in_the_same_order(root: str) -> None:
    """The indexed iterator against the filter it replaces (Mine rails, 2026-11-08): every id the
    fixture's bridges name, alone and in unions, same tuples in the same order."""
    ids = sorted(
        {
            x
            for _p, b in cil_queries.iter_cil_bridge_bundles(root, root)
            for x in cil_queries._bridge_all_ids(b)
        }
    )
    cases = [{x} for x in ids] + [set(ids[i : i + 5]) for i in range(0, len(ids), 5)]
    cases += [{"topic:no-such-topic"}, set()]
    several = 0
    for case in cases:

        def keep(b: dict, c: set[str] = case) -> bool:
            return bool(c & cil_queries._bridge_all_ids(b))

        want = list(cil_queries.iter_cil_episode_bundles_where(root, root, keep))
        got = list(cil_queries.iter_cil_episode_bundles_naming(root, root, case))
        several += len(want) >= 2
        assert [(p, id(b), g, k) for p, b, g, k in got] == [
            (p, id(b), g, k) for p, b, g, k in want
        ], sorted(case)
    assert several, "no id is named by two bridges, so order is not compared"


def test_naming_never_reads_gi_or_kg_for_an_unnamed_episode(
    root: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    list(cil_queries.iter_cil_episode_bundles_naming(root, root, {"topic:warm"}))  # warm the index
    reads: list[str] = []
    real = cil_queries._read_json

    def spy(path: str) -> dict | None:
        reads.append(path)
        return real(path)

    monkeypatch.setattr(cil_queries, "_read_json", spy)
    assert list(cil_queries.iter_cil_episode_bundles_naming(root, root, {"topic:no-such"})) == []
    assert reads == []
