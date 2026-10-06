"""The cache warmer builds the entity id map in a child process, not on the api's GIL.

Prod 2026-10-06: while the warmer built the map in-process (minutes of difflib), every endpoint
that walks the corpus stalled -- search 186-195 s, /api/index/stats 84-173 s, /api/artifacts up
to 300 s -- and every post-deploy smoke failed in that window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.kg import entity_clusters

pytestmark = pytest.mark.unit


def _seed_two_spellings(root: Path) -> None:
    """A corpus the worker can scan (it may or may not cluster these; the test asserts it RAN)."""
    meta = root / "feeds" / "f" / "run_1" / "metadata"
    meta.mkdir(parents=True)
    (meta / "0001 - E.metadata.json").write_text("{}", encoding="utf-8")


@pytest.fixture(autouse=True)
def _fresh_cache():
    perf_cache.clear()
    yield
    perf_cache.clear()


def test_the_isolated_path_builds_once_and_caches(tmp_path, monkeypatch) -> None:
    calls = []

    def isolated(root, same_show_required):
        calls.append((Path(root), same_show_required))
        return {"person:a": "person:b"}

    monkeypatch.setattr(entity_clusters, "_build_entity_id_map_isolated", isolated)
    monkeypatch.setattr(
        entity_clusters,
        "build_entity_id_map",
        lambda *a, **k: pytest.fail("the warmer path must not build in-process"),
    )

    first = entity_clusters.cached_entity_id_map(tmp_path, isolate_process=True)
    second = entity_clusters.cached_entity_id_map(tmp_path)

    assert first == second == {"person:a": "person:b"}
    assert calls == [(tmp_path, True)]


def test_a_child_that_cannot_start_falls_back_to_in_process(tmp_path, monkeypatch) -> None:
    import subprocess

    def broken(*a, **k):
        raise OSError("no processes here")

    monkeypatch.setattr(subprocess, "run", broken)
    monkeypatch.setattr(
        entity_clusters, "build_entity_id_map", lambda root, same_show_required: {"x": "y"}
    )

    assert entity_clusters._build_entity_id_map_isolated(tmp_path, True) == {"x": "y"}


def test_a_real_child_process_builds_the_map(tmp_path, monkeypatch) -> None:
    """Runs the worker for real. The parent's own builder is made to FAIL, so only a child that
    actually ran can produce the answer -- the first version of this test passed through the
    in-process fallback while the child was crashing."""
    _seed_two_spellings(tmp_path)
    monkeypatch.setattr(
        entity_clusters,
        "build_entity_id_map",
        lambda *a, **k: pytest.fail("built in the parent: the child process did not run"),
    )
    id_map = entity_clusters._build_entity_id_map_isolated(tmp_path, True)
    assert isinstance(id_map, dict)


def test_the_warmer_asks_for_the_isolated_build(tmp_path, monkeypatch) -> None:
    from podcast_scraper.server import app_cache_warm

    seen = {}

    def fake(root, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(entity_clusters, "cached_entity_id_map", fake)
    app_cache_warm._warm_entity_id_map(tmp_path)
    assert seen == {"isolate_process": True}
