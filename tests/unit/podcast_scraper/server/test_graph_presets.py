"""Presets once per revision and build (ADR-183 decision 11): the cache key, its token, the ETag,
and what is not cached. The engine's reads are faked; db.graph's own tests cover them."""

from __future__ import annotations

from typing import Any, Iterator

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import graph_presets

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _empty() -> Iterator[None]:
    perf_cache.clear(graph_presets.NAMESPACE)
    yield
    perf_cache.clear(graph_presets.NAMESPACE)


class _Db:
    """The probe's (revision, build) and a count of slice reads."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.revision, self.build, self.reads = 1, 10, 0
        monkeypatch.setattr(graph_presets, "_probe", lambda engine: (self.revision, self.build))
        monkeypatch.setattr(graph_presets.graph, "read_slice", self._read)

    def _read(self, engine: Any, fn: Any) -> dict[str, Any]:
        self.reads += 1
        return {"corpus_revision": self.revision, "graph_build_id": self.build, "n": self.reads}


def test_a_preset_is_read_once_per_revision_and_build(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _Db(monkeypatch)
    first = graph_presets.serve(None, "slice.recent", {"window": "7d"})  # type: ignore[arg-type]
    again = graph_presets.serve(None, "slice.recent", {"window": "7d"})  # type: ignore[arg-type]
    assert db.reads == 1 and again.body is first.body and again.etag == first.etag
    assert first.etag.startswith('"1:')
    graph_presets.serve(None, "slice.recent", {"window": "30d"})  # type: ignore[arg-type]
    assert db.reads == 2
    db.build = 11  # a projections publish does not move the revision (rule 7)
    rebuilt = graph_presets.serve(None, "slice.recent", {"window": "7d"})  # type: ignore[arg-type]
    assert db.reads == 3 and rebuilt.etag != first.etag
    db.revision = 2
    bumped = graph_presets.serve(None, "slice.recent", {"window": "7d"})  # type: ignore[arg-type]
    assert db.reads == 4 and bumped.etag.startswith('"2:')


def test_a_request_with_held_items_is_read_every_time(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _Db(monkeypatch)
    params = {"natural_ids": ["episode:1"], "exclude_items": ["episode:2"]}
    graph_presets.serve(None, "slice.items", params)  # type: ignore[arg-type]
    graph_presets.serve(None, "slice.items", params)  # type: ignore[arg-type]
    assert db.reads == 2


def test_an_unknown_preset_is_refused_before_any_read(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _Db(monkeypatch)
    with pytest.raises(graph_presets.UnknownPreset):
        graph_presets.serve(None, "slice.everything", {})  # type: ignore[arg-type]
    assert db.reads == 0


def test_the_cache_is_cleared_past_its_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _Db(monkeypatch)
    monkeypatch.setattr(graph_presets, "MAX_ENTRIES", 2)
    for window in ("7d", "30d", "90d"):
        graph_presets.serve(None, "slice.recent", {"window": window})  # type: ignore[arg-type]
    assert perf_cache.stats()[graph_presets.NAMESPACE]["entries"] == 1
    graph_presets.serve(None, "slice.recent", {"window": "90d"})  # type: ignore[arg-type]
    assert db.reads == 3
