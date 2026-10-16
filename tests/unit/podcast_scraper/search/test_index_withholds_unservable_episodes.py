"""An episode marked unusable is not indexed, and its rows leave search (#2375, D-44).

The catalog already hides such an episode. Search did not: the indexer indexed every metadata file,
so a refused or untranslated episode stayed findable. Rows are withdrawn only for an episode with
no servable run in the build, so an older unusable run never takes a newer good run's rows with it.
"""

from __future__ import annotations

import json

import pytest

from podcast_scraper.search.two_tier_indexer import _prune_withheld, _servable_docs

pytestmark = pytest.mark.unit


def _meta(tmp_path, name, episode_id, unusable=False):
    episode = {"episode_id": episode_id, "title": name}
    if unusable:
        episode.update({"unusable": True, "unusable_reason": "language_refused"})
    path = tmp_path / f"{name}.metadata.json"
    path.write_text(json.dumps({"episode": episode, "feed": {"title": "Show"}}), encoding="utf-8")
    return path


def test_an_unusable_run_is_not_handed_to_the_indexer(tmp_path) -> None:
    paths = [_meta(tmp_path, "good", "e1"), _meta(tmp_path, "refused", "e2", unusable=True)]
    served, withheld = set(), set()
    got = [doc["episode"]["episode_id"] for _, doc in _servable_docs(paths, {}, served, withheld)]
    assert got == ["e1"]
    assert served == {"e1"} and withheld == {"e2"}


def test_its_fingerprint_is_forgotten_so_it_is_embedded_again_once_servable(tmp_path) -> None:
    fps = {"feed-a\x1fe2": "fp2", "e2": "fp2b", "feed-a\x1fe22": "keep", "e1": "fp1"}
    list(_servable_docs([_meta(tmp_path, "refused", "e2", unusable=True)], fps, set(), set()))
    assert fps == {"feed-a\x1fe22": "keep", "e1": "fp1"}


def test_an_unreadable_metadata_file_is_skipped(tmp_path) -> None:
    bad = tmp_path / "bad.metadata.json"
    bad.write_text("{not json", encoding="utf-8")
    assert list(_servable_docs([bad], {}, set(), set())) == []


class _Backend:
    def __init__(self):
        self.calls = []

    def prune_episode_rows(self, tier, episode_id, keep_ids):
        self.calls.append((tier, episode_id, set(keep_ids)))
        return 2


def test_rows_leave_every_tier_for_an_episode_with_no_servable_run() -> None:
    backend = _Backend()
    removed = _prune_withheld(backend, served={"e1"}, withheld={"e2"}, clear_requested=False)
    assert {(t, e) for t, e, _ in backend.calls} == {
        ("segment", "e2"),
        ("segment_nonen", "e2"),
        ("insight", "e2"),
        ("aux", "e2"),
    }
    assert all(keep == set() for _, _, keep in backend.calls)
    assert removed == 8


def test_an_episode_with_a_servable_run_keeps_its_rows() -> None:
    """An older run refused, a newer one good: the newer run's rows must stay."""
    backend = _Backend()
    assert _prune_withheld(backend, served={"e2"}, withheld={"e2"}, clear_requested=False) == 0
    assert backend.calls == []


def test_a_full_rebuild_needs_no_pruning() -> None:
    backend = _Backend()
    assert _prune_withheld(backend, served=set(), withheld={"e2"}, clear_requested=True) == 0
    assert backend.calls == []
