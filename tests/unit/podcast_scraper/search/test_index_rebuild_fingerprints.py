"""#1865: ``index_corpus(rebuild=True)`` must remove the fingerprint sidecar.

The per-episode fingerprint sidecar (``episode_fingerprints.json``) lives BESIDE the LanceDB index,
not inside it, so ``rmtree(lance_index)`` leaves it. ``build_two_tier_index`` then loads it, matches
every episode, skips them all, and writes nothing — a delete-then-guaranteed-no-op that took prod
search down (``--rebuild`` -> index deleted, EXIT_NO_ARTIFACTS). These tests pin that a rebuild
starts fingerprint-clean, including recovery from the orphaned-sidecar state (index already gone).
"""

from __future__ import annotations

import pytest

from podcast_scraper import config
from podcast_scraper.search import indexer, two_tier_indexer

pytestmark = pytest.mark.unit


def _cfg() -> config.Config:
    return config.Config(rss="https://example.com/feed.xml")


def _stub_build(monkeypatch, seen: dict) -> None:
    """Replace the heavy builder with a stub that records whether the sidecar still exists when
    the build runs (it must not — the rebuild should have removed it first)."""

    def _fake(corpus, lance, **_kwargs):
        seen["sidecar_at_build"] = two_tier_indexer._fingerprints_path(lance).exists()
        return two_tier_indexer.TwoTierIndexStats(episodes=0, segments=0, insights=0)

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _fake)


def test_rebuild_removes_the_fingerprint_sidecar(tmp_path, monkeypatch) -> None:
    lance_path = tmp_path / "search" / "lance_index"
    lance_path.mkdir(parents=True)
    sidecar = two_tier_indexer._fingerprints_path(lance_path)
    sidecar.write_text('{"ep1": "fp1"}', encoding="utf-8")

    seen: dict = {}
    _stub_build(monkeypatch, seen)
    indexer.index_corpus(str(tmp_path), _cfg(), rebuild=True)

    assert seen["sidecar_at_build"] is False, (
        "rebuild left the stale fingerprint sidecar in place; build_two_tier_index would load it, "
        "skip every episode, and produce an empty index (#1865)."
    )
    assert not sidecar.exists()
    assert not lance_path.exists()  # rmtree still nukes the index dir


def test_rebuild_recovers_orphaned_sidecar_when_index_already_gone(tmp_path, monkeypatch) -> None:
    """The prod 18:14 state: a prior --rebuild already deleted the index but stranded the sidecar,
    so every subsequent --rebuild skipped everything. The sidecar removal must not be gated on the
    index dir existing."""
    lance_path = tmp_path / "search" / "lance_index"
    (tmp_path / "search").mkdir(parents=True)  # index dir itself does NOT exist
    sidecar = two_tier_indexer._fingerprints_path(lance_path)
    sidecar.write_text('{"ep1": "fp1"}', encoding="utf-8")
    assert not lance_path.exists()

    seen: dict = {}
    _stub_build(monkeypatch, seen)
    indexer.index_corpus(str(tmp_path), _cfg(), rebuild=True)

    assert seen["sidecar_at_build"] is False
    assert not sidecar.exists()


def test_non_rebuild_keeps_the_sidecar(tmp_path, monkeypatch) -> None:
    """An incremental (non-rebuild) index must NOT delete the sidecar — that is what lets it skip
    unchanged episodes. Guards against over-correcting the fix into "always clear"."""
    lance_path = tmp_path / "search" / "lance_index"
    lance_path.mkdir(parents=True)
    sidecar = two_tier_indexer._fingerprints_path(lance_path)
    sidecar.write_text('{"ep1": "fp1"}', encoding="utf-8")

    seen: dict = {}
    _stub_build(monkeypatch, seen)
    indexer.index_corpus(str(tmp_path), _cfg(), rebuild=False)

    assert seen["sidecar_at_build"] is True, "incremental index dropped the fingerprint cache"
    assert sidecar.exists()
