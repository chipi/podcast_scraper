"""Unit tests for :mod:`podcast_scraper.server.app_discover_view` (#1098).

Covers the two pure-ish pieces of the personalized-discovery ranker:

* :func:`_significance` — the provisional content-depth weighting
  (``+2`` GI, ``+1`` KG, ``+0.2`` per summary bullet capped at five).
* :func:`rank_discover` — significance × interest-affinity re-ranking with a
  recency (``-idx``) tie-break, exercised against a tiny on-disk fixture corpus
  so the per-episode KG loads run through the real loader (direct row
  construction would not exercise ``_episode_features``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server import app_discover_view
from podcast_scraper.server.app_discover_view import _significance, rank_discover
from podcast_scraper.server.corpus_catalog import (
    build_catalog_rows_cumulative,
    CatalogEpisodeRow,
)

pytestmark = [pytest.mark.unit]


def _row(*, has_gi: bool, has_kg: bool, bullets: tuple[str, ...]) -> CatalogEpisodeRow:
    """Minimal catalog row carrying only the fields ``_significance`` reads."""
    return CatalogEpisodeRow(
        metadata_relative_path="metadata/x.metadata.json",
        feed_id="f",
        feed_title=None,
        episode_id="e",
        episode_title="E",
        publish_date=None,
        summary_title=None,
        summary_bullets=bullets,
        summary_text=None,
        gi_relative_path="metadata/x.gi.json",
        kg_relative_path="metadata/x.kg.json",
        bridge_relative_path="metadata/x.bridge.json",
        has_gi=has_gi,
        has_kg=has_kg,
        has_bridge=False,
    )


# --------------------------------------------------------------------------- #
# _significance
# --------------------------------------------------------------------------- #


def test_significance_baseline_is_one() -> None:
    assert _significance(_row(has_gi=False, has_kg=False, bullets=())) == 1.0


def test_significance_gi_adds_two() -> None:
    assert _significance(_row(has_gi=True, has_kg=False, bullets=())) == 3.0


def test_significance_kg_adds_one() -> None:
    assert _significance(_row(has_gi=False, has_kg=True, bullets=())) == 2.0


def test_significance_bullets_add_point_two_each() -> None:
    row = _row(has_gi=False, has_kg=False, bullets=("a", "b", "c"))
    assert _significance(row) == pytest.approx(1.0 + 3 * 0.2)


def test_significance_bullets_capped_at_five() -> None:
    row = _row(has_gi=False, has_kg=False, bullets=tuple("abcdefghij"))  # 10 bullets
    # capped at 5 → 1.0 + 5 * 0.2 == 2.0; the extra five bullets are inert.
    assert _significance(row) == pytest.approx(2.0)


def test_significance_combines_all_weights() -> None:
    row = _row(has_gi=True, has_kg=True, bullets=("a", "b"))
    # 1 (base) + 2 (gi) + 1 (kg) + 2 * 0.2 (bullets) == 4.4
    assert _significance(row) == pytest.approx(4.4)


# --------------------------------------------------------------------------- #
# rank_discover (small on-disk fixture corpus)
# --------------------------------------------------------------------------- #


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    topics: list[tuple[str, str]],
    published: str,
    with_gi: bool = False,
    persons: list[tuple[str, str]] | None = None,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "myfeed", "title": "My Show", "url": "https://pod.example/feed.xml"},
        "episode": {
            "episode_id": episode_id,
            "title": f"Episode {episode_id}",
            "published_date": published,
            "duration_seconds": 1000,
        },
        "summary": {"title": "Sum", "bullets": ["a"]},
        "content": {"transcript_file_path": f"transcripts/{stem}.txt"},
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    (root / "transcripts" / f"{stem}.txt").write_text("hello", encoding="utf-8")
    nodes = [{"id": tid, "type": "Topic", "properties": {"label": label}} for tid, label in topics]
    nodes += [
        {"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in (persons or [])
    ]
    (root / "metadata" / f"{stem}.kg.json").write_text(
        json.dumps({"episode_id": episode_id, "nodes": nodes}), encoding="utf-8"
    )
    if with_gi:
        gi = {"episode_id": episode_id, "nodes": [], "edges": []}
        (root / "metadata" / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


def _corpus(root: Path) -> None:
    # epOld is older but about AI (+GI, +person Jane); epNew is newer but about Health.
    _write_episode(
        root,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
        with_gi=True,
        persons=[("person:jane", "Jane")],
    )
    _write_episode(
        root,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:health", "Health")],
        published="2024-06-01T00:00:00",
    )
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "AI",
                "member_count": 3,
                "members": [{"topic_id": "topic:ai", "label": "AI"}],
            },
            {
                "graph_compound_parent_id": "tc:health",
                "canonical_label": "Health",
                "member_count": 1,
                "members": [{"topic_id": "topic:health", "label": "Health"}],
            },
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _rows_newest_first(root: Path) -> list[CatalogEpisodeRow]:
    rows = build_catalog_rows_cumulative(root)
    titles = [r.episode_title for r in rows]
    assert titles == ["Episode new", "Episode old"]  # catalog is recency (newest-first)
    return rows


def test_empty_interests_is_recency_passthrough(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, [], rows, limit=10)
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def test_only_empty_string_tokens_collapse_to_recency(tmp_path: Path) -> None:
    # ``rank_discover`` keeps a token only when ``str(i)`` is truthy; an all-empty-string
    # interest list yields an empty set → the recency passthrough (no scoring at all).
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["", ""], rows, limit=10)
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def test_cluster_interest_reranks_matching_episode_first(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["tc:ai"], rows, limit=10)
    # epOld matches the followed cluster AND has GI → leads despite being older.
    assert [s.title for s in out] == ["Episode old", "Episode new"]


def test_topic_interest_reranks(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["topic:ai"], rows, limit=10)
    assert [s.title for s in out] == ["Episode old", "Episode new"]


def test_person_interest_reranks(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["person:jane"], rows, limit=10)
    assert [s.title for s in out] == ["Episode old", "Episode new"]


def test_unknown_prefix_token_grants_no_affinity(tmp_path: Path) -> None:
    # An unknown-prefix token (lands in cluster_interests, matches nothing) gives zero
    # affinity to BOTH episodes — so the order is pure significance, not interest-driven.
    # epOld carries GI (+2) and so leads on depth alone; flip it via two equal-depth rows
    # to prove the unknown token added no per-episode boost (order would be recency then).
    _write_episode(
        tmp_path,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        tmp_path,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )
    rows = build_catalog_rows_cumulative(tmp_path)
    assert [r.episode_title for r in rows] == ["Episode new", "Episode old"]
    out = rank_discover(tmp_path, ["genre:jazz"], rows, limit=10)
    # Equal significance + zero affinity for both → recency tie-break preserved.
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def test_recency_tie_break_keeps_equal_score_newest_first(tmp_path: Path) -> None:
    # Two episodes with identical depth (no GI, same bullets, both with KG) and an
    # interest that neither matches → equal score; the -idx tie-break must preserve
    # the incoming newest-first order.
    _write_episode(
        tmp_path,
        stem="0001-old",
        episode_id="old",
        topics=[("topic:ai", "AI")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        tmp_path,
        stem="0002-new",
        episode_id="new",
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )
    rows = build_catalog_rows_cumulative(tmp_path)
    assert [r.episode_title for r in rows] == ["Episode new", "Episode old"]
    # Interest no episode matches → both score == _significance (equal) → tie.
    out = rank_discover(tmp_path, ["person:nobody"], rows, limit=10)
    assert [s.title for s in out] == ["Episode new", "Episode old"]


def test_limit_truncates_after_ranking(tmp_path: Path) -> None:
    _corpus(tmp_path)
    rows = _rows_newest_first(tmp_path)
    out = rank_discover(tmp_path, ["tc:ai"], rows, limit=1)
    assert [s.title for s in out] == ["Episode old"]  # top-ranked survives the cap


def test_affinity_weight_constant_is_two() -> None:
    # Guards the documented "fully on-interest episode → (1 + AFFINITY_WEIGHT)x" contract.
    assert app_discover_view._AFFINITY_WEIGHT == 2.0
