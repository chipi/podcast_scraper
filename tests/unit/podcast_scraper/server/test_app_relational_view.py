"""Unit tests for :mod:`podcast_scraper.server.app_relational_view` (#1095/#1096/#1097).

Exercise the KG-grounded person/topic card projections and the entity resolver directly
against a tiny on-disk fixture corpus (no TestClient) so the corpus-scan + co-occurrence
counting + cluster enrichment paths are covered by the unit suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server.app_relational_view import (
    build_person_card,
    build_topic_card,
    resolve_entity,
)

pytestmark = [pytest.mark.unit]


def _write_episode(
    root: Path,
    *,
    stem: str,
    episode_id: str,
    persons: list[tuple[str, str]],
    topics: list[tuple[str, str]],
    published: str = "2024-03-10T00:00:00",
    write_kg: bool = True,
    corrupt_kg: bool = False,
) -> None:
    """Write one episode (metadata + KG with the given person/topic nodes)."""
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
    if not write_kg:
        return
    if corrupt_kg:
        (root / "metadata" / f"{stem}.kg.json").write_text("{not json", encoding="utf-8")
        return
    nodes = [{"id": pid, "type": "Person", "properties": {"name": name}} for pid, name in persons]
    nodes += [{"id": tid, "type": "Topic", "properties": {"label": label}} for tid, label in topics]
    kg = {"episode_id": episode_id, "nodes": nodes}
    (root / "metadata" / f"{stem}.kg.json").write_text(json.dumps(kg), encoding="utf-8")


def _write_clusters(root: Path) -> None:
    (root / "search").mkdir(parents=True, exist_ok=True)
    payload = {
        "clusters": [
            {
                "graph_compound_parent_id": "tc:ai",
                "canonical_label": "Artificial Intelligence",
                "members": [
                    {"topic_id": "topic:ai", "label": "AI"},
                    {"topic_id": "topic:ml", "label": "Machine Learning"},
                ],
            }
        ]
    }
    (root / "search" / "topic_clusters.json").write_text(json.dumps(payload), encoding="utf-8")


def _two_episode_corpus(root: Path) -> None:
    _write_episode(
        root,
        stem="0001-a",
        episode_id="ep1",
        persons=[("person:jane-doe", "Jane Doe"), ("person:bob", "Bob")],
        topics=[("topic:ai", "AI"), ("topic:ml", "Machine Learning")],
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        root,
        stem="0002-b",
        episode_id="ep2",
        persons=[("person:jane-doe", "Jane Doe"), ("person:carol", "Carol")],
        topics=[("topic:ai", "AI")],
        published="2024-06-01T00:00:00",
    )


def test_build_person_card_aggregates_and_excludes_self(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    card = build_person_card(tmp_path, "person:jane-doe")
    assert card is not None
    assert card.label == "Jane Doe"
    assert card.episode_count == 2
    # newest-first ordering: ep2 (June) before ep1 (Jan).
    assert [e.title for e in card.episodes] == ["Episode ep2", "Episode ep1"]
    assert {p.id for p in card.related_people} == {"person:bob", "person:carol"}
    assert {t.id for t in card.related_topics} == {"topic:ai", "topic:ml"}


def test_build_person_card_topics_carry_cluster_info(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    _write_clusters(tmp_path)
    card = build_person_card(tmp_path, "person:jane-doe")
    assert card is not None
    ai = next(t for t in card.related_topics if t.id == "topic:ai")
    assert ai.cluster_id == "tc:ai"
    assert ai.cluster_label == "Artificial Intelligence"
    assert ai.cluster_size == 2


def test_build_person_card_unknown_returns_none(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    assert build_person_card(tmp_path, "person:nobody") is None


def test_build_topic_card_episodes_siblings_people(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    _write_clusters(tmp_path)
    card = build_topic_card(tmp_path, "topic:ai")
    assert card is not None
    assert card.label == "AI"
    assert card.cluster_id == "tc:ai"
    assert card.cluster_size == 2
    assert card.episode_count == 2
    assert {s.id for s in card.sibling_topics} == {"topic:ml"}
    assert {p.id for p in card.related_people} == {
        "person:jane-doe",
        "person:bob",
        "person:carol",
    }


def test_build_topic_card_without_clusters_has_no_siblings(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    card = build_topic_card(tmp_path, "topic:ml")
    assert card is not None
    assert card.episode_count == 1
    assert card.sibling_topics == []
    assert card.cluster_id is None
    assert card.cluster_size == 0


def test_build_topic_card_unknown_returns_none(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    assert build_topic_card(tmp_path, "topic:nope") is None


def test_iter_kg_skips_episodes_without_or_with_unreadable_kg(tmp_path: Path) -> None:
    # ep with no KG (skipped via has_kg) + ep with corrupt KG (skipped via None artifact).
    _write_episode(
        tmp_path,
        stem="0001-nokg",
        episode_id="nokg",
        persons=[],
        topics=[],
        write_kg=False,
    )
    _write_episode(
        tmp_path,
        stem="0002-bad",
        episode_id="bad",
        persons=[("person:x", "X")],
        topics=[],
        corrupt_kg=True,
    )
    _write_episode(
        tmp_path,
        stem="0003-ok",
        episode_id="ok",
        persons=[("person:jane", "Jane")],
        topics=[("topic:t", "T")],
    )
    # Only the readable episode contributes; the person from the corrupt KG is invisible.
    assert build_person_card(tmp_path, "person:x") is None
    card = build_person_card(tmp_path, "person:jane")
    assert card is not None and card.episode_count == 1


def test_resolve_entity_person_exact_and_near_exact(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    exact = resolve_entity(tmp_path, "Jane Doe")
    assert exact is not None
    assert (exact.id, exact.kind) == ("person:jane-doe", "person")
    near = resolve_entity(tmp_path, "jane-doe")  # punctuation/case-insensitive
    assert near is not None and near.id == "person:jane-doe"


def test_resolve_entity_topic(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    ref = resolve_entity(tmp_path, "machine learning")
    assert ref is not None
    assert (ref.id, ref.kind) == ("topic:ml", "topic")


def test_resolve_entity_blank_and_no_match_return_none(tmp_path: Path) -> None:
    _two_episode_corpus(tmp_path)
    assert resolve_entity(tmp_path, "   ") is None  # normalizes to empty
    assert resolve_entity(tmp_path, "quantum chromodynamics") is None


def test_resolve_entity_prefers_person_over_topic_on_collision(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        stem="0001-c",
        episode_id="ep1",
        persons=[("person:focus", "Focus")],
        topics=[("topic:focus", "Focus")],
    )
    ref = resolve_entity(tmp_path, "focus")
    assert ref is not None
    assert ref.kind == "person" and ref.id == "person:focus"


def test_card_builders_accept_precomputed_rows(tmp_path: Path) -> None:
    # Passing rows explicitly skips the internal catalog build (the `rows is not None` branch).
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    _two_episode_corpus(tmp_path)
    rows = build_catalog_rows_cumulative(tmp_path)
    pcard = build_person_card(tmp_path, "person:jane-doe", rows=rows)
    assert pcard is not None and pcard.episode_count == 2
    tcard = build_topic_card(tmp_path, "topic:ai", rows=rows)
    assert tcard is not None and tcard.episode_count == 2
    assert resolve_entity(tmp_path, "Bob", rows=rows) is not None
