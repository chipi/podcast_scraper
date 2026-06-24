"""#1058 chunk 3 — unit tests for ``kg.topic_clustering``.

Exercises gather → cluster → apply with a stub embedder so no real
model load happens. The stub returns hand-crafted vectors so cluster
membership is deterministic and the cosine cutoff is testable
explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from podcast_scraper.kg.topic_clustering import (
    apply_concept_topics_to_corpus,
    cluster_and_apply_corpus_topics,
    cluster_topics,
    ClusteringSummary,
    gather_corpus_topics,
    slug_for_concept,
    TopicCluster,
    TopicMention,
)

pytestmark = pytest.mark.unit


def _stub_embedder(label_to_vec: Dict[str, List[float]]):
    def _embed(labels: List[str]) -> List[List[float]]:
        return [label_to_vec[label] for label in labels]

    return _embed


def _write_kg(corpus: Path, podcast_id: str, episode_id: str, topics: List[str]) -> Path:
    """Write a minimal KG artifact with the given Topic labels under a
    feed/run dir tree shaped like the real corpus."""
    feed = podcast_id.replace("podcast:", "")
    run = episode_id.replace("episode:", "run-")
    target = corpus / "feeds" / feed / run / "metadata"
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{episode_id.replace(':', '_')}.kg.json"
    data = {
        "schema_version": "2.0",
        "model_version": "stub",
        "prompt_version": "v1",
        "episode_id": episode_id,
        "extraction": {
            "model_version": "stub",
            "extracted_at": "2024-01-01T00:00:00Z",
            "transcript_ref": "transcript.txt",
        },
        "nodes": [
            {
                "id": episode_id,
                "type": "Episode",
                "properties": {
                    "podcast_id": podcast_id,
                    "title": "T",
                    "publish_date": "2024-01-01T00:00:00Z",
                },
            },
            {
                "id": podcast_id,
                "type": "Podcast",
                "properties": {"title": podcast_id.split(":")[-1]},
            },
        ],
        "edges": [],
    }
    for i, label in enumerate(topics):
        topic_id = f"topic:{episode_id}-t{i}"
        data["nodes"].append(
            {
                "id": topic_id,
                "type": "Topic",
                "properties": {"label": label},
            }
        )
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class TestSlugForConcept:
    def test_basic_lowercase(self) -> None:
        assert slug_for_concept("AI Safety") == "ai-safety"

    def test_collapses_punctuation(self) -> None:
        assert slug_for_concept("AI: alignment / safety!") == "ai-alignment-safety"

    def test_empty_input(self) -> None:
        assert slug_for_concept("") == ""
        assert slug_for_concept("   ") == ""


class TestGatherCorpusTopics:
    def test_collects_topics_with_episode_and_podcast_ids(self, tmp_path: Path) -> None:
        _write_kg(tmp_path, "podcast:show-a", "episode:e1", ["AI safety", "energy"])
        _write_kg(tmp_path, "podcast:show-b", "episode:e2", ["AI alignment"])

        mentions = gather_corpus_topics(tmp_path)

        labels = {m.label for m in mentions}
        assert labels == {"AI safety", "energy", "AI alignment"}
        for m in mentions:
            assert m.episode_id.startswith("episode:")
            assert m.podcast_id is not None
            assert m.podcast_id.startswith("podcast:")
            assert m.kg_path.is_file()

    def test_empty_corpus_returns_empty(self, tmp_path: Path) -> None:
        assert gather_corpus_topics(tmp_path) == []

    def test_skips_malformed_kg_files(self, tmp_path: Path) -> None:
        bad = tmp_path / "feeds" / "broken" / "metadata"
        bad.mkdir(parents=True)
        (bad / "broken.kg.json").write_text("{ not valid", encoding="utf-8")
        _write_kg(tmp_path, "podcast:p", "episode:e", ["topic"])
        assert len(gather_corpus_topics(tmp_path)) == 1


class TestClusterTopics:
    def _mention(
        self, episode_id: str, label: str, podcast_id: Optional[str] = None
    ) -> TopicMention:
        return TopicMention(
            episode_id=episode_id,
            podcast_id=podcast_id or f"podcast:{episode_id.split(':')[-1]}",
            label=label,
            topic_id=f"topic:{episode_id}-{label}",
            kg_path=Path("/fake.kg.json"),
        )

    def test_returns_empty_for_no_topics(self) -> None:
        assert cluster_topics([], _stub_embedder({}), threshold=0.5) == []

    def test_clusters_two_similar_labels_from_different_episodes(self) -> None:
        mentions = [
            self._mention("episode:1", "AI safety"),
            self._mention("episode:2", "AI alignment"),
        ]
        # Both label vectors point in the same direction → cosine 1.0
        # → above threshold → cluster.
        embedder = _stub_embedder({"AI safety": [1.0, 0.0], "AI alignment": [1.0, 0.0]})
        clusters = cluster_topics(mentions, embedder, threshold=0.75)
        assert len(clusters) == 1
        assert {m.episode_id for m in clusters[0].members} == {"episode:1", "episode:2"}

    def test_rejects_cluster_below_min_episodes(self) -> None:
        """Single-episode cluster with min_episodes=2 must drop."""
        mentions = [self._mention("episode:1", "lone topic")]
        embedder = _stub_embedder({"lone topic": [1.0, 0.0]})
        assert cluster_topics(mentions, embedder, threshold=0.5, min_episodes=2) == []

    def test_does_not_cluster_when_cosine_below_threshold(self) -> None:
        """Orthogonal vectors (cosine 0) must stay in separate clusters
        even with min_episodes=1 — so the only clusters returned span
        ≥ min_episodes distinct episodes."""
        mentions = [
            self._mention("episode:1", "tech"),
            self._mention("episode:2", "cooking"),
        ]
        embedder = _stub_embedder({"tech": [1.0, 0.0], "cooking": [0.0, 1.0]})
        clusters = cluster_topics(mentions, embedder, threshold=0.5, min_episodes=2)
        # Two singleton clusters; neither meets min_episodes=2.
        assert clusters == []

    def test_canonical_label_is_most_frequent(self) -> None:
        mentions = [
            self._mention("episode:1", "AI safety"),
            self._mention("episode:2", "AI safety"),
            self._mention("episode:3", "AI alignment"),
        ]
        embedder = _stub_embedder(
            {
                "AI safety": [1.0, 0.0],
                "AI alignment": [1.0, 0.0],
            }
        )
        clusters = cluster_topics(mentions, embedder, threshold=0.5)
        assert clusters[0].canonical_label == "AI safety"
        assert clusters[0].concept_id == "concept:topic-ai-safety"

    def test_three_label_cluster_one_concept_id(self) -> None:
        mentions = [
            self._mention("episode:1", "energy policy"),
            self._mention("episode:2", "clean energy"),
            self._mention("episode:3", "renewable energy"),
        ]
        embedder = _stub_embedder(
            {
                "energy policy": [1.0, 0.0],
                "clean energy": [1.0, 0.0],
                "renewable energy": [1.0, 0.0],
            }
        )
        clusters = cluster_topics(mentions, embedder, threshold=0.5)
        assert len(clusters) == 1
        assert clusters[0].episode_count == 3

    def test_embedder_size_mismatch_raises(self) -> None:
        mentions = [self._mention("episode:1", "a"), self._mention("episode:2", "b")]
        bad_embedder = lambda labels: [[1.0]]  # noqa: E731
        with pytest.raises(ValueError):
            cluster_topics(mentions, bad_embedder)


class TestApplyConceptTopicsToCorpus:
    def _kg_with_topic(
        self, tmp_path: Path, episode_id: str, podcast_id: str, label: str
    ) -> tuple[Path, str]:
        path = _write_kg(tmp_path, podcast_id, episode_id, [label])
        data = json.loads(path.read_text())
        topic_id = next(n["id"] for n in data["nodes"] if n.get("type") == "Topic")
        return path, topic_id

    def test_adds_concept_topic_node_and_related_to_edge(self, tmp_path: Path) -> None:
        p1, t1 = self._kg_with_topic(tmp_path, "episode:e1", "podcast:a", "AI safety")
        p2, t2 = self._kg_with_topic(tmp_path, "episode:e2", "podcast:b", "AI alignment")

        cluster = TopicCluster(
            canonical_label="AI safety",
            concept_id="concept:topic-ai-safety",
            members=[
                TopicMention("episode:e1", "podcast:a", "AI safety", t1, p1),
                TopicMention("episode:e2", "podcast:b", "AI alignment", t2, p2),
            ],
        )
        summary = apply_concept_topics_to_corpus([cluster])

        assert isinstance(summary, ClusteringSummary)
        assert summary.concept_topics_added == 2  # one per source artifact
        assert summary.related_to_edges_added == 2
        assert summary.artifacts_mutated == 2

        for path in (p1, p2):
            data = json.loads(path.read_text())
            concept_node = next(
                n for n in data["nodes"] if n.get("id") == "concept:topic-ai-safety"
            )
            assert concept_node["type"] == "Topic"
            assert concept_node["properties"]["is_concept"] is True
            related_edges = [e for e in data["edges"] if e["type"] == "RELATED_TO"]
            assert len(related_edges) == 1
            assert related_edges[0]["to"] == "concept:topic-ai-safety"

    def test_idempotent_on_second_apply(self, tmp_path: Path) -> None:
        p1, t1 = self._kg_with_topic(tmp_path, "episode:e1", "podcast:a", "AI safety")
        p2, t2 = self._kg_with_topic(tmp_path, "episode:e2", "podcast:b", "AI alignment")
        cluster = TopicCluster(
            canonical_label="AI safety",
            concept_id="concept:topic-ai-safety",
            members=[
                TopicMention("episode:e1", "podcast:a", "AI safety", t1, p1),
                TopicMention("episode:e2", "podcast:b", "AI alignment", t2, p2),
            ],
        )
        apply_concept_topics_to_corpus([cluster])
        # Re-apply with the same input — nothing new should land.
        summary2 = apply_concept_topics_to_corpus([cluster])
        assert summary2.concept_topics_added == 0
        assert summary2.related_to_edges_added == 0
        assert summary2.artifacts_mutated == 0

    def test_dry_run_does_not_touch_disk(self, tmp_path: Path) -> None:
        p1, t1 = self._kg_with_topic(tmp_path, "episode:e1", "podcast:a", "AI safety")
        p2, t2 = self._kg_with_topic(tmp_path, "episode:e2", "podcast:b", "AI alignment")
        before_1 = p1.read_text()
        before_2 = p2.read_text()
        cluster = TopicCluster(
            canonical_label="AI safety",
            concept_id="concept:topic-ai-safety",
            members=[
                TopicMention("episode:e1", "podcast:a", "AI safety", t1, p1),
                TopicMention("episode:e2", "podcast:b", "AI alignment", t2, p2),
            ],
        )
        summary = apply_concept_topics_to_corpus([cluster], write=False)
        assert summary.artifacts_mutated == 2  # would have mutated
        assert p1.read_text() == before_1  # but didn't
        assert p2.read_text() == before_2


class TestClusterAndApplyCorpusTopics:
    def test_end_to_end_empty_corpus(self, tmp_path: Path) -> None:
        summary = cluster_and_apply_corpus_topics(tmp_path, embedder=_stub_embedder({}))
        assert summary.clusters_found == 0
        assert summary.concept_topics_added == 0

    def test_end_to_end_one_cross_show_cluster(self, tmp_path: Path) -> None:
        _write_kg(tmp_path, "podcast:show-a", "episode:e1", ["AI safety"])
        _write_kg(tmp_path, "podcast:show-b", "episode:e2", ["AI alignment"])
        _write_kg(tmp_path, "podcast:show-c", "episode:e3", ["cooking"])

        embedder = _stub_embedder(
            {
                "AI safety": [1.0, 0.0],
                "AI alignment": [1.0, 0.0],
                "cooking": [0.0, 1.0],
            }
        )
        summary = cluster_and_apply_corpus_topics(
            tmp_path, embedder=embedder, threshold=0.75, min_episodes=2
        )
        assert summary.clusters_found == 1  # only the AI cluster
        # Two source artifacts mutated; each gets the concept-Topic
        # node + a RELATED_TO edge.
        assert summary.artifacts_mutated == 2
        assert summary.concept_topics_added == 2
        assert summary.related_to_edges_added == 2
