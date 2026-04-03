"""Integration tests for KG artifacts (write/read/validate, no ML)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper.kg import (
    build_artifact,
    read_artifact,
    scan_kg_artifact_paths,
    validate_artifact,
    write_artifact,
)
from podcast_scraper.kg.quality_metrics import compute_kg_quality_metrics


@pytest.mark.integration
class TestKGArtifactIntegration:
    """End-to-end kg.json write/read/validate."""

    def test_transcript_to_artifact_to_file_roundtrip(self, tmp_path):
        """Transcript -> build_artifact -> write -> read -> validate."""
        transcript = "Short transcript mentioning ACME Corp and inflation."
        episode_id = "episode:kg-int-1"
        path = tmp_path / "test.kg.json"
        payload = build_artifact(
            episode_id,
            transcript,
            podcast_id="podcast:test",
            episode_title="KG Integration Episode",
            publish_date="2025-02-01T00:00:00Z",
            topic_label="Inflation",
            detected_hosts=["Host One"],
        )
        validate_artifact(payload, strict=False)
        write_artifact(path, payload, validate=True)
        assert path.exists()
        read_back = read_artifact(path)
        validate_artifact(read_back, strict=False)
        assert read_back["episode_id"] == episode_id
        assert any(n.get("type") == "Episode" for n in read_back["nodes"])
        assert any(n.get("type") == "Topic" for n in read_back["nodes"])

    def test_build_artifact_with_summary_bullets(self):
        """summary_bullets source with topic_labels → Topic nodes derived from bullets."""
        cfg = MagicMock()
        cfg.kg_extraction_source = "summary_bullets"
        cfg.kg_max_topics = 10
        cfg.kg_max_entities = 15
        cfg.kg_merge_pipeline_entities = True
        cfg.kg_extraction_model = None

        payload = build_artifact(
            "episode:bullets-1",
            "Some transcript text about markets.",
            podcast_id="podcast:test",
            episode_title="Bullets Episode",
            publish_date="2025-03-01T00:00:00Z",
            topic_labels=["Topic A discussed", "Topic B mentioned"],
            detected_hosts=["Host One"],
            cfg=cfg,
        )
        validate_artifact(payload, strict=False)
        topic_nodes = [n for n in payload["nodes"] if n.get("type") == "Topic"]
        assert len(topic_nodes) == 2
        topic_labels = {n["properties"]["label"] for n in topic_nodes}
        assert "Topic A discussed" in topic_labels
        assert "Topic B mentioned" in topic_labels

    def test_validate_artifact_strict_mode(self):
        """build_artifact → validate_artifact(strict=True) passes without error."""
        payload = build_artifact(
            "episode:strict-1",
            "Transcript for strict validation.",
            podcast_id="podcast:test",
            episode_title="Strict Validation Episode",
            publish_date="2025-04-01T00:00:00Z",
            topic_label="Economics",
            detected_hosts=["Host A"],
        )
        validate_artifact(payload, strict=True)

    def test_quality_metrics_on_artifact(self, tmp_path):
        """build_artifact → write → compute_kg_quality_metrics returns expected keys."""
        payload = build_artifact(
            "episode:qm-1",
            "Transcript for quality metrics.",
            podcast_id="podcast:test",
            episode_title="Quality Metrics Episode",
            publish_date="2025-05-01T00:00:00Z",
            topic_label="Technology",
            detected_hosts=["Host Q"],
        )
        path = tmp_path / "qm.kg.json"
        write_artifact(path, payload, validate=True)

        metrics = compute_kg_quality_metrics([tmp_path])
        result = metrics.to_dict()
        expected_keys = {
            "artifact_paths",
            "total_nodes",
            "total_edges",
            "avg_nodes_per_artifact",
            "avg_edges_per_artifact",
            "artifacts_with_extraction",
            "extraction_coverage",
            "errors",
        }
        assert expected_keys.issubset(result.keys())
        assert result["artifact_paths"] == 1
        assert result["total_nodes"] >= 1
        assert result["total_edges"] >= 0
        assert result["extraction_coverage"] == 1.0

    def test_corpus_scan_finds_artifacts(self, tmp_path):
        """Write two KG artifacts → scan_kg_artifact_paths finds both."""
        for i in range(1, 3):
            payload = build_artifact(
                f"episode:scan-{i}",
                f"Transcript number {i}.",
                podcast_id="podcast:test",
                episode_title=f"Scan Episode {i}",
                publish_date="2025-06-01T00:00:00Z",
                topic_label="Finance",
                detected_hosts=[],
            )
            path = tmp_path / "metadata" / f"ep{i}.kg.json"
            write_artifact(path, payload, validate=True)

        found = scan_kg_artifact_paths(tmp_path)
        assert len(found) == 2
        assert all(p.name.endswith(".kg.json") for p in found)

    def test_build_artifact_empty_transcript_produces_stub(self):
        """Empty transcript with stub source → valid artifact with Episode node."""
        cfg = MagicMock()
        cfg.kg_extraction_source = "stub"
        cfg.kg_max_topics = 10
        cfg.kg_max_entities = 15
        cfg.kg_merge_pipeline_entities = True
        cfg.kg_extraction_model = None

        payload = build_artifact(
            "episode:empty-1",
            "",
            podcast_id="podcast:test",
            episode_title="Empty Transcript Episode",
            publish_date="2025-07-01T00:00:00Z",
            topic_label=None,
            detected_hosts=[],
            cfg=cfg,
        )
        validate_artifact(payload, strict=False)
        assert payload["episode_id"] == "episode:empty-1"
        assert payload["extraction"]["model_version"] == "stub"
        assert any(n.get("type") == "Episode" for n in payload["nodes"])
        topic_nodes = [n for n in payload["nodes"] if n.get("type") == "Topic"]
        assert len(topic_nodes) == 0
