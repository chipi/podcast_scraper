"""Integration tests for KG artifacts (write/read/validate, no ML)."""

from __future__ import annotations

import pytest

from podcast_scraper.kg import build_artifact, read_artifact, validate_artifact, write_artifact


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
