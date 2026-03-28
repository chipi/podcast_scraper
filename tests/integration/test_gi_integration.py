"""Integration tests for GIL pipeline (Issue #356).

Build artifact from transcript fixture, write to file, read back, validate.
No ML models required.
"""

import pytest

from podcast_scraper.gi import build_artifact, read_artifact, validate_artifact, write_artifact


@pytest.mark.integration
class TestGILArtifactIntegration:
    """End-to-end artifact write/read/validate."""

    def test_transcript_to_artifact_to_file_roundtrip(self, tmp_path):
        """Transcript -> build_artifact -> write -> read -> validate."""
        transcript = "This is a short transcript for GIL integration test."
        episode_id = "episode:test-1"
        path = tmp_path / "test.gi.json"
        payload = build_artifact(
            episode_id,
            transcript,
            model_version="test-model",
            prompt_version="v1",
            episode_title="Integration Episode",
            publish_date="2025-02-01T00:00:00Z",
        )
        validate_artifact(payload, strict=False)
        write_artifact(path, payload, validate=True)
        assert path.exists()
        read_back = read_artifact(path)
        validate_artifact(read_back, strict=False)
        assert read_back["episode_id"] == episode_id
        assert read_back["model_version"] == "test-model"
        assert any(
            n["type"] == "Quote" and transcript[:20] in n["properties"]["text"]
            for n in read_back["nodes"]
        )
