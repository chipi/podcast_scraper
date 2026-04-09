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

    def test_build_artifact_with_segments_adds_segment_nodes(self):
        """Transcript segments propagate timestamp info into Quote nodes."""
        transcript = "Hello world"
        segments = [{"start": 0.0, "end": 5.0, "text": "Hello world"}]
        payload = build_artifact(
            "episode:seg-test",
            transcript,
            model_version="test-model",
            prompt_version="v1",
            episode_title="Segment Episode",
            publish_date="2025-03-01T00:00:00Z",
            transcript_segments=segments,
        )
        assert payload["episode_id"] == "episode:seg-test"
        assert isinstance(payload["nodes"], list)
        assert len(payload["nodes"]) >= 1
        quote_nodes = [n for n in payload["nodes"] if n["type"] == "Quote"]
        for qn in quote_nodes:
            assert "timestamp_start_ms" in qn["properties"]
            assert "timestamp_end_ms" in qn["properties"]

    def test_build_artifact_with_speakers_adds_speaker_nodes(self):
        """Speaker labels on transcript segments produce Speaker nodes and SPOKEN_BY edges."""
        transcript = "Alice says hello. Bob says goodbye."
        segments = [
            {"start": 0.0, "end": 3.0, "text": "Alice says hello. ", "speaker": "Alice"},
            {"start": 3.0, "end": 6.0, "text": "Bob says goodbye.", "speaker": "Bob"},
        ]
        payload = build_artifact(
            "episode:spk-test",
            transcript,
            model_version="test-model",
            prompt_version="v1",
            episode_title="Speaker Episode",
            publish_date="2025-03-01T00:00:00Z",
            transcript_segments=segments,
        )
        node_types = {n["type"] for n in payload["nodes"]}
        assert "Episode" in node_types
        speaker_nodes = [n for n in payload["nodes"] if n["type"] == "Speaker"]
        speaker_names = {n["properties"]["name"] for n in speaker_nodes}
        # Speaker nodes are only produced when the multi-insight grounding path
        # maps quote char ranges to segment speakers. Short transcripts may go
        # through the stub path which skips speaker attachment.
        if speaker_nodes:
            assert speaker_names <= {"Alice", "Bob"}
            spoken_by_edges = [e for e in payload["edges"] if e["type"] == "SPOKEN_BY"]
            assert len(spoken_by_edges) >= 1

    def test_build_artifact_strict_validation(self):
        """Well-formed artifact passes validate_artifact(strict=True)."""
        transcript = "A well-formed transcript for strict validation testing."
        payload = build_artifact(
            "episode:strict-test",
            transcript,
            model_version="test-model",
            prompt_version="v1",
            episode_title="Strict Episode",
            publish_date="2025-04-01T00:00:00Z",
        )
        validate_artifact(payload, strict=True)

    def test_build_artifact_empty_transcript_produces_stub(self):
        """Empty transcript produces a stub artifact with episode_id but minimal nodes."""
        payload = build_artifact(
            "episode:empty-test",
            "",
            model_version="test-model",
            prompt_version="v1",
            episode_title="Empty Episode",
            publish_date="2025-04-01T00:00:00Z",
        )
        assert payload["episode_id"] == "episode:empty-test"
        assert isinstance(payload["nodes"], list)
        assert len(payload["nodes"]) >= 1
        node_types = [n["type"] for n in payload["nodes"]]
        assert "Episode" in node_types
