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

    def test_speaker_attribution_survives_artifact_file_roundtrip(self, tmp_path):
        """Diarized speaker attribution survives a write -> read -> validate roundtrip.

        Drives the *real* multi-insight grounding path (``_artifact_from_multi_insight``)
        rather than the stub path: a grounded quote whose char range lands in a speaker's
        segment must produce a ``Person`` node + ``SPOKEN_BY`` edge, named by the human
        ``speaker_label`` (not the raw ``SPEAKER_00`` id), and that attribution must
        round-trip through disk intact. The old test ran build_artifact's stub path,
        which by design never attributes (its single quote spans both speakers), so its
        ``if speaker_nodes:`` block silently never executed — a hollow assertion.
        """
        from podcast_scraper.gi.grounding import GroundedQuote
        from podcast_scraper.gi.pipeline import _artifact_from_multi_insight

        transcript = "Alice says hello. Bob says goodbye."
        segments = [
            {"text": "Alice says hello. ", "speaker": "SPEAKER_00", "speaker_label": "Alice"},
            {"text": "Bob says goodbye.", "speaker": "SPEAKER_01", "speaker_label": "Bob"},
        ]
        gq = GroundedQuote(
            char_start=18,
            char_end=34,
            text="Bob says goodbye.",
            qa_score=0.9,
            nli_score=0.85,
        )
        payload = _artifact_from_multi_insight(
            "episode:spk-test",
            [("Insight", "unknown")],
            [[gq]],
            model_version="test-model",
            prompt_version="v1",
            podcast_id="podcast:p",
            episode_title="Speaker Episode",
            date_str="2025-03-01T00:00:00Z",
            transcript_ref="t.txt",
            transcript_text=transcript,
            transcript_segments=segments,
        )
        # Person node + SPOKEN_BY exist pre-write, named by the human label not the id.
        person_nodes = [n for n in payload["nodes"] if n["type"] == "Person"]
        assert {n["properties"]["name"] for n in person_nodes} == {"Bob"}

        path = tmp_path / "spk.gi.json"
        write_artifact(path, payload, validate=True)
        read_back = read_artifact(path)
        validate_artifact(read_back, strict=True)

        rt_persons = [n for n in read_back["nodes"] if n["type"] == "Person"]
        assert {n["properties"]["name"] for n in rt_persons} == {"Bob"}
        quote = next(n for n in read_back["nodes"] if n["type"] == "Quote")
        assert quote["properties"]["speaker_id"] == "person:bob"
        spoken_by = [e for e in read_back["edges"] if e["type"] == "SPOKEN_BY"]
        assert len(spoken_by) == 1
        assert spoken_by[0]["from"] == quote["id"]
        assert spoken_by[0]["to"] == rt_persons[0]["id"]

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
