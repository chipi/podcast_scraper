"""Tests for GIL load layer (build_inspect_output, evidence span, find artifact)."""

import json
from pathlib import Path

import pytest

from podcast_scraper.gi import (
    build_artifact,
    build_inspect_output,
    find_artifact_by_episode_id,
    find_artifact_by_insight_id,
    load_artifact_and_transcript,
)
from podcast_scraper.gi.load import (
    _transcript_path_from_artifact_path,
    get_evidence_span,
    load_transcript_for_evidence,
)


@pytest.mark.unit
class TestGILLoad:
    """Load layer and build_inspect_output."""

    def test_transcript_path_from_artifact_path(self):
        """Transcript path is output_dir/transcripts/<base>.txt."""
        p = Path("/out/metadata/1 - episode.gi.json")
        t = _transcript_path_from_artifact_path(p)
        assert t == Path("/out/transcripts/1 - episode.txt")

    def test_get_evidence_span_excerpt(self):
        """Evidence span excerpt is transcript slice."""
        text = "Hello world here is the quote."
        span = get_evidence_span(text, 6, 11, transcript_ref="ep.txt")
        assert span.char_start == 6
        assert span.char_end == 11
        assert span.excerpt == "world"

    def test_get_evidence_span_out_of_range_excerpt_none(self):
        """When char_start/char_end are out of range, excerpt is None."""
        text = "hello"
        span = get_evidence_span(text, 0, 100, transcript_ref="ep.txt")
        assert span.char_start == 0
        assert span.char_end == 100
        assert span.excerpt is None

    def test_load_transcript_for_evidence_missing_returns_none(self, tmp_path):
        """load_transcript_for_evidence returns None when file is missing."""
        missing = tmp_path / "missing.txt"
        assert not missing.exists()
        assert load_transcript_for_evidence(missing) is None

    def test_load_transcript_for_evidence_existing_returns_content(self, tmp_path):
        """load_transcript_for_evidence returns file content when file exists."""
        path = tmp_path / "transcript.txt"
        path.write_text("Evidence here.", encoding="utf-8")
        assert load_transcript_for_evidence(path) == "Evidence here."

    def test_build_inspect_output_from_artifact(self):
        """build_inspect_output produces InspectOutput with insights and stats."""
        artifact = build_artifact("ep:1", "Some transcript.", prompt_version="v1")
        out = build_inspect_output(artifact, "Some transcript.")
        assert out.episode_id == "ep:1"
        assert len(out.insights) == 1
        assert out.insights[0].grounded is True
        assert out.stats["insight_count"] == 1
        assert out.stats["quote_count"] >= 1

    def test_load_artifact_and_transcript_roundtrip(self, tmp_path):
        """Load artifact from path; transcript optional."""
        payload = build_artifact("ep:1", "Evidence here.", prompt_version="v1")
        gi_path = tmp_path / "metadata" / "1_ep.gi.json"
        gi_path.parent.mkdir(parents=True)
        with open(gi_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        (tmp_path / "transcripts").mkdir()
        (tmp_path / "transcripts" / "1_ep.txt").write_text("Evidence here.", encoding="utf-8")
        artifact, transcript, tpath = load_artifact_and_transcript(
            gi_path, validate=True, load_transcript=True
        )
        assert artifact["episode_id"] == "ep:1"
        assert transcript == "Evidence here."
        assert tpath == tmp_path / "transcripts" / "1_ep.txt"

    def test_find_artifact_by_insight_id(self, tmp_path):
        """find_artifact_by_insight_id returns path to artifact containing insight."""
        payload = build_artifact("ep:1", "Evidence here.", prompt_version="v1")
        insight_id = "insight:ep:1:0"
        gi_path = tmp_path / "metadata" / "ep1.gi.json"
        gi_path.parent.mkdir(parents=True)
        with open(gi_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        found = find_artifact_by_insight_id(tmp_path, insight_id)
        assert found == gi_path
        assert find_artifact_by_insight_id(tmp_path, "nonexistent") is None

    def test_find_artifact_by_episode_id(self, tmp_path):
        """find_artifact_by_episode_id returns path to artifact with given episode_id."""
        payload = build_artifact("ep:1", "Evidence here.", prompt_version="v1")
        metadata = tmp_path / "metadata"
        metadata.mkdir(parents=True)
        gi_path = metadata / "ep1.gi.json"
        with open(gi_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        found = find_artifact_by_episode_id(tmp_path, "ep:1")
        assert found == gi_path
        assert find_artifact_by_episode_id(tmp_path, "ep:nonexistent") is None

    def test_find_artifact_by_episode_id_no_metadata_dir_returns_none(self, tmp_path):
        """find_artifact_by_episode_id returns None when metadata dir does not exist."""
        assert find_artifact_by_episode_id(tmp_path, "ep:1") is None
