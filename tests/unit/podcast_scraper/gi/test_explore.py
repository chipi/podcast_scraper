"""Tests for GIL cross-episode explore (scan, collect, build_explore_output)."""

import pytest

from podcast_scraper.gi import (
    build_artifact,
    build_explore_output,
    collect_insights,
    load_artifacts,
    scan_artifact_paths,
    write_artifact,
)
from podcast_scraper.gi.explore import (
    _insight_matches_topic,
    _topic_labels_for_insight,
    EXIT_NO_ARTIFACTS,
    EXIT_NO_RESULTS,
    EXIT_SUCCESS,
)


@pytest.mark.unit
class TestExploreScan:
    """Scan and load artifacts."""

    def test_scan_artifact_paths_empty_dir(self, tmp_path):
        """Empty dir returns no paths."""
        assert scan_artifact_paths(tmp_path) == []

    def test_scan_artifact_paths_metadata_gi_json(self, tmp_path):
        """Finds metadata/*.gi.json."""
        (tmp_path / "metadata").mkdir()
        p = tmp_path / "metadata" / "ep1.gi.json"
        p.write_text("{}", encoding="utf-8")
        found = scan_artifact_paths(tmp_path)
        assert p in found

    def test_load_artifacts_skips_invalid(self, tmp_path):
        """Invalid artifact is skipped with warning."""
        (tmp_path / "metadata").mkdir()
        (tmp_path / "metadata" / "bad.gi.json").write_text("{", encoding="utf-8")
        paths = list((tmp_path / "metadata").glob("*.gi.json"))
        loaded = load_artifacts(paths, validate=False)
        assert len(loaded) == 0


@pytest.mark.unit
class TestExploreTopicMatch:
    """Topic label and text matching."""

    def test_topic_labels_for_insight_about_edge(self):
        """ABOUT edge from insight to Topic yields label."""
        artifact = {
            "nodes": [
                {"id": "insight:ep:0", "type": "Insight", "properties": {"text": "x"}},
                {
                    "id": "topic:ai",
                    "type": "Topic",
                    "properties": {"label": "AI Regulation"},
                },
            ],
            "edges": [{"type": "ABOUT", "from": "insight:ep:0", "to": "topic:ai"}],
        }
        labels = _topic_labels_for_insight(artifact, "insight:ep:0")
        assert "AI Regulation" in labels

    def test_insight_matches_topic_substring_in_text(self):
        """Topic substring in insight text matches."""
        artifact = {}
        assert (
            _insight_matches_topic(
                artifact, "i:1", "AI regulation will lag behind innovation", "regulation"
            )
            is True
        )
        assert (
            _insight_matches_topic(artifact, "i:1", "Weather today is nice", "regulation") is False
        )

    def test_insight_matches_topic_none_always_true(self):
        """No topic filter matches all."""
        assert _insight_matches_topic({}, "i:1", "Any text", None) is True


@pytest.mark.unit
class TestExploreCollectAndOutput:
    """Collect insights and build ExploreOutput."""

    def test_collect_insights_from_artifact(self, tmp_path):
        """Collect insights from one artifact."""
        artifact = build_artifact("ep:1", "Evidence here.", prompt_version="v1")
        (tmp_path / "metadata").mkdir()
        gi_path = tmp_path / "metadata" / "ep1.gi.json"
        write_artifact(gi_path, artifact, validate=True)
        loaded = load_artifacts([gi_path], validate=False)
        insights = collect_insights(loaded, topic=None, limit=10)
        assert len(insights) == 1
        assert insights[0].episode_id == "ep:1"
        assert insights[0].grounded is True

    def test_collect_insights_grounded_only(self, tmp_path):
        """grounded_only filters ungrounded."""
        artifact = build_artifact("ep:1", "Text.", prompt_version="v1")
        # Stub has grounded=True; force ungrounded by editing
        for n in artifact["nodes"]:
            if n.get("type") == "Insight":
                n["properties"]["grounded"] = False
                break
        (tmp_path / "metadata").mkdir()
        gi_path = tmp_path / "metadata" / "ep1.gi.json"
        write_artifact(gi_path, artifact, validate=False)
        loaded = load_artifacts([gi_path], validate=False)
        insights = collect_insights(loaded, grounded_only=True)
        assert len(insights) == 0

    def test_build_explore_output_shape(self):
        """ExploreOutput has topic, insights, summary, episodes_searched."""
        from podcast_scraper.gi.contracts import InsightSummary

        insights = [
            InsightSummary(
                insight_id="i:1",
                text="One",
                grounded=True,
                episode_id="ep:1",
                supporting_quotes=[],
            ),
        ]
        out = build_explore_output(insights, episodes_searched=1, topic="AI")
        assert out.topic == "AI"
        assert out.episodes_searched == 1
        assert out.summary["insight_count"] == 1
        assert out.summary["grounded_insight_count"] == 1

    def test_exit_codes_constants(self):
        """Exit code constants match RFC."""
        assert EXIT_SUCCESS == 0
        assert EXIT_NO_ARTIFACTS == 3
        assert EXIT_NO_RESULTS == 4
