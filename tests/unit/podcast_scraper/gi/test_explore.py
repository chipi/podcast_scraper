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
from podcast_scraper.gi.contracts import EvidenceSpan, InsightSummary, SupportingQuote
from podcast_scraper.gi.explore import (
    _insight_matches_topic,
    _topic_labels_for_insight,
    EXIT_NO_ARTIFACTS,
    EXIT_NO_RESULTS,
    EXIT_SUCCESS,
    explore_output_to_rfc_dict,
    map_uc4_question_to_params,
    run_uc4_semantic_qa,
    run_uc4_topic_leaderboard,
    sort_insights,
    topic_slug_for_rfc,
)
from podcast_scraper.gi.grounding import GroundedQuote
from podcast_scraper.gi.pipeline import _artifact_from_multi_insight


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
        """ExploreOutput has topic, insights, summary, episodes_searched, top_speakers."""
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
        assert out.summary["speaker_count"] == 0
        assert out.top_speakers == []

    def test_exit_codes_constants(self):
        """Exit code constants match RFC."""
        assert EXIT_SUCCESS == 0
        assert EXIT_NO_ARTIFACTS == 3
        assert EXIT_NO_RESULTS == 4

    def test_sort_insights_by_confidence_desc(self):
        """Higher confidence first; None last."""
        lo = InsightSummary(
            insight_id="i1",
            text="a",
            grounded=True,
            confidence=0.3,
            episode_id="e",
            supporting_quotes=[],
        )
        hi = InsightSummary(
            insight_id="i2",
            text="b",
            grounded=True,
            confidence=0.9,
            episode_id="e",
            supporting_quotes=[],
        )
        none_c = InsightSummary(
            insight_id="i3",
            text="c",
            grounded=False,
            confidence=None,
            episode_id="e",
            supporting_quotes=[],
        )
        out = sort_insights([lo, none_c, hi], sort_by="confidence")
        assert out[0].insight_id == "i2"
        assert out[1].insight_id == "i1"
        assert out[2].insight_id == "i3"

    def test_sort_insights_by_time_desc(self):
        """Newer publish_date first."""
        old = InsightSummary(
            insight_id="a",
            text="t",
            grounded=True,
            episode_id="e1",
            publish_date="2020-01-01T00:00:00Z",
            supporting_quotes=[],
        )
        new = InsightSummary(
            insight_id="b",
            text="t",
            grounded=True,
            episode_id="e2",
            publish_date="2025-06-01T12:00:00Z",
            supporting_quotes=[],
        )
        out = sort_insights([old, new], sort_by="time")
        assert out[0].insight_id == "b"

    def test_explore_output_to_rfc_dict_shape(self):
        """RFC payload includes topic object, episode nest, top_speakers."""
        ins = InsightSummary(
            insight_id="ins:1",
            text="Hello world",
            grounded=True,
            confidence=0.8,
            episode_id="ep:1",
            episode_title="Ep title",
            publish_date="2025-01-01T00:00:00Z",
            supporting_quotes=[],
        )
        from podcast_scraper.gi.explore import build_explore_output

        out = build_explore_output([ins], episodes_searched=3, topic="world")
        d = explore_output_to_rfc_dict(out)
        assert d["topic"]["label"] == "world"
        assert d["topic"]["topic_id"] == f"topic:{topic_slug_for_rfc('world')}"
        assert d["insights"][0]["episode"]["title"] == "Ep title"
        assert "top_speakers" in d
        assert d["episodes_searched"] == 3

    def test_map_uc4_question_to_params(self):
        """UC4 pattern maps to topic or speaker."""
        assert map_uc4_question_to_params("What insights about inflation?") == {
            "topic": "inflation"
        }
        assert map_uc4_question_to_params("What did Sam say?") == {"speaker": "Sam"}
        assert map_uc4_question_to_params("Show me insights about trade?") == {"topic": "trade"}
        assert map_uc4_question_to_params("Tell me about insights on climate") == {
            "topic": "climate"
        }
        assert map_uc4_question_to_params("What insights are there about tariffs?") == {
            "topic": "tariffs"
        }
        assert map_uc4_question_to_params("What are insights about rates?") == {"topic": "rates"}
        assert map_uc4_question_to_params("Top topics?") == {"topic_leaderboard": True}
        assert map_uc4_question_to_params("Which topics have the most insights?") == {
            "topic_leaderboard": True
        }
        assert map_uc4_question_to_params("What did Sam Altman say about innovation?") == {
            "speaker": "Sam Altman",
            "topic": "innovation",
        }

    def test_run_uc4_topic_leaderboard_untagged(self, tmp_path):
        """Leaderboard buckets untagged insights under (untagged)."""
        (tmp_path / "metadata").mkdir()
        write_artifact(
            tmp_path / "metadata" / "a.gi.json",
            build_artifact("ep:1", "T1", prompt_version="v1"),
            validate=True,
        )
        out = run_uc4_topic_leaderboard(tmp_path, "Top topics?", limit=10)
        assert out["answer"]["summary"]["episodes_searched"] == 1
        topics = out["answer"]["topics"]
        assert len(topics) >= 1
        assert topics[0]["topic_label"] == "(untagged)"
        assert topics[0]["insight_count"] >= 1

    def test_run_uc4_semantic_qa_compound_speaker_topic(self, tmp_path):
        """Compound UC4 question applies both filters."""
        (tmp_path / "metadata").mkdir()
        art = build_artifact("ep:1", "Nothing.", prompt_version="v1")
        for n in art["nodes"]:
            if n.get("type") == "Quote":
                n["properties"]["speaker_id"] = "SPEAKER_HOST"
                break
        write_artifact(tmp_path / "metadata" / "a.gi.json", art, validate=True)
        q = "What did HOST say about stub?"
        res = run_uc4_semantic_qa(tmp_path, q, limit=20)
        assert res is not None
        assert res["question"] == q
        assert "speaker" in res["explanation"].lower() or "topic" in res["explanation"].lower()

    def test_explore_output_to_rfc_dict_speaker_name_without_diarization_id(self):
        """RFC quote speaker uses graph name when speaker_id is absent."""
        ev = EvidenceSpan(transcript_ref="t.txt", char_start=0, char_end=3)
        q = SupportingQuote(
            quote_id="quote:1:0",
            text="Hi",
            speaker_id=None,
            speaker_name="Pat Example",
            evidence=ev,
        )
        ins = InsightSummary(
            insight_id="ins:1",
            text="Hello world",
            grounded=True,
            episode_id="ep:1",
            supporting_quotes=[q],
        )
        from podcast_scraper.gi.explore import build_explore_output

        out = build_explore_output([ins], episodes_searched=1)
        d = explore_output_to_rfc_dict(out)
        sp = d["insights"][0]["supporting_quotes"][0]["speaker"]
        assert sp["speaker_id"] == "Pat Example"
        assert sp["name"] == "Pat Example"

    def test_collect_insights_speaker_filter(self, tmp_path):
        """speaker= matches quote speaker_id substring."""
        artifact = build_artifact("ep:1", "Evidence here.", prompt_version="v1")
        for n in artifact["nodes"]:
            if n.get("type") == "Quote":
                n["properties"]["speaker_id"] = "SPEAKER_HOST"
                break
        (tmp_path / "metadata").mkdir()
        gi_path = tmp_path / "metadata" / "ep1.gi.json"
        write_artifact(gi_path, artifact, validate=True)
        loaded = load_artifacts([gi_path], validate=False)
        assert len(collect_insights(loaded, speaker="host")) == 1
        assert len(collect_insights(loaded, speaker="nobody")) == 0

    def test_collect_insights_speaker_filter_graph_name_only(self, tmp_path):
        """--speaker matches SupportingQuote.speaker_name from SPOKEN_BY when id cleared."""
        transcript = "First segment. Second segment."
        segments = [
            {"start": 0.0, "end": 1.5, "text": "First segment. ", "speaker": "Host"},
            {"start": 1.5, "end": 3.0, "text": "Second segment.", "speaker": "Guest"},
        ]
        gq = GroundedQuote(
            char_start=15,
            char_end=32,
            text="Second segment.",
            qa_score=0.9,
            nli_score=0.85,
        )
        out = _artifact_from_multi_insight(
            "ep:1",
            ["Insight"],
            [[gq]],
            model_version="m",
            prompt_version="v1",
            podcast_id="p",
            episode_title="T",
            date_str="2025-01-01T00:00:00Z",
            transcript_ref="t.txt",
            transcript_text=transcript,
            transcript_segments=segments,
        )
        for n in out["nodes"]:
            if n.get("type") == "Quote":
                n["properties"]["speaker_id"] = None
        (tmp_path / "metadata").mkdir()
        gi_path = tmp_path / "metadata" / "ep1.gi.json"
        write_artifact(gi_path, out, validate=True)
        loaded = load_artifacts([gi_path], validate=False)
        assert len(collect_insights(loaded, speaker="guest")) == 1
        assert len(collect_insights(loaded, speaker="host")) == 0
