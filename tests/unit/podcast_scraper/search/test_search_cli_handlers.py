"""Unit tests for search CLI handler (corpus search / lift_stats paths)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper.search import cli_handlers
from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.search.protocol import SearchResult

pytestmark = [pytest.mark.unit]


def _log() -> logging.Logger:
    return logging.getLogger("test_search_cli_handlers")


def test_run_search_cli_rejects_empty_query() -> None:
    args = argparse.Namespace(query=[], output_dir="/tmp/out")
    assert cli_handlers.run_search_cli(args, _log()) == cli_handlers.EXIT_INVALID_ARGS


def test_run_search_cli_requires_output_dir() -> None:
    args = argparse.Namespace(query=["hello"], output_dir=None)
    assert cli_handlers.run_search_cli(args, _log()) == cli_handlers.EXIT_INVALID_ARGS


@patch("podcast_scraper.search.corpus_search.run_corpus_search")
def test_run_search_cli_no_index(mock_run, tmp_path, caplog) -> None:
    mock_run.return_value = CorpusSearchOutcome(error="no_index", detail="/no/index")
    args = argparse.Namespace(
        query=["q"],
        output_dir=str(tmp_path),
        doc_type=None,
        feed=None,
        since=None,
        speaker=None,
        grounded_only=False,
        top_k=5,
        index_path=None,
        embedding_model=None,
        no_dedupe_kg_surfaces=False,
        format="pretty",
    )
    with caplog.at_level(logging.ERROR):
        code = cli_handlers.run_search_cli(args, _log())
    assert code == cli_handlers.EXIT_NO_ARTIFACTS
    joined = " ".join(r.message.lower() for r in caplog.records)
    assert "index" in joined or "vector" in joined


@patch("podcast_scraper.search.corpus_search.run_corpus_search")
def test_run_search_cli_json_includes_lift_stats(mock_run, tmp_path, capsys) -> None:
    mock_run.return_value = CorpusSearchOutcome(
        results=[{"score": 1.0, "metadata": {"doc_type": "x"}, "text": "t"}],
        lift_stats={"transcript_hits_returned": 1, "lift_applied": 0},
    )
    args = argparse.Namespace(
        query=["q"],
        output_dir=str(tmp_path),
        doc_type=None,
        feed=None,
        since=None,
        speaker=None,
        grounded_only=False,
        top_k=5,
        index_path=None,
        embedding_model=None,
        no_dedupe_kg_surfaces=False,
        format="json",
    )
    assert cli_handlers.run_search_cli(args, _log()) == cli_handlers.EXIT_SUCCESS
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["lift_stats"]["transcript_hits_returned"] == 1


def test_parse_search_argv_minimal(tmp_path) -> None:
    ns = cli_handlers.parse_search_argv(
        ["hello", "world", "--output-dir", str(tmp_path)],
    )
    assert ns.query == ["hello", "world"]
    assert ns.output_dir == str(tmp_path)


def test_parse_index_argv_requires_output_dir(tmp_path) -> None:
    ns = cli_handlers.parse_index_argv(["--output-dir", str(tmp_path)])
    assert ns.output_dir == str(tmp_path)


def test_resolve_index_dir_default_under_output(tmp_path: Path) -> None:
    p = cli_handlers._resolve_index_dir(tmp_path, None)
    assert p.name == "search"


def test_resolve_index_dir_custom_relative(tmp_path: Path) -> None:
    sub = tmp_path / "idx"
    sub.mkdir()
    p = cli_handlers._resolve_index_dir(tmp_path, "idx")
    assert p.resolve() == sub.resolve()


def test_parse_since_valid() -> None:
    from datetime import timezone

    dt = cli_handlers._parse_since("2024-06-01")
    assert dt is not None
    assert dt.tzinfo == timezone.utc
    assert dt.year == 2024


def test_parse_since_invalid() -> None:
    assert cli_handlers._parse_since("not-a-date") is None


def test_quotes_for_insight_supported_by() -> None:
    art = {
        "nodes": [
            {"id": "q1", "type": "Quote", "properties": {"text": "hi", "char_start": 0}},
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "ins1", "to": "q1"}],
    }
    qs = cli_handlers._quotes_for_insight(art, "ins1")
    assert len(qs) == 1
    assert qs[0]["quote_id"] == "q1"


def test_hit_passes_cli_feed_filter() -> None:
    hit = SearchResult(
        "d",
        1.0,
        {"doc_type": "insight", "feed_id": "my-pod", "episode_id": "e1"},
    )
    assert cli_handlers._hit_passes_cli_filters(
        hit,
        feed_substr="pod",
        since_dt=None,
        speaker_substr=None,
        grounded_only=False,
        gi_by_episode={},
    )
    assert not cli_handlers._hit_passes_cli_filters(
        hit,
        feed_substr="nomatch",
        since_dt=None,
        speaker_substr=None,
        grounded_only=False,
        gi_by_episode={},
    )


def test_insight_passes_speaker_filter() -> None:
    art = {
        "nodes": [
            {
                "id": "q1",
                "type": "Quote",
                "properties": {"speaker_id": "person:AliceJones"},
            },
        ],
        "edges": [{"type": "SUPPORTED_BY", "from": "i1", "to": "q1"}],
    }
    assert cli_handlers._insight_passes_speaker_filter(art, "i1", "alice") is True
    assert cli_handlers._insight_passes_speaker_filter(art, "i1", "zzz") is False


def test_insight_passes_topic_filter_id_match() -> None:
    """topic_key matches when the insight ABOUTs a topic whose id contains it."""
    art = {
        "nodes": [
            {
                "id": "topic:compute-governance",
                "type": "Topic",
                "properties": {"label": "Compute Governance"},
            },
        ],
        "edges": [{"type": "ABOUT", "from": "i1", "to": "topic:compute-governance"}],
    }
    assert cli_handlers._insight_passes_topic_filter(art, "i1", "compute") is True
    assert cli_handlers._insight_passes_topic_filter(art, "i1", "governance") is True
    assert cli_handlers._insight_passes_topic_filter(art, "i1", "climate") is False


def test_insight_passes_topic_filter_label_fallback() -> None:
    """When the id slug doesn't carry the term, match against the topic label."""
    art = {
        "nodes": [
            {"id": "topic:t2b7f9", "type": "Topic", "properties": {"label": "Silicon Supply"}},
        ],
        "edges": [{"type": "ABOUT", "from": "i2", "to": "topic:t2b7f9"}],
    }
    assert cli_handlers._insight_passes_topic_filter(art, "i2", "silicon") is True
    assert cli_handlers._insight_passes_topic_filter(art, "i2", "SUPPLY") is True
    assert cli_handlers._insight_passes_topic_filter(art, "i2", "nope") is False


def test_insight_passes_topic_filter_returns_false_when_no_about_edges() -> None:
    art = {"nodes": [{"id": "topic:x", "type": "Topic"}], "edges": []}
    assert cli_handlers._insight_passes_topic_filter(art, "i3", "x") is False


def test_hit_passes_cli_topic_filter_kg_topic() -> None:
    """kg_topic hits match on source_id (slug) and topic_label."""
    hit = SearchResult(
        "d",
        1.0,
        {
            "doc_type": "kg_topic",
            "source_id": "topic:compute-governance",
            "topic_label": "Compute Governance",
        },
    )
    assert cli_handlers._hit_passes_cli_filters(
        hit,
        feed_substr=None,
        since_dt=None,
        speaker_substr=None,
        topic_substr="compute",
        grounded_only=False,
        gi_by_episode={},
    )
    assert not cli_handlers._hit_passes_cli_filters(
        hit,
        feed_substr=None,
        since_dt=None,
        speaker_substr=None,
        topic_substr="climate",
        grounded_only=False,
        gi_by_episode={},
    )


def test_hit_passes_cli_topic_filter_drops_off_topic_doc_types() -> None:
    """Segments / quotes / aux rows are dropped when a topic filter is set (same
    shape as the speaker filter — those doc_types don't carry topic linkage)."""
    for dt in ("segment", "quote", "kg_entity", "summary"):
        hit = SearchResult("d", 1.0, {"doc_type": dt})
        assert not cli_handlers._hit_passes_cli_filters(
            hit,
            feed_substr=None,
            since_dt=None,
            speaker_substr=None,
            topic_substr="anything",
            grounded_only=False,
            gi_by_episode={},
        )


@patch("podcast_scraper.search.cli_handlers.index_corpus")
def test_run_index_cli_success(mock_index, tmp_path, caplog) -> None:
    from types import SimpleNamespace

    mock_index.return_value = SimpleNamespace(
        errors=[],
        episodes_scanned=3,
        episodes_skipped_unchanged=1,
        episodes_reindexed=2,
        vectors_upserted=10,
    )
    import logging

    args = argparse.Namespace(
        output_dir=str(tmp_path),
        vector_index_path=None,
        embedding_model=None,
        vector_index_types=None,
        stats=False,
        rebuild=False,
    )
    log = logging.getLogger("t_idx")
    with caplog.at_level(logging.INFO):
        code = cli_handlers.run_index_cli(args, log)
    assert code == cli_handlers.EXIT_SUCCESS
    mock_index.assert_called_once()


def test_run_index_cli_missing_output_dir() -> None:
    import logging

    args = argparse.Namespace(output_dir=None)
    code = cli_handlers.run_index_cli(args, logging.getLogger("t"))
    assert code == cli_handlers.EXIT_INVALID_ARGS


@patch("podcast_scraper.search.lance_index_stats.read_lance_index_stats")
def test_run_index_cli_stats_reads_index(mock_stats, tmp_path) -> None:
    from podcast_scraper.search.lance_index_stats import LanceIndexStats

    mock_stats.return_value = LanceIndexStats(
        total_vectors=5,
        doc_type_counts={"insight": 5},
        feeds_indexed=["f1"],
        embedding_model="m",
        embedding_provider="sentence_transformers",
        embedding_dim=4,
        last_updated="t",
        index_size_bytes=100,
    )

    import logging

    args = argparse.Namespace(
        output_dir=str(tmp_path),
        vector_index_path=None,
        embedding_model=None,
        vector_index_types=None,
        stats=True,
        rebuild=False,
    )
    code = cli_handlers.run_index_cli(args, logging.getLogger("t"))
    assert code == cli_handlers.EXIT_SUCCESS


@patch("podcast_scraper.search.corpus_search.run_corpus_search")
def test_run_search_cli_load_failed(mock_run, tmp_path, caplog) -> None:
    import logging

    mock_run.return_value = CorpusSearchOutcome(error="load_failed", detail="x")
    args = argparse.Namespace(
        query=["q"],
        output_dir=str(tmp_path),
        doc_type=None,
        feed=None,
        since=None,
        speaker=None,
        grounded_only=False,
        top_k=5,
        index_path=None,
        embedding_model=None,
        no_dedupe_kg_surfaces=False,
        format="pretty",
    )
    with caplog.at_level(logging.ERROR):
        code = cli_handlers.run_search_cli(args, logging.getLogger("t"))
    assert code == cli_handlers.EXIT_NO_ARTIFACTS


@patch("podcast_scraper.search.corpus_search.run_corpus_search")
def test_run_search_cli_embed_failed(mock_run, tmp_path, caplog) -> None:
    import logging

    mock_run.return_value = CorpusSearchOutcome(error="embed_failed", detail="e")
    args = argparse.Namespace(
        query=["q"],
        output_dir=str(tmp_path),
        doc_type=None,
        feed=None,
        since=None,
        speaker=None,
        grounded_only=False,
        top_k=5,
        index_path=None,
        embedding_model=None,
        no_dedupe_kg_surfaces=False,
        format="pretty",
    )
    with caplog.at_level(logging.ERROR):
        code = cli_handlers.run_search_cli(args, logging.getLogger("t"))
    assert code == cli_handlers.EXIT_INVALID_ARGS


@patch("podcast_scraper.search.corpus_search.run_corpus_search")
def test_run_search_cli_pretty_truncates_long_text(mock_run, tmp_path, capsys) -> None:
    long_t = "x" * 250
    mock_run.return_value = CorpusSearchOutcome(
        results=[
            {
                "doc_id": "d",
                "score": 0.5,
                "metadata": {"doc_type": "insight", "episode_id": "e"},
                "text": long_t,
                "supporting_quotes": [{"quote_id": "q"}],
            },
        ],
    )
    args = argparse.Namespace(
        query=["q"],
        output_dir=str(tmp_path),
        doc_type=None,
        feed=None,
        since=None,
        speaker=None,
        grounded_only=False,
        top_k=5,
        index_path=None,
        embedding_model=None,
        no_dedupe_kg_surfaces=False,
        format="pretty",
    )
    assert cli_handlers.run_search_cli(args, _log()) == cli_handlers.EXIT_SUCCESS
    out = capsys.readouterr().out
    assert "..." in out
    assert "quotes: 1" in out


def test_merged_episode_gi_paths_flat_metadata(tmp_path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    gi = tmp_path / "metadata" / "ep1.gi.json"
    gi.write_text("{}", encoding="utf-8")
    md = {
        "episode": {"episode_id": "ep1"},
        "grounded_insights": {"artifact_path": "metadata/ep1.gi.json"},
    }
    (meta / "x.metadata.json").write_text(json.dumps(md), encoding="utf-8")
    m = cli_handlers.merged_episode_gi_paths(tmp_path)
    assert m.get("ep1") == gi.resolve()
