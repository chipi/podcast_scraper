"""Unit tests for search CLI handler (corpus search / lift_stats paths)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper.search import cli_handlers
from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.search.faiss_store import VECTORS_FILE
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
        vector_faiss_index_mode=None,
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


@patch("podcast_scraper.search.cli_handlers.FaissVectorStore.load")
def test_run_index_cli_stats_reads_index(mock_load, tmp_path) -> None:
    from types import SimpleNamespace

    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / VECTORS_FILE).write_bytes(b"")

    st = SimpleNamespace(
        total_vectors=5,
        doc_type_counts={"insight": 5},
        feeds_indexed=["f1"],
        embedding_model="m",
        embedding_dim=4,
        last_updated="t",
        index_size_bytes=100,
    )
    store = MagicMock()
    store.stats.return_value = st
    mock_load.return_value = store

    import logging

    args = argparse.Namespace(
        output_dir=str(tmp_path),
        vector_index_path=None,
        embedding_model=None,
        vector_faiss_index_mode=None,
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
