"""Unit tests for search CLI handler (corpus search / lift_stats paths)."""

from __future__ import annotations

import argparse
import json
import logging
from unittest.mock import patch

import pytest

from podcast_scraper.search import cli_handlers
from podcast_scraper.search.corpus_search import CorpusSearchOutcome

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
