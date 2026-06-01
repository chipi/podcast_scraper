"""Unit tests for the `index-two-tier` CLI handler (RFC-090 Phase 2 / B)."""

from __future__ import annotations

import logging

import pytest

from podcast_scraper.search import two_tier_indexer
from podcast_scraper.search.cli_handlers import (
    EXIT_INVALID_ARGS,
    EXIT_SUCCESS,
    parse_index_two_tier_argv,
    run_index_two_tier_cli,
)

pytestmark = pytest.mark.unit
log = logging.getLogger("test")


def test_parse_sets_command_and_defaults():
    args = parse_index_two_tier_argv(["--output-dir", "/c"])
    assert args.command == "index-two-tier"
    assert args.output_dir == "/c" and args.lance_path is None


def test_run_builds_and_reports(tmp_path, monkeypatch):
    captured = {}

    def _fake(corpus, lance, **k):
        captured["lance"] = str(lance)
        return two_tier_indexer.TwoTierIndexStats(episodes=2, segments=9, insights=4)

    monkeypatch.setattr(two_tier_indexer, "build_two_tier_index", _fake)
    args = parse_index_two_tier_argv(["--output-dir", str(tmp_path)])
    assert run_index_two_tier_cli(args, log) == EXIT_SUCCESS
    assert captured["lance"].endswith("search/lance_index")  # default co-located path


def test_run_requires_output_dir():
    args = parse_index_two_tier_argv(["--output-dir", "/c"])
    args.output_dir = None  # simulate missing
    assert run_index_two_tier_cli(args, log) == EXIT_INVALID_ARGS
