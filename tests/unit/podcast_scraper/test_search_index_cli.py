"""Unit tests for search/index CLI parsing and main() dispatch."""

from __future__ import annotations

import tempfile
import unittest

from podcast_scraper import cli


class TestSearchIndexSubcommand(unittest.TestCase):
    """Test search and index subcommands."""

    def test_parse_args_search(self) -> None:
        args = cli.parse_args(["search", "hello", "world", "--output-dir", "/tmp/x"])
        self.assertEqual(args.command, "search")
        self.assertEqual(args.query, ["hello", "world"])

    def test_parse_args_index_stats(self) -> None:
        args = cli.parse_args(["index", "--output-dir", "/out", "--stats"])
        self.assertEqual(args.command, "index")
        self.assertTrue(args.stats)

    def test_main_search_missing_index(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            code = cli.main(["search", "q", "--output-dir", d])
        self.assertEqual(code, 3)

    def test_main_index_stats_missing(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            code = cli.main(["index", "--output-dir", d, "--stats"])
        self.assertEqual(code, 3)
