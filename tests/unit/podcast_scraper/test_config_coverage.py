#!/usr/bin/env python3
"""Additional unit tests for config.py -- patch coverage for new validators.

Covers: _normalize_multi_rss_input, _coerce_rss_urls_list,
_validate_rss_urls_entries, _normalize_episode_order, _coerce_episode_offset,
_multi_feed_requires_output_dir, episode_since/episode_until cross-validation.
"""

from __future__ import annotations

import unittest
from datetime import date

import pytest

from podcast_scraper.config import Config

pytestmark = [pytest.mark.unit]

# Pydantic alias + populate_by_name=True: mypy plugin doesn't recognise
# the field name ``rss_url`` (only the alias ``rss``), so we suppress
# [call-arg] on Config() calls that use the field name.
_CFG = {"rss_url": "https://a.example/rss"}


def _cfg(**kw):  # type: ignore[no-untyped-def]
    """Shortcut that merges defaults and suppresses mypy alias noise."""
    return Config(**{**_CFG, **kw})  # type: ignore[call-arg]


class TestMultiRssInput(unittest.TestCase):
    """Cover _normalize_multi_rss_input pre-validator."""

    def test_rss_list_promoted_to_rss_urls(self) -> None:
        cfg = _cfg(
            rss_urls=["https://a.example/rss", "https://b.example/rss"],
            output_dir="/tmp/out",
        )
        assert cfg.rss_urls is not None
        self.assertIn("https://a.example/rss", cfg.rss_urls)
        self.assertIn("https://b.example/rss", cfg.rss_urls)

    def test_rss_urls_strips_whitespace(self) -> None:
        cfg = _cfg(
            rss_urls=["  https://a.example/rss  ", "https://b.example/rss"],
            output_dir="/tmp/out",
        )
        assert cfg.rss_urls is not None
        self.assertTrue(
            all(u == u.strip() for u in cfg.rss_urls),
            "URLs should be stripped of whitespace",
        )

    def test_rss_urls_none_stays_none(self) -> None:
        cfg = _cfg()
        self.assertIsNone(cfg.rss_urls)


class TestRssUrlsValidation(unittest.TestCase):
    """Cover _coerce_rss_urls_list and _validate_rss_urls_entries."""

    def test_invalid_scheme_raises(self) -> None:
        with self.assertRaises(Exception) as ctx:
            _cfg(rss_urls=["ftp://bad.example/rss"])
        self.assertIn("http", str(ctx.exception).lower())

    def test_missing_netloc_raises(self) -> None:
        with self.assertRaises(Exception) as ctx:
            _cfg(rss_urls=["https:///no-host"])
        self.assertIn("hostname", str(ctx.exception).lower())

    def test_empty_list_coerced_to_none(self) -> None:
        cfg = _cfg(rss_urls=[])
        self.assertIsNone(cfg.rss_urls)


class TestEpisodeOrder(unittest.TestCase):
    """Cover _normalize_episode_order validator."""

    def test_default_newest(self) -> None:
        self.assertEqual(_cfg().episode_order, "newest")

    def test_oldest_accepted(self) -> None:
        cfg = _cfg(episode_order="oldest")
        self.assertEqual(cfg.episode_order, "oldest")

    def test_case_insensitive(self) -> None:
        cfg = _cfg(episode_order="NEWEST")  # type: ignore[arg-type]
        self.assertEqual(cfg.episode_order, "newest")

    def test_invalid_raises(self) -> None:
        with self.assertRaises(Exception) as ctx:
            _cfg(episode_order="random")  # type: ignore[arg-type]
        self.assertIn("newest", str(ctx.exception).lower())


class TestEpisodeOffset(unittest.TestCase):
    """Cover _coerce_episode_offset validator."""

    def test_default_zero(self) -> None:
        self.assertEqual(_cfg().episode_offset, 0)

    def test_positive_accepted(self) -> None:
        self.assertEqual(_cfg(episode_offset=5).episode_offset, 5)

    def test_negative_raises(self) -> None:
        with self.assertRaises(Exception) as ctx:
            _cfg(episode_offset=-1)
        self.assertIn("non-negative", str(ctx.exception).lower())

    def test_string_coerced(self) -> None:
        cfg = _cfg(episode_offset="3")  # type: ignore[arg-type]
        self.assertEqual(cfg.episode_offset, 3)

    def test_non_numeric_raises(self) -> None:
        with self.assertRaises(Exception):
            _cfg(episode_offset="abc")  # type: ignore[arg-type]


class TestMultiFeedRequiresOutputDir(unittest.TestCase):
    """Cover _multi_feed_requires_output_dir model validator."""

    def test_single_feed_no_output_dir_ok(self) -> None:
        self.assertIsNotNone(_cfg())

    def test_multi_feed_without_output_dir_raises(self) -> None:
        with self.assertRaises(Exception) as ctx:
            _cfg(
                rss_urls=[
                    "https://a.example/rss",
                    "https://b.example/rss",
                ],
            )
        self.assertIn("output_dir", str(ctx.exception).lower())

    def test_multi_feed_with_output_dir_ok(self) -> None:
        cfg = _cfg(
            rss_urls=[
                "https://a.example/rss",
                "https://b.example/rss",
            ],
            output_dir="/tmp/corpus",
        )
        self.assertIsNotNone(cfg)


class TestEpisodeDateCrossValidation(unittest.TestCase):
    """Cover episode_since > episode_until validation."""

    def test_since_before_until_ok(self) -> None:
        cfg = _cfg(
            episode_since=date(2024, 1, 1),
            episode_until=date(2024, 12, 31),
        )
        self.assertIsNotNone(cfg)

    def test_since_equals_until_ok(self) -> None:
        cfg = _cfg(
            episode_since=date(2024, 6, 15),
            episode_until=date(2024, 6, 15),
        )
        self.assertIsNotNone(cfg)

    def test_since_after_until_raises(self) -> None:
        with self.assertRaises(Exception) as ctx:
            _cfg(
                episode_since=date(2025, 1, 1),
                episode_until=date(2024, 1, 1),
            )
        self.assertIn("episode_since", str(ctx.exception))
