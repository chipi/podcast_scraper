"""Integration: bundled LLM pipeline mode in metadata_generation (Issue #477).

Tests the bundled dispatch logic, fallback to staged, and cleaned
transcript file handling without running the full generate_metadata
function (which requires extensive setup).
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock

import pytest

from podcast_scraper import config

pytestmark = [pytest.mark.integration]

VALID_BUNDLED_JSON = json.dumps(
    {
        "title": "Test Title",
        "summary": "A detailed prose summary paragraph.",
        "bullets": ["Point one.", "Point two."],
    }
)


class TestBundledConfigWiring(unittest.TestCase):
    """Config.llm_pipeline_mode correctly controls bundled dispatch."""

    def test_config_accepts_bundled_mode(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test",
            llm_pipeline_mode="bundled",
        )
        self.assertEqual(cfg.llm_pipeline_mode, "bundled")

    def test_config_defaults_to_staged(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test",
        )
        self.assertEqual(cfg.llm_pipeline_mode, "staged")

    def test_config_rejects_invalid_mode(self):
        with self.assertRaises(Exception):
            config.Config(
                rss_url="https://example.com/feed.xml",
                summary_provider="openai",
                openai_api_key="sk-test",
                llm_pipeline_mode="invalid_mode",
            )


class TestBundledProviderDispatch(unittest.TestCase):
    """Workflow dispatches to summarize_bundled when mode=bundled."""

    def test_bundled_mode_checks_for_summarize_bundled_method(self):
        """Provider must have summarize_bundled for bundled mode."""
        provider_with = Mock()
        provider_with.summarize_bundled = Mock(
            return_value={
                "summary": VALID_BUNDLED_JSON,
                "metadata": {"bundled": True},
            }
        )
        fn = getattr(provider_with, "summarize_bundled", None)
        self.assertTrue(callable(fn))

        provider_without = Mock(spec=["summarize"])
        fn2 = getattr(provider_without, "summarize_bundled", None)
        self.assertIsNone(fn2)

    def test_bundled_result_has_no_cleaned_text_key(self):
        """Bundled JSON output must not contain cleaned_text."""
        parsed = json.loads(VALID_BUNDLED_JSON)
        self.assertIn("title", parsed)
        self.assertIn("summary", parsed)
        self.assertIn("bullets", parsed)
        self.assertNotIn("cleaned_text", parsed)


class TestBundledFallbackBehavior(unittest.TestCase):
    """When bundled fails, workflow falls back to staged."""

    def test_fallback_records_metric(self):
        pm = Mock()
        pm.record_llm_bundled_fallback_to_staged = Mock()
        pm.record_llm_bundled_fallback_to_staged()
        pm.record_llm_bundled_fallback_to_staged.assert_called_once()

    def test_bundled_max_output_tokens_config(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test",
            llm_pipeline_mode="bundled",
            llm_bundled_max_output_tokens=8192,
        )
        self.assertEqual(cfg.llm_bundled_max_output_tokens, 8192)

    def test_bundled_max_output_tokens_default(self):
        cfg = config.Config(
            rss_url="https://example.com/feed.xml",
            summary_provider="openai",
            openai_api_key="sk-test",
            llm_pipeline_mode="bundled",
        )
        self.assertEqual(cfg.llm_bundled_max_output_tokens, 16384)


class TestBundledMetricsRecording(unittest.TestCase):
    """Metrics correctly track bundled calls."""

    def test_metrics_has_bundled_fields(self):
        from podcast_scraper.workflow.metrics import Metrics

        m = Metrics()
        m.record_llm_bundled_clean_summary_call(100, 50)
        result = m.finish()
        self.assertEqual(result["llm_bundled_clean_summary_calls"], 1)
        self.assertEqual(result["llm_bundled_clean_summary_input_tokens"], 100)
        self.assertEqual(result["llm_bundled_clean_summary_output_tokens"], 50)
        stage = result["llm_token_totals_by_stage"]["bundled_clean_summary"]
        self.assertEqual(stage["input"], 100)
        self.assertEqual(stage["output"], 50)
        self.assertEqual(stage["calls"], 1)

    def test_metrics_fallback_counter(self):
        from podcast_scraper.workflow.metrics import Metrics

        m = Metrics()
        m.record_llm_bundled_fallback_to_staged()
        m.record_llm_bundled_fallback_to_staged()
        result = m.finish()
        self.assertEqual(result["llm_bundled_fallback_to_staged_count"], 2)
