#!/usr/bin/env python3
"""GitHub #539: multi-feed ML batch defers shared ML singleton cleanup."""

import unittest
from unittest.mock import Mock

import pytest

from podcast_scraper import config
from podcast_scraper.workflow import orchestration as orch
from podcast_scraper.workflow.orchestration import _cleanup_providers
from podcast_scraper.workflow.stages.setup import _configs_match_for_ml_preload


def _sample_ml_config(
    *, rss_url: str, output_dir: str, whisper_model: str = "tiny.en"
) -> config.Config:
    return config.Config(
        rss=rss_url,
        gemini_api_key="test-key-placeholder",
        transcription_provider="whisper",
        speaker_detector_provider="spacy",
        summary_provider="transformers",
        whisper_model=whisper_model,
        whisper_device="cpu",
        summary_model="bart-small",
        summary_device="cpu",
        summary_reduce_model="long-fast",
        ner_model="en_core_web_sm",
        transcribe_missing=True,
        auto_speakers=True,
        generate_summaries=True,
        generate_metadata=True,
        preload_models=True,
        output_dir=output_dir,
    )


@pytest.mark.unit
class TestMultiFeedMlCleanup539(unittest.TestCase):
    """Deferral skips per-feed cleanup for the shared ML preload singleton."""

    def tearDown(self) -> None:
        try:
            orch.end_multi_feed_ml_batch()
        except Exception:
            pass
        with orch._preloaded_ml_provider_lock:
            orch._preloaded_ml_provider = None

    def test_cleanup_skips_shared_ml_while_deferred_then_finalizes(self) -> None:
        shared = Mock()
        orch.begin_multi_feed_ml_batch()
        try:
            with orch._preloaded_ml_provider_lock:
                orch._preloaded_ml_provider = shared
            tr = Mock()
            tr.transcription_provider = shared
            _cleanup_providers(tr, shared)
            self.assertEqual(shared.cleanup.call_count, 0)
        finally:
            orch.end_multi_feed_ml_batch()
        self.assertEqual(shared.cleanup.call_count, 1)

    def test_cleanup_runs_when_not_deferred(self) -> None:
        shared = Mock()
        with orch._preloaded_ml_provider_lock:
            orch._preloaded_ml_provider = shared
        try:
            tr = Mock()
            tr.transcription_provider = shared
            _cleanup_providers(tr, shared)
            self.assertGreaterEqual(shared.cleanup.call_count, 1)
        finally:
            with orch._preloaded_ml_provider_lock:
                orch._preloaded_ml_provider = None


@pytest.mark.unit
class TestMlPreloadFingerprint539(unittest.TestCase):
    """ML preload fingerprint ignores per-feed-only config fields."""

    def test_same_ml_different_output_dir_matches(self) -> None:
        a = _sample_ml_config(rss_url="https://a.example/feed.xml", output_dir="out/feeds/a")
        b = a.model_copy(update={"output_dir": "out/feeds/b", "rss": "https://b.example/feed.xml"})
        self.assertTrue(_configs_match_for_ml_preload(a, b))

    def test_whisper_model_change_no_match(self) -> None:
        a = _sample_ml_config(rss_url="https://a.example/feed.xml", output_dir="out/a")
        b = a.model_copy(update={"whisper_model": "base.en"})
        self.assertFalse(_configs_match_for_ml_preload(a, b))
