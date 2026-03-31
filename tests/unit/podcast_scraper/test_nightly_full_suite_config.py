"""Unit tests for nightly E2E production-style config (no full pipeline)."""

from __future__ import annotations

import unittest

from podcast_scraper import config
from tests.e2e.test_nightly_full_suite_e2e import create_nightly_config


class TestNightlyFullSuiteConfig(unittest.TestCase):
    """Nightly suite should mirror production ML defaults (not TEST_DEFAULT_*)."""

    def test_nightly_config_uses_prod_whisper_and_promoted_summary_mode(self) -> None:
        cfg = create_nightly_config("/tmp/nightly-out", "http://127.0.0.1:9/feed.xml")
        self.assertEqual(cfg.whisper_model, config.PROD_DEFAULT_WHISPER_MODEL)
        self.assertEqual(
            cfg.summary_mode_id,
            config.config_constants.PROD_DEFAULT_SUMMARY_MODE_ID,
        )
        self.assertEqual(cfg.summary_mode_precedence, "mode")
        self.assertIsNone(cfg.summary_model)
        self.assertIsNone(cfg.summary_reduce_model)
        # CI preloads en_core_web_sm only; prod NER (en_core_web_trf) is not cached in preload.
        self.assertEqual(cfg.ner_model, config.TEST_DEFAULT_NER_MODEL)
