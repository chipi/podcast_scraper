#!/usr/bin/env python3
"""Additional unit tests for workflow.metadata_generation -- patch coverage.

Covers: _hybrid_ml_layered_summarize_params (pattern vs hybrid strategy,
non-HybridMLProvider, HybridMLProvider with pattern strategy).
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config

pytestmark = [pytest.mark.unit]


def _cfg(**kw):  # type: ignore[no-untyped-def]
    return config.Config(**{"rss_url": "https://a.example/rss", **kw})  # type: ignore[call-arg]


class TestHybridMlLayeredSummarizeParams(unittest.TestCase):
    """Cover _hybrid_ml_layered_summarize_params helper."""

    def _import_fn(self):  # type: ignore[no-untyped-def]
        from podcast_scraper.workflow.metadata_generation import (
            _hybrid_ml_layered_summarize_params,
        )

        return _hybrid_ml_layered_summarize_params

    def test_non_hybrid_provider_returns_empty(self) -> None:
        fn = self._import_fn()
        result = fn(_cfg(), Mock())
        self.assertEqual(result, {})

    def test_hybrid_provider_non_pattern_strategy(self) -> None:
        fn = self._import_fn()
        cfg = _cfg(transcript_cleaning_strategy="hybrid")
        mock_hybrid_cls = type("HybridMLProvider", (), {})
        mock_provider = mock_hybrid_cls()
        with patch(
            "podcast_scraper.providers.ml.hybrid_ml_provider" ".HybridMLProvider",
            mock_hybrid_cls,
        ):
            result = fn(cfg, mock_provider)
        self.assertEqual(result, {})

    def test_hybrid_provider_pattern_strategy(self) -> None:
        fn = self._import_fn()
        cfg = _cfg(transcript_cleaning_strategy="pattern")
        mock_hybrid_cls = type("HybridMLProvider", (), {})
        mock_provider = mock_hybrid_cls()
        with patch(
            "podcast_scraper.providers.ml.hybrid_ml_provider" ".HybridMLProvider",
            mock_hybrid_cls,
        ):
            result = fn(cfg, mock_provider)
        self.assertIn("preprocessing_profile", result)
