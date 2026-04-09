#!/usr/bin/env python3
"""E2E tests for hybrid MAP-REDUCE provider (RFC-042, Issue #352).

Verifies HybridMLProvider with Tier 1 (LongT5 MAP + FLAN-T5 REDUCE via transformers)
using real models. Uses same pattern as test_ml_models_e2e for transformers.
"""

import sys
import tempfile
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

integration_dir = Path(__file__).parent.parent / "integration"
if str(integration_dir) not in sys.path:
    sys.path.insert(0, str(integration_dir))

from ml_model_cache_helpers import require_transformers_model_cached  # noqa: E402

from podcast_scraper import Config
from podcast_scraper.summarization.factory import create_summarization_provider

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: F401

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


@pytest.mark.e2e
@pytest.mark.ml_models
@pytest.mark.skipif(not TRANSFORMERS_AVAILABLE, reason="Transformers not available")
class TestHybridMLProviderE2E:
    """E2E tests for HybridMLProvider (Tier 1: LongT5 + FLAN-T5)."""

    def test_hybrid_ml_provider_summarize(self, e2e_server):
        """HybridMLProvider (Tier 1) summarizes text with real MAP + REDUCE models.

        Requires longt5-base and google/flan-t5-base to be cached (e.g. via preload).
        """
        require_transformers_model_cached("longt5-base")
        require_transformers_model_cached("google/flan-t5-base")

        fixture_root = Path(__file__).parent.parent / "fixtures"
        transcript_file = fixture_root / "transcripts" / "p01_e01_fast.txt"
        if not transcript_file.exists():
            pytest.skip(f"Transcript fixture not found: {transcript_file}")

        transcript_text = transcript_file.read_text(encoding="utf-8")

        with tempfile.TemporaryDirectory():
            cfg = Config(
                rss="",
                generate_metadata=True,
                generate_summaries=True,
                summary_provider="hybrid_ml",
                hybrid_map_model="longt5-base",
                hybrid_reduce_model="google/flan-t5-base",
                hybrid_reduce_backend="transformers",
                language="en",
            )

            provider = create_summarization_provider(cfg)
            provider.initialize()

            result = provider.summarize(
                text=transcript_text,
                episode_title="Test Episode",
            )

            assert isinstance(result, dict)
            assert "summary" in result
            assert isinstance(result["summary"], str)
            assert len(result["summary"]) > 0

            provider.cleanup()

    def test_hybrid_ml_provider_cleanup_is_idempotent(self, e2e_server):
        """Calling cleanup() twice after summarize does not raise."""
        require_transformers_model_cached("longt5-base")
        require_transformers_model_cached("google/flan-t5-base")

        fixture_root = Path(__file__).parent.parent / "fixtures"
        transcript_file = fixture_root / "transcripts" / "p01_e01_fast.txt"
        if not transcript_file.exists():
            pytest.skip(f"Transcript fixture not found: {transcript_file}")

        transcript_text = transcript_file.read_text(encoding="utf-8")

        cfg = Config(
            rss="",
            generate_metadata=True,
            generate_summaries=True,
            summary_provider="hybrid_ml",
            hybrid_map_model="longt5-base",
            hybrid_reduce_model="google/flan-t5-base",
            hybrid_reduce_backend="transformers",
            language="en",
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()
        provider.summarize(text=transcript_text, episode_title="Test Episode")

        provider.cleanup()
        provider.cleanup()

    def test_hybrid_ml_provider_result_has_expected_keys(self, e2e_server):
        """Summarize result dict contains 'summary' key with a non-empty string."""
        require_transformers_model_cached("longt5-base")
        require_transformers_model_cached("google/flan-t5-base")

        fixture_root = Path(__file__).parent.parent / "fixtures"
        transcript_file = fixture_root / "transcripts" / "p01_e01_fast.txt"
        if not transcript_file.exists():
            pytest.skip(f"Transcript fixture not found: {transcript_file}")

        transcript_text = transcript_file.read_text(encoding="utf-8")

        cfg = Config(
            rss="",
            generate_metadata=True,
            generate_summaries=True,
            summary_provider="hybrid_ml",
            hybrid_map_model="longt5-base",
            hybrid_reduce_model="google/flan-t5-base",
            hybrid_reduce_backend="transformers",
            language="en",
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        result = provider.summarize(
            text=transcript_text,
            episode_title="Test Episode",
        )

        assert "summary" in result
        assert isinstance(result["summary"], str)
        assert len(result["summary"].strip()) > 0

        provider.cleanup()

    def test_hybrid_ml_layered_cleaning_pattern_strategy_e2e(self, e2e_server):
        """Issue #419: workflow-style pattern clean + cleaning_hybrid_after_pattern in summarize."""
        require_transformers_model_cached("longt5-base")
        require_transformers_model_cached("google/flan-t5-base")

        from podcast_scraper.workflow.metadata_generation import (
            _hybrid_ml_layered_summarize_params,
        )

        fixture_root = Path(__file__).parent.parent / "fixtures"
        transcript_file = fixture_root / "transcripts" / "p01_e01_fast.txt"
        if not transcript_file.exists():
            pytest.skip(f"Transcript fixture not found: {transcript_file}")

        transcript_text = transcript_file.read_text(encoding="utf-8")

        cfg = Config(
            rss="",
            generate_metadata=True,
            generate_summaries=True,
            summary_provider="hybrid_ml",
            transcript_cleaning_strategy="pattern",
            hybrid_internal_preprocessing_after_pattern="cleaning_hybrid_after_pattern",
            hybrid_map_model="longt5-base",
            hybrid_reduce_model="google/flan-t5-base",
            hybrid_reduce_backend="transformers",
            language="en",
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        layered_params = _hybrid_ml_layered_summarize_params(cfg, provider)
        assert layered_params == {"preprocessing_profile": "cleaning_hybrid_after_pattern"}

        cleaned_text = provider.cleaning_processor.clean(transcript_text)
        result = provider.summarize(
            text=cleaned_text,
            episode_title="Layered cleaning E2E",
            params=layered_params,
        )

        assert result.get("metadata", {}).get("preprocessing_profile") == (
            "cleaning_hybrid_after_pattern"
        )
        assert isinstance(result.get("summary"), str)
        assert len((result.get("summary") or "").strip()) > 0

        provider.cleanup()

    def test_hybrid_ml_provider_extract_kg_graph_returns_none(self, e2e_server):
        """extract_kg_graph() returns None (not implemented for hybrid)."""
        require_transformers_model_cached("longt5-base")
        require_transformers_model_cached("google/flan-t5-base")

        cfg = Config(
            rss="",
            generate_metadata=True,
            generate_summaries=True,
            summary_provider="hybrid_ml",
            hybrid_map_model="longt5-base",
            hybrid_reduce_model="google/flan-t5-base",
            hybrid_reduce_backend="transformers",
            language="en",
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        result = provider.extract_kg_graph("some text")
        assert result is None

        provider.cleanup()

    def test_hybrid_ml_provider_generate_insights_returns_empty(self, e2e_server):
        """generate_insights() returns empty list (not implemented for hybrid)."""
        require_transformers_model_cached("longt5-base")
        require_transformers_model_cached("google/flan-t5-base")

        cfg = Config(
            rss="",
            generate_metadata=True,
            generate_summaries=True,
            summary_provider="hybrid_ml",
            hybrid_map_model="longt5-base",
            hybrid_reduce_model="google/flan-t5-base",
            hybrid_reduce_backend="transformers",
            language="en",
        )

        provider = create_summarization_provider(cfg)
        provider.initialize()

        result = provider.generate_insights("some text")
        assert result == []

        provider.cleanup()
