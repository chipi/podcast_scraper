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
