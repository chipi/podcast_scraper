#!/usr/bin/env python3
"""Unit tests for HybridMLProvider and factory wiring (issue #352)."""

import unittest
from unittest.mock import MagicMock

import pytest

from podcast_scraper import config
from podcast_scraper.providers.ml.hybrid_ml_provider import LlamaCppReduceBackend
from podcast_scraper.providers.params import SummarizationParams
from podcast_scraper.summarization.factory import create_summarization_provider

pytestmark = [pytest.mark.unit, pytest.mark.module_summarization]


class TestHybridProviderFactory(unittest.TestCase):
    """Factory wiring for hybrid_ml provider."""

    def test_factory_creates_hybrid_provider_config_mode(self):
        """Config-based factory should create HybridMLProvider."""
        cfg = config.Config(rss_url="https://example.com/feed.xml", summary_provider="hybrid_ml")
        provider = create_summarization_provider(cfg)
        self.assertEqual(provider.__class__.__name__, "HybridMLProvider")

    def test_factory_hybrid_experiment_mode_creates_provider(self):
        """Experiment mode creates HybridMLProvider with minimal Config from params."""
        provider = create_summarization_provider("hybrid_ml")
        self.assertEqual(provider.__class__.__name__, "HybridMLProvider")
        self.assertEqual(provider.cfg.summary_provider, "hybrid_ml")
        self.assertEqual(provider.cfg.hybrid_reduce_backend, "transformers")

    def test_factory_hybrid_experiment_mode_with_params(self):
        """Experiment mode with explicit params sets hybrid_map_model, reduce_model, backend."""
        params = SummarizationParams(
            model_name="longt5-base",
            reduce_model="qwen2.5:7b",
            reduce_backend="ollama",
        )
        provider = create_summarization_provider("hybrid_ml", params)
        self.assertEqual(provider.__class__.__name__, "HybridMLProvider")
        self.assertEqual(provider.cfg.hybrid_map_model, "longt5-base")
        self.assertEqual(provider.cfg.hybrid_reduce_model, "qwen2.5:7b")
        self.assertEqual(provider.cfg.hybrid_reduce_backend, "ollama")


try:
    import llama_cpp  # noqa: F401

    HAS_LLAMA_CPP = True
except ImportError:
    HAS_LLAMA_CPP = False


class TestLlamaCppReduceBackend(unittest.TestCase):
    """Unit tests for LlamaCppReduceBackend (optional llama.cpp REDUCE)."""

    @unittest.skipIf(HAS_LLAMA_CPP, "llama_cpp installed; ImportError tested when not installed")
    def test_initialize_raises_import_error_when_llama_cpp_not_installed(self):
        """Without llama-cpp-python, initialize raises ImportError with install hint."""
        backend = LlamaCppReduceBackend(model_path="/fake/model.gguf", n_ctx=2048)
        with self.assertRaises(ImportError) as ctx:
            backend.initialize()
        self.assertIn("llama-cpp-python", str(ctx.exception))
        self.assertIn("podcast-scraper[llama]", str(ctx.exception))

    def test_reduce_returns_result_when_llm_mocked(self):
        """reduce() returns HybridReduceResult with text from llm output."""
        backend = LlamaCppReduceBackend(model_path="/fake/model.gguf", n_ctx=2048)
        mock_llm = MagicMock()
        mock_llm.return_value = {
            "choices": [{"text": "## Takeaways\n- One.\n\n## Outline\n- Section.\n"}],
        }
        backend._llm = mock_llm
        result = backend.reduce(
            notes="Some notes.",
            instruction="Summarize into markdown.",
            params={"max_new_tokens": 256},
        )
        self.assertEqual(result.backend, "llama_cpp")
        self.assertEqual(result.model, "/fake/model.gguf")
        self.assertIn("Takeaways", result.text)
        mock_llm.assert_called_once()
        call_kw = mock_llm.call_args[1]
        self.assertEqual(call_kw["max_tokens"], 256)
