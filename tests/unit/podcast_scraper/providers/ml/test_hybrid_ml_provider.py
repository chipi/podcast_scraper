"""Unit tests for hybrid_ml_provider (MAP/REDUCE backends and HybridMLProvider)."""

from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch

import pytest

from podcast_scraper import Config
from podcast_scraper.providers.ml.hybrid_ml_provider import (
    HybridMLProvider,
    HybridReduceResult,
    OllamaReduceBackend,
    TransformersReduceBackend,
)

pytestmark = [pytest.mark.unit]


class TestTransformersReduceBackend(unittest.TestCase):
    def test_reduce_raises_when_not_initialized(self) -> None:
        backend = TransformersReduceBackend(
            model_name="google/flan-t5-small",
            device="cpu",
            cache_dir=None,
        )
        with self.assertRaises(RuntimeError) as ctx:
            backend.reduce("n", "i")
        self.assertIn("initialize()", str(ctx.exception))

    def test_reduce_uses_pipeline_and_handles_empty_outputs(self) -> None:
        backend = TransformersReduceBackend("m", "cpu", None)
        mock_pipe = MagicMock(return_value=[])
        backend._pipeline = mock_pipe
        out = backend.reduce("notes", "instr", params={"max_new_tokens": 10})
        self.assertEqual(out.text, "")
        self.assertEqual(out.backend, "transformers")
        mock_pipe.assert_called_once()

    def test_reduce_returns_generated_text(self) -> None:
        backend = TransformersReduceBackend("m", "cpu", None)
        backend._pipeline = MagicMock(return_value=[{"generated_text": "  ## Takeaways\n- x  "}])
        out = backend.reduce("n", "i")
        self.assertIn("Takeaways", out.text)

    def test_initialize_short_circuits_when_pipeline_already_set(self) -> None:
        backend = TransformersReduceBackend("m", "cpu", None)
        backend._pipeline = MagicMock()
        with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
            backend.initialize()
        mock_tok.assert_not_called()

    def test_cleanup_clears_pipeline(self) -> None:
        backend = TransformersReduceBackend("m", "cpu", None)
        backend._pipeline = MagicMock()
        backend.cleanup()
        self.assertIsNone(backend._pipeline)


class TestOllamaReduceBackend(unittest.TestCase):
    def test_reduce_raises_when_not_initialized(self) -> None:
        cfg = Config(rss="https://example.com/f.xml", summary_provider="hybrid_ml")
        backend = OllamaReduceBackend(cfg=cfg, model_name="qwen2.5:7b")
        with self.assertRaises(RuntimeError) as ctx:
            backend.reduce("n", "i")
        self.assertIn("initialize()", str(ctx.exception))

    def test_reduce_maps_max_new_tokens_to_max_length(self) -> None:
        cfg = Config(rss="https://example.com/f.xml", summary_provider="hybrid_ml")
        backend = OllamaReduceBackend(cfg=cfg, model_name="m")
        mock_provider = MagicMock()
        mock_provider.summarize.return_value = {"summary": "  ok  "}
        backend._provider = mock_provider
        out = backend.reduce("notes", "instr", params={"max_new_tokens": 128})
        self.assertEqual(out.backend, "ollama")
        self.assertEqual(out.text, "ok")
        _args, kw = mock_provider.summarize.call_args
        self.assertEqual(kw["params"]["max_length"], 128)
        self.assertEqual(kw["params"]["prompt"], "instr")


class TestHybridMLProviderHelpers(unittest.TestCase):
    def test_build_reduce_instruction_includes_episode_title(self) -> None:
        text = HybridMLProvider._build_reduce_instruction("My Podcast", None)
        self.assertIn("My Podcast", text)
        self.assertIn("## Takeaways", text)

    def test_build_reduce_instruction_paragraph_style(self) -> None:
        text = HybridMLProvider._build_reduce_instruction_paragraph("T", None)
        self.assertIn("paragraphs", text.lower())
        self.assertNotIn("## Takeaways", text)

    def test_postprocess_reduce_output_strips_notes_prefix(self) -> None:
        self.assertEqual(
            HybridMLProvider._postprocess_reduce_output("NOTES: hello"),
            "hello",
        )


class TestHybridMLProviderBehavior(unittest.TestCase):
    def test_generate_insights_returns_empty(self) -> None:
        cfg = Config(rss="https://example.com/f.xml", summary_provider="hybrid_ml")
        p = HybridMLProvider(cfg)
        self.assertEqual(p.generate_insights("any"), [])

    def test_summarize_requires_initialize(self) -> None:
        cfg = Config(rss="https://example.com/f.xml", summary_provider="hybrid_ml")
        p = HybridMLProvider(cfg)
        with self.assertRaises(RuntimeError) as ctx:
            p.summarize("text")
        self.assertIn("initialized", str(ctx.exception).lower())

    @patch(
        "podcast_scraper.providers.ml.hybrid_ml_provider.summarizer._join_summaries_with_structure"
    )
    @patch("podcast_scraper.providers.ml.hybrid_ml_provider.summarizer._summarize_chunks_map")
    @patch("podcast_scraper.providers.ml.hybrid_ml_provider.summarizer._merge_tiny_chunks")
    @patch("podcast_scraper.providers.ml.hybrid_ml_provider.summarizer._prepare_chunks")
    @patch("podcast_scraper.providers.ml.hybrid_ml_provider.ModelRegistry.get_capabilities")
    @patch("podcast_scraper.providers.ml.hybrid_ml_provider.apply_profile_with_stats")
    def test_summarize_happy_path_with_mocks(
        self,
        mock_profile: MagicMock,
        mock_caps: MagicMock,
        mock_prepare: MagicMock,
        mock_merge: MagicMock,
        mock_map: MagicMock,
        mock_join: MagicMock,
    ) -> None:
        mock_profile.return_value = ("cleaned body", {})
        caps = MagicMock()
        caps.max_input_tokens = 2048
        mock_caps.return_value = caps
        mock_prepare.return_value = (["chunk-a"], 512)
        mock_merge.side_effect = lambda _m, chunks: chunks
        mock_map.return_value = ["bullet one"]
        mock_join.return_value = "joined map notes"

        cfg = Config(
            rss="https://example.com/f.xml",
            summary_provider="hybrid_ml",
            summary_chunk_size=800,
        )
        provider = HybridMLProvider(cfg)
        provider._initialized = True
        mm = MagicMock()
        mm.model_name = "facebook/bart-base-cnn"
        mm.model = MagicMock()
        mm.tokenizer = MagicMock()
        provider._map_model = mm
        backend = MagicMock()
        backend.reduce.return_value = HybridReduceResult(
            text="## Takeaways\n- final",
            backend="transformers",
            model="flan",
        )
        provider._reduce_backend = backend

        out = provider.summarize("long transcript", episode_title="Ep1")

        self.assertIn("Takeaways", out["summary"])
        self.assertEqual(out["metadata"]["provider"], "hybrid_ml")
        self.assertEqual(out["metadata"]["map_chunks"], 1)
        backend.reduce.assert_called_once()
        instr = backend.reduce.call_args.kwargs["instruction"]
        self.assertIn("Ep1", instr)

    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_extract_quotes_returns_quote_candidate(self, mock_answer: MagicMock) -> None:
        span = types.SimpleNamespace(start=0, end=5, score=0.88, answer="hello")
        mock_answer.return_value = span
        cfg = Config(rss="https://example.com/f.xml", summary_provider="hybrid_ml")
        p = HybridMLProvider(cfg)
        quotes = p.extract_quotes("hello world", "insight text")
        self.assertEqual(len(quotes), 1)
        self.assertEqual(quotes[0].char_start, 0)
        self.assertEqual(quotes[0].char_end, 5)

    @patch("podcast_scraper.providers.ml.extractive_qa.answer")
    def test_extract_quotes_swallows_qa_errors(self, mock_answer: MagicMock) -> None:
        mock_answer.side_effect = RuntimeError("boom")
        cfg = Config(rss="https://example.com/f.xml", summary_provider="hybrid_ml")
        p = HybridMLProvider(cfg)
        self.assertEqual(p.extract_quotes("t", "i"), [])

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    def test_score_entailment_delegates_to_nli(self, mock_score: MagicMock) -> None:
        mock_score.return_value = 0.73
        cfg = Config(rss="https://example.com/f.xml", summary_provider="hybrid_ml")
        p = HybridMLProvider(cfg)
        self.assertAlmostEqual(p.score_entailment("p", "h"), 0.73)

    @patch("podcast_scraper.providers.ml.nli_loader.entailment_score")
    def test_score_entailment_empty_inputs(self, mock_score: MagicMock) -> None:
        cfg = Config(rss="https://example.com/f.xml", summary_provider="hybrid_ml")
        p = HybridMLProvider(cfg)
        self.assertEqual(p.score_entailment("", "h"), 0.0)
        mock_score.assert_not_called()
