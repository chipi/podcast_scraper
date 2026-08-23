"""Unit tests for podcast_scraper.workflow.stages.setup module.

This module tests the pipeline setup functionality, including ML model
caching and environment initialization.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from podcast_scraper.exceptions import ProviderDependencyError
from podcast_scraper.workflow.stages import setup as setup_stage
from podcast_scraper.workflow.stages.setup import (
    _append_gil_evidence_downloads,
    _caused_by_missing_import,
    _collect_hybrid_ml_models_to_download,
    ensure_ml_models_cached,
    set_reproducibility_seeds,
    should_preload_ml_models,
)


@pytest.mark.unit
class TestSetReproducibilitySeeds(unittest.TestCase):
    """Tests for set_reproducibility_seeds function."""

    def test_no_seed_returns_early(self):
        """When cfg has no seed, function returns without setting seeds."""
        cfg = Mock(seed=None)
        set_reproducibility_seeds(cfg)

    def test_with_seed_completes_without_error(self):
        """When cfg has seed, function completes (may set torch/numpy/transformers seeds)."""
        cfg = Mock(seed=42)
        set_reproducibility_seeds(cfg)

    def test_with_seed_sets_torch_seed_when_available(self):
        """When cfg has seed and torch is available, manual_seed is called with seed."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            set_reproducibility_seeds(Mock(seed=42))
            mock_torch.manual_seed.assert_any_call(42)


@pytest.mark.unit
class TestShouldPreloadMLModels(unittest.TestCase):
    """Tests for should_preload_ml_models function."""

    def test_returns_true_when_transcribe_missing_with_whisper(self):
        """Test returns True when transcribe_missing is True and provider is whisper."""
        cfg = Mock()
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"
        cfg.generate_summaries = False
        cfg.summary_provider = "openai"
        cfg.generate_gi = False

        result = should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_returns_true_when_generate_summaries_with_transformers(self):
        """Test returns True when generate_summaries is True and provider is transformers."""
        cfg = Mock()
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "transformers"
        cfg.generate_gi = False

        result = should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_returns_true_when_generate_summaries_with_hybrid_ml(self):
        """Test returns True when generate_summaries is True and provider is hybrid_ml."""
        cfg = Mock()
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "hybrid_ml"
        cfg.generate_gi = False

        result = should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_returns_false_when_no_ml_providers(self):
        """Test returns False when no ML providers are configured."""
        cfg = Mock()
        cfg.transcribe_missing = True
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "openai"
        cfg.generate_gi = False

        result = should_preload_ml_models(cfg)
        self.assertFalse(result)

    def test_returns_false_when_features_disabled(self):
        """Test returns False when both features are disabled."""
        cfg = Mock()
        cfg.transcribe_missing = False
        cfg.transcription_provider = "whisper"
        cfg.generate_summaries = False
        cfg.summary_provider = "transformers"
        cfg.generate_gi = False

        result = should_preload_ml_models(cfg)
        self.assertFalse(result)

    def test_returns_true_when_generate_gi_with_transformers_evidence(self):
        """Test returns True when generate_gi is True and evidence provider is transformers."""
        cfg = Mock()
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = False
        cfg.summary_provider = "openai"
        cfg.generate_gi = True
        cfg.quote_extraction_provider = "transformers"
        cfg.entailment_provider = "transformers"

        result = should_preload_ml_models(cfg)
        self.assertTrue(result)

    def test_returns_true_when_vector_search_enabled(self):
        """vector_search=True preloads embedding model path."""
        cfg = Mock()
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = False
        cfg.summary_provider = "openai"
        cfg.generate_gi = False
        cfg.vector_search = True

        result = should_preload_ml_models(cfg)
        self.assertTrue(result)


@pytest.mark.unit
class TestCollectHybridMlModelsToDownload(unittest.TestCase):
    """Tests for _collect_hybrid_ml_models_to_download (hybrid_ml preload)."""

    @patch("podcast_scraper.providers.ml.summarizer.resolve_model_name")
    def test_returns_map_model_when_not_cached(self, mock_resolve):
        """When MAP model not cached, it is added to the list."""
        mock_resolve.side_effect = lambda x: (
            "google/long-t5-tglobal-base" if x == "longt5-base" else x
        )
        cfg = Mock(hybrid_map_model="longt5-base", hybrid_reduce_backend="ollama")
        cache_dir = MagicMock(spec=Path)
        cache_dir.__truediv__ = Mock(return_value=MagicMock(exists=Mock(return_value=False)))
        result = _collect_hybrid_ml_models_to_download(cfg, cache_dir)
        self.assertEqual(result, [("transformers", "google/long-t5-tglobal-base")])

    @patch("podcast_scraper.providers.ml.summarizer.resolve_model_name")
    def test_skips_reduce_when_backend_not_transformers(self, mock_resolve):
        """When reduce_backend is ollama, REDUCE model is not added."""
        mock_resolve.return_value = "google/long-t5-tglobal-base"
        cfg = Mock(hybrid_map_model="longt5-base", hybrid_reduce_backend="ollama")
        cache_dir = MagicMock(spec=Path)
        cache_dir.__truediv__ = Mock(return_value=MagicMock(exists=Mock(return_value=True)))
        result = _collect_hybrid_ml_models_to_download(cfg, cache_dir)
        self.assertEqual(result, [])

    @patch("podcast_scraper.providers.ml.summarizer.resolve_model_name")
    def test_adds_reduce_model_when_different_and_not_cached(self, mock_resolve):
        """When REDUCE backend is transformers and reduce model not cached, add it."""

        def resolve(name):
            if "longt5" in str(name) or "long-t5" in str(name):
                return "google/long-t5-tglobal-base"
            return "google/flan-t5-base"

        mock_resolve.side_effect = resolve
        cfg = Mock(
            hybrid_map_model="longt5-base",
            hybrid_reduce_model="google/flan-t5-base",
            hybrid_reduce_backend="transformers",
        )
        map_path = MagicMock(exists=Mock(return_value=True))
        reduce_path = MagicMock(exists=Mock(return_value=False))
        cache_dir = MagicMock(spec=Path)
        cache_dir.__truediv__.side_effect = [map_path, reduce_path]
        result = _collect_hybrid_ml_models_to_download(cfg, cache_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "transformers")
        self.assertEqual(result[0][1], "google/flan-t5-base")  # resolved_reduce

    @patch("podcast_scraper.providers.ml.summarizer.resolve_model_name")
    def test_handles_value_error_uses_passthrough_for_raw_id(self, mock_resolve):
        """When resolve_model_name raises ValueError, raw HF ID (with /) is used."""
        mock_resolve.side_effect = ValueError("unknown alias")
        cfg = Mock(
            hybrid_map_model="google/custom-model",
            hybrid_reduce_backend="ollama",
        )
        cache_dir = MagicMock(spec=Path)
        cache_dir.__truediv__ = Mock(return_value=MagicMock(exists=Mock(return_value=False)))
        result = _collect_hybrid_ml_models_to_download(cfg, cache_dir)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], "google/custom-model")


@pytest.mark.unit
class TestAppendGilEvidenceDownloads(unittest.TestCase):
    """Tests for _append_gil_evidence_downloads (vector_search vs full GIL ML)."""

    @patch("podcast_scraper.providers.ml.model_loader.is_evidence_model_cached")
    def test_vector_search_only_queues_embedding_not_qa_nli(self, mock_cached):
        """vector_search without generate_gi queues vector_embedding_model (not QA/NLI)."""
        mock_cached.return_value = False
        cfg = Mock()
        cfg.generate_gi = False
        cfg.vector_search = True
        cfg.vector_embedding_model = "minilm-l6"
        cfg.gi_embedding_model = "other-embedding"
        models: list = []
        _append_gil_evidence_downloads(cfg, models)
        kinds = [m[0] for m in models]
        self.assertIn("evidence_embedding", kinds)
        self.assertNotIn("evidence_qa", kinds)
        self.assertNotIn("evidence_nli", kinds)
        self.assertEqual(
            [c.args[0] for c in mock_cached.call_args_list],
            ["minilm-l6"],
        )

    @patch("podcast_scraper.providers.ml.model_loader.is_evidence_model_cached")
    def test_vector_search_and_gil_ml_queue_distinct_embedding_models(self, mock_cached):
        """When both vector_search and GIL transformers run, check both embedding ids."""
        mock_cached.return_value = False
        cfg = Mock()
        cfg.generate_gi = True
        cfg.vector_search = True
        cfg.quote_extraction_provider = "transformers"
        cfg.entailment_provider = "transformers"
        cfg.vector_embedding_model = "mpnet-base"
        cfg.gi_embedding_model = "minilm-l6"
        cfg.gi_qa_model = "roberta-squad2"
        cfg.gi_nli_model = "nli-deberta-base"
        models: list = []
        _append_gil_evidence_downloads(cfg, models)
        emb = [m for m in models if m[0] == "evidence_embedding"]
        self.assertEqual(len(emb), 2)
        self.assertEqual({emb[0][1], emb[1][1]}, {"mpnet-base", "minilm-l6"})

    @patch("podcast_scraper.providers.ml.model_loader.is_evidence_model_cached")
    def test_generate_gil_transformers_queues_embedding_qa_nli(self, mock_cached):
        """GIL with transformers evidence queues embedding plus QA and NLI when missing."""
        mock_cached.return_value = False
        cfg = Mock()
        cfg.generate_gi = True
        cfg.vector_search = False
        cfg.quote_extraction_provider = "transformers"
        cfg.entailment_provider = "transformers"
        cfg.gi_embedding_model = "minilm-l6"
        cfg.gi_qa_model = "roberta-squad2"
        cfg.gi_nli_model = "nli-deberta-base"
        models: list = []
        _append_gil_evidence_downloads(cfg, models)
        kinds = [m[0] for m in models]
        self.assertIn("evidence_embedding", kinds)
        self.assertIn("evidence_qa", kinds)
        self.assertIn("evidence_nli", kinds)


@pytest.mark.unit
class TestEnsureMLModelsCached(unittest.TestCase):
    """Tests for ensure_ml_models_cached function."""

    def test_skips_when_preload_disabled(self):
        """Test that function returns early when preload_models is False."""
        cfg = Mock()
        cfg.preload_models = False
        cfg.dry_run = False

        # Should return without doing anything
        ensure_ml_models_cached(cfg)
        # No assertions needed - just shouldn't raise

    def test_skips_when_dry_run(self):
        """Test that function returns early during dry run."""
        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = True

        # Should return without doing anything
        ensure_ml_models_cached(cfg)
        # No assertions needed - just shouldn't raise

    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    def test_skips_when_no_ml_models_needed(self, mock_should_preload):
        """Test that function returns early when no ML models are needed."""
        mock_should_preload.return_value = False
        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False

        ensure_ml_models_cached(cfg)
        mock_should_preload.assert_called_once_with(cfg)

    @patch("podcast_scraper.workflow.stages.setup.config._is_pytest_run")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    def test_skips_in_test_environment(self, mock_should_preload, mock_is_test):
        """Test that function returns early in test environment."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = True
        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False

        ensure_ml_models_cached(cfg)
        mock_is_test.assert_called_once()

    @patch("podcast_scraper.config._is_pytest_run")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_checks_whisper_model_cache(self, mock_get_whisper, mock_should_preload, mock_is_test):
        """Test that function checks if Whisper model is cached."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        # Create a mock path that returns False for exists()
        mock_cache_dir = MagicMock(spec=Path)
        mock_model_file = MagicMock(spec=Path)
        mock_model_file.exists.return_value = True  # Model exists, no download needed
        mock_cache_dir.__truediv__ = Mock(return_value=mock_model_file)
        mock_get_whisper.return_value = mock_cache_dir

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"
        cfg.whisper_model = "tiny.en"
        cfg.generate_summaries = False
        cfg.summary_provider = "openai"

        ensure_ml_models_cached(cfg)

        # Should check if model file exists
        mock_model_file.exists.assert_called()

    @patch("podcast_scraper.config._is_pytest_run")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.providers.ml.model_loader.preload_whisper_models")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_downloads_missing_whisper_model(
        self, mock_get_whisper, mock_preload_whisper, mock_should_preload, mock_is_test
    ):
        """Test that function downloads missing Whisper model."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        # Create a mock path that returns False for exists() - model not cached
        mock_cache_dir = MagicMock(spec=Path)
        mock_model_file = MagicMock(spec=Path)
        mock_model_file.exists.return_value = False  # Model NOT cached
        mock_cache_dir.__truediv__ = Mock(return_value=mock_model_file)
        mock_get_whisper.return_value = mock_cache_dir

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"
        cfg.whisper_model = "tiny.en"
        cfg.generate_summaries = False
        cfg.summary_provider = "openai"

        ensure_ml_models_cached(cfg)

        # Should call preload_whisper_models
        mock_preload_whisper.assert_called_once_with(["tiny.en"])

    @patch("podcast_scraper.config._is_pytest_run")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.providers.ml.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.summarizer.select_summary_model")
    @patch("podcast_scraper.cache.get_transformers_cache_dir")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_checks_transformers_model_cache(
        self,
        mock_get_whisper,
        mock_get_transformers,
        mock_select_summary,
        mock_select_reduce,
        mock_should_preload,
        mock_is_test,
    ):
        """Test that function checks if Transformers model is cached."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        # Setup summarizer mocks
        mock_select_summary.return_value = "bart-small"
        mock_select_reduce.return_value = "bart-small"

        # Create mock paths
        mock_transformers_cache = MagicMock(spec=Path)
        mock_model_path = MagicMock(spec=Path)
        mock_model_path.exists.return_value = True  # Model exists
        mock_transformers_cache.__truediv__ = Mock(return_value=mock_model_path)
        mock_get_transformers.return_value = mock_transformers_cache

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "transformers"

        ensure_ml_models_cached(cfg)

        # Should check model selection
        mock_select_summary.assert_called_once_with(cfg)

    @patch("podcast_scraper.config._is_pytest_run")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.providers.ml.model_loader.preload_transformers_models")
    @patch("podcast_scraper.providers.ml.summarizer.select_reduce_model")
    @patch("podcast_scraper.providers.ml.summarizer.select_summary_model")
    @patch("podcast_scraper.cache.get_transformers_cache_dir")
    @patch("podcast_scraper.cache.get_whisper_cache_dir")
    def test_downloads_missing_transformers_model(
        self,
        mock_get_whisper,
        mock_get_transformers,
        mock_select_summary,
        mock_select_reduce,
        mock_preload_transformers,
        mock_should_preload,
        mock_is_test,
    ):
        """Test that function downloads missing Transformers model."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        # Setup summarizer mocks
        mock_select_summary.return_value = "bart-small"
        mock_select_reduce.return_value = "bart-small"

        # Create mock paths - model NOT cached
        mock_transformers_cache = MagicMock(spec=Path)
        mock_model_path = MagicMock(spec=Path)
        mock_model_path.exists.return_value = False  # Model NOT cached
        mock_transformers_cache.__truediv__ = Mock(return_value=mock_model_path)
        mock_get_transformers.return_value = mock_transformers_cache

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = False
        cfg.transcription_provider = "openai"
        cfg.generate_summaries = True
        cfg.summary_provider = "transformers"

        ensure_ml_models_cached(cfg)

        # Should call preload_transformers_models
        mock_preload_transformers.assert_called_once_with(["bart-small"])

    @patch("podcast_scraper.config._is_pytest_run")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.workflow.stages.setup.logger")
    def test_handles_import_error_gracefully(self, mock_logger, mock_should_preload, mock_is_test):
        """Test that function handles ImportError gracefully."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"

        # Patch to raise ImportError when importing cache module
        with patch(
            "podcast_scraper.cache.get_whisper_cache_dir",
            side_effect=ImportError("cache module not available"),
        ):
            # Should not raise
            ensure_ml_models_cached(cfg)

    @patch("podcast_scraper.config._is_pytest_run")
    @patch("podcast_scraper.workflow.stages.setup.should_preload_ml_models")
    @patch("podcast_scraper.workflow.stages.setup.logger")
    def test_handles_general_exception_gracefully(
        self, mock_logger, mock_should_preload, mock_is_test
    ):
        """Test that function handles general exceptions gracefully."""
        mock_should_preload.return_value = True
        mock_is_test.return_value = False

        cfg = Mock()
        cfg.preload_models = True
        cfg.dry_run = False
        cfg.transcribe_missing = True
        cfg.transcription_provider = "whisper"

        # Patch to raise general exception at the source module level
        with patch(
            "podcast_scraper.cache.get_whisper_cache_dir",
            side_effect=RuntimeError("Unexpected error"),
        ):
            # Should not raise
            ensure_ml_models_cached(cfg)
            # Should log debug message
            mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()


@pytest.mark.unit
class TestPreloadTreatsAMissingPackageAsNonFatal(unittest.TestCase):
    """A missing optional PACKAGE degrades to on-demand loading; a missing MODEL still fails fast.

    ``preload_ml_models_if_needed`` has always documented the split — *"ImportError: If ML
    dependencies are not installed"* vs *"RuntimeError: If required model cannot be loaded"* — and
    catches ``ImportError`` with a "non-fatal, ML models will be loaded on-demand" warning. But
    ``MLProvider.preload()`` translates every failure into ``ProviderDependencyError``, which is a
    ``ProviderError``, not an ``ImportError``. So the missing-package case fell through to the
    generic handler and killed the run: the handler written to tolerate a missing optional
    dependency hard-failed on the exception that means exactly that.
    """

    @staticmethod
    def _run_preload_raising(exc: BaseException) -> None:
        cfg = Mock(preload_models=True, dry_run=False)
        with (
            patch.object(setup_stage, "should_preload_ml_models", return_value=True),
            patch.object(setup_stage, "ensure_ml_models_cached"),
            patch("podcast_scraper.providers.ml.ml_provider.MLProvider") as provider_cls,
        ):
            provider_cls.return_value.preload.side_effect = exc
            setup_stage.preload_ml_models_if_needed(cfg)

    def _dependency_error(self, cause: BaseException | None, *, chain: str = "cause"):
        """Build the wrapped shape MLProvider produces, with explicit or implicit chaining."""
        inner = ProviderDependencyError(message="inner", provider="p", dependency="d")
        if cause is not None:
            if chain == "cause":
                inner.__cause__ = cause
            else:  # bare `raise` inside `except ImportError` — cause is None, context is set
                inner.__context__ = cause
        outer = ProviderDependencyError(message="outer", provider="p", dependency="d")
        outer.__cause__ = inner
        return outer

    def test_a_missing_package_does_not_kill_the_run(self):
        self._run_preload_raising(
            self._dependency_error(ModuleNotFoundError("No module named 'x'"))
        )

    def test_implicit_chaining_counts_too(self):
        """The raise sites are NOT consistent, and that inconsistency is the whole trap.

        ``MLProvider.preload`` uses ``raise ... from e``, but ``_initialize_whisper`` uses a bare
        ``raise`` inside ``except ImportError`` — recording the ImportError as ``__context__`` with
        ``__cause__`` left None. A check that trusts explicit chaining alone silently mis-classifies
        whichever sites happen to omit ``from``, which is exactly what happened for whisper.
        """
        self._run_preload_raising(
            self._dependency_error(ImportError("no whisper"), chain="context")
        )

    def test_a_missing_model_still_fails_fast(self):
        """The other half. "Re-raise to fail fast for required models" must keep meaning something —
        a downloaded-model problem is not a dependency problem and must not be swallowed."""
        with self.assertRaises(ProviderDependencyError):
            self._run_preload_raising(self._dependency_error(FileNotFoundError("model.bin")))

    def test_an_unrelated_error_still_fails_fast(self):
        with self.assertRaises(ValueError):
            self._run_preload_raising(ValueError("something else entirely"))


@pytest.mark.unit
class TestCausedByMissingImport(unittest.TestCase):
    """The classifier itself, including the shapes that made the first two attempts wrong."""

    def test_direct_cause(self):
        exc = RuntimeError()
        exc.__cause__ = ImportError()
        self.assertTrue(_caused_by_missing_import(exc))

    def test_nested_cause_two_links_deep(self):
        # The real whisper shape: ProviderDependencyError -> ProviderDependencyError -> ImportError.
        inner, outer = RuntimeError(), RuntimeError()
        inner.__cause__ = ModuleNotFoundError()
        outer.__cause__ = inner
        self.assertTrue(_caused_by_missing_import(outer))

    def test_context_only_chaining(self):
        exc = RuntimeError()
        exc.__context__ = ImportError()
        self.assertTrue(_caused_by_missing_import(exc))

    def test_no_import_anywhere(self):
        exc = RuntimeError()
        exc.__cause__ = FileNotFoundError()
        self.assertFalse(_caused_by_missing_import(exc))

    def test_bare_exception(self):
        self.assertFalse(_caused_by_missing_import(RuntimeError()))

    def test_a_cycle_terminates(self):
        a, b = RuntimeError(), RuntimeError()
        a.__cause__ = b
        b.__cause__ = a
        self.assertFalse(_caused_by_missing_import(a))
