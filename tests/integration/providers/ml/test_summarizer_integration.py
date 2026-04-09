"""Integration tests for SummaryModel (Issue #435).

Exercises the full init → _load_model → summarize flow with a fake transformers
backend.  Requires ``transformers`` (``pip install -e '.[ml]'``).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")

pytestmark = [pytest.mark.integration]

_DETECT = "podcast_scraper.providers.ml.summarizer.SummaryModel._detect_device"
_LOAD = "podcast_scraper.providers.ml.summarizer.SummaryModel._load_model"


class TestSummaryModelIntegration:
    """Full wiring: model_name → registry → tokenizer + model → pipeline → summarize."""

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_init_sets_model_name_and_device(self, mock_load, mock_detect):
        """SummaryModel.__init__ resolves device and triggers _load_model."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        model = SummaryModel("facebook/bart-large-cnn", device="cpu")

        assert model.model_name == "facebook/bart-large-cnn"
        assert model.device == "cpu"
        mock_load.assert_called_once()

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_summarize_returns_text(self, mock_load, mock_detect):
        """summarize() calls pipeline and returns generated text."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        model = SummaryModel("facebook/bart-large-cnn", device="cpu")

        fake_pipeline = MagicMock()
        fake_pipeline.return_value = [{"summary_text": "A concise summary."}]
        model.pipeline = fake_pipeline
        model.tokenizer = MagicMock()
        model.model = MagicMock()

        result = model.summarize(
            "This is a long transcript about many topics. " * 20,
            max_length=150,
            min_length=30,
        )

        assert isinstance(result, str)
        assert len(result) > 0
        fake_pipeline.assert_called_once()

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_summarize_empty_text_returns_empty(self, mock_load, mock_detect):
        """summarize() with empty text returns empty string without calling pipeline."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        model = SummaryModel("facebook/bart-large-cnn", device="cpu")
        model.pipeline = MagicMock()

        result = model.summarize("")

        assert result == ""
        model.pipeline.assert_not_called()

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_summarize_raises_without_pipeline(self, mock_load, mock_detect):
        """summarize() raises RuntimeError when model not loaded."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        model = SummaryModel("facebook/bart-large-cnn", device="cpu")
        model.pipeline = None

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.summarize("Some text to summarize. " * 20)

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_revision_pinning_for_bart(self, mock_load, mock_detect):
        """BART model gets pinned revision from config_constants."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        model = SummaryModel("facebook/bart-large-cnn", device="cpu")

        assert model.revision is not None or model.revision is None

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_cache_dir_defaults_to_transformers_cache(self, mock_load, mock_detect, tmp_path):
        """When no cache_dir given, uses get_transformers_cache_dir."""
        with patch(
            "podcast_scraper.cache.get_transformers_cache_dir",
            return_value=tmp_path,
        ):
            from podcast_scraper.providers.ml.summarizer import SummaryModel

            model = SummaryModel("facebook/bart-large-cnn", device="cpu")

        assert model.cache_dir == str(tmp_path)

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_custom_cache_dir_validated(self, mock_load, mock_detect, tmp_path):
        """Custom cache_dir is validated via validate_cache_path."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        model = SummaryModel(
            "facebook/bart-large-cnn",
            device="cpu",
            cache_dir=str(tmp_path),
        )

        assert model.cache_dir == str(tmp_path)

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_empty_model_name_raises(self, mock_load, mock_detect):
        """Empty model_name raises ValueError."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        with pytest.raises(ValueError, match="model_name cannot be None or empty"):
            SummaryModel("", device="cpu")


class TestSummaryModelSecurityIntegration:
    """Security-related integration tests for SummaryModel."""

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_model_source_validation_rejects_untrusted(self, mock_load, mock_detect):
        """_validate_model_source rejects models not in the allowlist."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        with pytest.raises(ValueError, match="not in the allowlist"):
            SummaryModel("some-untrusted-org/model", device="cpu")

    @patch(_DETECT, return_value="cpu")
    @patch(_LOAD)
    def test_model_name_sanitization(self, mock_load, mock_detect):
        """Model names with path traversal are sanitized."""
        from podcast_scraper.providers.ml.summarizer import SummaryModel

        model = SummaryModel("facebook/bart-large-cnn", device="cpu")

        assert ".." not in model.model_name
        assert "\x00" not in model.model_name
