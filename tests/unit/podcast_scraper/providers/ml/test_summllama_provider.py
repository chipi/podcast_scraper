"""Unit tests for SummLlama provider (#571)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from podcast_scraper.config import Config


def _make_config(**overrides):
    defaults = {
        "rss_url": "https://example.com/feed.xml",
        "summary_provider": "summllama",
        "generate_summaries": True,
        "generate_metadata": True,
    }
    defaults.update(overrides)
    return Config(**defaults)


class TestSummLlamaProvider:
    """Unit tests for SummLlamaProvider (no real model loading)."""

    def test_config_accepts_summllama(self):
        cfg = _make_config()
        assert cfg.summary_provider == "summllama"

    def test_factory_creates_provider(self):
        from podcast_scraper.summarization.factory import create_summarization_provider

        cfg = _make_config()
        provider = create_summarization_provider(cfg)
        assert type(provider).__name__ == "SummLlamaProvider"
        assert provider._model_id == "DISLab/SummLlama3.2-3B"
        assert provider._style == "bullets"

    def test_default_style_is_bullets(self):
        from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider

        provider = SummLlamaProvider(_make_config())
        assert provider._style == "bullets"

    def test_not_initialized_by_default(self):
        from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider

        provider = SummLlamaProvider(_make_config())
        assert not provider.is_initialized

    @patch("transformers.AutoModelForCausalLM")
    @patch("transformers.AutoTokenizer")
    def test_summarize_returns_dict(self, mock_tokenizer_cls, mock_model_cls):
        """Summarize returns dict with expected keys (mocked model)."""
        from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt text"
        mock_tokenizer.return_tensors = "pt"
        mock_tokenizer.eos_token_id = 0
        mock_inputs = MagicMock()
        mock_inputs.input_ids.shape = [1, 10]
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.__getitem__ = lambda self, idx: MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        mock_model_cls.from_pretrained.return_value = mock_model

        # Mock decode
        mock_tokenizer.decode.return_value = "- Bullet 1\n- Bullet 2"

        cfg = _make_config()
        provider = SummLlamaProvider(cfg)
        provider._device = "cpu"  # Force CPU for test
        provider.initialize()
        assert provider.is_initialized

        result = provider.summarize("Test transcript text here.")
        assert "summary" in result
        assert "model" in result
        assert "style" in result
        assert result["model"] == "DISLab/SummLlama3.2-3B"
        assert result["style"] == "bullets"

    def test_cleanup(self):
        from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider

        provider = SummLlamaProvider(_make_config())
        provider._model = MagicMock()
        provider._tokenizer = MagicMock()
        provider._initialized = True
        provider.cleanup()
        assert not provider.is_initialized
        assert provider._model is None
        assert provider._tokenizer is None
