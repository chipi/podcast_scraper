"""Unit tests for SummLlama provider (#571)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper.config import Config

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm] or viewer HTTP
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


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


def _loaded(**overrides):
    """A provider with its tokenizer and model mocked, and a stand-in ``torch`` (only ``no_grad``
    is used), so ``summarize`` runs without weights."""
    from podcast_scraper.providers.ml.summllama_provider import SummLlamaProvider

    provider = SummLlamaProvider(_make_config(**overrides))
    tok = MagicMock()
    tok.apply_chat_template.side_effect = lambda messages, **_: repr(messages)
    tok.decode.return_value = "  A short summary.  "
    tok.return_value.input_ids.shape = (1, 7)
    provider._tokenizer = tok
    provider._model = MagicMock()
    provider._initialized = True
    fake_torch = MagicMock()
    fake_torch.no_grad.return_value.__enter__.return_value = None
    fake_torch.no_grad.return_value.__exit__.return_value = False
    return provider, fake_torch


def _messages(provider):
    return provider._tokenizer.apply_chat_template.call_args.args[0]


class TestSummLlamaSummarize:
    def test_the_episode_summary_uses_the_tuned_prompts_and_budget(self, monkeypatch):
        import sys

        from podcast_scraper.providers.ml import summllama_provider as sp

        provider, fake_torch = _loaded()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        out = provider.summarize("A transcript about chains.", params={"style": "paragraph"})

        assert out["summary"] == "A short summary."
        assert "metadata" not in out  # the episode result is what it always was
        system, user = _messages(provider)
        assert system["content"] == sp._SYSTEM
        assert user["content"] == sp._USER_PARAGRAPH + "A transcript about chains."
        assert provider._model.generate.call_args.kwargs["max_new_tokens"] == 600

    def test_a_board_use_case_runs_the_same_flow_with_its_own_prompts_and_budget(self, monkeypatch):
        """RFC-127: SummLlama is a local chat model, so the board summary is a use case of its
        ``summarize`` exactly as on every LLM provider."""
        import sys

        from podcast_scraper.providers.common.summary_use_case import SummaryUseCase

        provider, fake_torch = _loaded()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        material = {
            "episodes": [
                {
                    "title": "Quokka Hour",
                    "show": "S",
                    "summary": "",
                    "key_points": [],
                    "insights": [],
                }
            ],
            "moments": [{"text": "Counting is a moral act.", "speaker": "Ada", "episode": "Q"}],
        }
        uc = SummaryUseCase(
            system_prompt="summarization/board_system_v1",
            user_prompt="summarization/board_user_v1",
            stage="board_summary",
            max_length=222,
            params=material,
        )
        out = provider.summarize(text="", params={"use_case": uc})

        system, user = _messages(provider)
        assert "a short summary of a listener's board" in system["content"]
        assert "Quokka Hour" in user["content"] and "Counting is a moral act." in user["content"]
        assert "transcript" not in user["content"].lower()
        assert provider._model.generate.call_args.kwargs["max_new_tokens"] == 222
        prompts = out["metadata"]["prompts"]
        assert prompts["system"]["name"] == "summllama/summarization/board_system_v1"
        assert prompts["system"]["file"] == "shared/summarization/board_system_v1.j2"
        assert prompts["user"]["params"] == {"episodes": 1, "moments": 1}
        assert out["metadata"]["model"] == provider.summary_model == "DISLab/SummLlama3.2-3B"
