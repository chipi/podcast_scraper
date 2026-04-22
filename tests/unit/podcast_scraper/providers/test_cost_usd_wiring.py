"""#651 Part B regression guard: every billable provider records ``cost_usd``
on the pipeline_metrics call for cleaning, generate_insights, and
extract_kg_graph.

Pre-Part-B, only OpenAI recorded these costs. The other 5 providers silently
dropped cost to $0 whenever a non-OpenAI profile ran. This test locks in the
wiring so a future refactor can't regress the 5 providers × 3 capabilities =
15 recording sites.

Design: mock the client + response shape for each provider, call the
capability method with a mock ``pipeline_metrics``, and assert the expected
``record_llm_*_call`` was invoked with a non-None numeric ``cost_usd``.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import Mock, patch

import pytest

from podcast_scraper import config


def _mock_metrics_with_recorders(*recorder_names: str) -> Mock:
    """Build a Mock that only exposes the named record_llm_*_call methods.

    Uses ``spec=recorder_names`` so ``hasattr(pm, "record_llm_foo")`` returns
    True only for explicitly listed names — matches how providers gate the
    cost-recording branch.
    """
    pm = Mock(spec=list(recorder_names))
    # Each recorder becomes a Mock attribute when listed in spec.
    return pm


def _assert_cost_recorded(pm: Mock, recorder_name: str) -> None:
    """The named recorder was called once with a non-None numeric cost_usd."""
    recorder = getattr(pm, recorder_name)
    assert recorder.call_count >= 1, f"{recorder_name} never invoked"
    call = recorder.call_args_list[-1]
    cost = call.kwargs.get("cost_usd")
    assert cost is not None, f"{recorder_name} called with cost_usd=None"
    assert isinstance(cost, (int, float)), f"cost_usd not numeric: {cost!r}"


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def _make_anthropic_response(in_tok: int = 120, out_tok: int = 45) -> Mock:
    resp = Mock()
    resp.content = [Mock(text="Insight one\nInsight two")]
    resp.usage = Mock(input_tokens=in_tok, output_tokens=out_tok)
    return resp


@pytest.mark.unit
class TestAnthropicCostWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = config.Config(
            transcription_provider="whisper",
            speaker_detector_provider="anthropic",
            summary_provider="anthropic",
            anthropic_api_key="test-key",
            transcribe_missing=False,
            auto_speakers=True,
            generate_summaries=True,
        )

    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_generate_insights_records_cost(self, mock_anthropic: Mock) -> None:
        from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

        client = Mock()
        mock_anthropic.return_value = client
        client.messages.create.return_value = _make_anthropic_response()
        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_gi_call")
        provider.generate_insights("transcript", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_gi_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_extract_kg_graph_records_cost(self, mock_anthropic: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

        resp = Mock()
        resp.content = [Mock(text='{"topics": [], "entities": []}')]
        resp.usage = Mock(input_tokens=200, output_tokens=80)
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_anthropic.return_value = client
        client.messages.create.return_value = resp
        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_kg_call")
        provider.extract_kg_graph("transcript enough text to matter.", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_kg_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_clean_transcript_records_cost(self, mock_anthropic: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

        resp = _make_anthropic_response(in_tok=500, out_tok=400)
        resp.content = [Mock(text="cleaned transcript body")]
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_anthropic.return_value = client
        client.messages.create.return_value = resp

        provider = AnthropicProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_cleaning_call")
        provider.clean_transcript("some transcript text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_cleaning_call")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def _make_gemini_response(
    in_tok: int = 120, out_tok: int = 45, text: str = "Insight one\nInsight two"
) -> Mock:
    resp = Mock()
    resp.text = text
    resp.usage_metadata = Mock(prompt_token_count=in_tok, candidates_token_count=out_tok)
    return resp


@pytest.mark.unit
class TestGeminiCostWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = config.Config(
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            gemini_api_key="test-key",
            transcribe_missing=False,
            auto_speakers=True,
            generate_summaries=True,
        )

    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_generate_insights_records_cost(self, mock_genai: Mock) -> None:
        from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

        client = Mock()
        mock_genai.Client.return_value = client
        client.models.generate_content.return_value = _make_gemini_response()
        provider = GeminiProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_gi_call")
        provider.generate_insights("transcript", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_gi_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_extract_kg_graph_records_cost(self, mock_genai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

        resp = _make_gemini_response(in_tok=220, out_tok=88, text='{"topics": [], "entities": []}')
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_genai.Client.return_value = client
        client.models.generate_content.return_value = resp
        provider = GeminiProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_kg_call")
        provider.extract_kg_graph("transcript enough text to matter.", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_kg_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_clean_transcript_records_cost(self, mock_genai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

        resp = _make_gemini_response(in_tok=500, out_tok=400, text="cleaned transcript body")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_genai.Client.return_value = client
        client.models.generate_content.return_value = resp

        provider = GeminiProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_cleaning_call")
        provider.clean_transcript("some transcript text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_cleaning_call")


# ---------------------------------------------------------------------------
# OpenAI-compat shape (Mistral / DeepSeek / Grok)
# ---------------------------------------------------------------------------


def _make_openai_shape_response(
    in_tok: int = 120, out_tok: int = 45, content: str = "Insight one\nInsight two"
) -> Mock:
    resp = Mock()
    choice = Mock()
    choice.message.content = content
    resp.choices = [choice]
    resp.usage = Mock(prompt_tokens=in_tok, completion_tokens=out_tok)
    return resp


def _make_openai_compat_cfg(provider_name: str) -> config.Config:
    kwargs: dict[str, Any] = {
        "transcription_provider": "whisper",
        "speaker_detector_provider": provider_name,
        "summary_provider": provider_name,
        "transcribe_missing": False,
        "auto_speakers": True,
        "generate_summaries": True,
    }
    kwargs[f"{provider_name}_api_key"] = "test-api-key-123"
    return config.Config(**kwargs)


@pytest.mark.unit
class TestMistralCostWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _make_openai_compat_cfg("mistral")

    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_generate_insights_records_cost(self, mock_mistral: Mock) -> None:
        from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

        client = Mock()
        mock_mistral.return_value = client
        client.chat.complete.return_value = _make_openai_shape_response()
        provider = MistralProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_gi_call")
        provider.generate_insights("transcript", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_gi_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_extract_kg_graph_records_cost(self, mock_mistral: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_mistral.return_value = client
        client.chat.complete.return_value = _make_openai_shape_response(
            content='{"topics": [], "entities": []}'
        )
        provider = MistralProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_kg_call")
        provider.extract_kg_graph("transcript enough text to matter.", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_kg_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_clean_transcript_records_cost(self, mock_mistral: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_mistral.return_value = client
        client.chat.complete.return_value = _make_openai_shape_response(
            content="cleaned transcript body"
        )

        provider = MistralProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_cleaning_call")
        provider.clean_transcript("some transcript text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_cleaning_call")


@pytest.mark.unit
class TestDeepSeekCostWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _make_openai_compat_cfg("deepseek")

    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_generate_insights_records_cost(self, mock_openai: Mock) -> None:
        from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider

        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response()
        provider = DeepSeekProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_gi_call")
        provider.generate_insights("transcript", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_gi_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_extract_kg_graph_records_cost(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider

        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            content='{"topics": [], "entities": []}'
        )
        provider = DeepSeekProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_kg_call")
        provider.extract_kg_graph("transcript enough text to matter.", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_kg_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_clean_transcript_records_cost(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider

        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            content="cleaned transcript body"
        )

        provider = DeepSeekProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_cleaning_call")
        provider.clean_transcript("some transcript text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_cleaning_call")


@pytest.mark.unit
class TestGrokCostWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = _make_openai_compat_cfg("grok")

    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_generate_insights_records_cost(self, mock_openai: Mock) -> None:
        from podcast_scraper.providers.grok.grok_provider import GrokProvider

        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response()
        provider = GrokProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_gi_call")
        provider.generate_insights("transcript", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_gi_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_extract_kg_graph_records_cost(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.grok.grok_provider import GrokProvider

        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            content='{"topics": [], "entities": []}'
        )
        provider = GrokProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_kg_call")
        provider.extract_kg_graph("transcript enough text to matter.", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_kg_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_clean_transcript_records_cost(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.grok.grok_provider import GrokProvider

        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            content="cleaned transcript body"
        )

        provider = GrokProvider(self.cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_cleaning_call")
        provider.clean_transcript("some transcript text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_cleaning_call")


# ---------------------------------------------------------------------------
# Bundle-mode cost wiring (audit fix)
#
# Pre-audit: ``summarize_mega_bundled`` and ``summarize_extraction_bundled``
# made real LLM calls but never invoked any ``record_llm_*_call``. Every
# cloud_balanced / mega_bundled / extraction_bundled run silently dropped
# the summary-side cost to $0. This section locks the new wiring in: both
# methods now record to ``record_llm_summarization_call`` with cost_usd.
# ---------------------------------------------------------------------------


_VALID_MEGABUNDLE_JSON = (
    '{"title": "T", "summary": "s", "bullets": ["b1", "b2"], '
    '"insights": [{"text": "ins one"}], '
    '"topics": [{"label": "t1"}], '
    '"entities": [{"name": "E1", "kind": "org"}]}'
)
_VALID_EXTRACTION_JSON = (
    '{"insights": [{"text": "ins one"}], '
    '"topics": [{"label": "t1"}], '
    '"entities": [{"name": "E1", "kind": "org"}]}'
)


@pytest.mark.unit
class TestBundleModeSummarizationCostWiring(unittest.TestCase):
    """Each provider's mega / extraction bundle path records cost_usd via
    ``record_llm_summarization_call``.
    """

    # --- Anthropic -----------------------------------------------------------
    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_anthropic_mega_bundled(self, mock_anthropic: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

        cfg = config.Config(
            transcription_provider="whisper",
            speaker_detector_provider="anthropic",
            summary_provider="anthropic",
            anthropic_api_key="test-api-key-123",
            transcribe_missing=False,
            auto_speakers=True,
            generate_summaries=True,
        )
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_anthropic.return_value = client
        resp = Mock()
        resp.content = [Mock(text=_VALID_MEGABUNDLE_JSON)]
        resp.usage = Mock(input_tokens=800, output_tokens=200)
        client.messages.create.return_value = resp
        provider = AnthropicProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_mega_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.anthropic.anthropic_provider.Anthropic")
    def test_anthropic_extraction_bundled(self, mock_anthropic: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider

        cfg = config.Config(
            transcription_provider="whisper",
            speaker_detector_provider="anthropic",
            summary_provider="anthropic",
            anthropic_api_key="test-api-key-123",
            transcribe_missing=False,
            auto_speakers=True,
            generate_summaries=True,
        )
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_anthropic.return_value = client
        resp = Mock()
        resp.content = [Mock(text=_VALID_EXTRACTION_JSON)]
        resp.usage = Mock(input_tokens=800, output_tokens=200)
        client.messages.create.return_value = resp
        provider = AnthropicProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_extraction_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    # --- Gemini --------------------------------------------------------------
    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_mega_bundled(self, mock_genai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

        cfg = config.Config(
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=False,
            auto_speakers=True,
            generate_summaries=True,
        )
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_genai.Client.return_value = client
        resp = Mock()
        resp.text = _VALID_MEGABUNDLE_JSON
        resp.usage_metadata = Mock(prompt_token_count=800, candidates_token_count=200)
        client.models.generate_content.return_value = resp
        provider = GeminiProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_mega_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.gemini.gemini_provider.genai")
    def test_gemini_extraction_bundled(self, mock_genai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

        cfg = config.Config(
            transcription_provider="gemini",
            speaker_detector_provider="gemini",
            summary_provider="gemini",
            gemini_api_key="test-api-key-123",
            transcribe_missing=False,
            auto_speakers=True,
            generate_summaries=True,
        )
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_genai.Client.return_value = client
        resp = Mock()
        resp.text = _VALID_EXTRACTION_JSON
        resp.usage_metadata = Mock(prompt_token_count=800, candidates_token_count=200)
        client.models.generate_content.return_value = resp
        provider = GeminiProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_extraction_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    # --- OpenAI --------------------------------------------------------------
    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("openai.OpenAI")
    def test_openai_mega_bundled(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

        cfg = _make_openai_compat_cfg("openai")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            in_tok=800, out_tok=200, content=_VALID_MEGABUNDLE_JSON
        )
        provider = OpenAIProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_mega_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("openai.OpenAI")
    def test_openai_extraction_bundled(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.openai.openai_provider import OpenAIProvider

        cfg = _make_openai_compat_cfg("openai")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            in_tok=800, out_tok=200, content=_VALID_EXTRACTION_JSON
        )
        provider = OpenAIProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_extraction_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    # --- Mistral / DeepSeek / Grok (OpenAI-compat) ---------------------------
    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_mistral_mega_bundled(self, mock_mistral: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

        cfg = _make_openai_compat_cfg("mistral")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_mistral.return_value = client
        client.chat.complete.return_value = _make_openai_shape_response(
            in_tok=800, out_tok=200, content=_VALID_MEGABUNDLE_JSON
        )
        provider = MistralProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_mega_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.mistral.mistral_provider.Mistral")
    def test_mistral_extraction_bundled(self, mock_mistral: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.mistral.mistral_provider import MistralProvider

        cfg = _make_openai_compat_cfg("mistral")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_mistral.return_value = client
        client.chat.complete.return_value = _make_openai_shape_response(
            in_tok=800, out_tok=200, content=_VALID_EXTRACTION_JSON
        )
        provider = MistralProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_extraction_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_deepseek_mega_bundled(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider

        cfg = _make_openai_compat_cfg("deepseek")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            in_tok=800, out_tok=200, content=_VALID_MEGABUNDLE_JSON
        )
        provider = DeepSeekProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_mega_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.deepseek.deepseek_provider.OpenAI")
    def test_deepseek_extraction_bundled(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.deepseek.deepseek_provider import DeepSeekProvider

        cfg = _make_openai_compat_cfg("deepseek")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            in_tok=800, out_tok=200, content=_VALID_EXTRACTION_JSON
        )
        provider = DeepSeekProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_extraction_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_grok_mega_bundled(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.grok.grok_provider import GrokProvider

        cfg = _make_openai_compat_cfg("grok")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            in_tok=800, out_tok=200, content=_VALID_MEGABUNDLE_JSON
        )
        provider = GrokProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_mega_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")

    @patch("podcast_scraper.utils.provider_metrics.retry_with_metrics")
    @patch("podcast_scraper.providers.grok.grok_provider.OpenAI")
    def test_grok_extraction_bundled(self, mock_openai: Mock, mock_retry: Mock) -> None:
        from podcast_scraper.providers.grok.grok_provider import GrokProvider

        cfg = _make_openai_compat_cfg("grok")
        mock_retry.side_effect = lambda fn, **kwargs: fn()
        client = Mock()
        mock_openai.return_value = client
        client.chat.completions.create.return_value = _make_openai_shape_response(
            in_tok=800, out_tok=200, content=_VALID_EXTRACTION_JSON
        )
        provider = GrokProvider(cfg)
        provider.initialize()

        pm = _mock_metrics_with_recorders("record_llm_summarization_call")
        provider.summarize_extraction_bundled("transcript body text", pipeline_metrics=pm)
        _assert_cost_recorded(pm, "record_llm_summarization_call")
