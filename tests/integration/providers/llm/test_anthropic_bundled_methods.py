"""Unit tests for Anthropic's GIL-evidence bundled methods (#698).

Anthropic's SDK shape (``client.messages.create`` returning
``response.content[0].text``) differs from the OpenAI Chat Completions
shape, so it gets its own test file. The shared parsers, prompts, and
QuoteCandidate output are identical to the other providers.
"""

from __future__ import annotations

import importlib.util
import json
from unittest.mock import MagicMock, Mock, patch

import pytest

mock_anthropic = MagicMock()
mock_anthropic.__spec__ = importlib.util.spec_from_loader("anthropic", loader=None)
patch.dict("sys.modules", {"anthropic": mock_anthropic}).start()

from podcast_scraper import config
from podcast_scraper.providers.anthropic.anthropic_provider import AnthropicProvider
from podcast_scraper.providers.common.bundle_extract_parser import BundleExtractParseError
from podcast_scraper.providers.common.bundle_nli_parser import BundleNliParseError

pytestmark = [pytest.mark.integration]


def _make_provider() -> AnthropicProvider:
    cfg = config.Config(
        summary_provider="anthropic",
        anthropic_api_key="test-key",
        anthropic_summary_model="claude-haiku-4-5",
    )
    provider = AnthropicProvider(cfg)
    provider.client = MagicMock()
    provider._summarization_initialized = True
    provider.summary_model = "claude-haiku-4-5"
    return provider


def _make_anthropic_response(text: str) -> Mock:
    """Build a fake response matching ``client.messages.create`` shape."""
    block = Mock()
    block.text = text
    response = Mock()
    response.content = [block]
    response.usage = Mock(input_tokens=100, output_tokens=50)
    return response


def _patch_retry(text: str):
    return patch(
        "podcast_scraper.utils.provider_metrics.retry_with_metrics",
        return_value=_make_anthropic_response(text),
    )


def _patch_retry_raises(exc: Exception):
    return patch(
        "podcast_scraper.utils.provider_metrics.retry_with_metrics",
        side_effect=exc,
    )


# ---------------------------------------------------------------------------
# extract_quotes_bundled


class TestExtractQuotesBundledEarlyReturn:
    def test_not_initialized_returns_empty_per_insight(self) -> None:
        provider = _make_provider()
        provider._summarization_initialized = False
        out = provider.extract_quotes_bundled("transcript", ["i1", "i2"])
        assert out == {0: [], 1: []}

    def test_empty_transcript_returns_empty_per_insight(self) -> None:
        provider = _make_provider()
        out = provider.extract_quotes_bundled("", ["i1", "i2"])
        assert out == {0: [], 1: []}

    def test_no_insights_returns_empty_dict(self) -> None:
        provider = _make_provider()
        out = provider.extract_quotes_bundled("transcript", [])
        assert out == {}


class TestExtractQuotesBundledHappy:
    def test_parses_response_and_resolves_quote_spans(self) -> None:
        provider = _make_provider()
        transcript = "The cat sat on the mat. The dog barked loudly."
        bundled_payload = json.dumps(
            {
                "0": ["The cat sat on the mat."],
                "1": ["The dog barked loudly."],
            }
        )
        with _patch_retry(bundled_payload):
            out = provider.extract_quotes_bundled(transcript, ["cat", "dog"])
        assert set(out.keys()) == {0, 1}
        for idx in (0, 1):
            qc = out[idx][0]
            assert transcript[qc.char_start : qc.char_end] == qc.text

    def test_dedups_repeated_quotes(self) -> None:
        provider = _make_provider()
        with _patch_retry(json.dumps({"0": ["Alpha.", "Alpha.", "Beta."]})):
            out = provider.extract_quotes_bundled("Alpha. Beta.", ["i"])
        texts = [qc.text for qc in out[0]]
        assert texts.count("Alpha.") == 1
        assert "Beta." in texts

    def test_handles_empty_response_content(self) -> None:
        provider = _make_provider()
        # response.content == [] should fall through to parser with empty string.
        empty_response = Mock()
        empty_response.content = []
        empty_response.usage = Mock(input_tokens=10, output_tokens=0)
        with patch(
            "podcast_scraper.utils.provider_metrics.retry_with_metrics",
            return_value=empty_response,
        ):
            with pytest.raises(BundleExtractParseError):
                provider.extract_quotes_bundled("transcript", ["i"])


class TestExtractQuotesBundledErrors:
    def test_parser_failure_raises(self) -> None:
        provider = _make_provider()
        with _patch_retry("not valid JSON {"):
            with pytest.raises(BundleExtractParseError):
                provider.extract_quotes_bundled("transcript", ["i"])

    def test_sdk_exception_propagates(self) -> None:
        provider = _make_provider()
        with _patch_retry_raises(RuntimeError("upstream 503")):
            with pytest.raises(RuntimeError, match="upstream 503"):
                provider.extract_quotes_bundled("transcript", ["i"])


# ---------------------------------------------------------------------------
# score_entailment_bundled


class TestScoreEntailmentBundled:
    def test_not_initialized_returns_empty(self) -> None:
        provider = _make_provider()
        provider._summarization_initialized = False
        out = provider.score_entailment_bundled([("p", "h")])
        assert out == {}

    def test_no_pairs_returns_empty(self) -> None:
        provider = _make_provider()
        out = provider.score_entailment_bundled([])
        assert out == {}

    def test_chunks_pairs_and_merges_indices(self) -> None:
        provider = _make_provider()
        pairs = [(f"p{i}", f"h{i}") for i in range(3)]
        with patch.object(
            provider,
            "_score_entailment_bundled_chunk",
            side_effect=[{0: 0.7, 1: 0.8}, {0: 0.6}],
        ):
            out = provider.score_entailment_bundled(pairs, chunk_size=2)
        assert out == {0: 0.7, 1: 0.8, 2: 0.6}


class TestScoreEntailmentBundledChunk:
    def test_parses_chunk_response(self) -> None:
        provider = _make_provider()
        pairs = [("p1", "h1"), ("p2", "h2")]
        with _patch_retry(json.dumps({"0": 0.65, "1": 0.40})):
            scores = provider._score_entailment_bundled_chunk(
                chunk_pairs=pairs, pipeline_metrics=None
            )
        assert scores == {0: 0.65, 1: 0.40}

    def test_parser_failure_raises(self) -> None:
        provider = _make_provider()
        with _patch_retry("not JSON"):
            with pytest.raises(BundleNliParseError):
                provider._score_entailment_bundled_chunk(
                    chunk_pairs=[("p", "h")], pipeline_metrics=None
                )

    def test_sdk_exception_propagates(self) -> None:
        provider = _make_provider()
        with _patch_retry_raises(RuntimeError("503")):
            with pytest.raises(RuntimeError, match="503"):
                provider._score_entailment_bundled_chunk(
                    chunk_pairs=[("p", "h")], pipeline_metrics=None
                )
