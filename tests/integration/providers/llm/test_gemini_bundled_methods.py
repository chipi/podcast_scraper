"""Unit tests for Gemini's GIL-evidence bundled methods (#698).

Covers ``extract_quotes_bundled`` + ``score_entailment_bundled`` +
``_score_entailment_bundled_chunk`` on
:class:`podcast_scraper.providers.gemini.gemini_provider.GeminiProvider`.

The Gemini SDK is mocked at sys.modules import so the actual ``google.genai``
package isn't required. Each test wires a fake client + summary state on the
provider and asserts the bundled wrapper:

- Builds the right prompt via ``providers.common.bundled_prompts``.
- Calls the SDK once per chunk.
- Parses the response via ``providers.common.bundle_*_parser``.
- Resolves verbatim quote spans against the transcript (Layer A only).
- Returns the correctly-shaped dict.
- Falls through (raises) on parse errors so the dispatcher can record a
  fallback metric.
"""

from __future__ import annotations

import importlib.util
import json
from unittest.mock import MagicMock, Mock, patch

import pytest

mock_google = MagicMock()
mock_genai_module = MagicMock()
mock_api_core = MagicMock()
mock_google.__spec__ = importlib.util.spec_from_loader("google", loader=None)
mock_genai_module.__spec__ = importlib.util.spec_from_loader("google.genai", loader=None)
mock_api_core.__spec__ = importlib.util.spec_from_loader("google.api_core", loader=None)
_patch_google = patch.dict(
    "sys.modules",
    {
        "google": mock_google,
        "google.genai": mock_genai_module,
        "google.api_core": mock_api_core,
    },
)


def setUpModule():
    # Scope the SDK mocks to this module only — otherwise they leak into other
    # integration test files that need the real SDK. Same pattern as
    # tests/integration/providers/llm/test_gemini_provider.py.
    _patch_google.start()


def tearDownModule():
    _patch_google.stop()


from podcast_scraper import config
from podcast_scraper.providers.common.bundle_extract_parser import BundleExtractParseError
from podcast_scraper.providers.common.bundle_nli_parser import BundleNliParseError
from podcast_scraper.providers.gemini.gemini_provider import GeminiProvider

pytestmark = [pytest.mark.integration]


def _make_provider() -> GeminiProvider:
    """Construct a GeminiProvider in the post-initialize() state with no real SDK."""
    cfg = config.Config(
        summary_provider="gemini",
        gemini_summary_model="gemini-2.5-flash-lite",
        gemini_api_key="test-key",
    )
    provider = GeminiProvider(cfg)
    provider.client = MagicMock()
    provider._summarization_initialized = True
    provider.summary_model = "gemini-2.5-flash-lite"
    return provider


def _patch_gemini_call(text: str):
    """Patch the in-method imports so the actual SDK call is bypassed.

    ``retry_with_metrics`` is the abstraction the bundled methods call into.
    Patching it lets tests control the response object directly without needing
    to model the genai client's exact SDK shape.
    """
    fake_response = Mock()
    fake_response.text = text
    return patch(
        "podcast_scraper.utils.provider_metrics.retry_with_metrics",
        return_value=fake_response,
    )


def _patch_gemini_tokens(in_tok: int = 100, out_tok: int = 50):
    return patch(
        "podcast_scraper.utils.provider_metrics.gemini_generate_usage_tokens",
        return_value=(in_tok, out_tok),
    )


# ---------------------------------------------------------------------------
# extract_quotes_bundled — early-return paths


class TestExtractQuotesBundledEarlyReturn:
    def test_returns_empty_dict_for_each_insight_when_not_initialized(self) -> None:
        provider = _make_provider()
        provider._summarization_initialized = False
        out = provider.extract_quotes_bundled("transcript", ["i1", "i2", "i3"])
        assert out == {0: [], 1: [], 2: []}

    def test_returns_empty_dict_for_each_insight_when_transcript_empty(self) -> None:
        provider = _make_provider()
        out = provider.extract_quotes_bundled("", ["i1", "i2"])
        assert out == {0: [], 1: []}

    def test_returns_empty_dict_when_no_insights(self) -> None:
        provider = _make_provider()
        out = provider.extract_quotes_bundled("some transcript", [])
        assert out == {}


# ---------------------------------------------------------------------------
# extract_quotes_bundled — happy path


class TestExtractQuotesBundledHappy:
    def test_parses_bundled_response_and_resolves_quote_spans(self) -> None:
        provider = _make_provider()
        transcript = "The cat sat on the mat. The dog barked loudly. The bird flew away."
        # 3 insights, model returns one quote per insight; all are verbatim slices.
        bundled_payload = json.dumps(
            {
                "0": ["The cat sat on the mat."],
                "1": ["The dog barked loudly."],
                "2": ["The bird flew away."],
            }
        )
        with _patch_gemini_call(bundled_payload), _patch_gemini_tokens():
            out = provider.extract_quotes_bundled(
                transcript, ["cat insight", "dog insight", "bird insight"]
            )
        assert set(out.keys()) == {0, 1, 2}
        for idx in (0, 1, 2):
            assert len(out[idx]) == 1
            qc = out[idx][0]
            # Verbatim resolution: char_start/char_end must point into transcript.
            assert transcript[qc.char_start : qc.char_end] == qc.text
            assert qc.qa_score == 1.0

    def test_dedups_repeated_quotes_per_insight(self) -> None:
        provider = _make_provider()
        transcript = "Alpha. Beta. Gamma."
        # Model returns the same quote twice for one insight; bundled wrapper
        # should keep only one verbatim per insight.
        bundled_payload = json.dumps({"0": ["Alpha.", "Alpha.", "Beta."]})
        with _patch_gemini_call(bundled_payload), _patch_gemini_tokens():
            out = provider.extract_quotes_bundled(transcript, ["insight"])
        assert len(out[0]) == 2
        texts = [qc.text for qc in out[0]]
        assert texts.count("Alpha.") == 1
        assert "Beta." in texts

    def test_drops_quotes_that_dont_resolve_in_transcript(self) -> None:
        provider = _make_provider()
        transcript = "Real text only here."
        bundled_payload = json.dumps({"0": ["hallucinated quote that never appeared", "Real text"]})
        with _patch_gemini_call(bundled_payload), _patch_gemini_tokens():
            out = provider.extract_quotes_bundled(transcript, ["insight"])
        # First quote can't be resolved → dropped. Second is verbatim → kept.
        assert len(out[0]) == 1
        assert out[0][0].text == "Real text"

    def test_drops_blank_quote_strings(self) -> None:
        provider = _make_provider()
        transcript = "Real text."
        bundled_payload = json.dumps({"0": ["", "  ", "Real text."]})
        with _patch_gemini_call(bundled_payload), _patch_gemini_tokens():
            out = provider.extract_quotes_bundled(transcript, ["insight"])
        assert len(out[0]) == 1

    def test_missing_insight_index_returns_empty_list(self) -> None:
        # Parser returns dict; bundled wrapper iterates 0..len(insights). If parser
        # missed an index, that idx maps to [].
        provider = _make_provider()
        bundled_payload = json.dumps({"0": ["text"], "2": ["text"]})
        with _patch_gemini_call(bundled_payload), _patch_gemini_tokens():
            out = provider.extract_quotes_bundled("text", ["i0", "i1", "i2"])
        assert out[1] == []


# ---------------------------------------------------------------------------
# extract_quotes_bundled — error paths


class TestExtractQuotesBundledErrors:
    def test_parser_failure_raises_so_caller_can_fall_back(self) -> None:
        provider = _make_provider()
        with _patch_gemini_call("not valid JSON {"), _patch_gemini_tokens():
            with pytest.raises(BundleExtractParseError):
                provider.extract_quotes_bundled("transcript", ["i"])

    def test_sdk_exception_propagates(self) -> None:
        provider = _make_provider()
        with patch(
            "podcast_scraper.utils.provider_metrics.retry_with_metrics",
            side_effect=RuntimeError("upstream 503"),
        ):
            with pytest.raises(RuntimeError, match="upstream 503"):
                provider.extract_quotes_bundled("transcript", ["i"])


# ---------------------------------------------------------------------------
# score_entailment_bundled — early-return + chunking


class TestScoreEntailmentBundled:
    def test_returns_empty_dict_when_not_initialized(self) -> None:
        provider = _make_provider()
        provider._summarization_initialized = False
        out = provider.score_entailment_bundled([("p", "h")])
        assert out == {}

    def test_returns_empty_dict_for_no_pairs(self) -> None:
        provider = _make_provider()
        out = provider.score_entailment_bundled([])
        assert out == {}

    def test_chunks_pairs_by_chunk_size_and_merges_indices(self) -> None:
        # 3 pairs with chunk_size=2 → 2 chunks; first chunk indices 0,1 → second 2.
        provider = _make_provider()
        pairs = [(f"p{i}", f"h{i}") for i in range(3)]
        # Each chunk call returns local indices 0..len(chunk_pairs)-1; the wrapper
        # offsets them by chunk_start.
        with patch.object(
            provider,
            "_score_entailment_bundled_chunk",
            side_effect=[{0: 0.7, 1: 0.8}, {0: 0.6}],
        ):
            out = provider.score_entailment_bundled(pairs, chunk_size=2)
        assert out == {0: 0.7, 1: 0.8, 2: 0.6}

    def test_chunk_size_floored_at_1(self) -> None:
        provider = _make_provider()
        with patch.object(
            provider,
            "_score_entailment_bundled_chunk",
            return_value={0: 0.5},
        ) as chunk_fn:
            provider.score_entailment_bundled([("p", "h")], chunk_size=0)
        assert chunk_fn.call_count == 1


# ---------------------------------------------------------------------------
# _score_entailment_bundled_chunk


class TestScoreEntailmentBundledChunk:
    def test_parses_chunk_response_into_index_score_map(self) -> None:
        provider = _make_provider()
        pairs = [("p1", "h1"), ("p2", "h2")]
        with _patch_gemini_call(json.dumps({"0": 0.65, "1": 0.40})), _patch_gemini_tokens():
            scores = provider._score_entailment_bundled_chunk(
                chunk_pairs=pairs, pipeline_metrics=None
            )
        assert scores == {0: 0.65, 1: 0.40}

    def test_parser_failure_raises(self) -> None:
        provider = _make_provider()
        with _patch_gemini_call("not JSON"), _patch_gemini_tokens():
            with pytest.raises(BundleNliParseError):
                provider._score_entailment_bundled_chunk(
                    chunk_pairs=[("p", "h")], pipeline_metrics=None
                )

    def test_sdk_exception_propagates(self) -> None:
        provider = _make_provider()
        with patch(
            "podcast_scraper.utils.provider_metrics.retry_with_metrics",
            side_effect=RuntimeError("503"),
        ):
            with pytest.raises(RuntimeError, match="503"):
                provider._score_entailment_bundled_chunk(
                    chunk_pairs=[("p", "h")], pipeline_metrics=None
                )
