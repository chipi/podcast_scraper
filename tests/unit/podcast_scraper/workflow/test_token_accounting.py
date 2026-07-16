"""Token accounting: normalise every provider's usage shape, then derive cost from tokens.

The core of the Problem-1 fix — tokens are the ground truth, cost is a projection. These tests pin
the two things that are easy to get wrong: the differing cached-token conventions (OpenAI/DeepSeek/
Gemini report cached as a SUBSET of input; Anthropic reports it SEPARATELY), and the cost math that
bills cached/cache-write tokens at their own rates.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from podcast_scraper.workflow.token_accounting import (
    cost_from_tokens,
    extract_token_usage,
    TokenUsage,
)

pytestmark = pytest.mark.unit


def test_openai_style_cached_is_a_subset_of_prompt_tokens() -> None:
    resp = NS(
        usage=NS(
            prompt_tokens=1000, completion_tokens=200, prompt_tokens_details=NS(cached_tokens=800)
        )
    )
    u = extract_token_usage("openai", resp)
    assert u.input_tokens == 1000 and u.output_tokens == 200
    assert u.cached_input_tokens == 800
    assert u.uncached_input_tokens == 200  # 1000 total - 800 cached


def test_deepseek_cache_hit_field_is_read() -> None:
    resp = NS(usage=NS(prompt_tokens=5000, completion_tokens=100, prompt_cache_hit_tokens=4608))
    u = extract_token_usage("deepseek", resp)
    assert u.input_tokens == 5000 and u.cached_input_tokens == 4608
    assert u.uncached_input_tokens == 392


def test_anthropic_cache_fields_are_separate_and_folded_into_total() -> None:
    """Anthropic input_tokens is non-cached; cache_read/creation add on top → total = sum."""
    resp = NS(
        usage=NS(
            input_tokens=300,
            output_tokens=150,
            cache_read_input_tokens=700,
            cache_creation_input_tokens=100,
        )
    )
    u = extract_token_usage("anthropic", resp)
    assert u.input_tokens == 1100  # 300 + 700 + 100
    assert u.cached_input_tokens == 700
    assert u.cache_write_tokens == 100
    assert u.uncached_input_tokens == 300  # 1100 - 700 - 100


def test_gemini_cached_content_is_a_subset() -> None:
    resp = NS(
        usage_metadata=NS(
            prompt_token_count=2000, candidates_token_count=400, cached_content_token_count=1500
        )
    )
    u = extract_token_usage("gemini", resp)
    assert u.input_tokens == 2000 and u.output_tokens == 400
    assert u.cached_input_tokens == 1500 and u.uncached_input_tokens == 500


def test_missing_usage_is_empty_not_an_error() -> None:
    assert extract_token_usage("openai", None).is_empty()
    assert extract_token_usage("anthropic", NS()).is_empty()
    assert extract_token_usage("mistral", NS(usage=None)).is_empty()


def test_cost_bills_cached_and_write_at_their_own_rates() -> None:
    # 200 uncached @ $3/M + 700 cached @ $0.30/M + 100 write @ $3.75/M + 150 out @ $15/M
    usage = TokenUsage(
        input_tokens=1000, output_tokens=150, cached_input_tokens=700, cache_write_tokens=100
    )
    rates = {
        "input_cost_per_1m_tokens": 3.0,
        "output_cost_per_1m_tokens": 15.0,
        "cached_input_cost_per_1m_tokens": 0.30,
        "cache_write_cost_per_1m_tokens": 3.75,
    }
    expected = (200 / 1e6) * 3.0 + (700 / 1e6) * 0.30 + (100 / 1e6) * 3.75 + (150 / 1e6) * 15.0
    assert cost_from_tokens(usage, rates) == round(expected, 6)


def test_cost_defaults_cached_rate_to_input_rate_when_absent() -> None:
    usage = TokenUsage(input_tokens=1000, output_tokens=0, cached_input_tokens=800)
    rates = {"input_cost_per_1m_tokens": 1.0, "output_cost_per_1m_tokens": 2.0}
    # cached defaults to input rate → whole 1000 input billed at $1/M
    assert cost_from_tokens(usage, rates) == round((1000 / 1e6) * 1.0, 6)


def test_cost_is_none_when_no_rates_known() -> None:
    """Cost genuinely unknown → None (keep the tokens, drop the projection), never a fake 0."""
    usage = TokenUsage(input_tokens=1000, output_tokens=200)
    assert cost_from_tokens(usage, {}) is None
    assert cost_from_tokens(usage, {"unrelated": 1}) is None
