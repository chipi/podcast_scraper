"""The quote-extraction transcript budget must fit the context window (#1975).

``GI_QUOTE_TRANSCRIPT_MAX_CHARS`` was a literal 150_000, chosen when the narrowest model in use
had a 64k-token window. Prod now serves NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 at **32,768**, so
that budget was larger than the entire context — a 141-minute Dwarkesh episode (~118k chars)
passed the budget check with no warning and then failed the call outright:

    context limit 32768 cannot fit this request: the prompt alone is ~43300 tokens,
    leaving -10532 for the reply. Clamping the output budget cannot [help]

238 such failures in 24h. A budget larger than the window guarantees the call yields NOTHING;
truncating is a real loss but strictly better.

These tests pin the invariant rather than the number, so the next model change cannot silently
re-open the hole.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config_constants as c

pytestmark = pytest.mark.unit


def test_the_whole_prompt_fits_the_narrowest_context() -> None:
    """Transcript + instructions + reply must all fit, or every long episode fails."""
    transcript_tokens = c.GI_QUOTE_TRANSCRIPT_MAX_CHARS / c.CHARS_PER_TOKEN_ESTIMATE
    total = transcript_tokens + c.GI_QUOTE_RESPONSE_TOKENS + c.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE
    assert total <= c.LLM_NARROWEST_CONTEXT_TOKENS, (
        f"prompt budget {total:.0f} tokens exceeds the {c.LLM_NARROWEST_CONTEXT_TOKENS}-token "
        "window — long episodes will fail the call rather than truncate"
    )


def test_a_safety_margin_is_retained() -> None:
    """chars-per-token is an approximation; tokenisers vary by content, so do not sail close."""
    transcript_tokens = c.GI_QUOTE_TRANSCRIPT_MAX_CHARS / c.CHARS_PER_TOKEN_ESTIMATE
    used = transcript_tokens + c.GI_QUOTE_RESPONSE_TOKENS + c.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE
    assert used <= c.LLM_NARROWEST_CONTEXT_TOKENS * 0.95


def test_the_budget_is_still_large_enough_for_a_normal_episode() -> None:
    """The guard must not re-create the problem it replaced.

    A previous 50_000 hid the last third of every episode — zero of 1,418 grounded quotes fell
    beyond the cut, which looked like "insight density is concentrated early" and was truncation.
    At ~840 chars/min of speech the budget must comfortably clear a typical 45-90 minute show.
    """
    chars_per_minute = 840
    ninety_minutes = 90 * chars_per_minute  # ~75,600 chars
    assert c.GI_QUOTE_TRANSCRIPT_MAX_CHARS > ninety_minutes


def test_the_budget_is_derived_not_hardcoded() -> None:
    """Recomputing from the parts must reproduce it — otherwise it can drift silently again."""
    expected = int(
        (
            c.LLM_NARROWEST_CONTEXT_TOKENS
            - c.GI_QUOTE_RESPONSE_TOKENS
            - c.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE
        )
        * c.CHARS_PER_TOKEN_ESTIMATE
        * 0.9
    )
    assert c.GI_QUOTE_TRANSCRIPT_MAX_CHARS == expected
