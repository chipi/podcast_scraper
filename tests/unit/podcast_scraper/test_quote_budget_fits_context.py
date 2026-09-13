"""The quote-extraction transcript budget must fit the window it is derived from (#1975, #2050).

History, because it explains why this file pins an invariant rather than a number:

``GI_QUOTE_TRANSCRIPT_MAX_CHARS`` was a literal 150,000, chosen when the narrowest model in use had
a 64k-token window. Prod moved to NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4 at **32,768** and the
budget became larger than the entire context — a 141-minute Dwarkesh episode (~118k chars) passed
the budget check with no warning and then failed the call outright:

    context limit 32768 cannot fit this request: the prompt alone is ~43300 tokens,
    leaving -10532 for the reply. Clamping the output budget cannot [help]

238 such failures in 24h. It was then "fixed" by deriving the number from a global constant sized
to the narrowest model — which made one DGX container's ``--max-model-len`` the corpus policy for
every provider in the fleet, including the 1M-token ones, and left the derivation 3% too generous
so #1893 reopened four more times.

#2050 deleted the global constant. The budget is now a function of the window the deployment
actually serves, so these tests check the FUNCTION across the range of windows the registry serves
rather than one materialized figure.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config_constants as c

pytestmark = pytest.mark.unit

# Windows the registry actually serves, narrow to wide.
SERVED_WINDOWS = [32_768, 65_536, 128_000, 200_000, 1_000_000]


@pytest.mark.parametrize("window", SERVED_WINDOWS)
def test_the_whole_prompt_fits_the_window_it_was_derived_from(window: int) -> None:
    """Transcript + instructions + reply must all fit, or every long episode fails."""
    budget = c.transcript_budget_chars(window, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
    assert budget is not None
    total = (
        budget / c.CHARS_PER_TOKEN_BUDGET_RATIO
        + c.GI_QUOTE_RESPONSE_TOKENS
        + c.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE
    )
    assert (
        total <= window
    ), f"prompt budget {total:.0f} tokens exceeds the {window}-token window it was derived from"


@pytest.mark.parametrize("window", SERVED_WINDOWS)
def test_it_keeps_real_headroom_not_just_arithmetic_headroom(window: int) -> None:
    """The chars/token ratio varies with speaker density; an exact fit is not a fit."""
    budget = c.transcript_budget_chars(window, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
    assert budget is not None
    used = budget / 3.48 + c.GI_QUOTE_RESPONSE_TOKENS + c.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE
    assert used <= window * 0.95


def test_the_dgx_window_still_holds_a_normal_episode() -> None:
    """A 90-minute episode is ~75,600 chars at 840 chars/min. The narrowest served window must
    not have quietly become unable to process ordinary content."""
    ninety_minutes = 90 * c.CHARS_PER_MINUTE_OF_SPEECH
    budget = c.transcript_budget_chars(32_768, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
    assert budget is not None and budget > ninety_minutes


def test_the_budget_ratio_is_the_one_production_proved() -> None:
    """MEASURED FROM PRODUCTION: vLLM rejected the 106,905-char prompt 508 times in 30 days,
    each time reporting 30,721 input tokens. 106,905 / 30,721 = 3.48 chars/token.

    The v2 fixtures measure 4.44 — same format, simpler vocabulary — and that figure would
    "prove" the old budget fits. It does not. Never re-derive this from fixtures.
    """
    assert c.CHARS_PER_TOKEN_BUDGET_RATIO == 3.5
    # 3.5 is fractionally ABOVE production's 3.48 — the 0.9 margin is what makes it safe,
    # so that is the invariant, not the bare ratio.
    effective = c.CHARS_PER_TOKEN_BUDGET_RATIO * c.CONTEXT_BUDGET_SAFETY_MARGIN
    assert effective < 106_905 / 30_721
    assert c.CHARS_PER_TOKEN_BUDGET_RATIO < c.CHARS_PER_TOKEN_ESTIMATE


def test_there_is_no_global_budget_constant_to_reach_for() -> None:
    for gone in ("GI_QUOTE_TRANSCRIPT_MAX_CHARS", "LLM_NARROWEST_CONTEXT_TOKENS"):
        assert not hasattr(c, gone), f"{gone} is back; see the module docstring for why it went"
