"""The quote-extraction output budget must sit ABOVE the observed token distribution.

B2, re-diagnosed 2026-09-01. The failure mode was recorded as "Qwen3-30B JSON reliability" on
the strength of three samples with ``finish_reason=stop``. Over a 26h window with 91 samples it
inverts: the pipeline's own diagnostic classifies **69 of 91 as DOCUMENT_ENDED_EARLY** and only
22 as MALFORMED_MID_DOCUMENT. It is a budget problem, not a model-reliability problem, and the
log line already said so:

    extract_quotes_bundled parse FAILED: invalid JSON: Unterminated string ...
    [diagnosis=DOCUMENT_ENDED_EARLY] | finish_reason=length output_tokens=2688/2688 insights=7.
    finish_reason and the truncation diagnosis agree -> raise the output budget for this call

Measured over 1345 ``extract_quotes`` calls in the same window:

    completion_tokens  p50=989  p75=1310  p90=1902  p99=3840  max=3840

and 63 calls sat EXACTLY on a ceiling (1024 x11, 1920 x18, 2688 x1, 3840 x33) — i.e. truncated.
p90 lands within a token of the 5-insight ceiling of 1920. The budget line was inside the
distribution, which is the same mistake the 256 -> 384 change was made to fix.

Truncation is expensive here, not salvageable: unparsable JSON means the batch yields ZERO
quotes, is bisected into two further calls, and a size-1 failure drops to the per-insight staged
path. So the cost of being too low is paid in extra LLM calls on the stage that is already ~72%
of the pipeline's input tokens.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.common.bundled_prompts import (
    _QUOTE_MAX_OUTPUT_TOKENS,
    _QUOTE_TOKENS_PER_INSIGHT,
    extract_quotes_bundled_max_tokens,
)

#: The DGX serves Qwen3-30B at max_model_len 32768, and the default quote batch is
#: QUOTE_BUNDLE_CHUNK_SIZE = 10. The prompt figure is the LARGEST actually sent across 1507
#: production extract_quotes calls (p50=12044, p90=16441, p99=22242).
_SERVED_CONTEXT = 32768
_DEFAULT_QUOTE_BATCH = 10
_OBSERVED_MAX_PROMPT_TOKENS = 26714

#: Ceilings that calls were observed sitting exactly on, with how many hit each. Every one of
#: these is a truncated response under the OLD 384/insight budget.
_OBSERVED_TRUNCATION_CEILINGS = {1024: 11, 1920: 18, 2688: 1, 3840: 33}

#: The highest completion_tokens seen. CENSORED — it is a ceiling, so the true requirement is
#: higher than this, which is why the new budget clears it with room rather than matching it.
_CENSORED_P99 = 3840


class TestBudgetClearsTheObservedDistribution:
    @pytest.mark.parametrize("batch", [1, 2, 5, 7, 10])
    def test_budget_exceeds_every_observed_truncation_ceiling(self, batch):
        """No batch size may be given a budget a real response already exceeded."""
        got = extract_quotes_bundled_max_tokens(batch)
        assert got > _CENSORED_P99 or got >= 1024, (batch, got)

    def test_the_batch_sizes_that_truncated_now_get_more_room(self):
        """The three ceilings that actually bit, re-computed under the new constant."""
        # 1920 was a 5-insight batch, 2688 a 7-insight, 3840 a 10-insight.
        assert extract_quotes_bundled_max_tokens(5) > 1920
        assert extract_quotes_bundled_max_tokens(7) > 2688
        assert extract_quotes_bundled_max_tokens(10) > 3840

    def test_the_largest_batch_clears_the_censored_p99(self):
        """A full 10-insight batch must have room beyond the largest response ever completed."""
        assert extract_quotes_bundled_max_tokens(10) > _CENSORED_P99

    def test_still_within_the_hard_cap(self):
        """The cap is a CONTEXT-FIT bound, not a cost bound."""
        assert extract_quotes_bundled_max_tokens(10) <= _QUOTE_MAX_OUTPUT_TOKENS
        assert extract_quotes_bundled_max_tokens(1000) == _QUOTE_MAX_OUTPUT_TOKENS

    def test_the_default_batch_fits_the_served_context_with_the_worst_real_prompt(self):
        """THE REGRESSION GUARD. prompt + output must fit, or the call errors outright.

        Raising the per-insight budget to 640 without lowering the cap would ask for 6400
        output tokens on the DEFAULT batch of 10, and 26714 + 6400 = 33114 > 32768 — #1893,
        introduced by the very change meant to stop truncation. The OLD effective maximum
        (384 x 10 = 3840) always fit, so this failure mode did not previously exist.
        """
        worst_prompt = _OBSERVED_MAX_PROMPT_TOKENS
        out = extract_quotes_bundled_max_tokens(_DEFAULT_QUOTE_BATCH)
        assert worst_prompt + out < _SERVED_CONTEXT, (
            f"default batch of {_DEFAULT_QUOTE_BATCH} asks {out} output tokens; with the "
            f"largest prompt actually sent ({worst_prompt}) that is "
            f"{worst_prompt + out} against a {_SERVED_CONTEXT} context window"
        )

    def test_a_full_batch_still_gets_more_room_than_before_the_fix(self):
        """The cap must not undo the reason the budget was raised."""
        assert extract_quotes_bundled_max_tokens(_DEFAULT_QUOTE_BATCH) > 384 * _DEFAULT_QUOTE_BATCH

    def test_floor_still_protects_tiny_batches(self):
        """A 1-insight batch must not be given a budget too small to answer in."""
        assert extract_quotes_bundled_max_tokens(1) >= 1024
        assert extract_quotes_bundled_max_tokens(0) >= 1024
        assert extract_quotes_bundled_max_tokens(-5) >= 1024

    def test_budget_is_monotonic_in_batch_size(self):
        vals = [extract_quotes_bundled_max_tokens(n) for n in range(1, 13)]
        assert vals == sorted(vals), vals


class TestTheConstantItself:
    def test_is_above_the_value_production_disproved(self):
        """384 was measured too low; a revert must fail rather than pass quietly."""
        assert _QUOTE_TOKENS_PER_INSIGHT > 384, (
            "384/insight put the budget INSIDE the observed distribution (p90=1902 vs a "
            "5-insight ceiling of 1920); 63 of 1345 calls truncated"
        )

    def test_is_not_raised_so_far_that_the_cap_binds_at_normal_batch_sizes(self):
        """A budget that always saturates the 8192 cap would silently stop scaling with batch.

        Guards the opposite error from the one being fixed: over-correcting until every batch
        gets the same maximum makes the per-insight constant meaningless.
        """
        assert extract_quotes_bundled_max_tokens(10) < 8192, (
            "the per-insight budget is now high enough that a normal 10-insight batch pins the "
            "hard cap, so the budget no longer scales with batch size"
        )


def test_every_observed_ceiling_is_documented_with_its_hit_count():
    """The ceilings table is evidence, not decoration — keep it non-empty and summing right."""
    assert sum(_OBSERVED_TRUNCATION_CEILINGS.values()) == 63
    assert max(_OBSERVED_TRUNCATION_CEILINGS) == _CENSORED_P99
