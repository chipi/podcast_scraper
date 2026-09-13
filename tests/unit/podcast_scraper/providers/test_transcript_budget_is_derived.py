"""Transcript budgets derive from the SERVED window, not from a global constant (#2050).

Six sites each carried their own literal — 120,000 chars for insight extraction, 50,000 for
bundled quotes, 8,000 for a speaker description, nothing at all for ``summarize()`` — and none of
them could see the window they were supposed to fit. Fixing one moved the overflow to the next,
which is why #1893 reopened four times after being closed as fixed twice.

The literal was wrong in BOTH directions at once: 120,000 chars is ~34,000 tokens at the 3.5
chars/token ratio prod actually runs, so it did not fit the DGX's 32,768-token window at all —
while simultaneously clipping a 1M-token Gemini deployment to a fraction of what it could hold.
"""

from __future__ import annotations

import pytest

from podcast_scraper import config_constants as c

pytestmark = pytest.mark.unit


class TestTheBudgetRatioMatchesTheOverflowCheck:
    """The inconsistency that made #1893 reopen four times.

    Budgets were derived at 4 chars/token with a 0.9 margin — an effective 3.6 — while
    ``_context_clamp_token_budget`` judged overflow at 3.5. So a budget could be satisfied and the
    call still rejected. These pin the two to the same figure.
    """

    def test_the_two_ratios_agree(self) -> None:
        # If someone "optimises" the budget ratio upward, this is the test that should stop them.
        assert c.CHARS_PER_TOKEN_BUDGET_RATIO == 3.5

    def test_the_quote_budget_now_fits_at_the_ratio_prod_actually_tokenises_at(self) -> None:
        used = (
            c.GI_QUOTE_TRANSCRIPT_MAX_CHARS / 3.5
            + c.GI_QUOTE_RESPONSE_TOKENS
            + c.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE
        )
        assert used <= c.LLM_NARROWEST_CONTEXT_TOKENS, (
            "the pre-#2050 value of 106,905 chars needed 33,616 tokens of a 32,768 window — it "
            "passed the budget check and then 400'd the call"
        )

    def test_the_ceiling_tightened_and_that_is_the_point(self) -> None:
        # 127 -> 111 min. Be honest about what this costs: an episode at ~127 min genuinely could
        # not fit, but one at 115 min would have. The 111-125 band is headroom deliberately given
        # up, because the chars/token ratio varies with speaker density and timestamps and
        # under-estimating costs the whole stage. #1985 (64k) returns it as ~234 min.
        assert c.MAX_PROCESSABLE_EPISODE_SECONDS // 60 == 111

    def test_raising_the_window_more_than_repays_the_tightening(self) -> None:
        at_64k = c.transcript_budget_chars(65_536, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
        assert int((at_64k / c.CHARS_PER_MINUTE_OF_SPEECH) * 60) // 60 > 127


class TestTheBudgetTracksTheWindow:
    @pytest.mark.parametrize(
        "window,expected_at_least",
        [
            (32_768, 90_000),
            (65_536, 190_000),
            (128_000, 390_000),
            (1_000_000, 3_100_000),
        ],
    )
    def test_a_wider_window_buys_a_wider_budget(self, window: int, expected_at_least: int) -> None:
        got = c.transcript_budget_chars(window, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
        assert got >= expected_at_least

    def test_the_old_literal_did_not_fit_the_window_prod_actually_serves(self) -> None:
        # The finding that makes this issue urgent rather than cosmetic: 120,000 chars was ALREADY
        # over the DGX window, so insight extraction was handing the server prompts it had to
        # reject. Deriving it TIGHTENS the clip here — that is the fix, not a regression.
        derived = c.transcript_budget_chars(32_768, response_tokens=1024)
        assert derived < 120_000

    def test_raising_the_window_alone_lifts_the_episode_ceiling(self) -> None:
        # #1985's whole premise: the flag raise must move the ceiling with it, or long episodes
        # get admitted by one number and truncated by six others.
        at_64k = c.transcript_budget_chars(65_536, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
        ceiling_min = int((at_64k / c.CHARS_PER_MINUTE_OF_SPEECH) * 60) // 60
        assert ceiling_min > 200, "64k must clear Dwarkesh's 141-minute episodes with room"


class TestItRefusesRatherThanSendingAPromptThatCannotFit:
    def test_a_window_smaller_than_its_reserves_yields_zero_not_a_negative_slice(self) -> None:
        # A negative budget used as `text[:budget]` silently slices from the END of the
        # transcript — the model would receive the tail of the episode and nothing else.
        assert c.transcript_budget_chars(512, response_tokens=2048) == 0
        assert c.transcript_budget_chars(0, response_tokens=2048) == 0

    def test_a_zero_budget_clips_to_empty_rather_than_to_everything(self) -> None:
        from podcast_scraper.providers.common.bundled_prompts import transcript_clip

        assert transcript_clip("x" * 1000, max_chars=0) == ""

    def test_transcript_clip_still_has_its_legacy_default_for_callers_with_no_provider(
        self,
    ) -> None:
        from podcast_scraper.providers.common.bundled_prompts import transcript_clip

        assert len(transcript_clip("x" * 100_000)) == 50_000


class TestTheProviderPrefersWhatTheServerSaid:
    def _provider(self, **overrides):
        from unittest.mock import MagicMock, patch

        from podcast_scraper.config import Config
        from podcast_scraper.providers.vllm import VLLMProvider

        m = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"
        base = dict(
            rss_url="https://example.com/feed.xml",
            summary_provider="vllm",
            speaker_detector_provider="vllm",
            generate_summaries=True,
            generate_metadata=True,
            vllm_api_base="http://dgx-llm-1:8003/v1",
            vllm_summary_model=m,
            vllm_speaker_model=m,
            vllm_verify_served_model=False,
        )
        base.update(overrides)
        with patch("openai.OpenAI", return_value=MagicMock()):
            return VLLMProvider(Config(**base))

    def test_a_declared_window_beats_the_openai_native_default(self) -> None:
        p = self._provider(vllm_max_context_tokens=32_768)
        assert p.served_context_tokens() == 32_768

    def test_a_limit_learned_from_a_400_beats_the_declared_one(self) -> None:
        # The server said this. Nothing outranks it.
        p = self._provider(vllm_max_context_tokens=128_000)
        p._context_limits[p.summary_model] = 32_768
        assert p.served_context_tokens() == 32_768

    def test_with_nothing_declared_it_falls_back_rather_than_guessing_upward(self) -> None:
        p = self._provider()
        assert p.served_context_tokens() == c.DEFAULT_DECLARED_CONTEXT_TOKENS

    def test_the_budget_method_uses_that_window(self) -> None:
        p = self._provider(vllm_max_context_tokens=65_536)
        assert p.transcript_budget_chars(
            response_tokens=c.GI_QUOTE_RESPONSE_TOKENS
        ) == c.transcript_budget_chars(65_536, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
