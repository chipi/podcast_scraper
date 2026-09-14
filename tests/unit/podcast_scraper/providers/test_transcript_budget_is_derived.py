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


class TestThereIsNoGlobalContextBoundLeft:
    """The design decision, pinned so it cannot be quietly undone (#2050).

    A context window is a property of the model AS DEPLOYED. There must be no module-level
    constant anyone can reach for instead — every one of these previously existed, and each was a
    single deployment's serving flag acting as policy for the entire fleet.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "LLM_NARROWEST_CONTEXT_TOKENS",
            "GI_QUOTE_TRANSCRIPT_MAX_CHARS",
            "MAX_PROCESSABLE_EPISODE_SECONDS",
            "DEFAULT_DECLARED_CONTEXT_TOKENS",
        ],
    )
    def test_the_global_bounds_are_gone_and_must_not_come_back(self, name: str) -> None:
        assert not hasattr(c, name), (
            f"{name} is back. A global window constant makes one deployment's serving flag into "
            "corpus policy for every provider — a 1M-token model clipped by a DGX container's "
            "--max-model-len. Declare it on the StageOption instead."
        )

    def test_the_evidence_prompt_has_no_default_budget_either(self) -> None:
        from podcast_scraper.providers.common import evidence_prompts

        assert not hasattr(evidence_prompts, "DEFAULT_TRANSCRIPT_BUDGET_CHARS")


class TestAnUnknownWindowMeansUnboundedNotGuessed:
    """If nothing declared a window, we do not know it — and must not invent one."""

    @pytest.mark.parametrize("window", [None, 0])
    def test_an_unknown_window_yields_no_bound(self, window) -> None:
        assert c.transcript_budget_chars(window, response_tokens=2048) is None

    def test_a_known_but_hopeless_window_yields_zero_not_none(self) -> None:
        # 0 and None are different answers: "nothing fits, refuse" vs "we do not know, do not
        # clip". Collapsing them would silently send an empty transcript.
        assert c.transcript_budget_chars(512, response_tokens=2048) == 0

    def test_an_unknown_budget_does_not_clip_the_transcript(self) -> None:
        from podcast_scraper.providers.common.bundled_prompts import transcript_clip

        assert len(transcript_clip("x" * 300_000, max_chars=None)) == 300_000

    def test_the_quote_prompt_does_not_clip_on_an_unknown_budget(self) -> None:
        from podcast_scraper.providers.common.evidence_prompts import (
            render_extract_quote_prompt,
        )

        long = "word " * 60_000
        _, user = render_extract_quote_prompt("openai", long, "an insight", budget_chars=None)
        assert len(user) > 250_000, "an unknown window must send the transcript, not truncate it"


class TestTheBudgetRatioIsMeasuredNotAssumed:
    """The ratio governs how much of every episode the model sees, so it is measured, not argued.

    ``_context_clamp_token_budget`` separately estimates prompts at 3.5 chars/token. That is a
    RECOVERY estimate — deliberately pessimistic, because over-estimating there shrinks a reply
    while under-estimating sends a request that 400s. It is not the sizing ratio and the two do
    not have to match.
    """

    def test_the_budget_ratio_is_the_one_production_proved(self) -> None:
        """MEASURED FROM PRODUCTION, not from fixtures.

        vLLM rejected the 106,905-char staged-quote prompt 508 times in 30 days with an identical
        count: 30,721 input tokens. 106,905 / 30,721 = 3.48 chars/token on real transcripts.

        The fixtures measure 4.44 — they carry prod's format but simpler vocabulary — and that
        figure "proves" the 106,905 budget fits when production proves it does not. Anyone who
        raises this constant on a fixture measurement re-ships the overflow.
        """
        assert c.CHARS_PER_TOKEN_BUDGET_RATIO == 3.5
        # 3.5 is fractionally ABOVE production's 3.48. The 0.9 margin is what buys the
        # headroom, so the EFFECTIVE rate is the invariant worth pinning.
        real_ratio = 106_905 / 30_721
        effective = c.CHARS_PER_TOKEN_BUDGET_RATIO * c.CONTEXT_BUDGET_SAFETY_MARGIN
        assert effective < real_ratio, (
            "sizing at or above the rate production actually exhibits is how 508 prompts "
            "overflowed by exactly one token"
        )

    @pytest.mark.parametrize("window", [32_768, 65_536, 128_000])
    def test_a_derived_budget_fits_at_the_measured_ratio(self, window: int) -> None:
        budget = c.transcript_budget_chars(window, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
        assert budget is not None
        # 3.48 is what production actually exhibits (106,905 chars -> 30,721 tokens, 508x).
        used = budget / 3.48 + c.GI_QUOTE_RESPONSE_TOKENS + c.GI_QUOTE_INSTRUCTION_TOKEN_RESERVE
        assert used <= window


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
        assert got is not None and got >= expected_at_least

    def test_the_old_literal_did_not_fit_the_window_prod_actually_serves(self) -> None:
        # The finding that makes this issue urgent rather than cosmetic: 120,000 chars was ALREADY
        # over the DGX window, so insight extraction was handing the server prompts it had to
        # reject. Deriving it TIGHTENS the clip here — that is the fix, not a regression.
        derived = c.transcript_budget_chars(32_768, response_tokens=1024)
        assert derived is not None and derived < 120_000

    def test_raising_the_window_alone_lifts_the_episode_ceiling(self) -> None:
        # #1985's whole premise: the flag raise must move the ceiling with it, or long episodes
        # get admitted by one number and truncated by six others.
        at_64k = c.transcript_budget_chars(65_536, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
        assert at_64k is not None
        ceiling_min = int((at_64k / c.CHARS_PER_MINUTE_OF_SPEECH) * 60) // 60
        assert ceiling_min > 200, "64k must clear Dwarkesh's 141-minute episodes with room"


class TestItRefusesRatherThanSendingAPromptThatCannotFit:
    def test_a_window_smaller_than_its_reserves_yields_zero_not_a_negative_slice(self) -> None:
        # A negative budget used as `text[:budget]` silently slices from the END of the
        # transcript — the model would receive the tail of the episode and nothing else.
        assert c.transcript_budget_chars(512, response_tokens=2048) == 0
        assert c.transcript_budget_chars(0, response_tokens=2048) is None

    def test_a_zero_budget_clips_to_empty_rather_than_to_everything(self) -> None:
        from podcast_scraper.providers.common.bundled_prompts import transcript_clip

        assert transcript_clip("x" * 1000, max_chars=0) == ""

    def test_transcript_clip_has_no_default_budget_at_all(self) -> None:
        # It used to default to 50,000 — sized for no model in particular, and applied to every
        # caller that omitted the window.
        from podcast_scraper.providers.common.bundled_prompts import transcript_clip

        assert len(transcript_clip("x" * 100_000)) == 100_000


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

    def test_with_nothing_declared_the_window_is_None_not_a_guess(self) -> None:
        # The provider used to assume 128,000 here — against a real 32,768 on the DGX, wrong by 4x.
        p = self._provider()
        assert p.served_context_tokens() is None
        assert p.transcript_budget_chars(response_tokens=2048) is None

    def test_the_budget_method_uses_that_window(self) -> None:
        p = self._provider(vllm_max_context_tokens=65_536)
        assert p.transcript_budget_chars(
            response_tokens=c.GI_QUOTE_RESPONSE_TOKENS
        ) == c.transcript_budget_chars(65_536, response_tokens=c.GI_QUOTE_RESPONSE_TOKENS)
