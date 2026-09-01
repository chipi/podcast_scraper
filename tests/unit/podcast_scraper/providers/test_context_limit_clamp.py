"""prompt + output must fit the served context, checked BEFORE the call (#1893).

THE FAILURE. Production, `sha-a624e77`:

    BadRequestError 400 — This model's maximum context length is 32768 tokens.
    However, you requested 2048 output tokens and your prompt contains at least
    30721 input tokens

Over by ONE token, and **56 errors in a 13-second burst** — every episode in flight failing
identically, because nothing checked that the request fit before sending it.

THE APPROACH. The limit is LEARNED from the server's own 400, not configured. A per-model
context table maintained by hand goes stale silently the moment a model is swapped behind a
gateway alias — and this repo has already been bitten by exactly that class of drift more than
once today. vLLM knows the true number; it just communicates it by failing. So: parse it once,
clamp and retry that call, and clamp every later call for that model up front. Same
self-healing shape as the existing `_temp_fixed_at_default` cache next to it.

WHAT IT DELIBERATELY WILL NOT DO. If the prompt alone leaves less than a usable reply's worth
of room, no output budget makes the request valid — the INPUT has to shrink, which is a caller
decision (transcript clipping, smaller batches). Clamping to 12 tokens would turn a loud 400
into a truncated unparsable answer, i.e. trade a visible failure for a silent one. That case
logs at ERROR and sends unchanged so the server's explicit message is what surfaces.
"""

from __future__ import annotations

import pytest

from podcast_scraper.providers.openai.openai_provider import (
    _context_clamp_token_budget,
    _learn_context_limit_from_error,
    _MIN_USEFUL_OUTPUT_TOKENS,
)

#: The production error, verbatim in shape.
_REAL_ERROR = (
    "error code: 400 - {'error': {'message': \"this model's maximum context length is 32768 "
    "tokens. however, you requested 2048 output tokens and your prompt contains at least 30721 "
    'input tokens"}}'
)


class TestLearningTheLimit:
    def test_parses_the_real_production_error(self):
        assert _learn_context_limit_from_error(_REAL_ERROR) == 32768

    @pytest.mark.parametrize(
        "msg",
        [
            "This model's maximum context length is 8192 tokens",
            "MAXIMUM CONTEXT LENGTH IS 128000 TOKENS",
            "...maximum context length is    4096   tokens...",
        ],
    )
    def test_case_and_spacing_tolerant(self, msg):
        assert _learn_context_limit_from_error(msg) is not None

    @pytest.mark.parametrize(
        "msg",
        [
            "",
            "rate limit exceeded",
            "temperature does not support 0.3",
            "maximum context length is unknown tokens",
        ],
    )
    def test_unrelated_errors_learn_nothing(self, msg):
        """A wrong 'limit' would clamp every later call for that model to nonsense."""
        assert _learn_context_limit_from_error(msg) is None


class TestClamping:
    def test_the_production_case_now_fits(self):
        kw = {"max_tokens": 2048, "messages": [{"content": "x"}]}
        assert _context_clamp_token_budget(kw, 32768, prompt_hint=_REAL_ERROR) is True
        assert kw["max_tokens"] == 32768 - 30721
        assert 30721 + kw["max_tokens"] <= 32768

    def test_a_request_that_already_fits_is_untouched(self):
        kw = {"max_tokens": 2048, "messages": [{"content": "short"}]}
        assert _context_clamp_token_budget(kw, 32768) is False
        assert kw["max_tokens"] == 2048

    def test_it_honours_the_o1_style_kwarg_name(self):
        """o1/o3/gpt-5 renamed max_tokens; clamping the wrong key would do nothing at all."""
        kw = {"max_completion_tokens": 2048, "messages": [{"content": "x"}]}
        assert _context_clamp_token_budget(kw, 32768, prompt_hint=_REAL_ERROR) is True
        assert kw["max_completion_tokens"] == 32768 - 30721

    def test_prompt_size_is_estimated_when_the_server_did_not_say(self):
        """The up-front path has no error text — it must estimate from the messages."""
        kw = {"max_tokens": 4000, "messages": [{"content": "x" * 35000}]}  # ~10000 tokens
        assert _context_clamp_token_budget(kw, 12000) is True
        assert kw["max_tokens"] < 4000

    def test_the_estimate_is_pessimistic_not_optimistic(self):
        """Under-estimating sends a request that 400s; over-estimating costs a little headroom.

        Only one of those two loses an episode's stage, so the estimate must err toward
        assuming MORE prompt tokens than a generous chars/token ratio would suggest.
        """
        chars = 35000
        kw = {"max_tokens": 8000, "messages": [{"content": "x" * chars}]}
        _context_clamp_token_budget(kw, 20000)
        assumed_prompt = 20000 - kw["max_tokens"]
        assert assumed_prompt >= chars / 4, "estimate is looser than 4 chars/token — optimistic"


class TestItRefusesToPaperOverAnUnfixableRequest:
    def test_no_clamp_when_the_prompt_alone_swamps_the_context(self, caplog):
        kw = {"max_tokens": 2048, "messages": [{"content": "x" * 200000}]}
        caplog.set_level("ERROR")

        assert _context_clamp_token_budget(kw, 32768) is False
        assert kw["max_tokens"] == 2048, "must send unchanged so the server's 400 surfaces"
        msg = " ".join(r.getMessage() for r in caplog.records)
        assert "INPUT must shrink" in msg

    @staticmethod
    def _hint(prompt_tokens: int, limit: int) -> str:
        """A server error naming an EXACT prompt size.

        Driving the boundary through the hint rather than the char estimate removes the
        estimator's rounding from the assertion — a first version of this test computed chars
        from the target token count and landed one token the other side of the boundary,
        testing the arithmetic of the fixture instead of the behaviour of the code.
        """
        return (
            f"maximum context length is {limit} tokens. however, you requested 4096 output "
            f"tokens and your prompt contains at least {prompt_tokens} input tokens"
        )

    def test_it_will_not_clamp_below_a_usable_reply(self):
        """Clamping to a handful of tokens converts a loud 400 into a silent truncation."""
        limit = 10_000
        prompt_tokens = limit - (_MIN_USEFUL_OUTPUT_TOKENS - 1)  # one token too little room
        kw = {"max_tokens": 4096, "messages": [{"content": "x"}]}
        assert (
            _context_clamp_token_budget(kw, limit, prompt_hint=self._hint(prompt_tokens, limit))
            is False
        )
        assert kw["max_tokens"] == 4096

    def test_it_does_clamp_when_there_is_just_enough_room(self):
        limit = 10_000
        prompt_tokens = limit - _MIN_USEFUL_OUTPUT_TOKENS  # exactly enough
        kw = {"max_tokens": 4096, "messages": [{"content": "x"}]}
        assert (
            _context_clamp_token_budget(kw, limit, prompt_hint=self._hint(prompt_tokens, limit))
            is True
        )
        assert kw["max_tokens"] == _MIN_USEFUL_OUTPUT_TOKENS


class TestDegenerateInputs:
    @pytest.mark.parametrize("limit", [None, 0, -1])
    def test_no_known_limit_means_no_clamp(self, limit):
        """Unknown context must never silently shrink a reply."""
        kw = {"max_tokens": 2048, "messages": [{"content": "x" * 999999}]}
        assert _context_clamp_token_budget(kw, limit) is False
        assert kw["max_tokens"] == 2048

    @pytest.mark.parametrize("budget", [None, 0, -5, "2048"])
    def test_a_missing_or_odd_budget_is_left_alone(self, budget):
        kw = {"max_tokens": budget, "messages": [{"content": "x" * 999999}]}
        assert _context_clamp_token_budget(kw, 32768) is False

    def test_empty_messages_do_not_crash(self):
        kw = {"max_tokens": 2048, "messages": []}
        assert _context_clamp_token_budget(kw, 32768) is False

    def test_none_content_does_not_crash(self):
        kw = {"max_tokens": 2048, "messages": [{"content": None}, {}]}
        assert _context_clamp_token_budget(kw, 32768) is False
