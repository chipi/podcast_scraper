"""A truncated GI call must bill ONCE, not twice (#1891, prod 2026-08-31).

``generate_insights`` emitted a cost event from the guardrail branch and then fell through
to the normal emit, so exactly the truncated calls were double-counted. On prod that turned
35,601 real output tokens into 65,601 reported — a 84% overstatement, and it landed on the
one stage that was already the most expensive. Cost dashboards, per-stage share, and the
analysis that set the token budget were all reading inflated numbers.

The bias direction is what makes it worse than a random error: only FAILING calls were
double-billed, so the data most likely to trigger an investigation was the most distorted.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from podcast_scraper import Config


def _cfg() -> Config:
    return Config.model_validate(
        {
            "rss_url": "https://example.com/f.rss",
            "summary_provider": "vllm",
            "vllm_api_base": "http://dgx-llm-1:8003/v1",
        }
    )


def _response(content: str, *, finish_reason: str, in_tok: int = 100, out_tok: int = 9000):
    resp = Mock()
    choice = Mock()
    choice.message = Mock(content=content)
    choice.finish_reason = finish_reason
    resp.choices = [choice]
    resp.usage = Mock(prompt_tokens=in_tok, completion_tokens=out_tok)
    resp.model = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"
    return resp


@pytest.fixture
def emitted():
    """Capture every llm_cost event generate_insights emits."""
    calls: list[dict] = []

    def _capture(cfg, **kwargs):
        calls.append(kwargs)

    with patch(
        "podcast_scraper.workflow.cost_monitoring.emit_llm_cost_event", side_effect=_capture
    ):
        yield calls


def _provider(response):
    from podcast_scraper.providers.vllm.vllm_provider import VLLMProvider

    p = VLLMProvider(_cfg())
    p._summarization_initialized = True
    p.insight_model = "NVFP4/Qwen3-30B-A3B-Instruct-2507-FP4"
    p._chat_create = Mock(return_value=response)
    return p


def test_a_clean_call_emits_exactly_one_cost_event(emitted):
    p = _provider(_response("Insight one\nInsight two", finish_reason="stop", out_tok=40))
    p.generate_insights("t" * 500, max_insights=5)
    assert len(emitted) == 1, f"expected 1 cost event, got {len(emitted)}"
    assert emitted[0]["triggered_guardrail"] is False


def test_a_truncated_call_still_emits_exactly_one(emitted):
    """THE REGRESSION: this used to emit twice, inflating the most expensive stage."""
    truncated = "\n".join(f"Insight {i}" for i in range(400))
    p = _provider(_response(truncated, finish_reason="length", out_tok=9000))
    p.generate_insights("t" * 500, max_insights=60)

    assert len(emitted) == 1, (
        f"truncated GI call emitted {len(emitted)} cost events; "
        "one API call must bill once or spend data is inflated by exactly the failing calls"
    )


def test_the_single_event_reports_the_guardrail_trip(emitted):
    """Billing once must not cost us the signal that it was truncated."""
    truncated = "\n".join(f"Insight {i}" for i in range(400))
    p = _provider(_response(truncated, finish_reason="length", out_tok=9000))
    p.generate_insights("t" * 500, max_insights=60)

    assert emitted, "no cost event emitted at all"
    assert emitted[0]["triggered_guardrail"] is True
    assert emitted[0]["completion_tokens"] == 9000


def test_tokens_are_not_double_counted(emitted):
    """The concrete prod symptom: summed completion_tokens must equal the call's own."""
    truncated = "\n".join(f"Insight {i}" for i in range(400))
    p = _provider(_response(truncated, finish_reason="length", out_tok=9000))
    p.generate_insights("t" * 500, max_insights=60)

    assert sum(e["completion_tokens"] for e in emitted) == 9000
