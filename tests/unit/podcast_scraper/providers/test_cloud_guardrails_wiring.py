"""Confirm the cloud-LLM-provider guardrail wiring fires on bad responses (#1003).

Unit-level checks for the OpenAI / Anthropic / Gemini / DeepSeek providers:
each one's ``summarize`` (or equivalent) path should raise
``GuardrailViolation`` when the SDK returns an empty content / thinking-prose
/ finish_reason=length response. Doesn't exercise the full FallbackAware
provider — that's the integration layer's job; here we just confirm the
``check_chat_response`` call lands at the right site with the right
``service`` label.

The unit-level coverage of the helper itself (per-mode raises) lives in
``test_resilience_and_guardrails.py``. This file is the per-provider
wiring spec.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from podcast_scraper.providers import guardrails


def _openai_response(content: str, finish_reason: str = "stop") -> MagicMock:
    """Shape a fake OpenAI SDK chat-completion response (also matches DeepSeek)."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    response.choices[0].finish_reason = finish_reason
    return response


def _anthropic_response(text: str, stop_reason: str = "end_turn") -> MagicMock:
    response = MagicMock()
    block = MagicMock()
    block.text = text
    response.content = [block]
    response.stop_reason = stop_reason
    return response


def _gemini_response(text: str, finish_reason: str = "STOP") -> MagicMock:
    response = MagicMock()
    response.text = text
    candidate = MagicMock()
    candidate.finish_reason = MagicMock()
    candidate.finish_reason.name = finish_reason
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# Cross-provider wiring smoke-tests — direct check_chat_response invocations
# matching the call shape each provider's wiring uses. Confirms the helper
# accepts every shape and emits the right service label on violation.
# ---------------------------------------------------------------------------


class TestOpenAIWiring:
    def test_empty_summary_fires_openai_violation(self):
        response = _openai_response("", finish_reason="stop")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.choices[0].message.content,
                service="openai",
                finish_reason=response.choices[0].finish_reason,
            )
        assert exc_info.value.service == "openai"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_finish_length_fires_openai_violation(self):
        response = _openai_response("truncated mid", finish_reason="length")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.choices[0].message.content,
                service="openai",
                finish_reason=response.choices[0].finish_reason,
            )
        assert exc_info.value.reason == guardrails.REASON_CHAT_FINISH_LENGTH

    def test_thinking_prose_fires_openai_violation(self):
        response = _openai_response("<think>reasoning</think>", finish_reason="stop")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.choices[0].message.content,
                service="openai",
                finish_reason=response.choices[0].finish_reason,
            )
        assert exc_info.value.reason == guardrails.REASON_CHAT_THINKING_PROSE

    def test_normal_response_no_raise(self):
        response = _openai_response(
            "This is a perfectly reasonable summary of the episode.",
            finish_reason="stop",
        )
        guardrails.check_chat_response(
            response.choices[0].message.content,
            service="openai",
            finish_reason=response.choices[0].finish_reason,
        )  # no raise


class TestAnthropicWiring:
    def test_empty_text_fires_anthropic_violation(self):
        response = _anthropic_response("", stop_reason="end_turn")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.content[0].text,
                service="anthropic",
                finish_reason=getattr(response, "stop_reason", None),
            )
        assert exc_info.value.service == "anthropic"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_max_tokens_stop_reason_normalised_to_length(self):
        # Anthropic's "max_tokens" is normalised to "length" by the
        # provider's _anthropic_finish_reason helper so the guardrail
        # trips (closing the documented limitation from #1003).
        from podcast_scraper.providers.anthropic.anthropic_provider import (
            _anthropic_finish_reason,
        )

        response = _anthropic_response("truncated", stop_reason="max_tokens")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.content[0].text,
                service="anthropic",
                finish_reason=_anthropic_finish_reason(response),
            )
        assert exc_info.value.reason == guardrails.REASON_CHAT_FINISH_LENGTH

    def test_thinking_prose_fires_anthropic_violation(self):
        response = _anthropic_response("Let me think about this.")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.content[0].text,
                service="anthropic",
                finish_reason=getattr(response, "stop_reason", None),
            )
        assert exc_info.value.reason == guardrails.REASON_CHAT_THINKING_PROSE


class TestGeminiWiring:
    def test_empty_text_fires_gemini_violation(self):
        # The provider's _gemini_finish_reason normalizes the enum.name to lowercase
        from podcast_scraper.providers.gemini.gemini_provider import (
            _gemini_finish_reason,
        )

        response = _gemini_response("", finish_reason="STOP")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.text,
                service="gemini",
                finish_reason=_gemini_finish_reason(response),
            )
        assert exc_info.value.service == "gemini"
        assert exc_info.value.reason == guardrails.REASON_CHAT_EMPTY

    def test_finish_length_fires_gemini_violation(self):
        from podcast_scraper.providers.gemini.gemini_provider import (
            _gemini_finish_reason,
        )

        # Gemini's MAX_TOKENS now normalises to "length" in the helper
        # (closing the documented limitation from #1003).
        response = _gemini_response("truncated mid", finish_reason="MAX_TOKENS")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.text,
                service="gemini",
                finish_reason=_gemini_finish_reason(response),
            )
        assert exc_info.value.reason == guardrails.REASON_CHAT_FINISH_LENGTH

        # Direct "length" also fires.
        response2 = _gemini_response("truncated", finish_reason="length")
        with pytest.raises(guardrails.GuardrailViolation):
            guardrails.check_chat_response(
                response2.text,
                service="gemini",
                finish_reason=_gemini_finish_reason(response2),
            )

    def test_gemini_finish_reason_helper_handles_missing_candidates(self):
        from podcast_scraper.providers.gemini.gemini_provider import (
            _gemini_finish_reason,
        )

        response = MagicMock()
        response.candidates = []
        assert _gemini_finish_reason(response) is None


class TestDeepSeekWiring:
    """DeepSeek uses the OpenAI-compatible SDK shape."""

    def test_empty_summary_fires_deepseek_violation(self):
        response = _openai_response("", finish_reason="stop")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.choices[0].message.content,
                service="deepseek",
                finish_reason=response.choices[0].finish_reason,
            )
        assert exc_info.value.service == "deepseek"

    def test_finish_length_fires_deepseek_violation(self):
        response = _openai_response("cut off", finish_reason="length")
        with pytest.raises(guardrails.GuardrailViolation) as exc_info:
            guardrails.check_chat_response(
                response.choices[0].message.content,
                service="deepseek",
                finish_reason=response.choices[0].finish_reason,
            )
        assert exc_info.value.reason == guardrails.REASON_CHAT_FINISH_LENGTH


class TestCostAttributionFlag:
    """The llm_cost log event now carries a triggered_guardrail boolean."""

    def test_emit_llm_cost_event_includes_guardrail_flag_default_false(self, caplog):
        from podcast_scraper import config as cfg_module
        from podcast_scraper.workflow.cost_monitoring import emit_llm_cost_event

        cfg = cfg_module.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "openai_api_key": "sk-test",
            }
        )
        with caplog.at_level("INFO"):
            emit_llm_cost_event(
                cfg,
                provider="openai",
                stage="summarize",
                model="gpt-4o-mini",
                estimated_cost_usd=0.0123,
            )
        import json as _json

        # Find the llm_cost line
        events = [
            _json.loads(r.message)
            for r in caplog.records
            if r.message.startswith("{") and '"event_type"' in r.message
        ]
        assert len(events) == 1
        assert events[0]["triggered_guardrail"] is False

    def test_emit_llm_cost_event_with_guardrail_flag_true(self, caplog):
        from podcast_scraper import config as cfg_module
        from podcast_scraper.workflow.cost_monitoring import emit_llm_cost_event

        cfg = cfg_module.Config.model_validate(
            {
                "rss_url": "https://example.com/feed.xml",
                "openai_api_key": "sk-test",
            }
        )
        with caplog.at_level("INFO"):
            emit_llm_cost_event(
                cfg,
                provider="gemini",
                stage="summarize",
                model="gemini-2.5-flash",
                estimated_cost_usd=0.0007,
                triggered_guardrail=True,
            )
        import json as _json

        events = [
            _json.loads(r.message)
            for r in caplog.records
            if r.message.startswith("{") and '"event_type"' in r.message
        ]
        assert len(events) == 1
        assert events[0]["triggered_guardrail"] is True
