"""Chat-completion-shape guardrail (any provider — self-hosted or cloud).

Covers Ollama, vLLM, OpenAI, Anthropic, Gemini, DeepSeek, Mistral — anything
whose response carries a ``content`` field plus an optional ``finish_reason``.
Callers extract ``content`` and ``finish_reason`` from their SDK's response
object, then pass the strings to :func:`check_chat_response` along with the
service label.

Used to be split into ``check_ollama_response`` + ``check_vllm_response`` in
the old ``tailnet_dgx/resilience.py`` module — unified here because the
failure modes are the same shape across all chat providers; only the SDK
extraction differs.
"""

from __future__ import annotations

from typing import Optional

from ._telemetry import raise_violation

REASON_CHAT_EMPTY = "empty_content"
REASON_CHAT_THINKING_PROSE = "thinking_prose_detected"
REASON_CHAT_BAD_JSON = "json_parse_failed"
REASON_CHAT_FINISH_LENGTH = "finish_reason_length"

# Thinking-prose markers seen in qwen3.5 thinking-budget failures and some
# cloud thinking-mode model responses. Match only against the response head
# (first 200 chars) so we don't false-fire on transcripts that legitimately
# quote a thinking-prose phrase later in the body.
_THINKING_MARKERS: tuple[str, ...] = (
    "<think>",
    "<think ",
    "Okay, so I need to",
    "Let me think",
)


def check_chat_response(
    content: Optional[str],
    *,
    service: str,
    finish_reason: Optional[str] = None,
    expect_json: bool = False,
) -> None:
    """Raise :class:`..exceptions.GuardrailViolation` if the response fails a
    structural sanity check; return silently otherwise.

    Args:
        content: the chat completion's ``message.content`` string (or
            equivalent in the provider's SDK shape). May be None.
        service: the service identifier — used as the Prometheus label and
            as ``GuardrailViolation.service``. Pick a fixed short string per
            provider: ``"ollama"`` / ``"openai"`` / ``"anthropic"`` /
            ``"gemini"`` / ``"deepseek"`` / ``"vllm"``. Don't put deployment
            details here (no ``"openai-on-dgx"`` — the service is OpenAI,
            where it runs is in the log body).
        finish_reason: optional. The completion's stop reason. When set to
            ``"length"`` indicates the response was truncated mid-output.
        expect_json: when True, parse ``content`` as JSON and treat parse
            failure as a violation. Set by callers that requested
            structured output.
    """
    # finish_reason check first — even non-empty content with finish='length'
    # is structurally truncated and unusable downstream.
    if finish_reason == "length":
        summary = f"finish_reason=length content_head={(content or '')[:80]!r}"
        raise_violation(service, REASON_CHAT_FINISH_LENGTH, summary)

    # Empty content
    if content is None or content == "":
        raise_violation(service, REASON_CHAT_EMPTY, "")

    # Thinking-prose markers in the response head
    head = (content or "")[:200]
    for marker in _THINKING_MARKERS:
        if marker in head:
            raise_violation(service, REASON_CHAT_THINKING_PROSE, head)

    # JSON-parse check (only when structured output was requested)
    if expect_json:
        import json as _json

        try:
            # ``content`` is non-empty by the early-return above; type-narrow
            # for mypy (Optional[str] -> str).
            assert content is not None
            _json.loads(content)
        except (ValueError, TypeError) as exc:
            summary = f"parse_error={exc!s} head={(content or '')[:80]!r}"
            raise_violation(service, REASON_CHAT_BAD_JSON, summary)


__all__ = [
    "REASON_CHAT_BAD_JSON",
    "REASON_CHAT_EMPTY",
    "REASON_CHAT_FINISH_LENGTH",
    "REASON_CHAT_THINKING_PROSE",
    "check_chat_response",
]
