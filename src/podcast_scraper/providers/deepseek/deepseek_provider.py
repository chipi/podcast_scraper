"""DeepSeekProvider — direct DeepSeek serving over its OpenAI-compatible API (ADR-147).

A SIBLING of :class:`OpenAIProvider` / :class:`VLLMProvider` / :class:`LiteLLMProvider`, not a
standalone re-implementation. It shares the OpenAI-compatible transport via the common
:class:`~podcast_scraper.providers.openai.openai_provider.OpenAICompatibleProvider` base and only
changes *identity* (the ``deepseek_*`` config namespace + telemetry), the *default endpoint*
(``api.deepseek.com``, so the "direct, no gateway" path needs zero config), and the *reasoning
token headroom* that DeepSeek's v4/r1/reasoner models require.

LiteLLM-independent by design: this talks straight to DeepSeek. Pointing ``deepseek_api_base`` at a
LiteLLM gateway (with a ``deepseek-*`` alias) routes the same class *through* the gateway instead —
so DeepSeek runs both directly and via LiteLLM with only a config change, never a code change.

Superseded the former ~2,100-line standalone class (pre-ADR-147); cost/pricing attribution,
transport, cleaning, and stage methods now come from the shared base, correctly namespaced to
``deepseek`` via ``_TELEMETRY_PROVIDER``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ... import config
from ...utils.cleaning_max_tokens import DEEPSEEK_CLEANING_MAX_TOKENS
from ...utils.provider_metadata import validate_api_key_format
from ..openai.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# DeepSeek models that emit a reasoning block BEFORE the answer: deepseek-v4-* (flash/pro), the r1
# line, and deepseek-reasoner all spend the token budget on ``reasoning_content`` first.
# does not. Substring match so a dated snapshot ("deepseek-v4-flash-2026...") still hits.
_REASONING_MODEL_MARKERS = ("v4", "-r1", "reasoner", "reasoning")

# Room for the reasoning that must precede the answer WHEN reasoning is left ON. A tight
# ``max_tokens`` (score_entailment used 10) is otherwise consumed entirely by reasoning and
# ``content`` comes back EMPTY with finish_reason="length" — silently disconnecting the whole
# grounding stack (0 quotes, every insight unsupported). Reasoning CAN be disabled on the native
# api.deepseek.com — verified live: ``reasoning_effort:none`` AND ``thinking:{type:disabled}`` both
# zero ``reasoning_content`` (documented form: ``thinking:{type:disabled}``; ``enable_thinking`` /
# ``reasoning:{enabled:false}`` are ignored — those are OpenRouter/vLLM shapes). Set one via
# ``deepseek_extra_body`` and no headroom is needed. This headroom is the FALLBACK for a run that
# leaves reasoning on. DeepSeek caps chat max_tokens at 8192.
_REASONING_TOKEN_HEADROOM = 2048
_DEEPSEEK_MAX_TOKENS = 8192


def _model_reasons(model: Optional[str]) -> bool:
    """Does this DeepSeek model emit a reasoning block before its answer?"""
    name = (model or "").lower()
    return any(marker in name for marker in _REASONING_MODEL_MARKERS)


def _extra_body_disables_thinking(extra_body: Any) -> bool:
    """True if ``extra_body`` carries a directive that turns DeepSeek's thinking OFF.

    Recognises every shape a profile might reasonably use — the two that actually work on
    api.deepseek.com (``reasoning_effort: none``, ``thinking: {type: disabled}``) AND the
    OpenRouter/vLLM shapes (``reasoning: {enabled: false}``, ``enable_thinking: false``,
    ``chat_template_kwargs: {enable_thinking: false}``) so a gateway-routed profile is also read as
    "intended off". The point is to detect INTENT, not to validate the endpoint.
    """
    if not isinstance(extra_body, dict):
        return False
    if str(extra_body.get("reasoning_effort", "")).strip().lower() == "none":
        return True
    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and str(thinking.get("type", "")).lower() == "disabled":
        return True
    reasoning = extra_body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("enabled") is False:
        return True
    if extra_body.get("enable_thinking") is False:
        return True
    ctk = extra_body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and ctk.get("enable_thinking") is False:
        return True
    return False


class DeepSeekProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider talking direct to DeepSeek's API (or a LiteLLM gateway alias).

    Overrides only what differs from OpenAI-native: the ``deepseek_*`` config namespace, the
    telemetry identity, the default ``api.deepseek.com`` endpoint, the 8192-token cleaning cap, a
    key-required auth (no ``sk-`` format assumption), and reasoning-model token headroom. Everything
    else — transport, summary/speaker/GI/KG/grounding chat calls, cost recording — is inherited.
    """

    _CONFIG_NS: str = "deepseek"
    _TELEMETRY_PROVIDER: str = "deepseek"
    _PROVIDER_LABEL: str = "DeepSeek"
    # Direct endpoint so "no gateway" needs zero config; deepseek_api_base overrides it.
    _DEFAULT_API_BASE: str = "https://api.deepseek.com"
    # DeepSeek chat supports an 8192-token output — double the OpenAI-native 4096 cap. Cleaning a
    # full transcript needs the headroom or it truncates (finish_reason=length) and the guardrail
    # discards the cleaned text.
    _CLEANING_MAX_TOKENS_CAP: int = DEEPSEEK_CLEANING_MAX_TOKENS

    def __init__(self, cfg: config.Config):
        super().__init__(cfg)
        # DeepSeek is an open model: none of the OpenAI-native temperature-rejection heuristics
        # apply, so start with an empty "temperature-fixed" set (the base seeds it from the
        # OpenAI-only _TEMPERATURE_FIXED_MODELS constant).
        self._temp_fixed_at_default = set()
        # Cleaning defaults to the summary model unless a profile pins deepseek_cleaning_model — the
        # base's default is the OpenAI cleaning model, which is wrong for DeepSeek.
        self.cleaning_model = getattr(cfg, "deepseek_cleaning_model", None) or self.summary_model
        # v4/r1/reasoner emit reasoning before the answer; a tight budget starves ``content``.
        self._is_reasoning_model = _model_reasons(self.summary_model)
        # B7 guard: a reasoning model whose thinking was NOT disabled emits reasoning_content into
        # every JSON-extraction stage (summary/GI/KG/grounding). On the tight summary budget that
        # consumes the whole allowance and content returns EMPTY (finish_reason=length) -> the
        # episode fails (exactly the 8/9-episode failure this session). Warn LOUDLY at construction
        # so the misconfig is caught before a paid run, not after N failed episodes. Not a hard
        # error: the reasoning-token headroom is the intentional (if inferior) fallback for a run
        # that deliberately leaves thinking on.
        if self._thinking_left_on():
            logger.warning(
                "DeepSeek reasoning model %r has thinking LEFT ON (no reasoning-off directive in "
                "deepseek_extra_body): JSON-extraction stages may truncate to empty content "
                "(finish_reason=length) and fail. Set deepseek_extra_body: "
                "{thinking: {type: disabled}} (or reasoning_effort: none) to disable it.",
                self.summary_model,
            )

    def _thinking_left_on(self) -> bool:
        """A reasoning model whose thinking is NOT disabled via deepseek_extra_body (B7 guard)."""
        if not self._is_reasoning_model:
            return False
        return not _extra_body_disables_thinking(getattr(self.cfg, "deepseek_extra_body", None))

    def _authenticate(self, cfg: "config.Config") -> None:
        """A DeepSeek API key is REQUIRED (unlike a local vLLM bearer). It is an OpenAI-compatible
        key but we do not assume a specific prefix, so validate leniently and warn (never log the
        key). Raise when the key is missing so a misconfigured run fails fast."""
        api_key = getattr(cfg, "deepseek_api_key", None)
        if not api_key:
            raise ValueError(
                "DeepSeek API key required for DeepSeek provider. "
                "Set DEEPSEEK_API_KEY environment variable or deepseek_api_key in config."
            )
        is_valid, _ = validate_api_key_format(api_key, "DeepSeek", expected_prefixes=None)
        if not is_valid:
            # Do not log validation detail: CodeQL taints any message from this API-key path.
            logger.warning(
                "DeepSeek API key validation failed (missing or too short); "
                "credentials are never logged."
            )

    def _token_kwarg(self, n: int, model: Optional[str] = None) -> Dict[str, Any]:
        """DeepSeek open models use classic ``max_tokens`` (no o1/o3/gpt-5 rename). On a
        model, add headroom so ``reasoning_content`` (emitted first) does not truncate the answer to
        empty; capped at DeepSeek's 8192 chat limit. On deepseek-chat this is a plain passthrough.
        """
        if self._is_reasoning_model:
            n = min(n + _REASONING_TOKEN_HEADROOM, _DEEPSEEK_MAX_TOKENS)
        return {"max_tokens": n}
