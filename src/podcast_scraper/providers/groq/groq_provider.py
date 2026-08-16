"""GroqProvider — Groq's low-latency OpenAI-compatible chat API (ADR-147).

A SIBLING of :class:`OpenAIProvider` / :class:`VLLMProvider` / :class:`DeepSeekProvider` /
:class:`QwenProvider`, not a standalone re-implementation. It shares the OpenAI-compatible transport
via the common :class:`~podcast_scraper.providers.openai.openai_provider.OpenAICompatibleProvider`
base and only changes *identity* (the ``groq_*`` config namespace + telemetry), the *default
endpoint* (``api.groq.com/openai/v1``, so the "direct, no gateway" path needs zero config), *auth*
(a bearer is required — Groq is cloud-only, no unauthenticated local-serving mode — but construction
warns rather than raises, unlike DeepSeek), and the *reasoning token headroom* that Groq's hosted
reasoning models (gpt-oss, Qwen3, DeepSeek-R1-distill, compound) require.

LiteLLM-independent by design: this talks straight to Groq. Pointing ``groq_api_base`` at a LiteLLM
gateway (with a ``groq-*`` alias) routes the same class *through* the gateway instead — so Groq runs
both directly and via LiteLLM with only a config change, never a code change.

Groq hosts a wide catalog of THIRD-PARTY open models (Llama, gpt-oss, Qwen3, DeepSeek-R1-distill)
rather than a single vendor's own tuned family, so — unlike DeepSeek, which ships its own
``deepseek/*`` prompt templates — this sibling reuses the generic ``openai/*`` prompt_store
templates by default, matching QwenProvider (also a multi-model host with no dedicated templates).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from ... import config
from ...utils.cleaning_max_tokens import GROQ_CLEANING_MAX_TOKENS
from ...utils.provider_metadata import validate_api_key_format
from ..openai.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Groq models that emit a reasoning block BEFORE the answer: OpenAI's gpt-oss family, the Qwen3
# family, DeepSeek-R1-distill variants, generic "reasoning"-tagged ids, and the agentic "compound" /
# "compound-mini" systems (which run internal tool-use/reasoning steps). Substring match so a dated
# snapshot still hits.
_REASONING_MODEL_MARKERS = ("gpt-oss", "qwen3", "r1", "reasoning", "compound")

# Room for the reasoning that must precede the answer WHEN reasoning is left ON. A tight
# ``max_tokens`` is otherwise consumed entirely by reasoning and ``content`` comes back EMPTY with
# finish_reason="length" — silently disconnecting the whole grounding stack (mirrors the DeepSeek
# B7 failure mode this guards against). Reasoning CAN usually be disabled per-model on Groq (e.g.
# ``reasoning_effort: none`` for gpt-oss); set one via ``groq_extra_body`` and no headroom is
# needed.
# This headroom is the FALLBACK for a run that leaves reasoning on. Groq has no single vendor-wide
# chat max_tokens cap (it varies per model), so this reuses the same conservative ceiling as
# transcript cleaning (``GROQ_CLEANING_MAX_TOKENS``) — exactly the value DeepSeek reuses for both.
_REASONING_TOKEN_HEADROOM = 2048


def _model_reasons(model: Optional[str]) -> bool:
    """Does this Groq-hosted model emit a reasoning block before its answer?"""
    name = (model or "").lower()
    return any(marker in name for marker in _REASONING_MODEL_MARKERS)


def _extra_body_disables_thinking(extra_body: Any) -> bool:
    """True if ``extra_body`` carries a directive that turns reasoning OFF.

    Recognises every shape a profile might reasonably use — ``reasoning_effort: none`` (the shape
    Groq's own gpt-oss/qwen3 docs use), ``thinking: {type: disabled}`` (DeepSeek-shape, relevant for
    Groq's R1-distill models), and the OpenRouter/vLLM shapes (``reasoning: {enabled: false}``,
    ``enable_thinking: false``, ``chat_template_kwargs: {enable_thinking: false}``) so a
    gateway-routed profile is also read as "intended off". The point is to detect INTENT, not to
    validate the endpoint.
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


# An endpoint served without auth still needs the OpenAI SDK client to carry *some* bearer so it
# does not silently fall back to reading an unrelated ``OPENAI_API_KEY`` env var (the OpenAI SDK's
# default behaviour when ``api_key=None``). Groq is cloud-only — unlike a local vLLM/Qwen slot, this
# dummy never actually authenticates; it only exists to keep construction offline-safe (ADR-147 /
# Qwen precedent). Never a real secret.
_GROQ_DUMMY_BEARER = "EMPTY"


class GroqProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider talking direct to Groq's API (or a LiteLLM gateway alias).

    Overrides only what differs from OpenAI-native: the ``groq_*`` config namespace, the telemetry
    identity, the default ``api.groq.com`` endpoint, the 8192-token cleaning cap, a key-required
    (warn-not-raise) auth with no ``sk-`` format assumption, and reasoning-model token headroom.
    Everything else — transport, summary/speaker/GI/KG/grounding chat calls, cost recording — is
    inherited.
    """

    _CONFIG_NS: str = "groq"
    _TELEMETRY_PROVIDER: str = "groq"
    _PROVIDER_LABEL: str = "Groq"
    # Direct endpoint so "no gateway" needs zero config; groq_api_base overrides it.
    _DEFAULT_API_BASE: str = "https://api.groq.com/openai/v1"
    # Groq chat comfortably emits an 8192-token output for cleaning a full transcript.
    _CLEANING_MAX_TOKENS_CAP: int = GROQ_CLEANING_MAX_TOKENS

    def __init__(self, cfg: config.Config):
        super().__init__(cfg)
        # Groq's open/hosted models don't reject a non-default temperature the way some OpenAI
        # models do, so start with an empty "temperature-fixed" set (the base seeds it from the
        # OpenAI-only _TEMPERATURE_FIXED_MODELS constant).
        self._temp_fixed_at_default = set()
        # Cleaning defaults to the summary model unless a profile pins groq_cleaning_model — the
        # base's default is the OpenAI cleaning model, which is wrong for Groq.
        self.cleaning_model = getattr(cfg, "groq_cleaning_model", None) or self.summary_model
        # gpt-oss/qwen3/r1-distill/compound emit reasoning before the answer; a tight budget
        # starves ``content``.
        self._is_reasoning_model = _model_reasons(self.summary_model)
        # B7-style guard (mirrors DeepSeek): a reasoning model whose thinking was NOT disabled
        # emits reasoning content into every JSON-extraction stage (summary/GI/KG/grounding). On a
        # tight summary budget that consumes the whole allowance and content returns EMPTY
        # (finish_reason=length) -> the episode fails. Warn LOUDLY at construction so the misconfig
        # is caught before a paid run, not after N failed episodes. Not a hard error: the
        # reasoning-token headroom is the intentional (if inferior) fallback for a run that
        # deliberately leaves thinking on.
        if self._thinking_left_on():
            logger.warning(
                "Groq reasoning model %r has thinking LEFT ON (no reasoning-off directive in "
                "groq_extra_body): JSON-extraction stages may truncate to empty content "
                "(finish_reason=length) and fail. Set groq_extra_body: {reasoning_effort: none} "
                "(or the model's own disable directive) to turn it off.",
                self.summary_model,
            )

    def _thinking_left_on(self) -> bool:
        """A reasoning model whose thinking is NOT disabled via groq_extra_body (B7-style guard)."""
        if not self._is_reasoning_model:
            return False
        return not _extra_body_disables_thinking(getattr(self.cfg, "groq_extra_body", None))

    def _authenticate(self, cfg: "config.Config") -> None:
        """A Groq API key is effectively required (Groq is cloud-only — no unauthenticated local
        serving mode like a vLLM/Qwen slot) — but unlike DeepSeek, construction WARNS instead of
        raising, so an offline import (or a test that never calls the API) still works without a
        network key. It is an OpenAI-compatible key but Groq issues ``gsk_...`` ids, not ``sk-``, so
        we do not assume a specific prefix (validated leniently, like DeepSeek/Qwen open-model
        keys)."""
        api_key = self._resolve_api_key(cfg)
        if api_key == _GROQ_DUMMY_BEARER:
            logger.warning(
                "Groq API key not configured (set groq_api_key or the env named by "
                "groq_api_key_env, default GROQ_API_KEY); Groq calls will fail authentication "
                "until one is set."
            )
            return
        is_valid, _ = validate_api_key_format(api_key, "Groq", expected_prefixes=None)
        if not is_valid:
            # Do not log validation detail: CodeQL taints any message from this API-key path.
            logger.warning(
                "Groq API key validation failed (missing or too short); "
                "credentials are never logged."
            )

    def _resolve_api_key(self, cfg: "config.Config") -> Optional[str]:
        """Bearer for the client: ``groq_api_key``, else the env named by ``groq_api_key_env``
        (default ``GROQ_API_KEY``), else a dummy. The dummy exists ONLY to stop the OpenAI SDK from
        silently falling back to an unrelated ``OPENAI_API_KEY`` env var when ``api_key=None`` is
        passed — Groq itself always 401s on it, since (unlike a local vLLM/Qwen slot) there is no
        unauthenticated serving mode to fall back to."""
        explicit: Optional[str] = getattr(cfg, "groq_api_key", None)
        if explicit:
            return explicit
        env_name: str = getattr(cfg, "groq_api_key_env", None) or "GROQ_API_KEY"
        from_env = os.getenv(env_name)
        if from_env:
            return from_env
        return _GROQ_DUMMY_BEARER

    def _token_kwarg(self, n: int, model: Optional[str] = None) -> Dict[str, Any]:
        """Groq's hosted models use classic ``max_tokens`` (no o1/o3/gpt-5 rename). On a reasoning
        model, add headroom so reasoning content (emitted first) does not truncate the answer to
        empty; capped at the same conservative ceiling as cleaning. On a non-reasoning model this is
        a plain passthrough.
        """
        if self._is_reasoning_model:
            n = min(n + _REASONING_TOKEN_HEADROOM, GROQ_CLEANING_MAX_TOKENS)
        return {"max_tokens": n}
