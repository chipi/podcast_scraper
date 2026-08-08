"""QwenProvider — Qwen3-family serving over any OpenAI-compatible endpoint (ADR-147).

A SIBLING of :class:`OpenAIProvider` / :class:`VLLMProvider` / :class:`LiteLLMProvider` /
:class:`DeepSeekProvider`, not a subclass. Qwen3 is an open family we serve either from a cloud host
(DeepInfra/Together/Fireworks) or the DGX vLLM slot; both speak the OpenAI-compatible wire protocol,
so this class shares the transport via the common
:class:`~podcast_scraper.providers.openai.openai_provider.OpenAICompatibleProvider` base and only
changes *identity* (the ``qwen_*`` config namespace + ``qwen`` telemetry), *auth* (a bearer is
optional for a local vLLM, required by a cloud host), and the open-model *heuristics* that do not
apply to OpenAI-native models.

Why its own provider and not just vllm/litellm pointed at a Qwen endpoint: cost/telemetry
attribution. "we cannot attribute vllm and litellm to openai" (ADR-147) applies here too — running
Qwen through the vllm provider bills it to ``vllm``, through litellm to ``litellm``; neither says
``qwen``. This sibling gives Qwen its own cost namespace and a cloud-direct path to a single
fixed-price host, sidestepping OpenRouter's per-host price variance.

There is NO ``_DEFAULT_API_BASE``: Qwen has no single vendor endpoint we commit to (Alibaba
DashScope is explicitly out of scope), so the profile always names ``qwen_api_base`` — exactly like
vllm. Like the model fields elsewhere in the vLLM/DeepSeek siblings, the model fields name the REAL
model id on the wire (no ``--served-model-name`` alias) and are verified fail-closed against the
served model at init (ADR-143/144).

The Qwen3 "thinking" toggle is host-specific and lives in ``qwen_extra_body`` (a self-hosted vLLM
wants ``{chat_template_kwargs: {enable_thinking: false}}``; DashScope ``{enable_thinking: false}``;
OpenRouter ``{reasoning: {enabled: false}}``) — the base merges it into every request. Thinking MUST
be off for JSON extraction stages.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional, Set

from ... import config
from ...utils.cleaning_max_tokens import QWEN_CLEANING_MAX_TOKENS
from ..openai.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# An endpoint served without auth still needs the OpenAI SDK client to carry *some* bearer; it is
# ignored server-side. Never a real secret.
_QWEN_DUMMY_BEARER = "EMPTY"


class QwenServedModelMismatch(RuntimeError):
    """The Qwen endpoint serves a different model than the profile pins (ADR-147 B3).

    Raised fail-closed so a wrong model behind the endpoint stops the run instead of silently
    producing a corpus attributed to the wrong model.
    """


def _served_matches(expected: str, served: Set[str]) -> bool:
    """True if the configured model id matches one the endpoint advertises. Casefold + dated/version
    suffix tolerance (startswith either way); the org prefix is NOT stripped, so ``Qwen/...`` never
    matches ``someoneelse/...``."""
    e = expected.casefold()
    for s in served:
        sc = s.casefold()
        if sc == e or sc.startswith(e) or e.startswith(sc):
            return True
    return False


class QwenProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for a Qwen3 endpoint (cloud host or DGX-local vLLM).

    Overrides only what differs from OpenAI-native: the ``qwen_*`` config namespace, ``qwen``
    telemetry identity, optional-bearer auth, the 8192-token cleaning cap, and the open-model
    token/temperature heuristics. Everything else (summary/speaker/GI/KG/grounding chat calls, cost
    recording) is inherited unchanged from the shared transport base.
    """

    _CONFIG_NS: str = "qwen"
    _TELEMETRY_PROVIDER: str = "qwen"
    _PROVIDER_LABEL: str = "Qwen"
    # No vendor default endpoint (DashScope is out of scope): the profile always sets qwen_api_base.
    _CLEANING_MAX_TOKENS_CAP: int = QWEN_CLEANING_MAX_TOKENS

    def __init__(self, cfg: config.Config):
        super().__init__(cfg)
        # Open models (Qwen3) do not reject a non-default temperature the way some OpenAI models do,
        # so start with an empty "temperature-fixed" set (the base seeds it from the OpenAI-only
        # _TEMPERATURE_FIXED_MODELS constant).
        self._temp_fixed_at_default = set()
        # Cleaning defaults to the summary model unless a profile pins qwen_cleaning_model — one
        # served model handles the whole cascade unless explicitly split.
        if not self.cleaning_model:
            self.cleaning_model = self.summary_model

    def _authenticate(self, cfg: "config.Config") -> None:
        """A bearer is optional — a local vLLM Qwen needs none; a cloud host supplies one via
        ``qwen_api_key`` / ``qwen_api_key_env``. No required-key / ``sk-`` validation (ADR-147)."""
        return None

    def _resolve_api_key(self, cfg: "config.Config") -> Optional[str]:
        """Bearer for the client: ``qwen_api_key``, else the env named by ``qwen_api_key_env``
        (default ``QWEN_API_KEY``), else a dummy (a local vLLM ignores it when served without
        auth)."""
        explicit: Optional[str] = getattr(cfg, "qwen_api_key", None)
        if explicit:
            return explicit
        env_name: str = getattr(cfg, "qwen_api_key_env", None) or "QWEN_API_KEY"
        from_env = os.getenv(env_name)
        if from_env:
            return from_env
        return _QWEN_DUMMY_BEARER

    def _token_kwarg(self, n: int, model: Optional[str] = None) -> Dict[str, Any]:
        """Qwen open models use the classic ``max_tokens``; there is no o1/o3/gpt-5
        ``max_completion_tokens`` rename to honour."""
        return {"max_tokens": n}

    def initialize(self) -> None:
        """Fail-closed served-model check before first use (ADR-147 B3), then the normal init."""
        if getattr(self.cfg, "qwen_verify_served_model", True):
            self._verify_served_model()
        super().initialize()

    def _verify_served_model(self) -> None:
        """Assert the endpoint actually serves the model this profile pins (real Qwen id).

        A wrong model behind the endpoint must fail the run, not produce a corpus attributed to the
        wrong model (ADR-143/144). An UNREACHABLE endpoint only warns — the real inference
        call surfaces a connection error anyway, and hard-failing here would make an offline import
        of the provider impossible. A REACHABLE endpoint serving a different model raises.
        """
        base = getattr(self.cfg, "qwen_api_base", None)
        expected = self.summary_model
        if not base or not expected:
            return
        url = f"{base.rstrip('/')}/models"
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {self._resolve_api_key(self.cfg)}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — fixed profile URL
                data = json.loads(resp.read().decode("utf-8")).get("data", [])
        except Exception as exc:  # noqa: BLE001 — unreachable != mismatch; surface at call time
            logger.warning(
                "qwen: could not verify served model at %s (%s); the inference call will surface "
                "any real connectivity problem",
                url,
                type(exc).__name__,
            )
            return
        served: Set[str] = set()
        for entry in data if isinstance(data, list) else []:
            for key in ("id", "root"):
                val = entry.get(key) if isinstance(entry, dict) else None
                if isinstance(val, str) and val:
                    served.add(val)
        if not _served_matches(expected, served):
            raise QwenServedModelMismatch(
                f"Qwen at {base} serves {sorted(served) or '<none>'} but this profile pins "
                f"{expected!r}. Load/point at the right Qwen model (or fix qwen_summary_model). "
                f"Refusing to run to avoid corpus corruption (ADR-147 B3)."
            )
        logger.info("qwen: served-model check OK (%s advertised at %s)", expected, base)
