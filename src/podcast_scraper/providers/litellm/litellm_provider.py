"""LiteLLMProvider — LLM calls via the homelab LiteLLM GATEWAY (#1356).

A SIBLING of :class:`OpenAIProvider` / :class:`VLLMProvider` over the shared
:class:`~podcast_scraper.providers.openai.openai_provider.OpenAICompatibleProvider` transport. The
gateway (``homelab:4001``) proxies to OpenRouter / direct vendors behind swappable ALIASES, so this
provider's model fields name aliases (``homelab-qwen``, ``homelab-flash``, …), not vendor ids — the
gateway decides the route. Named ``litellm`` (the stable component we run), not ``openrouter``
(today's swappable backend), so telemetry/cost logs stay honest when a route moves.

Everything (clean / quotes / entailment / speaker-detect / labeling / summary / GI / KG) is
inherited unchanged from the base — this class only changes *identity* (``litellm_*`` config
namespace + telemetry), *auth* (a gateway virtual key, not an ``sk-`` OpenAI key), and the
served-model check (assert the gateway advertises the pinned alias).

Reasoning models (qwen/glm/kimi) return only ``reasoning_content`` at normal token budgets; a
profile sets ``litellm_extra_body={"reasoning": {"enabled": false}}`` and the base injects it into
every ``chat.completions`` call — mirroring the vLLM ``enable_thinking=false`` precedent.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional, Set

from ... import config
from ..openai.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# The gateway 401s without a valid virtual key. If none is configured we still let the client
# construct (so an offline import of the provider works) with a placeholder — the 401 surfaces
# clearly at call time. Never a real secret.
_MISSING_KEY = "MISSING_LITELLM_KEY"


class LiteLLMServedModelMismatch(RuntimeError):
    """The gateway does not advertise the alias this profile pins (fail-closed).

    Raised so a mis-typed or absent alias stops the run instead of silently routing to nothing / a
    wrong model and producing a corpus attributed to the wrong candidate.
    """


def _served_matches(expected: str, served: Set[str]) -> bool:
    """True iff the configured alias EXACTLY matches (casefold) one the gateway advertises.

    Unlike the vLLM check (dated HF ids with version-suffix tolerance), gateway aliases are exact
    contract names — so no ``startswith`` tolerance, which also avoids the both-directions
    ``startswith`` trap (``homelab-flash`` spuriously matching ``homelab-flash-thinking``).
    """
    e = expected.casefold()
    return any(s.casefold() == e for s in served)


class LiteLLMProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for the homelab LiteLLM gateway (alias-routed, multi-vendor)."""

    _CONFIG_NS: str = "litellm"
    _TELEMETRY_PROVIDER: str = "litellm"
    _PROVIDER_LABEL: str = "LiteLLM"
    # Gateway routes to deepseek/qwen/glm/kimi, which support an 8192-token output — double the
    # OpenAI-native 4096 cap. Cleaning a full transcript needs the headroom or it truncates
    # (finish_reason=length) and the guardrail discards the cleaned text (#1356).
    _CLEANING_MAX_TOKENS_CAP: int = 8192

    def __init__(self, cfg: config.Config):
        super().__init__(cfg)
        # Gateway routes open / reasoning models; none of the OpenAI-native temperature-rejection
        # heuristics apply, so start with an empty "temperature-fixed" set.
        self._temp_fixed_at_default = set()
        # Cleaning defaults to the summary alias unless a profile pins litellm_cleaning_model.
        if not self.cleaning_model:
            self.cleaning_model = self.summary_model

    def _authenticate(self, cfg: "config.Config") -> None:
        """A gateway virtual key is required at call time, but it is NOT an ``sk-`` OpenAI key, so
        skip the OpenAI-format validation. Warn (never raise) if none is resolvable so an offline
        import still works; the 401 surfaces the real problem at call time."""
        if not self._resolve_api_key(cfg) or self._resolve_api_key(cfg) == _MISSING_KEY:
            logger.warning(
                "litellm: no gateway virtual key resolved (set litellm_api_key or the env named by "
                "litellm_api_key_env, default LITELLM_API_KEY); calls will 401 until it is set."
            )
        return None

    def _resolve_api_key(self, cfg: "config.Config") -> Optional[str]:
        """Bearer for the gateway: ``litellm_api_key``, else the env named by
        ``litellm_api_key_env`` (default ``LITELLM_API_KEY``), else a placeholder so construction
        succeeds (the gateway 401 then surfaces the missing key clearly)."""
        explicit: Optional[str] = getattr(cfg, "litellm_api_key", None)
        if explicit:
            return explicit
        env_name: str = getattr(cfg, "litellm_api_key_env", None) or "LITELLM_API_KEY"
        return os.getenv(env_name) or _MISSING_KEY

    def _token_kwarg(self, n: int, model: Optional[str] = None) -> Dict[str, Any]:
        """Gateway-routed models use the classic ``max_tokens`` (no o1/o3/gpt-5 rename)."""
        return {"max_tokens": n}

    def initialize(self) -> None:
        """Fail-closed served-alias check before first use, then the normal init."""
        if getattr(self.cfg, "litellm_verify_served_model", True):
            self._verify_served_model()
        super().initialize()

    def _verify_served_model(self) -> None:
        """Assert the gateway advertises the alias this profile pins.

        An UNREACHABLE gateway only warns — the real inference call surfaces a connection error
        anyway, and hard-failing here would make an offline import impossible. A REACHABLE gateway
        that does not advertise the alias raises.
        """
        base = getattr(self.cfg, "litellm_api_base", None)
        expected = self.summary_model
        if not base or not expected:
            return
        url = f"{base.rstrip('/')}/models"
        key = self._resolve_api_key(self.cfg)
        try:
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 — fixed profile URL
                data = json.loads(resp.read().decode("utf-8")).get("data", [])
        except Exception as exc:  # noqa: BLE001 — unreachable != mismatch; surface at call time
            logger.warning(
                "litellm: could not verify served alias at %s (%s); the inference call will "
                "surface any real connectivity problem",
                url,
                type(exc).__name__,
            )
            return
        served: Set[str] = set()
        for entry in data if isinstance(data, list) else []:
            for entry_key in ("id", "root"):
                val = entry.get(entry_key) if isinstance(entry, dict) else None
                if isinstance(val, str) and val:
                    served.add(val)
        if not _served_matches(expected, served):
            raise LiteLLMServedModelMismatch(
                f"LiteLLM gateway at {base} advertises {sorted(served) or '<none>'} but this "
                f"profile pins alias {expected!r}. Add the alias to infra/litellm/config.yaml "
                f"(+ reload the gateway) or fix litellm_summary_model. Refusing to run."
            )
        logger.info("litellm: served-alias check OK (%s advertised at %s)", expected, base)
