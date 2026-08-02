"""VLLMProvider — DGX-local open-model serving over vLLM's OpenAI-compatible API (ADR-144).

A SIBLING of :class:`OpenAIProvider`, not a subclass: with vLLM we serve a wide family of
*non-OpenAI* open models (Qwen/DeepSeek/Llama), so it must not be modelled as "an OpenAI thing".
Both share the OpenAI-compatible transport via the common
:class:`~podcast_scraper.providers.openai.openai_provider.OpenAICompatibleProvider` base; this
class only changes *identity* (config namespace + telemetry), *auth* (a local vLLM bearer is
optional), and the OpenAI-native *heuristics* that do not apply to open models.

Unlike the ``openai`` provider, the model fields name the **real HF model id** on the wire — there
is no ``--served-model-name`` alias — so a profile is self-describing and reproducible (ADR-143).
The fail-closed served-model verification against ``GET /v1/models`` is wired in a follow-up step.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from ... import config
from ..openai.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# vLLM served without auth still requires the OpenAI SDK client to carry *some* bearer; it is
# ignored server-side. Never a real secret.
_VLLM_DUMMY_BEARER = "EMPTY"


class VLLMProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider for a vLLM endpoint serving DGX-local open models.

    Overrides only what differs from OpenAI-native: the ``vllm_*`` config namespace, ``vllm``
    telemetry identity, optional-bearer auth, and the open-model token/temperature heuristics.
    Everything else (transcription is unused here; summary/speaker/GI/KG/grounding chat calls) is
    inherited unchanged from the shared transport base.
    """

    _CONFIG_NS: str = "vllm"
    _TELEMETRY_PROVIDER: str = "vllm"
    _PROVIDER_LABEL: str = "vLLM"

    def __init__(self, cfg: config.Config):
        super().__init__(cfg)
        # Open models (Qwen/DeepSeek/Llama) do not reject a non-default temperature the way some
        # OpenAI models do, so start with an empty "temperature-fixed" set (the base seeds it from
        # the OpenAI-only _TEMPERATURE_FIXED_MODELS constant).
        self._temp_fixed_at_default = set()
        # Cleaning defaults to the summary model when a profile does not pin vllm_cleaning_model —
        # one served model handles the whole cascade unless explicitly split.
        if not self.cleaning_model:
            self.cleaning_model = self.summary_model

    def _authenticate(self, cfg: "config.Config") -> None:
        """A local vLLM bearer is optional — no required-key / ``sk-`` validation (ADR-144)."""
        return None

    def _resolve_api_key(self, cfg: "config.Config") -> Optional[str]:
        """Bearer for the client: ``vllm_api_key``, else the env named by ``vllm_api_key_env``
        (default ``VLLM_API_KEY``), else a dummy (vLLM ignores it when served without auth)."""
        explicit: Optional[str] = getattr(cfg, "vllm_api_key", None)
        if explicit:
            return explicit
        env_name: str = getattr(cfg, "vllm_api_key_env", None) or "VLLM_API_KEY"
        from_env = os.getenv(env_name)
        if from_env:
            return from_env
        return _VLLM_DUMMY_BEARER

    def _token_kwarg(self, n: int, model: Optional[str] = None) -> Dict[str, Any]:
        """vLLM-served open models use the classic ``max_tokens``; there is no o1/o3/gpt-5
        ``max_completion_tokens`` rename to honour."""
        return {"max_tokens": n}
