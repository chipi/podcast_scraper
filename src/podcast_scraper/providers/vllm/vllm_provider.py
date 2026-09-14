"""VLLMProvider — DGX-local open-model serving over vLLM's OpenAI-compatible API (ADR-147).

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

import json
import logging
import os
import urllib.request
from typing import Any, Dict, Optional, Set, Tuple

from ... import config
from ..openai.openai_provider import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# vLLM served without auth still requires the OpenAI SDK client to carry *some* bearer; it is
# ignored server-side. Never a real secret.
_VLLM_DUMMY_BEARER = "EMPTY"


class VLLMServedModelMismatch(RuntimeError):
    """The vLLM endpoint serves a different model than the profile pins (ADR-147 B3).

    Raised fail-closed so a wrong model loaded on the DGX slot stops the run instead of silently
    producing a corpus attributed to the wrong model.
    """


def _served_matches(expected: str, served: Set[str]) -> bool:
    """True if the configured model id matches one the endpoint advertises. Casefold + dated/version
    suffix tolerance (startswith either way); the org prefix is NOT stripped, so
    ``Qwen/...`` never matches ``someoneelse/...``."""
    e = expected.casefold()
    for s in served:
        sc = s.casefold()
        if sc == e or sc.startswith(e) or e.startswith(sc):
            return True
    return False


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
        # (model, hash, len) -> exact token count. The same transcript is budgeted at several
        # stages and the answer cannot change between them, so ask the server once (#2050).
        self._token_count_cache: Dict[Tuple[str, int, int], int] = {}
        # Cleaning defaults to the summary model when a profile does not pin vllm_cleaning_model —
        # one served model handles the whole cascade unless explicitly split.
        if not self.cleaning_model:
            self.cleaning_model = self.summary_model

    def _authenticate(self, cfg: "config.Config") -> None:
        """A local vLLM bearer is optional — no required-key / ``sk-`` validation (ADR-147)."""
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

    def initialize(self) -> None:
        """Fail-closed served-model check before first use (ADR-147 B3), then the normal init."""
        if getattr(self.cfg, "vllm_verify_served_model", True):
            self._verify_served_model()
        super().initialize()

    def count_tokens(self, text: str) -> Optional[int]:
        """Exact token count from vLLM's own ``POST /tokenize`` (#2050).

        This is the number the server will use to accept or reject the request, so budgeting
        against it removes the guess entirely. Every context-overflow bug in this repo came from a
        chars-per-token constant that was fractionally optimistic: measured 2026-09-13, vLLM
        rejected the same 106,905-char prompt **508 times in 30 days**, each at 30,721 input
        tokens against a 32,768 limit — over by exactly one token, deterministically.

        Returns ``None`` on any failure. A tokenizer outage must degrade to the pessimistic
        estimate, never take down the stage it was meant to protect.

        Cached per (model, text) because the same transcript is budgeted at several stages and the
        answer cannot change between them.
        """
        base = getattr(self.cfg, "vllm_api_base", None)
        model = self.summary_model
        if not base or not model or not text:
            return None
        key = (model, hash(text), len(text))
        cached = self._token_count_cache.get(key)
        if cached is not None:
            return cached
        # /tokenize is a vLLM extension and sits at the server root, NOT under /v1.
        root = base.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        url = f"{root}/tokenize"
        payload = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self._resolve_api_key(self.cfg)}",
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 — profile URL
                count = json.loads(resp.read().decode("utf-8")).get("count")
        except Exception as exc:  # noqa: BLE001 — never fail a stage over a token count
            logger.warning(
                "vllm: /tokenize unavailable at %s (%s); budgeting falls back to the pessimistic "
                "character estimate (#2050)",
                url,
                type(exc).__name__,
            )
            return None
        if not isinstance(count, int) or count <= 0:
            return None
        self._token_count_cache[key] = count
        return count

    def _verify_served_model(self) -> None:
        """Assert the DGX slot actually serves the model this profile pins (real HF id).

        A wrong model loaded on the slot must fail the run, not silently produce a corpus attributed
        to the wrong model (ADR-143/144). An UNREACHABLE endpoint only warns — the real inference
        call surfaces a connection error anyway, and hard-failing here would make an offline import
        of the provider impossible. A REACHABLE endpoint serving a different model raises.
        """
        base = getattr(self.cfg, "vllm_api_base", None)
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
                "vllm: could not verify served model at %s (%s); the inference call will surface "
                "any real connectivity problem",
                url,
                type(exc).__name__,
            )
            return
        served: Set[str] = set()
        advertised_window: Optional[int] = None
        for entry in data if isinstance(data, list) else []:
            if not isinstance(entry, dict):
                continue
            names = {entry.get(key) for key in ("id", "root")}
            for val in names:
                if isinstance(val, str) and val:
                    served.add(val)
            # #2050: vLLM advertises the window it is actually serving. Taking it here means the
            # budget is a DISCOVERED property of this deployment instead of a module constant
            # sized to the narrowest model in the fleet — which is how one eval harness's
            # `--max-model-len=32768` became the episode-length policy for every provider.
            if advertised_window is None and _served_matches(expected, {n for n in names if n}):
                mml = entry.get("max_model_len")
                if isinstance(mml, int) and mml > 0:
                    advertised_window = mml
        if advertised_window is not None:
            if advertised_window != self.max_context_tokens:
                # %s, not %d: the declared window is Optional and is None when nothing declared
                # one (#2050). Formatting None with %d raises inside logging and would take down
                # the served-model check that this whole method exists to perform.
                logger.info(
                    "vllm: served context window is %s tokens (declared: %s); every transcript "
                    "budget now derives from the served figure (#2050)",
                    advertised_window,
                    self.max_context_tokens if self.max_context_tokens else "none",
                )
            self.max_context_tokens = advertised_window
        else:
            logger.warning(
                "vllm: %s advertises no max_model_len; keeping the declared window (%s). "
                "Budgets stay conservative rather than guessing upward (#2050).",
                base,
                self.max_context_tokens if self.max_context_tokens else "none — nothing is clipped",
            )
        if not _served_matches(expected, served):
            raise VLLMServedModelMismatch(
                f"vLLM at {base} serves {sorted(served) or '<none>'} but this profile pins "
                f"{expected!r}. Load the right model on the DGX slot (or fix vllm_summary_model). "
                f"Refusing to run to avoid corpus corruption (ADR-147 B3)."
            )
        logger.info("vllm: served-model check OK (%s advertised at %s)", expected, base)
