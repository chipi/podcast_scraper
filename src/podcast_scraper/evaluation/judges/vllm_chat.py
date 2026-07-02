"""Generic vLLM chat judge — OpenAI-compatible.

Sibling of :mod:`ollama_chat`. Same chat-completions shape, different base URL
(8003 not 11434), and an explicit ``thinking`` toggle for reasoning-capable
candidates (Qwen3 / Qwen3.5 with ``enable_thinking`` chat_template_kwarg).
Why thinking matters here: per :doc:`MODEL_PLAYBOOK`, thinking-enabled judges
enumerate flaws before scoring, which materially improves discrimination over
single-pass calls — exactly the W27 saturation problem (judge_mean
0.925-0.975 across the entire 7-candidate cohort) that motivated the swap.

Wire it via ``provider: vllm`` in ``judge_config_*.yaml``; the dispatcher in
:mod:`autoresearch_track_a._call_judge` routes accordingly.
"""

from __future__ import annotations

import logging
import os
from typing import Any, cast, Optional

import httpx

from podcast_scraper.evaluation.judges.base import JudgeUnavailableError
from podcast_scraper.evaluation.judges.ollama_chat import _parse_score
from podcast_scraper.utils.retry import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


def _resolve_vllm_base() -> str:
    """Resolve the vLLM autoresearch base URL.

    Precedence:
    1. ``VLLM_API_BASE`` — established project convention (see
       ``scripts/eval/onboard_model_smoke.py``, ``autoresearch/MODEL_PLAYBOOK.md``).
    2. ``DGX_TAILNET_FQDN`` → ``http://<fqdn>:8003/v1`` — autoresearch vLLM
       lives on port 8003 (the ``coder-next`` IDE vLLM lives on a different
       port and is OFF-LIMITS per [[project_dgx_vllm_distinction]]).
    3. ``http://localhost:8003/v1`` — tests / local dev fallback.
    """
    direct = os.environ.get("VLLM_API_BASE", "").strip()
    if direct:
        return direct.rstrip("/")
    fqdn = os.environ.get("DGX_TAILNET_FQDN", "").strip()
    if fqdn:
        return f"http://{fqdn}:8003/v1"
    return "http://localhost:8003/v1"


def _resolve_vllm_api_key() -> str:
    """Resolve the vLLM API key from ``VLLM_API_KEY``. Empty string when
    unset — the transport then omits the Authorization header entirely,
    which is what a no-auth vLLM (local dev, unauthenticated tests) wants.
    Homelab vLLM autoresearch on DGX serves under ``--api-key <token>``;
    export that token as ``VLLM_API_KEY`` in the shell / GHA env / .env."""
    return os.environ.get("VLLM_API_KEY", "").strip()


class VllmChatJudge:
    """vLLM judge over OpenAI-compatible ``/chat/completions``.

    The same scalar-score contract as :class:`OllamaChatJudge`: returns a
    float in [0, 10] parsed from the model's reply. The pairwise judging
    rubric (separate follow-up) layers on top of this transport — the
    transport stays scalar-or-pairwise-agnostic so both use the same
    retry, parsing, and error-handling.
    """

    def __init__(
        self,
        *,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str,
        request_timeout_s: float = 180.0,
        thinking: bool = False,
        client: object | None = None,
    ) -> None:
        self.api_base = (api_base or _resolve_vllm_base()).rstrip("/")
        self.api_key = api_key if api_key is not None else _resolve_vllm_api_key()
        self.model = model
        self.request_timeout_s = request_timeout_s
        self.thinking = thinking
        self._client = client

    def _headers(self) -> dict[str, str]:
        """Bearer-token auth when ``api_key`` is set; empty when not.
        Homelab vLLM autoresearch on DGX serves under ``--api-key <token>``
        so requests without the header return 401; local no-auth vLLM
        (dev / test) returns fine either way."""
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _do_request(self, *, user_content: str) -> dict[str, Any]:
        url = f"{self.api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.0,
        }
        if self.thinking:
            # vLLM passes ``chat_template_kwargs`` straight through to the
            # tokenizer's chat template. Qwen3 / Qwen3.5 expose
            # ``enable_thinking`` there; other reasoning-capable models
            # accept the same flag. No-op for models that don't read it.
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        headers = self._headers()
        if self._client is not None:
            return cast(
                dict,
                self._client.post(  # type: ignore[attr-defined]
                    url, json=payload, headers=headers
                ).json(),
            )
        with httpx.Client(timeout=self.request_timeout_s) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return cast(dict, resp.json())

    def raw(self, *, user_content: str) -> str:
        """POST to ``/chat/completions``; return the raw assistant text.

        Same retry policy as ``score``. Pairwise judging (which parses
        its own JSON schema — see :mod:`podcast_scraper.evaluation.pairwise`)
        uses this directly; scalar callers go through ``score``.
        """
        retryable = (
            httpx.RequestError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            httpx.HTTPStatusError,
            OSError,
        )
        try:
            reply = retry_with_exponential_backoff(
                lambda: self._do_request(user_content=user_content),
                max_retries=3,
                initial_delay=2.0,
                max_delay=15.0,
                retryable_exceptions=retryable,
            )
        except (httpx.HTTPError, OSError) as exc:
            raise JudgeUnavailableError(
                f"vLLM judge ({self.model} at {self.api_base}) unreachable " f"after retries: {exc}"
            ) from exc

        choices = reply.get("choices") or []
        if not choices:
            raise JudgeUnavailableError(f"vLLM judge ({self.model}) returned no choices")
        return (choices[0].get("message") or {}).get("content") or ""

    def score(self, *, user_content: str) -> float:
        """POST to ``/chat/completions``; return the parsed float in [0, 10].

        Thin wrapper over :meth:`raw`. Scalar callers use this; pairwise
        callers use ``raw`` directly.
        """
        return _parse_score(self.raw(user_content=user_content))
