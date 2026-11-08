"""Module-level helpers ollama_provider and its GI/naming methods share (moved verbatim from
``ollama_provider``, which imports them back)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ...models import Episode
else:
    from ... import models

    Episode = models.Episode  # type: ignore[assignment]

logger = logging.getLogger("podcast_scraper.providers.ollama.ollama_provider")


def _ollama_openai_chat_extra_kwargs(model: str, num_ctx: Optional[int] = None) -> Dict[str, Any]:
    """Extra kwargs for Ollama's OpenAI-compatible ``/v1/chat/completions``.

    Two Ollama-specific concerns handled here:

    1. **Context window (num_ctx)** — Ollama defaults to 2048 tokens, which silently
       truncates any prompt longer than that. For summarisation we need the full
       transcript, so we pass ``num_ctx`` explicitly via ``extra_body``. Models have
       different maximums (e.g. gemma2 = 8k, qwen3.5 = 256k); pick a value large
       enough for your expected prompt length but within the model's range.

    2. **Qwen 3.x chain-of-thought** — Qwen 3.x reasoning models (3.5, 3.6, and
       newer variants in the qwen3 family) emit a separate ``reasoning`` /
       ``thinking`` channel by default. Without ``reasoning_effort: none``,
       ``message.content`` can stay empty while the model consumes
       ``max_tokens`` on thinking tokens (finish_reason ``length``).
       Observed on 2026-06-08 sweep: qwen3.6:latest produced 0-token content
       for every episode until we added it here. See:
       https://docs.ollama.com/capabilities/thinking
    """
    m = (model or "").lower()
    extra_body: Dict[str, Any] = {}
    if num_ctx is not None:
        # Ollama's OpenAI-compat endpoint accepts native Ollama options via extra_body.
        extra_body["options"] = {"num_ctx": int(num_ctx)}
    # Match any qwen3.x reasoning model — qwen3.5, qwen3.6, qwen3-coder, etc.
    if any(tag in m for tag in ("qwen3.5", "qwen3.6", "qwen3-", "qwen3:")):
        extra_body["reasoning_effort"] = "none"
    return {"extra_body": extra_body} if extra_body else {}


def _flatten_json_speaker_names(value: Any) -> List[str]:
    """Flatten nested JSON name lists from Ollama JSON speaker responses."""
    if value is None:
        return []
    if isinstance(value, str):
        t = value.strip()
        return [t] if t else []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for v in value:
            out.extend(_flatten_json_speaker_names(v))
        return out
    t = str(value).strip()
    return [t] if t else []
