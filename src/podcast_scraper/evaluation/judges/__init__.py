"""LLM judge clients for the G-Eval finale tier (#932 + #940 Track 1).

Each judge exposes a uniform ``score`` callable:

    JudgeResult = score(prompt: str, *, max_tokens: int = 1024) -> JudgeResult

where ``JudgeResult`` carries the raw assistant text, an estimated token cost
(prompt+completion tokens when the API reports them), and an estimated USD
cost.  G-Eval parsing happens at the caller (``g_eval.py``), so judges are
deliberately thin — they own *only* the transport and pricing-instrumentation.

The three concrete implementations cover the finale plan:

- :mod:`sonnet46`     — Anthropic Claude Sonnet 4.6 (primary judge)
- :mod:`gemini25pro`  — Google Gemini 2.5 Pro       (cross-check on top-2)
- :mod:`deepseek_r1`  — DeepSeek-R1 32B via DGX Ollama (cheap tertiary, #940)
"""

from __future__ import annotations

from podcast_scraper.evaluation.judges.base import JudgeResult, JudgeUnavailableError
from podcast_scraper.evaluation.judges.deepseek_r1 import DeepSeekR1Judge
from podcast_scraper.evaluation.judges.gemini25pro import Gemini25ProJudge
from podcast_scraper.evaluation.judges.sonnet46 import Sonnet46Judge

__all__ = [
    "DeepSeekR1Judge",
    "Gemini25ProJudge",
    "JudgeResult",
    "JudgeUnavailableError",
    "Sonnet46Judge",
]
