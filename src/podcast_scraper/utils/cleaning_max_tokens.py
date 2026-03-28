"""Clamp max_tokens for LLM transcript cleaning (provider API limits).

Long transcripts can produce huge ``word_count * 0.85 * 1.3`` estimates that exceed
provider limits (DeepSeek [1, 8192], OpenAI per-model output caps).
"""

from __future__ import annotations

# Rough output token budget: cleaned text ~85% of words, plus buffer
_CLEANING_WORD_FACTOR = 0.85 * 1.3


def estimate_cleaning_output_tokens(word_count: int) -> int:
    """Estimate max output tokens from word count (legacy heuristic)."""
    return max(1, int(word_count * _CLEANING_WORD_FACTOR))


def clamp_cleaning_max_tokens(estimated: int, cap: int) -> int:
    """Clamp estimated tokens to ``[1, cap]``."""
    return max(1, min(cap, estimated))


# Provider-specific output caps for cleaning calls (conservative; avoids 400s)
OPENAI_CLEANING_MAX_TOKENS = 4096
DEEPSEEK_CLEANING_MAX_TOKENS = 8192
ANTHROPIC_CLEANING_MAX_TOKENS = 8192
MISTRAL_CLEANING_MAX_TOKENS = 8192
GROK_CLEANING_MAX_TOKENS = 8192
OLLAMA_CLEANING_MAX_TOKENS = 8192
# Gemini ``max_output_tokens`` (Generative API; long transcripts need a cap)
GEMINI_CLEANING_MAX_OUTPUT_TOKENS = 8192
