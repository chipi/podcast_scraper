"""Shared bundled-mode prompts for the GIL evidence stack (#698).

These are the system + user prompt fragments shared by every provider's
``extract_quotes_bundled`` / ``score_entailment_bundled`` implementation.
Provider methods only differ in the SDK call shape; the prompts are
identical because the parsers in
``providers/common/{bundle_extract_parser,bundle_nli_parser}.py`` expect a
specific JSON contract.

Keeping these here means a prompt tweak (e.g. RFC-073 Track A optimization)
lands in one place and applies to all providers, not six.
"""

from __future__ import annotations

from typing import List, Tuple

EXTRACT_QUOTES_BUNDLED_SYSTEM = (
    "For EACH insight below, extract 3-5 short verbatim quotes from the "
    "transcript that support it. Each quote MUST be a different passage — "
    "never repeat. Reply with ONLY a JSON object mapping the integer "
    "insight index (as a string) to an array of quote strings: "
    '{"0": ["quote A", "quote B"], "1": ["quote C"], ...}. '
    "If an insight has no supporting quote, return an empty array for it."
)


SCORE_ENTAILMENT_BUNDLED_SYSTEM = (
    "For each numbered (premise, hypothesis) pair below, rate how much the "
    "premise supports the hypothesis on a scale from 0 (not at all) to 1 "
    "(fully supports). Reply with ONLY a JSON object mapping the integer "
    'index (as a string) to its score: {"0": 0.85, "1": 0.42, ...}.'
)


def extract_quotes_bundled_user(transcript: str, insight_texts: List[str]) -> str:
    """Render the user message for ``extract_quotes_bundled``.

    Caller is responsible for clipping ``transcript`` to a budget appropriate
    for the provider's context window (Gemini uses 50_000 chars; smaller
    models may need less).
    """
    numbered_insights = "\n".join(
        f"{idx}: {text.strip()}" for idx, text in enumerate(insight_texts)
    )
    return (
        f"Transcript (excerpt):\n{transcript.strip()}\n\n"
        f"Insights:\n{numbered_insights}\n\n"
        "Return JSON only."
    )


def score_entailment_bundled_user(pairs: List[Tuple[str, str]]) -> str:
    """Render the user message for one chunk of ``score_entailment_bundled`` pairs."""
    numbered_pairs_lines = []
    for idx, (premise, hypothesis) in enumerate(pairs):
        numbered_pairs_lines.append(
            f"{idx}:\n  premise: {premise.strip()}\n  hypothesis: {hypothesis.strip()}"
        )
    numbered_pairs = "\n".join(numbered_pairs_lines)
    return f"Pairs:\n{numbered_pairs}\n\nReturn JSON only."


def extract_quotes_bundled_max_tokens(num_insights: int) -> int:
    """Default output budget for ``extract_quotes_bundled``.

    Roughly: 5 quotes × 100 chars × N insights × ~1.3 tokens-per-char.
    Floored at 1024, capped at 8192.
    """
    return max(1024, min(8192, 256 * max(1, num_insights)))


def score_entailment_bundled_max_tokens(chunk_size: int) -> int:
    """Default output budget for one bundled-NLI chunk.

    Roughly: 25 chars per pair line + envelope.
    Floored at 256, capped at 8192.
    """
    return max(256, min(8192, 30 * max(1, chunk_size)))


def transcript_clip(transcript: str, max_chars: int = 50_000) -> str:
    """Clip transcript to provider-appropriate budget. Default matches Gemini's 50k."""
    return transcript.strip()[:max_chars]
