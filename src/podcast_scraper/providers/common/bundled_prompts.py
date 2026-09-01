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


#: Output tokens budgeted per insight. Was 256, which production disproved: on 2026-08-30
#: (prod_dgx_full / Qwen3-30B-A3B) ten-insight batches ended at exactly 2560/2560 with
#: ``finish_reason == "length"`` on three of eight episodes. Five verbatim quotes of 20-40
#: words each is 150-300 tokens before the JSON envelope, so 256 sat under the mean, not
#: above it. 384 puts the budget above the observed distribution instead of inside it.
#:
#: Raising this is close to free: ``max_tokens`` is a ceiling, and billing follows tokens
#: actually emitted. The overflow PATH still matters and is still tested — a verbose model or
#: a larger chunk can exceed any fixed number, which is what the bisect exists for.
#:
#: 384 -> 640, measured on the 2026-08-31 DGX batch over 1345 ``extract_quotes`` calls. 384 was
#: still INSIDE the distribution, not above it:
#:
#:     completion_tokens  p50=989  p75=1310  p90=1902  p99=3840  max=3840
#:
#: p90 (1902) lands within a token of the 5-insight ceiling (1920), and 63 calls sat EXACTLY on
#: a ceiling — 1024 x11, 1920 x18, 2688 x1, 3840 x33 — i.e. truncated. That matches the 68
#: ``DOCUMENT_ENDED_EARLY`` parse failures in the same window. The apparent p99/max of 3840 is
#: censored BY the ceiling, so the true requirement is higher than anything measurable here.
#:
#: 640 clears the censored p99 at every batch size (10 x 640 = 6400, still under the 8192 cap)
#: while staying below it.
#:
#: Why raising the ceiling is right HERE but was wrong for insight generation (where the fix
#: was to LOWER the ask): quote extraction is not runaway. Its p50 is 989 against ceilings of
#: 1024-3840 — a model that were filling its budget would sit AT the ceiling, and it does not.
#: The response is a fixed-shape JSON of ~5 quotes per insight, not an open-ended list. And
#: truncation here is expensive rather than salvageable: the JSON is unparsable, so the batch
#: yields ZERO quotes, gets bisected into two more calls, and a size-1 failure drops to the
#: much costlier per-insight staged path.
_QUOTE_TOKENS_PER_INSIGHT = 640


#: Hard ceiling on the quote-extraction output budget, 8192 -> 5120.
#:
#: This exists because the request has to FIT: prompt + output must stay inside the served
#: context window, and the DGX serves Qwen3-30B at ``max_model_len`` 32768. Measured prompt
#: sizes for ``extract_quotes`` over 1507 production calls:
#:
#:     p50=12044   p90=16441   p99=22242   MAX=26714
#:
#: The default batch is ``QUOTE_BUNDLE_CHUNK_SIZE = 10``, so raising the per-insight budget to
#: 640 would ask for 6400 output tokens, and 26714 + 6400 = 33114 — OVER the window, on the
#: DEFAULT path, for the longest prompts we actually send. That is #1893's error
#: ("Chat completion exceeds Qwen3-30B model context length limit") and it would have been
#: introduced by the 384 -> 640 raise: the OLD effective maximum was 384 x 10 = 3840, which
#: always fit.
#:
#: 5120 keeps the worst observed prompt inside the window (26714 + 5120 = 31834, ~930 to
#: spare) while still giving a full batch a third more room than the 3840 it had before. Small
#: batches are unaffected — the per-insight rate only stops applying at 8+ insights.
#:
#: This is a STATIC bound standing in for the right fix. The budget should be computed against
#: the served context minus the ACTUAL prompt size at call time, which is the only thing that
#: is correct for every model and every transcript length; a fixed number fitted to one model's
#: window and one corpus's prompt distribution will drift. Tracked on #1893.
_QUOTE_MAX_OUTPUT_TOKENS = 5120


def extract_quotes_bundled_max_tokens(num_insights: int) -> int:
    """Default output budget for ``extract_quotes_bundled``.

    Roughly: 5 quotes × 20-40 words × N insights, plus JSON envelope.
    Floored at 1024, capped at ``_QUOTE_MAX_OUTPUT_TOKENS`` so prompt + output fits the served
    context window (see that constant — the cap is a context-fit bound, not a cost bound).
    """
    return max(
        1024, min(_QUOTE_MAX_OUTPUT_TOKENS, _QUOTE_TOKENS_PER_INSIGHT * max(1, num_insights))
    )


def score_entailment_bundled_max_tokens(chunk_size: int) -> int:
    """Default output budget for one bundled-NLI chunk.

    Roughly: 25 chars per pair line + envelope.
    Floored at 256, capped at 8192.
    """
    return max(256, min(8192, 30 * max(1, chunk_size)))


def transcript_clip(transcript: str, max_chars: int = 50_000) -> str:
    """Clip transcript to provider-appropriate budget. Default matches Gemini's 50k."""
    return transcript.strip()[:max_chars]
