"""Post-process DeepSeek-R1-Distill summary outputs.

The R1-Distill family emits a "thinking" preamble before the actual
answer by default — e.g. an opening ``<think>...</think>`` block or
free-form prose like *"Okay, so I need to summarize..."*. The
companion prompt at ``prompts/vllm/r1_distill_32b/summarization/`` asks
the model to wrap its summary in ``<summary>...</summary>`` tags to
make extraction unambiguous, but small models still leak preamble or
forget the closing tag in some cases.

This module is the belt-and-braces layer: a single ``strip_r1_reasoning``
function that tries several extraction strategies in order, returning
the cleanest possible summary text. Used by the autoresearch
experiment runner when the experiment YAML asks for it — see #961.
"""

from __future__ import annotations

import re

# Order matters: try the cleanest extractor first, fall back to coarser
# strips. Each regex must be tolerant of leading/trailing whitespace.
_SUMMARY_BLOCK_RE = re.compile(
    r"<summary>\s*(.*?)\s*</summary>",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_SUMMARY_ONLY_RE = re.compile(
    r"<summary>\s*(.*)\Z",
    re.IGNORECASE | re.DOTALL,
)
_THINK_BLOCK_RE = re.compile(
    r"<think>.*?</think>\s*",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_THINK_NO_CLOSE_RE = re.compile(
    r"\A<think>.*\Z",
    re.IGNORECASE | re.DOTALL,
)

# Common free-form preamble openers R1-Distill emits when it forgets
# the tag contract. Kept anchored to start-of-text to avoid eating
# legitimate paragraph openers mid-summary.
_PREAMBLE_OPENERS = (
    "okay, so",
    "okay so",
    "alright, so",
    "alright so",
    "let me think",
    "let's think",
    "first, i",
    "first i",
    "i need to",
    "i'll start by",
    "to summarize",
)


def strip_r1_reasoning(text: str) -> str:
    """Return the cleaned summary text from an R1-Distill response.

    Strategies, tried in order:
      1. If a complete ``<summary>...</summary>`` block is present,
         return its contents (most preferred — the prompt's
         tag-contract was honored).
      2. If only an opening ``<summary>`` tag is present (closing tag
         forgotten), return everything after it.
      3. If a complete ``<think>...</think>`` block is present, strip
         it and return the rest.
      4. If text starts with an opening ``<think>`` but has no close
         tag (model hit max_tokens mid-reasoning), return empty —
         there's no actual summary to extract.
      5. If the text starts with a known preamble opener and contains
         a double newline, drop the leading paragraph and return the
         rest. (Heuristic; only fires when the prompt failed entirely.)
      6. Otherwise, return the input stripped of leading/trailing
         whitespace.
    """
    if not text:
        return ""

    block_match = _SUMMARY_BLOCK_RE.search(text)
    if block_match:
        return block_match.group(1).strip()

    open_only_match = _OPEN_SUMMARY_ONLY_RE.search(text)
    if open_only_match:
        return open_only_match.group(1).strip()

    if _OPEN_THINK_NO_CLOSE_RE.match(text) and "</think>" not in text.lower():
        return ""

    if _THINK_BLOCK_RE.search(text):
        stripped = _THINK_BLOCK_RE.sub("", text, count=1).lstrip()
        return stripped.strip()

    lower = text.lstrip().lower()
    if any(lower.startswith(opener) for opener in _PREAMBLE_OPENERS):
        # Drop the leading paragraph if there's a clear break.
        parts = text.lstrip().split("\n\n", 1)
        if len(parts) == 2:
            return parts[1].strip()

    return text.strip()


__all__ = ["strip_r1_reasoning"]
