"""ONE grounding prompt, for every provider. Nothing hardcoded.

A prompt in a file is a tuned parameter: versioned, diffable, A/B-testable, reviewable. A prompt in
code is none of those, and the bugs hide inside it. Ollama's inline quote prompt was replaced for
exactly that reason (#1179) — it carried three silent evidence-destroying defects at once:

    * it truncated the transcript to 50 000 chars, below our episode length, so the tail of every
      episode was invisible to quote extraction;
    * its system and user messages disagreed about the output shape (``{"quotes": [...]}`` in
      one, "quote_text only" in the other) — the model was told to return a list AND a scalar;
    * it embedded a copyable example string, which local models reproduce verbatim.

Every other provider still carried a buried variant of that prompt, and they were NOT the same
prompt. Measured before this module existed:

    ollama      a prompt FILE: "sweep the whole transcript... two quotes is usually too few"
    openai      a prompt FILE asking for `quote_text` — ONE quote, while its own parser reads a list
    gemini      an INLINE string asking for a list AND for "quote_text only" — contradicting itself
    the rest    the same inline string, unversioned and untunable

So a bake-off across providers was really a bake-off across prompts we happened to give them. Worse,
the ENTAILMENT wording is itself a gate: asking for strict textual entailment ("does the premise
entail the hypothesis") is not the question the pipeline means — a quote can be excellent evidence
for an insight without logically entailing it — and asking it strictly cost **60% of the evidence**
a trusted annotator had accepted (#1179). Only ollama had the corrected wording.

One template, rendered for all seven providers. The per-provider prompt files are byte-identical
copies, so a provider can still diverge deliberately — but it can no longer diverge by accident.
"""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# The transcript we can actually show the model. Cloud context windows are large; the local path
# computes its own budget from num_ctx and passes it in.
DEFAULT_TRANSCRIPT_BUDGET_CHARS = 120_000


def render_extract_quote_prompt(
    provider: str,
    transcript: str,
    insight: str,
    budget_chars: int = DEFAULT_TRANSCRIPT_BUDGET_CHARS,
) -> Tuple[str, str]:
    """``(system, user)`` for GIL quote extraction, from ``<provider>/evidence/extract_quote/v1``.

    The system message is empty on purpose: the template carries the whole instruction, so there is
    exactly one place to read and one place to change.
    """
    from ...prompts.store import render_prompt

    text = (transcript or "").strip()
    if len(text) > budget_chars:
        logger.warning(
            "transcript %d chars exceeds the %d-char budget; quote extraction will not see the "
            "tail of this episode",
            len(text),
            budget_chars,
        )
        text = text[:budget_chars]

    return "", render_prompt(
        f"{provider}/evidence/extract_quote/v1",
        transcript=text,
        insight=(insight or "").strip(),
    )


def render_entailment_prompt(provider: str, premise: str, hypothesis: str) -> Tuple[str, str]:
    """``(system, user)`` for GIL entailment, from ``<provider>/evidence/entailment/v1``.

    THE WORDING IS THE GATE. Strict textual entailment is not the question the pipeline means, and
    asking it strictly cost 60% of the evidence a trusted annotator had accepted (#1179). Keeping it
    in a template is what lets it be calibrated instead of rediscovered.
    """
    from ...prompts.store import render_prompt

    return "", render_prompt(
        f"{provider}/evidence/entailment/v1",
        premise=(premise or "").strip(),
        hypothesis=(hypothesis or "").strip(),
    )
