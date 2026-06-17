"""Per-model output post-processors for eval predictions.

Each candidate in the autoresearch eval can declare a ``postprocessor:``
in its experiment YAML config — see
``PromptConfig.postprocessor`` in ``experiment_config.py``. The
post-processor is applied to the LLM's raw response text BEFORE the
prediction is written to ``predictions.jsonl``, so all downstream
consumers (scorer, judges, manual inspection) see the cleaned text.

Why this lives with the model adapter, not the scorer:

- The prompt contract is part of the model's *harness shape*. R1-Distill,
  Magistral, and Mistral-Small (when run with R1-borrowed prompts) all
  emit ``<summary>...</summary>`` wrapping and may leak reasoning prose;
  Qwen with ``enable_thinking=False`` and gemini don't. The strip
  strategy belongs alongside the prompt+sampling that produced the
  output, not as a scorer-side concern.
- Keeps ``predictions.jsonl`` faithful to "what a downstream consumer
  would actually use" — if production routes summary to this model with
  this prompt, the same strip applies in prod via the same module.

Registry
========

``REGISTRY`` maps a string key (set in YAML's ``prompts.postprocessor``)
to a function ``Callable[[str], str]``. Add new entries here when a new
model's prompt contract introduces a new tag convention.

Currently registered:

- ``strip_r1_reasoning`` — the original #961 R1 post-processor. Handles
  ``<summary>...</summary>`` wrapping (closed or only-open or only-close),
  ``<think>...</think>`` blocks, leaked reasoning preambles. Idempotent
  and safe on already-clean output (verified empirically on
  Mistral-Small-3.2 outputs where no tags were emitted).
"""

from __future__ import annotations

from typing import Callable, Dict

from podcast_scraper.evaluation.r1_postprocess import strip_r1_reasoning


def noop(text: str) -> str:
    """Identity post-processor for models that emit clean text already."""
    return text


# Byte-level BPE tokenizer artifacts that vLLM 26.05-py3 emits raw for
# DeepSeek-R1-Distill on the autoresearch stack (observed 2026-06-16).
# Root cause: the tokenizer's byte-level decode step isn't being applied
# in the chat-completions response path. Until the upstream image
# bumps fix this, we strip them here. Each entry is
# ``(raw_token, decoded_char)``.
_R1_BYTE_LEVEL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # Multi-byte first so they're not eaten by single-char rules.
    ("âĢĻ", "’"),  # right single quotation mark (’)
    ("âĢľ", "“"),  # left double quotation mark (“)
    ("âĢĿ", "”"),  # right double quotation mark (”)
    ("âĢĶ", "—"),  # em dash (—)
    ("âĢĵ", "–"),  # en dash (–)
    ("âĢ¦", "…"),  # horizontal ellipsis (…)
    # Single-char rules — order doesn't matter at this point.
    ("Ġ", " "),
    ("Ċ", "\n"),
)


def decode_r1_byte_level(text: str) -> str:
    """Decode raw byte-level BPE token artifacts.

    See ``_R1_BYTE_LEVEL_REPLACEMENTS`` for the mapping rationale.
    No-op on clean text (idempotent), so safe to apply as a tail
    post-processor on any model's output.
    """
    for raw, decoded in _R1_BYTE_LEVEL_REPLACEMENTS:
        if raw in text:
            text = text.replace(raw, decoded)
    return text


def strip_r1_reasoning_and_decode(text: str) -> str:
    """Compose: strip R1 reasoning prose/tags THEN byte-level decode.

    Applied to DeepSeek-R1-Distill outputs on vLLM 26.05-py3 where
    the served-completion path leaks byte-level BPE artifacts even
    after ``--reasoning-parser=deepseek_r1`` separates think blocks.
    """
    return decode_r1_byte_level(strip_r1_reasoning(text))


def extract_json_summary_field(text: str) -> str:
    """Extract the ``summary`` field from a JSON-mode response.

    Used with prompts that ask the model to emit ``{"summary": "..."}``
    (e.g. Kimi-Linear Round 3 v3_json — JSON mode side-steps the model's
    refusal + task-narration failure modes by forcing structure).

    Behavior:

    - Strips a single leading ```json fenced block if present.
    - Parses the JSON object and returns the value of ``summary``.
    - On any parse/lookup failure, returns the original text unchanged so
      the eval scorer + manual inspection see what the model actually
      emitted (don't silently lose the raw output).
    """
    import json
    import re

    candidate = text.strip()
    # Strip a fenced code block if the model decided to be helpful.
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and isinstance(obj.get("summary"), str):
            return obj["summary"]
    except (json.JSONDecodeError, ValueError):
        pass
    return text


REGISTRY: Dict[str, Callable[[str], str]] = {
    "strip_r1_reasoning": strip_r1_reasoning,
    "strip_r1_reasoning_and_decode": strip_r1_reasoning_and_decode,
    "decode_r1_byte_level": decode_r1_byte_level,
    "extract_json_summary_field": extract_json_summary_field,
    "noop": noop,
}


def get_postprocessor(name: str | None) -> Callable[[str], str]:
    """Look up a post-processor by name.

    ``None`` or missing keys return :func:`noop`. Raise ``KeyError`` only
    on an explicit unknown name — defaulting to no-op on missing entries
    lets configs omit the field for the common no-postprocessing case.
    """
    if not name:
        return noop
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown output postprocessor: {name!r}. "
            f"Known: {sorted(REGISTRY.keys())}. "
            f"Register new entries in "
            f"``podcast_scraper.evaluation.output_postprocess.REGISTRY``."
        )
    return REGISTRY[name]


__all__ = ["REGISTRY", "get_postprocessor", "noop"]
