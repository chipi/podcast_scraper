"""RFC-111: shared transcript-prefix relocation for prompt caching, used by every LLM provider.

Providers cache an identical LEADING token prefix at ~0.1x price. The transcript-bearing LLM stages
of an episode (summary, GI insights, quote extraction, KG) all consume the SAME cleaned transcript,
so putting the transcript as the byte-stable leading block of the prompt makes it cache across those
stages (and across reprocessing runs, where the transcript is frozen).

The core (:func:`relocate_transcript`) is provider-family-agnostic: it finds the transcript verbatim
in the rendered user prompt and moves it to a byte-stable leading block, leaving a short marker in
its place. Each provider family then APPLIES that block to its own request container:

* OpenAI-style (openai/deepseek/qwen/litellm/vllm/grok/mistral/ollama): the transcript block leads
  the SYSTEM message — :func:`openai_style_messages`. Layout alone enables caching; no API fields.
* Anthropic: the block becomes a ``system`` content block carrying ``cache_control`` — handled in
  the anthropic provider (the block text + marker come from here so the wording is identical).
* Gemini: the block seeds a native ``cached_content`` handle — handled in the gemini provider.

The block wrapper MUST be assembled identically on every stage of an episode — any per-stage
interpolation before the transcript would shift the shared prefix and kill the cache (RFC-111 §8).
Caching reuses the model's compute (KV state), never the answer: this changes cost/latency only.
Moving the transcript DOES change the prompt wording (and thus can change the generated text), which
is why RFC-111 §6.8 gates rollout on a quality-parity check — but it does not change correctness.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Constant, byte-stable wrapper. Header + the same cleaned transcript + separator form the shared
# cache prefix; stage-specific instructions follow and diverge only after the cache boundary.
TRANSCRIPT_BLOCK_HEADER = (
    "EPISODE TRANSCRIPT (shared source text for all analysis stages of this episode):"
)
TRANSCRIPT_BLOCK_SEPARATOR = "\n\n=== END TRANSCRIPT ===\n\n"
# Left in the user message where the transcript used to sit so the task stays coherent (the model is
# told the transcript is above) without duplicating the transcript text.
TRANSCRIPT_MOVED_MARKER = (
    "[The full episode transcript is provided at the top of the system message above.]"
)


def cacheable_transcript_prefix(transcript: str) -> str:
    """The byte-stable leading block that carries the transcript into the prompt (RFC-111)."""
    return f"{TRANSCRIPT_BLOCK_HEADER}\n\n{transcript}{TRANSCRIPT_BLOCK_SEPARATOR}"


def relocate_transcript(
    transcript: str,
    system_prompt: str,
    user_prompt: str,
    *,
    enabled: bool,
) -> Tuple[Optional[str], str, str]:
    """Move the transcript from the user prompt to a leading cache block (RFC-111 core).

    Returns ``(transcript_block, new_system, new_user)``. When ``enabled`` and the transcript
    appears in ``user_prompt``, ``transcript_block`` is the byte-stable leading block and
    ``new_user`` is the user prompt with the transcript replaced by a short marker (no duplication).
    Otherwise — disabled, or the transcript not present (a custom prompt) — returns
    ``(None, system_prompt, user_prompt)`` unchanged, so callers fall back to the exact legacy
    layout. Content is only reordered; nothing is dropped.

    The transcript is NORMALISED (``strip``) before it becomes the block, so stages that embed it
    raw (summary) and stages that pre-strip it (GI insights, KG) emit a BYTE-IDENTICAL block and
    therefore share one cached prefix across the episode. Without this, a stray leading/trailing
    newline on the cleaned transcript would silently split the cache per stage (RFC-111 §8).
    """
    if not enabled or not transcript:
        return None, system_prompt, user_prompt
    canonical = transcript.strip()
    # Match the normalised transcript first (GI/KG embed the stripped form); fall back to the raw
    # string (summary embeds it verbatim, which still CONTAINS the stripped form). Either way the
    # block is built from ``canonical`` so it is identical across stages.
    if canonical and canonical in user_prompt:
        target = canonical
    elif transcript in user_prompt:
        target = transcript
    else:
        return None, system_prompt, user_prompt
    new_user = user_prompt.replace(target, TRANSCRIPT_MOVED_MARKER, 1)
    return cacheable_transcript_prefix(canonical), system_prompt, new_user


def openai_style_messages(
    transcript: str,
    system_prompt: str,
    user_prompt: str,
    *,
    enabled: bool,
) -> List[Dict[str, str]]:
    """OpenAI-family application: transcript block leads the SYSTEM message; task stays in user.

    Used by every provider that speaks the OpenAI chat wire format (the OpenAICompatibleProvider
    siblings plus grok/mistral/ollama). Auto-cache providers need no extra API fields — the layout
    alone caches. With caching off or the transcript absent, this is today's exact legacy layout.
    """
    block, sys_prompt, usr_prompt = relocate_transcript(
        transcript, system_prompt, user_prompt, enabled=enabled
    )
    system_content = (block + sys_prompt) if block is not None else sys_prompt
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": usr_prompt},
    ]
