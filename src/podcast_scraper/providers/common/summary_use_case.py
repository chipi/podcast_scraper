"""A summary other than the episode's, made by the same ``summarize`` (RFC-127).

The episode summary is one use of a provider's ``summarize``: its prompts, its parameters, its cost
stage. A board summary is another — "just another use case" (operator 2026-10-13) — so it goes
through the very same method, and with it the same retries, call metrics, cost recording, response
guardrails and prompt metadata every provider already has in production. Only what differs is
passed in:

- the system and user prompt names, LOGICAL (``summarization/board_user_v1``): each provider
  resolves them under its own prompt folder (``<ns>/summarization/board_user_v1.j2``), falling
  back to ``shared/summarization/board_user_v1.j2`` — the store's rule for every summarization
  prompt;
- the template parameters (the episode prompts render a transcript; this one renders its material);
- the ``llm_cost`` stage, so the call is never counted as episode summarization;
- the reply budget.

Passed as ``summarize(text="", params={"use_case": SummaryUseCase(...)})``. Absent, ``summarize``
is exactly what it was.

A seq2seq summarizer (BART/LED, the ``transformers`` provider) follows no instructions: it
summarises a document. For it the use case names a ``document_prompt`` — a template that writes
the material as plain prose — and that document goes through its own map/reduce in place of the
transcript. The chat providers render ``system_prompt`` + ``user_prompt``; the seq2seq one renders
``document_prompt``. :func:`use_case_input_kind` tells which a provider renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Mapping, Optional, Tuple

InputKind = Literal["chat", "document"]


@dataclass(frozen=True)
class SummaryUseCase:
    """What a non-episode summary changes in ``summarize``, and nothing else."""

    system_prompt: str
    user_prompt: str
    stage: str
    max_length: int
    params: Mapping[str, Any] = field(default_factory=dict)
    #: For seq2seq providers, which take a document and no instructions; None when unsupported.
    document_prompt: Optional[str] = None


def use_case_from_params(params: Optional[Dict[str, Any]]) -> Optional[SummaryUseCase]:
    """The use case ``summarize`` was called with, or None for the episode summary."""
    uc = (params or {}).get("use_case")
    return uc if isinstance(uc, SummaryUseCase) else None


def reply_budget(
    use_case: Optional[SummaryUseCase],
    default: int,
    clamp: Callable[[int], int] = lambda n: n,
) -> int:
    """The reply budget: the episode's ``default``, or the use case's own through the provider's
    ``clamp``. A function, not a branch in ``summarize``, so the method stays as it is."""
    return default if use_case is None else clamp(use_case.max_length)


def use_case_stage(use_case: Optional[SummaryUseCase]) -> Optional[str]:
    """The ``llm_cost`` stage: the use case's, or None for the episode summary's default."""
    return None if use_case is None else use_case.stage


def prompt_name(namespace: str, logical: str) -> str:
    """``summarization/board_user_v1`` under ``openai`` → ``openai/summarization/board_user_v1``;
    the store falls back to ``shared/summarization/board_user_v1`` when the provider has none."""
    return f"{namespace}/{logical}"


def build_use_case_prompts(namespace: str, use_case: SummaryUseCase) -> Tuple[str, str, str, str]:
    """``(system_prompt, user_prompt, system_name, user_name)`` for a use case, rendered by the
    prompt store as every other prompt is."""
    from ...prompts.store import render_prompt

    system_name = prompt_name(namespace, use_case.system_prompt)
    user_name = prompt_name(namespace, use_case.user_prompt)
    return (
        render_prompt(system_name),
        render_prompt(user_name, **dict(use_case.params)),
        system_name,
        user_name,
    )


def build_use_case_document(namespace: str, use_case: SummaryUseCase) -> Tuple[str, str]:
    """``(document, document_name)`` for a seq2seq provider: the material written as prose by the
    use case's ``document_prompt``. ValueError when the use case has none."""
    from ...prompts.store import render_prompt

    if not use_case.document_prompt:
        raise ValueError(f"use case {use_case.stage!r} has no document_prompt for a seq2seq model")
    name = prompt_name(namespace, use_case.document_prompt)
    return render_prompt(name, **dict(use_case.params)), name


def use_case_input_kind(provider: Any) -> Optional[InputKind]:
    """What a provider renders for a use case: ``chat`` (system + user prompts), ``document`` (a
    seq2seq model), or None when it cannot run one at all."""
    if hasattr(provider, "_build_summarization_prompts"):
        return "chat"
    if hasattr(provider, "_build_use_case_document"):
        return "document"
    return None


def use_case_prompt_metadata(
    namespace: str, use_case: Optional[SummaryUseCase], kind: InputKind = "chat"
) -> Optional[Dict[str, Any]]:
    """The ``prompts`` block of ``summarize``'s metadata for a use case: each template the provider
    rendered, with name, file and hash, as for the episode prompts. The material itself is not
    copied into it, only its size. None for the episode summary, so a caller writes
    ``use_case_prompt_metadata(...) or <its own>``.
    """
    from ...prompts.store import get_prompt_metadata

    if use_case is None:
        return None
    sizes = {k: len(v) for k, v in use_case.params.items() if isinstance(v, (list, tuple))}
    if kind == "document":
        name = prompt_name(namespace, str(use_case.document_prompt))
        return {"document": get_prompt_metadata(name, params=sizes)}
    return {
        "system": get_prompt_metadata(prompt_name(namespace, use_case.system_prompt)),
        "user": get_prompt_metadata(prompt_name(namespace, use_case.user_prompt), params=sizes),
    }
