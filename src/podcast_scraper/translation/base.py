"""TranslationProvider protocol definition (RFC-124 / S2.3).

The ``translation`` operation, defined the same way ``summarization`` is: a runtime-checkable
Protocol every provider satisfies, so the pipeline depends on the capability rather than on a
provider name. See :mod:`podcast_scraper.summarization.base` for the sibling it mirrors.

WHY A SEPARATE OPERATION RATHER THAN A SUMMARISER METHOD. Translation is not a summary variant:
it runs at a different point in the pipeline (before summary, so every later stage reads
English), it is served by a different model on a different endpoint, its unit of work is a
turn-bounded span rather than a whole transcript, and its output is an ARTIFACT the corpus keeps
rather than a field on a document. Folding it into ``SummarizationProvider`` would have made
"which model translated this episode" unanswerable from the provenance.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class TranslationProvider(Protocol):
    """Protocol for translation providers.

    A provider translates ONE unit of text. Batching, ordering and per-unit failure accounting
    belong to the caller (S2.4) — a provider that decided an episode's fate from inside a single
    call could not see the other units.
    """

    def initialize(self) -> None:
        """Initialize provider (clients, served-model verification, etc.). Idempotent."""
        ...

    def translate(
        self,
        text: str,
        *,
        source_language: str,
        target_language: str = "en",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Translate one unit of text.

        Args:
            text: The source-language text. Never carries a speaker label — labels bypass the
                translator entirely (D-24) and are re-applied by the caller.
            source_language: BCP-47-ish tag of the source. Must be a language declared in
                ``config/languages.yaml``; an undeclared one is refused rather than guessed.
            target_language: Tag of the target. ``en`` for the whole arc.
            params: Optional overrides (``max_tokens``, ``temperature``).

        Returns:
            ``{"text": str | None, "metadata": {...}}`` — ``text`` is ``None`` when this unit
            could not be translated. The metadata carries the model, the prompt's name and
            SHA256, token usage and the finish reason, so an English artifact can be tied to the
            exact prompt and model that produced it (S2.11).

        Raises:
            RuntimeError: provider misconfigured, or the endpoint serves the wrong model.
        """
        ...

    def cleanup(self) -> None:
        """Release resources. Idempotent."""
        ...
