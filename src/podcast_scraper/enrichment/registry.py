"""Enricher registry — register / get / list_enabled with double-opt-in.

LLM-tier enrichers require BOTH ``manifest.requires_opt_in == True``
AND the ``EnricherSet``'s ``opt_in_flags[id] == True``; missing either
logs a WARNING and the enricher is skipped (never raised — matches
the executor's safety-net contract).

Tests register enrichers directly via pytest fixtures (per chunk-1
lock audit §B7); profile presets register enrichers via the chunk-7
``ProfilePreset`` → ``EnricherSet`` wiring.
"""

from __future__ import annotations

import logging

from podcast_scraper.enrichment.protocol import Enricher, EnricherSet

logger = logging.getLogger(__name__)


# Enrichers that need an injected provider / scorer at construction
# time and so can't auto-register the way the deterministic enrichers
# do. When ``list_enabled`` finds one of these absent from the registry,
# the WARNING includes the wiring hint so an operator running with
# ``--profile cloud_thin`` (which lists them in ``enabled_enrichers``)
# sees the actionable reason rather than a generic "not registered"
# silence. Keep this map narrow — only known optional enrichers.
_PROVIDER_WIRING_HINT: dict[str, str] = {
    "topic_similarity": "an EmbeddingProvider (see scorers/embedding.py)",
    "nli_contradiction": "an NliScorer (see scorers/nli.py)",
    "stance_disagreement": "an NliScorer (see scorers/nli.py)",
}


class EnricherRegistry:
    """A flat registry of ``Enricher`` instances keyed by ``manifest.id``.

    Use a fresh ``EnricherRegistry()`` per test fixture; production
    code reuses a single module-scope instance attached to the
    application state.
    """

    def __init__(self) -> None:
        self._enrichers: dict[str, Enricher] = {}

    def register(self, enricher: Enricher) -> None:
        """Register an enricher under its ``manifest.id``.

        Raises ``ValueError`` if the id is already registered. Use
        ``clear()`` between test runs.
        """
        mid = enricher.manifest.id
        if mid in self._enrichers:
            raise ValueError(f"enricher already registered: {mid!r}")
        self._enrichers[mid] = enricher

    def get(self, enricher_id: str) -> Enricher:
        """Lookup by id (raises ``KeyError`` if absent)."""
        return self._enrichers[enricher_id]

    def all_ids(self) -> list[str]:
        """All registered ids (insertion order)."""
        return list(self._enrichers.keys())

    def list_enabled(self, enricher_set: EnricherSet) -> list[Enricher]:
        """Enrichers enabled by the ``EnricherSet``, filtered for opt-in gating.

        For enrichers with ``manifest.requires_opt_in == True`` (LLM
        tier today), the ``EnricherSet`` must additionally carry
        ``opt_in_flags[id] == True``. Failures log a WARNING and skip
        the enricher (matches the executor's safety contract: never
        raise into the caller).

        Unregistered ids in ``enabled_enrichers`` are also logged +
        skipped (mismatched profile-preset config, typo, etc.).
        """
        out: list[Enricher] = []
        for eid in enricher_set.enabled_enrichers:
            enr = self._enrichers.get(eid)
            if enr is None:
                hint = _PROVIDER_WIRING_HINT.get(eid)
                if hint:
                    logger.warning(
                        "EnricherSet enables %r but it is not registered "
                        "(this enricher requires %s injected at the call "
                        "site — the CLI auto-registers deterministic "
                        "enrichers only; the workflow / API path is the "
                        "canonical entry point that wires providers); "
                        "skipping",
                        eid,
                        hint,
                    )
                else:
                    logger.warning(
                        "EnricherSet enables %r but it is not registered; skipping",
                        eid,
                    )
                continue
            manifest = enr.manifest
            if manifest.requires_opt_in and not enricher_set.has_opt_in(eid):
                logger.warning(
                    "enricher %r (tier=%s) requires opt_in flag but EnricherSet "
                    "did not set it; skipping",
                    eid,
                    manifest.tier.value,
                )
                continue
            out.append(enr)
        return out

    def clear(self) -> None:
        """Clear all registered enrichers (test fixture cleanup)."""
        self._enrichers.clear()
