"""``--with-ml`` wiring: construct provider-injected enrichers from the EnricherSet.

Called by the CLI when ``--with-ml`` is set. Walks every enricher in the
active :class:`EnricherSet` whose manifest declares a
:class:`ProviderRequirement`, looks up the provider type from
``per_enricher_config[id].provider.type``, instantiates via the
:mod:`provider_types` registry, constructs the enricher with the
provider plus any other declared knobs, and registers on the
:class:`EnricherRegistry`.

Lookups that fail (missing provider block, unknown type, factory error)
log a WARNING and skip the enricher. The enricher-side registry path
already emits a hinted warning for un-registered enrichers; this layer
adds detail when an attempt was made to wire one but the config was
malformed.

The set of enricher ids this helper knows how to wire is closed today
(``topic_similarity``, ``nli_contradiction``). Future ML enrichers add
themselves to the dispatcher map below.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from podcast_scraper.enrichment.enrichers.nli_contradiction import (
    NliContradictionEnricher,
)
from podcast_scraper.enrichment.enrichers.topic_similarity import TopicSimilarityEnricher
from podcast_scraper.enrichment.protocol import EnricherSet
from podcast_scraper.enrichment.provider_types import get_global_registry
from podcast_scraper.enrichment.registry import EnricherRegistry

logger = logging.getLogger(__name__)


def _build_topic_similarity(provider: Any, knobs: dict[str, Any]) -> TopicSimilarityEnricher:
    top_k_raw = knobs.get("top_k", 10)
    try:
        top_k = int(top_k_raw)
    except (TypeError, ValueError):
        top_k = 10
    if top_k < 1 or top_k > 100:
        top_k = 10
    return TopicSimilarityEnricher(provider=provider, top_k=top_k)


def _build_nli_contradiction(scorer: Any, knobs: dict[str, Any]) -> NliContradictionEnricher:
    threshold_raw = knobs.get("threshold", 0.5)
    try:
        threshold = float(threshold_raw)
    except (TypeError, ValueError):
        threshold = 0.5
    if not 0.0 <= threshold <= 1.0:
        threshold = 0.5
    return NliContradictionEnricher(scorer=scorer, threshold=threshold)


# Each entry: enricher_id → builder taking (provider/scorer, knobs dict)
# and returning the constructed enricher. Future ML enrichers add a row
# here + a class-side __init__ that accepts the same shape.
_ML_ENRICHER_BUILDERS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "topic_similarity": _build_topic_similarity,
    "nli_contradiction": _build_nli_contradiction,
}


def register_ml_enrichers(
    enricher_registry: EnricherRegistry,
    enricher_set: EnricherSet,
) -> None:
    """Register provider-injected enrichers from the active EnricherSet."""
    provider_registry = get_global_registry()
    for eid in enricher_set.enabled_enrichers:
        builder = _ML_ENRICHER_BUILDERS.get(eid)
        if builder is None:
            continue
        if eid in enricher_registry.all_ids():
            # Already registered (e.g. test fixture supplied it).
            continue
        cfg = enricher_set.get_config(eid) or {}
        provider_block = cfg.get("provider")
        if not isinstance(provider_block, dict):
            logger.warning(
                "enrichment: --with-ml: %r has no 'provider' block in "
                "per_enricher_config; skipping. Add 'enrichers.%s.provider: "
                "{type: <name>, ...}' to your YAML to wire it.",
                eid,
                eid,
            )
            continue
        ptype_name = provider_block.get("type")
        if not isinstance(ptype_name, str) or not ptype_name:
            logger.warning(
                "enrichment: --with-ml: %r provider block missing 'type'; "
                "skipping. Known types: %s",
                eid,
                [t.name for t in provider_registry.all_types()],
            )
            continue
        ptype_params = {k: v for k, v in provider_block.items() if k != "type"}
        try:
            provider_instance = provider_registry.instantiate(ptype_name, ptype_params)
        except KeyError:
            logger.warning(
                "enrichment: --with-ml: %r requested provider type %r which "
                "is not registered. Known types: %s",
                eid,
                ptype_name,
                [t.name for t in provider_registry.all_types()],
            )
            continue
        except (ValueError, RuntimeError) as exc:
            logger.warning(
                "enrichment: --with-ml: %r provider %r construction failed: %s",
                eid,
                ptype_name,
                exc,
            )
            continue
        # Strip provider + enabled keys from knob set; pass the rest.
        knobs = {k: v for k, v in cfg.items() if k not in ("provider", "enabled", "opt_in")}
        try:
            enricher_instance = builder(provider_instance, knobs)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "enrichment: --with-ml: %r builder rejected its knobs %r: %s",
                eid,
                knobs,
                exc,
            )
            continue
        enricher_registry.register(enricher_instance)
        logger.info(
            "enrichment: --with-ml: registered %r with provider %r",
            eid,
            ptype_name,
        )


__all__ = ["register_ml_enrichers"]
