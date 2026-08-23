"""Derive the enrichment child's ``--profile`` / ``--with-ml`` from the operator YAML.

The pipeline's post-run auto-chain has always derived these from config
(``_maybe_spawn_enrichment_after_pipeline``): the PROFILE decides which enrichers run,
and ``--with-ml`` follows from whether any enabled enricher declares a
``provider_requirement`` (or the operator YAML carries an explicit ``provider:``
block). The operator-triggered surfaces — ``POST /api/jobs/enrichment`` and the MCP
``reenrich`` tool — enqueue the same child but had no equivalent derivation, so a
UI/MCP force re-derive silently warn-skipped ``topic_similarity`` /
``topic_consensus``: the exact two enrichers RFC-118 exists for. One derivation,
every trigger surface.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unreadable/corrupt YAML fails open (child stays loud)
        return {}
    return raw if isinstance(raw, dict) else {}


def derive_enrichment_job_params(operator_yaml: Optional[Path]) -> Tuple[Optional[str], bool]:
    """``(profile, with_ml)`` for an operator-triggered enrichment job.

    Mirrors the pipeline auto-chain's derivation, sourced from the operator YAML:

    * ``profile`` — the YAML's top-level ``profile:`` (or ``enrichment.profile``).
    * ``with_ml`` — True when the YAML declares an explicit per-enricher
      ``provider:`` block, OR the resolved profile's enricher set enables an
      enricher whose manifest declares ``provider_requirement``.

    Fail-open to ``(None, False)`` on any parse/resolution problem — the child then
    behaves exactly as before this derivation existed, and the #1648 loud-failure
    still catches a truly empty configuration.
    """
    if operator_yaml is None or not operator_yaml.is_file():
        return None, False
    doc = _load_yaml(operator_yaml)
    block_raw = doc.get("enrichment")
    block: dict[str, Any] = block_raw if isinstance(block_raw, dict) else {}
    profile_raw = doc.get("profile") or block.get("profile")
    profile = str(profile_raw) if profile_raw else None

    enrichers_raw = block.get("enrichers")
    enrichers_block: dict[str, Any] = enrichers_raw if isinstance(enrichers_raw, dict) else {}
    operator_has_provider = any(
        isinstance(cfg, dict) and isinstance(cfg.get("provider"), dict)
        for cfg in enrichers_block.values()
    )

    profile_needs_ml = False
    if profile and not operator_has_provider:
        try:
            from podcast_scraper.enrichment.eval.admission import known_enricher_manifests
            from podcast_scraper.enrichment.profile_sets import enricher_set_for_profile

            resolved = enricher_set_for_profile(profile)
            manifests = known_enricher_manifests()
            for eid in resolved.enabled_enrichers:
                m = manifests.get(eid)
                if m is not None and m.provider_requirement is not None:
                    profile_needs_ml = True
                    break
        except Exception:  # noqa: BLE001 — unknown profile etc.: fail open, child stays loud
            logger.debug("enrichment spawn params: profile %r did not resolve", profile)
            profile_needs_ml = False

    return profile, bool(operator_has_provider or profile_needs_ml)


__all__ = ["derive_enrichment_job_params"]
