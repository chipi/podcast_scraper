"""Consumer enrichment read surface (P3 Consolidation, #1121 / RFC-088 envelopes).

The operator routes (``/api/enrichment/*``) and the corpus-scope reader
(``/api/corpus/enrichments*``) are ops/global; these are the **consumer projection** under
``/api/app/*``, addressed by the consumer episode *slug* and shaped for the player + recall.

Read-only over the on-disk envelopes the executor produced (ADR-104 boundary — never recompute).
Each envelope is ``{enricher_id, schema_version, status, data, …}``; we surface only the ``data`` of
enrichers that ran OK, keyed by ``enricher_id``. Envelope ids are **discovered** from disk (a glob),
so the surface stays correct as the enricher set evolves — no hardcoded id list.
"""

from __future__ import annotations

import glob as globmod
import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.schemas import (
    AppCorpusEnrichmentResponse,
    AppEpisodeEnrichmentResponse,
)

router = APIRouter(tags=["app"])

_ENRICHER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")
_SUMMARY_FILES = {"run_summary.json"}


def _parse_envelope(path: Path) -> dict[str, Any] | None:
    """Parsed envelope dict for an OK enricher, or ``None`` (absent / unparsable / not OK)."""
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("status") not in (None, "ok"):  # tolerate envelopes that omit status
        return None
    return parsed


def _envelope_data(path: Path) -> Any | None:
    """The ``data`` payload of an OK envelope, or ``None``."""
    parsed = _parse_envelope(path)
    return parsed.get("data") if parsed is not None else None


@router.get("/episodes/{slug}/enrichment", response_model=AppEpisodeEnrichmentResponse)
async def episode_enrichment(request: Request, slug: str) -> AppEpisodeEnrichmentResponse:
    """Per-episode enrichment signals for the episode the user is viewing (404 if no such slug).

    Episode-scope envelopes live at ``<metadata_dir>/enrichments/<stem>.<enricher_id>.json``.
    """
    root = corpus_root_or_503(request)
    row = resolve_slug(root, slug)
    if row is None:
        raise HTTPException(status_code=404, detail="Unknown episode slug.")
    meta_path = root / row.metadata_relative_path
    enrich_dir = meta_path.parent / "enrichments"
    signals: dict[str, Any] = {}
    if enrich_dir.is_dir() and meta_path.name.endswith(".metadata.json"):
        stem = meta_path.name[: -len(".metadata.json")]
        for path in sorted(
            Path(p) for p in globmod.glob(globmod.escape(str(enrich_dir / stem)) + ".*.json")
        ):
            enricher_id = path.name[len(stem) + 1 : -len(".json")]
            if not _ENRICHER_ID_PATTERN.match(enricher_id):
                continue
            data = _envelope_data(path)
            if data is not None:
                signals[enricher_id] = data
    return AppEpisodeEnrichmentResponse(slug=slug, signals=signals)


@router.get("/corpus/enrichment", response_model=AppCorpusEnrichmentResponse)
async def corpus_enrichment(request: Request) -> AppCorpusEnrichmentResponse:
    """Corpus-scope enrichment signals (temporal velocity, topic similarity, …) for the consumer."""
    root = corpus_root_or_503(request)
    enrich_dir = root / "enrichments"
    signals: dict[str, Any] = {}
    if enrich_dir.is_dir():
        for path in sorted(enrich_dir.glob("*.json")):
            if path.name in _SUMMARY_FILES:
                continue
            parsed = _parse_envelope(path)
            if parsed is None or parsed.get("data") is None:
                continue
            signals[str(parsed.get("enricher_id") or path.stem)] = parsed["data"]
    return AppCorpusEnrichmentResponse(signals=signals)
