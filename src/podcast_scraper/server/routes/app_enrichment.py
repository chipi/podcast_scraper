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
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.app_user_store import User
from podcast_scraper.server.corpus_signals import (
    _SUMMARY_FILES,
    corpus_signals as _corpus_signals,
    filtered_entity_signals,
    parse_envelope as _parse_envelope,
)
from podcast_scraper.server.routes.app_auth import get_current_user, get_optional_user
from podcast_scraper.server.schemas import (
    AppCorpusEnrichmentResponse,
    AppEntitySignalsResponse,
    AppEpisodeEnrichmentResponse,
    AppStorylineDetail,
    AppStorylineMember,
    AppTrendingTopicRow,
    AppTrendingTopicsResponse,
)

router = APIRouter(tags=["app"])

_ENRICHER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


def _envelope_data(path: Path) -> Any | None:
    """The ``data`` payload of an OK envelope, or ``None``."""
    parsed = _parse_envelope(path)
    return parsed.get("data") if parsed is not None else None


@router.get("/episodes/{slug}/enrichment", response_model=AppEpisodeEnrichmentResponse)
def episode_enrichment(
    request: Request, slug: str, _user: User = Depends(get_current_user)
) -> AppEpisodeEnrichmentResponse:
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
def corpus_enrichment(
    request: Request, _user: User = Depends(get_current_user)
) -> AppCorpusEnrichmentResponse:
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


# --------------------------------------------------------------------------- #
# Lean corpus projections (#perf)
#
# The two routes below serve what the Home trending rail and an entity card actually render — a
# top-N slice, and one entity's rows — instead of the whole ~25 MB corpus-enrichment payload the
# client used to download to show ~12 rows / one card. Same on-disk envelopes, same discovery.
# --------------------------------------------------------------------------- #

#: Default ``min_velocity`` — 0.0, i.e. velocity does NOT gate the rail (#1931).
#:
#: This was 1.5 ("heating up", mirroring an old client filter) and it silently un-did the fix
#: above it. Velocity is an acceleration RATIO; ``trend_score`` ranks by volume-with-recency. The
#: two disagree by construction, and the filter ran BEFORE the sort — so the rail kept selecting
#: on the signal the branch had just proved unusable, then ranked whatever survived.
#:
#: Executed against the live 1,066-episode artifact with the post-#1931 shrinkage applied:
#:
#:     min_velocity=1.5  ->  2 of 602 topics pass    (ai and productivity, inflation persistence)
#:     min_velocity=0.0  ->  open source ai models, ai regulation, ai in education,
#:                           ai job displacement, federal reserve policy, us-china ai competition
#:
#: Every topic in that second list scores velocity 0.25-0.33 — the ratio calls the corpus's
#: most-discussed topics "cooling", which is arithmetically true (fewer mentions last month than
#: their own 6-month average) and useless as a discovery filter. Gating on it kept exactly the
#: sparse, spiky topics #1931 set out to demote.
#:
#: The parameter stays, so a caller that genuinely wants accelerating-only topics can ask for it.
#: ``min_total`` remains the real quality gate: a topic still needs 3 mentions to appear at all.
_RISING_DEFAULT = 0.0
_MIN_TOTAL_DEFAULT = 3  # ignore topics too sparse to read anything into
_TRENDING_LIMIT_DEFAULT = 12  # rows the rail shows


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    """``float`` or ``None`` — distinguishes "absent" from "zero" (see the call site)."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    return float(value)


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@router.get("/corpus/trending-topics", response_model=AppTrendingTopicsResponse)
def corpus_trending_topics(
    request: Request,
    limit: int = Query(default=_TRENDING_LIMIT_DEFAULT, ge=1, le=100),
    min_velocity: float = Query(
        default=_RISING_DEFAULT,
        ge=0.0,
        description=(
            "Optional acceleration filter on velocity_last_over_6mo. Defaults to 0.0 (off): "
            "velocity is a ratio and calls the most-discussed topics 'cooling' (#1931). Raise it "
            "only to ask specifically for accelerating topics."
        ),
    ),
    min_total: int = Query(default=_MIN_TOTAL_DEFAULT, ge=0),
    user: User | None = Depends(get_optional_user),
) -> AppTrendingTopicsResponse:
    """Top-N rising topics for the Home trending rail — a lean projection of ``temporal_velocity``.

    The rail rendered ~12 rows out of a ~25 MB corpus-velocity artifact; here we filter (velocity ≥
    ``min_velocity`` and total ≥ ``min_total``), sort by velocity desc, trim to ``limit``, and drop
    the per-topic weekly series the client never reads. ``has_velocity_data`` separates "no
    enricher" (render nothing) from "ran, nothing rising" (show the quiet state).
    """
    if user is None:
        # Anon teaser (RFC-120): lock to the default top-N slice. Clamp the count AND ignore the
        # filter params — otherwise sweeping min_velocity/min_total enumerates different 8-topic
        # slices past the clamp, leaking the corpus label-set to the public (Fable-5 review M1).
        limit = min(limit, 8)
        min_velocity = _RISING_DEFAULT
        min_total = _MIN_TOTAL_DEFAULT
    root = corpus_root_or_503(request)
    signals = _corpus_signals(root, {"temporal_velocity", "topic_theme_clusters"})

    tv = signals.get("temporal_velocity")
    tv = tv if isinstance(tv, dict) else {}
    rows_any = tv.get("topics")
    rows = [r for r in rows_any if isinstance(r, dict)] if isinstance(rows_any, list) else []

    # #1931 — rank on ``trend_score``, not on ``velocity_last_over_6mo``.
    #
    # Velocity is an acceleration RATIO, and on a sparse corpus a ratio cannot separate "discussed
    # once, recently" from "discussed all year": a single recent mention scores the maximum while
    # every sustained topic sits at ~1.0. Measured on the 1,066-episode corpus, the old ordering
    # put SEVEN single-mention topics in its top ten — ``fiscal dominance`` and ``gdp measurement``
    # above ``monetary policy`` (16 mentions). Shrinking the ratio (#1931) made the value honest
    # but could not reorder it; no prior from 3 to 10 changed that top ten.
    #
    # ``trend_score`` asks the question a discovery rail actually wants — "what is being talked
    # about, lately, repeatedly" — as recency-decayed volume scaled by weekly spread. Same corpus,
    # new top: open source ai models, ai regulation, ai in education, federal reserve policy,
    # us-china ai competition. Zero single-mention topics.
    #
    # ``min_velocity`` still filters (a caller can ask for accelerating topics) but no longer
    # ORDERS. Rows lacking ``trend_score`` — an artifact written before #1931 — fall back to
    # velocity so an un-re-enriched corpus still renders.
    rising = [
        r
        for r in rows
        if _as_float(r.get("velocity_last_over_6mo")) >= min_velocity
        and _as_int(r.get("total")) >= min_total
    ]

    def _rank(r: dict[str, Any]) -> float:
        score = r.get("trend_score")
        if isinstance(score, (int, float)):
            return float(score)
        return _as_float(r.get("velocity_last_over_6mo"))

    rising.sort(key=_rank, reverse=True)
    top = rising[:limit]

    window_any = tv.get("window_months")
    window_months = [str(m) for m in window_any] if isinstance(window_any, list) else []

    def _monthly(r: dict[str, Any]) -> dict[str, int]:
        mc = r.get("monthly_counts")
        if not isinstance(mc, dict):
            return {}
        return {str(k): _as_int(v) for k, v in mc.items()}

    ttc = signals.get("topic_theme_clusters")
    ttc = ttc if isinstance(ttc, dict) else {}
    clusters_any = ttc.get("clusters")
    clusters = [
        AppStorylineDetail(
            graph_compound_parent_id=(c.get("graph_compound_parent_id") or None),
            canonical_label=(c.get("canonical_label") or None),
            members=[
                AppStorylineMember(topic_id=str(m.get("topic_id")))
                for m in (c.get("members") or [])
                if isinstance(m, dict) and m.get("topic_id")
            ],
        )
        for c in (clusters_any if isinstance(clusters_any, list) else [])
        if isinstance(c, dict)
    ]

    return AppTrendingTopicsResponse(
        has_velocity_data=bool(rows),
        window_months=window_months,
        topics=[
            AppTrendingTopicRow(
                topic_id=str(r.get("topic_id") or ""),
                topic_label=(str(r["topic_label"]) if r.get("topic_label") else None),
                velocity_last_over_6mo=_as_float(r.get("velocity_last_over_6mo")),
                # NOT _as_float: that collapses a MISSING trend_score to 0.0, which is a real
                # value, so the client could never tell "pre-#1931 artifact" from "scored zero"
                # and its documented fallback to velocity was unreachable.
                trend_score=_as_optional_float(r.get("trend_score")),
                total=_as_int(r.get("total")),
                monthly_counts=_monthly(r),
            )
            for r in top
        ],
        storylines=clusters,
    )


@router.get("/corpus/entity-signals", response_model=AppEntitySignalsResponse)
def corpus_entity_signals(
    request: Request,
    kind: str = Query(..., pattern="^(person|topic)$"),
    id: str = Query(..., min_length=1),
    _user: User = Depends(get_current_user),
) -> AppEntitySignalsResponse:
    """Corpus enrichment signals filtered to ONE person/topic, for its entity card.

    Every list in ``/corpus/enrichment`` is pre-filtered to the rows that touch the focused entity,
    so the card fetches a few KB instead of the whole ~25 MB corpus payload. The client keeps its
    own a/b orientation over this subset.
    """
    return AppEntitySignalsResponse(
        signals=filtered_entity_signals(corpus_root_or_503(request), kind, id)
    )
