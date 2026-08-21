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

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper import perf_cache
from podcast_scraper.server.app_corpus_access import corpus_root_or_503
from podcast_scraper.server.app_slugs import resolve_slug
from podcast_scraper.server.schemas import (
    AppCorpusEnrichmentResponse,
    AppEntitySignalsResponse,
    AppEpisodeEnrichmentResponse,
    AppThemeCluster,
    AppThemeClusterMember,
    AppTrendingTopicRow,
    AppTrendingTopicsResponse,
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
def episode_enrichment(request: Request, slug: str) -> AppEpisodeEnrichmentResponse:
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
def corpus_enrichment(request: Request) -> AppCorpusEnrichmentResponse:
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

_RISING_DEFAULT = 1.5  # velocity_last_over_6mo bar for "heating up" (mirrors the old client filter)
_MIN_TOTAL_DEFAULT = 3  # ignore topics too sparse to read anything into
_TRENDING_LIMIT_DEFAULT = 12  # rows the rail shows

_PERSON_ENRICHERS = {"grounding_rate", "guest_coappearance", "topic_consensus"}
_TOPIC_ENRICHERS = {"temporal_velocity", "topic_similarity", "topic_cooccurrence_corpus"}
_ID_PREFIX_RE = re.compile(r"^(?:g:|k:|kg:)+")


_ENRICHER_SIGNALS_NS = "app_corpus_signals"


def _read_corpus_signals(root: Path, wanted: set[str]) -> dict[str, Any]:
    """``{enricher_id: envelope data}`` for the *wanted* corpus enrichers that ran OK.

    Filename-first: the enrichment filename **is** the enricher id for every current enricher, so we
    skip the glob's non-wanted files *before* parsing them. That is the fix for the old behaviour of
    JSON-parsing the multi-MB ``topic_cooccurrence_corpus.json`` on a trending-topics request
    just to discard it — now only the wanted files are read. The envelope's ``enricher_id`` keys
    the result, so discovery semantics match :func:`corpus_enrichment`.
    """
    enrich_dir = root / "enrichments"
    out: dict[str, Any] = {}
    if not enrich_dir.is_dir():
        return out
    for path in sorted(enrich_dir.glob("*.json")):
        if path.name in _SUMMARY_FILES:
            continue
        if path.stem not in wanted:  # filename-first — do not parse files we would only discard
            continue
        parsed = _parse_envelope(path)
        if parsed is None or parsed.get("data") is None:
            continue
        enricher_id = str(parsed.get("enricher_id") or path.stem)
        if enricher_id in wanted:
            out[enricher_id] = parsed["data"]
    return out


def _corpus_signals(root: Path, wanted: set[str]) -> dict[str, Any]:
    """:func:`_read_corpus_signals`, cached by corpus mtime (bumps on ingest).

    Keyed by ``(root, wanted)`` so trending-topics and entity-signals keep separate warmed subsets;
    the parsed envelopes are held once per ingest instead of re-read+re-parsed on every request.
    """
    # root is the platform corpus (corpus_root_or_503) or a _resolve_corpus-validated ?path.
    # codeql[py/path-injection] -- root validated by corpus_root_or_503 / _resolve_corpus (Type 1).
    resolved_root = str(Path(root).resolve())
    key = f"{resolved_root}::{','.join(sorted(wanted))}"
    signals: dict[str, Any] = perf_cache.get_or_compute(
        _ENRICHER_SIGNALS_NS,
        key,
        perf_cache.corpus_mtime(root),
        lambda: _read_corpus_signals(root, wanted),
    )
    return signals


def _norm_entity_id(value: Any) -> str:
    """Drop graph id prefixes (``g:`` / ``k:`` / ``kg:``) so ids compare like the client norm."""
    return _ID_PREFIX_RE.sub("", str(value or ""))


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@router.get("/corpus/trending-topics", response_model=AppTrendingTopicsResponse)
def corpus_trending_topics(
    request: Request,
    limit: int = Query(default=_TRENDING_LIMIT_DEFAULT, ge=1, le=100),
    min_velocity: float = Query(default=_RISING_DEFAULT, ge=0.0),
    min_total: int = Query(default=_MIN_TOTAL_DEFAULT, ge=0),
) -> AppTrendingTopicsResponse:
    """Top-N rising topics for the Home trending rail — a lean projection of ``temporal_velocity``.

    The rail rendered ~12 rows out of a ~25 MB corpus-velocity artifact; here we filter (velocity ≥
    ``min_velocity`` and total ≥ ``min_total``), sort by velocity desc, trim to ``limit``, and drop
    the per-topic weekly series the client never reads. ``has_velocity_data`` separates "no
    enricher" (render nothing) from "ran, nothing rising" (show the quiet state).
    """
    root = corpus_root_or_503(request)
    signals = _corpus_signals(root, {"temporal_velocity", "topic_theme_clusters"})

    tv = signals.get("temporal_velocity")
    tv = tv if isinstance(tv, dict) else {}
    rows_any = tv.get("topics")
    rows = [r for r in rows_any if isinstance(r, dict)] if isinstance(rows_any, list) else []

    rising = [
        r
        for r in rows
        if _as_float(r.get("velocity_last_over_6mo")) >= min_velocity
        and _as_int(r.get("total")) >= min_total
    ]
    rising.sort(key=lambda r: _as_float(r.get("velocity_last_over_6mo")), reverse=True)
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
        AppThemeCluster(
            graph_compound_parent_id=(c.get("graph_compound_parent_id") or None),
            canonical_label=(c.get("canonical_label") or None),
            members=[
                AppThemeClusterMember(topic_id=str(m.get("topic_id")))
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
                total=_as_int(r.get("total")),
                monthly_counts=_monthly(r),
            )
            for r in top
        ],
        theme_clusters=clusters,
    )


def filtered_entity_signals(root: Path, kind: str, id: str) -> dict[str, Any]:
    """Corpus enrichment signals filtered to ONE person/topic (the entity-card projection).

    Every corpus-scope enricher list is pre-filtered to the rows that touch the focused entity, so a
    caller reads a few KB instead of the whole (up to ~25 MB) corpus payload. Shared by the consumer
    ``/api/app/corpus/entity-signals`` (single platform corpus) and the operator
    ``/api/corpus/entity-signals`` (``?path=``-scoped viewer) — same filter, different root.
    """
    self_id = _norm_entity_id(id)
    raw = _corpus_signals(root, _PERSON_ENRICHERS if kind == "person" else _TOPIC_ENRICHERS)
    out: dict[str, Any] = {}

    def _hit(*ids: Any) -> bool:
        return any(_norm_entity_id(i) == self_id for i in ids)

    def _filtered(enricher: str, list_key: str, keep: Any) -> None:
        env = raw.get(enricher)
        if not isinstance(env, dict):
            return
        items_any = env.get(list_key)
        if not isinstance(items_any, list):
            return
        kept = [it for it in items_any if isinstance(it, dict) and keep(it)]
        if kept:
            out[enricher] = {list_key: kept}

    if kind == "person":
        _filtered("grounding_rate", "persons", lambda r: _hit(r.get("person_id")))
        _filtered(
            "guest_coappearance",
            "pairs",
            lambda r: _hit(r.get("person_a_id"), r.get("person_b_id")),
        )
        _filtered(
            "topic_consensus",
            "consensus",
            lambda r: _hit(r.get("person_a_id"), r.get("person_b_id")),
        )
    else:
        _filtered("temporal_velocity", "topics", lambda r: _hit(r.get("topic_id")))
        _filtered("topic_similarity", "topics", lambda r: _hit(r.get("topic_id")))
        _filtered(
            "topic_cooccurrence_corpus",
            "pairs",
            lambda r: _hit(r.get("topic_a_id"), r.get("topic_b_id")),
        )

    return out


@router.get("/corpus/entity-signals", response_model=AppEntitySignalsResponse)
def corpus_entity_signals(
    request: Request,
    kind: str = Query(..., pattern="^(person|topic)$"),
    id: str = Query(..., min_length=1),
) -> AppEntitySignalsResponse:
    """Corpus enrichment signals filtered to ONE person/topic, for its entity card.

    Every list in ``/corpus/enrichment`` is pre-filtered to the rows that touch the focused entity,
    so the card fetches a few KB instead of the whole ~25 MB corpus payload. The client keeps its
    own a/b orientation over this subset.
    """
    return AppEntitySignalsResponse(
        signals=filtered_entity_signals(corpus_root_or_503(request), kind, id)
    )
