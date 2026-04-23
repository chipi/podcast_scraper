"""GET /api/corpus/persons/top — grounded insight counts per Person (dashboard)."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Any, DefaultDict

from fastapi import APIRouter, Query, Request

from podcast_scraper.gi.edge_normalization import normalize_gil_edge_type
from podcast_scraper.server.cil_queries import (
    _insight_ids_supported_by_quotes,
    _quote_ids_spoken_by_person,
    canonical_cil_entity_id,
)
from podcast_scraper.server.corpus_catalog import build_catalog_rows
from podcast_scraper.server.pathutil import resolved_corpus_root_str
from podcast_scraper.server.routes.corpus_library import _resolve_corpus_root
from podcast_scraper.server.schemas import CorpusTopPersonsResponse, TopPersonItem
from podcast_scraper.utils.path_validation import safe_relpath_under_corpus_root

router = APIRouter(tags=["corpus"])

_MAX_GI_BYTES = 50 * 1024 * 1024


def _person_display_name(gi: dict[str, Any], person_id: str) -> str:
    for n in gi.get("nodes") or []:
        if not isinstance(n, dict) or n.get("type") != "Person":
            continue
        rid = n.get("id")
        if rid is None:
            continue
        if canonical_cil_entity_id(str(rid)) != person_id:
            continue
        props = n.get("properties")
        if isinstance(props, dict):
            raw = props.get("name") or props.get("display_name")
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
        return person_id
    return person_id


def _topics_for_insights(gi: dict[str, Any], insight_ids: set[str]) -> Counter[str]:
    """Sum ABOUT-edge confidences per topic across the given insights (#664).

    Confidence weighting means topics with strong semantic links rank above
    topics that are only tangentially connected. Edges without a
    ``properties.confidence`` field (legacy cross-product, pre-#664) count as
    1.0 so older gi.json files behave as before.
    """
    topics: Counter[str] = Counter()
    for e in gi.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if normalize_gil_edge_type(e.get("type")) != "ABOUT":
            continue
        if str(e.get("from")) not in insight_ids:
            continue
        to_raw = e.get("to")
        if to_raw is None:
            continue
        tid = canonical_cil_entity_id(str(to_raw))
        if not tid:
            continue
        props = e.get("properties") if isinstance(e.get("properties"), dict) else {}
        conf_raw = props.get("confidence") if props else None
        try:
            conf = float(conf_raw) if conf_raw is not None else 1.0
        except (TypeError, ValueError):
            conf = 1.0
        # Counter is typed for int values; scale float confidence to preserve
        # 4-decimal ranking precision while keeping the Counter interface
        # (``most_common`` ordering is what we need downstream).
        topics[tid] += int(round(conf * 10000))
    return topics


def _load_gi_dict(path_str: str) -> dict[str, Any] | None:
    try:
        sz = os.path.getsize(path_str)
        if sz > _MAX_GI_BYTES:
            return None
        with open(path_str, encoding="utf-8") as fh:
            raw_any: Any = json.load(fh)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return raw_any if isinstance(raw_any, dict) else None


@router.get("/corpus/persons/top", response_model=CorpusTopPersonsResponse)
async def corpus_persons_top(
    request: Request,
    path: str | None = Query(default=None, description="Corpus root."),
    limit: int = Query(default=5, ge=1, le=50, description="Max persons to return."),
) -> CorpusTopPersonsResponse:
    """Scan catalog GI files; rank Person nodes by grounded (quote-backed) insight counts."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    root_safe = resolved_corpus_root_str(root, anchor)
    root_prefix = os.path.normpath(str(root_safe)) + os.sep

    rows = build_catalog_rows(root)
    all_person_ids: set[str] = set()
    episode_by_person: DefaultDict[str, set[str]] = defaultdict(set)
    insight_keys_by_person: DefaultDict[str, set[str]] = defaultdict(set)
    topic_hits: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    display_name_by_person: dict[str, str] = {}

    for row in rows:
        if not row.has_gi:
            continue
        gi_rel = (row.gi_relative_path or "").strip()
        if not gi_rel:
            continue
        gi_abs = safe_relpath_under_corpus_root(root, gi_rel)
        if not gi_abs or not gi_abs.startswith(root_prefix) or not os.path.isfile(gi_abs):
            continue
        gi = _load_gi_dict(gi_abs)
        if gi is None:
            continue
        eid = (row.episode_id or "").strip() or os.path.basename(gi_rel).replace(".gi.json", "")

        persons_here: set[str] = set()
        for n in gi.get("nodes") or []:
            if not isinstance(n, dict):
                continue
            if n.get("type") != "Person":
                continue
            raw_id = n.get("id")
            if raw_id is None:
                continue
            pid = canonical_cil_entity_id(str(raw_id))
            if not pid:
                continue
            persons_here.add(pid)
            all_person_ids.add(pid)
            if pid not in display_name_by_person:
                display_name_by_person[pid] = _person_display_name(gi, pid)

        for pid in persons_here:
            spoken = _quote_ids_spoken_by_person(gi, pid)
            supported = _insight_ids_supported_by_quotes(gi, spoken)
            if not supported:
                continue
            episode_by_person[pid].add(eid)
            for iid in supported:
                insight_keys_by_person[pid].add(f"{eid}\0{iid}")
            for tid, c in _topics_for_insights(gi, supported).items():
                topic_hits[pid][tid] += c

    ranked: list[tuple[str, int]] = []
    for pid, keys in insight_keys_by_person.items():
        ranked.append((pid, len(keys)))
    ranked.sort(key=lambda x: (-x[1], x[0]))

    persons_out: list[TopPersonItem] = []
    for pid, icount in ranked[:limit]:
        top_t = [t for t, _ in topic_hits[pid].most_common(3)]
        persons_out.append(
            TopPersonItem(
                person_id=pid,
                display_name=display_name_by_person.get(pid, pid),
                episode_count=len(episode_by_person[pid]),
                insight_count=icount,
                top_topics=top_t,
            )
        )

    return CorpusTopPersonsResponse(
        path=str(root_safe),
        persons=persons_out,
        total_persons=len(all_person_ids),
    )
