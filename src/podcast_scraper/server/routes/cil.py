"""RFC-072 cross-layer CIL query API (GitHub #527)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.server import cil_queries
from podcast_scraper.server.pathutil import resolve_corpus_path_param, resolved_corpus_root_str
from podcast_scraper.server.schemas import (
    CilArcEpisodeBlock,
    CilIdListResponse,
    CilPersonProfileInsightRow,
    CilPersonProfileQuoteRow,
    CilPersonProfileResponse,
    CilPositionArcResponse,
    CilTopicTimelineMergedResponse,
    CilTopicTimelineMergeRequest,
    CilTopicTimelineResponse,
)

router = APIRouter(tags=["cil"])


def _resolve_corpus_root(path: str | None, fallback: Path | None) -> Path | None:
    if path is not None and str(path).strip():
        return resolve_corpus_path_param(path, fallback)
    return fallback


def _parse_insight_types(
    raw: str | None,
    *,
    default: tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    """Comma-separated types, or ``all`` / ``*`` / empty → no filter (``None``)."""
    if raw is None:
        return default
    s = raw.strip().lower()
    if s in ("", "all", "*"):
        return None
    parts = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    return parts if parts else default


def _require_root_and_anchor(request: Request, path: str | None) -> tuple[str, str]:
    """Return ``(root_safe, anchor_safe)`` — sanitised strings.

    ``root_safe`` comes from ``resolved_corpus_root_str`` which applies
    ``os.path.normpath`` + ``str.startswith`` — the sanitiser that CodeQL
    recognises for ``py/path-injection``.  ``anchor_safe`` is the untainted
    server ``output_dir`` (normalised).
    """
    anchor: Path | None = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    if root is None or anchor is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "path query parameter is required when the server has no default output_dir "
                "(set PODCAST_SERVE_OUTPUT_DIR or pass output_dir to create_app)."
            ),
        )
    anchor_safe = os.path.normpath(str(anchor.resolve()))
    root_safe = resolved_corpus_root_str(root, anchor)
    return root_safe, anchor_safe


@router.get("/persons/{person_id}/positions", response_model=CilPositionArcResponse)
async def person_positions(
    request: Request,
    person_id: str,
    topic: str = Query(..., description="Canonical topic id (e.g. topic:climate)."),
    path: str | None = Query(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    ),
    insight_types: str | None = Query(
        default=None,
        description="Comma-separated insight_type values; omit for default ``claim`` only; "
        "``all`` or ``*`` for no filter.",
    ),
) -> CilPositionArcResponse:
    """Position arc — insights by person + topic across episodes (RFC-072 Pattern A)."""
    root_safe, anchor_safe = _require_root_and_anchor(request, path)
    types = _parse_insight_types(insight_types, default=("claim",))
    raw = cil_queries.position_arc(root_safe, anchor_safe, person_id, topic, insight_types=types)
    episodes = [
        CilArcEpisodeBlock(
            episode_id=str(b["episode_id"]),
            publish_date=b.get("publish_date"),
            insights=list(b.get("insights") or []),
        )
        for b in raw
    ]
    return CilPositionArcResponse(
        path=root_safe,
        person_id=person_id.strip(),
        topic_id=topic.strip(),
        episodes=episodes,
    )


@router.get("/persons/{person_id}/brief", response_model=CilPersonProfileResponse)
async def person_profile(
    request: Request,
    person_id: str,
    path: str | None = Query(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    ),
) -> CilPersonProfileResponse:
    """Person profile — insights grouped by topic (RFC-072 Pattern B)."""
    root_safe, anchor_safe = _require_root_and_anchor(request, path)
    raw = cil_queries.person_profile(root_safe, anchor_safe, person_id)
    topics_raw = raw.get("topics") or {}
    topics_out: dict[str, list[CilPersonProfileInsightRow]] = {}
    if isinstance(topics_raw, dict):
        for tid, rows in topics_raw.items():
            if not isinstance(rows, list):
                continue
            out_rows: list[CilPersonProfileInsightRow] = []
            for row in rows:
                if isinstance(row, dict):
                    out_rows.append(CilPersonProfileInsightRow.model_validate(row))
            topics_out[str(tid)] = out_rows
    quotes_raw = raw.get("quotes") or []
    quotes: list[CilPersonProfileQuoteRow] = []
    if isinstance(quotes_raw, list):
        for row in quotes_raw:
            if isinstance(row, dict):
                quotes.append(CilPersonProfileQuoteRow.model_validate(row))
    return CilPersonProfileResponse(
        path=root_safe,
        person_id=str(raw.get("person_id") or person_id.strip()),
        topics=topics_out,
        quotes=quotes,
    )


@router.post("/topics/timeline", response_model=CilTopicTimelineMergedResponse)
async def topic_timeline_merge(
    request: Request,
    body: CilTopicTimelineMergeRequest,
) -> CilTopicTimelineMergedResponse:
    """Merged topic timeline — one corpus scan for multiple topic ids (cluster scope)."""
    root_safe, anchor_safe = _require_root_and_anchor(request, body.path)
    types = _parse_insight_types(body.insight_types, default=None)
    seen: set[str] = set()
    ordered_ids: list[str] = []
    for raw_tid in body.topic_ids:
        tid = cil_queries.canonical_cil_entity_id(str(raw_tid))
        if tid in seen:
            continue
        seen.add(tid)
        ordered_ids.append(tid)
    raw = cil_queries.topic_timeline_merged(
        root_safe, anchor_safe, ordered_ids, insight_types=types
    )
    episodes = [
        CilArcEpisodeBlock(
            episode_id=str(b["episode_id"]),
            publish_date=b.get("publish_date"),
            episode_title=b.get("episode_title"),
            feed_title=b.get("feed_title"),
            episode_number=b.get("episode_number"),
            episode_image_url=b.get("episode_image_url"),
            episode_image_local_relpath=b.get("episode_image_local_relpath"),
            feed_image_url=b.get("feed_image_url"),
            feed_image_local_relpath=b.get("feed_image_local_relpath"),
            insights=list(b.get("insights") or []),
        )
        for b in raw
    ]
    return CilTopicTimelineMergedResponse(
        path=root_safe,
        topic_ids=ordered_ids,
        episodes=episodes,
    )


@router.get("/topics/{topic_id}/timeline", response_model=CilTopicTimelineResponse)
async def topic_timeline(
    request: Request,
    topic_id: str,
    path: str | None = Query(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    ),
    insight_types: str | None = Query(
        default=None,
        description="Comma-separated insight_type filter; omit for all types; "
        "``all`` or ``*`` for all.",
    ),
) -> CilTopicTimelineResponse:
    """Topic evolution across episodes (RFC-072 Pattern C)."""
    root_safe, anchor_safe = _require_root_and_anchor(request, path)
    types = _parse_insight_types(insight_types, default=None)
    tid = cil_queries.canonical_cil_entity_id(topic_id)
    raw = cil_queries.topic_timeline(root_safe, anchor_safe, tid, insight_types=types)
    episodes = [
        CilArcEpisodeBlock(
            episode_id=str(b["episode_id"]),
            publish_date=b.get("publish_date"),
            episode_title=b.get("episode_title"),
            feed_title=b.get("feed_title"),
            episode_number=b.get("episode_number"),
            episode_image_url=b.get("episode_image_url"),
            episode_image_local_relpath=b.get("episode_image_local_relpath"),
            feed_image_url=b.get("feed_image_url"),
            feed_image_local_relpath=b.get("feed_image_local_relpath"),
            insights=list(b.get("insights") or []),
        )
        for b in raw
    ]
    return CilTopicTimelineResponse(
        path=root_safe,
        topic_id=tid,
        episodes=episodes,
    )


@router.get("/topics/{topic_id}/persons", response_model=CilIdListResponse)
async def topic_persons(
    request: Request,
    topic_id: str,
    path: str | None = Query(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    ),
) -> CilIdListResponse:
    """Person ids that discuss this topic (via grounded GI quotes)."""
    root_safe, anchor_safe = _require_root_and_anchor(request, path)
    tid = cil_queries.canonical_cil_entity_id(topic_id)
    ids = cil_queries.topic_person_ids(root_safe, anchor_safe, tid)
    return CilIdListResponse(
        path=root_safe,
        anchor_id=tid,
        ids=ids,
    )


@router.get("/persons/{person_id}/topics", response_model=CilIdListResponse)
async def person_topics(
    request: Request,
    person_id: str,
    path: str | None = Query(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    ),
) -> CilIdListResponse:
    """Topic ids this person discusses (via grounded GI edges)."""
    root_safe, anchor_safe = _require_root_and_anchor(request, path)
    ids = cil_queries.person_topic_ids(root_safe, anchor_safe, person_id)
    return CilIdListResponse(
        path=root_safe,
        anchor_id=person_id.strip(),
        ids=ids,
    )
