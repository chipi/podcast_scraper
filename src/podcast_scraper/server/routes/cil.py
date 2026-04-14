"""RFC-072 cross-layer CIL query API (GitHub #527)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.server import cil_queries
from podcast_scraper.server.pathutil import resolve_corpus_path_param, resolved_corpus_root_str
from podcast_scraper.server.schemas import (
    CilArcEpisodeBlock,
    CilGuestBriefInsightRow,
    CilGuestBriefQuoteRow,
    CilGuestBriefResponse,
    CilIdListResponse,
    CilPositionArcResponse,
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


def _require_root_safe(request: Request, path: str | None) -> tuple[str, Path | None]:
    """Return ``(root_safe, anchor)`` — sanitised corpus root string and anchor.

    ``root_safe`` comes from ``resolved_corpus_root_str`` which applies
    ``os.path.normpath`` + ``str.startswith`` — the sanitiser that CodeQL
    recognises for ``py/path-injection``.
    """
    anchor = getattr(request.app.state, "output_dir", None)
    root = _resolve_corpus_root(path, anchor)
    if root is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "path query parameter is required when the server has no default output_dir "
                "(set PODCAST_SERVE_OUTPUT_DIR or pass output_dir to create_app)."
            ),
        )
    return resolved_corpus_root_str(root, anchor), anchor


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
    root_safe, _anchor = _require_root_safe(request, path)
    types = _parse_insight_types(insight_types, default=("claim",))
    raw = cil_queries.position_arc(root_safe, person_id, topic, insight_types=types)
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


@router.get("/persons/{person_id}/brief", response_model=CilGuestBriefResponse)
async def person_brief(
    request: Request,
    person_id: str,
    path: str | None = Query(
        default=None,
        description="Corpus root. Omit when server default output_dir is set.",
    ),
) -> CilGuestBriefResponse:
    """Guest intelligence brief — insights grouped by topic (RFC-072 Pattern B)."""
    root_safe, _anchor = _require_root_safe(request, path)
    brief = cil_queries.guest_brief(root_safe, person_id)
    topics_raw = brief.get("topics") or {}
    topics_out: dict[str, list[CilGuestBriefInsightRow]] = {}
    if isinstance(topics_raw, dict):
        for tid, rows in topics_raw.items():
            if not isinstance(rows, list):
                continue
            out_rows: list[CilGuestBriefInsightRow] = []
            for row in rows:
                if isinstance(row, dict):
                    out_rows.append(CilGuestBriefInsightRow.model_validate(row))
            topics_out[str(tid)] = out_rows
    quotes_raw = brief.get("quotes") or []
    quotes: list[CilGuestBriefQuoteRow] = []
    if isinstance(quotes_raw, list):
        for row in quotes_raw:
            if isinstance(row, dict):
                quotes.append(CilGuestBriefQuoteRow.model_validate(row))
    return CilGuestBriefResponse(
        path=root_safe,
        person_id=str(brief.get("person_id") or person_id.strip()),
        topics=topics_out,
        quotes=quotes,
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
    root_safe, _anchor = _require_root_safe(request, path)
    types = _parse_insight_types(insight_types, default=None)
    raw = cil_queries.topic_timeline(root_safe, topic_id, insight_types=types)
    episodes = [
        CilArcEpisodeBlock(
            episode_id=str(b["episode_id"]),
            publish_date=b.get("publish_date"),
            insights=list(b.get("insights") or []),
        )
        for b in raw
    ]
    return CilTopicTimelineResponse(
        path=root_safe,
        topic_id=topic_id.strip(),
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
    root_safe, _anchor = _require_root_safe(request, path)
    ids = cil_queries.topic_person_ids(root_safe, topic_id)
    return CilIdListResponse(
        path=root_safe,
        anchor_id=topic_id.strip(),
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
    root_safe, _anchor = _require_root_safe(request, path)
    ids = cil_queries.person_topic_ids(root_safe, person_id)
    return CilIdListResponse(
        path=root_safe,
        anchor_id=person_id.strip(),
        ids=ids,
    )
