"""GET/PUT /api/feeds — corpus RSS URL list (``rss_urls.list.txt``).

``GET`` returns URLs as stored in the file (including duplicate lines if present).
``PUT`` deduplicates while preserving first-seen order (convenient for editor saves).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request

from podcast_scraper.feed_list_text import parse_rss_list_file_lines
from podcast_scraper.server.atomic_write import atomic_write_text
from podcast_scraper.server.pathutil import resolve_corpus_path_param
from podcast_scraper.server.schemas import FeedsListResponse, FeedsPutBody

router = APIRouter(tags=["feeds"])

FEEDS_LIST_BASENAME = "rss_urls.list.txt"


@router.get("/feeds", response_model=FeedsListResponse)
async def get_feeds(
    request: Request,
    path: str = Query(..., description="Corpus root directory (resolved under server anchor)."),
) -> FeedsListResponse:
    """Return ``rss_urls.list.txt`` URLs for the resolved corpus root."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = resolve_corpus_path_param(path, anchor)
    fp = root / FEEDS_LIST_BASENAME
    urls: list[str] = []
    if fp.is_file():
        text = fp.read_text(encoding="utf-8", errors="replace")
        urls = parse_rss_list_file_lines(text)
    return FeedsListResponse(
        path=str(root.resolve()),
        file_relpath=FEEDS_LIST_BASENAME,
        urls=urls,
    )


@router.put("/feeds", response_model=FeedsListResponse)
async def put_feeds(
    request: Request,
    body: FeedsPutBody,
    path: str = Query(..., description="Corpus root directory (resolved under server anchor)."),
) -> FeedsListResponse:
    """Persist deduped feed URLs to ``rss_urls.list.txt`` under the corpus root."""
    anchor = getattr(request.app.state, "output_dir", None)
    root = resolve_corpus_path_param(path, anchor)
    if len(body.urls) > 5000:
        raise HTTPException(status_code=400, detail="Too many feed URLs (max 5000).")
    cleaned: list[str] = []
    seen: set[str] = set()
    for u in body.urls:
        s = str(u).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    fp = root / FEEDS_LIST_BASENAME
    text = "\n".join(cleaned)
    if cleaned:
        text += "\n"
    atomic_write_text(fp, text)
    return FeedsListResponse(
        path=str(root.resolve()),
        file_relpath=FEEDS_LIST_BASENAME,
        urls=cleaned,
    )
