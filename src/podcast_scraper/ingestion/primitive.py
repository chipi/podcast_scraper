"""Corpus ingestion primitive (#1069) — ``ingest_feed``.

The durable write-path spine both PRD-037 phases share. It composes existing
machinery rather than reinventing it:

* :func:`podcast_scraper.service.run` runs the single-feed pipeline **and**
  upserts the corpus manifest (the merge) — and the pipeline is incrementally
  deduped per episode, so a re-run only processes new episodes.
* :func:`~podcast_scraper.utils.filesystem.feed_workspace_dirname` +
  :func:`~podcast_scraper.workflow.corpus_operations.corpus_parent_for_manifest_stamp_from_cfg`
  give the stable ``<corpus>/feeds/<dir>`` location for the feed-level dedup check.

What this adds on top: the :class:`~podcast_scraper.ingestion.policy.IngestPolicy`
authorization seam (so phase-2 guardrails slot in without touching this path), a
feed-level already-present short-circuit, and an ingestion-framed result. The
pipeline runner is injected (defaults to ``service.run``) so unit tests exercise
the orchestration — authorize / dedup / result mapping — without invoking the
real ML pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, TYPE_CHECKING

from podcast_scraper.ingestion.policy import AllowAllPolicy, IngestPolicy, IngestRequest
from podcast_scraper.utils.filesystem import feed_workspace_dirname
from podcast_scraper.workflow.corpus_operations import (
    corpus_parent_for_manifest_stamp_from_cfg,
)

if TYPE_CHECKING:  # avoid importing the heavy service/config modules at import time
    from podcast_scraper import config as _config

#: ``service.run(cfg) -> ServiceResult`` — injected so tests can stub the pipeline.
ServiceRunner = Callable[["_config.Config"], object]

STATUS_INGESTED = "ingested"
STATUS_ALREADY_PRESENT = "already_present"
STATUS_FAILED = "failed"


@dataclass(frozen=True)
class IngestResult:
    """Outcome of one :func:`ingest_feed` call.

    ``status`` is ``ingested`` (the pipeline ran and merged), ``already_present``
    (the feed was already in the corpus and ``force`` was not set — a no-op), or
    ``failed`` (rejected shape or a pipeline error; ``error`` explains).
    """

    status: str
    feed_url: str
    feed_dir: str
    episodes_added: int = 0
    summary: Optional[str] = None
    error: Optional[str] = None


def _feed_dir_path(cfg: "_config.Config", feed_url: str) -> Optional[Path]:
    """``<corpus>/feeds/<stable_dir>`` for this feed, or ``None`` if not corpus-layout."""
    parent = corpus_parent_for_manifest_stamp_from_cfg(cfg)
    if not parent:
        return None
    return Path(parent) / "feeds" / feed_workspace_dirname(feed_url)


def ingest_feed(
    cfg: "_config.Config",
    *,
    run: Optional[ServiceRunner] = None,
    policy: Optional[IngestPolicy] = None,
    actor: Optional[str] = None,
    force: bool = False,
) -> IngestResult:
    """Ingest the feed described by ``cfg`` into its corpus — authorized + deduped.

    ``cfg`` is a single-feed, corpus-layout :class:`~podcast_scraper.config.Config`
    (``rss_url`` set, ``output_dir`` under ``<corpus>/feeds/<dir>``,
    ``single_feed_uses_corpus_layout=True``). ``run`` defaults to
    :func:`podcast_scraper.service.run` (pipeline + manifest merge); tests inject a
    stub. ``policy`` gates the request (default :class:`AllowAllPolicy`); a
    rejection propagates as :class:`~podcast_scraper.ingestion.policy.IngestNotAuthorized`.
    ``force`` re-runs even when the feed dir already exists (the incremental
    pipeline still skips episodes already processed).
    """
    feed_url = (getattr(cfg, "rss_url", None) or "").strip()
    if not feed_url:
        return IngestResult(
            status=STATUS_FAILED,
            feed_url="",
            feed_dir="",
            error="cfg has no rss_url — ingestion targets a single feed",
        )

    # Authorization seam — runs BEFORE any dedup or pipeline work (phase-2 guardrails).
    (policy or AllowAllPolicy()).authorize(IngestRequest(feed_url=feed_url, actor=actor))

    feed_dir_path = _feed_dir_path(cfg, feed_url)
    feed_dir = feed_dir_path.name if feed_dir_path else feed_workspace_dirname(feed_url)

    if feed_dir_path is not None and feed_dir_path.is_dir() and not force:
        return IngestResult(
            status=STATUS_ALREADY_PRESENT,
            feed_url=feed_url,
            feed_dir=feed_dir,
            summary="feed already in corpus; pass force=True to re-run (incremental)",
        )

    if run is None:
        from podcast_scraper import service

        run = service.run

    result = run(cfg)
    if not getattr(result, "success", False):
        return IngestResult(
            status=STATUS_FAILED,
            feed_url=feed_url,
            feed_dir=feed_dir,
            error=str(getattr(result, "error", None) or "pipeline run failed"),
        )
    return IngestResult(
        status=STATUS_INGESTED,
        feed_url=feed_url,
        feed_dir=feed_dir,
        episodes_added=int(getattr(result, "episodes_processed", 0) or 0),
        summary=getattr(result, "summary", None),
    )


__all__ = [
    "IngestResult",
    "ingest_feed",
    "STATUS_ALREADY_PRESENT",
    "STATUS_FAILED",
    "STATUS_INGESTED",
]
