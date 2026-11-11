"""The text-in entry point: run a source whose items arrive as text (ADR-177 decision 4).

It reuses what a podcast run does around its items and replaces only how they arrive:

- the same start (:func:`orchestration._begin_run`), event log, run manifest, processing worker
  and finalisation (``_finalize_pipeline``, which also rewrites the run summary the caches key on);
- in place of the feed fetch and the download and transcription threads, the source's transport
  lists the new items. Each becomes an RSS-shaped ``<item>`` (its guid is the item id) and its
  body file under ``transcripts/``, so metadata, summary and KG read it as they read a transcript.

Each processing job carries the item's context (type, id, body path); the type's stage profile
decides which stages run (ADR-177 decision 7). Only the summary provider is created: a text item
never reaches transcription or speaker detection.

The transport's cursor is kept in one small file per source, beside its runs, written after the
run finishes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import xml.etree.ElementTree as ET  # nosec B405 - builds elements, parses nothing
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Literal, Optional, Tuple

from .. import config, content_types, stage_hooks
from ..models import Episode, RssFeed
from ..utils import filesystem
from .types import FeedMetadata, HostDetectionResult, ProcessingJob

logger = logging.getLogger(__name__)

TEXT_BODY: Literal["text_body"] = "text_body"
CURSOR_FILE = "source_cursor.json"


def _item_element(item: content_types.SourceItem, title: str, minted_id: str) -> ET.Element:
    """An RSS-shaped ``<item>`` for a text item: what the metadata path reads from an episode."""
    elem = ET.Element("item")
    ET.SubElement(elem, "title").text = title
    guid = ET.SubElement(elem, "guid", isPermaLink="false")
    guid.text = minted_id
    if item.url:
        ET.SubElement(elem, "link").text = item.url
    description = item.metadata.get("description")
    if description:
        ET.SubElement(elem, "description").text = str(description)
    published = _as_datetime(item.metadata.get("published_at"))
    if published is not None:
        ET.SubElement(elem, "pubDate").text = format_datetime(published)
    return elem


def _as_datetime(value: Any) -> Optional[datetime]:
    """A transport's publish time as a datetime (RSS ``pubDate`` is RFC 2822, so an ISO string
    is converted); an unreadable value is dropped rather than written as a date nobody parses."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            return _as_datetime(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
        except ValueError:
            logger.warning("text-in: unreadable published_at %r; no date recorded", value)
    return None


def _read_cursor(source_dir: str, transport: str) -> Optional[str]:
    path = Path(source_dir) / CURSOR_FILE
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("transport") != transport:
        return None
    cursor = doc.get("cursor")
    return cursor if isinstance(cursor, str) else None


def _write_cursor(source_dir: str, transport: str, cursor: Optional[str]) -> None:
    if cursor is None:
        return
    path = Path(source_dir) / CURSOR_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"transport": transport, "cursor": cursor}), encoding="utf-8")
    os.replace(tmp, path)


def _jobs_for_items(
    items: List[content_types.SourceItem],
    ctype: content_types.ContentType,
    effective_output_dir: str,
    run_suffix: Optional[str],
) -> Tuple[List[Episode], List[ProcessingJob]]:  # type: ignore[valid-type]
    episodes: List[Episode] = []  # type: ignore[valid-type]
    jobs: List[ProcessingJob] = []
    for item in items:
        if item.text is None:
            # Fetching and extracting a page is the news type's work (News 1.0), not the seam's.
            logger.warning("text-in: item %s has no text (url=%s); skipped", item.item_id, item.url)
            continue
        idx = len(episodes)
        minted = content_types.item_id(ctype, item)
        title = str(item.metadata.get("title") or item.item_id)
        title_safe = filesystem.sanitize_filename(title)
        episode = Episode(
            idx=idx,
            title=title,
            title_safe=title_safe,
            item=_item_element(item, title, minted),
            transcript_urls=[],
            media_url=None,
            media_type=None,
        )
        rel = os.path.join(
            filesystem.TRANSCRIPTS_SUBDIR,
            filesystem.build_whisper_output_name(idx, title_safe, run_suffix),
        )
        body = Path(effective_output_dir) / rel
        body.parent.mkdir(parents=True, exist_ok=True)
        body.write_text(item.text, encoding="utf-8")
        episodes.append(episode)
        jobs.append(
            ProcessingJob(
                episode=episode,
                transcript_path=rel,
                transcript_source=TEXT_BODY,
                detected_names=None,
                whisper_model=None,
                queued_at=time.monotonic(),
                item=content_types.ItemContext(content_type=ctype, item_id=minted, body_path=rel),
            )
        )
    return episodes, jobs


def run_text_source(
    cfg: config.Config,
    entry: Any,
    ctype: content_types.ContentType,
    transport: content_types.Transport,
) -> Tuple[int, str]:
    """List the source's new items through ``transport`` and run ``ctype``'s stages on each."""
    from . import orchestration as orch
    from .stages import processing as processing_stage

    if transport.list_items is None:
        raise content_types.NoTextRunner(
            f"transport {transport.name!r} lists no items itself; it cannot feed the text-in path"
        )
    source_dir = cfg.output_dir or "."
    cfg, effective_output_dir, run_suffix, _cfg_str, pipeline_metrics = orch._begin_run(cfg)
    try:
        stage_hooks.run_started(cfg)
        jsonl_emitter = orch._setup_jsonl_emitter(cfg, effective_output_dir, pipeline_metrics)
        run_manifest = orch._create_run_manifest(cfg, effective_output_dir)

        cursor = _read_cursor(source_dir, transport.name)
        items, next_cursor = transport.list_items(entry, cursor)
        episodes, jobs = _jobs_for_items(items, ctype, effective_output_dir, run_suffix)
        logger.info(
            "text-in: %s over %s: %d new item(s), %d with text",
            ctype.name,
            transport.name,
            len(items),
            len(jobs),
        )

        summary_provider = orch._create_summarization_provider(cfg)
        resources = processing_stage.setup_processing_resources(cfg)
        resources.processing_jobs.extend(jobs)
        done = threading.Event()
        done.set()  # no transcription: every job is queued before the worker starts
        title = str(getattr(entry, "title", None) or cfg.rss_url or ctype.name)
        feed = RssFeed(title=title, items=[], base_url=cfg.rss_url or "")
        processing_stage.process_processing_jobs_concurrent(
            processing_resources=resources,
            feed=feed,
            cfg=cfg,
            effective_output_dir=effective_output_dir,
            run_suffix=run_suffix,
            feed_metadata=FeedMetadata(None, None, None),
            host_detection_result=HostDetectionResult(
                cached_hosts=set(), heuristics=None, feed_title=title
            ),
            pipeline_metrics=pipeline_metrics,
            summary_provider=summary_provider,
            transcription_complete_event=done,
        )
        result = orch._finalize_pipeline(
            cfg=cfg,
            saved=len(jobs),
            transcription_resources=SimpleNamespace(temp_dir=None),
            effective_output_dir=effective_output_dir,
            run_suffix=run_suffix,
            pipeline_metrics=pipeline_metrics,
            episodes=episodes,
            jsonl_emitter=jsonl_emitter,
            run_manifest=run_manifest,
            summary_provider=summary_provider,
            transcription_provider=None,
        )
        stage_hooks.run_completed(cfg, effective_output_dir)
        _write_cursor(source_dir, transport.name, next_cursor)
        return result
    finally:
        from .show_events import unbind_show

        unbind_show()
        try:
            config.reset_screenplay_issue_562_gates()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("reset_screenplay_issue_562_gates failed", exc_info=True)
