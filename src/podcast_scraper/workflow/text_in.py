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

import dataclasses
import hashlib
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
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Tuple

from .. import config, content_types, stage_hooks
from ..kg.v3 import ITEM_TIMES
from ..models import Episode, RssFeed
from ..utils import filesystem
from ..utils.corpus_walk import corpus_rglob, SOURCES_DIR
from .types import FeedMetadata, HostDetectionResult, ProcessingJob

logger = logging.getLogger(__name__)

TEXT_BODY: Literal["text_body"] = "text_body"
CURSOR_FILE = "source_cursor.json"
#: The pipeline stage that re-runs a source's stored items instead of listing new ones.
REDERIVE_ONLY = "rederive_only"


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
            logger.warning("text-in: unreadable item time %r; not recorded", value)
    return None


def _item_times(item: content_types.SourceItem) -> Dict[str, str]:
    """The item's v3 times its transport gave, as ISO 8601 UTC; unreadable ones are dropped."""
    out: Dict[str, str] = {}
    for name in ITEM_TIMES:
        value = _as_datetime(item.metadata.get(name))
        if value is not None:
            out[name] = value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return out


def _source_url(cfg: config.Config, entry: Any) -> str:
    return str(getattr(entry, "url", None) or cfg.rss_url or "")


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


def _item_doc_id(doc: Dict[str, Any]) -> Optional[str]:
    raw_episode = doc.get("episode")
    episode: Dict[str, Any] = raw_episode if isinstance(raw_episode, dict) else {}
    found = doc.get("item_id") or episode.get("guid")
    return found if isinstance(found, str) and found else None


def stored_items(
    source_dir: str, ctype: content_types.ContentType
) -> List[content_types.SourceItem]:
    """This source's items as they are stored: the newest run of each (by run directory name), its
    body from ``transcripts/``, listed again by the type's ``restore_item``. What ``rederive_only``
    re-runs the stages on, so nothing is fetched."""
    return list(stored_items_by_id(source_dir, ctype).values())


def _newest_copies(source_dir: str) -> Dict[str, Tuple[str, Path, Dict[str, Any]]]:
    """Each item's newest stored copy (by run directory name) by its written id: the run, the
    metadata file and its document."""
    newest: Dict[str, Tuple[str, Path, Dict[str, Any]]] = {}
    for path in corpus_rglob(Path(source_dir), "*.metadata.json"):
        if path.parent.name != "metadata":
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        iid = _item_doc_id(doc)
        run = path.parent.parent.name
        if iid is None or (iid in newest and newest[iid][0] >= run):
            continue
        newest[iid] = (run, path, doc)
    return newest


def stored_metadata_by_id(source_dir: str) -> Dict[str, Path]:
    """The metadata file of each item's newest stored copy, by its written id: the copy
    :func:`stored_items_by_id` lists, and the one the surfaces read."""
    return {iid: path for iid, (_run, path, _doc) in sorted(_newest_copies(source_dir).items())}


def stored_items_by_id(
    source_dir: str, ctype: content_types.ContentType
) -> Dict[str, content_types.SourceItem]:
    """:func:`stored_items` by each item's written id (its metadata's ``item_id``, the rows'
    ``content_items.natural_id``), in id order. ``restore_item`` gives an item its transport's id
    again, so this is how a reader of the rows finds an item's stored copy (the news re-check)."""
    newest = _newest_copies(source_dir)
    items: Dict[str, content_types.SourceItem] = {}
    for iid, (_run, path, doc) in sorted(newest.items()):
        stem = path.name[: -len(".metadata.json")]
        body = path.parent.parent / filesystem.TRANSCRIPTS_SUBDIR / f"{stem}.txt"
        try:
            text = body.read_text(encoding="utf-8")
        except OSError:
            logger.warning("text-in: rederive: %s has no stored body; skipped", path)
            continue
        if ctype.restore_item is not None:
            item = ctype.restore_item(doc, text)
        else:
            item = content_types.SourceItem(
                item_id=_item_doc_id(doc) or stem,
                metadata={"title": doc.get("title")},
                text=text,
                url=doc.get("url"),
            )
        raws = sorted(body.parent.glob(f"{stem}.raw.*"))
        if raws and item.raw is None:
            item = dataclasses.replace(
                item, raw=raws[0].read_text(encoding="utf-8"), raw_suffix=raws[0].suffix
            )
        items[iid] = item
    return items


def written_items(source_dir: str, ctype: content_types.ContentType) -> Dict[str, List[Path]]:
    """Every item of ``ctype`` the corpus holds, by id, with its metadata files, read under
    ``sources/<type>/`` (all sources, all runs). ``source_dir`` is this source's workspace; when
    it is not inside a corpus, only that workspace is read."""
    workspace = Path(source_dir)
    root = filesystem.corpus_root_of_workspace(workspace)
    base = root / SOURCES_DIR / ctype.name if root is not None else workspace
    ids: Dict[str, List[Path]] = {}
    for path in corpus_rglob(base, "*.metadata.json"):
        if path.parent.name != "metadata":
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        episode = doc.get("episode") if isinstance(doc.get("episode"), dict) else {}
        found = doc.get("item_id") or episode.get("guid")
        if isinstance(found, str) and found:
            ids.setdefault(found, []).append(path)
    return ids


def _note_also_listed(
    ctype: content_types.ContentType, paths: List[Path], listed: content_types.SourceItem
) -> None:
    """Give each written file of an item what a later listing of it adds (``also_listed``),
    rewritten atomically with its key order kept. Best-effort: a file that cannot be read or
    written is left as it was."""
    if ctype.also_listed is None:
        return
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        updated = ctype.also_listed(doc, listed) if isinstance(doc, dict) else None
        if updated is None:
            continue
        tmp = path.with_name(path.name + ".also-listed.tmp")
        try:
            tmp.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            logger.warning("text-in: could not record a second listing on %s", path)
            continue
        logger.info(
            "text-in: %s is also listed (url=%s); recorded on %s", ctype.name, listed.url, path
        )


def _jobs_for_items(
    items: List[content_types.SourceItem],
    ctype: content_types.ContentType,
    effective_output_dir: str,
    run_suffix: Optional[str],
    *,
    source_id: str,
    transport: str,
    entry: Any = None,
    written: Optional[Dict[str, List[Path]]] = None,
) -> Tuple[List[Episode], List[ProcessingJob]]:  # type: ignore[valid-type]
    """``written``: the items the corpus already holds for this type, with their metadata files.
    One article is one item, so an item already written, by this source or another (two feeds
    carrying one article), or listed twice in this batch, is skipped before its body is fetched:
    the first source to write it keeps it, and takes what the later listing adds (its type's
    ``also_listed``: news, the listing's transport id). The set grows with this batch."""
    episodes: List[Episode] = []  # type: ignore[valid-type]
    jobs: List[ProcessingJob] = []
    on_disk: Dict[str, List[Path]] = written or {}
    seen: Set[str] = set(on_disk)
    for listed in items:
        try:
            listed_id: Optional[str] = content_types.item_id(ctype, listed)
        except ValueError:
            listed_id = None
        if listed_id is not None and listed_id in seen:
            logger.info("text-in: %s is already written; skipped (url=%s)", listed_id, listed.url)
            _note_also_listed(ctype, on_disk.get(listed_id, []), listed)
            continue
        item: Optional[content_types.SourceItem] = listed
        if listed.text is None and ctype.fetch_body is not None:
            item = ctype.fetch_body(listed, entry)
        if item is None or item.text is None:
            logger.warning(
                "text-in: item %s has no text (url=%s); skipped", listed.item_id, listed.url
            )
            continue
        idx = len(episodes)
        minted = content_types.item_id(ctype, item)
        if minted in seen:
            logger.info("text-in: %s is already written; skipped (url=%s)", minted, item.url)
            _note_also_listed(ctype, on_disk.get(minted, []), listed)
            continue
        seen.add(minted)
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
        if item.raw is not None:
            raw = os.path.splitext(str(body))[0] + ".raw" + item.raw_suffix
            Path(raw).write_text(item.raw, encoding="utf-8")
        episodes.append(episode)
        jobs.append(
            ProcessingJob(
                episode=episode,
                transcript_path=rel,
                transcript_source=TEXT_BODY,
                detected_names=None,
                whisper_model=None,
                queued_at=time.monotonic(),
                item=content_types.ItemContext(
                    content_type=ctype,
                    item_id=minted,
                    body_path=rel,
                    source_id=source_id,
                    transport=transport,
                    times=_item_times(item),
                    fields={
                        **item.metadata,
                        "url": item.url,
                        "body_sha256": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
                    },
                ),
            )
        )
    return episodes, jobs


#: What a run's items are: ``(items, the cursor to keep, whether every item is rewritten)``.
ItemsOf = Callable[[str], Tuple[List[content_types.SourceItem], Optional[str], bool]]


def run_text_source(
    cfg: config.Config,
    entry: Any,
    ctype: content_types.ContentType,
    transport: content_types.Transport,
) -> Tuple[int, str]:
    """List the source's new items through ``transport`` and run ``ctype``'s stages on each."""
    if transport.list_items is None:
        raise content_types.NoTextRunner(
            f"transport {transport.name!r} lists no items itself; it cannot feed the text-in path"
        )
    list_items = transport.list_items

    def items_of(source_dir: str) -> Tuple[List[content_types.SourceItem], Optional[str], bool]:
        if str(getattr(cfg, "pipeline_stage", "") or "") == REDERIVE_ONLY:
            # The stored items again, under this run's profile: nothing fetched, the cursor kept,
            # and every item rewritten (a new run supersedes the old one, newest run per item).
            return stored_items(source_dir, ctype), None, True
        listed, next_cursor = list_items(entry, _read_cursor(source_dir, transport.name))
        return listed, next_cursor, False

    return _run_text(cfg, entry, ctype, transport, items_of)


def run_text_items(
    cfg: config.Config,
    entry: Any,
    ctype: content_types.ContentType,
    transport: content_types.Transport,
    items: Sequence[content_types.SourceItem],
) -> Tuple[int, str]:
    """Run ``ctype``'s stages on ``items``, items the source already holds, as one run: the
    rederive path on just these (CRFC-006 B.2 step 5, the news re-check's changed pages). Nothing
    is listed or fetched and the cursor is kept; each item is written again in this run, which
    supersedes its older run (newest run per item), and on rows its article syncs as its next
    version. ``cfg.pipeline_stage`` must be ``rederive_only``: that stage is what the processing
    path treats as a reprocess, and what the row sync records the change under."""
    stage = str(getattr(cfg, "pipeline_stage", "") or "")
    if stage != REDERIVE_ONLY:
        raise ValueError(f"run_text_items needs pipeline_stage={REDERIVE_ONLY!r}, not {stage!r}")
    given = list(items)
    return _run_text(cfg, entry, ctype, transport, lambda _source_dir: (given, None, True))


def _run_text(
    cfg: config.Config,
    entry: Any,
    ctype: content_types.ContentType,
    transport: content_types.Transport,
    items_of: ItemsOf,
) -> Tuple[int, str]:
    """One text-in run over the items ``items_of`` gives for the source directory: the shared
    start, the processing worker and finalisation of :func:`run_text_source` and
    :func:`run_text_items`."""
    from . import orchestration as orch
    from .stages import processing as processing_stage

    source_dir = cfg.output_dir or "."
    cfg, effective_output_dir, run_suffix, _cfg_str, pipeline_metrics = orch._begin_run(cfg)
    orch._corpus_deferred("retry_deferred", source_dir)
    try:
        stage_hooks.run_started(cfg)
        jsonl_emitter = orch._setup_jsonl_emitter(cfg, effective_output_dir, pipeline_metrics)
        run_manifest = orch._create_run_manifest(cfg, effective_output_dir)

        items, next_cursor, rewrite = items_of(source_dir)
        episodes, jobs = _jobs_for_items(
            items,
            ctype,
            effective_output_dir,
            run_suffix,
            source_id=content_types.source_id(ctype, _source_url(cfg, entry)),
            transport=transport.name,
            entry=entry,
            written=None if rewrite else written_items(source_dir, ctype),
        )
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
        orch._corpus_deferred("flush_deferred", effective_output_dir)
        stage_hooks.run_completed(cfg, effective_output_dir)
        _write_cursor(source_dir, transport.name, next_cursor)
        return result
    except BaseException:
        # Kept for the next run, not retried: the process may be stopping.
        orch._corpus_deferred("flush_deferred", effective_output_dir, retry=False)
        raise
    finally:
        from .show_events import unbind_show

        unbind_show()
        try:
            config.reset_screenplay_issue_562_gates()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("reset_screenplay_issue_562_gates failed", exc_info=True)
