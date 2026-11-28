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
run finishes. Like an unprocessed podcast episode, a listed item that did not write (its summary
failed, or it was left past the source's cap) is offered again: the cursor stops short of the
oldest such item, which the next run lists again while the items already written are skipped.

Failures go through the podcast path's own mechanisms, not copies of them:

- the body's fetch is the shared downloader's (retries, backoff, Retry-After, circuit breaker,
  throttle); a failed fetch is classified by its rule (``downloader.is_transient_failure``), so a
  transient one waits for the next run and a permanent one is skipped;
- every fetch that fails is a row in ``corpus_incidents.jsonl`` (soft or policy), and the batch's
  rollup counts it as it counts an episode's;
- an item whose fetch or metadata keeps failing is parked by ``episode_park`` (stages ``fetch``
  and ``metadata``), with its outage rule, alert and ``parked-episodes --release``; a parked item
  is skipped before its fetch and no longer holds the cursor;
- a summary provider's ``ResilienceFuseOpenError`` halts the run (ADR-122), so nothing is written
  and the cursor is kept.
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
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

from .. import config, content_types, stage_hooks
from ..kg.v3 import ITEM_TIMES
from ..models import Episode, RssFeed
from ..utils import filesystem
from ..utils.corpus_incidents import append_corpus_incident
from ..utils.corpus_walk import corpus_rglob, SOURCES_DIR
from . import episode_park
from .types import FeedMetadata, HostDetectionResult, ProcessingJob

logger = logging.getLogger(__name__)

TEXT_BODY: Literal["text_body"] = "text_body"
CURSOR_FILE = "source_cursor.json"
#: The pipeline stage that re-runs a source's stored items instead of listing new ones.
REDERIVE_ONLY = "rederive_only"
#: Listed items still to write, each with the id its metadata will carry (``None``: not fetched).
_Waiting = List[Tuple[content_types.SourceItem, Optional[str]]]
#: The park's stages for a text item (``episode_park``): its body's fetch, and its metadata.
FETCH = "fetch"
METADATA = "metadata"


def park_key(ctype: content_types.ContentType, listed: content_types.SourceItem) -> str:
    """The id an item is parked and recorded under: its minted id as listed, else its URL."""
    try:
        return content_types.item_id(ctype, listed)
    except ValueError:
        return str(listed.url or listed.item_id)


def _skip_parked(incidents: Optional[str], key: str, feed_url: Optional[str]) -> bool:
    """Skip, before its fetch, an item parked after repeated failures (as a podcast episode is
    skipped before its download), with the same log marker and policy incident."""
    entry = episode_park.parked_entry(incidents, key)
    if entry is None:
        return False
    reason = (
        f"parked {entry['parked_at']} after {len(entry.get('failures') or [])} failures; "
        "release with `parked-episodes --release`"
    )
    logger.warning("text-in: %s episode_id=%s skipped: %s", episode_park.LOG_MARKER, key, reason)
    append_corpus_incident(
        incidents,
        scope="episode",
        category="policy",
        message=reason,
        exception_type="EpisodeParked",
        stage=FETCH,
        feed_url=feed_url,
        episode_id=key,
    )
    return True


def _note_not_fetched(
    incidents: Optional[str],
    key: str,
    feed_url: Optional[str],
    listed: content_types.SourceItem,
    why: content_types.NotFetched,
) -> None:
    """Record a body the type could not fetch. Transient: a soft incident, and a failure in the
    park's batch (it counts only when another item's fetch passed). Permanent: a policy incident;
    the item is skipped for good."""
    logger.warning(
        "text-in: item %s not fetched (%s, %s; url=%s)",
        key,
        why.reason,
        "waits for the next run" if why.transient else "skipped",
        listed.url,
    )
    append_corpus_incident(
        incidents,
        scope="episode",
        category="soft" if why.transient else "policy",
        message=f"not fetched: {why.reason}",
        exception_type="NotFetched",
        stage=FETCH,
        feed_url=feed_url,
        episode_id=key,
    )
    if why.transient:
        episode_park.note_failure(
            incidents,
            key,
            feed_url=feed_url,
            exception_type="NotFetched",
            message=why.reason,
            stage=FETCH,
        )


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
    items: List[content_types.SourceItem] = []
    for _iid, (_run, path, doc) in sorted(newest.items()):
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
        items.append(item)
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
    cap: Optional[int] = None,
    incidents: Optional[str] = None,
    feed_url: Optional[str] = None,
) -> Tuple[List[Episode], List[ProcessingJob], _Waiting]:  # type: ignore[valid-type]
    """``written``: the items the corpus already holds for this type, with their metadata files.
    One article is one item, so an item already written, by this source or another (two feeds
    carrying one article), or listed twice in this batch, is skipped before its body is fetched:
    the first source to write it keeps it, and takes what the later listing adds (its type's
    ``also_listed``: news, the listing's transport id). The set grows with this batch.

    ``cap``: the most jobs made; an item already written does not count toward it. Also returned:
    the listed items still to write, each with the id its metadata will carry (``None``: not
    fetched, past the cap or after a transient fetch failure), which the cursor must not pass
    until they write.

    ``incidents``: the run's ``corpus_incidents.jsonl``. With it, an item parked after repeated
    failures (``episode_park``, as a podcast episode is) is skipped before its fetch, and each
    fetch is recorded there and in the park's batch, under stage ``fetch``."""
    episodes: List[Episode] = []  # type: ignore[valid-type]
    jobs: List[ProcessingJob] = []
    waiting: _Waiting = []
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
        key = park_key(ctype, listed)
        if _skip_parked(incidents, key, feed_url):
            continue
        if cap is not None and len(jobs) >= cap:
            waiting.append((listed, None))
            continue
        item: Optional[content_types.SourceItem] = listed
        if listed.text is None and ctype.fetch_body is not None:
            fetched = ctype.fetch_body(listed, entry)
            if isinstance(fetched, content_types.NotFetched):
                _note_not_fetched(incidents, key, feed_url, listed, fetched)
                if fetched.transient:
                    waiting.append((listed, None))
                continue
            item = fetched
            if item is not None and item.text is not None:
                episode_park.note_passed(incidents, key, FETCH)
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
        waiting.append((listed, minted))
    return episodes, jobs, waiting


def _note_metadata_outcomes(
    incidents: Optional[str],
    ctype: content_types.ContentType,
    waiting: _Waiting,
    now_written: Dict[str, List[Path]],
    feed_url: Optional[str],
) -> None:
    """Each fetched item's metadata outcome in the park's batch, under stage ``metadata``: one
    that wrote passed, one that did not failed (its processing already recorded the hard
    incident). Counted at the batch's end only when another item's metadata wrote."""
    for listed, minted in waiting:
        if minted is None:
            continue
        key = park_key(ctype, listed)
        if minted in now_written:
            episode_park.note_passed(incidents, key, METADATA)
        else:
            episode_park.note_failure(
                incidents,
                key,
                feed_url=feed_url,
                exception_type="NotWritten",
                message="its metadata was not written",
                stage=METADATA,
            )


def _next_cursor(
    cursor: Optional[str],
    listed_next: Optional[str],
    items: List[content_types.SourceItem],
    unwritten: List[content_types.SourceItem],
    key: Callable[[str], Any],
) -> Optional[str]:
    """Where the cursor may go after a run. Past every item when all of them wrote (the
    transport's own next cursor); else to the newest item older than the oldest one that did not
    write, so the next run lists that one again; with none older, where it was. Undated items
    (no cursor value) are listed every run and never hold it."""
    held = [key(i.cursor) for i in unwritten if i.cursor is not None]
    if not held:
        return listed_next
    oldest = min(held)
    older = [i.cursor for i in items if i.cursor is not None and key(i.cursor) < oldest]
    return max(older, key=key) if older else cursor


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

        rederive = str(getattr(cfg, "pipeline_stage", "") or "") == REDERIVE_ONLY
        if rederive:
            # The stored items again, under this run's profile: nothing fetched, the cursor kept,
            # and every item rewritten (a new run supersedes the old one, newest run per item).
            cursor, items, next_cursor = None, stored_items(source_dir, ctype), None
        else:
            cursor = _read_cursor(source_dir, transport.name)
            items, next_cursor = transport.list_items(entry, cursor)
        cap = None if rederive or transport.max_items is None else transport.max_items(entry)
        incidents = getattr(cfg, "incident_log_path", None)
        episodes, jobs, waiting = _jobs_for_items(
            items,
            ctype,
            effective_output_dir,
            run_suffix,
            source_id=content_types.source_id(ctype, _source_url(cfg, entry)),
            transport=transport.name,
            entry=entry,
            written=None if rederive else written_items(source_dir, ctype),
            cap=cap,
            incidents=None if rederive else incidents,
            feed_url=_source_url(cfg, entry),
        )
        logger.info(
            "text-in: %s over %s: %d new item(s), %d with text, %d waiting",
            ctype.name,
            transport.name,
            len(items),
            len(jobs),
            sum(1 for _listed, minted in waiting if minted is None),
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
        if not rederive:
            now_written = written_items(source_dir, ctype)
            unwritten = [listed for listed, minted in waiting if minted not in now_written]
            _note_metadata_outcomes(incidents, ctype, waiting, now_written, _source_url(cfg, entry))
            next_cursor = _next_cursor(cursor, next_cursor, items, unwritten, transport.cursor_key)
            if unwritten:
                logger.warning(
                    "text-in: %s over %s: %d listed item(s) not written; cursor held at %s, so "
                    "the next run lists them again",
                    ctype.name,
                    transport.name,
                    len(unwritten),
                    next_cursor,
                )
        _write_cursor(source_dir, transport.name, next_cursor)
        return result
    finally:
        from .show_events import unbind_show

        unbind_show()
        try:
            config.reset_screenplay_issue_562_gates()
        except Exception:  # pragma: no cover - defensive cleanup
            logger.debug("reset_screenplay_issue_562_gates failed", exc_info=True)
