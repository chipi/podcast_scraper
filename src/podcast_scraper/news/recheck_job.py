"""The news re-check job (CRFC-006 Part B, B.2; tracker n-w9): one run over the items due.

A run enrols the served articles inside their source's window, takes the items due, and checks
each against the page it came from (``recheck_fetch.check``: robots.txt, the source's agent, a
conditional GET with the stored validators). ``recheck.advance`` says what follows:

- the check is recorded (the same text, a held candidate, one gone answer, a failed fetch);
- a changed page is written as the item's next version: every changed item of a source in one
  ``text_in.run_text_items`` run, the stored item with the page's new text and raw cache (B.2 step
  5), so its files get a new run and its row sync writes the article's next version;
- the second gone answer in a row withdraws it (``db.news_recheck.withdraw``).

Its text is compared with the served text, unless the served text came from an extractor other
than today's (``extractor_version`` differs): then with the stored raw page extracted again first,
so a library upgrade writes no version (B.2 step 2).

A page's text is compared as ``recheck_fetch.check`` compares it, given the served text's
extraction method, so a page that only flips between trafilatura and readability is the same (B.2
step 3). An item of a Guardian source is re-read by id instead (``recheck_fetch.check_guardian``),
and its next version is the API record made into an item as the listing makes one
(``guardian.item_from_record``).

Not re-checked: an item whose served text is the feed's (a page fetch does not reproduce it, and a
feed drops old items within days). It is recorded as a check that found ``unsupported``, and an
item whose stored copy is missing as one that found ``no_body``, so its schedule moves on and it
never holds the head of the queue. A run takes only the items of the sources it is given.

A changed item is recorded only when the rows serve a new article document after the write: the
pipeline run reports a refused or deferred row sync as an outcome, never an exception, so such an
item is counted ``not_synced``, logged as an error and left due (a deferred sync is retried at the
next pipeline run's start). A check that raises is recorded as ``failed``.

Leases: the run holds ``corpus`` shared and ``build:news_recheck`` exclusive (one run at a time),
extended while a source's write runs; a withdrawal also takes its source's exclusive lease, as an
ingest of the source does. The version write takes its own leases in the row sync. Each item's
state is written in its own transaction, after its write, so a run stopped midway re-checks what it
did not record.
"""

from __future__ import annotations

import copy
import logging
import threading
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from sqlalchemy import exists, select
from sqlalchemy.engine import Engine

from .. import content_types
from ..db import leases, news_recheck as rows
from ..db.importers.corpus_documents import ARTICLE
from ..db.schema import documents
from . import extract, fetch, guardian, recheck, recheck_fetch

logger = logging.getLogger(__name__)

#: The run's own lease: one re-check run at a time.
RESOURCE = leases.build_resource("news_recheck")
#: The transport whose items are re-read by id.
GUARDIAN = "guardian"
#: Extractions a page fetch reproduces.
PAGE_EXTRACTIONS = frozenset({extract.TRAFILATURA, extract.READABILITY, extract.STANDFIRST})
#: What a check records for an item it could not check.
UNSUPPORTED = "unsupported"
NO_BODY = "no_body"

Check = Callable[..., recheck_fetch.Check]
Write = Callable[..., Tuple[int, str]]


@dataclass(frozen=True)
class SourceRun:
    """One configured source, as the batch loop runs it: its config (``output_dir`` is its
    directory), its feeds entry, its type and its transport."""

    cfg: Any
    entry: Any
    ctype: content_types.ContentType
    transport: content_types.Transport

    @property
    def natural_id(self) -> str:
        url = str(getattr(self.entry, "url", None) or getattr(self.cfg, "rss_url", None) or "")
        return content_types.source_id(self.ctype, url)


def _served_text(item: content_types.SourceItem, source: SourceRun) -> str:
    """The text to compare with: the served one, or its raw page extracted again when today's
    extractor is not the one that made it (B.2 step 2)."""
    text = item.text or ""
    method = str(item.metadata.get("extraction") or "")
    made_by = item.metadata.get("extractor_version")
    if not item.raw or not made_by or made_by == extract.extractor_version(method):
        return text
    found = extract.main_text(
        item.raw, item.url or "", text_from=fetch.source_spec(source.entry).text_from
    )
    return found.text if found else text


def _new_version(
    item: content_types.SourceItem, source: SourceRun, got: recheck_fetch.Check
) -> Optional[content_types.SourceItem]:
    """The stored item with the page's text, its fields and its raw cache, as the ingest's page
    fetch completes it (a Guardian item: its API record, as the listing makes it); ``None`` when
    there is no text or it is in a language we do not ingest."""
    if source.transport.name == GUARDIAN:
        return guardian.item_from_record(got.record, source.entry) if got.record else None
    url, html = item.url or "", got.html or ""
    found = extract.main_text(html, url, text_from=fetch.source_spec(source.entry).text_from)
    if found is None:
        return None
    return fetch.complete(
        item,
        source.entry,
        found.text,
        method=found.method,
        declared_language=found.language,
        byline_raw=item.metadata.get("byline_raw") or extract.page_author(html, url),
        raw=html,
        raw_suffix=".html",
    )


def _unsupported(item: content_types.SourceItem, source: SourceRun) -> bool:
    return (
        source.transport.name != GUARDIAN
        and str(item.metadata.get("extraction") or "") not in PAGE_EXTRACTIONS
    )


def _check(
    item: content_types.SourceItem,
    source: SourceRun,
    found: rows.Due,
    check: Check,
    check_guardian: Check,
) -> recheck_fetch.Check:
    if source.transport.name == GUARDIAN:
        return check_guardian(item.text or "", item.item_id, source.entry, found.validators)
    return check(
        _served_text(item, source),
        item.url or "",
        source.entry,
        found.validators,
        served_method=str(item.metadata.get("extraction") or "") or None,
    )


def run(
    engine: Engine,
    sources: Sequence[SourceRun],
    now: Optional[datetime] = None,
    *,
    check: Check = recheck_fetch.check,
    check_guardian: Check = recheck_fetch.check_guardian,
    write: Optional[Write] = None,
    limit: int = 500,
) -> Counter[str]:
    """One run at ``now`` (default: the clock) over the due items of ``sources``; returns what
    it did, by outcome."""
    from ..workflow import text_in

    when = now or datetime.now(timezone.utc)
    write_items = write or text_in.run_text_items
    done: Counter[str] = Counter()
    holder = f"news_recheck:{uuid.uuid4().hex[:12]}"
    held = leases.acquire(
        engine,
        [
            leases.LeaseRequest(leases.CORPUS, leases.SHARED),
            leases.LeaseRequest(RESOURCE, leases.EXCLUSIVE),
        ],
        holder,
    )
    try:
        by_id = {source.natural_id: source for source in sources}
        with engine.begin() as conn:
            leases.heartbeat(conn, holder, held)
            done["enrolled"] += rows.enrol(conn, when)
            due = rows.due(conn, when, limit, source_natural_ids=list(by_id))
        grouped: Dict[str, List[rows.Due]] = {}
        for found in due:
            grouped.setdefault(found.source_natural_id, []).append(found)
        for source_id, items in grouped.items():
            _run_source(
                engine,
                holder,
                held,
                by_id[source_id],
                items,
                when,
                (check, check_guardian),
                write_items,
                done,
            )
    finally:
        leases.release(engine, holder)
    logger.info("news re-check: %s", dict(sorted(done.items())))
    return done


def _run_source(
    engine: Engine,
    holder: str,
    held: int,
    source: SourceRun,
    due: List[rows.Due],
    when: datetime,
    checks: Tuple[Check, Check],
    write: Write,
    done: Counter[str],
) -> None:
    from ..workflow import text_in

    stored = text_in.stored_items_by_id(source.cfg.output_dir or ".", source.ctype)
    changed: List[Tuple[rows.Due, recheck.RecheckState, recheck_fetch.Check]] = []
    versions: List[content_types.SourceItem] = []
    for found in due:
        item = stored.get(found.natural_id)
        if item is None or not item.text:
            logger.warning("news re-check: %s has no stored body", found.natural_id)
            _skip(engine, holder, held, found, NO_BODY, when, done)
            continue
        if _unsupported(item, source):
            _skip(engine, holder, held, found, UNSUPPORTED, when, done)
            continue
        try:
            got = _check(item, source, found, *checks)
        except Exception:  # noqa: BLE001 - one item's failed check is a failed check, nothing more
            logger.exception("news re-check: checking %s failed", found.natural_id)
            got = recheck_fetch.Check(recheck.FAILED, found.validators, 0)
        state, action = recheck.advance(found.state, got.status)
        if action == recheck.WRITE_VERSION:
            version = _new_version(item, source, got)
            if version is not None:
                changed.append((found, state, got))
                versions.append(version)
                continue
            # Changed, but the page completes to nothing (a language we do not ingest): the
            # served version stays and the check is recorded.
            _record(engine, holder, held, found, state, got, when)
            done["not_written"] += 1
            continue
        if action == recheck.WITHDRAW:
            try:
                withdrawn = _withdraw(engine, holder, held, found, state, got, when)
            except leases.LeaseRefused as exc:
                # The source is being ingested: the item stays due, its gone count unchanged, so
                # the next run's gone answer withdraws it.
                done["withdraw_deferred"] += 1
                logger.info("news re-check: %s withdraw deferred: %s", found.natural_id, exc)
                continue
            done["withdrawn" if withdrawn else "already_withdrawn"] += 1
            continue
        _record(engine, holder, held, found, state, got, when)
        done[got.status] += 1
    if not versions:
        return
    item_ids = [found.item_id for found, _state, _got in changed]
    before = _served_articles(engine, item_ids)
    try:
        with _kept_alive(engine, holder, held):
            write(_rederive(source.cfg), source.entry, source.ctype, source.transport, versions)
    except leases.LeaseLost:
        raise
    except Exception:  # noqa: BLE001 - one source's failed write leaves its items due, nothing else
        logger.exception(
            "news re-check: writing %d version(s) of %s failed", len(versions), source.natural_id
        )
        done["write_failed"] += len(versions)
        return
    after = _served_articles(engine, item_ids)
    for found, state, got in changed:
        # The pipeline run never raises for its row sync (a refused or deferred sync is an
        # outcome): a change counts as written only when the rows serve a new article document.
        if after.get(found.item_id, frozenset()) - before.get(found.item_id, frozenset()):
            _record(engine, holder, held, found, state, got, when)
            done[recheck.CHANGED] += 1
        else:
            done["not_synced"] += 1
            logger.error(
                "news re-check: %s changed and its files were written, but the rows hold no "
                "new version (the row sync was refused or deferred); left due",
                found.natural_id,
            )


def _served_articles(engine: Engine, item_ids: Sequence[int]) -> Dict[int, frozenset[int]]:
    """Each item's served ``article`` documents: the ones no row supersedes."""
    d, newer = documents.c, documents.alias("newer")
    with engine.connect() as conn:
        found = conn.execute(
            select(d.item_id, d.document_id).where(
                d.item_id.in_(list(item_ids)),
                d.kind == ARTICLE,
                ~exists().where(newer.c.supersedes_id == d.document_id),
            )
        ).all()
    out: Dict[int, set[int]] = {}
    for item_id, document_id in found:
        out.setdefault(int(item_id), set()).add(int(document_id))
    return {k: frozenset(v) for k, v in out.items()}


def _rederive(cfg: Any) -> Any:
    """``cfg`` as a rederive run of its source: ``run_text_items`` refuses any other stage."""
    from ..workflow.text_in import REDERIVE_ONLY

    if str(getattr(cfg, "pipeline_stage", "") or "") == REDERIVE_ONLY:
        return cfg
    if hasattr(cfg, "model_copy"):
        return cfg.model_copy(update={"pipeline_stage": REDERIVE_ONLY})
    again = copy.copy(cfg)
    setattr(again, "pipeline_stage", REDERIVE_ONLY)
    return again


@contextmanager
def _kept_alive(engine: Engine, holder: str, held: int, every_s: float = 60.0) -> Iterator[None]:
    """Extend the run's leases every ``every_s`` while a source's write runs: one write can take
    longer than the leases' 10 minutes. A lease found lost is raised when the write returns."""
    stop = threading.Event()
    lost: List[leases.LeaseLost] = []

    def beat() -> None:
        while not stop.wait(every_s):
            try:
                with engine.begin() as conn:
                    leases.heartbeat(conn, holder, held)
            except leases.LeaseLost as exc:
                lost.append(exc)
                return
            except Exception:  # noqa: BLE001 - a missed beat is retried on the next one
                logger.warning("news re-check: a lease heartbeat failed", exc_info=True)

    thread = threading.Thread(target=beat, name="news-recheck-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
    if lost:
        raise lost[0]


def _record(
    engine: Engine,
    holder: str,
    held: int,
    found: rows.Due,
    state: recheck.RecheckState,
    got: recheck_fetch.Check,
    when: datetime,
) -> None:
    with engine.begin() as conn:
        leases.heartbeat(conn, holder, held)
        rows.record(conn, found.item_id, state, got.validators, when)


def _skip(
    engine: Engine,
    holder: str,
    held: int,
    found: rows.Due,
    status: str,
    when: datetime,
    done: Counter[str],
) -> None:
    """Record a check that could not be made, so the item's schedule moves on."""
    state, _ = recheck.advance(found.state, status)
    with engine.begin() as conn:
        leases.heartbeat(conn, holder, held)
        rows.record(conn, found.item_id, state, found.validators, when)
    done[status] += 1


def _withdraw(
    engine: Engine,
    holder: str,
    held: int,
    found: rows.Due,
    state: recheck.RecheckState,
    got: recheck_fetch.Check,
    when: datetime,
) -> bool:
    """Withdraw the item and record its last check in one transaction, under its source's
    exclusive lease (taken by a holder of its own, as the row sync takes it)."""
    source_holder = f"{holder}:{found.source_natural_id}"
    source_held = leases.acquire(
        engine,
        [
            leases.LeaseRequest(leases.CORPUS, leases.SHARED),
            leases.LeaseRequest(leases.source_resource(found.source_natural_id), leases.EXCLUSIVE),
        ],
        source_holder,
    )
    try:
        with engine.begin() as conn:
            leases.heartbeat(conn, holder, held)
            leases.heartbeat(conn, source_holder, source_held)
            withdrawn = rows.withdraw(conn, found.item_id)
            rows.record(conn, found.item_id, state, got.validators, when)
        return withdrawn
    finally:
        leases.release(engine, source_holder)


__all__ = [
    "GUARDIAN",
    "NO_BODY",
    "PAGE_EXTRACTIONS",
    "RESOURCE",
    "run",
    "SourceRun",
    "UNSUPPORTED",
]
