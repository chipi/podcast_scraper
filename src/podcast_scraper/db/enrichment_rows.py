"""An enricher's outputs onto the corpus rows after its run (tracker w-public, w-common split).

Every enricher's envelope, Common's included, is written by ``enrichment.executor``; after one
enricher's run there, :func:`publish_after_run` publishes that enricher's outputs as a new
enrichment build (``importers.corpus_enrichments.publish_enrichments``). A build holds the whole
enricher, so it is read from every served item's metadata file, not only the episodes this run
touched: the staleness gate skips unchanged episodes, and a build of only the touched ones would
drop the rest.

After ``topic_theme_clusters`` (its envelope holds the storylines and super-themes) the groupings
are published too (``importers.corpus_groupings.publish_groupings``: every grouping family as one
build), and then the projections (``db.projections``), so ``cluster_reach`` counts the build just
published. Leases: ``corpus`` shared, then ``build:enrichment:<id>`` exclusive, and
``build:groupings`` exclusive for the groupings (w-leases).

A failure is logged and counted; it raises only with ``PODCAST_CORPUS_WRITE_STRICT`` set, as the
row writer (``corpus_writer``) does. An enricher whose publish did not land stays in the pending
file beside the health file until one does, and the executor publishes a pending enricher after
its run even when it wrote nothing: the staleness gate skips an enricher whose inputs did not
change, so waiting for its next envelope could leave its build behind its files for weeks. A
publish that left out an episode the run wrote (its item row not synced yet, so not served) has
not landed either: the pending entry keeps those metadata files, and every later publish checks
them again, so a retry the staleness gate leaves with nothing to write still sees them. An
unreadable pending file is set aside, not overwritten (:func:`take_unreadable`), and that run
publishes every enricher it runs.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import select
from sqlalchemy.engine import Connection

from podcast_scraper.db import leases
from podcast_scraper.db.schema import content_items

logger = logging.getLogger(__name__)

#: The enricher whose envelope holds the storylines and super-themes.
GROUPINGS_ENRICHER = "topic_theme_clusters"
GROUPINGS_RESOURCE = leases.build_resource("groupings")

#: How each :func:`publish_after_run` ended, since the process started.
OUTCOMES: collections.Counter[str] = collections.Counter()


_PENDING_LOCK = threading.Lock()


def pending_path(corpus_root: Path) -> Path:
    """The enrichers whose last publish did not land: beside ``enrichment_health.json``."""
    from podcast_scraper.enrichment.paths import enrichment_health_path

    return enrichment_health_path(Path(corpus_root)).with_name("enrichment_rows_pending.json")


#: How many left-out metadata files a warning names.
_NAMED = 5


def _entry_ok(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    left_out = entry.get("left_out", [])
    since = entry.get("since")
    return (
        isinstance(left_out, list)
        and all(isinstance(r, str) for r in left_out)
        and (since is None or isinstance(since, str))
    )


def _read(path: Path) -> Optional[dict[str, dict[str, Any]]]:
    """The pending entries by enricher id, each ``{"since": <UTC ISO>, "left_out": [relpath]}``
    (a list of ids, #2620's form, reads as entries without either); ``None`` when the file is
    there but is neither."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("enrichment rows: %s is unreadable: %s", path, exc)
        return None
    if isinstance(raw, list) and all(isinstance(i, str) for i in raw):
        return {i: {} for i in raw}
    if isinstance(raw, dict) and all(_entry_ok(e) for e in raw.values()):
        return {str(k): dict(v) for k, v in raw.items()}
    logger.warning("enrichment rows: %s is not a map of pending enrichers", path)
    return None


def pending(corpus_root: Path) -> set[str]:
    """The enrichers to publish after their run whether or not they wrote an envelope. An
    unreadable file reads as empty, logged: it must not fail the run (the executor sets it aside
    first, :func:`take_unreadable`)."""
    return set(_read(pending_path(corpus_root)) or {})


def left_out(corpus_root: Path, enricher_id: str) -> set[str]:
    """The metadata files ``enricher_id``'s last publish left out (not served then)."""
    entry = (_read(pending_path(corpus_root)) or {}).get(enricher_id, {})
    return set(entry.get("left_out", []))


def take_unreadable(corpus_root: Path) -> bool:
    """Set an unreadable pending file aside (``<name>.unreadable``, the last one kept) and return
    ``True``: which enrichers it named is lost, so the caller publishes every enricher it runs.
    ``False`` when the file is readable or absent."""
    path = pending_path(corpus_root)
    with _PENDING_LOCK:
        if _read(path) is not None:
            return False
        _set_aside(path)
        return True


def _set_aside(path: Path) -> None:
    aside = path.with_name(path.name + ".unreadable")
    try:
        path.replace(aside)
    except OSError as exc:
        logger.warning("enrichment rows: could not set %s aside: %s", path, exc)
    else:
        logger.warning("enrichment rows: set %s aside as %s", path, aside.name)


def _mark(
    corpus_root: Path,
    enricher_id: str,
    *,
    landed: bool,
    left: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Clear ``enricher_id``'s entry (``landed``) or keep it pending, with ``left`` as its
    left-out files when given (``None`` keeps the ones it has); returns the entry, empty once
    cleared. A file that became unreadable since the run started is set aside first."""
    from podcast_scraper.utils.atomic_io import write_json_atomic

    # Enrichers publish from worker threads at once (executor ``asyncio.to_thread``).
    with _PENDING_LOCK:
        path = pending_path(corpus_root)
        entries = _read(path)
        if entries is None:
            _set_aside(path)
            entries = {}
        before = entries.get(enricher_id)
        if landed:
            if before is None:
                return {}
            del entries[enricher_id]
            entry: dict[str, Any] = {}
        else:
            entry = dict(before or {})
            entry.setdefault("since", datetime.now(timezone.utc).isoformat(timespec="seconds"))
            if left is not None:
                entry["left_out"] = sorted(set(left))
            if entry == before:
                return entry
            entries[enricher_id] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, {k: entries[k] for k in sorted(entries)})
        return entry


def enricher_resource(enricher_id: str) -> str:
    """The lease resource of one enricher's build."""
    return leases.build_resource(f"enrichment:{enricher_id}")


def served_metadata_relpaths(conn: Connection) -> list[str]:
    """Every served item's metadata file, relative to the corpus root (``type_attrs``'
    ``metadata_relative_path``, the field the catalog reads)."""
    relpath = content_items.c.type_attrs["metadata_relative_path"].astext
    rows = conn.execute(
        select(relpath).where(content_items.c.served > 0, relpath.is_not(None)).order_by(relpath)
    )
    return [str(row[0]) for row in rows]


def _publish(corpus_root: Path, enricher_id: str, touched: set[str]) -> tuple[int, set[str]]:
    """Publish in one transaction under the leases; the builds written and the episodes of
    ``touched`` the build left out."""
    from podcast_scraper.db.engines import pipeline_rw_engine
    from podcast_scraper.db.importers.corpus_enrichments import publish_enrichments
    from podcast_scraper.db.importers.corpus_groupings import publish_groupings

    groupings = enricher_id == GROUPINGS_ENRICHER
    wanted = [
        leases.LeaseRequest(leases.CORPUS, leases.SHARED),
        leases.LeaseRequest(enricher_resource(enricher_id), leases.EXCLUSIVE),
    ]
    if groupings:
        wanted.append(leases.LeaseRequest(GROUPINGS_RESOURCE, leases.EXCLUSIVE))
    engine = pipeline_rw_engine()
    holder = f"enrichment:{uuid.uuid4().hex[:12]}:{enricher_id}"
    taken = leases.acquire(engine, wanted, holder)
    try:
        with engine.begin() as conn:
            leases.heartbeat(conn, holder, taken)
            relpaths = served_metadata_relpaths(conn)
            written = publish_enrichments(conn, corpus_root, relpaths, enricher_id)
            logger.info("enrichment rows: %s published %s", enricher_id, written)
            if groupings and written.builds:
                logger.info("groupings rows: published %s", publish_groupings(conn, corpus_root))
    finally:
        leases.release(engine, holder)
    left_out = set(touched) if not written.builds else set(touched) - set(relpaths)
    return written.builds, left_out


def publish_after_run(corpus_root: Path, enricher_id: str, *, touched: Iterable[str] = ()) -> str:
    """Publish ``enricher_id``'s outputs under ``corpus_root`` as its new build (and, after the
    groupings' enricher, the groupings and then the projections); returns and counts the
    outcome. ``touched`` is the metadata files (relative to the root) of the episodes the run
    wrote an envelope for; the ones an earlier publish left out (:func:`left_out`) are checked
    with them.

    - ``published`` clears :func:`pending`.
    - ``left_out``: a build was published (or none could be) without one of those episodes,
      whose item is not served yet. The enricher stays pending with them, so a later run
      publishes again and checks them.
    - ``nothing_to_publish``: no readable ``ok`` output and nothing to check; clears pending.
    - ``lease_refused`` and ``failed`` leave it pending, its left-out files kept.
    - ``not_configured``: no pipeline database URL.
    """
    from podcast_scraper.db import corpus_writer
    from podcast_scraper.db.settings import pipeline_is_configured

    if not pipeline_is_configured():
        OUTCOMES["not_configured"] += 1
        return "not_configured"
    check = set(touched) | left_out(Path(corpus_root), enricher_id)
    try:
        builds, missing = _publish(Path(corpus_root), enricher_id, check)
    except leases.LeaseRefused:
        OUTCOMES["lease_refused"] += 1
        _mark(corpus_root, enricher_id, landed=False)
        return "lease_refused"
    except Exception as exc:  # noqa: BLE001 - the files stay; the next run publishes again
        logger.warning("enrichment rows: %s failed: %s", enricher_id, exc)
        OUTCOMES["failed"] += 1
        _mark(corpus_root, enricher_id, landed=False)
        if corpus_writer._strict():
            raise corpus_writer.SyncFailed(f"enrichment rows for {enricher_id}: {exc}") from exc
        return "failed"
    if missing:
        outcome = "left_out"
        entry = _mark(corpus_root, enricher_id, landed=False, left=missing)
        named = sorted(missing)
        logger.warning(
            "enrichment rows: %s left out %d episode(s) not served yet, pending since %s: %s%s",
            enricher_id,
            len(named),
            entry.get("since"),
            ", ".join(named[:_NAMED]),
            f" and {len(named) - _NAMED} more" if len(named) > _NAMED else "",
        )
    else:
        outcome = "published" if builds else "nothing_to_publish"
        _mark(corpus_root, enricher_id, landed=True)
    OUTCOMES[outcome] += 1
    if builds and enricher_id == GROUPINGS_ENRICHER:
        from podcast_scraper.db import projections

        projections.after_batch()
    return outcome


__all__ = [
    "enricher_resource",
    "GROUPINGS_ENRICHER",
    "GROUPINGS_RESOURCE",
    "left_out",
    "OUTCOMES",
    "pending",
    "pending_path",
    "publish_after_run",
    "served_metadata_relpaths",
    "take_unreadable",
]
