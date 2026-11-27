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
change, so waiting for its next envelope could leave its build behind its files for weeks.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import uuid
from pathlib import Path

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


def pending(corpus_root: Path) -> set[str]:
    """The enrichers to publish after their run whether or not they wrote an envelope. A file that
    is not a list of ids reads as empty, logged: it must not fail the run, and the next refused or
    failed publish rewrites it."""
    path = pending_path(corpus_root)
    try:
        ids = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    except (OSError, ValueError) as exc:
        logger.warning("enrichment rows: %s is unreadable, read as empty: %s", path, exc)
        return set()
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        logger.warning("enrichment rows: %s is not a list of enricher ids, read as empty", path)
        return set()
    return set(ids)


def _mark(corpus_root: Path, enricher_id: str, *, landed: bool) -> None:
    from podcast_scraper.utils.atomic_io import write_json_atomic

    # Enrichers publish from worker threads at once (executor ``asyncio.to_thread``).
    with _PENDING_LOCK:
        ids = pending(corpus_root)
        if (enricher_id in ids) != landed:
            return
        ids.symmetric_difference_update({enricher_id})
        path = pending_path(corpus_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, sorted(ids))


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


def _publish(corpus_root: Path, enricher_id: str) -> None:
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
            if groupings:
                logger.info("groupings rows: published %s", publish_groupings(conn, corpus_root))
    finally:
        leases.release(engine, holder)


def publish_after_run(corpus_root: Path, enricher_id: str) -> str:
    """Publish ``enricher_id``'s outputs under ``corpus_root`` as its new build (and, after the
    groupings' enricher, the groupings and then the projections); returns and counts the
    outcome: ``published``, ``not_configured`` (no pipeline database URL), ``lease_refused``
    or ``failed``. The last two leave the enricher :func:`pending`, so the next run publishes it
    again; ``published`` clears it."""
    from podcast_scraper.db import corpus_writer
    from podcast_scraper.db.settings import pipeline_is_configured

    if not pipeline_is_configured():
        OUTCOMES["not_configured"] += 1
        return "not_configured"
    try:
        _publish(Path(corpus_root), enricher_id)
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
    OUTCOMES["published"] += 1
    _mark(corpus_root, enricher_id, landed=True)
    if enricher_id == GROUPINGS_ENRICHER:
        from podcast_scraper.db import projections

        projections.after_batch()
    return "published"


__all__ = [
    "enricher_resource",
    "GROUPINGS_ENRICHER",
    "GROUPINGS_RESOURCE",
    "OUTCOMES",
    "pending",
    "pending_path",
    "publish_after_run",
    "served_metadata_relpaths",
]
