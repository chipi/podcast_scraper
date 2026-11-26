"""Corpus job leases (DECISIONS section 4): ``job_leases`` rows replace the file lock.

A lease is on a *resource*: ``corpus`` (the whole corpus), ``source:`` plus a source's natural id,
or ``build:`` plus a corpus-wide build's kind. Its *mode* is ``shared`` or ``exclusive``: an
exclusive lease is granted only when the resource has no unexpired holder, a shared one only when
it has no unexpired exclusive holder. An expired lease counts as free and is removed when the
resource is next taken. A request that cannot be granted is refused at once, never waited for.

:func:`acquire` takes several leases all or nothing, in the order given (``corpus`` first, then a
source or build lease), each under a transaction-scoped advisory lock on its resource so two
takers cannot both see it free. Leases are taken outside the job's own transactions. A job's
chunk transaction starts with :func:`heartbeat`, which extends its leases only while they are
unexpired: when it touches nothing the job has lost them and must stop without committing.

Who takes what after the cut-over: an ingest of a source, ``corpus`` shared then its
``source:`` exclusive; an enricher, ``corpus`` shared; a corpus-wide build, ``corpus`` shared then
its ``build:`` exclusive; a row migration and the schema migrate step, ``corpus`` exclusive.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Sequence

from sqlalchemy import and_, delete, func, insert, select, text, update
from sqlalchemy.engine import Connection, Engine

from podcast_scraper.db.schema import job_leases

CORPUS = "corpus"
SHARED = "shared"
EXCLUSIVE = "exclusive"

_MODES = (SHARED, EXCLUSIVE)


class LeaseRefused(RuntimeError):
    """A lease is held in a conflicting mode; nothing was taken."""


@dataclass(frozen=True)
class LeaseRequest:
    """One lease to take: its resource and mode."""

    resource: str
    mode: str


def source_resource(source_natural_id: str) -> str:
    """The lease resource of one source."""
    return f"source:{source_natural_id}"


def build_resource(kind: str) -> str:
    """The lease resource of a corpus-wide build (``identity``, ``clusters``, ``projections``)."""
    return f"build:{kind}"


def _take(
    conn: Connection, request: LeaseRequest, holder: str, ttl: timedelta, job_id: Optional[int]
) -> None:
    if request.mode not in _MODES:
        raise ValueError(f"lease mode {request.mode!r} is not shared or exclusive")
    conn.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:r, 0))"), {"r": request.resource}
    )
    conn.execute(
        delete(job_leases).where(
            and_(job_leases.c.resource == request.resource, job_leases.c.expires_at <= func.now())
        )
    )
    held = conn.execute(
        select(job_leases.c.holder, job_leases.c.mode).where(
            and_(job_leases.c.resource == request.resource, job_leases.c.holder != holder)
        )
    ).all()
    conflicts = [
        f"{row.holder} ({row.mode})"
        for row in held
        if request.mode == EXCLUSIVE or row.mode == EXCLUSIVE
    ]
    if conflicts:
        raise LeaseRefused(
            f"{request.resource} {request.mode} refused: held by {', '.join(conflicts)}"
        )
    expires = func.now() + ttl
    existing = conn.execute(
        update(job_leases)
        .where(and_(job_leases.c.resource == request.resource, job_leases.c.holder == holder))
        .values(mode=request.mode, expires_at=expires, heartbeat_at=func.now(), job_id=job_id)
    ).rowcount
    if not existing:
        conn.execute(
            insert(job_leases).values(
                resource=request.resource,
                mode=request.mode,
                holder=holder,
                job_id=job_id,
                expires_at=expires,
            )
        )


def acquire(
    engine: Engine,
    requests: Sequence[LeaseRequest],
    holder: str,
    *,
    ttl: timedelta = timedelta(minutes=10),
    job_id: Optional[int] = None,
) -> None:
    """Take every lease in ``requests`` for ``holder``, all or nothing, in one short transaction.

    Raises :class:`LeaseRefused` (and takes none) when any is held in a conflicting mode.
    """
    with engine.begin() as conn:
        for request in requests:
            _take(conn, request, holder, ttl, job_id)


def heartbeat(conn: Connection, holder: str, *, ttl: timedelta = timedelta(minutes=10)) -> int:
    """Extend ``holder``'s unexpired leases; the first statement of each chunk transaction.

    Returns how many leases it extended: 0 means the leases expired and the job must stop.
    """
    return int(
        conn.execute(
            update(job_leases)
            .where(and_(job_leases.c.holder == holder, job_leases.c.expires_at > func.now()))
            .values(expires_at=func.now() + ttl, heartbeat_at=func.now())
        ).rowcount
    )


def release(engine: Engine, holder: str) -> int:
    """Drop every lease ``holder`` holds; returns how many."""
    with engine.begin() as conn:
        return int(conn.execute(delete(job_leases).where(job_leases.c.holder == holder)).rowcount)
