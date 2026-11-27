"""Corpus job leases (DECISIONS section 4): ``job_leases`` rows replace the file lock.

A lease is on a *resource*: ``corpus`` (the whole corpus), ``source:`` plus a source's natural id,
or ``build:`` plus a corpus-wide build's kind. Its *mode* is ``shared`` or ``exclusive``: an
exclusive lease is granted only when the resource has no unexpired holder, a shared one only when
it has no unexpired exclusive holder. An expired lease counts as free and is removed when the
resource is next taken. A request that cannot be granted is refused at once, never queued: the
only wait is for a chunk transaction that is extending the lease to commit (its heartbeat holds
the row), after which the lease is no longer expired and the request is refused.

:func:`acquire` takes several leases all or nothing, in the order given (``corpus`` first, then a
source or build lease), each under a transaction-scoped advisory lock on its resource so two
takers cannot both see it free, and returns how many leases the holder then holds. Leases are
taken outside the job's own transactions. A job's chunk transaction starts with :func:`heartbeat`
given that number: it raises :class:`LeaseLost` unless every one of them is still unexpired, so a
job that lost any lease (one expired while another was refreshed) stops without committing.

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


class LeaseLost(RuntimeError):
    """A job's lease expired; its transaction must roll back and the job stop."""


@dataclass(frozen=True)
class LeaseRequest:
    """One lease to take: its resource and mode."""

    resource: str
    mode: str


def source_resource(source_natural_id: str) -> str:
    """The lease resource of one source."""
    return f"source:{source_natural_id}"


def build_resource(kind: str) -> str:
    """The lease resource of a corpus-wide build: ``identity``, ``groupings`` (both groupings
    writers) or ``projections``."""
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
) -> int:
    """Take every lease in ``requests`` for ``holder``, all or nothing, in one short transaction,
    and return how many leases ``holder`` holds afterwards (what :func:`heartbeat` checks).

    Raises :class:`LeaseRefused` (and takes none) when any is held in a conflicting mode.
    """
    with engine.begin() as conn:
        for request in requests:
            _take(conn, request, holder, ttl, job_id)
        return int(
            conn.execute(select(func.count()).where(job_leases.c.holder == holder)).scalar_one()
        )


def heartbeat(
    conn: Connection, holder: str, held: int, *, ttl: timedelta = timedelta(minutes=10)
) -> None:
    """Extend ``holder``'s ``held`` leases (``acquire``'s count); the first statement of each
    chunk transaction.

    Raises :class:`LeaseLost` when fewer than ``held`` are unexpired: the caller's transaction
    then rolls back, the partial extension with it.
    """
    extended = conn.execute(
        update(job_leases)
        .where(and_(job_leases.c.holder == holder, job_leases.c.expires_at > func.now()))
        .values(expires_at=func.now() + ttl, heartbeat_at=func.now())
    ).rowcount
    if extended != held:
        raise LeaseLost(f"{holder}: {extended} of its {held} leases are unexpired")


def release(engine: Engine, holder: str) -> int:
    """Drop every lease ``holder`` holds; returns how many."""
    with engine.begin() as conn:
        return int(conn.execute(delete(job_leases).where(job_leases.c.holder == holder)).rowcount)
