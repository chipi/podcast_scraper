"""The graph API's presets served once per revision (ADR-183 decision 11).

A preset is computed once per corpus revision and projections build in each process, keyed on its
name and parameters, with one build per key at a time (``perf_cache``'s single flight). Its ETag is
``"<revision>:<hash>"``; a client sending it back as ``If-None-Match`` gets 304. Rule 7 (database,
2026-11-27): a build publish does not move ``corpus_revision``, so the key's token is the pair
(revision, ``graph_build_id``), read by one probe statement before the cache is asked.

Not cached: a request with ``exclude_items`` (the held set is one client's, so its key would never
be asked again). The namespace is cleared when it passes :data:`MAX_ENTRIES` keys: crude, but
bounded; the next requests rebuild what they ask.

Generic queries (``POST /api/graph/query``) are not served here: they are computed per request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from sqlalchemy import select
from sqlalchemy.engine import Connection, Engine

from podcast_scraper import perf_cache
from podcast_scraper.db import graph

NAMESPACE = "graph_presets"
MAX_ENTRIES = 512
_TOKEN_BITS = (1 << 52) - 1

_FUNCS: dict[str, Callable[..., dict[str, Any]]] = {
    "slice.recent": graph.slice_recent,
    "slice.items": graph.slice_items,
    "slice.neighbourhood": graph.slice_neighbourhood,
    "slice.group": graph.slice_group,
    "groups.summary": graph.groups_summary,
}


class UnknownPreset(ValueError):
    """A preset name the engine does not serve (422 at the route)."""


@dataclass(frozen=True)
class Served:
    """A preset's answer and the ETag naming it."""

    body: dict[str, Any]
    etag: str


def _probe(engine: Engine) -> tuple[Optional[int], Optional[int]]:
    from podcast_scraper.db.engines import read_connection

    r = graph._revision()
    with read_connection(engine) as conn:
        row = conn.execute(select(r.c.revision, r.c.graph_build_id)).one_or_none()
    return (row.revision, row.graph_build_id) if row is not None else (None, None)


def etag(name: str, params: dict[str, Any], revision: Any, build: Any) -> str:
    """``"<revision>:<hash>"``: the hash covers the preset, its parameters and the build."""
    canon = json.dumps([name, params, build], sort_keys=True, default=str)
    return f'"{revision}:{hashlib.sha256(canon.encode()).hexdigest()[:16]}"'


def serve(engine: Engine, name: str, params: dict[str, Any]) -> Served:
    """Preset ``name`` with ``params`` (its function's keyword arguments, ``conn`` aside), from
    the cache when this revision and build already answered it. A bad parameter raises the
    engine's ``ValueError`` (or :class:`UnknownPreset`), which the route turns into 422."""
    fn = _FUNCS.get(name)
    if fn is None:
        raise UnknownPreset(f"preset {name!r} is not one of {sorted(_FUNCS)}")

    def compute() -> dict[str, Any]:
        def run(conn: Connection) -> dict[str, Any]:
            return fn(conn, **params)

        return graph.read_slice(engine, run)

    revision, build = _probe(engine)
    if params.get("exclude_items"):
        body = compute()
    else:
        if perf_cache.stats().get(NAMESPACE, {}).get("entries", 0) >= MAX_ENTRIES:
            perf_cache.clear(NAMESPACE)
        key = json.dumps([name, params], sort_keys=True, default=str)
        token = float(hash((revision, build)) & _TOKEN_BITS)
        body = perf_cache.get_or_compute(NAMESPACE, key, token, compute)
    # The body's own revision names it: a write between the probe and the read is in the body.
    return Served(
        body,
        etag(
            name, params, body.get("corpus_revision", revision), body.get("graph_build_id", build)
        ),
    )
