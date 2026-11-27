"""The graph engine (ADR-183): bounded walks over the corpus rows, answered as ``graph-slice/1``.

One module of functions over the corpus tables (decision 1). Each request runs in one read-only,
repeatable-read transaction (:func:`read_slice`); inside it every step of a walk is one statement,
a ``UNION ALL`` over the step's edges, each frontier node joined ``LATERAL`` with at most
:data:`FANOUT` edges per edge type (decision 9, CRFC-005 §4). No recursive statement is used: it
cannot cap a level's frontier.

Built here, the walk core:

- the edges of decision 10 whose tables exist (:data:`EDGES`); ``CONTRIBUTES_TO``, ``RELATED_TO``
  and ``IN_GROUP`` are not built yet;
- the presets ``slice.items``, ``slice.recent`` (without ``expand_groups`` or an extension score)
  and ``slice.neighbourhood`` with the named paths of decision 6;
- a merged entity read per member and answered under its canonical id (CRFC-005 §4).

Not built yet: the groups catalogue and node ``groups`` (no ``clusters`` table until s-clusters),
``with: counts`` (``entity_item_counts``, r-projections), ``exclude_items`` and held stubs, the
structured query, the routes, caching and the replica retry.

Statement counts, against decision 9's table: the tail is 3 (nodes, items, induced edges), since
there is no groups catalogue to read yet; aggregated edges are counted during the walk, not by a
statement of their own; a path whose last step reaches entities reads their types in one more
statement, so ``keep`` can choose among them. ``tests/unit/podcast_scraper/db/test_db_graph.py``
asserts each count.

Node and edge ids in a slice are natural ids; an entity's is its canonical row's. Insights and
quotes are served only when visible and cut from their item's served document (CRFC-005 §3.4),
and items only when served.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar

from sqlalchemy import BigInteger, bindparam, exc as sa_exc, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.engine import Connection, Engine

from podcast_scraper.db.schema import CORPUS_SCHEMA

FORMAT = "graph-slice/1"
#: The most edges of one type a frontier node contributes to a step.
FANOUT = 200
DEFAULT_BUDGET = 1_500
MAX_BUDGET = 3_000
DEFAULT_EDGE_BUDGET = 10_000
MAX_EDGE_BUDGET = 20_000
MAX_ITEM_SEEDS = 200
#: CRFC-004's deadline; CRFC-005 §4 lowers it for presets only once the rehearsal confirms 2 s.
STATEMENT_TIMEOUT_MS = 5_000
#: ``slice.recent``'s windows, widened in this order while too few items match.
WINDOWS: dict[str, Optional[timedelta]] = {
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "90d": timedelta(days=90),
    "all": None,
}

ITEM, SOURCE, ENTITY, INSIGHT, EVIDENCE = "item", "source", "entity", "insight", "evidence"
#: A node inside the engine: its kind and its row's key.
Ref = tuple[str, int]

#: ``entities.type`` as an ontology v3 node type.
ENTITY_TYPES = {"person": "Person", "org": "Organization", "topic": "Topic", "object": "Object"}
NODE_TYPES = {ITEM: "ContentItem", SOURCE: "Source", INSIGHT: "Insight", EVIDENCE: "Quote"}

T = TypeVar("T")
_S = CORPUS_SCHEMA


class GraphTimeout(RuntimeError):
    """The request ran past its statement timeout: answered 503, never as a partial slice."""


class UnknownPath(ValueError):
    """A named path the engine does not have, or one not defined for the seed's type."""


def _current(alias: str) -> str:
    """The row was cut from its item's served document: nothing supersedes it."""
    return (
        f"NOT EXISTS (SELECT 1 FROM {_S}.documents d WHERE d.item_id = {alias}.item_id "
        f"AND d.supersedes_id = {alias}.document_id)"
    )


def _frontier(p: str) -> str:
    return f"unnest(CAST(:{p} AS bigint[])) AS f(id)"


def _canonical(column: str) -> str:
    return f"(SELECT e.canonical_id FROM {_S}.entities e WHERE e.entity_id = {column})"


def _per_member(p: str, inner: str, order: str) -> str:
    """A canonical entity's edges, read per member (one index range each), merged in ``order``
    and capped again, so a merged node fans out like any other."""
    return (
        f"FROM {_frontier(p)} CROSS JOIN LATERAL (SELECT y.* FROM {_S}.entities m "
        f"CROSS JOIN LATERAL ({inner} LIMIT {FANOUT}) y WHERE m.canonical_id = f.id "
        f"ORDER BY {order} LIMIT {FANOUT}) x"
    )


_SERVED = f"JOIN {_S}.content_items c ON c.item_id = {{0}} AND c.served > 0"
_VISIBLE = f"JOIN {_S}.insights i ON i.insight_id = {{0}} AND i.visible AND {_current('i')}"
_PARTICIPANT_ROLES = "('host', 'guest')"


def _sel(frm: str, dst: str, item: str, prop: str = "NULL::text") -> str:
    return f"SELECT {frm} AS frm, {dst} AS dst, {item} AS item_id, {prop} AS prop "


@dataclass(frozen=True)
class EdgeType:
    """One edge of decision 10: its ends, class, whether it belongs to an item, and the
    statement fragment that reads it from a frontier in each direction. ``out`` starts from the
    ``source`` end, ``in`` from the ``target`` end; both answer ``(frm, dst)`` in the edge's own
    direction."""

    name: str
    cls: str
    source: str
    target: str
    item_scoped: bool
    out: Callable[[str], str]
    in_: Callable[[str], str]


def _mention(kind: str) -> EdgeType:
    return EdgeType(
        kind,
        "descriptive",
        INSIGHT,
        ENTITY,
        True,
        lambda p: _sel("f.id", _canonical("x.entity_id"), "x.item_id")
        + f"FROM {_frontier(p)} CROSS JOIN LATERAL (SELECT im.entity_id, im.item_id "
        f"FROM {_S}.insight_mentions im WHERE im.insight_id = f.id AND im.kind = '{kind}' "
        f"ORDER BY im.entity_id LIMIT {FANOUT}) x",
        lambda p: _sel("x.insight_id", "f.id", "x.item_id")
        + _per_member(
            p,
            f"SELECT im.insight_id, im.item_id, im.rank FROM {_S}.insight_mentions im "
            f"{_VISIBLE.format('im.insight_id')} WHERE im.entity_id = m.entity_id "
            f"AND im.kind = '{kind}' ORDER BY im.rank, im.insight_id",
            "y.rank, y.insight_id",
        ),
    )


_EDGE_LIST = [
    EdgeType(
        "HAS_ITEM",
        "structural",
        SOURCE,
        ITEM,
        True,
        lambda p: _sel("f.id", "x.item_id", "x.item_id")
        + f"FROM {_frontier(p)} CROSS JOIN LATERAL (SELECT c.item_id FROM {_S}.content_items c "
        f"WHERE c.source_id = f.id AND c.served > 0 "
        f"ORDER BY c.published_at DESC NULLS LAST, c.item_id DESC LIMIT {FANOUT}) x",
        lambda p: _sel("c.source_id", "f.id", "f.id")
        + f"FROM {_frontier(p)} {_SERVED.format('f.id')}",
    ),
    EdgeType(
        "HAS_INSIGHT",
        "structural",
        ITEM,
        INSIGHT,
        True,
        lambda p: _sel("f.id", "x.insight_id", "f.id")
        + f"FROM {_frontier(p)} CROSS JOIN LATERAL (SELECT i.insight_id FROM {_S}.insights i "
        f"WHERE i.item_id = f.id AND i.visible AND {_current('i')} "
        f"ORDER BY i.rank, i.insight_id LIMIT {FANOUT}) x",
        lambda p: _sel("i.item_id", "f.id", "i.item_id")
        + f"FROM {_frontier(p)} {_VISIBLE.format('f.id')} {_SERVED.format('i.item_id')}",
    ),
    EdgeType(
        "PUBLISHED_BY",
        "structural",
        SOURCE,
        ENTITY,
        False,
        lambda p: _sel("f.id", _canonical("s.publisher_id"), "NULL::bigint")
        + f"FROM {_frontier(p)} JOIN {_S}.sources s ON s.source_id = f.id "
        f"AND s.publisher_id IS NOT NULL",
        lambda p: _sel("x.source_id", "f.id", "NULL::bigint")
        + _per_member(
            p,
            f"SELECT s.source_id FROM {_S}.sources s WHERE s.publisher_id = m.entity_id "
            f"ORDER BY s.source_id",
            "y.source_id",
        ),
    ),
    EdgeType(
        "PARTICIPATES_IN",
        "attribution",
        ENTITY,
        ITEM,
        True,
        lambda p: _sel("f.id", "x.item_id", "x.item_id", "x.role")
        + _per_member(
            p,
            f"SELECT r.item_id, r.role, c.published_at FROM {_S}.person_roles r "
            f"{_SERVED.format('r.item_id')} WHERE r.person_id = m.entity_id "
            f"AND r.role IN {_PARTICIPANT_ROLES} "
            f"ORDER BY c.published_at DESC NULLS LAST, r.item_id DESC",
            "y.published_at DESC NULLS LAST, y.item_id DESC",
        ),
        lambda p: _sel(_canonical("x.person_id"), "f.id", "f.id", "x.role")
        + f"FROM {_frontier(p)} CROSS JOIN LATERAL (SELECT r.person_id, r.role "
        f"FROM {_S}.person_roles r WHERE r.item_id = f.id AND r.role IN {_PARTICIPANT_ROLES} "
        f"ORDER BY r.person_id LIMIT {FANOUT}) x",
    ),
    EdgeType(
        "MENTIONS",
        "discovery",
        ENTITY,
        ITEM,
        True,
        # The index's order, (published_at DESC, item_id): Postgres puts undated rows first.
        lambda p: _sel("f.id", "x.item_id", "x.item_id")
        + _per_member(
            p,
            f"SELECT ie.item_id, ie.published_at FROM {_S}.item_entities ie "
            f"{_SERVED.format('ie.item_id')} WHERE ie.entity_id = m.entity_id "
            f"AND ie.role = 'mentions' ORDER BY ie.published_at DESC, ie.item_id",
            "y.published_at DESC, y.item_id",
        ),
        lambda p: _sel(_canonical("x.entity_id"), "f.id", "f.id")
        + f"FROM {_frontier(p)} CROSS JOIN LATERAL (SELECT ie.entity_id "
        f"FROM {_S}.item_entities ie WHERE ie.item_id = f.id AND ie.role = 'mentions' "
        f"ORDER BY ie.entity_id LIMIT {FANOUT}) x",
    ),
    _mention("ABOUT"),
    _mention("MENTIONS_PERSON"),
    _mention("MENTIONS_ORG"),
    _mention("MENTIONS_OBJECT"),
    EdgeType(
        "SUPPORTED_BY",
        "evidentiary",
        INSIGHT,
        EVIDENCE,
        True,
        lambda p: _sel("f.id", "x.evidence_id", "x.item_id")
        + f"FROM {_frontier(p)} CROSS JOIN LATERAL (SELECT s.evidence_id, ev.item_id "
        f"FROM {_S}.insight_support s JOIN {_S}.evidence ev ON ev.evidence_id = s.evidence_id "
        f"WHERE s.insight_id = f.id ORDER BY s.ord, s.evidence_id LIMIT {FANOUT}) x",
        lambda p: _sel("x.insight_id", "f.id", "x.item_id")
        + f"FROM {_frontier(p)} CROSS JOIN LATERAL (SELECT s.insight_id, i.item_id "
        f"FROM {_S}.insight_support s {_VISIBLE.format('s.insight_id')} "
        f"WHERE s.evidence_id = f.id ORDER BY i.rank, s.insight_id LIMIT {FANOUT}) x",
    ),
    EdgeType(
        "ATTRIBUTED_TO",
        "attribution",
        EVIDENCE,
        ENTITY,
        True,
        lambda p: _sel("f.id", _canonical("ev.speaker_id"), "ev.item_id")
        + f"FROM {_frontier(p)} JOIN {_S}.evidence ev ON ev.evidence_id = f.id "
        f"AND ev.speaker_id IS NOT NULL",
        lambda p: _sel("x.evidence_id", "f.id", "x.item_id")
        + _per_member(
            p,
            f"SELECT ev.evidence_id, ev.item_id FROM {_S}.evidence ev "
            f"WHERE ev.speaker_id = m.entity_id AND {_current('ev')} "
            f"ORDER BY ev.item_id DESC, ev.evidence_id",
            "y.item_id DESC, y.evidence_id",
        ),
    ),
    EdgeType(
        "STATED_BY",
        "attribution",
        INSIGHT,
        ENTITY,
        True,
        lambda p: _sel("f.id", _canonical("a.person_id"), "a.item_id")
        + f"FROM {_frontier(p)} JOIN {_S}.insight_attribution a ON a.insight_id = f.id",
        lambda p: _sel("x.insight_id", "f.id", "x.item_id")
        + _per_member(
            p,
            f"SELECT a.insight_id, a.item_id FROM {_S}.insight_attribution a "
            f"{_VISIBLE.format('a.insight_id')} WHERE a.person_id = m.entity_id "
            f"ORDER BY a.item_id DESC, a.insight_id",
            "y.item_id DESC, y.insight_id",
        ),
    ),
]
EDGES: dict[str, EdgeType] = {e.name: e for e in _EDGE_LIST}

#: A step: the edges it crosses, each with its direction.
Step = tuple[tuple[str, str], ...]
#: A named path: per seed type, its branches; a branch is a list of steps.
Branches = tuple[tuple[Step, ...], ...]


def _b(*steps: tuple[str, str]) -> tuple[Step, ...]:
    return tuple(((name, direction),) for name, direction in steps)


def _items_of(mention: str) -> Branches:
    return (_b(("MENTIONS", "out")), _b((mention, "in"), ("HAS_INSIGHT", "in")))


#: Decision 6's named paths: per seed type, the branches, and the node types the slice keeps.
PATHS: dict[str, tuple[dict[str, Branches], frozenset[str]]] = {
    "node.items": (
        {
            "Topic": _items_of("ABOUT"),
            "Person": (
                _b(("MENTIONS", "out")),
                _b(("PARTICIPATES_IN", "out")),
                _b(("MENTIONS_PERSON", "in"), ("HAS_INSIGHT", "in")),
                _b(("ATTRIBUTED_TO", "in"), ("SUPPORTED_BY", "in"), ("HAS_INSIGHT", "in")),
            ),
            "Organization": _items_of("MENTIONS_ORG"),
            "Object": _items_of("MENTIONS_OBJECT"),
        },
        frozenset({"ContentItem"}),
    ),
    "person.topics": (
        {"Person": (_b(("STATED_BY", "in"), ("ABOUT", "out")),)},
        frozenset({"Topic"}),
    ),
    "person.co_speakers": (
        {
            "Person": (
                _b(("STATED_BY", "in"), ("ABOUT", "out"), ("ABOUT", "in"), ("STATED_BY", "out")),
            )
        },
        frozenset({"Person"}),
    ),
    "topic.people": (
        {"Topic": (_b(("ABOUT", "in"), ("STATED_BY", "out")),)},
        frozenset({"Person"}),
    ),
    "topic.entities": (
        {
            "Topic": (
                (
                    (("ABOUT", "in"),),
                    (("MENTIONS_PERSON", "out"), ("MENTIONS_ORG", "out")),
                ),
            )
        },
        frozenset({"Person", "Organization"}),
    ),
    "org.topics": (
        {"Organization": (_b(("MENTIONS_ORG", "in"), ("ABOUT", "out")),)},
        frozenset({"Topic"}),
    ),
}

#: An item slice's two steps: the items' own edges, then their insights' edges.
_ITEM_STEP: Step = (
    ("HAS_ITEM", "in"),
    ("MENTIONS", "in"),
    ("PARTICIPATES_IN", "in"),
    ("HAS_INSIGHT", "out"),
)
_INSIGHT_STEP: Step = (
    ("ABOUT", "out"),
    ("MENTIONS_PERSON", "out"),
    ("MENTIONS_ORG", "out"),
    ("MENTIONS_OBJECT", "out"),
    ("STATED_BY", "out"),
)


def _ids(name: str) -> Any:
    return bindparam(name, type_=ARRAY(BigInteger))


def read_slice(
    engine: Engine,
    fn: Callable[[Connection], T],
    *,
    timeout_ms: int = STATEMENT_TIMEOUT_MS,
) -> T:
    """Run ``fn`` in one read-only, repeatable-read transaction with ``timeout_ms`` as its
    statement timeout (``SET LOCAL``: nothing outlives the transaction). A timeout raises
    :class:`GraphTimeout`."""
    with engine.connect() as raw:
        conn = raw.execution_options(isolation_level="REPEATABLE READ")
        try:
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                conn.execute(text(f"SET LOCAL statement_timeout = {int(timeout_ms)}"))
                return fn(conn)
        except sa_exc.DBAPIError as exc:
            if getattr(exc.orig, "sqlstate", None) == "57014":
                raise GraphTimeout(f"the graph read ran past {timeout_ms} ms") from exc
            raise


@dataclass
class _Hit:
    edge: str
    frm: Ref
    dst: Ref
    item_id: Optional[int]
    prop: Optional[str]


def _kind_end(edge: EdgeType, direction: str) -> tuple[str, str]:
    """The kinds of the node a part starts from and of the node it reaches."""
    return (edge.source, edge.target) if direction == "out" else (edge.target, edge.source)


def _step(
    conn: Connection, parts: Sequence[tuple[str, str, Sequence[int]]]
) -> list[tuple[int, _Hit]]:
    """One statement for a whole step: ``parts`` are ``(edge, direction, frontier ids)``. Each
    hit carries the index of its part."""
    selects: list[str] = []
    params: dict[str, list[int]] = {}
    binds = []
    for n, (name, direction, ids) in enumerate(parts):
        edge = EDGES[name]
        fragment = edge.out(f"p{n}") if direction == "out" else edge.in_(f"p{n}")
        selects.append(f"SELECT {n} AS part, q.* FROM ({fragment}) q")
        params[f"p{n}"] = list(ids)
        binds.append(_ids(f"p{n}"))
    statement = text(" UNION ALL ".join(selects)).bindparams(*binds)
    hits = []
    for row in conn.execute(statement, params):
        name, direction, _ids_ = parts[row.part]
        edge = EDGES[name]
        hits.append(
            (
                row.part,
                _Hit(
                    name,
                    (edge.source, row.frm),
                    (edge.target, row.dst),
                    row.item_id,
                    row.prop,
                ),
            )
        )
    return hits


@dataclass
class _Slice:
    """What a request collects before the tail statements turn it into ``graph-slice/1``."""

    preset: str
    params: dict[str, Any]
    revision: Optional[int] = None
    budget: int = DEFAULT_BUDGET
    edge_budget: int = DEFAULT_EDGE_BUDGET
    nodes: dict[Ref, None] = field(default_factory=dict)
    items: dict[int, None] = field(default_factory=dict)
    aggregated: dict[tuple[Ref, Ref], int] = field(default_factory=dict)
    reasons: set[str] = field(default_factory=set)
    fanout_capped: set[Ref] = field(default_factory=set)
    nodes_left: int = 0
    items_left: Optional[int] = None
    applied_window: Optional[str] = None
    unknown_seeds: list[str] = field(default_factory=list)

    def admit(self, ref: Ref) -> None:
        self.nodes.setdefault(ref, None)
        if ref[0] == ITEM:
            self.items.setdefault(ref[1], None)


def _note_fanout(s: _Slice, parts: Sequence[tuple[str, str, Sequence[int]]], hits: Any) -> None:
    counts: dict[tuple[int, int], int] = {}
    for part, hit in hits:
        name, direction, _ = parts[part]
        start = hit.frm if direction == "out" else hit.dst
        counts[(part, start[1])] = counts.get((part, start[1]), 0) + 1
    for (part, start_id), n in counts.items():
        if n >= FANOUT:
            name, direction, _ = parts[part]
            kind, _ = _kind_end(EDGES[name], direction)
            s.fanout_capped.add((kind, start_id))
            s.reasons.add("fanout")


def _check_budgets(budget: int, edge_budget: int) -> None:
    if not 1 <= budget <= MAX_BUDGET:
        raise ValueError(f"budget {budget} is not between 1 and {MAX_BUDGET}")
    if not 1 <= edge_budget <= MAX_EDGE_BUDGET:
        raise ValueError(f"edge_budget {edge_budget} is not between 1 and {MAX_EDGE_BUDGET}")


# --- item slices -----------------------------------------------------------------------------


def _fill_items(conn: Connection, s: _Slice, item_ids: Sequence[int]) -> None:
    """Decision 9's item mode: the items in order, then round-robin across them, each item's KG
    nodes first, then its insights by rank; admitting an insight admits the nodes it reaches."""
    items = list(dict.fromkeys(item_ids))[: s.budget]
    if not items:
        return
    for item in items:
        s.admit((ITEM, item))
    first_parts = [(name, direction, items) for name, direction in _ITEM_STEP]
    first = _step(conn, first_parts)
    _note_fanout(s, first_parts, first)
    kg: dict[int, list[Ref]] = {item: [] for item in items}
    ranked: dict[int, list[int]] = {item: [] for item in items}
    for _, hit in first:
        if hit.edge == "HAS_INSIGHT":
            ranked[hit.frm[1]].append(hit.dst[1])
        else:
            item = hit.dst[1]
            other = hit.frm
            if other not in kg[item]:
                kg[item].append(other)
    insights = [i for item in items for i in ranked[item]]
    reach: dict[int, list[Ref]] = {i: [] for i in insights}
    if insights:
        second_parts = [(name, direction, insights) for name, direction in _INSIGHT_STEP]
        second = _step(conn, second_parts)
        _note_fanout(s, second_parts, second)
        for _, hit in second:
            if hit.dst not in reach[hit.frm[1]]:
                reach[hit.frm[1]].append(hit.dst)
    queues = {
        item: [[ref] for ref in kg[item]] + [[(INSIGHT, i), *reach[i]] for i in ranked[item]]
        for item in items
    }
    left = 0
    stopped = False
    for depth in range(max((len(q) for q in queues.values()), default=0)):
        for item in items:
            if depth >= len(queues[item]):
                continue
            unit = [ref for ref in queues[item][depth] if ref not in s.nodes]
            if stopped or len(s.nodes) + len(unit) > s.budget:
                stopped = True
                left += len(unit)
                continue
            for ref in unit:
                s.admit(ref)
    if stopped:
        s.reasons.add("budget")
        s.nodes_left = left
    # The items that own an admitted insight are in the slice already: every insight came from
    # one of ``items``.


def _first_statement_items(conn: Connection, natural_ids: Sequence[str]) -> tuple[Any, dict]:
    statement = text(
        f"SELECT r.revision, x.item_id, x.natural_id FROM "
        f"(SELECT revision FROM {_S}.corpus_revision WHERE id = 1) r "
        f"LEFT JOIN LATERAL (SELECT c.item_id, c.natural_id FROM {_S}.content_items c "
        f"WHERE c.natural_id = ANY(:ids) AND c.served > 0) x ON true"
    )
    rows = conn.execute(statement, {"ids": list(natural_ids)}).all()
    revision = rows[0].revision if rows else None
    return revision, {row.natural_id: row.item_id for row in rows if row.item_id is not None}


def slice_items(
    conn: Connection,
    natural_ids: Sequence[str],
    *,
    budget: int = DEFAULT_BUDGET,
    edge_budget: int = DEFAULT_EDGE_BUDGET,
) -> dict[str, Any]:
    """``slice.items``: these items' graph. Unknown or unserved items are ``unknown_seeds``."""
    _check_budgets(budget, edge_budget)
    seeds = list(dict.fromkeys(natural_ids))
    if len(seeds) > MAX_ITEM_SEEDS:
        raise ValueError(f"{len(seeds)} item ids is more than {MAX_ITEM_SEEDS}")
    s = _Slice("slice.items", {"items": seeds}, budget=budget, edge_budget=edge_budget)
    s.revision, found = _first_statement_items(conn, seeds)
    s.unknown_seeds = [n for n in seeds if n not in found]
    _fill_items(conn, s, [found[n] for n in seeds if n in found])
    return _finish(conn, s)


def slice_recent(
    conn: Connection,
    *,
    window: str = "7d",
    content_types: Optional[Sequence[str]] = None,
    sources: Optional[Sequence[str]] = None,
    min_items: int = 10,
    budget: int = DEFAULT_BUDGET,
    edge_budget: int = DEFAULT_EDGE_BUDGET,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """``slice.recent``: the newest served items. The window widens (7d, 30d, 90d, all) while
    fewer than ``min_items`` match; every widening is answered by the one candidate statement,
    newest first and at most ``budget`` items, and the response names ``applied_window``."""
    _check_budgets(budget, edge_budget)
    if window not in WINDOWS:
        raise ValueError(f"window {window!r} is not one of {sorted(WINDOWS)}")
    if min_items < 1:
        raise ValueError("min_items must be at least 1")
    instant = now or datetime.now(timezone.utc)
    params = {
        "window": window,
        "content_types": list(content_types) if content_types is not None else None,
        "sources": list(sources) if sources is not None else None,
        "min_items": min_items,
    }
    s = _Slice("slice.recent", params, budget=budget, edge_budget=edge_budget)
    conditions = ["c.served > 0"]
    values: dict[str, Any] = {"budget": budget}
    if content_types is not None:
        conditions.append("c.content_type = ANY(:types)")
        values["types"] = list(content_types)
    if sources is not None:
        conditions.append(
            f"c.source_id IN (SELECT so.source_id FROM {_S}.sources so "
            f"WHERE so.natural_id = ANY(:sources))"
        )
        values["sources"] = list(sources)
    statement = text(
        f"SELECT r.revision, x.item_id, x.published_at FROM "
        f"(SELECT revision FROM {_S}.corpus_revision WHERE id = 1) r "
        f"LEFT JOIN LATERAL (SELECT c.item_id, c.published_at FROM {_S}.content_items c "
        f"WHERE {' AND '.join(conditions)} "
        f"ORDER BY c.published_at DESC NULLS LAST, c.item_id DESC LIMIT :budget) x ON true"
    )
    rows = conn.execute(statement, values).all()
    s.revision = rows[0].revision if rows else None
    candidates = [(row.item_id, row.published_at) for row in rows if row.item_id is not None]
    names = list(WINDOWS)
    chosen: list[int] = []
    for name in names[names.index(window) :]:
        span = WINDOWS[name]
        start = instant - span if span is not None else None
        chosen = [
            item
            for item, published in candidates
            if start is None or (published is not None and published >= start)
        ]
        s.applied_window = name
        if len(chosen) >= min_items:
            break
    _fill_items(conn, s, chosen)
    return _finish(conn, s)


# --- path walks ------------------------------------------------------------------------------


def slice_neighbourhood(
    conn: Connection,
    node_id: str,
    path: str,
    *,
    budget: int = DEFAULT_BUDGET,
    edge_budget: int = DEFAULT_EDGE_BUDGET,
) -> dict[str, Any]:
    """``slice.neighbourhood``: the node, what the named path reaches from it, and the items
    those draw from. Nodes of a type the path does not keep are walked through, not returned;
    one aggregated edge (type = the path's name, weight = the number of paths) joins the seed
    to each kept node reached through them. A union path runs one statement per level."""
    _check_budgets(budget, edge_budget)
    if path not in PATHS:
        raise UnknownPath(f"{path!r} is not a named path")
    by_type, keep = PATHS[path]
    s = _Slice(
        "slice.neighbourhood",
        {"node": node_id, "path": path},
        budget=budget,
        edge_budget=edge_budget,
    )
    statement = text(
        f"SELECT r.revision, x.canonical_id, x.type FROM "
        f"(SELECT revision FROM {_S}.corpus_revision WHERE id = 1) r "
        f"LEFT JOIN LATERAL (SELECT e.canonical_id, k.type FROM {_S}.entities e "
        f"JOIN {_S}.entities k ON k.entity_id = e.canonical_id WHERE e.natural_id = :id) x ON true"
    )
    row = conn.execute(statement, {"id": node_id}).first()
    s.revision = row.revision if row is not None else None
    if row is None or row.canonical_id is None:
        s.unknown_seeds = [node_id]
        return _finish(conn, s)
    seed_type = ENTITY_TYPES.get(row.type)
    if seed_type not in by_type:
        raise UnknownPath(f"{path!r} is not defined for a {seed_type} seed")
    seed: Ref = (ENTITY, row.canonical_id)
    s.admit(seed)
    _walk(conn, s, seed, by_type[seed_type], keep)
    return _finish(conn, s)


def _type_of(ref: Ref, entity_types: dict[int, str]) -> Optional[str]:
    if ref[0] == ENTITY:
        return entity_types.get(ref[1])
    return NODE_TYPES[ref[0]]


def _walk(conn: Connection, s: _Slice, seed: Ref, branches: Branches, keep: frozenset) -> None:
    """Every branch from ``seed``, level by level. A frontier entry carries its branch and the
    number of paths that reach it; after each level the new frontier is cut to the budget left."""
    frontier: list[tuple[int, Ref, int, frozenset[int]]] = [
        (b, seed, 1, frozenset()) for b in range(len(branches))
    ]
    # A reached node: the number of paths to it and the items those paths draw from, so the
    # slice's items are those of the nodes it returns, not of the paths the budget cut.
    reached: dict[Ref, tuple[int, set[int]]] = {}
    depth = 0
    while frontier:
        parts: list[tuple[str, str, Sequence[int]]] = []
        owners: list[int] = []
        for b, branch in enumerate(branches):
            if depth >= len(branch):
                continue
            for name, direction in branch[depth]:
                start_kind, _ = _kind_end(EDGES[name], direction)
                ids = sorted(
                    {ref[1] for fb, ref, _, _ in frontier if fb == b and ref[0] == start_kind}
                )
                if ids:
                    parts.append((name, direction, ids))
                    owners.append(b)
        if not parts:
            break
        hits = _step(conn, parts)
        _note_fanout(s, parts, hits)
        paths_at = {(fb, ref): (n, items) for fb, ref, n, items in frontier}
        nxt: dict[tuple[int, Ref], tuple[int, set[int]]] = {}
        for part, hit in hits:
            b = owners[part]
            _, direction, _ = parts[part]
            start, end = (hit.frm, hit.dst) if direction == "out" else (hit.dst, hit.frm)
            n, items = paths_at.get((b, start), (0, frozenset()))
            step_items = set(items) | ({hit.item_id} if hit.item_id is not None else set())
            target = reached if depth + 1 == len(branches[b]) else nxt
            key: Any = end if target is reached else (b, end)
            count, seen = target.get(key, (0, set()))
            target[key] = (count + n, seen | step_items)
        room = max(s.budget - len(s.nodes) - len(reached), 0)
        if len({ref for _, ref in nxt}) > room:
            s.reasons.add("budget")
        kept_refs: set[Ref] = set()
        cut: list[tuple[int, Ref, int, frozenset[int]]] = []
        for (b, ref), (n, along) in nxt.items():
            if ref in kept_refs or len(kept_refs) < room:
                kept_refs.add(ref)
                cut.append((b, ref, n, frozenset(along)))
        frontier = cut
        depth += 1
    types = _entity_types(conn, [ref[1] for ref in reached if ref[0] == ENTITY])
    ordered = sorted(reached.items(), key=lambda kv: -kv[1][0])
    room = max(s.budget - len(s.nodes), 0)
    kept = [(ref, v) for ref, v in ordered if _type_of(ref, types) in keep and ref != seed]
    if len(kept) > room:
        s.reasons.add("budget")
        s.nodes_left = len(kept) - room
        kept = kept[:room]
    for ref, (n, drawn) in kept:
        s.admit(ref)
        s.aggregated[(seed, ref)] = n
        for item in sorted(drawn):
            s.items.setdefault(item, None)


def _entity_types(conn: Connection, ids: Sequence[int]) -> dict[int, str]:
    """The node type of each entity reached on a path's last step, read only when the path keeps
    some entity type and ends on entities."""
    if not ids:
        return {}
    rows = conn.execute(
        text(f"SELECT entity_id, type FROM {_S}.entities WHERE entity_id = ANY(:ids)").bindparams(
            _ids("ids")
        ),
        {"ids": sorted(set(ids))},
    )
    return {row.entity_id: ENTITY_TYPES.get(row.type, row.type) for row in rows}


# --- the tail: nodes, items, induced edges ---------------------------------------------------


def _by_kind(refs: Iterable[Ref], kind: str) -> list[int]:
    return sorted(ref[1] for ref in refs if ref[0] == kind)


def _node_rows(conn: Connection, s: _Slice) -> dict[Ref, dict[str, Any]]:
    """The returned nodes other than items, one statement."""
    groups = {
        ENTITY: _by_kind(s.nodes, ENTITY),
        INSIGHT: _by_kind(s.nodes, INSIGHT),
        EVIDENCE: _by_kind(s.nodes, EVIDENCE),
        SOURCE: _by_kind(s.nodes, SOURCE),
    }
    if not any(groups.values()):
        return {}
    statement = text(
        f"SELECT 'entity' AS kind, e.entity_id AS id, e.natural_id, e.type, e.label, "
        f"NULL::bigint AS item_id, '{{}}'::jsonb AS props FROM {_S}.entities e "
        f"WHERE e.entity_id = ANY(:entity) "
        f"UNION ALL SELECT 'insight', i.insight_id, i.natural_id, 'Insight', i.text, i.item_id, "
        f"jsonb_build_object('insight_type', i.insight_type, 'rank', i.rank, "
        f"'grounding_confidence', i.grounding_confidence, 'position_hint', i.position_hint) "
        f"FROM {_S}.insights i WHERE i.insight_id = ANY(:insight) "
        f"UNION ALL SELECT 'evidence', ev.evidence_id, ev.natural_id, 'Quote', ev.text, "
        f"ev.item_id, jsonb_build_object('start_ms', ev.start_ms, 'end_ms', ev.end_ms) "
        f"FROM {_S}.evidence ev WHERE ev.evidence_id = ANY(:evidence) "
        f"UNION ALL SELECT 'source', so.source_id, so.natural_id, 'Source', so.title, "
        f"NULL::bigint, jsonb_build_object('content_type', so.content_type) "
        f"FROM {_S}.sources so WHERE so.source_id = ANY(:source)"
    ).bindparams(*(_ids(k) for k in groups))
    out = {}
    for row in conn.execute(statement, groups):
        node_type = ENTITY_TYPES.get(row.type, row.type) if row.kind == ENTITY else row.type
        out[(row.kind, row.id)] = {
            "natural_id": row.natural_id,
            "type": node_type,
            "label": row.label,
            "item_id": row.item_id,
            "properties": dict(row.props or {}),
        }
    return out


def _item_rows(conn: Connection, s: _Slice) -> dict[int, dict[str, Any]]:
    ids = sorted(s.items)
    if not ids:
        return {}
    statement = text(
        f"SELECT c.item_id, c.natural_id, c.content_type, c.published_at, c.title, "
        f"so.natural_id AS source_natural_id, so.title AS source_title "
        f"FROM {_S}.content_items c JOIN {_S}.sources so ON so.source_id = c.source_id "
        f"WHERE c.item_id = ANY(:items)"
    ).bindparams(_ids("items"))
    return {row.item_id: dict(row._mapping) for row in conn.execute(statement, {"items": ids})}


def _induced(conn: Connection, s: _Slice) -> list[_Hit]:
    """Every stored edge with both ends returned, one statement. Each edge type is probed from
    its bounded, indexed end: the returned items, insights, quotes or sources."""
    ids = {
        "items": _by_kind(s.nodes, ITEM),
        "insights": _by_kind(s.nodes, INSIGHT),
        "evidence": _by_kind(s.nodes, EVIDENCE),
        "sources": _by_kind(s.nodes, SOURCE),
        "entities": _by_kind(s.nodes, ENTITY),
    }
    if sum(1 for ref in s.nodes) < 2:
        return []
    ent = "= ANY(:entities)"
    parts = [
        f"SELECT 'HAS_ITEM' AS edge, c.source_id AS frm, c.item_id AS dst, c.item_id AS item_id, "
        f"NULL::text AS prop FROM {_S}.content_items c WHERE c.item_id = ANY(:items) "
        f"AND c.source_id = ANY(:sources)",
        f"SELECT 'HAS_INSIGHT', i.item_id, i.insight_id, i.item_id, NULL FROM {_S}.insights i "
        f"WHERE i.insight_id = ANY(:insights) AND i.item_id = ANY(:items)",
        f"SELECT 'PUBLISHED_BY', so.source_id, {_canonical('so.publisher_id')}, NULL, NULL "
        f"FROM {_S}.sources so WHERE so.source_id = ANY(:sources) "
        f"AND {_canonical('so.publisher_id')} {ent}",
        f"SELECT 'PARTICIPATES_IN', {_canonical('r.person_id')}, r.item_id, r.item_id, r.role "
        f"FROM {_S}.person_roles r WHERE r.item_id = ANY(:items) "
        f"AND r.role IN {_PARTICIPANT_ROLES} AND {_canonical('r.person_id')} {ent}",
        f"SELECT 'MENTIONS', {_canonical('ie.entity_id')}, ie.item_id, ie.item_id, NULL "
        f"FROM {_S}.item_entities ie WHERE ie.item_id = ANY(:items) AND ie.role = 'mentions' "
        f"AND {_canonical('ie.entity_id')} {ent}",
        f"SELECT im.kind, im.insight_id, {_canonical('im.entity_id')}, im.item_id, NULL "
        f"FROM {_S}.insight_mentions im WHERE im.insight_id = ANY(:insights) "
        f"AND {_canonical('im.entity_id')} {ent}",
        f"SELECT 'SUPPORTED_BY', s.insight_id, s.evidence_id, ev.item_id, NULL "
        f"FROM {_S}.insight_support s JOIN {_S}.evidence ev ON ev.evidence_id = s.evidence_id "
        f"WHERE s.insight_id = ANY(:insights) AND s.evidence_id = ANY(:evidence)",
        f"SELECT 'ATTRIBUTED_TO', ev.evidence_id, {_canonical('ev.speaker_id')}, ev.item_id, NULL "
        f"FROM {_S}.evidence ev WHERE ev.evidence_id = ANY(:evidence) "
        f"AND {_canonical('ev.speaker_id')} {ent}",
        f"SELECT 'STATED_BY', a.insight_id, {_canonical('a.person_id')}, a.item_id, NULL "
        f"FROM {_S}.insight_attribution a WHERE a.insight_id = ANY(:insights) "
        f"AND {_canonical('a.person_id')} {ent}",
    ]
    statement = text(" UNION ALL ".join(parts)).bindparams(*(_ids(k) for k in ids))
    hits = []
    seen = set()
    for row in conn.execute(statement, ids):
        edge = EDGES[row.edge]
        key = (row.edge, row.frm, row.dst)
        if key in seen:
            continue
        seen.add(key)
        hits.append(
            _Hit(row.edge, (edge.source, row.frm), (edge.target, row.dst), row.item_id, row.prop)
        )
    return hits


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finish(conn: Connection, s: _Slice) -> dict[str, Any]:
    """The tail: nodes' rows, items' rows, the induced edges; then ``graph-slice/1``."""
    nodes = _node_rows(conn, s)
    items = _item_rows(conn, s)
    induced = _induced(conn, s)

    def node_id(ref: Ref) -> str:
        if ref[0] == ITEM:
            return str(items[ref[1]]["natural_id"])
        return str(nodes[ref]["natural_id"])

    def item_natural(item_id: Optional[int]) -> Optional[str]:
        if item_id is None or item_id not in items:
            return None
        return str(items[item_id]["natural_id"])

    out_nodes = []
    for ref in s.nodes:
        if ref[0] == ITEM:
            row = items.get(ref[1])
            if row is None:
                continue
            out_nodes.append(
                {
                    "id": row["natural_id"],
                    "type": "ContentItem",
                    "label": row["title"],
                    "properties": {
                        "content_type": row["content_type"],
                        "published_at": _iso(row["published_at"]),
                    },
                    "item_id": row["natural_id"],
                    "groups": [],
                }
            )
            continue
        node = nodes.get(ref)
        if node is None:
            continue
        entry: dict[str, Any] = {
            "id": node["natural_id"],
            "type": node["type"],
            "label": node["label"],
            "properties": node["properties"],
            "groups": [],
        }
        if node["item_id"] is not None:
            entry["item_id"] = item_natural(node["item_id"])
        out_nodes.append(entry)

    def recency(item_id: Optional[int]) -> float:
        if item_id is None or item_id not in items or items[item_id]["published_at"] is None:
            return float("-inf")
        return float(items[item_id]["published_at"].timestamp())

    candidates: list[tuple[float, float, str, dict[str, Any]]] = []
    for hit in induced:
        edge = EDGES[hit.edge]
        frm, to = node_id(hit.frm), node_id(hit.dst)
        entry = {
            "type": hit.edge,
            "class": edge.cls,
            "from": frm,
            "to": to,
            "weight": 1,
            "properties": {"role": hit.prop} if hit.prop else {},
        }
        if edge.item_scoped:
            entry["item_id"] = item_natural(hit.item_id)
        candidates.append((1.0, recency(hit.item_id), f"{hit.edge}:{frm}:{to}", entry))
    for (a, b), n in s.aggregated.items():
        frm, to = node_id(a), node_id(b)
        entry = {
            "type": s.params.get("path", s.preset),
            "class": "discovery",
            "from": frm,
            "to": to,
            "weight": n,
            "properties": {},
        }
        candidates.append((float(n), float("-inf"), f"{entry['type']}:{frm}:{to}", entry))
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
    if len(candidates) > s.edge_budget:
        s.reasons.add("edge_budget")
    out_edges = [c[3] for c in candidates[: s.edge_budget]]

    capped = sorted(node_id(ref) for ref in s.fanout_capped if ref in s.nodes)
    truncated: dict[str, Any] = {"reasons": sorted(s.reasons), "fanout_capped": capped}
    if s.nodes_left:
        truncated["nodes_left"] = s.nodes_left
    if s.items_left is not None:
        truncated["items_left"] = s.items_left
    return {
        "format": FORMAT,
        "corpus_revision": s.revision,
        "preset": s.preset,
        "params": s.params,
        "applied_window": s.applied_window,
        "unknown_seeds": s.unknown_seeds,
        "nodes": out_nodes,
        "edges": out_edges,
        "items": [
            {
                "item_id": row["natural_id"],
                "content_type": row["content_type"],
                "published_at": _iso(row["published_at"]),
                "source_id": row["source_natural_id"],
                "source_title": row["source_title"],
            }
            for row in sorted(
                items.values(), key=lambda r: (-recency(r["item_id"]), r["natural_id"])
            )
        ],
        "groups": [],
        "truncated": truncated,
    }
