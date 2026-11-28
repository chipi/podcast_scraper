"""The graph engine (ADR-183): bounded walks over the corpus rows, answered as ``graph-slice/1``.

One module of functions over the corpus tables (decision 1). Each request runs in one read-only,
repeatable-read transaction (:func:`read_slice`); inside it every step of a walk is one statement,
a ``UNION ALL`` over the step's edges, each frontier node joined ``LATERAL`` with at most
:data:`FANOUT` edges per edge type (decision 9, CRFC-005 §4). No recursive statement is used: it
cannot cap a level's frontier. Statements are SQLAlchemy Core with bound arrays: no request value
reaches SQL text.

Built here, the walk core:

- the edges of decision 10 whose tables exist (:data:`EDGES`); ``CONTRIBUTES_TO`` and
  ``RELATED_TO`` are not built yet. ``IN_GROUP`` (a topic to its groups, a group to its parent)
  comes with the groups (below): it is not walked, only returned;
- the presets ``slice.items``, ``slice.recent`` (with ``expand_groups``; without an extension
  score),
  ``slice.group`` (one group of the published cluster build: its topics, or its child groups',
  and the newest items that carry them) and ``slice.neighbourhood`` with the named paths of
  decision 6;
- a merged entity read per member and answered under its canonical id (CRFC-005 §4);
- ``with: counts``: an entity node's ``item_count`` from the published projections build
  (``db.projections``), one more statement; every slice names that build as ``graph_build_id``,
  read in its seed statement;
- node ``groups`` and the slice's groups catalogue (CRFC-004 §2) from the published cluster build:
  a topic is in a group when a member of its canonical is, and the catalogue holds every group a
  served node is in and that group's parent;
- ``groups.summary``: every group of one kind with its parent and child count, and
  ``cluster_links``' weights between them, from the published cluster build, in one statement.

``exclude_items`` (decision 8), on every item-bearing preset: the held items resolve in the seed
statement and leave its candidates; their nodes, their item-owned nodes and their item-scoped edges
are not returned; a shared node a held item also reaches is a stub (``id``, ``type``,
``held: true``), read by an ``EXISTS`` in the nodes statement, so no statement is added.

The routes are ``server/routes/graph.py`` (the catalogue and the presets, cached by
``server/graph_presets.py``). Not built yet: the structured query and the replica retry.

Statement counts, against decision 9's table, are an upper bound: a step whose frontier is empty
sends nothing. The tail is 4 (nodes, items, induced edges, groups; groups only when the slice has
an entity or insight node), decision 9's 4; aggregated edges are counted during the walk, not by
a statement of their own; a path whose last step reaches entities reads their types in one more
statement, so ``keep`` can choose among them. ``tests/unit/podcast_scraper/db/test_db_graph.py``
asserts each count.

What is never served: an item that is not served or is withheld, an insight that is hidden
(``visible`` false) or cut from a document something supersedes, a quote cut from such a
document (CRFC-005 §3.4), and anything of an item that is not served. Node and edge ids in a slice
are natural ids; an entity's is its canonical row's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Sequence, TypeVar

from sqlalchemy import (
    and_,
    BigInteger,
    bindparam,
    cast,
    exc as sa_exc,
    exists,
    func,
    Integer,
    literal,
    null,
    or_,
    select,
    Text,
    text,
    true,
    union_all,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.sql import ColumnElement, Select

from podcast_scraper.db.schema import (
    cluster_builds,
    cluster_ids,
    cluster_links,
    cluster_members,
    clusters,
    content_items,
    corpus_revision,
    documents,
    entities,
    entity_item_counts,
    evidence,
    insight_attribution,
    insight_mentions,
    insight_support,
    insights,
    item_entities,
    person_roles,
    projection_builds,
    sources,
)

FORMAT = "graph-slice/1"
#: Membership: a topic or group to a group of the catalogue (decision 10).
IN_GROUP = "IN_GROUP"
#: The most edges of one type a frontier node contributes to a step.
FANOUT = 200
DEFAULT_BUDGET = 1_500
MAX_BUDGET = 3_000
DEFAULT_EDGE_BUDGET = 10_000
MAX_EDGE_BUDGET = 20_000
MAX_ITEM_SEEDS = 200
#: Decision 5's other static bounds (the structured query's; ``exclude_items`` every preset's).
MAX_NODE_SEEDS = 50
MAX_PATH_STEPS = 3
MAX_EXCLUDE_ITEMS = 2_000
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
_PARTICIPANT_ROLES = ("host", "guest")

T = TypeVar("T")
#: ``slice_recent`` takes a ``sources`` filter, so the table has a second name there.
_SOURCES = sources


class GraphTimeout(RuntimeError):
    """The request ran past its statement timeout: answered 503, never as a partial slice."""


class UnknownPath(ValueError):
    """A named path the engine does not have, or one not defined for the seed's type."""


#: The grouping kinds (``cluster_ids.kind``'s check, corpus_0013).
GROUP_KINDS = ("theme", "storyline", "super_theme", "insight")


class UnknownKind(ValueError):
    """A grouping kind that is not one of :data:`GROUP_KINDS`: answered 422, as a bad path."""


# --- statement pieces ------------------------------------------------------------------------


def _frontier(p: str) -> Any:
    """``unnest(:p) AS f(id)``: one step's frontier of one kind, bound as an array."""
    return (
        func.unnest(bindparam(p, type_=ARRAY(BigInteger)))
        .table_valued("id")
        .render_derived(name="f")
    )


def _shown(item_id: Any) -> ColumnElement[bool]:
    """The item is served and not withheld."""
    c = content_items.alias()
    return exists().where(c.c.item_id == item_id, c.c.served > 0, ~c.c.withheld)


def _current(row: Any) -> ColumnElement[bool]:
    """The row was cut from its item's served document: nothing supersedes it."""
    d = documents.alias()
    return ~exists().where(d.c.item_id == row.c.item_id, d.c.supersedes_id == row.c.document_id)


def _insight_ok(i: Any) -> ColumnElement[bool]:
    return and_(i.c.visible, _current(i), _shown(i.c.item_id))


def _quote_ok(ev: Any) -> ColumnElement[bool]:
    return and_(_current(ev), _shown(ev.c.item_id))


def _cols(frm: Any, dst: Any, item: Any = None, prop: Any = None) -> list[Any]:
    return [
        frm.label("frm"),
        dst.label("dst"),
        (item if item is not None else cast(null(), BigInteger)).label("item_id"),
        (prop if prop is not None else cast(null(), Text)).label("prop"),
    ]


def _per_member(
    p: str, inner: Callable[[Any], Select[Any]], order: Callable[[Any], list[Any]]
) -> tuple[Any, Any]:
    """A canonical entity's edges, read per member (one index range each), merged in ``order``
    and capped again, so a merged node fans out like any other. Each cap reads one row past
    :data:`FANOUT`, so :func:`_step` can tell a capped node from one with exactly that many."""
    f = _frontier(p)
    m = entities.alias("m")
    y = inner(m).limit(FANOUT + 1).lateral("y")
    x = (
        select(*y.c)
        .select_from(m.join(y, true()))
        .where(m.c.canonical_id == f.c.id)
        .order_by(*order(y))
        .limit(FANOUT + 1)
        .lateral("x")
    )
    return f, x


@dataclass(frozen=True)
class EdgeType:
    """One edge of decision 10: its ends, class, whether it belongs to an item, and the
    statement that reads it from a frontier in each direction. ``out`` starts from the
    ``source`` end, ``in`` from the ``target`` end; both answer ``(frm, dst, item_id, prop)`` in
    the edge's own direction."""

    name: str
    cls: str
    source: str
    target: str
    item_scoped: bool
    out: Callable[[str], Select[Any]]
    in_: Callable[[str], Select[Any]]


def _has_item_out(p: str) -> Select[Any]:
    f = _frontier(p)
    c = content_items.alias("c")
    x = (
        select(c.c.item_id)
        .where(c.c.source_id == f.c.id, c.c.served > 0, ~c.c.withheld)
        .order_by(c.c.published_at.desc().nulls_last(), c.c.item_id.desc())
        .limit(FANOUT + 1)
        .lateral("x")
    )
    return select(*_cols(f.c.id, x.c.item_id, x.c.item_id)).select_from(f.join(x, true()))


def _has_item_in(p: str) -> Select[Any]:
    f = _frontier(p)
    c = content_items.alias("c")
    on = and_(c.c.item_id == f.c.id, c.c.served > 0, ~c.c.withheld)
    return select(*_cols(c.c.source_id, f.c.id, f.c.id)).select_from(f.join(c, on))


def _has_insight_out(p: str) -> Select[Any]:
    f = _frontier(p)
    i = insights.alias("i")
    x = (
        select(i.c.insight_id)
        .where(i.c.item_id == f.c.id, i.c.visible, _current(i))
        .order_by(i.c.rank, i.c.insight_id)
        .limit(FANOUT + 1)
        .lateral("x")
    )
    return select(*_cols(f.c.id, x.c.insight_id, f.c.id)).select_from(f.join(x, true()))


def _has_insight_in(p: str) -> Select[Any]:
    f = _frontier(p)
    i = insights.alias("i")
    on = and_(i.c.insight_id == f.c.id, _insight_ok(i))
    return select(*_cols(i.c.item_id, f.c.id, i.c.item_id)).select_from(f.join(i, on))


def _published_by_out(p: str) -> Select[Any]:
    f = _frontier(p)
    so = sources.alias("so")
    k = entities.alias("k")
    return select(*_cols(f.c.id, k.c.canonical_id)).select_from(
        f.join(so, so.c.source_id == f.c.id).join(k, k.c.entity_id == so.c.publisher_id)
    )


def _published_by_in(p: str) -> Select[Any]:
    so = sources.alias("so")
    f, x = _per_member(
        p,
        lambda m: select(so.c.source_id)
        .where(so.c.publisher_id == m.c.entity_id)
        .order_by(so.c.source_id),
        lambda y: [y.c.source_id],
    )
    return select(*_cols(x.c.source_id, f.c.id)).select_from(f.join(x, true()))


def _participates_out(p: str) -> Select[Any]:
    r = person_roles.alias("r")
    c = content_items.alias("c")
    f, x = _per_member(
        p,
        lambda m: select(r.c.item_id, r.c.role, c.c.published_at)
        .join(c, and_(c.c.item_id == r.c.item_id, c.c.served > 0, ~c.c.withheld))
        .where(r.c.person_id == m.c.entity_id, r.c.role.in_(_PARTICIPANT_ROLES))
        .order_by(c.c.published_at.desc().nulls_last(), r.c.item_id.desc(), r.c.role),
        lambda y: [y.c.published_at.desc().nulls_last(), y.c.item_id.desc(), y.c.role],
    )
    return select(*_cols(f.c.id, x.c.item_id, x.c.item_id, x.c.role)).select_from(f.join(x, true()))


def _participates_in(p: str) -> Select[Any]:
    f = _frontier(p)
    r = person_roles.alias("r")
    k = entities.alias("k")
    x = (
        select(r.c.person_id, r.c.role)
        .where(r.c.item_id == f.c.id, r.c.role.in_(_PARTICIPANT_ROLES))
        .order_by(r.c.person_id, r.c.role)
        .limit(FANOUT + 1)
        .lateral("x")
    )
    return select(*_cols(k.c.canonical_id, f.c.id, f.c.id, x.c.role)).select_from(
        f.join(x, true()).join(k, k.c.entity_id == x.c.person_id)
    )


def _mentions_out(p: str) -> Select[Any]:
    ie = item_entities.alias("ie")
    # The index's order, (published_at DESC, item_id): Postgres puts undated rows first.
    f, x = _per_member(
        p,
        lambda m: select(ie.c.item_id, ie.c.published_at)
        .where(ie.c.entity_id == m.c.entity_id, ie.c.role == "mentions", _shown(ie.c.item_id))
        .order_by(ie.c.published_at.desc(), ie.c.item_id),
        lambda y: [y.c.published_at.desc(), y.c.item_id],
    )
    return select(*_cols(f.c.id, x.c.item_id, x.c.item_id)).select_from(f.join(x, true()))


def _mentions_in(p: str) -> Select[Any]:
    f = _frontier(p)
    ie = item_entities.alias("ie")
    k = entities.alias("k")
    x = (
        select(ie.c.entity_id)
        .where(ie.c.item_id == f.c.id, ie.c.role == "mentions")
        .order_by(ie.c.entity_id)
        .limit(FANOUT + 1)
        .lateral("x")
    )
    return select(*_cols(k.c.canonical_id, f.c.id, f.c.id)).select_from(
        f.join(x, true()).join(k, k.c.entity_id == x.c.entity_id)
    )


def _mention(kind: str) -> EdgeType:
    def out(p: str) -> Select[Any]:
        f = _frontier(p)
        im = insight_mentions.alias("im")
        k = entities.alias("k")
        x = (
            select(im.c.entity_id, im.c.item_id)
            .where(im.c.insight_id == f.c.id, im.c.kind == kind)
            .order_by(im.c.entity_id)
            .limit(FANOUT + 1)
            .lateral("x")
        )
        return select(*_cols(f.c.id, k.c.canonical_id, x.c.item_id)).select_from(
            f.join(x, true()).join(k, k.c.entity_id == x.c.entity_id)
        )

    def in_(p: str) -> Select[Any]:
        im = insight_mentions.alias("im")
        i = insights.alias("i")
        f, x = _per_member(
            p,
            lambda m: select(im.c.insight_id, im.c.item_id, im.c.rank)
            .join(i, i.c.insight_id == im.c.insight_id)
            .where(im.c.entity_id == m.c.entity_id, im.c.kind == kind, _insight_ok(i))
            .order_by(im.c.rank, im.c.insight_id),
            lambda y: [y.c.rank, y.c.insight_id],
        )
        return select(*_cols(x.c.insight_id, f.c.id, x.c.item_id)).select_from(f.join(x, true()))

    return EdgeType(kind, "descriptive", INSIGHT, ENTITY, True, out, in_)


def _supported_out(p: str) -> Select[Any]:
    f = _frontier(p)
    s = insight_support.alias("s")
    ev = evidence.alias("ev")
    x = (
        select(s.c.evidence_id, ev.c.item_id)
        .join(ev, ev.c.evidence_id == s.c.evidence_id)
        .where(s.c.insight_id == f.c.id, _quote_ok(ev))
        .order_by(s.c.ord, s.c.evidence_id)
        .limit(FANOUT + 1)
        .lateral("x")
    )
    return select(*_cols(f.c.id, x.c.evidence_id, x.c.item_id)).select_from(f.join(x, true()))


def _supported_in(p: str) -> Select[Any]:
    f = _frontier(p)
    s = insight_support.alias("s")
    i = insights.alias("i")
    x = (
        select(s.c.insight_id, i.c.item_id)
        .join(i, i.c.insight_id == s.c.insight_id)
        .where(s.c.evidence_id == f.c.id, _insight_ok(i))
        .order_by(i.c.rank, s.c.insight_id)
        .limit(FANOUT + 1)
        .lateral("x")
    )
    return select(*_cols(x.c.insight_id, f.c.id, x.c.item_id)).select_from(f.join(x, true()))


def _attributed_out(p: str) -> Select[Any]:
    f = _frontier(p)
    ev = evidence.alias("ev")
    k = entities.alias("k")
    on = and_(ev.c.evidence_id == f.c.id, _quote_ok(ev))
    return select(*_cols(f.c.id, k.c.canonical_id, ev.c.item_id)).select_from(
        f.join(ev, on).join(k, k.c.entity_id == ev.c.speaker_id)
    )


def _attributed_in(p: str) -> Select[Any]:
    ev = evidence.alias("ev")
    f, x = _per_member(
        p,
        lambda m: select(ev.c.evidence_id, ev.c.item_id)
        .where(ev.c.speaker_id == m.c.entity_id, _quote_ok(ev))
        .order_by(ev.c.item_id.desc(), ev.c.evidence_id),
        lambda y: [y.c.item_id.desc(), y.c.evidence_id],
    )
    return select(*_cols(x.c.evidence_id, f.c.id, x.c.item_id)).select_from(f.join(x, true()))


def _stated_out(p: str) -> Select[Any]:
    f = _frontier(p)
    a = insight_attribution.alias("a")
    k = entities.alias("k")
    return select(*_cols(f.c.id, k.c.canonical_id, a.c.item_id)).select_from(
        f.join(a, a.c.insight_id == f.c.id).join(k, k.c.entity_id == a.c.person_id)
    )


def _stated_in(p: str) -> Select[Any]:
    a = insight_attribution.alias("a")
    i = insights.alias("i")
    f, x = _per_member(
        p,
        lambda m: select(a.c.insight_id, a.c.item_id)
        .join(i, i.c.insight_id == a.c.insight_id)
        .where(a.c.person_id == m.c.entity_id, _insight_ok(i))
        .order_by(a.c.item_id.desc(), a.c.insight_id),
        lambda y: [y.c.item_id.desc(), y.c.insight_id],
    )
    return select(*_cols(x.c.insight_id, f.c.id, x.c.item_id)).select_from(f.join(x, true()))


_EDGE_LIST = [
    EdgeType("HAS_ITEM", "structural", SOURCE, ITEM, True, _has_item_out, _has_item_in),
    EdgeType("HAS_INSIGHT", "structural", ITEM, INSIGHT, True, _has_insight_out, _has_insight_in),
    EdgeType(
        "PUBLISHED_BY", "structural", SOURCE, ENTITY, False, _published_by_out, _published_by_in
    ),
    EdgeType(
        "PARTICIPATES_IN", "attribution", ENTITY, ITEM, True, _participates_out, _participates_in
    ),
    EdgeType("MENTIONS", "discovery", ENTITY, ITEM, True, _mentions_out, _mentions_in),
    _mention("ABOUT"),
    _mention("MENTIONS_PERSON"),
    _mention("MENTIONS_ORG"),
    _mention("MENTIONS_OBJECT"),
    EdgeType("SUPPORTED_BY", "evidentiary", INSIGHT, EVIDENCE, True, _supported_out, _supported_in),
    EdgeType(
        "ATTRIBUTED_TO", "attribution", EVIDENCE, ENTITY, True, _attributed_out, _attributed_in
    ),
    EdgeType("STATED_BY", "attribution", INSIGHT, ENTITY, True, _stated_out, _stated_in),
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


def read_slice(
    engine: Engine,
    fn: Callable[[Connection], T],
    *,
    timeout_ms: int = STATEMENT_TIMEOUT_MS,
) -> T:
    """Run ``fn`` in one read-only, repeatable-read transaction with ``timeout_ms`` as its
    statement timeout (``SET LOCAL``: nothing outlives the transaction). A timeout raises
    :class:`GraphTimeout`."""
    ms = int(timeout_ms)
    with engine.connect() as raw:
        conn = raw.execution_options(isolation_level="REPEATABLE READ")
        try:
            with conn.begin():
                conn.execute(text("SET TRANSACTION READ ONLY"))
                # set_config(..., true) is SET LOCAL with a bound value.
                conn.execute(
                    select(func.set_config("statement_timeout", bindparam("ms", str(ms)), True))
                )
                return fn(conn)
        except sa_exc.DBAPIError as exc:
            if getattr(exc.orig, "sqlstate", None) == "57014":
                raise GraphTimeout(f"the graph read ran past {ms} ms") from exc
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


Parts = Sequence[tuple[str, str, Sequence[int]]]


def _step(conn: Connection, parts: Parts) -> tuple[list[tuple[int, _Hit]], set[Ref]]:
    """One statement for a whole step: ``parts`` are ``(edge, direction, frontier ids)``. Each
    hit carries the index of its part; one edge between the same two nodes comes once (a person
    who is host and guest of one item, two merged members mentioning it). The second value is
    the frontier nodes past the fan-out cap: each read one row more than :data:`FANOUT`, and
    that row is dropped."""
    selects = []
    params: dict[str, list[int]] = {}
    for n, (name, direction, ids) in enumerate(parts):
        edge = EDGES[name]
        q = (edge.out if direction == "out" else edge.in_)(f"p{n}").subquery(f"q{n}")
        selects.append(select(literal(n, Integer).label("part"), *q.c))
        params[f"p{n}"] = list(ids)
    statement = selects[0] if len(selects) == 1 else union_all(*selects)
    hits: list[tuple[int, _Hit]] = []
    seen: set[tuple[int, int, int]] = set()
    counts: dict[tuple[int, int], int] = {}
    capped: set[Ref] = set()
    for row in conn.execute(statement, params):
        name, direction, _ = parts[row.part]
        edge = EDGES[name]
        start = row.frm if direction == "out" else row.dst
        n = counts[(row.part, start)] = counts.get((row.part, start), 0) + 1
        if n > FANOUT:
            capped.add((_kind_end(edge, direction)[0], start))
            continue
        if (row.part, row.frm, row.dst) in seen:
            continue
        seen.add((row.part, row.frm, row.dst))
        hit = _Hit(name, (edge.source, row.frm), (edge.target, row.dst), row.item_id, row.prop)
        hits.append((row.part, hit))
    return hits, capped


@dataclass
class _Slice:
    """What a request collects before the tail statements turn it into ``graph-slice/1``."""

    preset: str
    params: dict[str, Any]
    revision: Optional[int] = None
    graph_build_id: Optional[int] = None
    counts: bool = False
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
    #: The items the client holds (``exclude_items``), resolved in the seed statement.
    held: frozenset[int] = field(default_factory=frozenset)

    def admit(self, ref: Ref) -> None:
        self.nodes.setdefault(ref, None)
        if ref[0] == ITEM:
            self.items.setdefault(ref[1], None)

    def capped(self, refs: set[Ref]) -> None:
        if refs:
            self.fanout_capped |= refs
            self.reasons.add("fanout")


def _check_budgets(budget: int, edge_budget: int) -> None:
    if not 1 <= budget <= MAX_BUDGET:
        raise ValueError(f"budget {budget} is not between 1 and {MAX_BUDGET}")
    if not 1 <= edge_budget <= MAX_EDGE_BUDGET:
        raise ValueError(f"edge_budget {edge_budget} is not between 1 and {MAX_EDGE_BUDGET}")


def _revision() -> Any:
    """The corpus revision and the published projections build (``graph_build_id``), read in the
    seed statement, so a slice names the build its caches and ETags key on (CRFC-005 §4)."""
    built = (
        select(projection_builds.c.build_id)
        .where(projection_builds.c.state == "published")
        .scalar_subquery()
    )
    return (
        select(corpus_revision.c.revision, built.label("graph_build_id"))
        .where(corpus_revision.c.id == 1)
        .subquery("r")
    )


#: What ``with`` may name (CRFC-004 §2).
WITH = frozenset({"counts"})


def _with(s: "_Slice", with_: Sequence[str]) -> None:
    unknown = sorted(set(with_) - WITH)
    if unknown:
        raise ValueError(f"with {unknown} is not one of {sorted(WITH)}")
    if with_:
        s.counts = "counts" in with_
        s.params["with"] = sorted(set(with_))


def _exclude(s: "_Slice", exclude_items: Sequence[str]) -> list[str]:
    """The held items' natural ids, bounded (decision 5), named in the slice's params."""
    held = list(dict.fromkeys(exclude_items))
    if len(held) > MAX_EXCLUDE_ITEMS:
        raise ValueError(f"{len(held)} exclude_items is more than {MAX_EXCLUDE_ITEMS}")
    if held:
        s.params["exclude_items"] = held
    return held


def _held_col(held: list[str]) -> Any:
    """The held items' keys, a column of the seed statement."""
    hc = content_items.alias("hc")
    return (
        select(func.array_agg(hc.c.item_id))
        .where(hc.c.natural_id == func.any(literal(held, ARRAY(Text))))
        .scalar_subquery()
        .label("held")
    )


def _not_held(natural_id: Any, held: list[str]) -> ColumnElement[bool]:
    return natural_id != func.all(literal(held, ARRAY(Text)))  # type: ignore[no-any-return]


def _set_held(s: "_Slice", value: Any) -> None:
    s.held = frozenset(int(i) for i in (value or []))


# --- item slices -----------------------------------------------------------------------------


def _fill_items(conn: Connection, s: _Slice, item_ids: Sequence[int]) -> None:
    """Decision 9's item mode: the items in order, then round-robin across them, each item's KG
    nodes first, then its insights by rank; admitting an insight admits the nodes it reaches.
    Items past what the budget has left are cut and counted in ``items_left``."""
    items = [i for i in dict.fromkeys(item_ids) if i not in s.items and i not in s.held]
    room = max(s.budget - len(s.nodes), 0)
    if len(items) > room:
        s.reasons.add("budget")
        s.items_left = (s.items_left or 0) + len(items) - room
        items = items[:room]
    if not items:
        return
    for item in items:
        s.admit((ITEM, item))
    first, capped = _step(conn, [(name, direction, items) for name, direction in _ITEM_STEP])
    s.capped(capped)
    kg: dict[int, list[Ref]] = {item: [] for item in items}
    ranked: dict[int, list[int]] = {item: [] for item in items}
    for _, hit in first:
        if hit.edge == "HAS_INSIGHT":
            ranked[hit.frm[1]].append(hit.dst[1])
        elif hit.frm not in kg[hit.dst[1]]:
            kg[hit.dst[1]].append(hit.frm)
    insights_ = [i for item in items for i in ranked[item]]
    reach: dict[int, list[Ref]] = {i: [] for i in insights_}
    if insights_:
        second, capped = _step(
            conn, [(name, direction, insights_) for name, direction in _INSIGHT_STEP]
        )
        s.capped(capped)
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


def slice_items(
    conn: Connection,
    natural_ids: Sequence[str],
    *,
    budget: int = DEFAULT_BUDGET,
    edge_budget: int = DEFAULT_EDGE_BUDGET,
    with_: Sequence[str] = (),
    exclude_items: Sequence[str] = (),
) -> dict[str, Any]:
    """``slice.items``: these items' graph. Unknown, unserved or withheld items are
    ``unknown_seeds``; a held one is left out."""
    _check_budgets(budget, edge_budget)
    seeds = list(dict.fromkeys(natural_ids))
    if len(seeds) > MAX_ITEM_SEEDS:
        raise ValueError(f"{len(seeds)} item ids is more than {MAX_ITEM_SEEDS}")
    s = _Slice("slice.items", {"items": seeds}, budget=budget, edge_budget=edge_budget)
    _with(s, with_)
    held = _exclude(s, exclude_items)
    r = _revision()
    c = content_items.alias("c")
    x = (
        select(c.c.item_id, c.c.natural_id)
        .where(c.c.natural_id.in_(seeds), c.c.served > 0, ~c.c.withheld)
        .lateral("x")
    )
    rows = conn.execute(
        select(
            r.c.revision, r.c.graph_build_id, _held_col(held), x.c.item_id, x.c.natural_id
        ).select_from(r.outerjoin(x, true()))
    ).all()
    s.revision = rows[0].revision if rows else None
    s.graph_build_id = rows[0].graph_build_id if rows else None
    _set_held(s, rows[0].held if rows else None)
    found = {row.natural_id: row.item_id for row in rows if row.item_id is not None}
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
    with_: Sequence[str] = (),
    expand_groups: int = 0,
    exclude_items: Sequence[str] = (),
) -> dict[str, Any]:
    """``slice.recent``: the newest served items. The window widens (7d, 30d, 90d, all) while
    fewer than ``min_items`` match; every widening is answered by the one candidate statement,
    newest first and at most ``budget`` items (one more is read to know whether the budget cut),
    and the response names ``applied_window``. With ``expand_groups: n`` (:func:`_expand_groups`),
    every group with a member topic in the slice adds its ``n`` newest other items after them."""
    _check_budgets(budget, edge_budget)
    if expand_groups < 0:
        raise ValueError("expand_groups must be at least 0")
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
    if expand_groups:
        params["expand_groups"] = expand_groups
    s = _Slice("slice.recent", params, budget=budget, edge_budget=edge_budget)
    _with(s, with_)
    held = _exclude(s, exclude_items)
    c = content_items.alias("c")
    conditions: list[Any] = [c.c.served > 0, ~c.c.withheld, _not_held(c.c.natural_id, held)]
    if content_types is not None:
        conditions.append(c.c.content_type.in_(list(content_types)))
    if sources is not None:
        so = _SOURCES.alias("so")
        conditions.append(
            c.c.source_id.in_(select(so.c.source_id).where(so.c.natural_id.in_(list(sources))))
        )
    r = _revision()
    x = (
        select(c.c.item_id, c.c.published_at)
        .where(*conditions)
        .order_by(c.c.published_at.desc().nulls_last(), c.c.item_id.desc())
        .limit(budget + 1)
        .lateral("x")
    )
    rows = conn.execute(
        select(
            r.c.revision, r.c.graph_build_id, _held_col(held), x.c.item_id, x.c.published_at
        ).select_from(r.outerjoin(x, true()))
    ).all()
    s.revision = rows[0].revision if rows else None
    s.graph_build_id = rows[0].graph_build_id if rows else None
    _set_held(s, rows[0].held if rows else None)
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
    if len(chosen) > budget:
        # More match than the budget holds; how many more is not read.
        s.reasons.add("budget")
        chosen = chosen[:budget]
    _fill_items(conn, s, chosen)
    if expand_groups and len(s.nodes) < s.budget:
        _expand_groups(conn, s, expand_groups, conditions, c)
    return _finish(conn, s)


def _expand_groups(conn: Connection, s: _Slice, n: int, conditions: list[Any], c: Any) -> None:
    """``expand_groups`` (CRFC-004 §2, the sibling pull): every group of the published cluster
    build with a member topic among the slice's entities adds its ``n`` newest other items of the
    slice's sources and content types (not its window: the siblings are what the window left
    out), groups in id order, inside what the budget has left, after the primary items. One
    statement chooses them (a ``LATERAL`` per group), then the item steps
    (ADR-183 decision 9: +1, +2)."""
    held = [ref[1] for ref in s.nodes if ref[0] == ENTITY]
    if not held:
        return
    build = (
        select(cluster_builds.c.build_id)
        .where(cluster_builds.c.state == "published")
        .scalar_subquery()
    )
    m, k = cluster_members.alias("xm"), entities.alias("xk")
    groups = (
        select(m.c.cluster_id)
        .select_from(m.join(k, k.c.entity_id == m.c.entity_id))
        .where(
            m.c.build_id == build,
            k.c.canonical_id == func.any(bindparam("held", type_=ARRAY(BigInteger))),
        )
        .distinct()
        .subquery("xg")
    )
    gm, gk = cluster_members.alias("xgm"), entities.alias("xgk")
    canon = (
        select(gk.c.canonical_id)
        .select_from(gm.join(gk, gk.c.entity_id == gm.c.entity_id))
        .where(gm.c.build_id == build, gm.c.cluster_id == groups.c.cluster_id)
        .correlate_except(gm, gk)
    )
    ie, ke = item_entities.alias("xe"), entities.alias("xq")
    carriers = (
        select(ie.c.item_id)
        .select_from(ie.join(ke, ke.c.entity_id == ie.c.entity_id))
        .where(ke.c.canonical_id.in_(canon))
        .correlate_except(ie, ke)
    )
    x = (
        select(c.c.item_id, c.c.published_at)
        .where(
            c.c.item_id.in_(carriers),
            c.c.item_id != func.all(bindparam("primary", type_=ARRAY(BigInteger))),
            *conditions,
        )
        .order_by(c.c.published_at.desc().nulls_last(), c.c.item_id.desc())
        .limit(n)
        .lateral("xi")
    )
    own = cluster_ids.alias("xo")
    rows = conn.execute(
        select(own.c.natural_id, x.c.item_id)
        .select_from(groups.join(own, own.c.cluster_id == groups.c.cluster_id).join(x, true()))
        .order_by(own.c.natural_id, x.c.published_at.desc().nulls_last(), x.c.item_id.desc()),
        {"held": held, "primary": list(s.items)},
    ).all()
    added = list(dict.fromkeys(row.item_id for row in rows))
    _fill_items(conn, s, added)


def slice_group(
    conn: Connection,
    group_id: str,
    *,
    budget: int = DEFAULT_BUDGET,
    edge_budget: int = DEFAULT_EDGE_BUDGET,
    with_: Sequence[str] = (),
    exclude_items: Sequence[str] = (),
) -> dict[str, Any]:
    """``slice.group``: one group of any kind in the published cluster build (CRFC-004 §2): its
    member topics, or its child groups' for a group with children (a super-theme), and the
    served items that carry them, newest first, at most ``budget`` items, and their graph. An
    item carries a topic when an ``item_entities`` row names a member of its canonical; an
    insight cluster's items are its shown insights'. The seed statement reads the group, its
    topics and its items; a group not in that build is ``unknown_seeds``."""
    _check_budgets(budget, edge_budget)
    s = _Slice("slice.group", {"group": group_id}, budget=budget, edge_budget=edge_budget)
    _with(s, with_)
    held = _exclude(s, exclude_items)
    build = (
        select(cluster_builds.c.build_id)
        .where(cluster_builds.c.state == "published")
        .scalar_subquery()
    )
    own = cluster_ids.alias("sg")
    group = (
        select(clusters.c.cluster_id)
        .select_from(clusters.join(own, own.c.cluster_id == clusters.c.cluster_id))
        .where(clusters.c.build_id == build, own.c.natural_id == group_id)
        .scalar_subquery()
    )
    sub = clusters.alias("sc")
    scope = select(sub.c.cluster_id).where(
        sub.c.build_id == build, (sub.c.cluster_id == group) | (sub.c.parent_id == group)
    )
    m, k = cluster_members.alias("sm"), entities.alias("sk")
    topics = (
        select(k.c.canonical_id)
        .select_from(m.join(k, k.c.entity_id == m.c.entity_id))
        .where(m.c.build_id == build, m.c.cluster_id.in_(scope))
        .distinct()
    )
    mi, i = cluster_members.alias("si"), insights.alias("sn")
    ie, ke = item_entities.alias("se"), entities.alias("sq")
    carriers = union_all(
        select(ie.c.item_id)
        .select_from(ie.join(ke, ke.c.entity_id == ie.c.entity_id))
        .where(ke.c.canonical_id.in_(topics)),
        select(i.c.item_id)
        .select_from(mi.join(i, i.c.insight_id == mi.c.insight_id))
        .where(mi.c.build_id == build, mi.c.cluster_id.in_(scope), _insight_ok(i)),
    )
    c = content_items.alias("c")
    x = (
        select(c.c.item_id)
        .where(
            c.c.item_id.in_(carriers),
            c.c.served > 0,
            ~c.c.withheld,
            _not_held(c.c.natural_id, held),
        )
        .order_by(c.c.published_at.desc().nulls_last(), c.c.item_id.desc())
        .limit(budget + 1)
        .lateral("x")
    )
    tk = entities.alias("st")
    members = (
        select(func.array_agg(tk.c.entity_id)).where(tk.c.entity_id.in_(topics)).scalar_subquery()
    )
    r = _revision()
    rows = conn.execute(
        select(
            r.c.revision,
            r.c.graph_build_id,
            _held_col(held),
            group.label("group"),
            members.label("topics"),
            x.c.item_id,
        ).select_from(r.outerjoin(x, true()))
    ).all()
    s.revision = rows[0].revision if rows else None
    s.graph_build_id = rows[0].graph_build_id if rows else None
    _set_held(s, rows[0].held if rows else None)
    if not rows or rows[0].group is None:
        s.unknown_seeds = [group_id]
        return _finish(conn, s)
    # An item slice (CRFC-004 §5): the topics take at most half the budget while there are
    # items for the rest, so a group with more topics than the budget still shows its items.
    # One more item than the budget is read, so a cut is known, but not how many more there are.
    chosen = [row.item_id for row in rows if row.item_id is not None]
    topic_room = max(budget - len(chosen), budget // 2)
    for topic in sorted(rows[0].topics or []):
        if len(s.nodes) >= topic_room:
            s.reasons.add("budget")
            break
        s.admit((ENTITY, int(topic)))
    room = budget - len(s.nodes)
    if len(chosen) > room:
        s.reasons.add("budget")
        chosen = chosen[: max(room, 0)]
    _fill_items(conn, s, chosen)
    return _finish(conn, s)


def groups_summary(conn: Connection, kind: str) -> dict[str, Any]:
    """``groups.summary``: every group of ``kind`` in the published cluster build (CRFC-004 §2,
    the top-down bubbles and their links), in one statement with the revision and build ids: each
    group's ``id``, ``kind``, ``label``, ``parent_id`` and ``child_count`` (the groups whose
    parent it is), and ``links``, ``cluster_links``' weights between two groups of the kind, from
    the same build, so the two never mix builds. Weights are corpus-wide, not the loaded graph's.
    With no published build, both lists are empty. Nothing writes ``cluster_links`` yet (the
    viewer derives its top-down links itself, ``topDownSlice.ts``), so ``links`` is empty until
    a writer exists; one would be the groupings importer's (CRFC-005 §8)."""
    if kind not in GROUP_KINDS:
        raise UnknownKind(f"grouping kind {kind!r} is not one of {list(GROUP_KINDS)}")
    build = (
        select(cluster_builds.c.build_id)
        .where(cluster_builds.c.state == "published")
        .scalar_subquery()
    )
    c, child = clusters.alias("gc"), clusters.alias("gk")
    own, up = cluster_ids.alias("go"), cluster_ids.alias("gu")
    children = (
        select(func.count())
        .where(child.c.build_id == c.c.build_id, child.c.parent_id == c.c.cluster_id)
        .scalar_subquery()
    )
    la, lb = cluster_ids.alias("la"), cluster_ids.alias("lb")
    links = cluster_links
    r = _revision()
    statement = union_all(
        select(
            literal("head").label("row"),
            r.c.revision,
            r.c.graph_build_id,
            cast(null(), Text).label("id"),
            cast(null(), Text).label("label"),
            cast(null(), Text).label("other"),
            cast(null(), BigInteger).label("n"),
        ).select_from(r),
        select(
            literal("group"),
            cast(null(), BigInteger),
            cast(null(), BigInteger),
            own.c.natural_id,
            c.c.label,
            up.c.natural_id,
            children,
        )
        .select_from(
            c.join(own, own.c.cluster_id == c.c.cluster_id).outerjoin(
                up, up.c.cluster_id == c.c.parent_id
            )
        )
        .where(c.c.build_id == build, c.c.kind == kind),
        select(
            literal("link"),
            cast(null(), BigInteger),
            cast(null(), BigInteger),
            la.c.natural_id,
            cast(null(), Text),
            lb.c.natural_id,
            cast(links.c.weight, BigInteger),
        )
        .select_from(
            links.join(la, la.c.cluster_id == links.c.a).join(lb, lb.c.cluster_id == links.c.b)
        )
        .where(links.c.build_id == build, links.c.kind == kind),
    )
    revision = graph_build_id = None
    groups: list[dict[str, Any]] = []
    weights: list[dict[str, Any]] = []
    for row in conn.execute(statement):
        if row.row == "head":
            revision, graph_build_id = row.revision, row.graph_build_id
        elif row.row == "group":
            groups.append(
                {
                    "id": row.id,
                    "kind": kind,
                    "label": row.label,
                    "parent_id": row.other,
                    "child_count": int(row.n),
                }
            )
        else:
            a, b = sorted((str(row.id), str(row.other)))
            weights.append({"a": a, "b": b, "weight": int(row.n)})
    return {
        "format": FORMAT,
        "corpus_revision": revision,
        "graph_build_id": graph_build_id,
        "preset": "groups.summary",
        "params": {"kind": kind},
        "groups": sorted(groups, key=lambda g: g["id"]),
        "links": sorted(weights, key=lambda w: (-w["weight"], w["a"], w["b"])),
    }


# --- path walks ------------------------------------------------------------------------------


def slice_neighbourhood(
    conn: Connection,
    node_id: str,
    path: str,
    *,
    budget: int = DEFAULT_BUDGET,
    edge_budget: int = DEFAULT_EDGE_BUDGET,
    with_: Sequence[str] = (),
    exclude_items: Sequence[str] = (),
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
    _with(s, with_)
    held = _exclude(s, exclude_items)
    r = _revision()
    e = entities.alias("e")
    k = entities.alias("k")
    x = (
        select(e.c.canonical_id, k.c.type)
        .select_from(e.join(k, k.c.entity_id == e.c.canonical_id))
        .where(e.c.natural_id == node_id)
        .lateral("x")
    )
    row = conn.execute(
        select(
            r.c.revision, r.c.graph_build_id, _held_col(held), x.c.canonical_id, x.c.type
        ).select_from(r.outerjoin(x, true()))
    ).first()
    s.revision = row.revision if row is not None else None
    s.graph_build_id = row.graph_build_id if row is not None else None
    _set_held(s, row.held if row is not None else None)
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
    """Every branch from ``seed``, level by level. A frontier entry carries its branch, the
    number of paths that reach it and the items those paths draw from; after each level the new
    frontier is cut to the budget left. Ties in a cut go to the node with more paths, then the
    newer row (the higher key)."""
    frontier: list[tuple[int, Ref, int, frozenset[int]]] = [
        (b, seed, 1, frozenset()) for b in range(len(branches))
    ]
    # The slice's items are those of the nodes it returns, not of the paths the budget cut.
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
        hits, capped = _step(conn, parts)
        s.capped(capped)
        paths_at = {(fb, ref): (n, along) for fb, ref, n, along in frontier}
        nxt: dict[tuple[int, Ref], tuple[int, set[int]]] = {}
        for part, hit in hits:
            b = owners[part]
            _, direction, _ = parts[part]
            start, end = (hit.frm, hit.dst) if direction == "out" else (hit.dst, hit.frm)
            n, along = paths_at.get((b, start), (0, frozenset()))
            drawn = set(along) | ({hit.item_id} if hit.item_id is not None else set())
            if depth + 1 == len(branches[b]):
                count, seen = reached.get(end, (0, set()))
                reached[end] = (count + n, seen | drawn)
            else:
                count, seen = nxt.get((b, end), (0, set()))
                nxt[(b, end)] = (count + n, seen | drawn)
        room = max(s.budget - len(s.nodes) - len(reached), 0)
        ranked = sorted(nxt.items(), key=lambda kv: (-kv[1][0], -kv[0][1][1], kv[0][0]))
        kept_refs: set[Ref] = set()
        cut: list[tuple[int, Ref, int, frozenset[int]]] = []
        for (b, ref), (n, items_on) in ranked:
            if ref in kept_refs or len(kept_refs) < room:
                kept_refs.add(ref)
                cut.append((b, ref, n, frozenset(items_on)))
            else:
                s.reasons.add("budget")
        frontier = cut
        depth += 1
    types = _entity_types(conn, [ref[1] for ref in reached if ref[0] == ENTITY])
    ordered = sorted(reached.items(), key=lambda kv: (-kv[1][0], -kv[0][1]))
    room = max(s.budget - len(s.nodes), 0)
    kept = [
        (ref, v)
        for ref, v in ordered
        if _type_of(ref, types) in keep
        and ref != seed
        and not (ref[0] == ITEM and ref[1] in s.held)
    ]
    if len(kept) > room:
        s.reasons.add("budget")
        s.nodes_left = len(kept) - room
        kept = kept[:room]
    for ref, (n, drawn) in kept:
        s.admit(ref)
        s.aggregated[(seed, ref)] = n
        for item in sorted(drawn - s.held):
            s.items.setdefault(item, None)


def _entity_types(conn: Connection, ids: Sequence[int]) -> dict[int, str]:
    """The node type of each entity a path's last step reached, so ``keep`` can choose."""
    if not ids:
        return {}
    rows = conn.execute(
        select(entities.c.entity_id, entities.c.type).where(
            entities.c.entity_id.in_(sorted(set(ids)))
        )
    )
    return {row.entity_id: ENTITY_TYPES.get(row.type, row.type) for row in rows}


# --- the tail: nodes, items, induced edges ---------------------------------------------------


def _by_kind(refs: Iterable[Ref], kind: str) -> list[int]:
    return sorted(ref[1] for ref in refs if ref[0] == kind)


def _node_rows(conn: Connection, s: _Slice) -> dict[Ref, dict[str, Any]]:
    """The returned nodes other than items, one statement."""
    ids = {kind: _by_kind(s.nodes, kind) for kind in (ENTITY, INSIGHT, EVIDENCE, SOURCE)}
    if not any(ids.values()):
        return {}

    def node(
        kind: str, key: Any, nat: Any, typ: Any, label: Any, item: Any, props: Any, held: Any
    ) -> Any:
        return [
            literal(kind, Text).label("kind"),
            key.label("id"),
            nat.label("natural_id"),
            typ.label("type"),
            label.label("label"),
            item.label("item_id"),
            props.label("props"),
            held.label("held"),
        ]

    i, ev, so = insights, evidence, sources
    no = literal(False)
    entity_held: Any = no
    source_held: Any = no
    if s.held:
        # A shared node a held item reaches (decision 8): by any member of the canonical, through
        # the item's own edges or its insights' and quotes'.
        hl = literal(sorted(s.held), ARRAY(BigInteger))
        reaches = []
        for table, ent, item in (
            (item_entities, "entity_id", "item_id"),
            (person_roles, "person_id", "item_id"),
            (insight_mentions, "entity_id", "item_id"),
            (insight_attribution, "person_id", "item_id"),
            (evidence, "speaker_id", "item_id"),
        ):
            t, m = table.alias(f"h_{table.name}"), entities.alias(f"hm_{table.name}")
            reaches.append(
                exists()
                .select_from(t.join(m, m.c.entity_id == t.c[ent]))
                .where(m.c.canonical_id == entities.c.entity_id, t.c[item] == func.any(hl))
            )
        entity_held = or_(*reaches)
        hs = content_items.alias("hs")
        source_held = exists().where(hs.c.source_id == so.c.source_id, hs.c.item_id == func.any(hl))
    statement = union_all(
        select(
            *node(
                ENTITY,
                entities.c.entity_id,
                entities.c.natural_id,
                entities.c.type,
                entities.c.label,
                cast(null(), BigInteger),
                func.jsonb_build_object(),
                entity_held,
            )
        ).where(entities.c.entity_id.in_(ids[ENTITY])),
        select(
            *node(
                INSIGHT,
                i.c.insight_id,
                i.c.natural_id,
                literal("Insight", Text),
                i.c.text,
                i.c.item_id,
                func.jsonb_build_object(
                    "insight_type",
                    i.c.insight_type,
                    "rank",
                    i.c.rank,
                    "grounding_confidence",
                    i.c.grounding_confidence,
                    "position_hint",
                    i.c.position_hint,
                ),
                no,
            )
        ).where(i.c.insight_id.in_(ids[INSIGHT])),
        select(
            *node(
                EVIDENCE,
                ev.c.evidence_id,
                ev.c.natural_id,
                literal("Quote", Text),
                ev.c.text,
                ev.c.item_id,
                func.jsonb_build_object("start_ms", ev.c.start_ms, "end_ms", ev.c.end_ms),
                no,
            )
        ).where(ev.c.evidence_id.in_(ids[EVIDENCE])),
        select(
            *node(
                SOURCE,
                so.c.source_id,
                so.c.natural_id,
                literal("Source", Text),
                so.c.title,
                cast(null(), BigInteger),
                func.jsonb_build_object("content_type", so.c.content_type),
                source_held,
            )
        ).where(so.c.source_id.in_(ids[SOURCE])),
    )
    out = {}
    for row in conn.execute(statement):
        node_type = ENTITY_TYPES.get(row.type, row.type) if row.kind == ENTITY else row.type
        out[(row.kind, row.id)] = {
            "natural_id": row.natural_id,
            "type": node_type,
            "label": row.label,
            "item_id": row.item_id,
            "properties": dict(row.props or {}),
            "held": bool(row.held),
        }
    return out


def _item_rows(conn: Connection, s: _Slice) -> dict[int, dict[str, Any]]:
    """The slice's items, served and not withheld only: a node of any other item is dropped."""
    ids = sorted(s.items)
    if not ids:
        return {}
    c, so = content_items, sources
    statement = (
        select(
            c.c.item_id,
            c.c.natural_id,
            c.c.content_type,
            c.c.published_at,
            c.c.title,
            so.c.natural_id.label("source_natural_id"),
            so.c.title.label("source_title"),
        )
        .join(so, so.c.source_id == c.c.source_id)
        .where(c.c.item_id.in_(ids), c.c.served > 0, ~c.c.withheld)
    )
    return {row.item_id: dict(row._mapping) for row in conn.execute(statement)}


def _induced(conn: Connection, s: _Slice) -> list[_Hit]:
    """Every stored edge with both ends returned, one statement. Each edge type is probed from
    its bounded, indexed end (the returned items, insights, quotes or sources), and an entity end
    is matched through its member rows' canonical id."""
    if len(s.nodes) < 2:
        return []
    items = _by_kind(s.nodes, ITEM)
    ins = _by_kind(s.nodes, INSIGHT)
    quotes = _by_kind(s.nodes, EVIDENCE)
    srcs = _by_kind(s.nodes, SOURCE)
    ents = _by_kind(s.nodes, ENTITY)

    def edge(name: Any, frm: Any, dst: Any, item: Any = None, prop: Any = None) -> list[Any]:
        label = name if not isinstance(name, str) else literal(name, Text)
        return [label.label("edge"), *_cols(frm, dst, item, prop)]

    c, i, so = content_items, insights, sources
    r, ie, im = person_roles, item_entities, insight_mentions
    sp, ev, a = insight_support, evidence, insight_attribution
    k = entities.alias("k")
    statement = union_all(
        select(*edge("HAS_ITEM", c.c.source_id, c.c.item_id, c.c.item_id)).where(
            c.c.item_id.in_(items), c.c.source_id.in_(srcs)
        ),
        select(*edge("HAS_INSIGHT", i.c.item_id, i.c.insight_id, i.c.item_id)).where(
            i.c.insight_id.in_(ins), i.c.item_id.in_(items)
        ),
        select(*edge("PUBLISHED_BY", so.c.source_id, k.c.canonical_id))
        .select_from(so.join(k, k.c.entity_id == so.c.publisher_id))
        .where(so.c.source_id.in_(srcs), k.c.canonical_id.in_(ents)),
        select(*edge("PARTICIPATES_IN", k.c.canonical_id, r.c.item_id, r.c.item_id, r.c.role))
        .select_from(r.join(k, k.c.entity_id == r.c.person_id))
        .where(
            r.c.item_id.in_(items),
            r.c.role.in_(_PARTICIPANT_ROLES),
            k.c.canonical_id.in_(ents),
        ),
        select(*edge("MENTIONS", k.c.canonical_id, ie.c.item_id, ie.c.item_id))
        .select_from(ie.join(k, k.c.entity_id == ie.c.entity_id))
        .where(ie.c.item_id.in_(items), ie.c.role == "mentions", k.c.canonical_id.in_(ents)),
        select(*edge(im.c.kind, im.c.insight_id, k.c.canonical_id, im.c.item_id))
        .select_from(im.join(k, k.c.entity_id == im.c.entity_id))
        .where(im.c.insight_id.in_(ins), k.c.canonical_id.in_(ents)),
        select(*edge("SUPPORTED_BY", sp.c.insight_id, sp.c.evidence_id, ev.c.item_id))
        .select_from(sp.join(ev, ev.c.evidence_id == sp.c.evidence_id))
        .where(sp.c.insight_id.in_(ins), sp.c.evidence_id.in_(quotes)),
        select(*edge("ATTRIBUTED_TO", ev.c.evidence_id, k.c.canonical_id, ev.c.item_id))
        .select_from(ev.join(k, k.c.entity_id == ev.c.speaker_id))
        .where(ev.c.evidence_id.in_(quotes), k.c.canonical_id.in_(ents)),
        select(*edge("STATED_BY", a.c.insight_id, k.c.canonical_id, a.c.item_id))
        .select_from(a.join(k, k.c.entity_id == a.c.person_id))
        .where(a.c.insight_id.in_(ins), k.c.canonical_id.in_(ents)),
    )
    hits = []
    seen = set()
    for row in conn.execute(statement):
        key = (row.edge, row.frm, row.dst)
        if key in seen:
            continue
        seen.add(key)
        e = EDGES[row.edge]
        hits.append(_Hit(row.edge, (e.source, row.frm), (e.target, row.dst), row.item_id, row.prop))
    return hits


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _groups(conn: Connection, s: _Slice) -> tuple[dict[Ref, list[str]], dict[str, dict[str, Any]]]:
    """Node ``groups`` and the groups catalogue (CRFC-004 §2), from the published cluster build, in
    one statement: each entity node's groups (a topic is in a group when a member of its canonical
    is), each insight node's, and a catalogue row (``id``, ``kind``, ``label``, ``parent_id``) for
    every group a node is in and for its parent, so a storyline's super-theme can be drawn. Ids are
    ``cluster_ids.natural_id``. Nothing is read for a slice with no entity or insight node."""
    ents = sorted(ref[1] for ref in s.nodes if ref[0] == ENTITY)
    ins = sorted(ref[1] for ref in s.nodes if ref[0] == INSIGHT)
    if not ents and not ins:
        return {}, {}
    m, k = cluster_members, entities.alias("gk")
    build = (
        select(cluster_builds.c.build_id)
        .where(cluster_builds.c.state == "published")
        .scalar_subquery()
    )
    members = union_all(
        select(
            literal(ENTITY).label("node_kind"),
            k.c.canonical_id.label("node_key"),
            m.c.cluster_id,
        )
        .select_from(m.join(k, k.c.entity_id == m.c.entity_id))
        .where(
            m.c.build_id == build,
            k.c.canonical_id == func.any(bindparam("group_ents", ents, type_=ARRAY(BigInteger))),
        ),
        select(literal(INSIGHT), m.c.insight_id, m.c.cluster_id).where(
            m.c.build_id == build,
            m.c.insight_id == func.any(bindparam("group_ins", ins, type_=ARRAY(BigInteger))),
        ),
    ).cte("gm")
    c, own, up = clusters, cluster_ids.alias("gi"), cluster_ids.alias("gp")
    referenced = select(members.c.cluster_id)
    parents = select(c.c.parent_id).where(
        c.c.build_id == build, c.c.cluster_id.in_(referenced), c.c.parent_id.is_not(None)
    )
    statement = union_all(
        select(
            literal("member").label("row"),
            members.c.node_kind,
            members.c.node_key,
            own.c.natural_id,
            cast(null(), Text).label("kind"),
            cast(null(), Text).label("label"),
            cast(null(), Text).label("parent"),
        ).select_from(members.join(own, own.c.cluster_id == members.c.cluster_id)),
        select(
            literal("group"),
            cast(null(), Text),
            cast(null(), BigInteger),
            own.c.natural_id,
            c.c.kind,
            c.c.label,
            up.c.natural_id,
        )
        .select_from(
            c.join(own, own.c.cluster_id == c.c.cluster_id).outerjoin(
                up, up.c.cluster_id == c.c.parent_id
            )
        )
        .where(
            c.c.build_id == build,
            c.c.cluster_id.in_(referenced) | c.c.cluster_id.in_(parents),
        ),
    )
    of: dict[Ref, set[str]] = {}
    catalogue: dict[str, dict[str, Any]] = {}
    for row in conn.execute(statement):
        if row.row == "member":
            of.setdefault((str(row.node_kind), int(row.node_key)), set()).add(str(row.natural_id))
        else:
            catalogue[str(row.natural_id)] = {
                "id": str(row.natural_id),
                "kind": row.kind,
                "label": row.label,
                "parent_id": row.parent,
            }
    return {ref: sorted(ids) for ref, ids in of.items()}, catalogue


def _item_counts(conn: Connection, s: _Slice) -> Optional[dict[int, int]]:
    """``with: counts``: each entity node's corpus-wide item count, from the published
    projections build (``entity_item_counts``, keyed by canonical id), in one statement. None
    when counts were not asked for or no build is published: the nodes then carry no
    ``item_count``. An entity the build has no row for has 0 items."""
    if not s.counts or s.graph_build_id is None:
        return None
    ids = sorted(ref[1] for ref in s.nodes if ref[0] == ENTITY)
    if not ids:
        return {}
    rows = conn.execute(
        select(entity_item_counts.c.entity_id, entity_item_counts.c["items"]).where(
            entity_item_counts.c.build_id == s.graph_build_id,
            entity_item_counts.c.entity_id
            == func.any(bindparam("count_ids", ids, type_=ARRAY(BigInteger))),
        )
    )
    return {int(row.entity_id): int(row[1]) for row in rows}


def _finish(conn: Connection, s: _Slice) -> dict[str, Any]:
    """The tail: nodes' rows, items' rows, the induced edges; then ``graph-slice/1``. A node
    whose item is not in the items' rows is dropped, with every edge that touches it."""
    nodes = _node_rows(conn, s)
    items = _item_rows(conn, s)
    induced = _induced(conn, s)
    counts = _item_counts(conn, s)
    node_groups, catalogue = _groups(conn, s)

    def present(ref: Ref) -> bool:
        if ref[0] == ITEM:
            return ref[1] in items
        node = nodes.get(ref)
        return node is not None and (node["item_id"] is None or node["item_id"] in items)

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
        if not present(ref):
            continue
        if ref[0] == ITEM:
            row = items[ref[1]]
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
        node = nodes[ref]
        if node.get("held"):
            # A stub (decision 8): the client holds the node; the new edge does not dangle.
            out_nodes.append({"id": node["natural_id"], "type": node["type"], "held": True})
            continue
        entry: dict[str, Any] = {
            "id": node["natural_id"],
            "type": node["type"],
            "label": node["label"],
            "properties": node["properties"],
            "groups": node_groups.get(ref, []),
        }
        if node["item_id"] is not None:
            entry["item_id"] = item_natural(node["item_id"])
        if counts is not None and ref[0] == ENTITY:
            entry["item_count"] = counts.get(ref[1], 0)
        out_nodes.append(entry)

    def recency(item_id: Optional[int]) -> float:
        if item_id is None or item_id not in items or items[item_id]["published_at"] is None:
            return float("-inf")
        return float(items[item_id]["published_at"].timestamp())

    candidates: list[tuple[float, float, str, dict[str, Any]]] = []
    for hit in induced:
        if not (present(hit.frm) and present(hit.dst)):
            continue
        if hit.item_id is not None and hit.item_id not in items:
            continue
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
        if not (present(a) and present(b)):
            continue
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
    # IN_GROUP (decision 10): a topic to each group it is in, a group to its parent; read with
    # the groups, so no statement of its own. Its ``to`` resolves in the groups catalogue.
    groups_out = _catalogue(out_nodes, catalogue)
    links = [(n["id"], g) for n in out_nodes if n["type"] == "Topic" for g in n.get("groups", [])]
    links += [(g["id"], g["parent_id"]) for g in groups_out if g["parent_id"] is not None]
    for frm, to in links:
        entry = {
            "type": IN_GROUP,
            "class": "structural",
            "from": frm,
            "to": to,
            "weight": 1,
            "properties": {},
        }
        candidates.append((1.0, float("-inf"), f"{IN_GROUP}:{frm}:{to}", entry))
    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
    if len(candidates) > s.edge_budget:
        s.reasons.add("edge_budget")
    out_edges = [c[3] for c in candidates[: s.edge_budget]]

    capped = sorted(node_id(ref) for ref in s.fanout_capped if ref in s.nodes and present(ref))
    truncated: dict[str, Any] = {"reasons": sorted(s.reasons), "fanout_capped": capped}
    if s.nodes_left:
        truncated["nodes_left"] = s.nodes_left
    if s.items_left is not None:
        truncated["items_left"] = s.items_left
    return {
        "format": FORMAT,
        "corpus_revision": s.revision,
        "graph_build_id": s.graph_build_id,
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
        "groups": groups_out,
        "truncated": truncated,
    }


def _catalogue(
    out_nodes: list[dict[str, Any]], catalogue: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """The catalogue rows of the groups the served nodes are in, and of their parents, by id."""
    wanted = {group for node in out_nodes for group in node.get("groups", [])}
    wanted |= {catalogue[g]["parent_id"] for g in wanted if g in catalogue} - {None}
    return [catalogue[g] for g in sorted(wanted) if g in catalogue]


#: Decision 7's presets and the parameters each takes besides ``budget`` and ``edge_budget``.
PRESETS: dict[str, tuple[str, ...]] = {
    "slice.recent": ("window", "sources", "content_types", "min_items", "expand_groups", "with"),
    "slice.items": ("items", "with"),
    "slice.neighbourhood": ("node", "path", "with"),
    "slice.group": ("group", "with"),
    "groups.summary": ("kind",),
}


def catalogue() -> dict[str, Any]:
    """``GET /api/graph/catalogue`` (decision 11): what this build serves, from the engine's own
    constants, so it cannot drift from them: the format, the presets, the node types, the edge
    types with class, ends and whether they are item-scoped, the named paths with their steps, the
    grouping kinds and the static bounds. It reads no row: the same for every revision."""
    kinds = {ITEM: "ContentItem", SOURCE: "Source", INSIGHT: "Insight", EVIDENCE: "Quote"}

    def end(kind: str) -> list[str]:
        return sorted(ENTITY_TYPES.values()) if kind == ENTITY else [kinds[kind]]

    edges = [
        {
            "type": e.name,
            "class": e.cls,
            "from": end(e.source),
            "to": end(e.target),
            "item_scoped": e.item_scoped,
        }
        for e in EDGES.values()
    ]
    edges.append(
        {
            "type": IN_GROUP,
            "class": "structural",
            "from": ["Topic", "group"],
            "to": ["group"],
            "item_scoped": False,
        }
    )
    paths = {
        name: {
            "seeds": {
                seed: [[[list(part) for part in step] for step in branch] for branch in branches]
                for seed, branches in by_seed.items()
            },
            "keep": sorted(keep),
        }
        for name, (by_seed, keep) in PATHS.items()
    }
    return {
        "format": FORMAT,
        "presets": {name: list(params) for name, params in PRESETS.items()},
        "node_types": sorted({*ENTITY_TYPES.values(), *NODE_TYPES.values()}),
        "edge_types": sorted(edges, key=lambda e: str(e["type"])),
        "named_paths": paths,
        "group_kinds": list(GROUP_KINDS),
        "windows": list(WINDOWS),
        "with": sorted(WITH),
        "bounds": {
            "budget": {"default": DEFAULT_BUDGET, "max": MAX_BUDGET},
            "edge_budget": {"default": DEFAULT_EDGE_BUDGET, "max": MAX_EDGE_BUDGET},
            "item_seeds": MAX_ITEM_SEEDS,
            "node_seeds": MAX_NODE_SEEDS,
            "path_steps": MAX_PATH_STEPS,
            "exclude_items": MAX_EXCLUDE_ITEMS,
            "fanout": FANOUT,
            "statement_timeout_ms": STATEMENT_TIMEOUT_MS,
        },
    }
