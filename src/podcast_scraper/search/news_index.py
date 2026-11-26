"""News rows in the search index: one table of their own (News 1.0, RFC-134 §9).

The ``news`` table sits in the corpus's LanceDB directory beside the podcast tables but outside
the backend's tier map (``LanceDBBackend.TABLES``), so no podcast search, reindex sweep or tier
list can reach it: podcast search cannot return news. It is rebuilt whole from the corpus at the
end of each index build (News 1.0's corpus is small; per-item fingerprints can come later) and
dropped by itself.

A row is one paragraph of an item's body. The chunk text is stored for retrieval only (keyword
and vector), never served: :func:`search_news` returns item ids and scores, and the route builds
each hit from the item's metadata (:func:`podcast_scraper.news_items.served_item`), so the
never-served rule (RFC-134 §10, ADR-179) holds at the response layer.

Indexed: every item a surface may show (``serveable`` not ``none``, not marked ``unusable``) whose
``mineable`` is not ``false``: its title as one row, and its body paragraphs when ``text_status`` is
``full``. A teaser or a video's text is never chunked as if it were the whole item (the news item
contract), so such an item is found by its title only.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from podcast_scraper.news_items import current_copies, F_TEXT_STATUS, is_shown
from podcast_scraper.utils.utc import utc_day, utc_sort_key

logger = logging.getLogger(__name__)

TABLE = "news"
#: KG topic rows for news items (theme and storyline building), the ``aux`` schema.
AUX_TABLE = "news_aux"
_RRF_K = 60
#: ``since`` as the search route takes it: a day, compared as text against ``published_at``.
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
#: Each signal's candidate pool, before collapsing to one hit per item.
_POOL = 50


def _body_paragraphs(meta_path: Path) -> List[str]:
    """The item's body (``transcripts/<stem>.txt`` beside its run's ``metadata/``) split into
    paragraphs at blank lines (ontology v3, untimed documents)."""
    stem = meta_path.name[: -len(".metadata.json")]
    body = meta_path.parent.parent / "transcripts" / f"{stem}.txt"
    if not body.is_file():
        return []
    text = body.read_text(encoding="utf-8")
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def indexable(doc: Dict[str, Any]) -> bool:
    """Whether an item goes into the index: one a surface may show and the source lets us mine.
    Rows are for retrieval only: a hit never carries their text."""
    return is_shown(doc) and doc.get("mineable") is not False


def _whole_text(doc: Dict[str, Any]) -> bool:
    """Whether the stored body is the whole article. A ``teaser`` or ``video`` body is never chunked
    as if whole (news-item-1.0, ``text_status``)."""
    raw = doc.get("news_article")
    own: Dict[str, Any] = raw if isinstance(raw, dict) else {}
    return own.get(F_TEXT_STATUS) == "full"


def news_rows(corpus_root: Path) -> List[Dict[str, Any]]:
    """For every indexable news item, a row holding its title, then, for a whole text only, one
    row per body paragraph (no embeddings yet). The title row finds an item whose body never
    repeats its title, and is the only row of a teaser or a video item."""
    rows: List[Dict[str, Any]] = []
    for f in current_copies(corpus_root).values():
        doc = dict(f.doc)
        if not indexable(doc):
            continue
        item_id = str(doc.get("item_id") or "")
        title = str(doc.get("title") or "").strip()
        paras = _body_paragraphs(f.path) if _whole_text(doc) else []
        texts = ([("title", title)] if title else []) + [
            (f"p{n}", para) for n, para in enumerate(paras)
        ]
        for suffix, text in texts:
            rows.append(
                {
                    "id": f"{item_id}#{suffix}",
                    "item_id": item_id,
                    "text": text,
                    "language": str(f.doc.get("language") or ""),
                    # UTC with Z, so the `since` filter (text) compares UTC days.
                    "published_at": utc_sort_key(f.doc.get("published_at")),
                    "metadata_relative_path": f.relative_path,
                }
            )
    return rows


def build_news_table(
    lance_dir: Path, corpus_root: Path, model_id: str, *, allow_download: bool
) -> int:
    """Rebuild the ``news`` table from the corpus; returns the row count. No news drops it."""
    import lancedb  # type: ignore[import-untyped]

    from podcast_scraper.search.two_tier_indexer import _embed

    db = lancedb.connect(str(lance_dir))
    rows = news_rows(corpus_root)
    if not rows:
        if TABLE in db.table_names():
            db.drop_table(TABLE)
        return 0
    # Embed everything before touching the table: a failure here keeps the old table serving.
    for row in rows:
        row["embedding"] = _embed(row["text"], model_id, allow_download=allow_download)
    table = db.create_table(TABLE, data=rows, mode="overwrite")
    try:
        table.create_fts_index("text", replace=True)
    except Exception as exc:  # noqa: BLE001 - keyword leg degrades to vector-only
        logger.warning("news FTS index not built: %s", exc)
    return len(rows)


def news_topic_rows(corpus_root: Path) -> List[Dict[str, Any]]:
    """One ``kg_topic`` row per (news item, KG Topic), the podcast rows' shape in the ``aux``
    schema: ``episode_id`` holds the item id, ``show_id`` the news source id, ``source_id`` the
    topic id.
    Every item a surface may show (its current copy, :func:`current_copies`) whose ``mineable``
    is not false. A withheld or unusable item's topics are left out at the source: the theme
    builder copies a news-only topic's label into the theme file whatever its items' rights, so a
    later reader of that file could otherwise show it (the database session, 2026-11-19)."""
    from podcast_scraper.search.indexer import kg_topic_embed_text

    latest = {
        item_id: f
        for item_id, f in current_copies(corpus_root).items()
        if f.doc.get("mineable") is not False and is_shown(f.doc)
    }
    rows: List[Dict[str, Any]] = []
    for item_id, f in sorted(latest.items()):
        kg_path = f.path.with_name(f.path.name.replace(".metadata.json", ".kg.json"))
        if not kg_path.is_file():
            continue
        try:
            art = json.loads(kg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        published = utc_day(f.doc.get("published_at")) or ""
        for node in art.get("nodes") or []:
            if not isinstance(node, dict) or node.get("type") != "Topic":
                continue
            raw_props = node.get("properties")
            props: Dict[str, Any] = raw_props if isinstance(raw_props, dict) else {}
            text = kg_topic_embed_text(props)
            topic_id = str(node.get("id") or "")
            if not text or not topic_id:
                continue
            rows.append(
                {
                    "id": f"kg_topic:{item_id}:{topic_id}",
                    "text": text,
                    "show_id": str(f.doc.get("source_id") or ""),
                    "episode_id": item_id,
                    "doc_type": "kg_topic",
                    "source_tier": "aux",
                    "publish_date": published,
                    "source_id": topic_id,
                }
            )
    return rows


def build_news_aux_table(
    lance_dir: Path, corpus_root: Path, model_id: str, *, allow_download: bool
) -> int:
    """Rebuild the ``news_aux`` table (``aux`` schema) from the corpus; returns the row count."""
    import lancedb

    from podcast_scraper.search.backends.lancedb_backend import _aux_schema
    from podcast_scraper.search.two_tier_indexer import _embed

    db = lancedb.connect(str(lance_dir))
    rows = news_topic_rows(corpus_root)
    if not rows:
        if AUX_TABLE in db.table_names():
            db.drop_table(AUX_TABLE)
        return 0
    # Embed everything before touching the table: a failure here keeps the old table serving.
    for row in rows:
        row["embedding"] = _embed(row["text"], model_id, allow_download=allow_download)
    dim = len(rows[0]["embedding"])
    schema = _aux_schema(dim)
    ordered = [{name: row[name] for name in schema.names} for row in rows]
    db.create_table(AUX_TABLE, data=ordered, schema=schema, mode="overwrite")
    return len(rows)


def _ranked(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(r.get("item_id") or "") for r in rows]


def search_news(
    lance_dir: Path,
    query: str,
    model_id: str,
    *,
    top_k: int = 10,
    since: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """``(item_id, score)`` for the best-matching news items, one per item, best first. Scores are
    reciprocal-rank fusion of the keyword and vector legs (``1/(60 + rank)`` summed), the same
    form the podcast search uses, so the two lists merge by score. No table: no news."""
    import lancedb

    from podcast_scraper.search.two_tier_indexer import _embed

    db = lancedb.connect(str(lance_dir))
    if TABLE not in db.table_names():
        return []
    table = db.open_table(TABLE)
    if since and not _DAY.fullmatch(since):
        raise ValueError(f"since must be a day, YYYY-MM-DD: {since!r}")
    where = f"published_at >= '{since}'" if since else None
    legs: List[List[str]] = []
    vector = _embed(query, model_id, allow_download=False)
    vq = table.search(vector, vector_column_name="embedding").limit(_POOL)
    if where:
        vq = vq.where(where, prefilter=True)
    legs.append(_ranked(vq.to_list()))
    try:
        kq = table.search(query, query_type="fts").limit(_POOL)
        if where:
            kq = kq.where(where, prefilter=True)
        legs.append(_ranked(kq.to_list()))
    except Exception as exc:  # noqa: BLE001 - no FTS index: vector only
        logger.warning("news keyword search failed, vector results only: %s", exc)
    scores: Dict[str, float] = {}
    for leg in legs:
        seen: set = set()
        rank = 0
        for item_id in leg:
            if item_id in seen:
                continue
            seen.add(item_id)
            rank += 1
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (_RRF_K + rank)
    ordered = sorted(scores, key=lambda i: (-scores[i], i))
    return [(i, scores[i]) for i in ordered[:top_k]]
