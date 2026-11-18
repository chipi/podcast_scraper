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

Indexed: news items whose ``serveable`` is not ``none``, ``mineable`` is not ``false`` and
``text_status`` is ``full``. A teaser or a video's text is never chunked as if it were the whole
item (the news item contract).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from podcast_scraper.news_items import iter_news_metadata, serveable

logger = logging.getLogger(__name__)

TABLE = "news"
#: KG topic rows for news items (theme and storyline building), the ``aux`` schema.
AUX_TABLE = "news_aux"
_RRF_K = 60
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
    Its title always does; its body too, whatever its ``text_status`` (a teaser's or a video's text
    is still the item's own). Rows are for retrieval only: a hit never carries their text."""
    return serveable(doc) != "none" and doc.get("mineable") is not False


def news_rows(corpus_root: Path) -> List[Dict[str, Any]]:
    """For every indexable news item, a row holding its title, then one row per body paragraph
    (no embeddings yet). The title row is what finds an item whose body never repeats its title,
    and the only row of an item stored without a body."""
    rows: List[Dict[str, Any]] = []
    for f in iter_news_metadata(corpus_root):
        if not indexable(dict(f.doc)):
            continue
        item_id = str(f.doc.get("item_id") or "")
        title = str(f.doc.get("title") or "").strip()
        texts = ([("title", title)] if title else []) + [
            (f"p{n}", para) for n, para in enumerate(_body_paragraphs(f.path))
        ]
        for suffix, text in texts:
            rows.append(
                {
                    "id": f"{item_id}#{suffix}",
                    "item_id": item_id,
                    "text": text,
                    "language": str(f.doc.get("language") or ""),
                    "published_at": str(f.doc.get("published_at") or ""),
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
    if TABLE in db.table_names():
        db.drop_table(TABLE)
    if not rows:
        return 0
    for row in rows:
        row["embedding"] = _embed(row["text"], model_id, allow_download=allow_download)
    table = db.create_table(TABLE, data=rows)
    try:
        table.create_fts_index("text", replace=True)
    except Exception as exc:  # noqa: BLE001 - keyword leg degrades to vector-only
        logger.warning("news FTS index not built: %s", exc)
    return len(rows)


def news_topic_rows(corpus_root: Path) -> List[Dict[str, Any]]:
    """One ``kg_topic`` row per (news item, KG Topic), the podcast rows' shape in the ``aux``
    schema: ``episode_id`` holds the item id, ``show_id`` the news source id, ``source_id`` the
    topic id.
    Every item whose ``mineable`` is not false, served or not (theme building is analysis; a
    surface drops a ``serveable: none`` item when it reads). One item written twice counts once."""
    from podcast_scraper.search.indexer import kg_topic_embed_text

    latest: Dict[str, Any] = {}
    for f in iter_news_metadata(corpus_root):
        if f.doc.get("mineable") is False:
            continue
        item_id = str(f.doc.get("item_id") or "")
        if item_id and (item_id not in latest or f.relative_path > latest[item_id].relative_path):
            latest[item_id] = f
    rows: List[Dict[str, Any]] = []
    for item_id, f in sorted(latest.items()):
        kg_path = f.path.with_name(f.path.name.replace(".metadata.json", ".kg.json"))
        if not kg_path.is_file():
            continue
        try:
            art = json.loads(kg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        published = str(f.doc.get("published_at") or "")[:10]
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
    if AUX_TABLE in db.table_names():
        db.drop_table(AUX_TABLE)
    if not rows:
        return 0
    for row in rows:
        row["embedding"] = _embed(row["text"], model_id, allow_download=allow_download)
    dim = len(rows[0]["embedding"])
    schema = _aux_schema(dim)
    ordered = [{name: row[name] for name in schema.names} for row in rows]
    db.create_table(AUX_TABLE, data=ordered, schema=schema)
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
    where = f"published_at >= '{since.replace(chr(39), '')}'" if since else None
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
        logger.debug("news keyword leg skipped: %s", exc)
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
