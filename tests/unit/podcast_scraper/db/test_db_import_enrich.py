"""The clustering's and the enrichers' files as rows (db.importers.corpus_groupings,
db.importers.corpus_enrichments, through import_corpus)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.engine import Connection

from podcast_scraper.db.importers.corpus_items import import_corpus
from podcast_scraper.db.schema import content_items, documents
from tests.unit.podcast_scraper.db.test_corpus_rows import _episode
from tests.unit.podcast_scraper.db.test_db_corpus import conn, url  # noqa: F401
from tests.unit.podcast_scraper.db.test_db_import_corpus import _kg, _REGISTRY, _write, _write_kg

_RUN = "run_20261101-000000"
_ITEM = "flightcast:01"
_EP = "ep-1"  # the item's natural id (episode.episode_id)


def _corpus(root: Path) -> None:
    _write(root, _RUN, "0001 - One", _episode(), "Hello.\n")
    _write_kg(root, _RUN, "0001 - One", _kg(_ITEM))


def _json(root: Path, relpath: str, doc: Any) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _envelope(enricher_id: str, data: Any, status: str = "ok") -> dict[str, Any]:
    return {
        "enricher_id": enricher_id,
        "enricher_version": "1.0.0",
        "computed_at": "2026-11-16T03:56:23Z",
        "schema_version": "1.0",
        "status": status,
        "data": data,
    }


def _rows(conn: Connection, sql: str) -> list[tuple[Any, ...]]:  # noqa: F811
    return [tuple(r) for r in conn.execute(text(sql)).all()]


def _groupings(root: Path) -> None:
    _json(
        root,
        "search/topic_clusters.json",
        {
            "schema_version": "3",
            "cluster_count": 1,
            "clusters": [
                {
                    "canonical_label": "ai",
                    "cil_alias_target_topic_id": "topic:ai",
                    "graph_compound_parent_id": "tc:ai",
                    "member_count": 3,
                    "members": [
                        {
                            "topic_id": "topic:ai",
                            "similarity_to_centroid": 0.9,
                            "episode_ids": [_ITEM],
                        },
                        {
                            "topic_id": "topic:ml",
                            "similarity_to_centroid": 0.8,
                            "episode_ids": [_ITEM],
                        },
                        {"topic_id": "topic:nowhere", "similarity_to_centroid": 0.7},
                    ],
                    "news_members": [],
                }
            ],
            "id_history": {"successors": {"tc:old-ai": "tc:ai"}, "gone": ["tc:gone"]},
        },
    )
    _json(
        root,
        "enrichments/topic_theme_clusters.json",
        _envelope(
            "topic_theme_clusters",
            {
                "schema_version": "1",
                "cluster_count": 1,
                "clusters": [
                    {
                        "cluster_type": "theme",
                        "canonical_label": "machines",
                        "graph_compound_parent_id": "thc:machines",
                        "member_count": 2,
                        "members": [
                            {
                                "topic_id": "topic:ai",
                                "strength": 1.0,
                                "anchor": True,
                                "lift_to_cluster": 5.0,
                            },
                            {
                                "topic_id": "topic:ml",
                                "strength": 0.5,
                                "anchor": False,
                                "lift_to_cluster": 2.0,
                            },
                        ],
                        "state": "active",
                        "aliases": ["thinking machines"],
                        "created_at": 1794801386,
                        "last_active_at": 1794801386,
                        "super_theme_id": "sth:tech",
                        "super_theme_label": "tech",
                        "super_theme_is_long_tail": False,
                        "renamed_from": "computers",
                    }
                ],
                "id_history": {"successors": {}, "gone": []},
                "registry": {
                    "as_of": 1794801386.8,
                    "entries": {},
                    "candidates": [],
                    "seeded_from": "x",
                },
            },
        ),
    )


def test_the_clustering_files_become_one_published_build_of_groups(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _corpus(tmp_path)
    _groupings(tmp_path)
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.groups, report.group_members) == (3, 4)
    assert _rows(
        conn, "SELECT state, registry_state->>'seeded_from' FROM corpus.cluster_builds"
    ) == [("published", "x")]
    groups = _rows(
        conn,
        "SELECT i.natural_id, c.kind, c.label, p.natural_id, a.natural_id, c.state, c.aliases,"
        " c.detail->>'renamed_from' FROM corpus.clusters c"
        " JOIN corpus.cluster_ids i USING (cluster_id)"
        " LEFT JOIN corpus.cluster_ids p ON p.cluster_id = c.parent_id"
        " LEFT JOIN corpus.entities a ON a.entity_id = c.anchor_entity_id ORDER BY 1",
    )
    assert groups == [
        ("sth:tech", "super_theme", "tech", None, None, None, [], None),
        ("tc:ai", "theme", "ai", None, "topic:ai", None, [], None),
        (
            "thc:machines",
            "storyline",
            "machines",
            "sth:tech",
            "topic:ai",
            "active",
            ["thinking machines"],
            "computers",
        ),
    ]
    members = _rows(
        conn,
        "SELECT i.natural_id, e.natural_id, m.is_anchor, round(m.strength::numeric, 2)::float8"
        " FROM corpus.cluster_members m JOIN corpus.cluster_ids i USING (cluster_id)"
        " JOIN corpus.entities e USING (entity_id) ORDER BY 1, 2",
    )
    assert members == [
        ("tc:ai", "topic:ai", False, 0.9),
        ("tc:ai", "topic:ml", False, 0.8),
        ("thc:machines", "topic:ai", True, 1.0),
        ("thc:machines", "topic:ml", False, 0.5),
    ]
    assert (
        "search/topic_clusters.json",
        "1 members name a topic the corpus has no row for, e.g. topic:nowhere",
    ) in report.refused
    history = _rows(
        conn,
        "SELECT old_natural_id, new_natural_id, change FROM corpus.cluster_id_history ORDER BY 1",
    )
    assert history == [("tc:gone", None, "gone"), ("tc:old-ai", "tc:ai", "successor")]
    conn.rollback()


def test_a_second_import_writes_no_group_or_output_twice(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _corpus(tmp_path)
    _groupings(tmp_path)
    _json(
        tmp_path,
        "enrichments/grounding_rate.json",
        _envelope("grounding_rate", {"corpus_rate": 0.9}),
    )
    import_corpus(conn, tmp_path, registry=_REGISTRY)
    again = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (again.groups, again.group_members, again.enrichment_outputs) == (0, 0, 0)
    assert _rows(conn, "SELECT count(*) FROM corpus.clusters") == [(3,)]
    assert _rows(conn, "SELECT count(*) FROM corpus.enrichment_outputs") == [(1,)]
    conn.rollback()


def test_the_clustering_history_lines_become_events_with_their_kind(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _corpus(tmp_path)
    lines = [
        {"ts": 1791979247.5, "kind": "theme", "changes": {"gone": [{"id": "tc:q"}]}, "kept": 2},
        {"ts": 1791979300.0, "changes": {"created": [{"id": "thc:new"}]}, "active": 1},
        {"changes": {}},
    ]
    path = tmp_path / "enrichments" / "clustering_history.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    events = _rows(
        conn,
        "SELECT kind, changes->>'kept', changes->>'active', source FROM corpus.cluster_events"
        " ORDER BY at",
    )
    assert events == [
        ("theme", "2", None, "clustering_history.jsonl"),
        ("storyline", None, "1", "clustering_history.jsonl"),
    ]
    assert (
        "enrichments/clustering_history.jsonl:3",
        "the line names no time or no kind",
    ) in report.refused
    conn.rollback()


def test_an_insight_cluster_member_is_its_insight_row_and_one_with_no_row_is_refused(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _corpus(tmp_path)
    import_corpus(conn, tmp_path, registry=_REGISTRY)
    item_id = conn.execute(select(content_items.c.item_id)).scalar_one()
    document_id = conn.execute(select(documents.c.document_id)).scalar_one()
    conn.execute(
        text(
            "INSERT INTO corpus.insights (natural_id, item_id, document_id, insight_type, rank,"
            " routing_tag, text_origin, text) VALUES ('insight:a', :i, :d, 'claim', 0, 'surface',"
            " 'authored', 'A claim.')"
        ),
        {"i": item_id, "d": document_id},
    )
    _json(
        tmp_path,
        "search/insight_clusters.json",
        {
            "clusters": [
                {
                    "cluster_id": "ic:a-claim",
                    "canonical_insight": "A claim.",
                    "member_count": 2,
                    "episode_count": 1,
                    "cross_episode": False,
                    "episode_ids": [_ITEM],
                    "members": [
                        {
                            "insight_id": "insight:a",
                            "similarity_to_centroid": 0.95,
                            "supporting_quotes": [],
                        },
                        {"insight_id": "insight:gone", "similarity_to_centroid": 0.9},
                    ],
                }
            ]
        },
    )
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.groups, report.group_members) == (1, 1)
    got = _rows(
        conn,
        "SELECT c.label, c.detail->>'cross_episode', i.natural_id FROM corpus.cluster_members m"
        " JOIN corpus.clusters c USING (build_id, cluster_id)"
        " JOIN corpus.insights i USING (insight_id)",
    )
    assert got == [("A claim.", "false", "insight:a")]
    assert (
        "search/insight_clusters.json",
        "1 members name a insight the corpus has no row for, e.g. insight:gone",
    ) in report.refused
    conn.rollback()


def test_enricher_outputs_are_written_per_corpus_entity_pair_and_item(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _corpus(tmp_path)
    _json(
        tmp_path,
        "enrichments/grounding_rate.json",
        _envelope("grounding_rate", {"corpus_rate": 0.98}),
    )
    _json(
        tmp_path,
        "enrichments/person_web.json",
        _envelope(
            "person_web",
            {
                "provider": "wikidata",
                "persons": [
                    {"person_id": "person:ann", "bio": "Ann writes."},
                    {"person_id": "person:nobody", "bio": "?"},
                ],
            },
        ),
    )
    _json(
        tmp_path,
        "enrichments/topic_similarity.json",
        _envelope(
            "topic_similarity",
            {
                "top_k": 7,
                "topics": [
                    {
                        "topic_id": "topic:ml",
                        "top_k": [{"topic_id": "topic:ai", "similarity": 0.6}],
                    },
                ],
            },
        ),
    )
    _json(
        tmp_path,
        "enrichments/topic_cooccurrence_corpus.json",
        _envelope(
            "topic_cooccurrence_corpus",
            {
                "episode_count": 1,
                "pairs": [
                    {
                        "topic_a_id": "topic:ml",
                        "topic_b_id": "topic:ai",
                        "episode_count": 1,
                        "lift": 2.5,
                        "pmi": 1.3,
                        "npmi": 0.4,
                    },
                ],
            },
        ),
    )
    _json(tmp_path, "enrichments/temporal_velocity.json", _envelope("temporal_velocity", {}))
    _json(tmp_path, "enrichments/topic_similarity.delta_cursor.json", {"x": 1})
    _json(tmp_path, "enrichments/broken.json", _envelope("broken", None, status="error"))
    _json(
        tmp_path,
        f"feeds/rss_example/{_RUN}/metadata/enrichments/0001 - One.insight_density.json",
        _envelope("insight_density", {"episode_id": _EP, "total_insights": 3}),
    )
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    outputs = _rows(
        conn,
        "SELECT o.enricher_id, o.scope, i.natural_id, e.natural_id, o.output::text"
        " FROM corpus.enrichment_outputs o LEFT JOIN corpus.content_items i USING (item_id)"
        " LEFT JOIN corpus.entities e USING (entity_id) ORDER BY 1, 2",
    )
    assert outputs == [
        ("grounding_rate", "corpus", None, None, '{"corpus_rate": 0.98}'),
        ("insight_density", "episode", _EP, None, '{"episode_id": "ep-1", "total_insights": 3}'),
        ("person_web", "corpus", None, None, '{"provider": "wikidata"}'),
        (
            "person_web",
            "entity",
            None,
            "person:ann",
            '{"bio": "Ann writes.", "person_id": "person:ann"}',
        ),
        ("topic_cooccurrence_corpus", "corpus", None, None, '{"episode_count": 1}'),
        ("topic_similarity", "corpus", None, None, '{"top_k": 7}'),
    ]
    pairs = _rows(
        conn,
        "SELECT p.enricher_id, a.natural_id, b.natural_id, round(p.score::numeric, 1)::float8,"
        " p.detail->>'npmi'"
        " FROM corpus.entity_pairs p JOIN corpus.entities a ON a.entity_id = p.a"
        " JOIN corpus.entities b ON b.entity_id = p.b ORDER BY 1",
    )
    ai_first = (
        conn.execute(
            text(
                "SELECT natural_id FROM corpus.entities"
                " WHERE natural_id IN ('topic:ai', 'topic:ml')"
                " ORDER BY entity_id"
            )
        )
        .scalars()
        .first()
    )
    co_a, co_b = ("topic:ai", "topic:ml") if ai_first == "topic:ai" else ("topic:ml", "topic:ai")
    assert pairs == [
        ("topic_cooccurrence_corpus", co_a, co_b, 2.5, "0.4"),
        ("topic_similarity", "topic:ml", "topic:ai", 0.6, None),
    ]
    assert _rows(conn, "SELECT enricher_id, state FROM corpus.enrichment_builds ORDER BY 1") == [
        ("grounding_rate", "published"),
        ("insight_density", "published"),
        ("person_web", "published"),
        ("topic_cooccurrence_corpus", "published"),
        ("topic_similarity", "published"),
    ]
    assert ("enrichments/broken.json", "status 'error'") in report.refused
    assert (
        "enrichments/person_web.json",
        "1 rows name an entity the corpus has no row for, e.g. person:nobody",
    ) in report.refused
    conn.rollback()


def test_an_episode_output_of_an_older_run_is_not_read(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _corpus(tmp_path)
    old = "run_20261001-000000"
    _write(tmp_path, old, "0001 - One", _episode(), "Hello.\n")
    _json(
        tmp_path,
        f"feeds/rss_example/{old}/metadata/enrichments/0001 - One.insight_density.json",
        _envelope("insight_density", {"episode_id": _EP, "total_insights": 1}),
    )
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.enrichment_outputs == 0
    assert _rows(conn, "SELECT count(*) FROM corpus.enrichment_builds") == [(0,)]
    conn.rollback()
