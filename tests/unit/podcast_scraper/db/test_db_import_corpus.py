"""A corpus root's sources, items and documents as rows (db.importers.corpus_items)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import func, select, text
from sqlalchemy.engine import Connection

from podcast_scraper.db.importers.corpus_items import import_corpus
from podcast_scraper.db.schema import content_items, documents, item_cards, sources
from podcast_scraper.publishers import Registry
from tests.unit.podcast_scraper.db.test_corpus_rows import _episode
from tests.unit.podcast_scraper.db.test_db_corpus import conn, url  # noqa: F401

_REGISTRY = Registry(publishers=())


def _write(root: Path, run: str, base: str, doc: dict[str, Any], body: str | None = None) -> None:
    run_dir = root / "feeds" / "rss_example" / run
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata" / f"{base}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    if body is not None:
        (run_dir / "transcripts").mkdir(exist_ok=True)
        (run_dir / "transcripts" / f"{base}.txt").write_text(body, encoding="utf-8")


def _count(conn: Connection, table: Any) -> int:  # noqa: F811
    return int(conn.execute(select(func.count()).select_from(table)).scalar_one())


def test_a_corpus_becomes_one_source_its_items_and_their_documents(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\nWorld.\n")
    _write(
        tmp_path,
        "run_20261101-000000",
        "0002 - Two",
        _episode(episode_id="ep-2", guid="guid-2", title="Two"),
        "Second.\n",
    )
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.sources, report.items, report.documents, report.refused) == (1, 2, 2, [])
    assert report.segments == 3
    assert (_count(conn, sources), _count(conn, content_items), _count(conn, documents)) == (
        1,
        2,
        2,
    )
    assert conn.execute(select(sources.c.title)).scalar_one() == "A show"
    conn.rollback()


def test_an_episode_gets_its_card_and_its_flags(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    doc = _episode()
    doc["content"]["transcript_file_path"] = "transcripts/0001 - One.txt"
    doc["summary"] = {"title": "A lede", "bullets": ["First point.", "Second point."]}
    _write(tmp_path, "run_20261101-000000", "0001 - One", doc, "Hello.\n")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.cards == 1
    card = conn.execute(select(item_cards)).mappings().one()
    assert (card["title"], card["source_title"], card["has_transcript"]) == (
        "An episode",
        "A show",
        True,
    )
    assert card["summary_bullets"] and card["lede"]
    item = conn.execute(select(content_items.c.has_transcript)).scalar_one()
    assert item is True
    conn.rollback()


def test_the_newest_run_of_an_episode_is_the_one_imported(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(title="Old"), "Old body.\n")
    _write(tmp_path, "run_20261102-000000", "0001 - One", _episode(title="New"), "New body.\n")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.items, report.documents) == (1, 1)
    assert conn.execute(select(content_items.c.title)).scalar_one() == "New"
    body = conn.execute(text("SELECT string_agg(text, '' ORDER BY seq) FROM corpus.segments"))
    assert body.scalar_one() == "New body.\n"
    conn.rollback()


def test_an_unusable_episode_is_written_not_served(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    doc = _episode(unusable=True, unusable_reason="no English after translation")
    _write(tmp_path, "run_20261101-000000", "0001 - One", doc)
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.items, report.refused) == (1, [])
    assert conn.execute(select(content_items.c.served)).scalar_one() == 0
    conn.rollback()


def test_a_refused_item_writes_nothing_and_the_rest_is_written(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    bad = "0002 - Two"
    _write(
        tmp_path,
        "run_20261101-000000",
        bad,
        _episode(episode_id="ep-2", guid="guid-2"),
        "Body.\n",
    )
    transcripts = tmp_path / "feeds" / "rss_example" / "run_20261101-000000" / "transcripts"
    (transcripts / f"{bad}.cleaned.adfree.txt").write_text("stacked\n", encoding="utf-8")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.items == 1
    assert [reason for _, reason in report.refused] == [f"{bad}: a stacked .cleaned.adfree.txt"]
    assert _count(conn, content_items) == 1 and _count(conn, documents) == 1
    conn.rollback()


def test_a_second_import_writes_nothing_new(conn: Connection, tmp_path: Path) -> None:  # noqa: F811
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    import_corpus(conn, tmp_path, registry=_REGISTRY)
    again = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (again.sources, again.items, again.refused) == (0, 1, [])
    assert (_count(conn, sources), _count(conn, content_items), _count(conn, documents)) == (
        1,
        1,
        1,
    )
    conn.rollback()


def _kg(item_id: str) -> dict[str, Any]:
    return {
        "schema_version": "3.0",
        "item_id": item_id,
        "content_type": "podcast_episode",
        "nodes": [
            {"id": f"episode:{item_id}", "type": "ContentItem", "properties": {"title": "One"}},
            {"id": "podcast:show", "type": "Source", "properties": {"title": "A show"}},
            {"id": "person:ann", "type": "Person", "properties": {"name": "Ann", "role": "host"}},
            {"id": "topic:ai", "type": "Topic", "properties": {"label": "AI"}},
            {"id": "topic:ml", "type": "Topic", "properties": {"label": "ML"}},
        ],
        "edges": [
            {
                "type": "HAS_ITEM",
                "from": "podcast:show",
                "to": f"episode:{item_id}",
                "properties": {},
            },
            {"type": "MENTIONS", "from": "topic:ai", "to": f"episode:{item_id}", "properties": {}},
            {
                "type": "MENTIONS",
                "from": "person:ann",
                "to": f"episode:{item_id}",
                "properties": {},
            },
            {"type": "MENTIONS", "from": "topic:ml", "to": "podcast:show", "properties": {}},
            {
                "type": "CONTRIBUTES_TO",
                "from": "person:ann",
                "to": "podcast:show",
                "properties": {"role": "host"},
            },
        ],
    }


def _write_kg(root: Path, run: str, base: str, kg: dict[str, Any]) -> None:
    path = root / "feeds" / "rss_example" / run / "metadata" / f"{base}.kg.json"
    path.write_text(json.dumps(kg), encoding="utf-8")


def test_an_episodes_kg_gives_its_entities_edges_and_roles(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    import_corpus(conn, tmp_path, registry=_REGISTRY)
    natural_id = "flightcast:01"  # the root item_id; the ContentItem node is episode:<it>
    conn.rollback()
    _write_kg(tmp_path, "run_20261101-000000", "0001 - One", _kg(natural_id))
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.entities, report.entity_edges, report.person_roles) == (3, 3, 1)
    edges = conn.execute(
        text(
            "SELECT e.natural_id, ie.role FROM corpus.item_entities ie "
            "JOIN corpus.entities e USING (entity_id) ORDER BY 1, 2"
        )
    ).all()
    assert [tuple(r) for r in edges] == [
        ("person:ann", "contributes_to"),
        ("person:ann", "mentions"),
        ("topic:ai", "mentions"),
    ]
    roles = conn.execute(text("SELECT role FROM corpus.person_roles")).scalars().all()
    assert roles == ["host"]
    conn.rollback()


def test_a_kg_node_with_no_label_refuses_its_item(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    kg = _kg("episode:x")
    kg["nodes"][3]["properties"] = {}  # topic:ai
    _write_kg(tmp_path, "run_20261101-000000", "0001 - One", kg)
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.items == 0 and len(report.refused) == 1
    assert "no label" in report.refused[0][1]
    assert _count(conn, content_items) == 0
    conn.rollback()


def test_an_episodes_transcripts_are_found_by_its_transcript_path_not_its_metadata_name(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    # The metadata name truncates the title; the transcripts carry it whole, and the metadata
    # names them (content.transcript_file_path), as on snapshot-prod-20261124 (200 episodes).
    run = "run_20261101-000000"
    full = "0001 - How to Fund a Hospital in a Warzone_x"
    doc = _episode()
    doc["content"]["transcript_file_path"] = f"../{run}/transcripts/{full}.txt"
    _write(tmp_path, run, "0001 - How to Fund a Hospital in a Warz_x", doc)
    transcripts = tmp_path / "feeds" / "rss_example" / run / "transcripts"
    transcripts.mkdir()
    (transcripts / f"{full}.txt").write_text("Hello.\nWorld.\n", encoding="utf-8")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.items, report.documents, report.refused) == (1, 1, [])
    conn.rollback()


def test_an_episode_reusing_an_earlier_runs_transcript_is_found_in_that_run(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    # Two episodes of snapshot-prod-20261124 name a transcript in a sibling run of their feed.
    doc = _episode()
    doc["content"]["transcript_file_path"] = "../run_20261101-000000/transcripts/0001 - One.txt"
    _write(
        tmp_path,
        "run_20261101-000000",
        "0002 - Other",
        _episode(episode_id="ep-2", guid="g-2", title="Other"),
        "Other.\n",
    )
    _write(tmp_path, "run_20261102-000000", "0001 - One", doc)
    earlier = tmp_path / "feeds" / "rss_example" / "run_20261101-000000" / "transcripts"
    (earlier / "0001 - One.txt").write_text("Hello.\n", encoding="utf-8")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.items, report.documents, report.refused) == (2, 2, [])
    conn.rollback()


def test_a_transcript_path_leading_out_of_the_feed_refuses_the_episode(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    doc = _episode()
    doc["content"]["transcript_file_path"] = "../../../elsewhere/0001 - One.txt"
    _write(tmp_path, "run_20261101-000000", "0001 - One", doc)
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.items == 0
    assert [reason for _, reason in report.refused] == [
        "its transcript path '../../../elsewhere/0001 - One.txt' leads out of its feed"
    ]
    conn.rollback()


def test_an_absolute_container_transcript_path_is_read_under_this_corpus_root(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    # 121 episodes of snapshot-prod-20261124 state the path the pipeline wrote in its container.
    run = "run_20261101-000000"
    doc = _episode()
    doc["content"][
        "transcript_file_path"
    ] = f"/app/output/feeds/rss_example/{run}/transcripts/0001 - One.txt"
    _write(tmp_path, run, "0001 - One", doc, "Hello.\n")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.items, report.documents, report.refused) == (1, 1, [])
    conn.rollback()


def test_an_absolute_transcript_path_of_another_feed_refuses_the_episode(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    doc = _episode()
    doc["content"]["transcript_file_path"] = "/app/output/feeds/rss_other/run_1/transcripts/x.txt"
    _write(tmp_path, "run_20261101-000000", "0001 - One", doc, "Hello.\n")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.items == 0 and len(report.refused) == 1
    assert "leads out of its feed" in report.refused[0][1]
    conn.rollback()


def test_an_items_has_transcript_follows_its_documents_not_its_card(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    # The metadata names a transcript the run does not hold: the card says what today's card
    # says (from the metadata), the item says there is no transcript document.
    doc = _episode()
    doc["content"]["transcript_file_path"] = "transcripts/0001 - One.txt"
    _write(tmp_path, "run_20261101-000000", "0001 - One", doc)
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.items, report.documents, report.cards) == (1, 0, 1)
    assert conn.execute(select(item_cards.c.has_transcript)).scalar_one() is True
    assert conn.execute(select(content_items.c.has_transcript)).scalar_one() is False


def test_an_unusable_episode_with_a_transcript_has_one(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    # No card (the catalog does not show it), so the flag comes from its documents, as the
    # pipeline's writer sets it (snapshot parity: 2 such items differed).
    doc = _episode(unusable=True, unusable_reason="no English after translation")
    _write(tmp_path, "run_20261101-000000", "0001 - One", doc, "Hello.\n")
    import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert conn.execute(select(content_items.c.has_transcript)).scalar_one() is True
    assert _count(conn, item_cards) == 0
    conn.rollback()


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_FINGERPRINTS = "feeds/rss_example/.podcast_scraper/audio-fingerprints.json"


def _write_fingerprints(root: Path, index: dict[str, Any], feed: str = "rss_example") -> None:
    path = root / "feeds" / feed / ".podcast_scraper" / "audio-fingerprints.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index), encoding="utf-8")


def _fingerprint_rows(conn: Connection) -> list[tuple[Any, ...]]:  # noqa: F811
    return [
        tuple(r)
        for r in conn.execute(
            text(
                "SELECT f.digest, f.identity, i.external_item_id, f.episode_title, "
                "f.transcript_path, f.recorded_at IS NOT NULL, f.duplicates "
                "FROM corpus.audio_fingerprints f "
                "LEFT JOIN corpus.content_items i USING (item_id) ORDER BY f.digest"
            )
        ).all()
    ]


def test_a_feeds_fingerprint_file_gives_its_rows_under_the_feeds_source(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    _write_fingerprints(
        tmp_path,
        {
            _DIGEST_A: {
                "identity": "guid-1",
                "episode_title": "One",
                "transcript_path": "transcripts/0001 - One.txt",
                "recorded_at": "2026-10-18T06:59:10+00:00",
                "duplicates": ["guid-9", "guid-8", "guid-9"],
            },
            _DIGEST_B: {"identity": "https://cdn.example/two.mp3"},
        },
    )
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert (report.fingerprints, report.refused) == (2, [])
    assert _fingerprint_rows(conn) == [
        (
            _DIGEST_A,
            "guid-1",
            "guid-1",
            "One",
            "transcripts/0001 - One.txt",
            True,
            ["guid-8", "guid-9"],
        ),
        (_DIGEST_B, "https://cdn.example/two.mp3", None, None, None, False, []),
    ]
    source = conn.execute(select(sources.c.source_id)).scalar_one()
    scopes = conn.execute(text("SELECT DISTINCT source_id FROM corpus.audio_fingerprints"))
    assert scopes.scalars().all() == [source]
    conn.rollback()


def test_a_fingerprint_file_of_a_feed_with_no_item_is_refused(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    _write_fingerprints(tmp_path, {_DIGEST_A: {"identity": "guid-1"}}, feed="rss_other")
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.fingerprints == 0
    assert report.refused == [
        (
            "feeds/rss_other/.podcast_scraper/audio-fingerprints.json",
            "its feed directory holds items of 0 sources",
        )
    ]
    assert _fingerprint_rows(conn) == []
    conn.rollback()


def test_a_fingerprint_file_of_a_feed_directory_with_two_sources_is_refused(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    other = _episode(episode_id="ep-2", guid="guid-2", title="Two")
    other["feed"] = {**other["feed"], "feed_id": "sha256:" + "c" * 64, "title": "Another show"}
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    _write(tmp_path, "run_20261101-000000", "0002 - Two", other, "Second.\n")
    _write_fingerprints(tmp_path, {_DIGEST_A: {"identity": "guid-1"}})
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.sources == 2
    assert (report.fingerprints, report.refused) == (
        0,
        [(_FINGERPRINTS, "its feed directory holds items of 2 sources")],
    )
    conn.rollback()


def test_a_fingerprint_entry_with_no_identity_or_a_bad_digest_is_refused_alone(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    _write_fingerprints(
        tmp_path,
        {
            _DIGEST_A: {"identity": "guid-1"},
            _DIGEST_B: {"episode_title": "x"},
            "nothex": {"identity": "g"},
        },
    )
    report = import_corpus(conn, tmp_path, registry=_REGISTRY)
    assert report.fingerprints == 1
    assert sorted(report.refused) == [
        (f"{_FINGERPRINTS}#{_DIGEST_B}", "the entry names no identity"),
        (f"{_FINGERPRINTS}#nothex", "the key is not a sha256 digest"),
    ]
    assert [r[0] for r in _fingerprint_rows(conn)] == [_DIGEST_A]
    conn.rollback()


def test_a_second_import_writes_no_fingerprint_twice(
    conn: Connection, tmp_path: Path  # noqa: F811
) -> None:
    _write(tmp_path, "run_20261101-000000", "0001 - One", _episode(), "Hello.\n")
    _write_fingerprints(tmp_path, {_DIGEST_A: {"identity": "guid-1"}})
    assert import_corpus(conn, tmp_path, registry=_REGISTRY).fingerprints == 1
    assert import_corpus(conn, tmp_path, registry=_REGISTRY).fingerprints == 0
    assert len(_fingerprint_rows(conn)) == 1
    conn.rollback()
