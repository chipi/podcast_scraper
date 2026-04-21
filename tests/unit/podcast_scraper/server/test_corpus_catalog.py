"""Unit tests for Corpus Library catalog scan."""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.server.corpus_catalog import (
    aggregate_feeds,
    build_catalog_rows,
    catalog_row_for_metadata_path,
    CatalogEpisodeRow,
    decode_catalog_cursor,
    encode_catalog_cursor,
    episode_list_summary_preview,
    episode_list_topics,
    feed_display_title_by_feed_id,
    filter_rows,
    index_rows_by_feed_episode,
    resolve_feed_display_title,
    slice_page,
)


def _write_meta(
    meta: Path,
    *,
    feed_id: str | None = "feed_a",
    feed_title: str = "Show A",
    episode_id: str = "ep1",
    episode_title: str = "Episode One",
    published: str = "2024-06-15T12:00:00",
    bullets: list[str] | None = None,
) -> None:
    meta.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {
        "feed": {"feed_id": feed_id, "title": feed_title},
        "episode": {
            "episode_id": episode_id,
            "title": episode_title,
            "published_date": published,
        },
    }
    if bullets:
        doc["summary"] = {"title": "S", "bullets": bullets}
    meta.write_text(json.dumps(doc), encoding="utf-8")


def test_summary_text_prefers_raw_then_short(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    mdir.mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {"feed_id": "f", "title": "S"},
        "episode": {"episode_id": "e", "title": "T", "published_date": "2024-01-01"},
        "summary": {
            "title": "Head",
            "bullets": ["b"],
            "short_summary": "  short body  ",
        },
    }
    (mdir / "x.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    rows = build_catalog_rows(root)
    assert len(rows) == 1
    assert rows[0].summary_text == "short body"
    doc2 = {
        "feed": {"feed_id": "f", "title": "S"},
        "episode": {"episode_id": "e2", "title": "T2", "published_date": "2024-02-01"},
        "summary": {
            "title": "H",
            "bullets": [],
            "raw_text": "  raw  ",
            "short_summary": "ignored",
        },
    }
    (mdir / "y.metadata.json").write_text(json.dumps(doc2), encoding="utf-8")
    rows2 = build_catalog_rows(root)
    by_path = {r.metadata_relative_path: r for r in rows2}
    assert by_path["metadata/y.metadata.json"].summary_text == "raw"


def test_build_catalog_rows_sorts_newest_first(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    _write_meta(
        mdir / "old.metadata.json",
        episode_id="o",
        episode_title="Old",
        published="2020-01-01T00:00:00",
    )
    _write_meta(
        mdir / "new.metadata.json",
        episode_id="n",
        episode_title="New",
        published="2024-12-01T00:00:00",
    )
    rows = build_catalog_rows(root)
    assert [r.episode_title for r in rows] == ["New", "Old"]


def test_aggregate_feeds_counts(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    _write_meta(mdir / "a.metadata.json", feed_id="f1", episode_id="1")
    _write_meta(mdir / "b.metadata.json", feed_id="f1", episode_id="2")
    _write_meta(mdir / "c.metadata.json", feed_id="f2", episode_id="3")
    rows = build_catalog_rows(root)
    agg = aggregate_feeds(rows)
    by = {x["feed_id"]: x["episode_count"] for x in agg}
    assert by["f1"] == 2 and by["f2"] == 1


def test_build_catalog_rows_latest_feed_run_only(tmp_path: Path) -> None:
    """Two ``run_*`` trees under the same feed dir: catalog sees the lexicographically last run."""
    root = tmp_path
    doc = {
        "feed": {"feed_id": "feed_a", "title": "Show"},
        "episode": {
            "episode_id": "ep_dup",
            "title": "Same episode",
            "published_date": "2024-06-01T00:00:00",
        },
        "summary": {"title": "S", "bullets": ["a"]},
    }
    text = json.dumps(doc)
    old = (
        root
        / "feeds"
        / "rss_pod"
        / "run_20260416-120000_x"
        / "metadata"
        / "0001 - Same_20260416-120000_x.metadata.json"
    )
    new = (
        root
        / "feeds"
        / "rss_pod"
        / "run_20260417-120000_y"
        / "metadata"
        / "0001 - Same_20260417-120000_y.metadata.json"
    )
    old.parent.mkdir(parents=True, exist_ok=True)
    new.parent.mkdir(parents=True, exist_ok=True)
    old.write_text(text, encoding="utf-8")
    new.write_text(text, encoding="utf-8")

    rows = build_catalog_rows(root)
    assert len(rows) == 1
    assert "run_20260417-120000_y" in rows[0].metadata_relative_path


def test_build_catalog_rows_visual_metadata(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    mdir.mkdir(parents=True, exist_ok=True)
    doc = {
        "feed": {
            "feed_id": "f1",
            "title": "Show",
            "image_url": "https://cdn.example/feed.jpg",
        },
        "episode": {
            "episode_id": "e1",
            "title": "Ep",
            "published_date": "2024-01-01",
            "image_url": "https://cdn.example/ep.jpg",
            "duration_seconds": 125,
            "episode_number": 7,
        },
    }
    (mdir / "x.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    rows = build_catalog_rows(root)
    assert len(rows) == 1
    r = rows[0]
    assert r.feed_image_url == "https://cdn.example/feed.jpg"
    assert r.episode_image_url == "https://cdn.example/ep.jpg"
    assert r.duration_seconds == 125
    assert r.episode_number == 7


def test_aggregate_feeds_image_url_first_non_empty(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "a.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "f", "title": "S"},
                "episode": {"episode_id": "1", "title": "A", "published_date": "2024-01-01"},
            },
        ),
        encoding="utf-8",
    )
    (mdir / "b.metadata.json").write_text(
        json.dumps(
            {
                "feed": {
                    "feed_id": "f",
                    "title": "S",
                    "image_url": "https://cdn.example/cover.png",
                },
                "episode": {"episode_id": "2", "title": "B", "published_date": "2024-02-01"},
            },
        ),
        encoding="utf-8",
    )
    rows = build_catalog_rows(root)
    agg = aggregate_feeds(rows)
    assert len(agg) == 1
    assert agg[0]["image_url"] == "https://cdn.example/cover.png"


def test_index_rows_by_feed_episode(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    _write_meta(mdir / "a.metadata.json", feed_id="f1", episode_id="e1")
    rows = build_catalog_rows(root)
    idx = index_rows_by_feed_episode(rows)
    assert ("f1", "e1") in idx
    assert idx[("f1", "e1")].metadata_relative_path.endswith("a.metadata.json")


def test_episode_list_topics_caps_count_and_length() -> None:
    bullets = tuple(f"b{i}" for i in range(10))
    assert episode_list_topics(bullets, max_items=3) == ["b0", "b1", "b2"]
    long_b = "x" * 200
    assert len(episode_list_topics((long_b,), max_len=12)[0]) == 12
    assert episode_list_topics((long_b,), max_len=12)[0].endswith("…")


def test_filter_rows_topic_q_matches_bullet_or_summary_title(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    _write_meta(
        mdir / "a.metadata.json",
        episode_id="1",
        bullets=["climate policy debate", "other"],
    )
    _write_meta(mdir / "b.metadata.json", episode_id="2", bullets=["unrelated"])
    _write_meta(
        mdir / "c.metadata.json",
        episode_id="3",
        episode_title="No bullets",
        bullets=None,
    )
    # title-only row: add summary title via custom write
    (mdir / "d.metadata.json").write_text(
        json.dumps(
            {
                "feed": {"feed_id": "f", "title": "S"},
                "episode": {
                    "episode_id": "4",
                    "title": "T",
                    "published_date": "2024-01-01",
                },
                "summary": {"title": "Quantum computing overview", "bullets": []},
            },
        ),
        encoding="utf-8",
    )
    rows = build_catalog_rows(root)
    f = filter_rows(rows, topic_q="climate")
    assert {r.episode_id for r in f} == {"1"}
    f2 = filter_rows(rows, topic_q="QUANTUM")
    assert {r.episode_id for r in f2} == {"4"}


def test_filter_rows_until_and_has_gi(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    _write_meta(
        mdir / "early.metadata.json",
        episode_id="1",
        published="2024-01-10T00:00:00",
    )
    (mdir / "early.gi.json").write_text("{}", encoding="utf-8")
    _write_meta(
        mdir / "late.metadata.json",
        episode_id="2",
        published="2024-03-20T00:00:00",
    )
    (mdir / "late.gi.json").write_text("{}", encoding="utf-8")
    rows = build_catalog_rows(root)
    f = filter_rows(rows, since="2024-01-01", until="2024-02-01")
    assert {r.episode_id for r in f} == {"1"}
    _write_meta(
        mdir / "nogi.metadata.json",
        episode_id="3",
        published="2024-06-01T00:00:00",
    )
    rows2 = build_catalog_rows(root)
    missing = filter_rows(rows2, has_gi=False)
    assert {r.episode_id for r in missing} == {"3"}
    assert all(not r.has_gi for r in missing)


def test_filter_rows_feed_id_empty_string(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    _write_meta(mdir / "a.metadata.json", feed_id="", episode_id="1")
    _write_meta(mdir / "b.metadata.json", feed_id="x", episode_id="2")
    rows = build_catalog_rows(root)
    f = filter_rows(rows, feed_id="")
    assert len(f) == 1 and f[0].episode_id == "1"


def test_slice_page_and_cursor_roundtrip() -> None:
    rows = [type("R", (), {"metadata_relative_path": str(i)})() for i in range(5)]
    page, nxt = slice_page(rows, 0, 2)
    assert len(page) == 2
    assert nxt is not None
    off = decode_catalog_cursor(nxt)
    page2, nxt2 = slice_page(rows, off, 2)
    assert len(page2) == 2
    assert nxt2 is not None
    off2 = decode_catalog_cursor(nxt2)
    page3, nxt3 = slice_page(rows, off2, 10)
    assert len(page3) == 1
    assert nxt3 is None


def test_encode_decode_cursor_zero() -> None:
    assert decode_catalog_cursor(None) == 0
    assert decode_catalog_cursor("") == 0
    assert decode_catalog_cursor("not-valid") == 0
    c = encode_catalog_cursor(3)
    assert decode_catalog_cursor(c) == 3


def _row(
    *,
    summary_title: str | None = None,
    summary_bullets: tuple[str, ...] = (),
    summary_text: str | None = None,
) -> CatalogEpisodeRow:
    return CatalogEpisodeRow(
        metadata_relative_path="m.json",
        feed_id="f",
        feed_title=None,
        episode_id="e",
        episode_title="Ep",
        publish_date=None,
        summary_title=summary_title,
        summary_bullets=summary_bullets,
        summary_text=summary_text,
        gi_relative_path="m.gi.json",
        kg_relative_path="m.kg.json",
        bridge_relative_path="m.bridge.json",
        has_gi=False,
        has_kg=False,
        has_bridge=False,
    )


def test_episode_list_summary_preview_title_bullets() -> None:
    r = _row(summary_title="Head", summary_bullets=("one", "two"))
    assert episode_list_summary_preview(r) == "Head — one · two"


def test_episode_list_summary_preview_bullets_only() -> None:
    r = _row(summary_bullets=("only",))
    assert episode_list_summary_preview(r) == "only"


def test_episode_list_summary_preview_body_fallback() -> None:
    r = _row(summary_text="x" * 250)
    prev = episode_list_summary_preview(r)
    assert prev is not None
    assert prev.endswith("…")
    assert len(prev) == 201


def test_catalog_row_for_metadata_path_detects_gi_kg(tmp_path: Path) -> None:
    root = tmp_path
    mdir = root / "metadata"
    _write_meta(mdir / "ep.metadata.json")
    (mdir / "ep.gi.json").write_text("{}", encoding="utf-8")
    (mdir / "ep.kg.json").write_text("{}", encoding="utf-8")
    (mdir / "ep.bridge.json").write_text("{}", encoding="utf-8")
    rel = "metadata/ep.metadata.json"
    row = catalog_row_for_metadata_path(root, rel)
    assert row is not None
    assert row.has_gi and row.has_kg
    assert row.has_bridge
    assert row.bridge_relative_path == "metadata/ep.bridge.json"


def test_feed_display_title_by_feed_id_first_nonempty_wins() -> None:
    rows = [
        CatalogEpisodeRow(
            metadata_relative_path="a.json",
            feed_id="f",
            feed_title=None,
            episode_id="1",
            episode_title="A",
            publish_date=None,
            summary_title=None,
            summary_bullets=(),
            summary_text=None,
            gi_relative_path="a.gi.json",
            kg_relative_path="a.kg.json",
            bridge_relative_path="a.bridge.json",
            has_gi=False,
            has_kg=False,
            has_bridge=False,
        ),
        CatalogEpisodeRow(
            metadata_relative_path="b.json",
            feed_id="f",
            feed_title="Podcast Name",
            episode_id="2",
            episode_title="B",
            publish_date=None,
            summary_title=None,
            summary_bullets=(),
            summary_text=None,
            gi_relative_path="b.gi.json",
            kg_relative_path="b.kg.json",
            bridge_relative_path="b.bridge.json",
            has_gi=False,
            has_kg=False,
            has_bridge=False,
        ),
    ]
    assert feed_display_title_by_feed_id(rows) == {"f": "Podcast Name"}


def test_resolve_feed_display_title_prefers_row_then_map() -> None:
    row = CatalogEpisodeRow(
        metadata_relative_path="x.json",
        feed_id="f",
        feed_title="Local",
        episode_id="e",
        episode_title="E",
        publish_date=None,
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="x.gi.json",
        kg_relative_path="x.kg.json",
        bridge_relative_path="x.bridge.json",
        has_gi=False,
        has_kg=False,
        has_bridge=False,
    )
    assert resolve_feed_display_title(row, {"f": "Other"}) == "Local"
    row2 = CatalogEpisodeRow(
        metadata_relative_path="y.json",
        feed_id="f",
        feed_title=None,
        episode_id="e2",
        episode_title="E2",
        publish_date=None,
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="y.gi.json",
        kg_relative_path="y.kg.json",
        bridge_relative_path="y.bridge.json",
        has_gi=False,
        has_kg=False,
        has_bridge=False,
    )
    assert resolve_feed_display_title(row2, {"f": "Shared"}) == "Shared"
