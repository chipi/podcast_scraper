"""Unit tests for the pluggable consumer content source (#1078).

Builds a tiny on-disk corpus and exercises LocalCorpusSource directly (no HTTP), plus the
get_content_source factory + override seam for the future DiscoverySource (#1069).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from podcast_scraper.server.app_content_source import (
    _card_bullets,
    _card_lede,
    EpisodeListResult,
    get_content_source,
    LocalCorpusSource,
    transcript_corpus_relpath,
    transcript_relpath,
)
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow


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


class TestCardLede:
    def test_prefers_summary_title(self) -> None:
        assert _card_lede(_row(summary_title="Head", summary_bullets=("b1",))) == "Head"

    def test_falls_back_to_first_bullet_when_no_title(self) -> None:
        r = _row(summary_bullets=(" ", "First bullet.", "Second."))
        assert _card_lede(r) == "First bullet."

    def test_falls_back_to_body_first_sentence(self) -> None:
        lede = _card_lede(_row(summary_text="A first sentence. A second one."))
        assert lede == "A first sentence."

    def test_body_without_sentence_break_used_whole(self) -> None:
        assert _card_lede(_row(summary_text="no period here")) == "no period here"

    def test_returns_none_when_no_summary_content(self) -> None:
        assert _card_lede(_row()) is None
        assert _card_lede(_row(summary_title="   ", summary_text="   ")) is None

    def test_truncates_long_lede_with_ellipsis(self) -> None:
        out = _card_lede(_row(summary_title="x" * 200), max_len=10)
        assert out is not None and len(out) == 10 and out.endswith("…")


class TestCardBullets:
    def test_strips_blanks_and_caps(self) -> None:
        bullets = tuple(f"b{i}" for i in range(20))
        out = _card_bullets(_row(summary_bullets=("  ", *bullets)))
        assert out == [f"b{i}" for i in range(8)]  # blanks dropped, capped at 8

    def test_empty_when_no_bullets(self) -> None:
        assert _card_bullets(_row()) == []


def test_transcript_corpus_relpath_resolves_run_relative() -> None:
    # Nested run layout (prod): transcript_file_path is relative to the run dir.
    assert (
        transcript_corpus_relpath("feeds/F/run_R/metadata/ep.metadata.json", "transcripts/ep.txt")
        == "feeds/F/run_R/transcripts/ep.txt"
    )
    # Flat layout: run dir is "" → just the transcript path.
    assert (
        transcript_corpus_relpath("metadata/ep.metadata.json", "transcripts/ep.txt")
        == "transcripts/ep.txt"
    )


def test_transcript_relpath_prefers_canonical_key() -> None:
    # Canonical key written by the pipeline + read by the search indexer.
    assert transcript_relpath({"transcript_file_path": "t/0001.txt"}) == "t/0001.txt"
    # Defensive fallback for the legacy/fixture key.
    assert transcript_relpath({"transcript_file": "t/0001.txt"}) == "t/0001.txt"
    # Canonical wins when both present.
    assert transcript_relpath({"transcript_file_path": "a", "transcript_file": "b"}) == "a"
    # None / blank → no transcript.
    assert transcript_relpath({}) is None
    assert transcript_relpath({"transcript_file_path": "  "}) is None


def _write_episode(
    root: Path,
    *,
    stem: str,
    feed_id: str,
    feed_title: str,
    episode_id: str,
    title: str,
    published: str,
    with_transcript: bool = True,
    with_gi: bool = True,
) -> None:
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    content: dict = {}
    if with_transcript:
        (root / "transcripts").mkdir(parents=True, exist_ok=True)
        (root / "transcripts" / f"{stem}.txt").write_text("hi", encoding="utf-8")
        content["transcript_file_path"] = f"transcripts/{stem}.txt"
    doc = {
        "feed": {
            "feed_id": feed_id,
            "title": feed_title,
            "url": f"https://{feed_id}.example/f.xml",
        },
        "episode": {
            "episode_id": episode_id,
            "title": title,
            "published_date": published,
            "duration_seconds": 1800,
        },
        "summary": {"title": f"{title} summary", "bullets": ["point one", "point two"]},
        "content": content,
    }
    (root / "metadata" / f"{stem}.metadata.json").write_text(json.dumps(doc), encoding="utf-8")
    if with_gi:
        gi = {
            "episode_id": episode_id,
            "nodes": [
                {
                    "id": "insight:1",
                    "type": "Insight",
                    "properties": {"text": "A claim.", "insight_type": "claim"},
                }
            ],
            "edges": [],
        }
        (root / "metadata" / f"{stem}.gi.json").write_text(json.dumps(gi), encoding="utf-8")


def _corpus(root: Path) -> None:
    # Two feeds; newest-first ordering by publish date. ep3 has no transcript (pending).
    _write_episode(
        root,
        stem="0001",
        feed_id="showa",
        feed_title="Show A",
        episode_id="a1",
        title="A One",
        published="2024-01-01T00:00:00",
    )
    _write_episode(
        root,
        stem="0002",
        feed_id="showa",
        feed_title="Show A",
        episode_id="a2",
        title="A Two",
        published="2024-03-01T00:00:00",
    )
    _write_episode(
        root,
        stem="0003",
        feed_id="showb",
        feed_title="Show B",
        episode_id="b1",
        title="B One",
        published="2024-02-01T00:00:00",
        with_transcript=False,
        with_gi=False,
    )


def test_lists_all_episodes_newest_first(tmp_path: Path) -> None:
    _corpus(tmp_path)
    res = LocalCorpusSource(tmp_path).list_episodes()
    assert isinstance(res, EpisodeListResult)
    assert res.total == 3
    # Newest-first: A Two (Mar) > B One (Feb) > A One (Jan).
    assert [i.title for i in res.items] == ["A Two", "B One", "A One"]


def test_summary_card_fields_are_populated(tmp_path: Path) -> None:
    _corpus(tmp_path)
    items = {i.title: i for i in LocalCorpusSource(tmp_path).list_episodes().items}
    a_two = items["A Two"]
    assert a_two.slug  # deterministic, non-empty
    assert a_two.podcast_title == "Show A"
    assert a_two.feed_id == "showa"
    assert a_two.duration_seconds == 1800
    assert a_two.status == "ready"
    assert a_two.has_transcript is True
    assert a_two.has_gi is True
    # Clean one-line lede = the summary title (NOT the bullets jammed together).
    assert a_two.summary_preview == "A Two summary"
    # Full bullets carried for the card's expand-on-demand insights view.
    assert a_two.summary_bullets == ["point one", "point two"]


def test_pagination_offset_limit(tmp_path: Path) -> None:
    _corpus(tmp_path)
    src = LocalCorpusSource(tmp_path)
    first = src.list_episodes(offset=0, limit=2)
    assert [i.title for i in first.items] == ["A Two", "B One"]
    assert first.total == 3
    second = src.list_episodes(offset=2, limit=2)
    assert [i.title for i in second.items] == ["A One"]
    assert second.total == 3


def test_feed_id_scope(tmp_path: Path) -> None:
    _corpus(tmp_path)
    res = LocalCorpusSource(tmp_path).list_episodes(feed_id="showa")
    assert res.total == 2
    assert {i.feed_id for i in res.items} == {"showa"}


def test_status_filter_pending_vs_ready(tmp_path: Path) -> None:
    _corpus(tmp_path)
    src = LocalCorpusSource(tmp_path)
    pending = src.list_episodes(status="pending")
    assert [i.title for i in pending.items] == ["B One"]
    assert pending.total == 1
    ready = src.list_episodes(status="ready")
    assert {i.title for i in ready.items} == {"A One", "A Two"}
    assert ready.total == 2


def test_empty_corpus(tmp_path: Path) -> None:
    res = LocalCorpusSource(tmp_path).list_episodes()
    assert res.total == 0 and res.items == []


def test_get_content_source_defaults_to_local(tmp_path: Path) -> None:
    state = SimpleNamespace()  # no content_source attribute
    src = get_content_source(state, tmp_path)
    assert isinstance(src, LocalCorpusSource)


def test_get_content_source_honors_override(tmp_path: Path) -> None:
    sentinel = object()
    state = SimpleNamespace(content_source=sentinel)
    assert get_content_source(state, tmp_path) is sentinel
