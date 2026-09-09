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
    _clean_bullets,
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
    episode_title: str = "Ep",
) -> CatalogEpisodeRow:
    return CatalogEpisodeRow(
        metadata_relative_path="m.json",
        feed_id="f",
        feed_title=None,
        episode_id="e",
        episode_title=episode_title,
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
    def test_uses_the_summary_title(self) -> None:
        assert _card_lede(_row(summary_title="Head", summary_bullets=("b1",))) == "Head"

    def test_does_NOT_fall_back_to_a_bullet(self) -> None:
        """One slot, one shape.

        This used to fall through to the first bullet when there was no title. Every branch is a
        different SHAPE, so the same card line rendered a headline on one episode and a bullet on
        the next — reported from the outside as the line "feeling like it has 2 parts sometimes".
        """
        assert _card_lede(_row(summary_bullets=("First bullet.",))) is None

    def test_does_NOT_fall_back_to_the_prose_body(self) -> None:
        """The prose body is the BIG summary and belongs on the episode detail surface.

        Squeezed into a card line it arrived as half a sentence with an ellipsis, which is the
        third shape the same slot used to take.
        """
        assert _card_lede(_row(summary_text="A first sentence. And more.")) is None
        assert _card_lede(_row(summary_text="no period here")) is None

    def test_returns_none_when_there_is_no_summary_title(self) -> None:
        assert _card_lede(_row()) is None
        assert _card_lede(_row(summary_title="   ", summary_text="   ")) is None

    def test_truncates_a_long_title_with_ellipsis(self) -> None:
        out = _card_lede(_row(summary_title="x" * 40), max_len=10)
        assert out is not None and len(out) == 10 and out.endswith("…")

    def test_skips_a_summary_title_that_just_restates_the_episode_title(self) -> None:
        """The card renders the episode title directly above this line.

        Echoing it prints the same sentence twice and spends the card's only descriptive line
        saying nothing new. Rare — 1 of 36 on the validation corpus — but it is the case that looks
        broken, and with no fallback chain the correct answer is now to render nothing.
        """
        r = _row(
            episode_title="Risk Is a Systems Property", summary_title="Risk Is a Systems Property"
        )
        assert _card_lede(r) is None

    def test_restatement_check_ignores_case_spacing_and_edge_punctuation(self) -> None:
        r = _row(
            episode_title="Risk Is a Systems Property",
            summary_title="  risk is a systems property.  ",
        )
        assert _card_lede(r) is None

    def test_a_title_that_merely_contains_the_episode_title_is_kept(self) -> None:
        r = _row(
            episode_title="Risk",
            summary_title="Risk, and why couplings matter more than components",
        )
        assert _card_lede(r) == "Risk, and why couplings matter more than components"

    def test_strips_blanks_and_caps(self) -> None:
        bullets = tuple(f"b{i}" for i in range(20))
        out = _card_bullets(_row(summary_bullets=("  ", *bullets)))
        assert out == [f"b{i}" for i in range(8)]  # blanks dropped, capped at 8

    def test_empty_when_no_bullets(self) -> None:
        assert _card_bullets(_row()) == []

    def test_card_preview_caps_but_true_count_is_uncapped(self) -> None:
        # The "every episode has 8 key points" bug (BE.1): the card preview pins at 8, so its
        # length is not the real count. `_clean_bullets` is the uncapped count that
        # summary_bullet_count reports, so a 12-bullet episode reads "12", not "8".
        row = _row(summary_bullets=tuple(f"b{i}" for i in range(12)))
        assert len(_card_bullets(row)) == 8
        assert len(_clean_bullets(row)) == 12


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
