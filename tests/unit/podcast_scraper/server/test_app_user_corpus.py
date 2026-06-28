"""Unit tests for the user's heard∪captured episode set (P3 #1120, RFC-101 §1)."""

from __future__ import annotations

from podcast_scraper.server.app_user_corpus import derive_episode_set


def test_heard_requires_threshold_of_known_duration() -> None:
    playback = [
        {"slug": "ep-30pct", "position_seconds": 300},  # 300/1000 = 30% → heard
        {"slug": "ep-29pct", "position_seconds": 290},  # below 30% → not heard
        {"slug": "ep-nodur", "position_seconds": 999},  # unknown duration → not heard alone
    ]
    durations = {"ep-30pct": 1000.0, "ep-29pct": 1000.0}  # ep-nodur absent
    got = derive_episode_set(playback, [], durations)
    assert got == {"ep-30pct"}


def test_captured_always_qualifies_even_without_playback() -> None:
    got = derive_episode_set([], ["ep-hl", "ep-fav", ""], {})
    assert got == {"ep-hl", "ep-fav"}  # blanks dropped


def test_union_of_heard_and_captured() -> None:
    playback = [{"slug": "ep-heard", "position_seconds": 600}]
    durations = {"ep-heard": 1000.0}
    got = derive_episode_set(playback, ["ep-cap"], durations)
    assert got == {"ep-heard", "ep-cap"}


def test_custom_threshold() -> None:
    playback = [{"slug": "ep", "position_seconds": 100}]  # 10%
    durations = {"ep": 1000.0}
    assert derive_episode_set(playback, [], durations, threshold=0.05) == {"ep"}
    assert derive_episode_set(playback, [], durations, threshold=0.5) == set()


def test_malformed_playback_rows_are_skipped() -> None:
    playback: list[dict] = [
        {"position_seconds": 500},
        {"slug": "", "position_seconds": 5},
        {"slug": "ok"},
    ]
    durations = {"ok": 10.0}
    # 'ok' has no position_seconds → defaults 0 → not heard; no crash on missing slug/position
    assert derive_episode_set(playback, [], durations) == set()


def test_user_episode_set_heard_via_playback(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # The heard-via-listening path end-to-end: ≥30% played of a known-duration episode qualifies,
    # below-threshold does not — exercising user_episode_set + slug_durations over a real corpus.
    import json
    from pathlib import Path

    from podcast_scraper.server import app_user_state
    from podcast_scraper.server.app_slugs import slug_for_row
    from podcast_scraper.server.app_user_corpus import user_episode_set
    from podcast_scraper.server.corpus_catalog import build_catalog_rows_cumulative

    root = Path(tmp_path) / "corpus"
    (root / "metadata").mkdir(parents=True)
    for eid, dur in [("ep-heard", 1000), ("ep-skim", 1000)]:
        (root / "metadata" / f"{eid}.metadata.json").write_text(
            json.dumps(
                {
                    "feed": {"feed_id": "f", "title": "S", "url": "https://f.ex/f.xml"},
                    "episode": {
                        "episode_id": eid,
                        "title": eid,
                        "published_date": "2024-01-01T00:00:00",
                        "duration_seconds": dur,
                    },
                    "content": {},
                }
            ),
            encoding="utf-8",
        )
    slugs = {r.episode_id: slug_for_row(r) for r in build_catalog_rows_cumulative(root)}
    data_dir = Path(tmp_path) / "appdata"
    app_user_state.set_playback(data_dir, "u1", slugs["ep-heard"], 400.0, 1)  # 40% → heard
    app_user_state.set_playback(data_dir, "u1", slugs["ep-skim"], 100.0, 1)  # 10% → not heard
    assert user_episode_set(root, data_dir, "u1") == {slugs["ep-heard"]}
