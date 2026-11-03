"""The reel-clip warm scores in a child process, and reads each episode's files once (2026-10-16).

Prod, that day: the warm scored every episode's reel clips in the api process for 243 s, holding
the GIL, and ``/api/index/stats`` (~4 s alone) gave no answer until it ended, so the post-deploy
smoke failed. The scoring now runs in a child and the api only files the results. Over the
COMMITTED fixture corpus, read only.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import app_cache_warm
from podcast_scraper.server.app_episode_search import (
    episode_ad_spans,
    episode_segments,
    episode_segments_relpath,
    episode_timeline,
    episode_turn_starts,
)
from podcast_scraper.server.app_goal_reel import (
    _EPISODE_CLIPS_NS,
    _episode_clips_uncached,
    _episode_token,
)
from podcast_scraper.server.app_moments import MomentsConfig
from podcast_scraper.server.catalog_cache import cached_catalog

pytestmark = [pytest.mark.integration]

CORPUS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def _rows(root: Path) -> list:
    return [r for r in cached_catalog(root) if r.has_gi]


def test_one_read_gives_what_the_three_reads_gave() -> None:
    rows = _rows(CORPUS)
    assert rows
    with_ads = 0
    for row in rows:
        three = (
            episode_segments(CORPUS, row),
            episode_ad_spans(CORPUS, row),
            episode_turn_starts(CORPUS, row),
        )
        with_ads += bool(three[1])
        assert episode_timeline(CORPUS, row) == three, row.metadata_relative_path
    assert with_ads, "the corpus must exercise ad spans, or this proves nothing about them"


def test_an_episode_with_only_its_ad_free_segments_has_no_ad_spans(tmp_path: Path) -> None:
    root = tmp_path / "v3"
    shutil.copytree(CORPUS, root, ignore=shutil.ignore_patterns("lance_index"))
    row = next(
        r
        for r in _rows(root)
        if (episode_segments_relpath(root, r) or "").endswith(".segments.json")
        and episode_ad_spans(root, r)
    )
    raw = root / str(episode_segments_relpath(root, row))
    adfree = raw.with_name(raw.name[: -len(".segments.json")] + ".adfree.segments.json")
    if not adfree.is_file():
        shutil.copyfile(raw, adfree)
    raw.unlink()
    assert str(episode_segments_relpath(root, row)).endswith(".adfree.segments.json")
    segments, ads, turns = episode_timeline(root, row)
    assert ads == [] == episode_ad_spans(root, row)
    assert segments == episode_segments(root, row) and segments
    assert turns == episode_turn_starts(root, row)


def _cached(root: Path, cfg: MomentsConfig) -> dict:
    return {
        r.metadata_relative_path: perf_cache.cache_peek(
            _EPISODE_CLIPS_NS,
            (str(root), r.metadata_relative_path, repr(cfg)),
            _episode_token(root, r),
        )
        for r in _rows(root)
    }


def test_the_warm_scores_in_a_child_and_files_what_a_request_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from podcast_scraper.server import app_goal_reel

    cfg = MomentsConfig.from_env()
    perf_cache.clear(_EPISODE_CLIPS_NS)

    def no_in_process(*_a: object, **_k: object) -> None:
        raise AssertionError("scored in the api process; the child must do it")

    # Every in-process route to a score is closed: only the child can fill the cache.
    monkeypatch.setattr(app_goal_reel, "_episode_clips_uncached", no_in_process)
    app_cache_warm._warm_reel_clips(CORPUS, cfg)
    monkeypatch.undo()

    cached = _cached(CORPUS, cfg)
    assert all(v is not None for v in cached.values()), "an episode was not filed"
    for row in _rows(CORPUS):
        assert cached[row.metadata_relative_path] == _episode_clips_uncached(CORPUS, row, cfg)
    assert sum(len(v) for v in cached.values()), "no clips at all: the comparison proves nothing"


def test_a_child_that_cannot_run_falls_back_to_scoring_in_process(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = MomentsConfig.from_env()
    perf_cache.clear(_EPISODE_CLIPS_NS)

    def broken(*_a: object, **_k: object) -> None:
        raise subprocess.CalledProcessError(1, "worker", stderr=b"boom")

    monkeypatch.setattr(subprocess, "run", broken)
    app_cache_warm._warm_reel_clips(CORPUS, cfg)
    assert "isolated scoring failed" in caplog.text and "boom" in caplog.text
    assert all(v is not None for v in _cached(CORPUS, cfg).values())
