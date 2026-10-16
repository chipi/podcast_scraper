"""#2359 option C: short listens and Learn leave out episodes whose served audio has drifted."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from podcast_scraper.server.app_audio_drift import DRIFT_FILE, drifted_episodes
from podcast_scraper.server.app_catalog_cache import cached_catalog
from podcast_scraper.server.app_goal_reel import goal_reel
from podcast_scraper.server.app_slugs import slug_for_row

pytestmark = pytest.mark.integration

SRC = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus" / "v3"


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "v3"
    shutil.copytree(SRC, root, ignore=shutil.ignore_patterns("lance_index"))
    return root


def test_no_drift_file_means_nothing_has_drifted(corpus: Path) -> None:
    assert drifted_episodes(corpus) == frozenset()


def test_a_drifted_episode_leaves_the_reel(corpus: Path) -> None:
    before = goal_reel(corpus, "topic:systems-thinking", minutes=15)
    assert before
    gone = before[0]
    meta = next(
        r.metadata_relative_path
        for r in cached_catalog(corpus)
        if slug_for_row(r) == gone.episode_slug
    )
    (corpus / DRIFT_FILE).write_text(
        json.dumps({"episodes": {meta: {"served_s": 400.0, "ref_s": 300.0}}}), encoding="utf-8"
    )
    after = goal_reel(corpus, "topic:systems-thinking", minutes=15)
    assert gone.episode_slug not in {c.episode_slug for c in after}
    assert after  # the rest of the reel is still there


def test_an_unreadable_drift_file_is_ignored(corpus: Path) -> None:
    (corpus / DRIFT_FILE).write_text("{not json", encoding="utf-8")
    assert drifted_episodes(corpus) == frozenset()


def test_a_drifted_episode_with_a_time_map_stays_in_the_reel(corpus: Path) -> None:
    from podcast_scraper.server.app_audio_drift import MAPS_FILE

    before = goal_reel(corpus, "topic:systems-thinking", minutes=15)
    kept = before[0]
    meta = next(
        r.metadata_relative_path
        for r in cached_catalog(corpus)
        if slug_for_row(r) == kept.episode_slug
    )
    (corpus / DRIFT_FILE).write_text(
        json.dumps({"episodes": {meta: {"served_s": 400.0, "ref_s": 300.0}}}), encoding="utf-8"
    )
    (corpus / MAPS_FILE).write_text(
        json.dumps({meta: {"anchors": [[0.0, 12.0]]}}), encoding="utf-8"
    )
    after = goal_reel(corpus, "topic:systems-thinking", minutes=15)
    assert kept.episode_slug in {c.episode_slug for c in after}
