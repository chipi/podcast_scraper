"""#2359: a drifted episode is never hidden. It stays in short listens and Learn, and plays in sync
through its time map (server) or the listener's own measurement (RFC-129)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from podcast_scraper.server.app_audio_drift import DRIFT_FILE, MAPS_FILE
from podcast_scraper.server.app_goal_reel import goal_reel
from podcast_scraper.server.catalog_cache import cached_catalog
from podcast_scraper.server.slugs import slug_for_row

pytestmark = pytest.mark.integration

SRC = Path(__file__).resolve().parents[2] / "fixtures" / "app-validation-corpus" / "v3"


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "v3"
    shutil.copytree(SRC, root, ignore=shutil.ignore_patterns("lance_index"))
    return root


def test_a_drifted_episode_stays_in_the_reel(corpus: Path) -> None:
    before = goal_reel(corpus, "topic:systems-thinking", minutes=15)
    assert before
    drifted = before[0]
    meta = next(
        r.metadata_relative_path
        for r in cached_catalog(corpus)
        if slug_for_row(r) == drifted.episode_slug
    )
    (corpus / DRIFT_FILE).write_text(
        json.dumps({"episodes": {meta: {"served_s": 400.0, "ref_s": 300.0}}}), encoding="utf-8"
    )
    (corpus / MAPS_FILE).write_text(
        json.dumps({meta: {"anchors": [[0.0, 12.0]]}}), encoding="utf-8"
    )
    after = goal_reel(corpus, "topic:systems-thinking", minutes=15)
    assert [(c.episode_slug, c.moment.insight_id) for c in after] == [
        (c.episode_slug, c.moment.insight_id) for c in before
    ]
