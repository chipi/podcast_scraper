"""A rising theme's row carries its size, whatever its rank by size (2026-10-12).

The client looked a theme's "(N)" up in the 50 widest themes, so any rising theme narrower than
those showed no count: on the committed corpus (82 themes) the top rising one, and on prod nearly
all of them. The size now rides on the trending row, from the same source the picker lists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.search.topic_clusters import top_themes_by_member_count
from podcast_scraper.server.app_momentum import MomentumConfig, trending

pytestmark = [pytest.mark.integration]

CORPUS = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "app-validation-corpus" / "v3"


def test_every_trending_theme_carries_its_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20T00:00:00Z")
    rows = trending(
        CORPUS,
        tmp_path,
        kind="theme",
        limit=1000,
        window="3m",
        config=MomentumConfig.from_dict({"trend": {"min_total": 1}}),
    )
    sizes = {c["id"]: c["size"] for c in top_themes_by_member_count(CORPUS, 1_000_000)}
    widest = {c["id"] for c in top_themes_by_member_count(CORPUS, 50)}
    assert rows and all(r.size == sizes[r.entity_id] for r in rows)
    assert any(r.entity_id not in widest for r in rows), "a rising theme outside the widest 50"


def test_other_kinds_carry_no_size(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("APP_TRENDING_NOW", "2026-07-20T00:00:00Z")
    rows = trending(
        CORPUS,
        tmp_path,
        kind="topic",
        limit=50,
        config=MomentumConfig.from_dict({"trend": {"min_total": 1}}),
    )
    assert rows and all(r.size is None for r in rows)
