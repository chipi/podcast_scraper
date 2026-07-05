"""Unit tests for ranking-config persistence (``app_ranking_config_store``)."""

from __future__ import annotations

from pathlib import Path

from podcast_scraper.server import app_ranking_config_store as store
from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    ranking_config_from_dict,
    SIGNAL_TREND_VELOCITY,
)


def test_load_absent_returns_default(tmp_path: Path) -> None:
    assert store.load_ranking_config(tmp_path) is DEFAULT_RANKING_CONFIG


def test_save_then_load_roundtrips(tmp_path: Path) -> None:
    cfg = ranking_config_from_dict(
        {"signals": [{"name": "trend_velocity", "enabled": True, "weight": 0.7}]}
    )
    store.save_ranking_config(tmp_path, cfg)
    loaded = store.load_ranking_config(tmp_path)
    assert loaded.is_enabled(SIGNAL_TREND_VELOCITY) is True
    assert loaded.weight_of(SIGNAL_TREND_VELOCITY) == 0.7
    assert (tmp_path / "ranking_config.json").is_file()


def test_load_malformed_returns_default(tmp_path: Path) -> None:
    (tmp_path / "ranking_config.json").write_text("{ not json", encoding="utf-8")
    assert store.load_ranking_config(tmp_path) is DEFAULT_RANKING_CONFIG
