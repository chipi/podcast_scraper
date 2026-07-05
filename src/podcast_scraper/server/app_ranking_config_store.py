"""Persistence for the active ranking-signal config (#11 / B2) — one JSON file per instance.

The operator edits the ranking registry (which signals are on, their weights + params) through the
admin ranking-config endpoint; it persists here and the discovery feed loads it per request. An
absent or malformed file falls back to :data:`DEFAULT_RANKING_CONFIG`, so ranking always has a
valid config — the same total-parsing guarantee as ``ranking_config_from_dict``.
"""

from __future__ import annotations

import json
from pathlib import Path

from podcast_scraper.server.app_ranking_config import (
    DEFAULT_RANKING_CONFIG,
    ranking_config_from_dict,
    ranking_config_to_dict,
    RankingConfig,
)
from podcast_scraper.server.atomic_write import atomic_write_text


def _config_path(data_dir: Path) -> Path:
    return data_dir / "ranking_config.json"


def load_ranking_config(data_dir: Path) -> RankingConfig:
    """The persisted ranking config, or :data:`DEFAULT_RANKING_CONFIG` when absent/unreadable."""
    path = _config_path(data_dir)
    if not path.is_file():
        return DEFAULT_RANKING_CONFIG
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_RANKING_CONFIG
    return ranking_config_from_dict(data)


def save_ranking_config(data_dir: Path, config: RankingConfig) -> RankingConfig:
    """Persist *config* atomically; returns it for convenience."""
    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        _config_path(data_dir),
        json.dumps(ranking_config_to_dict(config), indent=2, ensure_ascii=False),
    )
    return config


__all__ = ["load_ranking_config", "save_ranking_config"]
