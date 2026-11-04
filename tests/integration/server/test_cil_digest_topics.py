"""Unit tests for CIL digest topic pills (bridge + topic clusters)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server.cil_digest_topics import (
    build_cil_digest_topics_for_row,
    CIL_TOPIC_PILL_CAP,
    load_topic_cluster_index,
)
from podcast_scraper.server.corpus_catalog import CatalogEpisodeRow

# Moved from tests/unit/ — RFC-081 PR-A1: tests that import [ml]/[llm] or viewer HTTP
# gated modules belong in the integration tier per UNIT_TESTING_GUIDE.md.
pytestmark = [pytest.mark.integration]


def _row(
    *,
    bridge_rel: str = "ep.bridge.json",
    has_bridge: bool = True,
    episode_id: str | None = "e1",
) -> CatalogEpisodeRow:
    return CatalogEpisodeRow(
        metadata_relative_path="ep.metadata.json",
        feed_id="f",
        feed_title=None,
        episode_id=episode_id,
        episode_title="T",
        publish_date="2024-06-01",
        summary_title=None,
        summary_bullets=(),
        summary_text=None,
        gi_relative_path="ep.gi.json",
        kg_relative_path="ep.kg.json",
        bridge_relative_path=bridge_rel,
        has_gi=True,
        has_kg=True,
        has_bridge=has_bridge,
    )


def test_load_topic_cluster_index_missing_file(tmp_path: Path) -> None:
    idx = load_topic_cluster_index(tmp_path)
    assert idx.cluster_for_topic("topic:any", "e1") == (False, None)


def test_cap_respected(tmp_path: Path) -> None:
    identities = [
        {"id": f"topic:t{i}", "display_name": f"T{i}"} for i in range(CIL_TOPIC_PILL_CAP + 3)
    ]
    (tmp_path / "ep.bridge.json").write_text(
        json.dumps({"identities": identities}),
        encoding="utf-8",
    )
    idx = load_topic_cluster_index(tmp_path)
    pills = build_cil_digest_topics_for_row(tmp_path, _row(), idx)
    assert len(pills) == CIL_TOPIC_PILL_CAP
