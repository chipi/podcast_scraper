"""R3b's writer switch on a real run: a new episode's KG file is KG 3.0 and valid (0051's record).

0051 rewrites the corpus's KG files; the writer switch makes every file a run writes afterwards KG
3.0 too. This runs a podcast feed through ``run_source`` (the real HTTP client against the local
test server, metadata and KG on, no model) and reads every KG file the run wrote: each is KG 3.0,
validates strictly against kg.v3.schema.json, names its item at the root, and carries none of KG
2.x's names (``Episode``/``Podcast`` nodes, ``HOSTS``/``GUESTS_ON``/``HAS_EPISODE`` edges).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper import content_types as ct
from podcast_scraper.kg.schema import validate_artifact
from tests.conftest import create_test_config

pytestmark = [pytest.mark.e2e]

KG_2_NODE_TYPES = {"Episode", "Podcast"}
KG_2_EDGE_TYPES = {"HOSTS", "GUESTS_ON", "HAS_EPISODE"}


def test_a_new_runs_kg_files_are_kg_3_0_and_valid(e2e_server, tmp_path: Path) -> None:
    out = tmp_path / "run"
    cfg = create_test_config(
        rss_url=e2e_server.urls.feed("podcast1_multi_episode"),
        output_dir=str(out),
        max_episodes=2,
        generate_metadata=True,
        metadata_format="json",
        generate_summaries=False,
        generate_kg=True,
        kg_extraction_source="metadata_only",
        auto_speakers=False,
        transcribe_missing=False,
    )
    count, _ = ct.run_source(cfg, None)
    assert count > 0
    written = sorted(out.rglob("*.kg.json"))
    assert len(written) == count, f"{count} episodes, {len(written)} KG files"
    for path in written:
        kg = json.loads(path.read_text(encoding="utf-8"))
        assert kg["schema_version"] == "3.0", path.name
        validate_artifact(kg, strict=True)
        assert kg.get("item_id"), f"{path.name}: the root names no item"
        types = {n.get("type") for n in kg.get("nodes", [])}
        edges = {e.get("type") for e in kg.get("edges", [])}
        assert (
            not types & KG_2_NODE_TYPES
        ), f"{path.name}: KG 2.x node types {types & KG_2_NODE_TYPES}"
        assert (
            not edges & KG_2_EDGE_TYPES
        ), f"{path.name}: KG 2.x edge types {edges & KG_2_EDGE_TYPES}"
