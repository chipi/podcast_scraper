"""Test the chunk-8 catalog-row enrichments_available field."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from podcast_scraper.server.routes.corpus_library import (
    _resolve_episode_enrichments_available,
    episode_scope_enricher_ids,
)

pytestmark = pytest.mark.integration


def _seed_episode(corpus_root: Path, stem: str, *, enrichers: list[str]) -> str:
    meta_dir = corpus_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    md = meta_dir / f"{stem}.metadata.json"
    md.write_text(
        json.dumps(
            {
                "feed": {"feed_id": "feed1", "title": "Feed One"},
                "episode": {
                    "episode_id": f"episode:{stem}",
                    "title": f"Episode {stem}",
                    "published_date": "2026-06-01T00:00:00Z",
                },
            }
        ),
        encoding="utf-8",
    )
    if enrichers:
        out = meta_dir / "enrichments"
        out.mkdir(parents=True, exist_ok=True)
        for eid in enrichers:
            (out / f"{stem}.{eid}.json").write_text(
                json.dumps({"enricher_id": eid, "data": {}}), encoding="utf-8"
            )
    return f"metadata/{stem}.metadata.json"


def test_resolve_returns_all_false_when_no_enrichments(tmp_path: Path) -> None:
    rel = _seed_episode(tmp_path, "ep1", enrichers=[])
    flags = _resolve_episode_enrichments_available(tmp_path, rel)
    assert flags == {eid: False for eid in episode_scope_enricher_ids()}


def test_resolve_handles_empty_relpath(tmp_path: Path) -> None:
    flags = _resolve_episode_enrichments_available(tmp_path, "")
    assert all(v is False for v in flags.values())


def test_resolve_handles_non_metadata_relpath(tmp_path: Path) -> None:
    flags = _resolve_episode_enrichments_available(tmp_path, "feeds.spec.yaml")
    assert all(v is False for v in flags.values())
