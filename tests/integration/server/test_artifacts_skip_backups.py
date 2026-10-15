"""``/api/artifacts`` never lists a backup or trash copy as a corpus artifact.

Prod 2026-10-07: 1,444 of the 2,587 artifacts it listed to the operator viewer's graph were
``.podcast_scraper/upgrade-backups/`` copies from BEFORE a repair (see utils/corpus_walk.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = [pytest.mark.integration]

LIVE = "feeds/f/run_1/metadata/ep"
BOOKKEEPING = (
    ".podcast_scraper/upgrade-backups/0020/feeds/f/run_1/metadata/ep",
    ".trash/20261001T000000Z/feeds/f/run_1/metadata/ep",
)


def test_the_artifact_listing_ignores_backups(tmp_path: Path) -> None:
    for stem in (LIVE, *BOOKKEEPING):
        for suffix in (".gi.json", ".kg.json", ".metadata.json"):
            path = tmp_path / f"{stem}{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    client = TestClient(create_app(tmp_path, static_dir=False))
    resp = client.get("/api/artifacts", params={"path": str(tmp_path)})
    assert resp.status_code == 200
    listed = {a["relative_path"] for a in resp.json()["artifacts"]}
    assert listed == {f"{LIVE}.gi.json", f"{LIVE}.kg.json"}


def test_only_the_latest_run_is_listed_and_only_it_costs_a_publish_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The latest-run filter runs BEFORE the per-file work (2026-10-15: a publish date for each of
    9,531 files, of which 1,191 were listed, was 2.7 s of an 8.5 s listing)."""
    for run in ("run_1", "run_2", "run_3"):
        for suffix in (".gi.json", ".kg.json", ".bridge.json"):
            path = tmp_path / f"feeds/f/{run}/metadata/ep{suffix}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    dated: list[str] = []

    def spy(base, rel, mtime):
        dated.append(rel)
        return "2026-10-15"

    monkeypatch.setattr(
        "podcast_scraper.server.routes.artifacts.publish_calendar_date_for_artifact_listing", spy
    )
    client = TestClient(create_app(tmp_path, static_dir=False))
    resp = client.get("/api/artifacts", params={"path": str(tmp_path)})
    assert resp.status_code == 200
    latest = {f"feeds/f/run_3/metadata/ep{s}" for s in (".bridge.json", ".gi.json", ".kg.json")}
    assert [a["relative_path"] for a in resp.json()["artifacts"]] == sorted(latest)
    assert sorted(dated) == sorted(latest)
