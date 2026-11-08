"""``/api/health`` reports the insight stage's routes and artifact kinds from what this server
actually has (E04): the operator viewer gates its CIL, top-voices and GI-coverage surfaces on them.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from podcast_scraper import builtin_read_hooks, builtin_stage_hooks
from podcast_scraper.extensions import use_extensions
from podcast_scraper.server.app import create_app
from podcast_scraper.stage_hooks import PipelineHooks

pytestmark = [pytest.mark.integration]

_STAGE_FLAGS = ("explore_api", "cil_queries_api", "corpus_persons_api")


def _health(tmp_path: Path) -> dict:
    client = TestClient(create_app(tmp_path, static_dir=False))
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body: dict = resp.json()
    return body


def test_with_the_insight_stage_its_routes_and_kinds_are_advertised(tmp_path: Path) -> None:
    body = _health(tmp_path)
    assert {flag: body[flag] for flag in _STAGE_FLAGS} == dict.fromkeys(_STAGE_FLAGS, True)
    assert body["artifact_kinds"] == ["kg", "gi", "bridge"]


def test_without_the_insight_stage_health_says_so(tmp_path: Path) -> None:
    with (
        patch.object(builtin_read_hooks, "ROUTERS", lambda: ()),
        patch.object(builtin_stage_hooks, "BUILTIN", PipelineHooks(name="b")),
        use_extensions([]),
    ):
        body = _health(tmp_path)
        client = TestClient(create_app(tmp_path, static_dir=False))
        topic_timeline = client.get("/api/topics/topic:ai/timeline", params={"path": str(tmp_path)})
    assert {flag: body[flag] for flag in _STAGE_FLAGS} == dict.fromkeys(_STAGE_FLAGS, False)
    assert body["artifact_kinds"] == ["kg"]
    assert topic_timeline.status_code == 404
