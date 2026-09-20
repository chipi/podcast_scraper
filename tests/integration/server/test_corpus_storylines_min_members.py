"""Themes are a NAVIGATION surface, so tiny ones are filtered at serve time (not in the artifact).

Measured on the 1,066-episode corpus: of 54 themes, 27 have exactly 2 members and a median of 3
episodes — a co-occurrence pair, not a destination. Filtering at >= 4 keeps 18 themes and still
reaches 192 of 286 episodes, so it drops 67% of themes for 33% of coverage.

Filtering here rather than in the enricher is deliberate: the artifact keeps every theme (other
consumers read it, and #1929's cluster-count diagnostics need the full set) and the threshold can
change without recomputing enrichment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration


def _theme(label: str, members: int, episodes: int = 2) -> dict:
    return {
        "cluster_type": "theme",
        "canonical_label": label,
        "graph_compound_parent_id": f"thc:{label.replace(' ', '-')}",
        "member_count": members,
        "members": [
            {
                "topic_id": f"topic:{label}-{i}",
                "label": f"{label} {i}",
                "episode_ids": [f"ep-{label}-{j}" for j in range(episodes)],
            }
            for i in range(members)
        ],
    }


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "metadata").mkdir()
    enr = tmp_path / "enrichments"
    enr.mkdir()
    payload = {
        "schema_version": "1",
        "clusters": [
            _theme("big", 9),
            _theme("mid", 4),
            _theme("pair-a", 2),
            _theme("pair-b", 2),
            _theme("triple", 3),
        ],
    }
    (enr / "topic_theme_clusters.json").write_text(json.dumps({"data": payload}), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def app(corpus: Path) -> FastAPI:
    return create_app(corpus, static_dir=False, enable_jobs_api=True)


def _labels(body: dict) -> set[str]:
    return {c["canonical_label"] for c in body.get("clusters") or []}


def test_default_surfaces_only_navigable_themes(app: FastAPI, corpus: Path) -> None:
    """The regression: 2- and 3-member themes must not reach the navigation surface."""
    body = TestClient(app).get("/api/corpus/theme-clusters", params={"path": str(corpus)}).json()
    assert _labels(body) == {"big", "mid"}
    assert body["withheld_below_min_members"] == 3
    assert body["min_members"] == 4


def test_zero_returns_the_unfiltered_artifact(app: FastAPI, corpus: Path) -> None:
    """Diagnostics need the full set — #1929 is about telling 'no themes' from 'not computed'."""
    body = (
        TestClient(app)
        .get("/api/corpus/theme-clusters", params={"path": str(corpus), "min_members": 0})
        .json()
    )
    assert len(body["clusters"]) == 5
    assert "withheld_below_min_members" not in body


def test_threshold_is_tunable_without_recomputing(app: FastAPI, corpus: Path) -> None:
    client = TestClient(app)
    at3 = client.get(
        "/api/corpus/theme-clusters", params={"path": str(corpus), "min_members": 3}
    ).json()
    assert _labels(at3) == {"big", "mid", "triple"}
    at9 = client.get(
        "/api/corpus/theme-clusters", params={"path": str(corpus), "min_members": 9}
    ).json()
    assert _labels(at9) == {"big"}


def test_filtering_does_not_poison_the_cache(app: FastAPI, corpus: Path) -> None:
    """``perf_cache`` returns the SAME object each hit, so filtering must never mutate it.

    Without a copy, the first filtered request would permanently shrink the cached payload and
    every later caller — including ``min_members=0`` — would see the filtered list.
    """
    client = TestClient(app)
    client.get("/api/corpus/theme-clusters", params={"path": str(corpus)})  # prime + filter
    full = client.get(
        "/api/corpus/theme-clusters", params={"path": str(corpus), "min_members": 0}
    ).json()
    assert len(full["clusters"]) == 5, "the cached artifact was mutated by an earlier filter"
    again = client.get("/api/corpus/theme-clusters", params={"path": str(corpus)}).json()
    assert _labels(again) == {"big", "mid"}


def test_no_filtering_metadata_when_nothing_is_withheld(app: FastAPI, corpus: Path) -> None:
    body = (
        TestClient(app)
        .get("/api/corpus/theme-clusters", params={"path": str(corpus), "min_members": 2})
        .json()
    )
    assert len(body["clusters"]) == 5
    assert "withheld_below_min_members" not in body
