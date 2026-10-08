"""Themes, storylines, their search operators and the share-card extras come from an extension.

ADR-158: these are private features. The platform reads them through ``search.groupings`` and the
share-card contribution; with no extension installed every reader is empty, nothing is built, the
operators are not offered and the storyline card does not exist. With one, the platform passes
through exactly what the extension returns.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.extensions import (
    _from_module,
    Extension,
    ShareCardContribution,
    use_extensions,
)
from podcast_scraper.search import groupings
from podcast_scraper.search.corpus_search import (
    _attach_storyline_metadata,
    _attach_topic_cluster_metadata,
    CorpusSearchOutcome,
)
from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration

_CORPUS = Path("tests/fixtures/app-validation-corpus/v3")


def _intelligence() -> Extension:
    ext = _from_module("podcast_scraper.enrichment.intelligence_extension")
    assert ext is not None
    return ext


def test_without_an_extension_every_reader_is_empty() -> None:
    with use_extensions([]):
        assert not groupings.available()
        assert groupings.theme_map_by_topic(_CORPUS) == {}
        assert groupings.storyline_map_by_topic(_CORPUS) == {}
        assert groupings.top_themes_by_member_count(_CORPUS, 50) == []
        assert groupings.top_storylines_by_member_count(_CORPUS, 50, min_members=1) == []
        assert groupings.storyline_index_rows(_CORPUS) == []
        assert groupings.load_theme_payload(_CORPUS) is None
        assert groupings.build_topic_clusters_for_corpus(_CORPUS) is None
        assert groupings.search_operators() == {}


def test_the_intelligence_extension_serves_the_real_groupings() -> None:
    from podcast_scraper.search import storylines, topic_clusters

    with use_extensions([_intelligence()]):
        assert groupings.available()
        themes = groupings.theme_map_by_topic(_CORPUS)
        assert themes and themes == topic_clusters.theme_map_by_topic(_CORPUS)
        lines = groupings.top_storylines_by_member_count(_CORPUS, 50, min_members=1)
        assert lines and lines == storylines.top_storylines_by_member_count(
            _CORPUS, 50, min_members=1
        )
        assert set(groupings.search_operators()) == {"cluster", "consensus"}


def test_search_hits_lose_their_grouping_fields_without_an_extension() -> None:
    topic_id = next(iter(groupings.theme_map_by_topic(_CORPUS)))
    rows: list[dict[str, Any]] = [
        {"metadata": {"doc_type": "kg_topic", "source_id": topic_id}},
        {"metadata": {"doc_type": groupings.STORYLINE_DOC_TYPE, "source_id": "thc:anything"}},
    ]
    with use_extensions([]):
        _attach_topic_cluster_metadata(rows, _CORPUS)
        kept = _attach_storyline_metadata(rows, _CORPUS)
    assert "topic_cluster" not in rows[0]["metadata"]
    assert [r["metadata"]["doc_type"] for r in kept] == ["kg_topic"]

    with use_extensions([_intelligence()]):
        _attach_topic_cluster_metadata(rows, _CORPUS)
    assert rows[0]["metadata"]["topic_cluster"]


def _search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operator: str) -> dict[str, Any]:
    hits = [{"doc_id": "d:1", "score": 0.5, "metadata": {"doc_type": "insight"}, "text": "x"}]
    monkeypatch.setattr(
        "podcast_scraper.search.capability.run_corpus_search",
        lambda *a, **kw: CorpusSearchOutcome(
            results=hits, lift_stats={"transcript_hits_returned": 0, "lift_applied": 0}
        ),
    )
    client = TestClient(create_app(tmp_path, static_dir=False))
    params = {"q": "x", "path": str(tmp_path), "operator": operator}
    body: dict[str, Any] = client.get("/api/search", params=params).json()
    return body


def test_search_offers_only_the_operators_an_extension_provides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with use_extensions([]):
        body = _search(tmp_path, monkeypatch, "cluster")
    assert body["operator"] is None and body["clusters"] is None

    seen: list[int] = []

    def _cluster(hits: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
        seen.append(len(hits))
        return [
            {
                "cluster_id": "tc:k",
                "cluster_kind": "topic",
                "label": "L",
                "size": 1,
                "hit_indices": [0],
            }
        ]

    fake = groupings.TopicGroupings(search_operators=lambda: {"cluster": _cluster})
    with use_extensions([Extension(name="fake", groupings=fake)]):
        body = _search(tmp_path, monkeypatch, "cluster")
        assert _search(tmp_path, monkeypatch, "consensus")["operator"] is None
    assert seen == [1]
    assert body["operator"] == "cluster" and body["clusters"][0]["label"] == "L"


def test_the_themes_rebuild_route_does_not_exist_without_themes(tmp_path: Path) -> None:
    from podcast_scraper.server.routes.app_auth import require_viewer_access

    with use_extensions([]):
        app = create_app(tmp_path, static_dir=False)
        app.dependency_overrides[require_viewer_access] = lambda: object()
        r = TestClient(app).post(
            "/api/corpus/topic-clusters/rebuild", params={"path": str(tmp_path)}
        )
    assert r.status_code == 404


def test_the_pipeline_builds_themes_only_through_an_extension(tmp_path: Path) -> None:
    from podcast_scraper.workflow.orchestration import _maybe_build_topic_clusters_after_index

    lance = tmp_path / "search" / "lance_index"
    lance.mkdir(parents=True)
    (lance / "table.lance").write_text("x", encoding="utf-8")

    metrics: Any = SimpleNamespace()
    with use_extensions([]):
        _maybe_build_topic_clusters_after_index(str(tmp_path), metrics)
    assert metrics.topic_clusters_built is False

    calls: list[dict[str, Any]] = []

    def _build(output_dir: Any, **kw: Any) -> dict[str, Any]:
        calls.append(kw)
        return {"cluster_count": 3, "topic_count": 9, "singletons": 1, "schema_version": "2"}

    fake = groupings.TopicGroupings(build_topic_clusters_for_corpus=_build)
    metrics = SimpleNamespace()
    with use_extensions([Extension(name="fake", groupings=fake)]):
        _maybe_build_topic_clusters_after_index(str(tmp_path), metrics, threshold=0.6)
    assert calls == [{"index_dir": (tmp_path / "search").resolve(), "threshold": 0.6}]
    assert metrics.topic_clusters_built is True and metrics.topic_cluster_count == 3


def test_the_storyline_share_card_exists_only_with_storylines() -> None:
    from podcast_scraper.server.og.build import build_og_model

    with use_extensions([]):
        assert build_og_model(_CORPUS, "storyline", "topic:risk-management") is None
    with use_extensions([_intelligence()]):
        assert build_og_model(_CORPUS, "storyline", "topic:risk-management") is not None


def test_share_cards_take_their_trend_and_photo_from_an_extension(tmp_path: Path) -> None:
    from podcast_scraper.server.app_kg_index import build_kg_index
    from podcast_scraper.server.og.build import _person_photo, _trend

    topic_id = next(iter(build_kg_index(_CORPUS).topic_to_eps))
    with use_extensions([]):
        assert _trend(_CORPUS, "topic", topic_id) == (None, None, None)
        assert _person_photo(_CORPUS, "person:anyone") is None

    photo = tmp_path / "p.png"
    photo.write_bytes(b"\x89PNG-bytes")
    cards = ShareCardContribution(
        trends=lambda root, kind: (
            {topic_id: (2.0, (1.0, 2.0, 3.0, 4.0))} if kind == "topic" else {}
        ),
        person_image_path=lambda root, pid: (photo, "image/png") if pid == "person:a" else None,
    )
    with use_extensions([Extension(name="fake", share_cards=cards)]):
        assert _trend(_CORPUS, "topic", topic_id) == ("↑ 2.0× rising", (1.0, 2.0, 3.0, 4.0), 2.0)
        assert _person_photo(_CORPUS, "person:a") == b"\x89PNG-bytes"
        assert _person_photo(_CORPUS, "person:b") is None


def test_the_intelligence_extension_serves_trends_photos_and_logos(tmp_path: Path) -> None:
    from podcast_scraper.enrichment.enrichers.org_web import _logo_dir
    from podcast_scraper.enrichment.enrichers.person_web import _image_dir
    from podcast_scraper.server.og.build import _org_logo, _person_photo, _trend

    root = tmp_path / "corpus"
    _image_dir(root).mkdir(parents=True)
    (_image_dir(root) / "bob.jpg").write_bytes(b"photo-of-bob")
    _logo_dir(root).mkdir(parents=True)
    (_logo_dir(root) / "globex.png").write_bytes(b"logo-of-globex")
    with use_extensions([_intelligence()]):
        assert _person_photo(root, "person:bob") == b"photo-of-bob"
        assert _org_logo(root, "org:globex") == b"logo-of-globex"
        # The fixture's endurance-sport topic is rising (10x over the past year).
        hot, spark, mult = _trend(_CORPUS, "topic", "topic:endurance-sport")
    assert hot == "↑ 10.0× rising" and mult == 10.0 and spark
