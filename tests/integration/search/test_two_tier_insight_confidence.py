"""#2521: the built ``insights`` table holds each insight's grounding confidence."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.search import two_tier_indexer as tti  # noqa: E402
from podcast_scraper.search.backends.lancedb_backend import LanceDBBackend  # noqa: E402


def test_insights_table_stores_grounding_confidence(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "metadata").mkdir(parents=True)
    meta_path = corpus / "metadata" / "ep1.metadata.json"
    rows = [
        (
            "insight:1",
            "grounded claim",
            {
                "doc_type": "insight",
                "episode_id": "ep1",
                "feed_id": "show1",
                "grounded": True,
                "grounding_confidence": 0.87,
            },
        ),
        (
            "insight:2",
            "ungraded claim",
            {"doc_type": "insight", "episode_id": "ep1", "feed_id": "show1"},
        ),
    ]
    monkeypatch.setattr(tti, "discover_metadata_files", lambda root: [meta_path])
    monkeypatch.setattr(tti, "_load_metadata_file", lambda p: {"episode": {"episode_id": "ep1"}})
    monkeypatch.setattr(tti, "episode_root_from_metadata_path", lambda p: corpus)
    monkeypatch.setattr(tti, "_collect_docs_for_episode", lambda *a, **k: rows)
    monkeypatch.setattr(tti, "_embed", lambda text, model_id, *, allow_download: [0.1] * 8)

    lance = corpus / "search" / "lance_index"
    tti.build_two_tier_index(corpus, lance, drop_existing=True)

    table = LanceDBBackend(str(lance)).db.open_table("insights")
    by_id = {r["id"]: r["confidence"] for r in table.to_arrow().to_pylist()}
    assert by_id == {"insight:1": 0.87, "insight:2": 0.0}
