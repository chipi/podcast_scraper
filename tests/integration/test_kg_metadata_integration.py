"""Integration: metadata generation writes kg.json when kg_extraction_source=provider (mock)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# Match unit metadata tests: avoid importing heavy NLP before spacy is mocked.
if "spacy" not in sys.modules:
    sys.modules["spacy"] = MagicMock()

from podcast_scraper.utils import filesystem
from podcast_scraper.workflow import metadata_generation as mg, metrics as metrics_mod
from tests.conftest import create_test_config, create_test_episode, create_test_feed, TEST_FEED_URL


@pytest.mark.integration
def test_generate_episode_metadata_kg_provider_calls_extract_kg_graph(
    tmp_path,
    monkeypatch,
) -> None:
    """Provider KG path: extract_kg_graph on summary_provider; kg.json on disk; metrics updated."""
    out = tmp_path / "run"
    out.mkdir(parents=True)
    (out / "transcripts").mkdir()
    ep = create_test_episode(idx=1, title="Test Episode", title_safe="Episode_Title")
    tname = filesystem.build_whisper_output_name(1, "Episode_Title", None)
    trans_path = out / "transcripts" / tname
    trans_path.write_text("Discussing ACME Corp and markets.", encoding="utf-8")

    cfg = create_test_config(
        generate_metadata=True,
        generate_kg=True,
        generate_summaries=False,
        kg_extraction_source="provider",
    )
    prov = MagicMock()
    prov.summary_model = "mock-kg-model"
    prov.extract_kg_graph.return_value = {
        "topics": [{"label": "Markets"}],
        "entities": [{"name": "ACME Corp", "entity_kind": "organization"}],
    }
    pm = metrics_mod.Metrics()

    def _noop_serialize(_doc, _path, _cfg, pipeline_metrics=None):
        return None

    monkeypatch.setattr(mg, "_serialize_metadata", _noop_serialize)

    result = mg.generate_episode_metadata(
        feed=create_test_feed(),
        episode=ep,
        feed_url=TEST_FEED_URL,
        cfg=cfg,
        output_dir=str(out),
        transcript_file_path=f"transcripts/{tname}",
        summary_provider=prov,
        pipeline_metrics=pm,
    )

    assert result is not None
    kg_paths = list((out / "metadata").rglob("*.kg.json"))
    assert len(kg_paths) == 1, f"expected one kg.json, got {kg_paths}"
    prov.extract_kg_graph.assert_called_once()
    assert pm.kg_artifacts_generated == 1
    assert pm.kg_extractions_provider == 1
    assert pm.kg_topic_nodes_total >= 1
    assert pm.kg_entity_nodes_total >= 1
