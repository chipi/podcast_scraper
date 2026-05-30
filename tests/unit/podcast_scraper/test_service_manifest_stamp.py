"""Single-feed service manifest stamp (#807)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from podcast_scraper import config, service


@pytest.mark.unit
def test_service_run_stamps_corpus_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    cfg = config.Config.model_validate(
        {
            "rss_url": "https://example.com/feed.xml",
            "output_dir": str(corpus),
            "single_feed_uses_corpus_layout": True,
            "openai_api_key": "sk-test",
        }
    )

    with patch.object(service, "workflow") as mock_workflow:
        mock_workflow.run_pipeline.return_value = (3, "ok")
        result = service.run(cfg)

    assert result.success
    manifest_path = corpus / "corpus_manifest.json"
    assert manifest_path.is_file()
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert doc["produced_by"]["code_version"]
    assert doc["feeds"][0]["feed_url"] == "https://example.com/feed.xml"
    assert doc["feeds"][0]["episodes_processed"] == 3
