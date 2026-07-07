"""Unit tests for the ``insight_sentiment`` enricher (VADER, deterministic)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from podcast_scraper.enrichment.enrichers.insight_sentiment import InsightSentimentEnricher
from podcast_scraper.enrichment.protocol import EpisodeArtifactBundle, RunContext, STATUS_OK


def _bundle(meta_dir: Path, gi: dict[str, Any]) -> EpisodeArtifactBundle:
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "e1.metadata.json").write_text("{}", encoding="utf-8")
    gi_path = meta_dir / "e1.gi.json"
    gi_path.write_text(json.dumps(gi), encoding="utf-8")
    return EpisodeArtifactBundle(
        metadata_path=meta_dir / "e1.metadata.json",
        gi_path=gi_path,
        kg_path=None,
        bridge_path=None,
        episode_id="episode:e1",
        stem="e1",
    )


def _ctx() -> RunContext:
    return RunContext(
        run_id="r1",
        parent_run_id=None,
        enricher_id="insight_sentiment",
        enricher_version="1.0.0",
        tier="deterministic",
        attempt=1,
        job_id="r1",
        cancel_event=asyncio.Event(),
    )


def _gi(texts: dict[str, str]) -> dict[str, Any]:
    return {
        "nodes": [
            {"type": "Insight", "id": iid, "properties": {"text": t}} for iid, t in texts.items()
        ],
        "edges": [],
    }


def _run(bundle: EpisodeArtifactBundle) -> dict[str, Any]:
    result = asyncio.run(
        InsightSentimentEnricher().enrich(
            bundle=bundle, corpus_root=Path("/tmp"), all_bundles=None, config={}, ctx=_ctx()
        )
    )
    assert result.status == STATUS_OK and isinstance(result.data, dict)
    return result.data


def test_labels_pos_neg_neutral(tmp_path: Path) -> None:
    gi = _gi(
        {
            "insight:pos": "This breakthrough is an exciting and wonderful leap forward.",
            "insight:neg": "This is a terrible, disastrous failure with serious harm.",
            "insight:neu": "The meeting is scheduled for the third quarter.",
        }
    )
    data = _run(_bundle(tmp_path / "e1", gi))
    by_id = {r["insight_id"]: r for r in data["insights"]}
    assert by_id["insight:pos"]["label"] == "positive" and by_id["insight:pos"]["compound"] > 0
    assert by_id["insight:neg"]["label"] == "negative" and by_id["insight:neg"]["compound"] < 0
    assert by_id["insight:neu"]["label"] == "neutral"
    assert data["counts"] == {"negative": 1, "neutral": 1, "positive": 1}
    assert data["total_insights"] == 3


def test_deterministic_same_text_same_score(tmp_path: Path) -> None:
    gi = _gi({"insight:a": "An exciting, wonderful breakthrough."})
    a = _run(_bundle(tmp_path / "a", gi))["insights"][0]["compound"]
    b = _run(_bundle(tmp_path / "b", gi))["insights"][0]["compound"]
    assert a == b  # lexicon is fixed → deterministic


def test_empty_or_missing_text_skipped(tmp_path: Path) -> None:
    gi = _gi({"insight:blank": "   ", "insight:ok": "A wonderful, delightful result."})
    data = _run(_bundle(tmp_path / "e1", gi))
    assert data["total_insights"] == 1
    assert data["insights"][0]["insight_id"] == "insight:ok"


def test_manifest_deterministic_no_gate() -> None:
    m = InsightSentimentEnricher.manifest
    assert m.id == "insight_sentiment"
    assert m.tier.value == "deterministic" and m.scope.value == "episode"
    assert m.accuracy_gate is None  # decoration → no gate → always admitted
    assert m.provider_requirement is None
