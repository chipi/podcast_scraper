"""Unit coverage for hybrid_search pure helpers (RFC-090 Phase 2)."""

from __future__ import annotations

import pytest

from podcast_scraper.search import hybrid_search as hs
from podcast_scraper.search.backend import CompoundResult, ScoredResult

pytestmark = pytest.mark.unit


def test_lance_index_dir():
    assert str(hs.lance_index_dir("/c")).endswith("/c/search/lance_index")


def test_load_search_config_present_and_invalid(tmp_path, monkeypatch):
    cfg = tmp_path / "search.yaml"
    monkeypatch.setenv("PODCAST_SEARCH_CONFIG", str(cfg))
    cfg.write_text("serving:\n  hybrid_enabled: true\nrouter:\n  mode: rules\n", encoding="utf-8")
    doc = hs._load_search_config()
    assert doc["serving"]["hybrid_enabled"] is True and doc["router"]["mode"] == "rules"

    cfg.write_text(": : not yaml : :", encoding="utf-8")
    assert hs._load_search_config() == {}  # parse error → {}

    cfg.write_text("- a list not a dict", encoding="utf-8")
    assert hs._load_search_config() == {}  # non-dict → {}


def test_to_search_result_segment_carries_timestamps_and_speaker():
    seg = ScoredResult(
        "chunk:1",
        0.7,
        1,
        {
            "text": "t",
            "episode_id": "e1",
            "show_id": "A",
            "start_time": 2.0,
            "end_time": 5.0,
            "speaker_id": "spk",
        },
        "bm25",
        "segment",
    )
    out = hs._to_search_result(seg)
    assert out.metadata["doc_type"] == "transcript"
    assert out.metadata["timestamp_start_ms"] == 2000 and out.metadata["timestamp_end_ms"] == 5000
    assert out.metadata["speaker_id"] == "spk"
    assert out.metadata["feed_id"] == "A"


def test_flatten_handles_all_result_types():
    seg = ScoredResult("s1", 0.8, 1, {}, "bm25", "segment")
    ins = ScoredResult("i1", 0.9, 1, {}, "vector", "insight")
    comp = CompoundResult(doc_id="s1", score=0.9, rank=1, segment=seg, insight=ins)
    assert hs._flatten(comp) == [ins, seg]  # compound → both
    assert hs._flatten(ins) == [ins]  # scored → itself
    assert hs._flatten("not a result") == []  # unknown → empty


def test_tier_for_aux_doc_types():
    assert hs._tier_for(["kg_topic"]) == "aux"
    assert hs._tier_for(["kg_entity", "quote"]) == "aux"
    assert hs._tier_for(["insight", "kg_topic"]) == "all"  # mixed → all


def test_to_search_result_aux_uses_payload_doc_type():
    aux = ScoredResult(
        "kg_topic:1",
        0.5,
        1,
        {"text": "x", "episode_id": "e1", "show_id": "A", "doc_type": "kg_topic"},
        "bm25",
        "aux",
    )
    out = hs._to_search_result(aux)
    assert out.metadata["doc_type"] == "kg_topic"  # aux carries its real doc_type
