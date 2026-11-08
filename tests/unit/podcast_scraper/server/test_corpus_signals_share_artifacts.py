"""Corpus signals read each enrichment envelope through the per-file artifact cache: the same data
as parsing each file, and one parsed copy shared with every other reader of that file."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from podcast_scraper import perf_cache
from podcast_scraper.server import corpus_signals as cs
from podcast_scraper.server.corpus_access import cached_json_artifact

pytestmark = [pytest.mark.unit]

CORPUS = Path(__file__).resolve().parents[4] / "tests/fixtures/viewer-validation-corpus/v3"
WANTED = {"temporal_velocity", "topic_cooccurrence_corpus", "guest_coappearance"}


@pytest.fixture(autouse=True)
def _empty_caches() -> Iterator[None]:
    perf_cache.clear()
    yield
    perf_cache.clear()


def _parsed_one_by_one(wanted: set[str]) -> dict:
    out = {}
    for path in sorted((CORPUS / "enrichments").glob("*.json")):
        if path.stem not in wanted or path.name in cs._SUMMARY_FILES:
            continue
        parsed = cs.parse_envelope(path)
        if parsed is not None and parsed.get("data") is not None:
            out[str(parsed.get("enricher_id") or path.stem)] = parsed["data"]
    return out


def test_the_signals_equal_parsing_each_file() -> None:
    got = cs.read_corpus_signals(CORPUS, WANTED)
    assert got and got == _parsed_one_by_one(WANTED)


def test_overlapping_requests_and_other_readers_share_one_copy() -> None:
    a = cs.corpus_signals(CORPUS, {"temporal_velocity", "guest_coappearance"})
    b = cs.corpus_signals(CORPUS, {"temporal_velocity"})
    artifact = cached_json_artifact(CORPUS, "enrichments/temporal_velocity.json")
    assert artifact is not None
    assert "temporal_velocity" in a
    assert a["temporal_velocity"] is b["temporal_velocity"] is artifact["data"]
