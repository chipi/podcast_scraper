"""Regression: ``/api/corpus/digest`` warms the native search stack with ONE
serial band before fanning the rest out concurrently.

Prior to #1205 the endpoint ran up to eight ``_band`` invocations in a
naked ``ThreadPoolExecutor.map`` — each of which called
``run_corpus_search`` and, transitively, sentence-transformers / sentencepiece
/ LanceDB. Concurrent first-touch of those native layers SIGSEGVed the api
container mid-request (exit 139). See ``docs/wip/DIGEST-TOPICBAND-THREAD-UNSAFETY-ARM64.md``
for the original arm64 report and commits ``b6955854`` / ``8b5a1c07`` /
``62c049e5`` / ``0fe0854b`` for the LanceDB-side story.

The current shape is: run the first band **inline** (serial warmup so the
model + LanceDB are loaded once, single-threaded), THEN fan out bands
2..N via ``ThreadPoolExecutor.map``. This test catches an accidental
regression back to unconditional parallel-map by asserting that the FIRST
``run_corpus_search`` invocation completes before ANY second invocation
starts — i.e., the warmup is genuinely serial.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.search.corpus_search import CorpusSearchOutcome
from podcast_scraper.server.app import create_app

pytestmark = pytest.mark.integration


def _row(published: str, *, eid: str = "ep1", feed: str = "feed_a") -> dict:
    return {
        "feed": {"feed_id": feed, "title": "Show"},
        "episode": {
            "episode_id": eid,
            "title": "Hello",
            "published_date": published,
        },
        "summary": {"title": "Sum", "bullets": ["a", "b", "c", "d", "e"]},
    }


def test_first_topic_band_completes_before_any_second_band_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The FIRST ``_band`` call must finish before ANY subsequent
    ``_band`` begins. This is the warmup invariant that keeps
    sentence-transformers / sentencepiece / LanceDB native code out of
    the concurrent first-touch SIGSEGV shape.

    Rationale in the module docstring above."""
    today = datetime.now(timezone.utc).date().isoformat()
    meta = tmp_path / "metadata"
    meta.mkdir()
    (meta / "one.metadata.json").write_text(
        json.dumps(_row(f"{today}T12:00:00Z")),
        encoding="utf-8",
    )
    lance_idx = tmp_path / "search" / "lance_index"
    lance_idx.mkdir(parents=True)
    (lance_idx / "marker").write_text("x", encoding="utf-8")

    # 8 topic bands so a restored naked
    # ``ThreadPoolExecutor(max_workers=min(8, ...))`` would fan every one
    # out to a distinct worker thread from t=0 — making the warmup-shape
    # violation trivial to detect via arrival timestamps.
    monkeypatch.setattr(
        "podcast_scraper.server.routes.corpus_digest.load_digest_topics",
        lambda: [{"id": f"t{i}", "label": f"L{i}", "query": f"q{i}"} for i in range(8)],
    )

    # Every ``run_corpus_search`` invocation records enter+exit
    # timestamps. Band-level calls (queries q0..q7) are separated from
    # the pre-loop ``digest`` probe by the query string prefix.
    band_events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        del output_dir
        start = time.perf_counter()
        # Small delay so any concurrent second call would demonstrably
        # overlap. Larger than any single Python instruction, small
        # enough to keep the whole test snappy.
        time.sleep(0.05)
        end = time.perf_counter()
        if query.startswith("q"):
            with events_lock:
                band_events.append({"query": query, "start": start, "end": end})
        if query == "digest":
            return CorpusSearchOutcome(
                results=[{"score": 1.0, "metadata": {"episode_id": "ep1", "feed_id": "feed_a"}}],
            )
        return CorpusSearchOutcome(
            results=[
                {
                    "score": 0.9,
                    "text": "hit",
                    "metadata": {
                        "doc_type": "summary",
                        "episode_id": "ep1",
                        "feed_id": "feed_a",
                    },
                },
            ],
        )

    monkeypatch.setattr(
        "podcast_scraper.server.routes.corpus_digest.run_corpus_search",
        fake_run,
    )

    client = TestClient(create_app(tmp_path, static_dir=False))
    r = client.get(
        "/api/corpus/digest",
        params={"path": str(tmp_path), "window": "7d", "include_topics": "true"},
    )
    assert r.status_code == 200, r.text

    # Sanity: at least 2 band calls fired so the warmup shape is
    # observable. Some bands may short-circuit before ``run_corpus_search``
    # if their query mock returns empty results; the invariant we're
    # testing is about the FIRST vs SUBSEQUENT search timing.
    assert len(band_events) >= 2, band_events

    # THE regression assertion. Order by arrival (start time).
    band_events.sort(key=lambda e: e["start"])
    first = band_events[0]
    for later in band_events[1:]:
        assert later["start"] >= first["end"], (
            f"regression: /api/corpus/digest fanned band {later['query']!r} out "
            f"BEFORE the warmup band {first['query']!r} completed "
            f"(later.start={later['start']:.4f}, first.end={first['end']:.4f}). "
            f"This is the concurrent-first-touch shape that SIGSEGVs the api "
            f"container on arm64 and (per #1205) on x86_64 CI too. See "
            f"docs/wip/DIGEST-TOPICBAND-THREAD-UNSAFETY-ARM64.md and commits "
            f"b6955854 / 8b5a1c07 / 62c049e5 / 0fe0854b."
        )
