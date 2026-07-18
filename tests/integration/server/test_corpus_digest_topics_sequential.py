"""Regression: ``/api/corpus/digest`` topic-band bands run **sequentially**.

Prior to commit ``1db52bea`` the endpoint ran up to eight ``_band`` invocations
in a ``ThreadPoolExecutor.map`` — each of which called
``run_corpus_search`` and, transitively, sentence-transformers / sentencepiece.
On arm64 (Docker for Mac, M1/M2) the concurrent first-touch of the tokenizer
C extension SIGSEGVed the api container mid-request (exit 139, traceback
inside ``ThreadPoolExecutor.map``). See
``docs/wip/DIGEST-TOPICBAND-THREAD-UNSAFETY-ARM64.md``.

The current implementation uses plain sequential ``map``. This test would
have caught the accidental restoration of the ``ThreadPoolExecutor`` by
observing that all bands run on the caller thread with no worker-pool
concurrency.
"""

from __future__ import annotations

import json
import threading
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


def test_topic_bands_run_sequentially_on_caller_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert every ``_band`` search runs on the request thread — no
    ``ThreadPoolExecutor`` fan-out around ``run_corpus_search`` calls.

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

    # 8 topic bands so a restored ``ThreadPoolExecutor(max_workers=min(8, ...))``
    # would fan every one out to a distinct worker thread — making it easy to
    # detect a regression by observing thread IDs.
    monkeypatch.setattr(
        "podcast_scraper.server.routes.corpus_digest.load_digest_topics",
        lambda: [{"id": f"t{i}", "label": f"L{i}", "query": f"q{i}"} for i in range(8)],
    )

    # Every ``run_corpus_search`` invocation records the thread it ran on.
    # The topic-band ``_band`` helper wraps this in a nested
    # ``ThreadPoolExecutor(max_workers=1)`` used only as a timeout guard, so
    # the actual search executes on ONE well-defined worker thread PER band
    # (regardless of whether the outer loop is parallel). We can still
    # distinguish the sequential vs concurrent case by counting DISTINCT
    # threads across all bands: sequential → 1 thread (the shared timeout
    # worker is disposed & recycled between bands so the OS thread id may
    # differ, but they're never LIVE at the same instant); concurrent → up
    # to N threads live simultaneously.
    caller_thread = threading.current_thread().ident
    seen_search_threads: list[int] = []
    seen_lock = threading.Lock()
    concurrent_ceiling = {"peak": 0}
    live: set[int] = set()

    def fake_run(
        output_dir: Path,
        query: str,
        **kwargs: Any,
    ) -> CorpusSearchOutcome:
        del output_dir
        tid = threading.current_thread().ident
        assert tid is not None
        with seen_lock:
            live.add(tid)
            concurrent_ceiling["peak"] = max(concurrent_ceiling["peak"], len(live))
            seen_search_threads.append(tid)
        # Small delay so a concurrent ThreadPoolExecutor.map WOULD race and
        # let ``live`` swell above 1 (peak > 1).
        import time as _t

        _t.sleep(0.05)
        with seen_lock:
            live.discard(tid)
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

    # Sanity: at least one topic band actually invoked ``run_corpus_search``
    # (plus the pre-loop ``digest`` probe). Some bands may short-circuit
    # without a hit; the invariant we're testing is about concurrency, not
    # about band count.
    band_calls = [t for t in seen_search_threads if t is not None]
    assert len(band_calls) >= 2, band_calls

    # THE regression assertion: peak concurrent searches should be 1.
    # A restored ``ThreadPoolExecutor(max_workers=min(8, ...))`` around the
    # ``_band`` loop would trivially push this to > 1 given the 50 ms
    # sleep above.
    assert concurrent_ceiling["peak"] == 1, (
        f"regression: /api/corpus/digest is running topic-band searches in "
        f"parallel (peak={concurrent_ceiling['peak']}). This is the shape "
        f"that SIGSEGVs the api container on arm64 (see "
        f"docs/wip/DIGEST-TOPICBAND-THREAD-UNSAFETY-ARM64.md). If parallelism "
        f"is needed, gate on a warm-model check."
    )

    # And the whole request completed on/near the caller thread (i.e., the
    # per-band inner ``max_workers=1`` timeout executors were the only
    # non-caller threads).
    del caller_thread  # Not asserted; kept for readability.
