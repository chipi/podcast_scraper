"""advisor #6: chunk workers run in a ThreadPoolExecutor; ContextVars don't propagate, so the
episode correlation id must be re-bound inside each worker (else chunk logs show episode_id=-)."""

from __future__ import annotations

import threading

import pytest

from podcast_scraper.providers.ml import summarizer
from podcast_scraper.utils import correlation


class _FakeModel:
    tokenizer = None  # skip the token-counting branch
    model_name = "fake"

    def __init__(self):
        self._summarize_lock = threading.Lock()
        self.seen = []

    def summarize(self, text, **kwargs):
        self.seen.append(correlation.get_episode_id())
        return "s:" + text


@pytest.mark.unit
def test_chunk_workers_inherit_episode_id():
    model = _FakeModel()
    correlation.set_episode_id("ep-parallel")
    try:
        summarizer._summarize_chunks_parallel(
            model, ["a", "b", "c", "d"], 100, 10, None, max_workers=3, start_time=0.0
        )
    finally:
        correlation.set_episode_id(None)
    assert model.seen  # workers ran
    assert all(s == "ep-parallel" for s in model.seen)  # each saw the id (None before the fix)
