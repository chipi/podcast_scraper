"""The GI encoder cache must be thread-safe (#1179).

The GI stage runs across worker threads. `_get_encoder` cached into a plain dict with no lock, so
two threads could miss together, both construct a SentenceTransformer, and torch's lazy
meta-device init would race — raising "Cannot copy out of meta tensor; no data!".

It did not fail loudly. `_rank_about_edges_for_insights` threw, the GI artifact build for that
episode collapsed, and the episode landed with 1 insight instead of 10 while the run reported
success. One episode in three, every run, on the DGX pilot.

Every sibling loader (nli_loader, embedding_loader, extractive_qa) already guarded its cache this
way; this one was missed.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from podcast_scraper.gi import about_edges

pytestmark = pytest.mark.unit


def test_concurrent_get_encoder_constructs_the_model_once(monkeypatch) -> None:
    """N threads racing into a cold cache must build exactly ONE model.

    The construction is deliberately slow so every thread lands inside the window that used to be
    unguarded. Without the lock this constructs the model N times — which is what raced in torch.
    """
    monkeypatch.setattr(about_edges, "_encoder_cache", {})
    monkeypatch.setattr(about_edges, "_encoder_cache_lock", threading.Lock())

    constructions = []
    barrier = threading.Barrier(8)

    class _SlowEncoder:
        def __init__(self, model_id: str) -> None:
            constructions.append(model_id)
            # Widen the race window: torch's real init is slow, which is why this ever raced.
            threading.Event().wait(0.05)

    def _fake_import(model_id: str) -> _SlowEncoder:
        return _SlowEncoder(model_id)

    with patch.object(about_edges, "_get_encoder", wraps=about_edges._get_encoder):
        with patch.dict(
            "sys.modules",
            {"sentence_transformers": type("M", (), {"SentenceTransformer": _fake_import})},
        ):
            results = []

            def worker() -> None:
                barrier.wait()  # all threads start together
                results.append(about_edges._get_encoder("model-x"))

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

    assert len(constructions) == 1, (
        f"the model was constructed {len(constructions)} times — the cache is unguarded, and "
        "concurrent torch init is what raises 'Cannot copy out of meta tensor'"
    )
    assert len(results) == 8
    assert all(r is results[0] for r in results), "every thread must get the same instance"
