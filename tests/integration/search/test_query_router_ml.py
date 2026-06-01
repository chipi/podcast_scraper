"""Integration test: MLQueryRouter real joblib load + embed (RFC-092, #860).

Needs scikit-learn + joblib (ML extras), so it lives in integration — unit tests run
[dev]-only and would fail to import sklearn.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

from podcast_scraper.search.query_router import MLQueryRouter  # noqa: E402


def test_loads_persisted_model_and_embeds(tmp_path, monkeypatch):
    # Real joblib round-trip + real _embed (encode monkeypatched) → covers load + embed.
    import joblib
    from sklearn.dummy import DummyClassifier

    clf = DummyClassifier(strategy="constant", constant="cross_show_synthesis")
    clf.fit([[0.0, 0.0, 0.0]], ["cross_show_synthesis"])
    model_path = tmp_path / "router.joblib"
    joblib.dump(clf, model_path)

    monkeypatch.setattr(
        "podcast_scraper.providers.ml.embedding_loader.encode",
        lambda *a, **k: [0.0, 0.0, 0.0],
    )
    router = MLQueryRouter(model_path)
    assert router.classify("compare A and B") == "cross_show_synthesis"
