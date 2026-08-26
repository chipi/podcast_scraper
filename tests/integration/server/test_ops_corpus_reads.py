"""GET /api/ops/corpus/* — inspect-prod-corpus's measurements as side-effect-free reads (#1688).

The load-bearing property is READ-ONLY: the workflow's write_worklist=true wrote a file INTO
the corpus during a supposedly read-only audit and killed a backup mid-window (2026-08-18,
'tar: file changed as we read it'). Every test here snapshots the corpus file-set + mtimes
and asserts the request changed nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from podcast_scraper.server.routes import ops  # noqa: E402

# critical_path: operator-surface coverage must land on PRs (same rule as test_ops.py).
pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _client(corpus: Path) -> TestClient:
    app = FastAPI()
    app.state.output_dir = corpus
    app.include_router(ops.router, prefix="/api")
    return TestClient(app)


def _write_episode(
    corpus: Path,
    *,
    run: str,
    name: str,
    episode_id: str,
    insight_texts: list | None = None,
    declare_gi: bool = True,
    write_artifact: bool = True,
    preprocess_metrics: Dict[str, Any] | None = None,
) -> None:
    meta_dir = corpus / "feeds" / "feed_a" / run / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    gi_rel = f"metadata/{name}.gi.json"
    meta: Dict[str, Any] = {
        "episode": {"episode_id": episode_id, "title": name},
        "content": {"transcript_file_path": f"transcripts/{name}.txt"},
    }
    if declare_gi:
        meta["grounded_insights"] = {"artifact_path": gi_rel, "version": "1.0"}
    (meta_dir / f"{name}.metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    if write_artifact:
        nodes = [
            {"id": f"i{n}", "type": "Insight", "text": t} for n, t in enumerate(insight_texts or [])
        ]
        (meta_dir / f"{name}.gi.json").write_text(
            json.dumps({"nodes": nodes, "edges": []}), encoding="utf-8"
        )
    if preprocess_metrics is not None:
        run_dir = corpus / "feeds" / "feed_a" / run
        (run_dir / "run_summary.json").write_text(
            json.dumps({"preprocessing": preprocess_metrics}), encoding="utf-8"
        )


def _snapshot(corpus: Path) -> Dict[str, float]:
    return {str(p): p.stat().st_mtime for p in sorted(corpus.rglob("*")) if p.is_file()}


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    _write_episode(
        tmp_path,
        run="run_20260820-100000",
        name="0001 - Healthy",
        episode_id="ep-1",
        insight_texts=["a real insight"],
    )
    _write_episode(
        tmp_path,
        run="run_20260820-100000",
        name="0002 - NoBlock",
        episode_id="ep-2",
        declare_gi=False,
        write_artifact=False,
    )
    return tmp_path


class TestIntegrity:
    def test_counts_and_verdict(self, corpus: Path) -> None:
        before = _snapshot(corpus)
        r = _client(corpus).get("/api/ops/corpus/integrity")
        assert r.status_code == 200
        body = r.json()
        assert body["healthy_gi"] == 1
        assert body["no_gi_block"] == 1
        assert body["legacy_placeholders"] == 0
        assert body["verdict"] == "FAIL"  # the no-gi episode is a defect
        assert _snapshot(corpus) == before, "a READ endpoint changed the corpus"

    def test_clean_corpus_passes(self, tmp_path: Path) -> None:
        _write_episode(
            tmp_path,
            run="run_20260820-100000",
            name="0001 - Only",
            episode_id="ep-1",
            insight_texts=["one insight"],
        )
        body = _client(tmp_path).get("/api/ops/corpus/integrity").json()
        assert body["verdict"] == "PASS" and body["healthy_gi"] == 1


class TestPreprocessing:
    def test_worklist_is_json_not_a_file(self, corpus: Path) -> None:
        before = _snapshot(corpus)
        r = _client(corpus).get("/api/ops/corpus/preprocessing")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["unpreprocessed_episodes"], list)
        assert isinstance(body["damaged_runs"], list)
        assert body["verdict"] in ("PASS", "FAIL")
        after = _snapshot(corpus)
        assert after == before, (
            "the preprocessing read wrote into the corpus — the exact defect (worklist file "
            "during a backup window) this endpoint exists to remove"
        )


class TestUsage:
    def test_bytes_by_directory(self, corpus: Path) -> None:
        before = _snapshot(corpus)
        body = _client(corpus).get("/api/ops/corpus/usage").json()
        assert body["total_bytes"] > 0
        names = {d["path"] for d in body["by_directory"]}
        assert "feeds" in names
        assert sum(d["bytes"] for d in body["by_directory"]) == body["total_bytes"]
        assert _snapshot(corpus) == before


class TestNoCorpus:
    def test_400_when_server_has_no_corpus(self) -> None:
        app = FastAPI()
        app.state.output_dir = None
        app.include_router(ops.router, prefix="/api")
        r = TestClient(app).get("/api/ops/corpus/integrity")
        assert r.status_code == 400


class TestOperatorGuard:
    def test_403_without_operator_key(self, corpus: Path, monkeypatch) -> None:
        """/api/ops is an _OPERATOR_BASES member: the full app must refuse keyless reads."""
        from podcast_scraper.server.app import create_app

        monkeypatch.setenv("APP_OPERATOR_API_KEY", "test-operator-key")
        app = create_app(corpus, static_dir=False)
        client = TestClient(app)
        r = client.get("/api/ops/corpus/integrity")
        assert r.status_code in (401, 403), f"keyless operator read got {r.status_code}"
        r2 = client.get(
            "/api/ops/corpus/integrity", headers={"X-Operator-Key": "test-operator-key"}
        )
        assert r2.status_code == 200
