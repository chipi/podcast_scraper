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


class TestSecretsStatus:
    """#1690: the endpoint that would have ended the 2026-08-18 outage in one request."""

    def _client_with_dir(self, monkeypatch, secrets_dir: Path) -> TestClient:
        monkeypatch.setenv("PODCAST_SECRETS_STATUS_DIR", str(secrets_dir))
        app = FastAPI()
        app.include_router(ops.router, prefix="/api")
        return TestClient(app)

    def test_missing_empty_unreadable_are_three_distinct_states(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        (tmp_path / "openai_api_key").write_text("sk-test-value-1234567890")
        (tmp_path / "anthropic_api_key").write_text("")  # staged but EMPTY
        unreadable = tmp_path / "gemini_api_key"
        unreadable.write_text("cannot-see-me")
        unreadable.chmod(0o000)
        try:
            body = (
                self._client_with_dir(monkeypatch, tmp_path).get("/api/ops/secrets/status").json()
            )
        finally:
            unreadable.chmod(0o600)
        by_name = {row["name"]: row for row in body["secrets"]}
        assert by_name["openai_api_key"] == {
            "name": "openai_api_key",
            "present": True,
            "readable": True,
            "bytes": 24,
            "sha256_prefix": by_name["openai_api_key"]["sha256_prefix"],
        }
        assert len(by_name["openai_api_key"]["sha256_prefix"]) == 12
        assert by_name["anthropic_api_key"]["present"] is True
        assert by_name["anthropic_api_key"]["bytes"] == 0
        assert by_name["gemini_api_key"]["present"] is True
        assert by_name["gemini_api_key"]["readable"] is False
        assert by_name["deepgram_api_key"]["present"] is False  # missing — never staged
        assert len(body["secrets"]) == 11

    def test_no_secret_value_ever_crosses_the_boundary(self, tmp_path: Path, monkeypatch) -> None:
        secret_value = "sk-live-EXTREMELY-SECRET-VALUE-42"
        (tmp_path / "litellm_api_key").write_text(secret_value)
        r = self._client_with_dir(monkeypatch, tmp_path).get("/api/ops/secrets/status")
        assert secret_value not in r.text
        assert "EXTREMELY" not in r.text


class TestGatewayAuth:
    """#1689: the auth probe tests the credential THIS process actually holds."""

    def _fresh_client(self, monkeypatch) -> TestClient:
        monkeypatch.setattr(ops, "_gateway_probe_last", [])
        app = FastAPI()
        app.include_router(ops.router, prefix="/api")
        return TestClient(app)

    def test_ok_and_401_both_reported_faithfully(self, monkeypatch) -> None:
        import httpx

        monkeypatch.setenv("LITELLM_API_BASE", "http://gw:4001/v1")
        monkeypatch.setenv("LITELLM_API_KEY", "sk-litellm-test-key")
        for status, ok in ((200, True), (401, False)):
            monkeypatch.setattr(
                httpx,
                "get",
                lambda *a, _s=status, **k: type("R", (), {"status_code": _s})(),
            )
            body = self._fresh_client(monkeypatch).get("/api/ops/gateway/auth").json()
            assert body["http_status"] == status and body["ok"] is ok
            assert "sk-litellm-test-key" not in json.dumps(body), "key crossed the boundary"

    def test_missing_key_is_the_2026_08_18_shape_not_a_500(self, monkeypatch) -> None:
        monkeypatch.setenv("LITELLM_API_BASE", "http://gw:4001/v1")
        monkeypatch.delenv("LITELLM_API_KEY", raising=False)
        body = self._fresh_client(monkeypatch).get("/api/ops/gateway/auth").json()
        assert body["key_present"] is False and body["ok"] is False

    def test_rate_limited_second_call(self, monkeypatch) -> None:
        import httpx

        monkeypatch.setenv("LITELLM_API_BASE", "http://gw:4001/v1")
        monkeypatch.setenv("LITELLM_API_KEY", "sk-litellm-test-key")
        monkeypatch.setattr(httpx, "get", lambda *a, **k: type("R", (), {"status_code": 200})())
        client = self._fresh_client(monkeypatch)
        assert client.get("/api/ops/gateway/auth").status_code == 200
        assert client.get("/api/ops/gateway/auth").status_code == 429
