"""Operator corpus rollback API — DELETE runs/episodes to trash + reindex (incremental-add P0.2).

Covers the safe-rollback contract without a real LanceDB build: the index rebuild is delegated to
the shared _spawn_rebuild_thread machinery (tested in test_index_rebuild / the fold test), so here
we mock it and assert it is kicked with rebuild=True. Everything else — trash-move recoverability,
dry_run, confirm token, 404/409, the operator guard, episode-scoped delete — is asserted for real.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from podcast_scraper.server.app import create_app
from podcast_scraper.server.index_rebuild import gate_for_corpus
from podcast_scraper.server.routes import corpus_rollback

pytestmark = [pytest.mark.integration, pytest.mark.critical_path]


def _seed_run(corpus: Path, run_id: str, idx: int, guid: str, episode_id: str) -> None:
    run = corpus / "feeds" / "rss_show" / f"run_{run_id}"
    (run / "metadata").mkdir(parents=True, exist_ok=True)
    (run / "transcripts").mkdir(parents=True, exist_ok=True)
    name = f"{idx:04d} - Ep {idx}"
    (run / "transcripts" / f"{name}.txt").write_text("hello", encoding="utf-8")
    (run / "metadata" / f"{name}.metadata.json").write_text(
        json.dumps({"episode": {"guid": guid, "episode_id": episode_id}}), encoding="utf-8"
    )


@pytest.fixture
def client_and_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    corpus = tmp_path / "corpus"
    _seed_run(corpus, "R1", 1, "g1", "ep-1")
    spawned: Dict[str, Any] = {}

    def _fake_spawn(**kwargs: Any) -> None:
        spawned.update(kwargs)
        kwargs["gate"].end(None)  # release the rebuild gate as the real thread would

    monkeypatch.setattr(corpus_rollback, "_spawn_rebuild_thread", _fake_spawn)
    app = create_app(corpus, static_dir=False)
    # A destructive DELETE fails closed without a configured credential — set one + present it so
    # the happy-path tests exercise the real gate (the no-credential 403 is asserted separately).
    app.state.operator_api_key = "test-op-key"
    client = TestClient(app)
    client.headers["X-Operator-Key"] = "test-op-key"
    return client, corpus, spawned, app


def _wait(cond, timeout: float = 2.0) -> None:
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.02)


def test_delete_run_moves_to_trash_and_triggers_rebuild(client_and_corpus):
    client, corpus, spawned, _app = client_and_corpus
    run_dir = corpus / "feeds" / "rss_show" / "run_R1"
    assert run_dir.is_dir()

    r = client.request(
        "DELETE", "/api/corpus/runs/R1", params={"path": str(corpus), "confirm": "R1"}
    )
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["episodes_dropped"] == 1
    assert body["rebuild"]["status"] == "in_progress"

    # Live dir gone; contents recoverable under .trash/ (not hard-rm'd).
    assert not run_dir.exists()
    trashed = list((corpus / ".trash").rglob("run_R1/metadata/0001 - Ep 1.metadata.json"))
    assert trashed, "run must be recoverable from .trash/"

    # Full rebuild kicked (sweeps orphaned vectors — the two-tier index has no orphan sweep).
    _wait(lambda: "rebuild" in spawned)
    assert spawned["rebuild"] is True


def test_dry_run_changes_nothing(client_and_corpus):
    client, corpus, _spawned, _app = client_and_corpus
    r = client.request(
        "DELETE", "/api/corpus/runs/R1", params={"path": str(corpus), "dry_run": "true"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["dry_run"] is True
    assert any("run_R1" in p for p in body["would_remove"])
    assert (corpus / "feeds" / "rss_show" / "run_R1").is_dir()  # untouched


def test_confirm_token_required(client_and_corpus):
    client, corpus, _spawned, _app = client_and_corpus
    r = client.request(
        "DELETE", "/api/corpus/runs/R1", params={"path": str(corpus), "confirm": "wrong"}
    )
    assert r.status_code == 400
    assert (corpus / "feeds" / "rss_show" / "run_R1").is_dir()  # nothing removed


def test_unknown_run_id_404(client_and_corpus):
    client, corpus, _spawned, _app = client_and_corpus
    r = client.request(
        "DELETE", "/api/corpus/runs/NOPE", params={"path": str(corpus), "confirm": "NOPE"}
    )
    assert r.status_code == 404


def test_concurrent_rebuild_409_and_no_delete(client_and_corpus):
    client, corpus, _spawned, app = client_and_corpus
    # Simulate an index rebuild already running for this corpus.
    gate_for_corpus(app, corpus.resolve()).try_begin()
    r = client.request(
        "DELETE", "/api/corpus/runs/R1", params={"path": str(corpus), "confirm": "R1"}
    )
    assert r.status_code == 409
    assert (corpus / "feeds" / "rss_show" / "run_R1").is_dir()  # not moved — 409 before any change


def test_episode_scoped_delete_removes_only_that_episode(tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    _seed_run(corpus, "append_h", 1, "g1", "ep-1")
    _seed_run(corpus, "append_h", 2, "g2", "ep-2")  # same run dir, two appended episodes
    monkeypatch.setattr(corpus_rollback, "_spawn_rebuild_thread", lambda **k: k["gate"].end(None))
    app = create_app(corpus, static_dir=False)
    app.state.operator_api_key = "k"
    client = TestClient(app)
    client.headers["X-Operator-Key"] = "k"

    r = client.request(
        "DELETE", "/api/corpus/episodes/ep-1", params={"path": str(corpus), "confirm": "ep-1"}
    )
    assert r.status_code == 202, r.text
    run = corpus / "feeds" / "rss_show" / "run_append_h"
    assert not (run / "transcripts" / "0001 - Ep 1.txt").exists()  # ep-1 gone
    assert (run / "transcripts" / "0002 - Ep 2.txt").exists()  # ep-2 kept


def test_operator_guard_blocks_without_key(tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus"
    _seed_run(corpus, "R1", 1, "g1", "ep-1")
    monkeypatch.setattr(corpus_rollback, "_spawn_rebuild_thread", lambda **k: k["gate"].end(None))
    app = create_app(corpus, static_dir=False)
    app.state.operator_api_key = "secret-key"  # enable enforcement
    client = TestClient(app)

    denied = client.request(
        "DELETE", "/api/corpus/runs/R1", params={"path": str(corpus), "confirm": "R1"}
    )
    assert denied.status_code == 403
    assert (corpus / "feeds" / "rss_show" / "run_R1").is_dir()  # guard blocked before the handler

    ok = client.request(
        "DELETE",
        "/api/corpus/runs/R1",
        params={"path": str(corpus), "confirm": "R1"},
        headers={"X-Operator-Key": "secret-key"},
    )
    assert ok.status_code == 202


def test_delete_run_reaggregates_manifest_cost_downward(tmp_path: Path, monkeypatch):
    """Acceptance: after deleting a run, corpus_manifest.json cost_rollup recomputes from the
    REMAINING run metrics (drops by the deleted run's cost)."""
    from podcast_scraper.workflow import corpus_operations as cops

    corpus = tmp_path / "corpus"
    _seed_run(corpus, "R1", 1, "g1", "ep-1")
    _seed_run(corpus, "R2", 2, "g2", "ep-2")
    show = corpus / "feeds" / "rss_show"
    (show / "run_R1" / "metrics.json").write_text('{"llm_gi_cost_usd": 2.0}', encoding="utf-8")
    (show / "run_R2" / "metrics.json").write_text('{"llm_gi_cost_usd": 3.0}', encoding="utf-8")
    cops.write_corpus_manifest(str(corpus), [])  # seed manifest (cost_rollup = 5.0 from disk)

    monkeypatch.setattr(corpus_rollback, "_spawn_rebuild_thread", lambda **k: k["gate"].end(None))
    app = create_app(corpus, static_dir=False)
    app.state.operator_api_key = "k"
    client = TestClient(app)
    client.headers["X-Operator-Key"] = "k"

    r = client.request(
        "DELETE", "/api/corpus/runs/R1", params={"path": str(corpus), "confirm": "R1"}
    )
    assert r.status_code == 202, r.text
    assert r.json()["cost_rollup_total_usd"] == 3.0  # 5.0 - deleted R1's 2.0
    manifest = json.loads((corpus / cops.CORPUS_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["cost_rollup"]["total_cost_usd"] == 3.0


@pytest.mark.parametrize("bad", ["../../etc", "..%2f..", "a/b", "..", "with space", ""])
def test_delete_run_rejects_unsafe_run_id(client_and_corpus, bad):
    """Path-traversal / malformed run_id → 400, nothing moved (defense-in-depth on a DELETE)."""
    client, corpus, _spawned, _app = client_and_corpus
    r = client.request(
        "DELETE", f"/api/corpus/runs/{bad}", params={"path": str(corpus), "confirm": bad}
    )
    assert r.status_code in (400, 404)  # rejected or simply not found — never a traversal delete
    assert (corpus / "feeds" / "rss_show" / "run_R1").is_dir()  # untouched


def test_destructive_delete_fails_closed_without_credential(tmp_path: Path, monkeypatch):
    """Fable-5 #1: a DELETE with NO operator credential configured is refused (fail-closed), so a
    bare/misconfigured deploy can't be corpus-wiped by anyone who merely reaches the tailnet."""
    corpus = tmp_path / "corpus"
    _seed_run(corpus, "R1", 1, "g1", "ep-1")
    monkeypatch.setattr(corpus_rollback, "_spawn_rebuild_thread", lambda **k: k["gate"].end(None))
    app = create_app(corpus, static_dir=False)  # no operator_api_key, no session secret
    client = TestClient(app)
    r = client.request(
        "DELETE", "/api/corpus/runs/R1", params={"path": str(corpus), "confirm": "R1"}
    )
    assert r.status_code == 403
    assert (corpus / "feeds" / "rss_show" / "run_R1").is_dir()  # untouched


def test_episode_scoped_delete_also_sweeps_media(tmp_path: Path, monkeypatch):
    """Fable-5 #4: episode delete must also remove the episode's audio in media/, not orphan it."""
    corpus = tmp_path / "corpus"
    _seed_run(corpus, "append_h", 1, "g1", "ep-1")
    media = corpus / "feeds" / "rss_show" / "run_append_h" / "media"
    media.mkdir(parents=True, exist_ok=True)
    (media / "0001 - Ep 1.mp3").write_text("audio", encoding="utf-8")
    monkeypatch.setattr(corpus_rollback, "_spawn_rebuild_thread", lambda **k: k["gate"].end(None))
    app = create_app(corpus, static_dir=False)
    app.state.operator_api_key = "k"
    client = TestClient(app)
    client.headers["X-Operator-Key"] = "k"

    r = client.request(
        "DELETE", "/api/corpus/episodes/ep-1", params={"path": str(corpus), "confirm": "ep-1"}
    )
    assert r.status_code == 202, r.text
    assert not (media / "0001 - Ep 1.mp3").exists()  # audio swept, not orphaned
    trashed = list((corpus / ".trash").rglob("media/0001 - Ep 1.mp3"))
    assert trashed  # recoverable


def test_unknown_episode_id_404(client_and_corpus):
    client, corpus, _spawned, _app = client_and_corpus
    r = client.request(
        "DELETE", "/api/corpus/episodes/nope-9", params={"path": str(corpus), "confirm": "nope-9"}
    )
    assert r.status_code == 404


def test_flat_layout_run_dir_is_found(tmp_path: Path):
    """A run dir directly under the corpus root (non-feeds layout) is also located."""
    from podcast_scraper.server.routes.corpus_rollback import _run_dirs_for_id

    corpus = tmp_path / "corpus"
    (corpus / "run_FLAT" / "metadata").mkdir(parents=True)
    found = _run_dirs_for_id(corpus, "FLAT")
    assert [p.name for p in found] == ["run_FLAT"]


def test_assert_under_root_rejects_outside_path(tmp_path: Path):
    from fastapi import HTTPException

    from podcast_scraper.server.routes.corpus_rollback import _assert_under_root

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(HTTPException) as ei:
        _assert_under_root(corpus, [tmp_path / "elsewhere" / "x"])
    assert ei.value.status_code == 400


def test_move_to_trash_rejects_symlink_escape(tmp_path: Path):
    from fastapi import HTTPException

    from podcast_scraper.server.routes.corpus_rollback import _move_to_trash

    corpus = tmp_path / "corpus"
    (corpus / "feeds").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    escape = corpus / "feeds" / "link"
    escape.symlink_to(outside)  # a target that realpath-resolves outside the corpus
    with pytest.raises(HTTPException) as ei:
        _move_to_trash(corpus, [escape], "ts")
    assert ei.value.status_code == 400


def test_reaggregate_manifest_none_and_bad(tmp_path: Path):
    from podcast_scraper.server.routes.corpus_rollback import _reaggregate_manifest
    from podcast_scraper.workflow import corpus_operations as cops

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assert _reaggregate_manifest(corpus) is None  # no manifest → None
    (corpus / cops.CORPUS_MANIFEST_FILE).write_text("{ not json", encoding="utf-8")
    assert _reaggregate_manifest(corpus) is None  # unreadable manifest → best-effort None


def test_rollback_aborts_and_releases_gate_on_move_failure(tmp_path: Path, monkeypatch):
    """If the move/re-aggregate fails, the gate is released (gate.end) and the error propagates —
    never a half-deleted corpus holding the rebuild gate."""
    from podcast_scraper.server.index_rebuild import gate_for_corpus

    corpus = tmp_path / "corpus"
    _seed_run(corpus, "R1", 1, "g1", "ep-1")
    monkeypatch.setattr(
        corpus_rollback, "_move_to_trash", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    app = create_app(corpus, static_dir=False)
    app.state.operator_api_key = "k"
    client = TestClient(app, raise_server_exceptions=False)
    client.headers["X-Operator-Key"] = "k"
    r = client.request(
        "DELETE", "/api/corpus/runs/R1", params={"path": str(corpus), "confirm": "R1"}
    )
    assert r.status_code == 500
    # gate released so a later rebuild can start
    assert gate_for_corpus(app, corpus.resolve()).try_begin() is True


def test_consumer_episodes_get_stays_open_under_write_gate(tmp_path: Path):
    """The write-only gate must NOT lock the consumer Library GET /api/corpus/episodes."""
    corpus = tmp_path / "corpus"
    _seed_run(corpus, "R1", 1, "g1", "ep-1")
    app = create_app(corpus, static_dir=False)
    app.state.operator_api_key = "secret-key"
    client = TestClient(app)
    r = client.get("/api/corpus/episodes", params={"path": str(corpus)})
    assert r.status_code != 403  # consumer read is not operator-gated
