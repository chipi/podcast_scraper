"""End-to-end test of the real 0001 migration + CLI dispatch (#862).

Builds a tiny FAISS corpus, drives the actual ``podcast upgrade`` CLI path
(parse → run → status → verify), and asserts the ledger + LanceDB index land.
"""

from __future__ import annotations

import logging

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")
pytest.importorskip("faiss")

from podcast_scraper.search.faiss_store import FaissVectorStore  # noqa: E402
from podcast_scraper.upgrade.cli_handlers import parse_upgrade_argv, run_upgrade_cli  # noqa: E402
from podcast_scraper.upgrade.state import FilesystemStateStore  # noqa: E402

log = logging.getLogger("test")


def _tiny_corpus(root):
    search = root / "search"
    store = FaissVectorStore(4, index_dir=search)
    store.upsert(
        "insight:1",
        [0.1, 0.2, 0.3, 0.4],
        {"doc_type": "insight", "text": "Altman on AI", "episode_id": "ep1", "feed_id": "A"},
    )
    store.upsert(
        "chunk:1",
        [0.9, 0.1, 0.0, 0.1],
        {
            "doc_type": "transcript",
            "text": "a chunk",
            "episode_id": "ep1",
            "feed_id": "A",
            "timestamp_start_ms": 0,
            "timestamp_end_ms": 1000,
        },
    )
    store.persist()
    (root / "corpus_manifest.json").write_text(
        '{"produced_by": {"code_version": "2.6.0"}}', encoding="utf-8"
    )


def _run(corpus, *argv):
    args = parse_upgrade_argv([argv[0], "--corpus-dir", str(corpus), *argv[1:]])
    return run_upgrade_cli(args, log)


def test_status_then_run_then_verify(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _tiny_corpus(corpus)

    # status before: pending → exit 2.
    assert _run(corpus, "status") == 2

    # run --yes applies the chain: 0001 migrates from FAISS, 0002 sees the index and
    # no-ops.
    assert _run(corpus, "run", "--yes") == 0
    store = FilesystemStateStore(corpus)
    assert store.applied_migration_ids() == {
        "0001_faiss_to_lance",
        "0002_two_tier_native_reindex",
    }
    assert store.current_version() == "2.7.0"
    assert (corpus / "search" / "lance_index").exists()

    # status after: up to date → exit 0.
    assert _run(corpus, "status") == 0

    # verify passes.
    assert _run(corpus, "verify") == 0

    # idempotent: a second run is a no-op (still exit 0, ledger unchanged).
    assert _run(corpus, "run", "--yes") == 0
    assert len(store.applied_records()) == 2


def test_run_dry_run_writes_nothing(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    _tiny_corpus(corpus)
    assert _run(corpus, "run", "--dry-run") == 0
    assert FilesystemStateStore(corpus).applied_migration_ids() == set()
    assert not (corpus / "search" / "lance_index").exists()
    assert not (corpus / "upgrade_ledger.json").exists()


def test_no_faiss_index_is_clean_noop(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "corpus_manifest.json").write_text(
        '{"produced_by": {"code_version": "2.6.0"}}', encoding="utf-8"
    )
    # No FAISS index → migration applies as a no-op, version still advances.
    assert _run(corpus, "run", "--yes") == 0
    assert FilesystemStateStore(corpus).current_version() == "2.7.0"
    assert _run(corpus, "verify") == 0  # no-op verifies ok
