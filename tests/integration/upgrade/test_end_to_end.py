"""End-to-end test of the upgrade migration chain + CLI dispatch (#862, #995).

Builds a tiny corpus of artifacts (metadata + gi.json + transcript), drives the actual
``podcast upgrade`` CLI path (parse → run → status → verify), and asserts the ledger +
LanceDB index land. FAISS was retired (#995): 0001 is a no-op, 0002 builds the two-tier
LanceDB index natively from the artifacts.
"""

from __future__ import annotations

import json
import logging

import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("lancedb")

from podcast_scraper.upgrade.cli_handlers import parse_upgrade_argv, run_upgrade_cli  # noqa: E402
from podcast_scraper.upgrade.state import FilesystemStateStore  # noqa: E402

log = logging.getLogger("test")


def _tiny_corpus(root):
    # Artifacts the native two-tier build (0002) indexes: metadata + gi.json + transcript.
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (root / "ep1.txt").write_text(
        "The central bank signaled a policy shift in markets today.", encoding="utf-8"
    )
    (meta / "ep1.gi.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "insight:n1", "type": "Insight", "properties": {"text": "policy shift"}},
                    {
                        "id": "quote:q1",
                        "type": "Quote",
                        "properties": {"timestamp_start_ms": 0, "timestamp_end_ms": 4000},
                    },
                ],
                "edges": [{"type": "SUPPORTED_BY", "from": "insight:n1", "to": "quote:q1"}],
            }
        ),
        encoding="utf-8",
    )
    (meta / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "ep1"},
                "feed": {"feed_id": "A"},
                "content": {"transcript_file_path": "ep1.txt"},
                "grounded_insights": {"artifact_path": "metadata/ep1.gi.json"},
            }
        ),
        encoding="utf-8",
    )
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

    # run --yes applies the chain: 0001 is a no-op (FAISS retired), 0002 builds the
    # two-tier LanceDB index natively from the corpus artifacts, 0003 is a no-op
    # on a tiny corpus with no .gi.json files (still records to the ledger).
    assert _run(corpus, "run", "--yes") == 0
    store = FilesystemStateStore(corpus)
    assert store.applied_migration_ids() == {
        "0001_faiss_to_lance",
        "0002_two_tier_native_reindex",
        "0003_gi_v3_typed_mentions",
    }
    assert store.current_version() == "2.7.1"
    assert (corpus / "search" / "lance_index").exists()

    # status after: up to date → exit 0.
    assert _run(corpus, "status") == 0

    # verify passes.
    assert _run(corpus, "verify") == 0

    # idempotent: a second run is a no-op (still exit 0, ledger unchanged).
    assert _run(corpus, "run", "--yes") == 0
    assert len(store.applied_records()) == 3


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
    # No FAISS index → 0001 no-ops, 0002 builds natively, 0003 no-ops (no
    # .gi.json files in this tiny corpus). Version advances to the last
    # migration's to_version.
    assert _run(corpus, "run", "--yes") == 0
    assert FilesystemStateStore(corpus).current_version() == "2.7.1"
    assert _run(corpus, "verify") == 0  # no-op verifies ok
