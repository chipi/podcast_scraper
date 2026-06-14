"""Pytest E2E: ``upgrade`` + ``index-two-tier`` CLI entrypoints (RFC-090 / #862).

Exercises the real operator commands end-to-end via subprocess (coverage is captured
through coverage.py's subprocess patching) so the migration / LanceDB / upgrade-runner
code is covered at the E2E layer, not just unit/integration.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

pytestmark = [pytest.mark.e2e, pytest.mark.critical_path]

pytest.importorskip("lancedb")


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "podcast_scraper.cli", *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(cwd), timeout=180, env=os.environ.copy()
    )


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _build_corpus(root: Path) -> None:
    """A minimal corpus of artifacts (metadata + gi.json + transcript) the native two-tier
    build (0002) indexes into LanceDB, so a post-migration ``search`` exercises the real path
    (#995 — FAISS retired; the index is built natively from artifacts, not migrated).
    """
    meta = root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    (root / "ep1.txt").write_text(
        "Markets moved as the central bank signaled a rate policy shift.", encoding="utf-8"
    )
    (meta / "ep1.gi.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "insight:n1",
                        "type": "Insight",
                        "properties": {"text": "the central bank shifted monetary policy"},
                    },
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
                "feed": {"feed_id": "show1"},
                "content": {"transcript_file_path": "ep1.txt"},
                "grounded_insights": {"artifact_path": "metadata/ep1.gi.json"},
            }
        ),
        encoding="utf-8",
    )
    (root / "corpus_manifest.json").write_text(
        json.dumps({"produced_by": {"code_version": "2.6.0"}}), encoding="utf-8"
    )


class TestUpgradeCliE2E:
    def test_upgrade_full_chain_via_cli(self, project_root: Path, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        _build_corpus(corpus)

        # status → pending (exit 2)
        r = _run_cli(["upgrade", "status", "--corpus-dir", str(corpus)], project_root)
        assert r.returncode == 2, f"status: {r.returncode} {r.stderr!r}"

        # list → both migrations shown
        r = _run_cli(["upgrade", "list", "--corpus-dir", str(corpus)], project_root)
        assert r.returncode == 0 and "0001_faiss_to_lance" in r.stdout

        # dry-run → writes nothing, exit 0
        r = _run_cli(["upgrade", "run", "--corpus-dir", str(corpus), "--dry-run"], project_root)
        assert r.returncode == 0
        assert not (corpus / "search" / "lance_index").exists()

        # run --yes → 0001 no-op (FAISS retired), 0002 builds LanceDB natively, records ledger
        r = _run_cli(["upgrade", "run", "--corpus-dir", str(corpus), "--yes"], project_root)
        assert r.returncode == 0, f"run: {r.returncode} {r.stderr!r}"
        assert (corpus / "search" / "lance_index").exists()
        assert (corpus / "upgrade_ledger.json").is_file()

        # status after → up to date (exit 0); verify → ok
        assert (
            _run_cli(["upgrade", "status", "--corpus-dir", str(corpus)], project_root).returncode
            == 0
        )
        assert (
            _run_cli(["upgrade", "verify", "--corpus-dir", str(corpus)], project_root).returncode
            == 0
        )

        # JSON surface
        r = _run_cli(["upgrade", "status", "--corpus-dir", str(corpus), "--json"], project_root)
        assert json.loads(r.stdout)["up_to_date"] is True

        # Search path: `search` hits the LanceDB two-tier index built above — the single
        # search path (FAISS retired, #995). Exercises hybrid_search + corpus_search
        # through the real CLI.
        env = os.environ.copy()
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "podcast_scraper.cli",
                "search",
                "central bank policy",
                "--output-dir",
                str(corpus),
                "--top-k",
                "5",
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=180,
            env=env,
        )
        assert r.returncode == 0, f"hybrid search: {r.returncode} {r.stderr!r}"

    def test_upgrade_no_faiss_corpus(self, project_root: Path, tmp_path: Path) -> None:
        # Corpus with a manifest but no FAISS index: 0001 no-ops, 0002 native-builds
        # (empty → nothing) — exercises the no-FAISS migration branches.
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "corpus_manifest.json").write_text(
            json.dumps({"produced_by": {"code_version": "2.6.0"}}), encoding="utf-8"
        )
        assert (
            _run_cli(
                ["upgrade", "run", "--corpus-dir", str(corpus), "--yes"], project_root
            ).returncode
            == 0
        )
        assert (
            _run_cli(["upgrade", "verify", "--corpus-dir", str(corpus)], project_root).returncode
            == 0
        )

    def test_upgrade_status_missing_corpus_dir(self, project_root: Path) -> None:
        env = os.environ.copy()
        env.pop("CORPUS_DIR", None)
        env.pop("OUTPUT_DIR", None)
        r = subprocess.run(
            [sys.executable, "-m", "podcast_scraper.cli", "upgrade", "status"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=60,
            env=env,
        )
        assert r.returncode == 1  # no --corpus-dir → error


def _build_real_corpus(root: Path) -> None:
    """A minimal corpus the from-corpus indexer can walk: metadata → gi.json + transcript."""
    (root / "metadata").mkdir(parents=True)
    (root / "ep1.txt").write_text(
        "The central bank signaled a major rate policy shift today. " * 8, encoding="utf-8"
    )
    (root / "ep1.gi.json").write_text(
        json.dumps(
            {
                "episode_id": "ep1",
                "nodes": [
                    {
                        "id": "insight:n1",
                        "type": "Insight",
                        "properties": {"text": "central bank shifted policy", "grounded": True},
                    },
                    {
                        "id": "quote:q1",
                        "type": "Quote",
                        "properties": {
                            "text": "we are shifting policy",
                            "timestamp_start_ms": 1000,
                            "timestamp_end_ms": 4000,
                        },
                    },
                ],
                "edges": [{"type": "SUPPORTED_BY", "from": "insight:n1", "to": "quote:q1"}],
            }
        ),
        encoding="utf-8",
    )
    (root / "metadata" / "ep1.metadata.json").write_text(
        json.dumps(
            {
                "episode": {"episode_id": "ep1", "published_date": "2026-01-01"},
                "feed": {"feed_id": "show1"},
                "grounded_insights": {"artifact_path": "ep1.gi.json"},
                "content": {"transcript_file_path": "ep1.txt"},
            }
        ),
        encoding="utf-8",
    )


class TestIndexTwoTierCliE2E:
    def test_index_two_tier_real_corpus(self, project_root: Path, tmp_path: Path) -> None:
        corpus = tmp_path / "corpus"
        _build_real_corpus(corpus)
        r = _run_cli(
            ["index-two-tier", "--output-dir", str(corpus), "--allow-download"], project_root
        )
        assert r.returncode == 0, f"index-two-tier: {r.returncode} {r.stderr!r}"
        assert "segments=1" in r.stdout and "insights=1" in r.stdout
        assert (corpus / "search" / "lance_index").exists()

    def test_index_two_tier_empty_corpus(self, project_root: Path, tmp_path: Path) -> None:
        r = _run_cli(["index-two-tier", "--output-dir", str(tmp_path)], project_root)
        assert r.returncode == 0
        assert "episodes=0" in r.stdout  # empty corpus → nothing indexed, clean exit

    def test_index_two_tier_requires_output_dir(self, project_root: Path) -> None:
        r = _run_cli(["index-two-tier"], project_root)
        assert r.returncode == 2  # argparse: --output-dir required
