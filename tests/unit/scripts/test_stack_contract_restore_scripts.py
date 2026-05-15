"""Unit checks for stack-contract VPS restore helpers (ADR-093 / #762)."""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RESTORE = REPO_ROOT / "scripts" / "ops" / "restore_corpus_from_tarball_host.sh"
RESOLVE = REPO_ROOT / "scripts" / "ops" / "resolve_latest_snapshot_prod_tag.sh"
SELECT = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "select_release_tag.sh"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    assert script.is_file(), f"missing {script}"
    return subprocess.run(
        ["/usr/bin/env", "bash", str(script), *args],
        cwd=str(REPO_ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )


def test_restore_script_requires_tarball_argument() -> None:
    proc = _run(RESTORE)
    assert proc.returncode != 0
    assert "usage" in (proc.stderr + proc.stdout).lower()


def test_restore_script_declares_vps_compose_stack_and_health_probe() -> None:
    text = RESTORE.read_text(encoding="utf-8")
    assert "docker-compose.stack.yml" in text
    assert "docker-compose.prod.yml" in text
    assert "docker-compose.vps-prod.yml" in text
    assert "http://127.0.0.1:8000/api/health" in text
    assert "PODCAST_REPO_DIR:-/srv/podcast-scraper" in text
    assert "expected top-level corpus/" in text
    assert "RESTORE_EXTRACT_ONLY" in text


def test_restore_script_rejects_tarball_without_prod_corpus_layout(tmp_path: Path) -> None:
    tarball = tmp_path / "snapshot.tgz"
    empty = tmp_path / "empty"
    empty.mkdir()
    subprocess.run(
        ["tar", "-czf", str(tarball), "-C", str(empty), "."],
        check=True,
    )
    env = os.environ.copy()
    env["PODCAST_REPO_DIR"] = str(tmp_path)
    proc = subprocess.run(
        ["/usr/bin/env", "bash", str(RESTORE), str(tarball)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "expected top-level corpus/" in proc.stderr


def test_restore_script_extract_only_round_trip_prod_layout(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "build" / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "marker.txt").write_text("ok", encoding="utf-8")
    tarball = tmp_path / "snapshot.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(corpus_dir, arcname="corpus")

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "corpus" / "old.txt").parent.mkdir(parents=True)
    (repo_dir / "corpus" / "old.txt").write_text("stale", encoding="utf-8")

    env = os.environ.copy()
    env["PODCAST_REPO_DIR"] = str(repo_dir)
    env["RESTORE_EXTRACT_ONLY"] = "1"
    proc = subprocess.run(
        ["/usr/bin/env", "bash", str(RESTORE), str(tarball)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (repo_dir / "corpus" / "marker.txt").read_text(encoding="utf-8") == "ok"
    assert "Restore extract OK" in proc.stdout
    assert list(repo_dir.glob("corpus.bak.*"))


def test_resolve_prod_tag_wrapper_delegates_to_corpus_snapshot_selector() -> None:
    text = RESOLVE.read_text(encoding="utf-8")
    assert "corpus_snapshot/select_release_tag.sh" in text
    assert "TAG_REGEX='^snapshot-prod-[0-9]{8}$'" in text
    assert SELECT.is_file()


@pytest.mark.parametrize(
    "workflow",
    [
        REPO_ROOT / ".github/workflows/prod-restore-corpus.yml",
        REPO_ROOT / ".github/workflows/drill-restore-corpus.yml",
    ],
)
def test_restore_workflows_download_and_verify_on_runner(workflow: Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    assert "download_and_verify_snapshot.sh" in text
    assert "resolve_latest_snapshot_prod_tag.sh" in text


@pytest.mark.parametrize(
    "workflow",
    [
        REPO_ROOT / ".github/workflows/backup-corpus.yml",
        REPO_ROOT / ".github/workflows/backup-corpus-prod.yml",
    ],
)
def test_backup_workflows_finalize_before_upload(workflow: Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    assert "finalize_backup_bundle.sh" in text
    assert "snapshot.manifest.json" in text


@pytest.mark.parametrize(
    "script",
    [RESTORE, RESOLVE],
)
def test_restore_ops_scripts_pass_bash_syntax_check(script: Path) -> None:
    proc = subprocess.run(
        ["/usr/bin/env", "bash", "-n", str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
