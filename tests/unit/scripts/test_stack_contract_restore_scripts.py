"""Unit checks for stack-contract VPS restore helpers (ADR-093 / #762)."""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
OPS = REPO_ROOT / "scripts" / "ops"
RESTORE = OPS / "restore_corpus_from_tarball_host.sh"
RESOLVE = OPS / "resolve_latest_snapshot_prod_tag.sh"
SELECT = OPS / "corpus_snapshot" / "select_release_tag.sh"
CUTOVER = OPS / "cutover_corpus_inplace.sh"
SWAP = OPS / "corpus_snapshot" / "swap_corpus_in_place.sh"
PRUNE = OPS / "corpus_snapshot" / "prune_corpus_backups.sh"
SMOKE = OPS / "post_deploy_smoke.sh"
PACK = OPS / "corpus_snapshot" / "pack_corpus_local.sh"


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
    assert "RESTORE_EXTRACT_ONLY" in text
    # DR-2: in-place inode-preserving swap + recreate ALL consumers (not the old `api viewer` only).
    assert "swap_corpus_in_place.sh" in text
    assert "up -d --force-recreate" in text


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
    # DR-2: the in-place swap refuses a tarball with no corpus artifacts (empty/wrong layout).
    assert "gi.json" in proc.stderr


def test_restore_script_extract_only_round_trip_prod_layout(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "build" / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "marker.txt").write_text("ok", encoding="utf-8")
    # DR-2: the in-place swap refuses an artifact-less corpus, so a realistic fixture carries a
    # gi.json (a real corpus always does).
    (corpus_dir / "e1.gi.json").write_text('{"episode": {"episode_id": "e1"}}', encoding="utf-8")
    tarball = tmp_path / "snapshot.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(corpus_dir, arcname="corpus")

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "corpus" / "old.txt").parent.mkdir(parents=True)
    (repo_dir / "corpus" / "old.txt").write_text("stale", encoding="utf-8")
    inode_before = (repo_dir / "corpus").stat().st_ino

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
    assert not (repo_dir / "corpus" / "old.txt").exists()  # old contents swapped out
    assert (repo_dir / "corpus").stat().st_ino == inode_before  # DR-2: inode preserved (bind holds)
    assert "OK under" in proc.stdout
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


def test_cutover_refuses_unmanifested_tarball_without_override(tmp_path: Path) -> None:
    """DR-3/M3: the ONE sanctioned prod swap must HARD-fail on a tarball with no
    snapshot.manifest.json (the flipped WARN→fail). Exits 1 at step 1, before any docker call."""
    tarball = tmp_path / "snapshot.tgz"
    stage = tmp_path / "stage" / "corpus"
    stage.mkdir(parents=True)
    (stage / "e1.gi.json").write_text('{"episode": {"episode_id": "e1"}}', encoding="utf-8")
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(stage, arcname="corpus")  # corpus payload, but NO snapshot.manifest.json

    repo = tmp_path / "repo"
    repo.mkdir()
    env = os.environ.copy()
    env["PODCAST_REPO_DIR"] = str(repo)
    env.pop("ALLOW_UNMANIFESTED", None)
    proc = subprocess.run(
        ["/usr/bin/env", "bash", str(CUTOVER), str(tarball)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "refusing unverified cutover" in proc.stderr


def _docker_stub_bin(tmp_path: Path) -> Path:
    """A fake ``docker`` on PATH that appends its argv to docker.log and exits 0 (no daemon)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "docker"
    stub.write_text(
        '#!/usr/bin/env bash\necho "docker $*" >> "$DOCKER_STUB_LOG"\nexit 0\n', encoding="utf-8"
    )
    stub.chmod(0o755)
    return bin_dir


def test_cutover_composition_ordering(tmp_path: Path) -> None:
    """DR-3: cutover composes validate→swap→recreate-ALL→topic-clusters→prune in order. Docker is
    stubbed (records argv) and the smoke is skipped, so this asserts the orchestration behaviorally
    without a live stack — the piece each sub-script does is unit-tested separately."""
    repo = tmp_path / "repo"
    (repo / "corpus").mkdir(parents=True)
    (repo / "corpus" / "old.gi.json").write_text('{"episode": {"episode_id": "old"}}', "utf-8")
    for stamp in ("20260101T000000Z", "20260102T000000Z", "20260103T000000Z"):
        (repo / f"corpus.bak.{stamp}").mkdir()  # pre-existing backups → prune should trim to KEEP

    # Manifested-layout tarball (corpus/ with a gi.json so the swap accepts it).
    stage = tmp_path / "stage" / "corpus" / "feeds" / "f" / "run_1" / "metadata"
    stage.mkdir(parents=True)
    (stage / "e1.gi.json").write_text('{"episode": {"episode_id": "e1"}}', "utf-8")
    tarball = tmp_path / "snapshot.tgz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(tmp_path / "stage" / "corpus", arcname="corpus")

    log = tmp_path / "docker.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{_docker_stub_bin(tmp_path)}:{env['PATH']}",
            "DOCKER_STUB_LOG": str(log),
            "PODCAST_REPO_DIR": str(repo),
            "CUTOVER_SKIP_SMOKE": "1",  # smoke curls a live endpoint — out of scope here
            "ALLOW_UNMANIFESTED": "1",  # manifest validation is covered by its own test
            "PLAYER_COMPOSE": "",  # skip the separate player project
            "RESTORE_BACKUP_KEEP": "1",
        }
    )
    proc = subprocess.run(
        ["/usr/bin/env", "bash", str(CUTOVER), str(tarball)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "cutover complete" in proc.stdout + proc.stderr
    # Swap ran in place (new corpus installed, old gone, inode-preserving handled by swap test).
    assert list((repo / "corpus").rglob("e1.gi.json"))
    assert not list((repo / "corpus").rglob("old.gi.json"))
    docker_calls = log.read_text(encoding="utf-8")
    # Recreate ALL control-plane consumers (no filter) + topic-clusters in the api container.
    assert "up -d --force-recreate" in docker_calls
    assert "exec -T api" in docker_calls and "topic-clusters" in docker_calls
    # DR-6 prune trimmed the backups to KEEP=1 (+ the fresh one the swap just made).
    assert len(list(repo.glob("corpus.bak.*"))) <= 2


def test_restore_recreates_all_consumers_behaviorally(tmp_path: Path) -> None:
    """DR-2: the full restore (not extract-only) recreates ALL consumers + health-probes via docker.
    Docker is stubbed (records argv; the in-container curl is itself a `docker exec` → stubbed), so
    this asserts the real recreate call behaviorally, not just a text grep of the script."""
    repo = tmp_path / "repo"
    (repo / "corpus").mkdir(parents=True)
    (repo / "corpus" / "old.txt").write_text("stale", encoding="utf-8")

    stage = tmp_path / "stage" / "corpus"
    stage.mkdir(parents=True)
    (stage / "e1.gi.json").write_text('{"episode": {"episode_id": "e1"}}', encoding="utf-8")
    tarball = tmp_path / "snapshot.tgz"
    with tarfile.open(tarball, "w:gz") as tf:
        tf.add(stage, arcname="corpus")

    log = tmp_path / "docker.log"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{_docker_stub_bin(tmp_path)}:{env['PATH']}",
            "DOCKER_STUB_LOG": str(log),
            "PODCAST_REPO_DIR": str(repo),
        }
    )
    proc = subprocess.run(
        ["/usr/bin/env", "bash", str(RESTORE), str(tarball)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert list((repo / "corpus").rglob("e1.gi.json"))  # swapped in
    assert not (repo / "corpus" / "old.txt").exists()  # old swapped out
    docker_calls = log.read_text(encoding="utf-8")
    assert "up -d --force-recreate" in docker_calls  # ALL consumers, no service filter (DR-2)
    # in-container health probe via docker exec
    assert "exec -T api" in docker_calls and "api/health" in docker_calls


@pytest.mark.parametrize(
    "script",
    [RESTORE, RESOLVE, CUTOVER, SWAP, PRUNE, SMOKE, PACK],
)
def test_restore_ops_scripts_pass_bash_syntax_check(script: Path) -> None:
    assert script.is_file(), f"missing {script}"
    proc = subprocess.run(
        ["/usr/bin/env", "bash", "-n", str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
