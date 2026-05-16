"""Tests for corpus snapshot manifest scripts (RFC-084 / ADR-092)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "corpus_snapshot"
VALIDATE = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "validate_snapshot_manifest.sh"
EMIT = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "emit_manifest.sh"
FINALIZE = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "finalize_backup_bundle.sh"
SELECT = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "select_release_tag.sh"
DOWNLOAD = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "download_and_verify_snapshot.sh"
RESOLVE = REPO_ROOT / "scripts" / "ops" / "resolve_latest_snapshot_prod_tag.sh"
RESTORE_RELEASE = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "restore_corpus_release.sh"


def _script_env(**overrides: str) -> dict[str, str]:
    """Subprocess env for bash ops scripts.

    Drop GITHUB_OUTPUT so local tests match stdout contract.
    """
    env = os.environ.copy()
    env.pop("GITHUB_OUTPUT", None)
    env.update(overrides)
    return env


def _run(
    script: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    assert script.is_file(), f"missing {script}"
    full_env = _script_env(**(env or {}))
    return subprocess.run(
        ["/usr/bin/env", "bash", str(script), *args],
        cwd=str(REPO_ROOT),
        env=full_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _write_mock_gh(
    tmp_path: Path,
    *,
    manifest_path: Path,
    tarball_path: Path,
    releases_json: str,
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "release" && "$2" == "list" ]]; then
  cat <<'EOF'
{releases_json}
EOF
  exit 0
fi
if [[ "$1" == "release" && "$2" == "view" ]]; then
  exit 0
fi
if [[ "$1" == "release" && "$2" == "download" ]]; then
  shift 2
  out_dir=""
  out_file=""
  pattern=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --repo) shift 2 ;;
      --pattern|-p) pattern="$2"; shift 2 ;;
      --output|-o) out_file="$2"; shift 2 ;;
      --output-dir|-D) out_dir="$2"; shift 2 ;;
      *) shift ;;
    esac
  done
  if [[ -n "$out_file" ]]; then
    dest="$out_file"
  else
    if [[ -z "$out_dir" ]]; then
      out_dir="$PWD"
    fi
    dest="$out_dir/$pattern"
  fi
  mkdir -p "$(dirname "$dest")"
  if [[ "$pattern" == "snapshot.manifest.json" ]]; then
    cp "{manifest_path}" "$dest"
  elif [[ "$pattern" == "snapshot.tgz" ]]; then
    cp "{tarball_path}" "$dest"
  else
    echo "unexpected pattern: $pattern" >&2
    exit 1
  fi
  exit 0
fi
echo "unhandled gh: $*" >&2
exit 1
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return bin_dir


def _prod_snapshot_tarball(tmp_path: Path) -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "marker.txt").write_text("ok", encoding="utf-8")
    tarball = tmp_path / "snapshot.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(corpus_dir, arcname="corpus")
    return tarball


def _manifest_for_tarball(tarball: Path) -> dict[str, object]:
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    return {
        "schema_version": 1,
        "corpus_format_version": 1,
        "created_at": "2026-05-12T12:34:56Z",
        "producer": {"git_sha": "a" * 40},
        "archive": {"relative_path": "snapshot.tgz", "sha256": digest},
    }


def test_validate_accepts_fixture_manifest() -> None:
    proc = _run(VALIDATE, str(FIXTURES / "manifest_v1_ok.json"))
    assert proc.returncode == 0, proc.stderr
    assert "OK:" in proc.stdout


def test_validate_rejects_missing_producer() -> None:
    proc = _run(VALIDATE, str(FIXTURES / "manifest_v1_missing_producer.json"))
    assert proc.returncode == 1
    assert "producer" in proc.stderr.lower()


def test_validate_rejects_missing_sha256_when_required(tmp_path: Path) -> None:
    data = json.loads((FIXTURES / "manifest_v1_ok.json").read_text(encoding="utf-8"))
    del data["archive"]["sha256"]
    manifest = tmp_path / "snapshot.manifest.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    proc = _run(VALIDATE, str(manifest), env={"CORPUS_SNAPSHOT_REQUIRE_SHA256": "1"})
    assert proc.returncode == 1
    assert "sha256" in proc.stderr.lower()


def test_emit_manifest_writes_required_fields(tmp_path: Path) -> None:
    out = tmp_path / "snapshot.manifest.json"
    proc = _run(
        EMIT,
        "--output",
        str(out),
        "--git-sha",
        "a1b2c3d4e5f6789012345678901234567890abcd",
        env={"CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT)},
    )
    assert proc.returncode == 0, proc.stderr
    assert out.is_file()
    proc2 = _run(VALIDATE, str(out))
    assert proc2.returncode == 0, proc2.stderr


def test_finalize_injects_manifest_at_tarball_root(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "marker.txt").write_text("ok", encoding="utf-8")
    tarball = tmp_path / "snapshot.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(corpus_dir, arcname="corpus")

    proc = _run(
        FINALIZE,
        str(tarball),
        env={
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
            "GIT_SHA": "a1b2c3d4e5f6789012345678901234567890abcd",
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    sibling = tmp_path / "snapshot.manifest.json"
    assert sibling.is_file()
    sibling_data = json.loads(sibling.read_text(encoding="utf-8"))
    assert sibling_data["archive"]["sha256"]
    with tarfile.open(tarball, "r:gz") as tar:
        names = tar.getnames()
    assert any(name.endswith("snapshot.manifest.json") for name in names)


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_select_release_tag_chooses_newest_compatible(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    shutil.copy(FIXTURES / "manifest_v1_ok.json", manifest)
    tarball = tmp_path / "snapshot.tgz"
    tarball.write_bytes(b"snapshot")
    releases = json.dumps(
        [
            {"tagName": "snapshot-prod-20260101", "publishedAt": "2026-01-01T00:00:00Z"},
            {"tagName": "snapshot-prod-20261201", "publishedAt": "2026-12-01T00:00:00Z"},
        ]
    )
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=manifest,
        tarball_path=tarball,
        releases_json=releases,
    )
    proc = _run(
        SELECT,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "TAG_REGEX": r"^snapshot-prod-[0-9]{8}$",
            "BACKUP_REPO": "owner/backup",
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert proc.stdout.strip().splitlines()[-1] == "snapshot-prod-20261201"


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_resolve_latest_snapshot_prod_tag_honors_pin(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    shutil.copy(FIXTURES / "manifest_v1_ok.json", manifest)
    tarball = tmp_path / "snapshot.tgz"
    tarball.write_bytes(b"snapshot")
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=manifest,
        tarball_path=tarball,
        releases_json="[]",
    )
    proc = _run(
        RESOLVE,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "BACKUP_TAG": "snapshot-prod-20260511",
            "BACKUP_REPO": "owner/backup",
            "GH_TOKEN": "test-token",
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = [
        line for line in proc.stdout.splitlines() if line.strip() and not line.startswith("OK:")
    ]
    assert lines[0] == "snapshot-prod-20260511"
    assert lines[1] == "owner/backup"


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_download_rejects_sha256_mismatch(tmp_path: Path) -> None:
    tarball = tmp_path / "snapshot.tgz"
    tarball.write_bytes(b"payload")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus_format_version": 1,
                "created_at": "2026-05-12T12:34:56Z",
                "producer": {"git_sha": "a" * 40},
                "archive": {
                    "relative_path": "snapshot.tgz",
                    "sha256": "0" * 64,
                },
            }
        ),
        encoding="utf-8",
    )
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=manifest,
        tarball_path=tarball,
        releases_json="[]",
    )
    out_dir = tmp_path / "out"
    proc = _run(
        DOWNLOAD,
        "--tag",
        "snapshot-prod-20260511",
        "--output-dir",
        str(out_dir),
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "BACKUP_REPO": "owner/backup",
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 4
    assert "sha256 mismatch" in proc.stderr.lower()


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_download_accepts_matching_sha256(tmp_path: Path) -> None:
    tarball = _prod_snapshot_tarball(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_for_tarball(tarball)), encoding="utf-8")
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=manifest,
        tarball_path=tarball,
        releases_json="[]",
    )
    out_dir = tmp_path / "out"
    proc = _run(
        DOWNLOAD,
        "--tag",
        "snapshot-prod-20260511",
        "--output-dir",
        str(out_dir),
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "BACKUP_REPO": "owner/backup",
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (out_dir / "snapshot.tgz").is_file()
    assert (out_dir / "snapshot.manifest.json").is_file()
    assert "sha256 matches manifest" in proc.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_resolve_latest_snapshot_prod_tag_defaults_to_newest_compatible(
    tmp_path: Path,
) -> None:
    tarball = _prod_snapshot_tarball(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_for_tarball(tarball)), encoding="utf-8")
    releases = json.dumps(
        [
            {"tagName": "snapshot-prod-20260101", "publishedAt": "2026-01-01T00:00:00Z"},
            {"tagName": "snapshot-prod-20261201", "publishedAt": "2026-12-01T00:00:00Z"},
        ]
    )
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=manifest,
        tarball_path=tarball,
        releases_json=releases,
    )
    proc = _run(
        RESOLVE,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "BACKUP_REPO": "owner/backup",
            "GH_TOKEN": "test-token",
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    lines = [
        line for line in proc.stdout.splitlines() if line.strip() and not line.startswith("OK:")
    ]
    assert lines[0] == "snapshot-prod-20261201"
    assert lines[1] == "owner/backup"


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_restore_corpus_release_prod_layout_round_trip(tmp_path: Path) -> None:
    tarball = _prod_snapshot_tarball(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_for_tarball(tarball)), encoding="utf-8")
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=manifest,
        tarball_path=tarball,
        releases_json="[]",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    proc = _run(
        RESTORE_RELEASE,
        "--layout",
        "prod",
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "WORKSPACE_DIR": str(workspace),
            "PODCAST_BACKUP_TAG": "snapshot-prod-20260511",
            "BACKUP_REPO": "owner/backup",
            "TAG_REGEX": r"^snapshot-prod-[0-9]{8}$",
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (workspace / "corpus" / "marker.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_scripts_are_executable() -> None:
    for script in (VALIDATE, EMIT, FINALIZE, SELECT, DOWNLOAD, RESOLVE, RESTORE_RELEASE):
        assert os.access(script, os.X_OK), f"{script} should be executable"
