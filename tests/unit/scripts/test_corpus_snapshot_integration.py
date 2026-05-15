"""Offline integration tests for corpus snapshot backup/restore scripts."""

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
FINALIZE = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "finalize_backup_bundle.sh"
SELECT = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "select_release_tag.sh"
DOWNLOAD = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "download_and_verify_snapshot.sh"
RESTORE_RELEASE = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "restore_corpus_release.sh"
GIT_SHA = "a1b2c3d4e5f6789012345678901234567890abcd"


def _safe_extract_test_tarball(tarball: Path, dest: Path) -> None:
    """Extract a test-built tarball with path traversal checks (bandit B202)."""
    dest_root = dest.resolve()
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise AssertionError(f"unexpected link in test tarball: {member.name}")
            target = (dest_root / member.name).resolve()
            if target != dest_root and not str(target).startswith(f"{dest_root}{os.sep}"):
                raise AssertionError(f"unsafe tar member path: {member.name}")
            tar.extract(member, path=dest_root)


def _run(
    script: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    assert script.is_file(), f"missing {script}"
    full_env = {**os.environ, **(env or {})}
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
    tags_without_manifest: frozenset[str] = frozenset(),
) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    skip_tags = " ".join(sorted(tags_without_manifest))
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
  tag="$3"
  shift 3
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
  if [[ "$pattern" == "snapshot.manifest.json" ]]; then
    for skip in {skip_tags}; do
      [[ "$tag" == "$skip" ]] && exit 1
    done
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


def _finalize_layout_tarball(tmp_path: Path, *, arcname: str) -> tuple[Path, Path]:
    payload_dir = tmp_path / arcname
    payload_dir.mkdir()
    (payload_dir / "marker.txt").write_text("ok", encoding="utf-8")
    tarball = tmp_path / "snapshot.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add(payload_dir, arcname=arcname)
    proc = _run(
        FINALIZE,
        str(tarball),
        env={
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
            "GIT_SHA": GIT_SHA,
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    sibling = tmp_path / "snapshot.manifest.json"
    assert sibling.is_file()
    return tarball, sibling


@pytest.mark.parametrize("arcname", ["corpus", ".codespace_corpus"])
def test_finalize_round_trip_extracts_layout(tmp_path: Path, arcname: str) -> None:
    tarball, sibling = _finalize_layout_tarball(tmp_path, arcname=arcname)
    sibling_data = json.loads(sibling.read_text(encoding="utf-8"))
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert sibling_data["archive"]["sha256"] == digest

    extract_root = tmp_path / "extract"
    extract_root.mkdir()
    _safe_extract_test_tarball(tarball, extract_root)
    assert (extract_root / arcname / "marker.txt").read_text(encoding="utf-8") == "ok"
    assert (extract_root / "snapshot.manifest.json").is_file()
    proc = _run(VALIDATE, str(extract_root / "snapshot.manifest.json"))
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_select_skips_release_without_sibling_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    shutil.copy(FIXTURES / "manifest_v1_ok.json", manifest)
    tarball = tmp_path / "snapshot.tgz"
    tarball.write_bytes(b"snapshot")
    releases = json.dumps(
        [
            {"tagName": "snapshot-prod-20261201", "publishedAt": "2026-12-01T00:00:00Z"},
            {"tagName": "snapshot-prod-20260101", "publishedAt": "2026-01-01T00:00:00Z"},
        ]
    )
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=manifest,
        tarball_path=tarball,
        releases_json=releases,
        tags_without_manifest=frozenset({"snapshot-prod-20261201"}),
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
    assert proc.stdout.strip().splitlines()[-1] == "snapshot-prod-20260101"
    assert "skip: snapshot-prod-20261201" in proc.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_select_fails_closed_when_only_incompatible_manifests(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    shutil.copy(FIXTURES / "manifest_v1_incompatible.json", manifest)
    tarball = tmp_path / "snapshot.tgz"
    tarball.write_bytes(b"snapshot")
    releases = json.dumps(
        [{"tagName": "snapshot-prod-20261201", "publishedAt": "2026-12-01T00:00:00Z"}]
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
    assert proc.returncode == 3
    assert "no compatible snapshot" in proc.stderr.lower()


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_select_pinned_legacy_without_manifest_warns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    shutil.copy(FIXTURES / "manifest_v1_ok.json", manifest)
    tarball = tmp_path / "snapshot.tgz"
    tarball.write_bytes(b"snapshot")
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=manifest,
        tarball_path=tarball,
        releases_json="[]",
        tags_without_manifest=frozenset({"snapshot-prod-legacy"}),
    )
    proc = _run(
        SELECT,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "PODCAST_BACKUP_TAG": "snapshot-prod-legacy",
            "BACKUP_REPO": "owner/backup",
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (
        "WARNING: pinned release snapshot-prod-legacy has no snapshot.manifest.json" in proc.stderr
    )
    assert proc.stdout.strip().splitlines()[-1] == "snapshot-prod-legacy"


@pytest.mark.skipif(sys.platform == "win32", reason="bash script")
def test_download_accepts_matching_sha256(tmp_path: Path) -> None:
    tarball, sibling = _finalize_layout_tarball(tmp_path, arcname="corpus")
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=sibling,
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
@pytest.mark.parametrize(
    ("layout", "arcname", "expected_marker"),
    [
        ("prod", "corpus", "corpus/marker.txt"),
        ("codespace", ".codespace_corpus", ".codespace_corpus/marker.txt"),
    ],
)
def test_restore_corpus_release_layouts(
    tmp_path: Path, layout: str, arcname: str, expected_marker: str
) -> None:
    tarball, sibling = _finalize_layout_tarball(tmp_path, arcname=arcname)
    bin_dir = _write_mock_gh(
        tmp_path,
        manifest_path=sibling,
        tarball_path=tarball,
        releases_json="[]",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    proc = _run(
        RESTORE_RELEASE,
        "--layout",
        layout,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "WORKSPACE_DIR": str(workspace),
            "PODCAST_BACKUP_TAG": "snapshot-prod-20260511",
            "BACKUP_REPO": "owner/backup",
            "CORPUS_SNAPSHOT_REPO_ROOT": str(REPO_ROOT),
        },
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert (workspace / expected_marker).read_text(encoding="utf-8") == "ok"
