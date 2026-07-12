"""Tests for the local corpus export/import scripts (#1175).

Covers pack_corpus_local.sh + import_local_snapshot.sh:

* Round-trip: pack a synthetic corpus, import into a tmpdir, assert byte-for-byte
  identical layout.
* Sanity refusals on the pack side: empty corpus (no ``*.gi.json``), missing
  ``feeds.spec.yaml``.
* Manifest / integrity refusals on the import side: sha256 mismatch, layout
  mismatch.
* Layout matrix: both ``codespace`` and ``prod`` layouts produce a working
  archive and extract to the correct root.

Reuses the ``_run`` / ``_script_env`` pattern from
``test_corpus_snapshot_manifest.py`` so the scripts execute with the same
environment stability guarantees.
"""

from __future__ import annotations

import filecmp
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PACK = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "pack_corpus_local.sh"
IMPORT = REPO_ROOT / "scripts" / "ops" / "corpus_snapshot" / "import_local_snapshot.sh"

# 1 KiB floor is too strict for tiny synthetic fixtures. Tests use 256 bytes;
# the production floor lives in ``pack_corpus_local.sh`` (default 1024).
TEST_MIN_BYTES = "256"


def _script_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GITHUB_OUTPUT", None)
    env.update(overrides)
    return env


def _run(
    script: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    assert script.is_file(), f"missing {script}"
    return subprocess.run(
        ["/usr/bin/env", "bash", str(script), *args],
        cwd=str(REPO_ROOT),
        env=_script_env(**(env or {})),
        capture_output=True,
        text=True,
        check=False,
    )


def _make_corpus(root: Path, *, include_gi: bool = True, include_spec: bool = True) -> Path:
    """Build a minimal but structurally valid synthetic corpus."""
    root.mkdir(parents=True, exist_ok=True)
    if include_spec:
        (root / "feeds.spec.yaml").write_text(
            "schema_version: 1\n" "feeds:\n" "  - id: showA\n" "    url: https://example.com/rss\n"
        )
    if include_gi:
        show_dir = root / "transcripts" / "showA"
        show_dir.mkdir(parents=True, exist_ok=True)
        (show_dir / "ep1.gi.json").write_text(
            json.dumps({"episode_id": "ep1", "insights": [{"text": "i1"}]}) + "\n"
        )
    return root


def _pack_env(git_sha: str = "0" * 40) -> dict[str, str]:
    return {
        "GIT_SHA": git_sha,
        "CORPUS_SNAPSHOT_MIN_TARBALL_BYTES": TEST_MIN_BYTES,
    }


# ---------- pack (export) ---------------------------------------------------


def test_pack_refuses_empty_corpus(tmp_path: Path) -> None:
    """No ``*.gi.json`` under corpus dir → pack refuses (matches backup-corpus.yml)."""
    src = _make_corpus(tmp_path / "src", include_gi=False)
    result = _run(
        PACK,
        "--corpus-dir",
        str(src),
        "--out",
        str(tmp_path / "snap.tgz"),
        env=_pack_env(),
    )
    assert result.returncode != 0
    assert "no *.gi.json artifacts" in result.stderr


def test_pack_refuses_missing_feeds_spec(tmp_path: Path) -> None:
    """No ``feeds.spec.yaml`` → pack refuses with a clear error."""
    src = _make_corpus(tmp_path / "src", include_spec=False)
    result = _run(
        PACK,
        "--corpus-dir",
        str(src),
        "--out",
        str(tmp_path / "snap.tgz"),
        env=_pack_env(),
    )
    assert result.returncode != 0
    assert "feeds.spec.yaml missing" in result.stderr


def test_pack_rejects_unknown_layout(tmp_path: Path) -> None:
    src = _make_corpus(tmp_path / "src")
    result = _run(
        PACK,
        "--corpus-dir",
        str(src),
        "--out",
        str(tmp_path / "snap.tgz"),
        "--layout",
        "nonsense",
        env=_pack_env(),
    )
    assert result.returncode != 0
    assert "layout" in result.stderr.lower()


# ---------- round-trip ------------------------------------------------------


def _assert_dirs_identical(a: Path, b: Path) -> None:
    """diff -r equivalent — walk both trees, compare each pair."""
    cmp = filecmp.dircmp(str(a), str(b))
    assert not cmp.left_only, f"only in {a}: {cmp.left_only}"
    assert not cmp.right_only, f"only in {b}: {cmp.right_only}"
    assert not cmp.diff_files, f"content differs: {cmp.diff_files}"
    for sub in cmp.common_dirs:
        _assert_dirs_identical(a / sub, b / sub)


@pytest.mark.parametrize(
    "layout,expected_root",
    [("codespace", ".codespace_corpus"), ("prod", "corpus")],
)
def test_round_trip_layouts(tmp_path: Path, layout: str, expected_root: str) -> None:
    """Pack → import → resulting tree is byte-identical to the source. Both layouts."""
    src = _make_corpus(tmp_path / "src")
    tarball = tmp_path / "snap.tgz"
    workspace = tmp_path / "target"

    pack_result = _run(
        PACK,
        "--corpus-dir",
        str(src),
        "--out",
        str(tarball),
        "--layout",
        layout,
        env=_pack_env(),
    )
    assert pack_result.returncode == 0, pack_result.stderr

    assert tarball.exists()
    assert (tmp_path / "snapshot.manifest.json").exists()

    import_result = _run(
        IMPORT,
        "--file",
        str(tarball),
        "--workspace-dir",
        str(workspace),
        "--layout",
        layout,
    )
    assert import_result.returncode == 0, import_result.stderr

    extracted = workspace / expected_root
    assert extracted.is_dir()
    _assert_dirs_identical(src, extracted)


# ---------- import: integrity refusals --------------------------------------


def _pack_ok(tmp_path: Path, layout: str = "codespace") -> tuple[Path, Path, Path]:
    src = _make_corpus(tmp_path / "src")
    tarball = tmp_path / "snap.tgz"
    result = _run(
        PACK,
        "--corpus-dir",
        str(src),
        "--out",
        str(tarball),
        "--layout",
        layout,
        env=_pack_env(),
    )
    assert result.returncode == 0, result.stderr
    return src, tarball, tmp_path / "snapshot.manifest.json"


def test_import_refuses_sha256_mismatch(tmp_path: Path) -> None:
    """Tamper the sibling manifest's ``archive.sha256`` → import refuses."""
    _src, tarball, manifest = _pack_ok(tmp_path)

    doc = json.loads(manifest.read_text())
    doc["archive"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(doc))

    result = _run(
        IMPORT,
        "--file",
        str(tarball),
        "--workspace-dir",
        str(tmp_path / "target"),
        "--layout",
        "codespace",
    )
    assert result.returncode != 0
    assert "sha256 mismatch" in result.stderr


def test_import_refuses_layout_mismatch(tmp_path: Path) -> None:
    """Pack codespace, import as prod → refuse (archive root does not match)."""
    _pack_ok(tmp_path, layout="codespace")
    result = _run(
        IMPORT,
        "--file",
        str(tmp_path / "snap.tgz"),
        "--workspace-dir",
        str(tmp_path / "target"),
        "--layout",
        "prod",
    )
    assert result.returncode != 0
    assert "layout mismatch" in result.stderr


def test_import_refuses_existing_layout_root(tmp_path: Path) -> None:
    """A pre-existing target layout dir must not be overwritten silently."""
    _pack_ok(tmp_path)
    workspace = tmp_path / "target"
    (workspace / ".codespace_corpus").mkdir(parents=True)
    (workspace / ".codespace_corpus" / "canary.txt").write_text("live data")

    result = _run(
        IMPORT,
        "--file",
        str(tmp_path / "snap.tgz"),
        "--workspace-dir",
        str(workspace),
        "--layout",
        "codespace",
    )
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert (workspace / ".codespace_corpus" / "canary.txt").read_text() == "live data"


def test_import_skip_sha256_verify_env(tmp_path: Path) -> None:
    """CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY=1 lets tampered-sha manifests through."""
    _pack_ok(tmp_path)

    manifest = tmp_path / "snapshot.manifest.json"
    doc = json.loads(manifest.read_text())
    doc["archive"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(doc))

    result = _run(
        IMPORT,
        "--file",
        str(tmp_path / "snap.tgz"),
        "--workspace-dir",
        str(tmp_path / "target"),
        "--layout",
        "codespace",
        env={"CORPUS_SNAPSHOT_SKIP_SHA256_VERIFY": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "target" / ".codespace_corpus").is_dir()


def test_import_uses_inner_manifest_when_sibling_absent(tmp_path: Path) -> None:
    """Delete the sibling manifest → import falls back to the inner one at archive root."""
    _pack_ok(tmp_path)
    (tmp_path / "snapshot.manifest.json").unlink()

    result = _run(
        IMPORT,
        "--file",
        str(tmp_path / "snap.tgz"),
        "--workspace-dir",
        str(tmp_path / "target"),
        "--layout",
        "codespace",
    )
    assert result.returncode == 0, result.stderr
    assert "inner manifest" in result.stdout


def test_import_refuses_when_no_manifest_anywhere(tmp_path: Path) -> None:
    """Non-conformant tarball → refuse."""
    fake = tmp_path / "notasnapshot.tgz"
    subprocess.check_call(
        ["/usr/bin/env", "tar", "-czf", str(fake), "-C", str(tmp_path), "."],
        cwd=str(tmp_path),
    )
    result = _run(
        IMPORT,
        "--file",
        str(fake),
        "--workspace-dir",
        str(tmp_path / "target"),
        "--layout",
        "codespace",
    )
    assert result.returncode != 0
    assert "not RFC-084 conformant" in result.stderr
