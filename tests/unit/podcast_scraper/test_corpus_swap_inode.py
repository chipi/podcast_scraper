"""DR-7: the bind-volume inode trap — repro + guard for swap_corpus_in_place.sh.

The corpus is a bind-backed docker volume (device=.../corpus) shared across 3 stacks. `mv corpus
corpus.bak` swaps the DIR INODE, so the bind keeps serving the moved .bak inode while /api/health
reports green — the trap that cost real time twice (the L64 migration-noop + the task-#14 swap).
The fix preserves the inode by emptying the dir and extracting INTO it. These tests assert that
property directly (os.stat().st_ino), so a refactor that reintroduces `mv` fails here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SWAP = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "ops"
    / "corpus_snapshot"
    / "swap_corpus_in_place.sh"
)

_needs_tools = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("tar") is None,
    reason="needs bash + tar",
)


def _make_corpus_tarball(dst: Path, gi_id: str, marker: str) -> Path:
    """Prod-layout tarball: top-level corpus/ with one episode's gi.json carrying *marker*."""
    stage = dst.parent / f"_stage_{gi_id}"
    meta = stage / "corpus" / "feeds" / "f" / "run_1" / "metadata"
    meta.mkdir(parents=True)
    (meta / f"{gi_id}.gi.json").write_text(
        json.dumps({"episode": {"episode_id": gi_id}, "marker": marker}), encoding="utf-8"
    )
    tgz = dst / "snapshot.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(stage / "corpus", arcname="corpus")
    shutil.rmtree(stage)
    return tgz


@_needs_tools
def test_swap_preserves_corpus_dir_inode(tmp_path: Path):
    """The core property: after the swap, corpus/ is the SAME inode (bind keeps resolving)."""
    corpus = tmp_path / "corpus"
    (corpus / "old").mkdir(parents=True)
    (corpus / "old" / "e0.gi.json").write_text('{"marker": "OLD"}', encoding="utf-8")
    inode_before = os.stat(corpus).st_ino

    tgz = _make_corpus_tarball(tmp_path, "e1", "NEW")
    res = subprocess.run(
        ["bash", str(_SWAP), str(tgz), str(corpus)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr

    # Inode preserved → a shared bind volume re-resolves to the new contents (no orphaning).
    assert os.stat(corpus).st_ino == inode_before
    # New contents installed, old contents gone from the live dir.
    assert list(corpus.rglob("e1.gi.json"))
    assert not list(corpus.rglob("e0.gi.json"))
    # Old contents preserved in a backup (moved, not lost).
    baks = list(tmp_path.glob("corpus.bak.*"))
    assert baks and list(baks[0].rglob("e0.gi.json"))


@_needs_tools
def test_swap_refuses_empty_corpus_tarball(tmp_path: Path):
    """A tarball with no gi.json must NOT wipe the live corpus (safe rollback = do nothing)."""
    corpus = tmp_path / "corpus"
    (corpus / "e0.gi.json").parent.mkdir(parents=True, exist_ok=True)
    (corpus / "e0.gi.json").write_text('{"marker": "OLD"}', encoding="utf-8")

    empty = tmp_path / "empty"
    (empty / "corpus").mkdir(parents=True)
    tgz = tmp_path / "empty.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(empty / "corpus", arcname="corpus")

    res = subprocess.run(
        ["bash", str(_SWAP), str(tgz), str(corpus)], capture_output=True, text=True
    )
    assert res.returncode != 0
    assert list(corpus.rglob("e0.gi.json"))  # live corpus untouched


def test_mv_dir_changes_inode_the_trap(tmp_path: Path):
    """Control: the OLD `mv corpus corpus.bak; mkdir corpus` DOES change the inode — this is
    exactly why a bind volume orphaned onto the stale copy. Documents what the fix must avoid."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    inode_before = os.stat(corpus).st_ino
    corpus.rename(tmp_path / "corpus.bak")
    (tmp_path / "corpus").mkdir()
    assert os.stat(tmp_path / "corpus").st_ino != inode_before
