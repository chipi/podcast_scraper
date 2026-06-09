"""Round-trip integration test for the cache backup/restore scripts.

Regression guard for the silent-no-op bug found in review: ``restore_cache.py``
hardcoded a ``.cache/`` member prefix while ``backup_cache.py`` prefixes arcnames
with the *source* cache dir's basename (``relative_to(cache_dir.parent)``). So
backing up any dir NOT literally named ``.cache`` restored 0 files yet still
printed "Restore completed successfully". This drives both scripts end-to-end and
asserts a byte-for-byte round-trip for multiple cache dir names.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / "cache" / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_backup = _load("backup_cache")
_restore = _load("restore_cache")

pytestmark = pytest.mark.unit


def _build_cache(root: Path) -> None:
    blobs = root / "models--sentence-transformers--all-MiniLM-L6-v2" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "abc123").write_bytes(b"fake-weights")
    (root / "whisper").mkdir()
    (root / "whisper" / "tiny.en.pt").write_bytes(b"fake-whisper-model")


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.mark.parametrize("basename", [".cache", "huggingface"])
def test_backup_restore_roundtrip(tmp_path, basename):
    src = tmp_path / "src" / basename
    _build_cache(src)
    out = tmp_path / "backups"
    out.mkdir()

    backup_path = _backup.create_backup(src, out)
    assert backup_path is not None and backup_path.is_file()

    target = tmp_path / "dst" / basename
    assert _restore.restore_backup(backup_path, target, force=True) is True

    # The bug: for basename="huggingface" the old restore extracted 0 files.
    assert _tree(target) == _tree(src), f"round-trip mismatch for cache dir named {basename!r}"
