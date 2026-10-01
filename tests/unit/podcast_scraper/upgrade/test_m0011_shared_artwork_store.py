"""m0011: per-run artwork moves to the one corpus-root store the readers use (#2204)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from podcast_scraper.server.corpus_catalog import _verified_artwork_relpath
from podcast_scraper.upgrade.migration import MigrationContext
from podcast_scraper.upgrade.migrations.m0011_shared_artwork_store import (
    SharedArtworkStoreMigration,
)
from podcast_scraper.upgrade.registry import get_migrations

pytestmark = [pytest.mark.unit]

ART = ".podcast_scraper/corpus-art"


def _rel(body: bytes) -> str:
    h = hashlib.sha256(body).hexdigest()
    return f"{ART}/sha256/{h[:2]}/{h[2:4]}/{h}.jpg"


def _put(run_dir: Path, rel: str, body: bytes) -> Path:
    path = run_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _corpus(tmp_path: Path) -> tuple[Path, str, str]:
    """Two feeds, three runs; one image stored in all three runs, one in a single run."""
    a, b = b"\xff\xd8\xffshow-a" * 40, b"\xff\xd8\xffshow-b" * 40
    for run in ("feeds/f1/run_1", "feeds/f1/run_2", "feeds/f2/run_1"):
        _put(tmp_path / run, _rel(a), a)
    _put(tmp_path / "feeds/f2/run_1", _rel(b), b)
    return tmp_path, _rel(a), _rel(b)


def _run(root: Path, dry_run: bool = False):
    return SharedArtworkStoreMigration().apply(MigrationContext(corpus_root=root, dry_run=dry_run))


def _per_run_files(root: Path) -> list[Path]:
    return [p for p in root.glob("feeds/*/run_*/**/*") if p.is_file()]


def test_one_copy_of_each_image_lands_at_the_corpus_root(tmp_path: Path) -> None:
    root, rel_a, rel_b = _corpus(tmp_path)
    result = _run(root)
    assert (root / rel_a).is_file() and (root / rel_b).is_file()
    assert _per_run_files(root) == []
    assert result.details["copied"] == 2
    assert result.details["removed"] == 4
    assert result.details["conflicts"] == []


def test_the_readers_resolve_the_moved_art_with_no_metadata_rewrite(tmp_path: Path) -> None:
    """The stored relpath was always corpus-relative; only the file was in the wrong place."""
    root, rel_a, _ = _corpus(tmp_path)
    assert _verified_artwork_relpath(root, rel_a) is None
    _run(root)
    assert _verified_artwork_relpath(root, rel_a) == rel_a


def test_emptied_run_stores_are_removed(tmp_path: Path) -> None:
    root, _, _ = _corpus(tmp_path)
    _run(root)
    assert not list(root.glob("feeds/*/run_*/.podcast_scraper"))


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root, rel_a, _ = _corpus(tmp_path)
    before = sorted(str(p) for p in root.rglob("*"))
    result = _run(root, dry_run=True)
    assert sorted(str(p) for p in root.rglob("*")) == before
    assert not (root / rel_a).exists()
    assert result.details["copied"] == 2
    assert result.details["removed"] == 4


def test_second_run_is_a_no_op(tmp_path: Path) -> None:
    root, _, _ = _corpus(tmp_path)
    _run(root)
    after = sorted(str(p) for p in root.rglob("*"))
    result = _run(root)
    assert sorted(str(p) for p in root.rglob("*")) == after
    assert result.details["files"] == 0


def test_a_copy_that_does_not_match_its_hash_is_never_deleted(tmp_path: Path) -> None:
    """A corrupt shared file must not cost the good per-run copy."""
    root, rel_a, _ = _corpus(tmp_path)
    (root / rel_a).parent.mkdir(parents=True)
    (root / rel_a).write_bytes(b"truncated")
    result = _run(root)
    assert len(result.details["conflicts"]) == 3
    assert len([p for p in _per_run_files(root) if p.name == Path(rel_a).name]) == 3


def test_verify_reads_the_corpus(tmp_path: Path) -> None:
    root, _, _ = _corpus(tmp_path)
    migration = SharedArtworkStoreMigration()
    ok, message = migration.verify(MigrationContext(corpus_root=root))
    assert not ok and "4 artwork file(s)" in message
    _run(root)
    assert migration.verify(MigrationContext(corpus_root=root)) == (
        True,
        "all artwork is in the corpus-root store",
    )


def test_plan_reports_without_writing(tmp_path: Path) -> None:
    root, _, _ = _corpus(tmp_path)
    before = sorted(str(p) for p in root.rglob("*"))
    text = SharedArtworkStoreMigration().plan(MigrationContext(corpus_root=root))
    assert "3 per-run store(s), 4 file(s); 2 distinct image(s)" in text
    assert sorted(str(p) for p in root.rglob("*")) == before


def test_registered_after_0010() -> None:
    ids = [m.id for m in get_migrations()]
    assert ids.index("0011_shared_artwork_store") == ids.index("0010_canonical_person_names") + 1
