"""Where a migration's reviewed verdicts live (E04 move): beside the package holding it, as its
shipped outcome does (``outcome_hook.outcome_path``)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from podcast_scraper.upgrade import reviewer_verdicts as rv

pytestmark = pytest.mark.unit


def test_an_extensions_migration_finds_its_verdicts_beside_its_own_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A migration an extension adds ships its verdicts in ITS package's ``upgrade/verdicts``; a
    caller that names no directory (``rv.load(mid, ...)``) must still find them there."""
    from podcast_scraper.extensions import Extension, use_extensions

    pkg = tmp_path / "ext_verdicts_pkg" / "upgrade" / "migrations"
    pkg.mkdir(parents=True)
    for d in (pkg.parents[1], pkg.parent, pkg):
        (d / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "m0997_ext_verdicts.py").write_text(
        "from podcast_scraper.upgrade.migration import Migration\n"
        "class Probe(Migration):\n"
        "    id = '0997_ext_verdicts'\n"
        "    def apply(self, ctx):\n"
        "        raise AssertionError('never run')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("ext_verdicts_pkg.upgrade.migrations.m0997_ext_verdicts")
    try:
        ext = Extension(name="probe", corpus_migrations=lambda: [module.Probe()])
        with use_extensions([ext]):
            path = rv.verdicts_path("0997_ext_verdicts")
        assert (
            path == tmp_path / "ext_verdicts_pkg" / "upgrade" / "verdicts" / "0997_ext_verdicts.tsv"
        )
    finally:
        for name in [n for n in sys.modules if n.startswith("ext_verdicts_pkg")]:
            del sys.modules[name]


def test_an_explicit_directory_still_wins(tmp_path: Path) -> None:
    assert rv.verdicts_path("0021_anything", tmp_path) == tmp_path / "0021_anything.tsv"


def test_an_unregistered_id_reads_from_this_package() -> None:
    assert rv.verdicts_path("0000_not_registered") == rv.VERDICTS_DIR / "0000_not_registered.tsv"
