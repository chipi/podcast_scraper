"""Where a migration's shipped outcome lives (E04 move, ADR-174): beside the package holding it."""

from __future__ import annotations

from pathlib import Path

import pytest

from podcast_scraper.upgrade import outcome_hook as oh

pytestmark = pytest.mark.unit


def test_an_extensions_migration_finds_its_outcome_beside_its_own_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E04 move: a migration an extension adds keeps its shipped outcome in ITS package's
    ``upgrade/outcomes``, the place public's own migrations use in public's package."""
    import importlib
    import sys

    from podcast_scraper.extensions import Extension, use_extensions

    pkg = tmp_path / "ext_probe_pkg" / "upgrade" / "migrations"
    pkg.mkdir(parents=True)
    for d in (pkg.parents[1], pkg.parent, pkg):
        (d / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "m0998_ext_probe.py").write_text(
        "from podcast_scraper.upgrade.migration import Migration\n"
        "class Probe(Migration):\n"
        "    id = '0998_ext_probe'\n"
        "    def apply(self, ctx):\n"
        "        raise AssertionError('never run')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("ext_probe_pkg.upgrade.migrations.m0998_ext_probe")
    try:
        ext = Extension(name="probe", corpus_migrations=lambda: [module.Probe()])
        with use_extensions([ext]):
            path = oh.outcome_path("0998_ext_probe")
        assert path == (tmp_path / "ext_probe_pkg" / "upgrade" / "outcomes" / "0998_ext_probe.json")
    finally:
        for name in [n for n in sys.modules if n.startswith("ext_probe_pkg")]:
            del sys.modules[name]


def test_public_migrations_keep_their_outcomes_in_public() -> None:
    import importlib.resources

    from podcast_scraper.upgrade.registry import get_migrations

    public = Path(str(importlib.resources.files("podcast_scraper.upgrade"))) / "outcomes"
    for m in get_migrations():
        assert oh.outcome_path(m.id).resolve() == (public / f"{m.id}.json").resolve(), m.id
