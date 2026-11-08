"""Corpus migrations an installed extension adds (``Extension.corpus_migrations``): one sequence
with this package's, and an outcome-hook closure inside the extension's own package."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from podcast_scraper.extensions import Extension, use_extensions
from podcast_scraper.upgrade import outcome_hook, registry
from podcast_scraper.upgrade.migration import Migration, MigrationContext, MigrationResult

pytestmark = [pytest.mark.unit]


class _Step(Migration):
    def __init__(self, mid: str) -> None:
        self.id = mid

    def apply(self, ctx: MigrationContext) -> MigrationResult:
        return MigrationResult(migration_id=self.id, applied=False)


def _ext(*ids: str) -> Extension:
    return Extension(name="fake", corpus_migrations=lambda: [_Step(i) for i in ids])


def test_no_extension_keeps_the_in_tree_sequence() -> None:
    with use_extensions([]):
        ids = [m.id for m in registry.get_migrations()]
    assert ids == sorted(m.id for m in registry._MIGRATIONS)


def test_an_extension_migration_sorts_into_the_one_sequence() -> None:
    with use_extensions([_ext("9001_fake_step")]):
        ids = [m.id for m in registry.get_migrations()]
    assert ids[-1] == "9001_fake_step"
    assert ids[:-1] == sorted(m.id for m in registry._MIGRATIONS)


@pytest.mark.parametrize(
    "mid, refusal",
    [
        ("0031_again", "is already"),  # a number this package uses
        ("0032_reused", "retired"),
        ("9001", "NNNN_name"),
        ("12_short", "NNNN_name"),
        ("9001_Upper", "NNNN_name"),
    ],
)
def test_a_number_used_twice_retired_or_malformed_is_refused(mid: str, refusal: str) -> None:
    with use_extensions([_ext(mid)]), pytest.raises(registry.RegistryError, match=refusal):
        registry.get_migrations()


def test_two_extensions_cannot_share_a_number() -> None:
    a = Extension(name="a", corpus_migrations=lambda: [_Step("9001_a")])
    b = Extension(name="b", corpus_migrations=lambda: [_Step("9001_b")])
    with use_extensions([a, b]), pytest.raises(registry.RegistryError, match="is already"):
        registry.get_migrations()


def test_an_object_that_is_not_a_migration_is_refused() -> None:
    ext = Extension(name="fake", corpus_migrations=lambda: [object()])
    with use_extensions([ext]), pytest.raises(registry.RegistryError, match="not a Migration"):
        registry.get_migrations()


@pytest.fixture
def fake_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """An installed package ``fakeext_s5`` with one migration that imports a helper of its own and
    a module of this package."""
    pkg = tmp_path / "fakeext_s5"
    (pkg / "migrations").mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (pkg / "migrations" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "migrations" / "m9001_fake_step.py").write_text(
        "from podcast_scraper.upgrade.migration import Migration\n" "from ..helper import VALUE\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield "fakeext_s5.migrations.m9001_fake_step"
    for name in [n for n in sys.modules if n.startswith("fakeext_s5")]:
        del sys.modules[name]


def test_an_extension_migration_closure_holds_its_own_files_and_the_platform_ones(
    fake_package: str,
) -> None:
    closure = outcome_hook.module_closure(fake_package)
    for own in ("fakeext_s5", "fakeext_s5.migrations", fake_package, "fakeext_s5.helper"):
        assert own in closure, own
    assert "podcast_scraper.upgrade.migration" in closure
    assert closure["fakeext_s5.helper"].name == "helper.py"


def test_an_extension_file_is_named_by_its_package_path_in_the_digest(fake_package: str) -> None:
    before = outcome_hook.code_digest(fake_package)
    helper = Path(outcome_hook.module_closure(fake_package)["fakeext_s5.helper"])
    helper.write_text("VALUE = 2\n", encoding="utf-8")
    assert outcome_hook.code_digest(fake_package) != before
    roots = outcome_hook._package_roots(fake_package)
    assert outcome_hook._digest_path(helper.resolve(), roots) == "fakeext_s5/helper.py"


def test_a_platform_migration_follows_only_the_platform_package() -> None:
    mod = type(registry._MIGRATIONS[0]).__module__
    assert set(outcome_hook._package_roots(mod)) == {"podcast_scraper"}
    assert all(name.split(".")[0] == "podcast_scraper" for name in outcome_hook.module_closure(mod))
