"""Module-owned migration histories, against the test database (ADR-172)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from sqlalchemy import Column, create_engine, Integer, MetaData, Table, text

from podcast_scraper.db import init_engines, migrate, reset_engines
from podcast_scraper.db.cli_handlers import parse_db_argv, run_db_cli
from podcast_scraper.db.engines import DatabaseNotReady
from podcast_scraper.db.importers import registry
from podcast_scraper.db.settings import DatabaseSettings, normalize_url
from podcast_scraper.extensions import use_extensions
from tests._database import worker_database_url
from tests.unit.podcast_scraper.db import demo_module

PLANE = "player"


def db_main(argv: list[str]) -> int:
    return run_db_cli(parse_db_argv(argv), logging.getLogger("test"))


@pytest.fixture
def url() -> str:
    found = worker_database_url()
    assert found, "set PODCAST_TEST_DATABASE_URL (make test-db-up)"
    return found


def _sql(url: str, statement: str, **params: object) -> list[tuple]:
    engine = create_engine(normalize_url(url))
    try:
        with engine.begin() as conn:
            result = conn.execute(text(statement), params)
            return list(result) if result.returns_rows else []
    finally:
        engine.dispose()


@pytest.fixture
def demo(url: str):
    """The demo module installed, its table and version table absent at the start and the end."""

    def drop() -> None:
        _sql(url, "DROP TABLE IF EXISTS player.demo_items, player.alembic_version_demo")
        _sql(url, "DELETE FROM player.schema_history WHERE revision LIKE 'demo\\_%'")

    reset_engines()
    migrate.upgrade(url, PLANE)
    drop()
    with use_extensions([demo_module.extension()]):
        yield
    reset_engines()
    drop()


def test_upgrade_runs_the_module_after_the_kernel_with_its_own_version_table(
    url: str, demo: None
) -> None:
    applied = migrate.upgrade_all(url, PLANE)
    assert applied["demo"] == ["demo_0001_items"]
    assert _sql(url, "SELECT version_num FROM player.alembic_version_demo") == [
        ("demo_0001_items",)
    ]
    # The kernel's own version table is untouched by the module.
    assert _sql(url, "SELECT version_num FROM player.alembic_version") == [
        (migrate.head_revision(),)
    ]
    assert _sql(
        url, "SELECT count(*) FROM player.schema_history WHERE revision = 'demo_0001_items'"
    ) == [(1,)]
    assert migrate.upgrade_all(url, PLANE)["demo"] == []
    assert migrate.module_status(url, PLANE) == [("demo", "demo_0001_items", "demo_0001_items")]


def test_a_module_does_not_run_on_a_plane_it_does_not_name(url: str, demo: None) -> None:
    assert migrate.modules("operator") == []
    assert migrate.module_status(url, "operator") == []


def test_account_deletion_cascades_into_module_tables(url: str, demo: None) -> None:
    migrate.upgrade_all(url, PLANE)
    _sql(
        url,
        "INSERT INTO player.users (user_id, provider, subject) "
        "VALUES ('u-demo', 'email', 'demo@example.test')",
    )
    _sql(url, "INSERT INTO player.demo_items (id, user_id, body) VALUES (1, 'u-demo', 'x')")
    _sql(url, "DELETE FROM player.users WHERE user_id = 'u-demo'")
    assert _sql(url, "SELECT count(*) FROM player.demo_items") == [(0,)]


def test_readiness_refuses_a_module_behind_and_starts_when_it_is_ahead(
    url: str, demo: None
) -> None:
    settings = DatabaseSettings(rw_url=url, ro_url=url, plane=PLANE)
    with pytest.raises(DatabaseNotReady, match="module demo on plane 'player'.*behind"):
        init_engines(settings)
    migrate.upgrade_all(url, PLANE)
    init_engines(settings)
    reset_engines()
    # A later release of the module migrated past this code: expand-only, so this code starts.
    _sql(url, "UPDATE player.alembic_version_demo SET version_num = 'demo_0002_later'")
    init_engines(settings)


def test_sql_render_covers_the_module(url: str, demo: None) -> None:
    rendered = migrate.upgrade_sql(None, PLANE)
    assert "-- module demo on plane player: base -> demo_0001_items" in rendered
    assert "CREATE TABLE player.demo_items" in rendered
    assert "alembic_version_demo" in rendered


def test_db_status_reports_the_module_and_exits_2_when_it_is_behind(
    url: str, demo: None, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PODCAST_DATABASE_URL", url)
    assert db_main(["status", "--plane", PLANE]) == 2
    assert "module demo on plane player: current none, head demo_0001_items" in (
        capsys.readouterr().out
    )
    assert db_main(["upgrade", "--plane", PLANE]) == 0
    assert db_main(["status", "--plane", PLANE]) == 0


def test_module_importers_join_the_registry_after_the_kernel(demo: None) -> None:
    class DemoImporter:
        name = "demo_items"

        def claims(self) -> list:
            return []

        def run(self, data_dir: Path, engine: object, *, dry_run: bool) -> object:
            raise NotImplementedError

    with use_extensions([demo_module.extension(importers=lambda: [DemoImporter()])]):
        names = list(registry(PLANE))
        assert names[0] == "users" and names[-1] == "demo_items"
        assert "demo_items" not in registry("operator")


@pytest.mark.parametrize(
    "override, message",
    [
        ({"name": "shared"}, "module name"),
        ({"name": "Demo"}, "module name"),
        ({"planes": ("corpus",)}, "plane"),
    ],
)
def test_a_module_breaking_the_rules_is_refused(override: dict, message: str) -> None:
    ext = demo_module.extension(**override)
    with pytest.raises(ValueError, match=message):
        migrate.validate_module(ext.database)


def test_a_module_table_without_its_prefix_is_refused() -> None:
    stray = MetaData()
    Table("items", stray, Column("id", Integer, primary_key=True))
    with pytest.raises(ValueError, match="demo_"):
        migrate.validate_module(demo_module.extension(metadata=stray).database)


def test_a_module_revision_without_its_prefix_is_refused(tmp_path: Path) -> None:
    (tmp_path / "versions").mkdir()
    (tmp_path / "script.py.mako").write_text((demo_module.SCRIPTS / "script.py.mako").read_text())
    (tmp_path / "versions" / "0001_items.py").write_text(
        'revision = "0001_items"\ndown_revision = None\n' "def upgrade():\n    pass\n"
    )
    with pytest.raises(ValueError, match="must start with demo_"):
        migrate.validate_module(demo_module.extension(scripts=tmp_path).database)


def _named(name: str, **overrides: object):
    from podcast_scraper.extensions import DbModule, Extension

    fields: dict[str, object] = {
        "name": name,
        "scripts": demo_module.SCRIPTS,
        "metadata": MetaData(),
        "planes": ("player",),
    }
    fields.update(overrides)
    return Extension(name=name, database=DbModule(**fields))  # type: ignore[arg-type]


def test_a_module_runs_after_the_modules_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(migrate, "module_history", lambda m: migrate.History(m.name, Path(".")))
    with use_extensions([_named("late", after=("early",)), _named("early")]):
        assert [h.name for h in migrate.modules("player")] == ["early", "late"]


def test_a_missing_or_cyclic_dependency_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(migrate, "module_history", lambda m: migrate.History(m.name, Path(".")))
    with use_extensions([_named("late", after=("absent",))]):
        with pytest.raises(ValueError, match="not on plane player"):
            migrate.modules("player")
    with use_extensions([_named("a", after=("b",)), _named("b", after=("a",))]):
        with pytest.raises(ValueError, match="cycle"):
            migrate.modules("player")
